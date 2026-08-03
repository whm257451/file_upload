#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PaddleOCR 2.7.3 / PaddlePaddle 2.6.2 可复现手写体黄色框标注。"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from paddleocr import PaddleOCR

YELLOW = (0, 255, 255)


@dataclass
class OCRItem:
    polygon: list[list[float]]
    text: str
    confidence: float
    handwriting_score: float = 0.0
    is_handwriting: bool = False
    features: dict[str, float] | None = None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def imread(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"无法读取图片: {path}")
    return image


def imwrite(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 96])
    if not ok:
        raise RuntimeError(f"无法写入图片: {path}")
    buf.tofile(str(path))


def iter_tiles(image: np.ndarray, size: int, overlap: int):
    h, w = image.shape[:2]
    if size <= 0 or (h <= size and w <= size):
        yield image, 0, 0
        return
    step = size - overlap
    xs = list(range(0, max(1, w - size + 1), step))
    ys = list(range(0, max(1, h - size + 1), step))
    lx, ly = max(0, w - size), max(0, h - size)
    if xs[-1] != lx:
        xs.append(lx)
    if ys[-1] != ly:
        ys.append(ly)
    for y in ys:
        for x in xs:
            yield image[y:min(y + size, h), x:min(x + size, w)], x, y


def parse_v2_result(raw) -> list[OCRItem]:
    if raw is None:
        return []
    page = raw
    if isinstance(raw, list) and len(raw) == 1 and isinstance(raw[0], list):
        page = raw[0]
    out: list[OCRItem] = []
    if not isinstance(page, list):
        return out
    for line in page:
        if not isinstance(line, (list, tuple)) or len(line) < 2:
            continue
        try:
            poly = np.asarray(line[0], dtype=np.float32).reshape(-1, 2)
            rec = line[1]
            text = str(rec[0])
            conf = float(rec[1])
        except Exception:
            continue
        if len(poly) >= 4 and np.all(np.isfinite(poly)):
            out.append(OCRItem(poly.tolist(), text, conf))
    return out


def bbox(poly: list[list[float]]) -> tuple[float, float, float, float]:
    a = np.asarray(poly, np.float32)
    return float(a[:, 0].min()), float(a[:, 1].min()), float(a[:, 0].max()), float(a[:, 1].max())


def iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(aa + bb - inter, 1e-6)


def dedup(items: list[OCRItem], threshold: float = 0.55) -> list[OCRItem]:
    kept: list[OCRItem] = []
    for item in sorted(items, key=lambda x: x.confidence, reverse=True):
        if not any(iou(bbox(item.polygon), bbox(old.polygon)) >= threshold for old in kept):
            kept.append(item)
    kept.sort(key=lambda x: (bbox(x.polygon)[1], bbox(x.polygon)[0]))
    return kept


def crop_for(image: np.ndarray, poly: list[list[float]]) -> np.ndarray:
    a = np.asarray(poly, np.float32)
    x1 = max(0, int(math.floor(a[:, 0].min())) - 3)
    y1 = max(0, int(math.floor(a[:, 1].min())) - 3)
    x2 = min(image.shape[1], int(math.ceil(a[:, 0].max())) + 4)
    y2 = min(image.shape[0], int(math.ceil(a[:, 1].max())) + 4)
    return image[y1:y2, x1:x2]


def features(crop: np.ndarray, box_h: float, confidence: float, text: str) -> dict[str, float]:
    if crop.size == 0:
        return {k: 0.0 for k in ["dark_ratio", "blue_ratio", "red_ratio", "stroke_width", "irregularity", "contrast", "box_height", "inverse_confidence", "short_text"]}
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    bg = float(np.percentile(gray, 90))
    ink = (gray < max(35.0, bg - 38.0)).astype(np.uint8)
    b, g, r = cv2.split(crop.astype(np.int16))
    blue_ratio = float(np.mean((b > g + 7) & (b > r + 12) & (gray < 230)))
    red_ratio = float(np.mean((r > g + 15) & (r > b + 15) & (gray < 235)))
    dark_ratio = float(np.mean(gray < min(185.0, bg - 20.0)))
    clean = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    dist = cv2.distanceTransform(clean, cv2.DIST_L2, 3)
    vals = dist[clean > 0] * 2.0
    stroke = float(np.median(vals)) if vals.size else 0.0
    n, _, stats, _ = cv2.connectedComponentsWithStats(clean, 8)
    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float32) if n > 1 else np.array([], np.float32)
    irregular = float(np.clip(np.std(areas) / max(float(np.mean(areas)), 1e-6), 0, 3) / 3) if len(areas) >= 2 else 0.0
    return {
        "dark_ratio": dark_ratio,
        "blue_ratio": blue_ratio,
        "red_ratio": red_ratio,
        "stroke_width": stroke,
        "irregularity": irregular,
        "contrast": float(np.clip(np.std(gray) / 64.0, 0, 1)),
        "box_height": float(box_h),
        "inverse_confidence": float(np.clip((0.96 - confidence) / 0.45, 0, 1)),
        "short_text": float(np.clip((10 - len(text.strip())) / 10, 0, 1)),
    }


def score(f: dict[str, float]) -> float:
    blue = min(f["blue_ratio"] / 0.045, 1.0)
    red = min(f["red_ratio"] / 0.04, 1.0)
    dark = min(f["dark_ratio"] / 0.24, 1.0)
    stroke = float(np.clip((f["stroke_width"] - 1.2) / 3.1, 0, 1))
    height = float(np.clip((f["box_height"] - 18.0) / 34.0, 0, 1))
    return float(np.clip(
        0.31 * max(blue, red) + 0.18 * dark + 0.18 * stroke + 0.12 * f["irregularity"] +
        0.07 * f["contrast"] + 0.07 * height + 0.05 * f["inverse_confidence"] + 0.02 * f["short_text"],
        0, 1,
    ))


def classify(image: np.ndarray, items: list[OCRItem], threshold: float) -> None:
    for item in items:
        x1, y1, x2, y2 = bbox(item.polygon)
        f = features(crop_for(image, item.polygon), y2 - y1, item.confidence, item.text)
        item.features = f
        item.handwriting_score = score(f)
        item.is_handwriting = item.handwriting_score >= threshold


def draw(image: np.ndarray, items: Iterable[OCRItem], thickness: int = 4) -> np.ndarray:
    out = image.copy()
    h, w = out.shape[:2]
    for item in items:
        p = np.asarray(item.polygon, np.float32).reshape(-1, 2)
        p[:, 0] = np.clip(p[:, 0], 0, w - 1)
        p[:, 1] = np.clip(p[:, 1], 0, h - 1)
        cv2.polylines(out, [np.rint(p).astype(np.int32)], True, YELLOW, thickness, cv2.LINE_AA)
    return out


def process(engine: PaddleOCR, input_path: Path, output_dir: Path, tile_size: int, overlap: int,
            min_confidence: float, handwriting_threshold: float) -> dict:
    image = imread(input_path)
    items: list[OCRItem] = []
    tile_count = 0
    for tile, dx, dy in iter_tiles(image, tile_size, overlap):
        tile_count += 1
        raw = engine.ocr(tile, cls=False)
        for item in parse_v2_result(raw):
            if item.confidence < min_confidence:
                continue
            item.polygon = [[x + dx, y + dy] for x, y in item.polygon]
            items.append(item)
    items = dedup(items)
    classify(image, items, handwriting_threshold)
    selected = [x for x in items if x.is_handwriting]
    output_dir.mkdir(parents=True, exist_ok=True)
    out_img = output_dir / f"{input_path.stem}_yellow.jpg"
    out_json = output_dir / f"{input_path.stem}_ocr.json"
    imwrite(out_img, draw(image, selected))
    payload = {
        "input": str(input_path),
        "input_sha256": sha256_file(input_path),
        "output_image": str(out_img),
        "output_sha256": sha256_file(out_img),
        "image_size": {"width": image.shape[1], "height": image.shape[0]},
        "parameters": {"tile_size": tile_size, "overlap": overlap, "min_confidence": min_confidence,
                       "handwriting_threshold": handwriting_threshold, "yellow_bgr": list(YELLOW)},
        "tile_count": tile_count,
        "detected_count": len(items),
        "selected_count": len(selected),
        "selected_items": [asdict(x) for x in selected],
        "all_items": [asdict(x) for x in items],
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+", type=Path)
    ap.add_argument("-o", "--output-dir", type=Path, default=Path("real_results"))
    ap.add_argument("--tile-size", type=int, default=1400)
    ap.add_argument("--overlap", type=int, default=160)
    ap.add_argument("--min-confidence", type=float, default=0.30)
    ap.add_argument("--handwriting-threshold", type=float, default=0.38)
    args = ap.parse_args()

    engine = PaddleOCR(use_angle_cls=False, lang="ch", use_gpu=False, show_log=True)
    results = [process(engine, p, args.output_dir, args.tile_size, args.overlap,
                       args.min_confidence, args.handwriting_threshold) for p in args.images]
    import paddle
    import paddleocr
    summary = {
        "python": sys.version,
        "platform": platform.platform(),
        "paddle": paddle.__version__,
        "paddleocr": getattr(paddleocr, "__version__", "unknown"),
        "pipeline": "PaddleOCR 2.x official Chinese OCR models + deterministic handwriting feature filter",
        "results": results,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    for r in results:
        print(f"OK {r['input']} detected={r['detected_count']} selected={r['selected_count']} -> {r['output_image']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
