#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""固定版本 PP-OCRv6 手写内容检测与黄色框标注。"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

YELLOW = (0, 255, 255)  # OpenCV BGR
DET_MODEL = "PP-OCRv6_medium_det"
REC_MODEL = "PP-OCRv6_medium_rec"


@dataclass
class OCRItem:
    polygon: list[list[float]]
    text: str
    confidence: float
    score: float = 0.0
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
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 96])
    if not ok:
        raise RuntimeError(f"无法写入图片: {path}")
    encoded.tofile(str(path))


def pythonize(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): pythonize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [pythonize(v) for v in value]
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    return value


def result_dict(result: Any) -> dict[str, Any] | None:
    if isinstance(result, dict):
        return pythonize(result)
    for attr in ("json", "to_dict", "dict"):
        if hasattr(result, attr):
            value = getattr(result, attr)
            try:
                value = value() if callable(value) else value
            except Exception:
                continue
            if isinstance(value, dict):
                return pythonize(value)
    return None


def valid_polygon(poly: Any) -> list[list[float]] | None:
    try:
        arr = np.asarray(poly, np.float32).reshape(-1, 2)
    except Exception:
        return None
    if len(arr) < 4 or not np.all(np.isfinite(arr)):
        return None
    return arr.tolist()


def parse_result(raw: Any) -> list[OCRItem]:
    out: list[OCRItem] = []
    candidates = raw if isinstance(raw, (list, tuple)) else [raw]
    for obj in candidates:
        data = result_dict(obj)
        if not data:
            continue
        if isinstance(data.get("res"), dict):
            data = data["res"]
        polys = data.get("rec_polys") or data.get("dt_polys") or data.get("polys") or data.get("boxes")
        texts = pythonize(data.get("rec_texts") or data.get("texts") or [])
        scores = pythonize(data.get("rec_scores") or data.get("scores") or [])
        if polys is None:
            continue
        for i, poly in enumerate(pythonize(polys)):
            p = valid_polygon(poly)
            if p is None:
                continue
            text = str(texts[i]) if i < len(texts) else ""
            try:
                conf = float(scores[i]) if i < len(scores) else 1.0
            except Exception:
                conf = 0.0
            out.append(OCRItem(p, text, conf))
    return out


def iter_tiles(image: np.ndarray, size: int, overlap: int):
    h, w = image.shape[:2]
    if size <= 0 or (h <= size and w <= size):
        yield image, 0, 0
        return
    step = size - overlap
    xs = list(range(0, max(1, w - size + 1), step))
    ys = list(range(0, max(1, h - size + 1), step))
    last_x = max(0, w - size)
    last_y = max(0, h - size)
    if not xs or xs[-1] != last_x:
        xs.append(last_x)
    if not ys or ys[-1] != last_y:
        ys.append(last_y)
    for y in ys:
        for x in xs:
            yield image[y:min(y + size, h), x:min(x + size, w)], x, y


def bbox(poly: list[list[float]]) -> tuple[float, float, float, float]:
    a = np.asarray(poly, np.float32)
    return float(a[:, 0].min()), float(a[:, 1].min()), float(a[:, 0].max()), float(a[:, 1].max())


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(aa + bb - inter, 1e-6)


def dedup(items: list[OCRItem], threshold: float = 0.55) -> list[OCRItem]:
    kept: list[OCRItem] = []
    for item in sorted(items, key=lambda z: z.confidence, reverse=True):
        if not any(iou(bbox(item.polygon), bbox(old.polygon)) >= threshold for old in kept):
            kept.append(item)
    kept.sort(key=lambda z: (bbox(z.polygon)[1], bbox(z.polygon)[0]))
    return kept


def crop_for(image: np.ndarray, poly: list[list[float]]) -> np.ndarray:
    a = np.asarray(poly, np.float32)
    x1 = max(0, int(math.floor(a[:, 0].min())) - 3)
    y1 = max(0, int(math.floor(a[:, 1].min())) - 3)
    x2 = min(image.shape[1], int(math.ceil(a[:, 0].max())) + 4)
    y2 = min(image.shape[0], int(math.ceil(a[:, 1].max())) + 4)
    return image[y1:y2, x1:x2]


def handwriting_features(crop: np.ndarray, box_height: float, confidence: float, text: str) -> dict[str, float]:
    if crop.size == 0:
        return {k: 0.0 for k in ("dark_ratio", "blue_ratio", "stroke_width", "irregularity", "contrast", "box_height", "inverse_confidence", "short_text")}
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    bg = float(np.percentile(gray, 90))
    threshold = max(35.0, bg - 38.0)
    ink = (gray < threshold).astype(np.uint8)
    dark_ratio = float(np.mean(gray < min(180.0, bg - 22.0)))
    b, g, r = cv2.split(crop.astype(np.int16))
    blue_ratio = float(np.mean((b > g + 8) & (b > r + 13) & (gray < 225)))
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
        "stroke_width": stroke,
        "irregularity": irregular,
        "contrast": float(np.clip(np.std(gray) / 64.0, 0, 1)),
        "box_height": float(box_height),
        "inverse_confidence": float(np.clip((0.96 - confidence) / 0.45, 0, 1)),
        "short_text": float(np.clip((10 - len(text.strip())) / 10, 0, 1)),
    }


def handwriting_score(f: dict[str, float]) -> float:
    blue = min(f["blue_ratio"] / 0.05, 1.0)
    dark = min(f["dark_ratio"] / 0.24, 1.0)
    stroke = float(np.clip((f["stroke_width"] - 1.25) / 3.2, 0, 1))
    height = float(np.clip((f["box_height"] - 19.0) / 30.0, 0, 1))
    value = (0.30 * blue + 0.20 * dark + 0.18 * stroke + 0.12 * f["irregularity"] +
             0.07 * f["contrast"] + 0.07 * height + 0.04 * f["inverse_confidence"] + 0.02 * f["short_text"])
    return float(np.clip(value, 0, 1))


def classify(image: np.ndarray, items: list[OCRItem], threshold: float) -> None:
    for item in items:
        x1, y1, x2, y2 = bbox(item.polygon)
        f = handwriting_features(crop_for(image, item.polygon), y2 - y1, item.confidence, item.text)
        item.features = f
        item.score = handwriting_score(f)
        item.is_handwriting = item.score >= threshold


def draw(image: np.ndarray, items: Iterable[OCRItem], thickness: int = 4) -> np.ndarray:
    out = image.copy()
    h, w = out.shape[:2]
    for item in items:
        p = np.asarray(item.polygon, np.float32).reshape(-1, 2)
        p[:, 0] = np.clip(p[:, 0], 0, w - 1)
        p[:, 1] = np.clip(p[:, 1], 0, h - 1)
        cv2.polylines(out, [np.rint(p).astype(np.int32)], True, YELLOW, thickness, cv2.LINE_AA)
    return out


def build_ocr():
    from paddleocr import PaddleOCR
    kwargs = dict(
        text_detection_model_name=DET_MODEL,
        text_recognition_model_name=REC_MODEL,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        device="cpu",
        engine="paddle",
    )
    try:
        return PaddleOCR(**kwargs)
    except TypeError:
        kwargs.pop("engine", None)
        return PaddleOCR(**kwargs)


def process(engine: Any, input_path: Path, output_dir: Path, tile_size: int, overlap: int,
            min_confidence: float, handwriting_threshold: float) -> dict[str, Any]:
    image = imread(input_path)
    found: list[OCRItem] = []
    tile_count = 0
    for tile, dx, dy in iter_tiles(image, tile_size, overlap):
        tile_count += 1
        raw = engine.predict(input=tile)
        for item in parse_result(raw):
            if item.confidence < min_confidence:
                continue
            item.polygon = [[x + dx, y + dy] for x, y in item.polygon]
            found.append(item)
    found = dedup(found)
    classify(image, found, handwriting_threshold)
    selected = [x for x in found if x.is_handwriting]
    output_dir.mkdir(parents=True, exist_ok=True)
    out_image = output_dir / f"{input_path.stem}_yellow.jpg"
    out_json = output_dir / f"{input_path.stem}_ocr.json"
    imwrite(out_image, draw(image, selected))
    payload = {
        "input": str(input_path),
        "input_sha256": sha256_file(input_path),
        "output_image": str(out_image),
        "output_sha256": sha256_file(out_image),
        "image_size": {"width": image.shape[1], "height": image.shape[0]},
        "models": {"detection": DET_MODEL, "recognition": REC_MODEL},
        "parameters": {"tile_size": tile_size, "overlap": overlap, "min_confidence": min_confidence,
                       "handwriting_threshold": handwriting_threshold, "yellow_bgr": list(YELLOW)},
        "tile_count": tile_count,
        "detected_count": len(found),
        "selected_count": len(selected),
        "selected_items": [asdict(x) for x in selected],
        "all_items": [asdict(x) for x in found],
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+", type=Path)
    ap.add_argument("-o", "--output-dir", type=Path, default=Path("ocr_output"))
    ap.add_argument("--tile-size", type=int, default=1400)
    ap.add_argument("--overlap", type=int, default=160)
    ap.add_argument("--min-confidence", type=float, default=0.30)
    ap.add_argument("--handwriting-threshold", type=float, default=0.38)
    args = ap.parse_args()
    engine = build_ocr()
    summaries = [process(engine, p, args.output_dir, args.tile_size, args.overlap,
                         args.min_confidence, args.handwriting_threshold) for p in args.images]
    import paddle
    import paddleocr
    meta = {
        "python": sys.version,
        "platform": platform.platform(),
        "paddle": paddle.__version__,
        "paddleocr": getattr(paddleocr, "__version__", "unknown"),
        "models": {"detection": DET_MODEL, "recognition": REC_MODEL},
        "results": summaries,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    for s in summaries:
        print(f"OK {s['input']} detected={s['detected_count']} selected={s['selected_count']} -> {s['output_image']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
