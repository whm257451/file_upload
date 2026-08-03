#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Any
import cv2, numpy as np
DET_MODEL='PP-OCRv6_medium_det'; REC_MODEL='PP-OCRv6_medium_rec'
def read_img(p):
 a=cv2.imdecode(np.fromfile(str(p),np.uint8),cv2.IMREAD_COLOR)
 if a is None: raise RuntimeError(p)
 return a
def py(v:Any):
 if isinstance(v,np.ndarray): return v.tolist()
 if isinstance(v,np.generic): return v.item()
 if isinstance(v,dict): return {str(k):py(x) for k,x in v.items()}
 if isinstance(v,(list,tuple)): return [py(x) for x in v]
 return v
def data_of(o):
 if isinstance(o,dict): d=o
 elif hasattr(o,'json'):
  d=o.json() if callable(o.json) else o.json
 else:return None
 d=py(d); return d.get('res',d) if isinstance(d,dict) else None
def tiles(im,size=960,overlap=128):
 h,w=im.shape[:2]; step=size-overlap
 xs=list(range(0,max(1,w-size+1),step)); ys=list(range(0,max(1,h-size+1),step))
 if not xs or xs[-1]!=max(0,w-size): xs.append(max(0,w-size))
 if not ys or ys[-1]!=max(0,h-size): ys.append(max(0,h-size))
 for y in ys:
  for x in xs: yield im[y:min(y+size,h),x:min(x+size,w)],x,y
def iou(a,b):
 x1=max(a[0],b[0]);y1=max(a[1],b[1]);x2=min(a[2],b[2]);y2=min(a[3],b[3]); inter=max(0,x2-x1)*max(0,y2-y1);aa=max(0,a[2]-a[0])*max(0,a[3]-a[1]);bb=max(0,b[2]-b[0])*max(0,b[3]-b[1]);return inter/max(aa+bb-inter,1e-6)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('images',nargs='+',type=Path);ap.add_argument('-o','--output',type=Path,default=Path('probe_results'));a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 from paddleocr import PaddleOCR
 ocr=PaddleOCR(text_detection_model_name=DET_MODEL,text_recognition_model_name=REC_MODEL,use_doc_orientation_classify=False,use_doc_unwarping=False,use_textline_orientation=False,device='cpu',engine='paddle')
 summary=[]
 for p in a.images:
  im=read_img(p);found=[]
  for tile,dx,dy in tiles(im):
   rr=ocr.predict(input=tile,text_det_limit_side_len=1280,text_det_limit_type='max',text_det_thresh=.15,text_det_box_thresh=.25,text_det_unclip_ratio=1.6,text_rec_score_thresh=0.0)
   for r in rr:
    d=data_of(r)
    if not d:continue
    polys=d.get('rec_polys',[]) or [];texts=d.get('rec_texts',[]) or [];scores=d.get('rec_scores',[]) or []
    for i,q in enumerate(polys):
     z=np.asarray(q,dtype=float).reshape(-1,2);z[:,0]+=dx;z[:,1]+=dy;box=[float(z[:,0].min()),float(z[:,1].min()),float(z[:,0].max()),float(z[:,1].max())]
     found.append({'polygon':z.tolist(),'bbox':box,'text':str(texts[i]) if i<len(texts) else '', 'confidence':float(scores[i]) if i<len(scores) else 0.0})
  kept=[]
  for x in sorted(found,key=lambda z:z['confidence'],reverse=True):
   if not any(iou(x['bbox'],y['bbox'])>.55 for y in kept):kept.append(x)
  kept.sort(key=lambda z:(z['bbox'][1],z['bbox'][0]));out={'input':str(p),'size':[im.shape[1],im.shape[0]],'models':[DET_MODEL,REC_MODEL],'count':len(kept),'items':kept}
  (a.output/f'{p.stem}_v6_raw.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');vis=im.copy()
  for x in kept:cv2.polylines(vis,[np.rint(np.asarray(x['polygon'])).astype(np.int32)],True,(0,255,255),2,cv2.LINE_AA)
  ok,b=cv2.imencode('.jpg',vis,[cv2.IMWRITE_JPEG_QUALITY,94]);b.tofile(str(a.output/f'{p.stem}_v6_all.jpg'));summary.append({'image':str(p),'count':len(kept)});print(p,len(kept))
 (a.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
if __name__=='__main__':main()
