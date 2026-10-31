"""Actual first tracked frames before/after selection; red boxes are selected plates."""
from pathlib import Path
import json,sys
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from core.io.rows import read_rows


def panel(cap,rows,title,width=540,height=440):
 first=next((r for r in rows if r.get('plate',{}).get('used')),None)
 if first is None:
  im=np.zeros((height,width,3),np.uint8);cv2.putText(im,title+' no plate',(10,30),0,.6,(0,255,255),1);return im
 cap.set(cv2.CAP_PROP_POS_MSEC,max(0,first['t'])*1000);ok,im=cap.read()
 if not ok:return np.zeros((height,width,3),np.uint8)
 p=first['plate'];x0,y0,x1,y1=map(int,p['box']);cv2.rectangle(im,(x0,y0),(x1,y1),(0,0,255),max(2,im.shape[1]//250))
 scale=min(width/im.shape[1],(height-35)/im.shape[0]);im=cv2.resize(im,(round(scale*im.shape[1]),round(scale*im.shape[0])))
 canvas=np.zeros((height,width,3),np.uint8);canvas[35:35+im.shape[0],:im.shape[1]]=im
 cv2.putText(canvas,f'{title}: frame {first["seq"]}, {first["t"]:.2f}s',(8,25),0,.6,(0,255,255),1,cv2.LINE_AA)
 return canvas

if __name__=='__main__':
 data=json.loads((ROOT/'eval_out/mc2/confirmed/evaluation.json').read_text());base={r['key']:r for r in json.loads((ROOT/'results/mc/baseline.json').read_text())}
 out=ROOT/'eval_out/mc2/selection';out.mkdir(parents=True,exist_ok=True)
 for r in data:
  if not r['selection_changed']:continue
  b=base[r['key']];old=read_rows(Path(b['result_path'])/'rows.jsonl.gz');new=read_rows(ROOT/'eval_out/mc2/confirmed/rows'/b['dataset']/(r['key'].replace('/','__')+'.jsonl.gz'))
  cap=cv2.VideoCapture(b['video']);a=panel(cap,old,'Mc before');c=panel(cap,new,'Mc2 after');cap.release()
  cv2.imwrite(str(out/(r['key'].replace('/','__')+'.jpg')),np.hstack([a,c]),[int(cv2.IMWRITE_JPEG_QUALITY),88])
 print(sum(r['selection_changed'] for r in data),'images',out)
