"""Three full-frame contact images per input for scope review."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import json,sys
import cv2,numpy as np
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[2];DATA=json.loads((ROOT/'results/mc/baseline.json').read_text());TEMP=ROOT/'eval_out/mc/view_review';TEMP.mkdir(parents=True,exist_ok=True)
def make(item):
 cv2.setNumThreads(1);cap=cv2.VideoCapture(item['video']);fps=cap.get(cv2.CAP_PROP_FPS);n=cap.get(cv2.CAP_PROP_FRAME_COUNT);images=[]
 for pct in (.2,.5,.8):
  t=(n/fps)*pct;cap.set(cv2.CAP_PROP_POS_MSEC,t*1000);ok,im=cap.read();canvas=np.zeros((150,200,3),np.uint8)
  if ok:
   scale=min(200/im.shape[1],150/im.shape[0]);im=cv2.resize(im,(max(1,int(im.shape[1]*scale)),max(1,int(im.shape[0]*scale))));x=(200-im.shape[1])//2;y=(150-im.shape[0])//2;canvas[y:y+im.shape[0],x:x+im.shape[1]]=im
  cv2.putText(canvas,f'{t:.1f}s',(3,14),0,.38,(255,255,255),1);images.append(canvas)
 cap.release();return item['key'],cv2.cvtColor(np.hstack(images),cv2.COLOR_BGR2RGB)
if __name__=='__main__':
 with ProcessPoolExecutor(max_workers=3) as pool:out=list(pool.map(make,DATA))
 for start in range(0,len(out),8):
  sheet=Image.new('RGB',(1200,720),'#111');draw=ImageDraw.Draw(sheet)
  for j,(key,im) in enumerate(out[start:start+8]):
   x=(j%2)*600;y=(j//2)*180;draw.text((x+4,y+4),f'{start+j+1}: {key}',fill='white');sheet.paste(Image.fromarray(im),(x,y+24))
  sheet.save(TEMP/f'{start//8+1:02}.jpg',quality=92)
 print('contact sheets',len(list(TEMP.glob('*.jpg'))))
