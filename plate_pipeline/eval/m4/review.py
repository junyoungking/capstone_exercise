import sys,json
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw
sys.path.insert(0,str(Path.cwd()))
from core.io.rows import read_rows
from core.io import video
from eval.m4.cohort import HOME,TMP,OUT,M2
from concurrent.futures import ProcessPoolExecutor
cv2.setNumThreads(1)

def create(item):
 key=item['key'];rows=read_rows(TMP/key/'rows.jsonl.gz');path=HOME/'new_data/new_dataset'/key
 if key.startswith('team9/'):path=HOME/'plate_pipeline/data/team/test9'/key.removeprefix('team9/')
 info=video.info(path);step=max(1,round(info.fps*.25));used=[i for i,r in enumerate(rows) if r.get('plate',{}).get('used')];targets=[min(used,key=lambda i:abs(i-round(used[round((len(used)-1)*q)]/step)*step)) for q in (.2,.5,.8)];imgs={}
 for i,(_,img) in enumerate(video.frames(path)):
  if i in targets:imgs[i]=img.copy()
  if i>=max(targets):break
 panels=[]
 for i in targets:
  img=imgs.get(i);p=rows[i].get('plate',{});e=p.get('ellipse',{});canvas=np.zeros((200,200,3),np.uint8)
  if img is not None and p.get('used'):
   x0,y0,x1,y1=map(int,p['box']);cv2.rectangle(img,(x0,y0),(x1,y1),(255,100,30),2)
   if e.get('source')=='ellipse':cv2.ellipse(img,tuple(int(round(x)) for x in e['center']),(int(e['major']/2),int(e['minor']/2)),float(np.degrees(e['angle'])),0,360,(40,245,190),2)
   cx=(x0+x1)/2;cy=(y0+y1)/2;r=max(x1-x0,y1-y0)*.9;H,W=img.shape[:2]
   a,b,c,d=max(0,int(cx-r)),max(0,int(cy-r)),min(W,int(cx+r)),min(H,int(cy+r));roi=img[b:d,a:c]
   if roi.size:
    scale=min(200/roi.shape[1],200/roi.shape[0]);roi=cv2.resize(roi,(max(1,int(roi.shape[1]*scale)),max(1,int(roi.shape[0]*scale))));y=(200-roi.shape[0])//2;x=(200-roi.shape[1])//2;canvas[y:y+roi.shape[0],x:x+roi.shape[1]]=roi
  cv2.putText(canvas,e.get('source','none'),(5,16),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1);panels.append(canvas)
 return key,cv2.cvtColor(np.hstack(panels),cv2.COLOR_BGR2RGB)
if __name__=='__main__':
 items=json.loads(M2.read_text())['videos']
 with ProcessPoolExecutor(max_workers=4) as pool:out=list(pool.map(create,items))
 dest=OUT/'review_sheets';dest.mkdir(exist_ok=True)
 for first in range(0,len(out),8):
  sheet=Image.new('RGB',(1200,920),'#161616');draw=ImageDraw.Draw(sheet)
  for j,(key,im) in enumerate(out[first:first+8]):
   x=(j%2)*600;y=(j//2)*230;draw.text((x+4,y+4),f'{first+j+1}: {key}',fill='white');sheet.paste(Image.fromarray(im),(x,y+25))
  sheet.save(dest/f'{first//8+1:02}.jpg',quality=92)
 print('sheets',len(list(dest.glob('*.jpg'))))
