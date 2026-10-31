"""Five standing frames: plate-box height and visible knee-to-ankle lengths."""
from pathlib import Path
import json,sys
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from core.geometry import standing_frames
from core.io.rows import read_rows
from core.io import video
from eval.m4b.cohort import M2,BASE

KEYS=['deadlift/PB_S01_R10.mp4','deadlift/PB_S07_R19.mp4','deadlift/GB_S01_R07.mp4','deadlift/LLK_S01_R04.mp4','deadlift/LLK_S05_R17.mp4']
DATA=Path('/home/kojy1009/capstone_exercise/new_data/new_dataset');OUT=ROOT/'results/m4b';entries={x['key']:x for x in json.loads(M2.read_text())['videos']};cohort={x['key']:x for x in json.loads((OUT/'cohort.json').read_text())}

for number,key in enumerate(KEYS,1):
 item=entries[key];source=Path(item['source']);source=source if source.is_absolute() else Path('/home/kojy1009/work/m2-3d/plate_pipeline')/source
 rows=read_rows(BASE/key/'plate/rows.jsonl.gz')
 with np.load(source/'raw'/key/'pose.npz') as f:
  raw=f['raw_lms'].copy();w=int(f['w']);h=int(f['h']);raw[:,:,:2]*=[w,h]
 stand=standing_frames(raw);idx=[]
 for i,row in enumerate(rows):
  p=row.get('plate',{});b=p.get('box')
  if stand[i] and p.get('used') and b and b[0]>1 and b[1]>1 and b[2]<w-1 and b[3]<h-1:idx.append(i)
 if not idx:raise ValueError('no standing plate frame: '+key)
 target=idx[len(idx)//2]
 frame=next(im for i,(_,im) in enumerate(video.frames(DATA/key)) if i==target)
 b=rows[target]['plate']['box'];x=int(round((b[0]+b[2])/2));a=(x,int(round(b[1])));z=(x,int(round(b[3])))
 side=max(((25,27),(26,28)),key=lambda s:raw[target,s[0],3]+raw[target,s[1],3]);q=raw[target,side,:2].round().astype(int);thick=max(2,round(w/450))
 cv2.rectangle(frame,(int(b[0]),int(b[1])),(int(b[2]),int(b[3])),(255,120,20),thick)
 cv2.line(frame,a,z,(0,255,255),thick);cv2.line(frame,tuple(q[0]),tuple(q[1]),(40,255,150),thick)
 length=np.linalg.norm(q[0]-q[1]);header=np.zeros((max(95,int(h*.13)),w,3),np.uint8)
 for k,text in enumerate([f'{key} | frame {target}',f'plate box height {b[3]-b[1]:.1f} px (yellow) | knee-ankle {length:.1f} px (green)',f'implied {cohort[key]["implied_mm"]:.0f} mm with 170 cm assumption']):
  cv2.putText(header,text,(10,25+k*27),cv2.FONT_HERSHEY_SIMPLEX,max(.45,w/1500),(235,235,235),max(1,round(w/900)),cv2.LINE_AA)
 image=np.vstack([header,frame]);scale=min(1,1100/image.shape[1]);image=cv2.resize(image,(int(image.shape[1]*scale),int(image.shape[0]*scale)))
 cv2.imwrite(str(OUT/f'inspection_{number:02d}.jpg'),image,[cv2.IMWRITE_JPEG_QUALITY,86]);print(key,target)
