"""Per-video evidence: plate height, GT/back markers, and each unmatched-event frame."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json,sys,math
import cv2,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from eval.mc.baseline import read_analysis,get_clocks


def draw(x):
 cv2.setNumThreads(1)
 p=Path(x['result_path']);res,rows=read_analysis(p);t,y,sig,_=get_clocks(rows,res);origin=x['origin'];t=t-origin
 events=[('FN',v) for v in (x['fn'] or [])]+[('FP',v) for v in (x['fp'] or [])]
 if not events:events=[('frame',0.25*max(0,float(t[-1]))),('frame',0.65*max(0,float(t[-1])))]
 events.sort(key=lambda a:a[1]);nrows=math.ceil(len(events)/6);fig=plt.figure(figsize=(15,3+2.0*nrows));gs=fig.add_gridspec(nrows+1,6,height_ratios=[2.0]+[1]*nrows)
 ax=fig.add_subplot(gs[0,:]);ax.plot(t,100*y,color='#546878',lw=1,label='raw');ax.plot(sig.t-origin,sig.y*100,color='#9abb21',lw=1.4,label='clean')
 for i,v in enumerate(x['gt'] or []):ax.axvline(v,color='#168e69',lw=1,ls='--',label='GT back' if i==0 else None)
 for i,v in enumerate(x['pred']):ax.axvline(v,color='#d35b62',lw=1,ls=':',label='pred back' if i==0 else None)
 ax.set_title(f'{x["key"]} | {x["scope"]} | GT {x["gt_count"]} predicted {len(x["pred"])} | event labels {"yes" if x["gt"] is not None else "NO"}');ax.set_xlabel('seconds');ax.set_ylabel('bar height cm');ax.grid(alpha=.2);ax.legend(loc='best',fontsize=8)
 cap=cv2.VideoCapture(x['video'])
 for j,(kind,tm) in enumerate(events):
  ax=fig.add_subplot(gs[1+j//6,j%6]);cap.set(cv2.CAP_PROP_POS_MSEC,max(0,tm-.25)*1000);ok,im=cap.read()
  if ok:
   idx=min(range(len(rows)),key=lambda i:abs(rows[i]['t']-origin-(tm-.25)));plate=rows[idx].get('plate',{});box=plate.get('box')
   if box:
    x0,y0,x1,y1=map(int,box);cv2.rectangle(im,(x0,y0),(x1,y1),(255,100,30),max(1,int(im.shape[1]/450)))
   ax.imshow(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
  ax.set_title(f'{kind} {tm:.2f}s; frame {max(0,tm-.25):.2f}s',fontsize=8);ax.axis('off')
 cap.release();fig.tight_layout();out=ROOT/'eval_out/mc/evidence'/x['dataset'];out.mkdir(parents=True,exist_ok=True);name=x['key'].replace('/','__').replace('.mp4','')+'.png';fig.savefig(out/name,dpi=90);plt.close(fig)
 return str(out/name)
if __name__=='__main__':
 data=json.loads((ROOT/'results/mc/baseline.json').read_text())
 with ProcessPoolExecutor(max_workers=3) as pool:
  for i,p in enumerate(pool.map(draw,data),1):
   if i%10==0:print(i,p,flush=True)
