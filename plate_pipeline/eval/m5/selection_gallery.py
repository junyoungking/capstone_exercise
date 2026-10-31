"""Visual comparison of neutral and exercise-specific bench lifter selection."""
from pathlib import Path
import json,sys
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
base={r['key']:r for r in json.loads((ROOT/'results/mc/baseline.json').read_text())};reports=json.loads((ROOT/'results/m5/auto_evaluation.json').read_text());out=ROOT/'eval_out/m5/selection_review';out.mkdir(parents=True,exist_ok=True)
for r in reports:
 review=r.get('selection_review') or {}
 if not review.get('center_diff_gt_0_1'):continue
 b=base[r['key']];path=ROOT/'eval_out/m5/poses'/b['dataset']/(r['key'].replace('/','__')+'.npz');mpath=path.with_name(path.stem+'_manual.npz')
 with np.load(path,allow_pickle=False) as a,np.load(mpath,allow_pickle=False) as c:
  x=a['raw_lms'];y=c['raw_lms'];valid=a['valid']&c['valid'];idx=[11,12,23,24]
  d=np.linalg.norm(np.nanmean(x[:,idx,:2],axis=1)-np.nanmean(y[:,idx,:2],axis=1),axis=1);d[~valid]=0;i=int(np.nanargmax(d));poses=(x[i],y[i]);fps=float(a['fps'])
 cap=cv2.VideoCapture(b['video']);cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,frame=cap.read();cap.release()
 if not ok:continue
 panels=[]
 for label,p in zip(('neutral','bench selection'),poses):
  im=frame.copy();h,w=im.shape[:2]
  for k in (0,11,12,13,14,15,16,23,24,25,26,27,28):
   if np.isfinite(p[k]).all() and p[k,3]>=.3:cv2.circle(im,(int(p[k,0]*w),int(p[k,1]*h)),max(2,w//250),(0,0,255),-1)
  cv2.putText(im,label,(10,25),0,.6,(0,255,255),2);panels.append(im)
 both=np.hstack(panels);scale=min(1,1400/both.shape[1]);both=cv2.resize(both,None,fx=scale,fy=scale)
 cv2.imwrite(str(out/(r['key'].replace('/','__')+'.jpg')),both,[int(cv2.IMWRITE_JPEG_QUALITY),85])
