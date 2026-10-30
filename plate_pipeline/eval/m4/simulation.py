"""Projected real trajectory and noisy rim frames through the production fitter."""
from pathlib import Path
import sys,json
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from core.io.rows import read_rows
from core.signal.height import height_signal,corrected_height_signal
from core.recog.ellipse import fit_plate_ellipse,interpolate_ellipses,smooth_ellipses
cv2.setNumThreads(1)
session=Path('/home/kojy1009/work/m3c-ui/plate_pipeline/data/sessions/9deee7471cea4b339a7bc8e80ad70d71')
rows=read_rows(session/'analysis/user/plate/rows.jsonl.gz');t,y,x=height_signal(rows)
reps=json.loads((session/'result_v3.json').read_text())['user']['reps']
X=x-x.mean();Y=y-(y.max()+y.min())/2
R=.225;F=1000.;D=3.;P0=np.array([0.,1.,0.]);ang=np.linspace(0,2*np.pi,240,endpoint=False)

def camera(yaw,pitch,roll):
 yaw,pitch,roll=np.radians([yaw,pitch,roll]);d=np.array([np.sin(yaw)*np.cos(pitch),np.sin(pitch),np.cos(yaw)*np.cos(pitch)]);C=P0+D*d
 forward=(P0-C)/np.linalg.norm(P0-C);right=np.cross(forward,[0,1,0]);right/=np.linalg.norm(right);up=np.cross(right,forward)
 return C,np.cos(roll)*right+np.sin(roll)*up,-np.sin(roll)*right+np.cos(roll)*up,forward

def project(p,cam):
 c,r,u,f=cam;v=p-c;z=v@f;return np.c_[F*(v@r)/z,-F*(v@u)/z]+[256,256]

def rep_rom(values):
 return np.array([np.ptp(values[(t>=r['start'])&(t<=r['back'])]) for r in reps])

truth=rep_rom(Y);output=[]
for pitch in (0,20):
 for yaw in (0,15,30,45,60):
  for roll in (0,8):
   if roll and pitch:continue
   cam=camera(yaw,pitch,roll); rr=[]; clipped=0
   out=ROOT/'eval_out/m4/synthetic';out.mkdir(parents=True,exist_ok=True)
   writer=cv2.VideoWriter(str(out/f'y{yaw}_p{pitch}_r{roll}.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),4.,(512,512))
   for i,(ti,xi,yi) in enumerate(zip(t,X,Y)):
    p=P0+np.array([xi,yi,0]);rim=p+R*np.c_[np.cos(ang),np.sin(ang),np.zeros_like(ang)];pr=project(rim,cam)
    lo,hi=pr.min(0),pr.max(0);c=(lo+hi)/2;size=hi-lo
    row={'t':float(ti),'cam':[0.,0.],'plate':{'used':True,'src':'det','box':[float(lo[0]),float(lo[1]),float(hi[0]),float(hi[1])],'center':c.tolist(),'size_px':size.tolist()}}
    sampled=i%6==0
    if sampled:
     # A short part of the rim is covered and the sleeve protrudes. At a few
     # samples translate the entire image to the edge and record camera motion.
     shift=np.array([0.,0.])
     if i%90==0:shift[0]=-lo[0]-12; clipped+=1
     pts=np.round(pr+shift).astype(np.int32);image=np.full((512,512,3),155,np.uint8)
     cv2.fillPoly(image,[pts],(58,58,58));cv2.polylines(image,[pts],True,(22,22,22),2,cv2.LINE_AA)
     cc=np.round(c+shift).astype(int);cv2.line(image,tuple(cc),tuple(cc+[70,0]),(195,195,195),6)
     if i%36==0:cv2.rectangle(image,tuple(cc+[-50,-100]),tuple(cc+[-5,-40]),(100,100,100),-1)
     noise=np.random.default_rng(i).normal(0,2,image.shape[:2]);image=np.clip(image.astype(np.int16)+noise[:,:,None],0,255).astype(np.uint8)
     row['cam']=shift.tolist();row['plate']['center']=(c+shift).tolist();row['plate']['box']=[float(lo[0]+shift[0]),float(lo[1]),float(hi[0]+shift[0]),float(hi[1])]
     writer.write(image)
     e=fit_plate_ellipse(image,row['plate']['box']);row['plate']['ellipse']={**e,'sampled':True} if e else {'source':'box','sampled':True,'quality':0.}
    else:row['plate']['ellipse']={'source':'box','sampled':False,'quality':0.}
    rr.append(row)
   writer.release()
   interpolate_ellipses(rr);smooth_ellipses(rr)
   # Method A and B use the instantaneous unmodified projected boxes, as in angle_sim.
   centers=np.array([r['plate']['center'] for r in rr])-np.array([r['cam'] for r in rr]);sizes=np.array([r['plate']['size_px'] for r in rr])
   dy=-np.diff(centers[:,1]);sa=.45/np.mean(sizes,axis=1);sb=.45/sizes[:,1]
   ya=np.r_[0,np.cumsum(dy*(sa[1:]+sa[:-1])/2)];yb=np.r_[0,np.cumsum(dy*(sb[1:]+sb[:-1])/2)]
   _,yd,_=corrected_height_signal(rr,scale_source='ellipse')
   errs={k:float(np.max(abs(rep_rom(v)/truth-1))*100) for k,v in [('A',ya),('B',yb),('D',yd)]}
   success=sum(r['plate']['ellipse']['source']=='ellipse' for r in rr)/len(rr)
   output.append({'yaw':yaw,'pitch':pitch,'roll':roll,'rom_error_pct':errs,'ellipse_frame_fraction':success,'clipped_samples':clipped})
   print(output[-1],flush=True)
path=ROOT/'results/m4/simulation.json';path.write_text(json.dumps({'truth_rom_cm':(truth*100).tolist(),'conditions':output},indent=2))
