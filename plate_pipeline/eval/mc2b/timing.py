"""Sequential same-machine, six-thread end-to-end plate processing benchmark."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
VIDEO=Path('/home/kojy1009/Downloads/squat_user_set.mp4')
MODEL=Path('/home/kojy1009/capstone_exercise/plate_pipeline/models/plate_tiny.onnx')

def child(root):
 sys.path.insert(0,str(root))
 import cv2,onnxruntime as ort
 from dataclasses import replace
 from core.config import DEFAULT
 from server.jobs import process_video
 cv2.setNumThreads(1);original=ort.InferenceSession
 def session(*a,**kw):
  options=ort.SessionOptions();options.intra_op_num_threads=6;options.inter_op_num_threads=1
  return original(*a,sess_options=options,**kw)
 ort.InferenceSession=session
 cfg=replace(DEFAULT,track=replace(DEFAULT.track,plate_pick=True,plate_pick_weak=False))
 t=time.perf_counter();result,rows,_=process_video(VIDEO,'squat',model=str(MODEL),device='cpu',cfg=cfg)
 print(json.dumps({'seconds':time.perf_counter()-t,'frames':len(rows),'recog_s':result['recog_s'],'reps':sum(v['back'] is not None for s in result['sets'] for v in s['rep_values'])}))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--child',type=Path);a=p.parse_args()
 if a.child:child(a.child);raise SystemExit
 out={}
 for version,root in [('mc2',Path('/home/kojy1009/work/mc-count/plate_pipeline')),('mc2b',ROOT)]:
  env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
  q=subprocess.run([sys.executable,__file__,'--child',str(root)],capture_output=True,text=True,env=env,check=True)
  out[version]=json.loads(q.stdout.strip().splitlines()[-1]);print(version,out[version],flush=True)
 (ROOT/'results/mc2b/timing.json').write_text(json.dumps(out,indent=2))
