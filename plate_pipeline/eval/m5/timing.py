"""End-to-end manual-exercise 61.5 s benchmark including pose, six threads."""
from pathlib import Path
import argparse,json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[2]
VIDEO='/home/kojy1009/Downloads/squat_user_set.mp4'
MODEL='/home/kojy1009/capstone_exercise/plate_pipeline/models/plate_tiny.onnx'

def child(root,dest):
 sys.path.insert(0,str(root));sys.path.insert(0,str(root.parent))
 from server.compare import compare
 start=time.perf_counter();result=compare(VIDEO,None,'squat',dest,threads=6,model=MODEL,plate_pick=True)
 print(json.dumps({'seconds':time.perf_counter()-start,'reps':len(result['user']['reps']),'stages':result['timing']['stages']},ensure_ascii=False))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--child',type=Path);p.add_argument('--dest',type=Path);a=p.parse_args()
 if a.child:child(a.child,a.dest);raise SystemExit
 out={}
 for name,root in [('mc2b',Path('/home/kojy1009/work/mc2b-plate-pick/plate_pipeline')),('m5',ROOT)]:
  dest=ROOT/'eval_out/m5/timing'/name;dest.mkdir(parents=True,exist_ok=True)
  env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
  p=subprocess.run([sys.executable,__file__,'--child',str(root),'--dest',str(dest)],capture_output=True,text=True,env=env,check=True)
  out[name]=json.loads(p.stdout.strip().splitlines()[-1]);print(name,out[name],flush=True)
 (ROOT/'results/m5/timing.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
