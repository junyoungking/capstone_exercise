"""Actual pose-only automatic-evidence evaluation; never fabricates missing model votes."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import json,sys,time
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT.parent))
from core.config import DEFAULT
from core.io.rows import read_rows
from core.auto.judge import assess
from eval.mc2b.evaluate import raw_path
from feedback.pose import extract_pose


def work(item):
 b,m=item;cv2.setNumThreads(1)
 import onnxruntime as ort
 original=ort.InferenceSession
 def session(*args,**kwargs):
  opts=kwargs.pop('sess_options',None) or ort.SessionOptions();opts.intra_op_num_threads=1;opts.inter_op_num_threads=1
  return original(*args,sess_options=opts,**kwargs)
 ort.InferenceSession=session
 rowfile=(raw_path(b['result_path'])/'rows.jsonl.gz') if m['start_same'] else ROOT/'eval_out/m5/plate_rows'/b['dataset']/(b['key'].replace('/','__')+'.jsonl.gz')
 rows=read_rows(rowfile);path=ROOT/'eval_out/m5/poses'/b['dataset']/(b['key'].replace('/','__')+'.npz');path.parent.mkdir(parents=True,exist_ok=True)
 start=time.perf_counter()
 try:
  pose=extract_pose(Path(b['video']),path,'squat',no_cache=False,plate_rows=rows,pose_cfg=DEFAULT.pose)
  judgment=assess(rows,pose);out={'key':b['key'],'dataset':b['dataset'],'scope':b['scope'],'truth':'benchpress' if b['exercise']=='bench' else b['exercise'],'judgment':judgment,'pose_valid':int(pose['valid'].sum()),'frames':len(rows),'seconds':round(time.perf_counter()-start,2),'pose_path':str(path)}
  if b['exercise']=='bench':
   manualpath=path.with_name(path.stem+'_manual.npz');manual=extract_pose(Path(b['video']),manualpath,'benchpress',no_cache=False,plate_rows=rows,pose_cfg=DEFAULT.pose)
   a=np.asarray(pose['raw_lms']);c=np.asarray(manual['raw_lms']);valid=np.asarray(pose['valid'])&np.asarray(manual['valid']);idx=[11,12,23,24]
   dist=np.linalg.norm(np.nanmean(a[:,idx,:2],axis=1)-np.nanmean(c[:,idx,:2],axis=1),axis=1)
   out['selection_review']={'overlap_frames':int(valid.sum()),'center_diff_gt_0_1':int(np.sum(valid&(dist>.1))),'neutral_only':int(np.sum(pose['valid']&~manual['valid'])),'manual_only':int(np.sum(manual['valid']&~pose['valid']))}
  return out
 except Exception as exc:
  return {'key':b['key'],'dataset':b['dataset'],'scope':b['scope'],'truth':'benchpress' if b['exercise']=='bench' else b['exercise'],'error':f'{type(exc).__name__}: {exc}',
          'judgment':{'status':'needs_confirmation','exercise':None,'candidate':None,'rest':'unknown','floor':'unknown','posture':'unknown','classifier':{'status':'unavailable','exercise':None}},'seconds':round(time.perf_counter()-start,2)}

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=4);p.add_argument('--keys',nargs='*');p.add_argument('--from-index',type=int,default=0);p.add_argument('--output',default='auto_evaluation.json');a=p.parse_args()
 base=json.loads((ROOT/'results/mc/baseline.json').read_text());m={r['key']:r for r in json.loads((ROOT/'eval_out/m5/eval0.json').read_text())};base=[b for b in base if b['dataset']!='edited' and b['key']!='team9/benchpress/stu8_7.mp4' and (not a.keys or b['key'] in a.keys)]
 base=base[a.from_index:];out=[]
 with ProcessPoolExecutor(max_workers=a.workers) as pool:
  for f in as_completed([pool.submit(work,(b,m[b['key']])) for b in base]):
   r=f.result();out.append(r);print(len(out),'/',len(base),r['key'],r.get('error') or r['judgment']['rest'],flush=True)
   dest=ROOT/'eval_out/m5'/a.output;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(out,ensure_ascii=False,indent=2))
