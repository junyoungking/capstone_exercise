"""M5-0: replay Mc detections through actual plate tracking, no model re-inference."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import replace
from pathlib import Path
import hashlib,json,sys,time
import cv2
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from core.config import DEFAULT
from core.io.rows import read_rows,write_rows
from core.recog.plate import Detection
from server.jobs import recognize,analyse
from eval.mc.baseline import matching
from eval.mc2b.evaluate import raw_path


def run(row):
 cv2.setNumThreads(1);source=raw_path(row['result_path']);old=read_rows(source/'rows.jsonl.gz')
 cache=[(r['seq'],r['t'],[Detection(tuple(d[:4]),d[4]) for d in r.get('dets',[])]) for r in old]
 calls=[];cfg=replace(DEFAULT.track,plate_pick=True)
 def attempt(index,count,seq,t,img,st):
  if seq==0:calls.append((index,count))
 t=time.perf_counter();new=recognize(row['video'],detector=lambda *_:[],rim=False,track_cfg=cfg,detections=cache,on_attempt=attempt)
 def serial(v):return json.dumps(v,ensure_ascii=False,separators=(',',':')).encode()
 def first(v):
  r=next((r for r in v if r.get('plate',{}).get('used')),None)
  return None if r is None else {'seq':r['seq'],'box':r['plate']['box'],'t':r['t']}
 oldstart,newstart=first(old),first(new)
 unchanged=oldstart==newstart
 if unchanged:assert serial(old)==serial(new),(row['key'],'same plate/start but row bytes changed')
 if not unchanged:
  out=ROOT/'eval_out/m5/plate_rows'/row['dataset']/(row['key'].replace('/','__')+'.jsonl.gz');out.parent.mkdir(parents=True,exist_ok=True);write_rows(out,new)
 res=json.loads((source/'result.json').read_text());sets,_=analyse(new,res['exercise'],res['fps'],res['plate_mm'])
 pred=[v['back']-row['origin'] for st in sets for v in st['rep_values'] if v['back'] is not None]
 if row['gt'] is None:tp=fp=fn=None
 else:
  pairs,fni,fpi=matching(row['gt'],pred);tp=len(pairs);fn=[row['gt'][i] for i in fni];fp=[pred[i] for i in fpi]
 return {'key':row['key'],'dataset':row['dataset'],'scope':row['scope'],'exercise':row['exercise'],'gt_count':row['gt_count'],'labeled':row['gt'] is not None,
  'pred':pred,'tp':tp,'fp':fp,'fn':fn,'mc_start':oldstart,'m5_start':newstart,'start_same':unchanged,'rows_equal':serial(old)==serial(new),
  'attempts':len(calls),'replay_s':round(time.perf_counter()-t,3)}

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--keys',nargs='*');p.add_argument('--workers',type=int,default=4);a=p.parse_args();rows=json.loads((ROOT/'results/mc/baseline.json').read_text());rows=[r for r in rows if not a.keys or r['key'] in a.keys];out=[]
 with ProcessPoolExecutor(max_workers=a.workers) as pool:
  for f in as_completed([pool.submit(run,r) for r in rows]):
   v=f.result();out.append(v);print(len(out),'/',len(rows),v['key'],'pred',len(v['pred']),'attempts',v['attempts'],'same',v['start_same'],flush=True)
 dest=ROOT/'eval_out/m5/eval0.json';dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(out,ensure_ascii=False,indent=2))
