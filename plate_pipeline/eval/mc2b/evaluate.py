"""Replay frozen Mc detections through Mc2b's actual legacy and final tracking passes."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import replace
from pathlib import Path
import hashlib,json,sys,time
import cv2
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from core.config import DEFAULT
from core.io.rows import read_rows,write_rows
from core.io.video import info
from core.recog.plate import Detection
from core.recog.plate_pick import choose_plate,safe_start,tracked_motion
from server.jobs import recognize,analyse
from eval.mc.baseline import matching


def raw_path(path):
 p=Path(path)
 if p.exists():return p
 old=Path('/home/kojy1009/work')
 if p.is_relative_to(old):
  archived=Path('/home/kojy1009/.local/share/Trash/files')/p.relative_to(old)
  if archived.exists():return archived
 raise FileNotFoundError(p)


def run(row):
 cv2.setNumThreads(1)
 src=raw_path(row['result_path']);old=read_rows(src/'rows.jsonl.gz')
 cache=[(r['seq'],r['t'],[Detection(tuple(d[:4]),d[4]) for d in r.get('dets',[])]) for r in old]
 cfg=replace(DEFAULT.track,plate_pick=True,plate_pick_weak=False)
 chosen,legacy,_=choose_plate(cache,cfg,legacy_rows=old)
 vi=info(row['video']);initial=safe_start(chosen,(vi.width,vi.height),cfg) if chosen else None
 hi=legacy.first_high(cfg) if legacy else None
 changed=initial is not None and (chosen is not legacy or hi is None or initial[0]!=hi[0])
 start=time.perf_counter()
 new=recognize(row['video'],detector=lambda *_:[],rim=False,track_cfg=cfg,detections=cache)
 def serial(rows):return json.dumps(rows,ensure_ascii=False,separators=(',',':')).encode()
 oldbytes,newbytes=serial(old),serial(new)
 if not changed:assert oldbytes==newbytes,(row['key'],'unchanged pick does not reproduce Mc rows')
 if changed:
  dest=ROOT/'eval_out/mc2b/rows'/row['dataset']/(row['key'].replace('/','__')+'.jsonl.gz');dest.parent.mkdir(parents=True,exist_ok=True);write_rows(dest,new)
 res=json.loads((src/'result.json').read_text());sets,_=analyse(new,res['exercise'],res['fps'],res['plate_mm'])
 pred=[v['back']-row['origin'] for st in sets for v in st['rep_values'] if v['back'] is not None]
 if row['gt'] is None:tp=fp=fn=None
 else:
  pairs,fni,fpi=matching(row['gt'],pred);tp=len(pairs);fn=[row['gt'][i] for i in fni];fp=[pred[i] for i in fpi]
 def obj(item):
  if item is None:return None
  s,t,d=item;return {'seq':s,'t':t,'box':list(d.box),'score':d.score}
 return {'key':row['key'],'dataset':row['dataset'],'exercise':row['exercise'],'scope':row['scope'],'gt':row['gt'],'gt_count':row['gt_count'],
         'selection_changed':changed,'legacy_motion_d':tracked_motion(old),'legacy_high':obj(hi),'selected_start':obj(initial),
         'selected_legacy':chosen is legacy if chosen else None,'rows_equal':oldbytes==newbytes,'old_sha256':hashlib.sha256(oldbytes).hexdigest(),
         'new_sha256':hashlib.sha256(newbytes).hexdigest(),'pred':pred,'tp':tp,'fp':fp,'fn':fn,'replay_s':round(time.perf_counter()-start,2)}

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--keys',nargs='*');p.add_argument('--workers',type=int,default=4);a=p.parse_args()
 rows=json.loads((ROOT/'results/mc/baseline.json').read_text());rows=[r for r in rows if not a.keys or r['key'] in a.keys];out=[]
 with ProcessPoolExecutor(max_workers=a.workers) as pool:
  for f in as_completed([pool.submit(run,r) for r in rows]):
   v=f.result();out.append(v);print(len(out),'/',len(rows),v['key'],'changed',v['selection_changed'],'pred',len(v['pred']),'motion',v['legacy_motion_d'],flush=True)
 dest=ROOT/'eval_out/mc2b/evaluation.json';dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(out,ensure_ascii=False,indent=2))
