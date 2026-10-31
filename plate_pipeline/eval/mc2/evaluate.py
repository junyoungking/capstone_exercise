"""Replay Mc's per-frame detections through KLT; the detector is not rerun."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
import hashlib,json,sys,time
import cv2
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from core.config import DEFAULT
from core.io.rows import read_rows,write_rows
from core.recog.plate import Detection
from core.recog.plate_pick import choose_plate
from server.jobs import recognize,analyse
from eval.mc.baseline import matching


def run(item):
 row, variant, force = item
 cv2.setNumThreads(1)
 old=read_rows(Path(row['result_path'])/'rows.jsonl.gz')
 cached=[(r['seq'],r['t'],[Detection(tuple(d[:4]),d[4]) for d in r.get('dets',[])]) for r in old]
 chosen,legacy,_=choose_plate(cached,replace(DEFAULT.track,plate_pick_weak=variant=='weak'))
 changed=chosen is not None and (chosen is not legacy or chosen.first[0]!=chosen.first_high(DEFAULT.track)[0])
 if not changed and not force:
  new=old
 else:
  started=time.perf_counter()
  new=recognize(row['video'],detector=lambda im,seq:[],rim=False,track_cfg=replace(DEFAULT.track,plate_pick=True,plate_pick_weak=variant=='weak'),detections=cached)
  path=ROOT/'eval_out/mc2'/variant/'rows'/row['dataset']/(row['key'].replace('/','__')+'.jsonl.gz');path.parent.mkdir(parents=True,exist_ok=True);write_rows(path,new)
  print('replay',row['key'],variant,'s',round(time.perf_counter()-started,1),flush=True)
 def serial(rows):return json.dumps(rows,ensure_ascii=False,separators=(',',':')).encode()
 oldbytes=serial(old);newbytes=serial(new)
 if not changed:
  assert newbytes==oldbytes,(row['key'],'unchanged initialization produced different rows')
 res=json.loads((Path(row['result_path'])/'result.json').read_text())
 sets,_=analyse(new,res['exercise'],res['fps'],res['plate_mm'])
 pred=[v['back']-row['origin'] for st in sets for v in st['rep_values'] if v['back'] is not None]
 if row['gt'] is None:tp=fp=fn=None
 else:
  pairs,fni,fpi=matching(row['gt'],pred);tp=len(pairs);fn=[row['gt'][i] for i in fni];fp=[pred[i] for i in fpi]
 def info(tr):
  if tr is None:return None
  s,t,d=tr.first;hi=tr.first_high(DEFAULT.track)
  return {'seq':s,'t':t,'box':list(d.box),'score':d.score,'high_seq':hi[0] if hi else None,'motion_d':tr.motion(),'observations':len(tr.items)}
 return {'key':row['key'],'dataset':row['dataset'],'exercise':row['exercise'],'scope':row['scope'],'gt':row['gt'],'gt_count':row['gt_count'],'chosen':info(chosen),'legacy':info(legacy),'selection_changed':changed,'rows_equal':oldbytes==newbytes,'old_sha256':hashlib.sha256(oldbytes).hexdigest(),'new_sha256':hashlib.sha256(newbytes).hexdigest(),'pred':pred,'tp':tp,'fp':fp,'fn':fn}

if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--variant',choices=['confirmed','weak'],default='confirmed');ap.add_argument('--all',action='store_true');ap.add_argument('--keys',nargs='*');ap.add_argument('--workers',type=int,default=4);a=ap.parse_args()
 rows=json.loads((ROOT/'results/mc/baseline.json').read_text());rows=[r for r in rows if not a.keys or r['key'] in a.keys]
 with ProcessPoolExecutor(max_workers=a.workers) as pool:
  pending=[pool.submit(run,(r,a.variant,a.all)) for r in rows];out=[]
  for f in as_completed(pending):out.append(f.result());print('done',len(out),'/',len(rows),flush=True)
 p=ROOT/'eval_out/mc2'/a.variant/'evaluation.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(out,ensure_ascii=False,indent=2))
