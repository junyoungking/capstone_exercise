"""Check whether experimental weak-only eligibility changes each selected candidate."""
from dataclasses import replace
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from core.io.rows import read_rows
from core.config import TrackCfg
from core.recog.plate import Detection
from core.recog.plate_pick import choose_plate

if __name__=='__main__':
 rows=json.loads((ROOT/'results/mc/baseline.json').read_text());out=[]
 def signature(tr):
  if tr is None:return None
  s,t,d=tr.first;return [s,t,list(d.box),d.score]
 for row in rows:
  saved=read_rows(Path(row['result_path'])/'rows.jsonl.gz')
  frames=[(r['seq'],r['t'],[Detection(tuple(d[:4]),d[4]) for d in r.get('dets',[])]) for r in saved]
  confirmed,_,_=choose_plate(frames,TrackCfg());weak,_,_=choose_plate(frames,replace(TrackCfg(),plate_pick_weak=True))
  out.append({'key':row['key'],'confirmed':signature(confirmed),'weak':signature(weak),'equal':signature(confirmed)==signature(weak)})
 p=ROOT/'results/mc2/weak_compare.json';p.write_text(json.dumps(out,ensure_ascii=False,indent=2))
 print('same',sum(x['equal'] for x in out),'/',len(out))
