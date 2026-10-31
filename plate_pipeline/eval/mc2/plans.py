"""Inspect picks using detections recorded in Mc, without running the model again."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from core.io.rows import read_rows
from core.recog.plate import Detection
from core.recog.plate_pick import choose_plate
from core.config import TrackCfg

def run(row):
 rows=read_rows(Path(row['result_path'])/'rows.jsonl.gz')
 frames=[(r['seq'],r['t'],[Detection(tuple(d[:4]),d[4]) for d in r.get('dets',[])]) for r in rows]
 chosen,legacy,tracks=choose_plate(frames)
 def info(tr):
  if tr is None:return None
  a=tr.first;h=tr.first_high(TrackCfg())
  return {'seq':a[0],'time':a[1],'box':a[2].box,'score':a[2].score,'high_seq':h[0] if h else None,'motion_d':tr.motion(),'n':len(tr.items)}
 return {'key':row['key'],'chosen':info(chosen),'legacy':info(legacy),'changed':chosen is not legacy or (chosen is not None and chosen.first[0]!=chosen.first_high(TrackCfg())[0]),'n_tracks':len(tracks)}
if __name__=='__main__':
 rows=json.loads((ROOT/'results/mc/baseline.json').read_text());result=[]
 for r in rows:
  v=run(r);result.append(v)
  if v['changed']:print(v,flush=True)
 (ROOT/'eval_out/mc2/plans.json').write_text(json.dumps(result,indent=2))
