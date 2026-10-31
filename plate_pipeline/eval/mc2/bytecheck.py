"""Replay streaming Mc with picker off, compare raw row bytes to saved Mc rows."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import replace
from pathlib import Path
import gzip,hashlib,json,sys
import cv2
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from core.config import DEFAULT
from core.io.rows import read_rows
from core.recog.plate import Detection
from server.jobs import recognize

def run(row):
 cv2.setNumThreads(1)
 old=read_rows(Path(row['result_path'])/'rows.jsonl.gz')
 cached=[(r['seq'],r['t'],[Detection(tuple(d[:4]),d[4]) for d in r.get('dets',[])]) for r in old]
 new=recognize(row['video'],detector=lambda *_:[],rim=False,track_cfg=replace(DEFAULT.track,plate_pick=False),detections=cached)
 def byt(x):return ''.join(json.dumps(r,separators=(',',':'))+'\n' for r in x).encode()
 a=gzip.decompress((Path(row['result_path'])/'rows.jsonl.gz').read_bytes());b=byt(new)
 assert a==byt(old),row['key']
 return {'key':row['key'],'equal':a==b,'old_sha256':hashlib.sha256(a).hexdigest(),'new_sha256':hashlib.sha256(b).hexdigest()}
if __name__=='__main__':
 rows=json.loads((ROOT/'results/mc/baseline.json').read_text())
 with ProcessPoolExecutor(max_workers=4) as pool:
  out=[]
  for f in as_completed([pool.submit(run,r) for r in rows]):
   v=f.result();out.append(v);print(len(out),'/',len(rows),v['key'],v['equal'],flush=True)
 p=ROOT/'results/mc2/bytecheck.json';p.write_text(json.dumps(out,indent=2))
 print('equal',sum(r['equal'] for r in out),'/',len(out),flush=True)
