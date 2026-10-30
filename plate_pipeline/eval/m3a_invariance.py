"""Read-only byte/source audit and exact reanalysis of 69 M2 plate records."""
import argparse,hashlib,json,subprocess
from pathlib import Path
from core.io.rows import read_rows
from server.jobs import analyse

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def audit(source,out):
    root=Path(__file__).resolve().parents[2];base=json.loads((source/'baseline.json').read_text());entries=[]
    for v in base['videos']:
        dest=source/'raw'/v['key']/'plate';files=[dest/'rows.jsonl.gz',dest/'result.json'];before=[sha(p) for p in files]
        rows=read_rows(files[0]);old=json.loads(files[1].read_text());sets,_=analyse(rows,old['exercise'],old['fps'],old['plate_mm'])
        assert sets==old['sets'],v['key']
        assert before==[sha(p) for p in files]
        entries.append(dict(key=v['key'],hashes=dict(zip(('rows.jsonl.gz','result.json'),before)),byte_unchanged=True,reanalysis_exact=True,counts=v['counts']))
    paths=subprocess.check_output(['git','ls-tree','-r','--name-only','c09f869','plate_pipeline/core','plate_pipeline/server/jobs.py','feedback'],cwd=root,text=True).splitlines();sources=[]
    for path in paths:
        old=subprocess.check_output(['git','show','c09f869:'+path],cwd=root)
        assert (root/path).read_bytes()==old,path
        sources.append(dict(path=path,sha256=hashlib.sha256(old).hexdigest()))
    assert len(entries)==69 and sum(v['counts'] is not None for v in entries)==61
    out.write_text(json.dumps(dict(videos=69,count_videos=61,exact=True,files=entries,sources=sources),ensure_ascii=False,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();audit(a.source,a.out)
