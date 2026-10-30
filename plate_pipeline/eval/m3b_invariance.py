"""M3b: exact reanalysis/byte audit of 69 saved plate records plus callback AST proof."""
import argparse,ast,hashlib,json,subprocess
from pathlib import Path
from core.io.rows import read_rows
from server.jobs import analyse


class WithoutCallbacks(ast.NodeTransformer):
    def visit_arguments(self,node):
        pairs=list(zip(node.args[-len(node.defaults):],node.defaults)) if node.defaults else []
        node.defaults=[default for arg,default in pairs if arg.arg!='on_frame']
        node.args=[arg for arg in node.args if arg.arg!='on_frame']
        pairs=list(zip(node.kwonlyargs,node.kw_defaults));node.kwonlyargs=[a for a,_ in pairs if a.arg!='on_frame'];node.kw_defaults=[d for a,d in pairs if a.arg!='on_frame']
        return self.generic_visit(node)
    def visit_If(self,node):
        if 'on_frame' in ast.unparse(node.test):return None
        return self.generic_visit(node)
    def visit_Call(self,node):
        node.keywords=[k for k in node.keywords if k.arg!='on_frame' and not (k.arg is None and 'on_frame' in ast.unparse(k.value))]
        return self.generic_visit(node)


def audit(source,out):
    root=Path(__file__).resolve().parents[2];base=json.loads((source/'baseline.json').read_text());files=[]
    for v in base['videos']:
        path=source/'raw'/v['key']/'plate';paths=[path/'rows.jsonl.gz',path/'result.json']
        hashes=[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
        old=json.loads(paths[1].read_text());sets,_=analyse(read_rows(paths[0]),old['exercise'],old['fps'],old['plate_mm'])
        assert sets==old['sets'],v['key']
        assert hashes==[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
        files.append(dict(key=v['key'],hashes=hashes,exact=True,has_counts=v['counts'] is not None))
    paths=subprocess.check_output(['git','ls-tree','-r','--name-only','41b39e3','plate_pipeline/core','plate_pipeline/server/jobs.py','feedback'],cwd=root,text=True).splitlines();sources=[]
    for path in paths:
        old=subprocess.check_output(['git','show','41b39e3:'+path],cwd=root);new=(root/path).read_bytes();exact=old==new
        if not exact:
            assert path in ('plate_pipeline/server/jobs.py','feedback/pose.py'),path
            normalize=lambda b:ast.dump(WithoutCallbacks().visit(ast.parse(b)),include_attributes=False)
            assert normalize(old)==normalize(new),path
        sources.append(dict(path=path,bytes_equal=exact,calculation_ast_equal=True))
    assert len(files)==69 and sum(f['has_counts'] for f in files)==61
    out.write_text(json.dumps(dict(videos=69,count_videos=61,all_saved_bytes_unchanged=True,reanalysis_exact=True,files=files,sources=sources),ensure_ascii=False,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();audit(a.source,a.out)
