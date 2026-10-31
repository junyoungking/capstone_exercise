"""Compare Mc, Mc2, and Mc2b on the fixed 73-video cohort."""
from pathlib import Path
import gzip,hashlib,json,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from eval.mc2b.evaluate import raw_path
from core.io.rows import read_rows
base=json.loads((ROOT/'results/mc/baseline.json').read_text())
mc2={r['key']:r for r in json.loads((ROOT/'results/mc2/evaluation.json').read_text())}
b={r['key']:r for r in json.loads((ROOT/'eval_out/mc2b/evaluation.json').read_text())}
rows=[]
for old in base:
 x=mc2[old['key']];y=b[old['key']]
 # Matching canonical row bytes means matching JSONL bytes when the saved
 # original was produced by the unchanged JSONL serializer. Check that premise.
 source=raw_path(old['result_path'])/'rows.jsonl.gz';original=gzip.decompress(source.read_bytes())
 decoded=read_rows(source);reencoded=''.join(json.dumps(r,separators=(',',':'))+'\n' for r in decoded).encode()
 assert original==reencoded,old['key']
 rows.append({'key':old['key'],'dataset':old['dataset'],'scope':old['scope'],'exercise':old['exercise'],'gt_count':old['gt_count'],'labeled':old['gt'] is not None,
              'mc':x['old'],'mc2':x['new'],'mc2b':{k:y[k] for k in ('pred','tp','fp','fn')},
              'selection_changed':y['selection_changed'],'selected_legacy':y['selected_legacy'],'legacy_motion_d':y['legacy_motion_d'],
              'legacy_high':y['legacy_high'],'selected_start':y['selected_start'],'rows_equal':y['rows_equal'],
              'original_jsonl_sha256':hashlib.sha256(original).hexdigest()})
unique=[r for r in rows if r['dataset']!='edited' and r['key']!='team9/benchpress/stu8_7.mp4']
groups={'전체':unique,'조건 안':[r for r in unique if r['scope']=='inside'],'조건 밖':[r for r in unique if r['scope']=='outside']}
for ds in ('midterm','team9','user_clip','edited'):groups[ds]=[r for r in rows if r['dataset']==ds]
def summ(items,mode):
 labeled=[r for r in items if r['labeled']];tp=sum(r[mode]['tp'] for r in labeled);fp=sum(len(r[mode]['fp']) for r in labeled);fn=sum(len(r[mode]['fn']) for r in labeled)
 return {'n':len(items),'labeled':len(labeled),'exact':sum(len(r[mode]['pred'])==r['gt_count'] for r in items),'tp':tp,'fp':fp,'fn':fn,
         'p':tp/(tp+fp) if tp+fp else None,'r':tp/(tp+fn) if tp+fn else None}
result={name:{mode:summ(group,mode) for mode in ('mc','mc2','mc2b')} for name,group in groups.items()}
out=ROOT/'results/mc2b';out.mkdir(parents=True,exist_ok=True)
(out/'evaluation.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2));(out/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
for name,s in result.items():
 print(name, end=' ')
 for mode,d in s.items():print(mode,f"{d['exact']}/{d['n']}",f"{d['tp']}/{d['fp']}/{d['fn']}",f"P={d['p']:.3f} R={d['r']:.3f}",end='; ')
 print()
print('changed',sum(r['selection_changed'] for r in rows),'unchanged equal',sum(not r['selection_changed'] and r['rows_equal'] for r in rows))
print('regressions',[(r['key'],r['gt_count'],len(r['mc']['pred']),len(r['mc2b']['pred'])) for r in rows if r['scope']=='inside' and len(r['mc']['pred'])==r['gt_count'] and len(r['mc2b']['pred'])!=r['gt_count']])
