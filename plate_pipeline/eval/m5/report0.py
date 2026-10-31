from pathlib import Path
import gzip,json,sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from eval.mc2b.evaluate import raw_path
from core.io.rows import read_rows
base={r['key']:r for r in json.loads((ROOT/'results/mc2b/evaluation.json').read_text())};new=json.loads((ROOT/'eval_out/m5/eval0.json').read_text());rows=[]
source={r['key']:r for r in json.loads((ROOT/'results/mc/baseline.json').read_text())}
for n in new:
 p=raw_path(source[n['key']]['result_path'])/'rows.jsonl.gz'
 old=read_rows(p);assert gzip.decompress(p.read_bytes())==''.join(json.dumps(r,separators=(',',':'))+'\n' for r in old).encode()
 b=base[n['key']];rows.append({**n,'mc':b['mc'],'mc2b':b['mc2b'],'m5':{k:n[k] for k in ('pred','tp','fp','fn')}})
unique=[r for r in rows if r['dataset']!='edited' and r['key']!='team9/benchpress/stu8_7.mp4']
groups={'전체':unique,'조건 안':[r for r in unique if r['scope']=='inside'],'조건 밖':[r for r in unique if r['scope']=='outside']}
for d in ('midterm','team9','user_clip','edited'):groups[d]=[r for r in rows if r['dataset']==d]
def aggregate(group,mode):
 lbl=[r for r in group if r['labeled']];tp=sum(r[mode]['tp'] for r in lbl);fp=sum(len(r[mode]['fp']) for r in lbl);fn=sum(len(r[mode]['fn']) for r in lbl)
 return {'n':len(group),'labeled':len(lbl),'exact':sum(len(r[mode]['pred'])==r['gt_count'] for r in group),'tp':tp,'fp':fp,'fn':fn,'p':tp/(tp+fp) if tp+fp else None,'r':tp/(tp+fn) if tp+fn else None}
summary={k:{m:aggregate(v,m) for m in ('mc','mc2b','m5')} for k,v in groups.items()}
(ROOT/'results/m5/plate_evaluation.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2));(ROOT/'results/m5/plate_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
for k,v in summary.items():
 print(k,end=' ')
 for m,d in v.items():print(m,f"{d['exact']}/{d['n']}",f"{d['tp']}/{d['fp']}/{d['fn']}",f"P={d['p']:.3f} R={d['r']:.3f}",end='; ')
 print()
print('count regressions',[(r['key'],r['gt_count'],len(r['m5']['pred'])) for r in unique if r['scope']=='inside' and len(r['mc']['pred'])==r['gt_count'] and len(r['m5']['pred'])!=r['gt_count']])
print('same initial byte equal',sum(r['start_same'] and r['rows_equal'] for r in rows),'attempts',sum(r['attempts']>0 for r in rows))
