"""Compare the confirmed/weak pick with Mc's enabled endpoint fix."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[2]
base=json.loads((ROOT/'results/mc/baseline.json').read_text())
mc={x['key']:x for x in json.loads((ROOT/'results/mc/variants.json').read_text())}
new={x['key']:x for x in json.loads((ROOT/'eval_out/mc2/confirmed/evaluation.json').read_text())}
# The weak-only rule has no eligible candidate in this evaluation cohort.
rows=[]
for b in base:
 m=mc[b['key']];n=new[b['key']]
 rows.append({'key':b['key'],'dataset':b['dataset'],'exercise':b['exercise'],'scope':b['scope'],'gt_count':b['gt_count'],
              'labeled':b['gt'] is not None,'old':m['on'],'new':{k:n[k] for k in ('pred','tp','fp','fn')},
              'selection_changed':n['selection_changed'],'chosen':n['chosen'],'legacy':n['legacy'],'unchanged_rows_byte_equal':not n['selection_changed'] and n['rows_equal']})
unique=[r for r in rows if r['dataset']!='edited' and r['key']!='team9/benchpress/stu8_7.mp4']
groups={'전체':unique,'조건 안':[r for r in unique if r['scope']=='inside'],'조건 밖':[r for r in unique if r['scope']=='outside']}
for ds in ('midterm','team9','user_clip','edited'):groups[ds]=[r for r in rows if r['dataset']==ds]
def summary(v,mode):
 v_label=[r for r in v if r['labeled']];tp=sum(r[mode]['tp'] for r in v_label);fp=sum(len(r[mode]['fp']) for r in v_label);fn=sum(len(r[mode]['fn']) for r in v_label)
 return {'n':len(v),'event_labeled':len(v_label),'exact':sum(len(r[mode]['pred'])==r['gt_count'] for r in v),'tp':tp,'fp':fp,'fn':fn,'p':tp/(tp+fp) if tp+fp else None,'r':tp/(tp+fn) if tp+fn else None}
summary_data={k:{m:summary(v,m) for m in ('old','new')} for k,v in groups.items()}
out=ROOT/'results/mc2';out.mkdir(parents=True,exist_ok=True)
(out/'evaluation.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
(out/'summary.json').write_text(json.dumps(summary_data,ensure_ascii=False,indent=2))
for k,s in summary_data.items():
 print(k,end=' ')
 for mode in ('old','new'):
  a=s[mode];print(f"{a['exact']}/{a['n']} {a['tp']}/{a['fp']}/{a['fn']} P={a['p']:.3f} R={a['r']:.3f}",end='  ')
 print()
