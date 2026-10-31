"""Re-evaluate saved real pose observations and aggregate all automatic evidence."""
from pathlib import Path
import json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT.parent))
from core.io.rows import read_rows
from core.auto.judge import assess
from eval.mc2b.evaluate import raw_path

base=json.loads((ROOT/'results/mc/baseline.json').read_text());plate={r['key']:r for r in json.loads((ROOT/'eval_out/m5/eval0.json').read_text())}
ref=Path('/home/kojy1009/capstone_exercise/new_data/experiments_athlete_pipeline_v40/athlete_yolo_balanced_phase_sameconditions_seed42/records.json')
records={r['key']:r for r in json.loads(ref.read_text())}
partial={}
for name in ('auto_tail.json','auto_evaluation.json'):
 p=ROOT/'eval_out/m5'/name
 if p.exists():partial.update({r['key']:r for r in json.loads(p.read_text())})
out=[]
for b in base:
 if b['dataset']=='edited' or b['key']=='team9/benchpress/stu8_7.mp4':continue
 r=partial.get(b['key']);rec=dict(key=b['key'],dataset=b['dataset'],scope=b['scope'],truth='benchpress' if b['exercise']=='bench' else b['exercise'])
 match=b['key'].removeprefix('team9/')
 if b['key'].startswith('user_clip_'):match='squat/squat_0'+str(int(b['key'][-1])+1)+'.mp4'
 rec['v40_split']=records.get(match,{}).get('split','unknown')
 if not r:rec.update(status='not_evaluated',error='포즈 추론 미완료')
 elif r.get('error'):rec.update(status='error',error=r['error'])
 else:
  m=plate[b['key']];rp=(raw_path(b['result_path'])/'rows.jsonl.gz') if m['start_same'] else ROOT/'eval_out/m5/plate_rows'/b['dataset']/(b['key'].replace('/','__')+'.jsonl.gz')
  rows=read_rows(rp)
  with np.load(r['pose_path'],allow_pickle=False) as f:pose={k:f[k] for k in f.files}
  for k in ('fps','w','h','total'):pose[k]=pose[k].item()
  decision=assess(rows,pose);rec.update(status='evaluated',judgment=decision,pose_valid=r.get('pose_valid'),frames=r.get('frames'),selection_review=r.get('selection_review'))
 out.append(rec)

def group(rows):
 conf_ok=conf_bad=need=rest_ok=rest_bad=rest_unknown=body_ok=body_bad=body_unknown=floor_ok=floor_bad=floor_unknown=0
 for r in rows:
  j=r.get('judgment') or {};expected='bottom' if r['truth']=='deadlift' else 'top';body='lying' if r['truth']=='benchpress' else 'standing'
  if j.get('status')=='confirmed':
   if j.get('exercise')==r['truth']:conf_ok+=1
   else:conf_bad+=1
  else:need+=1
  for name,truth in [('rest',expected),('posture',body),('floor',expected)]:
   value=j.get(name,'unknown');match= value==truth;unknown=value=='unknown'
   if name=='rest':rest_ok+=match;rest_unknown+=unknown;rest_bad+=not(match or unknown)
   elif name=='posture':body_ok+=match;body_unknown+=unknown;body_bad+=not(match or unknown)
   else:floor_ok+=match;floor_unknown+=unknown;floor_bad+=not(match or unknown)
 return {'n':len(rows),'evaluated':sum(r['status']=='evaluated' for r in rows),'confirmed_correct':conf_ok,'confirmed_wrong':conf_bad,'needs_confirmation':need,
         'rest_correct':rest_ok,'rest_wrong':rest_bad,'rest_unknown':rest_unknown,'body_correct':body_ok,'body_wrong':body_bad,'body_unknown':body_unknown,
         'floor_correct':floor_ok,'floor_wrong':floor_bad,'floor_unknown':floor_unknown,
         'classifier_votes_available':sum(r.get('judgment',{}).get('classifier',{}).get('status')=='ok' for r in rows)}
groups={'all':out,'inside':[r for r in out if r['scope']=='inside'],'outside':[r for r in out if r['scope']=='outside']}
for ex in ('squat','benchpress','deadlift'):groups[ex]=[r for r in out if r['truth']==ex]
for split in ('train','val','test','unknown'):groups[split]=[r for r in out if r['v40_split']==split]
summary={name:group(items) for name,items in groups.items()}
(ROOT/'results/m5/auto_evaluation.json').write_text(json.dumps(out,ensure_ascii=False,indent=2));(ROOT/'results/m5/auto_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
for name,v in summary.items():print(name,v)
print('wrong rest',[r['key'] for r in out if r.get('judgment',{}).get('rest') not in (None,'unknown','bottom' if r['truth']=='deadlift' else 'top')])
print('selection differences',[(r['key'],r['selection_review']) for r in out if (r.get('selection_review') or {}).get('center_diff_gt_0_1')])
