"""Frozen 61-video labels, team9 counts, and supplied separate-clip events."""
from pathlib import Path
import csv,importlib.util,json,sys
import numpy as np
import av
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from core.io.rows import read_rows
from core.signal.height import height_signal
from core.signal.final import clean_final
from core.judge.session import run_session
from eval.debug_track import diagnose
BASE=Path('/home/kojy1009/work/m0-baseline/plate_pipeline/eval_out/baseline_mp/raw')
HOME=Path('/home/kojy1009/capstone_exercise');DATA=HOME/'new_data/new_dataset';TEAM=HOME/'plate_pipeline/data/team/test9'
OUT=ROOT/'results/mc';TMP=ROOT/'eval_out/mc';OUT.mkdir(parents=True,exist_ok=True)
V40=HOME/'new_data/experiments_athlete_pipeline_v40/athlete_yolo_balanced_phase_sameconditions_seed42/decode.py'
spec=importlib.util.spec_from_file_location('v40decode',V40);v40=importlib.util.module_from_spec(spec);spec.loader.exec_module(v40)

def matching(gt,pred,tol=.5):
    n,m=len(gt),len(pred);S=[[None]*(m+1) for _ in range(n+1)];S[0][0]=(0,0.,[])
    def better(a,b):return a if b is None or (a[0],-a[1])>(b[0],-b[1]) else b
    for i in range(n+1):
     for j in range(m+1):
      if i==j==0:continue
      best=None
      if i:best=S[i-1][j]
      if j:best=better(S[i][j-1],best)
      if i and j and abs(gt[i-1]-pred[j-1])<=tol:
       a=S[i-1][j-1];best=better((a[0]+1,a[1]+abs(pred[j-1]-gt[i-1]),a[2]+[(i-1,j-1)]),best)
      S[i][j]=best
    pairs=S[n][m][2];a={i for i,_ in pairs};b={j for _,j in pairs}
    assert (len(pairs),m-len(pairs),n-len(pairs))==v40.match_events(gt,pred,tol)[:3]
    return pairs,[i for i in range(n) if i not in a],[j for j in range(m) if j not in b]


def read_analysis(path):
    result=json.loads((path/'result.json').read_text());rows=read_rows(path/'rows.jsonl.gz')
    return result,rows


def get_clocks(rows,result):
 t,y,x=height_signal(rows,result.get('plate_mm',450.));fr=clean_final(t,y,x,result['fps']);sets=run_session(fr.sig.t,fr.sig.y,fr.sig.x,fr.sig.gap_before,result['exercise'])
 trips=[{'kind':tr.kind,'flags':tr.flags,'entry':tr.entry_t,'far':tr.far_t,'end':tr.end_t,'amp_cm':100*tr.amp,'counted':tr.counted} for st in sets for tr in st.trips]
 return t,y,fr.sig,trips


def entry(key,kind,video,result_path,truth,truth_count,origin=0,scope='inside',scope_reason=''):
 result,rows=read_analysis(result_path);diag=diagnose(rows,result)
 predicted=[v['back']-origin for s in result['sets'] for v in s['rep_values'] if v['back'] is not None]
 pairs,fn,fp=matching(truth,predicted) if truth is not None else ([],[],[])
 _,_,_,trips=get_clocks(rows,result)
 return {'key':key,'dataset':kind,'exercise':result['exercise'],'video':str(video),'result_path':str(result_path),'scope':scope,'scope_reason':scope_reason,
         'gt':truth,'gt_count':truth_count,'pred':predicted,'tp':None if truth is None else len(pairs),'fn':None if truth is None else [truth[i] for i in fn],
         'fp':None if truth is None else [predicted[i] for i in fp],'pairs':pairs,'origin':origin,
         'count_error':len(predicted)-truth_count,'tracked':diag['used_ratio'],'source_share':diag['src_share'],
         'moving_candidates':diag['moving_cands']['n'],'jumps':len(diag['jumps']),'trips':trips}


def main():
 labels={r['key']:r for r in csv.DictReader((ROOT/'results/midterm/gt_clock.csv').open())}
 tags={r['key']:r for r in csv.DictReader((ROOT/'results/midterm/view_tags.csv').open())}
 records=list(csv.DictReader((ROOT/'results/midterm/per_video.csv').open()));items=[]
 # Tags are a bounding-box heuristic; explicit known edits/captions override it.
 outside={'benchpress/stu8_7.mp4':'자막으로 원판이 가려짐',
  'benchpress/stu8_15.mp4':'무게 자막·화면 전환으로 이어 붙임',
  'benchpress/bench press_40.mp4':'영상 상단 문자 오버레이',
  'benchpress/stu7_12_2.mp4':'화면 아래 자막',
  'squat/stu6_11_2.mp4':'무게 자막', 'squat/stu6_11_3.mp4':'무게 자막',
  'deadlift/deadlift_30.mp4':'영상 위 텍스트/편집',
  'deadlift/deadlift_sihoo02.mp4':'앱 화면 녹화 및 화면 자막'}
 for q in records:
  key=q['key'];tag=tags[key];ratio=float(tag['plate_ratio']) if tag['plate_ratio'] else None
  reason=outside.get(key,'')
  if ratio is None or ratio<.5:reason=reason or '상자 비율 기준 60° 초과/미측정'
  origin=float(labels[key]['clock_origin_s']);gt=json.loads(labels[key]['gt_events'])
  a=entry(key,'midterm',DATA/key,BASE/key/'plate',gt,int(q['gt_count']),origin,'outside' if reason else 'inside',reason or 'view_tags 0.5 이상; 편집 미확인')
  assert (len(a['pred']),a['tp'],len(a['fp']),len(a['fn']))==(int(q['pred_count']),int(q['tp']),int(q['fp']),int(q['fn'])),key
  items.append(a)
 teams=list(csv.DictReader((ROOT/'eval/lists/team9.csv').open()))
 v40_records={r['key']:r for r in json.loads((V40.parent/'records.json').read_text())}
 for q in teams:
  p=next(TEAM.rglob(q['path']));rel=str(p.relative_to(TEAM));key='team9/'+rel;rowpath=BASE/key/'plate'
  if not rowpath.exists():rowpath=BASE/rel/'plate'
  gt=json.loads(labels[rel]['gt_events']) if rel in labels else None
  origin=float(labels[rel]['clock_origin_s']) if rel in labels else 0
  # A complete v40 frame-label sequence is useful only when it agrees with
  # the subsequently reviewed team9 count. Do not invent missing event times.
  record=v40_records.get(rel)
  if gt is None and record is not None and len(record['reps'])==int(q['reps']):
   with av.open(str(p)) as container:
    timestamps=[float(frame.time) for frame in container.decode(video=0) if frame.time is not None]
   origin=timestamps[0]
   assert all(0 <= rep[2] < len(timestamps) for rep in record['reps']),key
   gt=[timestamps[rep[2]]-origin for rep in record['reps']]
  reason={'stu8_7.mp4':'자막 가림','stu3_64.mp4':'큰 자막·편집','stu4_2_-1.mp4':'정면·자막','stu9_10.mp4':'아이콘·인터넷 편집','stu2_10_edited.mp4':'편집'}.get(q['path'],'')
  items.append(entry(key,'team9',p,rowpath,gt,int(q['reps']),origin,'outside' if reason else 'inside',reason or 'side-view review pending'))
 offsets=[0,10.917,23.375,37.667];truths=[[10.],[11.21],[6.27,13.73],[14.75,22.21]]
 for i,gt in enumerate(truths,1):items.append(entry(f'user_clip_{i}','user_clip',TMP/'inputs'/f'user_clip_{i}.mp4',TMP/'baseline_user'/f'user_clip_{i}',gt,len(gt),0,'inside','분리한 옆 영상'))
 gt=[t+off for off,rs in zip(offsets,truths) for t in rs]
 concat=Path('/home/kojy1009/work/m3a-result/plate_pipeline/eval_out/m3a/concatenated/analysis/user/plate')
 items.append(entry('squat_user_set','edited',Path('/home/kojy1009/Downloads/squat_user_set.mp4'),concat,gt,6,0,'outside','이어 붙인 영상'))
 (OUT/'baseline.json').write_text(json.dumps(items,ensure_ascii=False,indent=2))
 print('total',len(items),'midterm',sum(x['dataset']=='midterm' for x in items),'team9 GT events',sum(x['gt'] is not None for x in items if x['dataset']=='team9'))
 for x in items:
  if x['fn'] or x['fp'] or (x['gt'] is None and x['count_error']):print(x['key'],'scope',x['scope'],'gt',x['gt_count'],'pred',len(x['pred']),'FN',x['fn'],'FP',x['fp'],'tracked',round(x['tracked'],2))
if __name__=='__main__':main()
