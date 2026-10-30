"""Produce auditable event classifications and grouped metrics from frozen data."""
import json
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
base=json.loads((ROOT/'results/mc/baseline.json').read_text())
variants=json.loads((ROOT/'results/mc/variants.json').read_text())
bykey={v['key']:v for v in variants}
# Initial diagnosis; ambiguous timestamp/track cases are explicitly marked as hypotheses.
why={
 'squat/val1235.mp4':('바','첫 동작이 영상 시작부터 진행 중이며 라벨에는 두 번째 완료만 있음; 경계/라벨 확인 필요'),
 'benchpress/stu10_11.mp4':('가','정면 촬영에서 추적률 5%; 원판 신호 부재'),
 'benchpress/stu8_15.mp4':('사','자막과 화면 전환; 편집 전후 추적 단절'),
 'benchpress/stu8_7.mp4':('가','자막이 원판을 가리고 초반 추적 누락'),
 'benchpress/test153.mp4':('가','정면 원판 추적률 6%'),
 'deadlift/deadlift_7.mp4':('아','실측 완료와 라벨이 0.530초 차이; 완료 시각 재확인 필요'),
 'deadlift/deadlift_23.mp4':('가','정면 원판 추적 61%, 신호 누락'),
 'squat/stu6_11_2.mp4':('나','움직이지 않는 보관 원판이 선택됨'),
 'squat/stu6_11_3.mp4':('나','움직이지 않는 보관 원판이 선택됨'),
 'squat/val1250.mp4':('가','비스듬/정면에서 추적률 41%, 원판 신호 누락'),
 'deadlift/deadlift_sihoo02.mp4':('아','화면 녹화; 완료 시각과 라벨 0.628초 차이'),
 'deadlift/GB_S07_R20.mp4':('가','하강 후반 추적 공백을 선형 보간, 완료 시각 조기 추정'),
 'deadlift/LLK_S06_R32.mp4':('바','초반 상승이 보이지 않아 첫 동작 경계 또는 초기 추적 확인 필요'),
 'deadlift/deadlift_02.mp4':('아','실측 바닥 접촉과 라벨이 0.608초 차이; 시점 재확인'),
 'deadlift/deadlift_03.mp4':('아','바닥 접촉과 라벨이 각각 0.62~0.82초 차이; 시점 재확인'),
 'benchpress/bench_01.mp4':('마','상태기계 뒤 완료점 측정이 라벨보다 빠르거나 늦음; 피크/완료 기준 검토 필요'),
 'squat/squat_01.mp4':('가','큰 추적 공백 후 완료점 계산 실패'),
 'squat/squat_03.mp4':('가','하강 구간 추적 공백과 완료점 계산 실패'),
 'team9/squat/stu10_70.mp4':('아','3번째 실제 복귀 약 11.7초, 프레임 라벨 12.96초; 라벨 검토 필요'),
 'team9/benchpress/stu8_7.mp4':('가','중간 평가의 동일 영상과 SHA-256 일치; 자막 가림'),
 'team9/benchpress/stu9_10.mp4':('마','완료점 조기 검출/누락 또는 추적 공백; 개별 시점 확인 필요'),
 'team9/deadlift/stu7_1_1.mp4':('아','2번째 바닥 복귀와 라벨 차이 0.66초; 완료 기준 검토 필요'),
 'team9/deadlift/stu4_2_-1.mp4':('가','정면·자막 영상에서 원판 추적 자체가 시작되지 않음'),
 'team9/deadlift/deadlift_08.mp4':('아','2번째 바닥 복귀와 라벨 차이 1.03초; 완료 기준 검토 필요'),
 'user_clip_2':('마','상태기계 복귀 확정 전에 첫 국소 최고점으로 완료 시각을 선택'),
 'user_clip_3':('가','초반 올바른 원판 후보가 낮은 점수이고 약 4.6초까지 추적 시작 실패'),
 'squat_user_set':('사','편집된 연결 지점의 추적 공백 및 클립 2 조기 완료점'),
}
fail=[];counts=Counter()
for b in base:
 if b['gt'] is None:
  fail.append({'key':b['key'],'scope':b['scope'],'type':'미분류','reason':'정답 종료 시각 미완성: 총횟수 차이만 확인 가능','gt_count':b['gt_count'],'pred_count':len(b['pred'])})
  continue
 for direction in ('fn','fp'):
  for t in b[direction]:
   if b['key'] not in why: raise ValueError((b['key'],direction,t))
   typ,reason=why[b['key']];rec={'key':b['key'],'scope':b['scope'],'direction':direction.upper(),'time_s':t,'type':typ,'reason':reason}
   fail.append(rec)
   if b['dataset']!='edited' and b['key']!='team9/benchpress/stu8_7.mp4': counts[(b['scope'],typ,direction)]+=1
(ROOT/'results/mc/failures.json').write_text(json.dumps(fail,ensure_ascii=False,indent=2))

def agg(items,mode):
 n=sum(x['gt_count'] for x in items);ex=sum(len(x[mode]['pred'])==x['gt_count'] for x in items)
 labeled=[x for x in items if x['gt'] is not None]
 tp=sum(x[mode]['tp'] for x in labeled);fp=sum(len(x[mode]['fp']) for x in labeled);fn=sum(len(x[mode]['fn']) for x in labeled)
 return {'videos':len(items),'exact_count':ex,'gt_count':n,'pred_count':sum(len(x[mode]['pred']) for x in items),'event_labeled_videos':len(labeled),'tp':tp,'fp':fp,'fn':fn,'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn) if tp+fn else None,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None}
# The duplicate video is counted once overall; the edited concatenation is separate.
unique=[v for v in variants if v['dataset']!='edited' and v['key']!='team9/benchpress/stu8_7.mp4']
groups={'전체(중복 제거)':unique,'조건 안':[x for x in unique if x['scope']=='inside'],'조건 밖':[x for x in unique if x['scope']=='outside']}
for ex in ('squat','bench','deadlift'):groups[ex]=[x for x in unique if x['exercise']==ex]
groups['1회 영상']=[x for x in unique if x['gt_count']==1]
groups['여러 rep']=[x for x in unique if x['gt_count']>1]
for ds in ('midterm','team9','user_clip','edited'):groups[ds]=[x for x in variants if x['dataset']==ds]
summ={k:{m:agg(v,m) for m in ('off','on')} for k,v in groups.items()}
(ROOT/'results/mc/summary.json').write_text(json.dumps(summ,ensure_ascii=False,indent=2))
print('FAILURE COUNTS', sorted((s,t,d,n) for (s,t,d),n in counts.items()))
for k,v in summ.items():
 def fmt(d):return f"{d['videos']} ({d['event_labeled_videos']} labeled); exact {d['exact_count']}; {d['tp']}/{d['fp']}/{d['fn']}; P/R/F1 {d['precision']:.3f}/{d['recall']:.3f}/{d['f1']:.3f}"
 print(k,fmt(v['off']),'->',fmt(v['on']))
