"""Summarize the frozen cohort and simulation results."""
from pathlib import Path
import csv,gzip,importlib.util,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from eval.m4.cohort import BASE,M2,OUT,TMP
r=json.loads((OUT/'cohort.json').read_text()); by={x['key']:x for x in r};sim=json.loads((OUT/'simulation.json').read_text())
old=list(csv.DictReader((ROOT/'results/midterm/per_video.csv').open()));gt={x['key']:x for x in csv.DictReader((ROOT/'results/midterm/gt_clock.csv').open())};m2=json.loads(M2.read_text())['videos']
v40=Path('/home/kojy1009/capstone_exercise/new_data/experiments_athlete_pipeline_v40/athlete_yolo_balanced_phase_sameconditions_seed42/decode.py');spec=importlib.util.spec_from_file_location('old_decode',v40);dec=importlib.util.module_from_spec(spec);spec.loader.exec_module(dec)
assert len(r)==69 and len(old)==61
other=Path('/home/kojy1009/work/m2-3d/plate_pipeline/eval_out/m2/selected2d/raw')
byte_equal=sum(gzip.decompress((BASE/x['key']/'plate/rows.jsonl.gz').read_bytes())==gzip.decompress((other/x['key']/'plate/rows.jsonl.gz').read_bytes()) for x in r)
assert all(by[x['key']]['box_reps']==int(x['pred_count']) for x in old)

def metrics(mode):
 errs=[]; tp=fp=fn=0
 for x in old:
  q=by[x['key']];g=gt[x['key']];a=json.loads(g['gt_events']);origin=float(g['clock_origin_s']);pred=[t-origin for t in q[mode+'_events']]
  t,p,n,_=dec.match_events(a,pred);tp+=t;fp+=p;fn+=n;errs.append(abs(len(a)-len(pred)))
 return sum(errs),tp,fp,fn,2*tp/(2*tp+fp+fn)

bad_indices=[1,3,4,5,9,10,15,16,22,23,26,27,28,29,30,31,32,33,34,36,37,38,40,41,42,43,44,45,48,49,50,51,52,55,57,58,59,60,61,62,65,66,69]
bad={r[i-1]['key'] for i in bad_indices}
(OUT/'manual_review.json').write_text(json.dumps([{'key':x['key'],'needs_review':x['key'] in bad,'note':'세 표본 중 원판 외곽과 불일치 또는 안쪽/장애물 경계 의심' if x['key'] in bad else '세 표본에서 명백한 잘못된 타원 미확인; box는 측정 실패 대체'} for x in r],ensure_ascii=False,indent=2))

def fmt(v,n=2):return '—' if v is None else f'{v:.{n}f}'
def quartiles(values):
 if not values:return '—'
 a=np.percentile(values,[25,50,75]);return '/'.join(f'{v:.2f}' for v in a)+f' (n={len(values)})'
L=['# M4 원판 타원 및 데드리프트 표시 결과','', '브랜치 `integ/m4-ellipse`, 기준 `integ/m3c-ui` (`becb8a4`). 기존 영상·모델·data·eval_out은 포함하지 않았다. 타원 결과는 검증에 실패하여 실험적이다. **기본값은 `scale_source=box`, `position_source=box`를 유지한다.**','',
'## 구현과 한계','', '상자를 30% 확대한 영역의 Canny 가장자리와 contour/고정 난수 RANSAC 후보를 사용한다. 중심은 상자 중심에서 정규화 거리 0.35 이내, 장축은 상자의 큰 변의 0.7~1.4배, 축비는 0.2 이상이어야 한다. 화면 경계에 닿는 상자나 추정 타원은 거절한다. 타원 둘레 96점 중 62% 및 24구역 중 70% 이상이 2px 안의 가장자리와 맞아야 한다. 0.25초 간격 관측 사이 두 성공 측정만 보간하고, 실패 구간은 box로 되돌린다. 성공 품질은 둘레 지지율에서 중심 이탈 페널티를 뺀 값이다. 크기와 2배 각도를 0.25초 중앙값으로 다듬는다. 이것은 가림·고정 장치·글자가 많은 실제 영상에서 충분히 강건하지 않다. 각도와 다시 찍기 안내 역시 잘못된 테두리에서 오판할 수 있다.','',
'타원 모드는 장·단축의 역수와 회전을 합친 대칭 행렬을 사용하고, 프레임 간 행렬 평균으로 이동량을 적분한다. 3 cm 이상이며 주축 분산이 부축의 2배 이상인 경로는 PCA를 위쪽 방향으로 쓰고 화면 위쪽으로 부호를 정한다. 짧거나 방향이 모호하면 화면 위쪽을 쓴다. 타원 중점 대신 상자 중점을 쓰는 것이 기본이다. 상자 모드의 원판 기록은 기존 함수를 그대로 사용한다. 선택 모드는 `python -m server.compare ... --scale-source ellipse --position-source box|ellipse`이다. `result_v3.json`에 프레임별 타원·품질·출처, geo 타원 통계·angle_source·retake_warn을 넣는다. 기존 기준 캐시는 타원 설정 시 다시 분석한다.','',
'데드리프트 표시 전후: `captures/deadlift_before_1920x1080.png`, `captures/deadlift_after_1920x1080.png`, `captures/deadlift_after_charts_1920x1080.png`. 3D와 영상 간 간격은 12px이다. 범례는 올림→정점 멈춤→내림이고 높이는 위쪽으로 증가한다. 수치는 바꾸지 않았다.','',
'## (a) 합성 투영','', '기존 실제 바 경로를 angle_sim과 같은 15개 yaw/pitch/roll 조합으로 투영했다. 각 프레임에 바 끝, 잡음과 일부 가림을 넣고 일부 표본은 화면 밖으로 이동시켰다. 합성 mp4는 로컬 eval_out에서만 생성하며 결과는 `simulation.json`에 남겼다. 각 칸은 rep별 최대 ROM 상대오차 %. A/B는 투영 상자, D는 실제 검출 및 실패 시 상자 대체를 통과한 결과다.','', '| yaw | pitch | roll | A | B | D | 타원 사용률 |','|---:|---:|---:|---:|---:|---:|---:|']
for z in sim['conditions']:L.append(f"| {z['yaw']} | {z['pitch']} | {z['roll']} | {fmt(z['rom_error_pct']['A'])} | {fmt(z['rom_error_pct']['B'])} | {fmt(z['rom_error_pct']['D'])} | {z['ellipse_frame_fraction']:.1%} |")
L+=['',f"D 최대 {max(z['rom_error_pct']['D'] for z in sim['conditions']):.2f}%: 1.1% 기준 미통과.",'','## (b) 69개 영상','',f"유효 프레임 중 타원 사용률의 영상별 중앙값 {np.median([x['ellipse_success'] for x in r]):.1%}. 타원 사용률에는 성공한 두 표본 사이 보간 프레임이 포함된다. 69개 기존 M0/M2 원판 행의 gzip 해제 후 내용은 {byte_equal}/69 바이트 일치했고, 기존 61개 완료 횟수는 box 재계산에서 61/61 일치했다.",'', '| 방법 | 횟수 절대오차 합 | TP | FP | FN | 이벤트 F1 |','|---|---:|---:|---:|---:|---:|']
for mode in ('box','ellipse','ellipse_center'):
 a,b,c,d,e=metrics(mode);L.append(f'| {mode} | {a} | {b} | {c} | {d} | {e:.3f} |')
L+=['','이벤트는 기존 0.5초 매칭과 영상 첫 PTS 보정을 사용한다. ellipse는 상자 중심, ellipse_center는 타원 중심이다. 오차 합은 나빠졌고 합성 목표도 넘었으므로 box/box를 유지한다. 61개 중 완료 횟수가 바뀐 영상은 다음과 같다. 원인은 확인된 신호 변화이며 개별 물체 오검출 원인은 세 표본만으로 확정하지 않았다.','', '| 영상 | 정답 | box | 타원·box 중심 | 타원·타원 중심 | 관찰 |','|---|---:|---:|---:|---:|---|']
for x in old:
 q=by[x['key']]
 if q['box_reps']!=q['ellipse_reps'] or q['box_reps']!=q['ellipse_center_reps']:
  L.append(f"| {x['key']} | {x['gt_count']} | {q['box_reps']} | {q['ellipse_reps']} | {q['ellipse_center_reps']} | 타원/상자 전환과 위치 변화로 신호·이벤트 변동 |")
L+=['','### 종목별 ROM과 속도 분포','', '모든 산출 rep의 25/50/75 백분위수다. 횟수가 다른 경우 모집단도 달라진다.','', '| 종목 | box ROM cm | 타원 ROM cm | box 속도 m/s | 타원 속도 m/s |','|---|---|---|---|---|']
for ex in ('squat','benchpress','deadlift'):
 group=[x for x in r if x['exercise']==ex]
 vals=lambda k:[v for x in group for v in x[k] if v is not None]
 L.append(f"| {ex} | {quartiles(vals('box_rom_cm'))} | {quartiles(vals('ellipse_rom_cm'))} | {quartiles(vals('box_velocity'))} | {quartiles(vals('ellipse_velocity'))} |")
L+=['','### 내부 키 점검','']
for field in ('stature_box_cm','stature_ellipse_cm'):
 vals=[x[field] for x in r if x[field] is not None];L.append(f"- {field}: 150~200cm {sum(150<=v<=200 for v in vals)}/{len(vals)}, 중앙값 {np.median(vals):.1f}cm. 동일 M2 선택 포즈를 사용했다.")
L+=['','| 같은 사람 클립 | box cm | 타원 cm |','|---|---:|---:|']
for name in ('squat_01','squat_02','squat_03','deadlift_01','deadlift_02','deadlift_03'):
 x=next(q for q in r if q['key'].endswith('/'+name+'.mp4'));L.append(f"| {name} | {fmt(x['stature_box_cm'],1)} | {fmt(x['stature_ellipse_cm'],1)} |")
L+=['','### 영상별 측정','', '| 영상 | 성공률 | 장축/상자 세로 | 상자 각도 | 타원 각도 |','|---|---:|---:|---:|---:|']
for x in r:L.append(f"| {x['key']} | {x['ellipse_success']:.1%} | {fmt(x['major_to_box_h'])} | {fmt(x['box_angle'],1)} | {fmt(x['ellipse_angle'],1)} |")
L+=['','### M2의 가로 > 세로 26개','', 'M2의 원본 통계에서 키 관측이 있고 상자 비가 1보다 큰 26개를 그대로 선택했다.','', '| 영상 | M2 상자 비 | 타원 사용률 | 장축/상자 세로 | 타원 각도 |','|---|---:|---:|---:|---:|']
for x in m2:
 g=x['geometry']
 if g['ratio']>1 and g['stature_cm'] is not None:
  z=by[x['key']];L.append(f"| {x['key']} | {g['ratio']:.3f} | {z['ellipse_success']:.1%} | {fmt(z['major_to_box_h'])} | {fmt(z['ellipse_angle'],1)} |")
L+=['','## (c) 확인 이미지','', '69개 각각 사용 프레임의 앞·중간·뒤 세 프레임을 `overlays/`에 넣었고, 원판 부근 확대 `review_sheets/` 9장을 직접 열어 확인했다. 측정이 거절된 프레임은 파랑 상자만 보인다. 한 프레임만 used인 영상은 같은 프레임을 반복한다. 다음은 세 표본 중 명백한 불일치 또는 안쪽 테두리·바 끝과 혼동이 의심된 영상이며, 세 표본 밖의 프레임까지 검증한 목록은 아니다.','']
L.extend('- `'+x['key']+'`' for x in r if x['key'] in bad)
L+=['','## (d) 동시 촬영','', '같은 세트를 0°·45°에서 동시에 촬영한 V 영상은 찾지 못했다. 대기.','', '## 처리 속도와 시험','', f"69개 저장 원판 행과 원본 영상 재생·타원 검출·보간을 4개 작업으로 처리했다. 작업별 경과 합계 {sum(x['elapsed_s'] for x in r):.1f}초, 영상 중앙값 {np.median([x['elapsed_s'] for x in r]):.2f}초다. 이 시간은 기존 검출/포즈를 제외한 추가 재생 단계이며 병렬 작업별 시간이다. 6스레드 전체 실행 시간은 `benchmark.json`을 참조한다.", '', '합성 테두리·잡음·잘림, 대칭 변환, 주축·짧은 경로, box 대체, 60° 임계와 스키마 시험을 포함했다. 전체 시험: 파이프라인 188 passed (Starlette 의존성 경고 1개), 루트 feedback 235 passed. Playwright에서 경고 문구와 콘솔 오류 0개도 확인했다.','', f"6스레드 `stu10_70` 단일 영상 전체 실행: box {json.loads((OUT/'benchmark.json').read_text())['box']['total_s']:.2f}초, ellipse {json.loads((OUT/'benchmark.json').read_text())['ellipse']['total_s']:.2f}초 (각 1회, 두 방식 모두 타원 측정을 수행하므로 이 차이는 순수 오버헤드가 아니다). 같은 456프레임 영상의 추가 타원 재생 3회는 {', '.join(f'{v:.2f}' for v in json.loads((OUT/'benchmark.json').read_text())['ellipse_replay_seconds'])}초. 원판 box 행은 M3c에서 같은 영상으로 생성한 행과 gzip 해제 후 SHA-256 `58a00cc104ab7938846bdde6b4158aa10fe7ab0c7e82f70eb1b44535a38a37e3`으로 일치했다.", '', '## 재현','', '```bash','cd plate_pipeline','python eval/m4/cohort.py','python eval/m4/simulation.py','python eval/m4/review.py','python eval/m4/report.py','python -m pytest -q tests eval/midterm/test_metrics.py','```','이 스크립트는 M0/M2 원본 행과 포즈 및 영상의 로컬 경로를 필요로 한다.']
(OUT/'REPORT.md').write_text('\n'.join(L)+'\n')
print('metrics',[metrics(x) for x in ('box','ellipse','ellipse_center')])
