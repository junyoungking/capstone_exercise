"""Build a concise, reproducible M4b report from the fixed M2 cohort."""
from pathlib import Path
import csv,gzip,importlib.util,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from core.geometry import plate_suggestion,PLATE_SUGGEST_FRACTION
from eval.m4b.cohort import BASE,OUT
r=json.loads((OUT/'cohort.json').read_text());m={x['key']:x for x in r};old=list(csv.DictReader((ROOT/'results/midterm/per_video.csv').open()));gt={x['key']:x for x in csv.DictReader((ROOT/'results/midterm/gt_clock.csv').open())};assert len(r)==69 and len(old)==61
v40=Path('/home/kojy1009/capstone_exercise/new_data/experiments_athlete_pipeline_v40/athlete_yolo_balanced_phase_sameconditions_seed42/decode.py');spec=importlib.util.spec_from_file_location('v40decode',v40);dec=importlib.util.module_from_spec(spec);spec.loader.exec_module(dec)

def metrics(mode):
 error=tp=fp=fn=0
 for x in old:
  q=m[x['key']];g=gt[x['key']];truth=json.loads(g['gt_events']);origin=float(g['clock_origin_s']);pred=[v-origin for v in q['stats'][mode]['events']]
  error+=abs(len(truth)-len(pred));a,b,c,_=dec.match_events(truth,pred);tp+=a;fp+=b;fn+=c
 return error,tp,fp,fn,2*tp/(2*tp+fp+fn)
def fmt(x,d=1):return '—' if x is None else f'{x:.{d}f}'
def quartile(vals):
 if not vals:return '—'
 return '/'.join(f'{a:.1f}' for a in np.percentile(vals,[25,50,75]))+f' (n={len(vals)})'
def collect(group,mode,key):return [v for x in group for v in x['stats'][mode][key]]
controls=[x for x in r if x['group']!='GB_LLK_PB'];small=[x for x in r if x['group']=='GB_LLK_PB']
check=[x for x in controls if x['implied_mm'] is not None];at=lambda x,t:plate_suggestion(x['implied_mm'],450,t)
other=Path('/home/kojy1009/work/m2-3d/plate_pipeline/eval_out/m2/selected2d/raw')
byte_equal=sum(gzip.decompress((BASE/x['key']/'plate/rows.jsonl.gz').read_bytes())==gzip.decompress((other/x['key']/'plate/rows.jsonl.gz').read_bytes()) for x in r)
assert all(m[x['key']]['stats']['before']['count']==int(x['pred_count']) for x in old)
timing=json.loads((OUT/'rescale_timing.json').read_text());cos=[(abs(x['cos_one_mm']-x['cos_box_ratio_mm']),x['key']) for x in r if not x['angle_reliable'] and x['cos_one_mm'] is not None and x['cos_box_ratio_mm'] is not None]
L=['# M4b 원판 지름 확인과 각도 표시','', '브랜치 `integ/m4b-plate-size`, 기준 `integ/m4-ellipse` (`8f1767f`). 지름은 영상에서 직접 측정한 정답이 아니라 키·폰 화각·바와 사람 사이 거리 가정에서 역산한 제안이다. 실측 지름으로 확인해야 한다.','',
'## 공식과 제안 기준','', 'M2에서 H = (코–바닥 px / 0.92) × (D / 원판 상자 세로 px + 0.70 × cos(각도) / 초점 px)이므로 D = 원판 상자 세로 px × (0.92 × H / 코–바닥 px − 0.70 × cos(각도) / 초점 px)다. 단위는 m. 선택 리프터의 서 있는 프레임 중 원판 상자가 관측되고 화면 밖으로 잘리지 않은 프레임만 사용해 프레임별 역산의 중앙값을 쓴다. 해당 프레임이 없거나 양의 해가 없으면 추정하지 않는다. 내 영상은 입력 키가 있으면 그 키, 없으면 170 cm; 기준 영상은 항상 170 cm다. 범위는 각각 160·180 cm를 넣은 값이다. 상자 비가 0.95 이상 또는 한 프레임이라도 화면 끝에 닿으면 각도는 측정 어려움이며 cos=1로 둔다.','',
'**제안 기준은 사용 지름과 추정 지름의 차이가 사용 지름의 30%를 초과할 때다.** GB/LLK/PB 30개를 작은 원판 후보, 나머지를 450 mm로 보이는 대조군으로 취급한 검증에서 추정 가능한 대조군 23개 중 잘못 제안 0개였다. 25%에서는 대조군 3개가 제안되어 30%를 택했다. 작은 원판 후보 30개 중 21개를 제안하고 9개를 놓쳤다. 이는 영상별 실측 지름이 없는 대리 분류 결과이며 실제 위양성·위음성의 확정 수가 아니다.','',
'| 차이 기준 | 대조군 잘못 제안 / 추정 가능 | 작은 원판 후보 제안 / 30 |','|---:|---:|---:|']
for threshold in (.20,.25,.30,.35,.40):L.append(f'| {threshold:.0%} | {sum(at(x,threshold) for x in check)} / {len(check)} | {sum(at(x,threshold) for x in small)} / {len(small)} |')
L+=['',f"각도를 믿을 수 없는 추정 가능 {len(cos)}개에서 cos=1과 상자 비를 그대로 cos로 쓴 역산 지름의 절대 차이는 중앙값 {np.median([x[0] for x in cos]):.1f} mm, 최대 {max(cos)[0]:.1f} mm ({max(cos)[1]})다. 이는 두 가정의 차이이지 정확도 정답이 아니다.",'',
'## 저장 결과 재계산','', 'POST `/api/sessions/{id}/rescale`의 `plate_mm_user`, `plate_mm_ref`는 각각 선택적 양수이며 하나 이상 필요하다. 저장된 원판 rows와 2D pose를 읽어 신호, 상태기계, rep, 문장·판정 및 HTML을 다시 만든다. 기존 3D 좌표·분절을 재사용하며 검출·포즈·3D 모델은 실행하지 않는다. 두 지름을 독립 저장한다. 이전 JSON과 HTML은 `data/sessions/<id>/history/<version>/`에 보관하고 새 결과로 세션 및 기록의 횟수·최고 속도를 갱신한다. 완료 rep이 0개면 0회 화면을 표시하며 지름을 다시 바꿀 수 있다. 처리 실패 시 기존 결과는 유지한다.','',
 f"실제 `GB_S01_R07` 단일 영상으로 450↔250 mm 네 번 요청했을 때 서버 전체 경과 {min(x['elapsed_s'] for x in timing):.3f}~{max(x['elapsed_s'] for x in timing):.3f}초, 계산 부분 {min(x['compute_s'] for x in timing):.3f}~{max(x['compute_s'] for x in timing):.3f}초였다. 각 요청의 기록은 `rescale_timing.json`에 있다. 1920×1080 실제 흐름에서 제안 버튼, 다시 계산 후 지름, 각도 측정 어려움과 콘솔 오류 0개를 확인했다.",'',
'캡처: `01_suggestion_and_angle.png`(450 mm·제안·각도 측정 어려움), `02_rescaled.png`(250 mm 재계산). `inspection_01.jpg`~`inspection_05.jpg`는 각각 PB 두 개, GB 한 개, LLK 두 개다. 파랑 상자는 원판 검출, 노랑은 상자 세로 픽셀, 연두는 2D 무릎–발목 길이이며 사람/원판의 실측 길이는 아니다. 직접 보면 원판이 정강이와 비슷하거나 작아 보이나 원근·자세와 가림의 영향을 받는다.','',
'## 69개 영상 평가','', '| 그룹 | 영상 | 지름 추정 가능 | 추정 지름 mm Q1/중앙/Q3 | 제안 |','|---|---:|---:|---|---:|']
for name in ('GB_LLK_PB','other_deadlift','squat','team9','benchpress'):
 g=[x for x in r if x['group']==name];vals=[x['implied_mm'] for x in g if x['implied_mm'] is not None];L.append(f'| {name} | {len(g)} | {len(vals)} | {quartile(vals)} | {sum(x["suggested"] for x in g)} |')
L+=['','### 지름 변경 전후 ROM·속도','', '제안된 영상만 추정값을 10 mm 단위로 반올림해 상태기계를 다시 돌렸다. 모든 rep의 Q1/중앙/Q3; 속도 단위 m/s. rep 집합이 달라지면 분포의 모집단도 달라진다.','', '| 그룹 | ROM 전 cm | ROM 후 cm | 속도 전 m/s | 속도 후 m/s |','|---|---|---|---|---|']
for name in ('GB_LLK_PB','other_deadlift','squat','team9','benchpress'):
 g=[x for x in r if x['group']==name];L.append(f'| {name} | {quartile(collect(g,"before","rom_cm"))} | {quartile(collect(g,"after","rom_cm"))} | {quartile(collect(g,"before","velocity"))} | {quartile(collect(g,"after","velocity"))} |')
other_dead=[x for x in r if x['exercise']=='deadlift' and x['group']!='GB_LLK_PB'];L+=['',f"작은 원판 후보의 ROM 중앙값은 {np.median(collect(small,'before','rom_cm')):.1f}→{np.median(collect(small,'after','rom_cm')):.1f} cm이고 나머지 데드 10개의 중앙값 {np.median(collect(other_dead,'before','rom_cm')):.1f} cm에 가까워졌다. 사람·동작이 달라 정상 범위나 실제 오차를 확정하지는 못한다.",'','### 원판으로 잰 키','']
for field in ('stature_before_cm','stature_after_cm'):
 vals=[x[field] for x in r if x[field] is not None];L.append(f'- {field}: 150~200 cm {sum(150<=x<=200 for x in vals)}/{len(vals)}, Q1/중앙/Q3 {quartile(vals)} cm.')
L+=['','제안된 영상은 키 170 cm 가정으로 지름을 역산했으므로 재계산한 키가 170 cm에 가까워지는 것은 계산상 당연하다. 이 키 분포를 독립적인 정확도 개선 증거로 쓰지 않는다.','', '### 61개 완료 횟수와 이벤트','', '| 설정 | 절대 횟수오차 합 | TP | FP | FN | event F1 |','|---|---:|---:|---:|---:|---:|']
for mode in ('before','after'):
 a,b,c,d,e=metrics(mode);L.append(f'| {mode} | {a} | {b} | {c} | {d} | {e:.3f} |')
changes=[m[x['key']] for x in old if m[x['key']]['stats']['before']['count']!=m[x['key']]['stats']['after']['count']]
L+=['','완료 횟수가 바뀐 영상: '+('없음. 재계산해도 이 61개에서는 0.13 m 상태기계 경계를 넘는 변화가 없었다.' if not changes else ', '.join(x['key'] for x in changes)),f'기본 상자 원판 행은 M0/M2 {byte_equal}/69 영상에서 gzip 해제 후 바이트가 같았고, 61개 재계산 횟수는 기준과 61/61 같았다. 실제 재처리한 GB 영상의 행도 기준과 바이트가 같았다.','',
'### 키 가정에 따라 제안이 바뀐 영상','', '| 영상 | 160 cm | 170 cm | 180 cm |','|---|---|---|---|']
for x in r:
 options=[plate_suggestion(x['by_height'][str(H)],450) for H in (160,170,180)]
 if len(set(options))>1:L.append(f'| {x["key"]} | {"제안" if options[0] else "없음"} | {"제안" if options[1] else "없음"} | {"제안" if options[2] else "없음"} |')
L+=['','### 영상별 지름과 제안','', '| 영상 | 추정 mm | 160–180 cm 범위 mm | 제안 | 적용 mm |','|---|---:|---:|---|---:|']
for x in r:
 low,high=x['range_mm'];L.append(f'| {x["key"]} | {fmt(x["implied_mm"])} | {fmt(low)}–{fmt(high)} | {"예" if x["suggested"] else "아니오"} | {x["chosen_mm"]} |')
L+=['','## 시험과 한계','', '- `cd plate_pipeline && python -m pytest -q tests eval/midterm/test_metrics.py`: 192 passed, Starlette 의존성 경고 1개. 루트 `python -m pytest -q tests/fb`: 235 passed. 합성 역산 왕복, 30% 경계, 잘림·각도, API 재계산/0회/이전 결과 보존 및 독립 지름을 포함한다.', '- 실제 원판 지름과 키의 측정 정답이 없어 제안 정확도와 ROM의 절대 정확도는 검증되지 않았다. 170 cm 가정, 코–바닥 관측, 바 깊이·초점, 리프터 선택 및 화면 잘림은 모두 오차 요인이다. 위 대조군 분류도 영상상 추정이다.', '- 완료 분석의 원판 행 또는 pose 파일이 없는 오래된 세션은 재계산할 수 없다. 독립 HTML은 제안을 보여 주지만 서버가 없으므로 버튼으로 저장 재계산할 수 없다.', '', '재현: `python eval/m4b/cohort.py`, `python eval/m4b/inspection.py`, `python eval/m4b/report.py` (M0/M2의 로컬 영상·원판 행·포즈가 필요). 결과 JSON과 꼭 필요한 캡처만 커밋하며 영상과 zip은 포함하지 않는다.']
(OUT/'REPORT.md').write_text('\n'.join(L)+'\n');print('metrics',metrics('before'),metrics('after'),'byte_equal',byte_equal)
