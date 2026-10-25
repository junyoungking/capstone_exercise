# 중간 발표 원판 횟수 평가 보고서

브랜치: `integ/midterm-eval`; 기준: `ae31956`. 기존 S4 작업 트리와 루트의 준비 파일은 보존했다. PLAN/core/feedback/server/model은 수정하지 않았다.

## 0. 재현 확인

reproduction.md와 reproduction_checks.csv: 원래 v40 전체 분할 해시 c91ed670… 및 B 세 묶음 횟수·이벤트 집계 일치. test 73개 키와 로컬 라벨의 완료 삼중 프레임도 records와 일치. train.summarize의 원본 AST 집계식을 직접 불러 쓰며, 매칭은 원본 decode.match_events다.
v40_clock_baseline_check.csv: 로컬 영상의 원본 PTS에서 영상 첫 PTS를 뺀 v40 기준 시계로 GT를 복원해 B의 저장 count_event와 다시 매칭했다. 73개 모두 TP/FP/FN이 원본과 일치. delay 재집계와 원본은 v40_clock_baseline_summary.csv에서 대조하며, count/event 및 delay 중앙값·p95를 1e-9 오차 안에서 대조해 통과해야 첨부를 만든다. 처음에는 PyAV 절대 PTS를 그대로 써서 B delay 중앙값이 −0.015132초(원본 −0.004671초)로 달랐고, 첫 PTS를 뺀 공통 변환으로 원본을 재현했다.

## 1. 이벤트·지표 정의

v40 data.make_pack은 L1/L2/L3 완료 삼중 라벨의 L3 원본 프레임 timestamp를 gt_events로 쓴다. 같은 로컬 영상의 PyAV 디코드 프레임 PTS(반올림 전)로 대응하고, GT와 예측 모두 첫 PTS를 뺐다. 이는 원본 v40 tracking_compare의 0시작 OpenCV 시계와 맞추는 영상 공통 변환이며 GT별 정렬이 아니다. 원판 원본 rows/result의 back은 원래 PTS 그대로 보존했다. 원판 rep_values.back은 쉬는 자리로 복귀한 완료 시각으로 선택했다. 내부 횟수는 증가했으나 back이 공란인 trip은 완료 이벤트에서 제외하고 별도 열로 남겼다. 숫자를 맞추기 위해 정답에 이벤트를 정렬하거나 옮기지 않았다.
count_mae/exact/obo는 영상별 오차의 평균; OBO는 |오차|≤1. event F1은 TP/FP/FN을 합친 micro 집계이며 tolerance=0.5초, 최대 매칭 수·최소 절대 시각 오차의 순서 매칭이다. delay는 pred−GT의 중앙값과 |delay|의 p95다. 오프라인 최종 back 시각의 오차이며 온라인 처리·통지 지연을 뜻하지 않는다.

## 2. 실행 조건

v40 원본 test 73개는 고정하고 명시적 원판 없음 12개만 제외한 61개를 평가한다. reference_test 52, custom_test 9, all_test 61이다. 다운로드·새 분할 없음. plate_tiny.onnx, device=auto, 실제 provider=CPUExecutionProvider. 라벨 종목을 그대로 사용(benchpress→bench). 포즈는 끄고 core 임계·선택·추적·세트 규칙은 기준 코드 그대로 사용했다. 평가 도구에서 ORT 실행 스레드만 세션당 1개, OpenCV 1개로 제한하고 영상 4개를 병렬 처리했다.

각도는 used인 det/det_low/rim 측정 상자의 가로/세로 비 중앙값이다. KLT의 추정 크기는 넣지 않았다. ≥0.75 옆, ≥0.4 비스듬, 그 아래 정면, 측정 없음은 없음. 물체 오검출이면 이 태그도 틀릴 수 있어 촬영 방향 정답으로 해석하지 않는다. 추적률은 used/전체 디코드 프레임이다.
개발용 team9와 파일명 일치: 1/61개. 제외 집계는 plate_summary.csv의 without_development이며, 0개라면 원래 값과 같다.

## 3. 결과

주 비교는 동일 61개 A/B/C/원판의 조건부 성능이다. 원판이 없는 영상까지 포함한 전체 시스템 성능이라고 주장하지 않는다. summary.md와 comparison.csv가 주 비교이며, 기존 73개 A/B/C는 baseline73_abc.csv 및 baseline73_abc_by_exercise.csv에 그대로 보존했다. event F1은 영상별 평균이 아니라 합계 TP/FP/FN으로 계산했다.
종목별 동일 집합 비교는 comparison_by_exercise.csv, 개발 영상 stu8_7을 추가로 제외한 60개 비교는 comparison_without_development.csv다. 종목 인식 A 0.863, B·C 0.973은 원본 73개 v40 수치이며 61개 집계나 원판 종목 인식 성능이 아니다.
제외 규칙이 추가되기 전 이미 실행한 12개 원판 산출물은 로컬 중간 파일로만 보존했다. 최종 원판 지표·실패 목록·공유 ZIP에서는 제외하며 0회 예측으로 넣지 않았다. 아래 제외 목록 외에 검출이 안 된 영상을 추가로 빼지 않았다.

### 원판 없음 명시적 제외 12개

| 영상 | 제외 사유 |
|---|---|
| benchpress/bench press_8.mp4 | 원판 없음 (사용자 지정) |
| benchpress/bench press_12.mp4 | 원판 없음 (사용자 지정) |
| benchpress/bench press_43.mp4 | 원판 없음 (사용자 지정) |
| benchpress/stu10_9.mp4 | 원판 없음 (사용자 지정) |
| benchpress/test138.mp4 | 원판 없음 (사용자 지정) |
| squat/stu2_67.mp4 | 원판 없음 (사용자 지정) |
| squat/stu2_68.mp4 | 원판 없음 (사용자 지정) |
| squat/stu8_65.mp4 | 원판 없음 (사용자 지정) |
| squat/stu9_66_2.mp4 | 원판 없음 (사용자 지정) |
| squat/stu9_66_3.mp4 | 원판 없음 (사용자 지정) |
| squat/stu9_66_4.mp4 | 원판 없음 (사용자 지정) |
| squat/stu10_68.mp4 | 원판 없음 (사용자 지정) |

### 동일 61개 중 추가 검출 실패


원판 측정 프레임이 완전히 0인 추가 영상은 없다. 몇 프레임이라도 검출된 것이 올바른 원판 identity/지속 추적/횟수를 보장하지는 않는다.

추적률 60% 미만인 영상(진단 표시만 하며 제외하지 않음):
- benchpress/stu10_11.mp4: 추적 4.6%, 완료 0/20, 태그 정면.
- benchpress/test153.mp4: 추적 6.3%, 완료 0/5, 태그 정면.
- squat/val1250.mp4: 추적 40.8%, 완료 0/3, 태그 비스듬.

### 원판 조건부 결과


### 원본 73개 A/B/C (보존, 주 비교 아님)

| 방법 | n | MAE | exact | OBO | event F1 |
|---|---:|---:|---:|---:|---:|
| A_existing | 73 | 1.397 | 0.370 | 0.808 | 0.665 |
| B_generic | 73 | 1.151 | 0.466 | 0.877 | 0.742 |
| C_athlete | 73 | 1.616 | 0.219 | 0.822 | 0.626 |

### 동일 61개 주 비교

| 묶음 | 방법 | n | MAE | exact | OBO | event F1 | TP/FP/FN | delay median/p95(s) |
|---|---|---:|---:|---:|---:|---:|---|---|
| reference_test | A_existing | 52 | 1.462 | 0.346 | 0.808 | 0.300 | 21/11/87 | -0.0083/0.3372 |
| custom_test | A_existing | 9 | 0.778 | 0.444 | 0.889 | 0.471 | 4/1/8 | -0.1875/0.3979 |
| all_test | A_existing | 61 | 1.361 | 0.361 | 0.820 | 0.318 | 25/12/95 | -0.0333/0.4008 |
| reference_test | B_generic | 52 | 1.173 | 0.442 | 0.865 | 0.516 | 40/7/68 | -0.0417/0.4471 |
| custom_test | B_generic | 9 | 0.444 | 0.556 | 1.000 | 0.455 | 5/5/7 | -0.1250/0.3667 |
| all_test | B_generic | 61 | 1.066 | 0.459 | 0.885 | 0.508 | 45/12/75 | -0.0417/0.4404 |
| reference_test | C_athlete | 52 | 1.615 | 0.154 | 0.827 | 0.288 | 19/5/89 | 0.0833/0.3955 |
| custom_test | C_athlete | 9 | 0.556 | 0.444 | 1.000 | 0.211 | 2/5/10 | 0.0833/0.0833 |
| all_test | C_athlete | 61 | 1.459 | 0.197 | 0.852 | 0.278 | 21/10/99 | 0.0833/0.3917 |
| reference_test | plate | 52 | 0.904 | 0.808 | 0.885 | 0.690 | 59/4/49 | -0.0572/0.3415 |
| custom_test | plate | 9 | 0.222 | 0.778 | 1.000 | 0.455 | 5/5/7 | -0.0401/0.3149 |
| all_test | plate | 61 | 0.803 | 0.803 | 0.902 | 0.663 | 64/9/56 | -0.0566/0.3484 |

### 원판 세부 집계

| 묶음 | scope | n | MAE | exact | OBO | event F1 | TP/FP/FN | delay median/p95(s) |
|---|---|---:|---:|---:|---:|---:|---|---|
| reference_test | all | 52 | 0.904 | 0.808 | 0.885 | 0.690 | 59/4/49 | -0.0572/0.3415 |
| reference_test | without_development | 51 | 0.863 | 0.824 | 0.902 | 0.695 | 57/4/46 | -0.0577/0.3495 |
| reference_test | squat | 7 | 1.286 | 0.429 | 0.714 | 0.571 | 6/1/8 | -0.2437/0.3969 |
| reference_test | benchpress | 11 | 3.273 | 0.636 | 0.636 | 0.561 | 23/0/36 | -0.0446/0.1794 |
| reference_test | deadlift | 34 | 0.059 | 0.941 | 1.000 | 0.882 | 30/3/5 | -0.0566/0.3358 |
| custom_test | all | 9 | 0.222 | 0.778 | 1.000 | 0.455 | 5/5/7 | -0.0401/0.3149 |
| custom_test | without_development | 9 | 0.222 | 0.778 | 1.000 | 0.455 | 5/5/7 | -0.0401/0.3149 |
| custom_test | squat | 3 | 0.667 | 0.333 | 1.000 | 0.500 | 1/0/2 | -0.1705/0.1705 |
| custom_test | benchpress | 3 | 0.000 | 1.000 | 1.000 | 0.600 | 3/2/2 | -0.0401/0.3200 |
| custom_test | deadlift | 3 | 0.000 | 1.000 | 1.000 | 0.250 | 1/3/3 | 0.0566/0.0566 |
| all_test | all | 61 | 0.803 | 0.803 | 0.902 | 0.663 | 64/9/56 | -0.0566/0.3484 |
| all_test | without_development | 60 | 0.767 | 0.817 | 0.917 | 0.667 | 62/9/53 | -0.0575/0.3502 |
| all_test | squat | 10 | 1.100 | 0.400 | 0.800 | 0.560 | 7/1/10 | -0.2322/0.3927 |
| all_test | benchpress | 14 | 2.571 | 0.714 | 0.714 | 0.565 | 26/2/38 | -0.0424/0.2093 |
| all_test | deadlift | 37 | 0.054 | 0.946 | 1.000 | 0.816 | 31/6/8 | -0.0561/0.3272 |

## 4. 횟수가 틀린 영상별 단서

아래 원인은 자동 관측으로 분류한 진단 단서이며 확정 원인이 아니다. 모든 영상의 timeline은 ZIP videos/ 아래에 있다.

| 영상 | 원인·단서 |
|---|---|
| squat/val1235.mp4 | 정답보다 많은 왕복을 rep으로 포함(과계수); 추가 왕복의 실제 동작/검출 identity는 미확인; 추적 100.0%, 높이 범위 41.8 cm, 완료 2/1. 자동 진단 단서이며 물체 정체/분할 오류의 수작업 확정은 안 함 |
| benchpress/stu10_11.mp4 | 추적 프레임 부족으로 유효 왕복 rep을 생성하지 못함; 추적 4.6%, 높이 범위 42.7 cm, 완료 0/20. 자동 진단 단서이며 물체 정체/분할 오류의 수작업 확정은 안 함 |
| benchpress/stu8_15.mp4 | 내부 trip 5개 중 종료 back 공란 1개를 완료 이벤트에서 제외; 추적 75.4%, 높이 범위 52.9 cm, 완료 4/12. 자동 진단 단서이며 물체 정체/분할 오류의 수작업 확정은 안 함 |
| benchpress/stu8_7.mp4 | 내부 trip 4개 중 종료 back 공란 2개를 완료 이벤트에서 제외; 추적 76.2%, 높이 범위 57.7 cm, 완료 2/5. 자동 진단 단서이며 물체 정체/분할 오류의 수작업 확정은 안 함 |
| benchpress/test153.mp4 | 추적 프레임 부족으로 유효 왕복 rep을 생성하지 못함; 추적 6.3%, 높이 범위 6.0 cm, 완료 0/5. 자동 진단 단서이며 물체 정체/분할 오류의 수작업 확정은 안 함 |
| deadlift/deadlift_23.mp4 | 정면·비스듬 조건에서 신호/가림 확인 필요; 추적 61.1%, 높이 범위 43.4 cm, 완료 0/1. 자동 진단 단서이며 물체 정체/분할 오류의 수작업 확정은 안 함 |
| squat/stu6_11_2.mp4 | 관측 원판 높이 변화가 1.0 cm로 최소 rep 진폭 13 cm보다 작아 rep 미생성; 추적 100.0%, 높이 범위 1.0 cm, 완료 0/4. 자동 진단 단서이며 물체 정체/분할 오류의 수작업 확정은 안 함 |
| squat/stu6_11_3.mp4 | 관측 원판 높이 변화가 1.0 cm로 최소 rep 진폭 13 cm보다 작아 rep 미생성; 추적 100.0%, 높이 범위 1.0 cm, 완료 0/1. 자동 진단 단서이며 물체 정체/분할 오류의 수작업 확정은 안 함 |
| squat/val1250.mp4 | 추적 프레임 부족으로 유효 왕복 rep을 생성하지 못함; 추적 40.8%, 높이 범위 20.9 cm, 완료 0/3. 자동 진단 단서이며 물체 정체/분할 오류의 수작업 확정은 안 함 |
| deadlift/LLK_S06_R32.mp4 | 검출 대상·세트/rep 분할 확인 필요; 추적 72.4%, 높이 범위 98.2 cm, 완료 0/1. 자동 진단 단서이며 물체 정체/분할 오류의 수작업 확정은 안 함 |
| squat/squat_01.mp4 | 내부 trip 1개 중 종료 back 공란 1개를 완료 이벤트에서 제외; 추적 81.3%, 높이 범위 21.3 cm, 완료 0/1. 자동 진단 단서이며 물체 정체/분할 오류의 수작업 확정은 안 함 |
| squat/squat_03.mp4 | 내부 trip 1개 중 종료 back 공란 1개를 완료 이벤트에서 제외; 추적 95.0%, 높이 범위 52.8 cm, 완료 0/1. 자동 진단 단서이며 물체 정체/분할 오류의 수작업 확정은 안 함 |

## 5. 검증·제한

루트 tests/fb: 235 passed; plate_pipeline/tests: 123 passed; eval/midterm/test_metrics.py: 2 passed. 입력 GT 시계·원본 집계·유한 표본을 확인한 시험이며 독립 촬영 데이터의 최종 성능 보장은 아니다.
Noto Sans KR 요청은 설치된 Noto Sans CJK KR 한국어 서브폰트를 사용했다. 세 발표 그림은 밝은 바탕 1600×900, 요청한 색상이다. 기존 v40 모델의 학습 데이터/원판 검출기 학습 데이터와의 추가 중복 감사는 하지 않았다.

## 재현 명령

```bash
export V40=/home/kojy1009/capstone_exercise/new_data/experiments_athlete_pipeline_v40/athlete_yolo_balanced_phase_sameconditions_seed42
python plate_pipeline/eval/midterm/run.py --local-root /home/kojy1009/capstone_exercise --model /home/kojy1009/capstone_exercise/plate_pipeline/models/plate_tiny.onnx --workers 4
python plate_pipeline/eval/midterm/present.py --v40 "$V40" --raw out/midterm
python plate_pipeline/eval/midterm/finish.py --v40 "$V40" --raw out/midterm
```
