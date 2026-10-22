# 0단계 재현 검증

결과: 통과.

v40 holdout.py의 effective_split_hash는 시험 73개만이 아니라 전체 462개 (key, split)를 정렬해 data.digest로 해시한 값이다. 이 원래 정의를 적용했다. 분할을 새로 만들지 않았다.

- 전체 분할 해시: `c91ed670f0851ca78f5358224b0084f6ce9f2a0310dcb3099943db196ab68ca0`
- 기준 해시: `c91ed670f0851ca78f5358224b0084f6ce9f2a0310dcb3099943db196ab68ca0`
- 시험 73개만의 별도 해시: `01e48a6962a134e00f76b21bd855382e51ba1c8a8f9d4ba7f8c35ca1078c0e1b` (위 전체 분할 해시와 정의가 다름)
- test 키는 records.json 및 B_generic all_test_per_video.csv와 정확히 일치. 로컬 new_labels.csv의 원본 parse_labels 결과도 records의 완료 rep 삼중 라벨과 73개 모두 일치.
- B_generic의 reference/custom/all 영상별 CSV를 train.summarize의 원본 집계 AST 식으로 재계산했다. 모델·추론·분할 생성 없이 횟수와 TP/FP/FN/event F1 집계를 검증했다. 원시 예측을 다시 디코딩한 검증은 아니다. saved per-video CSV에는 개별 delay가 없으므로 이 CSV 집계에는 delay를 넣지 않았다. 추가 clock_gate에서 원본 프레임 시계와 저장 count_event로 B 세 묶음의 delay 중앙값·p95까지 재현했다 (v40_clock_baseline_summary.csv). run.py는 원판 추론 전에 이 검사도 통과해야 진행한다.

세부 수치: reproduction_checks.csv.
