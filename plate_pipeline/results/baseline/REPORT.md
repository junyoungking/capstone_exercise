# M0 기준선 결과

- 브랜치: `integ/m0-baseline`, 작업 트리: `/home/kojy1009/work/m0-baseline`.
- 기준 `plate-pipeline(4605cd5)`에서 생성하고 로컬 `integ/midterm-eval(67e7173)`을 병합했다. 병합 커밋 `decff40`, 충돌 없음. PLAN.md·main·원판/포즈 운영 코드는 수정하지 않았다.
- 횟수 61개: MAE **0.803278689**, exact **0.803278689**, OBO **0.901639344**, event F1 **0.663212435**. 기존 midterm의 영상별 예측 횟수·TP/FP/FN·exact·OBO·오차가 61개 전부 일치한다.
- 포즈는 61개 + team9 9개 − 중복 stu8_7 1개 = **69개**다. 중복 영상의 SHA256도 같음을 확인했다.
- rep 구간 포즈 선택률 **96.799%**, 자동 몸통 점프 **89회**, 홀수/짝수 차이 **0/43칸**. CPU 전체 처리 **10.021 fps**.
- 단일 전체 실행 1991.4초(약 33.19분), 분석 패스 합계 1929.0초, CFR 준비 19.1초. 아래 표의 처리 fps는 분석 패스 합계 기준이다.
- 확인 이미지 69장(각 6프레임)을 모두 검토해 **6개 영상**에서 잘못된 리프터/허위 골격을 확인했다. 전체 프레임에 대한 정답 주석은 아니며, 이미지 표본에서 확인한 오류 목록이다.

# M0 MediaPipe baseline

Base `4605cd5`; merged local `integ/midterm-eval` (`67e7173`) without conflicts. PLAN.md and main unchanged.

Command: `python -m eval.baseline --out eval_out/baseline_mp --resume`
Invocation wall time: 1991.4 s

CPU: AMD Ryzen 5 7500F 6-Core Processor
Versions: `{"onnxruntime": "1.31.0", "mediapipe": "1.0.1", "numpy": "2.5.3", "av": "19.0.2", "opencv-python": "5.0.0.93"}`

Midterm per-video reproduction: **True** (61/61 videos).

Counts use original midterm match_events (0.5 s) and train.summarize AST; completed back events only. F1 pools TP/FP/FN.

|scope|videos|MAE|exact|OBO|event F1|
|---|---:|---:|---:|---:|---:|
|all|61|0.803|0.803|0.902|0.663|
|squat|10|1.100|0.400|0.800|0.560|
|benchpress|14|2.571|0.714|0.714|0.565|
|deadlift|37|0.054|0.946|1.000|0.816|

## Pose observations

Denominator: unique decoded frames in completed plate rep intervals. Missing pose is failure. Joint visibility >=0.5 on raw landmarks; pooled joint x frame counts. Jump flag: consecutive-frame torso center displacement > current used plate height; a flag is not confirmed identity ground truth.

|scope|videos|pose/rep frames|coverage|jump flags|different/judged|rate|
|---|---:|---:|---:|---:|---:|---:|
|all|69|9466/9779|0.968|89|0/43|0.000|
|squat|13|2505/2520|0.994|0|0/25|0.000|
|benchpress|16|3042/3297|0.923|70|0/18|0.000|
|deadlift|40|3919/3962|0.989|19|0/0|N/A|

Odd/even uses native S3b split/reindex and display metrics, >=4 measurable completed reps. Only judged cells enter denominator. Insufficient/advisory/disabled cells stay unjudged. This within-video diagnostic is not independent false-positive validation.

|scope|shoulder|elbow|wrist|hip|knee|ankle|foot|
|---|---:|---:|---:|---:|---:|---:|---:|
|all|0.968|0.622|0.589|0.968|0.733|0.734|0.778|
|squat|0.994|0.631|0.493|0.994|0.883|0.871|0.901|
|benchpress|0.923|0.656|0.673|0.922|0.716|0.609|0.641|
|deadlift|0.989|0.587|0.580|0.989|0.651|0.751|0.813|

## Rep variation (degrees)

Per-video sample SD (ddof=1) and median absolute difference over unique unordered pairs; summary = median of video statistics per position/metric. Same S3a available-cell population including excluded reps; n<2 is N/A. Unchanged smoothing/side/raw visibility/screen gates/five-frame medians. Bench angles suppressed by existing structural occlusion remain N/A.

|scope|position|metric|videos with >=2|SD median|pair-difference median|
|---|---|---|---:|---:|---:|
|all|START|knee_angle|17|5.905|7.023|
|all|START|hip_angle|20|9.572|11.292|
|all|START|trunk_lean|20|2.952|3.224|
|all|MID_DOWN|knee_angle|14|2.840|4.154|
|all|MID_DOWN|hip_angle|17|5.520|7.182|
|all|MID_DOWN|trunk_lean|18|2.999|3.789|
|all|BOTTOM|knee_angle|14|4.714|4.928|
|all|BOTTOM|hip_angle|15|4.748|4.803|
|all|BOTTOM|trunk_lean|17|1.926|2.536|
|all|MID_UP|knee_angle|15|6.270|7.018|
|all|MID_UP|hip_angle|18|9.254|12.581|
|all|MID_UP|trunk_lean|19|3.290|4.653|
|all|LOCKOUT|knee_angle|15|8.451|9.621|
|all|LOCKOUT|hip_angle|19|9.179|11.106|
|all|LOCKOUT|trunk_lean|19|2.599|2.528|
|squat|START|knee_angle|6|0.761|0.789|
|squat|START|hip_angle|6|3.627|4.620|
|squat|START|trunk_lean|6|1.238|1.847|
|squat|MID_DOWN|knee_angle|5|3.178|4.472|
|squat|MID_DOWN|hip_angle|6|6.315|8.348|
|squat|MID_DOWN|trunk_lean|6|3.162|4.143|
|squat|BOTTOM|knee_angle|5|4.679|6.144|
|squat|BOTTOM|hip_angle|6|7.167|10.135|
|squat|BOTTOM|trunk_lean|6|0.994|1.405|
|squat|MID_UP|knee_angle|6|5.417|7.661|
|squat|MID_UP|hip_angle|6|6.256|8.848|
|squat|MID_UP|trunk_lean|6|1.935|2.684|
|squat|LOCKOUT|knee_angle|5|3.307|2.921|
|squat|LOCKOUT|hip_angle|6|4.222|4.692|
|squat|LOCKOUT|trunk_lean|6|0.457|0.605|
|benchpress|START|knee_angle|7|13.741|7.629|
|benchpress|START|hip_angle|10|9.572|11.292|
|benchpress|START|trunk_lean|10|2.249|2.976|
|benchpress|MID_DOWN|knee_angle|5|7.012|7.311|
|benchpress|MID_DOWN|hip_angle|7|6.133|7.182|
|benchpress|MID_DOWN|trunk_lean|8|4.290|6.217|
|benchpress|BOTTOM|knee_angle|5|12.879|4.659|
|benchpress|BOTTOM|hip_angle|5|6.678|9.444|
|benchpress|BOTTOM|trunk_lean|7|4.680|8.048|
|benchpress|MID_UP|knee_angle|5|9.130|6.955|
|benchpress|MID_UP|hip_angle|8|16.021|20.422|
|benchpress|MID_UP|trunk_lean|9|9.758|11.635|
|benchpress|LOCKOUT|knee_angle|6|12.140|8.762|
|benchpress|LOCKOUT|hip_angle|9|9.179|11.106|
|benchpress|LOCKOUT|trunk_lean|9|2.637|3.381|
|deadlift|START|knee_angle|4|7.359|10.238|
|deadlift|START|hip_angle|4|18.154|25.217|
|deadlift|START|trunk_lean|4|9.300|13.063|
|deadlift|MID_DOWN|knee_angle|4|2.174|2.930|
|deadlift|MID_DOWN|hip_angle|4|1.737|2.545|
|deadlift|MID_DOWN|trunk_lean|4|2.761|3.789|
|deadlift|BOTTOM|knee_angle|4|2.576|3.498|
|deadlift|BOTTOM|hip_angle|4|1.605|2.472|
|deadlift|BOTTOM|trunk_lean|4|1.368|1.951|
|deadlift|MID_UP|knee_angle|4|4.981|6.027|
|deadlift|MID_UP|hip_angle|4|6.633|8.875|
|deadlift|MID_UP|trunk_lean|4|3.911|4.436|
|deadlift|LOCKOUT|knee_angle|4|11.142|16.549|
|deadlift|LOCKOUT|hip_angle|4|20.129|25.616|
|deadlift|LOCKOUT|trunk_lean|4|28.155|39.036|

## CPU speed

Serial, CPUExecutionProvider, ORT/OpenCV one thread. Original PTS plate pass for counts, then current feedback prepared CFR plate/pose passes where conversion is required; otherwise two passes: plate PyAV then feedback OpenCV pose every frame (2 poses, .45 confidence, current EMA/selection). Decode includes both passes; KLT includes tracker association/bookkeeping. Rest includes setup/selection/smoothing/analysis/plate I/O. fps = input frames / combined pass time; excludes CFR preparation and metric/contact-sheet/GT validation. Per-video prepare time is stored separately. External pose replay has no measured pose fps.

|scope|decode s|detector s|KLT s|pose s|rest s|total s|fps|
|---|---:|---:|---:|---:|---:|---:|---:|
|all|26.836|1425.071|98.805|350.084|28.207|1929.003|10.021|
|squat|6.688|329.094|23.029|87.616|6.089|452.515|10.565|
|benchpress|10.787|590.803|36.034|149.413|8.388|795.424|10.621|
|deadlift|9.361|505.174|39.742|113.055|13.731|681.063|8.960|

## Per video / lifter review

Six raw skeleton frames evenly sampled over rep intervals; no completed rep gives a labeled whole-video diagnostic sheet. Sparse sheets cannot prove identity at every frame. Review and jump flags are separate.

|video|coverage|jumps|fps|review|image|
|---|---:|---:|---:|---|---|
|squat/squat_11.mp4|1.000|0|10.840|correct in six sampled frames|images/squat__squat_11.mp4.jpg|
|squat/stu4_69.mp4|1.000|0|11.584|correct in six sampled frames|images/squat__stu4_69.mp4.jpg|
|squat/val1235.mp4|1.000|0|12.320|correct in six sampled frames|images/squat__val1235.mp4.jpg|
|squat/val1256.mp4|0.989|0|12.651|correct in six sampled frames; joint geometry may be unreliable|images/squat__val1256.mp4.jpg|
|benchpress/bench press_31.mp4|0.945|0|11.243|correct in six sampled frames; joint geometry may be unreliable|images/benchpress__bench press_31.mp4.jpg|
|benchpress/bench press_40.mp4|0.770|0|11.244|correct in five visible sampled frames; one sampled pose missing|images/benchpress__bench press_40.mp4.jpg|
|benchpress/stu10_11.mp4|N/A|0|13.186|wrong: spotter selected at diagnostic f359/f719/f1078/f1438; no completed plate reps, excluded from rep-interval denominators|images/benchpress__stu10_11.mp4.jpg|
|benchpress/stu8_15.mp4|0.531|0|12.798|correct in five visible sampled frames; one sampled pose missing|images/benchpress__stu8_15.mp4.jpg|
|benchpress/stu8_7.mp4|0.882|0|12.724|correct in five visible sampled frames; one sampled pose missing|images/benchpress__stu8_7.mp4.jpg|
|benchpress/test153.mp4|N/A|0|12.939|wrong: spotter selected at diagnostic f0/f239/f299; no completed plate reps|images/benchpress__test153.mp4.jpg|
|deadlift/deadlift_7.mp4|1.000|0|8.963|correct in visible sampled frames|images/deadlift__deadlift_7.mp4.jpg|
|deadlift/deadlift_23.mp4|N/A|0|9.759|correct in visible sampled frames|images/deadlift__deadlift_23.mp4.jpg|
|deadlift/deadlift_30.mp4|1.000|0|11.508|correct in visible sampled frames|images/deadlift__deadlift_30.mp4.jpg|
|squat/stu6_11_2.mp4|N/A|0|12.349|wrong: background standing person selected at diagnostic f127/f159; no completed plate reps|images/squat__stu6_11_2.mp4.jpg|
|squat/stu6_11_3.mp4|N/A|0|11.548|wrong: spurious torso on stored yellow plate at diagnostic f0/f38; no completed plate reps|images/squat__stu6_11_3.mp4.jpg|
|squat/val1250.mp4|N/A|0|12.598|correct in visible sampled frames|images/squat__val1250.mp4.jpg|
|benchpress/bench press_61.mp4|0.978|0|11.690|correct in six sampled frames; joint geometry may be unreliable|images/benchpress__bench press_61.mp4.jpg|
|benchpress/stu1_6_4.mp4|0.990|0|11.966|correct in six sampled frames; joint geometry may be unreliable|images/benchpress__stu1_6_4.mp4.jpg|
|benchpress/stu7_12_2.mp4|0.981|8|12.131|correct in visible sampled frames; occluded joints may be unreliable|images/benchpress__stu7_12_2.mp4.jpg|
|benchpress/stu7_7.mp4|0.723|6|12.030|correct in visible sampled frames; occluded joints may be unreliable|images/benchpress__stu7_7.mp4.jpg|
|benchpress/stu8_6_2.mp4|1.000|4|12.154|correct in visible sampled frames; occluded joints may be unreliable|images/benchpress__stu8_6_2.mp4.jpg|
|deadlift/deadlift_sihoo02.mp4|0.900|0|12.075|correct in visible sampled frames; occluded joints may be unreliable|images/deadlift__deadlift_sihoo02.mp4.jpg|
|deadlift/PB_S01_R10.mp4|1.000|0|7.019|correct in visible sampled frames|images/deadlift__PB_S01_R10.mp4.jpg|
|deadlift/PB_S03_R13.mp4|0.976|0|11.865|correct in visible sampled frames|images/deadlift__PB_S03_R13.mp4.jpg|
|deadlift/PB_S03_R14.mp4|0.989|0|11.894|correct in visible sampled frames|images/deadlift__PB_S03_R14.mp4.jpg|
|deadlift/PB_S04_R03.mp4|1.000|0|11.981|correct in visible sampled frames|images/deadlift__PB_S04_R03.mp4.jpg|
|deadlift/PB_S05_R22.mp4|0.978|0|12.125|correct in visible sampled frames|images/deadlift__PB_S05_R22.mp4.jpg|
|deadlift/PB_S06_R25.mp4|1.000|0|12.190|correct in visible sampled frames|images/deadlift__PB_S06_R25.mp4.jpg|
|deadlift/PB_S07_R19.mp4|0.974|0|12.276|correct in visible sampled frames|images/deadlift__PB_S07_R19.mp4.jpg|
|deadlift/GB_S01_R01.mp4|0.983|0|12.131|correct in six sampled frames|images/deadlift__GB_S01_R01.mp4.jpg|
|deadlift/GB_S01_R07.mp4|1.000|0|12.320|correct in six sampled frames|images/deadlift__GB_S01_R07.mp4.jpg|
|deadlift/GB_S01_R11.mp4|1.000|0|12.287|correct in six sampled frames|images/deadlift__GB_S01_R11.mp4.jpg|
|deadlift/GB_S01_R16.mp4|1.000|0|12.093|correct in six sampled frames|images/deadlift__GB_S01_R16.mp4.jpg|
|deadlift/GB_S01_R18.mp4|1.000|2|12.125|correct in six sampled frames|images/deadlift__GB_S01_R18.mp4.jpg|
|deadlift/GB_S04_R03.mp4|1.000|10|12.192|wrong: background standing person at far right selected at rep frame f36|images/deadlift__GB_S04_R03.mp4.jpg|
|deadlift/GB_S04_R04.mp4|1.000|0|12.138|correct in six sampled frames|images/deadlift__GB_S04_R04.mp4.jpg|
|deadlift/GB_S04_R07.mp4|1.000|4|12.018|correct in six sampled frames|images/deadlift__GB_S04_R07.mp4.jpg|
|deadlift/GB_S05_R47.mp4|0.936|0|11.954|correct in visible sampled frames; joint geometry may be unreliable|images/deadlift__GB_S05_R47.mp4.jpg|
|deadlift/GB_S06_R63.mp4|0.989|0|12.061|correct in visible sampled frames; joint geometry may be unreliable|images/deadlift__GB_S06_R63.mp4.jpg|
|deadlift/GB_S07_R12.mp4|N/A|0|7.037|correct in visible sampled frames; joint geometry may be unreliable|images/deadlift__GB_S07_R12.mp4.jpg|
|deadlift/GB_S07_R14.mp4|N/A|0|6.954|correct in visible sampled frames; joint geometry may be unreliable|images/deadlift__GB_S07_R14.mp4.jpg|
|deadlift/GB_S07_R20.mp4|N/A|0|7.042|correct in visible sampled frames|images/deadlift__GB_S07_R20.mp4.jpg|
|deadlift/LLK_S01_R02.mp4|0.992|0|6.915|correct in visible sampled frames|images/deadlift__LLK_S01_R02.mp4.jpg|
|deadlift/LLK_S01_R03.mp4|1.000|0|6.950|correct in visible sampled frames|images/deadlift__LLK_S01_R03.mp4.jpg|
|deadlift/LLK_S01_R04.mp4|1.000|0|6.927|correct in visible sampled frames|images/deadlift__LLK_S01_R04.mp4.jpg|
|deadlift/LLK_S01_R06.mp4|0.965|1|6.982|correct in visible sampled frames|images/deadlift__LLK_S01_R06.mp4.jpg|
|deadlift/LLK_S03_R11.mp4|0.963|0|11.773|correct in visible sampled frames|images/deadlift__LLK_S03_R11.mp4.jpg|
|deadlift/LLK_S04_R05.mp4|1.000|0|12.057|correct in visible sampled frames|images/deadlift__LLK_S04_R05.mp4.jpg|
|deadlift/LLK_S05_R15.mp4|1.000|0|11.982|correct in visible sampled frames|images/deadlift__LLK_S05_R15.mp4.jpg|
|deadlift/LLK_S05_R17.mp4|1.000|0|11.958|correct in visible sampled frames|images/deadlift__LLK_S05_R17.mp4.jpg|
|deadlift/LLK_S06_R31.mp4|1.000|2|12.114|correct in visible sampled frames|images/deadlift__LLK_S06_R31.mp4.jpg|
|deadlift/LLK_S06_R32.mp4|N/A|0|12.149|correct in visible sampled frames|images/deadlift__LLK_S06_R32.mp4.jpg|
|deadlift/deadlift_01.mp4|0.991|0|6.424|correct in six sampled frames|images/deadlift__deadlift_01.mp4.jpg|
|deadlift/deadlift_02.mp4|1.000|0|6.478|correct in six sampled frames|images/deadlift__deadlift_02.mp4.jpg|
|deadlift/deadlift_03.mp4|1.000|0|6.482|correct in six sampled frames|images/deadlift__deadlift_03.mp4.jpg|
|benchpress/bench_01.mp4|0.992|0|6.391|correct in six sampled frames|images/benchpress__bench_01.mp4.jpg|
|benchpress/bench_02.mp4|0.820|0|6.422|correct in three visible sampled frames; three sampled poses missing|images/benchpress__bench_02.mp4.jpg|
|benchpress/bench_03.mp4|0.860|0|6.351|correct in visible sampled frames|images/benchpress__bench_03.mp4.jpg|
|squat/squat_01.mp4|N/A|0|6.796|correct in six diagnostic frames; no completed plate reps|images/squat__squat_01.mp4.jpg|
|squat/squat_02.mp4|1.000|0|6.548|correct in six sampled frames|images/squat__squat_02.mp4.jpg|
|squat/squat_03.mp4|N/A|0|6.576|correct in six sampled frames|images/squat__squat_03.mp4.jpg|
|team9/squat/stu10_70.mp4|1.000|0|11.693|correct in six sampled frames|images/team9__squat__stu10_70.mp4.jpg|
|team9/squat/stu3_64.mp4|0.990|0|12.562|correct in six sampled frames|images/team9__squat__stu3_64.mp4.jpg|
|team9/squat/stu8_67.mp4|1.000|0|12.260|correct in six sampled frames|images/team9__squat__stu8_67.mp4.jpg|
|team9/benchpress/stu9_10.mp4|1.000|52|12.931|wrong: standing spotters selected in all six rep samples f243/f401/f558/f740/f904/f1085 while lifter lies on bench; coverage 100%|images/team9__benchpress__stu9_10.mp4.jpg|
|team9/benchpress/stu2_10_edited.mp4|1.000|0|7.098|correct in six sampled frames|images/team9__benchpress__stu2_10_edited.mp4.jpg|
|team9/deadlift/stu7_1_1.mp4|0.988|0|12.471|correct in six sampled frames|images/team9__deadlift__stu7_1_1.mp4.jpg|
|team9/deadlift/stu4_2_-1.mp4|N/A|0|7.356|correct in six diagnostic frames; no completed plate reps|images/team9__deadlift__stu4_2_-1.mp4.jpg|
|team9/deadlift/deadlift_08.mp4|0.980|0|6.422|correct in six sampled frames|images/team9__deadlift__deadlift_08.mp4.jpg|

Confirmed wrong lifter videos: benchpress/stu10_11.mp4 (images/benchpress__stu10_11.mp4.jpg), benchpress/test153.mp4 (images/benchpress__test153.mp4.jpg), squat/stu6_11_2.mp4 (images/squat__stu6_11_2.mp4.jpg), squat/stu6_11_3.mp4 (images/squat__stu6_11_3.mp4.jpg), deadlift/GB_S04_R03.mp4 (images/deadlift__GB_S04_R03.mp4.jpg), team9/benchpress/stu9_10.mp4 (images/team9__benchpress__stu9_10.mp4.jpg)

## Reuse

`--resume` requires matching source/model/evaluator hashes; timings retain original run values. `--pose-results DIR` reads DIR/<key>/pose.npz (normalized MP33 lms/raw_lms, valid, total, fps, w, h). Frame count must match plate rows. Recompute all metrics with the same command. `--videos` accepts alternate dataset/team9 roots containing the frozen cohort, not a new split.

Hashes, denominators, detailed cells and jump frames: baseline.json and raw/*/evaluation.json. Generated media/models/NPZ/eval_out are not committed.

## 검증과 재현 메모

시험: plate 132 passed, feedback 235 passed, 기존 midterm 지표 2 passed. 새 합성 시험 9개는 표본 SD·rep 쌍 중앙값·rep 구간 합집합·visibility/결측 분모·프레임 점프·홀수/짝수 재번호와 결측·완료 이벤트 시계·종목 요약 집계를 확인한다.

```bash
cd /home/kojy1009/work/m0-baseline/plate_pipeline
python -m eval.baseline --out eval_out/baseline_mp --resume
python -m pytest -q tests
python -m pytest -q eval/midterm/test_metrics.py
cd ..
PATH=/home/kojy1009/miniconda3/lib/python3.14/site-packages/playwright/driver:$PATH python -m pytest -q tests/fb
```

기본 명령 `python -m eval.baseline --out eval_out/baseline_mp`도 전체 평가를 수행한다. 성공한 위 전체 실행은 모든 영상의 코드 해시가 같고, 이전 시도와 해시가 달라 69개를 모두 새로 측정했다. --resume은 동일 해시의 완료 결과만 재사용한다. Node가 기본 PATH에 없어 최초 feedback 시험은 실행 환경 오류가 났으며, 설치된 Playwright의 Node를 PATH에 넣은 재실행은 235개 모두 통과했다.

원본 v40 분할/라벨/집계 gate 및 73개 원본 PTS clock gate도 통과했다. 중간 파일은 eval_out/baseline_mp/reproduction/에 있다. S3a 저장 NPZ를 모델 추론 없이 재생해 비벤치 기존 각도 칸이 동일함을 확인했고, S3b 네이티브 홀수/짝수 재생은 0/43이다. 새 실제 추론에서도 stu10_70의 무릎각 rep 쌍 차이가 바닥 38.351°, 상승 중간 79.580°로 S3a 기록을 재현했다.

벤치의 무릎각·엉덩이각·몸통 기울기는 기존 구조적 가림 규칙에 의해 N/A인 칸이 있다. 기존 품질 규칙과 판정 임계값을 유지했다. 거짓 차이 시험에는 S3b의 종목별 기존 표시 지표를 사용한다. 측정 가능한 칸이 적고 표본이 같은 영상 안의 rep이므로 0/43은 독립 검증에서 거짓 양성률 0%라는 뜻이 아니다.

속도는 이 단일 기준선 명령의 분석 비용이다. 원본 횟수 평가 패스에 더해 기존 feedback이 필요로 하는 회전/CFR 준비 영상의 원판·포즈 패스를 수행한다. 변환 영상에서는 원본 참조 평가 비용도 표의 decode/detector/KLT/total에 포함되어 있다. M1 비교에서도 동일한 입력과 평가 패스를 유지해야 한다. CFR 인코딩 자체는 prepare 시간으로 따로 남기고 fps에서 제외했다.

잘못 선택한 골격도 현재 동작의 일부이므로 포즈 통계에 포함했다. 특히 team9/stu9_10은 포즈 선택률 100%지만 확인 이미지 6장 전부 보조자를 골랐다. 높은 커버리지나 거짓 차이 0개를 올바른 리프터 선택의 근거로 해석할 수 없다.

## 확인한 리프터 오류

|영상|확인 내용|이미지 파일|
|---|---|---|
|benchpress/stu10_11.mp4|wrong: spotter selected at diagnostic f359/f719/f1078/f1438; no completed plate reps, excluded from rep-interval denominators|images/benchpress__stu10_11.mp4.jpg|
|benchpress/test153.mp4|wrong: spotter selected at diagnostic f0/f239/f299; no completed plate reps|images/benchpress__test153.mp4.jpg|
|squat/stu6_11_2.mp4|wrong: background standing person selected at diagnostic f127/f159; no completed plate reps|images/squat__stu6_11_2.mp4.jpg|
|squat/stu6_11_3.mp4|wrong: spurious torso on stored yellow plate at diagnostic f0/f38; no completed plate reps|images/squat__stu6_11_3.mp4.jpg|
|deadlift/GB_S04_R03.mp4|wrong: background standing person at far right selected at rep frame f36|images/deadlift__GB_S04_R03.mp4.jpg|
|team9/benchpress/stu9_10.mp4|wrong: standing spotters selected in all six rep samples f243/f401/f558/f740/f904/f1085 while lifter lies on bench; coverage 100%|images/team9__benchpress__stu9_10.mp4.jpg|

## 원본 횟수와 포즈 측정 rep 구간의 차이

횟수는 원본 PTS의 완료 back 이벤트이고, 포즈는 현재 feedback의 준비 영상·완료 rep·시작/먼 점 시점 검사 규칙을 따른다. 아래는 두 경로에서 측정 가능한 rep 수가 다른 영상이다. 종료 시점이 없는 rep 등은 기존 제외 이유를 그대로 기록했다. 원본 횟수 기준선에는 영향을 주지 않는다.

|영상|원본 완료 횟수|포즈 측정 rep|기존 제외 이유|
|---|---:|---:|---|
|deadlift/GB_S07_R12.mp4|1|0|rep 1: rep 끝 시점 없음|
|deadlift/GB_S07_R14.mp4|1|0|rep 1: rep 끝 시점 없음|
|deadlift/GB_S07_R20.mp4|1|0|rep 1: rep 끝 시점 없음|

환경의 ffmpeg: `ffmpeg version 4.4.2-0ubuntu0.22.04.1 Copyright (c) 2000-2021 the FFmpeg developers`.

산출물: `plate_pipeline/eval_out/baseline_mp/baseline.json`, `baseline.md`, `baseline.zip`, `images/`. ZIP은 baseline.json·baseline.md·확인 이미지 69장만 포함한다. 영상·데이터셋·모델·NPZ·eval_out 산출물은 커밋하지 않는다. 포즈 NPZ 계약과 M1 재생 방법은 `plate_pipeline/eval/BASELINE.md`에 있다.
