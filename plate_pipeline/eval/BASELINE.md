# M0 기준선 실행

저장소 루트에서 쓰는 Python 환경에 MediaPipe, ONNX Runtime CPU, PyAV,
OpenCV, NumPy, pandas, pytest가 필요하다. 영상 준비에는 ffmpeg/ffprobe,
기존 feedback JavaScript 시험에는 Node가 필요하다.

```bash
cd plate_pipeline
python -m eval.baseline --out eval_out/baseline_mp --backend mediapipe
```

M0 브랜치의 원래 명령에는 backend 옵션이 없었다. M1에서는 기본이 RTMPose라서
MediaPipe를 다시 측정할 때 위처럼 `--backend mediapipe`를 명시한다.

기본 입력은 Git 공통 디렉터리가 있는 원래 작업 트리의 다음 경로다.
별도 worktree에서도 같은 로컬 입력을 사용한다.

- `new_data/new_dataset`: 고정된 midterm 61개 영상
- `new_data/experiments_athlete_pipeline_v40/athlete_yolo_balanced_phase_sameconditions_seed42`: 원래 split, records, train.py 집계식, decode.py 이벤트 매칭
- `new_data/new_labels.csv`: 원래 라벨 대조
- `plate_pipeline/data/team/test9`: 개발 9개 영상 (61개와 같은 키는 한 번만 측정)
- `plate_pipeline/models/plate_tiny.onnx`: 기존 원판 검출기
- 실행 작업 트리의 `pose_landmarker_lite.task`: 기존 MediaPipe 모델

`--local-root PATH`로 입력 루트를, `--model PATH`로 원판 모델을 지정한다.
`--videos DIR [DIR ...]`는 영상 루트들을 바꾸며 고정된 평가 집합을 유지한다.
각 루트 아래 `종목/파일명` 또는 `파일명`으로 영상을 찾는다.

`baseline.json`, `baseline.md`, `REPORT.md`, `baseline.zip`과 `images/`를
출력한다. 중간 산출물은 `raw/<key>/`에 남긴다. 영상별 원본 해시와
모델/지표 코드 해시를 기록한다. 중단 후 `--resume`을 추가하면 해시가
같은 완료 영상만 재사용한다. 이 경우 영상별 시간은 첫 측정값이다.

## M1과 비교

포즈 계산과 지표는 `baseline.py`와 `baseline_metrics.py`로 분리했다.
기존 포즈 설정은 feedback 경로 그대로다: 매 프레임, 후보 최대 2명,
confidence 0.45, 원판 손목 선택, 기존 EMA 스무딩. S1의 2프레임 간격
진단용 포즈 대신 S3a/S3b의 실제 피드백 경로를 평가한다.

RTMPose 결과를 MediaPipe 33개 관절 배치로 변환해
`DIR/<key>/pose.npz`로 저장한 뒤 다음 명령을 사용한다.

```bash
python -m eval.baseline --out eval_out/baseline_rtm --pose-results DIR
```

NPZ 계약:

|키|내용|
|---|---|
|`lms`|기존 스무딩에 대응하는 `(N,33,4)` x/y/z/visibility, x/y는 화면 크기로 나눈 좌표|
|`raw_lms`|스무딩 전 선택 관절, 같은 shape|
|`valid`|선택한 포즈가 있는 프레임의 `(N,)` bool|
|`total`, `fps`, `w`, `h`|CFR 준비 영상의 프레임 수, fps, 표시 너비/높이|

없는 프레임은 NaN으로 두고 valid=False로 한다. 원판과 같은 준비 영상의
프레임을 사용해야 한다. M1 파일 재생에서는 포즈 추론 시간을 측정할 수
없으므로 fps를 N/A로 낸다. 실제 M1 속도는 추론 생산자를 연결해 측정해야 한다.

원본 PTS 기반 횟수 평가는 별도로 유지한다. 기존 feedback의 회전/CFR 변환이
필요하면 포즈는 준비 영상의 원판 결과와 함께 평가한다. 원본 횟수 숫자를
CFR 결과로 바꾸지 않는다.

## 지표 해석

- 횟수: 기존 midterm 집계식과 match_events를 직접 사용한다. 완료 `back`이
  없는 내부 trip은 완료 이벤트에서 제외한다. OBO는 오차 절댓값 ≤1,
  event F1은 영상별 평균이 아닌 TP/FP/FN micro 집계다.
- 흔들림: S3a와 같은 측정 가능한 칸 전체(제외 rep도 포함)에서 영상별로
  계산한다. 표본 표준편차(ddof=1)와 중복 없는 rep 쌍 차이의 중앙값을
  종목별로 다시 중앙값 요약한다. 표본이 2개 미만이면 N/A다.
- 거짓 차이: S3b와 같은 홀수/짝수 분리·재번호·세트 판정이다. 실제 판정한
  칸만 분모다. 4개 미만 rep 영상은 판단하지 않는다.
- 커버리지·관절: 완료 rep 구간의 프레임 합집합을 분모로 쓴다.
  관절 묶음 비율은 좌우 관절 × 프레임 관측을 합친 비율이며 미검출도
  실패로 센다. raw visibility ≥0.5다. 화면 밖 검사는 각도 칸 품질에
  별도로 적용되는 기존 규칙이다.
- 사람 변경: 연속 두 프레임의 몸통 중심 이동이 현재 원판 높이를 초과한
  횟수다. 누락 프레임을 건너 연결하지 않는다. 실제 identity 정답이 아닌
  자동 점검 신호다. 이미지 검토 결과와 구분한다.
- 속도: 한 영상씩 CPU 처리한다. 디코드는 두 경로의 디코드·색변환,
  KLT는 tracker.step의 연결/추적, 포즈는 MediaPipe 입력 변환·추론이다.
  나머지는 모델 준비·선택·스무딩·원판 분석·원판 결과 저장을 포함한다.
  CFR 변환과 최종 평가·확인 이미지 생성 시간은 처리 fps에서 제외한다.

확인 후 `eval_out/.../lifter_review.json`에 `{ "영상 key": "검토 결과" }`를
적고 `--resume`으로 보고서를 다시 만들 수 있다. 틀린 리프터는 결과를
`wrong ...`으로 시작하면 별도 목록에 들어간다. 여섯 장 확인은 영상 전체의
오류가 없다는 보장이 아니다. rep이 없으면 전체 영상 진단 이미지를 만들고
이미지에 `NO COMPLETED REP`이라고 명시한다.

## M1 RTMPose 실행

`rtmlib==0.0.16`을 같은 M0 Python 환경에 추가한다. ONNX Runtime CPU를
사용하고, BodyWithFeet의 Halpe26 모델과 사람 검출기는 첫 실행 때
`~/.cache/rtmlib/hub/checkpoints/`에 받는다. 저장소에 모델을 넣지 않는다.

```bash
python -m pip install --no-deps rtmlib==0.0.16  # ORT/OpenCV/NumPy가 있는 M0 환경에서
python -m eval.baseline --out eval_out/baseline_rtm/lightweight --mode lightweight
python -m eval.baseline --out eval_out/baseline_rtm/balanced --mode balanced
# M0의 MediaPipe 생산자와 선택 규칙을 그대로 다시 실행:
python -m eval.baseline --out eval_out/baseline_mp --backend mediapipe
```

기본 backend는 `rtmpose`, mode는 `lightweight`다. `core.config.PoseCfg`로
운영 경로도 바꿀 수 있다. 두 모드 모두 매 프레임 사람 검출 → 검출된 모든
사람에게 포즈 → 원판으로 리프터 선택 → 기존 feedback EMA 순서다.
`--pose-results` 재생도 유지한다. `pose.npz` 계약은 위 표와 같다.
RTMPose의 z=0은 2D 자리 채움 값이다. 3D 관측이 아니다.

RTMPose 시간은 `timing.person_detection`, `timing.pose`에 각각 저장한다.
모델의 입력/출력 전후처리를 포함하며, `rest`와 겹치지 않는다. MP33 어댑터, 점수 분포 계산, 사람 선택과 EMA는 `rest`에 포함된다. MediaPipe는
검출 시간이 내부에 포함돼 분리하지 않는다. `score_distribution`에는 모든
프레임·모든 사람·26점의 원점수 개수와 분위수를 영상별로 기록한다.
`--resume`에는 backend/mode와 생산자·어댑터·설정 코드 해시도 포함된다.

69개 확인 이미지를 두 모드 모두 검토해 각 출력의 `lifter_review.json`을
작성한 뒤 다음 비교 명령으로 표에 반영한다. 이 명령은 최초 실행의
`wall_s`를 보존하면서 전체·종목별 비교표와 두 모드 json·md·이미지 zip을 만든다.

```bash
python -m eval.m1_compare --mp /path/to/baseline_mp \
  --rtm eval_out/baseline_rtm --report eval/M1_NOTES.md
```

`invariance.json`은 69개 영상의 원판 프레임 기록을 gzip 해제 후 바이트로
대조하고 원판 결과 JSON의 모든 값(측정 실행 시간 두 필드 제외)을 대조한다.
CFR 사본이 필요한 영상의 `pose_plate/`도 별도로 대조한다. 횟수 61개는
영상별 전체 count 레코드도 비교한다. 원본 영상·원판 모델·지표 코드 SHA256과 확인 이미지의 프레임 목록도 같아야 한다. 다르면 비교 명령은 실패한다.


### 원판 core 합성 회귀 비교

동일한 스크립트로 M0/M1 각각의 실제 core를 import해 90조건(종목 3 × fps 3 × 잡음 5 × seed 2)의 최종·라이브 횟수, 횟수·시간 정확도, rep별 속도와 횟수 표시 시차 원값을 JSON으로 저장한다. 실제 영상의 rep·깊이·속도 전체 값 불변은 앞의 `invariance.json`으로 확인한다. `M0`와 `M1`은 각 worktree의 절대 경로다. 생성 파일은 추적하지 않는다.

```bash
PYTHONPATH="$M0/plate_pipeline" python "$M1/plate_pipeline/eval/m1_sim_regression.py" --out "$M1/plate_pipeline/eval_out/baseline_rtm/sim_m0.json"
PYTHONPATH="$M1/plate_pipeline" python "$M1/plate_pipeline/eval/m1_sim_regression.py" --out "$M1/plate_pipeline/eval_out/baseline_rtm/sim_m1.json"
cmp "$M1/plate_pipeline/eval_out/baseline_rtm/sim_m0.json" "$M1/plate_pipeline/eval_out/baseline_rtm/sim_m1.json"
```

두 JSON이 있으면 `eval.m1_compare`는 바이트 동일성을 확인하고 함께 zip에 넣는다.
