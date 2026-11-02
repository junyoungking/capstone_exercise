# 3대 운동 기록·피드백

스쿼트·벤치프레스·데드리프트를 옆에서 찍은 영상에서 **원판**을 따라가, 횟수와 rep마다의 깊이·템포·속도를 자동으로 기록한다. 기준 영상을 같이 올리면 두 영상을 비교해서 다른 점을 판정어 없는 문장과 화면으로 보여 준다.

> 이 브랜치(`plate-pipeline`)가 최신이다. `main`은 지난 학기 코드 그대로다.

## 지금 상태 (2026-11-02)

| 기능 | 상태 |
|---|---|
| 영상 올리기 → 처리 → 리포트 (기준 영상과 비교, 또는 기준 없이 기록만) | 됨 |
| 횟수 세기 (원판 추적 + 상태기계) | 됨. 촬영 조건을 지킨 영상 56개에서 rep 재현율 82%, 정밀도 85% |
| rep 값: 깊이(ROM), 내림·멈춤·올림 시간, 평균 속도, 세트 안 속도 저하 | 됨 |
| 2D 골격(RTMPose), 3D 골격(VideoPose3D, 벤치는 2D만) | 됨 |
| 운동 기록 달력·목록, 원판 지름 확인 후 다시 계산 | 됨 |
| 종목 자동 인식 | 작업 중 (M5) |
| 자세 문장(무릎·엉덩이·상체), LLM 문장 | 팀 검증 촬영 뒤 (M6) |

진행과 결정은 [`plate_pipeline/docs/PLAN.md`](plate_pipeline/docs/PLAN.md)에, 단계별 결과는 `plate_pipeline/results/<단계>/REPORT.md`에 있다.

## 바로 써 보기

### 1. 설치

Python 3.10 이상 (3.13에서 시험했다). 비교 리포트 시험에는 Node.js가 필요하다.

```bash
git clone https://github.com/junyoungking/capstone_exercise.git
cd capstone_exercise
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install numpy scipy pandas opencv-python av onnxruntime rtmlib torch mediapipe pytest
pip install -r plate_pipeline/server/requirements.txt
```

### 2. 모델 파일

용량 때문에 저장소에 넣지 않는다.

| 파일 | 놓는 곳 | 받는 곳 |
|---|---|---|
| `plate_tiny.onnx` (원판 검출기, 우리가 학습) | `plate_pipeline/models/` | 팀장에게 받는다 |
| `pretrained_h36m_detectron_coco.bin` (3D) | `~/.cache/videopose3d/` | [VideoPose3D](https://github.com/facebookresearch/VideoPose3D) 공개 가중치 ([내려받기](https://dl.fbaipublicfiles.com/video-pose-3d/pretrained_h36m_detectron_coco.bin)) |
| RTMPose (2D 포즈) | 자동 | 처음 실행할 때 rtmlib가 내려받는다 |

다른 곳에 두었으면 환경 변수 `PLATE_MODEL`, `POSE3D_CHECKPOINT`로 경로를 준다.

### 3. 실행

```bash
cd plate_pipeline
python -m server.app --port 8000
```

브라우저에서 http://127.0.0.1:8000 을 연다. 내 영상과 기준 영상을 올리고 종목과 원판 지름을 고르면 처리 진행이 보이고 리포트가 나온다. 기준 영상 없이 기록만 해도 된다. 60초 영상은 CPU 6스레드에서 2분쯤 걸린다.

리포트 파일만 만들 때:

```bash
cd plate_pipeline
python -m server.compare --user 내영상.mp4 --ref 기준영상.mp4 --exercise squat --out out
# 종목: squat / benchpress / deadlift
# 기준 없이 기록만: --ref 빼기, 원판 지름: --plate-mm 450, 키: --height-cm 175
```

`out/report.html`을 브라우저로 연다.

### 4. 시험

```bash
cd plate_pipeline && python -m pytest -q tests
cd .. && python -m pytest -q tests/fb
```

코드를 바꾸면 두 시험이 모두 통과해야 한다.

## 찍는 법

원판이 1차 신호라서 촬영이 결과를 좌우한다.

- 원판 쪽 옆에서 찍는다. 비스듬해도 되지만 정면에 가까우면 원판을 못 찾는다.
- 원판 전체와 사람 전신이 화면에 들어오게 하고, 폰은 고정한다 (삼각대나 바닥).
- 자막·편집·이어 붙이기 없는 원본을 올린다.
- 원판 지름을 입력한다 (기본 450 mm). 작은 원판인데 450으로 두면 깊이·속도 cm가 크게 틀린다.

## 팀원 할 일: 검증 촬영

자세 문장을 켜고 숫자를 검증하려면 우리가 직접 찍은 영상이 필요하다.

- 같은 세트를 폰 2대로 동시에 찍는다: 옆(0°)과 45°.
- 같은 사람이 일부러 다르게 3~5 rep씩: 평소 / 하프(덜 앉기) / 상체 더 숙이기.
- 세션 3번 이상, 횟수 적은 세트 20개. 세트 앞뒤로 바를 10초 그대로 둔다. 템포를 맞춘 세트 쌍과 일부러 다르게 한 쌍도 찍는다.
- 원판 지름을 줄자로 잰다 (mm).
- 세트마다 적는다: 날짜, 사람, 종목, 무게, 원판 지름, 횟수(두 사람이 따로 센 값), 변형(평소/하프/숙이기), 각도(0°/45°), 파일 이름.
- 영상은 저장소에 올리지 않고 팀장에게 전달한다.

## 폴더

```text
plate_pipeline/            지금 파이프라인 (여기서 작업)
  core/                    신호 처리, 상태기계, rep 값, 원판 인식·추적, 포즈, 기하(3D·각도)
  server/                  웹 앱(app.py), 두 영상 비교(compare.py), 판정(findings.py), 화면(web/)
  eval/  sim/  tests/      실제 영상 평가, 합성 시험, 단위 시험
  train/plate/             원판 검출기 학습
  docs/                    PLAN.md(진행·결정), DESIGN.md(화면 규칙), midterm/(중간 발표 데모·그림)
  results/                 단계별 결과 보고서
feedback/ run_demo.py web/ track3d/ docs/design.md
                           두 영상 비교 리포트 (feedback-ui에서 합침)
unified_feedback_v4.py offline_compare.py export_overlay.py phase_bridge.py model/ ...
                           지난 학기 코드 중 지금도 쓰는 것
legacy/                    지난 학기 코드 중 안 쓰는 것 (노트북, 실시간 비교 스크립트, 실험)
tests/fb/                  비교 리포트 시험 (tests/의 나머지는 지난 학기 시험)
tools/                     영상 점검 등 도구
```

## 브랜치와 규칙

- `plate-pipeline`: 기본 브랜치, 최신. 팀원 작업은 여기서 새 브랜치를 만들어 하고 Pull Request로 합친다.
- `main`: 지난 학기 상태 그대로 둔다.
- `feedback-ui`: 비교 리포트의 원래 브랜치 (plate-pipeline에 합쳐짐).
- `integ/<단계>`: 단계별 구현 브랜치. 검증해서 plate-pipeline에 합치면 지운다.
- 영상·데이터셋·모델 파일(`data/`, `datasets/`, `models/`, `eval_out/`)은 커밋하지 않는다. 공개 저장소라 API 키도 코드에 넣지 않는다.
- `.gitignore`가 `*.md`를 막는다. 이 README, `legacy/README.md`, `plate_pipeline/` 아래 md는 예외다. 다른 md는 `git add -f`로 올린다.
- VideoPose3D(3D 골격)는 CC BY-NC 4.0이라 비상업 용도로만 쓴다.

## 문서

- [`plate_pipeline/docs/PLAN.md`](plate_pipeline/docs/PLAN.md): 목표, 마일스톤, 결정, 단계별 결과 요약
- [`plate_pipeline/README.md`](plate_pipeline/README.md): 파이프라인 구성, 실행, 시험 기록
- [`plate_pipeline/docs/DESIGN.md`](plate_pipeline/docs/DESIGN.md): 화면 색·글자·배치 규칙
- [`docs/design.md`](docs/design.md): 비교 리포트 설계
- `plate_pipeline/docs/midterm/`: 중간 발표 데모(`squat_compare_demo_v3.html`, 내려받아 브라우저로 연다)와 그림
