# 작업 안내 (Claude 세션용)

이 저장소의 지금 작업은 **`plate-pipeline` 브랜치**다. 3대 운동(스쿼트·벤치·데드) 영상에서 원판을 따라가 횟수·템포·속도를 기록하고, 세트 안 변화와 기준 영상과의 차이를 판정어 없는 문장으로 주는 파이프라인이다.
- `plate_pipeline/`: 원판 인식·신호·판정·세트 기록.
- 저장소 루트의 `feedback/`, `run_demo.py`, `web/`, `track3d/`, `docs/design.md`: 팀이 만든 비교 리포트(feedback-ui, 2026-10-22에 합침). 원판 rep으로 돈다 (`run_demo.py --segment plate`). 기준 문서는 `docs/design.md`다.

먼저 읽을 것:
1. `plate_pipeline/docs/PLAN.md` — 지금까지 한 것, 구현 순서(1~6단계)와 완료 기준, 다음 할 일
2. `plate_pipeline/README.md` — 폴더 구성, 실행법, 시험 결과
3. `plate_pipeline/docs/DESIGN.md` — UI 색·글자·레이아웃·컴포넌트 규칙 (화면을 만들 때)
4. `plate_pipeline/train/plate/README.md` — 원판 검출기 학습 순서

규칙:
- 커밋과 푸시는 `plate-pipeline` 브랜치에만. `main`은 건드리지 않는다.
- 저장소 루트의 예전 코드(노트북, `realtime_compare_*.py`, `model/`, `phase_experiments/`)는 이전 학기 것이다. 특히 YOLO 바벨 검출(`model/extract_barbell_yolo_world.py`, `visualize_barbell_yolo_world.py`, `phase_experiments/barbell_yolo_world`)은 잘 안 돼서 보지 않는다.
- 코드를 바꾸면 `cd plate_pipeline && python -m pytest -q tests`와 루트의 `python -m pytest -q tests/fb`가 통과해야 한다. 판정 규칙이나 설정값(`core/config.py`)을 바꾸면 `sim/`의 해당 시험도 다시 돌리고 결과를 README에 적는다.
- 영상, 데이터셋, 모델 파일(`data/`, `datasets/`, `models/`, `eval_out/`, `train/plate/work/`)은 커밋하지 않는다.
- `.gitignore`가 `*.md`를 막는다. `plate_pipeline/` 아래 md는 예외라 그대로 커밋된다. 그 밖의 md는 `git add -f`.
- 설계의 기준 문서는 Claude Docs "3대 운동 기록·피드백 파이프라인 — 처음부터 설계"다. 이 세션에서 읽을 수 없으면 PLAN.md와 README를 기준으로 하고, 바꾼 결정은 PLAN.md에 적는다.
- 사용자와는 한국어 반말로, 짧고 직접적으로.
