# legacy

지난 학기(MediaPipe·ST-GCN 실시간 비교) 코드 중 지금 파이프라인이 쓰지 않는 파일이다. 2026-11-02에 저장소 루트에서 옮겼다. 변경 기록은 git에 그대로 남아 있다.

- 노트북 11개: 데이터 라벨링, 포즈 추출 학습, phase·종목 분류 실험, 결과 그림.
- `realtime_compare_*.py`, `realtime_rotation.py`: 웹캠 실시간 비교.
- `expert_preprocess.py`, `feedback_overlay.py`, `feedback_integration_patch.py`: 기준 영상 전처리와 화면 덧그림.
- `phase_experiments/`: phase 분류 실험 결과. YOLO 바벨 검출 실험(`barbell_yolo_world/`)은 잘 안 돼서 쓰지 않는다.

주의:
- 루트에서 돌던 코드라 그대로는 import가 안 맞을 수 있다. 저장소 루트에서 `PYTHONPATH=. python legacy/<파일>.py`로 돌린다.
- 실시간 비교 스크립트는 LLM 키를 코드 안 변수에 넣는 방식이다. 공개 저장소라 키를 넣은 채로 커밋하지 않는다.
- 지금 코드나 시험이 쓰는 지난 학기 코드(`unified_feedback_v4.py`, `offline_compare.py`, `export_overlay.py`, `phase_bridge.py`, `unified_feedback_v4_llm.py`, `model/`)는 루트에 남겼다. `model/`의 실험 스크립트는 결과를 `./phase_experiments/`에 쓴다.
