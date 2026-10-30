# M3b 서버·업로드·운동 기록

`integ/m3a-result`의 `41b39e3`을 origin에 push했다. 그 커밋에서 `integ/m3b-server`를 만들었다. `origin/plate-pipeline` e0e860e의 PLAN “M3a 결과”와 결정을 읽었다. PLAN.md와 main은 수정하지 않았다.

브라우저 업로드 → 실제 분석/진행률 → 리포트 → 운동 기록 → 다시 열기를 구현했다. 기준 없는 기록과 저장된 기준의 분석 재사용도 확인했다. 제품 경로 목업은 0개다.

## 실행

기존 원판/RTMPose/VideoPose3D 실행 환경에서 서버 의존성을 설치한다. FFmpeg/ffprobe도 필요하다.

```bash
cd plate_pipeline
python -m pip install -r server/requirements.txt
export PLATE_MODEL=/path/to/plate_tiny.onnx
export POSE3D_CHECKPOINT=$HOME/.cache/videopose3d/pretrained_h36m_detectron_coco.bin
OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 python -m server.app --port 8000
```

브라우저: `http://127.0.0.1:8000/`. 주소는 코드에서 127.0.0.1로 고정하고 worker는 1개다. 외부 client/Host와 다른 Origin의 업로드도 거른다. `--data DIR`로 저장 루트를 바꿀 수 있다. 기본은 `plate_pipeline/data/`. 서버 재시작 뒤 완료 기록과 기준 영상은 유지되고, 중단 당시 실행·대기 작업은 실패로 표시해 다시 시도하게 한다.

원판 모델 기본 위치는 `plate_pipeline/models/plate_tiny.onnx`다. RTMPose는 기존 tracked N=10, 벤치 guard 끔을 그대로 쓴다. 사용한 RTMPose 모델은 `~/.cache/rtmlib/hub/checkpoints/`의 HumanArt YOLOX-m과 Halpe26 RTMPose-m ONNX다. 모델·영상·저장 DB는 커밋하지 않았다. FastAPI/uvicorn/multipart/httpx의 실제 검증 버전은 server/requirements.txt에 고정했다.

자세 카드 설정은 기본 false. 서버에 `--posture-cards`를 붙이면 true다. CLI도 `python -m server.compare --user ... [--ref ...] --exercise ... --out ... [--posture-cards]`로 같은 경로를 쓴다.

## API

|요청|입력·응답|
|---|---|
|POST /api/jobs|multipart: user 필수, ref 파일 또는 ref_id 또는 둘 다 없음, exercise 필수, plate_mm 기본 450, load_kg/height_cm 선택. 202 `{id}`.|
|GET /api/jobs/{id}|status(queued/running/completed/failed), stage, stages의 progress(0~1)·elapsed_s·상태·processed/total, error. 완료 시 report_url/session_id/ref_cache_hit. 오류 원문은 error_detail.|
|GET /api/sessions|실제 완료 기록: id, date, exercise, reps, has_ref, vmax, report_url, ref_id.|
|GET /sessions/{id}|리포트. 상대 영상 경로를 위해 /sessions/{id}/로 정규화(307). 영상은 HTTP Range 응답을 지원.|
|GET /api/refs|실제 업로드된 기준의 id, 이름, 날짜, SHA-256. 서버 파일 경로는 노출하지 않음.|

새 기준과 ref_id를 함께 보내면 422, 없는 기준/작업/세션은 404, 빈 파일·잘못된 종목·양수가 아닌 입력은 422다. 업로드 파일 이름을 저장 경로로 사용하지 않는다.

```bash
curl -F user=@USER.mp4 -F ref=@REF.mp4 -F exercise=squat \
  -F plate_mm=450 -F height_cm=175 http://127.0.0.1:8000/api/jobs
# 기준 없이 기록만: ref와 ref_id를 모두 생략
# 재사용: ref 파일 대신 -F ref_id=저장된_ID
```

## 저장·대기열·제한

```text
data/
  index.sqlite3            # jobs / sessions / refs 색인
  uploads/<job_id>/         # 원본 업로드
  refs/<sha256>/source.mp4  # 저장된 기준 원본
  sessions/<id>/
    result_v3.json
    report.html            # JSON, Three.js, 라이선스 내장
    user.mp4 [ref.mp4]      # 실제 분석한 H.264 CFR 사본
    analysis/ [prepared/]   # 원판 rows/result, 포즈, 3D 중간 관측
  ref_cache/<key>/          # 완성된 기준 관측만 게시
```

SQLite를 선택했다. 완료 파일을 쓴 뒤 세션 색인을 만들고, 그 뒤에 작업을 완료로 공개한다. 세션 목록은 이 색인만 읽는다. 작업은 프로세스 내 FIFO 대기열 하나와 전용 분석 thread 하나로 순서대로 실행한다. 같은 데이터 디렉터리로 서버 프로세스를 여러 개 실행하는 구성은 지원 범위가 아니다.

영상 하나당 **512 MiB**, 전체 multipart 요청은 **1025 MiB** 제한이다. Content-Length가 없어도 ASGI stream에서 실제 수신 바이트를 센다. multipart 임시 저장 뒤 파일별 제한도 검사한다. 초과 시 413. 제한은 서버 create_app의 upload_limit으로 바꿀 수 있으며 기본 UI는 512 MiB를 안내한다.

## 기준 캐시 규칙

key = SHA-256(원본 기준 파일 SHA-256 + 설정 SHA-256). 설정에는 종목, 원판 지름, DEFAULT 설정, 원판/3D/RTMPose 모델 파일 해시, 계산·준비 코드 해시가 들어간다. 같은 파일을 새로 업로드해도 내용 해시가 같으면 같은 관측을 재사용한다. ref_id는 원본 파일을 다시 찾는 식별자이며 캐시 key 자체는 아니다.

캐시 내용: 준비된 ref.mp4, 원판 rows/result, pose.npz, 입력 키 적용 전 native.npy. .pending에 다 쓴 뒤 complete.json과 함께 원자적으로 게시한다. 불완전한 캐시는 적중으로 보지 않는다.

키·부하·posture_cards는 추론을 바꾸지 않으므로 캐시 설정 해시에서 뺐다. 기준 3D의 표시 방향·크기 맞춤과 최종 숫자 구성은 해당 세션 입력으로 다시 한다. 175→180 cm로 바꾼 추가 실제 업로드에서도 추론 캐시 적중과 양쪽 3D 높이 180 cm를 확인했다. 모델을 처음 받았다면 추론 뒤 실제 모델 해시로 캐시 key를 다시 확정한다.

캐시 적중 실행의 기준 rows/result/pose/native 파일은 첫 실행과 SHA-256까지 같았다. 원판·포즈·3D 모델 추론을 재실행하지 않았고, 로드/크기 맞춤만 했다.

## 실제 진행률과 계산 보존

단계는 영상 준비 → 원판 내 영상 → 원판 기준 영상 → 포즈 → 3D → 판정 → 리포트다. 원판은 기존 recognize의 프레임 콜백을 process_video로 전달했고, 포즈는 기존 프레임 루프의 끝에 콜백만 추가했다. 처리한 프레임 수/전체 프레임 수를 보낸다. 포즈 단계는 양쪽 영상 프레임 합계다. 기준 캐시 관측은 읽은 뒤 완료로 표시하고 캐시 여부를 별도로 남긴다.

준비·3D·판정·리포트는 실제 작업 단위 완료 시 갱신한다. 3D의 분모는 분석 대상 영상 수이며 가짜 시간 보간은 없다. 화면은 300 ms마다 API를 읽는다. 단계 시간은 monotonic 시계로 재고, 대기열/파일 업로드 시간은 처리 시간에서 제외한다. 기준이 없으면 해당 단계는 skipped다.

실제 스쿼트 쌍의 fps/t/원판/깊이/rep 수치/geo/2D/3D/set_median/문장/속도 저하는 M3a와 정확히 같았다. 바뀐 것은 요청한 crop, 자세 카드 정책, has_ref/cache/timing 메타데이터다. M3a 저장 JSON에 없던 선택 필드 blank도 현재 코드에서 유지한다. 자세 카드 true의 고정 기대값 시험은 그대로 통과한다.

## 실제 브라우저 흐름과 단계 시간

Chromium 1920×1080. M3a와 같은 스쿼트 쌍(stu8_67 → stu10_70), 기준 없는 deadlift_08, 저장된 stu10_70을 다시 선택한 스쿼트를 실제 업로드했다. 원판 지름 450 mm, 키 175 cm, 부하 미입력이다. 키는 입력 경로 검증용 값이며 실제 촬영 대상 신장을 확인한 값은 아니다. ORT/torch 6스레드, OpenCV 1스레드다.

|단계|첫 스쿼트 비교|기준 없는 데드|기준 캐시 적중 스쿼트|
|---|---:|---:|---:|
|영상 준비|0.363|1.458|0.272|
|원판 · 내 영상|9.260|9.519|8.546|
|원판 · 기준 영상|14.039|0.000|0.022|
|포즈|29.559|10.085|9.925|
|3D|1.154|0.189|0.227|
|판정|0.025|0.013|0.025|
|리포트|0.157|0.065|0.155|
|전체 처리 s|54.562|21.333|19.179|

완료 rep: 비교 3/5회, 데드 기록 3회, 캐시 비교 3/5회. 진행률 표본 180/72/65개에서 단계별/전체 진행률 감소가 없었다. 세 흐름 콘솔/pageerror 0개. 실제 세션 id와 시간은 runs.json, 관측 응답은 browser_checks.json에 있다.

빈 기록 → 새 기록 → 첫 비교 처리/리포트 → 달력·목록에서 1개 표시 → 날짜 클릭으로 리포트 다시 열기 → 기준 없는 기록 → 저장된 기준 재사용 → 기록 3개 표시를 확인했다. 서버 재시작 뒤에도 SQLite의 세션·기준 목록이 유지됐다. 추가로 손상 파일 실패/다시 시도와 키 180 cm의 실제 캐시 재사용을 확인했다.

## 화면·crop·자세 카드

기본 posture_cards=false에서 무릎·엉덩이·상체 카드/북마크/extra를 만들지 않는다. 원판 템포·속도 저하와 MAD 튀는 rep만 남는다. 스쿼트에서는 속도 저하 카드 1개와 템포 extra가 남았고, 데드는 속도 저하 카드 1개가 남았다. 자세 표는 추정 표시와 숫자만 쓴다. 위쪽 KPI는 마지막 rep 속도 저하다.

기준 없는 JSON은 ref=null, has_ref=false다. 기준 관측을 내 관측으로 복제하지 않았다. 기록 리포트는 내 영상, 내 3D(스쿼트·데드), 내 깊이·속도·템포·표·바 높이·이상치만 표시한다. 벤치는 기존처럼 2D만 표시한다.

crop은 원판 실제 상자와 표시 원, 전체 바 중심 경로를 모두 포함한다. 경계에 걸친 경우 원본 바깥 여유를 검은 배경으로 표시한다. 영상 배율은 가로/세로 중 작은 값을 쓰고 중앙에 배치한다. 원본에 없는 화면을 만들어 채우지는 않는다. 검증한 3세션의 모든 관측 프레임에서 원판 표시 원 전체가 영상 칸 안에 들어왔다(crop_checks.json).

캡처를 직접 열어 확인했다. 긴 입력 폼과 리포트 아래 내용은 스크롤로 이어지며 화면/문장 잘림은 없었다. 3D의 먼 쪽 발이 뜨는 기존 추정 한계는 남아 있고, 자세 문장은 기본 껐다. 검은 여백은 전체 원판을 담기 위한 의도된 표시다.

|캡처|직접 확인한 내용|
|---|---|
|01_empty.png|실제 세션 0개일 때 빈 상태. 예시 기록 없음.|
|02_squat_input.png|두 영상, 종목, 450 mm, 입력 키 칸. 자동 인식 항목 없음.|
|02_squat_processing.png|6/330 프레임 등 실제 처리 수와 단계 진행률. 시간으로 막대가 차는 연출 없음.|
|02_squat_report.png / 02_squat_cards.png|3/5회, 속도 저하 KPI, 원판 전체/바 경로, 자세 카드 제거. 속도 저하 ① 하나가 마지막 rep에 대응.|
|02_squat_estimate.png|추정 자세 표에 k1/k2 강조 없음. 원판 추정 키 차이 안내와 입력 키/분절은 분리됨.|
|03_history.png / 04_reopened.png|실제 완료 기록 1개와 해당 날짜. 클릭하면 같은 세션을 다시 엶.|
|05_deadlift_input.png / 05_deadlift_processing.png|기준 없이 기록만 선택, 내 영상 처리.|
|05_deadlift_report.png / 05_deadlift_cards.png / 05_deadlift_estimate.png|기록 리포트에 내 영상/내 3D 하나, 기준선 없는 그래프, 3회 표와 추정 자세 숫자만 표시.|
|06_cached_input.png / 06_cached_processing.png|API 기준 목록에서 stu10_70 선택. 처리 초반의 실제 진행률.|
|06_cached_report.png / 06_cached_cards.png / 06_cached_estimate.png|캐시 결과에서도 동일한 수치·카드·추정 표.|
|07_history_three.png|완료 세션 3개만 목록에 표시. 비교/기록 구분과 최고 속도 표시.|
|08_failed.png / 09_retry.png|손상 파일을 실제 거부하고 읽을 수 있는 오류 안내. 다시 시도하면 입력 화면으로 돌아감.|
|10_cache_hit_live.png|추가 실제 작업 중 기준 원판 “캐시 재사용” 표시. 키 180 cm로 바꿔도 적중.|

## 시험

- `cd plate_pipeline && python -m pytest -q tests eval/midterm/test_metrics.py`: 181 passed. 오류 안내 조정 후에도 전체 명령을 다시 확인했다.
- 루트 `python -m pytest -q tests/fb`: 235 passed. 이 환경은 Node가 기본 PATH에 없어 설치된 Playwright driver의 Node를 임시 PATH에 추가했다.
- 새 API 시험 6개: 작업 생명주기/진행률/직렬 대기열, 실패/재시도/선택 기준/업로드 제한/세션 목록, 자세 설정 끔·켬, 설정/모델 캐시 무효화, Content-Length 없는 chunked 요청 제한, 재시작 중단 작업 처리. 가짜 파이프라인은 이 시험 파일에서만 주입한다.
- M3a 고정 기대값은 posture_cards=true로 유지. 카드 모든 필드·북마크·extra·outliers·포지션 시각·각도 비교를 통과했다.
- 69개 원판 저장 기록(138파일)과 61개 counts 기록은 SHA-256 기준 바이트 변화 0개. rows 재계산의 sets도 정확히 같았다. core 소스는 바이트 일치이며, 변경된 jobs/feedback.pose는 진행 콜백 부분을 제거한 계산 AST가 M3a와 같다. 새로 69개 영상을 검출한 시험은 아니다.
- FastAPI 시험에 Starlette의 httpx TestClient 폐기 예정 경고 1개가 있다. 실패나 브라우저 콘솔 오류가 아니다.

```bash
cd plate_pipeline
python -m eval.m3b_invariance \
  --source /path/to/m2/selected2d --out results/m3b/invariance.json
# 빈 데이터 디렉터리로 서버를 띄운 뒤 실제 브라우저 시험
python -m eval.m3b_browser --url http://127.0.0.1:8000 \
  --videos /path/to/data/team/test9 --out results/m3b
```

## 남은 목업·산출물

제품 경로 목업: **0개**. 기록·기준 목록·진행률·시간·횟수·문장·리포트는 실제 저장/분석 값이다. 원래 midterm 데모 기준 자료의 예시 화면은 수정하지 않았고 제품 앱에서 사용하지 않는다.

추가 범위: 자세 추정 정확도(V 검증), 이어 붙인 영상 누락 rep(Mc), 원판 상자 환산 오차(M4)는 그대로 남는다. 이는 이번 서버·기록 기능의 목업으로 보정하지 않았다.

`m3b-server.zip`에는 화면 캡처, 실제 첫 스쿼트 세션의 result_v3.json, 이 REPORT와 검증 JSON을 넣었다. 영상·모델·data/·eval_out/은 포함하지 않았다.
