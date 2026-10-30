# M3a v3 결과 JSON과 오프라인 리포트

기준: `plate-pipeline`의 `c09f869`. 작업 브랜치: `integ/m3a-result`. `PLAN.md`, main, 원판·기존 feedback 소스는 수정하지 않았다.

결과 JSON과 HTML 생성, Python 판정 이식, 실제 3쌍과 추가 2건 검증을 끝냈다. 단, 요청한 영상 선정 조건을 모두 충족한 것은 벤치 쌍뿐이다. 61개 평가 자료에는 횟수 일치·완료 3회 이상인 스쿼트/데드 영상이 없다. 아래에 대체 자료와 예외를 명시했다.

## 실행

```bash
cd plate_pipeline
export PLATE_MODEL=/path/to/plate_tiny.onnx
export POSE3D_CHECKPOINT=$HOME/.cache/videopose3d/pretrained_h36m_detectron_coco.bin
OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 python -m server.compare \
  --user USER.mp4 --ref REF.mp4 --exercise squat \
  --plate-mm 450 --load-kg 80 --height-cm 175 --out OUT
```

`--exercise`는 squat, benchpress, deadlift. 원판 기본 경로는 `plate_pipeline/models/plate_tiny.onnx`이며 외부 경로는 환경변수로 지정한다. 키·중량은 선택이다. 원판 모델, RTMPose 모델 캐시와 VideoPose3D 체크포인트가 필요하다. 영상·모델 다운로드나 서버 시작은 이 명령의 작업이 아니다.

OUT: `result_v3.json`, `report.html`, `user.mp4`, `ref.mp4`. 중간 관측은 `analysis/`, 변환본은 `prepared/`에 둔다. 기존 준비 경로로 회전·VFR을 처리하고, H.264가 아니면 H.264 CFR로 변환한다. 이미 H.264 CFR이면 사본을 둔다. 분석·포즈·재생은 이 사본 하나의 프레임 시계를 쓴다. 원본 평균 fps를 유지한다.

HTML에는 JSON, Three.js r147와 MIT 라이선스가 내장된다. 영상은 상대 경로로 읽는다. 인터넷 없이 file://로 열린다. zip에는 영상이 없으므로 재생하려면 해당 쌍 폴더에 `user.mp4`, `ref.mp4`를 별도로 둬야 한다. 폰트는 로컬 글꼴을 쓴다.

## 스키마

`schema_version: 3`, `exercise`, `inputs {plate_mm, load_kg, height_cm}`. 선택 입력은 미입력 시 null이다.

- user/ref: fps, width, height, duration, t, plate, depth, reps, plate_used, crop, geo, j2, kp2, j3names, j3.
- reps: start, mid_down, far, far_in, far_out, mid_up, back, rom_cm, v, down, pause, up, n. 원판 공백으로 속도를 잴 수 없으면 v는 null이고 blank에 이유를 둔다. 완료 rep을 속도 누락 때문에 버리거나 속도 0으로 바꾸지 않는다.
- 단위: 시각·템포 s, 2D px + visibility, 깊이 cm, 속도 m/s, 3D m. 원판·포즈가 없는 프레임은 null. t/plate/depth/kp2/j3의 길이는 같다.
- 공통: sentences, set_median, v_loss, summary. set_median은 반올림 전 원판 rep 값의 중앙값이다.
- findings: cards, extra, outliers, per_rep_angles, pos_times. kpi는 FAR 무릎각 judge 결과(d, se, lim, k, mu, mr, hit). angle_judgments는 포지션 표용 전체 판정이다. 재생용 rep._t/_p/dmax도 Python에서 계산한다.
- show_3d는 벤치에서 false. pose3d_scale은 내 영상에 입력 키를 썼으면 input_height, 없으면 model_native. 기준 3D는 내 영상과 같은 키 척도에 맞춘다.
- geo의 각도·거리·kd는 core.geometry에서 가져온다. stature_cm은 입력 키만 사용한다. 원판 추정 키는 각 영상의 geo_check.stature_plate_cm으로 분리하고 15% 초과 차이면 warn=true다. 분절은 입력 키로 맞춘 3D에서만 계산한다.
- j3names는 M2의 MP33 매핑(0,11~16,23~28)이다. 없는 귀·발끝을 만들어 넣지 않는다. 2D는 리프터의 MP33 중 페이지 점을 사용한다. crop은 보이는 리프터 점의 분위수와 원판 전체 영역에 여유를 더한다.
- 추가 진단: timing, pose3d_height_cm, pose3d_warnings. 서 있는 근거가 없는데 입력 키가 있으면 가장 높은 유효 3D 프레임을 명시적 대안으로 쓰고 경고를 기록한다. 유효 3D 자체가 없으면 모델 관측을 만들어 넣지 않는다.

`server/schema.py`가 필수 키·타입, 배열 길이, 이벤트 순서, crop 범위, 유한 숫자를 내보내기 전에 검사한다. NaN/Infinity JSON은 허용하지 않는다.

## 판정 이식과 고정 기대값

`server/findings.py`로 FINDINGS, OUTLIERS, judge, repsShowing, largestRep, prepReps/posTime을 옮겼다. 세트 평균의 차이를 max(3·SE, 최소 변화)와 엄격한 >로 비교한다. SE 하한은 자세 2°, 템포 0.05초이고 최소 변화는 무릎·엉덩이 8°, 상체 6°, 템포 0.25초다. 템포 카드 생성 여부와 문장은 원판 core의 판정을 따른다. 북마크는 가장 가까운 기준 rep에도 기준선을 넘는 rep만 고르고, 전부면 all=true다. 북마크 있는 상위 3개 뒤에 10% 이상 속도 저하 카드를 붙인다. 나머지는 extra, 세트 내부 깊이·속도의 MAD 이상치는 outliers다.

데드에서는 첫 이동이 상승이므로 포지션 표시·북마크 문구·진행 바 색을 상승→정점→하강으로 표시한다. 스쿼트 기준 번들의 수치·규칙에는 영향이 없다.

`bundle3.json.gz` → `findings_v3_expected.json`:

- 카드 4개와 extra 2개의 no/kind/key/pos/reps/all/best/text/val: 완전 일치.
- score 최대 오차: 0.000041174 (허용 0.001).
- outliers, per_rep_angles(9칸 전체), pos_times: 완전 일치.
- 재현 카드: 바닥 무릎 모든 rep, 내림 rep 1·2·4·6, 상승 중간 상체 rep 1·2·5, 속도 저하 rep 6. 이상치 rep 2.

HTML은 findings/kpi/angle_judgments를 읽는다. JS 판정·북마크 계산은 제거했다. JS에는 재생 시각 보간과 표시값 환산만 남겼다. 지름 변경 시 깊이·속도·관련 파워/일과 문장 괄호의 원판 수치만 비례 환산한다. 판정, 북마크, 키, 분절, 카메라 기하, 3D는 바뀌지 않는다.

## 실제 영상 선정과 시간

원래 조건(61개 평가에서 횟수 일치, 완료 3회 이상, 옆 영상)을 세 종목 모두에 적용할 수 없었다. 평가표에서 횟수 일치·3회 이상은 벤치 4개뿐이고, 옆 조건까지 적용하면 벤치 3개다. squat_02는 정답·검출 모두 1회다. stu10_70은 61개 밖의 team9 영상이다. 완료 3회 이상을 우선하는 대체 검증을 진행했고, squat_02는 추가로 확인했다. 이 예외를 원래 조건 충족으로 계산하지 않았다.

|종목|내 영상 → 기준 영상|완료 rep|선정 근거와 예외|처리 시간|
|---|---|---|---|---|
|squat|stu8_67.mp4 → stu10_70.mp4|3/5|stu8_67 → stu10_70. team9에서 검출 3/5회. 두 영상 모두 61개 밖이고 내 영상은 뒤 비스듬(원판 비율 0.612), 기준은 옆(0.829). 완료 횟수는 team9/sets.csv의 기존 라벨과 일치. strict 선정 조건 예외.|52.63 s|
|benchpress|stu7_12_2.mp4 → stu8_6_2.mp4|5/3|stu7_12_2 → stu8_6_2. 61개 평가에서 각각 정답=검출 5/3회, 옆 태그(원판 비율 0.839/1.204). 요청 선정 조건 충족. 두 영상의 원판 가림·2D 오류는 남아 있다.|60.00 s|
|deadlift|deadlift_08.mp4 → stu7_1_1.mp4|3/5|deadlift_08 → stu7_1_1. team9에서 검출 3/5회. 두 영상 모두 61개 밖이고 내 영상은 앞 비스듬(0.685), 기준은 옆(1.046). 완료 횟수는 team9/sets.csv의 기존 라벨과 일치. strict 선정 조건 예외.|49.86 s|

원판·포즈 ORT 6스레드, torch 6스레드, OpenCV 1스레드, 영상 쌍은 순서대로 실행했다. 시간은 준비·모델 로딩·두 영상 추론·분석·판정까지이며 JSON/HTML 직렬화는 제외한다. 영상 변환은 기존 준비 경로의 ffmpeg 기본 스레드 설정을 그대로 쓴다. 실제 명령·경로·입력·사본 SHA-256은 runs.json에 있다.

스쿼트·데드의 175 cm는 입력 경로를 검증하기 위해 넣은 값이며 영상 속 사람의 실제 키를 확인한 값이 아니다. 벤치는 키를 넣지 않았다. 중량은 세 쌍 모두 미입력이다.

키 입력 쌍의 내/기준 3D 높이 척도는 모두 175 cm로 일치한다. 벤치는 내 영상 모델 자체 크기를 유지하고 기준만 그 척도에 맞췄다. 벤치의 모델 크기를 실제 신장으로 해석하지 않는다. 스쿼트 내 영상의 원판 추정 키는 236.8 cm라 15% 안내가 나온다. 키·분절 수치는 이 원판 추정치에 연동되지 않는다.

## 카드와 캡처 직접 확인

세 쌍 모두 file://, 오프라인 Chromium, 1920×1080. 각 쌍 콘솔/pageerror 0개, 외부 요청 0개. 두 영상 decoded readyState≥2를 확인하고 첫 rep FAR로 이동해 캡처했다. 카드 이동·기준 rep 변경·재생/정지도 확인했다. `browser_checks.json`에 결과를 남겼다.

450→360 mm 입력 시 depth/rom/v는 정확히 0.8배, geo/3D/kpi는 동일했다. 입력 키 없음에서는 키 안내와 분절 미표시, 벤치에서는 WebGL 대신 2D canvas를 확인했다. 모델이 누운 사람을 잘 복원하지 못하는 문제를 목업으로 보정하지 않았다.

### squat

- 1. 올라오는 중간에 무릎을 기준 영상보다 41° 덜 굽혔습니다 (내 세트 137°, 기준 95°). 북마크: 모든 rep.
- 2. 올라오는 중간에 엉덩이를 기준 영상보다 37° 덜 접었습니다 (내 세트 138°, 기준 100°). 북마크: 모든 rep.
- 3. 바닥에서 상체를 기준 영상보다 26° 덜 숙였습니다 (내 세트 11°, 기준 37°). 북마크: 모든 rep.
- 4. 마지막 rep의 올리는 속도가 가장 빠른 rep보다 12% 느렸습니다 (rep 1 0.38 → rep 3 0.33 m/s). 북마크: rep 3.

추가 차이 3개, MAD 이상치 0개.

캡처: `squat_top.png`, `squat_cards.png`, `squat_pose.png`.

top: 3회/5회 KPI와 두 영상이 보이고, 내 영상이 뒤 비스듬이라는 선정 예외가 화면에서도 확인된다. cards: 세트 전체 괄호 ①②③, 마지막 rep ④가 겹치지 않고 문장 4개와 진행 바를 읽을 수 있다. pose: 마네킹의 먼 쪽 발이 뜨고 팔이 몸통 근처에 겹친다. M2 3D 추정 한계이며 추가 보정하지 않았다. 위쪽 캡처에서는 진행 바 아래 조작부가 다음 화면으로 이어진다.

### benchpress

- 1. 올라오는 중간에 엉덩이를 기준 영상보다 78° 더 접었습니다 (내 세트 86°, 기준 163°). 북마크: 모든 rep.
- 2. 바닥에서 상체를 기준 영상보다 25° 덜 숙였습니다 (내 세트 64°, 기준 89°). 북마크: 모든 rep.
- 3. 바닥에서 엉덩이를 기준 영상보다 76° 더 접었습니다 (내 세트 88°, 기준 164°). 북마크: 모든 rep.
- 4. 마지막 rep의 올리는 속도가 가장 빠른 rep보다 38% 느렸습니다 (rep 1 0.52 → rep 5 0.33 m/s). 북마크: rep 5.

추가 차이 1개, MAD 이상치 2개.

캡처: `benchpress_top.png`, `benchpress_cards.png`, `benchpress_pose.png`.

top: 5회/3회와 2D 패널이 보이고 3D 시점 버튼은 없다. cards: ①②③ 세트 전체, ④ 마지막 rep과 ! 이상치가 보인다. pose: 2D 관절선이 누운 사람의 팔·다리 부근에서 교차하고, 기준 영상의 원판이 몸을 크게 가린다. 벤치 자세 카드의 큰 각도 차이는 검증되지 않은 VP3D 추정에서 나온 값이라 실제 자세 차이로 확정할 수 없다. 페이지에도 근거 한계를 표시했다. 벤치는 패널이 낮아 cards/pose 캡처가 같은 스크롤 범위를 포함한다.

### deadlift

- 1. 내려오는 중간에 무릎을 기준 영상보다 15° 덜 굽혔습니다 (내 세트 156°, 기준 142°). 북마크: rep 3.
- 2. 정점에서 엉덩이를 기준 영상보다 11° 더 접었습니다 (내 세트 155°, 기준 166°). 북마크: rep 1.
- 3. 마지막 rep의 올리는 속도가 가장 빠른 rep보다 20% 느렸습니다 (rep 1 0.37 → rep 3 0.30 m/s). 북마크: rep 3.

추가 차이 0개, MAD 이상치 0개.

캡처: `deadlift_top.png`, `deadlift_cards.png`, `deadlift_pose.png`.

top: 3회/5회와 정점 무릎각 KPI, 양쪽 영상이 보인다. cards: rep 3 무릎 ①, rep 1 엉덩이 ②, rep 3 속도 저하 ③의 번호·문장이 대응한다. pose: 정점의 두 3D 몸통이 거의 겹치지만 발목 지지와 실제 발바닥의 관계는 복원하지 못한다. M2 매핑에 발끝이 없어 발·귀를 임의 생성하지 않았다. 내 영상의 앞 비스듬 각도와 기준 측면의 차이는 남아 있다.

## 추가 입력 확인

- squat_02 → stu10_70: 완료 1/5회, 51.28 s. 따라서 두 영상 3회 이상 조건에 맞지 않는다.
- squat_user_set.mp4 → stu10_70: 완료 4/5회, 118.37 s. 데모 기준 내 세트 6회보다 2회 적다. 원판 검출기는 수동 초기 상자 없이 실행했다.
- 내 세트 검출 구간: 5.87–10.04, 13.85–20.44, 49.15–52.46, 55.90–59.93초. 두 번째는 ROM 22.6 cm로 다른 rep(약 56~59 cm)보다 작고, 데모 두 번째 구간 16.89–22.13초와 어긋난다. 23.375–37.667초 클립의 두 rep은 검출되지 않았다. 데모는 일부 클립에 수동 원판 초기 상자를 썼으므로 생성 경로가 다르다.
- 연속 추적의 클립 전환·대상 선택과 rep 분할 문제로 기록했다. 원인을 확정하거나 임계값을 바꾸지는 않았다. Mc에서 다룬다. 세부 수치는 additional_checks.json.

## 시험과 원판 불변성

- `cd plate_pipeline && python -m pytest -q tests eval/midterm/test_metrics.py`: 175 passed.
- 루트 `python -m pytest -q tests/fb`: 235 passed. 이 환경 기본 PATH에 node가 없어, 설치된 Playwright driver의 Node 실행 파일을 임시 PATH에 추가했다. 저장소 시험·코드는 바꾸지 않았다.
- 새 시험 15개: fixture의 모든 필드, 엄격한 경계값과 가장 가까운 기준 rep, 스키마 타입/누락, 입력 키 유무×3종목, 입력 키·기준 3D 맞춤(서 있는 근거 유무 포함), 오프라인 렌더링, 누락 속도 보존.
- 기존 원판 rows/result 69개 쌍(138파일)을 SHA-256으로 읽기 전후 대조했다. 61개의 counts 기록을 포함해 바이트 변화 0개. 모든 rows를 analyse로 다시 계산한 sets도 기존 값과 완전 일치했다. 원판 인식·신호·기하·기존 feedback 소스는 c09f869과 바이트 일치다.
- 69개 검출기를 다시 실행한 시험은 아니다. 원판 구현을 바꾸지 않았음을 소스 해시와 기존 저장 결과의 재계산으로 확인한 것이다. 근거는 invariance.json, 재현 명령은 아래와 같다.

```bash
cd plate_pipeline
python -m eval.m3a_invariance --source /path/to/m2/selected2d --out results/m3a/invariance.json
python -m eval.m3a_verify --runs eval_out/m3a --out results/m3a
```

## 기존 S3b와 겹치는 부분 (병합하지 않음)

- 세트 단위 기준 영상 비교: 무릎각·엉덩이각·몸통 기울기, k·SE와 최소 변화.
- 원판 rep으로 포지션을 정하고 내림·먼 점 멈춤 템포를 비교하는 경로.
- 세트에서 판정된 차이를 rep 목표로 연결하고 상위 3개 문장을 고르는 경로.
- 구체적인 통계는 다르다. S3b(`feedback/set_judge.py`)는 2D 셀 신뢰도·촬영 각도 불일치·참조 제외를 검사하고 core의 산포/배수 표를 쓴다. v3는 3D 관절의 양측 평균과 5프레임 창, 고정 3·SE, closest-reference 북마크를 쓴다. 이번에는 둘 다 유지했다. 벤치의 부정확한 3D 판정과 두 판정계의 차이는 알려진 한계다.

## 남은 목업과 다음 범위

- 제품 리포트의 분석 값 목업: 없음. 종목은 입력값이며 97% 자동 인식 칩을 제거했다. 입력 키·부하도 저장한 inputs에서 읽는다.
- 제품 경로에서 업로드·가짜 진행률·처리 시간 연출·예시 운동 기록의 HTML/JS를 제거했다. 첫 화면은 리포트다.
- 원래 midterm 데모 자료의 가짜 처리 화면, 종목 인식 97%, 과거 날짜 예시 기록은 기준 자료에 그대로 남아 있다. 제품 리포트에서는 사용하지 않는다.
- M3b: 실제 업로드·작업 서버·진행률·운동 기록 저장. 제거한 데모 흐름을 실제 작업으로 구현해야 한다.
- M4: 원판 상자 기반 환산과 카메라 가정의 오차. Mc: 이어 붙인 영상 및 기존 누락 rep. 3D 정확도와 벤치 자세 판정은 검증이 더 필요하다.

## 산출물

`m3a-result.zip`: squat/benchpress/deadlift 각각 result_v3.json + report.html, 캡처 9장과 REPORT.md·검증 근거. 영상·모델·eval_out은 포함하지 않는다. 이 폴더의 개별 JSON/HTML도 동일하다.
