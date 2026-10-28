# M1 RTMPose 결과

- 작업 브랜치: `integ/m1-rtmpose`. M0 `8e5cdb4`에서 만들었다. 사용자가 지정한 `decff40` 이후의 M0 평가 커밋까지 포함해 `integ/m0-baseline`을 origin에 push했다. PLAN.md와 main은 수정하지 않았다.
- PLAN.md의 기말 목표와 마일스톤을 읽었다. 요청한 “M0 결과” 절은 이 파일에 없어서 M0 출력의 REPORT.md를 읽고 기준으로 썼다.
- 원판 검출·추적·신호·rep 계산 코드는 유지했다. server.jobs의 분석 경로 변경은 포즈 누락 시 이전 몸통 기억을 지우는 것뿐이다. CLI에는 `--pose-backend`와 `--pose-mode`를 추가해 같은 선택을 명시적으로 실행할 수 있게 했다. 평가는 CLI 대신 기존 process_video API를 직접 호출한다.
- RTMPose는 rtmlib BodyWithFeet, Halpe26, ONNX Runtime CPU다. lightweight와 balanced 모두 매 프레임 사람 검출 후 모든 검출 상자에 top-down 포즈를 실행한다. 검출 0명일 때 전체 영상을 사람으로 간주하는 RTMPose fallback은 실행하지 않는다.
- 기본 경로와 평가 경로를 같이 연결했다. feedback의 포즈 입출력·캐시·백엔드 라우팅만 바꾸고 각도 계산, SideLock, EMA, 품질 검사, 판정 코드는 그대로 썼다. MediaPipe 비교 경로는 M0의 후보 최대 2명과 기존 선택 규칙을 유지한다. 원판 기반 경로 밖의 예전 `offline_compare.extract_sequence` 경로는 그대로다.

## 선택 규칙

원판이 있으면 기존 손목–원판 중심 거리 / 원판 높이, 최대 4배 규칙을 쓴다. 보이는 손목이 없으면 보이는 점 상자까지 거리로 돌아간다. 동률이면 이전 몸통 중심과의 거리, 후보 인덱스 순이다.

벤치는 네 몸통 점 visibility ≥0.3을 만족하고, 어깨 중점–엉덩이 중점의 픽셀 벡터가 수직에서 45° 초과 기운 사람이 있으면 그 후보들만 남긴다. 없으면 기존 규칙이다. 종목은 입력값을 쓴다.

새 선택의 몸통 중심이 이전 중심에서 현재 원판 높이 1배 초과 떨어지면, 이전 중심 1배 안의 후보를 기존 리프터의 대리로 삼는다. 둘 다 유효한 화면 안 손목이 있고, 새 후보의 정규화 손목 거리가 **0.5 원판 높이 이상 감소하며 기존 거리의 75% 이하**일 때만 바꾼다. 이전 중심이 있으나 새 후보의 몸통 중심을 계산할 수 없는 경우도 불확실한 교체로 취급해 같은 손목 개선 조건을 적용한다. 대리 후보가 없으면 그 프레임을 비운다. 누락 시 이전 중심은 지워 연속 관측 프레임 사이에만 규칙을 적용한다. identity 추적기가 아니므로 실제 리프터가 빠졌다 돌아오는 경우 다음 프레임에서 다시 초기화할 수 있다. 몸통 점프 수는 실제 identity 정답과 구분한다.

## 점수 → visibility와 어댑터

`visibility = clip(raw_score / 0.6, 0, 1)`이다. 음수·비유한 점수는 0, 비유한 좌표도 visibility 0이다. 기존 visibility 임계 0.3 / 0.45 / 0.5는 각각 원점수 0.18 / 0.27 / 0.30에 대응한다. 느슨한 후보 상자 / 안정적인 양쪽 점 / 측정 가능한 관절이라는 기존 임계의 역할을 유지하고, 원점수 0.30을 측정 지원의 기준으로 고정했다. 점수를 확률로 해석하지 않는다.

이 변환은 확률 보정이 아니다. 이 코호트에는 가림 정답이 없어 MediaPipe visibility와 통계적으로 같은 가림 확률이라는 주장은 할 수 없다. RTMPose는 가려진 점을 높은 localization 점수로 추정할 수 있다. 아래 점수 분포는 영상별 모든 프레임·모든 검출 사람·26점의 원점수다. 각 모드 baseline.json에 같은 원분포 요약이 있다. 선택률과 흔들림뿐 아니라 이미지에서의 사람·관절 배치를 함께 확인했다.

Halpe26의 코·중앙 눈·귀, 어깨·팔꿈치·손목·엉덩이·무릎·발목을 MP33 대응 위치에 넣는다. MP29/30은 Halpe24/25(뒤꿈치), MP31/32는 Halpe20/21(엄지발가락)이다. MP1/3/4/6(눈 안·바깥), MP9/10(입), MP17~22(손가락)는 x/y/z/visibility 모두 0이다. 없는 점을 대체 추정하지 않는다. z=0은 2D 자리 채움이고 3D 근거가 아니다.

## visibility 0 점을 쓰는 코드와 동작 시험

- `unified_feedback_v4.py:SIDE_IDS`, `SideLock._side_score`: 벤치에서 index(19/20)·thumb(21/22)를 더한다. RTMPose에서는 양쪽 모두 0이라 어깨·팔꿈치·손목 세 점으로 잠근다. 절대 hysteresis margin 0.25, hold 6은 그대로라 MediaPipe와 전환 시점은 달라질 수 있다. 합성 시험에서 초기 동률 선택과 6프레임 뒤 반대쪽 전환을 확인했다. 좌우 손가락 점수가 비대칭인 예에서는 손가락을 제거하면 초기 선택이 right에서 left로 달라지는 사례도 시험했다.
- `plate_pipeline/core/recog/pose.py:SideLock._side_score`: 운영 기록용으로 같은 손가락 점수를 더한다. 같은 영향이다.
- `unified_feedback_v4.py:get_side_points`, `hand_grip_point`, `bar_proxy`, `compute_metrics`: 손잡이는 wrist/index/thumb의 유효 점 가중 평균이다. 손가락이 빠지면 유효한 손목만으로 대리점을 구한다. 합성 시험에서 결과가 손목 좌표와 같고 confidence가 유효한 손목 점수임을 확인했다. 손목/검지/엄지 x=0.4/0.6/0.8인 예의 손잡이는 x=0.6에서, 손가락을 제거하면 x=0.4(손목)로 바뀐다.
- `feedback/quality.py:required_joints('bar_proxy_conf')`: 두 손목과 index/thumb 여섯 점을 모두 요구한다. RTMPose에서는 손가락 visibility 부족으로 이 칸은 unavailable이다. 다른 팔꿈치 각도 칸은 계속 잴 수 있음을 시험했다. `bar_proxy_conf`는 기존 `ADVISORY_METRICS`상 벤치·데드 모두 직접 자세 판정에서 제외된다. 다만 아래의 기준 rep 자격 검사는 보조 지표도 분모에 넣으므로 간접 영향이 있다. 기존 feedback 정책을 바꾸지 않았다.
- `plate_pipeline/core/recog/pose.py:POSE_CONNECTIONS`, `unified_feedback_v4.py:POSE_CONNECTIONS`와 이를 쓰는 골격 그리기: MP17~22를 잇는 손가락 선이 그려지지 않는다.
- `person_box` 및 보이는 전체 점으로 크기를 구하는 fallback: 없는 점은 visibility 0이라 상자·크기에 기여하지 않는다.
- `feedback/gate.py:_reference_available_ratio`, `feedback/compare.py:_reference_available_ratio`, `feedback/set_judge.py:_eligible`: display metric 전체를 기준 rep 가용 비율로 센다. bar_proxy_conf 누락은 벤치(5지표)에서 최대 20%p, 데드(6지표)에서 최대 약 16.7%p를 낮춘다. 합성 시험에서 벤치의 다른 2지표와 proxy만 유효한 기준 rep이 60% → 40%가 되어 기존 50% 기준을 못 넘고 판정 기준 rep에서 빠지는 것을 확인했다. 따라서 보조 지표라도 판정 가능한 표본 수에 간접 영향을 줄 수 있다.
- `unified_feedback_v4.py:LandmarkSmoother`는 33점을 모두 처리한다. 없는 점은 모든 프레임에서 0으로 들어가므로 EMA 후에도 visibility 0이다. 백엔드·모드·점수 변환 설정을 캐시 키에 넣어 이전 MediaPipe 점이 섞이지 않게 했다.
- `export_overlay.py:_side_connections`와 overlay/report 연결 목록은 SIDE_IDS·POSE_CONNECTIONS를 사용한다. 손가락 연결 메타데이터는 남지만 visibility 검사로 선은 생략된다.
- `feedback/stgcn_infer.py:_pose_kpts`, `unified_feedback_v4.py:_lms_to_realtime_kpts`는 33점 전체를 분류기 입력으로 전달하므로 없는 점도 0으로 들어간다. 이번 수동 종목·원판 segmentation 경로와 baseline에서는 분류기를 실행하지 않는다. 기존 MediaPipe33 체크포인트에 RTMPose 어댑터를 그대로 넣는 정확도는 검증하지 않았고, 자동 판단·새 모델 입력 연결은 M5에서 한다.
- 눈 안·바깥과 입 점을 직접 이용하는 현재 원판 feedback 지표는 없다. 중앙 눈은 MP2/5에만 넣는다.

## world landmarks 사용처

`plate_pipeline/docs/midterm/src/pose_all.py:53`에서 MediaPipe `res.pose_world_landmarks[best]`의 x/y/z를 읽는다. 중간 발표 그림 재생성용 코드다. 현재 `feedback/pose.py`, `core/recog/pose.py`의 운영 추출과 기준선 지표는 world landmarks를 읽지 않는다. 일반 MediaPipe의 normalized landmark z와 world landmark는 별개다. 3D 추가는 M2 범위다.

## 버전과 모델 근거

`unified_feedback_v4.py`의 기존 상단 import를 유지하므로 RTMPose 추론 경로에서도 MediaPipe Python 패키지는 설치돼 있어야 한다. RTMPose 경로에서 MediaPipe landmarker를 생성하거나 추론하지는 않는다.

rtmlib 0.0.16. M0와 같은 Python 환경의 onnxruntime 1.31.0, MediaPipe 1.0.1, NumPy 2.5.3, PyAV 19.0.2, OpenCV 5.0.0.93을 썼다. 모델은 저장소 밖 `~/.cache/rtmlib/hub/checkpoints/`에 있다.

|모드|사람 검출 모델|포즈 모델|
|---|---|---|
|lightweight|`yolox_tiny_8xb8-300e_humanart-6f3252f9.onnx`|`rtmpose-s_simcc-body7_pt-body7-halpe26_700e-256x192-7f134165_20230605.onnx`|
|balanced|`yolox_m_8xb8-300e_humanart-c2c7a14a.onnx`|`rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.onnx`|

모드·모델 출처는 [rtmlib BodyWithFeet](https://github.com/Tau-J/rtmlib/blob/main/rtmlib/tools/solution/body_with_feet.py), 점 대응은 [rtmlib Halpe26 정의](https://github.com/Tau-J/rtmlib/blob/main/rtmlib/visualization/skeleton/halpe26.py)를 확인했다. 실제 설치 버전의 소스와도 대조했다.

실제 모델 SHA256:

|모델|SHA256|
|---|---|
|`yolox_tiny_8xb8-300e_humanart-6f3252f9.onnx`|`ceb11c07298f95c50d7c5abeb906d03340c85f23aa79e3e66966e7fb6c307250`|
|`rtmpose-s_simcc-body7_pt-body7-halpe26_700e-256x192-7f134165_20230605.onnx`|`85c5d669aa0b0c7969799ea9e0620d3b8c1d3c2b6878243e98a1ef8351457986`|
|`yolox_m_8xb8-300e_humanart-c2c7a14a.onnx`|`3dea6513388889f0fff4b77bf7a26013600321b9eb9ceb0e9a400a82572f5f23`|
|`rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.onnx`|`26f3a19e61304a600dfb82d1001d41d24343b89fc70a33ffc84657e0b0bf2ecf`|

원점수 지원 기준은 두 모드·모든 영상에 동일하게 적용했다. 기본 사람 검출 NMS 0.45, score threshold 0.7은 설치된 rtmlib 기본값을 유지했다.

baseline.json의 기존 `fingerprint.pose_model_sha256` 필드는 비교 경로용 MediaPipe task 해시다. RTMPose 가중치 해시로 해석하면 안 된다. 이번 실행의 실제 RTMPose 네 모델 파일과 SHA256은 위 표에 기록했다. backend/mode와 생산자·어댑터·설정 코드는 영상별 fingerprint에도 들어간다. rtmlib 캐시의 가중치를 바꾼 경우에는 `--resume`을 쓰지 않고 다시 실행해야 한다.

## Lightweight 단독 확인

69개, 영상별 원판 gzip 해제 기록과 결과 JSON(실행 시간 2필드 제외)이 M0와 전부 같았다. 횟수 61개의 전체 count 레코드도 전부 같다. 원본 영상·원판 모델·지표 코드 SHA256과 확인 이미지 표본 프레임 목록도 69개 전부 같았다. 처리 합계 3,426.034초, 전체 실행 3,493.571초다. 확인 이미지 69장을 모두 봤고 표본 오류는 5개 영상에서 확인했다.

원점수 1,169,194개, 최소 0.014307, 최대 1.200601이다. 영상별 p50 범위는 0.239885~0.806839다. 위의 고정 0.6 기준은 점수가 높은 localization을 포화시키고, 측정 지원 기준을 원점수 0.3에 두는 운영용 스케일이다. 최대가 1을 넘는 실제 출력에도 적용된다. 영상별로 기준을 조절하지 않았다.

### Lightweight: M0 오류 6개 재확인

“고침”은 같은 6개 표본에서 원래 오류를 다시 보지 못했다는 뜻이다. 전체 프레임 정답은 아니다. 포즈 누락은 사람 선택 오류와 구분했다.

|영상|결과|확인 내용|확인 이미지 (`lightweight/images/`)|
|---|---|---|---|
|benchpress/stu10_11|다른 오류|원래 틀린 운동 중 f359/719/1078/1438은 리프터로 수정. 운동 전 f0에서 배경 사람을 새로 선택|`benchpress__stu10_11.mp4.jpg`|
|benchpress/test153|다른 오류|원래 f0/239/299는 수정. f60에서 보조자를 새로 선택|`benchpress__test153.mp4.jpg`|
|team9 benchpress/stu9_10|그대로 (부분 개선)|원래 6개 모두 오류. 이제 f243/904/1085는 리프터, f401/558/740은 주변 사람 오류가 남음|`team9__benchpress__stu9_10.mp4.jpg`|
|squat/stu6_11_2|다른 오류|원래 f127/159는 수정. f95에서 배경 사람을 새로 선택|`squat__stu6_11_2.mp4.jpg`|
|deadlift/GB_S04_R03|고침|f36을 포함해 표본 6개 모두 리프터 선택|`deadlift__GB_S04_R03.mp4.jpg`|
|squat/stu6_11_3|고침 (누락 있음)|원판 위 허위 골격이 사라짐. f0/15/38은 리프터, f8/23/30은 포즈 누락|`squat__stu6_11_3.mp4.jpg`|

새로 틀린 **영상**은 `squat/val1235.mp4` 하나다. f52/232에서 배경의 서 있는 사람을 선택했다. 이미지 파일은 `lightweight/images/squat__val1235.mp4.jpg`다. 원래 오류 영상에서 새로 틀린 프레임은 위 표의 “다른 오류”에 따로 표시했다. 69개 전체에서 표본 리프터 오류 영상 수는 M0 6개 → lightweight 5개다.

### 추가 좌표 확인 이미지

balanced `stu6_11_2` f95는 선택된 포즈가 있지만 몸통 visibility 0.30~0.39여서 기본 확인 이미지의 선 기준 0.5를 못 넘었다. 원시 코·어깨·엉덩이·손목 좌표를 원본에 표시해 리프터에 붙었음을 확인했다: `balanced/details/squat__stu6_11_2__f95.jpg`. 낮은 visibility의 관절을 측정 칸에 새로 넣지는 않았다.

balanced `val1235`의 f0/105/179는 원시 코·몸통 점이 배경 사람/거울 반사에 붙는 것을 따로 확인했다: `balanced/details/squat__val1235__f0_f105_f179.jpg`. 이 두 확대 진단 이미지도 zip에 넣었다.

### Balanced: M0 오류 6개 재확인

|영상|결과|확인 내용|확인 이미지 (`balanced/images/`)|
|---|---|---|---|
|benchpress/stu10_11|다른 오류|원래 운동 중 f359/719/1078/1438은 수정. lightweight처럼 운동 전 f0에서 배경 사람을 새로 선택|`benchpress__stu10_11.mp4.jpg`|
|benchpress/test153|고침|원래 오류와 lightweight의 새 오류 f60을 포함해 표본 6개 모두 리프터|`benchpress__test153.mp4.jpg`|
|team9 benchpress/stu9_10|고침|원래 6개 모두 오류, 이제 f243/401/558/740/904/1085 모두 리프터. lightweight에 남던 세 보조자 오류도 수정|`team9__benchpress__stu9_10.mp4.jpg`|
|squat/stu6_11_2|고침|원래 f127/159와 lightweight 오류 f95 모두 리프터. f95는 낮은 몸통 visibility를 추가 좌표 이미지로 확인|`squat__stu6_11_2.mp4.jpg`|
|deadlift/GB_S04_R03|고침|f36을 포함해 표본 6개 모두 리프터|`deadlift__GB_S04_R03.mp4.jpg`|
|squat/stu6_11_3|고침 (누락 있음)|허위 골격이 사라짐. f0/15/30/38은 리프터, f8/23은 포즈 누락|`squat__stu6_11_3.mp4.jpg`|

두 모드의 이미지 판독은 사람 선택을 평가한다. 같은 사람을 골랐어도 가려진 팔·다리 위치가 부정확할 수 있다. 예를 들어 balanced `PB_S06_R25` f7의 발 위치는 타당하지 않아 보이고, 벤치의 원판 뒤 상체는 추정 관절이 포함된다. 높은 coverage·visibility가 곧 관절 위치의 정답률은 아니다.

Balanced에서 새로 틀린 **영상**도 `squat/val1235.mp4` 하나다. f0/105/179에서 배경 사람/거울 반사를 골랐고, f52/232/284는 리프터다. 파일은 `balanced/images/squat__val1235.mp4.jpg`다. M0 오류 영상에서 새 프레임에 발생한 오류는 stu10_11의 f0이며 위 표에 “다른 오류”로 기록했다.

## 최종 검증과 기본 모드 제안

두 모드 각각 확인 이미지 69장, 총 138장을 검토했다. 종목별·전체 정량 비교는 다음 표에 있으며 영상별 판독은 각 모드 `lifter_review.json`에 있다. 표본에서 잘못 고른 영상은 M0 6개, lightweight 5개, balanced 2개다. balanced는 기존 오류 영상 6개 중 5개를 고쳤고, stu10_11은 운동 중 오류를 고쳤으나 운동 전 다른 사람을 고르는 오류가 남았다. 포즈 누락과 가려진 관절의 부정확한 위치는 이 사람 선택 오류 수와 별개다.

balanced의 원점수는 1,249,404개, 최소 0.021491, 최대 1.166933이고 영상별 p50 범위는 0.315878~0.821594다. lightweight와 같은 `clip(score / 0.6, 0, 1)`을 적용했다. 전체 실행은 15,402.074초, 처리 합계는 15,335.185초다. 속도 표의 사람 검출과 포즈 시간은 따로 측정한 값이다.

**CPU 운영 기본값은 lightweight를 제안하고 현재 설정도 lightweight로 뒀다.** balanced보다 전체 처리가 4.48배 빠르고(5.64 대 1.26 fps), M0보다 표본 리프터 오류와 몸통 점프가 줄었다. 다만 벤치 coverage·관절 지원과 사람 선택은 balanced가 더 좋으므로, 처리 시간을 감수하는 오프라인 분석에는 `--mode balanced`를 권한다. 두 모드 모두 이 CPU·단일 스레드 조건에서 실시간 속도는 아니다. balanced의 사람 검출만 약 11,917초여서 현재 속도 병목도 명확하다. 이후 최적화는 매 프레임 실행 조건을 바꾸지 않는 별도 실험으로 검증해야 한다.

홀짝 거짓 차이는 M0 0/43칸, lightweight 0/32칸, balanced 0/35칸이다. 특히 벤치는 판정 분모가 18 → 7/10칸으로 줄었다. 손가락 누락의 간접 영향과 낮은 관절 지원이 함께 있으므로 0건을 거짓 차이 성능 향상으로 주장하지 않는다. 흔들림도 칸·종목에 따라 개선과 악화가 섞여 있으며, 측정 가능한 영상 수를 표에 함께 썼다.

`eval.m1_compare` 재검증에서 두 모드 모두 69개 원판 기록·전체 결과 값(실행 시간 필드 두 개 제외), CFR 원판 결과, 입력 해시, 이미지 표본 프레임이 M0와 같았다. 횟수 61개의 전체 레코드도 같다. 현재 생산자·어댑터·설정 SHA256이 실제 측정 시 fingerprint와 모드별 69개 전부 같음을 확인했다. 기존 완료 산출물의 측정 시간은 보존했고 추론을 다시 실행한 시간으로 바꾸지 않았다.

원판 코어를 M0/M1에서 각각 import한 90조건(종목 3 × fps 3 × 잡음 5 × seed 2) 합성 회귀 JSON도 바이트 단위로 같았다. SHA256은 `5ccb08f4ab2342e6861cd271d8045dc108bf4f7ea9ae01d2ca2773ebb1499a6e`다.

시험 명령과 결과:

```bash
cd /home/kojy1009/work/m1-rtmpose/plate_pipeline
python -m pytest -q tests eval/midterm/test_metrics.py
# 145 passed in 9.24s

cd /home/kojy1009/work/m1-rtmpose
PATH=/home/kojy1009/miniconda3/lib/python3.14/site-packages/playwright/driver:$PATH python -m pytest -q tests/fb
# 235 passed in 2.77s
```

어댑터·점수 임계·벤치 누운 사람 우선/45도 경계·연속성의 절대/상대 개선 조건·검출된 세 번째 사람 선택·빈 검출·백엔드/모드 라우팅·캐시 구분·손가락 누락에 따른 SideLock/손잡이/판정 기준 rep 변화의 합성 시험이 포함된다.

최종 보고서는 `plate_pipeline/results/m1-rtmpose/REPORT.md`, 전체 패키지는 `plate_pipeline/eval_out/baseline_rtm/m1-rtmpose.zip`이다. zip에는 두 모드 json·md, 138개 확인 이미지, 추가 좌표 확인 이미지 2개, 영상별 검토 기록, 원판 불변 검증 JSON과 합성 회귀 JSON이 들어 있다. 모델·영상·eval_out은 커밋하지 않는다.
