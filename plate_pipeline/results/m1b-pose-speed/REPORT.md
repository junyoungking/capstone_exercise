# M1b 포즈 속도·정확도 조합 결과

완료 기준(리프터 오류 ≤2개, 1스레드 ≥4 fps, 원판 불변)을 함께 통과한 설정: **없음**. 기본 모드: **lightweight (unchanged)**.

원판 69개 기록·결과(실행 시간 두 필드 제외), 횟수 61개가 M0와 M1 balanced 양쪽에 일치한다. 준비 영상 SHA256과 확인 이미지 표본 프레임도 모두 같다. 새 확인 이미지 292장(4×69 + 벤치 대조 16장)을 검토했다.

|설정|표본 오류 영상|coverage|1스레드 fps|
|---|---:|---:|---:|
|M0|6|0.9680|10.0212|
|M1 lightweight|5|0.9743|5.6424|
|M1 balanced|2|0.9935|1.2606|
|n3|3|0.9601|2.6471|
|n5|3|0.9613|3.4809|
|n10|3|0.9617|4.2610|
|tiny_m|6|0.8985|4.3033|

## 작업 범위와 재현 조건

기준은 `plate-pipeline(69102a0)`, 작업 브랜치는 `integ/m1b-pose-speed`다. PLAN.md의 M1 결과를 읽었고 PLAN.md와 main은 수정하지 않았다. 모델·영상·eval_out은 새로 커밋하지 않는다. 원판 검출·추적·신호·세트·rep·깊이·속도 계산 코드는 바꾸지 않았다.

## 구현

rtmlib PoseTracker를 쓰지 않고 기존 BodyWithFeet 위에 직접 상태를 추가했다. `PoseCfg.mode="tracked"`, `det_frequency=3/5/10`은 YOLOX-m + RTMPose-m이다. 검출 프레임은 모든 검출 사람에게 포즈를 실행하고 M1의 손목 거리·벤치 누운 사람 우선·연속성 규칙으로 선택한다. 마지막 검출에서 N프레임 뒤 다시 검출한다. 추적을 잃으면 N과 관계없이 바로 다음 프레임에 검출한다.

사이 프레임은 이전 프레임에서 선택한 raw MP33 점 중 visibility ≥0.3이고 유한·화면 안인 점들의 최소 상자를 만든다. 중심 기준 가로·세로를 각각 **1.25배** 넓히고 화면 경계로 자른다(퇴화 축은 최소 2픽셀). 이 상자 하나에만 RTMPose-m을 실행한다. rtmlib RTMPose 자체도 입력 bbox에 padding=1.25와 종횡비 맞춤을 적용한다. 따라서 화면 절단·종횡비 맞춤이 없는 경우 우리 확장과 내부 padding의 결합은 축마다 1.5625배다. 검출 모델의 상자를 여러 프레임 고정해서 쓰지 않는다.

추적을 잃는 조건은 선택 실패, 몸통 네 점(양 어깨·양 엉덩이)의 비유한 값 또는 visibility <0.3, 유효 손목 없음, 손목–원판 중심 거리 > 원판 높이 4배, 유효 상자를 만들 수 없음이다. 손목은 기존 visibility ≥0.5·화면 안 검사다. 원판을 관측하지 못한 프레임에는 손목 거리 검사를 할 수 없으므로 몸통·상자 조건만 적용한다. 현재 프레임의 관절 지원이 부족하면 다음 프레임 검출을 예약하며, 기존 feedback의 관절별 품질 검사는 현재 프레임에도 그대로 적용한다. 검출 0명일 때 rtmlib의 전체 화면 포즈 fallback은 쓰지 않는다.

`tiny_m`은 rtmlib BodyWithFeet의 det만 lightweight의 YOLOX-tiny(416×416), pose는 balanced의 RTMPose-m(192×256)으로 명시한다. 매 프레임 모든 검출 사람에게 포즈를 실행한다.

벤치 guard는 원판 분석이 끝난 준비 영상의 **완료 rep 구간**에서 적용한다. 몸통 네 점 visibility ≥0.3, 어깨 중점–엉덩이 중점 벡터가 수직에서 45도 초과인 후보가 없으면 비운다. rep 밖은 M1 규칙이다. `feedback.plate.plate_rep_mask`가 원판 start/back을 같은 프레임 시계에 대응시키며 포즈 결과로 rep 구간을 바꾸지 않는다. `run_demo.py`와 eval 생산자에 이 mask를 전달한다. `select_lifter(..., in_rep=...)`는 rep 여부를 명시적으로 받는다. S1의 단일 패스 `server.jobs --pose` 진단은 최종 rep 구간을 알기 전이라 기존 선택을 기록하며, rep guard는 원판 분석 뒤의 실제 feedback 추출에서 적용된다. CLI와 feedback 양쪽에서 tracked 상자 갱신을 지원한다.

손가락 지원은 `pose['has_fingers']`로 전달·캐시한다. RTMPose Halpe26은 False, MediaPipe와 기존 메타데이터 없는 외부 포즈는 True다. 점이 일시적으로 안 보이는 것을 구조적인 미지원으로 오인하지 않는다. `bar_proxy_conf`의 필요 관절만 False일 때 `[15,16]`으로 바꾼다. 손잡이 계산·지표 값·기준 rep 가용 비율의 분모·50% 기준·SideLock은 유지한다. 그래서 손목이 실제로 관측된 경우 이 보조 지표가 다시 가용해지고 자격 비율을 회복한다. 두 손목 중 하나가 가려지거나 화면 밖이면 여전히 unavailable이다. 기존 M1 손가락 누락 부작용 시험은 명시적인 구 정책(True)에서 유지하고, 새 정책(False)과 MediaPipe 기본 정책의 동일성을 추가 시험했다.

점수는 M1과 같은 `clip(score/0.6,0,1)`이다. 어댑터·없는 점의 visibility 0·2D z=0·기존 EMA도 같다. 3D를 추가하지 않았다.

## 평가 조건과 해석

M0/M1의 고정된 69개 영상·61개 횟수·동일 표본 프레임·지표를 사용한다. (a)는 기존 M1 balanced 측정값을 재사용한다. (b) tracked N=3/5/10, (c) tiny_m은 원판부터 포즈까지 새로 실행한다. 벤치 guard의 효과는 balanced의 벤치 16개만 새로 실행해 M1 balanced의 같은 16개와 따로 비교한다. 손가락 변경은 같은 포즈를 고정한 상태에서 전후 필요 관절을 바꿔 비교하므로 검출기·사람 선택 효과와 구분한다.

최종 비교 입력은 `--prepared-reference`로 M1 balanced의 준비 영상을 그대로 사용한다. M0/M1 준비 영상 SHA256도 69개 모두 같음을 확인했다. 새 설정은 ORT intra-op 1, inter-op 1, OpenCV 1스레드다. 설정별 영상은 한 번에 하나씩 처리한다. 다섯 평가 작업을 서로 다른 물리 코어 CPU 0/1/2/3/4에 고정해 동시에 실행했다(SMT 형제 코어를 서로 배정하지 않음). M0/M1은 이전 단독 실행값이므로 동시 작업 부하·주파수·메모리 대역폭 차이가 있을 수 있다. 이 조건을 숨기거나 기존 측정 시간을 새 실행 시간으로 바꾸지 않는다. 완료 기준의 fps는 새 설정의 실제 원판+포즈 처리 시간으로 계산하고, 원판 시간을 다른 실행에서 가져와 더하지 않는다.

추가 6스레드 시험은 모든 1스레드 평가가 끝난 뒤 같은 기계에서 설정별로 순차·단독 실행한다. ORT intra-op 6, inter-op 1, OpenCV 1이다. 원본 `benchpress/stu10_11.mp4`를 fps=30으로 변환하고 끝을 clone padding해 정확히 60초·1,800프레임으로 만들었다. ffmpeg 변환 시간은 제외하고, 모델 준비·원판 분석/저장·포즈 추출/저장까지의 시간과 지표 계산까지 포함한 전체 시간을 모두 낸다. 완료 rep이 없는 경우도 원판·포즈는 영상 전체를 처리한다. 새로운 모델 추론은 수행하며 기존 포즈를 재생해 속도를 측정하지 않는다.

포즈 coverage의 분모는 완료 rep 구간 합집합이고 관절은 raw visibility ≥0.5다. 몸통 점프는 연속 프레임의 중심 이동 > 원판 높이인 자동 신호이며 실제 identity 오류와 다르다. 이미지 오류 수는 각 영상 여섯 표본의 하한이다. 포즈 누락·관절 위치 오차는 잘못된 사람 선택과 구분한다. 홀짝 거짓 차이는 판정 칸 수와 함께 읽어야 하며, 다른 관절 가용성에서 나온 SD 감소가 곧 위치 정답률 향상을 뜻하지 않는다.

원판 불변 검사는 M0와 M1 balanced 각각을 기준으로 69개 원판 rows를 gzip 해제 후 바이트 비교하고 result.json의 실행 시간 두 필드(`recog_s`, `recog_fps`)를 제외한 모든 값을 동일 순서로 직렬화해 바이트 비교한다. CFR용 pose_plate도 검사한다. 횟수 61개는 이벤트·지연을 포함한 전체 레코드를 대조한다. 원본 영상·원판 모델 SHA256과 확인 이미지 표본 프레임도 검사한다. 이 변경은 관절 필요 조건을 바꾸므로 지표 코드 해시가 M0와 같다고 주장하지 않는다.

## 재현

```bash
cd plate_pipeline
M1_BAL=/home/kojy1009/work/m1-rtmpose/plate_pipeline/eval_out/baseline_rtm/balanced
# 서로 다른 코어에서 실행한 실제 1스레드 명령
 taskset -c 0 python -m eval.baseline --out eval_out/m1b/n3 --mode tracked --det-frequency 3 --prepared-reference "$M1_BAL" --resume
 taskset -c 1 python -m eval.baseline --out eval_out/m1b/n5 --mode tracked --det-frequency 5 --prepared-reference "$M1_BAL" --resume
 taskset -c 2 python -m eval.baseline --out eval_out/m1b/n10 --mode tracked --det-frequency 10 --prepared-reference "$M1_BAL" --resume
 taskset -c 3 python -m eval.baseline --out eval_out/m1b/tiny_m --mode tiny_m --prepared-reference "$M1_BAL" --resume
 taskset -c 4 python -m eval.baseline --out eval_out/m1b/bench_guard --mode balanced --exercise-filter benchpress --prepared-reference "$M1_BAL" --resume
# bench guard를 끄는 대조 실행도 지원
python -m eval.baseline --out eval_out/control --mode balanced --no-bench-rep-guard
# 입력 영상 준비 (추론 시간에는 이 변환을 포함하지 않음)
ffmpeg -i /home/kojy1009/capstone_exercise/new_data/new_dataset/benchpress/stu10_11.mp4 \
  -vf 'fps=30,tpad=stop_mode=clone:stop_duration=1' -t 60 -an \
  -c:v libx264 -preset fast -threads 1 eval_out/m1b/benchmark_60s30.mp4
# 6스레드 단독 실행, 정확히 60초/1800프레임인지 assert
python -m eval.m1b_benchmark --video eval_out/m1b/benchmark_60s30.mp4 \
  --model /home/kojy1009/capstone_exercise/plate_pipeline/models/plate_tiny.onnx \
  --out eval_out/m1b/threads6 --threads 6
```

같은 포즈의 손가락 처리 전후는 `python -m eval.m1b_audit --source DIR --out FILE.json`으로 재생할 수 있다. `--base M0_DIR`를 더하면 원판 불변 검사다. 확인 이미지를 모두 검토한 `lifter_review.json`이 있어야 `eval.m1b_report`가 최종 표·REPORT·zip을 생성한다.

## 환경

M1과 같은 rtmlib 0.0.16, onnxruntime 1.31.0, MediaPipe 1.0.1, NumPy 2.5.3, PyAV 19.0.2, OpenCV 5.0.0.93, AMD Ryzen 5 7500F다. 사용한 파일은 저장소 밖 rtmlib 캐시에 있다.

- YOLOX-m: `yolox_m_8xb8-300e_humanart-c2c7a14a.onnx`
- YOLOX-tiny: `yolox_tiny_8xb8-300e_humanart-6f3252f9.onnx`
- RTMPose-m: `rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.onnx`
- 비교용 RTMPose-s: `rtmpose-s_simcc-body7_pt-body7-halpe26_700e-256x192-7f134165_20230605.onnx`

모델 SHA256은 M1 REPORT의 동일 파일과 같다. 원판 모델은 기존 plate_tiny.onnx다. 새로운 모델 파일·영상·eval_out을 커밋하지 않는다.

## 기본값 판단

요청한 새 설정 네 개 모두 같은 표본 검토에서 `stu10_11`, `val1235`, `stu6_11_2`의 최소 세 영상에 잘못된 리프터가 있다. tiny_m은 `test153`에서도 오류가 있다. 따라서 속도가 4 fps를 넘더라도 ≤2개 조건을 만족하지 못하며, **PoseCfg.mode는 기존 lightweight를 유지한다**. 이는 lightweight가 balanced보다 정확하다는 주장이 아니라 요청한 fallback 정책이다. 원판 결과를 바꾸거나, 오류 표본을 제외하거나, rep 밖 오류를 집계에서 빼서 통과시키지 않았다. 최종 전체 오류 수·추가 오류는 아래 설정별 표와 목록을 따른다.

`stu6_11_2` f95는 M1 balanced에서 낮은 visibility지만 실제 리프터 좌표였고, 네 새 조합에서는 배경의 파란 옷 사람을 선택했다. tracked는 관절 상자가 가까운 배경 사람에게 이동한 뒤 가짜로 안정적인 몸통/손목 지원을 얻으면 N까지 따라갈 수 있다. 손목 거리·몸통 visibility만으로 identity를 보장하지 못한다. N=5의 `stu10_11` f1438처럼 기존 오류 영상 안의 새 오류 프레임도 별도로 남겼다.

벤치 guard는 선 보조자 대신 빈 프레임을 만들지만, 카메라 투영상 몸통이 수직에 가까운 누운 리프터나 가려진 몸통도 배제할 수 있다. `stu1_6_4`의 여러 설정에서 여섯 rep 표본이 전부 누락됐다. 비움으로 오류 수만 낮추는 효과와 실제 관절 관측 확보를 구분해야 한다. 최종 벤치 16개 전후 표의 coverage를 함께 읽는다.

## 시험

```bash
cd /home/kojy1009/work/m1b-pose-speed/plate_pipeline
python -m pytest -q tests eval/midterm/test_metrics.py
# 153 passed in 10.87s
cd /home/kojy1009/work/m1b-pose-speed
PATH=/home/kojy1009/miniconda3/lib/python3.14/site-packages/playwright/driver:$PATH python -m pytest -q tests/fb
# 235 passed in 2.85s
```

추가 합성 시험은 N=3/5/10의 검출 간격·누락 다음 프레임 재검출, 여러 검출 상자와 사이 프레임 단일 상자, 상자 1.25배 확장과 경계 자르기, 몸통/손목 누락, 벤치 rep 안 비움·밖 유지·MediaPipe 예외, 구조적 손가락 미지원과 실제 손목 가림 구분, proxy 복구 시 고정 지표 분모와 정확한 자격 비율 증가, MP 기본 정책 동일성을 다룬다. 기존 M1 어댑터·선택·SideLock·스무딩·캐시·원판 시험도 유지했다.

### 선이 숨은 포즈의 추가 좌표 확인

tiny_m `deadlift/LLK_S01_R03.mp4` f59는 valid=True지만 몸통/손목 점수가 낮아 기본 이미지에서 선이 거의 그려지지 않았다. 원시 코·어깨·엉덩이·손목 좌표를 원본에 표시하니 몸통·엉덩이가 앞의 리프터 몸통보다 뒤의 사람 쪽에 붙은 혼합 골격이라 새 오류로 기록했다. M0/M1과 SHA256이 같은 준비 입력으로 다시 추론한 뒤에도 이 현상을 좌표로 재확인했다. 추가 확인 이미지는 `tiny_m/details/deadlift__LLK_S01_R03__f59.jpg`이며 zip에 포함한다. tiny_m의 최종 오류 수에는 이 영상도 들어간다.

## 준비 영상 동일성 보강

첫 실행에서 코어 고정이 ffmpeg libx264의 자동 스레드 수까지 바꿔, 일부 CFR 사본의 압축 결과가 M0/M1과 달라졌다. 원본 영상 원판/횟수는 같았지만 보조 pose_plate 기록까지 같아야 하므로 이 산출물은 최종 비교에서 제외했다. M0와 M1 balanced의 준비 영상 SHA256은 69개 모두 같았으며, 최종 실행은 이 동일 파일을 `--prepared-reference`로 직접 읽는다. 포즈/원판 결과를 재생하는 옵션이 아니며 CFR 영상에서도 두 모델을 새로 추론한다. 최종 불변 검사에는 준비 영상 SHA256과 보고한 입력 해시의 일치도 추가했다.

중단 전 완료한 **미변환 원본** 영상은 입력 경로/해시, 모델·어댑터·생산자·설정·관절 품질·지표 코드 해시가 같은 경우에만 실제 측정값을 유지했다. 이 결과의 원래 `fingerprint`를 덮어쓰지 않고, 새 평가 준비 경로의 `cache_fingerprint`와 `reuse_provenance`를 따로 남겼다. CFR 사본으로 측정한 결과는 이 재사용에서 제외하고 새로 측정·검토했다. `baseline.json.wall_s`는 마지막 resume 실행 시간이고, 비교표의 전체 처리 시간/fps는 각 영상의 실제 측정 시간 합계다. 중단된 미완료 영상의 시간은 합계에 넣지 않았다.

tracked N=3/5/10의 `benchpress/stu8_6_2.mp4` f157도 valid=True이지만 선이 숨었다. 세 모드의 원시 좌표 확인에서 몸통·엉덩이는 누운 리프터에 붙었고, 손목은 가림 때문에 다리 쪽으로 잘못 추정됐다. 사람 선택 오류와 관절 위치 오류를 구분해 새 오류 영상으로 세지 않았다. 낮은 손목 visibility로 기존 품질 검사는 이 관절을 관측으로 채택하지 않는다. 세 추가 이미지는 각 모드 `details/benchpress__stu8_6_2__f157.jpg`에 있으며 zip에 포함한다.

## 최종 보고서·zip 재생성

```bash
cd /home/kojy1009/work/m1b-pose-speed/plate_pipeline
M0_DIR=/home/kojy1009/work/m0-baseline/plate_pipeline/eval_out/baseline_mp
M1_ROOT=/home/kojy1009/work/m1-rtmpose/plate_pipeline/eval_out/baseline_rtm
python -m eval.m1b_audit --m1-root "$M1_ROOT" --out eval_out/m1b/m1_reference_eligibility.json
# 모든 설정의 확인 이미지를 검토한 lifter_review.json과 6스레드 시험 완료 후
python -m eval.m1b_report --m0 "$M0_DIR" --m1 "$M1_ROOT" --runs eval_out/m1b \
  --notes eval/M1B_NOTES.md --report results/m1b-pose-speed/REPORT.md
```

최종 zip은 `eval_out/m1b/m1b-pose-speed.zip`이다. 설정별 json·md·확인 이미지와 추가 좌표 이미지, 영상별 검토 기록, M0/M1 각각에 대한 원판 불변 증거, 기준 rep 자격 전후, 6스레드 원시 시간을 포함한다. M0/M1의 비교용 json·md·확인 이미지도 함께 넣는다. 모델·영상·raw pose NPZ는 zip과 커밋에 넣지 않는다.

### 누운 후보 각도의 반례

tiny_m 팀 벤치 `team9/benchpress/stu9_10.mp4` f558은 화면 왼쪽 주변 사람·허위 몸통을 선택했다. raw 양 어깨 x가 약 -10픽셀로 화면 밖인데도 visibility는 0.37~0.43이고 어깨–엉덩이 벡터가 45도를 넘어 누운 후보로 통과했다. 실제 리프터는 오른쪽 아래 벤치에 누워 있다. 같은 영상 f401/f740은 guard로 비었고 f243/f904/f1085는 리프터다. 기하학적 누움 판별만으로 잘못된 사람·혼합 골격을 모두 막을 수 없음을 보여준다. 확인 이미지는 `tiny_m/details/team9__benchpress__stu9_10__f558.jpg`다. 이 영상도 tiny_m 오류에 포함한다.

N=5 실행은 60개 완료 뒤 SIGTERM으로 중단되어 남은 9개를 같은 설정으로 재개했다. 완료 영상의 모델/입력/코드 해시를 확인해 실제 측정 시간을 유지했고, 중단된 미완료 영상은 새로 측정했다. 6스레드 시험은 다섯 설정의 최종 결과 파일과 모든 1스레드 프로세스 종료를 확인한 뒤 시작하도록 했다.

## 1스레드 최종 결과 해석

새 설정 네 개 모두 동일 표본의 오류 영상 수가 2개를 초과한다. N=3은 2.65 fps·오류 3개, N=5는 3.48 fps·오류 3개, N=10은 4.26 fps·오류 3개, tiny_m은 4.30 fps·오류 6개다. N=10/tiny_m은 속도 조건만 통과한다. 기본 모드는 지시대로 lightweight를 유지한다.

balanced의 벤치 16개 대조에서 표본 오류 영상은 1→1개이고 coverage는 98.18%→89.63%(-8.55%p)다. 몸통 점프는 14→4회지만 비운 프레임이 늘면 연속 프레임 점프도 줄 수 있으므로 안정성 향상이라고 단정하지 않는다. 특히 카메라 투영상 수직에 가까운 누운 사람과 가려진 몸통을 잃는 손실이 있다. tiny_m의 f558 반례는 이 규칙이 허위 몸통까지 모두 막지는 못함을 보여준다.

## 평가기 출처와 최종 캐시 정책

측정에 사용한 초기 평가기와 준비 입력을 고정한 평가기의 정확한 소스·SHA256을 `evaluator_sources.json`에 보존했다. 영상별 실제 측정 fingerprint도 유지했다. 최종 코드는 일반 resume에서 평가기 해시까지 일치하도록 엄격하게 검사한다. 준비 경로만 바꾸는 검증된 이행은 `--reuse-verified-originals`를 명시해야 하며, 원본 입력·모델·생산자·설정·관절 품질·지표 해시가 같아야 한다. 검증 없이 임의의 평가기 변경을 자동으로 허용하지 않는다.

최종 캐시 정책 변경 후 모델 속도를 새로 잰 것으로 표시하지 않는다. 캐시 분기와 CLI를 제외한 측정·계산 함수의 AST가 준비 입력 평가기와 같음을 확인했고, 그 최종 소스도 `measured: false`로 같은 출처 파일에 보존했다. 캐시 재사용 정책 외의 수치 코드는 같다. 추가 좌표 확인 이미지 다섯 장에는 raw MP33 좌표·픽셀 좌표·입력 해시를 담은 JSON을 함께 넣었다.


## 설정별 전체·종목별 비교

|범위|설정|오류 영상|coverage|점프|홀짝 다른/판정 칸|fps|
|---|---|---:|---:|---:|---:|---:|
|all|M0|6|0.9680|89|0/43|10.0212|
|all|M1 lightweight|5|0.9743|12|0/32|5.6424|
|all|M1 balanced|2|0.9935|21|0/35|1.2606|
|all|n3|3|0.9601|4|0/40|2.6471|
|all|n5|3|0.9613|7|0/40|3.4809|
|all|n10|3|0.9617|7|1/41|4.2610|
|all|tiny_m|6|0.8985|1|0/35|4.3033|
|squat|M0|2|0.9940|0|0/25|10.5654|
|squat|M1 lightweight|2|0.9956|0|0/25|6.2645|
|squat|M1 balanced|1|0.9996|3|0/25|1.3312|
|squat|n3|2|0.9996|1|0/25|2.7394|
|squat|n5|2|0.9996|1|0/25|3.4892|
|squat|n10|2|0.9996|1|0/25|4.1990|
|squat|tiny_m|2|0.9960|1|0/25|5.0724|
|benchpress|M0|3|0.9227|70|0/18|10.6207|
|benchpress|M1 lightweight|3|0.9287|9|0/7|5.7242|
|benchpress|M1 balanced|1|0.9818|14|0/10|1.2664|
|benchpress|n3|1|0.8829|2|0/15|2.6006|
|benchpress|n5|1|0.8863|2|0/15|3.3574|
|benchpress|n10|1|0.8872|2|1/16|3.9886|
|benchpress|tiny_m|3|0.7031|0|0/10|4.3855|
|deadlift|M0|1|0.9891|19|0/0|8.9595|
|deadlift|M1 lightweight|0|0.9987|3|0/0|5.1407|
|deadlift|M1 balanced|0|0.9992|4|0/0|1.2029|
|deadlift|n3|0|0.9992|1|0/0|2.6427|
|deadlift|n5|0|0.9995|4|0/0|3.6605|
|deadlift|n10|0|0.9995|4|0/0|4.7667|
|deadlift|tiny_m|1|0.9990|0|0/0|3.7592|

### 횟수 (원본 61개)

|범위|설정|MAE|exact|OBO|event F1|
|---|---|---:|---:|---:|---:|
|all|M0|0.8033|0.8033|0.9016|0.6632|
|all|M1 lightweight|0.8033|0.8033|0.9016|0.6632|
|all|M1 balanced|0.8033|0.8033|0.9016|0.6632|
|all|n3|0.8033|0.8033|0.9016|0.6632|
|all|n5|0.8033|0.8033|0.9016|0.6632|
|all|n10|0.8033|0.8033|0.9016|0.6632|
|all|tiny_m|0.8033|0.8033|0.9016|0.6632|
|squat|M0|1.1000|0.4000|0.8000|0.5600|
|squat|M1 lightweight|1.1000|0.4000|0.8000|0.5600|
|squat|M1 balanced|1.1000|0.4000|0.8000|0.5600|
|squat|n3|1.1000|0.4000|0.8000|0.5600|
|squat|n5|1.1000|0.4000|0.8000|0.5600|
|squat|n10|1.1000|0.4000|0.8000|0.5600|
|squat|tiny_m|1.1000|0.4000|0.8000|0.5600|
|benchpress|M0|2.5714|0.7143|0.7143|0.5652|
|benchpress|M1 lightweight|2.5714|0.7143|0.7143|0.5652|
|benchpress|M1 balanced|2.5714|0.7143|0.7143|0.5652|
|benchpress|n3|2.5714|0.7143|0.7143|0.5652|
|benchpress|n5|2.5714|0.7143|0.7143|0.5652|
|benchpress|n10|2.5714|0.7143|0.7143|0.5652|
|benchpress|tiny_m|2.5714|0.7143|0.7143|0.5652|
|deadlift|M0|0.0541|0.9459|1.0000|0.8158|
|deadlift|M1 lightweight|0.0541|0.9459|1.0000|0.8158|
|deadlift|M1 balanced|0.0541|0.9459|1.0000|0.8158|
|deadlift|n3|0.0541|0.9459|1.0000|0.8158|
|deadlift|n5|0.0541|0.9459|1.0000|0.8158|
|deadlift|n10|0.0541|0.9459|1.0000|0.8158|
|deadlift|tiny_m|0.0541|0.9459|1.0000|0.8158|

### 단계별 CPU 시간 (초)

|범위|설정|디코드|원판 검출|사람 검출|KLT|포즈|나머지|합계|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|all|M0|26.8357|1425.0710|N/A|98.8045|350.0840|28.2074|1929.0026|
|all|M1 lightweight|33.4355|1449.8043|1184.8633|97.9308|639.7896|20.2107|3426.0342|
|all|M1 balanced|34.3849|1445.3432|11916.5787|97.9474|1814.8867|26.0446|15335.1854|
|all|n3|79.4405|1428.9136|4520.0561|118.1548|1125.0076|31.1384|7302.7110|
|all|n5|76.9300|1388.3227|2977.8029|116.6216|963.1050|30.6933|5553.4755|
|all|n10|81.6826|1438.2136|1969.9573|125.0387|890.7831|31.0940|4536.7693|
|all|tiny_m|82.2761|1438.8067|1159.2833|127.7774|1658.4794|25.4773|4492.1002|
|squat|M0|6.6880|329.0937|N/A|23.0289|87.6157|6.0888|452.5151|
|squat|M1 lightweight|8.2445|333.7292|293.2490|23.7933|99.7599|4.4128|763.1887|
|squat|M1 balanced|8.8291|334.1445|2961.1303|23.8114|258.0244|5.5346|3591.4743|
|squat|n3|21.1521|325.8293|1137.8227|26.3807|227.4869|6.5752|1745.2469|
|squat|n5|20.4550|320.9345|774.6638|26.8413|220.2028|7.1196|1370.2170|
|squat|n10|22.2195|334.5861|520.9783|30.3154|223.4125|7.0840|1138.5958|
|squat|tiny_m|22.2827|335.5502|290.3310|30.5278|258.3405|5.5173|942.5495|
|benchpress|M0|10.7865|590.8027|N/A|36.0341|149.4134|8.3878|795.4245|
|benchpress|M1 lightweight|13.4959|601.2575|517.6321|36.7849|299.6067|7.0623|1475.8395|
|benchpress|M1 balanced|14.0052|600.5049|5208.2450|36.6756|802.9054|8.4263|6670.7625|
|benchpress|n3|32.2091|581.9506|2097.2839|44.9099|482.7706|9.3800|3248.5040|
|benchpress|n5|31.1838|565.9415|1452.1603|44.7527|413.0142|9.2008|2516.2533|
|benchpress|n10|32.9292|585.3057|1060.6957|47.6020|382.1188|9.3826|2118.0341|
|benchpress|tiny_m|33.2569|583.8305|494.5312|49.0268|757.6326|8.0723|1926.3504|
|deadlift|M0|9.3612|505.1745|N/A|39.7416|113.0550|13.7308|681.0630|
|deadlift|M1 lightweight|11.6951|514.8175|373.9822|37.3526|240.4229|8.7356|1187.0059|
|deadlift|M1 balanced|11.5506|510.6937|3747.2034|37.4603|753.9569|12.0836|5072.9485|
|deadlift|n3|26.0794|521.1337|1284.9495|46.8642|414.7501|15.1833|2308.9601|
|deadlift|n5|25.2913|501.4467|750.9788|45.0276|329.8880|14.3729|1667.0052|
|deadlift|n10|26.5339|518.3218|388.2833|47.1213|285.2518|14.6273|1280.1395|
|deadlift|tiny_m|26.7366|519.4259|374.4211|48.2228|642.5063|11.8877|1623.2003|

### 관절 묶음 visibility ≥0.5

|범위|설정|어깨|팔꿈치|손목|엉덩이|무릎|발목|발|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|all|M0|0.9680|0.6217|0.5891|0.9676|0.7327|0.7340|0.7778|
|all|M1 lightweight|0.8585|0.8361|0.7505|0.8626|0.8732|0.8108|0.7618|
|all|M1 balanced|0.9490|0.8985|0.8061|0.9712|0.9436|0.9107|0.8394|
|all|n3|0.9250|0.8738|0.7836|0.9407|0.9156|0.8877|0.8155|
|all|n5|0.9268|0.8761|0.7857|0.9415|0.9147|0.8860|0.8139|
|all|n10|0.9283|0.8784|0.7881|0.9423|0.9154|0.8833|0.8125|
|all|tiny_m|0.8764|0.8302|0.7276|0.8844|0.8593|0.8188|0.7377|
|squat|M0|0.9940|0.6308|0.4933|0.9937|0.8825|0.8708|0.9008|
|squat|M1 lightweight|0.9804|0.8732|0.6494|0.9567|0.9185|0.9109|0.8997|
|squat|M1 balanced|0.9966|0.8986|0.6478|0.9776|0.9571|0.9812|0.9575|
|squat|n3|0.9964|0.8998|0.6385|0.9760|0.9556|0.9843|0.9621|
|squat|n5|0.9964|0.9040|0.6401|0.9774|0.9560|0.9841|0.9633|
|squat|n10|0.9964|0.9093|0.6442|0.9776|0.9597|0.9843|0.9669|
|squat|tiny_m|0.9940|0.9169|0.5839|0.9782|0.9452|0.9540|0.9315|
|benchpress|M0|0.9227|0.6562|0.6729|0.9219|0.7164|0.6090|0.6410|
|benchpress|M1 lightweight|0.5980|0.6192|0.5934|0.6300|0.7073|0.5927|0.5135|
|benchpress|M1 balanced|0.8544|0.7845|0.7463|0.9360|0.8702|0.7555|0.5954|
|benchpress|n3|0.7822|0.7076|0.6764|0.8446|0.7859|0.6814|0.5090|
|benchpress|n5|0.7871|0.7102|0.6788|0.8455|0.7827|0.6767|0.5017|
|benchpress|n10|0.7910|0.7123|0.6808|0.8476|0.7815|0.6688|0.4923|
|benchpress|tiny_m|0.6401|0.5748|0.5728|0.6764|0.6333|0.5587|0.4034|
|deadlift|M0|0.9891|0.5872|0.5804|0.9891|0.6511|0.7510|0.8134|
|deadlift|M1 lightweight|0.9977|0.9931|0.9456|0.9962|0.9825|0.9287|0.8807|
|deadlift|M1 balanced|0.9975|0.9933|0.9565|0.9963|0.9961|0.9950|0.9674|
|deadlift|n3|0.9984|0.9956|0.9652|0.9982|0.9982|0.9979|0.9773|
|deadlift|n5|0.9989|0.9963|0.9672|0.9985|0.9984|0.9977|0.9787|
|deadlift|n10|0.9991|0.9968|0.9690|0.9987|0.9986|0.9975|0.9808|
|deadlift|tiny_m|0.9981|0.9878|0.9479|0.9979|0.9928|0.9494|0.8926|

### Rep 간 흔들림 (도)

M0와 같은 영상별 SD(ddof=1)와 모든 rep 쌍 절대 차이 중앙값, 그 뒤 범위별 영상 중앙값이다. n은 표본 2개 이상인 영상 수다. 포즈 가용성에 따라 모집단이 다르므로 SD만으로 정확도를 주장하지 않는다.

|범위|포지션|지표|설정|n|SD|쌍 차이|
|---|---|---|---|---:|---:|---:|
|all|START|knee_angle|M0|17|5.9050|7.0232|
|all|START|hip_angle|M0|20|9.5716|11.2919|
|all|START|trunk_lean|M0|20|2.9516|3.2243|
|all|MID_DOWN|knee_angle|M0|14|2.8398|4.1542|
|all|MID_DOWN|hip_angle|M0|17|5.5200|7.1822|
|all|MID_DOWN|trunk_lean|M0|18|2.9991|3.7890|
|all|BOTTOM|knee_angle|M0|14|4.7140|4.9279|
|all|BOTTOM|hip_angle|M0|15|4.7481|4.8028|
|all|BOTTOM|trunk_lean|M0|17|1.9257|2.5356|
|all|MID_UP|knee_angle|M0|15|6.2705|7.0176|
|all|MID_UP|hip_angle|M0|18|9.2538|12.5808|
|all|MID_UP|trunk_lean|M0|19|3.2903|4.6532|
|all|LOCKOUT|knee_angle|M0|15|8.4510|9.6208|
|all|LOCKOUT|hip_angle|M0|19|9.1792|11.1064|
|all|LOCKOUT|trunk_lean|M0|19|2.5986|2.5280|
|all|START|knee_angle|M1 lightweight|16|7.2500|8.2885|
|all|START|hip_angle|M1 lightweight|17|5.6603|6.1873|
|all|START|trunk_lean|M1 lightweight|16|1.8547|2.2855|
|all|MID_DOWN|knee_angle|M1 lightweight|12|3.6804|5.5240|
|all|MID_DOWN|hip_angle|M1 lightweight|14|4.6450|5.8095|
|all|MID_DOWN|trunk_lean|M1 lightweight|14|2.4110|3.0338|
|all|BOTTOM|knee_angle|M1 lightweight|10|1.8714|2.3998|
|all|BOTTOM|hip_angle|M1 lightweight|11|1.4276|1.7443|
|all|BOTTOM|trunk_lean|M1 lightweight|11|1.1303|1.4287|
|all|MID_UP|knee_angle|M1 lightweight|12|2.6489|3.1676|
|all|MID_UP|hip_angle|M1 lightweight|14|2.7937|3.2693|
|all|MID_UP|trunk_lean|M1 lightweight|13|1.4918|1.9541|
|all|LOCKOUT|knee_angle|M1 lightweight|16|6.4811|8.9264|
|all|LOCKOUT|hip_angle|M1 lightweight|16|5.7970|7.3179|
|all|LOCKOUT|trunk_lean|M1 lightweight|15|3.9573|4.4693|
|all|START|knee_angle|M1 balanced|16|6.0235|8.9575|
|all|START|hip_angle|M1 balanced|18|5.8278|6.9504|
|all|START|trunk_lean|M1 balanced|19|1.8358|2.1784|
|all|MID_DOWN|knee_angle|M1 balanced|16|2.7824|3.8648|
|all|MID_DOWN|hip_angle|M1 balanced|18|4.4408|6.2299|
|all|MID_DOWN|trunk_lean|M1 balanced|18|1.6523|2.3367|
|all|BOTTOM|knee_angle|M1 balanced|16|3.0669|4.1986|
|all|BOTTOM|hip_angle|M1 balanced|16|3.1223|4.2819|
|all|BOTTOM|trunk_lean|M1 balanced|16|1.2967|1.6438|
|all|MID_UP|knee_angle|M1 balanced|16|3.8653|5.4664|
|all|MID_UP|hip_angle|M1 balanced|18|6.5613|8.2839|
|all|MID_UP|trunk_lean|M1 balanced|17|3.3601|3.2042|
|all|LOCKOUT|knee_angle|M1 balanced|16|7.0537|9.3368|
|all|LOCKOUT|hip_angle|M1 balanced|19|7.0928|8.8442|
|all|LOCKOUT|trunk_lean|M1 balanced|20|1.8926|2.6110|
|all|START|knee_angle|n3|14|5.1797|6.9253|
|all|START|hip_angle|n3|18|4.7523|4.9688|
|all|START|trunk_lean|n3|19|1.7229|2.3627|
|all|MID_DOWN|knee_angle|n3|15|3.3383|4.7211|
|all|MID_DOWN|hip_angle|n3|18|4.1001|5.2189|
|all|MID_DOWN|trunk_lean|n3|18|1.3265|1.2946|
|all|BOTTOM|knee_angle|n3|15|3.4266|4.3083|
|all|BOTTOM|hip_angle|n3|15|3.1720|4.3157|
|all|BOTTOM|trunk_lean|n3|15|1.8020|2.2502|
|all|MID_UP|knee_angle|n3|14|3.2976|3.7876|
|all|MID_UP|hip_angle|n3|17|7.1171|10.0651|
|all|MID_UP|trunk_lean|n3|17|3.2361|3.4635|
|all|LOCKOUT|knee_angle|n3|17|6.1096|8.5256|
|all|LOCKOUT|hip_angle|n3|20|5.8795|5.8520|
|all|LOCKOUT|trunk_lean|n3|20|2.8197|3.3279|
|all|START|knee_angle|n5|16|5.7625|7.3635|
|all|START|hip_angle|n5|19|5.2915|5.7183|
|all|START|trunk_lean|n5|19|2.0247|1.7096|
|all|MID_DOWN|knee_angle|n5|15|2.9928|4.2324|
|all|MID_DOWN|hip_angle|n5|18|3.1999|4.2558|
|all|MID_DOWN|trunk_lean|n5|18|1.4930|2.1010|
|all|BOTTOM|knee_angle|n5|15|2.6532|3.7521|
|all|BOTTOM|hip_angle|n5|15|3.3605|4.7525|
|all|BOTTOM|trunk_lean|n5|15|2.2110|2.2917|
|all|MID_UP|knee_angle|n5|14|3.2524|3.8808|
|all|MID_UP|hip_angle|n5|17|4.6742|6.6103|
|all|MID_UP|trunk_lean|n5|16|2.8528|3.1181|
|all|LOCKOUT|knee_angle|n5|17|5.5454|7.8453|
|all|LOCKOUT|hip_angle|n5|20|7.4931|9.4191|
|all|LOCKOUT|trunk_lean|n5|20|3.3153|4.0835|
|all|START|knee_angle|n10|16|5.5213|7.0043|
|all|START|hip_angle|n10|19|4.2310|5.0216|
|all|START|trunk_lean|n10|19|1.6471|1.7409|
|all|MID_DOWN|knee_angle|n10|15|3.0204|4.2715|
|all|MID_DOWN|hip_angle|n10|18|3.0087|3.9074|
|all|MID_DOWN|trunk_lean|n10|18|2.0516|2.4632|
|all|BOTTOM|knee_angle|n10|15|3.6693|4.6098|
|all|BOTTOM|hip_angle|n10|15|3.4075|4.5481|
|all|BOTTOM|trunk_lean|n10|15|2.2071|2.4554|
|all|MID_UP|knee_angle|n10|14|3.2699|4.6244|
|all|MID_UP|hip_angle|n10|17|4.3263|6.1182|
|all|MID_UP|trunk_lean|n10|16|2.6740|2.9961|
|all|LOCKOUT|knee_angle|n10|17|6.5562|9.2718|
|all|LOCKOUT|hip_angle|n10|20|5.9601|8.2107|
|all|LOCKOUT|trunk_lean|n10|20|3.2072|3.9734|
|all|START|knee_angle|tiny_m|17|4.9234|5.1395|
|all|START|hip_angle|tiny_m|18|3.5015|4.6531|
|all|START|trunk_lean|tiny_m|20|1.6371|2.1115|
|all|MID_DOWN|knee_angle|tiny_m|15|2.7293|3.9681|
|all|MID_DOWN|hip_angle|tiny_m|16|4.0104|5.0909|
|all|MID_DOWN|trunk_lean|tiny_m|18|2.5060|2.9176|
|all|BOTTOM|knee_angle|tiny_m|16|3.9777|5.0049|
|all|BOTTOM|hip_angle|tiny_m|16|3.1490|4.1489|
|all|BOTTOM|trunk_lean|tiny_m|15|1.5047|1.6567|
|all|MID_UP|knee_angle|tiny_m|15|3.1728|3.4986|
|all|MID_UP|hip_angle|tiny_m|16|2.9798|2.9939|
|all|MID_UP|trunk_lean|tiny_m|18|2.0879|2.0593|
|all|LOCKOUT|knee_angle|tiny_m|16|5.2766|7.1775|
|all|LOCKOUT|hip_angle|tiny_m|19|4.1826|5.9151|
|all|LOCKOUT|trunk_lean|tiny_m|20|2.5583|3.2345|
|squat|START|knee_angle|M0|6|0.7606|0.7885|
|squat|START|hip_angle|M0|6|3.6271|4.6204|
|squat|START|trunk_lean|M0|6|1.2379|1.8473|
|squat|MID_DOWN|knee_angle|M0|5|3.1776|4.4719|
|squat|MID_DOWN|hip_angle|M0|6|6.3149|8.3484|
|squat|MID_DOWN|trunk_lean|M0|6|3.1625|4.1428|
|squat|BOTTOM|knee_angle|M0|5|4.6795|6.1437|
|squat|BOTTOM|hip_angle|M0|6|7.1666|10.1352|
|squat|BOTTOM|trunk_lean|M0|6|0.9937|1.4053|
|squat|MID_UP|knee_angle|M0|6|5.4173|7.6612|
|squat|MID_UP|hip_angle|M0|6|6.2562|8.8476|
|squat|MID_UP|trunk_lean|M0|6|1.9351|2.6841|
|squat|LOCKOUT|knee_angle|M0|5|3.3066|2.9207|
|squat|LOCKOUT|hip_angle|M0|6|4.2221|4.6917|
|squat|LOCKOUT|trunk_lean|M0|6|0.4573|0.6054|
|squat|START|knee_angle|M1 lightweight|5|1.3843|1.5522|
|squat|START|hip_angle|M1 lightweight|5|2.3609|3.8983|
|squat|START|trunk_lean|M1 lightweight|5|0.8863|1.2535|
|squat|MID_DOWN|knee_angle|M1 lightweight|4|3.8308|5.6619|
|squat|MID_DOWN|hip_angle|M1 lightweight|5|2.6127|3.0905|
|squat|MID_DOWN|trunk_lean|M1 lightweight|5|2.3157|2.8821|
|squat|BOTTOM|knee_angle|M1 lightweight|4|1.8714|2.3897|
|squat|BOTTOM|hip_angle|M1 lightweight|5|1.7763|2.1230|
|squat|BOTTOM|trunk_lean|M1 lightweight|4|0.7818|1.2345|
|squat|MID_UP|knee_angle|M1 lightweight|4|2.6387|2.8790|
|squat|MID_UP|hip_angle|M1 lightweight|5|1.2722|2.1113|
|squat|MID_UP|trunk_lean|M1 lightweight|5|0.7644|1.0654|
|squat|LOCKOUT|knee_angle|M1 lightweight|5|2.1735|3.0738|
|squat|LOCKOUT|hip_angle|M1 lightweight|5|4.6939|6.6381|
|squat|LOCKOUT|trunk_lean|M1 lightweight|5|2.0432|2.8895|
|squat|START|knee_angle|M1 balanced|5|0.6222|0.8799|
|squat|START|hip_angle|M1 balanced|5|1.4039|1.5431|
|squat|START|trunk_lean|M1 balanced|5|1.3662|1.7980|
|squat|MID_DOWN|knee_angle|M1 balanced|5|2.6307|4.1495|
|squat|MID_DOWN|hip_angle|M1 balanced|5|2.6229|3.0620|
|squat|MID_DOWN|trunk_lean|M1 balanced|5|1.3449|1.9020|
|squat|BOTTOM|knee_angle|M1 balanced|5|3.4375|4.5839|
|squat|BOTTOM|hip_angle|M1 balanced|5|3.4522|4.5053|
|squat|BOTTOM|trunk_lean|M1 balanced|5|0.4012|0.5293|
|squat|MID_UP|knee_angle|M1 balanced|5|2.2923|2.8029|
|squat|MID_UP|hip_angle|M1 balanced|5|2.0965|3.3234|
|squat|MID_UP|trunk_lean|M1 balanced|5|0.6288|0.8893|
|squat|LOCKOUT|knee_angle|M1 balanced|5|3.8197|3.6095|
|squat|LOCKOUT|hip_angle|M1 balanced|6|6.7092|8.0306|
|squat|LOCKOUT|trunk_lean|M1 balanced|6|1.0372|1.4668|
|squat|START|knee_angle|n3|4|2.8665|2.9439|
|squat|START|hip_angle|n3|5|1.4220|1.8208|
|squat|START|trunk_lean|n3|5|1.1686|1.2949|
|squat|MID_DOWN|knee_angle|n3|5|2.5051|3.8770|
|squat|MID_DOWN|hip_angle|n3|5|2.5017|2.7661|
|squat|MID_DOWN|trunk_lean|n3|5|0.6652|0.8570|
|squat|BOTTOM|knee_angle|n3|5|3.4266|4.3083|
|squat|BOTTOM|hip_angle|n3|5|3.6819|4.4859|
|squat|BOTTOM|trunk_lean|n3|5|0.2888|0.2451|
|squat|MID_UP|knee_angle|n3|4|2.3383|1.9569|
|squat|MID_UP|hip_angle|n3|5|1.5911|1.9226|
|squat|MID_UP|trunk_lean|n3|5|0.6378|0.9019|
|squat|LOCKOUT|knee_angle|n3|5|4.0445|3.8250|
|squat|LOCKOUT|hip_angle|n3|6|7.0658|8.4180|
|squat|LOCKOUT|trunk_lean|n3|6|1.7472|2.4709|
|squat|START|knee_angle|n5|5|0.4886|0.6910|
|squat|START|hip_angle|n5|5|1.2491|1.7284|
|squat|START|trunk_lean|n5|5|0.9456|1.3093|
|squat|MID_DOWN|knee_angle|n5|5|2.2696|3.7691|
|squat|MID_DOWN|hip_angle|n5|5|2.6568|2.9939|
|squat|MID_DOWN|trunk_lean|n5|5|0.9096|1.2863|
|squat|BOTTOM|knee_angle|n5|5|3.7358|4.6501|
|squat|BOTTOM|hip_angle|n5|5|3.8576|4.7807|
|squat|BOTTOM|trunk_lean|n5|5|0.4367|0.2637|
|squat|MID_UP|knee_angle|n5|4|2.4494|1.9100|
|squat|MID_UP|hip_angle|n5|5|1.8984|2.4490|
|squat|MID_UP|trunk_lean|n5|5|0.7965|1.1264|
|squat|LOCKOUT|knee_angle|n5|5|4.2018|4.1184|
|squat|LOCKOUT|hip_angle|n5|6|7.6206|9.3529|
|squat|LOCKOUT|trunk_lean|n5|6|2.3101|2.9649|
|squat|START|knee_angle|n10|5|0.5119|0.5412|
|squat|START|hip_angle|n10|5|1.8622|1.8763|
|squat|START|trunk_lean|n10|5|1.2578|1.3045|
|squat|MID_DOWN|knee_angle|n10|5|2.1460|3.6145|
|squat|MID_DOWN|hip_angle|n10|5|2.4242|2.9669|
|squat|MID_DOWN|trunk_lean|n10|5|0.9534|1.3483|
|squat|BOTTOM|knee_angle|n10|5|3.6693|4.6098|
|squat|BOTTOM|hip_angle|n10|5|3.9626|4.8190|
|squat|BOTTOM|trunk_lean|n10|5|0.4081|0.2965|
|squat|MID_UP|knee_angle|n10|4|2.3631|2.1474|
|squat|MID_UP|hip_angle|n10|5|1.5172|2.2053|
|squat|MID_UP|trunk_lean|n10|5|0.8229|1.1637|
|squat|LOCKOUT|knee_angle|n10|5|4.2835|4.1678|
|squat|LOCKOUT|hip_angle|n10|6|7.7752|9.7991|
|squat|LOCKOUT|trunk_lean|n10|6|2.1595|3.0541|
|squat|START|knee_angle|tiny_m|6|2.6530|2.9626|
|squat|START|hip_angle|tiny_m|6|2.2579|3.1931|
|squat|START|trunk_lean|tiny_m|6|1.6371|2.0537|
|squat|MID_DOWN|knee_angle|tiny_m|5|3.0675|4.4903|
|squat|MID_DOWN|hip_angle|tiny_m|5|3.8166|4.2362|
|squat|MID_DOWN|trunk_lean|tiny_m|6|1.5156|1.8633|
|squat|BOTTOM|knee_angle|tiny_m|6|3.3330|4.0932|
|squat|BOTTOM|hip_angle|tiny_m|6|2.5145|3.0976|
|squat|BOTTOM|trunk_lean|tiny_m|5|0.4812|0.6729|
|squat|MID_UP|knee_angle|tiny_m|5|2.5266|3.1981|
|squat|MID_UP|hip_angle|tiny_m|5|2.3864|2.6728|
|squat|MID_UP|trunk_lean|tiny_m|5|0.8469|1.1977|
|squat|LOCKOUT|knee_angle|tiny_m|5|2.9945|3.6700|
|squat|LOCKOUT|hip_angle|tiny_m|6|3.0244|4.2771|
|squat|LOCKOUT|trunk_lean|tiny_m|6|1.7378|2.1962|
|benchpress|START|knee_angle|M0|7|13.7415|7.6291|
|benchpress|START|hip_angle|M0|10|9.5716|11.2919|
|benchpress|START|trunk_lean|M0|10|2.2492|2.9760|
|benchpress|MID_DOWN|knee_angle|M0|5|7.0124|7.3114|
|benchpress|MID_DOWN|hip_angle|M0|7|6.1329|7.1822|
|benchpress|MID_DOWN|trunk_lean|M0|8|4.2896|6.2174|
|benchpress|BOTTOM|knee_angle|M0|5|12.8786|4.6589|
|benchpress|BOTTOM|hip_angle|M0|5|6.6776|9.4435|
|benchpress|BOTTOM|trunk_lean|M0|7|4.6796|8.0480|
|benchpress|MID_UP|knee_angle|M0|5|9.1303|6.9546|
|benchpress|MID_UP|hip_angle|M0|8|16.0206|20.4219|
|benchpress|MID_UP|trunk_lean|M0|9|9.7582|11.6349|
|benchpress|LOCKOUT|knee_angle|M0|6|12.1403|8.7624|
|benchpress|LOCKOUT|hip_angle|M0|9|9.1792|11.1064|
|benchpress|LOCKOUT|trunk_lean|M0|9|2.6365|3.3809|
|benchpress|START|knee_angle|M1 lightweight|7|9.3981|8.9228|
|benchpress|START|hip_angle|M1 lightweight|8|8.1855|7.0961|
|benchpress|START|trunk_lean|M1 lightweight|7|1.6352|2.3125|
|benchpress|MID_DOWN|knee_angle|M1 lightweight|4|9.1766|10.5661|
|benchpress|MID_DOWN|hip_angle|M1 lightweight|5|8.5242|12.0551|
|benchpress|MID_DOWN|trunk_lean|M1 lightweight|5|2.5250|3.5709|
|benchpress|BOTTOM|knee_angle|M1 lightweight|2|19.9689|18.2261|
|benchpress|BOTTOM|hip_angle|M1 lightweight|2|10.3320|13.2611|
|benchpress|BOTTOM|trunk_lean|M1 lightweight|3|2.1278|3.0092|
|benchpress|MID_UP|knee_angle|M1 lightweight|4|12.0190|6.2944|
|benchpress|MID_UP|hip_angle|M1 lightweight|5|10.7830|15.2494|
|benchpress|MID_UP|trunk_lean|M1 lightweight|4|1.2771|1.6321|
|benchpress|LOCKOUT|knee_angle|M1 lightweight|7|19.5479|10.1977|
|benchpress|LOCKOUT|hip_angle|M1 lightweight|7|4.6299|6.5477|
|benchpress|LOCKOUT|trunk_lean|M1 lightweight|6|2.7145|3.4519|
|benchpress|START|knee_angle|M1 balanced|7|15.5665|21.6487|
|benchpress|START|hip_angle|M1 balanced|9|12.3416|14.7508|
|benchpress|START|trunk_lean|M1 balanced|10|1.6881|2.1007|
|benchpress|MID_DOWN|knee_angle|M1 balanced|7|10.1330|14.3302|
|benchpress|MID_DOWN|hip_angle|M1 balanced|9|9.1244|11.4706|
|benchpress|MID_DOWN|trunk_lean|M1 balanced|9|1.7920|2.5343|
|benchpress|BOTTOM|knee_angle|M1 balanced|7|12.1157|14.7436|
|benchpress|BOTTOM|hip_angle|M1 balanced|7|2.4819|3.2838|
|benchpress|BOTTOM|trunk_lean|M1 balanced|7|2.6491|3.6839|
|benchpress|MID_UP|knee_angle|M1 balanced|7|18.6317|27.6377|
|benchpress|MID_UP|hip_angle|M1 balanced|9|10.3645|10.7697|
|benchpress|MID_UP|trunk_lean|M1 balanced|8|5.3937|7.0821|
|benchpress|LOCKOUT|knee_angle|M1 balanced|7|16.5095|21.7740|
|benchpress|LOCKOUT|hip_angle|M1 balanced|9|3.5302|5.3619|
|benchpress|LOCKOUT|trunk_lean|M1 balanced|10|1.8143|2.2906|
|benchpress|START|knee_angle|n3|6|7.9135|6.9005|
|benchpress|START|hip_angle|n3|9|5.8581|8.3205|
|benchpress|START|trunk_lean|n3|10|1.9009|2.6883|
|benchpress|MID_DOWN|knee_angle|n3|6|5.5726|7.8809|
|benchpress|MID_DOWN|hip_angle|n3|9|6.0130|8.5036|
|benchpress|MID_DOWN|trunk_lean|n3|9|0.8683|1.0086|
|benchpress|BOTTOM|knee_angle|n3|6|12.5743|14.0283|
|benchpress|BOTTOM|hip_angle|n3|6|1.9679|3.0710|
|benchpress|BOTTOM|trunk_lean|n3|6|2.2531|3.2895|
|benchpress|MID_UP|knee_angle|n3|6|10.3859|16.1599|
|benchpress|MID_UP|hip_angle|n3|8|15.1392|18.1650|
|benchpress|MID_UP|trunk_lean|n3|8|5.9324|6.7050|
|benchpress|LOCKOUT|knee_angle|n3|8|9.8574|13.4093|
|benchpress|LOCKOUT|hip_angle|n3|10|3.2969|5.0985|
|benchpress|LOCKOUT|trunk_lean|n3|10|1.8442|2.1714|
|benchpress|START|knee_angle|n5|7|13.3713|9.0597|
|benchpress|START|hip_angle|n5|10|8.6382|12.4615|
|benchpress|START|trunk_lean|n5|10|1.8624|2.6339|
|benchpress|MID_DOWN|knee_angle|n5|6|5.9761|8.4514|
|benchpress|MID_DOWN|hip_angle|n5|9|2.6971|3.8143|
|benchpress|MID_DOWN|trunk_lean|n5|9|1.2892|1.8232|
|benchpress|BOTTOM|knee_angle|n5|6|7.8511|9.9339|
|benchpress|BOTTOM|hip_angle|n5|6|2.2554|3.9902|
|benchpress|BOTTOM|trunk_lean|n5|6|2.8189|4.1465|
|benchpress|MID_UP|knee_angle|n5|6|9.1900|11.8254|
|benchpress|MID_UP|hip_angle|n5|8|11.2281|17.0408|
|benchpress|MID_UP|trunk_lean|n5|7|5.8611|8.0088|
|benchpress|LOCKOUT|knee_angle|n5|8|5.2140|7.8502|
|benchpress|LOCKOUT|hip_angle|n5|10|4.2724|6.4188|
|benchpress|LOCKOUT|trunk_lean|n5|10|1.4413|1.9088|
|benchpress|START|knee_angle|n10|7|14.9460|10.1165|
|benchpress|START|hip_angle|n10|10|4.7778|7.3395|
|benchpress|START|trunk_lean|n10|10|1.2827|1.7401|
|benchpress|MID_DOWN|knee_angle|n10|6|6.9984|9.8972|
|benchpress|MID_DOWN|hip_angle|n10|9|2.7871|3.9415|
|benchpress|MID_DOWN|trunk_lean|n10|9|2.2931|3.2429|
|benchpress|BOTTOM|knee_angle|n10|6|12.0596|17.4235|
|benchpress|BOTTOM|hip_angle|n10|6|2.4869|3.8481|
|benchpress|BOTTOM|trunk_lean|n10|6|2.5855|3.5460|
|benchpress|MID_UP|knee_angle|n10|6|10.6862|12.3944|
|benchpress|MID_UP|hip_angle|n10|8|10.4771|14.0136|
|benchpress|MID_UP|trunk_lean|n10|7|6.2200|8.7083|
|benchpress|LOCKOUT|knee_angle|n10|8|5.9672|8.6678|
|benchpress|LOCKOUT|hip_angle|n10|10|2.3763|3.1574|
|benchpress|LOCKOUT|trunk_lean|n10|10|1.5748|2.1532|
|benchpress|START|knee_angle|tiny_m|7|2.9286|4.1416|
|benchpress|START|hip_angle|tiny_m|8|4.1319|5.3342|
|benchpress|START|trunk_lean|tiny_m|10|1.2529|1.7498|
|benchpress|MID_DOWN|knee_angle|tiny_m|6|7.8118|10.6308|
|benchpress|MID_DOWN|hip_angle|tiny_m|7|4.2042|5.9456|
|benchpress|MID_DOWN|trunk_lean|tiny_m|8|2.2671|3.2061|
|benchpress|BOTTOM|knee_angle|tiny_m|6|7.2128|10.0016|
|benchpress|BOTTOM|hip_angle|tiny_m|6|2.5465|3.5141|
|benchpress|BOTTOM|trunk_lean|tiny_m|6|2.1863|2.3791|
|benchpress|MID_UP|knee_angle|tiny_m|6|9.6889|11.1202|
|benchpress|MID_UP|hip_angle|tiny_m|7|2.1187|2.4533|
|benchpress|MID_UP|trunk_lean|tiny_m|9|3.6492|3.2995|
|benchpress|LOCKOUT|knee_angle|tiny_m|7|5.7329|7.5382|
|benchpress|LOCKOUT|hip_angle|tiny_m|9|2.7105|3.7664|
|benchpress|LOCKOUT|trunk_lean|tiny_m|10|1.6936|2.4954|
|deadlift|START|knee_angle|M0|4|7.3591|10.2379|
|deadlift|START|hip_angle|M0|4|18.1541|25.2172|
|deadlift|START|trunk_lean|M0|4|9.3000|13.0630|
|deadlift|MID_DOWN|knee_angle|M0|4|2.1742|2.9296|
|deadlift|MID_DOWN|hip_angle|M0|4|1.7373|2.5451|
|deadlift|MID_DOWN|trunk_lean|M0|4|2.7615|3.7890|
|deadlift|BOTTOM|knee_angle|M0|4|2.5765|3.4980|
|deadlift|BOTTOM|hip_angle|M0|4|1.6045|2.4720|
|deadlift|BOTTOM|trunk_lean|M0|4|1.3677|1.9511|
|deadlift|MID_UP|knee_angle|M0|4|4.9810|6.0274|
|deadlift|MID_UP|hip_angle|M0|4|6.6332|8.8746|
|deadlift|MID_UP|trunk_lean|M0|4|3.9113|4.4359|
|deadlift|LOCKOUT|knee_angle|M0|4|11.1422|16.5486|
|deadlift|LOCKOUT|hip_angle|M0|4|20.1289|25.6161|
|deadlift|LOCKOUT|trunk_lean|M0|4|28.1546|39.0361|
|deadlift|START|knee_angle|M1 lightweight|4|9.5065|13.9115|
|deadlift|START|hip_angle|M1 lightweight|4|21.4215|30.2946|
|deadlift|START|trunk_lean|M1 lightweight|4|15.4867|21.9015|
|deadlift|MID_DOWN|knee_angle|M1 lightweight|4|3.3251|4.5253|
|deadlift|MID_DOWN|hip_angle|M1 lightweight|4|2.2613|2.5529|
|deadlift|MID_DOWN|trunk_lean|M1 lightweight|4|2.3548|3.0711|
|deadlift|BOTTOM|knee_angle|M1 lightweight|4|0.8377|1.3000|
|deadlift|BOTTOM|hip_angle|M1 lightweight|4|0.9818|1.3317|
|deadlift|BOTTOM|trunk_lean|M1 lightweight|4|0.8924|1.2314|
|deadlift|MID_UP|knee_angle|M1 lightweight|4|1.5277|1.7415|
|deadlift|MID_UP|hip_angle|M1 lightweight|4|2.5153|2.9482|
|deadlift|MID_UP|trunk_lean|M1 lightweight|4|2.8723|3.6635|
|deadlift|LOCKOUT|knee_angle|M1 lightweight|4|6.4811|10.0716|
|deadlift|LOCKOUT|hip_angle|M1 lightweight|4|17.3263|24.5031|
|deadlift|LOCKOUT|trunk_lean|M1 lightweight|4|8.6774|12.2717|
|deadlift|START|knee_angle|M1 balanced|4|7.0286|9.4683|
|deadlift|START|hip_angle|M1 balanced|4|5.9642|7.6654|
|deadlift|START|trunk_lean|M1 balanced|4|2.7353|3.6716|
|deadlift|MID_DOWN|knee_angle|M1 balanced|4|2.3971|3.0036|
|deadlift|MID_DOWN|hip_angle|M1 balanced|4|5.0786|6.2616|
|deadlift|MID_DOWN|trunk_lean|M1 balanced|4|3.7256|5.0031|
|deadlift|BOTTOM|knee_angle|M1 balanced|4|0.9582|1.1886|
|deadlift|BOTTOM|hip_angle|M1 balanced|4|2.3590|3.3866|
|deadlift|BOTTOM|trunk_lean|M1 balanced|4|1.1156|1.6836|
|deadlift|MID_UP|knee_angle|M1 balanced|4|2.7112|3.6962|
|deadlift|MID_UP|hip_angle|M1 balanced|4|5.7947|6.0266|
|deadlift|MID_UP|trunk_lean|M1 balanced|4|2.4543|2.7684|
|deadlift|LOCKOUT|knee_angle|M1 balanced|4|7.0537|8.6758|
|deadlift|LOCKOUT|hip_angle|M1 balanced|4|10.7526|14.5829|
|deadlift|LOCKOUT|trunk_lean|M1 balanced|4|6.7061|9.4838|
|deadlift|START|knee_angle|n3|4|6.5655|9.2851|
|deadlift|START|hip_angle|n3|4|5.8175|7.6876|
|deadlift|START|trunk_lean|n3|4|2.2591|3.1689|
|deadlift|MID_DOWN|knee_angle|n3|4|2.6176|3.4509|
|deadlift|MID_DOWN|hip_angle|n3|4|4.1725|5.2533|
|deadlift|MID_DOWN|trunk_lean|n3|4|2.6199|2.9602|
|deadlift|BOTTOM|knee_angle|n3|4|1.8363|2.3618|
|deadlift|BOTTOM|hip_angle|n3|4|3.3444|4.4369|
|deadlift|BOTTOM|trunk_lean|n3|4|1.8184|2.4999|
|deadlift|MID_UP|knee_angle|n3|4|2.9165|3.7876|
|deadlift|MID_UP|hip_angle|n3|4|6.3576|7.9890|
|deadlift|MID_UP|trunk_lean|n3|4|2.9738|3.6491|
|deadlift|LOCKOUT|knee_angle|n3|4|6.4859|8.3323|
|deadlift|LOCKOUT|hip_angle|n3|4|10.6213|14.6232|
|deadlift|LOCKOUT|trunk_lean|n3|4|6.9233|9.7911|
|deadlift|START|knee_angle|n5|4|6.9709|8.9773|
|deadlift|START|hip_angle|n5|4|6.0740|8.0368|
|deadlift|START|trunk_lean|n5|4|2.5080|3.5389|
|deadlift|MID_DOWN|knee_angle|n5|4|2.1081|2.8843|
|deadlift|MID_DOWN|hip_angle|n5|4|3.7855|4.9668|
|deadlift|MID_DOWN|trunk_lean|n5|4|2.5972|2.8376|
|deadlift|BOTTOM|knee_angle|n5|4|1.2905|1.7420|
|deadlift|BOTTOM|hip_angle|n5|4|3.0865|4.3650|
|deadlift|BOTTOM|trunk_lean|n5|4|2.3301|2.3696|
|deadlift|MID_UP|knee_angle|n5|4|2.3446|3.0615|
|deadlift|MID_UP|hip_angle|n5|4|5.2452|6.8276|
|deadlift|MID_UP|trunk_lean|n5|4|3.1053|4.0965|
|deadlift|LOCKOUT|knee_angle|n5|4|6.1975|8.3995|
|deadlift|LOCKOUT|hip_angle|n5|4|9.7895|13.8445|
|deadlift|LOCKOUT|trunk_lean|n5|4|7.2516|10.2553|
|deadlift|START|knee_angle|n10|4|6.6571|8.4662|
|deadlift|START|hip_angle|n10|4|5.4730|7.5087|
|deadlift|START|trunk_lean|n10|4|2.5829|3.7605|
|deadlift|MID_DOWN|knee_angle|n10|4|1.4470|1.9077|
|deadlift|MID_DOWN|hip_angle|n10|4|3.4853|4.5815|
|deadlift|MID_DOWN|trunk_lean|n10|4|2.7618|3.3099|
|deadlift|BOTTOM|knee_angle|n10|4|1.1775|1.7148|
|deadlift|BOTTOM|hip_angle|n10|4|3.0939|4.2158|
|deadlift|BOTTOM|trunk_lean|n10|4|1.3825|1.4850|
|deadlift|MID_UP|knee_angle|n10|4|2.4331|3.0967|
|deadlift|MID_UP|hip_angle|n10|4|5.2700|6.7780|
|deadlift|MID_UP|trunk_lean|n10|4|3.0423|4.1035|
|deadlift|LOCKOUT|knee_angle|n10|4|7.5436|9.7171|
|deadlift|LOCKOUT|hip_angle|n10|4|10.5931|14.1316|
|deadlift|LOCKOUT|trunk_lean|n10|4|7.2309|10.2260|
|deadlift|START|knee_angle|tiny_m|4|5.4232|7.2170|
|deadlift|START|hip_angle|tiny_m|4|5.8331|8.0361|
|deadlift|START|trunk_lean|tiny_m|4|2.7981|4.2134|
|deadlift|MID_DOWN|knee_angle|tiny_m|4|2.3152|2.9181|
|deadlift|MID_DOWN|hip_angle|tiny_m|4|5.9770|8.0403|
|deadlift|MID_DOWN|trunk_lean|tiny_m|4|2.6486|3.4466|
|deadlift|BOTTOM|knee_angle|tiny_m|4|1.1938|1.6949|
|deadlift|BOTTOM|hip_angle|tiny_m|4|3.2106|4.4881|
|deadlift|BOTTOM|trunk_lean|tiny_m|4|2.5466|3.2788|
|deadlift|MID_UP|knee_angle|tiny_m|4|3.6849|5.2112|
|deadlift|MID_UP|hip_angle|tiny_m|4|5.6808|7.1646|
|deadlift|MID_UP|trunk_lean|tiny_m|4|1.9045|2.2999|
|deadlift|LOCKOUT|knee_angle|tiny_m|4|8.3135|11.3623|
|deadlift|LOCKOUT|hip_angle|tiny_m|4|11.1235|11.8397|
|deadlift|LOCKOUT|trunk_lean|tiny_m|4|6.2092|8.0837|

## M1 balanced 대비 새 오류 영상

- n3: squat/stu6_11_2.mp4
- n5: squat/stu6_11_2.mp4
- n10: squat/stu6_11_2.mp4
- tiny_m: benchpress/test153.mp4, deadlift/LLK_S01_R03.mp4, squat/stu6_11_2.mp4, team9/benchpress/stu9_10.mp4

기존 오류 영상의 새로운 오류 프레임은 아래 영상별 검토 문구에 따로 기록한다.

## 검출 호출과 추적 누락

|설정|전체 프레임|검출 프레임|검출 비율|누락 판정 프레임|
|---|---:|---:|---:|---:|
|n3|19331|7475|0.3867|1702|
|n5|19331|5089|0.2633|1649|
|n10|19331|3324|0.1720|1618|
|tiny_m|19331|19331|1.0000|0|

누락 프레임은 몸통/손목 지원 부족 등으로 다음 프레임 재검출을 예약한 횟수다. 현재 포즈 valid=False 수와는 다르다. tiny_m은 추적을 하지 않아 이 수가 0이다.

## 리프터 오류 영상과 이미지


### M0

|영상|검토 결과|이미지|
|---|---|---|
|benchpress/stu10_11.mp4|wrong: spotter selected at diagnostic f359/f719/f1078/f1438; no completed plate reps, excluded from rep-interval denominators|M0/images/benchpress__stu10_11.mp4.jpg|
|benchpress/test153.mp4|wrong: spotter selected at diagnostic f0/f239/f299; no completed plate reps|M0/images/benchpress__test153.mp4.jpg|
|squat/stu6_11_2.mp4|wrong: background standing person selected at diagnostic f127/f159; no completed plate reps|M0/images/squat__stu6_11_2.mp4.jpg|
|squat/stu6_11_3.mp4|wrong: spurious torso on stored yellow plate at diagnostic f0/f38; no completed plate reps|M0/images/squat__stu6_11_3.mp4.jpg|
|deadlift/GB_S04_R03.mp4|wrong: background standing person at far right selected at rep frame f36|M0/images/deadlift__GB_S04_R03.mp4.jpg|
|team9/benchpress/stu9_10.mp4|wrong: standing spotters selected in all six rep samples f243/f401/f558/f740/f904/f1085 while lifter lies on bench; coverage 100%|M0/images/team9__benchpress__stu9_10.mp4.jpg|

### M1 lightweight

|영상|검토 결과|이미지|
|---|---|---|
|squat/val1235.mp4|wrong: selected skeleton on background standing person at f52/f232; visible lifter skeleton at f179/f284|M1_lightweight/images/squat__val1235.mp4.jpg|
|benchpress/stu10_11.mp4|wrong: different error at pre-lift f0 (background person); previous spotter errors f359/f719/f1078/f1438 fixed, lifter selected there and at f1797; no completed plate reps|M1_lightweight/images/benchpress__stu10_11.mp4.jpg|
|benchpress/test153.mp4|wrong: spotter at f60; previous errors f0/f239/f299 fixed, lifter also selected at f120/f179; no completed plate reps|M1_lightweight/images/benchpress__test153.mp4.jpg|
|squat/stu6_11_2.mp4|wrong: background person at f95; previous f127/f159 errors fixed; lifter at f0/f32/f64/f127/f159; no completed plate reps|M1_lightweight/images/squat__stu6_11_2.mp4.jpg|
|team9/benchpress/stu9_10.mp4|wrong: bystanders at f401/f558/f740 remain; previous errors f243/f904/f1085 fixed, lifter selected there|M1_lightweight/images/team9__benchpress__stu9_10.mp4.jpg|

### M1 balanced

|영상|검토 결과|이미지|
|---|---|---|
|squat/val1235.mp4|wrong: background standing person/reflection at f0/f105/f179 (confirmed raw nose/torso coordinates); lifter at f52/f232/f284|M1_balanced/images/squat__val1235.mp4.jpg|
|benchpress/stu10_11.mp4|wrong: different error at pre-lift f0 (background person); previous spotter errors f359/f719/f1078/f1438 fixed, lifter selected there and at f1797; no completed plate reps|M1_balanced/images/benchpress__stu10_11.mp4.jpg|

### n3

|영상|검토 결과|이미지|
|---|---|---|
|squat/val1235.mp4|wrong: background/reflection selected at f0; six-frame sheet reviewed|n3/images/squat__val1235.mp4.jpg|
|benchpress/stu10_11.mp4|wrong: background person at pre-lift f0; other five samples select lifter; no completed rep|n3/images/benchpress__stu10_11.mp4.jpg|
|squat/stu6_11_2.mp4|wrong: background blue-shirt standing person at f95; remaining sampled frames select lifter; no completed rep|n3/images/squat__stu6_11_2.mp4.jpg|

### n5

|영상|검토 결과|이미지|
|---|---|---|
|squat/val1235.mp4|wrong: background/reflection selected at f0; six-frame sheet reviewed|n5/images/squat__val1235.mp4.jpg|
|benchpress/stu10_11.mp4|wrong: background person at pre-lift f0 and standing blue-shirt spotter at f1438; f359/f719/f1078/f1797 lifter; no completed rep|n5/images/benchpress__stu10_11.mp4.jpg|
|squat/stu6_11_2.mp4|wrong: background blue-shirt standing person at f95; remaining sampled frames select lifter; no completed rep|n5/images/squat__stu6_11_2.mp4.jpg|

### n10

|영상|검토 결과|이미지|
|---|---|---|
|squat/val1235.mp4|wrong: background/reflection selected at f0; six-frame sheet reviewed|n10/images/squat__val1235.mp4.jpg|
|benchpress/stu10_11.mp4|wrong: pre-lift f0 selects background person; remaining five samples select lifter; no completed rep|n10/images/benchpress__stu10_11.mp4.jpg|
|squat/stu6_11_2.mp4|wrong: background blue-shirt standing person at f95; remaining sampled frames select lifter; no completed rep|n10/images/squat__stu6_11_2.mp4.jpg|

### tiny_m

|영상|검토 결과|이미지|
|---|---|---|
|squat/val1235.mp4|wrong: background standing person/reflection at f52/f105/f179; other occluded poses also have uncertain geometry|tiny_m/images/squat__val1235.mp4.jpg|
|benchpress/stu10_11.mp4|wrong: pre-lift f0 selects background person; f359/f719/f1078/f1438 and f1797 are lifter, same six-frame issue as M1 balanced; no completed rep|tiny_m/images/benchpress__stu10_11.mp4.jpg|
|benchpress/test153.mp4|wrong: standing spotter selected at f60; other five samples select lifter; no completed rep, so rep guard not active|tiny_m/images/benchpress__test153.mp4.jpg|
|squat/stu6_11_2.mp4|wrong: background blue-shirt standing person at f95; remaining sampled frames select lifter; no completed rep|tiny_m/images/squat__stu6_11_2.mp4.jpg|
|deadlift/LLK_S01_R03.mp4|wrong: raw torso/hip at f59 attaches to background person or mixed foreground/background skeleton, despite valid=True and nearly no drawn edges; confirmed canonical-input detail deadlift__LLK_S01_R03__f59.jpg; other five samples lifter|tiny_m/images/deadlift__LLK_S01_R03.mp4.jpg|
|team9/benchpress/stu9_10.mp4|wrong: foreground-left bystander/spurious off-screen torso at f558 passes >45deg guard (both shoulders x=-10px); f401/f740 missing, f243/f904/f1085 lifter; confirmed details/team9__benchpress__stu9_10__f558.jpg|tiny_m/images/team9__benchpress__stu9_10.mp4.jpg|

### bench_guard

|영상|검토 결과|이미지|
|---|---|---|
|benchpress/stu10_11.mp4|wrong: background person at pre-lift f0; other five samples select lifter; no completed rep|bench_guard/images/benchpress__stu10_11.mp4.jpg|

## 벤치 규칙만 바꾼 16개 비교

|설정|오류 영상|coverage|몸통 점프|
|---|---:|---:|---:|
|M1 balanced (전)|1|0.9818|14|
|balanced guard (후)|1|0.8963|4|

### 벤치 16개 영상별 coverage와 오류

|영상|coverage 전|coverage 후|오류 전|오류 후|후 확인 이미지|
|---|---:|---:|---|---|---|
|benchpress/bench press_31.mp4|1.0000|1.0000|False|False|bench_guard/images/benchpress__bench press_31.mp4.jpg|
|benchpress/bench press_40.mp4|1.0000|0.9401|False|False|bench_guard/images/benchpress__bench press_40.mp4.jpg|
|benchpress/stu10_11.mp4|N/A|N/A|True|True|bench_guard/images/benchpress__stu10_11.mp4.jpg|
|benchpress/stu8_15.mp4|1.0000|1.0000|False|False|bench_guard/images/benchpress__stu8_15.mp4.jpg|
|benchpress/stu8_7.mp4|1.0000|1.0000|False|False|bench_guard/images/benchpress__stu8_7.mp4.jpg|
|benchpress/test153.mp4|N/A|N/A|False|False|bench_guard/images/benchpress__test153.mp4.jpg|
|benchpress/bench press_61.mp4|1.0000|1.0000|False|False|bench_guard/images/benchpress__bench press_61.mp4.jpg|
|benchpress/stu1_6_4.mp4|1.0000|0.0000|False|False|bench_guard/images/benchpress__stu1_6_4.mp4.jpg|
|benchpress/stu7_12_2.mp4|0.9904|0.7083|False|False|bench_guard/images/benchpress__stu7_12_2.mp4.jpg|
|benchpress/stu7_7.mp4|0.9732|0.9107|False|False|bench_guard/images/benchpress__stu7_7.mp4.jpg|
|benchpress/stu8_6_2.mp4|0.8418|0.8316|False|False|bench_guard/images/benchpress__stu8_6_2.mp4.jpg|
|benchpress/bench_01.mp4|1.0000|0.9690|False|False|bench_guard/images/benchpress__bench_01.mp4.jpg|
|benchpress/bench_02.mp4|1.0000|1.0000|False|False|bench_guard/images/benchpress__bench_02.mp4.jpg|
|benchpress/bench_03.mp4|1.0000|1.0000|False|False|bench_guard/images/benchpress__bench_03.mp4.jpg|
|team9/benchpress/stu9_10.mp4|0.9708|0.8961|False|False|bench_guard/images/team9__benchpress__stu9_10.mp4.jpg|
|team9/benchpress/stu2_10_edited.mp4|1.0000|1.0000|False|False|bench_guard/images/team9__benchpress__stu2_10_edited.mp4.jpg|

## 손가락 의존성 수정 전후 기준 rep 자격

같은 raw/smoothed 포즈와 segmentation에 대해 필요 관절만 바꿔 native `_eligible` 결과를 센다. 모든 rep 중 실제 세트 비교에서 기준으로 선택할 수 있는 rep 수이며, 홀짝 진단의 판정 칸 수와 다르다.

|설정|종목|수정 전|수정 후|
|---|---|---:|---:|
|M1 lightweight|benchpress|22|28|
|M1 lightweight|deadlift|42|42|
|M1 balanced|benchpress|32|44|
|M1 balanced|deadlift|42|42|
|n3|benchpress|24|37|
|n3|deadlift|42|42|
|n5|benchpress|30|36|
|n5|deadlift|42|42|
|n10|benchpress|29|37|
|n10|deadlift|42|42|
|tiny_m|benchpress|20|22|
|tiny_m|deadlift|42|42|
|bench_guard|benchpress|25|36|

## ORT 6스레드: 60초·30 fps 영상

|설정|원판+포즈 처리 초|지표 포함 전체 초|처리 fps|
|---|---:|---:|---:|
|balanced|566.8002|567.3786|3.1757|
|n3|244.3721|244.9635|7.3658|
|n5|182.7087|183.2877|9.8517|
|n10|125.3975|125.9779|14.3544|
|tiny_m|240.4920|241.0731|7.4847|
|lightweight|178.9557|179.5055|10.0584|

입력 SHA256: `cc1d4eb5ccc63ac4bea6aa7754fc960dd698956ad79c6715d8f185639f208047`. 단계별 시간과 영상 경로는 `threads6/benchmark.json`에 있다.
