# M1b 포즈 속도·정확도 조합

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
