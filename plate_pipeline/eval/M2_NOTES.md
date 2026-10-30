# M2 3D와 원판 기하

기준 `plate-pipeline(d5e19be)`, 브랜치 `integ/m2-3d`. PLAN.md의 M1b 결정을 읽었고 PLAN.md/main은 수정하지 않는다. 모델·영상·eval_out은 커밋하지 않는다.

## 포즈와 입력

PoseCfg 기본값은 tracked, det_frequency=10, bench_rep_guard=False다. 벤치 16개는 이 설정으로 원판·포즈부터 다시 추론한다. 나머지 53개는 M1b n10의 선택된 raw MP33를 재사용한다. guard는 벤치에만 영향을 주므로 나머지는 같은 선택 경로다. VideoPose3D 입력은 이 raw 관절에서 COCO17을 가져오며 사람 검출이나 최대 상자 선택을 다시 하지 않는다. visibility가 낮다고 다른 사람 관절로 대체하지 않는다.

기존 검토 문구는 원본 그대로 보존하고, 잘못된 사람으로 확인한 표본 프레임을 완료 원판 rep 구간 안/밖으로 분류했다. 과거 문구에서 "고침"으로 나오는 프레임까지 오류로 세지 않는다. 완료 rep은 start/far_in/far_out/back이 모두 있는 경우이며 start~back 양 끝을 포함한다. 한 영상이 두 범주에 모두 나타날 수 있다. 이미지 표본을 기준으로 한 하한이며 전체 프레임의 정답 판정률이 아니다.

벤치 재평가는 ORT intra-op 1/inter-op 1, OpenCV 1이다. 영상 묶음을 네 개로 나눠 서로 다른 물리 코어 CPU 0/3/4/5에서 실행했다. 3D의 최종 69개 측정은 CPU 1에 고정해 1스레드로 순차 실행했다. 설정별 영상은 순차 처리했다. 6스레드 전체 처리 시간은 이 작업과 시험이 모두 끝난 뒤 단독 실행한다.

## 기하

`core.geometry.plate_geometry`는 사용된 원판 상자의 가로/세로 비의 중앙값을 [0,1]로 잘라 arccos로 각도를 구한다. 초점거리는 `0.77 × max(w,h)`, 거리는 `f × D / 상자 세로`, kd는 `(거리 + 0.70 × cos 각도) / 거리`다. 상자가 없는 프레임의 세로는 양쪽 관측값으로 선형 보간하고 짧은 끝 구간은 가까운 관측값을 쓴다. 상자 관측 수와 비율 원값도 저장한다. 원판 신호나 판정 코드는 바꾸지 않는다.

0.77은 초점거리 가정, 0.70m는 **바 기하 가정, 지름과 무관**, 0.92는 코–바닥 높이/키 비율 가정이다. 모두 GeometryCfg에 있다. 사용자의 추가 결정대로 시험을 나눴다. 원판 면의 거리·깊이·속도는 지름에 비례한다. 리프터 면 스케일은 `D/h + 0.70*cos(angle)/f`이며 고정 바 간격 때문에 지름에 정확히 비례하지 않는다. 보고서에 실제 영상의 450→400mm 예시를 넣는다.

서 있는 프레임은 선택된 리프터의 코·양 어깨·엉덩이·무릎·발목이 유한하고 visibility≥0.5, 양 무릎 각도>160°, 몸통이 화면 수직에서 30° 미만이며 어깨가 엉덩이 위인 프레임이다. 가시 발목·뒤꿈치·발끝 중 가장 아래를 바닥으로 보고 `(발 y−코 y) × 리프터 면 m/px / 0.92`의 중앙값을 키로 쓴다. 누운 벤치 영상에서 다른 선 사람을 키 측정 대상으로 다시 고르지 않는다. 서 있는 관측이 없으면 키는 null이고 원인을 기록한다. 숫자를 150~200cm로 강제로 자르지 않는다.

## 3D

`track3d`의 vendored VideoPose3D와 `pretrained_h36m_detectron_coco.bin`을 재사용한다. 수용 영역은 243프레임이다. track3d의 프레임 단위 최대 10프레임 빈칸 선형 보간·짧은 끝 구간 hold를 그대로 쓰며, 긴 빈칸은 null로 남기고 유효 구간을 분리한다. 추론은 128 출력 프레임씩 실행하되 앞뒤 121프레임 문맥을 보존한다. 작은 합성 TemporalModel에서 전체 구간 추론과 수치 일치를 시험한다.

Gaussian 시간 평활화는 sigma=max(1프레임, fps×0.06초), 반경 ceil(3sigma), 끝은 nearest다. 긴 빈칸을 가로질러 평활화하지 않는다. H36M tree의 16개 뼈를 각 세트의 평활화 후 길이 중앙값으로 고정한다. 각 뼈 방향을 유지하며 부모부터 재구축한다. rep 밖 프레임은 가장 가까운 세트 중심에 배정하며 완료 세트가 없으면 영상 전체를 한 그룹으로 쓴다. 서로 다른 세트의 목표 길이는 다를 수 있으므로 CV는 세트별로도 저장한다.

몸 좌표계 위쪽은 서 있는 3D 프레임의 발목 중점→neck/nose(9) 벡터 중앙값이다. 엉덩이 좌우선을 위쪽에 직교화해 lateral 축을 만들고 그 선과 위쪽의 외적으로 forward 축을 만든다. Halpe 발뒤꿈치→엄지발가락의 2D 방향과 forward의 영상 평면 투영 내적으로 부호를 정한다. 발 방향이 가려지거나 수직 투영이 0에 가까우면 부호 불확실을 기록한다. 서 있는 프레임이 없으면 camera-up [0,-1,0]을 사용했다는 경고를 남긴다.

측정 키의 0.92배와 3D neck/nose→낮은 발목 높이 중앙값이 같도록 스케일한다. 서 있는 관측/키가 없으면 **VideoPose3D의 모델 자체 미터 스케일**을 유지하고 `scaled_to_stature=false`로 명시한다. 이 경우 키로 교정된 분절 수치라고 해석하면 안 된다. 프레임마다 낮은 발목 y를 0으로 옮기고 서 있는 발목 중점의 x/z 중앙값을 원점으로 둔다. 양발을 동시에 강제로 바닥에 붙이지 않는다.

출력은 프레임별 H36M17 미터 좌표와 MP33 번호 테이블(`j3names`, `j3`)이다. H36M 9는 기존 track3d에서 neck라 부르는 neck/nose proxy다. 독립적인 COCO 코가 없으므로 MP0에 그 proxy를 넣었다고 명시한다. 없는 MP 관절은 생성하지 않는다. 무릎·엉덩이·몸통 기울기는 기존 다섯 포지션에서 계산해 별도 JSON에 저장하며 기존 판정에는 넣지 않는다.

확인 이미지는 첫 완료 rep에서 원판 중심 y가 가장 큰 **물리적으로 가장 낮은 원판 프레임**을 쓴다. 데드리프트는 far/top과 구분한다. 완료 rep이 없으면 영상 중간 프레임에 NO COMPLETED REP 진단 표시를 넣는다. 왼쪽은 기존 raw 2D, 오른쪽은 z-forward/y-up의 등방 측면 투영이다. 파랑은 왼쪽, 주황은 오른쪽이다. 측면 투영만으로 숨은 깊이나 좌우 정답을 확정할 수 없다.

분절은 보정한 3D의 양쪽 길이 중앙값이다. H36M hip→shoulder 몸통은 여러 tree 뼈를 가로지르므로 그 합성 선 길이까지 불변이라는 뜻은 아니다. 뼈 CV는 평활화 후 고정 전/후 비교다. 가속도는 같은 최종 스케일을 적용한 native/평활화 좌표의 2차 차분 norm×fps² 중앙값(m/s²)이며 바닥 고정으로 생기는 이동은 제외한다. 각도 흔들림은 M1b와 같은 ddof=1 SD·모든 rep 쌍 차이 중앙값이고 2D/3D 각각의 관측 수를 표시한다. 흔들림 감소가 3D 정답률 향상을 보증하지 않는다.

## 라이선스와 실행

VideoPose3D의 코드·가중치는 [공식 저장소](https://github.com/facebookresearch/VideoPose3D)와 vendored LICENSE의 **CC BY-NC 4.0** 조건을 따른다. COCO 입력용 가중치는 [공식 inference 안내](https://github.com/facebookresearch/VideoPose3D/blob/main/INFERENCE.md)의 파일이다. 상업적 사용 라이선스로 해석하지 않는다. 체크포인트는 저장소 밖 `~/.cache/videopose3d/`에 둔다.

```bash
cd plate_pipeline
python -m eval.baseline --out eval_out/m2/bench_n10 --mode tracked --det-frequency 10 \
  --no-bench-rep-guard --exercise-filter benchpress --prepared-reference M1_BALANCED_DIR --resume
python -m eval.m2_3d --source M1B_N10_DIR --bench eval_out/m2/bench_n10 \
  --out eval_out/m2/pose3d --checkpoint ~/.cache/videopose3d/pretrained_h36m_detectron_coco.bin --resume
# 실제 제품 경로의 선택된 pose로 동일한 진단 JSON 생성 (기존 판정에는 미사용)
cd ..
python run_demo.py --config CONFIG.json --segment plate --out OUT \
  --pose3d-checkpoint ~/.cache/videopose3d/pretrained_h36m_detectron_coco.bin
```

합성 시험은 기하 공식·두 면의 지름 법칙·뼈 고정과 방향 보존·회전 불변 관절각/몸통 기울기·크기/바닥·MP33/COCO 대응·짧은 보간/긴 빈칸·분할 추론·포즈 기본값을 다룬다. 확인 이미지와 수치 결과는 eval_out에 있고 최종 보고서만 results에 커밋한다.


## 측정 환경과 시험

AMD Ryzen 5 7500F, CPUExecutionProvider, rtmlib 0.0.16, onnxruntime 1.31.0,
Torch 2.14.1+cpu, SciPy 1.18.1, NumPy 2.5.3이다. 체크포인트 SHA256은
`d3219e005b50591f694da5cbaf6849f060d6b2cf895864a779f8a992ac63a232`다.
69개 표의 3D 시간은 배열 입력부터 추론·기하·후처리까지이며 일회성 모델 로드와
JSON/이미지 인코딩은 제외한다. 6스레드 전체 시간에는 모델 로드·저장·지표가 들어간다.

```bash
cd plate_pipeline
python -m pytest -q tests eval/midterm/test_metrics.py
# 160 passed in 10.88s
cd ..
PATH=/home/kojy1009/miniconda3/lib/python3.14/site-packages/playwright/driver:$PATH python -m pytest -q tests/fb
# 235 passed in 2.93s
```


### H36M → 데모 MP33 대응

|MP33|H36M17|의미|
|---|---|---|
|0|9|neck/nose proxy (독립 코 아님)|
|11, 12|11, 14|왼쪽·오른쪽 어깨|
|13, 14|12, 15|왼쪽·오른쪽 팔꿈치|
|15, 16|13, 16|왼쪽·오른쪽 손목|
|23, 24|4, 1|왼쪽·오른쪽 엉덩이|
|25, 26|5, 2|왼쪽·오른쪽 무릎|
|27, 28|6, 3|왼쪽·오른쪽 발목|

그 외 MP33 점은 null이며 발뒤꿈치/발끝의 3D를 만들어내지 않는다. 발 방향의 부호 판정에만 기존 Halpe 2D 발점을 쓴다.


## 보고서 재생성

```bash
cd plate_pipeline
python -m eval.m2_reviews --m0 M0_DIR --m1 M1_ROOT --m1b M1B_ROOT \
  --out eval_out/m2/lifter_review_intervals.json
python -m eval.m2_benchmark --video M1B_ROOT/benchmark_60s30.mp4 \
  --model PLATE_MODEL.onnx --checkpoint ~/.cache/videopose3d/pretrained_h36m_detectron_coco.bin \
  --out eval_out/m2/threads6
python -m eval.m2_report --m0 M0_DIR --m1 M1_ROOT --m1b M1B_ROOT \
  --runs eval_out/m2 --notes eval/M2_NOTES.md --report results/m2-3d/REPORT.md
```

실제 벤치 네 묶음의 명령·프레임별 측정 fingerprint는 `bench_n10/shards.json`에 있다.
그 결과를 같은 16개 순서로 모아 baseline을 만들었으며 `wall_s`는 네 명령 중 가장 긴
실행 시간이다. 병렬 묶음의 합성 fps로 속도 향상을 주장하지 않는다. 모든 입력은
M0/M1과 SHA256이 같은 준비 영상을 사용했고, 원판과 포즈는 새로 추론했다.
3D 단계는 최종 코드로 69개 모두 다시 생성했고 이미지 해시를 검토 기록과 대조한다.

zip은 설정별 JSON·MD와 이미지, 역사적 오류 표본의 안/밖 분류, 원판 불변 증거를 담는다.
모델·영상·NPZ는 zip에 넣지 않는다. 벤치에서 몸통/발이 화면 밖인 경우와 서 있는
프레임이 없는 경우를 3D 모델이 자동으로 해결했다고 해석하지 않는다.

3D 각도는 해당 17점 프레임이 유한하면 계산한다. track3d와 같이 낮은 2D 점수에
별도 confidence pruning을 추가하지 않았고, 짧은 빈칸 보간 프레임도 포함한다.
따라서 기존 관절별 visibility gate가 있는 2D보다 가용 rep 수가 많을 수 있다.
이 차이는 표의 n과 보간 인덱스로 드러내며 추가 관측이나 정확도 향상으로 보지 않는다.
6스레드 전체 시간의 M1b 대비 작은 차이는 실행 편차도 포함하므로 3D를 더해서
2D가 빨라졌다고 해석하지 않는다.
