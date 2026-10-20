# 원판 검출기 학습 (구현 순서 3단계)

원판 하나만 찾는 1클래스 검출기(YOLOX-Tiny)를 학습해서 `models/plate_tiny.onnx`로 내보낸다. 합격 기준은 학습에 쓰지 않은 세션의 300프레임에서 재현율 95% 이상(IoU 0.5)이고, 벤치는 따로 센다. 저장소 루트의 예전 YOLO 바벨 검출 코드와는 상관없다.

모든 명령은 `plate_pipeline/` 폴더에서 실행한다.

## 0. 환경

```bash
conda create -n plate python=3.10 -y && conda activate plate
# PyTorch: https://pytorch.org 에서 자기 환경에 맞는 명령을 쓴다. 예:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # GPU (CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu     # CPU만
pip install numpy scipy opencv-python av onnxruntime pillow transformers pycocotools pytest
python -m pytest -q tests        # 전부 통과해야 한다
```

GPU가 없어도 전부 돌아간다. 자동 라벨과 학습이 느려질 뿐이다 (7단계).

## 1. 촬영

- 세 종목, 옆이나 비스듬히(45° 안쪽). 20 kg 원판과 작은 원판을 섞는다.
- 폰을 고정하고 원판 전체가 처음부터 끝까지 보이게. 30 또는 60 fps, 슬로모션 끔.
- 세션 = 촬영한 날 + 장소. **세션을 3개 이상** 만든다. 시험용 세션은 학습용과 장소나 사람이 달라야 한다.
- 세트마다 실제 횟수를 적어 둔다 (횟수 시험에 쓴다). 바를 거치대에 올려 둔 채 10초 찍은 클립도 하나 찍는다 (위치 잡음 시험).

`videos.csv`:

```text
path,session,exercise
D:/lift/1021_gymA_squat1.mp4,1021_gymA,squat
D:/lift/1021_gymA_bench1.mp4,1021_gymA,bench
```

## 2. 프레임 뽑기

```bash
python -m train.plate.extract_frames --videos videos.csv --out data/own --per-video 15
```

영상을 같은 길이로 나눠 구간마다 앞에서 뽑은 프레임과 가장 다른 프레임을 고른다. 영상 25개면 약 375장이다.

## 3. 자동 라벨 (Grounding DINO)

```bash
python -m train.plate.autolabel --frames data/own/frames.json --images data/own/images --out data/own/auto.json
```

처음 실행할 때 Hugging Face에서 모델을 받는다. GPU가 있으면 base 모델, 없으면 tiny 모델을 쓴다 (`--device cpu`, `--model`로 바꿀 수 있음). 박스가 너무 적거나 많으면 `--box-thr`(기본 0.30)를 바꾼다.

## 4. 손으로 고치기

```bash
python -m train.plate.review --labels data/own/auto.json --images data/own/images --out data/own/labels.json
```

| 키 | 동작 |
| --- | --- |
| 마우스 끌기 | 새 박스 |
| 박스 클릭 → d | 박스 지우기 |
| a / Enter / Space | 이 프레임 확인, 다음 |
| u | 못 쓰는 프레임 (데이터셋에서 뺌) |
| n / p | 다음 / 이전 |
| s, q | 저장, 저장하고 끝 |

라벨 규칙:

- 절반 이상 보이는 원판은 모두 단다.
- **박스는 원판 테두리 전체를 감싼다.** 바 끝, 칼라, 원판 가운데에 붙은 것을 기준으로 하지 않는다. 바 끝은 원판보다 카메라 쪽으로 튀어나와 있어서, 그걸 따라가면 ROM이 부푼다 (공개 스쿼트 영상에서 63~69 cm가 95 cm로 나왔다).
- 한쪽에 원판이 여러 장 겹쳐 있으면 가장 바깥의 큰 원판 하나만 단다.

자신 없는 박스부터 보여 주고, 중간에 그만둬도 이어서 할 수 있다.

## 5. 공개 데이터

Roboflow의 두 데이터셋을 "COCO" 형식으로 받아 `data/public/` 아래에 푼다.

- https://universe.roboflow.com/personal-4yjis/plate-detection-xvycd (659장)
- https://universe.roboflow.com/computer-vision-1bdzc/barbell-detection (957장)

클래스 이름이 무엇인지는 다음 단계가 출력한다. 원판 클래스 이름을 `--public-classes`로 준다. 박스가 원판 테두리를 감싸는지 review.py로 몇 장 확인한다 (바 전체를 감싼 박스면 쓰지 않는다).

## 6. 데이터셋 만들기

```bash
python -m train.plate.make_dataset --own data/own/labels.json --own-images data/own/images \
    --public data/public/plate-detection --public data/public/barbell-detection \
    --out datasets/plate
```

내 프레임은 세션 단위로 train / val / test로 나뉜다 (test 300장 이상). 직접 정하려면 `--test-sessions`, `--val-sessions`. 공개 데이터는 train에만 들어간다. train의 30%에는 흐림·밝기를 바꾼 사본이 하나씩 더 생긴다.

## 7. 학습과 내보내기

```bash
python -m train.plate.train --data datasets/plate                # GPU가 있으면 GPU, 없으면 CPU
python -m train.plate.train --data datasets/plate --device cpu   # CPU로 강제
```

YOLOX를 `train/plate/work/`에 받아 그대로 쓰고(빌드 없음), COCO로 사전학습한 `yolox_tiny.pth`에서 미세조정한 뒤 `models/plate_tiny.onnx`로 내보낸다.

| | GPU | CPU |
| --- | --- | --- |
| 학습 코드 | YOLOX 원래 학습기 (fp16) | `cpu_train.py` (같은 모델·데이터·증강·학습률, 크기 고정) |
| 기본값 | 입력 416, 배치 32, 80 에폭 | 입력 320, 배치 8, 40 에폭 |
| 에폭마다 확인 | COCO mAP (val) | val의 재현율·정밀도 (IoU 0.5), 재현율이 가장 높은 에폭을 저장 |

CPU 속도는 아래 "CPU 시험"을 본다. 오래 걸리면 `--epochs`를 줄이거나 `--size 256`으로 내린다 (입력은 32의 배수). GPU 메모리가 모자라면 `--batch 16`. 중간에 멈췄으면 `--export-only --ckpt <경로>`로 내보내기만 할 수 있다. 내보낸 ONNX의 입력 크기는 검출기가 자동으로 읽는다.

## 8. 평가 (3단계 완료 기준)

```bash
# 원판 검출: 재현율 95% 이상, 벤치 따로
python -m eval.plate_recall --model models/plate_tiny.onnx \
    --ann datasets/plate/annotations/instances_test2017.json --images datasets/plate/test2017 --out eval_out/recall
# 위치 잡음: 0.5 cm 이하 (거치대에 올려 둔 바 10초)
python -m eval.noise static.mp4 --model models/plate_tiny.onnx
# 횟수: 세트 20개 모두 (sets.csv: path,exercise,reps)
python -m eval.counts --sets sets.csv --model models/plate_tiny.onnx --out eval_out/counts
# 영상 하나 처리 → eval_out/one/report.html
python -m server.jobs my_set.mp4 --exercise squat --model models/plate_tiny.onnx --out eval_out/one
```

출력(또는 `eval_out/`의 json·csv)을 공유하면 설계 문서에 반영한다.
