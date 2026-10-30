# M4 원판 타원 및 데드리프트 표시 결과

브랜치 `integ/m4-ellipse`, 기준 `integ/m3c-ui` (`becb8a4`). 기존 영상·모델·data·eval_out은 포함하지 않았다. 타원 결과는 검증에 실패하여 실험적이다. **기본값은 `scale_source=box`, `position_source=box`를 유지한다.**

## 구현과 한계

상자를 30% 확대한 영역의 Canny 가장자리와 contour/고정 난수 RANSAC 후보를 사용한다. 중심은 상자 중심에서 정규화 거리 0.35 이내, 장축은 상자의 큰 변의 0.7~1.4배, 축비는 0.2 이상이어야 한다. 화면 경계에 닿는 상자나 추정 타원은 거절한다. 타원 둘레 96점 중 62% 및 24구역 중 70% 이상이 2px 안의 가장자리와 맞아야 한다. 0.25초 간격 관측 사이 두 성공 측정만 보간하고, 실패 구간은 box로 되돌린다. 성공 품질은 둘레 지지율에서 중심 이탈 페널티를 뺀 값이다. 크기와 2배 각도를 0.25초 중앙값으로 다듬는다. 이것은 가림·고정 장치·글자가 많은 실제 영상에서 충분히 강건하지 않다. 각도와 다시 찍기 안내 역시 잘못된 테두리에서 오판할 수 있다.

타원 모드는 장·단축의 역수와 회전을 합친 대칭 행렬을 사용하고, 프레임 간 행렬 평균으로 이동량을 적분한다. 3 cm 이상이며 주축 분산이 부축의 2배 이상인 경로는 PCA를 위쪽 방향으로 쓰고 화면 위쪽으로 부호를 정한다. 짧거나 방향이 모호하면 화면 위쪽을 쓴다. 타원 중점 대신 상자 중점을 쓰는 것이 기본이다. 상자 모드의 원판 기록은 기존 함수를 그대로 사용한다. 선택 모드는 `python -m server.compare ... --scale-source ellipse --position-source box|ellipse`이다. `result_v3.json`에 프레임별 타원·품질·출처, geo 타원 통계·angle_source·retake_warn을 넣는다. 기존 기준 캐시는 타원 설정 시 다시 분석한다.

데드리프트 표시 전후: `captures/deadlift_before_1920x1080.png`, `captures/deadlift_after_1920x1080.png`, `captures/deadlift_after_charts_1920x1080.png`. 3D와 영상 간 간격은 12px이다. 범례는 올림→정점 멈춤→내림이고 높이는 위쪽으로 증가한다. 수치는 바꾸지 않았다.

## (a) 합성 투영

기존 실제 바 경로를 angle_sim과 같은 15개 yaw/pitch/roll 조합으로 투영했다. 각 프레임에 바 끝, 잡음과 일부 가림을 넣고 일부 표본은 화면 밖으로 이동시켰다. 합성 mp4는 로컬 eval_out에서만 생성하며 결과는 `simulation.json`에 남겼다. 각 칸은 rep별 최대 ROM 상대오차 %. A/B는 투영 상자, D는 실제 검출 및 실패 시 상자 대체를 통과한 결과다.

| yaw | pitch | roll | A | B | D | 타원 사용률 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 0.00 | 0.00 | 1.43 | 86.7% |
| 0 | 0 | 8 | 0.75 | 0.75 | 1.82 | 86.7% |
| 15 | 0 | 0 | 1.73 | 0.03 | 2.01 | 86.7% |
| 15 | 0 | 8 | 1.12 | 0.72 | 1.70 | 86.7% |
| 30 | 0 | 0 | 7.19 | 0.04 | 1.42 | 86.7% |
| 30 | 0 | 8 | 6.42 | 0.57 | 1.76 | 86.7% |
| 45 | 0 | 0 | 17.18 | 0.04 | 1.83 | 86.7% |
| 45 | 0 | 8 | 16.10 | 0.35 | 2.04 | 86.7% |
| 60 | 0 | 0 | 33.37 | 0.03 | 1.36 | 86.7% |
| 60 | 0 | 8 | 31.62 | 0.13 | 1.52 | 86.7% |
| 0 | 20 | 0 | 2.98 | 0.13 | 1.48 | 86.7% |
| 15 | 20 | 0 | 1.32 | 0.19 | 2.13 | 86.7% |
| 30 | 20 | 0 | 3.84 | 1.26 | 0.33 | 86.7% |
| 45 | 20 | 0 | 12.96 | 2.71 | 0.95 | 86.7% |
| 60 | 20 | 0 | 27.79 | 4.10 | 2.47 | 86.7% |

D 최대 2.47%: 1.1% 기준 미통과.

## (b) 69개 영상

유효 프레임 중 타원 사용률의 영상별 중앙값 30.0%. 타원 사용률에는 성공한 두 표본 사이 보간 프레임이 포함된다. 69개 기존 M0/M2 원판 행의 gzip 해제 후 내용은 69/69 바이트 일치했고, 기존 61개 완료 횟수는 box 재계산에서 61/61 일치했다.

| 방법 | 횟수 절대오차 합 | TP | FP | FN | 이벤트 F1 |
|---|---:|---:|---:|---:|---:|
| box | 49 | 64 | 9 | 56 | 0.663 |
| ellipse | 51 | 62 | 9 | 58 | 0.649 |
| ellipse_center | 51 | 58 | 13 | 62 | 0.607 |

이벤트는 기존 0.5초 매칭과 영상 첫 PTS 보정을 사용한다. ellipse는 상자 중심, ellipse_center는 타원 중심이다. 오차 합은 나빠졌고 합성 목표도 넘었으므로 box/box를 유지한다. 61개 중 완료 횟수가 바뀐 영상은 다음과 같다. 원인은 확인된 신호 변화이며 개별 물체 오검출 원인은 세 표본만으로 확정하지 않았다.

| 영상 | 정답 | box | 타원·box 중심 | 타원·타원 중심 | 관찰 |
|---|---:|---:|---:|---:|---|
| benchpress/stu8_7.mp4 | 5 | 2 | 2 | 1 | 타원/상자 전환과 위치 변화로 신호·이벤트 변동 |
| deadlift/deadlift_sihoo02.mp4 | 1 | 1 | 0 | 0 | 타원/상자 전환과 위치 변화로 신호·이벤트 변동 |
| deadlift/deadlift_03.mp4 | 2 | 2 | 1 | 2 | 타원/상자 전환과 위치 변화로 신호·이벤트 변동 |

### 종목별 ROM과 속도 분포

모든 산출 rep의 25/50/75 백분위수다. 횟수가 다른 경우 모집단도 달라진다.

| 종목 | box ROM cm | 타원 ROM cm | box 속도 m/s | 타원 속도 m/s |
|---|---|---|---|---|
| squat | 42.17/46.51/57.41 (n=35) | 40.94/46.08/59.18 (n=35) | 0.33/0.39/0.52 (n=33) | 0.34/0.41/0.53 (n=33) |
| benchpress | 26.43/30.10/33.59 (n=52) | 26.32/31.46/35.77 (n=52) | 0.18/0.35/0.46 (n=47) | 0.20/0.36/0.48 (n=47) |
| deadlift | 43.51/52.72/81.87 (n=45) | 48.07/56.93/92.68 (n=43) | 0.41/0.60/0.71 (n=42) | 0.45/0.63/0.88 (n=40) |

### 내부 키 점검

- stature_box_cm: 150~200cm 26/55, 중앙값 188.6cm. 동일 M2 선택 포즈를 사용했다.
- stature_ellipse_cm: 150~200cm 24/55, 중앙값 182.5cm. 동일 M2 선택 포즈를 사용했다.

| 같은 사람 클립 | box cm | 타원 cm |
|---|---:|---:|
| squat_01 | 180.3 | 182.0 |
| squat_02 | 165.9 | 170.2 |
| squat_03 | 155.9 | 165.8 |
| deadlift_01 | 153.0 | 142.5 |
| deadlift_02 | 152.2 | 160.2 |
| deadlift_03 | 140.8 | 134.7 |

### 영상별 측정

| 영상 | 성공률 | 장축/상자 세로 | 상자 각도 | 타원 각도 |
|---|---:|---:|---:|---:|
| squat/squat_11.mp4 | 1.4% | 0.88 | 4.7 | 28.8 |
| squat/stu4_69.mp4 | 0.3% | 0.77 | 11.9 | 72.7 |
| squat/val1235.mp4 | 99.0% | 1.14 | 0.0 | 42.3 |
| squat/val1256.mp4 | 99.0% | 1.17 | 0.0 | 52.4 |
| benchpress/bench press_31.mp4 | 57.9% | 0.80 | 60.7 | 73.9 |
| benchpress/bench press_40.mp4 | 1.3% | 0.96 | 0.0 | 23.2 |
| benchpress/stu10_11.mp4 | 0.0% | — | 70.4 | — |
| benchpress/stu8_15.mp4 | 0.7% | 0.96 | 0.0 | 69.0 |
| benchpress/stu8_7.mp4 | 73.2% | 1.24 | 25.7 | 48.5 |
| benchpress/test153.mp4 | 5.3% | 0.79 | 76.0 | 76.6 |
| deadlift/deadlift_7.mp4 | 0.0% | — | 0.0 | — |
| deadlift/deadlift_23.mp4 | 0.0% | — | 60.2 | — |
| deadlift/deadlift_30.mp4 | 4.9% | 0.91 | 17.5 | 27.5 |
| squat/stu6_11_2.mp4 | 5.6% | 1.13 | 0.0 | 50.9 |
| squat/stu6_11_3.mp4 | 84.6% | 1.29 | 0.0 | 42.8 |
| squat/val1250.mp4 | 81.7% | 1.12 | 63.4 | 53.2 |
| benchpress/bench press_61.mp4 | 0.0% | — | 49.5 | — |
| benchpress/stu1_6_4.mp4 | 0.9% | 0.72 | 55.2 | 50.7 |
| benchpress/stu7_12_2.mp4 | 6.6% | 1.03 | 32.8 | 40.2 |
| benchpress/stu7_7.mp4 | 0.0% | — | 0.0 | — |
| benchpress/stu8_6_2.mp4 | 0.0% | — | 0.0 | — |
| deadlift/deadlift_sihoo02.mp4 | 69.1% | 0.90 | 34.1 | 60.6 |
| deadlift/PB_S01_R10.mp4 | 83.9% | 0.94 | 0.0 | 26.9 |
| deadlift/PB_S03_R13.mp4 | 0.0% | — | 4.8 | — |
| deadlift/PB_S03_R14.mp4 | 0.0% | — | 13.9 | — |
| deadlift/PB_S04_R03.mp4 | 59.9% | 0.92 | 9.6 | 43.8 |
| deadlift/PB_S05_R22.mp4 | 63.3% | 1.00 | 0.0 | 52.2 |
| deadlift/PB_S06_R25.mp4 | 33.7% | 0.86 | 7.7 | 20.8 |
| deadlift/PB_S07_R19.mp4 | 43.8% | 0.81 | 0.0 | 35.0 |
| deadlift/GB_S01_R01.mp4 | 75.6% | 0.97 | 0.0 | 48.6 |
| deadlift/GB_S01_R07.mp4 | 94.2% | 0.98 | 0.0 | 28.1 |
| deadlift/GB_S01_R11.mp4 | 79.7% | 0.93 | 16.6 | 51.3 |
| deadlift/GB_S01_R16.mp4 | 82.4% | 0.96 | 0.0 | 40.4 |
| deadlift/GB_S01_R18.mp4 | 46.9% | 0.91 | 0.0 | 21.3 |
| deadlift/GB_S04_R03.mp4 | 11.4% | 0.97 | 0.0 | 22.4 |
| deadlift/GB_S04_R04.mp4 | 20.0% | 0.92 | 2.3 | 17.3 |
| deadlift/GB_S04_R07.mp4 | 30.0% | 0.88 | 14.1 | 37.4 |
| deadlift/GB_S05_R47.mp4 | 13.6% | 0.93 | 0.0 | 33.7 |
| deadlift/GB_S06_R63.mp4 | 27.8% | 0.87 | 0.0 | 25.2 |
| deadlift/GB_S07_R12.mp4 | 28.0% | 1.03 | 23.3 | 48.0 |
| deadlift/GB_S07_R14.mp4 | 41.6% | 1.00 | 0.0 | 47.0 |
| deadlift/GB_S07_R20.mp4 | 54.1% | 1.01 | 39.2 | 49.8 |
| deadlift/LLK_S01_R02.mp4 | 51.5% | 1.00 | 0.0 | 32.4 |
| deadlift/LLK_S01_R03.mp4 | 51.4% | 0.99 | 0.0 | 39.0 |
| deadlift/LLK_S01_R04.mp4 | 56.2% | 1.09 | 0.0 | 43.6 |
| deadlift/LLK_S01_R06.mp4 | 50.0% | 1.00 | 0.0 | 24.6 |
| deadlift/LLK_S03_R11.mp4 | 1.7% | 1.04 | 16.9 | 49.3 |
| deadlift/LLK_S04_R05.mp4 | 34.3% | 1.34 | 0.0 | 60.9 |
| deadlift/LLK_S05_R15.mp4 | 46.7% | 0.97 | 0.0 | 41.3 |
| deadlift/LLK_S05_R17.mp4 | 61.3% | 1.04 | 0.0 | 35.1 |
| deadlift/LLK_S06_R31.mp4 | 69.1% | 0.90 | 0.0 | 54.7 |
| deadlift/LLK_S06_R32.mp4 | 23.7% | 1.06 | 0.0 | 37.4 |
| deadlift/deadlift_01.mp4 | 1.0% | 1.00 | 50.6 | 67.0 |
| deadlift/deadlift_02.mp4 | 10.1% | 0.80 | 51.3 | 35.8 |
| deadlift/deadlift_03.mp4 | 28.1% | 0.85 | 52.5 | 67.0 |
| benchpress/bench_01.mp4 | 1.5% | 0.84 | 41.7 | 44.0 |
| benchpress/bench_02.mp4 | 31.4% | 1.02 | 34.7 | 45.7 |
| benchpress/bench_03.mp4 | 99.4% | 0.87 | 39.3 | 47.0 |
| squat/squat_01.mp4 | 59.1% | 0.81 | 48.6 | 66.0 |
| squat/squat_02.mp4 | 77.1% | 0.91 | 59.1 | 73.9 |
| squat/squat_03.mp4 | 75.4% | 0.95 | 59.4 | 60.7 |
| team9/squat/stu10_70.mp4 | 0.4% | 0.86 | 34.0 | 44.7 |
| team9/squat/stu3_64.mp4 | 8.6% | 0.93 | 41.5 | 46.4 |
| team9/squat/stu8_67.mp4 | 0.0% | — | 52.2 | — |
| team9/benchpress/stu9_10.mp4 | 54.2% | 1.07 | 16.6 | 46.7 |
| team9/benchpress/stu2_10_edited.mp4 | 1.3% | 0.86 | 40.6 | 64.0 |
| team9/deadlift/stu7_1_1.mp4 | 28.0% | 1.38 | 0.0 | 38.3 |
| team9/deadlift/stu4_2_-1.mp4 | 0.0% | — | 0.0 | — |
| team9/deadlift/deadlift_08.mp4 | 61.9% | 0.84 | 46.8 | 61.3 |

### M2의 가로 > 세로 26개

M2의 원본 통계에서 키 관측이 있고 상자 비가 1보다 큰 26개를 그대로 선택했다.

| 영상 | M2 상자 비 | 타원 사용률 | 장축/상자 세로 | 타원 각도 |
|---|---:|---:|---:|---:|
| squat/val1235.mp4 | 1.075 | 99.0% | 1.14 | 42.3 |
| squat/val1256.mp4 | 1.070 | 99.0% | 1.17 | 52.4 |
| deadlift/deadlift_7.mp4 | 1.382 | 0.0% | — | — |
| squat/stu6_11_2.mp4 | 1.004 | 5.6% | 1.13 | 50.9 |
| squat/stu6_11_3.mp4 | 1.108 | 84.6% | 1.29 | 42.8 |
| deadlift/PB_S01_R10.mp4 | 1.009 | 83.9% | 0.94 | 26.9 |
| deadlift/PB_S05_R22.mp4 | 1.040 | 63.3% | 1.00 | 52.2 |
| deadlift/PB_S07_R19.mp4 | 1.012 | 43.8% | 0.81 | 35.0 |
| deadlift/GB_S01_R01.mp4 | 1.096 | 75.6% | 0.97 | 48.6 |
| deadlift/GB_S01_R07.mp4 | 1.091 | 94.2% | 0.98 | 28.1 |
| deadlift/GB_S01_R16.mp4 | 1.009 | 82.4% | 0.96 | 40.4 |
| deadlift/GB_S01_R18.mp4 | 1.006 | 46.9% | 0.91 | 21.3 |
| deadlift/GB_S04_R03.mp4 | 1.043 | 11.4% | 0.97 | 22.4 |
| deadlift/GB_S05_R47.mp4 | 1.016 | 13.6% | 0.93 | 33.7 |
| deadlift/GB_S06_R63.mp4 | 1.059 | 27.8% | 0.87 | 25.2 |
| deadlift/GB_S07_R14.mp4 | 1.021 | 41.6% | 1.00 | 47.0 |
| deadlift/LLK_S01_R02.mp4 | 1.045 | 51.5% | 1.00 | 32.4 |
| deadlift/LLK_S01_R03.mp4 | 1.002 | 51.4% | 0.99 | 39.0 |
| deadlift/LLK_S01_R04.mp4 | 1.046 | 56.2% | 1.09 | 43.6 |
| deadlift/LLK_S01_R06.mp4 | 1.052 | 50.0% | 1.00 | 24.6 |
| deadlift/LLK_S04_R05.mp4 | 1.059 | 34.3% | 1.34 | 60.9 |
| deadlift/LLK_S05_R15.mp4 | 1.070 | 46.7% | 0.97 | 41.3 |
| deadlift/LLK_S05_R17.mp4 | 1.037 | 61.3% | 1.04 | 35.1 |
| deadlift/LLK_S06_R31.mp4 | 1.025 | 69.1% | 0.90 | 54.7 |
| deadlift/LLK_S06_R32.mp4 | 1.113 | 23.7% | 1.06 | 37.4 |
| team9/deadlift/stu7_1_1.mp4 | 1.046 | 28.0% | 1.38 | 38.3 |

## (c) 확인 이미지

69개 각각 사용 프레임의 앞·중간·뒤 세 프레임을 `overlays/`에 넣었고, 원판 부근 확대 `review_sheets/` 9장을 직접 열어 확인했다. 측정이 거절된 프레임은 파랑 상자만 보인다. 한 프레임만 used인 영상은 같은 프레임을 반복한다. 다음은 세 표본 중 명백한 불일치 또는 안쪽 테두리·바 끝과 혼동이 의심된 영상이며, 세 표본 밖의 프레임까지 검증한 목록은 아니다.

- `squat/squat_11.mp4`
- `squat/val1235.mp4`
- `squat/val1256.mp4`
- `benchpress/bench press_31.mp4`
- `benchpress/stu8_7.mp4`
- `benchpress/test153.mp4`
- `squat/stu6_11_3.mp4`
- `squat/val1250.mp4`
- `deadlift/deadlift_sihoo02.mp4`
- `deadlift/PB_S01_R10.mp4`
- `deadlift/PB_S04_R03.mp4`
- `deadlift/PB_S05_R22.mp4`
- `deadlift/PB_S06_R25.mp4`
- `deadlift/PB_S07_R19.mp4`
- `deadlift/GB_S01_R01.mp4`
- `deadlift/GB_S01_R07.mp4`
- `deadlift/GB_S01_R11.mp4`
- `deadlift/GB_S01_R16.mp4`
- `deadlift/GB_S01_R18.mp4`
- `deadlift/GB_S04_R04.mp4`
- `deadlift/GB_S04_R07.mp4`
- `deadlift/GB_S05_R47.mp4`
- `deadlift/GB_S07_R12.mp4`
- `deadlift/GB_S07_R14.mp4`
- `deadlift/GB_S07_R20.mp4`
- `deadlift/LLK_S01_R02.mp4`
- `deadlift/LLK_S01_R03.mp4`
- `deadlift/LLK_S01_R04.mp4`
- `deadlift/LLK_S04_R05.mp4`
- `deadlift/LLK_S05_R15.mp4`
- `deadlift/LLK_S05_R17.mp4`
- `deadlift/LLK_S06_R31.mp4`
- `deadlift/LLK_S06_R32.mp4`
- `deadlift/deadlift_03.mp4`
- `benchpress/bench_02.mp4`
- `benchpress/bench_03.mp4`
- `squat/squat_01.mp4`
- `squat/squat_02.mp4`
- `squat/squat_03.mp4`
- `team9/squat/stu10_70.mp4`
- `team9/benchpress/stu9_10.mp4`
- `team9/benchpress/stu2_10_edited.mp4`
- `team9/deadlift/deadlift_08.mp4`

## (d) 동시 촬영

같은 세트를 0°·45°에서 동시에 촬영한 V 영상은 찾지 못했다. 대기.

## 처리 속도와 시험

69개 저장 원판 행과 원본 영상 재생·타원 검출·보간을 4개 작업으로 처리했다. 작업별 경과 합계 35.2초, 영상 중앙값 0.16초다. 이 시간은 기존 검출/포즈를 제외한 추가 재생 단계이며 병렬 작업별 시간이다. 6스레드 전체 실행 시간은 `benchmark.json`을 참조한다.

합성 테두리·잡음·잘림, 대칭 변환, 주축·짧은 경로, box 대체, 60° 임계와 스키마 시험을 포함했다. 전체 시험: 파이프라인 188 passed (Starlette 의존성 경고 1개), 루트 feedback 235 passed. Playwright에서 경고 문구와 콘솔 오류 0개도 확인했다.

6스레드 `stu10_70` 단일 영상 전체 실행: box 34.89초, ellipse 34.41초 (각 1회, 두 방식 모두 타원 측정을 수행하므로 이 차이는 순수 오버헤드가 아니다). 같은 456프레임 영상의 추가 타원 재생 3회는 0.86, 0.81, 0.80초. 원판 box 행은 M3c에서 같은 영상으로 생성한 행과 gzip 해제 후 SHA-256 `58a00cc104ab7938846bdde6b4158aa10fe7ab0c7e82f70eb1b44535a38a37e3`으로 일치했다.

## 재현

```bash
cd plate_pipeline
python eval/m4/cohort.py
python eval/m4/simulation.py
python eval/m4/review.py
python eval/m4/report.py
python -m pytest -q tests eval/midterm/test_metrics.py
```
이 스크립트는 M0/M2 원본 행과 포즈 및 영상의 로컬 경로를 필요로 한다.
