# 피드백 모듈 설계 — 세트 후 리포트 데모

이 문서가 구현의 기준이다. 구현이 달라지면 이 문서를 같이 고친다.

---

## 0. 정의와 원칙

**한 줄 정의**: 사용자가 고른 레퍼런스(ref) 영상의 1 rep과 사용자 영상의 rep 하나하나를 같은 포지션끼리 맞춰 놓고, 어디가 얼마나 다른지를 영상 위 표시와 문장으로 보여준다. 맞다/틀리다는 판정하지 않는다.

**원칙**
1. **비교만 한다.** 정답 범위나 판정은 없다. 화면에 그리는 범위는 "ref ± 임계값"이다.
2. **비교 단위는 rep × 포지션 × 지표.** 사용자 rep 하나하나를 ref 1 rep과만 비교한다. rep끼리는 비교하지 않는다. rep이 1개만 남아도 피드백을 낸다.
3. **템포와 자세를 나눈다.** 포지션끼리 맞추면 속도 차이가 지워지고 자세만 남는다. 템포는 phase 길이로 따로 잰다.
4. **방향과 크기를 꼭 말한다.** 예: "바닥에서 무릎을 레퍼런스보다 22° 더 굽혔습니다."
5. **rep마다 목표는 1개.** k(차이의 절댓값 ÷ 임계값)가 가장 큰 차이.
6. **못 잰 것은 빼고 이유를 보여준다.**
7. **3D는 보기 전용.** 3D 좌표로 계산하지 않는다.
8. **문장은 템플릿.** LLM은 이번 범위 밖.

**이번 데모에서 하지 않는 것**: 판정·정답 범위, LLM 문장, rep끼리 비교, 날짜별 추적, 실시간 피드백, 3D 수치, 드릴 추천, 음성 질문, 그리기 도구, 몸 이동의 cm 단위.

---

## 1. 범위 (데모)

| 항목 | 데모 |
|---|---|
| 종목 | 스쿼트 (벤치·데드로 넓힐 수 있게 종목을 인자로 받는다) |
| 촬영 각도 | 측면. 사용자와 ref가 같은 각도여야 비교한다 |
| 입력 | 사용자 영상 1개, ref 영상 1개 (ref는 rep 1개만 쓴다) |
| 실행 | CLI → 정적 HTML (`file://`로 열림) |
| rep·포지션 분할 | 1~5단계는 수동 라벨, 6단계부터 ST-GCN |
| 임계값 | v4 `DELTA_THRESHOLDS` 그대로 (임시), 템포 임계는 임의값 |
| 3D | 8단계. RTMPose + 3D lifter, 별도 환경 |

---

## 2. 전체 흐름

```
data/demo.json
 │
 ├─ gate       입력 확인 ── 실패 ──▶ failure.html
 │
 ├─ Track A (메인, MediaPipe)
 │    pose       MediaPipe 33관절 추출 + 캐시
 │    segment    rep · phase · 포지션   (1~5단계 수동 라벨 → 6단계부터 ST-GCN)
 │    metrics    프레임별 지표 → 포지션 값, 템포
 │    compare    ref와 차이, k, 색
 │    select     rep마다 목표 1개 + 근거
 │    describe   템플릿 문장
 │    export     report.json → report.html
 │
 └─ Track B (3D 보기 전용, 별도 환경)
      RTMPose → 3D lifter (H36M 17관절) → pose3d_*.js → report.html 3D 탭
```

ST-GCN은 MediaPipe 33관절로 학습돼서 Track A는 MediaPipe를 쓴다. 3D 뷰는 앞으로 쓸 RTMPose로 따로 만든다.

---

## 3. 설정 파일 `data/demo.json`

```json
{
  "exercise": "squat",
  "user": {
    "video": "data/videos/user.mp4",
    "view": "side",
    "reps": [
      {"index": 1, "phases": {"down": [184, 231], "bottom": [231, 234], "up": [234, 276]}}
    ]
  },
  "ref": {
    "video": "data/videos/ref.mp4",
    "view": "side",
    "rep": 1,
    "reps": [
      {"index": 1, "phases": {"down": [42, 90], "bottom": [90, 92], "up": [92, 153]}}
    ]
  }
}
```

- 숫자는 예시다.
- 경로는 프로젝트 루트 기준.
- `view`: `side` | `front` | `diag`
- `ref.rep`: ref에서 비교에 쓸 rep 번호 (기본 1)
- `reps`: 수동 라벨. 있으면 labels 모드, 없으면 model 모드 (6단계 이후)
- phase 구간은 반열림 `[start, end)` 프레임 번호. 라벨은 `data/numbered/`의 프레임 번호 영상 (`tools/frame_number.py`로 만든 것)을 보고 적는다.
- `data/labels/`의 기존 라벨 파일 (`expert` 키)을 바탕으로 만든다.
- `trim` (선택): `[start, end)`. model 모드에서 이 구간만 분석한다.
- 예전 라벨 파일의 `expert` 키는 `ref`와 같은 뜻으로 받는다.

---

## 4. 데이터 형식 (`out/<run>/`)

JSON은 UTF-8, `ensure_ascii=False`. 값이 없으면 `null`. 아래 숫자는 예시.

### 4.1 `pose_user.npz`, `pose_ref.npz` (캐시)
- `lms`: float32 `(T, 33, 4)` — `[x, y, z, visibility]`, 정규화 이미지 좌표. 검출 실패 프레임은 NaN
- `valid`: bool `(T,)`
- `fps`, `w`, `h`, `total`
- 캐시 확인용 `video` (영상의 절대경로), `mtime_ns` (영상 파일 수정 시각, 나노초)
- 추출·스무딩은 `offline_compare.extract_sequence`와 같게 한다 (v4 `LandmarkSmoother`)
- 영상 파일 경로와 수정 시각이 같으면 캐시를 쓴다. `--no-cache`로 무시.

### 4.2 `segments_user.json`, `segments_ref.json`
```json
{
  "video": "data/videos/user.mp4", "fps": 29.97, "size": [406, 720], "total": 911,
  "exercise": "squat", "view": "side", "facing": "left", "source": "labels",
  "reps": [
    {
      "rep": 1,
      "phases": {"down": [184, 231], "bottom": [231, 234], "up": [234, 276]},
      "positions": {"START": 184, "MID_DOWN": 209, "BOTTOM": 232, "MID_UP": 255, "LOCKOUT": 275},
      "side": "left"
    }
  ]
}
```
- `source`: `labels` | `model`. 포지션 수정을 적용하면 `+override`를 붙인다 (예: `labels+override`)
- `side`: 그 rep에서 카메라 쪽 몸 측면. 지표 계산과 관절 표시에 쓴다
- `facing`: 사람이 화면 왼쪽/오른쪽 중 어디를 보는지. ref 좌우 반전에 쓴다

### 4.3 `metrics_user.json`, `metrics_ref.json`
```json
{
  "exercise": "squat", "fps": 29.97, "total": 911,
  "frame_metrics": {
    "knee_angle": [null, 171.2, 170.8],
    "hip_angle": [], "trunk_lean": [], "foot_flatness": [], "head_hip_line": []
  },
  "reps": [
    {
      "rep": 1,
      "detect": {"down": 0.98, "bottom": 1.0, "up": 0.95},
      "positions": {
        "START": {"frame": 184, "values": {"knee_angle": 172.0, "hip_angle": 170.1, "trunk_lean": 8.2, "foot_flatness": 0.03, "head_hip_line": 0.10}},
        "BOTTOM": {"frame": 232, "values": {"knee_angle": 71.5, "hip_angle": 58.0, "trunk_lean": 41.0, "foot_flatness": null, "head_hip_line": 0.31}}
      },
      "tempo": {"down_s": 1.57, "bottom_s": 0.10, "up_s": 1.40, "ratio": 1.12},
      "excluded": false, "exclude_reason": null
    }
  ]
}
```
- `frame_metrics[지표]` 길이 = `total` (영상 전체 프레임)
- `positions`는 5개 모두, `values`는 `DISPLAY_METRICS[exercise]` 전부 (못 재면 `null`)

### 4.4 `findings.json`
```json
{
  "exercise": "squat", "ref_rep": 1,
  "thresholds": {"knee_angle": 18.0, "hip_angle": 18.0, "trunk_lean": 10.0, "foot_flatness": 0.08, "head_hip_line": 0.18},
  "tempo_thresholds": {"down_s": 0.3, "bottom_s": 0.2, "up_s": 0.3},
  "reps": [
    {
      "rep": 1, "excluded": false, "exclude_reason": null,
      "cells": [
        {"where": "BOTTOM", "metric": "knee_angle", "status": "ok",
         "user": 71.5, "ref": 93.5, "diff": -22.0, "unit": "deg", "threshold": 18.0,
         "k": 1.22, "dir": "-", "color": "yellow", "range": [75.5, 111.5]}
      ],
      "tempo": [
        {"metric": "down_s", "user": 1.57, "ref": 1.60, "diff": -0.03, "unit": "s",
         "threshold": 0.3, "k": 0.1, "dir": "-", "color": "none"}
      ],
      "tempo_ratio": {"user": 1.12, "ref": 0.79},
      "goal": {
        "type": "group", "key": "depth", "where": "BOTTOM",
        "metrics": ["knee_angle", "hip_angle"], "dir": "-", "k": 1.22, "color": "yellow",
        "evidence": [{"where": "BOTTOM", "metric": "knee_angle"}, {"where": "BOTTOM", "metric": "hip_angle"}]
      },
      "tempo_goal": null
    }
  ]
}
```
- `cells`: 포지션 5개 × 지표 전부. `status`: `ok` | `unavailable` | `advisory`. `unavailable`이면 숫자 필드는 `null`
- `unit`: `deg` | `ratio` | `s`
- `dir`: `+` (사용자 > ref) | `-` | `0`
- `color`: `none` | `yellow` | `red` (5.3)
- `goal.type`: `single` | `group` | `none`. 제외된 rep은 `goal`, `tempo_goal` 모두 `null`
- `tempo_goal`: 템포 항목 하나 (`tempo`의 원소와 같은 형식) 또는 `null`

### 4.5 `report.json` (HTML에 주입하는 데이터)
findings + 문장 + 화면용 데이터. 템플릿의 `/*__DATA__*/null` 자리에 넣는다 (기존 방식).
```json
{
  "meta": {
    "exercise": "squat", "view": "side", "threshold_note": "임계값은 임시값(v4)",
    "positions": ["START", "MID_DOWN", "BOTTOM", "MID_UP", "LOCKOUT"],
    "position_labels": {"START": "시작", "MID_DOWN": "하강 중간", "BOTTOM": "바닥", "MID_UP": "상승 중간", "LOCKOUT": "락아웃"},
    "metric_labels": {"knee_angle": "무릎각"},
    "thresholds": {}, "tempo_thresholds": {}, "warnings": [],
    "angles": {
      "user": {"ratio": 0.25, "view": "side", "samples": [{"rep": 1, "frame": 184, "ratio": 0.25, "view": "side"}], "warning": null},
      "ref": {"ratio": 0.27, "view": "side", "samples": [{"rep": 1, "frame": 42, "ratio": 0.27, "view": "side"}], "warning": null}
    }
  },
  "user": {
    "video": "../../data/videos/user.mp4", "fps": 29.97, "size": [406, 720], "total": 911,
    "facing": "left", "frames": {"184": [[0.51, 0.32, 0.99]]}, "frame_metrics": {}, "reps": []
  },
  "ref": {
    "video": "../../data/videos/ref.mp4", "fps": 29.89, "size": [1080, 1920], "total": 342,
    "facing": "left", "mirror": false, "rep": 1, "frames": {}, "frame_metrics": {}, "reps": [], "sprite": {}
  },
  "connections": [[11, 12]],
  "metric_joints": {"knee_angle": {"left": [23, 25, 27], "right": [24, 26, 28]}},
  "reps": [
    {
      "rep": 1, "excluded": false, "exclude_reason": null, "tab_label": "바닥 · 더 깊게",
      "goal": {
        "type": "group", "where": "BOTTOM", "k": 1.22, "color": "yellow",
        "sentence": "바닥에서 레퍼런스보다 더 깊게 앉았습니다 (무릎 22°, 엉덩이 19°).",
        "frame": 232, "ref_frame": 91, "joints": [23, 25, 27], "evidence": []
      },
      "cells": [],
      "tempo": {"items": [], "ratio": {}, "goal": null, "sentence": null}
    }
  ],
  "pose3d": {"user": "pose3d_user.js", "ref": "pose3d_ref.js"}
}
```
- `video`는 **report.html 위치 기준 상대경로** (`os.path.relpath`). 틀리면 검은 화면이 된다
- `frames`: rep 구간 ±15프레임만. 관절마다 `[x, y, visibility]` (기존 `dump_frames`)
- `user.reps`, `ref.reps`: segments의 rep (ref는 비교에 쓰는 rep 하나만)
- `ref.sprite`: 기존 `build_sprite` 결과. ref는 스프라이트로 그려서 끊기지 않게 한다
- `ref.mirror`: 사용자와 ref의 `facing`이 다르면 `true`
- `meta.warnings`: 입력 검사와 각도 자동 확인의 경고 문장. 세트 바에 그대로 표시한다
- `meta.angles`: 영상별 각도 자동 확인 값·분류, START별 표본과 경고. 대표값은 유효한 START 표본의 중앙값이다. 표본이 없으면 `ratio`, `view`는 `null`이다
- `goal.evidence`, `cells`: findings의 cell 전체 + `label`
- `excluded`인 rep은 `goal: null`, `exclude_sentence`를 넣는다
- `tab_label`, `sentence`는 5.5 규칙

### 4.6 `positions_override.json` (포지션 수정 모드에서 저장)
```json
{"user": {"1": {"BOTTOM": 233}}, "ref": {"1": {"LOCKOUT": 152}}}
```
`--override <파일>`로 넣으면 segment 단계 끝에서 덮어쓰고 그 뒤 단계를 다시 계산한다.

### 4.7 `pose3d_user.js`, `pose3d_ref.js` (8단계)
```js
window.POSE3D = window.POSE3D || {};
window.POSE3D.user = {"joints": "h36m17", "fps": 29.97, "total": 911, "frames": [null, [[0.0, 0.0, 0.0]]]};
```
- `frames` 길이 = 영상 프레임 수, 인덱스 = 영상 프레임 번호, 없으면 `null`
- H36M 17관절 순서: 0 골반 · 1 오른엉덩이 · 2 오른무릎 · 3 오른발목 · 4 왼엉덩이 · 5 왼무릎 · 6 왼발목 · 7 척추 · 8 흉곽 · 9 목 · 10 머리 · 11 왼어깨 · 12 왼팔꿈치 · 13 왼손목 · 14 오른어깨 · 15 오른팔꿈치 · 16 오른손목
- `fetch()` 대신 `<script src>`로 읽는다. 파일이 없으면 3D 탭에 "3D 데이터 없음"

---

## 5. 규칙

### 5.1 포지션

포지션 정의는 스쿼트 기준이다. 데드리프트는 바닥에서 시작해서 순서가 다르다 (이번 범위 밖).

엉덩이 높이 `hip_y(t)`: `unified_feedback_v4.mid(lms, 23, 24)`의 y. 5프레임 이하 빈칸은 선형 보간, 5프레임 이동평균. 이미지 좌표라 **값이 클수록 낮다**.

**labels 모드**
- START = down 시작, LOCKOUT = up 끝 − 1
- BOTTOM = bottom 구간 안에서 `hip_y`가 최대인 프레임 (값이 없으면 구간 가운데)
- MID_DOWN = [START, BOTTOM]에서 `hip_y`가 처음으로 (hip_y(START) + hip_y(BOTTOM)) / 2 이상이 되는 프레임
- MID_UP = [BOTTOM, LOCKOUT]에서 `hip_y`가 처음으로 (hip_y(BOTTOM) + hip_y(LOCKOUT)) / 2 이하가 되는 프레임
- MID를 못 찾으면 해당 phase의 가운데 프레임

**model 모드 (6단계)**
1. ST-GCN 프레임별 phase를 정리하고 (7장 6단계), down 연속 구간 + 바로 뒤 up 연속 구간을 rep 후보로 본다. 창 W = [down 시작 − 10, up 끝 + 10), 영상 범위로 자른다
2. BOTTOM = W 안에서 `hip_y` 최대
3. 최고점(가장 작은 y)은 각각 W 시작~BOTTOM의 최소 `pre_min`, BOTTOM~W 끝의 최소 `post_min`이다. D1 = hip_y(BOTTOM) − pre_min, D2 = hip_y(BOTTOM) − post_min. 이동폭과 최고점은 탐색을 확장해도 W 기준을 유지한다
4. 속도 `v(t) = (hip_y(t+1) − hip_y(t−1)) / 2`. 정지는 `|v| < 기준 × D1`(START) 또는 `|v| < 기준 × D2`(LOCKOUT)다. START·LOCKOUT 각각 0.005로 전체 탐색 범위를 먼저 검사하고, 구간을 못 찾으면 0.01 → 0.02 순서로 다시 검사한다. 중앙차분에 필요한 양쪽 프레임의 값이 없으면 정지로 취급하지 않는다
5. START: BOTTOM 이전에서 거꾸로 가며 `hip_y ≤ pre_min + 0.2 × D1`이면서 정지인 프레임이 3개 연속인 첫 구간을 찾고, 그 구간에서 BOTTOM에 가장 가까운 프레임을 고른다. LOCKOUT: BOTTOM 이후에서 앞으로 가며 `hip_y ≤ post_min + 0.2 × D2`이면서 정지인 프레임이 3개 연속인 첫 구간의 첫 프레임을 고른다
6. 탐색 범위는 [W 시작 − 30, W 끝 + 30)을 영상 범위로 자른 구간이다. 이전 rep의 LOCKOUT보다 앞이나 다음 rep의 START보다 뒤는 탐색하지 않는다. 다음 후보의 START를 먼저 구해 LOCKOUT 탐색 상한으로 쓰고, 이전에 확정된 LOCKOUT으로 최종 START 탐색 하한을 정한다
7. 세 기준 모두 정지 구간을 찾지 못한 endpoint만 기존 5% 규칙으로 대체한다: START = W 안의 BOTTOM 이전에서 `hip_y ≤ pre_min + 0.05 × D1`인 마지막 프레임, LOCKOUT = W 안의 BOTTOM 이후에서 `hip_y ≤ post_min + 0.05 × D2`인 첫 프레임. 대체 프레임도 인접 rep 경계를 넘지 않는다. 유효한 순서·범위의 포지션을 구성할 수 없으면 해당 후보를 버린다
8. bottom phase = BOTTOM을 포함하고 hip_y ≥ hip_y(BOTTOM) − 0.03·min(D1, D2) 인 연속 구간. down = [START, bottom 시작), up = [bottom 끝, LOCKOUT + 1)
9. MID는 labels 모드와 같다
10. min(D1, D2) ÷ 몸통 길이(`v4.torso_len`) < 0.3 이면 rep이 아니라고 보고 버린다. 몸통 길이는 후보 창 W의 유효 프레임에서 구한 `torso_len`의 중앙값이다
- model rep의 `endpoint_methods`에는 START·LOCKOUT별 `velocity` 또는 `fallback_5pct`를 기록하고, `endpoint_speed_ratios`에는 선택된 정지 기준(0.005, 0.01, 0.02)을 기록한다. 해당 endpoint를 override로 바꿨으면 method는 `override`다. 5% 대체·override의 speed ratio는 `null`이다. 비교 표에는 선택 기준과 5% 대체 여부를 표시한다
- 상수 10, 0.05, 0.03, 0.3 및 정지 기준 목록 0.005·0.01·0.02, 위쪽 영역 0.2, 연속 프레임 3, 추가 탐색 30은 `config.py`에 둔다 (임의값)

**공통**
- `side`: rep 구간에서 `export_overlay.dominant_side`
- `facing`: 각 rep의 START~START+5 프레임에서, `side` 쪽 발끝(31/32) x − 뒤꿈치(29/30) x 평균의 부호 (+ → `right`). 영상 전체는 rep 다수결
- `positions_override.json`이 있으면 마지막에 덮어쓰고 순서 START < MID_DOWN < BOTTOM < MID_UP < LOCKOUT 를 검사한다. 깨지면 에러

### 5.2 측정
- 프레임 지표: `compute_metrics(lms, exercise, side)`. rep 구간(±15)은 그 rep의 `side`, 밖은 `SideLock`을 순서대로 적용한 값
- 포지션 값: 포지션 프레임 ±2프레임의 값 중 `null`이 아닌 것의 중앙값. 3개 미만이면 `null`
- 템포: `down_s`, `bottom_s`, `up_s` = phase 프레임 수 ÷ fps. `ratio` = down_s ÷ up_s. bottom은 라벨 오차가 커서 비율에 넣지 않는다
- `detect[phase]` = phase 안에서 포즈가 잡힌 프레임 비율
- 제외: 어느 phase든 `detect` < 0.60 (`offline_compare.MIN_VALID_RATIO`) → 제외. 이유 예: `"하강 구간 관절 검출 45% (기준 60%)"`

### 5.3 비교
- 대상: 포지션 5개 × `DISPLAY_METRICS[exercise]`
- 둘 다 값이 있으면: `diff = user − ref`, `k = abs(diff) ÷ threshold`, `range = [ref − threshold, ref + threshold]`
- 색: k ≤ 1 → `none`, 1 < k ≤ 1.5 → `yellow`, k > 1.5 → `red`
- `ADVISORY_METRICS[exercise]`에 있으면 `status: advisory` (값과 색은 보여주되 목표 후보가 아님)
- v4에서 임계값이 0인 표시용 `bar_proxy_conf`는 값과 차이를 유지하되 `k: null`, `color: none`으로 둔다. 목표 후보에는 넣지 않는다.
- 하나라도 `null`이면 `status: unavailable`
- 추가 제외 조건: `ok` + `advisory` 칸이 전체의 50% 미만 (`offline_compare.MIN_REP_METRIC_RATIO`) → 제외. 이유 예: `"측정 가능한 칸 40% (기준 50%)"`
- 템포: `down_s`, `bottom_s`, `up_s` 각각 같은 방식. 임계 0.3 / 0.2 / 0.3초 (임의값). `ratio`는 보여주기만 한다
- ref rep이 제외 조건에 걸리면 gate 실패 ("레퍼런스 rep 검출 부족")
- 단위: 각도 지표는 `deg`, `offline_compare.RATIO_METRICS`에 있으면 `ratio` (몸통 길이 대비)

### 5.4 선택
1. 후보 = `status: ok`이고 k > 1인 칸
2. 묶음 (`config.GROUPS`): 같은 포지션에서 묶음 지표가 모두 후보이고 방향이 같으면 묶음 후보 1개로 만든다 (k = 멤버 중 최대). 묶인 칸은 개별 후보에서 뺀다
   - 데모 묶음은 1개: `depth` — BOTTOM에서 `knee_angle` + `hip_angle`
3. 목표 = k가 최대인 후보. 동점이면 포지션 순서가 앞선 것, 그다음 `DISPLAY_METRICS` 순서
4. 후보가 없으면 `type: none`
5. 근거 (`evidence`, 최대 3개): 목표 칸 (묶음이면 멤버를 k 큰 순) → 같은 지표의 다른 포지션 중 k > 1인 칸 (k 큰 순)
6. 강조 관절 = 목표 지표들의 `export_overlay.joints_for(metric, side)` 합집합
7. `tempo_goal` = 템포 항목 중 k 최대, k ≤ 1이면 `null`. 템포는 자세 목표와 경쟁하지 않는다

### 5.5 서술

포지션 이름: 시작 · 하강 중간 · 바닥 · 상승 중간 · 락아웃

숫자: 각도는 `round(abs(diff))`° · 비율은 `round(abs(diff) × 100)`% (몸통 길이 대비) · 초는 소수 1자리

| 지표 | + (사용자 > ref) | − (사용자 < ref) |
|---|---|---|
| knee_angle | 무릎을 레퍼런스보다 {n}° 덜 굽혔습니다 | 무릎을 레퍼런스보다 {n}° 더 굽혔습니다 |
| hip_angle | 엉덩이를 레퍼런스보다 {n}° 덜 접었습니다 | 엉덩이를 레퍼런스보다 {n}° 더 접었습니다 |
| trunk_lean | 상체를 레퍼런스보다 {n}° 더 숙였습니다 | 상체를 레퍼런스보다 {n}° 덜 숙였습니다 |
| foot_flatness | 발 앞뒤 높이 차이가 레퍼런스보다 몸통 길이의 {n}%만큼 큽니다 | 발 앞뒤 높이 차이가 레퍼런스보다 몸통 길이의 {n}%만큼 작습니다 |
| head_hip_line | 머리와 엉덩이의 가로 거리가 레퍼런스보다 몸통 길이의 {n}%만큼 멉니다 | 머리와 엉덩이의 가로 거리가 레퍼런스보다 몸통 길이의 {n}%만큼 가깝습니다 |

v4 정의: `foot_flatness` = 뒤꿈치 y와 발끝 y 차이의 절댓값 ÷ 몸통 길이, `head_hip_line` = 코 x와 엉덩이 중점 x 차이의 절댓값 ÷ 몸통 길이. 둘 다 절댓값이라 앞/뒤, 뒤꿈치/발끝을 구분하지 않으니 문구도 구분하지 않는다.

- 단일: `"{포지션}에서 {문구}."` → "바닥에서 무릎을 레퍼런스보다 22° 더 굽혔습니다."
- 묶음 depth: "바닥에서 레퍼런스보다 더 깊게 앉았습니다 (무릎 22°, 엉덩이 19°)." — 방향이 +면 "덜 깊게"
- 없음: "모든 포지션에서 레퍼런스와의 차이가 임계값 안에 있습니다."
- 템포
  - 하강·상승: "하강이 레퍼런스보다 0.4초 느립니다 (하강:상승 1.3:1, 레퍼런스 1.0:1)." — +면 느립니다, −면 빠릅니다
  - 바닥: "바닥에서 레퍼런스보다 0.3초 더 머물렀습니다." — −면 덜
- 제외: "이 rep은 비교에서 뺐습니다 — {이유}."
- 탭 라벨: 단일 "바닥 · 무릎각 −22°" · 묶음 "바닥 · 더 깊게" · 없음 "차이 없음" · 제외 "제외"
- 금지어 (`config.BANNED`): 좋, 나쁘, 잘못, 틀렸, 틀린, 올바, 정확한 자세, 해야, 하세요, 하십시오, 권장, 추천
- 목표 문장과 템포 문장에는 숫자+단위가 반드시 있다 ("차이 없음", "제외" 문장만 예외)

---

## 6. 화면

### 6.1 리포트 구성 (`report.html`)
1. **세트 바**: 종목 · rep 수 (제외 n) · 각도 · 레퍼런스 파일명과 rep · "임계값 임시" · gate 경고
2. **rep 탭**: 이름 = `tab_label`, 목표 색 점. 누르면 그 rep START로 이동
3. **영상 영역**: 사용자 영상 + ref(스프라이트) 나란히, 각각 캔버스 스켈레톤. 아래에 현재 프레임 지표 카드 (3장 보이고 가로 스크롤, 목표 지표 먼저, 사용자 값 / ref 값 / 차이, k 색)
4. **타임라인**: rep 구간 막대 (phase 색) + 포지션 점 + 재생헤드. 아래 버튼 줄: 포지션 5개 · ▶ · 속도 · 동기화 · 보기 · 스켈레톤 · 수정
5. **오른쪽 패널**
   - **목표 카드**: 제목 "레퍼런스와 가장 다른 곳" · 문장 · 근거 카드 1~3개 (지표명 · 포지션 배지 · 사용자 값 · ref ± 임계 범위 막대와 그 위 사용자 값 마커 · 색) · [이 장면 보기] · 템포 문장 (있으면). 제외된 rep은 제외 문장만
   - **템포 패널**: 막대 두 줄 (사용자 / ref). 하강·바닥·상승 칸 길이 = 초 (두 줄이 같은 축), 칸마다 숫자, 하강:상승 비, k > 1인 칸은 색
   - **포지션 표**: 행 = 지표, 열 = 포지션 5개, 칸 = 부호 있는 차이 (`−22°`, `+9%`)와 색. 못 잰 칸은 `—`, 보조 지표는 회색 "참고". 칸을 누르면 그 포지션으로 이동하고 그래프 지표가 바뀐다
   - **그래프**: 가로축 = 포지션 기준 진행도 (0 START … 4 LOCKOUT, 포지션 사이는 프레임 선형). 사용자 곡선 · ref 곡선 · ref ± 임계 띠 · 포지션 세로선 · 재생헤드. 기본 지표 = 목표 지표. 라이브러리 없이 canvas나 SVG

색: down `#2E5C8A` · bottom `#6B4E9E` · up `#2F7A55` · 노랑 `#E3A008` · 빨강 `#D93F3F`. 노랑·빨강은 차이 표시에만 쓴다.

### 6.2 보기 모드
- **2D (기본)**: 원본 영상 나란히. 스켈레톤 켜기/끄기 (기본 켬). 현재 rep의 목표 관절·뼈는 목표 색 (노랑/빨강), 나머지는 흰색
- **겹쳐**: 사용자 영상 위에 ref 스켈레톤을 반투명으로. rep START 프레임에서 ref의 엉덩이–발목 길이를 사용자에 맞게 스케일하고 발목 위치를 맞춘다
- **3D**: 8단계 (6.6)
- `ref.mirror`면 ref 영상과 스켈레톤을 좌우 반전해서 그린다

### 6.3 동기화
- **켜짐 (기본)**: 사용자 프레임이 포지션 i와 i+1 사이 진행도 t에 있으면, ref 프레임 = ref 포지션 i와 i+1 사이의 같은 t. rep 밖에서는 ref START에 멈춘다
- **꺼짐**: 사용자 rep START에 ref START를 맞춘 뒤 각자 실제 시간으로 재생한다 (템포 차이가 그대로 보인다). ref rep이 끝나면 LOCKOUT에 멈춘다
- 속도: 0.25 · 0.5 · 1×
- ref는 스프라이트라 프레임 단위로 그린다 (기존 `player_template.html` 방식)

### 6.4 조작
Space 재생/정지 · ←/→ 1프레임 · Shift+←/→ 10프레임 · 1~5 포지션 · `[` `]` 이전/다음 rep.
[이 장면 보기] = 목표 포지션으로 두 영상 이동 + 정지 + 목표 관절 강조.

### 6.5 포지션 수정 모드
- [수정]을 켜고 포지션 버튼으로 하나를 고른 뒤 ←/→ (Shift는 5프레임)로 옮긴다. 사용자/ref 전환 가능. rep 구간 ±15프레임 안에서만
- 순서가 깨지는 이동은 막는다
- 옮기면 그 포지션 열의 표 칸, 근거 카드 값과 색을 브라우저에서 다시 계산한다 (`frame_metrics`로 ±2프레임 중앙값, 5.3 규칙). 목표 문장은 다시 만들지 않고 "수정됨 — 저장 후 다시 실행하면 문장이 갱신됩니다"를 띄운다
- [되돌리기]: 원래 값으로. [저장]: `positions_override.json` 다운로드 → `python run_demo.py ... --override <파일>`
- 팀이 테스트 영상의 포지션 정답을 만들 때도 이 모드를 쓴다

### 6.6 3D 탭 (8단계)
- 기본은 로봇 (뼈마다 캡슐 + 관절마다 구), 스틱 (선 + 작은 구)으로 전환
- 시점 프리셋: 측면 · 정면 · 위 · 뒤 + 드래그 회전
- 재생헤드와 같은 프레임을 보여준다
- 목표 관절은 빨간 구. 지표 → H36M 관절: knee_angle 오른쪽 1·2·3 / 왼쪽 4·5·6, hip_angle 오른쪽 14·1·2 / 왼쪽 11·4·5, trunk_lean 0·7·8, foot_flatness 발목 (오른쪽 3 / 왼쪽 6, H36M에 발이 없음), head_hip_line 10·0
- 작은 원본 영상 PIP
- 3D 좌표로 만든 숫자는 화면에 없다
- three.js는 `web/vendor/`에 둔다. `file://`에서는 ES 모듈(`type="module"`)이 막히므로 classic 빌드를 쓴다 (예: three 0.147.0의 `build/three.min.js`). 드래그 회전은 직접 구현하거나 같은 버전의 `examples/js/controls/OrbitControls.js`
- 위쪽 축을 확인해서 y가 위가 되게 바꾸고, 바닥 = 두 발목 중 낮은 쪽

### 6.7 실패 화면 (`failure.html`)
- 제목 "분석할 수 없습니다" · 이유 (처음 걸린 검사, 수치 포함) · 해결 방법 한 줄
- 검사 번호(1~5)를 함께 표시한다. 이유는 사람이 읽는 문장으로 만들며 원시 딕셔너리는 표시하지 않는다
- 체크리스트: 측면에서 90°로 찍었는지 · 전신과 원판이 다 보이는지 · 너무 멀지 않은지 · 뒤에 다른 사람이 없는지 · 고정 fps·H.264인지 (`tools/check_video.py --fix`)
- 다시 실행하는 명령

---

## 7. 구현 단계

한 단계씩 한다. 단계가 끝나면 보고하고, 사용자가 확인·커밋한 뒤 다음 단계로 간다.

모든 단계 공통 완료 조건: v4를 HEAD로 복원한 뒤 전체 테스트 `python -m pytest -q`에서 기존 실패 2개 (`tests/test_unified_feedback_v4_llm.py`의 `test_local_feedback_selects_largest_normalized_difference`, `test_install_variant_is_idempotent_and_adds_llm_output_suffix`) 말고 새 실패가 없다. 새 테스트는 `tests/fb/test_fb_*.py`.

### 0단계 — 뼈대
- 수정: 폴더 구조, `feedback/*.py` (빈 함수), `feedback/config.py` (5장 상수 전부), `run_demo.py` (인자 파싱, `--until`), `tests/fb/test_fb_smoke.py`, `.gitignore`, `data/demo.json` (`data/labels/`의 기존 라벨로 만든다: `expert` → `ref`, `view`는 둘 다 `side`, `ref.rep` = 1)
- `.gitignore`: 기존 `data/` 규칙을 `data/*`로 바꾸고 `!data/demo.json`, `!data/labels/`를 더한다 (나머지 `data/` 아래는 그대로 무시). `out/`, `player_baseline.html` 추가
- 영상 진단: `data/videos/`의 영상마다 `tools/check_video.py` 실행 결과를 보고한다. `--fix`는 하지 않는다. ffprobe가 없으면 코덱·가변 fps 검사는 건너뛴다
- `run_demo.py` 인자: `--config`, `--out`, `--until {segments,metrics,findings,report}`, `--segment {labels,model}` (기본: `reps`가 있으면 labels), `--override`, `--ref-width 300`, `--ref-quality 72`, `--no-cache`
- 완료 조건
  - `python -m pytest tests/fb -q` 통과, 공통 완료 조건, `python run_demo.py --help` 동작
  - `python export_overlay.py --labels data/labels/<기존 라벨>.json --out player_baseline.html`이 그대로 도는지 확인 (기존 코드 회귀)

### 1단계 — 포즈 + 분할 (labels 모드)
- 수정: `feedback/pose.py`, `feedback/segment.py`, `run_demo.py`, `tests/fb/test_fb_segment.py`
- 출력: `pose_*.npz`, `segments_*.json`
- 완료 조건
  - `python -m pytest tests/fb/test_fb_segment.py`: 합성 엉덩이 궤적 (코사인)으로 5.1 labels 규칙 검증 — START·LOCKOUT 위치, BOTTOM 최대, MID 교차, 순서, `facing` 부호, override 적용과 순서 에러
  - `--until segments` 실행 시 rep × 포지션 프레임 표 출력
- 사람 확인: `data/numbered/`의 프레임 번호 영상에서 표의 프레임 몇 개를 눈으로 확인

### 2단계 — 측정
- 수정: `feedback/metrics.py`, `run_demo.py`, `tests/fb/test_fb_metrics.py`
- 완료 조건
  - 테스트: 포지션 값 = 창 안 `compute_metrics` 값의 중앙값 · 유효값 3개 미만이면 `null` · 템포 = 프레임 수 ÷ fps · 검출 < 0.60이면 제외 + 이유 문자열
  - `--until metrics` 실행 시 rep × 포지션 × 지표 값 표와 템포 출력

### 3단계 — 비교·선택
- 수정: `feedback/compare.py`, `feedback/select.py`, `run_demo.py`, `tests/fb/test_fb_compare.py`, `tests/fb/test_fb_select.py`
- 완료 조건 (손으로 만든 metrics 입력)
  - k·색 경계 (k = 1 → none, k = 1.5 → yellow, 1.5 초과 → red) · unavailable · advisory는 목표에서 제외 · 50% 제외 규칙
  - depth 묶음 (방향이 같을 때만, 다르면 개별) · 단일 목표 · 동점 규칙 · 차이 없음 · 근거 최대 3개와 순서 · 템포 목표
  - `--until findings`로 `findings.json` 생성, rep별 목표 출력

### 4단계 — 문장
- 수정: `feedback/describe.py`, `tests/fb/test_fb_describe.py`
- 완료 조건: 고정 입력 → 기대 문장과 정확히 일치 (지표 5개 × ±, 묶음 ±, 없음, 템포 3종 × ±, 제외, 탭 라벨) · 금지어 없음 · 숫자+단위 포함

### 5단계 — 리포트 화면
- 수정: `feedback/export.py`, `web/report_template.html`, `run_demo.py`, `tests/fb/test_fb_export.py`
- 시작점: `player_template.html`의 영상·스프라이트·캔버스·`diagnose()`·키보드 코드를 복사해서 `web/report_template.html`을 만든다 (원본은 그대로)
- 버리는 것: phase 구간 min/max 비교, 지표별 peak 북마크 줄 → 포지션 기준으로 바뀜
- 완료 조건 (자동): `report.json` 값이 `findings.json`과 같음 · 영상 상대경로가 `report.html` 기준으로 실제 파일을 가리킴 · HTML에 데이터가 주입됨
- 완료 조건 (사람)
  - [ ] `file://`로 열었을 때 콘솔 에러 없음, 두 영상 표시
  - [ ] 포지션 버튼 → 두 영상이 segments의 프레임으로 이동
  - [ ] 표 색과 목표 카드 값이 `findings.json`과 같음
  - [ ] [이 장면 보기], rep 탭, 키보드
  - [ ] 동기화 켜기/끄기 차이가 보임, 속도 변경
  - [ ] 스켈레톤 켜기/끄기, 목표 관절 색, 겹쳐 보기
  - [ ] `facing`이 다르면 ref 좌우 반전
  - [ ] 현재 프레임 카드, 템포 패널, 그래프
  - [ ] 포지션 수정 → 값 변화 → 되돌리기 → 저장 → `--override`로 다시 실행하면 브라우저에서 본 값과 같음

### 6단계 — ST-GCN 연결 (model 모드)
- 준비: 9장의 모델 연결 정보를 확인하고 `--ckpt`로 체크포인트를 명시한다
- 수정: `feedback/stgcn_infer.py`, `feedback/segment.py` (model 모드), `feedback/config.py` (상수), `run_demo.py`, `tests/fb/test_fb_segment_model.py`
- 추론: 기존 `model/realtime_stgcn_infer.py` 어댑터를 import하고 체크포인트는 `run_demo.py --ckpt <경로>`로 받는다. 입력 `(C=3, T=16, V=33)`, 학습 때와 같은 전처리. 16프레임 창을 1프레임씩 밀면서 창 마지막 프레임 (학습과 동일)에 phase (ready/down/up)와 종목 확률을 붙인다. 영상 시작의 부족한 과거 프레임은 첫 프레임을 반복한다. 미검출 프레임은 학습 코드처럼 x·y·visibility를 0으로 유지한다. 종목 = 포즈가 있는 프레임의 다수결
- phase 정리: 5프레임 중앙값 필터 → 5프레임 미만 구간은 앞뒤 구간에 흡수 → 허용 순서 ready → down → up → (ready 또는 down)
- torch가 현재 환경에 설치되지 않으면 멈추고 보고한다
- 완료 조건
  - 테스트: 합성 phase 시퀀스로 정리 규칙 검증
  - 데모 영상에서 종목 = squat, rep 수 = 수동 라벨 rep 수
  - 수동 (1단계) vs 모델 포지션 프레임 오차 표 출력. 목표: START·BOTTOM·LOCKOUT ±3프레임 이내. 못 맞추면 표와 함께 보고한다 (기준을 바꾸지 않는다)
  - model 모드로 `report.html`까지 생성

### 7단계 — 입력 확인·실패 화면
- 수정: `feedback/gate.py`, `web/failure_template.html`, `feedback/export.py` (경고·실패 데이터), `feedback/config.py` (상수·실패 문구), `run_demo.py`, `tests/fb/test_fb_gate.py`. 기존 합성 CLI 테스트 5개의 영상 검사 mock도 사용자 승인 범위다
- 검사 순서 (처음 실패한 곳에서 멈춤). 각 검사는 필요한 데이터가 나온 시점에 실행한다 (1은 포즈 추출 전, 4는 포즈 추출 후, 5의 ref 검사는 비교 직전, 사용자 검사는 비교 후). model 모드의 종목·각도 검사는 양쪽 추론 후 2 → 3 순서로 실행한다
  1. 영상 열림·코덱: `tools/check_video.py`의 `ffprobe`, `opencv_info`를 import (HEVC·10bit·가변 fps·회전). ffprobe가 없으면 경고만
  2. 종목 일치 (사용자 vs ref)
  3. 각도 일치 (`view` 설정값)
  4. 세트 검출률 ≥ 0.60
  5. 비교 가능한 사용자 rep ≥ 1, ref rep은 제외되지 않음
- 각도 자동 확인 (경고만): START에서 어깨 너비 (11–12번 x 거리) ÷ v4 몸통 길이를 계산하고 유효한 START 값의 중앙값을 영상 대표값으로 쓴다 — 0.35 미만은 측면, 0.60 초과는 정면, 사이는 대각. 설정과 다르면 세트 바에 경고. 유효 START가 없으면 확인할 수 없다는 경고를 표시한다
- 완료 조건: 테스트로 검사별 실패 이유가 맞게 나옴 · 사람 없는 영상 → 검출률 실패 화면 · 정면 영상을 `side`로 넣으면 경고

### 8단계 — 3D (Track B, 별도 환경)
- 8-0 설치 확인: conda 환경 `pose3d` (Python 3.10)에 RTMPose + MMPose 3D lifter (VideoPose3D 또는 MotionBERT, H36M 17관절). 짧은 영상으로 돌아가는지 먼저 확인하고 보고한다. 안 되면 멈춘다
- 8-1: `track3d/run_pose3d.py --video <영상> --who user --out out/demo/pose3d_user.js`
- 8-2: 리포트 3D 탭 (6.6)
- Track A와 같은 영상 파일 (고정 fps)을 쓴다. 가변 fps면 먼저 `check_video.py --fix`
- 완료 조건: `frames` 길이 = 영상 프레임 수 (테스트) · 뼈 길이 변동계수 출력 (참고용) · 사람 확인: 재생헤드와 동기, 로봇/스틱 전환, 시점 프리셋, 목표 관절 빨간 구, 파일이 없을 때 "3D 데이터 없음"

### 9단계 (선택) — 입력·처리 화면
업로드 · 각도 선택 · 촬영 가이드 · 구간 자르기 · 진행률 (2D 먼저, 3D 나중)을 로컬 웹으로 만든다. 새 패키지가 필요하니 시작 전에 합의한다.

---

## 8. 기존 코드 재사용

| 필요한 것 | 가져올 곳 |
|---|---|
| 랜드마크 추출·스무딩 | `offline_compare.extract_sequence` (v4 `create_landmarker`, `extract_landmarks`, `LandmarkSmoother`) |
| 지표 계산 | `unified_feedback_v4.compute_metrics` |
| 지표 목록·임계값·라벨·보조 지표 | v4 `DISPLAY_METRICS`, `DELTA_THRESHOLDS`, `METRIC_LABELS_KO`, `ADVISORY_METRICS` |
| 비율 지표 목록, 제외 기준 | `offline_compare.RATIO_METRICS`, `MIN_VALID_RATIO`, `MIN_REP_METRIC_RATIO` |
| 엉덩이 중점, 몸통 길이 | v4 `mid`, `torso_len` |
| side 고정 | v4 `SideLock`, `export_overlay.dominant_side` |
| 관절·뼈대 | v4 `SIDE_IDS`, `POSE_CONNECTIONS`, `export_overlay.METRIC_JOINTS`, `joints_for`, `bones_within` |
| 스켈레톤 좌표 | `export_overlay.dump_frames` |
| ref 스프라이트 | `export_overlay.build_sprite` |
| 영상 정보 | `export_overlay.video_info` |
| 영상 진단 | `tools/check_video.py`의 `ffprobe`, `opencv_info` |
| 플레이어 UI | `player_template.html` (영상·스프라이트·캔버스·`diagnose()`·키보드) |

`MODEL_PATH`는 v4 기준 `pose_landmarker_lite.task` (프로젝트 루트, 환경변수 `HAND_PROJECT_DIR`로 바꿀 수 있음).

---

## 9. ST-GCN 연결 정보

6단계에서 확인한 연결 정보다. 원본 모델·실시간 코드는 수정하지 않는다.
- [x] ST-GCN 모델 코드: `model/realtime_stgcn_infer.py`의 `RealtimeSTGCNInfer`, `MultiTaskSTGCNMLP`, `build_model_from_cfg`. 실시간 연결은 `unified_feedback_v4.py:_create_realtime_infer`, `realtime_compare_side_infer.py:_init_rep_counter`에서 확인했다.
- [x] 체크포인트: **임시** `data/data/STGCN_LSTM/ablation/mlp_h64_c16_ts2_do03_aug1/best_ep009.pt`. 사용자가 선택했고 `run_demo.py --segment model --ckpt <경로>`로 명시한다. 기존 실시간 기본 체크포인트와 같다고 간주하지 않는다. 메타데이터: MLP, hidden=64, clip_len=16, dropout=0.3, pose 입력 3채널.
- [x] 학습 전처리: `model/train_ablation.py:normalize_kpts`, `build_pose_input_features`, `CausalWindowDataset` (target=`phases[end]`). 추론에서는 같은 전처리를 제공하는 `model/realtime_stgcn_infer.py:normalize_kpts`, `build_pose_input_features`를 import한다. 입력은 x·y·visibility, 골반 중심과 어깨–골반 길이로 정규화한다. 미검출은 학습 추출과 같이 0으로 유지한다.
- [x] phase 클래스 인덱스: `0=ready`, `1=down`, `2=up` (`model/realtime_stgcn_infer.py:PHASE_NAMES`, 체크포인트 메타데이터).
- [x] 종목 클래스 인덱스: `0=squat`, `1=benchpress`, `2=deadlift` (`model/realtime_stgcn_infer.py:CLASS_LIST`, 체크포인트 메타데이터).
- [x] 기존 rep 카운팅: `model/realtime_stgcn_infer.py:OnlinePhaseCounter` (`update`에서 UP 종료 시 증가). 오프라인 분할은 5.1·7장 6단계의 phase 정리와 엉덩이 궤적 규칙을 적용한다.

---

## 10. 결정 기록

- **판정 대신 비교**: 단안 영상의 관절 각도 오차가 커서 (Physio2.2M 기준 무릎 MAE 2D 9.3–21.9°, 3D 14.1–25.8°) 절대 판정은 근거가 약하다. 같은 추정기로 같은 각도에서 찍은 두 영상을 비교하면 계통 오차가 일부 상쇄된다.
- **목표**: 사용자가 이미 하는 일 (내 영상과 레퍼런스를 눈으로 비교)을 자동화한다. 템포를 지우고, 어디가 얼마나 다른지 짚는다.
- **rep마다 ref 1 rep과 비교**, rep끼리는 비교하지 않는다.
- **Sportsbox AI 참고**: 포지션 기준 비교, 포지션 버튼, 포지션 수정 모드, 템포 패널, 범위 카드, 실패 화면, 2D = 원본 영상, 3D = 기본 로봇·스틱 전환.
- **임계값**: v4 임시값. 템포 임계는 임의값.
- **3D를 RTMPose로 따로**: ST-GCN은 MediaPipe 33관절로 학습돼 입력을 바꿀 수 없다. 앞으로 RTMPose를 쓸 예정이라 3D 뷰로 먼저 써 본다.
- **문장은 템플릿**: LLM은 나중에 템플릿 문장을 다듬는 용도로만 검토한다.

---

## 11. 단계 프롬프트 형식

기본 형식:

```
docs/design.md 7장 {N}단계 구현해.
수정 가능 파일과 완료 조건은 7장 {N}단계 그대로.
끝나면 AGENTS.md "단계 끝 보고" 형식으로 보고.
```
