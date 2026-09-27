# -*- coding: utf-8 -*-
"""
unified_feedback_v2.py

3대 운동(squat / deadlift / benchpress) 통합 자세 비교 + 시각적 피드백 코드 v2

반영 내용
- user / expert 영상 기반 비교. expert JSON이 없으면 expert 영상을 자동 전처리해서 JSON 캐시 생성
- squat / deadlift / benchpress 모두 같은 코드로 처리
- 출력 파일명: output/{종목명}_unified_feedback2.mp4
- landmark EMA smoothing 적용
- 측면영상 기준 visible side lock 적용
- generic 각도기 느낌 제거: 관절 중심 true arc 렌더러 적용
- expert skeleton bbox 정규화 후 중앙 배치
- deadlift / benchpress bar proxy confidence 적용
- 운동별 주요 레이어 분리

필요 패키지
    pip install "numpy<2" opencv-python mediapipe pillow

실행 예시
    python unified_feedback_v2.py --exercise squat
    python unified_feedback_v2.py --exercise deadlift
    python unified_feedback_v2.py --exercise benchpress
    python unified_feedback_v2.py --all

주의
- pose_landmarker_lite.task 파일이 현재 폴더에 있어야 합니다.
- 영상 파일명이 다르면 아래 VIDEO_CONFIG만 수정하면 됩니다.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from phase_bridge import (
    PhaseModelAdapter,
    Rep,
    align_realtime_phase_to_expert,
    build_phase_alignment_map,
    build_user_frame_meta,
    bridge_ready_gaps,
    count_phases_like_training,
    debounce_phase,
    num_reps_completed_until,
    phase_name as model_phase_name,
    pick_reference_rep,
    pose_seq_from_landmarks,
    rep_from_dict,
    rep_to_dict,
    segment_reps,
    segment_up_reps,
    smooth_phase_like_training,
)


# ============================================================
# 0. 사용자 설정
# ============================================================

# 프로젝트 폴더. 기본값은 이 스크립트가 있는 폴더로 고정한다.
# (VS Code에서 다른 작업 디렉토리로 실행해도 파일을 정확히 찾도록)
# 필요하면 환경변수 HAND_PROJECT_DIR 로 덮어쓰거나 아래 경로를 직접 수정한다.
#   예: BASE_DIR = Path(r"C:\dev\hand_project")
BASE_DIR = Path(os.environ.get("HAND_PROJECT_DIR", Path(__file__).resolve().parent))
MODEL_PATH = BASE_DIR / "pose_landmarker_lite.task"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 새 벤치프레스 원본 반영 완료
# squat / deadlift 파일명은 본인 로컬 파일명에 맞게 필요 시 여기만 수정하세요.
VIDEO_CONFIG = {
    "squat": {
        "user_video": "./infer/squat_user1.mp4",
        "expert_video": "./infer/squat_15.mp4",
        "output": "./infer/output/squat_unified_feedback_v4.mp4",
        "expert_ref_rep_index": None,
    },
    "deadlift": {
        "user_video": "./infer/user_deadlift_02.mp4",
        "expert_video": "./infer/expert_deadlift.mp4",
        "output": "./infer/output/deadlift_unified_feedback_v4.mp4",
        "expert_ref_rep_index": None,
    },
    "benchpress": {
        "user_video": "./infer/user_benchpress_v3.mp4",
        "expert_video": "./infer/expert_benchpress_v3.mp4",
        "output": "./infer/output/benchpress_unified_feedback_v4.mp4",
        "expert_ref_rep_index": None,
    },
}

# ============================================================
# 실행 대상 설정
# ============================================================
# CLI 인자 파싱 없이 여기만 수정해서 실행합니다.
#
# 예시:
#   RUN_EXERCISES = ["squat"]
#   RUN_EXERCISES = ["deadlift"]
#   RUN_EXERCISES = ["benchpress"]
#   RUN_EXERCISES = ["deadlift", "benchpress"]
#
# 주의: 여기에 적은 이름은 VIDEO_CONFIG에 살아있는 key와 같아야 합니다.
RUN_EXERCISES = ['squat',"deadlift","benchpress"]
RUN_ALL_EXERCISES = False

# Runtime mode:
#   "offline"  : render configured input videos to output MP4 files.
#   "realtime" : open webcam/video source and render live feedback.
RUN_MODE = "realtime"

# Realtime input.
#   0              -> default webcam
#   "video_config" -> VIDEO_CONFIG의 user_video 사용
#                     (REALTIME_INITIAL_EXERCISE가 auto면 RUN_EXERCISES 첫 종목)
#   "./infer/..."   -> 직접 지정한 로컬 영상 파일
REALTIME_SOURCE = "video_config"
REALTIME_INITIAL_EXERCISE = "auto"    # "auto", "squat", "deadlift", "benchpress"
REALTIME_LOCK_EXERCISE = True          # once action is confidently selected, keep it until reset
REALTIME_ACTION_CONF_MIN = 0.40
REALTIME_ACTION_SMOOTH_WINDOW = 7
REALTIME_ACTION_MIN_VOTES = 3
REALTIME_MODEL_INFERENCE_INTERVAL = 1
REALTIME_DISPLAY = True
REALTIME_OUTPUT = None                 # e.g. "./infer/output/realtime_unified_feedback_v4.mp4"
REALTIME_MAX_FRAMES = None             # set an int for non-interactive smoke runs
REALTIME_LOOP_VIDEO = False
REALTIME_CAMERA_WIDTH = 1280
REALTIME_CAMERA_HEIGHT = 720
REALTIME_WINDOW_NAME = "Unified Feedback Realtime"

# 화면 구성
EXPERT_W = 360
HUD_W = 360
MAX_OUTPUT_H = 720        # 너무 큰 원본이면 높이 기준으로 축소. 원본 크기 유지 원하면 None
MIN_PANEL_H = 620         # HUD 지표 행이 잘리지 않도록 하는 최소 패널 높이
MATCH_MODE = "ratio"     # "ratio" 권장. user/expert 길이가 달라도 진행률 기준 매칭

# ST-GCN phase/action bridge. 모델/phase/alignment 실패는 즉시 중단한다.
USE_MODEL_PHASE = True
EXERCISE_FROM_MODEL = True          # True면 모델 action head가 feedback exercise를 덮어씀(--exercise는 입력 영상 선택)
PHASE_ALIGN = "ratio"                # PR1-PR3 범위는 ratio만 지원
PHASE_DEBOUNCE_MIN_LEN = 4
PHASE_READY_BRIDGE_MAX_LEN = 30       # model READY가 active phase 사이를 짧게 끊을 때 최대 보정 프레임 수
REALTIME_READY_BRIDGE_MAX_LEN = 10    # realtime count 지연을 줄이기 위한 짧은 READY 보정 프레임 수
PHASE_COUNT_SMOOTH_WINDOW = 5         # phase 실험 count_phases와 같은 majority smoothing window
PHASE_COUNT_MIN_UP_LEN = 3            # phase 실험 count_phases와 같은 최소 UP 길이
MODEL_CKPT = BASE_DIR / "phase_experiments" / "quality_filter_best_arch_lstm" / "20260615T020450Z" / "high_acceleration_lstm" / "mediapipe_mediapipe33_jall_lstm_h128_c16_pw_2.0_ts2_do03_aug1_l1_input_acceleration" / "best_ep010.pt"
MODEL_DEVICE = "auto"
MODEL_CLIP_LEN = None
MODEL_STRIDE = 2
MODEL_INPUT_KIND = None              # None이면 checkpoint metadata를 그대로 사용
MODEL_GRAPH = "mediapipe33"
WINDOW_ANCHOR = "last"               # 학습 데이터셋은 causal window의 end frame을 phase target으로 사용

# MediaPipe / smoothing
VIS_THR = 0.45
OCCLUSION_THR = 0.50     # 이 값 미만이면 가려진 관절로 보고 해당 지표를 측정 불가 처리
LANDMARK_EMA_ALPHA = 0.35        # 낮을수록 부드러움, 높을수록 즉각 반응
EXPERT_EMA_ALPHA = 0.30
SIDE_SWITCH_MARGIN = 0.25
SIDE_SWITCH_HOLD = 6

# 색상(BGR)
C_BG = (18, 18, 24)
C_PANEL = (36, 36, 50)
C_LINE = (90, 220, 95)
C_LINE_DIM = (60, 130, 70)
C_OK = (90, 220, 120)
C_WARN = (80, 100, 255)
C_BAD = (40, 70, 255)
C_YELLOW = (0, 215, 255)
C_CYAN = (255, 210, 60)
C_WHITE = (235, 235, 235)
C_GRAY = (145, 145, 155)
C_DARKGRAY = (75, 75, 90)
C_ORANGE = (0, 155, 255)

# 한국어 폰트 후보
FONT_CANDIDATES = [
    "C:/Windows/Fonts/malgun.ttf",
    "C:/Windows/Fonts/malgunbd.ttf",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]

# 운동별 HUD 표시 지표
DISPLAY_METRICS = {
    "squat": ["knee_angle", "hip_angle", "trunk_lean", "foot_flatness", "head_hip_line"],
    "deadlift": ["knee_angle", "hip_angle", "trunk_lean", "elbow_angle", "head_hip_line", "bar_proxy_conf"],
    "benchpress": ["elbow_angle", "wrist_elbow_x_diff", "lockout_angle_min", "bench_line_diff", "bar_proxy_conf"],
}

METRIC_LABELS_KO = {
    "knee_angle": "무릎각",
    "hip_angle": "엉덩이각",
    "trunk_lean": "몸통기울기",
    "foot_flatness": "발바닥밀착",
    "head_hip_line": "머리-엉덩이",
    "neck_lean": "목-머리각",
    "elbow_angle": "팔꿈치각",
    "elbow_angle_avg": "팔꿈치각",
    "wrist_elbow_x_diff": "손목-팔꿈치",
    "lockout_angle_min": "락아웃각",
    "elbow_above_shoulder": "팔꿈치-어깨",
    "bench_line_diff": "벤치라인",
    "bar_proxy_conf": "바벨Proxy",
}

# 전문가와의 차이 허용치. angle은 degree, 비율 지표는 torso length 정규화값.
DELTA_THRESHOLDS = {
    "squat": {
        "knee_angle": 18.0,
        "hip_angle": 18.0,
        "trunk_lean": 10.0,
        "foot_flatness": 0.08,
        "head_hip_line": 0.18,
    },
    "deadlift": {
        "knee_angle": 18.0,
        "hip_angle": 18.0,
        "trunk_lean": 10.0,
        "elbow_angle": 20.0,
        "head_hip_line": 0.18,
        "bar_proxy_conf": 0.0,  # confidence는 delta 판단용이 아니라 HUD용
    },
    "benchpress": {
        "elbow_angle": 18.0,
        "wrist_elbow_x_diff": 0.08,
        "lockout_angle_min": 18.0,
        "bench_line_diff": 0.20,
        "bar_proxy_conf": 0.0,
    },
}

# 절대 기준. 전문가와 비교가 애매한 항목 보완용.
ABSOLUTE_RULES = {
    "squat": {
        "bottom_knee_too_open": 105.0,
        "trunk_lean_max": 35.0,
        "head_hip_line_max": 0.55,
    },
    "deadlift": {
        "top_knee_lockout_min": 155.0,
        "top_hip_lockout_min": 155.0,
        "elbow_lock_min": 155.0,
        "trunk_lean_max": 45.0,
    },
    "benchpress": {
        "lockout_min": 165.0,            # PDF: 상단에서 165도 이하면 락아웃 부족
        "bottom_elbow_min": 45.0,
        "wrist_elbow_x_diff_max": 0.10,  # PDF: 손목-팔꿈치 수직정렬 이탈 0.07~0.10 상한
    },
}


# ============================================================
# 경로 1: 종목별 구조적 가림(structural occlusion) 설정
# ------------------------------------------------------------
# MediaPipe의 visibility 점수는 바벨/원판에 의한 가림을 감지하지 못한다
# (가려진 관절도 높은 visibility로 추정해 버린다). 따라서 종목별로
# "구조적으로 신뢰하기 어려운 관절"을 명시한다.
#
# 처리 원칙:
# - STRUCTURAL_OCCLUSION에 속한 관절에 의존하는 지표는 ADVISORY로 강등한다.
# - ADVISORY 지표는 HUD에 값은 보여주되 '주의/오류'로 판정하지 않고(보조 지표),
#   파란 회색 톤으로 "참고"로 표시한다. PDF에서 데드 wrist_shoulder_y_diff를
#   보조 지표로만 쓰라고 한 것과 같은 취지.
# - bar proxy처럼 가려진 손목 추정에 의존하는 레이어는 해당 종목에서 끈다.
# ============================================================

# 손목(15,16)·손끝(17~22)·팔꿈치(13,14)는 데드/벤치에서 바벨·원판에 자주 가린다.
# 다만 측면영상에서는 카메라 쪽 팔이 보이는 경우가 많아, 일률적으로 막기보다
# 종목 특성에 맞춰 지정한다.
# - 데드리프트: 팔은 "바를 거는 고리"라 각도 자체가 판정 의미가 적고(PDF: 보조 지표),
#   손/손목이 바벨·원판과 겹쳐 신뢰도가 낮다 → 손/손목/팔꿈치를 구조적 가림으로 둔다.
# - 벤치프레스: 측면에서 카메라 쪽 팔(어깨-팔꿈치-손목)이 보이는 경우가 많으므로
#   구조적 가림으로 일괄 차단하지 않고, 프레임별 visibility gating에 맡긴다.
#   (가려진 쪽은 SideLock이 자동으로 반대쪽을 선택한다.)
STRUCTURAL_OCCLUSION = {
    "squat": set(),
    "deadlift": {13, 14, 15, 16, 17, 18, 19, 20, 21, 22},
    "benchpress": set(),
}

# 강등(보조)으로 표시할 지표: HUD에 값은 보이되 오류 판정에서 제외.
# STRUCTURAL_OCCLUSION 관절에 의존하는 지표를 여기에 둔다.
ADVISORY_METRICS = {
    "squat": set(),
    "deadlift": {"elbow_angle", "elbow_angle_avg", "wrist_elbow_x_diff", "bar_proxy_conf"},
    # bench_line_diff(머리-어깨-엉덩이 라인)는 측면 벤치에서 nose-어깨-엉덩이 기하가
    # 카메라 각도에 민감해 user/expert 비교가 불안정하다(expert가 user의 ~2배로 측정됨).
    # PDF 0.08 절대 기준을 강제하면 거의 항상 '주의'가 되어 노이즈가 되므로 보조 지표로 둔다.
    "benchpress": {"bar_proxy_conf", "bench_line_diff"},
}

# bar proxy 레이어를 그릴 종목(가려짐이 심하면 끈다).
# 벤치는 양손이 비교적 보여 grip center가 의미 있을 수 있으나,
# 데드는 손이 바벨/원판 뒤라 추정 신뢰도가 낮아 기본 비활성.
DRAW_BAR_PROXY = {
    "squat": False,
    "deadlift": False,
    "benchpress": True,
}

# expert 패널을 user 기준 방향으로 회전 정렬할지.
# - 스쿼트: user/expert 모두 서있어 정렬 불필요 → False.
# - 벤치: 누운 자세가 동작 내내 거의 고정(전신축 std 작음)이라 회전 정렬이 안정적 → True.
# - 데드: 동작 중 상체가 숙임→직립으로 전신축이 크게 변한다(std 큼).
#   회전을 한 번 고정하면 직립 구간에서 expert만 기울어 보이고, 매 프레임 돌리면
#   빙글빙글 돌아 더 어지럽다. 따라서 회전 정렬을 끄고 bbox 정규화에만 맡긴다.
ALIGN_EXPERT_ORIENTATION = {
    "squat": False,
    "deadlift": False,
    "benchpress": True,
}

POSE_CONNECTIONS = [
    (11, 12),
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
]

SIDE_IDS = {
    "left": {"shoulder": 11, "elbow": 13, "wrist": 15, "hip": 23, "knee": 25, "ankle": 27, "heel": 29, "foot": 31, "index": 19, "thumb": 21},
    "right": {"shoulder": 12, "elbow": 14, "wrist": 16, "hip": 24, "knee": 26, "ankle": 28, "heel": 30, "foot": 32, "index": 20, "thumb": 22},
}


@dataclass
class Issue:
    key: str
    message: str
    severity: float
    landmarks: List[int] = field(default_factory=list)


def find_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for fp in FONT_CANDIDATES:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size)
            except Exception:
                pass
    return ImageFont.load_default()


_FONT_CACHE: Dict[int, ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}


def get_font(size: int):
    if size not in _FONT_CACHE:
        _FONT_CACHE[size] = find_font(size)
    return _FONT_CACHE[size]


def bgr_to_rgb(c: Tuple[int, int, int]) -> Tuple[int, int, int]:
    return int(c[2]), int(c[1]), int(c[0])


def draw_text(
    img: np.ndarray,
    text: str,
    xy: Tuple[int, int],
    size: int = 22,
    color: Tuple[int, int, int] = C_WHITE,
    bold: bool = False,
) -> np.ndarray:
    """PIL 기반 한글 텍스트 렌더링."""
    if not text:
        return img
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    font = get_font(size + (2 if bold else 0))
    draw.text(xy, text, font=font, fill=bgr_to_rgb(color))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def _measure_text_px(text: str, size: int, bold: bool = False) -> Tuple[int, int]:
    """Return rendered text size in pixels for the same PIL font used by draw_text()."""
    font = get_font(size + (2 if bold else 0))
    canvas = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(canvas)
    bbox = draw.textbbox((0, 0), str(text), font=font)
    return max(0, bbox[2] - bbox[0]), max(0, bbox[3] - bbox[1])


def _line_height_px(size: int, bold: bool = False) -> int:
    return max(size + 5, _measure_text_px("가Ay", size, bold)[1] + 7)


def _ellipsis_to_width(text: str, max_width: int, size: int, bold: bool = False) -> str:
    """Trim one rendered line to max_width and append an ellipsis when needed."""
    text = str(text).rstrip()
    if _measure_text_px(text, size, bold)[0] <= max_width:
        return text
    ellipsis = "…"
    while text and _measure_text_px(text + ellipsis, size, bold)[0] > max_width:
        text = text[:-1].rstrip()
    return (text + ellipsis) if text else ellipsis


def _wrap_text_to_width(
    text: str,
    max_width: int,
    size: int,
    bold: bool = False,
    max_lines: int = 2,
) -> List[str]:
    """Pixel-width wrap that works for Korean feedback without relying on spaces."""
    lines, _truncated = _wrap_text_to_width_status(text, max_width, size, bold, max_lines)
    return lines


def _wrap_text_to_width_status(
    text: str,
    max_width: int,
    size: int,
    bold: bool = False,
    max_lines: int = 2,
) -> Tuple[List[str], bool]:
    """Wrap text and report whether any content was actually omitted."""
    max_lines = max(1, int(max_lines))
    text = str(text or "").strip()
    if not text:
        return [""], False

    lines: List[str] = []
    truncated = False
    paragraphs = text.splitlines() or [""]
    for paragraph_index, paragraph in enumerate(paragraphs):
        if paragraph_index > 0 and len(lines) >= max_lines:
            truncated = any(p.strip() for p in paragraphs[paragraph_index:])
            break
        current = ""
        ch_index = 0
        while ch_index < len(paragraph):
            ch = paragraph[ch_index]
            candidate = current + ch
            if current and _measure_text_px(candidate, size, bold)[0] > max_width:
                lines.append(current.rstrip())
                current = "" if ch.isspace() else ch.lstrip()
                if len(lines) >= max_lines:
                    truncated = True
                    break
            else:
                current = candidate
                ch_index += 1
        if truncated:
            break
        if current or not lines:
            lines.append(current.rstrip())
            if len(lines) >= max_lines and paragraph_index < len(paragraphs) - 1:
                truncated = any(p.strip() for p in paragraphs[paragraph_index + 1:])
                break

    if len(lines) > max_lines:
        lines = lines[:max_lines]
        truncated = True
    if truncated and lines:
        lines[-1] = _ellipsis_to_width(lines[-1], max_width, size, bold)
    return lines or [""], truncated


def _fit_wrapped_text(
    text: str,
    max_width: int,
    max_height: int,
    max_size: int,
    min_size: int,
    bold: bool = False,
    max_lines: int = 3,
) -> Tuple[int, List[str], int]:
    """Pick the largest font size that fits; prefer smaller complete text over ellipsis."""
    first_truncated_fit: Optional[Tuple[int, List[str], int]] = None
    for size in range(int(max_size), int(min_size) - 1, -1):
        line_h = _line_height_px(size, bold)
        allowed_lines = max(1, min(int(max_lines), max_height // line_h))
        lines, truncated = _wrap_text_to_width_status(text, max_width, size, bold, allowed_lines)
        if len(lines) * line_h <= max_height:
            if not truncated:
                return size, lines, line_h
            if first_truncated_fit is None:
                first_truncated_fit = (size, lines, line_h)
    size = int(min_size)
    line_h = _line_height_px(size, bold)
    allowed_lines = max(1, min(int(max_lines), max_height // line_h))
    lines, truncated = _wrap_text_to_width_status(text, max_width, size, bold, allowed_lines)
    if not truncated:
        return size, lines, line_h
    return first_truncated_fit or (size, lines, line_h)


def _draw_wrapped_text(
    img: np.ndarray,
    lines: Sequence[str],
    xy: Tuple[int, int],
    size: int,
    color: Tuple[int, int, int],
    bold: bool = False,
    line_h: Optional[int] = None,
) -> np.ndarray:
    x, y = xy
    line_h = line_h or _line_height_px(size, bold)
    for i, line in enumerate(lines):
        img = draw_text(img, line, (x, y + i * line_h), size, color, bold=bold)
    return img


def draw_round_rect(
    img: np.ndarray,
    p1: Tuple[int, int],
    p2: Tuple[int, int],
    color: Tuple[int, int, int],
    radius: int = 12,
    thickness: int = -1,
    alpha: Optional[float] = None,
) -> np.ndarray:
    """OpenCV rectangle helper. radius는 단순 rectangle로 처리."""
    if alpha is None:
        cv2.rectangle(img, p1, p2, color, thickness)
        return img
    overlay = img.copy()
    cv2.rectangle(overlay, p1, p2, color, thickness)
    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def safe_float(v, default: float = 0.0) -> float:
    try:
        v = float(v)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return default


def angle_abc(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom < 1e-9:
        return 0.0
    cosv = float(np.dot(ba, bc) / denom)
    cosv = clamp(cosv, -1.0, 1.0)
    return float(math.degrees(math.acos(cosv)))


def dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def normalize_exercise_name(name: str) -> str:
    n = name.lower().strip().replace("_", "").replace("-", "")
    if n in ["bench", "benchpress", "bp"]:
        return "benchpress"
    if n in ["squat", "sq"]:
        return "squat"
    if n in ["deadlift", "dl"]:
        return "deadlift"
    raise ValueError(f"알 수 없는 운동명: {name}")


# ============================================================
# 2. Landmark 처리
# ============================================================


def lms_to_np(lms) -> Optional[np.ndarray]:
    """MediaPipe landmark list -> (33,4) [x,y,z,visibility]."""
    if lms is None:
        return None
    arr = np.zeros((33, 4), dtype=np.float32)
    for i, p in enumerate(lms[:33]):
        arr[i, 0] = float(getattr(p, "x", 0.0))
        arr[i, 1] = float(getattr(p, "y", 0.0))
        arr[i, 2] = float(getattr(p, "z", 0.0))
        arr[i, 3] = float(getattr(p, "visibility", 1.0))
    return arr


def valid_lms(arr: Optional[np.ndarray]) -> bool:
    return arr is not None and isinstance(arr, np.ndarray) and arr.shape[0] >= 33


def get_xy(lms: np.ndarray, idx: int) -> np.ndarray:
    return np.array([float(lms[idx, 0]), float(lms[idx, 1])], dtype=np.float32)


def get_v(lms: np.ndarray, idx: int) -> float:
    return float(lms[idx, 3]) if lms is not None and idx < len(lms) else 0.0


def visible(lms: np.ndarray, idx: int, thr: float = VIS_THR) -> bool:
    return get_v(lms, idx) >= thr


def joint_ok(lms: np.ndarray, idx: int, thr: float = OCCLUSION_THR) -> bool:
    """occlusion gating: 이 관절이 가려지지 않고 신뢰 가능한지."""
    return get_v(lms, idx) >= thr


def all_ok(lms: np.ndarray, ids: Sequence[int], thr: float = OCCLUSION_THR) -> bool:
    """주어진 관절들이 모두 신뢰 가능하면 True. 하나라도 가려지면 False."""
    return all(joint_ok(lms, i, thr) for i in ids)


def mid(lms: np.ndarray, a: int, b: int) -> np.ndarray:
    pa, pb = get_xy(lms, a), get_xy(lms, b)
    va, vb = get_v(lms, a), get_v(lms, b)
    if va >= VIS_THR and vb >= VIS_THR:
        return (pa + pb) / 2.0
    return pa if va >= vb else pb


def vis_score(lms: np.ndarray, ids: Sequence[int]) -> float:
    if not valid_lms(lms):
        return 0.0
    return float(sum(max(0.0, min(1.0, get_v(lms, i))) for i in ids))


class LandmarkSmoother:
    """landmark 좌표 자체를 부드럽게 만드는 EMA smoother."""

    def __init__(self, alpha: float = 0.35, vis_thr: float = VIS_THR):
        self.alpha = float(alpha)
        self.vis_thr = float(vis_thr)
        self.prev: Optional[np.ndarray] = None

    def reset(self):
        self.prev = None

    def update(self, lms: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if not valid_lms(lms):
            return self.prev.copy() if self.prev is not None else None

        cur = lms.copy()
        if self.prev is None:
            self.prev = cur
            return cur

        out = self.prev.copy()
        for i in range(33):
            v = float(cur[i, 3])
            pv = float(self.prev[i, 3])
            if v >= self.vis_thr:
                out[i, :3] = self.alpha * cur[i, :3] + (1.0 - self.alpha) * self.prev[i, :3]
                out[i, 3] = max(v, 0.85 * pv)
            else:
                # 안 보이는 landmark는 갑자기 튀지 않게 이전 위치 유지, visibility만 감소
                out[i, :3] = self.prev[i, :3]
                out[i, 3] = 0.75 * pv
        self.prev = out
        return out


class SideLock:
    """측면영상에서 left/right 대표 side가 프레임마다 바뀌는 문제를 완화."""

    def __init__(self, exercise: str, margin: float = SIDE_SWITCH_MARGIN, hold: int = SIDE_SWITCH_HOLD):
        self.exercise = normalize_exercise_name(exercise)
        self.margin = margin
        self.hold = hold
        self.side: Optional[str] = None
        self.candidate: Optional[str] = None
        self.candidate_count = 0

    def _side_score(self, lms: np.ndarray, side: str) -> float:
        ids = SIDE_IDS[side]
        if self.exercise == "benchpress":
            key_ids = [ids["shoulder"], ids["elbow"], ids["wrist"], ids["index"], ids["thumb"]]
        else:
            key_ids = [ids["shoulder"], ids["hip"], ids["knee"], ids["ankle"], ids["foot"], ids["elbow"], ids["wrist"]]
        return vis_score(lms, key_ids)

    def update(self, lms: Optional[np.ndarray]) -> str:
        if not valid_lms(lms):
            return self.side or "right"

        ls = self._side_score(lms, "left")
        rs = self._side_score(lms, "right")
        best = "left" if ls >= rs else "right"

        if self.side is None:
            self.side = best
            return self.side

        cur_score = ls if self.side == "left" else rs
        alt_score = rs if self.side == "left" else ls
        alt_side = "right" if self.side == "left" else "left"

        if alt_score > cur_score + self.margin:
            if self.candidate == alt_side:
                self.candidate_count += 1
            else:
                self.candidate = alt_side
                self.candidate_count = 1
            if self.candidate_count >= self.hold:
                self.side = alt_side
                self.candidate = None
                self.candidate_count = 0
        else:
            self.candidate = None
            self.candidate_count = 0

        return self.side


# ============================================================
# 3. MediaPipe PoseLandmarker
# ============================================================


def create_landmarker(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"PoseLandmarker 모델 파일이 없습니다: {model_path}")
    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=2,  # 보조자(스팟터)가 함께 잡히는 경우를 위해 2명까지 검출 후 선택
        min_pose_detection_confidence=0.45,
        min_pose_presence_confidence=0.45,
        min_tracking_confidence=0.45,
        output_segmentation_masks=False,
    )
    return vision.PoseLandmarker.create_from_options(options)


def extract_all_landmarks(landmarker, frame_bgr: np.ndarray, timestamp_ms: int) -> List[np.ndarray]:
    """검출된 모든 사람의 랜드마크 리스트를 반환(0~여러 명)."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect_for_video(mp_image, int(timestamp_ms))
    if not result.pose_landmarks:
        return []
    return [lms_to_np(p) for p in result.pose_landmarks]


def _torso_horizontalness(lms: np.ndarray) -> float:
    """척추(엉덩이중심→어깨중심)가 수평에 가까울수록 1, 수직이면 0."""
    sh = mid(lms, 11, 12)
    hp = mid(lms, 23, 24)
    v = sh - hp
    n = float(np.hypot(v[0], v[1]))
    if n < 1e-6:
        return 0.0
    return abs(float(v[0])) / n  # |dx|/len: 수평이면 1


def _person_size(lms: np.ndarray) -> float:
    """화면상 사람 크기(주요 관절 bbox 대각선). 클수록 카메라에 가깝고 주피사체일 확률↑."""
    idxs = [0, 11, 12, 23, 24, 25, 26, 27, 28]
    pts = [get_xy(lms, i) for i in idxs if get_v(lms, i) >= 0.30]
    if len(pts) < 3:
        return 0.0
    p = np.stack(pts)
    d = p.max(axis=0) - p.min(axis=0)
    return float(np.hypot(d[0], d[1]))


def _person_center(lms: np.ndarray) -> Optional[np.ndarray]:
    idxs = [11, 12, 23, 24]
    pts = [get_xy(lms, i) for i in idxs if get_v(lms, i) >= 0.30]
    if len(pts) < 2:
        return None
    return np.stack(pts).mean(axis=0)


class PersonSelector:
    """여러 명이 검출될 때 운동 주체(리프터)를 일관되게 선택한다.

    - 벤치: 누운 사람(척추 수평)을 우선. 보조자는 보통 서 있다.
    - 데드/스쿼트: 가장 큰(카메라에 가까운) 사람을 우선.
    - 공통: 직전 프레임에서 고른 사람과 가까운 후보를 선호(깜빡임 방지).
    """

    def __init__(self, exercise: str):
        self.exercise = normalize_exercise_name(exercise)
        self._prev_center: Optional[np.ndarray] = None

    def select(self, candidates: List[np.ndarray]) -> Optional[np.ndarray]:
        if not candidates:
            return None
        if len(candidates) == 1:
            best = candidates[0]
            c = _person_center(best)
            if c is not None:
                self._prev_center = c
            return best

        scored = []
        for lms in candidates:
            score = 0.0
            if self.exercise == "benchpress":
                # 누운 자세 강하게 선호
                score += 3.0 * _torso_horizontalness(lms)
            # 크기(주피사체) 선호
            score += 1.0 * _person_size(lms)
            # 연속성: 직전 선택과 가까우면 가점
            c = _person_center(lms)
            if c is not None and self._prev_center is not None:
                dist_prev = float(np.hypot(*(c - self._prev_center)))
                score += 1.5 * max(0.0, 1.0 - dist_prev / 0.5)
            scored.append((score, lms, c))

        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best_lms, best_c = scored[0]
        if best_c is not None:
            self._prev_center = best_c
        return best_lms


def extract_landmarks(landmarker, frame_bgr: np.ndarray, timestamp_ms: int) -> Optional[np.ndarray]:
    """하위호환: 단일 사람 반환(첫 번째). 새 코드는 extract_all_landmarks + PersonSelector 사용."""
    allp = extract_all_landmarks(landmarker, frame_bgr, timestamp_ms)
    return allp[0] if allp else None


# ============================================================
# 4. 지표 계산
# ============================================================


def torso_len(lms: np.ndarray) -> float:
    s = mid(lms, 11, 12)
    h = mid(lms, 23, 24)
    return max(dist(s, h), 1e-5)


def signed_trunk_theta(lms: np.ndarray) -> float:
    """hip_center -> shoulder_center 벡터의 화면좌표계 angle rad."""
    sh = mid(lms, 11, 12)
    hp = mid(lms, 23, 24)
    v = sh - hp
    return float(math.atan2(float(v[1]), float(v[0])))


def angle_to_vertical_signed(top: np.ndarray, bottom: np.ndarray) -> float:
    """
    bottom -> top 벡터가 수직축에서 얼마나 기울었는지 signed degree.
    오른쪽으로 기울면 +, 왼쪽으로 기울면 -에 가깝게 사용.
    """
    dx = float(top[0] - bottom[0])
    dy = float(top[1] - bottom[1])
    # 화면에서 위쪽은 dy < 0. 수직 위쪽 벡터 대비 x 편차를 각도로 변환.
    return float(math.degrees(math.atan2(dx, -dy + 1e-9)))


def get_side_points(lms: np.ndarray, side: str) -> Dict[str, np.ndarray]:
    ids = SIDE_IDS[side]
    return {name: get_xy(lms, idx) for name, idx in ids.items() if name in ids}


def hand_grip_point(lms: np.ndarray, side: str) -> Tuple[Optional[np.ndarray], float]:
    """
    손목만 쓰지 않고 wrist/index/thumb를 visibility weighted 평균.
    confidence가 낮으면 None에 가깝게 처리하여 엉뚱한 bar proxy 방지.
    """
    ids = SIDE_IDS[side]
    candidates = [ids["wrist"], ids["index"], ids["thumb"]]
    pts = []
    weights = []
    for idx in candidates:
        v = get_v(lms, idx)
        if v >= OCCLUSION_THR:
            pts.append(get_xy(lms, idx))
            weights.append(v)
    if not pts:
        return None, 0.0
    w = np.array(weights, dtype=np.float32)
    p = np.sum(np.stack(pts, axis=0) * w[:, None], axis=0) / max(float(np.sum(w)), 1e-6)
    conf = float(np.mean(weights))
    return p.astype(np.float32), conf


def bar_proxy(lms: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
    lp, lc = hand_grip_point(lms, "left")
    rp, rc = hand_grip_point(lms, "right")
    pts = []
    ws = []
    if lp is not None and lc >= VIS_THR:
        pts.append(lp); ws.append(lc)
    if rp is not None and rc >= VIS_THR:
        pts.append(rp); ws.append(rc)
    if not pts:
        return None, 0.0
    w = np.array(ws, dtype=np.float32)
    p = np.sum(np.stack(pts, axis=0) * w[:, None], axis=0) / max(float(np.sum(w)), 1e-6)
    return p.astype(np.float32), float(np.mean(ws))


def compute_metrics(lms: Optional[np.ndarray], exercise: str, side: str) -> Dict[str, float]:
    exercise = normalize_exercise_name(exercise)
    if not valid_lms(lms):
        return {}

    ids = SIDE_IDS[side]
    sh = get_xy(lms, ids["shoulder"])
    el = get_xy(lms, ids["elbow"])
    wr = get_xy(lms, ids["wrist"])
    hp = get_xy(lms, ids["hip"])
    kn = get_xy(lms, ids["knee"])
    an = get_xy(lms, ids["ankle"])
    heel = get_xy(lms, ids["heel"])
    foot = get_xy(lms, ids["foot"])

    shoulder_c = mid(lms, 11, 12)
    hip_c = mid(lms, 23, 24)
    nose = get_xy(lms, 0)
    tlen = torso_len(lms)

    # ── occlusion gating ──
    # 각 지표는 의존하는 관절이 모두 신뢰 가능할 때만 값을 넣는다.
    # 가려지면 None으로 두어 HUD에 "측정 불가"로 표시되고, 비정상값으로 오판하지 않는다.
    sid = ids["shoulder"]; eid = ids["elbow"]; wid = ids["wrist"]
    hid = ids["hip"]; kid = ids["knee"]; aid = ids["ankle"]
    heid = ids["heel"]; fid = ids["foot"]
    torso_core = [11, 12, 23, 24]  # torso_len / mid 계산 신뢰성

    out: Dict[str, float] = {}
    struct_occ = STRUCTURAL_OCCLUSION.get(exercise, set())
    advisory = ADVISORY_METRICS.get(exercise, set())

    def put(key: str, value: float, req_ids: Sequence[int]):
        occluded = any(i in struct_occ for i in req_ids)
        if occluded and key not in advisory:
            # 구조적 가림 + 보조 지표도 아님 → 신뢰 불가, 아예 제외.
            return
        if occluded and key in advisory:
            # 보조 지표는 값은 표시하되(choose_issue에서 판정 제외), 그대로 넣는다.
            out[key] = float(value)
            return
        if all_ok(lms, list(req_ids)):
            out[key] = float(value)
        # 아니면 키 자체를 넣지 않음 → user_m.get(key) == None

    knee_angle = angle_abc(hp, kn, an)
    hip_angle = angle_abc(sh, hp, kn)
    elbow_angle = angle_abc(sh, el, wr)
    trunk_lean = abs(angle_to_vertical_signed(shoulder_c, hip_c))
    neck_lean = abs(angle_to_vertical_signed(nose, shoulder_c))
    foot_flatness = abs(float(heel[1] - foot[1])) / tlen
    head_hip_line = abs(float(nose[0] - hip_c[0])) / tlen
    wrist_elbow_x_diff = abs(float(wr[0] - el[0])) / tlen

    v = shoulder_c - hip_c
    n = np.linalg.norm(v)
    bench_line_diff = 0.0 if n < 1e-6 else abs(float(np.cross(v, nose - hip_c))) / (float(n) * tlen)

    put("knee_angle", knee_angle, [hid, kid, aid])
    put("hip_angle", hip_angle, [sid, hid, kid])
    put("elbow_angle", elbow_angle, [sid, eid, wid])
    put("elbow_angle_avg", elbow_angle, [sid, eid, wid])
    put("lockout_angle_min", elbow_angle, [sid, eid, wid])
    put("trunk_lean", trunk_lean, torso_core)
    put("trunk_lean_signed", angle_to_vertical_signed(shoulder_c, hip_c), torso_core)
    put("trunk_theta", signed_trunk_theta(lms), torso_core)
    put("neck_lean", neck_lean, [0, 11, 12])
    put("foot_flatness", foot_flatness, [heid, fid] + torso_core)
    put("head_hip_line", head_hip_line, [0] + torso_core)
    put("wrist_elbow_x_diff", wrist_elbow_x_diff, [eid, wid] + torso_core)
    put("bench_line_diff", bench_line_diff, [0] + torso_core)
    if all_ok(lms, [sid, eid]):
        out["elbow_above_shoulder"] = 1.0 if float(el[1]) < float(sh[1]) else 0.0

    bp, bconf = bar_proxy(lms)
    out["bar_proxy_conf"] = float(bconf)

    return out


def compute_deltas(user_m: Dict[str, float], expert_m: Dict[str, float]) -> Dict[str, float]:
    out = {}
    for k, uv in user_m.items():
        if k in expert_m:
            out[k] = safe_float(uv) - safe_float(expert_m[k])
    return out


# ============================================================
# 5. 피드백 판단 / 안정화 / 카운터
# ============================================================


class FeedbackStabilizer:
    def __init__(self, min_frames: int = 3, hold_frames: int = 9):
        self.min_frames = min_frames
        self.hold_frames = hold_frames
        self.candidate_key: Optional[str] = None
        self.candidate_count = 0
        self.current: Optional[Issue] = None
        self.hold_left = 0

    def update(self, issue: Optional[Issue]) -> Optional[Issue]:
        if issue is None:
            if self.hold_left > 0 and self.current is not None:
                self.hold_left -= 1
                return self.current
            self.current = None
            self.candidate_key = None
            self.candidate_count = 0
            return None

        if self.current is not None and issue.key == self.current.key:
            self.current = issue
            self.hold_left = self.hold_frames
            return self.current

        if self.candidate_key == issue.key:
            self.candidate_count += 1
        else:
            self.candidate_key = issue.key
            self.candidate_count = 1

        if self.candidate_count >= self.min_frames:
            self.current = issue
            self.hold_left = self.hold_frames
            return self.current

        if self.current is not None and self.hold_left > 0:
            self.hold_left -= 1
            return self.current
        return None


def issue_from_metric(exercise: str, key: str, delta: float, severity: float) -> Optional[Issue]:
    exercise = normalize_exercise_name(exercise)
    if exercise == "squat":
        if key == "knee_angle":
            msg = "무릎이 전문가보다 많이 굽혀짐" if delta < 0 else "무릎을 더 깊게 굽혀야 함"
            return Issue(key, msg, severity, [23, 24, 25, 26, 27, 28])
        if key == "hip_angle":
            msg = "엉덩이각/힙힌지 확인"
            return Issue(key, msg, severity, [11, 12, 23, 24, 25, 26])
        if key == "trunk_lean":
            msg = "몸통 기울기 확인"
            return Issue(key, msg, severity, [11, 12, 23, 24])
        if key == "foot_flatness":
            msg = "발바닥 밀착 확인"
            return Issue(key, msg, severity, [27, 28, 29, 30, 31, 32])
        if key == "head_hip_line":
            msg = "머리-엉덩이 라인 확인"
            return Issue(key, msg, severity, [0, 23, 24])
    elif exercise == "deadlift":
        if key == "hip_angle":
            msg = "엉덩이각/힙힌지 확인"
            return Issue(key, msg, severity, [11, 12, 23, 24, 25, 26])
        if key == "knee_angle":
            msg = "무릎각 확인"
            return Issue(key, msg, severity, [23, 24, 25, 26, 27, 28])
        if key == "trunk_lean":
            msg = "몸통 기울기 확인"
            return Issue(key, msg, severity, [11, 12, 23, 24])
        if key == "elbow_angle":
            msg = "팔꿈치 각도 확인"
            return Issue(key, msg, severity, [11, 12, 13, 14, 15, 16])
        if key == "head_hip_line":
            msg = "머리-엉덩이 라인 확인"
            return Issue(key, msg, severity, [0, 23, 24])
    else:
        if key in ["elbow_angle", "lockout_angle_min"]:
            msg = "팔꿈치 각도 확인"
            return Issue(key, msg, severity, [11, 12, 13, 14, 15, 16])
        if key == "wrist_elbow_x_diff":
            msg = "손목-팔꿈치 정렬 확인"
            return Issue(key, msg, severity, [13, 14, 15, 16])
        if key == "bench_line_diff":
            msg = "벤치라인/상체 고정 확인"
            return Issue(key, msg, severity, [0, 11, 12, 23, 24])
    return None


def choose_issue(
    exercise: str,
    user_m: Dict[str, float],
    expert_m: Dict[str, float],
    deltas: Dict[str, float],
    phase: str,
    rep_count: Optional[int] = None,
) -> Optional[Issue]:
    exercise = normalize_exercise_name(exercise)
    phase_text = str(phase or "").strip().lower()
    is_bottom_phase = phase_text in {"최저점", "bottom", "down"}
    is_top_phase = phase_text in {"락아웃/상단", "top", "lockout", "up"}
    issues: List[Issue] = []
    thr = DELTA_THRESHOLDS.get(exercise, {})
    advisory = ADVISORY_METRICS.get(exercise, set())

    def um(key, default=None):
        """user metric 안전 조회. None(측정 불가)이면 default."""
        v = user_m.get(key)
        return default if v is None else v

    # 전문가 비교 기반 issue (보조 지표는 제외)
    for key, d in deltas.items():
        if key in advisory:
            continue
        if key not in thr or thr[key] <= 0:
            continue
        if d is None:
            continue
        sev = abs(float(d)) / float(thr[key])
        if sev >= 1.0:
            issue = issue_from_metric(exercise, key, float(d), sev)
            if issue:
                issues.append(issue)

    # 절대 기준 보완 (측정 불가 지표는 건너뜀)
    if exercise == "squat":
        kn = um("knee_angle")
        if is_bottom_phase and kn is not None and kn > ABSOLUTE_RULES["squat"]["bottom_knee_too_open"]:
            issues.append(Issue("knee_depth", "최저점 깊이 확인", 1.3, [23, 24, 25, 26, 27, 28]))
        tl = um("trunk_lean")
        if tl is not None and tl > ABSOLUTE_RULES["squat"]["trunk_lean_max"]:
            issues.append(Issue("trunk_lean_abs", "상체가 과도하게 숙여짐", 1.2, [11, 12, 23, 24]))
    elif exercise == "deadlift":
        if is_top_phase:
            kn = um("knee_angle")
            if kn is not None and kn < ABSOLUTE_RULES["deadlift"]["top_knee_lockout_min"]:
                issues.append(Issue("deadlift_knee_lockout", "무릎 락아웃 확인", 1.2, [23, 24, 25, 26, 27, 28]))
            hpa = um("hip_angle")
            if hpa is not None and hpa < ABSOLUTE_RULES["deadlift"]["top_hip_lockout_min"]:
                issues.append(Issue("deadlift_hip_lockout", "엉덩이 락아웃 확인", 1.2, [11, 12, 23, 24, 25, 26]))
        # 데드 팔꿈치는 구조적 가림 대상 → 절대 기준 판정에서 제외(보조 지표).
    else:  # benchpress
        ea = um("elbow_angle")
        if is_top_phase and ea is not None and ea < ABSOLUTE_RULES["benchpress"]["lockout_min"]:
            issues.append(Issue("bench_lockout", "락아웃 각도 확인", 1.25, [11, 12, 13, 14, 15, 16]))
        we = um("wrist_elbow_x_diff")
        if we is not None and we > ABSOLUTE_RULES["benchpress"]["wrist_elbow_x_diff_max"]:
            issues.append(Issue("bench_wrist_elbow", "손목-팔꿈치 정렬 확인", 1.2, [13, 14, 15, 16]))

    if not issues:
        return None
    issues.sort(key=lambda x: x.severity, reverse=True)
    return issues[0]


# ============================================================
# 6. 좌표 변환 / 렌더링 함수
# ============================================================


def make_user_transform(w: int, h: int):
    def tr(p: np.ndarray) -> Tuple[int, int]:
        return int(round(float(p[0]) * w)), int(round(float(p[1]) * h))
    return tr


def _bbox_from_lms(lms: Optional[np.ndarray]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """visibility가 충분한 점들의 (min, max) bbox. 없으면 None."""
    if not valid_lms(lms):
        return None
    pts = []
    for i in range(33):
        if get_v(lms, i) >= max(0.25, VIS_THR * 0.6):
            pts.append(get_xy(lms, i))
    if len(pts) < 5:
        return None
    pts_np = np.stack(pts, axis=0)
    return pts_np.min(axis=0), pts_np.max(axis=0)


def _body_axis_angle(lms: Optional[np.ndarray]) -> Optional[float]:
    """전신 주축 방향 각도(rad). 화면 위쪽 기준 atan2(dx, -dy).

    부호 모호성이 있는 PCA 대신, 방향이 명확한 '발목중심 → 머리(코)' 벡터를 쓴다.
    이렇게 하면 회전 정렬 시 머리 방향까지 user와 일치한다.
    누운 자세면 ±90도 부근, 서있으면 0도 부근.
    """
    if not valid_lms(lms):
        return None
    # 머리 끝점: nose(0). 발 끝점: 양 발목(27,28) 중심(없으면 무릎).
    if get_v(lms, 0) < 0.30:
        return None
    head = get_xy(lms, 0)
    foot_ids = [27, 28]
    fpts = [get_xy(lms, i) for i in foot_ids if get_v(lms, i) >= 0.30]
    if len(fpts) < 1:
        foot_ids = [25, 26]
        fpts = [get_xy(lms, i) for i in foot_ids if get_v(lms, i) >= 0.30]
    if len(fpts) < 1:
        return None
    foot = np.stack(fpts).mean(axis=0)
    v = head - foot  # 발 -> 머리 방향
    if float(np.hypot(v[0], v[1])) < 1e-5:
        return None
    return float(math.atan2(float(v[0]), -float(v[1])))


def _spine_axis_angle(lms: Optional[np.ndarray]) -> Optional[float]:
    """전신 주축 각도. (이름은 호환 유지, 내부는 PCA 주축 사용)"""
    return _body_axis_angle(lms)


def _rotate_pts(pts: np.ndarray, pivot: np.ndarray, ang_rad: float) -> np.ndarray:
    c, s = math.cos(ang_rad), math.sin(ang_rad)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    return (pts - pivot) @ R.T + pivot


def _prescan_user_axis(user_video_path: str, model_path: Path, max_frames: int = 40) -> Optional[float]:
    """user 영상 앞쪽 일부를 빠르게 훑어 전신축(rad) 중앙값을 구한다.
    회전 정렬의 기준 방향으로 쓴다."""
    cap, rot = open_video_normalized(str(user_video_path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    sm = LandmarkSmoother(LANDMARK_EMA_ALPHA, VIS_THR)
    selector = PersonSelector("benchpress")  # prescan은 벤치에서만 쓰이므로 누운사람 우선
    angs: List[float] = []
    try:
        with create_landmarker(model_path) as lm:
            i = 0
            while i < max_frames:
                ok, fr = cap.read()
                if not ok:
                    break
                fr = apply_rotation(fr, rot)
                fr = resize_frame_if_needed(fr)
                raw = selector.select(extract_all_landmarks(lm, fr, int(i * 1000 / fps)))
                s = sm.update(raw)
                if valid_lms(s):
                    a = _body_axis_angle(s)
                    if a is not None:
                        angs.append(a)
                i += 1
    finally:
        cap.release()
    if not angs:
        return None
    return float(np.median(angs))


class ExpertPaneNormalizer:
    """
    expert skeleton을 expert pane에 고정 배치한다.

    스쿼트 노트북의 anchor-lock 아이디어를 별도 패널 버전으로 이식한 것.
    - 매 프레임 bbox를 새로 계산하면 expert가 출렁이고 크기가 변한다.
    - 대신 첫 lock_frames개의 유효 프레임에서 동작 전체를 감싸는 bbox를
      누적(union)한 뒤 스케일/중심을 한 번 고정한다.
    - 고정 후에는 관절만 그 안에서 움직이므로 안정적으로 보인다.

    경로 2(방향 정규화):
    - expert와 user의 "기준 척추축 방향"이 다르면(예: user는 누워있는데
      expert는 서있는 영상) 두 골격이 제각각 방향으로 보인다.
    - lock 구간 동안 expert의 기준 척추각 중앙값을 구해, user의 기준 척추각에
      맞추는 회전량(align_rot)을 한 번 고정한다.
    - 이후 expert 좌표를 그 회전량만큼 돌려서 그린다. 동작 중 상대적 변화
      (데드의 숙임→직립 등)는 그대로 보존된다.
    """

    def __init__(self, w: int, h: int, pad: int = 42, lock_frames: int = 25,
                 align_orientation: bool = False):
        self.w = w
        self.h = h
        self.pad = pad
        self.lock_frames = lock_frames
        self.align_orientation = align_orientation
        self._seen = 0
        self._acc_mn: Optional[np.ndarray] = None
        self._acc_mx: Optional[np.ndarray] = None
        self._locked = False
        self._scale = 1.0
        self._center = np.array([0.5, 0.5], dtype=np.float32)
        # 회전 정렬
        self._user_angles: List[float] = []
        self._expert_angles: List[float] = []
        self._align_rot = 0.0          # expert에 적용할 회전량(rad)
        self._pivot = np.array([0.5, 0.5], dtype=np.float32)
        self._rot_finalized = not align_orientation  # 회전 안 쓰면 처음부터 확정 상태

    def observe_user(self, user_lms: Optional[np.ndarray]):
        """user의 기준 전신축을 warmup 동안 수집(회전 정렬용)."""
        if not self.align_orientation or self._rot_finalized:
            return
        a = _body_axis_angle(user_lms)
        if a is not None:
            self._user_angles.append(a)

    def _fit(self, mn: np.ndarray, mx: np.ndarray):
        center = (mn + mx) / 2.0
        size = mx - mn
        sx = (self.w - 2 * self.pad) / max(float(size[0]), 1e-4)
        sy = (self.h - 2 * self.pad) / max(float(size[1]), 1e-4)
        scale = min(sx, sy)
        scale = min(scale, self.h * 1.9)
        self._scale = scale
        self._center = center.astype(np.float32)

    def _rotated(self, lms: np.ndarray) -> np.ndarray:
        """회전 정렬이 확정됐으면 회전 적용한 좌표를, 아니면 원좌표를 반환."""
        if not (self.align_orientation and self._rot_finalized and abs(self._align_rot) > 1e-4):
            return lms[:, :2]
        return _rotate_pts(lms[:, :2].copy(), self._pivot, self._align_rot)

    def update(self, lms: Optional[np.ndarray]):
        """현재 expert lms를 반영하고, 화면좌표 변환 함수를 반환한다.

        2단계 처리:
        - warmup(회전 정렬 ON일 때): lock_frames 동안 expert/user 전신축만 모은다.
          이 구간에는 bbox를 쌓지 않는다(회전이 아직 안 정해져 좌표 기준이 흔들리므로).
        - warmup 끝나는 순간 회전량/피벗을 한 번 확정(_rot_finalized=True)한다.
        - 이후 회전 적용된 좌표로 bbox를 누적하고 _fit으로 스케일/중심을 정한다.
        - 회전 정렬 OFF면 warmup 없이 바로 bbox를 누적한다(기존 동작).
        """
        # 1) warmup: 회전용 각 수집
        if self.align_orientation and not self._rot_finalized:
            ea = _body_axis_angle(lms)
            if ea is not None:
                self._expert_angles.append(ea)
            self._seen += 1
            # warmup 종료 → 회전량/피벗 확정
            if self._seen >= self.lock_frames and self._expert_angles and self._user_angles:
                ua = float(np.median(self._user_angles))
                ea_med = float(np.median(self._expert_angles))
                self._align_rot = ua - ea_med
                if valid_lms(lms):
                    cidx = [0, 11, 12, 23, 24, 25, 26, 27, 28]
                    cpts = [get_xy(lms, i) for i in cidx if get_v(lms, i) >= 0.30]
                    if cpts:
                        self._pivot = np.stack(cpts).mean(axis=0).astype(np.float32)
                self._rot_finalized = True
                self._seen = 0  # bbox 누적 단계 카운터 재사용
            # warmup 중에는 원좌표로 임시 변환 반환(화면엔 대략 위치만)
            return self._make_tr()

        # 2) bbox 누적 단계 (회전 OFF면 처음부터 여기로 들어옴)
        if valid_lms(lms) and not self._locked:
            pts2d = self._rotated(lms)
            # visibility 필터로 bbox 계산
            vis = lms[:, 3]
            mask = vis >= max(0.25, VIS_THR * 0.6)
            if int(mask.sum()) >= 5:
                sel = pts2d[mask]
                mn = sel.min(axis=0); mx = sel.max(axis=0)
                if self._acc_mn is None:
                    self._acc_mn, self._acc_mx = mn.copy(), mx.copy()
                else:
                    self._acc_mn = np.minimum(self._acc_mn, mn)
                    self._acc_mx = np.maximum(self._acc_mx, mx)
                self._seen += 1
                self._fit(self._acc_mn, self._acc_mx)
                if self._seen >= self.lock_frames:
                    self._locked = True

        return self._make_tr()

    def prefit(self, expert_lms_list: List[Optional[np.ndarray]], user_ref_angle: Optional[float]):
        """expert 전체 프레임 + user 기준 전신축으로 회전량/피벗/bbox를 한 번에 확정한다.

        실시간 warmup의 한계(첫 lock_frames 동작 범위만 보는 문제)를 없앤다.
        expert 프로파일(JSON)이 전체 프레임을 갖고 있을 때 사용한다.
        """
        valids = [l for l in expert_lms_list if valid_lms(l)]
        if not valids:
            return

        # 1) 회전량: expert 전신축 중앙값 → user 기준으로
        if self.align_orientation and user_ref_angle is not None:
            ex_angs = [_body_axis_angle(l) for l in valids]
            ex_angs = [a for a in ex_angs if a is not None]
            if ex_angs:
                ea_med = float(np.median(ex_angs))
                self._align_rot = float(user_ref_angle) - ea_med
                # 피벗: 첫 유효 프레임의 전신 중심
                cidx = [0, 11, 12, 23, 24, 25, 26, 27, 28]
                l0 = valids[0]
                cpts = [get_xy(l0, i) for i in cidx if get_v(l0, i) >= 0.30]
                if cpts:
                    self._pivot = np.stack(cpts).mean(axis=0).astype(np.float32)
        self._rot_finalized = True

        # 2) 회전 적용한 좌표로 전체 bbox 계산
        mn = None; mx = None
        for l in valids:
            pts2d = self._rotated(l)
            vis = l[:, 3]
            mask = vis >= max(0.25, VIS_THR * 0.6)
            if int(mask.sum()) < 5:
                continue
            sel = pts2d[mask]
            cmn = sel.min(axis=0); cmx = sel.max(axis=0)
            mn = cmn if mn is None else np.minimum(mn, cmn)
            mx = cmx if mx is None else np.maximum(mx, cmx)
        if mn is not None:
            self._acc_mn, self._acc_mx = mn, mx
            self._fit(mn, mx)
            self._locked = True

    def _make_tr(self):
        scale = self._scale
        center = self._center
        align_rot = self._align_rot
        pivot_fixed = self._pivot
        dst_center = np.array([self.w / 2.0, self.h / 2.0], dtype=np.float32)
        do_rot = self.align_orientation and self._rot_finalized and abs(align_rot) > 1e-4

        if self._acc_mn is None:
            def tr_raw(p: np.ndarray) -> Tuple[int, int]:
                pp = p
                if do_rot:
                    pp = _rotate_pts(np.asarray(p, dtype=np.float32).reshape(1, 2), pivot_fixed, align_rot)[0]
                return int(float(pp[0]) * self.w), int(float(pp[1]) * self.h)
            return tr_raw

        def tr(p: np.ndarray) -> Tuple[int, int]:
            pp = p
            if do_rot:
                pp = _rotate_pts(np.asarray(p, dtype=np.float32).reshape(1, 2), pivot_fixed, align_rot)[0]
            q = (pp - center) * scale + dst_center
            return int(round(float(q[0]))), int(round(float(q[1])))

        return tr


def draw_skeleton(
    img: np.ndarray,
    lms: Optional[np.ndarray],
    transform,
    bad_indices: Optional[Sequence[int]] = None,
    line_color: Tuple[int, int, int] = C_LINE,
    point_color: Tuple[int, int, int] = (160, 235, 170),
    bad_color: Tuple[int, int, int] = C_BAD,
    thickness: int = 3,
    occ_thr: float = OCCLUSION_THR,
):
    if not valid_lms(lms):
        return img
    bad = set(bad_indices or [])
    for a, b in POSE_CONNECTIONS:
        va, vb = get_v(lms, a), get_v(lms, b)
        # 둘 중 하나라도 occlusion 임계값 미만이면 신뢰 불가 → 골격선 안 그림(귀신 방지)
        if va < occ_thr or vb < occ_thr:
            continue
        pa = transform(get_xy(lms, a))
        pb = transform(get_xy(lms, b))
        color = bad_color if (a in bad or b in bad) else line_color
        cv2.line(img, pa, pb, color, thickness, cv2.LINE_AA)

    for i in range(33):
        v = get_v(lms, i)
        if v < 0.30:
            continue
        p = transform(get_xy(lms, i))
        if v < occ_thr:
            # 가려진 관절: 작은 회색 빈 원으로만 표시(불확실 신호)
            cv2.circle(img, p, 3, C_DARKGRAY, 1, cv2.LINE_AA)
            continue
        if i in bad:
            cv2.circle(img, p, 10, bad_color, 3, cv2.LINE_AA)
            cv2.circle(img, p, 3, bad_color, -1, cv2.LINE_AA)
        else:
            cv2.circle(img, p, 4, point_color, -1, cv2.LINE_AA)
    return img


def shortest_arc_points(center: Tuple[int, int], p1: Tuple[int, int], p2: Tuple[int, int], radius: float, n: int = 48) -> np.ndarray:
    cx, cy = center
    a1 = math.atan2(p1[1] - cy, p1[0] - cx)
    a2 = math.atan2(p2[1] - cy, p2[0] - cx)
    da = (a2 - a1 + math.pi) % (2 * math.pi) - math.pi
    angles = np.linspace(a1, a1 + da, n)
    pts = np.stack([cx + radius * np.cos(angles), cy + radius * np.sin(angles)], axis=1)
    return pts.astype(np.int32)


def draw_angle_arc(
    img: np.ndarray,
    lms: Optional[np.ndarray],
    ids: Tuple[int, int, int],
    transform,
    label: Optional[str] = None,
    color: Tuple[int, int, int] = C_YELLOW,
    warn: bool = False,
    radius_scale: float = 0.36,
    min_radius: int = 28,
    max_radius: int = 82,
    thickness: int = 5,
):
    """A-B-C 각도를 B 중심 true arc로 표시."""
    if not valid_lms(lms):
        return img
    a, b, c = ids
    if min(get_v(lms, a), get_v(lms, b), get_v(lms, c)) < 0.25:
        return img

    pa = transform(get_xy(lms, a))
    pb = transform(get_xy(lms, b))
    pc = transform(get_xy(lms, c))
    angle = angle_abc(get_xy(lms, a), get_xy(lms, b), get_xy(lms, c))
    la = math.dist(pb, pa)
    lc = math.dist(pb, pc)
    radius = int(clamp(min(la, lc) * radius_scale, min_radius, max_radius))
    use_color = C_BAD if warn else color

    arc = shortest_arc_points(pb, pa, pc, radius, n=60)
    cv2.polylines(img, [arc], False, use_color, thickness, cv2.LINE_AA)

    # arc 양 끝 작은 원
    if len(arc) > 2:
        cv2.circle(img, tuple(arc[0]), 4, use_color, -1, cv2.LINE_AA)
        cv2.circle(img, tuple(arc[-1]), 4, use_color, -1, cv2.LINE_AA)

    # 중심 관절 강조
    cv2.circle(img, pb, 12, use_color, 3, cv2.LINE_AA)

    # 텍스트 위치: arc 중간점에서 바깥쪽
    mid_idx = len(arc) // 2
    tx, ty = arc[mid_idx]
    vx = tx - pb[0]
    vy = ty - pb[1]
    norm = math.sqrt(vx * vx + vy * vy) + 1e-6
    tx = int(tx + 18 * vx / norm)
    ty = int(ty + 18 * vy / norm)
    txt = label if label is not None else f"{int(round(angle))}deg"
    img = draw_text(img, txt, (tx, ty - 12), size=22, color=use_color, bold=True)
    return img


def draw_dashed_line(img, p1, p2, color, thickness=2, dash=12, gap=8):
    x1, y1 = p1
    x2, y2 = p2
    length = math.hypot(x2 - x1, y2 - y1)
    if length < 1:
        return img
    dx = (x2 - x1) / length
    dy = (y2 - y1) / length
    t = 0
    while t < length:
        s = t
        e = min(t + dash, length)
        ps = (int(x1 + dx * s), int(y1 + dy * s))
        pe = (int(x1 + dx * e), int(y1 + dy * e))
        cv2.line(img, ps, pe, color, thickness, cv2.LINE_AA)
        t += dash + gap
    return img


def draw_trunk_corridor(
    img: np.ndarray,
    user_lms: Optional[np.ndarray],
    expert_lms: Optional[np.ndarray],
    user_m: Dict[str, float],
    issue: Optional[Issue],
    transform,
    tol_deg: float = 10.0,
):
    if not valid_lms(user_lms):
        return img
    hp = mid(user_lms, 23, 24)
    sh = mid(user_lms, 11, 12)
    hp_px = transform(hp)
    sh_px = transform(sh)
    tlen_px = max(math.dist(hp_px, sh_px), 40.0)

    # expert trunk 방향을 user hip 위치에 얹어서 corridor 표시
    if valid_lms(expert_lms):
        theta = signed_trunk_theta(expert_lms)
    else:
        theta = signed_trunk_theta(user_lms)
    tol = math.radians(tol_deg)
    length = tlen_px * 1.25

    def endpoint(th):
        return (int(hp_px[0] + math.cos(th) * length), int(hp_px[1] + math.sin(th) * length))

    p_mid = endpoint(theta)
    p_lo = endpoint(theta - tol)
    p_hi = endpoint(theta + tol)

    overlay = img.copy()
    poly = np.array([hp_px, p_lo, p_hi], dtype=np.int32)
    cv2.fillConvexPoly(overlay, poly, (40, 120, 60))
    img = cv2.addWeighted(overlay, 0.18, img, 0.82, 0)
    draw_dashed_line(img, hp_px, p_mid, C_OK, thickness=2, dash=10, gap=8)
    cv2.line(img, hp_px, sh_px, C_BAD if issue and "trunk" in issue.key else C_LINE, 5, cv2.LINE_AA)
    return img


def draw_bar_proxy_layer(
    img: np.ndarray,
    lms: Optional[np.ndarray],
    exercise: str,
    transform,
):
    if not valid_lms(lms):
        return img
    exercise = normalize_exercise_name(exercise)
    bp, conf = bar_proxy(lms)
    if bp is None or conf < 0.55:
        return img
    p = transform(bp)
    h, w = img.shape[:2]
    if exercise == "deadlift":
        # 손 위치 기반 proxy임을 명확히: vertical dashed + 짧은 grip line
        draw_dashed_line(img, (p[0], max(0, p[1] - 120)), (p[0], min(h - 1, p[1] + 160)), C_YELLOW, 2, 12, 8)
        cv2.line(img, (p[0] - 55, p[1]), (p[0] + 55, p[1]), C_YELLOW, 3, cv2.LINE_AA)
        cv2.circle(img, p, 7, C_YELLOW, -1, cv2.LINE_AA)
        img = draw_text(img, "bar proxy", (p[0] + 8, p[1] - 30), 17, C_YELLOW)
    elif exercise == "benchpress":
        # 양손이 모두 보이면 wrist/grip 연결선을 표시. 아니면 grip center만 표시.
        lp, lc = hand_grip_point(lms, "left")
        rp, rc = hand_grip_point(lms, "right")
        if lp is not None and rp is not None and lc >= 0.55 and rc >= 0.55:
            pl = transform(lp)
            pr = transform(rp)
            cv2.line(img, pl, pr, C_YELLOW, 4, cv2.LINE_AA)
            cv2.circle(img, pl, 7, C_YELLOW, -1, cv2.LINE_AA)
            cv2.circle(img, pr, 7, C_YELLOW, -1, cv2.LINE_AA)
        else:
            cv2.circle(img, p, 7, C_YELLOW, -1, cv2.LINE_AA)
    return img


def draw_wrist_elbow_guide(img: np.ndarray, lms: Optional[np.ndarray], side: str, transform, warn: bool):
    if not valid_lms(lms):
        return img
    ids = SIDE_IDS[side]
    el = transform(get_xy(lms, ids["elbow"]))
    wr = transform(get_xy(lms, ids["wrist"]))
    color = C_BAD if warn else C_CYAN
    # 손목-팔꿈치 수직 정렬 guide
    draw_dashed_line(img, (el[0], min(el[1], wr[1]) - 35), (el[0], max(el[1], wr[1]) + 35), color, 2, 8, 6)
    cv2.line(img, el, wr, color, 3, cv2.LINE_AA)
    return img


def draw_exercise_layers(
    img: np.ndarray,
    lms: Optional[np.ndarray],
    expert_lms: Optional[np.ndarray],
    exercise: str,
    side: str,
    user_m: Dict[str, float],
    issue: Optional[Issue],
    transform,
    is_expert: bool = False,
):
    if not valid_lms(lms):
        return img
    exercise = normalize_exercise_name(exercise)
    ids = SIDE_IDS[side]
    issue_key = issue.key if issue else ""

    # 공통: 운동별 true arc
    if exercise == "squat":
        warn_knee = issue is not None and ("knee" in issue_key or issue.key == "knee_angle")
        warn_hip = issue is not None and ("hip" in issue_key or issue.key == "hip_angle")
        img = draw_angle_arc(img, lms, (ids["hip"], ids["knee"], ids["ankle"]), transform, color=C_YELLOW, warn=warn_knee)
        # hip arc는 핵심 오류일 때만 크게 표시해서 화면 복잡도 줄임
        if warn_hip:
            img = draw_angle_arc(img, lms, (ids["shoulder"], ids["hip"], ids["knee"]), transform, color=C_CYAN, warn=True, radius_scale=0.30)
        if not is_expert:
            img = draw_trunk_corridor(img, lms, expert_lms, user_m, issue, transform, tol_deg=10.0)

    elif exercise == "deadlift":
        warn_hip = issue is not None and ("hip" in issue_key or issue.key == "hip_angle")
        warn_knee = issue is not None and ("knee" in issue_key or issue.key == "knee_angle")
        img = draw_angle_arc(img, lms, (ids["shoulder"], ids["hip"], ids["knee"]), transform, color=C_BAD if warn_hip else C_CYAN, warn=warn_hip)
        img = draw_angle_arc(img, lms, (ids["hip"], ids["knee"], ids["ankle"]), transform, color=C_YELLOW, warn=warn_knee, radius_scale=0.30)
        # 데드 팔꿈치/손목은 바벨에 가려져 신뢰 불가 → arc/bar proxy 표시 안 함.
        if not is_expert:
            img = draw_trunk_corridor(img, lms, expert_lms, user_m, issue, transform, tol_deg=10.0)
            if DRAW_BAR_PROXY.get(exercise, False):
                img = draw_bar_proxy_layer(img, lms, exercise, transform)

    elif exercise == "benchpress":
        warn_elbow = issue is not None and ("elbow" in issue_key or "lockout" in issue_key)
        warn_we = issue is not None and "wrist" in issue_key
        img = draw_angle_arc(img, lms, (ids["shoulder"], ids["elbow"], ids["wrist"]), transform, color=C_CYAN, warn=warn_elbow, radius_scale=0.40)
        if not is_expert:
            img = draw_wrist_elbow_guide(img, lms, side, transform, warn=warn_we)
            if DRAW_BAR_PROXY.get(exercise, False):
                img = draw_bar_proxy_layer(img, lms, exercise, transform)
            # bench line: 머리-어깨-엉덩이 라인
            p0 = transform(get_xy(lms, 0))
            ps = transform(mid(lms, 11, 12))
            ph = transform(mid(lms, 23, 24))
            line_color = C_BAD if issue and "bench" in issue.key else C_LINE
            cv2.line(img, p0, ps, line_color, 2, cv2.LINE_AA)
            cv2.line(img, ps, ph, line_color, 2, cv2.LINE_AA)
    return img


def draw_feedback_banner(img: np.ndarray, exercise: str, phase: str, issue: Optional[Issue]):
    h, w = img.shape[:2]
    main = issue.message if issue else "정상 범위"
    color = C_BAD if issue else C_OK
    bg = (38, 38, 48) if issue else (30, 48, 36)
    text = f"{phase}  |  {main}"
    x1, y1, x2 = 10, 54, w - 10
    max_text_h = min(180, max(48, h - y1 - 18))
    font_size, lines, line_h = _fit_wrapped_text(
        text,
        max_width=max(80, x2 - x1 - 20),
        max_height=max_text_h,
        max_size=18,
        min_size=9,
        bold=True,
        max_lines=7,
    )
    box_h = 18 + line_h * len(lines) + 12
    y2 = min(h - 8, y1 + box_h)
    img = draw_round_rect(img, (x1, y1), (x2, y2), bg, alpha=0.78)
    img = _draw_wrapped_text(img, lines, (x1 + 10, y1 + 12), font_size, color, bold=True, line_h=line_h)
    return img


def draw_metric_row(
    img: np.ndarray,
    x: int,
    y: int,
    w: int,
    label: str,
    u: Optional[float],
    e: Optional[float],
    d: Optional[float],
    warn: bool,
    advisory: bool = False,
):
    measurable = u is not None
    if advisory:
        color = C_CYAN  # 보조 지표는 파란 톤으로 구분
    else:
        color = C_GRAY if not measurable else (C_BAD if warn else C_OK)
    img = draw_text(img, label, (x, y), 18, C_WHITE, bold=True)
    if not measurable:
        status = "측정 불가"
    elif advisory:
        status = "참고"
    else:
        status = "주의" if warn else "OK"
    img = draw_text(img, status, (x + w - 78, y), 17, color, bold=True)

    def fmt(v):
        if v is None:
            return "-"
        if abs(v) >= 10:
            return f"{v:.1f}"
        return f"{v:.2f}"

    txt = f"U {fmt(u)} / E {fmt(e)} / Δ {fmt(d)}"
    img = draw_text(img, txt, (x, y + 22), 14, C_GRAY)
    bar_x = x
    bar_y = y + 44
    bar_w = w - 22
    cv2.line(img, (bar_x, bar_y), (bar_x + bar_w, bar_y), C_DARKGRAY, 5, cv2.LINE_AA)
    if measurable and d is not None:
        # delta bar는 절대값 클수록 길게. 최대 1로 clamp.
        ratio = clamp(abs(d) / (abs(d) + 1.0), 0.08, 1.0)
        bar_color = C_CYAN if advisory else color
        cv2.line(img, (bar_x, bar_y), (bar_x + int(bar_w * ratio), bar_y), bar_color, 5, cv2.LINE_AA)
    return img


def draw_hud(
    img: np.ndarray,
    exercise: str,
    phase: str,
    rep_count: int,
    user_m: Dict[str, float],
    expert_m: Dict[str, float],
    deltas: Dict[str, float],
    issue: Optional[Issue],
    fps_now: float,
    frame_info: str,
):
    h, w = img.shape[:2]
    exercise = normalize_exercise_name(exercise)
    cv2.rectangle(img, (0, 0), (w, h), C_BG, -1)

    title = {"squat": "SQUAT", "deadlift": "DEADLIFT", "benchpress": "BENCH PRESS"}[exercise]
    img = draw_text(img, title, (18, 18), 29, C_YELLOW, bold=True)
    img = draw_text(img, f"횟수  {rep_count}", (w - 112, 24), 18, C_GRAY)
    img = draw_text(img, str(rep_count), (w - 42, 18), 30, C_CYAN, bold=True)

    img = draw_round_rect(img, (18, 66), (w - 18, 106), C_PANEL, alpha=0.95)
    img = draw_text(img, phase, (30, 76), 18, C_WHITE)

    fb_text = issue.message if issue else "정상 범위"
    fb_color = C_BAD if issue else C_OK
    fb_x1, fb_y1, fb_x2 = 18, 126, w - 18
    title_y = fb_y1 + 10
    text_y = fb_y1 + 36
    font_size, fb_lines, line_h = _fit_wrapped_text(
        fb_text,
        max_width=max(80, fb_x2 - fb_x1 - 24),
        max_height=190,
        max_size=16,
        min_size=8,
        bold=True,
        max_lines=9,
    )
    fb_y2 = min(h - 90, text_y + line_h * len(fb_lines) + 14)
    img = draw_round_rect(img, (fb_x1, fb_y1), (fb_x2, fb_y2), C_PANEL, alpha=0.95)
    cv2.rectangle(img, (fb_x1, fb_y1), (fb_x2, fb_y2), fb_color, 2)
    img = draw_text(img, "현재 핵심 피드백", (30, title_y), 15, C_GRAY)
    img = _draw_wrapped_text(img, fb_lines, (30, text_y), font_size, fb_color, bold=True, line_h=line_h)

    meta_y = fb_y2 + 18
    line_y = meta_y + 26
    img = draw_text(img, f"{frame_info}   FPS {fps_now:.1f}", (18, meta_y), 14, C_GRAY)
    cv2.line(img, (18, line_y), (w - 18, line_y), C_DARKGRAY, 1, cv2.LINE_AA)

    y = line_y + 16
    metrics = DISPLAY_METRICS.get(exercise, [])
    thr = DELTA_THRESHOLDS.get(exercise, {})
    advisory_set = ADVISORY_METRICS.get(exercise, set())
    for key in metrics:
        if y > h - 90:
            break
        label = METRIC_LABELS_KO.get(key, key)
        u = user_m.get(key)
        e = expert_m.get(key)
        d = deltas.get(key)
        is_adv = key in advisory_set
        warn = False
        if not is_adv:
            if key in thr and thr[key] > 0 and d is not None:
                warn = abs(d) >= thr[key]
            if issue and (issue.key == key or key in issue.key):
                warn = True
        img = draw_metric_row(img, 28, y, w - 46, label, u, e, d, warn, advisory=is_adv)
        y += 66

    cv2.line(img, (18, h - 55), (w - 18, h - 55), C_DARKGRAY, 1, cv2.LINE_AA)
    img = draw_text(img, "빨간 원 = 현재 우선 확인 관절", (20, h - 42), 14, C_GRAY)
    img = draw_text(img, "arc = 관절 중심 실제 각도", (20, h - 22), 14, C_GRAY)
    return img


# ============================================================
# 7. Expert profile 로드/생성
# ============================================================


def save_expert_profile(
    path: Path,
    frames: List[dict],
    fps: float,
    source_video: str,
    phase: Optional[List[int]] = None,
    reps: Optional[List[dict]] = None,
    ref_rep_index: Optional[int] = None,
    rep_rule: Optional[str] = None,
):
    data = {
        "version": "unified_feedback_v3" if phase is not None and reps is not None else "unified_feedback_v2",
        "fps": fps,
        "source_video": source_video,
        "frames": frames,
    }
    if phase is not None and reps is not None:
        data["phase"] = [int(x) for x in phase]
        data["reps"] = reps
        data["ref_rep_index"] = None if ref_rep_index is None else int(ref_rep_index)
        if rep_rule is not None:
            data["rep_rule"] = str(rep_rule)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def _profile_has_phase(data: dict) -> bool:
    return (
        data.get("version") == "unified_feedback_v3"
        and isinstance(data.get("phase"), list)
        and isinstance(data.get("reps"), list)
    )


def load_expert_profile(path: Path) -> Optional[Tuple[List[dict], float, dict]]:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "frames" in data:
            frames = data["frames"]
            fps = safe_float(data.get("fps", 30.0), 30.0)
            return frames, fps, data
        if isinstance(data, list):
            return data, 30.0, {"version": "legacy_list", "frames": data}
    except Exception as e:
        print(f"[경고] expert JSON 로드 실패: {path} | {e}")
    return None


def frame_landmarks_to_list(lms: Optional[np.ndarray]) -> Optional[List[List[float]]]:
    if not valid_lms(lms):
        return None
    return [[float(x) for x in row] for row in lms.tolist()]


def list_to_lms(x) -> Optional[np.ndarray]:
    if x is None:
        return None
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 2 and arr.shape[0] >= 33 and arr.shape[1] >= 4:
        return arr[:33, :4].copy()
    return None


def _phase_debug_summary(phase: Sequence[int] | np.ndarray) -> str:
    arr = np.asarray(phase, dtype=np.int16).reshape(-1)
    if arr.size == 0:
        return "counts={}, runs=[]"

    values, counts = np.unique(arr, return_counts=True)
    count_map = {model_phase_name(int(v)): int(c) for v, c in zip(values, counts)}
    runs = []
    start = 0
    for i in range(1, arr.size + 1):
        if i == arr.size or arr[i] != arr[start]:
            name = model_phase_name(int(arr[start]))
            runs.append(f"{name}:{start}-{i}({i - start})")
            start = i
    shown = runs[:24]
    suffix = "" if len(runs) <= len(shown) else f" ... +{len(runs) - len(shown)} runs"
    return f"counts={count_map}, runs={shown}{suffix}"


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else BASE_DIR / path


def _expert_json_path(config: dict) -> Path:
    return _resolve_project_path(config["expert_video"]).with_suffix(".json")


def _augment_expert_phase(exercise: str, frames: List[dict], phase_adapter: PhaseModelAdapter, config: dict) -> dict:
    """Attach required model phase/reps/ref metadata to expert landmark frames."""
    if phase_adapter is None or not getattr(phase_adapter, "available", False):
        raise RuntimeError("phase_adapter is not ready; model phase is required.")

    exercise = normalize_exercise_name(exercise)
    pose_seq = pose_seq_from_landmarks(list_to_lms(f.get("landmarks")) for f in frames)
    result = phase_adapter.infer(pose_seq)
    count_phase = smooth_phase_like_training(result.phase_per_frame, PHASE_COUNT_SMOOTH_WINDOW)
    model_rep_count, _model_count_transitions = count_phases_like_training(
        count_phase,
        PHASE_COUNT_MIN_UP_LEN,
    )
    phase = bridge_ready_gaps(
        debounce_phase(result.phase_per_frame, PHASE_DEBOUNCE_MIN_LEN),
        PHASE_READY_BRIDGE_MAX_LEN,
    )
    reps = segment_reps(phase)
    rep_rule = "down_up"
    if not reps and exercise == "deadlift":
        reps = segment_up_reps(phase, PHASE_COUNT_MIN_UP_LEN)
        rep_rule = "up_only"
    if not reps:
        raise RuntimeError(
            "expert phase has no complete down->up reps. "
            + _phase_debug_summary(phase)
        )

    ref_rep = pick_reference_rep(
        reps,
        frames,
        config.get("expert_ref_rep_index"),
        allow_single_phase=(rep_rule == "up_only"),
    )
    if ref_rep is None:
        raise RuntimeError(
            "failed to pick expert_ref_rep_index. "
            f"expert_ref_rep_index={config.get('expert_ref_rep_index')}, reps={len(reps)}"
        )

    meta = {
        "phase": [int(x) for x in phase.tolist()],
        "reps": [rep_to_dict(r) for r in reps],
        "ref_rep_index": int(ref_rep.index),
        "rep_rule": rep_rule,
    }
    print(
        f"[MODEL] expert phase: action={normalize_exercise_name(result.action)} "
        f"conf={result.action_conf:.3f}, reps={len(reps)}, training_count={model_rep_count}, "
        f"rule={rep_rule}, ref={meta['ref_rep_index']}"
    )
    return meta


def _extract_expert_phase_meta(profile_data: dict) -> dict:
    if not _profile_has_phase(profile_data):
        return {"phase": None, "reps": [], "ref_rep_index": None}
    return {
        "phase": profile_data.get("phase"),
        "reps": profile_data.get("reps") or [],
        "ref_rep_index": profile_data.get("ref_rep_index"),
        "rep_rule": profile_data.get("rep_rule", "down_up"),
    }


def _require_expert_phase_meta(phase_meta: dict, frames: List[dict], json_path: Path) -> None:
    phase = phase_meta.get("phase")
    if phase is None:
        raise RuntimeError(f"expert JSON has no model phase metadata: {json_path}")
    if len(phase) != len(frames):
        raise RuntimeError(
            f"expert phase/frame length mismatch: phase={len(phase)}, frames={len(frames)}, json={json_path}"
        )
    if not phase_meta.get("reps"):
        raise RuntimeError(
            f"expert phase reps=0: {json_path}. " + _phase_debug_summary(phase)
        )
    if phase_meta.get("ref_rep_index") is None:
        raise RuntimeError(f"expert ref_rep_index is missing: {json_path}")


def build_expert_profile(
    exercise: str,
    config: dict,
    phase_adapter: Optional[PhaseModelAdapter | LazyPhaseAdapter],
) -> Tuple[List[dict], float, dict]:
    exercise = normalize_exercise_name(exercise)
    json_path = _expert_json_path(config)

    loaded = load_expert_profile(json_path)
    if loaded is not None:
        frames, fps, profile_data = loaded
        if not frames:
            raise RuntimeError(f"expert JSON frames are empty: {json_path}")
        phase_meta = _extract_expert_phase_meta(profile_data)
        if phase_meta["phase"] is None:
            print(f"[EXPERT] cached JSON has no phase metadata; upgrading: {json_path}")
            if phase_adapter is None:
                phase_adapter = _create_phase_adapter()
            elif isinstance(phase_adapter, LazyPhaseAdapter):
                phase_adapter = phase_adapter.get()
            phase_meta = _augment_expert_phase(exercise, frames, phase_adapter, config)
            save_expert_profile(
                json_path,
                frames,
                fps,
                str(profile_data.get("source_video", _resolve_project_path(config["expert_video"]))),
                phase=phase_meta["phase"],
                reps=phase_meta["reps"],
                ref_rep_index=phase_meta["ref_rep_index"],
                rep_rule=phase_meta.get("rep_rule"),
            )
        _require_expert_phase_meta(phase_meta, frames, json_path)
        if phase_meta.get("rep_rule") == "up_only" and config.get("expert_ref_rep_index") is None:
            reps = [rep_from_dict(x) for x in phase_meta.get("reps") or []]
            ref_rep = pick_reference_rep(reps, frames, allow_single_phase=True)
            if ref_rep is not None and int(phase_meta.get("ref_rep_index")) != int(ref_rep.index):
                print(
                    f"[EXPERT] cached up-only ref rep normalized: "
                    f"{phase_meta.get('ref_rep_index')} -> {ref_rep.index}"
                )
                phase_meta["ref_rep_index"] = int(ref_rep.index)
                save_expert_profile(
                    json_path,
                    frames,
                    fps,
                    str(profile_data.get("source_video", _resolve_project_path(config["expert_video"]))),
                    phase=phase_meta["phase"],
                    reps=phase_meta["reps"],
                    ref_rep_index=phase_meta["ref_rep_index"],
                    rep_rule=phase_meta.get("rep_rule"),
                )
        print(f"[EXPERT] cache loaded: {json_path} ({len(frames)} frames)")
        return frames, fps, phase_meta

    video_path = _resolve_project_path(config["expert_video"])
    if not video_path.exists():
        raise FileNotFoundError(f"expert video/json missing. video={video_path}, json={json_path}")

    cap, exp_rot = open_video_normalized(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open expert video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    smoother = LandmarkSmoother(EXPERT_EMA_ALPHA, VIS_THR)
    side_lock = SideLock(exercise)
    expert_selector = PersonSelector(exercise)
    frames: List[dict] = []

    print(f"[EXPERT] preprocess start: {video_path} | fps={fps:.2f}, frames={total}")
    with create_landmarker(MODEL_PATH) as landmarker:
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = apply_rotation(frame, exp_rot)
            timestamp_ms = int(idx * 1000 / fps)
            raw = expert_selector.select(extract_all_landmarks(landmarker, frame, timestamp_ms))
            lms = smoother.update(raw)
            side = side_lock.update(lms)
            metrics = compute_metrics(lms, exercise, side) if valid_lms(lms) else {}
            frames.append({
                "frame_idx": idx,
                "timestamp_ms": timestamp_ms,
                "side": side,
                "landmarks": frame_landmarks_to_list(lms),
                "metrics": metrics,
            })
            idx += 1
            if idx % 50 == 0:
                print(f"  expert preprocess {idx}/{total if total else '?'}")
    cap.release()

    if not frames:
        raise RuntimeError(f"expert preprocessing produced no frames: {video_path}")

    if phase_adapter is None:
        phase_adapter = _create_phase_adapter()
    elif isinstance(phase_adapter, LazyPhaseAdapter):
        phase_adapter = phase_adapter.get()
    phase_meta = _augment_expert_phase(exercise, frames, phase_adapter, config)
    _require_expert_phase_meta(phase_meta, frames, json_path)
    save_expert_profile(
        json_path,
        frames,
        fps,
        str(video_path),
        phase=phase_meta["phase"],
        reps=phase_meta["reps"],
        ref_rep_index=phase_meta["ref_rep_index"],
        rep_rule=phase_meta.get("rep_rule"),
    )
    print(f"[EXPERT] cache saved: {json_path} ({len(frames)} frames)")
    return frames, fps, phase_meta


# ============================================================
# 8. 메인 처리
# ============================================================


def get_video_rotation(video_path: str) -> int:
    """영상의 회전 메타데이터(0/90/180/270)를 읽는다.

    핸드폰 등에서 세로로 찍은 영상을 '재생 시 회전' 메타데이터로만 가로처럼
    보이게 저장하는 경우가 있다. OpenCV는 환경(빌드)에 따라 이 메타데이터를
    적용하기도/무시하기도 해서, 같은 파일이 PC마다 가로/세로로 다르게 읽힌다.
    이를 코드에서 직접 보정해 어디서나 동일한 방향이 되도록 한다.

    ffprobe가 있으면 그것으로, 없으면 0을 반환(보정 안 함)."""
    try:
        import subprocess, json
        out = subprocess.check_output(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_streams", str(video_path)],
            stderr=subprocess.DEVNULL,
        ).decode()
        d = json.loads(out)
        for s in d.get("streams", []):
            if s.get("codec_type") != "video":
                continue
            tag = s.get("tags", {}).get("rotate")
            if tag is not None:
                return int(tag) % 360
            for sd in s.get("side_data_list", []):
                if "rotation" in sd:
                    # ffmpeg side_data rotation은 부호가 반대 관례일 수 있음
                    return (-int(sd["rotation"])) % 360
    except Exception:
        pass
    return 0


def apply_rotation(frame: np.ndarray, rotation: int) -> np.ndarray:
    """회전 메타데이터(시계방향 deg)를 프레임 픽셀에 적용해 의도된 방향으로 만든다."""
    if rotation == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if rotation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if rotation == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def open_video_normalized(video_path: str):
    """VideoCapture와 함께, 프레임을 의도된 방향으로 보정하기 위한 회전값을 반환.

    핵심: OpenCV가 이미 메타데이터를 적용해 가로로 읽고 있는지(=프레임이 이미
    가로인지) 확인해서, 이중 회전을 피한다.
    반환: (cap, need_rotation)  need_rotation은 픽셀에 추가로 적용할 회전(deg)."""
    cap = cv2.VideoCapture(str(video_path))
    meta_rot = get_video_rotation(video_path)
    if meta_rot in (90, 270):
        # 메타데이터상 세로→가로 회전이 필요한 영상.
        # OpenCV가 읽은 첫 프레임이 이미 가로면(=메타 적용됨) 추가 회전 불필요.
        ok, fr = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        if ok:
            h, w = fr.shape[:2]
            if w >= h:
                return cap, 0          # 이미 가로로 읽힘 → 보정 불필요
            return cap, meta_rot       # 세로로 읽힘 → 메타 회전 적용 필요
    return cap, 0


def resize_frame_if_needed(frame: np.ndarray) -> np.ndarray:
    if MAX_OUTPUT_H is None:
        return frame
    h, w = frame.shape[:2]
    if h <= MAX_OUTPUT_H:
        return frame
    scale = MAX_OUTPUT_H / float(h)
    return cv2.resize(frame, (int(w * scale), MAX_OUTPUT_H), interpolation=cv2.INTER_AREA)


def _create_phase_adapter() -> PhaseModelAdapter:
    if not USE_MODEL_PHASE:
        raise RuntimeError("USE_MODEL_PHASE=False? ???? ????. model phase? ?????.")

    ckpt = MODEL_CKPT if MODEL_CKPT else None
    adapter = PhaseModelAdapter(
        ckpt_path=ckpt,
        device=MODEL_DEVICE,
        clip_len=MODEL_CLIP_LEN,
        stride=MODEL_STRIDE,
        input_kind=MODEL_INPUT_KIND,
        graph=MODEL_GRAPH,
        anchor=WINDOW_ANCHOR,
        required=True,
    )
    if not adapter.available:
        raise RuntimeError(f"phase model unavailable: {adapter.unavailable_reason}")

    print(
        f"[MODEL] phase adapter ready: ckpt={adapter.resolved_checkpoint}, "
        f"clip={adapter.clip_len}, input={adapter.cfg.get('derivative_mode')}"
    )
    if adapter.warning:
        print(f"[MODEL][WARN] {adapter.warning}")
    return adapter

def _median_body_axis_from_lms_seq(lms_seq: Sequence[Optional[np.ndarray]], max_frames: int = 40) -> Optional[float]:
    angles = []
    for lms in lms_seq[:max_frames]:
        ang = _body_axis_angle(lms)
        if ang is not None and np.isfinite(ang):
            angles.append(float(ang))
    if not angles:
        return None
    return float(np.median(np.asarray(angles, dtype=np.float32)))


def _rep_from_meta(
    reps_meta: Sequence[dict],
    ref_rep_index: Optional[int],
    allow_single_phase: bool = False,
) -> Optional[Rep]:
    reps = [rep_from_dict(x) for x in reps_meta]
    if ref_rep_index is not None:
        for rep in reps:
            is_usable = (rep.down is not None and rep.up is not None) or (allow_single_phase and rep.up is not None)
            if rep.index == int(ref_rep_index) and is_usable:
                return rep
    return pick_reference_rep(reps, preferred_index=ref_rep_index, allow_single_phase=allow_single_phase)


@dataclass
class RealtimeExpertProfile:
    exercise: str
    frames: List[dict]
    fps: float
    phase_meta: dict
    ref_rep: Rep
    rep_rule: str


@dataclass
class LazyPhaseAdapter:
    adapter: Optional[PhaseModelAdapter] = None

    def get(self) -> PhaseModelAdapter:
        if self.adapter is None:
            self.adapter = _create_phase_adapter()
        return self.adapter


class RealtimeActionSmoother:
    """Small majority vote smoother for the model action head."""

    def __init__(self, window_size: int, min_confidence: float, min_votes: int = 1):
        self.window_size = max(1, int(window_size))
        self.min_confidence = float(min_confidence)
        self.min_votes = max(1, min(int(min_votes), self.window_size))
        self.items: List[str] = []
        self.current: Optional[str] = None

    def reset(self):
        self.items.clear()
        self.current = None

    def update(self, action: Optional[str], confidence: float) -> Optional[str]:
        if action is None or float(confidence) < self.min_confidence:
            return self.current
        try:
            normalized = normalize_exercise_name(str(action))
        except Exception:
            return self.current
        if normalized not in VIDEO_CONFIG:
            return self.current

        self.items.append(normalized)
        if len(self.items) > self.window_size:
            self.items = self.items[-self.window_size :]

        counts: Dict[str, int] = {}
        latest: Dict[str, int] = {}
        for idx, value in enumerate(self.items):
            counts[value] = counts.get(value, 0) + 1
            latest[value] = idx
        selected = max(counts, key=lambda value: (counts[value], latest[value]))
        if counts[selected] < self.min_votes:
            return self.current
        self.current = selected
        return self.current


def _normalize_realtime_initial_exercise(value) -> Optional[str]:
    text = str(value or "auto").strip().lower()
    if text in {"", "auto", "model"}:
        return None
    exercise = normalize_exercise_name(text)
    if exercise not in VIDEO_CONFIG:
        raise KeyError(f"REALTIME_INITIAL_EXERCISE is not in VIDEO_CONFIG: {exercise}")
    return exercise


def _is_video_config_realtime_source(source) -> bool:
    if source is None:
        return False
    return str(source).strip().lower() in {
        "video_config",
        "videoconfig",
        "config",
        "user_video",
    }


def _resolve_video_config_realtime_source(initial_value) -> Tuple[Path, str]:
    exercise = _normalize_realtime_initial_exercise(initial_value)
    if exercise is None:
        targets = _get_run_targets()
        if not targets:
            raise RuntimeError("RUN_EXERCISES is empty; cannot resolve VIDEO_CONFIG source.")
        exercise = targets[0]

    config = VIDEO_CONFIG[exercise]
    path = _resolve_project_path(config["user_video"])
    if not path.exists():
        raise FileNotFoundError(f"VIDEO_CONFIG user_video missing: exercise={exercise}, path={path}")
    return path, exercise


def _lms_to_realtime_kpts(lms: np.ndarray) -> np.ndarray:
    if not valid_lms(lms):
        raise ValueError("cannot convert invalid landmarks to realtime keypoints")
    arr = np.asarray(lms, dtype=np.float32)
    out = np.zeros((33, 3), dtype=np.float32)
    out[:, 0:2] = arr[:33, 0:2]
    if arr.shape[1] >= 4:
        out[:, 2] = arr[:33, 3]
    else:
        out[:, 2] = 1.0
    return out


def _has_raw_pose_detection(lms: Optional[np.ndarray]) -> bool:
    return bool(lms is not None and valid_lms(lms))


def _open_realtime_capture(source):
    is_camera = False
    rotation = 0
    label = str(source)

    if isinstance(source, int):
        is_camera = True
        cap = cv2.VideoCapture(int(source))
        label = f"camera:{int(source)}"
    else:
        text = str(source).strip()
        if text.isdigit():
            is_camera = True
            cap = cv2.VideoCapture(int(text))
            label = f"camera:{int(text)}"
        elif "://" in text:
            cap = cv2.VideoCapture(text)
            label = text
        else:
            path = _resolve_project_path(text)
            if not path.exists():
                raise FileNotFoundError(f"realtime source missing: {path}")
            cap, rotation = open_video_normalized(str(path))
            label = str(path)

    if is_camera:
        if REALTIME_CAMERA_WIDTH:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(REALTIME_CAMERA_WIDTH))
        if REALTIME_CAMERA_HEIGHT:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(REALTIME_CAMERA_HEIGHT))

    if not cap.isOpened():
        raise RuntimeError(f"failed to open realtime source: {label}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if not np.isfinite(fps) or fps <= 1e-6:
        fps = 30.0
    return cap, rotation, label, fps, is_camera


def _create_realtime_infer():
    from model.realtime_stgcn_infer import RealtimeSTGCNInfer

    infer = RealtimeSTGCNInfer(
        checkpoint_path=MODEL_CKPT if MODEL_CKPT else None,
        device=MODEL_DEVICE,
        clip_len_override=MODEL_CLIP_LEN,
        smooth_window=PHASE_COUNT_SMOOTH_WINDOW,
        min_up_len=PHASE_COUNT_MIN_UP_LEN,
        inference_interval=REALTIME_MODEL_INFERENCE_INTERVAL,
        ready_bridge_max_len=REALTIME_READY_BRIDGE_MAX_LEN,
        require_prior_down_actions=("squat", "benchpress"),
        required=True,
        repo_root=BASE_DIR,
    )
    if not infer.available:
        raise RuntimeError(f"realtime model unavailable: {infer.unavailable_reason}")
    print(
        f"[MODEL] realtime ready: ckpt={infer.checkpoint_path}, "
        f"clip={infer.clip_len}, input={infer.cfg.get('derivative_mode')}"
    )
    if infer.warning:
        print(f"[MODEL][WARN] {infer.warning}")
    return infer


def _get_realtime_expert_profile(
    exercise: str,
    phase_adapter: Optional[PhaseModelAdapter | LazyPhaseAdapter],
    cache: Dict[str, RealtimeExpertProfile],
) -> RealtimeExpertProfile:
    exercise = normalize_exercise_name(exercise)
    if exercise in cache:
        return cache[exercise]

    config = VIDEO_CONFIG[exercise]
    frames, fps, phase_meta = build_expert_profile(exercise, config, phase_adapter)
    rep_rule = str(phase_meta.get("rep_rule", "down_up"))
    ref_rep = _rep_from_meta(
        phase_meta.get("reps") or [],
        phase_meta.get("ref_rep_index"),
        allow_single_phase=(rep_rule == "up_only"),
    )
    if ref_rep is None:
        raise RuntimeError(
            f"failed to resolve realtime expert reference rep: exercise={exercise}, "
            f"ref={phase_meta.get('ref_rep_index')}, reps={len(phase_meta.get('reps') or [])}"
        )

    profile = RealtimeExpertProfile(
        exercise=exercise,
        frames=frames,
        fps=fps,
        phase_meta=phase_meta,
        ref_rep=ref_rep,
        rep_rule=rep_rule,
    )
    cache[exercise] = profile
    print(f"[ALIGN] realtime expert ready: exercise={exercise}, ref_rep={ref_rep.index}, rule={rep_rule}")
    return profile


def _resolve_realtime_output_path(output_value) -> Optional[Path]:
    if output_value is None:
        return None
    text = str(output_value).strip()
    if not text:
        return None
    path = Path(text)
    if path.is_absolute():
        return path
    if path.parent == Path("."):
        return OUTPUT_DIR / path
    return BASE_DIR / path


def _draw_realtime_status_frame(
    frame: np.ndarray,
    user_lms: Optional[np.ndarray],
    status_lines: Sequence[str],
    fps_now: float,
) -> np.ndarray:
    frame = resize_frame_if_needed(frame)
    user_h, user_w = frame.shape[:2]
    out_h = max(user_h, MIN_PANEL_H)
    out_w = user_w + EXPERT_W + HUD_W

    user_canvas = frame.copy()
    user_canvas = draw_text(user_canvas, "USER", (8, 8), 25, C_YELLOW, bold=True)
    user_canvas = draw_skeleton(user_canvas, user_lms, make_user_transform(user_w, user_h), line_color=C_LINE, thickness=3)
    if user_canvas.shape[0] != out_h:
        pad_total = out_h - user_canvas.shape[0]
        user_canvas = cv2.copyMakeBorder(
            user_canvas,
            max(0, pad_total // 2),
            max(0, pad_total - pad_total // 2),
            0,
            0,
            cv2.BORDER_CONSTANT,
            value=C_BG,
        )

    expert_canvas = np.zeros((out_h, EXPERT_W, 3), dtype=np.uint8)
    expert_canvas[:] = C_BG
    expert_canvas = draw_text(expert_canvas, "EXPERT", (14, 8), 25, C_OK, bold=True)
    expert_canvas = draw_text(expert_canvas, "waiting action", (24, 72), 18, C_GRAY)

    hud = np.zeros((out_h, HUD_W, 3), dtype=np.uint8)
    hud[:] = C_BG
    hud = draw_text(hud, "REALTIME", (18, 22), 24, C_CYAN, bold=True)
    hud = draw_text(hud, f"FPS {fps_now:.1f}", (18, 58), 16, C_GRAY)
    y = 96
    for line in status_lines:
        if y > out_h - 40:
            break
        hud = draw_text(hud, str(line), (18, y), 16, C_WHITE)
        y += 30
    hud = draw_text(hud, "q/ESC: quit   r: reset", (18, out_h - 34), 14, C_GRAY)

    sep1 = np.full((out_h, 2, 3), (72, 60, 66), dtype=np.uint8)
    sep2 = np.full((out_h, 2, 3), (72, 60, 66), dtype=np.uint8)
    combined = np.hstack([user_canvas, sep1, expert_canvas[:, :EXPERT_W - 2], sep2, hud[:, :HUD_W - 2]])
    if combined.shape[1] != out_w:
        combined = cv2.resize(combined, (out_w, out_h), interpolation=cv2.INTER_AREA)
    return combined


def _compose_realtime_feedback_frame(
    frame: np.ndarray,
    exercise: str,
    user_lms: np.ndarray,
    user_side: str,
    user_m: Dict[str, float],
    expert_lms: np.ndarray,
    expert_side: str,
    expert_m: Dict[str, float],
    banner_phase: str,
    rep_count: int,
    issue: Optional[Issue],
    fps_now: float,
    frame_info: str,
    expert_norm: ExpertPaneNormalizer,
    model_line: str,
) -> np.ndarray:
    frame = resize_frame_if_needed(frame)
    user_h, user_w = frame.shape[:2]
    out_h = max(user_h, MIN_PANEL_H)
    out_w = user_w + EXPERT_W + HUD_W

    deltas = compute_deltas(user_m, expert_m)
    bad = issue.landmarks if issue else []

    user_canvas = frame.copy()
    user_tr = make_user_transform(user_w, user_h)
    user_canvas = draw_text(user_canvas, "USER", (8, 8), 25, C_YELLOW, bold=True)
    user_canvas = draw_skeleton(user_canvas, user_lms, user_tr, bad_indices=bad, line_color=C_LINE, thickness=3)
    user_canvas = draw_exercise_layers(user_canvas, user_lms, expert_lms, exercise, user_side, user_m, issue, user_tr, is_expert=False)
    user_canvas = draw_feedback_banner(user_canvas, exercise, banner_phase, issue)
    if user_canvas.shape[0] != out_h:
        pad_total = out_h - user_canvas.shape[0]
        user_canvas = cv2.copyMakeBorder(
            user_canvas,
            max(0, pad_total // 2),
            max(0, pad_total - pad_total // 2),
            0,
            0,
            cv2.BORDER_CONSTANT,
            value=C_BG,
        )

    expert_canvas = np.zeros((out_h, EXPERT_W, 3), dtype=np.uint8)
    expert_canvas[:] = C_BG
    expert_canvas = draw_text(expert_canvas, "EXPERT", (14, 8), 25, C_OK, bold=True)
    expert_norm.observe_user(user_lms)
    ex_tr = expert_norm.update(expert_lms)
    expert_canvas = draw_skeleton(expert_canvas, expert_lms, ex_tr, bad_indices=bad, line_color=C_LINE, thickness=3)
    expert_canvas = draw_exercise_layers(expert_canvas, expert_lms, None, exercise, expert_side, expert_m, issue, ex_tr, is_expert=True)

    hud = np.zeros((out_h, HUD_W, 3), dtype=np.uint8)
    hud = draw_hud(hud, exercise, banner_phase, rep_count, user_m, expert_m, deltas, issue, fps_now, frame_info)
    hud = draw_text(hud, model_line, (18, max(250, out_h - 78)), 14, C_GRAY)
    hud = draw_text(hud, "q/ESC: quit   r: reset", (18, out_h - 34), 14, C_GRAY)

    sep1 = np.full((out_h, 2, 3), (72, 60, 66), dtype=np.uint8)
    sep2 = np.full((out_h, 2, 3), (72, 60, 66), dtype=np.uint8)
    combined = np.hstack([user_canvas, sep1, expert_canvas[:, :EXPERT_W - 2], sep2, hud[:, :HUD_W - 2]])
    if combined.shape[1] != out_w:
        combined = cv2.resize(combined, (out_w, out_h), interpolation=cv2.INTER_AREA)
    return combined


def _trim_invalid_user_edges(
    frames: List[np.ndarray],
    lms_seq: List[Optional[np.ndarray]],
    side_seq: List[str],
    metrics_seq: List[Dict[str, float]],
    ts_seq: List[int],
) -> Tuple[List[np.ndarray], List[Optional[np.ndarray]], List[str], List[Dict[str, float]], List[int], int, int]:
    valid_mask = [bool(m) and valid_lms(lms) for lms, m in zip(lms_seq, metrics_seq)]
    if not any(valid_mask):
        raise RuntimeError("user pose/metrics were never detected; cannot run without fallback.")

    first = next(i for i, ok in enumerate(valid_mask) if ok)
    last = len(valid_mask) - 1 - next(i for i, ok in enumerate(reversed(valid_mask)) if ok)
    internal_missing = [i for i in range(first, last + 1) if not valid_mask[i]]
    if internal_missing:
        preview = internal_missing[:20]
        suffix = "" if len(internal_missing) <= len(preview) else f" ... +{len(internal_missing) - len(preview)}"
        raise RuntimeError(
            "user pose/metrics missing inside usable range; cannot run without fallback. "
            f"frames={preview}{suffix}"
        )

    if first > 0 or last < len(valid_mask) - 1:
        print(
            f"[SCAN] trim invalid edge frames: start={first}, "
            f"end={len(valid_mask) - 1 - last}, kept={last - first + 1}/{len(valid_mask)}"
        )

    sl = slice(first, last + 1)
    return frames[sl], lms_seq[sl], side_seq[sl], metrics_seq[sl], ts_seq[sl], first, last


def process_exercise(exercise: str):
    requested_exercise = normalize_exercise_name(exercise)
    exercise = requested_exercise
    config = VIDEO_CONFIG[exercise]
    user_video = _resolve_project_path(config["user_video"])
    if not user_video.exists():
        raise FileNotFoundError(f"??? ??? ????: {user_video}")

    if PHASE_ALIGN != "ratio":
        raise ValueError("PR1-PR3??? --align ratio? ?????. DTW? ?? PR ?????.")

    phase_adapter = _create_phase_adapter()

    cap, user_rot = open_video_normalized(str(user_video))
    if not cap.isOpened():
        raise RuntimeError(f"??? ?? ?? ??: {user_video}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    source_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    smoother = LandmarkSmoother(LANDMARK_EMA_ALPHA, VIS_THR)
    side_lock = SideLock(exercise)
    user_selector = PersonSelector(exercise)

    user_frames: List[np.ndarray] = []
    user_lms_seq: List[Optional[np.ndarray]] = []
    user_side_seq: List[str] = []
    user_metrics_seq: List[Dict[str, float]] = []
    user_ts_seq: List[int] = []

    print(f"[SCAN] {exercise}: MediaPipe 1? ?? ??")
    t0 = time.perf_counter()
    with create_landmarker(MODEL_PATH) as landmarker:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = apply_rotation(frame, user_rot)
            frame = resize_frame_if_needed(frame)
            timestamp_ms = int(frame_idx * 1000 / src_fps)

            raw_lms = user_selector.select(extract_all_landmarks(landmarker, frame, timestamp_ms))
            user_lms = smoother.update(raw_lms)
            user_side = side_lock.update(user_lms)
            user_m = compute_metrics(user_lms, exercise, user_side) if valid_lms(user_lms) else {}

            user_frames.append(frame)
            user_lms_seq.append(user_lms)
            user_side_seq.append(user_side)
            user_metrics_seq.append(user_m)
            user_ts_seq.append(timestamp_ms)

            frame_idx += 1
            if frame_idx % 50 == 0:
                print(f"  scan {frame_idx}/{source_total if source_total else '?'}")
    cap.release()

    user_total = len(user_frames)
    if user_total == 0:
        raise RuntimeError(f"user video produced no frames: {user_video}")

    (
        user_frames,
        user_lms_seq,
        user_side_seq,
        user_metrics_seq,
        user_ts_seq,
        _trim_first,
        _trim_last,
    ) = _trim_invalid_user_edges(user_frames, user_lms_seq, user_side_seq, user_metrics_seq, user_ts_seq)
    user_total = len(user_frames)

    pose_seq = pose_seq_from_landmarks(user_lms_seq)
    phase_result = phase_adapter.infer(pose_seq)
    count_phase = smooth_phase_like_training(phase_result.phase_per_frame, PHASE_COUNT_SMOOTH_WINDOW)
    model_rep_count, _model_count_transitions = count_phases_like_training(
        count_phase,
        PHASE_COUNT_MIN_UP_LEN,
    )
    model_phase = bridge_ready_gaps(
        debounce_phase(phase_result.phase_per_frame, PHASE_DEBOUNCE_MIN_LEN),
        PHASE_READY_BRIDGE_MAX_LEN,
    )
    model_action = normalize_exercise_name(phase_result.action)
    user_rep_rule = "down_up"
    user_reps = segment_reps(model_phase)
    if not user_reps and (model_action if EXERCISE_FROM_MODEL else exercise) == "deadlift":
        user_reps = segment_up_reps(model_phase, PHASE_COUNT_MIN_UP_LEN)
        user_rep_rule = "up_only"
    print(
        f"[MODEL] user action={model_action} conf={phase_result.action_conf:.3f}, "
        f"reps={len(user_reps)}, training_count={model_rep_count}, rule={user_rep_rule}"
    )
    if not user_reps:
        raise RuntimeError(
            "user phase has no usable reps. "
            + _phase_debug_summary(model_phase)
        )

    if EXERCISE_FROM_MODEL:
        if model_action not in VIDEO_CONFIG:
            raise KeyError(f"model action is not in VIDEO_CONFIG: {model_action}")
        if model_action != exercise:
            print(f"[MODEL] exercise override: {exercise} -> {model_action}")
            exercise = model_action
            config = VIDEO_CONFIG[exercise]
            user_metrics_seq = [
                compute_metrics(lms, exercise, side) if valid_lms(lms) else {}
                for lms, side in zip(user_lms_seq, user_side_seq)
            ]

    expert_frames, expert_fps, expert_phase_meta = build_expert_profile(exercise, config, phase_adapter)
    expert_rep_rule = str(expert_phase_meta.get("rep_rule", "down_up"))
    expert_ref_rep = _rep_from_meta(
        expert_phase_meta.get("reps") or [],
        expert_phase_meta.get("ref_rep_index"),
        allow_single_phase=(expert_rep_rule == "up_only"),
    )
    if expert_ref_rep is None:
        raise RuntimeError(
            f"failed to resolve expert reference rep: ref={expert_phase_meta.get('ref_rep_index')}, "
            f"reps={len(expert_phase_meta.get('reps') or [])}"
        )

    user_frame_meta = build_user_frame_meta(model_phase, user_reps)
    expert_idx_for_user = build_phase_alignment_map(
        user_frame_meta,
        user_reps,
        expert_ref_rep,
        align=PHASE_ALIGN,
    )
    print(f"[ALIGN] phase-segmented ratio enabled: expert_ref_rep={expert_ref_rep.index}, expert_rule={expert_rep_rule}")

    first_frame = user_frames[0]
    user_h, user_w = first_frame.shape[:2]
    # HUD? ?? ?? ?? ????? ??? ??? ????.
    # user ??? ???(?: ?? ?? 378px) HUD? ???? ?? ??? ??? ??.
    out_h = max(user_h, MIN_PANEL_H)
    out_w = user_w + EXPERT_W + HUD_W

    output_config = Path(config["output"])
    if output_config.is_absolute():
        output_path = output_config
    elif output_config.parent == Path("."):
        output_path = OUTPUT_DIR / output_config
    else:
        output_path = BASE_DIR / output_config
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, src_fps, (out_w, out_h))
    if not writer.isOpened():
        raise RuntimeError(f"VideoWriter failed to open: {output_path}")

    feedback_stabilizer = FeedbackStabilizer(min_frames=3, hold_frames=9)
    expert_norm = ExpertPaneNormalizer(
        EXPERT_W, out_h, pad=42, lock_frames=25,
        align_orientation=ALIGN_EXPERT_ORIENTATION.get(exercise, False),
    )

    # ?? ?? ??(??)? pass1?? ?? ?? user landmark ??? prefit??.
    if ALIGN_EXPERT_ORIENTATION.get(exercise, False):
        user_ref_angle = _median_body_axis_from_lms_seq(user_lms_seq, max_frames=40)
        expert_lms_all = [list_to_lms(f.get("landmarks")) for f in expert_frames]
        expert_norm.prefit(expert_lms_all, user_ref_angle)
        print(f"[ALIGN] {exercise}: user?={None if user_ref_angle is None else round(math.degrees(user_ref_angle),1)}?, "
              f"??={round(math.degrees(expert_norm._align_rot),1)}?")

    last_time = time.perf_counter()
    fps_now = 0.0

    print(f"[RUN] {exercise} ??")
    print(f"  user   : {user_video}")
    print(f"  output : {output_path}")
    print(f"  fps={src_fps:.2f}, frames={user_total}, size={user_w}x{user_h}")

    for frame_idx, frame in enumerate(user_frames):
        timestamp_ms = user_ts_seq[frame_idx]
        user_lms = user_lms_seq[frame_idx]
        user_side = user_side_seq[frame_idx]
        user_m = user_metrics_seq[frame_idx]

        if frame_idx >= len(expert_idx_for_user):
            raise RuntimeError(f"phase alignment map too short: frame={frame_idx}, map_len={len(expert_idx_for_user)}")
        aligned_idx = expert_idx_for_user[frame_idx]
        if aligned_idx is None:
            raise RuntimeError(f"phase alignment missing for frame: frame={frame_idx}, phase={model_phase_name(int(model_phase[frame_idx]))}")
        if not expert_frames:
            raise RuntimeError("expert frames are empty.")

        ex_idx = max(0, min(int(aligned_idx), len(expert_frames) - 1))
        ex_pack = expert_frames[ex_idx]
        expert_lms = list_to_lms(ex_pack.get("landmarks"))
        expert_side = ex_pack.get("side", user_side)
        expert_m = ex_pack.get("metrics", {})
        if valid_lms(expert_lms) and not expert_m:
            expert_m = compute_metrics(expert_lms, exercise, expert_side)
        if not user_m:
            raise RuntimeError(f"user metrics are empty: frame={frame_idx}")
        if not expert_m:
            raise RuntimeError(f"expert metrics are empty: frame={frame_idx}, expert_frame={ex_idx}")

        deltas = compute_deltas(user_m, expert_m)
        banner_phase = model_phase_name(int(model_phase[frame_idx]))
        phase_pos = banner_phase
        rep_count = num_reps_completed_until(user_reps, frame_idx)

        raw_issue = choose_issue(exercise, user_m, expert_m, deltas, phase_pos, rep_count=rep_count) if user_m else None
        issue = feedback_stabilizer.update(raw_issue)
        bad = issue.landmarks if issue else []

        # USER panel
        user_canvas = frame.copy()
        user_tr = make_user_transform(user_w, user_h)
        user_canvas = draw_text(user_canvas, "USER", (8, 8), 25, C_YELLOW, bold=True)
        user_canvas = draw_skeleton(user_canvas, user_lms, user_tr, bad_indices=bad, line_color=C_LINE, thickness=3)
        user_canvas = draw_exercise_layers(user_canvas, user_lms, expert_lms, exercise, user_side, user_m, issue, user_tr, is_expert=False)
        user_canvas = draw_feedback_banner(user_canvas, exercise, banner_phase, issue)

        # out_h? user_h?? ??(?? ??) ??? ????? ?? ??? ???.
        if user_canvas.shape[0] != out_h:
            pad_total = out_h - user_canvas.shape[0]
            pad_top = max(0, pad_total // 2)
            pad_bot = max(0, pad_total - pad_top)
            user_canvas = cv2.copyMakeBorder(
                user_canvas, pad_top, pad_bot, 0, 0,
                cv2.BORDER_CONSTANT, value=C_BG,
            )

        # EXPERT panel
        expert_canvas = np.zeros((out_h, EXPERT_W, 3), dtype=np.uint8)
        expert_canvas[:] = C_BG
        expert_canvas = draw_text(expert_canvas, "EXPERT", (14, 8), 25, C_OK, bold=True)
        expert_norm.observe_user(user_lms)
        ex_tr = expert_norm.update(expert_lms)
        expert_canvas = draw_skeleton(expert_canvas, expert_lms, ex_tr, bad_indices=bad, line_color=C_LINE, thickness=3)
        expert_canvas = draw_exercise_layers(expert_canvas, expert_lms, None, exercise, expert_side, expert_m, issue, ex_tr, is_expert=True)

        # HUD panel
        now = time.perf_counter()
        dt = now - last_time
        if dt > 1e-6:
            fps_now = 0.9 * fps_now + 0.1 * (1.0 / dt) if fps_now > 0 else (1.0 / dt)
        last_time = now
        frame_info = f"??? {frame_idx + 1}/{user_total if user_total else '?'}"
        hud = np.zeros((out_h, HUD_W, 3), dtype=np.uint8)
        hud = draw_hud(hud, exercise, banner_phase, rep_count, user_m, expert_m, deltas, issue, fps_now, frame_info)

        # vertical separators
        sep1 = np.full((out_h, 2, 3), (72, 60, 66), dtype=np.uint8)
        sep2 = np.full((out_h, 2, 3), (72, 60, 66), dtype=np.uint8)
        combined = np.hstack([user_canvas, sep1, expert_canvas[:, :EXPERT_W - 2], sep2, hud[:, :HUD_W - 2]])
        if combined.shape[1] != out_w:
            combined = cv2.resize(combined, (out_w, out_h), interpolation=cv2.INTER_AREA)
        writer.write(combined)

        if (frame_idx + 1) % 50 == 0:
            elapsed = time.perf_counter() - t0
            print(f"  {exercise}: {frame_idx + 1}/{user_total if user_total else '?'} frames | elapsed {elapsed:.1f}s")

    writer.release()
    elapsed = time.perf_counter() - t0
    size_mb = output_path.stat().st_size / 1024 / 1024 if output_path.exists() else 0
    print(f"[DONE] {exercise}: {output_path} | {size_mb:.1f} MB | {elapsed:.1f}s")


def _format_realtime_model_line(state) -> str:
    if state is None:
        return "model inactive"
    if not getattr(state, "available", False):
        return f"model unavailable: {getattr(state, 'unavailable_reason', '') or 'unknown'}"
    if not getattr(state, "ready", False):
        return f"model warming clip={getattr(state, 'clip_len', '?')}"
    return (
        f"model {state.pred_class} {state.action_confidence:.2f} | "
        f"phase {state.phase} {state.phase_confidence:.2f} | "
        f"seg {getattr(state, 'segment_len', 0)}"
    )


def run_realtime_feedback(
    source=None,
    initial_exercise=None,
    display: Optional[bool] = None,
    output=None,
    max_frames: Optional[int] = None,
    loop_video: Optional[bool] = None,
) -> dict:
    source = REALTIME_SOURCE if source is None else source
    display = REALTIME_DISPLAY if display is None else bool(display)
    output = REALTIME_OUTPUT if output is None else output
    max_frames = REALTIME_MAX_FRAMES if max_frames is None else max_frames
    loop_video = REALTIME_LOOP_VIDEO if loop_video is None else bool(loop_video)
    initial_value = REALTIME_INITIAL_EXERCISE if initial_exercise is None else initial_exercise

    video_config_source_exercise: Optional[str] = None
    if _is_video_config_realtime_source(source):
        source_path, video_config_source_exercise = _resolve_video_config_realtime_source(initial_value)
        source = str(source_path)

    active_exercise = _normalize_realtime_initial_exercise(initial_value)
    action_locked = bool(active_exercise and REALTIME_LOCK_EXERCISE)
    action_smoother = RealtimeActionSmoother(
        REALTIME_ACTION_SMOOTH_WINDOW,
        REALTIME_ACTION_CONF_MIN,
        REALTIME_ACTION_MIN_VOTES,
    )
    if active_exercise is not None:
        action_smoother.current = active_exercise

    realtime_infer = _create_realtime_infer()
    phase_adapter = LazyPhaseAdapter()
    cap, source_rot, source_label, src_fps, is_camera = _open_realtime_capture(source)
    source_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) if not is_camera else 0

    output_path = _resolve_realtime_output_path(output)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    smoother = LandmarkSmoother(LANDMARK_EMA_ALPHA, VIS_THR)
    selector_exercise = active_exercise or "squat"
    side_lock = SideLock(selector_exercise)
    user_selector = PersonSelector(selector_exercise)
    feedback_stabilizer = FeedbackStabilizer(min_frames=3, hold_frames=9)
    expert_cache: Dict[str, RealtimeExpertProfile] = {}
    expert_norm_cache: Dict[Tuple[str, int], ExpertPaneNormalizer] = {}

    writer = None
    frame_idx = 0
    written_frames = 0
    fps_now = 0.0
    last_time = time.perf_counter()
    t0 = time.perf_counter()
    stop_reason = "unknown"

    print("[RUN] realtime unified feedback")
    print(f"  source : {source_label}")
    if video_config_source_exercise:
        print(f"  source config: VIDEO_CONFIG[{video_config_source_exercise}]['user_video']")
    print(f"  mode   : {'display' if display else 'headless'}")
    if source_total_frames:
        print(f"  source frames: {source_total_frames} @ {src_fps:.2f} fps")
    if output_path is not None:
        print(f"  output : {output_path}")
    if active_exercise:
        print(f"  initial exercise: {active_exercise} ({'locked' if action_locked else 'model may update'})")
    else:
        print("  initial exercise: model auto")

    try:
        with create_landmarker(MODEL_PATH) as landmarker:
            while True:
                if max_frames is not None and frame_idx >= int(max_frames):
                    stop_reason = f"max_frames={int(max_frames)}"
                    break

                ret, frame = cap.read()
                if not ret:
                    if loop_video and not is_camera:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    stop_reason = "eof" if not is_camera else "capture_read_failed"
                    break

                frame = apply_rotation(frame, source_rot)
                frame = resize_frame_if_needed(frame)
                timestamp_ms = int(frame_idx * 1000.0 / max(src_fps, 1.0))

                raw_lms = user_selector.select(extract_all_landmarks(landmarker, frame, timestamp_ms))
                raw_pose_detected = _has_raw_pose_detection(raw_lms)
                if hasattr(realtime_infer, "set_count_exercise"):
                    realtime_infer.set_count_exercise(active_exercise)
                user_lms = smoother.update(raw_lms) if raw_pose_detected else None
                if raw_pose_detected and valid_lms(user_lms):
                    state = realtime_infer.update_kpts(_lms_to_realtime_kpts(user_lms))
                else:
                    smoother.reset()
                    state = realtime_infer.clear_frame()

                now = time.perf_counter()
                dt = now - last_time
                if dt > 1e-6:
                    fps_now = 0.9 * fps_now + 0.1 * (1.0 / dt) if fps_now > 0 else (1.0 / dt)
                last_time = now

                exercise_changed = False
                if getattr(state, "ready", False) and not action_locked:
                    selected = action_smoother.update(state.pred_class, state.action_confidence)
                    if selected and selected != active_exercise:
                        active_exercise = selected
                        exercise_changed = True
                        if REALTIME_LOCK_EXERCISE:
                            action_locked = True
                            print(f"[MODEL] realtime exercise locked: {active_exercise}")
                        else:
                            print(f"[MODEL] realtime exercise update: {active_exercise}")
                        if hasattr(realtime_infer, "set_count_exercise"):
                            realtime_infer.set_count_exercise(active_exercise)

                if exercise_changed and active_exercise:
                    side_lock = SideLock(active_exercise)
                    user_selector = PersonSelector(active_exercise)
                    feedback_stabilizer = FeedbackStabilizer(min_frames=3, hold_frames=9)

                model_line = _format_realtime_model_line(state)
                status_lines = [
                    model_line,
                    f"exercise: {active_exercise or 'waiting model action'}",
                    f"lock: {'on' if action_locked else 'off'}",
                ]
                if not raw_pose_detected:
                    status_lines.append("pose: not detected")

                if active_exercise is None or not getattr(state, "ready", False) or not raw_pose_detected or not valid_lms(user_lms):
                    combined = _draw_realtime_status_frame(frame, user_lms, status_lines, fps_now)
                else:
                    user_side = side_lock.update(user_lms)
                    user_m = compute_metrics(user_lms, active_exercise, user_side)
                    if not user_m:
                        combined = _draw_realtime_status_frame(
                            frame,
                            user_lms,
                            status_lines + ["metrics unavailable on this frame"],
                            fps_now,
                        )
                    else:
                        expert_profile = _get_realtime_expert_profile(active_exercise, phase_adapter, expert_cache)
                        ex_idx = align_realtime_phase_to_expert(
                            int(state.phase_id),
                            max(1, int(getattr(state, "segment_len", 1))),
                            expert_profile.ref_rep,
                        )
                        ex_idx = max(0, min(int(ex_idx), len(expert_profile.frames) - 1))
                        ex_pack = expert_profile.frames[ex_idx]
                        expert_lms = list_to_lms(ex_pack.get("landmarks"))
                        expert_side = ex_pack.get("side", user_side)
                        expert_m = ex_pack.get("metrics", {})
                        if valid_lms(expert_lms) and not expert_m:
                            expert_m = compute_metrics(expert_lms, active_exercise, expert_side)
                        if not valid_lms(expert_lms):
                            raise RuntimeError(f"expert landmarks are empty: exercise={active_exercise}, frame={ex_idx}")
                        if not expert_m:
                            raise RuntimeError(f"expert metrics are empty: exercise={active_exercise}, frame={ex_idx}")

                        deltas = compute_deltas(user_m, expert_m)
                        banner_phase = str(state.phase)
                        raw_issue = choose_issue(
                            active_exercise,
                            user_m,
                            expert_m,
                            deltas,
                            banner_phase,
                            rep_count=int(state.count),
                        )
                        issue = feedback_stabilizer.update(raw_issue)

                        out_h = max(frame.shape[0], MIN_PANEL_H)
                        norm_key = (active_exercise, out_h)
                        expert_norm = expert_norm_cache.get(norm_key)
                        if expert_norm is None:
                            expert_norm = ExpertPaneNormalizer(
                                EXPERT_W,
                                out_h,
                                pad=42,
                                lock_frames=25,
                                align_orientation=ALIGN_EXPERT_ORIENTATION.get(active_exercise, False),
                            )
                            expert_norm_cache[norm_key] = expert_norm

                        combined = _compose_realtime_feedback_frame(
                            frame=frame,
                            exercise=active_exercise,
                            user_lms=user_lms,
                            user_side=user_side,
                            user_m=user_m,
                            expert_lms=expert_lms,
                            expert_side=expert_side,
                            expert_m=expert_m,
                            banner_phase=banner_phase,
                            rep_count=int(state.count),
                            issue=issue,
                            fps_now=fps_now,
                            frame_info=f"LIVE {frame_idx + 1}",
                            expert_norm=expert_norm,
                            model_line=model_line,
                        )

                if output_path is not None:
                    if writer is None:
                        out_h, out_w = combined.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = cv2.VideoWriter(str(output_path), fourcc, src_fps, (out_w, out_h))
                        if not writer.isOpened():
                            raise RuntimeError(f"VideoWriter failed to open: {output_path}")
                    writer.write(combined)
                    written_frames += 1

                if display:
                    cv2.imshow(REALTIME_WINDOW_NAME, combined)
                    wait_ms = 1 if is_camera else max(1, int(1000.0 / max(src_fps, 1.0)))
                    key = cv2.waitKey(wait_ms) & 0xFF
                    if key in (ord("q"), 27):
                        stop_reason = f"user_key={key}"
                        break
                    if key == ord("r"):
                        realtime_infer.reset()
                        smoother.reset()
                        action_smoother.reset()
                        active_exercise = _normalize_realtime_initial_exercise(initial_value)
                        action_locked = bool(active_exercise and REALTIME_LOCK_EXERCISE)
                        if active_exercise is not None:
                            action_smoother.current = active_exercise
                        selector_exercise = active_exercise or "squat"
                        side_lock = SideLock(selector_exercise)
                        user_selector = PersonSelector(selector_exercise)
                        feedback_stabilizer = FeedbackStabilizer(min_frames=3, hold_frames=9)
                        expert_norm_cache.clear()
                        print("[RUN] realtime state reset")

                frame_idx += 1
                if frame_idx % 100 == 0:
                    print(f"  realtime: {frame_idx} frames | exercise={active_exercise or 'auto'} | {model_line}")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if display:
            try:
                cv2.destroyWindow(REALTIME_WINDOW_NAME)
            except Exception:
                pass

    elapsed = time.perf_counter() - t0
    if output_path is not None and output_path.exists():
        size_mb = output_path.stat().st_size / 1024 / 1024
        expected = f"/{source_total_frames}" if source_total_frames else ""
        print(f"[DONE] realtime: {output_path} | {written_frames}{expected} frames | stop={stop_reason} | {size_mb:.1f} MB | {elapsed:.1f}s")
    else:
        expected = f"/{source_total_frames}" if source_total_frames else ""
        print(f"[DONE] realtime: {frame_idx}{expected} frames | stop={stop_reason} | {elapsed:.1f}s")
    return {
        "frames": int(frame_idx),
        "written_frames": int(written_frames),
        "source_total_frames": int(source_total_frames),
        "stop_reason": stop_reason,
        "source": source_label,
        "last_exercise": active_exercise,
        "output": None if output_path is None else str(output_path),
    }


# ============================================================
# 9. ???? ??
# ============================================================


def _get_run_targets() -> List[str]:
    if RUN_ALL_EXERCISES:
        return list(VIDEO_CONFIG.keys())

    if isinstance(RUN_EXERCISES, str):
        raw_targets = [RUN_EXERCISES]
    else:
        raw_targets = list(RUN_EXERCISES)

    targets = [normalize_exercise_name(ex) for ex in raw_targets]
    missing = [ex for ex in targets if ex not in VIDEO_CONFIG]
    if missing:
        available = ", ".join(VIDEO_CONFIG.keys()) or "(??)"
        raise KeyError(
            f"VIDEO_CONFIG? ?? ?????: {missing}. "
            f"RUN_EXERCISES ?? VIDEO_CONFIG? ????. ?? ??: {available}"
        )
    return targets


def main():
    mode = str(RUN_MODE).strip().lower()
    if mode == "realtime":
        run_realtime_feedback()
        return
    if mode != "offline":
        raise ValueError(f"RUN_MODE must be 'realtime' or 'offline', got {RUN_MODE!r}")

    targets = _get_run_targets()

    for ex in targets:
        try:
            process_exercise(ex)
        except Exception as e:
            print(f"[ERROR] {ex}: {e}")
            raise


if __name__ == "__main__":
    main()
