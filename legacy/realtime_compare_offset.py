"""
realtime_compare.py — 실시간 전문가 비교 + LLM 피드백
=========================================================
사용법:
  1. 아래 [설정] 섹션 변수를 직접 수정
  2. python realtime_compare.py 실행

설치:
  pip install mediapipe opencv-python numpy pillow anthropic

키 조작:
  R - 횟수 리셋    Q / ESC - 종료

전문가 스켈레톤 렌더링 방식:
  - 앵커: 사용자 힙 센터 (23,24 중점) → 카메라 위치 무관
  - 스케일: 사용자 몸통 높이 (어깨~힙) → 체형 무관
  - 각도: 전문가 JSON의 각도값으로 뼈대 역산 → 카메라 각도 무관
  - 가시성: 핵심 관절 6개 visibility 0.6 이상일 때만 표시

횟수 카운트:
  - 사용자 핵심 각도 임계값 기반 (전문가 루프 아님)
  - 스쿼트: 무릎각도 < DOWN_THRESHOLD 진입 후 UP_THRESHOLD 복귀
  - 푸시업: 팔꿈치각도 < DOWN_THRESHOLD 진입 후 UP_THRESHOLD 복귀
"""

import cv2
import numpy as np
import math
import json
import time
import os
import sys
import types
import threading
import urllib.request
from collections import deque


# ══════════════════════════════════════════════════════════════
#  [설정] — 여기만 수정하면 됩니다
# ══════════════════════════════════════════════════════════════

# 운동 종류: "squat" 또는 "pushup"
EXERCISE = "squat"

# Anthropic API 키 (없으면 규칙 기반 fallback)
ANTHROPIC_API_KEY = None           # 예: "sk-ant-api03-..."

# 전문가 JSON 경로 (None → 같은 폴더 자동 탐색)
EXPERT_JSON_PATH = r'expert_dead2.json'

# 모델 파일 경로 (None → 자동 탐색 + 없으면 다운로드)
MODEL_PATH = None

# 웹캠 번호 / 해상도
CAMERA_INDEX  = 0
CAMERA_WIDTH  = 1280
CAMERA_HEIGHT = 720

# 윈도우 평균 프레임 수
WINDOW_SIZE = 5

# LLM 피드백 쿨타임 (초)
LLM_COOLDOWN = 3.0

# 전문가 스켈레톤 투명도 (0.0=완전투명 ~ 1.0=불투명)
EXPERT_ALPHA = 0.7

# 전문가 스켈레톤 x축 오프셋 (화면 너비 대비 비율)
# 0.0 = 오버레이, 0.15~0.20 = 사용자 옆에 나란히
EXPERT_OFFSET_X = 0.18

# 가시성 임계값 — 이 값 미만 관절이 핵심 관절에 있으면 전문가 스켈레톤 숨김
VISIBILITY_THRESHOLD = 0.6

# ── 횟수 카운트 임계값 ────────────────────────────────────────
# 스쿼트: 무릎각도가 DOWN 이하로 내려갔다가 UP 이상으로 올라오면 1회
# 푸시업: 팔꿈치각도가 DOWN 이하로 내려갔다가 UP 이상으로 올라오면 1회
COUNT_THRESHOLDS = {
    "squat":  {"down": 110.0, "up": 150.0},  # 무릎 각도 (도)
    "pushup": {"down":  95.0, "up": 140.0},  # 팔꿈치 각도 (도)
}

# ── 피드백 임계값 ─────────────────────────────────────────────
THRESHOLDS = {
    "knee_angle":  12.0,
    "hip_angle":   10.0,
    "ankle_angle":  8.0,
    "elbow_angle": 12.0,
    "body_angle":  10.0,
    "spine_lean":  10.0,
    "foot_width":   0.15,
    "grip_width":   0.15,
    "knee_align":   0.10,
}

# 운동별 비교 지표
EXERCISE_METRICS = {
    "squat":  ["knee_angle", "hip_angle", "ankle_angle",
               "spine_lean", "foot_width", "knee_align"],
    "pushup": ["elbow_angle", "body_angle", "spine_lean",
               "grip_width"],
}

# ══════════════════════════════════════════════════════════════
#  이하 수정 불필요
# ══════════════════════════════════════════════════════════════

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/latest/"
    "pose_landmarker_lite.task"
)
_DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "pose_landmarker_lite.task")

METRIC_LABELS = {
    "knee_angle":  "무릎 각도",
    "hip_angle":   "엉덩이 각도",
    "ankle_angle": "발목 각도",
    "elbow_angle": "팔꿈치 각도",
    "body_angle":  "몸통 각도",
    "spine_lean":  "척추 기울기",
    "foot_width":  "발 넓이 비율",
    "grip_width":  "손 넓이 비율",
    "knee_align":  "무릎-발끝 정렬",
}

METRIC_TO_LM = {
    "knee_angle":  [25, 26, 27, 28],
    "hip_angle":   [23, 24, 25, 26],
    "ankle_angle": [27, 28, 31, 32],
    "elbow_angle": [13, 14, 15, 16],
    "body_angle":  [11, 12, 23, 24],
    "spine_lean":  [11, 12, 23, 24],
    "foot_width":  [27, 28, 29, 30],
    "grip_width":  [15, 16],
    "knee_align":  [25, 26, 31, 32],
}

# 핵심 관절: 이 중 하나라도 visibility 미달이면 전문가 스켈레톤 숨김
_KEY_JOINTS = [11, 12, ] # 23, 24, 25, 26, 27, 28


# ── mediapipe import (버전/OS 독립) ──────────────────────────
for _n in ["mediapipe.tasks.python.genai",
           "mediapipe.tasks.python.genai.bundler"]:
    sys.modules.setdefault(_n, types.ModuleType(_n))

from mediapipe.tasks.python.vision import pose_landmarker as _pl
from mediapipe.tasks.python.vision.core import vision_task_running_mode as _vtm
from mediapipe.tasks.python.core import base_options as _bo

def _find_image_classes():
    try:
        from mediapipe.tasks.python.vision.core import image as m
        return m.Image, m.ImageFormat
    except (ImportError, AttributeError):
        pass
    try:
        from mediapipe.python._framework_bindings import image       as _img
        from mediapipe.python._framework_bindings import image_frame as _imgf
        return _img.Image, _imgf.ImageFormat
    except (ImportError, AttributeError):
        pass
    raise ImportError("mediapipe Image 클래스를 찾을 수 없습니다.")

_MpImage, _ImageFormat = _find_image_classes()

PoseLandmarker        = _pl.PoseLandmarker
PoseLandmarkerOptions = _pl.PoseLandmarkerOptions
PoseLandmarksConns    = _pl.PoseLandmarksConnections.POSE_LANDMARKS
BaseOptions           = _bo.BaseOptions
MpImage               = _MpImage
ImageFormat           = _ImageFormat
RunningMode           = _vtm.VisionTaskRunningMode


# ══════════════════════════════════════════════════════════════
#  유틸
# ══════════════════════════════════════════════════════════════

def ensure_model(model_path: str):
    if os.path.exists(model_path):
        print(f"[모델] {os.path.basename(model_path)} 확인 완료")
        return
    print("[모델] 다운로드 중... (~7MB)")
    def _p(n, bs, total):
        if total > 0:
            print(f"\r  {min(n*bs*100//total, 100)}%", end="", flush=True)
    urllib.request.urlretrieve(_MODEL_URL, model_path, _p)
    print(f"\n[모델] 저장 완료 → {model_path}")

def load_expert(exercise: str, json_path: str) -> dict:
    if not os.path.exists(json_path):
        print(f"[오류] {json_path} 없음. expert_preprocess.py 를 먼저 실행하세요.")
        sys.exit(1)
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    print(f"[전문가] {os.path.basename(json_path)} 로드 완료"
          f"  ({data['total_frames']}프레임, {data['fps']}fps)")
    return data


# ══════════════════════════════════════════════════════════════
#  STEP 1 — 관절 추출
# ══════════════════════════════════════════════════════════════

def build_landmarker(model_path: str):
    opts = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.4,
        min_pose_presence_confidence=0.4,
        min_tracking_confidence=0.4,
    )
    return PoseLandmarker.create_from_options(opts)

def extract_landmarks(landmarker, bgr: np.ndarray, ts_ms: int):
    rgb    = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mp_img = MpImage(image_format=ImageFormat.SRGB, data=rgb)
    result = landmarker.detect_for_video(mp_img, ts_ms)
    return result.pose_landmarks[0] if result.pose_landmarks else None


# ══════════════════════════════════════════════════════════════
#  STEP 2 — 지표 계산
# ══════════════════════════════════════════════════════════════

def G(lms, i):
    p = lms[i]; return (p.x, p.y, p.z)

def calc_angle(a, b, c) -> float:
    """B 꼭짓점 A-B-C 각도(도)"""
    ba = np.array([a[0]-b[0], a[1]-b[1], a[2]-b[2]])
    bc = np.array([c[0]-b[0], c[1]-b[1], c[2]-b[2]])
    n  = np.linalg.norm(ba) * np.linalg.norm(bc)
    if n == 0: return 0.0
    return math.degrees(math.acos(np.clip(np.dot(ba, bc) / n, -1.0, 1.0)))

def compute_metrics(lms) -> dict:
    l_sh,   r_sh   = G(lms, 11), G(lms, 12)
    l_elbow,r_elbow= G(lms, 13), G(lms, 14)
    l_wrist,r_wrist= G(lms, 15), G(lms, 16)
    l_hip,  r_hip  = G(lms, 23), G(lms, 24)
    l_knee, r_knee = G(lms, 25), G(lms, 26)
    l_ankle,r_ankle= G(lms, 27), G(lms, 28)
    l_foot, r_foot = G(lms, 31), G(lms, 32)

    shoulder_w = max(abs(l_sh[0] - r_sh[0]), 1e-6)
    sx = (l_sh[0]+r_sh[0])/2 - (l_hip[0]+r_hip[0])/2
    sy = (l_sh[1]+r_sh[1])/2 - (l_hip[1]+r_hip[1])/2
    spine_lean = math.degrees(math.atan2(abs(sx), abs(sy)+1e-9))

    return {
        "knee_angle":  (calc_angle(l_hip,   l_knee,  l_ankle) +
                        calc_angle(r_hip,   r_knee,  r_ankle)) / 2,
        "hip_angle":   (calc_angle(l_knee,  l_hip,   l_sh)    +
                        calc_angle(r_knee,  r_hip,   r_sh))   / 2,
        "ankle_angle": (calc_angle(l_knee,  l_ankle, l_foot)  +
                        calc_angle(r_knee,  r_ankle, r_foot))  / 2,
        "elbow_angle": (calc_angle(l_wrist, l_elbow, l_sh)    +
                        calc_angle(r_wrist, r_elbow, r_sh))   / 2,
        "body_angle":  (calc_angle(l_sh,    l_hip,   l_ankle) +
                        calc_angle(r_sh,    r_hip,   r_ankle)) / 2,
        "spine_lean":  spine_lean,
        "foot_width":  abs(l_ankle[0]-r_ankle[0]) / shoulder_w,
        "grip_width":  abs(l_wrist[0]-r_wrist[0]) / shoulder_w,
        "knee_align":  ((l_knee[0]-l_foot[0]) +
                        (r_knee[0]-r_foot[0])) / 2 / shoulder_w,
    }


# ══════════════════════════════════════════════════════════════
#  STEP 3 — 윈도우 평균
# ══════════════════════════════════════════════════════════════

class MetricsBuffer:
    def __init__(self, window: int = 5):
        self._buf = {}
        self._win = window

    def push(self, metrics: dict):
        for k, v in metrics.items():
            if k not in self._buf:
                self._buf[k] = deque(maxlen=self._win)
            self._buf[k].append(v)

    def get_avg(self) -> dict:
        return {k: float(np.mean(list(v)))
                for k, v in self._buf.items() if v}


# ══════════════════════════════════════════════════════════════
#  STEP 3.5 — 횟수 카운터 (사용자 각도 기반)
# ══════════════════════════════════════════════════════════════

class RepCounter:
    """
    핵심 각도가 DOWN 임계값 이하로 떨어졌다가
    UP 임계값 이상으로 복귀하면 1회 카운트.

    스쿼트: 무릎각도 down=110 → up=150
    푸시업: 팔꿈치각도 down=95 → up=140
    """
    def __init__(self, exercise: str):
        cfg          = COUNT_THRESHOLDS[exercise]
        self._down   = cfg["down"]
        self._up     = cfg["up"]
        self._angle_key = "knee_angle" if exercise == "squat" else "elbow_angle"
        self._in_down   = False    # 현재 down 구간 진입 여부
        self.count      = 0

    def update(self, metrics: dict) -> bool:
        """1회 증가 시 True 반환"""
        angle = metrics.get(self._angle_key, 180.0)

        if not self._in_down and angle < self._down:
            # 서있다가 내려감 → down 진입
            self._in_down = True

        elif self._in_down and angle > self._up:
            # down 구간에 있다가 올라옴 → 1회 완료
            self._in_down = False
            self.count   += 1
            return True

        return False


# ══════════════════════════════════════════════════════════════
#  STEP 4 — 전문가 비교
# ══════════════════════════════════════════════════════════════

def get_expert_frame_idx(elapsed_ms: float, expert: dict) -> int:
    fps = expert["fps"]
    n   = expert["total_frames"]
    return int((elapsed_ms / 1000.0) * fps) % n

def compute_deltas(user_metrics: dict, expert_metrics: dict,
                   exercise: str) -> dict:
    return {
        key: user_metrics.get(key, 0.0) - expert_metrics.get(key, 0.0)
        for key in EXERCISE_METRICS[exercise]
    }

def select_worst(deltas: dict):
    worst_key, worst_ratio = None, 0.0
    for key, delta in deltas.items():
        ratio = abs(delta) / THRESHOLDS.get(key, 1.0)
        if ratio > worst_ratio:
            worst_ratio = ratio
            worst_key   = key
    return (worst_key, deltas[worst_key]) if worst_ratio >= 1.0 else None


# ══════════════════════════════════════════════════════════════
#  전문가 스켈레톤 — 좌표 기반 오버레이
#  ─────────────────────────────────────────────────────────────
#  전문가 JSON의 정규화 좌표(0~1)를 그대로 픽셀로 변환해서 그림.
#  가시성 조건: 핵심 관절(_KEY_JOINTS) 모두 VISIBILITY_THRESHOLD 이상
# ══════════════════════════════════════════════════════════════

def check_visibility(lms) -> bool:
    """핵심 관절이 모두 VISIBILITY_THRESHOLD 이상인지 확인"""
    return all(lms[i].visibility >= VISIBILITY_THRESHOLD
               for i in _KEY_JOINTS)

# ── 스케일/앵커 누적 평균 (전역) ─────────────────────────────
_scale_sum    = 0.0
_scale_count  = 0
_anchor_sum_x = 0.0
_anchor_sum_y = 0.0
_anchor_count = 0


def normalize_expert_landmarks(ex_lms_raw: list, lms) -> list:
    """
    전문가 좌표를 사용자 기준으로 정규화.

    앵커: 사용자 발 중점 (31, 32번) — 누적 평균으로 흔들림 제거
    스케일: 발~어깨 전신 길이 기준 — 누적 평균으로 안정화
    """
    global _scale_sum, _scale_count
    global _anchor_sum_x, _anchor_sum_y, _anchor_count

    # ── 사용자 발 앵커 (현재 프레임) ──
    u_foot_x = (lms[31].x + lms[32].x) / 2
    u_foot_y = (lms[31].y + lms[32].y) / 2

    # ── 스케일 계산 ──
    u_sh_y = (lms[11].y + lms[12].y) / 2
    u_body = max(abs(u_sh_y - u_foot_y), 1e-6)

    e_sh_y   = (ex_lms_raw[11][1] + ex_lms_raw[12][1]) / 2
    e_foot_x = (ex_lms_raw[31][0] + ex_lms_raw[32][0]) / 2
    e_foot_y = (ex_lms_raw[31][1] + ex_lms_raw[32][1]) / 2
    e_body   = max(abs(e_sh_y - e_foot_y), 1e-6)

    # ── 누적 평균 업데이트 ──
    _scale_sum    += u_body / e_body
    _scale_count  += 1
    _anchor_sum_x += u_foot_x
    _anchor_sum_y += u_foot_y
    _anchor_count += 1

    stable_scale  = _scale_sum   / _scale_count
    stable_anc_x  = _anchor_sum_x / _anchor_count
    stable_anc_y  = _anchor_sum_y / _anchor_count

    # ── 변환 적용 ──
    result = []
    for p in ex_lms_raw:
        nx = (p[0] - e_foot_x) * stable_scale + stable_anc_x
        ny = (p[1] - e_foot_y) * stable_scale + stable_anc_y
        result.append([nx, ny, p[2], p[3]])
    return result


def draw_expert_skeleton(frame, ex_lms_raw: list, lms,
                         bad_indices: set, alpha: float = 0.55,
                         offset_x: float = 0.0):
    """
    전문가 스켈레톤 렌더링.
    - offset_x > 0 이면 사용자 옆에 나란히 표시
    - 단색 (민트) 으로 그림
    """
    H, W    = frame.shape[:2]
    overlay = frame.copy()

    ex_lms = normalize_expert_landmarks(ex_lms_raw, lms)

    LINE_COLOR = (180, 220, 60)
    DOT_COLOR  = (150, 200, 40)

    for conn in PoseLandmarksConns:
        if ex_lms_raw[conn.start][3] < 0.4 or ex_lms_raw[conn.end][3] < 0.4:
            continue
        a, b = ex_lms[conn.start], ex_lms[conn.end]
        ax = int((a[0] + offset_x) * W)
        ay = int(a[1] * H)
        bx = int((b[0] + offset_x) * W)
        by = int(b[1] * H)
        if not (0 <= ax < W and 0 <= bx < W):
            continue
        cv2.line(overlay, (ax, ay), (bx, by), LINE_COLOR, 2, cv2.LINE_AA)

    for i, p in enumerate(ex_lms):
        if ex_lms_raw[i][3] < 0.4:
            continue
        cx = int((p[0] + offset_x) * W)
        cy = int(p[1] * H)
        if not (0 <= cx < W):
            continue
        cv2.circle(overlay, (cx, cy), 5, DOT_COLOR, -1, cv2.LINE_AA)
        cv2.circle(overlay, (cx, cy), 5, (0, 0, 0),  1, cv2.LINE_AA)

    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def draw_user_skeleton(frame, lms, bad_indices: set):
    """사용자 스켈레톤 (틀린 관절 빨간색)"""
    H, W = frame.shape[:2]
    for conn in PoseLandmarksConns:
        a, b = lms[conn.start], lms[conn.end]
        if a.visibility < 0.4 or b.visibility < 0.4:
            continue
        bad   = conn.start in bad_indices or conn.end in bad_indices
        color = (0, 0, 255) if bad else (255, 255, 255)   # bad=빨간, 정상=흰색
        cv2.line(frame,
                 (int(a.x*W), int(a.y*H)),
                 (int(b.x*W), int(b.y*H)),
                 color, 2, cv2.LINE_AA)
    for i, lm in enumerate(lms):
        if lm.visibility < 0.4:
            continue
        cx, cy = int(lm.x*W), int(lm.y*H)
        color = (0, 0, 255) if i in bad_indices else (200, 200, 200)  # bad=빨간, 정상=회색
        cv2.circle(frame, (cx, cy), 5, color,   -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 5, (0,0,0),  1, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════
#  LLM 피드백
# ══════════════════════════════════════════════════════════════

class LLMFeedback:
    def __init__(self, api_key, cooldown: float = 3.0):
        self._key      = api_key
        self._cooldown = cooldown
        self.text      = ""
        self._loading  = False
        self._last_ts  = 0.0

    @property
    def is_loading(self): return self._loading

    def request(self, exercise, worst_key, user_val, expert_val):
        now = time.perf_counter()
        if self._loading or (now - self._last_ts) < self._cooldown:
            return
        self._last_ts = now
        self._loading = True
        threading.Thread(
            target=self._call,
            args=(exercise, worst_key, user_val, expert_val),
            daemon=True
        ).start()

    def _call(self, exercise, worst_key, user_val, expert_val):
        try:
            label    = METRIC_LABELS.get(worst_key, worst_key)
            delta    = user_val - expert_val
            is_angle = "angle" in worst_key or "lean" in worst_key
            unit     = "도" if is_angle else "(비율)"
            prompt   = (
                f"운동: {exercise}\n"
                f"문제 부위: {label}\n"
                f"사용자 값: {user_val:.1f}{unit}, "
                f"전문가 값: {expert_val:.1f}{unit}, "
                f"차이: {delta:+.1f}{unit}\n"
                f"위 정보를 바탕으로 사용자에게 자세 교정 피드백을 "
                f"한 문장(20자 이내)으로만 말해주세요. 부드럽고 명확하게."
            )
            if self._key:
                import anthropic
                client = anthropic.Anthropic(api_key=self._key)
                msg    = client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=80,
                    messages=[{"role": "user", "content": prompt}]
                )
                self.text = msg.content[0].text.strip()
            else:
                self.text = self._fallback(worst_key, delta, is_angle)
        except Exception as e:
            self.text = f"[피드백 오류: {e}]"
        finally:
            self._loading = False

    @staticmethod
    def _fallback(key, delta, is_angle):
        label = METRIC_LABELS.get(key, key)
        if is_angle:
            return f"{label}을 {'더 굽혀주세요' if delta < 0 else '더 펴주세요'}"
        return f"{label}을 {'더 넓게' if delta < 0 else '더 좁게'} 해주세요"


# ══════════════════════════════════════════════════════════════
#  HUD 렌더링
# ══════════════════════════════════════════════════════════════

try:
    from PIL import Image as PilImg, ImageDraw, ImageFont as PilFont
    _FONTS = [
        "C:/Windows/Fonts/malgun.ttf",
        "C:/Windows/Fonts/gulim.ttc",
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    _fp = next((p for p in _FONTS if os.path.exists(p)), None)
    def _f(sz): return PilFont.truetype(_fp, sz) if _fp else PilFont.load_default()
    FSM, FMD = _f(16), _f(20)
    PIL_OK = True
except ImportError:
    PIL_OK = False

def put_kr(frame, text, xy, fnt, rgb):
    if not PIL_OK:
        cv2.putText(frame, text, xy, cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (int(rgb[2]),int(rgb[1]),int(rgb[0])), 1, cv2.LINE_AA)
        return frame
    pil = PilImg.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(pil).text(xy, text, font=fnt, fill=tuple(rgb))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

def render_hud(frame, exercise, count, worst, deltas,
               user_metrics, expert_metrics,
               llm: LLMFeedback, fps, ex_idx, ex_total,
               skeleton_visible: bool):
    H, W = frame.shape[:2]
    PW   = 270
    px   = W - PW + 12

    cv2.rectangle(frame, (W-PW, 0), (W, H), (18, 18, 30), -1)
    cv2.line(frame, (W-PW, 0), (W-PW, H), (55, 55, 75), 1)

    # 운동 종류
    frame = put_kr(frame,
                   "SQUAT  스쿼트" if exercise=="squat" else "PUSH-UP  푸시업",
                   (px, 14), FSM if PIL_OK else None, (255, 200, 50))

    # 전문가 진행도 바
    prog  = ex_idx / max(ex_total-1, 1)
    bar_w = PW - 24
    cv2.rectangle(frame, (px-4, 44), (px-4+bar_w, 52), (50, 50, 70), -1)
    cv2.rectangle(frame, (px-4, 44), (px-4+int(bar_w*prog), 52), (0, 200, 120), -1)
    frame = put_kr(frame, f"전문가 {ex_idx+1}/{ex_total}",
                   (px, 56), FSM if PIL_OK else None, (120, 120, 140))

    # 전문가 스켈레톤 가시성 상태
    vis_color = (0, 200, 100) if skeleton_visible else (100, 100, 180)
    vis_text  = "전문가 오버레이 ON" if skeleton_visible else "전신을 보여주세요"
    frame = put_kr(frame, vis_text, (px, 68),
                   FSM if PIL_OK else None, vis_color)

    # 횟수
    cv2.putText(frame, str(count), (px+45, 118),
                cv2.FONT_HERSHEY_DUPLEX, 3.0, (255,255,255), 4, cv2.LINE_AA)
    frame = put_kr(frame, "횟수", (px+60, 126),
                   FSM if PIL_OK else None, (140,140,140))

    # 자세 상태
    ok    = worst is None
    badge = (0, 200, 70) if ok else (30, 30, 220)
    cv2.rectangle(frame, (px-4, 148), (W-10, 178), badge, -1)
    status = "정자세" if ok else f"교정: {METRIC_LABELS.get(worst,'?')}"
    frame  = put_kr(frame, status, (px+8, 154),
                    FMD if PIL_OK else None, (255,255,255))

    # 지표 비교
    y = 192
    frame = put_kr(frame, "── 지표 비교 ──", (px, y),
                   FSM if PIL_OK else None, (90,120,160))
    y += 26
    for key in EXERCISE_METRICS[exercise]:
        uv   = user_metrics.get(key, 0.0)
        ev   = expert_metrics.get(key, 0.0)
        dv   = deltas.get(key, 0.0)
        over = abs(dv) >= THRESHOLDS.get(key, 1.0)
        col  = (100,100,255) if over else (180,180,180)
        is_a = "angle" in key or "lean" in key
        u    = "d" if is_a else "r"
        frame = put_kr(frame, f"{METRIC_LABELS.get(key,key)[:5]}: {uv:.1f}{u} ({dv:+.1f})",
                       (px, y), FSM if PIL_OK else None, col)
        y += 24
        if y > H - 120:
            break

    frame = put_kr(frame, "R:리셋  Q:종료",
                   (px, H-28), FSM if PIL_OK else None, (80,80,100))
    cv2.putText(frame, f"FPS {fps:.0f}", (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80,80,80), 1)

    # LLM 피드백 자막
    feedback = llm.text if not llm.is_loading else "분석 중..."
    if feedback:
        bh = 52
        cv2.rectangle(frame, (0, H-bh), (W-PW, H), (20,0,20), -1)
        cv2.rectangle(frame, (0, H-bh), (W-PW, H), (140,0,200), 1)
        frame = put_kr(frame, feedback, (14, H-bh+14),
                       FMD if PIL_OK else None, (255,200,255))

    return frame


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if EXERCISE not in ("squat", "pushup"):
        print(f"[오류] EXERCISE 는 'squat' 또는 'pushup': {EXERCISE!r}")
        sys.exit(1)

    model_path = MODEL_PATH or _DEFAULT_MODEL_PATH
    json_path  = EXPERT_JSON_PATH or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"expert_{EXERCISE}.json"
    )

    ensure_model(model_path)
    expert   = load_expert(EXERCISE, json_path)
    ex_total = expert["total_frames"]
    ex_frames= expert["frames"]

    # cap = cv2.VideoCapture(CAMERA_INDEX)
    cap = cv2.VideoCapture("./data/squat_user1.mp4")
    if not cap.isOpened():
        print(f"[오류] 웹캠 {CAMERA_INDEX}번을 열 수 없습니다.")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    print("[초기화] PoseLandmarker 로딩 중...")
    landmarker = build_landmarker(model_path)
    print("[초기화] 완료!  R=리셋  Q/ESC=종료\n")

    llm         = LLMFeedback(ANTHROPIC_API_KEY, cooldown=LLM_COOLDOWN)
    buf         = MetricsBuffer(window=WINDOW_SIZE)
    rep_counter = RepCounter(EXERCISE)
    fps         = 0.0
    t_prev      = time.perf_counter()
    start_ms    = int(time.time() * 1000)
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"영상 원본: {vid_w} x {vid_h}")

    # 목표 창 높이 고정 (720)
    TARGET_H = 720
    TARGET_W = int(vid_w * TARGET_H / vid_h)  # 비율 유지
    CAMERA_WIDTH = TARGET_W + 270  # HUD 패널 포함
    CAMERA_HEIGHT = TARGET_H

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = 1.0 / video_fps  # 한 프레임당 목표 시간

    while True:
        t_frame_start = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        h, w = frame.shape[:2]
        target_h = 720
        target_w = int(w * target_h / h)
        frame = cv2.resize(frame, (target_w, target_h))
        # frame = cv2.flip(frame, 1)

        H, W = frame.shape[:2]  # 리사이즈 후 실제 크기
        PW = 270

        # view는 HUD 붙이기 전에 먼저 잘라내기
        view = frame.copy()  # ← 이렇게 바꿔

        # HUD 공간 붙이기
        hud = np.zeros((H, PW, 3), dtype=np.uint8)
        frame = np.hstack([frame, hud])
        view = frame[:, :target_w].copy()

        # elapsed_ms = int(time.time()*1000) - start_ms
        # ex_idx     = get_expert_frame_idx(elapsed_ms, expert)
        # ex_metrics = ex_frames[ex_idx]["metrics"]
        # ex_lms_raw = ex_frames[ex_idx]["landmarks"]
        #
        # # STEP 1: 관절 추출
        # lms = extract_landmarks(landmarker, view, elapsed_ms)
        # 사용자 영상 현재 프레임 번호로 전문가 동기화
        user_frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        ex_idx = user_frame_idx % ex_total
        ex_metrics = ex_frames[ex_idx]["metrics"]
        ex_lms_raw = ex_frames[ex_idx]["landmarks"]

        # elapsed_ms는 landmarker용으로만 사용 (단조 증가 필요)
        elapsed_ms = int(time.time() * 1000) - start_ms

        # STEP 1: 관절 추출
        lms = extract_landmarks(landmarker, view, elapsed_ms)

        worst            = None
        deltas           = {}
        user_avg         = {}
        bad_indices      = set()
        skeleton_visible = False

        if lms:
            # STEP 2: 지표 계산
            try:
                raw = compute_metrics(lms)
            except Exception:
                raw = {}

            # STEP 3: 윈도우 평균
            buf.push(raw)
            user_avg = buf.get_avg()

            # STEP 3.5: 횟수 카운트 (사용자 각도 기반)
            if rep_counter.update(user_avg):
                print(f"  [{EXERCISE.upper()}] {rep_counter.count}회!")

            # STEP 4: delta + 선별
            deltas = compute_deltas(user_avg, ex_metrics, EXERCISE)
            result = select_worst(deltas)

            if result:
                worst_key, _ = result
                worst        = worst_key
                bad_indices  = set(METRIC_TO_LM.get(worst_key, []))
                llm.request(
                    exercise   = EXERCISE,
                    worst_key  = worst_key,
                    user_val   = user_avg.get(worst_key, 0.0),
                    expert_val = ex_metrics.get(worst_key, 0.0),
                )
            else:
                llm.text = ""

            # STEP 5a: 전문가 스켈레톤 (좌표 기반 오버레이)
            skeleton_visible = check_visibility(lms)

            if skeleton_visible:
                draw_expert_skeleton(view, ex_lms_raw, lms, bad_indices,
                                     alpha=EXPERT_ALPHA,
                                     offset_x=EXPERT_OFFSET_X)

            # STEP 5b: 사용자 스켈레톤
            draw_user_skeleton(view, lms, bad_indices)
            frame[:, :target_w] = view


        else:
            llm.text = "자세를 인식할 수 없습니다. 전신이 보이도록 서주세요."

        # FPS
        t_now  = time.perf_counter()
        fps    = 1.0 / max(t_now - t_prev, 1e-9)
        t_prev = t_now

        frame = render_hud(
            frame, EXERCISE, rep_counter.count, worst, deltas,
            user_avg, ex_metrics, llm, fps, ex_idx, ex_total,
            skeleton_visible
        )
        cv2.imshow("Workout Posture Correction", frame)

        # key = cv2.waitKey(1) & 0xFF
        elapsed = time.perf_counter() - t_frame_start
        remain_ms = max(1, int((frame_interval - elapsed) * 1000))

        key = cv2.waitKey(remain_ms) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('r'):
            rep_counter.count = 0
            rep_counter._in_down = False
            llm.text = ""
            print("[리셋] 횟수 초기화")

    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()
    print("종료.")