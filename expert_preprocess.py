"""
expert_preprocess.py — 전문가 영상 오프라인 전처리
=========================================================
사용법:
  1. 아래 [설정] 섹션 변수를 직접 수정
  2. python expert_preprocess.py 실행

출력 JSON 구조:
  {
    "exercise":     "squat",
    "fps":          30.0,
    "total_frames": 90,
    "frames": [
      {
        "frame_idx": 0,
        "metrics": {
          "knee_angle":  172.3,   # 도 (5프레임 슬라이딩 평균)
          ...
        },
        "landmarks": [[x, y, z, visibility], ...]  # 33개 (raw)
      }, ...
    ]
  }

수정 내역:
  - metrics에 5프레임 슬라이딩 평균 적용 (사용자와 동일 처리)
  - landmarks는 raw 유지 (시각화용)
"""

import cv2
import numpy as np
import math
import json
import os
import sys
import types
import urllib.request
from collections import deque


# ══════════════════════════════════════════════════════════════
#  [설정] — 여기만 수정하면 됩니다
# ══════════════════════════════════════════════════════════════

VIDEO_PATH           = r"squat_ex.mp4"
EXERCISE             = "dead"
OUTPUT_PATH          = None
DETECTION_CONFIDENCE = 0.5
PRESENCE_CONFIDENCE  = 0.5
TRACKING_CONFIDENCE  = 0.5
WINDOW_SIZE          = 5   # 사용자와 동일한 슬라이딩 윈도우 크기

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "pose_landmarker_lite.task")

# ══════════════════════════════════════════════════════════════
#  이하 수정 불필요
# ══════════════════════════════════════════════════════════════

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/latest/"
    "pose_landmarker_lite.task"
)

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
BaseOptions           = _bo.BaseOptions
MpImage               = _MpImage
ImageFormat           = _ImageFormat
RunningMode           = _vtm.VisionTaskRunningMode


def ensure_model():
    if os.path.exists(MODEL_PATH):
        print(f"[모델] {os.path.basename(MODEL_PATH)} 확인 완료")
        return
    print("[모델] pose_landmarker_lite.task 다운로드 중... (~7MB)")
    def _prog(n, bs, total):
        if total > 0:
            print(f"\r  {min(n*bs*100//total, 100)}%", end="", flush=True)
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH, _prog)
    print(f"\n[모델] 저장 완료 → {MODEL_PATH}")


def build_landmarker():
    opts = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=DETECTION_CONFIDENCE,
        min_pose_presence_confidence=PRESENCE_CONFIDENCE,
        min_tracking_confidence=TRACKING_CONFIDENCE,
    )
    return PoseLandmarker.create_from_options(opts)


def extract_landmarks(landmarker, bgr: np.ndarray):
    rgb    = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mp_img = MpImage(image_format=ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_img)
    return result.pose_landmarks[0] if result.pose_landmarks else None


def G(lms, i):
    p = lms[i]; return (p.x, p.y, p.z)

def calc_angle(a, b, c) -> float:
    ba = np.array([a[0]-b[0], a[1]-b[1], a[2]-b[2]])
    bc = np.array([c[0]-b[0], c[1]-b[1], c[2]-b[2]])
    n  = np.linalg.norm(ba) * np.linalg.norm(bc)
    if n == 0:
        return 0.0
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
        "knee_angle":  round((calc_angle(l_hip,   l_knee,  l_ankle) +
                               calc_angle(r_hip,   r_knee,  r_ankle)) / 2, 2),
        "hip_angle":   round((calc_angle(l_knee,  l_hip,   l_sh)    +
                               calc_angle(r_knee,  r_hip,   r_sh))   / 2, 2),
        "ankle_angle": round((calc_angle(l_knee,  l_ankle, l_foot)  +
                               calc_angle(r_knee,  r_ankle, r_foot))  / 2, 2),
        "elbow_angle": round((calc_angle(l_wrist, l_elbow, l_sh)    +
                               calc_angle(r_wrist, r_elbow, r_sh))   / 2, 2),
        "body_angle":  round((calc_angle(l_sh,    l_hip,   l_ankle) +
                               calc_angle(r_sh,    r_hip,   r_ankle)) / 2, 2),
        "spine_lean":  round(spine_lean, 2),
        "foot_width":  round(abs(l_ankle[0]-r_ankle[0]) / shoulder_w, 4),
        "grip_width":  round(abs(l_wrist[0]-r_wrist[0]) / shoulder_w, 4),
        "knee_align":  round(((l_knee[0]-l_foot[0]) +
                               (r_knee[0]-r_foot[0])) / 2 / shoulder_w, 4),
    }

def landmarks_to_list(lms) -> list:
    return [[round(p.x, 4), round(p.y, 4),
             round(p.z, 4), round(p.visibility, 3)]
            for p in lms]


# ── 5프레임 슬라이딩 평균 적용 ────────────────────────────────
def apply_sliding_average(frames_data: list, window: int = 5) -> list:
    """
    metrics에 슬라이딩 윈도우 평균 적용.
    landmarks는 시각화용이므로 raw 유지.
    사용자 실시간 파이프라인과 동일한 처리 → 공정한 delta 비교 가능.
    """
    if not frames_data:
        return frames_data

    metric_keys = list(frames_data[0]["metrics"].keys())

    # 키별 버퍼
    buffers = {k: deque(maxlen=window) for k in metric_keys}

    result = []
    for fd in frames_data:
        # 버퍼에 현재 프레임 지표 추가
        for k in metric_keys:
            buffers[k].append(fd["metrics"][k])

        # 슬라이딩 평균 계산
        avg_metrics = {
            k: round(float(np.mean(list(buffers[k]))), 4)
            for k in metric_keys
        }

        result.append({
            "frame_idx": fd["frame_idx"],
            "metrics":   avg_metrics,    # 평균값
            "landmarks": fd["landmarks"] # raw 유지
        })

    return result


def preprocess(video_path: str, exercise: str, output_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[오류] 영상 파일을 열 수 없습니다: {video_path}")
        sys.exit(1)

    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[전처리] {os.path.basename(video_path)}")
    print(f"         fps={fps:.1f}  총 {total_frames}프레임  운동={exercise}")
    print(f"         슬라이딩 윈도우: {WINDOW_SIZE}프레임 (사용자와 동일)")

    landmarker  = build_landmarker()
    frames_raw  = []   # raw metrics 먼저 수집
    frame_idx   = 0
    fail_count  = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        lms = extract_landmarks(landmarker, frame)

        if lms:
            frames_raw.append({
                "frame_idx": frame_idx,
                "metrics":   compute_metrics(lms),
                "landmarks": landmarks_to_list(lms),
            })
        else:
            fail_count += 1

        frame_idx += 1
        if frame_idx % 30 == 0:
            pct = frame_idx * 100 // max(total_frames, 1)
            print(f"\r  진행: {pct}%  ({frame_idx}/{total_frames})",
                  end="", flush=True)

    cap.release()
    landmarker.close()
    print(f"\n[전처리] 관절 추출 완료 — 성공 {len(frames_raw)}프레임 / 실패 {fail_count}프레임")

    # ── 5프레임 슬라이딩 평균 적용 ──
    print(f"[평균화] {WINDOW_SIZE}프레임 슬라이딩 평균 적용 중...")
    frames_data = apply_sliding_average(frames_raw, window=WINDOW_SIZE)
    print(f"[평균화] 완료")

    data = {
        "exercise":     exercise,
        "fps":          fps,
        "total_frames": len(frames_data),
        "window_size":  WINDOW_SIZE,   # 몇 프레임 평균인지 기록
        "frames":       frames_data,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[저장] {output_path}")


if __name__ == "__main__":
    if not os.path.exists(VIDEO_PATH):
        print(f"[오류] 영상 파일 없음: {VIDEO_PATH}")
        sys.exit(1)

    out = OUTPUT_PATH or os.path.join(
        os.path.dirname(os.path.abspath(VIDEO_PATH)),
        f"expert_{EXERCISE}2.json"
    )

    ensure_model()
    preprocess(VIDEO_PATH, EXERCISE, out)