"""MediaPipe pose on every frame; keep the person whose wrists/shoulders are nearest the plate.

pose_all.py <video> <plate_track.json> <out.npz> [model.task]
plate_track.json: {"t": [...], "plate": [[cx, cy, r] | null, ...]} in the video's time base.
Output: kp (T,33,3) px + visibility, world (T,33,3) metres (MediaPipe world landmarks), t (T,).
"""
import sys, json
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mpt
from mediapipe.tasks.python import vision

video, track, out = sys.argv[1], sys.argv[2], sys.argv[3]
model = sys.argv[4] if len(sys.argv) > 4 else "/home/claude/capstone_exercise/pose_landmarker_lite.task"
tr = json.load(open(track))
pt = np.array(tr["t"]); pp = tr["plate"]
opts = vision.PoseLandmarkerOptions(base_options=mpt.BaseOptions(model_asset_path=model),
                                    running_mode=vision.RunningMode.VIDEO, num_poses=3,
                                    min_pose_detection_confidence=0.3, min_pose_presence_confidence=0.3,
                                    min_tracking_confidence=0.3)
cap = cv2.VideoCapture(video)
fps = cap.get(cv2.CAP_PROP_FPS)
W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
kps, worlds, ts = [], [], []
prev = None
with vision.PoseLandmarker.create_from_options(opts) as lm:
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        t = i / fps
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        res = lm.detect_for_video(img, int(round(t * 1000)))
        k = np.full((33, 3), np.nan); w = np.full((33, 3), np.nan)
        if res.pose_landmarks:
            j = int(np.argmin(np.abs(pt - t)))
            p = pp[j]
            best, bd = None, 1e18
            for n, L in enumerate(res.pose_landmarks):
                a = np.array([[q.x * W, q.y * H, q.visibility] for q in L])
                if p:
                    d = min(np.hypot(a[15, 0] - p[0], a[15, 1] - p[1]), np.hypot(a[16, 0] - p[0], a[16, 1] - p[1]),
                            np.hypot(a[11, 0] - p[0], a[11, 1] - p[1]), np.hypot(a[12, 0] - p[0], a[12, 1] - p[1]))
                elif prev is not None:
                    d = np.nanmean(np.hypot(a[:, 0] - prev[:, 0], a[:, 1] - prev[:, 1]))
                else:
                    d = -np.ptp(a[:, 1])          # tallest person
                if d < bd:
                    best, bd = n, d
            k = np.array([[q.x * W, q.y * H, q.visibility] for q in res.pose_landmarks[best]])
            w = np.array([[q.x, q.y, q.z] for q in res.pose_world_landmarks[best]])
            prev = k
        kps.append(k); worlds.append(w); ts.append(t)
        i += 1
np.savez_compressed(out, kp=np.array(kps), world=np.array(worlds), t=np.array(ts))
print("frames", len(ts), "with pose", int(np.isfinite(np.array(kps)[:, 0, 0]).sum()))
