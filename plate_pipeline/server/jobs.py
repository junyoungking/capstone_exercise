"""Upload processing (구현 설계 tab, 구현 순서 3단계): video -> frame rows -> final calculation -> sets -> report.

python -m server.jobs VIDEO --exercise squat [--plate-mm 450] [--box x0,y0,x1,y1] [--boxes labels.json]
                      [--model plate.onnx] [--out DIR]
Without a model the plate is placed once (--box, or the largest dark circle in the first frame) and then
followed with KLT; --boxes gives per-frame boxes from a label file instead of a detector.
Writes DIR/rows.jsonl.gz, DIR/result.json and DIR/report.html.
"""
import argparse
import json
import os
import time
from dataclasses import replace

import cv2
import numpy as np

from core.config import DEFAULT, Config
from core.io import video
from core.io.rows import write_rows
from core.recog.plate import BoxFileDetector, OnnxPlateDetector, find_plate_circle, refine_circle, Detection
from core.recog.track import PlateTracker
from core.recog.pose import PoseEstimator, SideLock, person_box, person_center, select_lifter, feet_y
from core.signal.height import height_signal
from core.signal.final import clean_final
from core.judge.session import run_session, REST
from core.measure.times import all_rep_values
from core.compare.summary import summarize
from core.compare.judge import run_compare
from core.text.templates import render, summary_sentence


def pick_detection(dets, box):
    """While tracking: the detection closest to the tracked plate (the tracker still gates it).
    Before the first box: the biggest confident one (the near plate in a side view)."""
    if not dets:
        return None
    if box is None:
        return max(dets, key=lambda d: d.score * d.size[0] * d.size[1])
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return min(dets, key=lambda d: (d.center[0] - cx) ** 2 + (d.center[1] - cy) ** 2)


def recognize(path, detector=None, init_box=None, on_frame=None, rim=True, pose=None, exercise="squat"):
    """Frame rows for a whole video. detector(image, seq) -> [Detection].
    rim enables circle fallback both for initialization and around the tracked box.
    With rim=False, wait for a detection (or an explicit init_box) to initialize.
    """
    tr = PlateTracker()
    rows = []
    pose_cfg = getattr(pose, "cfg", DEFAULT.pose)
    every_n = getattr(pose, "every_n", pose_cfg.every_n)
    if pose is not None and every_n < 1:
        raise ValueError("pose.every_n must be positive")
    side_lock = SideLock(exercise) if pose is not None else None
    prev_center = None
    for seq, (t, img) in enumerate(video.frames(path)):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dets = detector(img, seq) if detector is not None else []
        det = None
        if tr.box is None and not any(d.score >= tr.cfg.score_high for d in dets):
            if init_box is not None:
                det = Detection(tuple(init_box), 1.0, "klt")
            elif rim and detector is None:
                det = find_plate_circle(img)
                if det is not None:
                    det.src = "rim"
        if det is None and rim and tr.box is not None:
            det = refine_circle(gray, tr.box)
        st = tr.step(gray, dets=dets + ([det] if det is not None else []), t=t)
        det = tr.selected
        if st is not None and st.drift:
            for previous in reversed(rows):
                plate = previous.get("plate", {})
                if plate.get("src") in ("det", "det_low", "rim"):
                    break
                if plate.get("src") == "klt":
                    plate.update(src="klt_drift", used=False)
        row = {"t": round(t, 4), "seq": seq}
        if detector is not None:
            ranked = sorted(dets, key=lambda d: d.score, reverse=True)[:10]
            row["dets"] = [[*map(float, d.box), float(d.score)] for d in ranked]
            row["pick"] = next((i for i,d in enumerate(ranked) if d is det), None)
        if st is not None:
            x0, y0, x1, y1 = st.box
            cx, cy = st.center
            row["plate"] = {"box": [round(v, 1) for v in st.box], "score": round(st.score, 3),
                            "used": st.src != "lost", "center": [round(cx, 2), round(cy, 2)],
                            "size_px": [round(x1 - x0, 2), round(y1 - y0, 2)], "src": st.src, "n_pts": st.n_pts}
        row["cam"] = [0.0, 0.0]
        if pose is not None and seq % every_n == 0:
            start = time.perf_counter()
            candidates = pose(img, t)
            elapsed_ms = (time.perf_counter() - start) * 1000
            row["pose"] = {"n": len(candidates), "ms": round(elapsed_ms, 3)}
            index = select_lifter(candidates, st.box if st is not None else None,
                                  prev_center, exercise, cfg=pose_cfg,
                                  image_size=(img.shape[1], img.shape[0]))
            if index is not None:
                lms = candidates[index]
                row["person"] = {
                    "box": [round(v, 1) for v in person_box(lms, pose_cfg.vis_thr)],
                    "vis": round(float(lms[[11, 12, 23, 24], 3].mean()), 2),
                    "feet_y": feet_y(lms, pose_cfg.vis_thr, img.shape[0]),
                }
                row["kp"] = [[round(float(p[0]), 1), round(float(p[1]), 1),
                              round(float(p[3]), 2)] for p in lms]
                row["side"] = side_lock.update(lms)
                center = person_center(lms, pose_cfg.vis_thr)
                if center is not None:
                    prev_center = center
            elif pose_cfg.backend == "rtmpose":
                prev_center = None
        rows.append(row)
        if on_frame is not None:
            on_frame(seq, t, img, st)
    return rows


def analyse(rows, exercise, fps, plate_mm=450.0, cfg: Config = DEFAULT):
    t, y, x = height_signal(rows, plate_mm)
    fr = clean_final(t, y, x, fps, cfg.final)
    s = fr.sig
    sets = run_session(s.t, s.y, s.x, s.gap_before, exercise, "final", cfg=cfg)
    rest = REST[exercise]
    out = []
    for k, st in enumerate(sets):
        reps = st.reps
        vals = all_rep_values(fr.sig, fr.raw_t, reps, rest, cfg.speed)
        sm = summarize(exercise, fps, reps, vals)
        comps = run_compare(sm)
        speeds = [v.v_mean for v in vals]
        vs = [v for v in speeds if v is not None]
        loss = None
        if len(vs) >= 2 and speeds[-1] is not None:
            loss = 100.0 * (max(vs) - speeds[-1]) / max(vs)
        sents = [summary_sentence(exercise, len(reps), speeds, loss)] + render(comps, exercise, cfg=cfg)
        out.append(dict(
            set=k + 1, exercise=exercise, reason=st.reason, reps=len(reps), failed=len(st.failed),
            partial=sum("partial" in tr.flags for tr in st.trips),
            t0=reps[0].entry_t if reps else None, t1=reps[-1].end_t if reps else None,
            rep_values=[dict(n=i + 1, start=v.start, far_in=v.far_in, far_out=v.far_out, back=v.back, rom_m=v.rom,
                             far_dist_m=v.far_dist, v_mean=v.v_mean, down=v.down, far_pause=v.far_pause, up=v.up,
                             blank=v.blank) for i, v in enumerate(vals)],
            v_loss_pct=loss,
            sentences=[dict(key=s_.key, text=s_.text, kind=s_.kind, shown=s_.shown) for s_ in sents],
        ))
    sig = dict(t=s.t.tolist(), y=s.y.tolist(), raw_t=t.tolist(), raw_y=y.tolist())
    return out, sig


def process_video(path, exercise, plate_mm=450.0, init_box=None, boxes=None, model=None, out_dir=None, rim=None,
                  pose=False, device="auto"):
    """rim: look for the plate rim when the detector gives nothing (default: only without a model)."""
    info = video.info(path)
    det = OnnxPlateDetector(model, device=device) if model else (BoxFileDetector(boxes) if boxes else None)
    estimator = PoseEstimator() if pose is True else (pose or None)
    t0 = time.time()
    try:
        rows = recognize(path, det, init_box, rim=(model is None) if rim is None else rim,
                         pose=estimator, exercise=exercise)
    finally:
        if pose is True:
            estimator.close()
    took = time.time() - t0
    fps = len(rows) / max(rows[-1]["t"] - rows[0]["t"], 1e-6) if len(rows) > 1 else info.fps
    sets, sig = analyse(rows, exercise, fps, plate_mm)
    used = sum(1 for r in rows if r.get("plate", {}).get("used"))
    res = dict(video=os.path.basename(str(path)), exercise=exercise, plate_mm=plate_mm, fps=fps,
               width=info.width, height=info.height, frames=len(rows), plate_frames=used,
               recog_s=round(took, 1), recog_fps=round(len(rows) / max(took, 1e-6), 1),
               plate_source="model" if model else ("label boxes" if boxes else
                                                   ("첫 프레임 박스(입력) + 원판 테두리 원 찾기 + KLT" if init_box
                                                    else "첫 프레임 원 찾기 + 원판 테두리 원 찾기 + KLT")),
               sets=sets, detector_providers=getattr(det, "providers", []))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        write_rows(os.path.join(out_dir, "rows.jsonl.gz"), rows)
        json.dump(res, open(os.path.join(out_dir, "result.json"), "w"), ensure_ascii=False, indent=1)
        from server.report import write_report
        write_report(os.path.join(out_dir, "report.html"), res, sig)
    return res, rows, sig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--exercise", required=True, choices=sorted(REST))
    ap.add_argument("--plate-mm", type=float, default=450.0)
    ap.add_argument("--box", default=None, help="x0,y0,x1,y1 of the plate in the first frame")
    ap.add_argument("--boxes", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    ap.add_argument("--pose", action="store_true", help="record optional pose observations (RTMPose by default)")
    ap.add_argument("--pose-backend", choices=("rtmpose", "mediapipe"), default=DEFAULT.pose.backend)
    ap.add_argument("--pose-mode", choices=("lightweight", "balanced"), default=DEFAULT.pose.mode)
    ap.add_argument("--out", default="out")
    a = ap.parse_args()
    box = tuple(float(v) for v in a.box.split(",")) if a.box else None
    estimator = PoseEstimator(cfg=replace(DEFAULT.pose, backend=a.pose_backend, mode=a.pose_mode)) if a.pose else None
    try:
        res, _, _ = process_video(a.video, a.exercise, a.plate_mm, box, a.boxes, a.model, a.out,
                                  pose=estimator, device=a.device)
    finally:
        if estimator is not None:
            estimator.close()
    for s in res["sets"]:
        print(f"set {s['set']}: {s['reps']} reps ({s['reason']})")
        for x in s["sentences"]:
            print("  ", x["text"])


if __name__ == "__main__":
    main()
