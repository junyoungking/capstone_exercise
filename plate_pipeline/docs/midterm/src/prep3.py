"""Demo bundle v3: joined user clips + reference, plate analysis, every-frame pose (2D overlay + 3D world).

prep3.py <config.json> <out.json>
config: {"user": {"clips": [[run_dir, n_frames], ...], "pose": npz, "fps": 24},
         "ref":  {"clips": [[run_dir, n_frames]], "pose": npz, "fps": 24}}

Plate geometry: per-axis plate scale (0.45 m / box height, / box width), camera angle from the box ratio,
distance d = f*0.45/h with f = 0.77*long side, lifter plane DELTA = 0.70 m behind the near plate.
3D: MediaPipe world landmarks, smoothed, bone lengths fixed to the set median, heading from standing frames,
scaled so the standing nose height matches the stature measured with the plate scale, feet on the floor.
"""
import sys, json, copy
import numpy as np
sys.path.insert(0, "/home/claude/capstone_exercise/plate_pipeline")
from core.io.rows import read_rows
from core.signal.height import height_signal
from core.signal.final import clean_final
from core.judge.session import run_session, REST
from core.measure.times import all_rep_values
from core.compare.summary import summarize
from core.compare.judge import run_compare
from core.text.templates import render, summary_sentence
from core.config import DEFAULT

CFG = json.load(open(sys.argv[1])); OUT = sys.argv[2]
EX, PLATE, DELTA = "squat", 0.45, 0.70
J2 = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]      # 2D overlay joints
J3 = [0, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]  # 3D joints
BONES = [(11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 25), (25, 27), (24, 26), (26, 28),
         (27, 29), (29, 31), (27, 31), (28, 30), (30, 32), (28, 32), (11, 12), (23, 24), (0, 7), (0, 8)]
KP = [0, .25, .49, .51, .75, 1]


def clip(run, offset):
    rows = read_rows(f"{run}/rows.jsonl.gz")
    res = json.load(open(f"{run}/result.json"))
    fps = res["fps"]
    t, y, x = height_signal(rows)
    fr = clean_final(t, y, x, fps, DEFAULT.final)
    sets = run_session(fr.sig.t, fr.sig.y, fr.sig.x, fr.sig.gap_before, EX, "final", cfg=DEFAULT)
    trips = [r for s in sets for r in s.reps]
    vals = all_rep_values(fr.sig, fr.raw_t, trips, REST[EX], DEFAULT.speed)
    T = np.array([r["t"] for r in rows])
    st, sy = fr.sig.t, fr.sig.y
    plate, boxes = [], []
    for r in rows:
        p = r.get("plate")
        if p and p.get("used"):
            plate.append([round(p["center"][0], 1), round(p["center"][1], 1), round((p["size_px"][0] + p["size_px"][1]) / 4, 1)])
            boxes.append(p["size_px"])
        else:
            plate.append(None); boxes.append([np.nan, np.nan])
    reps = []
    for v in vals:
        if v.back is None or v.far_out is None:
            continue
        s, fi, fo, b = v.start, v.far_in, v.far_out, v.back
        lo, hi = min(fi, fo), max(fi, fo)
        w = (st >= s) & (st <= b)
        hmin = np.min(sy[w]); h0 = np.interp(s, st, sy); h1 = np.interp(b, st, sy)

        def cross(a0, a1, level, down):
            m = (st >= a0) & (st <= a1); tt, yy = st[m], sy[m]
            idx = np.where(yy <= level)[0] if down else np.where(yy >= level)[0]
            return float(tt[idx[0]]) if len(idx) else .5 * (a0 + a1)
        o = offset
        reps.append(dict(start=round(s + o, 3), mid_down=round(cross(s, lo, .5 * (h0 + hmin), True) + o, 3),
                         far=round(.5 * (lo + hi) + o, 3), far_in=round(lo + o, 3), far_out=round(hi + o, 3),
                         mid_up=round(cross(hi, b, .5 * (h1 + hmin), False) + o, 3), back=round(b + o, 3),
                         rom_cm=round(v.rom * 100, 1), v=round(v.v_mean, 3), down=round(v.down, 2),
                         pause=round(v.far_pause or 0.0, 2), up=round(v.up, 2)))
    # depth (cm) from each rep's standing level; height in cm
    h = np.interp(T, st, sy) * 100
    kt, kb = [], []
    for q in reps:
        for tt in (q["start"] - offset, q["back"] - offset):
            kt.append(tt); kb.append(np.interp(tt, T, h))
    depth = np.clip(np.interp(T, kt, kb) - h, 0, None) if kt else np.zeros_like(h)
    # shift RepValues so rest times between clips are not mixed up
    sv = []
    for v in vals:
        v2 = copy.copy(v)
        for k in ("start", "far_in", "far_out", "back"):
            if getattr(v2, k, None) is not None:
                setattr(v2, k, getattr(v2, k) + offset)
        sv.append(v2)
    good = [(tr, v) for tr, v in zip(trips, sv) if v.back is not None and v.far_out is not None]
    return dict(T=T + offset, plate=plate, boxes=np.array(boxes, float), depth=depth, reps=reps, fps=fps,
                W=res["width"], H=res["height"], trips=[g[0] for g in good], vals=[g[1] for g in good])


def smooth(a, sig):
    out = a.copy()
    r = int(3 * sig + 1)
    k = np.exp(-.5 * (np.arange(-r, r + 1) / sig) ** 2); k /= k.sum()
    for j in range(a.shape[1]):
        for c in range(a.shape[2]):
            v = a[:, j, c].copy(); ok = np.isfinite(v)
            if ok.sum() < 2:
                continue
            idx = np.arange(len(v)); v = np.interp(idx, idx[ok], v[ok])
            out[:, j, c] = np.convolve(np.pad(v, r, mode="edge"), k, mode="valid")
    return out


def despike(kp, thr_px, win):
    """Per joint: drop points far from the rolling median (L/R flips, jumps), then interpolate."""
    out = kp.copy()
    T = len(kp)
    for j in range(33):
        for c in range(2):
            v = kp[:, j, c].copy()
            med = np.array([np.nanmedian(v[max(0, i - win):i + win + 1]) for i in range(T)])
            v[np.abs(v - med) > thr_px] = np.nan
            out[:, j, c] = v
    return out


def build(role):
    cfg = CFG[role]
    parts, off = [], 0.0
    for run, nf in cfg["clips"]:
        parts.append(clip(run, off)); off += nf / cfg["fps"]
    T = np.concatenate([p["T"] for p in parts]); plate = sum([p["plate"] for p in parts], [])
    boxes = np.vstack([p["boxes"] for p in parts]); depth = np.concatenate([p["depth"] for p in parts])
    reps = sum([p["reps"] for p in parts], [])
    for i, r in enumerate(reps):
        r["n"] = i + 1
    trips = sum([p["trips"] for p in parts], []); vals = sum([p["vals"] for p in parts], [])
    W, H, fps = parts[0]["W"], parts[0]["H"], cfg["fps"]
    clips = []
    o = 0.0
    for run, nf in cfg["clips"]:
        clips.append([round(o, 3), round(o + nf / fps, 3)]); o += nf / fps
    # local plate box size (+-0.25 s)
    bw, bh = np.full(len(T), np.nan), np.full(len(T), np.nan)
    for i, t in enumerate(T):
        m = (np.abs(T - t) <= .25) & np.isfinite(boxes[:, 0])
        if m.any():
            bw[i], bh[i] = np.median(boxes[m], axis=0)
    ok = np.isfinite(bw); idx = np.arange(len(T))
    bw = np.interp(idx, idx[ok], bw[ok]); bh = np.interp(idx, idx[ok], bh[ok])
    ratio = float(np.median(np.minimum(bw, bh) / np.maximum(bw, bh)))
    f = 0.77 * max(W, H)
    dist = f * PLATE / bh
    # the near plate is DELTA in front of the lifter along the bar; seen at angle a from the side,
    # the depth gap is DELTA*cos(a) (side view: full DELTA, front view: none)
    gap = DELTA * np.cos(np.arccos(min(1.0, ratio)))
    kd = (dist + gap) / dist
    # pose: align npz frames to T by time
    z = np.load(cfg["pose"])
    pt = z["t"]
    sel = np.array([int(np.argmin(np.abs(pt - t))) for t in T])
    kp = z["kp"][sel].astype(float); wl = z["world"][sel].astype(float)
    # 2D overlay: despike + smooth
    kp = despike(kp, thr_px=0.06 * H, win=int(fps * .25))
    kp2 = smooth(kp, sig=fps * .04)
    # 3D: smooth world landmarks, heading from standing frames, bone lengths fixed
    w3 = smooth(wl, sig=max(1.0, fps * .06))
    stand = np.array([not any(r["start"] - .4 <= t <= r["back"] + .4 for r in reps) and np.isfinite(w3[i, 0, 0]) for i, t in enumerate(T)])
    ws = np.nanmedian(w3[stand], axis=0)
    up = np.array([0, -1.0, 0])
    lat = (ws[24] - ws[23]) + (ws[12] - ws[11]); lat = lat - up * (lat @ up); lat /= np.linalg.norm(lat)
    fwd = np.cross(up, lat)
    # make "forward" point where the toes point
    toes = (ws[31] + ws[32]) / 2 - (ws[29] + ws[30]) / 2
    if toes @ fwd < 0:
        fwd = -fwd; lat = -lat
    R = np.vstack([lat, up, fwd])                      # rows: x lateral, y up, z forward
    P = np.einsum("ij,tkj->tki", R, w3)
    # bone lengths -> set median (keeps directions)
    med = {b: np.nanmedian(np.linalg.norm(P[:, b[1]] - P[:, b[0]], axis=1)) for b in BONES}
    order = [(23, 25), (25, 27), (27, 29), (29, 31), (24, 26), (26, 28), (28, 30), (30, 32), (23, 11), (24, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
    Q = P.copy()
    for a, b in order:
        L = med.get((a, b)) or med.get((b, a))
        d = P[:, b] - P[:, a]; n = np.linalg.norm(d, axis=1, keepdims=True)
        Q[:, b] = Q[:, a] + d / np.where(n > 1e-6, n, 1) * L
    # stature from 2D standing frames: nose to floor with the plate vertical scale and the depth factor
    sy = PLATE / bh
    nose_px = kp2[:, 0, 1]; floor_px = np.nanmax(kp2[:, [29, 30, 31, 32], 1], axis=1)
    stat = np.nanmedian(((floor_px - nose_px) * sy * kd)[stand]) / 0.92
    nose_w = np.nanmedian((Q[:, 0, 1] - np.nanmin(Q[:, [29, 30, 31, 32], 1], axis=1))[stand])
    scale = (stat * 0.92) / nose_w if np.isfinite(nose_w) and nose_w > 0 else 1.0
    Q = Q * scale
    floor = np.nanmin(Q[:, [29, 30, 31, 32], 1], axis=1)
    Q[:, :, 1] -= floor[:, None]
    mid = np.nanmedian(((Q[:, 29] + Q[:, 31] + Q[:, 30] + Q[:, 32]) / 4)[stand], axis=0)
    Q[:, :, 0] -= mid[0]; Q[:, :, 2] -= mid[2]
    seg = lambda a, b: round(float(np.nanmedian(np.linalg.norm(Q[:, b] - Q[:, a], axis=1))) * 100, 1)
    geo = dict(cam_angle=round(float(np.degrees(np.arccos(min(1, ratio)))), 1), box_ratio=round(ratio, 3),
               dist_m=round(float(np.median(dist)), 2), kd=round(float(np.median(kd)), 3), focal_px=round(f),
               stature_cm=round(float(stat) * 100), thigh_cm=round((seg(23, 25) + seg(24, 26)) / 2, 1),
               shin_cm=round((seg(25, 27) + seg(26, 28)) / 2, 1), torso_cm=round((seg(11, 23) + seg(12, 24)) / 2, 1),
               upper_cm=round((seg(11, 13) + seg(12, 14)) / 2, 1), fore_cm=round((seg(13, 15) + seg(14, 16)) / 2, 1),
               cm_per_px=round(float(np.median(100 * PLATE / bh)), 3))
    k2 = [[[round(float(kp2[i, j, 0]), 1), round(float(kp2[i, j, 1]), 1), round(float(min(1, kp[i, j, 2] if np.isfinite(kp[i, j, 2]) else 0)), 2)] for j in J2]
          if np.isfinite(kp2[i, 11, 0]) else None for i in range(len(T))]
    j3 = [[[round(float(Q[i, j, 0]), 3), round(float(Q[i, j, 1]), 3), round(float(Q[i, j, 2]), 3)] for j in J3]
          if np.isfinite(Q[i, 11, 0]) else None for i in range(len(T))]
    # crop around the lifter and plate
    xs = kp2[:, J2, 0][np.isfinite(kp2[:, J2, 0])]; ys = kp2[:, J2, 1][np.isfinite(kp2[:, J2, 1])]
    pp = np.array([p for p in plate if p])
    x0 = min(np.percentile(xs, 1), pp[:, 0].min() - pp[:, 2].max()); x1 = max(np.percentile(xs, 99), pp[:, 0].max() + pp[:, 2].max())
    y0 = min(np.percentile(ys, .5), pp[:, 1].min() - pp[:, 2].max()); y1 = max(np.percentile(ys, 99.5), pp[:, 1].max() + pp[:, 2].max())
    padx, pady = .10 * (x1 - x0), .06 * (y1 - y0)
    crop = [round(max(0, x0 - padx)), round(max(0, y0 - pady)), round(min(W, x1 + padx)), round(min(H, y1 + pady))]
    print(role, "frames", len(T), "reps", len(reps), "geo", geo, "crop", crop)
    return dict(fps=fps, width=W, height=H, duration=float(T[-1]), t=[round(float(x), 4) for x in T], plate=plate,
                depth=[round(float(x), 1) for x in depth], reps=reps,
                plate_used=round(100 * sum(p is not None for p in plate) / len(plate), 1),
                crop=crop, geo=geo, j2=J2, kp2=k2, j3names=J3, j3=j3, clips=clips if len(clips) > 1 else []), trips, vals


B = {}
B["user"], ut, uv = build("user")
B["ref"], rt, rv = build("ref")
usm, rsm = summarize(EX, B["user"]["fps"], ut, uv), summarize(EX, B["ref"]["fps"], rt, rv)
sents = render(run_compare(usm, rsm), EX, EX)
speeds = [r["v"] for r in B["user"]["reps"]]
loss = 100 * (max(speeds) - speeds[-1]) / max(speeds)
B["sentences"] = [dict(kind=s.kind, text=s.text, key=s.key, shown=s.shown) for s in sents]
# set medians exactly as the comparison uses them (unrounded rep values)
B["set_median"] = {k: {f: round(float(np.median([getattr(v, a) or 0.0 for v in vv])), 5) for f, a in (("down", "down"), ("pause", "far_pause"), ("up", "up"))}
                   for k, vv in (("user", uv), ("ref", rv))}
print(B["set_median"])
B["summary"] = summary_sentence(EX, len(speeds), speeds, loss).text
B["v_loss"] = round(loss, 1)
for s in B["sentences"]:
    print(s)
print(B["summary"])
json.dump(B, open(OUT, "w"), ensure_ascii=False, separators=(",", ":"))
print("KB", round(len(json.dumps(B, ensure_ascii=False)) / 1024))
