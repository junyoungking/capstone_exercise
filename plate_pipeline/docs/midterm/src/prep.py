"""Build the demo bundle (real plate/pose data for the two chosen videos)."""
import sys, json
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

D = sys.argv[1]
EX = "squat"
BONES = [(11, 23), (12, 24), (23, 25), (24, 26), (25, 27), (26, 28)]

def load(role):
    rows = read_rows(f"{D}/{role}/rows.jsonl.gz")
    res = json.load(open(f"{D}/{role}/result.json"))
    fps = res["fps"]
    t, y, x = height_signal(rows)
    fr = clean_final(t, y, x, fps, DEFAULT.final)
    sets = run_session(fr.sig.t, fr.sig.y, fr.sig.x, fr.sig.gap_before, EX, "final", cfg=DEFAULT)
    reps = [r for s in sets for r in s.reps]
    vals = all_rep_values(fr.sig, fr.raw_t, reps, REST[EX], DEFAULT.speed)
    sm = summarize(EX, fps, reps, vals)
    return rows, res, fps, (t, y), fr, vals, sm

def bundle(role, rows, res, fps, ty, fr, vals):
    t_used, y_used = ty
    T = np.array([r["t"] for r in rows])
    # plate per frame (used only)
    plate = []
    for r in rows:
        p = r.get("plate")
        if p and p.get("used"):
            plate.append([round(p["center"][0], 1), round(p["center"][1], 1), round((p["size_px"][0] + p["size_px"][1]) / 4, 1)])
        else:
            plate.append(None)
    # final (filtered) height resampled at frame times, cm, top rest = 0
    st, sy = fr.sig.t, fr.sig.y
    h = np.interp(T, st, sy)
    rest_level = np.median(sy[:max(5, int(fps))])
    h = (h - rest_level) * 100
    # skeleton: keep frames with consistent bone lengths
    kp_frames = [(i, np.array(r["kp"], float)) for i, r in enumerate(rows) if "kp" in r]
    lens = {b: [] for b in BONES}
    for _, k in kp_frames:
        for a, b in BONES:
            if k[a, 2] > .5 and k[b, 2] > .5:
                lens[(a, b)].append(np.hypot(*(k[a, :2] - k[b, :2])))
    med = {b: (np.median(v) if v else None) for b, v in lens.items()}
    skel = {}
    kept = 0
    for i, k in kp_frames:
        ok = True; n = 0
        for a, b in BONES:
            if k[a, 2] > .5 and k[b, 2] > .5 and med[(a, b)]:
                n += 1
                L = np.hypot(*(k[a, :2] - k[b, :2]))
                if not (0.65 * med[(a, b)] <= L <= 1.35 * med[(a, b)]):
                    ok = False
        if ok and n >= 3:
            skel[i] = [[round(float(p[0]), 1), round(float(p[1]), 1), round(float(p[2]), 2)] for p in k]
            kept += 1
    reps = []
    for n, v in enumerate(vals, 1):
        if v.back is None:
            continue
        s, fi, fo, b = v.start, v.far_in, v.far_out, v.back
        lo, hi = min(fi, fo), max(fi, fo)
        far = 0.5 * (lo + hi)
        w = (st >= s) & (st <= b)
        hmin = np.min(sy[w]); h0 = np.interp(s, st, sy); h1 = np.interp(b, st, sy)
        def cross(a0, a1, level, down):
            m = (st >= a0) & (st <= a1)
            tt, yy = st[m], sy[m]
            idx = np.where(yy <= level)[0] if down else np.where(yy >= level)[0]
            return float(tt[idx[0]]) if len(idx) else 0.5 * (a0 + a1)
        mid_d = cross(s, lo, 0.5 * (h0 + hmin), True)
        mid_u = cross(hi, b, 0.5 * (h1 + hmin), False)
        reps.append(dict(n=len(reps) + 1, start=round(s, 3), mid_down=round(mid_d, 3), far=round(far, 3),
                         far_in=round(lo, 3), far_out=round(hi, 3), mid_up=round(mid_u, 3), back=round(b, 3),
                         rom_cm=round(v.rom * 100, 1) if v.rom else None,
                         v=round(v.v_mean, 3) if v.v_mean else None,
                         down=round(v.down, 2) if v.down else None,
                         pause=round(v.far_pause, 2) if v.far_pause is not None else None,
                         up=round(v.up, 2) if v.up else None))
    print(role, "frames", len(rows), "skeleton kept", kept, "/", len(kp_frames), "reps", len(reps))
    return dict(fps=fps, width=res["width"], height=res["height"], duration=float(T[-1]),
                t=[round(float(x), 4) for x in T], plate=plate, h=[round(float(x), 2) for x in h],
                skel=skel, reps=reps,
                plate_used=round(100 * sum(p is not None for p in plate) / len(plate), 1),
                pose_frames=round(100 * kept / max(1, len(kp_frames)), 1))

out = {}
ur = load("user"); rr = load("ref")
out["user"] = bundle("user", *ur[:6]) if False else bundle("user", ur[0], ur[1], ur[2], ur[3], ur[4], ur[5])
out["ref"] = bundle("ref", rr[0], rr[1], rr[2], rr[3], rr[4], rr[5])
comps = run_compare(ur[6], rr[6])
sents = render(comps, EX, EX)
speeds = [v.v_mean for v in ur[5]]
vs = [x for x in speeds if x]
loss = 100 * (max(vs) - speeds[-1]) / max(vs) if len(vs) >= 2 and speeds[-1] else None
out["sentences"] = [dict(kind=s.kind, text=s.text, key=s.key, shown=s.shown) for s in sents]
out["summary"] = summary_sentence(EX, len(out["user"]["reps"]), speeds, loss).text
out["v_loss"] = round(loss, 1) if loss is not None else None
for s in out["sentences"]: print(s)
print(out["summary"])
json.dump(out, open(f"{D}/bundle.json", "w"), ensure_ascii=False, separators=(",", ":"))
print("bundle KB", round(len(json.dumps(out)) / 1024))
