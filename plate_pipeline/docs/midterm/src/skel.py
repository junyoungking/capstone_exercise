"""Add a smooth skeleton template and depth to the demo bundle.

Pose from MediaPipe is noisy at the bottom (plate occlusion, left/right flips) and was run on
every second frame only. For the demo overlay we take, for each joint, the median offset from
the plate centre (in plate radii) at the same rep progress across all reps, and draw it at the
real plate position. Depth is measured from the standing level of each rep.
"""
import sys, json
import numpy as np
sys.path.insert(0, "/home/claude/capstone_exercise/plate_pipeline")
from core.io.rows import read_rows

D = sys.argv[1]
B = json.load(open(f"{D}/bundle.json"))
KP = [0, .25, .49, .51, .75, 1]
GRID = np.linspace(0, 1, 101)
JOINTS = [0, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]


def progress(r, t):
    k = [r["start"], r["mid_down"], r["far_in"], r["far_out"], r["mid_up"], r["back"]]
    return float(np.interp(t, k, KP))


def build(role):
    V = B[role]
    rows = read_rows(f"{D}/{role}/rows.jsonl.gz")
    T = np.array(V["t"])
    plate = V["plate"]
    rmed = float(np.median([p[2] for p in plate if p]))
    reps = V["reps"]
    first, last = reps[0]["start"], reps[-1]["back"]
    samples = []  # (p or None, 33x3 offsets)
    for i, r in enumerate(rows):
        if "kp" not in r or plate[i] is None:
            continue
        t = T[i]
        k = np.array(r["kp"], float)
        off = np.c_[(k[:, 0] - plate[i][0]) / rmed, (k[:, 1] - plate[i][1]) / rmed, k[:, 2]]
        rep = next((q for q in reps if q["start"] <= t <= q["back"]), None)
        if rep is not None:
            samples.append((progress(rep, t), off))
        elif first - 2.0 <= t <= last + 1.0:
            samples.append((None, off))
    stand = [o for p, o in samples if p is None]
    moving = [(p, o) for p, o in samples if p is not None]

    def med(offs):
        A = np.array(offs)  # n x 33 x 3
        out = np.zeros((33, 3))
        for j in range(33):
            ok = A[:, j, 2] > .5
            if ok.sum() >= 3:
                out[j, :2] = np.median(A[ok, j, :2], axis=0)
                out[j, 2] = 1
        return out

    S = med(stand)
    tm = np.zeros((len(GRID), 33, 3))
    for gi, g in enumerate(GRID):
        w = 0.06
        pick = [o for p, o in moving if abs(p - g) <= w]
        if g <= .06 or g >= .94:
            pick += stand  # anchor the ends to the standing pose
        tm[gi] = med(pick) if len(pick) >= 3 else S
    # smooth across the grid
    ker = np.exp(-0.5 * (np.arange(-6, 7) / 2.5) ** 2); ker /= ker.sum()
    sm = tm.copy()
    for j in range(33):
        for c in range(2):
            x = np.pad(tm[:, j, c], 6, mode="edge")
            sm[:, j, c] = np.convolve(x, ker, mode="valid")
        sm[:, j, 2] = (np.convolve(np.pad(tm[:, j, 2], 6, mode="edge"), ker, mode="valid") > .5).astype(float)
    # depth from the standing level of each rep
    h = np.array(V["h"])
    kt, kb = [], []
    for q in reps:
        for tt in (q["start"], q["back"]):
            kt.append(tt); kb.append(np.interp(tt, T, h))
    base = np.interp(T, kt, kb)
    depth = np.clip(base - h, 0, None)
    V["depth"] = [round(float(x), 1) for x in depth]
    V["rmed"] = round(rmed, 1)
    V["tmpl"] = [[[round(float(a), 3), round(float(b), 3), int(c)] for a, b, c in sm[gi]] for gi in range(len(GRID))]
    V["stand"] = [[round(float(a), 3), round(float(b), 3), int(c)] for a, b, c in S]
    print(role, "samples moving", len(moving), "standing", len(stand), "rmed", rmed,
          "bottom depth", [round(float(np.interp(q["far"], T, depth)), 1) for q in reps])
    return V


for role in ("user", "ref"):
    build(role)
json.dump(B, open(f"{D}/bundle2.json", "w"), ensure_ascii=False, separators=(",", ":"))
print("KB", round(len(json.dumps(B, ensure_ascii=False)) / 1024))
