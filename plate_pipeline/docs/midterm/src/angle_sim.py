"""Camera-angle correction check with a real bar path.

Real in-plane bar path (stu10_70, side view, pipeline height_signal) is placed on the plate plane and
seen by a virtual pinhole camera at yaw/pitch/roll angles. For every frame the plate rim (r = 0.225 m)
is projected; the detector-like box (w, h, centre) and the rim ellipse are measured from the image.

Methods
  A  one scale for both axes: 0.45 / mean(w, h)                 (no angle handling)
  B  per-axis scale 0.45/h vertical, 0.45/w horizontal, local  (current pipeline)
  C  per-axis scale fixed from the first second                (no perspective update)
  D  rim-ellipse rectification (affine ellipse -> circle), local, up = bar-path principal axis
Errors are reported for per-rep ROM and mean concentric velocity against the true path.
"""
import sys, json
import numpy as np
sys.path.insert(0, "/home/claude/capstone_exercise/plate_pipeline")
from core.io.rows import read_rows
from core.signal.height import height_signal

rows = read_rows(sys.argv[1])
t, y, x = height_signal(rows)
reps = json.load(open(sys.argv[2]))["user"]["reps"]
R = 0.225
F = 1000.0          # focal length px (about 57 deg horizontal FOV on 1080 wide)
D = 3.0             # camera distance from the path centre (m)
P0 = np.array([0.0, 1.0, 0.0])   # path centre in world (plate plane z = 0)
X = x - x.mean()
Y = y - (y.max() + y.min()) / 2
ang = np.linspace(0, 2 * np.pi, 180, endpoint=False)


def camera(yaw, pitch, roll):
    yaw, pitch, roll = np.radians([yaw, pitch, roll])
    d = np.array([np.sin(yaw) * np.cos(pitch), np.sin(pitch), np.cos(yaw) * np.cos(pitch)])
    C = P0 + D * d
    fwd = (P0 - C) / np.linalg.norm(P0 - C)
    right = np.cross(fwd, [0, 1, 0]); right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    cr, sr = np.cos(roll), np.sin(roll)
    right, up = cr * right + sr * up, -sr * right + cr * up
    return C, right, up, fwd


def project(Pw, cam):
    C, right, up, fwd = cam
    v = Pw - C
    z = v @ fwd
    return np.c_[F * (v @ right) / z, -F * (v @ up) / z]


def fit_ellipse(p):
    c = p.mean(0)
    q = p - c
    cov = q.T @ q / len(q)
    w, V = np.linalg.eigh(cov)
    # for points sampled uniformly in the circle's angle, var along an axis = a^2/2 (approximately)
    a, b = np.sqrt(2 * w[1]), np.sqrt(2 * w[0])
    return c, a, b, V[:, 1], V[:, 0]


def measure(cam):
    centres, boxes, ell = [], [], []
    for xi, yi in zip(X, Y):
        P = P0 + np.array([xi, yi, 0.0])
        rim = P + R * np.c_[np.cos(ang), np.sin(ang), np.zeros_like(ang)]
        pr = project(rim, cam)
        lo, hi = pr.min(0), pr.max(0)
        centres.append((lo + hi) / 2)
        boxes.append(hi - lo)
        ell.append(fit_ellipse(pr))
    return np.array(centres), np.array(boxes), ell


def integrate(c, sx, sy):
    dy = -np.diff(c[:, 1]) * (sy[1:] + sy[:-1]) / 2
    dx = np.diff(c[:, 0]) * (sx[1:] + sx[:-1]) / 2
    return np.r_[0, np.cumsum(dx)], np.r_[0, np.cumsum(dy)]


def rectify(c, ell):
    # ellipse -> circle affine map per frame, symmetric form (no axis-order ambiguity):
    # A = R * Sigma^(-1/2) / sqrt(2) maps the rim ellipse to a circle of radius R (metres).
    As = []
    for ce, a, b, u, v in ell:
        S = (a * a / 2) * np.outer(u, u) + (b * b / 2) * np.outer(v, v)
        w, V = np.linalg.eigh(S)
        As.append(R * (V @ np.diag(w ** -0.5) @ V.T) / np.sqrt(2))
    As = np.array(As)
    d = np.diff(c, axis=0)
    Am = (As[1:] + As[:-1]) / 2
    steps = np.einsum("nij,nj->ni", Am, d)
    pts = np.vstack([[0, 0], np.cumsum(steps, axis=0)])
    # gravity: principal axis of the bar path (squat path is close to vertical)
    q = pts - pts.mean(0)
    w, V = np.linalg.eigh(q.T @ q)
    upv = V[:, 1]
    if upv @ (pts[np.argmax(Y)] - pts[np.argmin(Y)]) < 0:
        upv = -upv
    hor = np.array([upv[1], -upv[0]])
    return q @ hor, q @ upv


def rep_values(yy):
    out = []
    for r in reps:
        m = (t >= r["start"]) & (t <= r["back"])
        rom = yy[m].max() - yy[m].min()
        mu = (t >= r["far_out"]) & (t <= r["back"])
        tt, ys = t[mu], yy[mu]
        v = (ys[-1] - ys[0]) / (tt[-1] - tt[0])
        out.append((rom, v))
    return np.array(out)


truth = rep_values(Y)
res = []
for pitch in (0, 20):
    for yaw in (0, 15, 30, 45, 60):
        for roll in (0, 8):
            if roll and pitch:
                continue
            cam = camera(yaw, pitch, roll)
            c, box, ell = measure(cam)
            w, h = box[:, 0], box[:, 1]
            meth = {}
            s = 0.45 / ((w + h) / 2)
            meth["A"] = integrate(c, s, s)[1]
            meth["B"] = integrate(c, 0.45 / w, 0.45 / h)[1]
            n0 = t < t[0] + 1.0
            meth["C"] = integrate(c, np.full_like(w, 0.45 / w[n0].mean()), np.full_like(h, 0.45 / h[n0].mean()))[1]
            meth["D"] = rectify(c, ell)[1]
            row = dict(yaw=yaw, pitch=pitch, roll=roll, axis_ratio=float(np.median(np.minimum(w, h) / np.maximum(w, h))),
                       est_angle=float(np.degrees(np.arccos(np.median(np.minimum(w, h) / np.maximum(w, h))))))
            for k, yy in meth.items():
                v = rep_values(yy)
                row[k + "_rom"] = float(np.max(np.abs(v[:, 0] / truth[:, 0] - 1)) * 100)
                row[k + "_vel"] = float(np.max(np.abs(v[:, 1] / truth[:, 1] - 1)) * 100)
            res.append(row)
            print({k: (round(v, 1) if isinstance(v, float) else v) for k, v in row.items()})
json.dump(dict(truth_rom_cm=[round(a * 100, 1) for a in truth[:, 0]], rows=res), open(sys.argv[3], "w"), indent=1)
