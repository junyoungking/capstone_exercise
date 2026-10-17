"""Synthetic bar trajectories and observations (rebuilt 2026-10; the iteration 1-17 generator was lost).

Every generator value is listed in GEN below and in README.md. A recording is
  * a dense true trajectory (1 kHz): height y (m, up +) and horizontal x (m)
  * frames at 1/fps with timing jitter, observed with noise, dropouts, one gap and outliers
  * the truth: one entry per rep (far point interval, raise interval, ROM, true mean speed)

Shapes: lowering is a cosine; raising is a cosine (75%) or slows down in the middle (25%,
"sticking"). Squat and bench start racked, unrack, walk out sideways, do the reps, walk back and rerack.
Deadlift starts and ends on the floor.
"""
from dataclasses import dataclass, field, replace
from typing import List, Optional

import numpy as np

DT = 0.001  # dense trajectory step (s)

# ---- generator values (also in README.md) -------------------------------------------------
GEN = dict(
    reps=(2, 9),                      # reps per set, uniform integer, inclusive
    rom=dict(squat=(0.527, 0.062, 0.35, 0.70),   # mean, sd, clip lo, clip hi (m)
             bench=(0.355, 0.065, 0.22, 0.50),
             deadlift=(0.53, 0.05, 0.42, 0.65)),
    rom_rep_sd=0.03,                  # rep-to-rep ROM variation (fraction)
    # tempo: calibrated so the slow-motion statistics in the doc come out (4x: 17% asked, 8x: 97% asked,
    # 8x counted right 0.222); this generator gives 15%, 99%, 0.26
    down=dict(squat=(1.2, 3.0), bench=(1.0, 2.6), deadlift=(0.8, 1.8)),    # set-level lowering time (s)
    up=dict(squat=(0.7, 1.5), bench=(0.6, 1.3), deadlift=(0.8, 1.5)),      # set-level raise time, first rep (s)
    fatigue=(0.03, 0.10),             # raise time grows by this fraction per rep
    dur_rep_sd=0.08,                  # rep-to-rep duration variation (fraction)
    far_pause=dict(squat=(0.0, 0.3), bench=(0.0, 0.3), deadlift=(0.3, 1.0)),   # s at the far point
    rest_pause=dict(squat=(0.5, 2.5), bench=(0.5, 2.5), deadlift=(0.8, 2.5)),  # s between reps
    sticking=0.25,                    # share of raises that slow down in the middle
    sway=0.03,                        # max sideways bar drift within a rep (m)
    squat_rack=(1.25, 1.50), floor=0.225,
    unrack=(0.03, 0.10),              # lift off the hooks (m)
    walk=dict(squat=(0.12, 0.60), bench=(0.12, 0.30)),   # sideways walkout / move over the chest (m)
    walk_s=dict(squat=(1.5, 3.0), bench=(0.8, 1.5)),
    walk_bob=0.015,                   # vertical bob while walking (m, max)
    bench_lockout_over_hook=(0.03, 0.12),
    bench_chest=(0.62, 0.72),         # bar height at the chest, lying on the bench (m); lockout = chest + ROM
    lead=(1.0, 3.0), tail=(1.0, 3.0), settle=(1.0, 2.5),
    # observation
    jitter=0.2,                       # frame time jitter, fraction of 1/fps (uniform +-)
    slow_ratio=0.5, slow_tau=1.0,     # slow noise component: sd = ratio x noise, OU time constant (s)
    dropout=0.10, gap_s=0.4, outlier=0.01, outlier_m=(0.2, 1.0),
)


@dataclass
class TrueRep:
    far_in: float
    far_out: float
    raise0: float
    raise1: float
    rom: float
    v_true: float
    t0: float = None         # leaves the rest side
    t1: float = None         # back on the rest side


@dataclass
class Recording:
    exercise: str
    rest: str
    fps: float
    noise: float
    frame_t: np.ndarray      # all frame times
    true_y: np.ndarray       # true values at frame times
    true_x: np.ndarray
    t: np.ndarray            # observed samples (dropouts and the gap removed)
    y: np.ndarray
    x: np.ndarray
    reps: List[TrueRep]
    meta: dict = field(default_factory=dict)


class Timeline:
    def __init__(self, y0, x0):
        self.t = [np.array([0.0])]
        self.y = [np.array([y0])]
        self.x = [np.array([x0])]
        self.now, self.cy, self.cx = 0.0, y0, x0

    def _add(self, n, ys, xs):
        ts = self.now + DT * np.arange(1, n + 1)
        self.t.append(ts); self.y.append(ys); self.x.append(xs)
        self.now, self.cy, self.cx = float(ts[-1]), float(ys[-1]), float(xs[-1])

    def hold(self, dur, bob=0.0, rng=None):
        n = max(1, int(round(dur / DT)))
        ys = np.full(n, self.cy)
        if bob and rng is not None:
            u = np.arange(1, n + 1) / n
            ys = ys + bob * np.sin(2 * np.pi * u * rng.uniform(1, 3)) * np.sin(np.pi * u)
        self._add(n, ys, np.full(n, self.cx))

    def move(self, dur, y1=None, x1=None, shape="cos", bob=0.0, rng=None, sway=0.0):
        n = max(1, int(round(dur / DT)))
        u = np.arange(1, n + 1) / n
        p = profile(u, shape)
        y1 = self.cy if y1 is None else y1
        x1 = self.cx if x1 is None else x1
        ys = self.cy + (y1 - self.cy) * p
        if bob and rng is not None:
            ys = ys + bob * np.sin(2 * np.pi * u * rng.uniform(1.5, 3)) * np.sin(np.pi * u)
        xs = self.cx + (x1 - self.cx) * p + sway * np.sin(np.pi * u)
        self._add(n, ys, xs)

    def arrays(self):
        return np.concatenate(self.t), np.concatenate(self.y), np.concatenate(self.x)


def profile(u, shape):
    """Fraction of the distance covered at time fraction u."""
    if shape == "cos":
        return (1 - np.cos(np.pi * u)) / 2
    if shape == "stick":
        v = np.sin(np.pi * u) * (1 - 0.6 * np.exp(-((u - 0.45) / 0.12) ** 2))
        c = np.cumsum(v)
        return c / c[-1]
    if shape == "lin":
        return u
    raise ValueError(shape)


def _u(rng, ab):
    return float(rng.uniform(*ab))


@dataclass
class Opts:
    """Knobs used by the break-it scenarios. Defaults = the base recording."""
    reps: Optional[int] = None
    rom: Optional[float] = None
    rom_range: Optional[tuple] = None
    far_pause: Optional[tuple] = None
    down: Optional[tuple] = None
    up: Optional[tuple] = None
    rest_pause: Optional[tuple] = None
    unrack: Optional[tuple] = None
    walk: Optional[tuple] = None
    tail: Optional[tuple] = None
    lead: Optional[tuple] = None
    occlude: Optional[tuple] = None                 # extra occlusion (s, s) placed inside a random rep
    cut_in_first: bool = False                      # recording starts inside the first rep
    hooks: dict = field(default_factory=dict)       # scenario hooks, see scenarios.py
    gap: bool = True
    outliers_x: bool = False
    speed_factor: float = 1.0                       # slow-motion export (time stretched)


def _rom(rng, ex, o: Opts):
    if o.rom is not None:
        return o.rom
    if o.rom_range is not None:
        return _u(rng, o.rom_range)
    m, s, lo, hi = GEN["rom"][ex]
    return float(np.clip(rng.normal(m, s), lo, hi))


def build(ex, rng, o: Opts = Opts()):
    """Dense true trajectory and per-rep truth for one set."""
    g = GEN
    hk = o.hooks
    n_reps = o.reps if o.reps is not None else int(rng.integers(g["reps"][0], g["reps"][1] + 1))
    rom0 = _rom(rng, ex, o)
    t_down = _u(rng, o.down or g["down"][ex])
    t_up = _u(rng, o.up or g["up"][ex])
    fat = _u(rng, g["fatigue"])
    fp = o.far_pause or g["far_pause"][ex]
    rp = o.rest_pause or g["rest_pause"][ex]
    reps: List[TrueRep] = []

    if ex == "deadlift":
        rest_h = g["floor"]
        tl = Timeline(rest_h, 0.0)
        tl.hold(_u(rng, o.lead or g["lead"]))
        if "pre" in hk:
            hk["pre"](tl, rng, rest_h, rom0)
    else:
        if ex == "squat":
            rack = _u(rng, g["squat_rack"])
            unrack = _u(rng, o.unrack or g["unrack"])
            rest_h = rack + unrack
        else:
            chest = _u(rng, g["bench_chest"])
            rest_h = chest + rom0                       # lockout
            unrack = _u(rng, o.unrack or g["bench_lockout_over_hook"])
            hook = rest_h - unrack
            rack = hook
        walk = _u(rng, o.walk or g["walk"][ex]) * (-1 if ex == "squat" else 1)
        tl = Timeline(rack, 0.0)
        tl.hold(_u(rng, o.lead or g["lead"]))
        if "unrack" in hk:
            hk["unrack"](tl, rng, rack, rest_h, walk, rom0)
        else:
            tl.move(_u(rng, (0.6, 1.2)), y1=rest_h)
            if "walkout" in hk:
                hk["walkout"](tl, rng, rest_h, walk)
            else:
                tl.move(_u(rng, g["walk_s"][ex]), x1=walk, shape="cos", bob=g["walk_bob"], rng=rng)
            tl.hold(_u(rng, g["settle"]))
        if "pre" in hk:
            hk["pre"](tl, rng, rest_h, rom0)

    sgn = -1.0 if ex != "deadlift" else 1.0     # direction of the far point
    for k in range(n_reps):
        rom = rom0 * float(rng.normal(1, g["rom_rep_sd"]))
        jd = float(rng.normal(1, g["dur_rep_sd"]))
        ju = float(rng.normal(1, g["dur_rep_sd"]))
        d_dur = max(0.3, t_down * jd)
        u_dur = max(0.3, t_up * (1 + fat * k) * ju)
        shape = "stick" if rng.random() < g["sticking"] else "cos"
        sway = float(rng.uniform(-g["sway"], g["sway"]))
        if "rep" in hk:
            t0 = tl.now
            r = hk["rep"](tl, rng, k, n_reps, rest_h, rom, d_dur, u_dur, shape, sway, fp)
            if r is not None:
                for rr in (r if isinstance(r, list) else [r]):
                    if rr.t0 is None:
                        rr.t0, rr.t1 = t0, tl.now
                    reps.append(rr)
            if "after_rep" in hk:
                hk["after_rep"](tl, rng, k, n_reps, rest_h)
            elif k < n_reps - 1:
                tl.hold(_u(rng, rp))
            continue
        x0 = tl.cx
        t0 = tl.now
        if ex == "deadlift":
            r0 = tl.now
            tl.move(u_dur, y1=rest_h + rom, shape=shape, sway=sway)
            r1 = tl.now
            fin = tl.now
            tl.hold(_u(rng, fp))
            fout = tl.now
            tl.move(d_dur, y1=rest_h, x1=x0, shape="cos")
            reps.append(TrueRep(fin, fout, r0, r1, rom, rom / (r1 - r0), t0, tl.now))
        else:
            tl.move(d_dur, y1=rest_h - rom, shape="cos", sway=sway)
            fin = tl.now
            tl.hold(_u(rng, fp))
            fout = tl.now
            r0 = tl.now
            tl.move(u_dur, y1=rest_h, x1=x0, shape=shape)
            r1 = tl.now
            reps.append(TrueRep(fin, fout, r0, r1, rom, rom / (r1 - r0), t0, tl.now))
        if "after_rep" in hk:
            hk["after_rep"](tl, rng, k, n_reps, rest_h)
        elif k < n_reps - 1:
            tl.hold(_u(rng, rp))

    if "post" in hk:
        hk["post"](tl, rng, rest_h, rom0)
    else:
        tl.hold(_u(rng, (0.5, 2.0)))
        if ex != "deadlift":
            tl.move(_u(rng, g["walk_s"][ex]), x1=0.0, bob=g["walk_bob"], rng=rng)
            tl.move(_u(rng, (0.6, 1.2)), y1=rack)
    tl.hold(_u(rng, o.tail or g["tail"]))
    t, y, x = tl.arrays()
    return t, y, x, reps


def _ou(rng, n, dt, sd, tau):
    """Ornstein-Uhlenbeck (AR(1)) noise with stationary sd and time constant tau."""
    from scipy.signal import lfilter
    a = np.exp(-dt / tau)
    e = rng.normal(0, sd * np.sqrt(1 - a * a), n)
    x0 = rng.normal(0, sd)
    out, _ = lfilter([1.0], [1.0, -a], e, zi=[a * x0])
    return out


def observe(rng, dense_t, dense_y, dense_x, fps, noise, o: Opts = Opts(), reps=()):
    g = GEN
    T = dense_t[-1]
    n = int(np.floor(T * fps))
    ft = np.arange(n) / fps + rng.uniform(-g["jitter"], g["jitter"], n) / fps
    ft = np.clip(np.sort(ft), 0, T)
    ty = np.interp(ft, dense_t, dense_y)
    tx = np.interp(ft, dense_t, dense_x)
    dt = 1.0 / fps
    ny = rng.normal(0, noise, n) + _ou(rng, n, dt, g["slow_ratio"] * noise, g["slow_tau"])
    nx = rng.normal(0, noise, n) + _ou(rng, n, dt, g["slow_ratio"] * noise, g["slow_tau"])
    oy, ox = ty + ny, tx + nx
    out = rng.random(n) < g["outlier"]
    oy[out] += rng.choice([-1, 1], out.sum()) * rng.uniform(*g["outlier_m"], out.sum())
    if o.outliers_x:
        out2 = rng.random(n) < g["outlier"]
        ox[out2] += rng.choice([-1, 1], out2.sum()) * rng.uniform(*g["outlier_m"], out2.sum())
    keep = rng.random(n) >= g["dropout"]
    if o.gap and T > 3:
        g0 = rng.uniform(1.0, T - 1.0 - g["gap_s"])
        keep &= ~((ft >= g0) & (ft < g0 + g["gap_s"]))
    if o.occlude and reps:
        r = reps[int(rng.integers(len(reps)))]
        dur = rng.uniform(*o.occlude)
        # inside the rep: from leaving the rest side to coming back (a longer gap covers the whole rep)
        s0 = rng.uniform(r.t0, max(r.t0, r.t1 - dur))
        keep &= ~((ft >= s0) & (ft < s0 + dur))
    return ft, ty, tx, ft[keep], oy[keep], ox[keep]


def make_recording(ex, rng, fps=30.0, noise=0.01, o: Opts = Opts()) -> Recording:
    dt_, dy, dx, reps = build(ex, rng, o)
    if o.speed_factor != 1.0:
        dt_ = dt_ * o.speed_factor
        k = o.speed_factor
        reps = [replace(r, far_in=r.far_in * k, far_out=r.far_out * k, raise0=r.raise0 * k, raise1=r.raise1 * k,
                        t0=r.t0 * k, t1=r.t1 * k, v_true=r.v_true / k) for r in reps]
    ft, ty, tx, t, y, x = observe(rng, dt_, dy, dx, fps, noise, o, reps)
    rest = "bottom" if ex == "deadlift" else "top"
    meta = {}
    if o.cut_in_first and reps:
        r0 = reps[0]
        # the first rep runs from leaving the rest side to coming back; cut somewhere inside it
        c = rng.uniform(r0.t0, r0.t1)
        meta["cut"] = c
        m = t >= c
        t, y, x = t[m] - c, y[m], x[m]
        mf = ft >= c
        ft, ty, tx = ft[mf] - c, ty[mf], tx[mf]
        reps = [replace(r, far_in=r.far_in - c, far_out=r.far_out - c, raise0=r.raise0 - c,
                        raise1=r.raise1 - c, t0=r.t0 - c, t1=r.t1 - c) for r in reps[1:]]
    return Recording(ex, rest, fps, noise, ft, ty, tx, t, y, x, reps, meta)
