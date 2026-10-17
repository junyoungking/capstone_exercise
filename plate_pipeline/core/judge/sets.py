"""Final-mode exclusions over a whole set (pipeline tab: 부분 rep, 다른 자리, 리랙).

  partial    : amp < partial x median amp of all round trips in the set
  other_spot : amp < other_amp x median and the far point x differs from the set median by > other_dx_m
               (final only, only with >= 3 round trips) - re-gripping the bar in the rack
  far_gap    : far point next to a gap > gap_s -> depth unknown, neither check applies
  failed attempts that are smaller than partial x median or sit elsewhere (> other_dx_m) are reracks
"""
import numpy as np

from ..config import DEFAULT, RepCfg


def finalize(trips, cfg: RepCfg = DEFAULT.rep):
    reps = [r for r in trips if r.kind == "rep" and "bounce" not in r.flags]
    for r in reps:
        r.flags = [f for f in r.flags if f not in ("partial", "other_spot")]
    if reps:
        med = float(np.median([r.amp for r in reps]))
        medx = float(np.median([r.far_x for r in reps]))
        for r in reps:
            if r.far_gap:
                continue
            if r.amp < cfg.partial * med:
                r.flags.append("partial")
            elif (len(reps) >= 3 and r.amp < cfg.other_amp * med
                  and abs(r.far_x - medx) > cfg.other_dx_m):
                r.flags.append("other_spot")
    counted = [r for r in reps if r.counted]
    for f in (r for r in trips if r.kind == "failed"):
        f.flags = [g for g in f.flags if g != "rerack"]
        if counted:
            med = float(np.median([r.amp for r in counted]))
            medx = float(np.median([r.far_x for r in counted]))
            if f.amp < cfg.partial * med or abs(f.far_x - medx) > cfg.other_dx_m:
                f.flags.append("rerack")
    return trips


def counted(trips):
    return [r for r in trips if r.counted]


def failed(trips):
    return [r for r in trips if r.kind == "failed" and not r.flags]
