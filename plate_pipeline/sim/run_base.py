"""Base synthetic test: counts by fps x noise, and mean-speed error by noise.

python -m sim.run_base [--sets 900] [--quick]
Writes results/base_counts.json, results/base_speed.json and results/base.md.
Counts : share of sets with the right number of reps (final / live), and the stricter share where every
         counted rep's far point is within 1 s of the true one. Doc (iteration 16): 근거·출처 tab, 합성 시험.
Speed  : error of the final calculation against the same calculation on the noise-free signal.
"""
import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np

from sim.gen import make_recording
from sim.evaluate import evaluate

EXS = ["squat", "bench", "deadlift"]
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _job(args):
    ex, fps, noise, seed, speed = args
    rng = np.random.default_rng(seed)
    rec = make_recording(ex, rng, fps, noise)
    r = evaluate(rec, speed=speed)
    return dict(ex=ex, n=r.n_true, f=r.final_ok, l=r.live_ok, ft=r.final_time_ok, lt=r.live_time_ok,
                sp=r.speed, spl=r.speed_live)


def run(cond, sets, speed, pool):
    fps, noise = cond
    jobs = [(EXS[i % 3], fps, noise, hash((fps, noise, i)) % (2 ** 32), speed) for i in range(sets)]
    return pool.map(_job, jobs, chunksize=8)


def speed_stats(rows, key="sp"):
    pairs = [(a, b) for r in rows for a, b, *rest in r[key] if b is not None]
    have = [(a, b) for a, b in pairs if a is not None]
    blank_gap = sum(1 for r in rows for a, b, *rest in r[key] if b is not None and a is None and rest and rest[0] == "gap")
    blank_other = sum(1 for r in rows for a, b, *rest in r[key] if b is not None and a is None and (not rest or rest[0] != "gap"))
    if not have:
        return {}
    a, b = np.array(have).T
    e = a - b
    rel = e / b
    return dict(n=len(pairs), rmse=float(np.sqrt(np.mean(e ** 2))), bias_med_pct=float(100 * np.median(rel)),
                over10_pct=float(100 * np.mean(np.abs(rel) > 0.10)),
                blank_gap_pct=float(100 * blank_gap / len(pairs)), blank_other_pct=float(100 * blank_other / len(pairs)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", type=int, default=900)
    ap.add_argument("--speed-sets", type=int, default=450)
    ap.add_argument("--procs", type=int, default=os.cpu_count())
    ap.add_argument("--speed-only", action="store_true")
    args = ap.parse_args()
    out_dir = os.path.join(HERE, "results")
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    counts = {}
    with Pool(args.procs) as pool:
        for fps in (() if args.speed_only else (15, 30, 60)):
            for noise in (0.003, 0.01, 0.02, 0.03, 0.04):
                rows = run((fps, noise), args.sets, False, pool)
                key = f"{fps}fps_{noise*100:g}cm"
                counts[key] = dict(sets=len(rows), reps=int(sum(r["n"] for r in rows)),
                                   final=float(np.mean([r["f"] for r in rows])),
                                   live=float(np.mean([r["l"] for r in rows])),
                                   final_time=float(np.mean([r["ft"] for r in rows])),
                                   live_time=float(np.mean([r["lt"] for r in rows])))
                print(key, counts[key], f"{time.time()-t0:.0f}s", flush=True)
        speed = {}
        for fps in (30, 60):
            for noise in (0.002, 0.005, 0.01, 0.02):
                rows = run((fps, noise), args.speed_sets, True, pool)
                key = f"{fps}fps_{noise*100:g}cm"
                speed[key] = dict(final=speed_stats(rows, "sp"), live_recompute=speed_stats(rows, "spl"))
                print(key, speed[key], f"{time.time()-t0:.0f}s", flush=True)
    if counts:
        json.dump(counts, open(os.path.join(out_dir, "base_counts.json"), "w"), indent=1)
    json.dump(speed, open(os.path.join(out_dir, "base_speed.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
