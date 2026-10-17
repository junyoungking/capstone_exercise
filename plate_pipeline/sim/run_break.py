"""Break-it scenarios at 30 fps, noise 1 cm.

python -m sim.run_break [--sets 600] [--only TEXT]
Per scenario and exercise: share of sets with the right count (final / live), share of sets where a
rerack was written as a failed attempt, and (partial scenario) share where the partial rep was flagged.
Writes results/break.json and prints the comparison with the doc numbers.
"""
import argparse
import json
import os
import time
import zlib
from multiprocessing import Pool

import numpy as np

from sim.gen import make_recording
from sim.evaluate import run_final, run_live
from sim.scenarios import SCENARIOS
from core.judge.sets import counted, failed

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _job(args):
    name, ex, i, fps, noise = args
    exs, factory, *_ = SCENARIOS[name]
    seed = zlib.crc32(f"{name}|{ex}|{i}|{fps}|{noise}".encode())
    rng = np.random.default_rng(seed)
    o = factory(ex, rng)
    rec = make_recording(ex, rng, fps, noise, o)
    if name == "방향을 거꾸로 줌":
        rec.rest = "top" if rec.rest == "bottom" else "bottom"
    _, trips = run_final(rec)
    live = run_live(rec)
    n = len(rec.reps)
    cf, cl = counted(trips), counted(live)
    return dict(ex=ex, f=len(cf) == n, l=len(cl) == n,
                rerack_f=len(failed(trips)) > 0, rerack_l=len(failed(live)) > 0,
                part_f=any("partial" in t.flags for t in trips), part_l=any("partial" in t.flags for t in live))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", type=int, default=600, help="sets per exercise")
    ap.add_argument("--only", default=None)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--noise", type=float, default=0.01)
    ap.add_argument("--out", default="break.json")
    args = ap.parse_args()
    names = [n for n in SCENARIOS if args.only is None or args.only in n]
    res = {}
    t0 = time.time()
    with Pool(os.cpu_count()) as pool:
        for name in names:
            exs = SCENARIOS[name][0]
            jobs = [(name, ex, i, args.fps, args.noise) for ex in exs for i in range(args.sets)]
            rows = pool.map(_job, jobs, chunksize=8)
            d = dict(sets=len(rows), final=float(np.mean([r["f"] for r in rows])),
                     live=float(np.mean([r["l"] for r in rows])),
                     rerack_failed=float(np.mean([r["rerack_f"] for r in rows])),
                     partial_flag=[float(np.mean([r["part_f"] for r in rows])), float(np.mean([r["part_l"] for r in rows]))],
                     by_ex={ex: [float(np.mean([r["f"] for r in rows if r["ex"] == ex])),
                                 float(np.mean([r["l"] for r in rows if r["ex"] == ex]))] for ex in exs},
                     doc=[SCENARIOS[name][2], SCENARIOS[name][3]])
            res[name] = d
            print(f"{name}: {d['final']:.3f} / {d['live']:.3f}  doc {d['doc']}  rerack {d['rerack_failed']:.3f}  "
                  f"{ {k: [round(v, 3) for v in vv] for k, vv in d['by_ex'].items()} }  [{time.time()-t0:.0f}s]", flush=True)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    path = os.path.join(HERE, "results", args.out)
    old = json.load(open(path)) if os.path.exists(path) else {}
    old.update(res)
    json.dump(old, open(path, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
