"""Set tests (근거·출처 tab, 그 밖의 시험): two sets in one recording, failed last rep, when the count goes up.

python -m sim.run_sets [--n 400]
  two sets : gap 68-163 s (same / other exercise), same exercise 24-55 s, squat<->bench 32-55 s,
             deadlift with another exercise 29-51 s. 30 fps, noise 1 cm. Final and live.
  failed   : squat, bench, the last rep fails onto the safeties (sim.scenarios.fail_last).
  timing   : live count time - true time the bar is back (first rep / later reps), and when the live
             calculation knows the raise is over - true raise end (live speed).
Writes results/sets.json and results/sets.md.
"""
import argparse
import json
import os
import zlib
from multiprocessing import Pool

import numpy as np

from core.judge.sets import counted, failed
from sim.evaluate import run_final, run_live, run_final_sets, run_live_sets, live_raise_end
from sim.gen import make_recording, Opts
from sim.scenarios import fail_last
from sim.two_sets import make_two_sets

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXS = ("squat", "bench", "deadlift")
FPS, NOISE = 30.0, 0.01

GROUPS = {
    "long": dict(gap=(68, 163), pairs="any"),
    "same short": dict(gap=(24, 55), pairs="same"),
    "squat-bench short": dict(gap=(32, 55), pairs="sb"),
    "deadlift change short": dict(gap=(29, 51), pairs="dl"),
}


def _pair(rng, kind, i):
    if kind == "any":
        if i % 2 == 0:
            e = EXS[int(rng.integers(3))]
            return e, e
        kind = "other"
    if kind == "same":
        e = EXS[i % 3]
        return e, e
    if kind == "sb":
        return ("squat", "bench") if i % 2 == 0 else ("bench", "squat")
    if kind == "dl":
        o = ("squat", "bench")[i % 2]
        return ("deadlift", o) if (i // 2) % 2 == 0 else (o, "deadlift")
    a, b = rng.choice(3, 2, replace=False)
    return EXS[a], EXS[b]


def _two(args):
    name, i = args
    g = GROUPS[name]
    rng = np.random.default_rng(zlib.crc32(f"{name}|{i}".encode()))
    ex1, ex2 = _pair(rng, g["pairs"], i)
    ts = None
    while ts is None:
        ts = make_two_sets(ex1, ex2, float(rng.uniform(*g["gap"])), rng, FPS, NOISE)
    n1, n2 = len(ts.reps[0]), len(ts.reps[1])
    out = dict(ex=(ex1, ex2), gap=ts.gap, n=(n1, n2))
    _, sf = run_final_sets(ts.t, ts.y, ts.x, FPS, ex1, ts.frame_t, ts.labels)
    sl = run_live_sets(ts.t, ts.y, ts.x, FPS, ex1, ts.frame_t, ts.labels)
    for k, sets in (("final", sf), ("live", sl)):
        out[k] = [(s.exercise, len(s.reps), s.reason) for s in sets]
    return out


def _failed(args):
    ex, i = args
    rng = np.random.default_rng(zlib.crc32(f"fail|{ex}|{i}".encode()))
    rec = make_recording(ex, rng, FPS, NOISE, fail_last(ex, rng))
    _, trips = run_final(rec)
    live = run_live(rec)
    n = len(rec.reps)
    return dict(ex=ex, f=len(counted(trips)) == n, l=len(counted(live)) == n,
                ff=len(failed(trips)) == 1, fl=len(failed(live)) == 1)


def _timing(args):
    ex, i = args
    rng = np.random.default_rng(zlib.crc32(f"time|{ex}|{i}".encode()))
    rec = make_recording(ex, rng, FPS, NOISE, Opts())
    trips, sig = run_live(rec, with_signal=True)
    reps = counted(trips)
    if len(reps) != len(rec.reps):
        return dict(ex=ex, ok=False)
    lead = [(k, r.end_t - tr.t1) for k, (r, tr) in enumerate(zip(reps, rec.reps))]
    spd = []
    for r, tr in zip(reps, rec.reps):
        t1 = live_raise_end(sig, r, rec.rest)
        if t1 is not None:
            spd.append(t1 - tr.raise1)
    return dict(ex=ex, ok=True, lead=lead, spd=spd)


def _two_summary(rows):
    out = {}
    for mode in ("final", "live"):
        right = split = merged_ok = split_ok = 0
        gaps_split, gaps_merged = [], []
        for r in rows:
            got = r[mode]
            (e1, e2), (n1, n2) = r["ex"], r["n"]
            two_right = len(got) == 2 and got[0][:2] == (e1, n1) and got[1][:2] == (e2, n2)
            if len(got) == 2:
                split += 1
                gaps_split.append(r["gap"])
                split_ok += two_right
            elif len(got) == 1:
                gaps_merged.append(r["gap"])
                merged_ok += e1 == e2 and got[0][1] == n1 + n2
            right += two_right
        n = len(rows)
        out[mode] = dict(n=n, two_sets_right=right / n, split=split / n, split_counts_right=split_ok / max(split, 1),
                         merged=len(gaps_merged) / n, merged_sum_right=merged_ok / max(len(gaps_merged), 1),
                         split_gap=[min(gaps_split), max(gaps_split)] if gaps_split else None,
                         merged_gap=[min(gaps_merged), max(gaps_merged)] if gaps_merged else None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400, help="recordings per two-set group")
    ap.add_argument("--fail", type=int, default=300, help="sets per exercise, failed last rep")
    ap.add_argument("--time", type=int, default=300, help="sets per exercise, timing")
    a = ap.parse_args()
    res = {}
    with Pool(os.cpu_count()) as pool:
        for name in GROUPS:
            rows = pool.map(_two, [(name, i) for i in range(a.n)], chunksize=4)
            res[name] = _two_summary(rows)
            res[name]["gap"] = [min(r["gap"] for r in rows), max(r["gap"] for r in rows)]
            if name == "same short":
                # where the split happens: share split per 2 s bin
                bins = {}
                for r in rows:
                    b = int(r["gap"] // 2 * 2)
                    bins.setdefault(b, []).append(len(r["final"]) == 2)
                res[name]["split_by_gap_final"] = {str(k): float(np.mean(v)) for k, v in sorted(bins.items())}
            print(name, json.dumps(res[name]["final"]), json.dumps(res[name]["live"]))
        fr = pool.map(_failed, [(ex, i) for ex in ("squat", "bench") for i in range(a.fail)], chunksize=8)
        res["failed last rep"] = dict(n=len(fr), count_final=float(np.mean([r["f"] for r in fr])),
                                      count_live=float(np.mean([r["l"] for r in fr])),
                                      marked_final=float(np.mean([r["ff"] for r in fr])),
                                      marked_live=float(np.mean([r["fl"] for r in fr])))
        print("failed", res["failed last rep"])
        tr = pool.map(_timing, [(ex, i) for ex in EXS for i in range(a.time)], chunksize=8)
        ok = [r for r in tr if r["ok"]]
        first = [d for r in ok for k, d in r["lead"] if k == 0]
        later = [d for r in ok for k, d in r["lead"] if k > 0]
        allv = first + later
        spd = {ex: [d for r in ok if r["ex"] == ex for d in r["spd"]] for ex in EXS}
        pct = lambda v: [float(np.percentile(v, q)) for q in (5, 50, 95)]
        res["timing"] = dict(sets=len(ok), count_all=pct(allv), count_first=pct(first), count_later=pct(later),
                             speed={ex: pct(v) for ex, v in spd.items()},
                             speed_all=pct([d for v in spd.values() for d in v]))
        print("timing", res["timing"])
    json.dump(res, open(os.path.join(HERE, "results", "sets.json"), "w"), indent=1)
    report(res)


def report(res):
    L = ["# 세트 시험", "", f"`python -m sim.run_sets`, 30 fps, 잡음 1 cm. 최종 / 라이브.", "",
         "| 시험 | 녹화 | 간격 (초) | 두 세트로 나뉘고 종목·횟수 맞음 | 나뉨 | 합쳐짐 (합쳐진 것 중 횟수 합 맞음) | 나뉜 간격 | 합쳐진 간격 |",
         "|---|---|---|---|---|---|---|---|"]
    for name in GROUPS:
        r = res[name]
        f, l = r["final"], r["live"]
        rng = lambda v: "—" if v is None else f"{v[0]:.1f}~{v[1]:.1f}"
        L.append(f"| {name} | {f['n']} | {r['gap'][0]:.0f}~{r['gap'][1]:.0f} | {f['two_sets_right']:.3f} / {l['two_sets_right']:.3f} | "
                 f"{f['split']:.3f} / {l['split']:.3f} | {f['merged']:.3f} ({f['merged_sum_right']:.3f}) / "
                 f"{l['merged']:.3f} ({l['merged_sum_right']:.3f}) | {rng(f['split_gap'])} | {rng(f['merged_gap'])} |")
    if "split_by_gap_final" in res["same short"]:
        L += ["", "같은 종목 24~55초, 간격 2초 구간마다 두 세트로 나뉜 비율 (최종): " +
              ", ".join(f"{k}초 {v:.2f}" for k, v in res["same short"]["split_by_gap_final"].items())]
    fl = res["failed last rep"]
    L += ["", f"마지막 rep 실패 (스쿼트·벤치 {fl['n']}세트): 횟수 맞음 {fl['count_final']:.3f} / {fl['count_live']:.3f}, "
          f"미완료 시도 1개로 표시 {fl['marked_final']:.3f} / {fl['marked_live']:.3f}."]
    tm = res["timing"]
    f3 = lambda v: " / ".join(f"{x:+.2f}" for x in v)
    L += ["", f"횟수가 오른 시점 − 바가 다 돌아온 시점 (라이브, 횟수가 맞은 {tm['sets']}세트, 5 / 50 / 95 백분위, 초): "
          f"전체 {f3(tm['count_all'])}, 첫 rep {f3(tm['count_first'])}, 이후 rep {f3(tm['count_later'])}.",
          "", "올리는 동작이 끝났다고 아는 시점 − 실제로 끝난 시점 (라이브 속도, 초): 전체 " + f3(tm["speed_all"]) + ", " +
          ", ".join(f"{ex} {f3(v)}" for ex, v in tm["speed"].items()) + "."]
    open(os.path.join(HERE, "results", "sets.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
