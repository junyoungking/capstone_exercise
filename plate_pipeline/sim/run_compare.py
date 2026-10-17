"""Comparison-rule tests (구현 설계 tab, step 2 "비교와 문장"; 근거·출처 tab, "비교 판정 시험").

python -m sim.run_compare [--part table|normal|gen|all] [--n 20000] [--groups 1500]

table  : core/crit_table.json against the doc's tables A, B, C (must agree within 0.1).
normal : independent normal rep values through the real compare code (core.compare):
         (a) no floor, no minimum change: false "different" per value must stay at or under 2.5%;
         (b) the doc's "처음 규칙 → 지금 표" table (rep sd 0.1/0.2/0.3 s, floor 0.07 s, min change 0.15 s).
gen    : sets from sim.gen run through the whole final pipeline, then summary + compare. Four conditions:
         noise 0.5 / 1 cm x rep variation small (duration 5%, ROM 2%) / large (12%, 5%), 30 fps.
         Per group of the same exercise and lowering time x ("same habit"):
           R  reference set (x)                M0 my set, same habit (x)
           R1 short reference, 1-2 reps (x)    M5 / M10 my set, lowering x + 0.5 / x + 1.0 s
           T  lowering shortens by 1 s from the first to the last rep (start 1.4 s or more)
           S  the last 2 reps 4 cm less ROM
         The same compares are also judged with the first rule (fixed multipliers: in-set 2, vs ref 3,
         short reference max(3, t 97.5% with n-1)) so the two rules are seen on the same data.
         Also records the generator's measured rep scatter (median per-set second-difference s).
Writes results/cmp_table.json, cmp_normal.json, cmp_gen.json and results/compare_rules.md.
"""
import argparse
import json
import os
import time
from dataclasses import replace
from multiprocessing import Pool

import numpy as np
from scipy.stats import t as tdist

from core.config import DEFAULT
from core.compare.summary import from_values, summarize
from core.compare.judge import Multipliers, compare_inset, compare_ref, run_compare, judged
from core.judge.sets import counted
from core.measure.times import all_rep_values
from sim.gen import GEN, Opts, make_recording
from sim.evaluate import run_final

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, "results")
EXS = ("squat", "bench", "deadlift")

# ---- the doc's tables (근거·출처 tab, "판정 배수 다시 정하기") --------------------------------------
DOC_A = dict(zip(range(4, 13), [6.5, 4.9, 4.2, 3.7, 3.4, 3.3, 3.1, 3.0, 2.9]))
DOC_C = dict(zip(range(3, 11), [25.7, 7.2, 5.3, 4.4, 4.0, 3.6, 3.5, 3.3]))
DOC_B = {3: [6.5, 5.0, 4.8, 4.6, 4.5, 4.5, 4.5, 4.5], 4: [5.0, 4.2, 4.0, 3.8, 3.8, 3.7, 3.8, 3.7],
         5: [4.8, 4.0, 3.9, 3.7, 3.7, 3.6, 3.6, 3.6], 6: [4.5, 3.8, 3.7, 3.5, 3.4, 3.3, 3.4, 3.3],
         7: [4.5, 3.8, 3.7, 3.5, 3.4, 3.3, 3.4, 3.3], 8: [4.5, 3.7, 3.6, 3.3, 3.3, 3.2, 3.2, 3.2],
         9: [4.5, 3.8, 3.6, 3.4, 3.4, 3.2, 3.3, 3.2], 10: [4.5, 3.8, 3.6, 3.3, 3.3, 3.2, 3.2, 3.1]}

# doc table "처음 규칙과 지금 표": (rep sd 0.1, 0.2, 0.3 s) first rule -> table; false, 1 s, 0.5 s (%)
DOC_RULES = {
    "inset 5": ((7.7, 15.4, 16.5), (0.0, 1.1, 2.0), (100, 98, 87), (98, 65, 37), (99, 72, 49), (63, 23, 12)),
    "inset 8": ((7.0, 11.3, 11.5), (0.6, 2.2, 2.4), (100, 99, 88), (100, 88, 57), (99, 71, 45), (87, 35, 17)),
    "ref 3x3": ((2.5, 8.2, 10.0), (0.0, 0.4, 1.2), (100, 96, 79), (96, 58, 33), (96, 61, 39), (51, 18, 10)),
    "ref 5x5": ((1.9, 5.2, 5.6), (0.5, 2.0, 2.2), (100, 100, 93), (100, 98, 82), (100, 77, 47), (98, 58, 30)),
    "ref 8x8": ((0.6, 3.0, 3.1), (0.5, 2.3, 2.4), (100, 100, 100), (100, 100, 99), (100, 93, 63), (100, 91, 58)),
    "ref 1, me 4": ((1.5, 7.6, 9.5), (0.0, 0.2, 1.0), (100, 83, 59), (79, 37, 20), (82, 43, 27), (15, 9, 6)),
    "ref 1, me 5": ((2.0, 7.1, 8.2), (0.0, 1.0, 1.9), (100, 87, 62), (94, 53, 28), (86, 44, 26), (47, 16, 9)),
    "ref 1, me 8": ((1.6, 4.1, 4.3), (0.5, 2.2, 2.4), (100, 90, 61), (100, 80, 47), (90, 40, 21), (80, 28, 13)),
    "ref 2, me 5": ((2.3, 7.4, 8.7), (0.0, 1.1, 1.9), (100, 96, 79), (99, 72, 44), (96, 60, 37), (71, 27, 14)),
}
CELLS = {"inset 5": ("inset", 5, 0), "inset 8": ("inset", 8, 0), "ref 3x3": ("ref", 3, 3), "ref 5x5": ("ref", 5, 5),
         "ref 8x8": ("ref", 8, 8), "ref 1, me 4": ("ref", 4, 1), "ref 1, me 5": ("ref", 5, 1),
         "ref 1, me 8": ("ref", 8, 1), "ref 2, me 5": ("ref", 5, 2)}

# doc "비교 판정 시험" (first rule, generator lost): ranges over the four conditions, %
DOC_GEN = {
    "vs ref report false": "2.9~5.5",
    "vs ref down false": "0.9~3.0", "vs ref far_pause false": "2.2~3.1",
    "vs ref down 0.5 s": "36~87", "vs ref down 1 s": "87~100",
    "short ref 1 rep down false": "1.8 / 3.7 / 4.8 / 4.8", "short ref 2 rep down false": "1.5 / 3.2 / 2.8 / 3.0",
    "short ref 1 rep down 0.5 s": "41.6 / 14.7 / 23.8 / 12.1", "short ref 1 rep down 1 s": "89.6 / 46.4 / 69.3 / 43.0",
    "short ref 2 rep down 0.5 s": "61.1 / 20.7 / 39.8 / 17.9", "short ref 2 rep down 1 s": "96.6 / 61.0 / 82.2 / 55.5",
    "inset far_dist false": "0.1~3.1", "inset down false": "1.7~4.8", "inset report false": "2.1~7.5",
    "inset last 2 reps 4 cm less": "69~81 (even), 35~39 (uneven)", "inset down 1 s shorter": "76~82 (even), 52~57 (uneven)",
}


def old_k(c):
    """First rule: fixed multipliers."""
    if c["kind"] == "inset":
        return 2.0
    if c["nb"] < DEFAULT.cmp.short_ref_n:
        return max(3.0, float(tdist.ppf(0.975, c["na"] - 1)))
    return 3.0


# ---- table ----------------------------------------------------------------------------------

def part_table():
    K = Multipliers()
    diffs = [abs(K.inset(n) - v) for n, v in DOC_A.items()]
    diffs += [abs(K.short(n) - v) for n, v in DOC_C.items()]
    diffs += [abs(K.ref(a, 3 + j) - v) for a, row in DOC_B.items() for j, v in enumerate(row)]
    out = dict(cells=len(diffs), max_diff=float(max(diffs)), ok=bool(max(diffs) <= 0.1))
    json.dump(out, open(os.path.join(RES, "cmp_table.json"), "w"), indent=1)
    print("table", out)
    return out


# ---- normal ---------------------------------------------------------------------------------

def _cell(args):
    kind, n_me, n_ref, sd, shift, N, seed, pure = args
    rng = np.random.default_rng(seed)
    cfg = replace(DEFAULT.cmp, floor_s=0.0, min_change_s=0.0) if pure else DEFAULT.cmp
    new = old = judged_n = 0
    for _ in range(N):
        a = rng.normal(2.0, sd, n_me)
        if kind == "inset":
            if shift:
                a = a + shift * np.arange(n_me) / (n_me - 2)     # last-2 mean - first-2 mean = shift
            c = compare_inset(from_values("squat", 30, down=list(a)), "down", cfg)
        else:
            b = rng.normal(2.0, sd, n_ref)
            c = compare_ref(from_values("squat", 30, down=list(a + shift)), from_values("squat", 30, down=list(b)),
                            "down", cfg)
        if c.result == "skip":
            continue
        judged_n += 1
        new += c.result == "different"
        old += judged(c.delta, c.se, old_k(dict(kind=c.kind, na=c.n_a, nb=c.n_b)), c.min_change)
    return dict(kind=kind, n_me=n_me, n_ref=n_ref, sd=sd, shift=shift, N=N, judged=judged_n,
                new=100 * new / max(judged_n, 1), old=100 * old / max(judged_n, 1))


def part_normal(N, pool):
    jobs = []
    seed = 100
    for n in range(4, 13):
        jobs.append(("inset", n, 0, 1.0, 0.0, N, seed, True)); seed += 1
    for a, b in [(n, n) for n in range(3, 11)] + [(3, 10), (10, 3), (4, 7), (7, 4)]:
        jobs.append(("ref", a, b, 1.0, 0.0, N, seed, True)); seed += 1
    for nr in (1, 2):
        for nu in range(4, 11):
            jobs.append(("ref", nu, nr, 1.0, 0.0, N, seed, True)); seed += 1
    for name, (kind, nu, nr) in CELLS.items():
        for sd in (0.1, 0.2, 0.3):
            for shift in (0.0, 1.0, 0.5):
                jobs.append((kind, nu, nr, sd, shift, N, seed, False)); seed += 1
    rows = pool.map(_cell, jobs, chunksize=1)
    pure = [r for r in rows if r["sd"] == 1.0]
    worst = max(r["new"] for r in pure)
    mc = 100 * 2 * np.sqrt(0.025 * 0.975 / N)
    out = dict(N=N, pure=pure, pure_worst=worst, pure_ok=bool(worst <= 2.5 + mc), mc_2se=mc,
               rules=[r for r in rows if r["sd"] != 1.0])
    json.dump(out, open(os.path.join(RES, "cmp_normal.json"), "w"), indent=1)
    print("normal pure worst %.2f%% (limit 2.5 + %.2f)" % (worst, mc))
    return out


# ---- generator ------------------------------------------------------------------------------

COND = {"0.5cm even": (0.005, 0.05, 0.02), "0.5cm uneven": (0.005, 0.12, 0.05),
        "1cm even": (0.01, 0.05, 0.02), "1cm uneven": (0.01, 0.12, 0.05)}


def _summary(rec):
    fr, trips = run_final(rec)
    reps = counted(trips)
    vals = all_rep_values(fr.sig, fr.raw_t, reps, rec.rest, DEFAULT.speed) if reps else []
    return summarize(rec.exercise, rec.fps, reps, vals)


def _row(c):
    return dict(kind=c.kind, value=c.value, res=c.result, reason=c.reason, d=c.delta, se=c.se, k=c.k,
                mc=c.min_change, na=c.n_a, nb=c.n_b)


def _group(args):
    cond, i, seed = args
    noise, dsd, rsd = COND[cond]
    ex = EXS[i % 3]
    rng = np.random.default_rng(seed)
    lo, hi = GEN["down"][ex]
    x = float(rng.uniform(lo, hi))
    base = Opts(down=(x, x), dur_rep_sd=dsd, rom_rep_sd=rsd)
    xt = float(rng.uniform(max(lo, 1.4), max(hi, 2.0)))
    opts = dict(R=base, R1=replace(base, reps=int(rng.integers(1, 3))), M0=base,
                M5=replace(base, down=(x + 0.5, x + 0.5)), M10=replace(base, down=(x + 1.0, x + 1.0)),
                T=replace(base, down=(xt, xt), down_trend=-1.0), S=replace(base, rom_last=(2, -0.04)))
    sm = {k: _summary(make_recording(ex, np.random.default_rng(seed * 10 + j), 30.0, noise, o))
          for j, (k, o) in enumerate(opts.items())}
    out = dict(ex=ex, x=x, r1=opts["R1"].reps, pairs={})
    for my, ref in [("M0", "R"), ("M5", "R"), ("M10", "R"), ("M0", "R1"), ("M5", "R1"), ("M10", "R1")]:
        out["pairs"][f"{my}-{ref}"] = [_row(c) for c in run_compare(sm[my], sm[ref]) if c.kind == "vs_ref"]
    for k in ("M0", "R", "T", "S"):
        out["pairs"][k] = [_row(compare_inset(sm[k], v)) for v in ("far_dist", "down")]
    out["scatter"] = {v: [sm[k].get(v).scatter() for k in ("M0", "R")] for v in ("down", "far_dist", "far_pause")}
    return out


def _rate(rows, value=None, old=False):
    """% 'different' among judged compares of `value` (None: report level, any value), judged share."""
    per_pair = []
    for pr in rows:
        cs = [c for c in pr if c["res"] != "skip" and (value is None or c["value"] == value)]
        if not cs:
            per_pair.append(None)
            continue
        if old:
            hit = any(judged(c["d"], c["se"], old_k(c), c["mc"]) for c in cs)
        else:
            hit = any(c["res"] == "different" for c in cs)
        per_pair.append(hit)
    j = [p for p in per_pair if p is not None]
    return dict(rate=100 * sum(j) / max(len(j), 1), judged=100 * len(j) / max(len(per_pair), 1), n=len(j))


def part_gen(groups, pool):
    jobs = [(cond, i, 1000 * (ci + 1) + i) for ci, cond in enumerate(COND) for i in range(groups)]
    t0 = time.time()
    res = pool.map(_group, jobs, chunksize=4)
    print("gen %d groups in %.0f s" % (len(res), time.time() - t0))
    out = {}
    for cond in COND:
        G = [g for (c, _, _), g in zip(jobs, res) if c == cond]
        P = lambda key: [g["pairs"][key] for g in G]
        P1 = lambda key, n: [g["pairs"][key] for g in G if g["r1"] == n]
        r = {}
        for rule, old in (("table", False), ("first", True)):
            r[rule] = {
                "vs ref report false": _rate(P("M0-R"), None, old),
                "vs ref down false": _rate(P("M0-R"), "down", old),
                "vs ref far_pause false": _rate(P("M0-R"), "far_pause", old),
                "vs ref down 0.5 s": _rate(P("M5-R"), "down", old),
                "vs ref down 1 s": _rate(P("M10-R"), "down", old),
                "short ref 1 rep down false": _rate(P1("M0-R1", 1), "down", old),
                "short ref 2 rep down false": _rate(P1("M0-R1", 2), "down", old),
                "short ref 1 rep down 0.5 s": _rate(P1("M5-R1", 1), "down", old),
                "short ref 1 rep down 1 s": _rate(P1("M10-R1", 1), "down", old),
                "short ref 2 rep down 0.5 s": _rate(P1("M5-R1", 2), "down", old),
                "short ref 2 rep down 1 s": _rate(P1("M10-R1", 2), "down", old),
                "inset far_dist false": _rate(P("M0") + P("R"), "far_dist", old),
                "inset down false": _rate(P("M0") + P("R"), "down", old),
                "inset report false": _rate(P("M0") + P("R"), None, old),
                "inset last 2 reps 4 cm less": _rate(P("S"), "far_dist", old),
                "inset down 1 s shorter": _rate(P("T"), "down", old),
            }
        sc = {}
        for v in ("down", "far_dist", "far_pause"):
            s = [x for g in G for x in g["scatter"][v] if x is not None]
            sc[v] = float(np.median(s)) if s else None
        sc["down_true"] = float(np.median([COND[cond][1] * g["x"] for g in G]))
        r["scatter"] = sc
        out[cond] = r
    json.dump(dict(groups=groups, cond={k: dict(noise=v[0], dur_rep_sd=v[1], rom_rep_sd=v[2]) for k, v in COND.items()},
                   res=out), open(os.path.join(RES, "cmp_gen.json"), "w"), indent=1)
    return out


# ---- report ---------------------------------------------------------------------------------

def report():
    L = ["# 비교 판정 시험 (2단계)", "", "`python -m sim.run_compare` 결과. 값은 %.", ""]
    p = os.path.join(RES, "cmp_table.json")
    if os.path.exists(p):
        t = json.load(open(p))
        L += [f"## 배수 표", "", f"문서 표 A·B·C {t['cells']}칸과 core/crit_table.json의 최대 차이 {t['max_diff']:.2f} "
              f"({'통과' if t['ok'] else '실패'}, 기준 0.1).", ""]
    p = os.path.join(RES, "cmp_normal.json")
    if os.path.exists(p):
        n = json.load(open(p))
        L += ["## 정규분포 시험 (실제 비교 코드)", "",
              f"(a) 하한·최소 변화량 없이, 칸마다 {n['N']:,}번. 값 하나의 거짓 \"다르다\" 최대 {n['pure_worst']:.2f}% "
              f"(기준 2.5% + 몬테카를로 2SE {n['mc_2se']:.2f}%p, {'통과' if n['pure_ok'] else '실패'}).", "",
              "| 칸 | 거짓 % |", "|---|---|"]
        for r in n["pure"]:
            name = f"세트 안 {r['n_me']}" if r["kind"] == "inset" else f"기준 {r['n_ref']} × 내 {r['n_me']}"
            L.append(f"| {name} | {r['new']:.2f} |")
        L += ["", "(b) 문서 표 \"처음 규칙과 지금 표\"와 같은 조건 (rep 편차 0.1 / 0.2 / 0.3초, 하한 0.07초, "
              "최소 변화량 0.15초). 왼쪽이 이번 값, 괄호가 문서 값.", "",
              "| 비교 | 거짓, 처음 규칙 | 거짓, 지금 표 | 1초 검출, 처음 | 1초 검출, 지금 | 0.5초 검출, 처음 | 0.5초 검출, 지금 |",
              "|---|---|---|---|---|---|---|"]
        for name, doc in DOC_RULES.items():
            kind, nu, nr = CELLS[name]
            cells = []
            for shift, (do, dn) in ((0.0, doc[0:2]), (1.0, doc[2:4]), (0.5, doc[4:6])):
                rs = [r for r in n["rules"] if r["kind"] == kind and r["n_me"] == nu and r["n_ref"] == nr
                      and r["shift"] == shift]
                rs.sort(key=lambda r: r["sd"])
                f = "%.1f" if shift == 0 else "%.0f"
                cells.append(" / ".join(f % r["old"] for r in rs) + " (" + " / ".join(f % v for v in do) + ")")
                cells.append(" / ".join(f % r["new"] for r in rs) + " (" + " / ".join(f % v for v in dn) + ")")
            L.append(f"| {name} | " + " | ".join(cells) + " |")
        L.append("")
    p = os.path.join(RES, "cmp_gen.json")
    if os.path.exists(p):
        g = json.load(open(p))
        conds = list(g["res"])
        L += ["## 생성기 시험 (전체 최종 파이프라인)", "",
              f"조건마다 {g['groups']}묶음 (종목 셋을 돌아가며). 네 숫자는 " + " / ".join(conds) +
              " 순서. even = rep 편차 작음(시간 5%, ROM 2%), uneven = 큼(12%, 5%). "
              "괄호 안 \"판정\"은 판정까지 간 쌍의 비율.", "",
              "| 시험 | 지금 표 | 처음 규칙 (같은 데이터) | 문서의 처음 규칙 (지워진 생성기) | 판정 |", "|---|---|---|---|---|"]
        for key in g["res"][conds[0]]["table"]:
            new = " / ".join("%.1f" % g["res"][c]["table"][key]["rate"] for c in conds)
            old = " / ".join("%.1f" % g["res"][c]["first"][key]["rate"] for c in conds)
            jd = " / ".join("%.0f" % g["res"][c]["table"][key]["judged"] for c in conds)
            L.append(f"| {key} | {new} | {old} | {DOC_GEN.get(key, '—')} | {jd} |")
        L += ["", "생성기의 rep 편차 (판정에 쓰는 2차 차분 산포 s의 세트별 중앙값, 같은 버릇 세트):", "",
              "| 조건 | 내리는 시간 s (초) | 설정값 (편차 × 내리는 시간, 중앙값) | 먼 점 거리 s (cm) | 먼 점에서 멈춘 시간 s (초) |",
              "|---|---|---|---|---|"]
        for c in conds:
            s = g["res"][c]["scatter"]
            L.append(f"| {c} | {s['down']:.3f} | {s['down_true']:.3f} | {100 * s['far_dist']:.2f} | {s['far_pause']:.3f} |")
        L.append("")
    open(os.path.join(RES, "compare_rules.md"), "w").write("\n".join(L))
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--groups", type=int, default=1500)
    ap.add_argument("--procs", type=int, default=os.cpu_count())
    a = ap.parse_args()
    os.makedirs(RES, exist_ok=True)
    with Pool(a.procs) as pool:
        if a.part in ("table", "all"):
            part_table()
        if a.part in ("normal", "all"):
            part_normal(a.n, pool)
        if a.part in ("gen", "all"):
            part_gen(a.groups, pool)
    report()


if __name__ == "__main__":
    main()
