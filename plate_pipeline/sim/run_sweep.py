"""Threshold sweeps and rule-off runs (근거·출처 tab, 임계값 표와 규칙을 끈 표). 30 fps, noise 1 cm.

python -m sim.run_sweep [--sets 600]
Writes results/sweep.json.
"""
import argparse
import json
import os
import time
import zlib
from dataclasses import replace
from multiprocessing import Pool

import numpy as np

from core.config import DEFAULT
from sim.gen import make_recording
from sim.evaluate import run_final, run_live
from sim.scenarios import SCENARIOS, EXTRA
from core.judge.sets import counted

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALLS = {**{k: (v[0], v[1]) for k, v in SCENARIOS.items()}, **EXTRA}


def cfg_with(rep=None, final=None):
    c = DEFAULT
    if rep:
        c = replace(c, rep=replace(c.rep, **rep))
    if final:
        c = replace(c, final=replace(c.final, **final))
    return c


# (label, situation, list of (setting label, rep overrides, final overrides), doc final, doc live)
SWEEPS = [
    ("복귀율", "락아웃이 0~15% 모자란 스쿼트·벤치", [("0.6", {"ret": 0.6}, None), ("0.7", {"ret": 0.7}, None), ("0.8", {"ret": 0.8}, None), ("0.9", {"ret": 0.9}, None)], [1.000, 1.000, 0.993, 0.438], None),
    ("복귀율", "마지막 rep을 곧장 랙에 거는 벤치", [("0.6", {"ret": 0.6}, None), ("0.7", {"ret": 0.7}, None), ("0.8", {"ret": 0.8}, None)], [1.000, 0.998, 0.918], None),
    ("최소 폭", "ROM 0.15~0.22 m인 스쿼트·벤치", [("0.10", {"min_amp_m": 0.10}, None), ("0.13", {"min_amp_m": 0.13}, None), ("0.16", {"min_amp_m": 0.16}, None)], [1.000, 1.000, 0.917], None),
    ("최소 폭", "벤치: rep 사이에 바를 8~12 cm 고쳐 잡음", [("0.10", {"min_amp_m": 0.10}, None), ("0.13", {"min_amp_m": 0.13}, None), ("0.16", {"min_amp_m": 0.16}, None)], [0.982, 0.997, 1.000], None),
    ("최소 폭", "벤치: 고쳐 잡음 10 cm, ROM 0.15~0.22 m", [("0.10", {"min_amp_m": 0.10}, None), ("0.13", {"min_amp_m": 0.13}, None), ("0.16", {"min_amp_m": 0.16}, None)], [0.377, 0.837, 0.802], None),
    ("진입 폭", "기본 세트", [("0.03", {"entry_m": 0.03}, None), ("0.06", {"entry_m": 0.06}, None), ("0.09", {"entry_m": 0.09}, None)], [1.0, 1.0, 1.0], None),
    ("최소 시간", "벤치: rep당 0.85~1.21초로 쉬지 않고 반복", [("없음", {"bounce_s": 0.0}, None), ("0.3", {"bounce_s": 0.3}, None), ("0.4", {"bounce_s": 0.4}, None), ("0.5", {"bounce_s": 0.5}, None)], [0.940, 0.940, 0.940, 0.938], [0.918, 0.910, 0.885, 0.863]),
    ("최소 시간", "데드: 마지막 rep을 떨어뜨려 18~30 cm 튐", [("없음", {"bounce_s": 0.0}, None), ("0.3", {"bounce_s": 0.3}, None), ("0.4", {"bounce_s": 0.4}, None), ("0.5", {"bounce_s": 0.5}, None)], None, [1.0, 1.0, 1.0, 1.0]),
    ("최소 시간", "데드: 첫 rep 전에 바가 18~30 cm 튐", [("없음", {"bounce_s": 0.0}, None), ("0.3", {"bounce_s": 0.3}, None), ("0.4", {"bounce_s": 0.4}, None), ("0.5", {"bounce_s": 0.5}, None)], [0.875, 0.875, 0.897, 0.988], [0.022, 0.028, 0.372, 0.815]),
    ("공백 기준", "세트 중간에 부분 rep 1개", [("0.4", {"gap_s": 0.4}, {"interp_s": 0.4}), ("0.5", {"gap_s": 0.5}, {"interp_s": 0.5}), ("0.6", {"gap_s": 0.6}, {"interp_s": 0.6})], [0.988, 0.993, 1.000], None),
    ("공백 기준", "벤치: rep당 0.85~1.21초로 쉬지 않고 반복", [("0.4", {"gap_s": 0.4}, {"interp_s": 0.4}), ("0.5", {"gap_s": 0.5}, {"interp_s": 0.5}), ("0.6", {"gap_s": 0.6}, {"interp_s": 0.6})], [0.948, 0.940, 0.928], None),
    ("공백 기준", "rep 도중 원판이 1~2초 가림", [("0.4", {"gap_s": 0.4}, {"interp_s": 0.4}), ("0.5", {"gap_s": 0.5}, {"interp_s": 0.5}), ("0.6", {"gap_s": 0.6}, {"interp_s": 0.6})], [0.908, 0.909, 0.910], None),
    ("규칙을 끔", "데드: 마지막 rep을 떨어뜨려 8~18 cm 튐", [("지금", {}, None), ("바운스 끔", {"bounce_s": 0.0}, None), ("부분 끔", {"partial": 0.0}, None), ("둘 다 끔", {"bounce_s": 0.0, "partial": 0.0}, None)], [1.000, 1.000, 1.000, 0.990], [1.000, 1.000, 1.000, 0.998]),
    ("규칙을 끔", "데드: 마지막 rep을 떨어뜨려 18~30 cm 튐", [("지금", {}, None), ("바운스 끔", {"bounce_s": 0.0}, None), ("부분 끔", {"partial": 0.0}, None), ("둘 다 끔", {"bounce_s": 0.0, "partial": 0.0}, None)], [0.997, 0.992, 0.995, 0.282], [1.000, 1.000, 0.985, 0.322]),
    ("규칙을 끔", "데드: 첫 rep 전에 바가 18~30 cm 튐", [("지금", {}, None), ("바운스 끔", {"bounce_s": 0.0}, None), ("부분 끔", {"partial": 0.0}, None), ("둘 다 끔", {"bounce_s": 0.0, "partial": 0.0}, None)], [0.897, 0.875, 0.488, 0.013], [0.372, 0.022, 0.372, 0.022]),
]


def _job(args):
    sit, ex, i, rep_o, fin_o = args
    exs, factory = ALLS[sit]
    rng = np.random.default_rng(zlib.crc32(f"{sit}|{ex}|{i}".encode()))
    o = factory(ex, rng)
    rec = make_recording(ex, rng, 30.0, 0.01, o)
    cfg = cfg_with(rep_o, fin_o)
    _, trips = run_final(rec, cfg)
    live = run_live(rec, cfg)
    n = len(rec.reps)
    return len(counted(trips)) == n, len(counted(live)) == n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", type=int, default=600)
    args = ap.parse_args()
    res = []
    t0 = time.time()
    with Pool(os.cpu_count()) as pool:
        for label, sit, settings, doc_f, doc_l in SWEEPS:
            exs = ALLS[sit][0]
            row = dict(label=label, situation=sit, settings=[], doc_final=doc_f, doc_live=doc_l)
            for name, rep_o, fin_o in settings:
                jobs = [(sit, ex, i, rep_o, fin_o) for ex in exs for i in range(args.sets // len(exs) if len(exs) > 1 else args.sets)]
                rows = pool.map(_job, jobs, chunksize=8)
                f = float(np.mean([r[0] for r in rows])); l = float(np.mean([r[1] for r in rows]))
                row["settings"].append(dict(name=name, sets=len(rows), final=f, live=l))
            res.append(row)
            print(label, sit, [(s["name"], round(s["final"], 3), round(s["live"], 3)) for s in row["settings"]],
                  "doc", doc_f, doc_l, f"[{time.time()-t0:.0f}s]", flush=True)
    json.dump(res, open(os.path.join(HERE, "results", "sweep.json"), "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
