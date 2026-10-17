"""Build the multiplier table (core/crit_table.json).

For no true difference and rep values that are independent normal, |delta| / SE is simulated
(400,000 draws per cell, no scatter floor, no minimum change) and its (1 - alpha) quantile is the
multiplier, rounded up to 0.1. Short reference (1-2 values): the larger of the 1- and 2-value cells.
Same code as the design doc (근거·출처 tab, "판정 배수 다시 정하기"); seed 2026 reproduces it.

python -m sim.crit [--alpha 0.025]
"""
import argparse
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def scat(x):
    d2 = x[:, 2:] - 2 * x[:, 1:-1] + x[:, :-2]
    return np.sqrt((d2 ** 2).sum(1) / (6 * d2.shape[1]))


def build(alpha=0.025, N=400_000, seed=2026):
    rng = np.random.default_rng(seed)
    p = 1 - alpha

    def q(T):
        return np.quantile(np.abs(T), p)
    A = {n: q((x[:, -2:].mean(1) - x[:, :2].mean(1)) / scat(x))
         for n in range(4, 13) for x in [rng.normal(size=(N, n))]}
    C, B = {}, {}
    for nu in range(3, 11):
        a = rng.normal(size=(N, nu)); sa, ma = scat(a), np.median(a, 1)
        for nr in (1, 2):
            b = rng.normal(size=(N, nr))
            C[nu, nr] = q((ma - np.median(b, 1)) / (sa * np.sqrt(1 / nu + 1 / nr)))
        for nr in range(3, 11):
            b = rng.normal(size=(N, nr))
            B[nu, nr] = q((ma - np.median(b, 1)) / np.sqrt(sa ** 2 / nu + scat(b) ** 2 / nr))
    up = lambda v: math.ceil(v * 10 - 1e-9) / 10
    return dict(alpha=alpha, n_draws=N, seed=seed,
                inset={str(n): up(v) for n, v in A.items()},
                short={str(nu): up(max(C[nu, 1], C[nu, 2])) for nu in range(3, 11)},
                ref={f"{a},{b}": up(v) for (a, b), v in B.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.025)
    args = ap.parse_args()
    t = build(args.alpha)
    name = "crit_table.json" if args.alpha == 0.025 else f"crit_table_{args.alpha:g}.json"
    json.dump(t, open(os.path.join(HERE, "core", name), "w"), indent=1)
    print(json.dumps({k: t[k] for k in ("inset", "short")}))


if __name__ == "__main__":
    main()
