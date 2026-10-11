"""Null distribution for every (map, agg, h): REAL factors circularly shifted in time by a random offset
(>= 1 year), which keeps each factor's distribution / persistence / missing pattern but breaks any link with
future returns.  Also iid / AR(1) noise.  Compare the real-factor results with this null:
a map whose mean test Sharpe is negative on the null has a mechanical bias (e.g. bet-size feedback of
recursively estimated slopes), not negative information.
"""
import numpy as np
import pandas as pd
from multiprocessing import Pool

import timing_transform_study as T

SHIFTS_PER_FACTOR = 2
PHIS = (0.0, 0.9, 0.99)
NOISE_NAMES = [f"P:{i}_{PHIS[i % 3]}" for i in range(60)]


def init():
    T._init()
    for s in ("SPY", "QQQ"):
        rng = np.random.default_rng(0 if s == "SPY" else 1)
        F = T._G[s]["F"]; n = len(F); cols = {}
        for c in list(F.columns):
            for j in range(SHIFTS_PER_FACTOR):
                k = int(rng.integers(252, n - 252))
                cols[f"S{j}:{c}"] = np.roll(F[c].to_numpy(), k)
        for nm in NOISE_NAMES:
            phi = float(nm.split("_")[-1]); e = rng.standard_normal(n); x = np.zeros(n)
            for t in range(1, n):
                x[t] = phi * x[t - 1] + e[t]
            cols[nm] = x
        T._G[s]["F"] = pd.concat([F, pd.DataFrame(cols, index=F.index)], axis=1)


def job(nm):
    return T.factor_job(nm)[0]


if __name__ == "__main__":
    init()
    names = [c for c in T._G["SPY"]["F"].columns if c[:2] in ("S0", "S1") or c.startswith("P:")]
    print(len(names), "placebo series", flush=True)
    rows = []
    with Pool(8, initializer=init) as p:
        for i, r in enumerate(p.imap_unordered(job, names, chunksize=2)):
            rows += r
            if i % 50 == 0: print(i, flush=True)
    df = pd.DataFrame(rows)
    df["null"] = np.where(df.factor.str.startswith("P:"), "noise", "shift")
    df.to_parquet("outputs/full/transform/placebo.parquet")
    print(df.groupby(["null", "h", "map"]).sr_test.mean().unstack([0, 1]).round(3).to_string())
