"""Single-factor timing Sharpe (SPY / QQQ, 2019-01 .. 2026-10).

Every cached (orthogonalised) factor column -> a tradable position, PIT:
  s_t   = 2 * (trailing-252d percentile of F_t, past values only) - 1   in [-1, 1]
  sign  = sign of the factor's Pearson IC with the h-day next_open return, estimated on
          2019-2022 MATURED labels only (frozen; never uses the test period)
  w_t   = sign * mean(s_{t-h+1..t})   (overlapping-portfolio smoothing matching horizon h)
Portfolios (next_open execution: decision at close t, the gap is held at the old weight, the
new weight earns open->close of t+1; 1bp per unit of turnover):
  LS    pure timing overlay, weight w in [-1, 1]          (excess of cash, beta ~ 0)
  LC    long / cash, weight 1{w>0}                        (comparable with buy & hold)
  TILT  0.5 + 0.5 w  in [0, 1]                            (average exposure ~0.5 + timing)
Reported per period: dev 2019-22 (sign estimation period, in-sample) and test 2023-26.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from index_timing_framework import load_market  # noqa: E402
from multifrequency_models import target  # noqa: E402
from run_full_extended import config_from  # noqa: E402

OUT = Path("outputs/full"); REP = OUT / "report"
HS = (5, 20)
SPLIT = "2023-01-03"


def pct_past(x: pd.Series, n=252, minp=126) -> pd.Series:
    return x.rolling(n, min_periods=minp).apply(lambda a: (a[:-1] < a[-1]).mean() + 0.5 * (a[:-1] == a[-1]).mean()
                                                if np.isfinite(a[-1]) else np.nan, raw=True)


def run_book(panel, w: pd.Series, fee_bp=1.0):
    o, c = panel.daily_open, panel.close
    w = w.reindex(panel.dates).fillna(0.0).to_numpy()
    on = np.r_[np.nan, o[1:] / c[:-1] - 1]; intra = c / o - 1
    prev = np.r_[0.0, w[:-1]]          # weight decided at t-1 close, filled at t open
    prev2 = np.r_[0.0, prev[:-1]]      # weight held through the t gap
    r = (1 + prev2 * np.nan_to_num(on)) * (1 + prev * intra) - 1 - fee_bp * 1e-4 * np.abs(prev - prev2)
    return pd.Series(r, index=pd.to_datetime(panel.dates))


def sr(x):
    x = x.dropna(); return float(x.mean() / x.std() * np.sqrt(252)) if x.std() > 0 else np.nan


def main():
    cfg = config_from(Path("data"))
    rows, keep = [], {}
    for sym in ("SPY", "QQQ"):
        panel = load_market(f"data/{sym}_5min.csv", cfg)
        X = pd.concat([pd.read_pickle(OUT / "cache" / f"{sym}_A.pkl"), pd.read_pickle(OUT / "cache" / f"{sym}_M.pkl")], axis=1)
        split = int(np.searchsorted(panel.dates, SPLIT))
        bh = run_book(panel, pd.Series(1.0, index=panel.dates))
        P = X.apply(pct_past)
        for h in HS:
            y, mat = target(panel, h, "next_open")
            lab = (mat < split) & np.isfinite(y)
            for col in X.columns:
                x = X[col].to_numpy(float); m = lab & np.isfinite(x)
                if m.sum() < 200 or np.std(x[m]) == 0:
                    continue
                sign = np.sign(np.corrcoef(x[m], y[m])[0, 1])
                s = (2 * P[col] - 1).rolling(h, min_periods=max(1, h // 2)).mean() * sign
                books = {"LS": s.clip(-1, 1), "LC": (s > 0).astype(float).where(s.notna(), 0.0), "TILT": 0.5 + 0.5 * s.clip(-1, 1)}
                name = f"{col[0].replace('M:', '[M]')}:{col[1]}"
                for b, w in books.items():
                    r = run_book(panel, w)
                    rec = {"symbol": sym, "factor": name, "macro": col[0].startswith("M:"), "h": h, "book": b, "sign": sign,
                           "sr_dev": sr(r.loc[:"2022"]), "sr_test": sr(r.loc["2023":]), "sr_all": sr(r),
                           "ret_test": float(r.loc["2023":].mean() * 252), "vol_test": float(r.loc["2023":].std() * np.sqrt(252)),
                           "bh_sr_test": sr(bh.loc["2023":]), "bh_sr_dev": sr(bh.loc[:"2022"]),
                           "corr_bh_test": float(r.loc["2023":].corr(bh.loc["2023":]))}
                    rows.append(rec)
                    keep[(sym, name, h, b)] = r
        keep[(sym, "BH", 0, "BH")] = bh
    df = pd.DataFrame(rows); df.to_csv(REP / "factor_sharpe.csv", index=False)
    pd.to_pickle(keep, OUT / "cache" / "factor_books.pkl")
    return df, keep


if __name__ == "__main__":
    df, keep = main()
    pd.set_option("display.width", 250)
    for b in ("LS", "LC", "TILT"):
        d = df[df.book == b]
        q = d.groupby(["symbol", "h"]).agg(n=("sr_test", "size"), dev_median=("sr_dev", "median"), test_median=("sr_test", "median"),
                                           test_p90=("sr_test", lambda s: s.quantile(.9)), share_test_gt0=("sr_test", lambda s: (s > 0).mean()),
                                           share_test_gt_bh=("sr_test", lambda s: (s > d.loc[s.index, "bh_sr_test"]).mean()),
                                           bh_test=("bh_sr_test", "first"))
        print(f"\n=== book {b}: distribution of single-factor Sharpe ===\n", q.round(3).to_string())
        # factor kept if dev Sharpe > 0.5 on BOTH symbols (selection on dev only) -> test
        piv = d.pivot_table(index=["factor", "h"], columns="symbol", values=["sr_dev", "sr_test"])
        sel = piv[(piv["sr_dev"]["SPY"] > 0.5) & (piv["sr_dev"]["QQQ"] > 0.5)]
        print(f"  dev-selected (dev SR>0.5 on both): {len(sel)} factors; their test SR median SPY {sel['sr_test']['SPY'].median():.2f}, "
              f"QQQ {sel['sr_test']['QQQ'].median():.2f}; share test>0: {(sel['sr_test']>0).mean().round(2).to_dict()}")
        top = sel.assign(dev=sel["sr_dev"].mean(axis=1), test=sel["sr_test"].mean(axis=1)).sort_values("dev", ascending=False).head(15)
        print(top.round(2).to_string())
