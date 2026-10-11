"""Test the 东北证券 volume-distribution factors (pdf_volume_factors) where intraday data exists.

The report's factors need 5-minute bars.  We have those for SPY and QQQ only, so this is a TIME-SERIES
test (does the factor predict the index's own forward return) rather than the report's cross-sectional
stock-selection test.  It still answers two things:
  1. are the factors implemented correctly and do they carry any signal at all?
  2. does the report's STABILITY operator add to the level factor, as it does in the report
     (where e.g. 早盘成交量占比 ICIR goes 0.54 -> 1.06 when switched to its stability version)?

Null: the factor column is circularly shifted in time by >= 1 year (correct null for a time-series test).
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pdf_volume_factors as PV
import timing_transform_study as T
from factor_sharpe import run_book
from index_timing_framework import load_market
from multifrequency_models import target
from run_full_extended import config_from

OUT = Path("outputs/full/pdf"); OUT.mkdir(parents=True, exist_ok=True)
HS = (1, 2, 5, 10, 20)
N_NULL = 3
SPLIT = "2023-01-03"


def main():
    cfg = config_from(Path("data"))
    rows = []
    for sym in ("SPY", "QQQ"):
        panel = load_market(f"data/{sym}_5min.csv", cfg)
        F, _ = PV.build(f"data/{sym}_5min.csv", panel, cfg, market="US")
        split = int(np.searchsorted(panel.dates, SPLIT))
        n = len(panel.dates)
        rngs = [np.random.default_rng(31 + i) for i in range(N_NULL)]
        shifts = [int(r.integers(252, n - 252)) for r in rngs]
        for h in HS:
            y = target(panel, h, "next_open")[0]
            lab = (np.arange(n) + h < split) & np.isfinite(y)
            tst = np.arange(n) >= split
            for name in F.columns:
                base = F[name].to_numpy(float)
                def book(col):
                    pct, mu, sd, med, mad = T.prior_stats(col)
                    with np.errstate(all="ignore"):
                        z = np.clip((col - med) / mad, -3, 3) / 3      # robust z, the best map found
                    m = lab & np.isfinite(z)
                    if m.sum() < 200 or np.nanstd(z[m]) == 0:
                        return None, None, None, None
                    ic_dev = T.spearman(z[m], y[m])
                    sg = float(np.sign(ic_dev) or 1.0)
                    w = np.clip(T.smooth(z * sg, h), -1.5, 1.5)
                    r = run_book(panel, pd.Series(np.nan_to_num(w), index=panel.dates), fee_bp=1.0)
                    mt = tst & np.isfinite(y) & np.isfinite(z)
                    return ic_dev, T.spearman(z[mt], y[mt]), T.sr(r.loc["2023":]), sg
                ic_dev, ic_test, sr_test, sg = book(base)
                if ic_dev is None:
                    continue
                nulls = [book(np.roll(base, s))[2] for s in shifts]
                rows.append(dict(symbol=sym, factor=name, h=h,
                                 is_stab=name.endswith("_STAB"),
                                 family=("stability" if name.endswith("_STAB") else "level"),
                                 dev_sign=sg, report_sign=PV.REPORT_SIGN.get(name, np.nan),
                                 ic_dev=ic_dev, ic_test=ic_test, sr_test=sr_test,
                                 sr_null=float(np.nanmean(nulls))))
    S = pd.DataFrame(rows)
    S["d_sr"] = S.sr_test - S.sr_null
    S.to_csv(OUT / "pdf_factor_timing.csv", index=False)
    pd.set_option("display.width", 270); pd.set_option("display.max_rows", 400)

    print("=== level vs stability versions (mean over factors, symbols) ===")
    print(S.groupby(["family", "h"]).agg(n=("factor", "size"), ic_test=("ic_test", lambda s: s.abs().mean()),
                                         sr_test=("sr_test", "mean"), sr_null=("sr_null", "mean"),
                                         d_sr=("d_sr", "mean")).round(3).to_string())
    print("\n=== does the dev-period sign agree with the report's stated (negative) direction? ===")
    lev = S[~S.is_stab & S.report_sign.notna()]
    print(f"agreement: {(lev.dev_sign == lev.report_sign).mean():.1%} of {len(lev)} level cells")
    print(lev.groupby("factor").apply(lambda g: pd.Series({
        "agree": (g.dev_sign == g.report_sign).mean(), "ic_dev": g.ic_dev.mean()})).round(3).to_string())
    print("\n=== per-factor excess Sharpe vs null, pooled over horizons, both symbols must agree ===")
    def cons(g):
        per = g.groupby("symbol").d_sr.mean()
        return pd.Series(dict(d_sr=per.mean(), worst=per.min() if len(per) == 2 else np.nan,
                              ic_test=g.ic_test.abs().mean(), sr_test=g.sr_test.mean()))
    tab = S.groupby("factor").apply(cons).sort_values("worst", ascending=False)
    print(tab.round(3).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    ax = axes[0]
    g = S.groupby(["family", "h"]).d_sr.mean().unstack("family")
    xx = np.arange(len(g))
    for i, c in enumerate(g.columns):
        ax.bar(xx + (i - 0.5) * 0.35, g[c].values, 0.35, label=c)
    ax.set_xticks(xx); ax.set_xticklabels(g.index); ax.set_xlabel("horizon h")
    ax.axhline(0, color="k", lw=.6); ax.set_ylabel("excess LS Sharpe vs null")
    ax.set_title("report factors on SPY/QQQ: level vs stability operator"); ax.legend(); ax.grid(alpha=.3, axis="y")
    ax = axes[1]
    t2 = tab.dropna(subset=["worst"]).sort_values("worst")
    ax.barh(np.arange(len(t2)), t2.worst.values,
            color=["tab:green" if v > 0 else "tab:red" for v in t2.worst])
    ax.set_yticks(np.arange(len(t2))); ax.set_yticklabels(t2.index, fontsize=8)
    ax.axvline(0, color="k", lw=.6)
    ax.set_xlabel("excess Sharpe vs null, worse of the two symbols")
    ax.set_title("per factor (conservative: worse symbol)"); ax.grid(alpha=.3, axis="x")
    plt.tight_layout(); plt.savefig(OUT / "fig_pdf_factors.png", dpi=110); plt.close()
    print("\nfigure written")
    return S


if __name__ == "__main__":
    main()
