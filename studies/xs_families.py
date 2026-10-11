"""Single-factor cross-sectional results for all 77 factors, broken out by family.

Answers the specific question: does the microstructure / liquidity family -- the best family in the
time-series study -- also work in the cross-section, now that it is represented by proper daily-data
estimators (Corwin-Schultz, Abdi-Ranaldo, Roll, Amihud, Hasbrouck/Kyle, Pastor-Stambaugh,
Bao-Pan-Wang, Lesmond et al.) instead of being omitted?

Null is the cross-sectional permutation (stock identities shuffled within each day), which is the
correct null here; a circular time shift is not, because stock characteristics are persistent over
years and a shifted panel inherits most of the real cross-sectional ranking.
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

import cross_section_study as XS
import xs_factor_library as XL

OUT = Path("outputs/full/xs")
HS = (1, 5, 20)
N_NULL = 3


def main():
    px, op, hi, lo, vo = XS.load()
    sec = pd.Series({c: XS.SECTOR[c] for c in px.columns})
    F0, r = XL.build(px, op, hi, lo, vo)
    Z = {k: XS.xs_rank_z(v) for k, v in F0.items()}
    ZS = {k: XS.sector_demean(v, sec) for k, v in Z.items()}
    split = int(np.searchsorted(px.index, XS.SPLIT))
    n = len(px.columns)

    perms = []
    for s in range(N_NULL):
        rp = np.random.default_rng(900 + s)
        perms.append(np.stack([rp.permutation(n) for _ in range(len(px))]))

    rows = []
    for h in HS:
        fwd = XS.fwd_ret(px, h)
        for k, v in ZS.items():
            icd = XS.ic_series(v.iloc[:split], fwd.iloc[:split])
            ict = XS.ic_series(v.iloc[split:], fwd.iloc[split:])
            sd_, st_ = XS.ic_stats(icd, h), XS.ic_stats(ict, h)
            sgn = np.sign(sd_["ic"]) if np.isfinite(sd_["ic"]) and sd_["ic"] != 0 else 1.0
            pnl = XS.ls_portfolio(v * sgn, r, h)
            # permutation null for this same factor
            nsr, nic = [], []
            for P in perms:
                vp = pd.DataFrame(np.take_along_axis(v.to_numpy(), P, axis=1), index=v.index, columns=v.columns)
                icn = XS.ic_series(vp.iloc[split:], fwd.iloc[split:])
                nic.append(XS.ic_stats(icn, h)["ic"])
                nsr.append(XS.sharpe(XS.ls_portfolio(vp * sgn, r, h).iloc[split:]))
            rows.append(dict(h=h, factor=k, family=XL.FAM_OF[k], ic_dev=sd_["ic"], ic_test=st_["ic"],
                             t_test=st_["t"], sr_dev=XS.sharpe(pnl.iloc[:split]),
                             sr_test=XS.sharpe(pnl.iloc[split:]),
                             ic_null=float(np.nanmean(nic)), sr_null=float(np.nanmean(nsr))))
    S = pd.DataFrame(rows)
    S["d_ic"] = S.ic_test.abs() - S.ic_null.abs()
    S["d_sr"] = S.sr_test - S.sr_null
    S.to_csv(OUT / "xs_families.csv", index=False)
    pd.set_option("display.width", 270); pd.set_option("display.max_rows", 400)

    print("=== by family (sector-neutral, real minus permutation null) ===")
    fam = S.groupby(["family", "h"]).agg(n=("factor", "size"), ic_test=("ic_test", lambda s: s.abs().mean()),
                                         d_ic=("d_ic", "mean"), share_t2=("t_test", lambda s: (s.abs() > 2).mean()),
                                         sr_test=("sr_test", "mean"), d_sr=("d_sr", "mean"))
    print(fam.round(3).to_string())
    print("\n=== family ranking averaged over horizons, by excess long-short Sharpe ===")
    print(S.groupby("family").agg(n=("factor", "size"), ic_test=("ic_test", lambda s: s.abs().mean()),
                                  d_ic=("d_ic", "mean"), sr_test=("sr_test", "mean"), d_sr=("d_sr", "mean"),
                                  dev_test_corr=("ic_dev", lambda s: np.corrcoef(
                                      s.fillna(0), S.loc[s.index, "ic_test"].fillna(0))[0, 1])
                                  ).sort_values("d_sr", ascending=False).round(3).to_string())
    print("\n=== microstructure factors specifically (spread + impact + noise), h=5 and 20 ===")
    micro = S[S.family.isin(["spread", "impact", "noise"]) & S.h.isin([5, 20])]
    print(micro.sort_values("d_sr", ascending=False)[
        ["h", "factor", "family", "ic_dev", "ic_test", "t_test", "sr_test", "sr_null", "d_sr"]]
        .head(18).round(3).to_string(index=False))
    print("\n=== top 20 single factors overall by excess Sharpe ===")
    print(S.sort_values("d_sr", ascending=False)[
        ["h", "factor", "family", "ic_dev", "ic_test", "t_test", "sr_test", "sr_null", "d_sr"]]
        .head(20).round(3).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(17, 6))
    ax = axes[0]
    g = S.groupby(["family", "h"]).d_sr.mean().unstack("h")
    g = g.loc[g.mean(axis=1).sort_values().index]
    yy = np.arange(len(g))
    for i, h in enumerate(HS):
        ax.barh(yy + (i - 1) * 0.26, g[h].values, 0.26, label=f"h={h}")
    ax.set_yticks(yy); ax.set_yticklabels(g.index)
    ax.axvline(0, color="k", lw=.6); ax.set_xlabel("decile long-short Sharpe minus permutation null")
    ax.set_title("cross-section: excess Sharpe by factor family"); ax.legend(fontsize=8); ax.grid(alpha=.3, axis="x")
    ax = axes[1]
    for f in S.family.unique():
        q = S[S.family == f]
        ax.scatter(q.ic_dev, q.ic_test, s=36, label=f, alpha=.8)
    lim = [S.ic_dev.min(), S.ic_dev.max()]
    ax.plot(lim, lim, ls=":", c="grey"); ax.axhline(0, color="k", lw=.6); ax.axvline(0, color="k", lw=.6)
    cc = np.corrcoef(S.ic_dev.fillna(0), S.ic_test.fillna(0))[0, 1]
    ax.set_xlabel("dev IC (2019-2022)"); ax.set_ylabel("test IC (2023-2026)")
    ax.set_title(f"cross-sectional IC persists: corr = {cc:.2f}\n(the timing study's equivalent was 0.02-0.16)")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(OUT / "fig_xs_families.png", dpi=110); plt.close()
    print("\nfigure written")
    return S


if __name__ == "__main__":
    main()
