"""Stage 7: why the sleeve did not convert into return, and the construction that does convert it.

Diagnosis from stage 6: tilting the EXPOSURE of a long equity core with the sleeve adds ~0 Sharpe versus
constant leverage at the same average exposure.  The reason is arithmetic, not statistical:

    sleeve beta to the core is 0.20-0.31, sleeve alpha is 1-6%/yr.
    Raising exposure from 1.0 to 1.3 adds ~30% x 32% = 9.6pp of pure beta return.
    So the beta term is an order of magnitude larger than the alpha term and drowns it, while the
    leveraged-ETF drag (-4.4pp at 2x, -8.5pp at 3x) eats what is left.

The fix is to stop using the sleeve to modulate beta and instead hold it as a SEPARATE position whose
beta has been removed, then size core and sleeve by risk.  Two steps, neither of which estimates a return:

  1. beta hedge     sleeve_hedged_t = sleeve_t - beta_t * core_t, beta_t = trailing 126d OLS beta
                    (known at t-1).  This is what makes the sleeve diversifying rather than a beta proxy.
  2. risk parity    w_core / w_sleeve so that both contribute equal trailing risk, then scale the whole
                    thing to a volatility target and implement the resulting exposure on the real
                    1x/2x/3x ETF ladder.

Theory check: with core SR 1.50 and sleeve SR 1.30 at correlation rho, the equal-risk combination has
SR = (1.50+1.30)/sqrt(2+2 rho).  rho = 0.48 (unhedged) gives 1.63; rho = 0 (hedged) gives 1.98.
That is the "room for improvement" being tested here.
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

import leveraged_portfolio as LP
import timing_model_strategy as MS
from factor_sharpe import run_book

OUT = Path("outputs/full/strategy")
HS = LP.SLEEVE_HS
FEE = 1.0
POS_CACHE = OUT / "sleeve_positions.pkl"


def pit_beta(sleeve: pd.Series, core: pd.Series, win=126):
    cov = sleeve.rolling(win, min_periods=63).cov(core)
    var = core.rolling(win, min_periods=63).var()
    return (cov / var).shift(1).fillna(0.0).clip(-1.5, 1.5)


def roll_vol(r: pd.Series, win=63):
    return (r.rolling(win, min_periods=40).std() * np.sqrt(252)).shift(1)


def main():
    MS._init()
    lev_px = pd.read_csv("data/lev/levetf_close.csv", index_col=0)
    dates = MS._G["SPY"]["panel"].dates
    didx = pd.to_datetime(dates)

    if POS_CACHE.exists():
        POS = pd.read_pickle(POS_CACHE)
    else:
        POS = {(sym, sh): LP.sleeve_positions(sym, sh) for sym in ("SPY", "QQQ") for sh in (0, 1)}
        pd.to_pickle(POS, POS_CACHE)

    rows, curves = [], {}
    for sh in (0, 1):
        tag = "" if sh == 0 else " [null]"
        for core_sym in ("SPY", "QQQ"):
            panel = MS._G[core_sym]["panel"]
            core = run_book(panel, pd.Series(1.0, index=panel.dates), fee_bp=FEE)
            # ---- 1. risk-parity blend of the four horizon sleeves (inverse trailing vol, PIT)
            sr_pos = POS[(core_sym, sh)]
            rets = {h: run_book(panel, pd.Series(np.nan_to_num(p), index=panel.dates), fee_bp=FEE)
                    for h, p in sr_pos.items()}
            iv = pd.DataFrame({h: 1.0 / roll_vol(rets[h]) for h in HS})
            iv = iv.div(iv.sum(1), axis=0).fillna(1.0 / len(HS))
            blend_pos = sum(iv[h].to_numpy() * np.nan_to_num(sr_pos[h]) for h in HS)
            sleeve = run_book(panel, pd.Series(blend_pos, index=panel.dates), fee_bp=FEE)
            # ---- 2. beta hedge
            beta = pit_beta(sleeve, core)
            hedged = sleeve - beta * core
            for nm, s in [("sleeve raw", sleeve), ("sleeve beta-hedged", hedged)]:
                st, ct = s.loc["2023":], core.loc["2023":]
                rows.append(dict(group="sleeve", core=core_sym, name=f"{core_sym} {nm}" + tag, shift=sh,
                                 corr_core=float(st.corr(ct)), **LP.metrics(st, ct)))
            # ---- 3. risk parity core vs hedged sleeve, then vol target on the ETF ladder
            for use_hedge, hname in [(True, "hedged"), (False, "raw")]:
                sl = hedged if use_hedge else sleeve
                v_core, v_sl = roll_vol(core), roll_vol(sl)
                k = (v_core / v_sl).clip(0, 6).fillna(0)        # equal risk: k units of sleeve per unit core
                for vt in (0.0, 0.15, 0.20, 0.25):
                    scale = (vt / v_core).clip(0.2, 3.0).fillna(1.0) if vt else pd.Series(1.0, index=core.index)
                    # core leg traded on the ETF ladder at exposure scale; sleeve leg is a separate
                    # long/short book on the 1x ETF, sized k*scale
                    E = scale.to_numpy()
                    core_leg = LP.ladder_returns(core_sym, E, dates, lev_px)
                    w_sl = (k * scale).to_numpy() * blend_pos - (beta * k * scale).to_numpy() * (1 if use_hedge else 0)
                    sl_leg = run_book(panel, pd.Series(np.nan_to_num(w_sl), index=panel.dates), fee_bp=FEE)
                    r = core_leg + sl_leg
                    tot_E = E + np.abs(np.nan_to_num(w_sl))
                    nm = f"{core_sym} core + risk-parity {hname} sleeve" + (f", vt={vt:.0%}" if vt else ", no vt")
                    m = LP.metrics(r.loc["2023":], core.loc["2023":])
                    # exposure-matched constant-leverage control
                    rc = LP.ladder_returns(core_sym, np.full(len(dates), float(np.nanmean(E[MS._G[core_sym]['split']:]))),
                                           dates, lev_px)
                    mc = LP.metrics(rc.loc["2023":], core.loc["2023":])
                    rows.append(dict(group="portfolio", core=core_sym, name=nm + tag, shift=sh, hedge=use_hedge,
                                     vol_target=vt, avg_gross=float(np.nanmean(tot_E[MS._G[core_sym]["split"]:])),
                                     sr_matched=mc["sharpe"], d_sr_vs_matched=m["sharpe"] - mc["sharpe"], **m))
                    if sh == 0:
                        curves[nm] = r
            if sh == 0:
                curves[f"{core_sym} buy & hold"] = core
                rows.append(dict(group="benchmark", core=core_sym, name=f"{core_sym} buy & hold", shift=0,
                                 **LP.metrics(core.loc["2023":], core.loc["2023":])))
                curves[f"{core_sym} sleeve beta-hedged"] = hedged

    for c in ("TQQQ", "QLD"):
        r = lev_px[c].reindex(dates).pct_change(); r.index = didx
        base = lev_px["QQQ"].reindex(dates).pct_change(); base.index = didx
        rows.append(dict(group="benchmark", name=f"{c} buy & hold", shift=0, **LP.metrics(r.loc["2023":], base.loc["2023":])))
        curves[f"{c} buy & hold"] = r

    tab = pd.DataFrame(rows); tab.to_csv(OUT / "portfolio_construction_metrics.csv", index=False)
    pd.to_pickle(curves, OUT / "portfolio_construction_curves.pkl")
    pd.set_option("display.width", 290); pd.set_option("display.max_rows", 400)
    print("\n=== sleeves: effect of the beta hedge ===")
    print(tab[tab.group == "sleeve"][["name", "sharpe", "ann_ret", "vol", "mdd", "beta", "alpha", "ir", "corr_core"]]
          .round(3).to_string(index=False))
    print("\n=== benchmarks ===")
    print(tab[tab.group == "benchmark"][["name", "sharpe", "ann_ret", "vol", "mdd", "calmar"]].round(3).to_string(index=False))
    print("\n=== core + risk-parity sleeve ===")
    c = ["name", "sharpe", "ann_ret", "vol", "mdd", "calmar", "excess_ann", "alpha", "ir", "avg_gross",
         "sr_matched", "d_sr_vs_matched"]
    print(tab[(tab.group == "portfolio") & (tab["shift"] == 0)][c].round(3).to_string(index=False))
    print("\n=== null ===")
    print(tab[(tab.group == "portfolio") & (tab["shift"] == 1)][["name", "sharpe", "ann_ret", "mdd", "d_sr_vs_matched"]]
          .round(3).to_string(index=False))

    # ----------------------------------------------------------------- figures (always with benchmarks)
    BENCH = ["SPY buy & hold", "QQQ buy & hold", "QLD buy & hold", "TQQQ buy & hold"]
    pick = [n for n in curves if "risk-parity hedged" in n and ("no vt" in n or "vt=20%" in n)]
    fig, axes = plt.subplots(2, 2, figsize=(18, 10), gridspec_kw=dict(height_ratios=[3, 1.3]))
    for j, sym in enumerate(["SPY", "QQQ"]):
        ax, axd = axes[0, j], axes[1, j]
        show = [b for b in BENCH if sym in b or b in ("SPY buy & hold", "QQQ buy & hold")] + \
               [n for n in pick if n.startswith(sym)] + [f"{sym} sleeve beta-hedged"]
        for n in show:
            r = curves[n].loc["2023":]
            eq = (1 + r).cumprod(); dd = eq / eq.cummax() - 1
            m = LP.metrics(r)
            isb = "buy & hold" in n
            ax.plot(eq, lw=1.8 if isb else 1.2, ls="--" if isb else "-",
                    label=f"{n}: SR {m['sharpe']:.2f}, ann {m['ann_ret']:.1%}, MDD {m['mdd']:.1%}")
            axd.plot(dd, lw=1.2 if isb else .9, ls="--" if isb else "-")
        ax.set_title(f"{sym}: core + beta-hedged risk-parity sleeve vs index benchmarks (test 2023-2026)")
        ax.legend(fontsize=8); ax.grid(alpha=.3); ax.set_yscale("log")
        axd.set_title("drawdown"); axd.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(OUT / "fig_portfolio_construction.png", dpi=110); plt.close()

    # risk/return scatter with benchmarks labelled
    fig, ax = plt.subplots(figsize=(10, 7))
    P = tab[(tab["shift"] == 0) & tab.group.isin(["portfolio", "benchmark"])]
    for g, mk, col in [("benchmark", "D", "k"), ("portfolio", "o", "tab:blue")]:
        q = P[P.group == g]
        ax.scatter(-q.mdd, q.ann_ret, marker=mk, s=70, c=col, label=g)
        for _, r in q.iterrows():
            ax.annotate(r["name"].replace(" buy & hold", " B&H").replace(" core + risk-parity", ""),
                        (-r["mdd"], r["ann_ret"]), fontsize=7, xytext=(4, 3), textcoords="offset points")
    for cal in (1.0, 1.5, 2.0):
        x = np.linspace(0.05, 0.65, 10); ax.plot(x, cal * x, ls=":", lw=.8, c="grey")
        ax.annotate(f"Calmar {cal}", (x[-1], cal * x[-1]), fontsize=7, color="grey")
    ax.set_xlabel("max drawdown"); ax.set_ylabel("annualised return")
    ax.set_title("return vs drawdown, test 2023-2026 (benchmarks in black)"); ax.grid(alpha=.3); ax.legend()
    plt.tight_layout(); plt.savefig(OUT / "fig_risk_return.png", dpi=110); plt.close()
    print("\nfigures written")
    return tab


if __name__ == "__main__":
    main()
