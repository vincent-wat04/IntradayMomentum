"""Stage 8: the return / drawdown frontier with real leveraged ETFs, and the corrected verdict.

Stage 7 showed the alpha sleeve does not survive a beta hedge, so there is no market-neutral alpha to
lever.  What is left is a pure risk-appetite question: for a given tolerated drawdown, what is the most
return obtainable, and does any signal processing move that frontier?

Compared here, all on real prices, 1bp per unit turnover, test 2023-01..2026-10:
  * 1x / 2x / 3x buy & hold (SPY, QQQ ladder)
  * constant exposure E on the cheapest ETF ladder, E from 1.0 to 3.0
  * signal-tilted exposure E_t = (1 + lam * sleeve_t) * s, exposure-matched to the constant-E control
  * volatility-targeted exposure
Calmar (return / max drawdown) is the frontier statistic: if it is flat in E, leverage only moves you
along the frontier and nothing is gained or lost; if the tilted version sits above the constant-E line,
the signal genuinely improves the tradeoff.
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


def main():
    MS._init()
    lev_px = pd.read_csv("data/lev/levetf_close.csv", index_col=0)
    dates = MS._G["SPY"]["panel"].dates
    POS = pd.read_pickle(OUT / "sleeve_positions.pkl")
    rows, curves = [], {}

    for sym in ("SPY", "QQQ"):
        panel = MS._G[sym]["panel"]; rv = MS._G[sym]["rv"]; split = MS._G[sym]["split"]
        core = run_book(panel, pd.Series(1.0, index=panel.dates), fee_bp=1.0)
        curves[f"{sym} 1x B&H"] = core
        # equal-risk blend of the four horizon sleeves (inverse trailing vol, PIT)
        sr_pos = POS[(sym, 0)]
        rets = {h: run_book(panel, pd.Series(np.nan_to_num(p), index=panel.dates), fee_bp=1.0) for h, p in sr_pos.items()}
        iv = pd.DataFrame({h: 1.0 / (rets[h].rolling(63, min_periods=40).std() * np.sqrt(252)).shift(1) for h in HS})
        iv = iv.div(iv.sum(1), axis=0).fillna(1.0 / len(HS))
        blend = sum(iv[h].to_numpy() * np.nan_to_num(sr_pos[h]) for h in HS)

        for E in np.arange(1.0, 3.01, 0.25):
            r = LP.ladder_returns(sym, np.full(len(dates), E), dates, lev_px)
            rows.append(dict(symbol=sym, kind="constant E", E=E, **LP.metrics(r.loc["2023":], core.loc["2023":])))
            if abs(E - round(E)) < 1e-9 and E > 1:
                curves[f"{sym} {int(E)}x B&H"] = r
        for lam in (0.25, 0.5, 1.0):
            Et = np.clip(1.0 + lam * blend, 0, 3)
            r = LP.ladder_returns(sym, Et, dates, lev_px)
            avgE = float(np.nanmean(Et[split:]))
            rc = LP.ladder_returns(sym, np.full(len(dates), avgE), dates, lev_px)
            m, mc = LP.metrics(r.loc["2023":], core.loc["2023":]), LP.metrics(rc.loc["2023":], core.loc["2023":])
            rows.append(dict(symbol=sym, kind=f"signal tilt lam={lam:g}", E=avgE, calmar_matched=mc["calmar"],
                             d_calmar=m["calmar"] - mc["calmar"], d_sharpe=m["sharpe"] - mc["sharpe"], **m))
            curves[f"{sym} tilt lam={lam:g}"] = r
        for vt in (0.15, 0.20, 0.25, 0.30):
            s = np.clip(vt / np.sqrt(252 * rv), 0.2, 3.0)
            r = LP.ladder_returns(sym, s, dates, lev_px)
            avgE = float(np.nanmean(s[split:]))
            rc = LP.ladder_returns(sym, np.full(len(dates), avgE), dates, lev_px)
            m, mc = LP.metrics(r.loc["2023":], core.loc["2023":]), LP.metrics(rc.loc["2023":], core.loc["2023":])
            rows.append(dict(symbol=sym, kind=f"vol target {vt:.0%}", E=avgE, calmar_matched=mc["calmar"],
                             d_calmar=m["calmar"] - mc["calmar"], d_sharpe=m["sharpe"] - mc["sharpe"], **m))
            curves[f"{sym} vol target {vt:.0%}"] = r

    tab = pd.DataFrame(rows); tab.to_csv(OUT / "leverage_frontier.csv", index=False)
    pd.set_option("display.width", 280); pd.set_option("display.max_rows", 300)
    c = ["symbol", "kind", "E", "sharpe", "ann_ret", "vol", "mdd", "calmar", "d_sharpe", "d_calmar"]
    print(tab[[x for x in c if x in tab.columns]].round(3).to_string(index=False))
    print("\n=== Calmar vs leverage (constant E) ===")
    print(tab[tab.kind == "constant E"].pivot_table(index="E", columns="symbol", values="calmar").round(3).to_string())
    print("\n=== does the signal move the frontier? (vs exposure-matched constant E) ===")
    q = tab[tab.kind.str.contains("tilt|vol target")]
    print(q.groupby(["symbol", "kind"])[["d_sharpe", "d_calmar"]].mean().round(3).to_string())

    # ---------------- frontier figure
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.6))
    ax = axes[0]
    for sym, col in [("SPY", "tab:blue"), ("QQQ", "tab:orange")]:
        q = tab[(tab.symbol == sym) & (tab.kind == "constant E")].sort_values("E")
        ax.plot(-q.mdd, q.ann_ret, "-o", ms=4, color=col, label=f"{sym} constant leverage 1x->3x")
        for k, mk in [("signal tilt", "*"), ("vol target", "s")]:
            s = tab[(tab.symbol == sym) & tab.kind.str.startswith(k)]
            ax.scatter(-s.mdd, s.ann_ret, marker=mk, s=110 if mk == "*" else 45, color=col,
                       edgecolor="k", lw=.5, label=f"{sym} {k}", zorder=5)
    for cal in (1.0, 1.25, 1.5):
        x = np.linspace(0.1, 0.62, 10); ax.plot(x, cal * x, ls=":", lw=.9, c="grey")
        ax.annotate(f"Calmar {cal}", (x[-1], cal * x[-1]), fontsize=8, color="grey")
    ax.set_xlabel("max drawdown"); ax.set_ylabel("annualised return")
    ax.set_title("return / drawdown frontier (real leveraged ETFs)"); ax.legend(fontsize=7); ax.grid(alpha=.3)
    ax = axes[1]
    for sym, col in [("SPY", "tab:blue"), ("QQQ", "tab:orange")]:
        q = tab[(tab.symbol == sym) & (tab.kind == "constant E")].sort_values("E")
        ax.plot(q.E, q.sharpe, "-o", ms=4, color=col, label=f"{sym} Sharpe")
        ax.plot(q.E, q.calmar, "--s", ms=4, color=col, alpha=.6, label=f"{sym} Calmar")
    ax.set_xlabel("constant exposure E"); ax.set_title("Sharpe and Calmar are ~flat in leverage")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    ax = axes[2]
    show = ["SPY 1x B&H", "QQQ 1x B&H", "QQQ 2x B&H", "QQQ 3x B&H", "QQQ tilt lam=1", "QQQ vol target 20%"]
    for n in show:
        r = curves[n].loc["2023":]; eq = (1 + r).cumprod(); m = LP.metrics(r)
        ax.plot(eq, lw=1.7 if "B&H" in n else 1.1, ls="--" if "B&H" in n else "-",
                label=f"{n}: SR {m['sharpe']:.2f}, ann {m['ann_ret']:.0%}, MDD {m['mdd']:.0%}")
    ax.set_yscale("log"); ax.set_title("equity curves vs index benchmarks"); ax.legend(fontsize=7.5); ax.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(OUT / "fig_leverage_frontier.png", dpi=110); plt.close()
    pd.to_pickle(curves, OUT / "frontier_curves.pkl")
    print("\nfigure written")
    return tab


if __name__ == "__main__":
    main()
