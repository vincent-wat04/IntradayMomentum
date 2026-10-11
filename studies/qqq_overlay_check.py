"""Does the stage-5 "best" strategy (B&H + risk-budgeted overlay, GP partial trading) beat QQQ alone?

The 1.505 number quoted earlier was the 50/50 SPY+QQQ portfolio.  This runs the identical pipeline on
QQQ only (h=5, VOLSTD label, RIDGE+AVG, 5% overlay risk budget, Garleanu-Pedersen partial adjustment)
and compares it against three references, not one:

  1. QQQ buy & hold                          -- the benchmark the question asks about
  2. constant exposure E = the strategy's own realised average exposure, same cost
     This is the control that matters: the overlay runs at average exposure > 1, so part of any extra
     return is simply more beta.  Only the difference versus this control is attributable to timing.
  3. the same pipeline on circularly-shifted (null) features

Also reported with the exposure implemented two ways: on margin (notional QQQ) and on the real
1x/2x ETF ladder (QQQ/QLD), because above 100% the financing cost is not free.
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
import timing_transform_study as T
from factor_sharpe import run_book

OUT = Path("outputs/full/strategy")
H, MODELS = 5, ("RIDGE", "AVG")
OVERLAY_VOL, FEE = 0.05, 1.0
SYM = os.environ.get("CHECK_SYM", "QQQ")


def build(shift):
    G = MS._G[SYM]; panel, rv = G["panel"], G["rv"]
    X = G["X"].copy()
    if shift:
        X = np.roll(X, int(np.random.default_rng(7000 + 13 * shift + H).integers(252, len(X) - 252)), axis=0)
    y = G["Y"][H] / np.sqrt(H * rv)
    sig, _ = MS.build_signals(X, y, H, rv, np.random.default_rng(12345 + H))
    comps = []
    for m in MODELS:
        s = sig[m]; sd = pd.Series(s).rolling(252, min_periods=126).std().shift(1).to_numpy()
        comps.append(s / np.where(sd > 0, sd, np.nan))
    z = np.nanmean(np.vstack(comps), 0)
    sd = pd.Series(z).rolling(252, min_periods=126).std().shift(1).to_numpy()
    ls = np.clip(T.smooth(z / np.where(sd > 0, sd, np.nan) / 2.0, H), -1, 1)
    k = np.clip(OVERLAY_VOL / np.sqrt(252 * rv), 0, 1.0)
    w_ov = 1 + k * ls
    a = 1.0 / (1.0 + H / 2.0)
    w_gp = np.full(len(ls), np.nan); prev = 1.0
    for t in range(len(ls)):
        if np.isfinite(w_ov[t]):
            prev = (1 - a) * prev + a * w_ov[t]
        w_gp[t] = prev
    return panel, ls, w_ov, w_gp


def main():
    MS._init()
    lev_px = pd.read_csv("data/lev/levetf_close.csv", index_col=0)
    panel = MS._G[SYM]["panel"]; dates = panel.dates; split = MS._G[SYM]["split"]
    bh = run_book(panel, pd.Series(1.0, index=panel.dates), fee_bp=FEE)
    rows, curves = [], {f"{SYM} buy & hold": bh}
    rows.append(dict(name=f"{SYM} buy & hold", shift=0, avg_E=1.0, **LP.metrics(bh.loc["2023":], bh.loc["2023":])))

    for shift in (0, 1):
        tag = "" if shift == 0 else " [null]"
        _, ls, w_ov, w_gp = build(shift)
        for nm, w in [("overlay", w_ov), ("overlay + GP partial trading", w_gp)]:
            avgE = float(np.nanmean(w[split:]))
            # (a) margin implementation: notional QQQ at weight w
            r_m = run_book(panel, pd.Series(np.nan_to_num(w), index=panel.dates), fee_bp=FEE)
            # (b) real ETF ladder (QQQ / QLD), which pays the actual financing above 100%
            r_l = LP.ladder_returns(SYM, np.nan_to_num(w, nan=1.0), dates, lev_px, fee_bp=FEE)
            # controls at the SAME average exposure
            c_m = run_book(panel, pd.Series(np.full(len(dates), avgE), index=panel.dates), fee_bp=FEE)
            c_l = LP.ladder_returns(SYM, np.full(len(dates), avgE), dates, lev_px, fee_bp=FEE)
            for impl, r, c in [("margin", r_m, c_m), ("ETF ladder", r_l, c_l)]:
                m = LP.metrics(r.loc["2023":], bh.loc["2023":]); mc = LP.metrics(c.loc["2023":], bh.loc["2023":])
                rows.append(dict(name=f"{SYM} {nm} [{impl}]" + tag, shift=shift, avg_E=avgE,
                                 sr_ctrl=mc["sharpe"], ann_ctrl=mc["ann_ret"], mdd_ctrl=mc["mdd"],
                                 d_sr_vs_ctrl=m["sharpe"] - mc["sharpe"], d_ann_vs_ctrl=m["ann_ret"] - mc["ann_ret"],
                                 d_sr_vs_bh=m["sharpe"] - LP.metrics(bh.loc["2023":])["sharpe"], **m))
                if shift == 0:
                    curves[f"{SYM} {nm} [{impl}]"] = r
                    curves[f"constant E={avgE:.3f} [{impl}]"] = c
        if shift == 0:
            rows.append(dict(name=f"{SYM} pure timing sleeve (LS)", shift=0, avg_E=float(np.nanmean(ls[split:])),
                             **LP.metrics(run_book(panel, pd.Series(np.nan_to_num(ls), index=panel.dates),
                                                   fee_bp=FEE).loc["2023":], bh.loc["2023":])))

    tab = pd.DataFrame(rows); tab.to_csv(OUT / f"overlay_check_{SYM}.csv", index=False)
    pd.set_option("display.width", 300); pd.set_option("display.max_rows", 200)
    c = ["name", "avg_E", "sharpe", "ann_ret", "vol", "mdd", "calmar", "excess_ann", "beta", "alpha", "ir",
         "d_sr_vs_bh", "sr_ctrl", "ann_ctrl", "mdd_ctrl", "d_sr_vs_ctrl", "d_ann_vs_ctrl"]
    print(tab[[x for x in c if x in tab.columns]].round(3).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(17, 6), gridspec_kw=dict(width_ratios=[2, 1]))
    ax = axes[0]
    show = [f"{SYM} buy & hold", f"{SYM} overlay + GP partial trading [margin]",
            f"{SYM} overlay + GP partial trading [ETF ladder]"]
    show += [k for k in curves if k.startswith("constant E") and "ETF" in k][:1]
    for n in show:
        r = curves[n].loc["2023":]; m = LP.metrics(r)
        ax.plot((1 + r).cumprod(), lw=2.0 if "buy & hold" in n else 1.2,
                color="k" if "buy & hold" in n else None, ls="--" if "constant" in n else "-",
                label=f"{n}: SR {m['sharpe']:.3f}, ann {m['ann_ret']:.1%}, MDD {m['mdd']:.1%}")
    ax.set_title(f"{SYM}: overlay + GP vs buy & hold vs exposure-matched constant leverage (test 2023-2026)")
    ax.legend(fontsize=8.5); ax.grid(alpha=.3)
    ax = axes[1]
    P = tab[tab["shift"] == 0]
    ax.scatter(-P.mdd, P.ann_ret, s=60)
    for _, r in P.iterrows():
        ax.annotate(r["name"].replace(SYM+" ", "").replace(" partial trading", ""), (-r["mdd"], r["ann_ret"]),
                    fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel("max drawdown"); ax.set_ylabel("annualised return"); ax.grid(alpha=.3)
    ax.set_title("return vs drawdown")
    plt.tight_layout(); plt.savefig(OUT / f"fig_overlay_check_{SYM}.png", dpi=110); plt.close()
    print("\nfigure written")
    return tab


if __name__ == "__main__":
    main()
