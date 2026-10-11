"""Stage 5: the final strategy, built from the pipeline chosen on dev evidence only, plus all figures.

Choice fixed BEFORE looking at its test numbers, using the stage-1..3 evidence:
  horizon h = 5      (stage 1: the excess-IC/Sharpe profile is U-shaped; h=5 is where the model stage
                      shows the largest null-adjusted gain and turnover is still manageable)
  target VOLSTD      (stage 3: standardising the label by sigma_t beats the raw label at every horizon;
                      Cederburg, Johnson & O'Doherty 2023)
  map rolling z      (stage 1: rolling beats expanding, robust z is best)
  model RIDGE + AVG  (stage 3: shallow linear beats GBM and random-feature ridge)
  strategy           B&H + risk-budgeted LS overlay, traded with Garleanu-Pedersen partial adjustment

Portfolio = equal-weight SPY and QQQ sleeves.  Reported against: 50/50 buy & hold, and against the same
pipeline run on the circularly-shifted (null) feature panel.
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

import timing_model_strategy as MS
import timing_transform_study as T
from factor_sharpe import run_book

OUT = Path("outputs/full/strategy")
H, TGT = 5, "VOLSTD"
MODELS = ("RIDGE", "AVG")
OVERLAY_VOL = 0.05
FEE = 1.0


def metrics(r, bench=None):
    r = r.dropna(); eq = (1 + r).cumprod(); dd = eq / eq.cummax() - 1
    ann = float(eq.iloc[-1] ** (252 / len(r)) - 1)
    out = dict(sharpe=float(r.mean() / r.std() * np.sqrt(252)), ann_ret=ann, vol=float(r.std() * np.sqrt(252)),
               mdd=float(dd.min()), calmar=ann / abs(dd.min()), worst_day=float(r.min()),
               hit=float((r > 0).mean()), skew=float(r.skew()))
    if bench is not None:
        b = bench.reindex(r.index).fillna(0)
        beta = np.cov(r, b)[0, 1] / b.var(); res = r - beta * b
        out.update(excess_ann=ann - float((1 + b).prod() ** (252 / len(b)) - 1), beta=float(beta),
                   alpha=float(res.mean() * 252), ir=float(res.mean() / res.std() * np.sqrt(252)),
                   mdd_vs_bench=out["mdd"] - float((((1 + b).cumprod()) / ((1 + b).cumprod()).cummax() - 1).min()))
    return out


def sleeve(sym, shift):
    """Return (bh, overlay_ret, gp_ret, ls_pos) for one symbol."""
    G = MS._G[sym]; panel, rv = G["panel"], G["rv"]
    X = G["X"].copy()
    if shift:
        k = int(np.random.default_rng(7000 + 13 * shift + H).integers(252, len(X) - 252))
        X = np.roll(X, k, axis=0)
    y0 = G["Y"][H]; y = y0 / np.sqrt(H * rv)
    sig, _ = MS.build_signals(X, y, H, rv, np.random.default_rng(12345 + H))
    comps = []
    for m in MODELS:
        s = sig[m]; sd = pd.Series(s).rolling(252, min_periods=126).std().shift(1).to_numpy()
        comps.append(s / np.where(sd > 0, sd, np.nan))
    z = np.nanmean(np.vstack(comps), 0)
    sd = pd.Series(z).rolling(252, min_periods=126).std().shift(1).to_numpy()
    ls = np.clip(T.smooth(z / np.where(sd > 0, sd, np.nan) / 2.0, H), -1, 1)
    k_risk = np.clip(OVERLAY_VOL / np.sqrt(252 * rv), 0, 1.0)
    bh = run_book(panel, pd.Series(1.0, index=panel.dates), fee_bp=FEE)
    w_ov = 1 + k_risk * ls
    a = 1.0 / (1.0 + H / 2.0)                        # partial adjustment towards the aim
    w_gp = np.full(len(ls), np.nan); prev = 1.0
    for t in range(len(ls)):
        if np.isfinite(w_ov[t]):
            prev = (1 - a) * prev + a * w_ov[t]
        w_gp[t] = prev
    return dict(bh=bh, ov=run_book(panel, pd.Series(np.nan_to_num(w_ov), index=panel.dates), fee_bp=FEE),
                gp=run_book(panel, pd.Series(np.nan_to_num(w_gp), index=panel.dates), fee_bp=FEE),
                tilt=run_book(panel, pd.Series(np.nan_to_num(np.clip(1 + 0.5 * ls, .5, 1.5)), index=panel.dates), fee_bp=FEE),
                ls=run_book(panel, pd.Series(np.nan_to_num(ls), index=panel.dates), fee_bp=FEE),
                pos_ov=w_ov, pos_gp=w_gp, lspos=ls, turn_ov=float(np.nanmean(np.abs(np.diff(np.nan_to_num(w_ov)))) * 252),
                turn_gp=float(np.nanmean(np.abs(np.diff(np.nan_to_num(w_gp)))) * 252))


def main():
    MS._init()
    S = {(sym, sh): sleeve(sym, sh) for sym in ("SPY", "QQQ") for sh in (0, 1)}
    rows, series = [], {}
    for sh in (0, 1):
        tag = "" if sh == 0 else " [null]"
        P = {k: (S[("SPY", sh)][k] + S[("QQQ", sh)][k]) / 2 for k in ("bh", "ov", "gp", "tilt", "ls")}
        bh = P["bh"]
        for name, key in [("50/50 Buy & hold", "bh"), ("Pure timing sleeve (LS)", "ls"),
                          ("Tilt 100%+-50%", "tilt"), ("B&H + risk-budgeted overlay", "ov"),
                          ("B&H + overlay, GP partial trading", "gp")]:
            r = P[key].loc["2023":]
            m = metrics(r, bh.loc["2023":"2026"] if key != "bh" else bh.loc["2023":])
            rows.append(dict(scope="portfolio", name=name + tag, shift=sh, **m))
            if sh == 0:
                series[name] = r
        for sym in ("SPY", "QQQ"):
            for name, key in [("Buy & hold", "bh"), ("Overlay", "ov"), ("Overlay+GP", "gp"), ("Tilt", "tilt"), ("LS", "ls")]:
                r = S[(sym, sh)][key].loc["2023":]
                rows.append(dict(scope=sym, name=name + tag, shift=sh,
                                 **metrics(r, S[(sym, sh)]["bh"].loc["2023":])))
    tab = pd.DataFrame(rows); tab.to_csv(OUT / "final_strategy_metrics.csv", index=False)
    pd.set_option("display.width", 260)
    print(tab[tab.scope == "portfolio"].round(3).to_string(index=False))
    print(tab[tab.scope != "portfolio"].round(3).to_string(index=False))
    print("turnover SPY ov/gp", round(S[("SPY", 0)]["turn_ov"], 1), round(S[("SPY", 0)]["turn_gp"], 1),
          "| QQQ", round(S[("QQQ", 0)]["turn_ov"], 1), round(S[("QQQ", 0)]["turn_gp"], 1))

    # ---------------- figure: full tearsheet
    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(3, 2, height_ratios=[3, 1.2, 1.4])
    ax = fig.add_subplot(gs[0, :])
    for name, r in series.items():
        eq = (1 + r).cumprod()
        m = metrics(r, series["50/50 Buy & hold"])
        ax.plot(eq, lw=1.8 if "Buy & hold" in name else 1.2, color="k" if "Buy & hold" in name else None,
                label=f"{name}: SR {m['sharpe']:.2f}, ann {m['ann_ret']:.1%}, MDD {m['mdd']:.1%}")
    nullp = (S[("SPY", 1)]["gp"] + S[("QQQ", 1)]["gp"]) / 2
    ax.plot((1 + nullp.loc["2023":]).cumprod(), lw=1.0, ls="--", color="grey",
            label=f"same pipeline on shifted (null) features: SR {metrics(nullp.loc['2023':])['sharpe']:.2f}")
    ax.set_title("Final strategy, test 2023-01 .. 2026-10 (SPY+QQQ 50/50, 1bp per unit turnover)")
    ax.legend(fontsize=9); ax.grid(alpha=.3)
    ax = fig.add_subplot(gs[1, :])
    for name, r in series.items():
        eq = (1 + r).cumprod(); ax.plot(eq / eq.cummax() - 1, lw=1.2 if "Buy & hold" in name else .9,
                                        color="k" if "Buy & hold" in name else None, label=name)
    ax.set_title("drawdown"); ax.grid(alpha=.3)
    ax = fig.add_subplot(gs[2, 0])
    for sym, c in [("SPY", "tab:blue"), ("QQQ", "tab:orange")]:
        ax.plot(pd.Series(S[(sym, 0)]["pos_gp"], index=S[(sym, 0)]["bh"].index).loc["2023":], lw=.9, color=c, label=f"{sym} weight")
    ax.axhline(1, color="k", lw=.6); ax.set_title("position (B&H + overlay, partial trading)"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    ax = fig.add_subplot(gs[2, 1])
    roll = {n: r.rolling(252).mean() / r.rolling(252).std() * np.sqrt(252) for n, r in series.items()}
    for n, s in roll.items():
        ax.plot(s, lw=1.4 if "Buy & hold" in n else .9, color="k" if "Buy & hold" in n else None, label=n)
    ax.set_title("rolling 1y Sharpe"); ax.grid(alpha=.3); ax.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(OUT / "fig_final_strategy.png", dpi=110); plt.close()

    # ---------------- figure: horizon profile + pipeline-strategy map
    M = pd.read_parquet("outputs/full/transform/real_vs_null.parquet")
    Z = M[M["map"].isin(["Z252", "RZ252"])]
    R = pd.read_csv(OUT / "model_strategy_vs_null.csv")
    fig, axes = plt.subplots(1, 3, figsize=(19, 5))
    ax = axes[0]
    for grp, lab in [("R", "realized measures (new)"), ("I", "intraday/price"), ("M", "macro"), ("N", "noise band")]:
        g = Z[Z.group == grp].groupby("h").d_ic.mean()
        ax.plot(g.index, g.values, marker="o", label=lab)
    ax.axhline(0, color="k", lw=.6); ax.set_xscale("log"); ax.set_xticks([1, 2, 3, 5, 10, 20, 40, 60])
    ax.set_xticklabels([1, 2, 3, 5, 10, 20, 40, 60]); ax.set_xlabel("horizon h (days)")
    ax.set_title("single-factor excess IC vs null, by factor family"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    ax = axes[1]
    g = Z.groupby("h").apply(lambda x: pd.Series(dict(
        d_sr=x.d_sr.mean(), t=x.groupby(["factor", "symbol"]).d_sr.mean().pipe(lambda f: f.mean() / f.std() * np.sqrt(len(f))))))
    ax.bar(range(len(g)), g.d_sr, color=["tab:green" if t > 2 else "tab:olive" for t in g.t])
    for i, (v, t) in enumerate(zip(g.d_sr, g.t)):
        ax.text(i, v, f"t={t:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(g))); ax.set_xticklabels(g.index); ax.set_xlabel("horizon h")
    ax.set_title("excess LS Sharpe vs null by horizon (U-shaped)"); ax.grid(alpha=.3, axis="y")
    ax = axes[2]
    piv = R[(R.target == "VOLSTD") & (R.fee_bp == 1)].pivot_table(index="strategy", columns="h", values="d_sr", aggfunc="mean")
    piv = piv.reindex(["LS", "LONGCASH", "TILT", "VOLTGT", "OVERLAY", "GP"])
    im = ax.imshow(piv.values, cmap="RdYlGn", vmin=-0.3, vmax=0.5, aspect="auto")
    ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels([f"h={c}" for c in piv.columns])
    ax.set_yticks(range(len(piv))); ax.set_yticklabels(piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.values[i, j]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_title("pipeline -> strategy: excess Sharpe vs null"); plt.colorbar(im, ax=ax, fraction=.04)
    plt.tight_layout(); plt.savefig(OUT / "fig_horizon_and_strategy_map.png", dpi=110); plt.close()

    # ---------------- figure: model comparison
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for ax, strat in zip(axes, ["LS", "GP"]):
        p = R[(R.strategy == strat) & (R.fee_bp == 1)].pivot_table(index="model", columns=["target", "h"], values="d_sr")
        p = p.reindex(["EW", "VOTE", "RIDGE", "GBM", "RFF", "AVG"])
        im = ax.imshow(p.values, cmap="RdYlGn", vmin=-0.4, vmax=0.6, aspect="auto")
        ax.set_xticks(range(p.shape[1])); ax.set_xticklabels([f"{a}\nh{b}" for a, b in p.columns], fontsize=7)
        ax.set_yticks(range(len(p))); ax.set_yticklabels(p.index)
        for i in range(p.shape[0]):
            for j in range(p.shape[1]):
                if np.isfinite(p.values[i, j]):
                    ax.text(j, i, f"{p.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
        ax.set_title(f"{strat}: excess Sharpe vs null by model / target / horizon")
        plt.colorbar(im, ax=ax, fraction=.04)
    plt.tight_layout(); plt.savefig(OUT / "fig_models.png", dpi=110); plt.close()
    print("figures written")


if __name__ == "__main__":
    main()
