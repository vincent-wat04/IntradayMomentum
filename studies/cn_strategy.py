"""Stage 10: the same pipeline applied to A-share, with a null.

This is the more informative test of the two markets: A-share buy & hold had a test Sharpe of only
0.20-0.40 over 2023-2026, so a timing signal is not competing against a 1.5-Sharpe bull market the way
it is in the US.  Nothing is transferred from the US study -- factor signs and selection are re-derived
from A-share data only, which is the point of stage 9.

Null: every factor column is circularly shifted by a random offset >= 1 year, so the factor's
distribution and persistence survive but its link to future returns does not.
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

import cn_adaptivity_study as CN
import timing_transform_study as T

OUT = Path("outputs/full/cn")
HS = (1, 2, 5, 20)
AGGS = ("L1", "MA5", "MA20", "EWM10", "MED20")
IC_MIN, CORR_MAX, K_FEAT = 0.03, 0.7, 12
SCREEN_EVERY, MIN_TRAIN = 63, 504


def metrics(r, bench=None):
    r = r.dropna(); eq = (1 + r).cumprod(); dd = eq / eq.cummax() - 1
    ann = float(eq.iloc[-1] ** (252 / len(r)) - 1)
    out = dict(sharpe=float(r.mean() / r.std() * np.sqrt(252)), ann_ret=ann, vol=float(r.std() * np.sqrt(252)),
               mdd=float(dd.min()), calmar=ann / abs(dd.min()))
    if bench is not None:
        b = bench.reindex(r.index).fillna(0)
        beta = np.cov(r, b)[0, 1] / b.var(); res = r - beta * b
        out.update(excess_ann=ann - float((1 + b).prod() ** (252 / len(b)) - 1), beta=float(beta),
                   alpha=float(res.mean() * 252), ir=float(res.mean() / res.std() * np.sqrt(252)))
    return out


def feature_panel(F, panel):
    cols, names = [], []
    for fac in F.columns:
        for agg in AGGS:
            a = T.aggregate(F[fac], agg); av = a.to_numpy(float)
            pct, mu, sd, med, mad = T.prior_stats(av)
            with np.errstate(all="ignore"):
                cols.append(np.clip((av - mu) / sd, -3, 3) / 3)
            names.append(f"{fac}:{agg}")
    return np.column_stack(cols), names


def build(X, y, h, split, every=SCREEN_EVERY):
    """Equal-weight blend of the re-screened, data-signed factors. Returns (position, mean #features)."""
    n = len(y); s = np.full(n, np.nan); idx, sgn = np.array([], int), np.array([])
    Xf = np.nan_to_num(X); nf = []
    for t in range(n):
        if t % every == 0:
            k = t - h + 1
            if k >= MIN_TRAIN:
                fy = np.isfinite(y[:k])
                if fy.sum() >= MIN_TRAIN:
                    ry = pd.Series(y[:k]).rank().to_numpy(); RX = pd.DataFrame(X[:k]).rank().to_numpy()
                    half = np.flatnonzero(fy); mid = half[len(half) // 2]
                    ic = np.full(X.shape[1], np.nan); ia = ic.copy(); ib = ic.copy()
                    for j in range(X.shape[1]):
                        m = fy & np.isfinite(X[:k, j])
                        a = m.copy(); a[mid:] = False
                        b = m.copy(); b[:mid] = False
                        if m.sum() < MIN_TRAIN // 2 or a.sum() < 60 or b.sum() < 60:
                            continue
                        ic[j] = np.corrcoef(RX[m, j], ry[m])[0, 1]
                        ia[j] = np.corrcoef(RX[a, j], ry[a])[0, 1]
                        ib[j] = np.corrcoef(RX[b, j], ry[b])[0, 1]
                    ok = np.isfinite(ic) & (np.sign(ia) == np.sign(ib)) & (np.abs(ic) >= IC_MIN)
                    cand = np.flatnonzero(ok)[np.argsort(-np.abs(ic[ok]))][:120]
                    chosen = []
                    if len(cand):
                        C = np.abs(np.nan_to_num(pd.DataFrame(X[:k, cand]).corr().to_numpy()))
                        for i in range(len(cand)):
                            if all(C[i, q] < CORR_MAX for q in chosen):
                                chosen.append(i)
                            if len(chosen) >= K_FEAT:
                                break
                    idx, sgn = cand[chosen], np.sign(ic[cand[chosen]])
        nf.append(len(idx))
        if len(idx):
            s[t] = float(np.mean(Xf[t, idx] * sgn))
    sd = pd.Series(s).rolling(252, min_periods=126).std().shift(1).to_numpy()
    return np.clip(T.smooth(s / np.where(sd > 0, sd, np.nan) / 2.0, h), -1, 1), float(np.mean(nf[split:]))


def main():
    rows, curves, feats = [], {}, []
    for name, code in CN.SYMS.items():
        bars = CN.build_5min(code); panel = CN.panel_from(bars)
        F = CN.cn_factors(bars, panel)
        X0, names = feature_panel(F, panel)
        split = int(np.searchsorted(panel.dates, CN.SPLIT))
        bh = CN.run_book_cn(panel, np.ones(len(panel.dates)))
        rows.append(dict(symbol=name, strategy="buy & hold", shift=0, **metrics(bh.loc["2023":], bh.loc["2023":])))
        curves[(name, "buy & hold")] = bh
        for shift in (0, 1):
            tag = "" if shift == 0 else " [null]"
            X = X0.copy()
            if shift:
                X = np.roll(X, int(np.random.default_rng(99 + shift).integers(252, len(X) - 252)), axis=0)
            pos = {}
            for h in HS:
                y = CN.target(panel, h)
                p, nf = build(X, y, h, split)
                pos[h] = p
                r = CN.run_book_cn(panel, np.nan_to_num(p))
                rows.append(dict(symbol=name, strategy=f"LS h={h}" + tag, shift=shift, n_feat=nf,
                                 **metrics(r.loc["2023":], bh.loc["2023":])))
                if shift == 0:
                    feats.append(dict(symbol=name, h=h, n_feat=nf))
            # equal-risk blend of horizons, then the three position maps
            rr = {h: CN.run_book_cn(panel, np.nan_to_num(pos[h])) for h in HS}
            iv = pd.DataFrame({h: 1.0 / (rr[h].rolling(63, min_periods=40).std() * np.sqrt(252)).shift(1) for h in HS})
            iv = iv.div(iv.sum(1), axis=0).fillna(0.25)
            blend = sum(iv[h].to_numpy() * np.nan_to_num(pos[h]) for h in HS)
            for sname, w in [("LS blended", blend), ("Long/cash", (blend > 0).astype(float)),
                             ("Tilt 100%+-50%", np.clip(1 + 0.5 * blend, 0.5, 1.5))]:
                r = CN.run_book_cn(panel, np.nan_to_num(w))
                rows.append(dict(symbol=name, strategy=sname + tag, shift=shift, **metrics(r.loc["2023":], bh.loc["2023":])))
                if shift == 0:
                    curves[(name, sname)] = r
                else:
                    curves[(name, sname + " [null]")] = r

    tab = pd.DataFrame(rows); tab.to_csv(OUT / "cn_strategy_metrics.csv", index=False)
    pd.to_pickle(curves, OUT / "cn_curves.pkl")
    pd.set_option("display.width", 280); pd.set_option("display.max_rows", 400)
    c = ["symbol", "strategy", "sharpe", "ann_ret", "vol", "mdd", "calmar", "excess_ann", "beta", "alpha", "ir", "n_feat"]
    print(tab[[x for x in c if x in tab.columns]].round(3).to_string(index=False))
    print("\n=== real minus null, averaged over the four indices ===")
    R = tab[~tab.strategy.str.contains(r"\[null\]|buy & hold")].groupby("strategy")[["sharpe", "excess_ann", "ir"]].mean()
    N = tab[tab.strategy.str.contains(r"\[null\]")].copy()
    N["strategy"] = N.strategy.str.replace(" [null]", "", regex=False)
    N = N.groupby("strategy")[["sharpe", "excess_ann", "ir"]].mean()
    D = (R - N).add_prefix("d_").join(R).join(N.add_prefix("null_"))
    print(D.round(3).to_string())

    fig, axes = plt.subplots(2, 2, figsize=(17, 9))
    for ax, name in zip(axes.ravel(), CN.SYMS):
        for s in ["buy & hold", "LS blended", "Long/cash", "Tilt 100%+-50%", "LS blended [null]"]:
            if (name, s) not in curves:
                continue
            r = curves[(name, s)].loc["2023":]; m = metrics(r)
            ax.plot((1 + r).cumprod(), lw=1.8 if s == "buy & hold" else 1.1,
                    color="k" if s == "buy & hold" else None, ls="--" if "null" in s else "-",
                    label=f"{s}: SR {m['sharpe']:.2f}, ann {m['ann_ret']:.1%}, MDD {m['mdd']:.1%}")
        ax.set_title(f"{name} (A-share), test 2023-01 .. 2026-09"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(OUT / "fig_cn_strategy.png", dpi=110); plt.close()
    print("\nfigure written")
    return tab


if __name__ == "__main__":
    main()
