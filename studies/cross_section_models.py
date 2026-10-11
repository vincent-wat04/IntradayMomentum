"""Stage 11b: do GBM / random-feature ridge beat a linear model once we have a cross-section?

Time-series study:  ~1,000 training rows, 12 features  -> shallow linear won, GBM and RFF lost.
Cross-section here: ~130,000 training rows, 32 features -> same four model families, same discipline.

Models are refit every 21 sessions on the pooled panel using only labels that matured before the fit
date, then applied daily until the next refit.  Inputs are the cross-sectionally standardised factors,
optionally sector-demeaned and style-orthogonalised, so each stage of your preprocessing chain can be
compared on identical model code.

Also included: the SPY-vs-QQQ "two asset cross-section", which is just the QQQ-minus-SPY spread.
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
from sklearn.linear_model import Ridge

import cross_section_study as XS

try:
    import lightgbm as lgb
    HAVE_LGB = True
except Exception:
    HAVE_LGB = False

OUT = XS.OUT
MODELS = ("EW", "RIDGE", "GBM", "RFF")
HS = (5, 20)
FIT_EVERY, MIN_TRAIN_ROWS = 21, 20000
NULL_SHIFTS = 1


def fit_model(kind, A, y, rng):
    if kind == "RIDGE":
        return ("lin", Ridge(alpha=10.0).fit(A, y))
    if kind == "GBM":
        m = lgb.LGBMRegressor(n_estimators=300, num_leaves=31, max_depth=6, learning_rate=0.03,
                              min_child_samples=100, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                              reg_lambda=1.0, verbose=-1, n_jobs=1, force_col_wise=True)
        return ("lin", m.fit(A, y))
    if kind == "RFF":
        W = rng.standard_normal((A.shape[1], XS.RFF_P // 2)) * np.sqrt(XS.RFF_GAMMA)
        b = rng.uniform(0, 2 * np.pi, XS.RFF_P // 2)
        phi = lambda Z: np.c_[np.cos(Z @ W + b), np.sin(Z @ W + b)] * np.sqrt(2.0 / XS.RFF_P)
        return ("rff", Ridge(alpha=XS.RFF_ALPHA).fit(phi(A), y), phi)
    raise KeyError(kind)


def predict(m, B):
    return m[1].predict(B) if m[0] == "lin" else m[1].predict(m[2](B))


def walk_forward(F: dict, fwd: pd.DataFrame, h: int, rng, ew_signs=None):
    keys = sorted(F)
    cube = np.stack([F[k].to_numpy(float) for k in keys], -1)
    Y = fwd.to_numpy(float)
    T_, N, K = cube.shape
    dates, cols = fwd.index, fwd.columns
    out = {m: np.full((T_, N), np.nan) for m in MODELS}
    mu = sd = None
    fitted = {}
    for t in range(T_):
        if t % FIT_EVERY == 0 and t - h - 1 > 0:
            X = cube[: t - h - 1].reshape(-1, K); y = Y[: t - h - 1].reshape(-1)
            m = np.isfinite(y) & np.isfinite(X).all(1)
            if m.sum() >= MIN_TRAIN_ROWS:
                A0, y0 = X[m], y[m]
                mu, sd = A0.mean(0), A0.std(0) + 1e-9
                A = (A0 - mu) / sd
                y0 = (y0 - y0.mean()) / (y0.std() + 1e-12)
                # EW sign per factor = sign of its pooled rank correlation with the label
                sg = np.array([np.sign(np.corrcoef(pd.Series(A[:, j]).rank(),
                                                   pd.Series(y0).rank())[0, 1]) for j in range(K)])
                fitted = {"EW": ("ew", sg)}
                for nm in MODELS:
                    if nm == "EW":
                        continue
                    if nm == "GBM" and not HAVE_LGB:
                        continue
                    try:
                        fitted[nm] = fit_model(nm, A, y0, rng)
                    except Exception:
                        pass
        if not fitted or mu is None:
            continue
        Xt = cube[t]
        ok = np.isfinite(Xt).all(1)
        if ok.sum() < 20:
            continue
        B = (Xt[ok] - mu) / sd
        for nm, m in fitted.items():
            try:
                out[nm][t, np.flatnonzero(ok)] = (B * m[1]).mean(1) if m[0] == "ew" else predict(m, B)
            except Exception:
                pass
    return {m: pd.DataFrame(v, index=dates, columns=cols) for m, v in out.items()}


def main():
    px, op, hi, lo, vo = XS.load()
    sec = pd.Series({c: XS.SECTOR[c] for c in px.columns})
    F0, r = XS.build_factors(px, op, hi, lo, vo)
    Z = {k: XS.xs_rank_z(v) for k, v in F0.items()}
    ZS = {k: XS.sector_demean(v, sec) for k, v in Z.items()}
    print("orthogonalising styles...", flush=True)
    ZSO = XS.style_orthogonalize(ZS)
    stages = {"xs_z": Z, "xs_z+sector": ZS, "xs_z+sector+style": ZSO}
    split_i = int(np.searchsorted(px.index, XS.SPLIT))
    rng = np.random.default_rng(7)

    rows, curves = [], {}
    for stage, F in stages.items():
        for h in HS:
            fwd = XS.fwd_ret(px, h)
            for shift in range(NULL_SHIFTS + 1):
                Fs = F
                if shift:
                    k = int(np.random.default_rng(500 + shift).integers(252, len(px) - 252))
                    Fs = {kk: pd.DataFrame(np.roll(v.to_numpy(), k, axis=0), index=v.index, columns=v.columns)
                          for kk, v in F.items()}
                scores = walk_forward(Fs, fwd, h, rng)
                for nm, sc in scores.items():
                    if sc.notna().sum().sum() == 0:
                        continue
                    icd = XS.ic_series(sc.iloc[:split_i], fwd.iloc[:split_i])
                    ict = XS.ic_series(sc.iloc[split_i:], fwd.iloc[split_i:])
                    sd_, st_ = XS.ic_stats(icd, h), XS.ic_stats(ict, h)
                    pnl = XS.ls_portfolio(sc, r, h)
                    tag = "" if shift == 0 else " [null]"
                    rows.append(dict(stage=stage, h=h, model=nm + tag, shift=shift,
                                     ic_dev=sd_["ic"], ic_test=st_["ic"], t_test=st_["t"],
                                     sr_dev=XS.sharpe(pnl.iloc[:split_i]), sr_test=XS.sharpe(pnl.iloc[split_i:])))
                    if shift == 0:
                        curves[(stage, h, nm)] = pnl

    M = pd.DataFrame(rows); M.to_csv(OUT / "xs_models.csv", index=False)
    pd.to_pickle(curves, OUT / "xs_curves.pkl")
    pd.set_option("display.width", 260); pd.set_option("display.max_rows", 300)
    print("\n=== cross-sectional models: test IC and long-short Sharpe ===")
    print(M[["stage", "h", "model", "ic_dev", "ic_test", "t_test", "sr_dev", "sr_test"]].round(3).to_string(index=False))
    print("\n=== model ranking, real minus null (averaged over stage and h) ===")
    real = M[M["shift"] == 0].copy()
    null = M[M["shift"] == 1].copy(); null["model"] = null.model.str.replace(" [null]", "", regex=False)
    j = real.merge(null, on=["stage", "h", "model"], suffixes=("", "_n"))
    j["d_ic"] = j.ic_test - j.ic_test_n; j["d_sr"] = j.sr_test - j.sr_test_n
    print(j.groupby("model")[["ic_test", "ic_test_n", "d_ic", "sr_test", "sr_test_n", "d_sr"]].mean().round(3).to_string())
    print("\n=== preprocessing stage ranking (averaged over models) ===")
    print(j.groupby("stage")[["ic_test", "d_ic", "sr_test", "d_sr"]].mean().round(3).to_string())
    print("\n=== GBM/RFF minus RIDGE, per cell (positive = non-linear wins) ===")
    p = j.pivot_table(index=["stage", "h"], columns="model", values="sr_test")
    for c in ("GBM", "RFF", "EW"):
        if c in p:
            p[f"{c}-RIDGE"] = p[c] - p["RIDGE"]
    print(p.round(3).to_string())

    # ---- SPY vs QQQ as a two-asset cross-section = the QQQ-SPY spread
    print("\n=== SPY vs QQQ pair: see pair_cross_section.py ===")

    # ---- figures
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))
    ax = axes[0]
    S = pd.read_csv(OUT / "xs_single_factor.csv")
    for stage in ["raw", "xs_z+sector", "xs_z+sector+style"]:
        g = S[S.stage == stage].groupby("h").apply(lambda x: x.ic_test.abs().mean())
        ax.plot(g.index, g.values, "-o", label=stage)
    ax.set_xlabel("horizon h"); ax.set_ylabel("mean |test IC|")
    ax.set_title("each neutralisation step removes signal magnitude"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    ax = axes[1]
    w = 0.2
    stages_l = list(stages)
    for i, nm in enumerate([m for m in MODELS if m in set(j.model)]):
        g = j[j.model == nm].groupby("stage").d_sr.mean().reindex(stages_l)
        ax.bar(np.arange(len(stages_l)) + (i - 1.5) * w, g.values, w, label=nm)
    ax.set_xticks(range(len(stages_l))); ax.set_xticklabels(stages_l, rotation=15, fontsize=8)
    ax.axhline(0, color="k", lw=.6); ax.set_ylabel("long-short Sharpe minus null")
    ax.set_title("do non-linear models win in the cross-section?"); ax.legend(fontsize=8); ax.grid(alpha=.3, axis="y")
    ax = axes[2]
    for (stage, h, nm), pnl in curves.items():
        if stage != "xs_z+sector" or h != 5:
            continue
        p = pnl.loc[XS.SPLIT:]
        ax.plot((1 + p).cumprod(), lw=1.2, label=f"{nm}: SR {XS.sharpe(p):.2f}")
    ax.set_title("decile long-short, sector-neutral, h=5 (test)"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(OUT / "fig_xs_models.png", dpi=110); plt.close()
    print("\nfigure written")
    return M


if __name__ == "__main__":
    main()
