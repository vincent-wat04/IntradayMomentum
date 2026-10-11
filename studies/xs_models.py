"""Cross-sectional models on the 77-factor library: does complexity pay once N x T is large?

Time-series study:  ~1,000 rows x 12 features   -> shallow linear beat GBM and random-feature ridge.
Here:               ~150,000 rows x 77 features -> same four families, same walk-forward discipline.

Preprocessing stages compared on identical model code (your chain):
  xs_z              cross-sectional rank -> z each day        (removes the market factor)
  +sector           minus the sector mean each day            (industry neutral)
  +style            residual after regressing out SIZE/BETA/VOL_21 each day   (style orthogonal)

Refit every 63 sessions on pooled rows whose labels matured before the fit date; training rows are
subsampled to TRAIN_CAP to keep the sweep tractable.  Null = whole factor panel shifted >= 1 year.
Long-short deciles pay 5bp per leg (single stocks).
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

import cross_section_study as XS
import xs_factor_library as XL

try:
    import lightgbm as lgb
    HAVE_LGB = True
except Exception:
    HAVE_LGB = False

OUT = Path("outputs/full/xs"); OUT.mkdir(parents=True, exist_ok=True)
MODELS = ("EW", "RIDGE", "GBM", "RFF")
HS = (5, 20)
FIT_EVERY, TRAIN_CAP, MIN_ROWS = 63, 60000, 20000
RFF_P, RFF_GAMMA, RFF_ALPHA = 1000, 0.5, 200.0
NULL_SHIFTS = 2
STAGES = ("xs_z+sector", "xs_z+sector+style")
_G = {}


def _init():
    px, op, hi, lo, vo = XS.load()
    sec = pd.Series({c: XS.SECTOR[c] for c in px.columns})
    F0, r = XL.build(px, op, hi, lo, vo)
    Z = {k: XS.xs_rank_z(v) for k, v in F0.items()}
    ZS = {k: XS.sector_demean(v, sec) for k, v in Z.items()}
    ZSO = XL.style_orthogonalize_fast(ZS)
    _G.update(px=px, r=r, sec=sec, stages={"xs_z": Z, "xs_z+sector": ZS, "xs_z+sector+style": ZSO},
              split=int(np.searchsorted(px.index, XS.SPLIT)))


def fit(kind, A, y, rng):
    if kind == "RIDGE":
        return ("lin", Ridge(alpha=10.0).fit(A, y))
    if kind == "GBM":
        m = lgb.LGBMRegressor(n_estimators=200, num_leaves=31, max_depth=6, learning_rate=0.05,
                              min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                              reg_lambda=5.0, verbose=-1, n_jobs=1, force_col_wise=True)
        return ("lin", m.fit(A, y))
    W = rng.standard_normal((A.shape[1], RFF_P // 2)) * np.sqrt(RFF_GAMMA)
    b = rng.uniform(0, 2 * np.pi, RFF_P // 2)
    phi = lambda Zm: np.c_[np.cos(Zm @ W + b), np.sin(Zm @ W + b)] * np.sqrt(2.0 / RFF_P)
    return ("rff", Ridge(alpha=RFF_ALPHA).fit(phi(A), y), phi)


def job(args):
    stage, h, shift = args
    F = _G["stages"][stage]
    px, r, split = _G["px"], _G["r"], _G["split"]
    if shift:
        # CROSS-SECTIONAL PERMUTATION NULL.  A circular time shift is the wrong null here: stock
        # characteristics (volatility, size, spread) are persistent over years, so a shifted panel still
        # ranks stocks almost correctly and the null inherits the real characteristic premia.  Permuting
        # stock identities WITHIN each day destroys the characteristic->return link while preserving each
        # factor's daily cross-sectional distribution and the correlation structure across factors
        # (the same permutation is applied to every factor on a given day).
        rngp = np.random.default_rng(500 + shift)
        n = len(F[sorted(F)[0]].columns)
        perm = np.stack([rngp.permutation(n) for _ in range(len(px))])
        F = {kk: pd.DataFrame(np.take_along_axis(v.to_numpy(), perm, axis=1), index=v.index,
                              columns=v.columns) for kk, v in F.items()}
    fwd = XS.fwd_ret(px, h)
    keys = sorted(F)
    cube = np.stack([F[k].to_numpy(float) for k in keys], -1)
    Y = fwd.to_numpy(float)
    T_, N, K = cube.shape
    rng = np.random.default_rng(11 + h)
    out = {m: np.full((T_, N), np.nan) for m in MODELS}
    fitted, mu, sd = {}, None, None
    for t in range(T_):
        if t % FIT_EVERY == 0 and t - h - 1 > 0:
            X = cube[: t - h - 1].reshape(-1, K); y = Y[: t - h - 1].reshape(-1)
            m = np.isfinite(y) & np.isfinite(X).all(1)
            if m.sum() >= MIN_ROWS:
                A0, y0 = X[m], y[m]
                if len(A0) > TRAIN_CAP:
                    sel = rng.choice(len(A0), TRAIN_CAP, replace=False)
                    A0, y0 = A0[sel], y0[sel]
                mu, sd = A0.mean(0), A0.std(0) + 1e-9
                A = (A0 - mu) / sd
                y0 = (y0 - y0.mean()) / (y0.std() + 1e-12)
                rk = pd.DataFrame(A).rank().to_numpy(); ry = pd.Series(y0).rank().to_numpy()
                sg = np.sign([np.corrcoef(rk[:, j], ry)[0, 1] for j in range(K)])
                fitted = {"EW": ("ew", np.nan_to_num(sg))}
                for nm in MODELS:
                    if nm == "EW" or (nm == "GBM" and not HAVE_LGB):
                        continue
                    try:
                        fitted[nm] = fit(nm, A, y0, rng)
                    except Exception:
                        pass
        if not fitted or mu is None:
            continue
        Xt = cube[t]; ok = np.isfinite(Xt).all(1)
        if ok.sum() < 20:
            continue
        B = (Xt[ok] - mu) / sd
        j = np.flatnonzero(ok)
        for nm, mdl in fitted.items():
            try:
                if mdl[0] == "ew":
                    out[nm][t, j] = (B * mdl[1]).mean(1)
                elif mdl[0] == "lin":
                    out[nm][t, j] = mdl[1].predict(B)
                else:
                    out[nm][t, j] = mdl[1].predict(mdl[2](B))
            except Exception:
                pass
    rows, curves = [], {}
    for nm, v in out.items():
        sc = pd.DataFrame(v, index=fwd.index, columns=fwd.columns)
        if sc.notna().sum().sum() == 0:
            continue
        icd = XS.ic_series(sc.iloc[:split], fwd.iloc[:split])
        ict = XS.ic_series(sc.iloc[split:], fwd.iloc[split:])
        sd_, st_ = XS.ic_stats(icd, h), XS.ic_stats(ict, h)
        pnl = XS.ls_portfolio(sc, r, h)
        rows.append(dict(stage=stage, h=h, model=nm, shift=shift, ic_dev=sd_["ic"], ic_test=st_["ic"],
                         t_test=st_["t"], sr_dev=XS.sharpe(pnl.iloc[:split]), sr_test=XS.sharpe(pnl.iloc[split:])))
        if shift == 0:
            curves[(stage, h, nm)] = pnl
    return rows, curves


def main(workers=6):
    jobs = [(s, h, sh) for s in STAGES for h in HS for sh in range(NULL_SHIFTS + 1)]
    rows, curves = [], {}
    with Pool(workers, initializer=_init) as p:
        for rw, cv in p.imap_unordered(job, jobs):
            rows += rw; curves.update(cv)
            print("done", rw[0]["stage"], rw[0]["h"], rw[0]["shift"], flush=True)
    M = pd.DataFrame(rows); M.to_csv(OUT / "xs_models_big.csv", index=False)
    pd.to_pickle(curves, OUT / "xs_curves_big.pkl")
    pd.set_option("display.width", 270); pd.set_option("display.max_rows", 300)
    print("\n=== all cells ===")
    print(M.sort_values(["stage", "h", "model", "shift"])[
        ["stage", "h", "model", "shift", "ic_dev", "ic_test", "t_test", "sr_dev", "sr_test"]].round(3).to_string(index=False))
    real = M[M["shift"] == 0]; null = M[M["shift"] == 1]
    j = real.merge(null, on=["stage", "h", "model"], suffixes=("", "_n"))
    j["d_ic"] = j.ic_test - j.ic_test_n; j["d_sr"] = j.sr_test - j.sr_test_n
    j.to_csv(OUT / "xs_models_vs_null.csv", index=False)
    print("\n=== model ranking (real minus null, averaged over stage and horizon) ===")
    print(j.groupby("model")[["ic_test", "ic_test_n", "d_ic", "sr_test", "sr_test_n", "d_sr"]].mean()
          .sort_values("d_sr", ascending=False).round(3).to_string())
    print("\n=== preprocessing stage ranking ===")
    print(j.groupby("stage")[["ic_test", "d_ic", "sr_test", "d_sr"]].mean().round(3).to_string())
    print("\n=== non-linear minus RIDGE, per cell (positive = complexity pays) ===")
    p = j.pivot_table(index=["stage", "h"], columns="model", values="sr_test")
    for c in ("GBM", "RFF", "EW"):
        if c in p and "RIDGE" in p:
            p[f"{c}-RIDGE"] = p[c] - p["RIDGE"]
    print(p.round(3).to_string())

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))
    ax = axes[0]
    w = 0.2
    for i, nm in enumerate([m for m in MODELS if m in set(j.model)]):
        g = j[j.model == nm].groupby("stage").d_sr.mean().reindex(STAGES)
        ax.bar(np.arange(len(STAGES)) + (i - 1.5) * w, g.values, w, label=nm)
    ax.set_xticks(range(len(STAGES))); ax.set_xticklabels(STAGES, rotation=12, fontsize=8)
    ax.axhline(0, color="k", lw=.6); ax.set_ylabel("long-short Sharpe minus null")
    ax.set_title("cross-section: does complexity pay?"); ax.legend(fontsize=8); ax.grid(alpha=.3, axis="y")
    ax = axes[1]
    for nm in [m for m in MODELS if m in set(j.model)]:
        g = j[j.model == nm]
        ax.scatter(g.ic_dev, g.ic_test, s=60, label=nm)
    lim = [j[["ic_dev", "ic_test"]].min().min(), j[["ic_dev", "ic_test"]].max().max()]
    ax.plot(lim, lim, ls=":", c="grey"); ax.axhline(0, color="k", lw=.6); ax.axvline(0, color="k", lw=.6)
    ax.set_xlabel("dev IC"); ax.set_ylabel("test IC"); ax.set_title("cross-sectional IC is persistent")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    ax = axes[2]
    for (stage, h, nm), pnl in sorted(curves.items()):
        if stage != "xs_z+sector" or h != 5:
            continue
        pp = pnl.loc[XS.SPLIT:]
        ax.plot((1 + pp).cumprod(), lw=1.3, label=f"{nm}: SR {XS.sharpe(pp):.2f}")
    ax.set_title("decile long-short, sector-neutral, h=5 (test, 5bp/leg)")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(OUT / "fig_xs_models_big.png", dpi=110); plt.close()
    print("\nfigure written")
    return M


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
