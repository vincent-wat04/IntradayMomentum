"""Stage 2: combining many time-series predictors into ONE timing signal (SPY / QQQ, h = 5, 20, next_open).

Input: the PIT rolling z-scores Z252 of every raw factor x aggregation (signal_store.pkl from
timing_transform_study.py).  Pools: INTRA (intraday/price + noise-band), MACRO, ALL.

Methods (all point-in-time; anything estimated uses only labels matured by the signal close):
  EW_DEV    equal weight of dev-signed z (sign = Spearman IC on 2019-22 matured labels), no selection
  EW_SEL    same, only columns whose dev IC has the same sign in 2019-20 and 2021-22 and |IC| >= 0.05
  VOTE      diffusion index = mean of dev-signed sign(z)            (Neely, Rapach, Tu & Zhou 2014)
  RIC_W     sum_j ric_j z_j / sum_j |ric_j|, ric = expanding rank IC (refit every 21d)   ("IC weighting")
  RSZ_MEAN  mean of univariate OLS forecasts (deviation from hist. mean)  (Rapach, Strauss & Zhou 2010)
  RSZ_MED   median of the same
  PCA3      expanding PCA (3 comps) of z -> OLS of y on scores, refit every 63d   (Neely et al. 2014)
  PLS1      expanding 1-component PLS of y on z (= 3PRF), refit every 63d  (Kelly & Pruitt 2015; Huang et al. 2015)
  CENET     positive LASSO (BIC) of y on the univariate forecasts, average the selected ones, refit 63d
            (Rapach & Zhou C-ENet; Dong, Li, Rapach & Zhou 2022)
Position maps for every composite s_t:  LS = clip(s / (2 sd_252(s)), -1, 1) (sd from t-252..t-1), h-day smoothed;
  TILT = clip(1 + 0.5 LS, 0.5, 1.5) (AQR "Sin a little": 100% +/- tilt, floor 50%, cap 150%);
  MV   = clip(mu_hat/(gamma sigma^2), 0, 1.5) with mu_hat = hist. mean + forecast  (Campbell-Thompson; gamma = 3)
        vs MV0 (hist. mean only), for forecast-type methods.
Null: the whole z panel of a symbol is circularly shifted by a random offset (cross-correlations kept, link to
future returns broken), NULL_SHIFTS times; every method is recomputed and evaluated on the real returns.
"""
from __future__ import annotations

import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import LassoLarsIC

import timing_transform_study as T
from factor_sharpe import run_book

warnings.filterwarnings("ignore")
OUT = T.OUT
HS2 = (5, 20)
POOLS = ("INTRA", "MACRO", "ALL")
NULL_SHIFTS = 4
GAMMA = 3.0
REFIT = 63
_G = {}


# ------------------------------------------------------------------ helpers
def roll_sd(s, n=252, minp=126):
    return pd.Series(s).rolling(n, min_periods=minp).std().shift(1).to_numpy()


def to_ls(s, h):
    p = np.clip(s / (2 * roll_sd(s)), -1, 1)
    return np.clip(T.smooth(p, h), -1, 1)


def matured_reg_mat(Z, y, h, minn=T.MIN_REG):
    """Column-wise expanding OLS deviation forecasts d_jt = b_jt (z_jt - zbar_jt), PIT."""
    n = len(y); ok = np.isfinite(Z) & np.isfinite(y)[:, None]
    zs, ys = np.where(ok, Z, 0.0), np.where(ok, y[:, None], 0.0)
    c = lambda v: np.vstack([np.zeros((1, v.shape[1])), np.cumsum(v, 0)])
    N, SX, SY, SXX, SXY = c(ok.astype(float)), c(zs), c(ys), c(zs * zs), c(zs * ys)
    k = np.clip(np.arange(n) - h + 1, 0, n)
    Nk, sx, sy, sxx, sxy = N[k], SX[k], SY[k], SXX[k], SXY[k]
    with np.errstate(all="ignore"):
        xb, yb = sx / Nk, sy / Nk
        b = (sxy - Nk * xb * yb) / (sxx - Nk * xb * xb)
    b[Nk < minn] = np.nan
    return b * (Z - xb)


def rank_ic_mat(Z, y, h, every=21, minn=T.MIN_REG):
    n, P = Z.shape; out = np.full((n, P), np.nan); cur = np.full(P, np.nan)
    for t in range(n):
        k = t - h + 1
        if t % every == 0 and k > minn:
            Zk, yk = Z[:k], y[:k]
            fy = np.isfinite(yk)
            ry = pd.Series(yk).rank().to_numpy()
            RZ = pd.DataFrame(Zk).rank().to_numpy()
            for j in range(P):
                m = fy & np.isfinite(Zk[:, j])
                if m.sum() >= minn:
                    a, b = RZ[m, j], ry[m]
                    cur[j] = np.corrcoef(a, b)[0, 1]
                else:
                    cur[j] = np.nan
        out[t] = cur
    return out


def dev_ic(Z, y, h, split, lo=None, hi=None):
    n = len(y); t = np.arange(n)
    m0 = (t + h < split) & np.isfinite(y)
    if lo is not None: m0 &= (t >= lo) & (t < hi)
    out = np.full(Z.shape[1], np.nan)
    for j in range(Z.shape[1]):
        m = m0 & np.isfinite(Z[:, j])
        if m.sum() > 100 and np.nanstd(Z[m, j]) > 0:
            out[j] = pd.Series(Z[m, j]).rank().corr(pd.Series(y[m]).rank())
    return out


def refit_models(Z, y, h, ybar, D, every=REFIT, minn=T.MIN_REG):
    """PCA3 / PLS1 / CENET forecasts (deviation from hist. mean), PIT, refit every `every` sessions."""
    n, P = Z.shape
    out = {k: np.full(n, np.nan) for k in ("PCA3", "PLS1", "CENET")}
    Zf = np.nan_to_num(Z); Df = np.nan_to_num(D)
    models = {}
    for t in range(n):
        k = t - h + 1
        if t % every == 0 and k > minn:
            rows = np.arange(k)[np.isfinite(y[:k]) & (np.isfinite(Z[:k]).mean(1) > 0.5)]
            if len(rows) >= minn:
                yy = y[rows] - ybar[t]
                cols = np.isfinite(Z[rows]).mean(0) > 0.8
                pca = PCA(3).fit(Zf[: t + 1][:, cols])            # unsupervised: may use z up to t
                S = pca.transform(Zf[rows][:, cols])
                bS = np.linalg.lstsq(np.c_[np.ones(len(rows)), S], yy, rcond=None)[0]
                pls = PLSRegression(1, scale=False).fit(Zf[rows][:, cols], yy)
                rowsD = rows[np.isfinite(D[rows]).mean(1) > 0.5]
                en = None
                if len(rowsD) >= minn:
                    yd = y[rowsD] - ybar[t]
                    en = LassoLarsIC(criterion="bic", positive=True, fit_intercept=False, noise_variance=float(np.var(yd)))
                    en.fit(Df[rowsD], yd)
                models = dict(cols=cols, pca=pca, bS=bS, pls=pls, en=en)
        if models:
            c = models["cols"]; z = Zf[t:t + 1][:, c]
            out["PCA3"][t] = models["bS"][0] + models["bS"][1:] @ models["pca"].transform(z)[0]
            out["PLS1"][t] = float(models["pls"].predict(z).ravel()[0])
            if models["en"] is not None:
                sel = models["en"].coef_ > 0
                out["CENET"][t] = float(Df[t, sel].mean()) if sel.any() else 0.0
    return out


def book_metrics(panel, w, bh, y, split):
    r = run_book(panel, pd.Series(np.nan_to_num(w), index=panel.dates))
    rt, rd = r.loc["2023":], r.loc[:"2022"]
    eq = (1 + rt).cumprod()
    return dict(sr_dev=T.sr(rd), sr_test=T.sr(rt), ann_test=float(eq.iloc[-1] ** (252 / len(rt)) - 1),
                vol_test=float(rt.std() * np.sqrt(252)), mdd_test=float((eq / eq.cummax() - 1).min()),
                cer_test=float((rt.mean() - GAMMA / 2 * rt.var()) * 252), ic_test=T.spearman(w[split:], y[split:]),
                avg_w=float(np.nanmean(w[split:]))), r


# ------------------------------------------------------------------ one (sym, h, pool, shift) run
def run_combo(job):
    sym, h, pool, shift = job
    G = _G[sym]; panel, split, y = G["panel"], G["split"], G["Y"][h]
    Z = G["Z"][pool].copy()
    if shift:
        k = int(np.random.default_rng(1000 * shift + h + len(pool)).integers(252, len(Z) - 252))
        Z = np.roll(Z, k, axis=0)
    ybar = T.ybar_matured(y, h)
    ic_dev = dev_ic(Z, y, h, split)
    yrs = np.array([d[:4] for d in panel.dates])
    lo21 = int(np.searchsorted(yrs, "2021"))
    ic_a, ic_b = dev_ic(Z, y, h, split, 0, lo21), dev_ic(Z, y, h, split, lo21, split)
    sgn = np.sign(np.nan_to_num(ic_dev))
    sel = (np.sign(ic_a) == np.sign(ic_b)) & (np.abs(ic_dev) >= 0.05)
    D = matured_reg_mat(Z, y, h)
    RIC = rank_ic_mat(Z, y, h)
    comp = {
        "EW_DEV": np.nanmean(Z * sgn, 1),
        "EW_SEL": np.nanmean(Z[:, sel] * sgn[sel], 1) if sel.any() else np.full(len(y), np.nan),
        "VOTE": np.nanmean(np.sign(Z) * sgn, 1),
        "RIC_W": np.nansum(RIC * Z, 1) / np.nansum(np.abs(RIC) * np.isfinite(Z), 1),
    }
    fc = {"RSZ_MEAN": np.nanmean(D, 1), "RSZ_MED": np.nanmedian(D, 1)}
    fc.update(refit_models(Z, y, h, ybar, D))
    comp.update(fc)
    rows, curves = [], {}
    bh = G["bh"]
    for name, s in comp.items():
        s = np.where(np.isfinite(s), s, np.nan)
        ls = to_ls(s, h)
        base = dict(symbol=sym, h=h, pool=pool, shift=shift, method=name, n_cols=Z.shape[1], n_sel=int(sel.sum()))
        m, r = book_metrics(panel, ls, bh, y, split); rows.append({**base, "book": "LS", **m}); curves[(name, "LS")] = r
        tilt = np.clip(1 + 0.5 * np.nan_to_num(ls), 0.5, 1.5)
        m, r = book_metrics(panel, tilt, bh, y, split); rows.append({**base, "book": "TILT", **m}); curves[(name, "TILT")] = r
        if name in fc:
            rows[-2]["r2os_test"] = T.r2_os(y, ybar, s, split, h)
            mu = (ybar + np.nan_to_num(s)) / h
            w = np.clip(T.smooth(mu / (GAMMA * G["rv"]), h), 0, 1.5)
            m, r = book_metrics(panel, w, bh, y, split); rows.append({**base, "book": "MV", **m}); curves[(name, "MV")] = r
    w0 = np.clip(T.smooth(ybar / h / (GAMMA * G["rv"]), h), 0, 1.5)
    m, r = book_metrics(panel, w0, bh, y, split)
    rows.append(dict(symbol=sym, h=h, pool=pool, shift=shift, method="HISTMEAN", book="MV", **m)); curves[("HISTMEAN", "MV")] = r
    return rows, (curves if shift == 0 else None), (sym, h, pool, shift)


def _init():
    store = pd.read_pickle(OUT / "signal_store.pkl")
    for sym in ("SPY", "QQQ"):
        G = T.load_all(sym)
        keys = sorted(k for k in store if k[0] == sym and k[3] == "Z252")
        pools = {"INTRA": [k for k in keys if k[1][0] in "IN"], "MACRO": [k for k in keys if k[1][0] == "M"], "ALL": keys}
        G["Z"] = {p: np.column_stack([store[k].astype(float) for k in ks]) for p, ks in pools.items()}
        G["cols"] = {p: [(k[1], k[2]) for k in ks] for p, ks in pools.items()}
        _G[sym] = G


def main(workers=8):
    jobs = [(s, h, p, sh) for s in ("SPY", "QQQ") for h in HS2 for p in POOLS for sh in range(NULL_SHIFTS + 1)]
    rows, curves = [], {}
    with Pool(workers, initializer=_init) as pool:
        for r, c, key in pool.imap_unordered(run_combo, jobs):
            rows += r
            if c: curves[key[:3]] = c
            print("done", key, flush=True)
    df = pd.DataFrame(rows); df.to_csv(OUT / "combination_results.csv", index=False)
    pd.to_pickle(curves, OUT / "combination_curves.pkl")
    return df


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8)
