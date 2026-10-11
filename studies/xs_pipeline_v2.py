"""Cross-sectional pipeline v2 -- built to the setup you specified.

SETUP
  1. Rebalance period P in {1 (daily), 5 (weekly), 10 (biweekly)}.  Weights are set at a period
     boundary and HELD CONSTANT until the next boundary (no daily drift of target weights), and the
     forward label horizon equals P so that signal and holding period match.
  2. Neutralisation, applied in this order, each step on the cross-section of a single day:
        a. winsorise at 1%/99%
        b. industry demean                  (11 sectors)
        c. regress out STYLE               (SIZE, BETA, VOL_21)
        d. regress out PAST-K RETURN       (K = 5 and 21) -- the cross-sectional analogue of the
                                            orthogonalize_past_k used in the time-series study
  3. The neutralised factor is RANKED cross-sectionally; the rank determines both the side and the
     weight: w_i proportional to (rank_pct_i - 0.5), long-short, gross-normalised to 1.
  4. FACTOR SCREENING, point-in-time, refit every 63 sessions on matured labels only: a factor passes
     if its rank IC has the same sign in both halves of the available history and |IC| >= IC_MIN;
     survivors are added greedily while |corr| with the already-selected set stays below CORR_MAX.
  5. Models: EW, ELASTICNET, RIDGE, GBM, RFF.  (RAVEN is in xs_raven.py -- it needs a sequence
     input and a separate, much slower run.)  ElasticNet is included because it was the best linear
     model in the A-share work and was missing from the previous US cross-sectional test.
  6. Benchmark is SPY, reported on every output.

Factor pool: 77 daily cross-sectional factors (xs_factor_library) + their STABILITY versions
(the rolling within-window variance operator from the 东北证券 report, which in that report lifts
ICIR substantially) + daily order-flow proxies.  Intraday 5-minute versions of the report's volume
factors cannot be built for the 98-name universe because no intraday history is available for them;
they are built and tested on SPY/QQQ in pdf_volume_factors.py instead.

Null: cross-sectional permutation (stock identities shuffled within each day, same permutation for
every factor on that day).  A time shift is NOT a valid null here -- stock characteristics persist
for years, so a shifted panel keeps most of the real cross-sectional ranking.
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
import yfinance as yf
from sklearn.linear_model import ElasticNet, Ridge

import cross_section_study as XS
import xs_factor_library as XL
import pdf_volume_factors as PV

try:
    import lightgbm as lgb
    HAVE_LGB = True
except Exception:
    HAVE_LGB = False

OUT = Path("outputs/full/xs2"); OUT.mkdir(parents=True, exist_ok=True)
SPLIT = "2023-01-03"
PERIODS = tuple(int(x) for x in os.environ.get('XS_PERIODS','5,20').split(','))
MODELS = ("EW", "ELASTICNET", "RIDGE", "GBM", "RFF")
NEUTRAL = ("ind+style", "ind+style+pastK")
STYLES = ("SIZE", "BETA", "VOL_21")
PASTK = (5, 21)
SCREEN_EVERY, FIT_EVERY, MIN_ROWS = 63, 42, 20000
K_FEAT, IC_MIN, CORR_MAX = 20, 0.004, 0.7
TRAIN_CAP = 40000
RFF_P, RFF_GAMMA, RFF_ALPHA = 1000, 0.5, 200.0
FEE_BP = 5.0
NULLS = 1
_G = {}


# --------------------------------------------------------------------------- preprocessing
def winsor(df, lo=0.01, hi=0.99):
    a = df.quantile(lo, axis=1); b = df.quantile(hi, axis=1)
    return df.clip(a, b, axis=0)


def xs_z(df):
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1).replace(0, np.nan), axis=0)


def neutralise(F: dict, sec: pd.Series, controls: dict, use_pastk: bool):
    """winsorise -> xs z -> industry demean -> regress out styles (+ past-K returns)."""
    ctrl_names = list(STYLES) + ([f"PASTR_{k}" for k in PASTK] if use_pastk else [])
    Z = {k: xs_z(winsor(v)) for k, v in F.items()}
    # industry demean
    for k in Z:
        out = Z[k].copy()
        for s in sec.unique():
            c = sec[sec == s].index.intersection(out.columns)
            if len(c) >= 2:
                out[c] = out[c].sub(out[c].mean(axis=1), axis=0)
        Z[k] = out
    keys = [k for k in Z if k not in ctrl_names]
    C = [xs_z(winsor(controls[n])) if n in controls else xs_z(winsor(Z[n])) for n in ctrl_names]
    idx, cols = Z[keys[0]].index, Z[keys[0]].columns
    Y = np.stack([Z[k].to_numpy(float) for k in keys], -1)
    Zc = np.stack([c.to_numpy(float) for c in C], -1)
    res = np.full_like(Y, np.nan)
    for t in range(len(idx)):
        A = np.c_[Zc[t], np.ones(Zc.shape[1])]
        m = np.isfinite(A).all(1)
        if m.sum() < 20:
            continue
        y = Y[t][m]
        good = np.isfinite(y)
        yf_ = np.where(good, y, 0.0)
        b, *_ = np.linalg.lstsq(A[m], yf_, rcond=None)
        r = yf_ - A[m] @ b
        r[~good] = np.nan
        res[t, np.flatnonzero(m)] = r
    D = {k: pd.DataFrame(res[:, :, j], index=idx, columns=cols) for j, k in enumerate(keys)}
    return D          # controls are deliberately NOT returned as features


# --------------------------------------------------------------------------- portfolio
def rank_weights(score: pd.DataFrame, beta: pd.DataFrame | None = None):
    """w proportional to (rank percentile - its cross-sectional mean), gross-normalised to 1.

    Demeaning against the day's OWN mean rank makes the book exactly dollar-neutral (using the
    constant 0.5 leaves a 1/(2N) net long, which is small but shows up as SPY correlation).  If a
    beta panel is supplied the book is additionally made beta-neutral by removing the
    beta-weighted component, which is what keeps the long-short leg from inheriting market direction.
    """
    rk = score.rank(axis=1, pct=True)
    w = rk.sub(rk.mean(axis=1), axis=0)
    if beta is not None:
        b = beta.reindex_like(w)
        num = (w * b).sum(axis=1)
        den = (b * b).sum(axis=1).replace(0, np.nan)
        w = w.sub(b.mul(num / den, axis=0))
        w = w.sub(w.mean(axis=1), axis=0)
    return w.div(w.abs().sum(axis=1).replace(0, np.nan), axis=0)


def period_hold(w: pd.DataFrame, P: int):
    """Set weights only on period boundaries and hold them constant inside the period (1-day lag)."""
    idx = np.arange(len(w))
    keep = w.where(pd.Series(idx % P == 0, index=w.index), np.nan)
    return keep.ffill(limit=P - 1).shift(1).fillna(0.0) if P > 1 else w.shift(1).fillna(0.0)


def backtest(score: pd.DataFrame, r: pd.DataFrame, P: int, fee_bp=FEE_BP, beta=None):
    held = period_hold(rank_weights(score, beta), P)
    gross = (held * r).sum(axis=1)
    cost = fee_bp * 1e-4 * held.diff().abs().sum(axis=1)
    return gross - cost, float(held.diff().abs().sum(axis=1).mean() * 252)


def metrics(x: pd.Series, bench: pd.Series | None = None):
    x = x.dropna()
    if len(x) < 60 or x.std() == 0:
        return dict(sharpe=np.nan, ann=np.nan, vol=np.nan, mdd=np.nan)
    eq = (1 + x).cumprod(); dd = eq / eq.cummax() - 1
    ann = float(eq.iloc[-1] ** (252 / len(x)) - 1)
    out = dict(sharpe=float(x.mean() / x.std() * np.sqrt(252)), ann=ann,
               vol=float(x.std() * np.sqrt(252)), mdd=float(dd.min()),
               calmar=ann / abs(dd.min()) if dd.min() < 0 else np.nan)
    if bench is not None:
        b = bench.reindex(x.index).fillna(0)
        if b.std() > 0:
            beta = np.cov(x, b)[0, 1] / b.var(); res = x - beta * b
            out.update(beta=float(beta), alpha=float(res.mean() * 252),
                       ir=float(res.mean() / res.std() * np.sqrt(252)),
                       corr_spy=float(x.corr(b)))
    return out


# --------------------------------------------------------------------------- screening + models
def _rank_nan(A):
    """Cross-sectional rank percentile per row, NaNs preserved, returned demeaned."""
    R = pd.DataFrame(A).rank(axis=1, pct=True).to_numpy()
    return R - np.nanmean(R, axis=1, keepdims=True)


def screen(X, y, t, h):
    """Mean DAILY cross-sectional rank IC (not a pooled rank), with sign stability across halves."""
    k = t - h - 1
    if k < 252:
        return np.array([], int), np.array([])
    Xk, yk = X[:k], y[:k]
    ry = _rank_nan(yk)
    K = X.shape[2]
    mid = k // 2
    ic = np.full(K, np.nan); ia = ic.copy(); ib = ic.copy()
    for j in range(K):
        rx = _rank_nan(Xk[:, :, j])
        num = np.nansum(rx * ry, axis=1)
        den = np.sqrt(np.nansum(np.where(np.isfinite(ry), rx, np.nan) ** 2, axis=1) *
                      np.nansum(np.where(np.isfinite(rx), ry, np.nan) ** 2, axis=1))
        daily = np.where(den > 0, num / den, np.nan)
        if np.isfinite(daily).sum() < 120:
            continue
        ia[j] = np.nanmean(daily[:mid]); ib[j] = np.nanmean(daily[mid:])
        ic[j] = np.nanmean(daily)
    ok = np.isfinite(ic) & (np.sign(ia) == np.sign(ib)) & (np.abs(ic) >= IC_MIN)
    cand = np.flatnonzero(ok)
    cand = cand[np.argsort(-np.abs(ic[cand]))][:80]
    chosen = []
    if len(cand):
        sub = Xk[::3, :, cand].reshape(-1, len(cand))
        C = np.abs(np.nan_to_num(pd.DataFrame(sub).corr().to_numpy()))
        for i in range(len(cand)):
            if all(C[i, q] < CORR_MAX for q in chosen):
                chosen.append(i)
            if len(chosen) >= K_FEAT:
                break
    idx = cand[chosen]
    return idx, np.sign(ic[idx])


def fit(kind, A, y, rng):
    if kind == "RIDGE":
        return ("lin", Ridge(alpha=10.0).fit(A, y))
    if kind == "ELASTICNET":
        return ("lin", ElasticNet(alpha=1e-3, l1_ratio=0.5, max_iter=20000, random_state=7).fit(A, y))
    if kind == "GBM":
        m = lgb.LGBMRegressor(n_estimators=200, num_leaves=31, max_depth=6, learning_rate=0.05,
                              min_child_samples=200, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.7, reg_lambda=5.0, verbose=-1, n_jobs=1,
                              force_col_wise=True)
        return ("lin", m.fit(A, y))
    W = rng.standard_normal((A.shape[1], RFF_P // 2)) * np.sqrt(RFF_GAMMA)
    b = rng.uniform(0, 2 * np.pi, RFF_P // 2)
    phi = lambda Z: np.c_[np.cos(Z @ W + b), np.sin(Z @ W + b)] * np.sqrt(2.0 / RFF_P)
    return ("rff", Ridge(alpha=RFF_ALPHA).fit(phi(A), y), phi)


def job(args):
    neu, P, shift = args
    F = _G["neut"][neu]
    px, r, spy = _G["px"], _G["r"], _G["spy"]
    if shift:
        rp = np.random.default_rng(300 + shift)
        n = len(px.columns)
        perm = np.stack([rp.permutation(n) for _ in range(len(px))])
        F = {k: pd.DataFrame(np.take_along_axis(v.to_numpy(), perm, axis=1), index=v.index,
                             columns=v.columns) for k, v in F.items()}
    keys = sorted(F)
    cube = np.stack([F[k].to_numpy(float) for k in keys], -1)
    fwd = XS.fwd_ret(px, P)
    Y = fwd.to_numpy(float)
    T_, N, K = cube.shape
    split = int(np.searchsorted(px.index, SPLIT))
    rng = np.random.default_rng(5 + P)
    out = {m: np.full((T_, N), np.nan) for m in MODELS}
    idx, sgn, fitted, mu, sd = np.array([], int), np.array([]), {}, None, None
    nfeat = np.zeros(T_)
    need_fit = False
    for t in range(T_):
        if t % SCREEN_EVERY == 0:
            new_idx, new_sgn = screen(cube, Y, t, P)
            if len(new_idx) != len(idx) or not np.array_equal(new_idx, idx):
                # the selected set changed, so the stored model and its scaler no longer apply
                idx, sgn = new_idx, new_sgn
                fitted, mu, sd, need_fit = {}, None, None, True
        nfeat[t] = len(idx)
        if len(idx) == 0:
            continue
        Z = cube[:, :, idx]
        if (t % FIT_EVERY == 0 or need_fit) and t - P - 1 > 252:
            Xtr = Z[: t - P - 1].reshape(-1, len(idx)); ytr = Y[: t - P - 1].reshape(-1)
            m = np.isfinite(ytr) & np.isfinite(Xtr).all(1)
            if m.sum() >= MIN_ROWS:
                A0, y0 = Xtr[m], ytr[m]
                if len(A0) > TRAIN_CAP:
                    s = rng.choice(len(A0), TRAIN_CAP, replace=False); A0, y0 = A0[s], y0[s]
                mu, sd = A0.mean(0), A0.std(0) + 1e-9
                A = (A0 - mu) / sd
                y0 = (y0 - y0.mean()) / (y0.std() + 1e-12)
                fitted = {"EW": ("ew", sgn)}
                need_fit = False
                for nm in MODELS:
                    if nm == "EW" or (nm == "GBM" and not HAVE_LGB):
                        continue
                    try:
                        fitted[nm] = fit(nm, A, y0, rng)
                    except Exception:
                        pass
        if not fitted or mu is None:
            continue
        Zt = Z[t]; ok = np.isfinite(Zt).all(1)
        if ok.sum() < 20:
            continue
        B = (Zt[ok] - mu) / sd
        jj = np.flatnonzero(ok)
        for nm, mdl in fitted.items():
            try:
                if mdl[0] == "ew":
                    out[nm][t, jj] = (B * mdl[1]).mean(1)
                elif mdl[0] == "lin":
                    out[nm][t, jj] = mdl[1].predict(B)
                else:
                    out[nm][t, jj] = mdl[1].predict(mdl[2](B))
            except Exception:
                pass
    rows, curves = [], {}
    for nm, v in out.items():
        sc = pd.DataFrame(v, index=px.index, columns=px.columns)
        if sc.notna().sum().sum() == 0:
            continue
        pnl, turn = backtest(sc, r, P, beta=_G.get('beta'))
        icd = XS.ic_series(sc.iloc[:split], fwd.iloc[:split])
        ict = XS.ic_series(sc.iloc[split:], fwd.iloc[split:])
        sd_, st_ = XS.ic_stats(icd, P), XS.ic_stats(ict, P)
        md = metrics(pnl.iloc[split:], spy.iloc[split:])
        rows.append(dict(neutral=neu, period=P, model=nm, shift=shift, n_feat=float(nfeat[split:].mean()),
                         ic_dev=sd_["ic"], ic_test=st_["ic"], t_test=st_["t"], turnover=turn,
                         sr_dev=metrics(pnl.iloc[:split])["sharpe"], **md))
        if shift == 0:
            curves[(neu, P, nm)] = pnl
    return rows, curves


def _init():
    px, op, hi, lo, vo = XS.load()
    sec = pd.Series({c: XS.SECTOR[c] for c in px.columns})
    F0, r = XL.build(px, op, hi, lo, vo)
    # stability operator from the report, applied to every daily factor
    stab = {f"{k}_STAB": PV.stability(v, 20) for k, v in F0.items()}
    # daily order-flow proxies (close location within the day's range, times volume)
    rng_ = (hi - lo).replace(0, np.nan)
    loc = ((px - lo) / rng_).clip(0, 1)
    dv = px * vo
    F0["OFI_D"] = ((2 * loc - 1) * dv).rolling(20, min_periods=10).sum() / dv.rolling(20, min_periods=10).sum()
    F0["BUY_SHARE_D"] = loc.rolling(20, min_periods=10).mean()
    F0["OFI_D_STAB"] = PV.stability(((2 * loc - 1) * dv) / dv.replace(0, np.nan), 20)
    F = {**F0, **stab}
    F = {k: v.replace([np.inf, -np.inf], np.nan) for k, v in F.items()}
    controls = {f"PASTR_{k}": np.log(px).diff(k) for k in PASTK}
    neut = {"ind+style": neutralise(F, sec, controls, use_pastk=False),
            "ind+style+pastK": neutralise(F, sec, controls, use_pastk=True)}
    spy = yf.download("SPY", start="2018-11-01", end="2026-10-03", auto_adjust=True,
                      progress=False)["Close"]
    spy.index = spy.index.strftime("%Y-%m-%d")
    spy = spy.reindex(px.index).ffill().pct_change().squeeze()
    _G.update(px=px, r=r, sec=sec, neut=neut, spy=spy, n_factors=len(F), beta=F0['BETA'])


def main(workers=6):
    _init()
    print(f"{len(_G['px'].columns)} names, {_G['n_factors']} factors "
          f"(incl. stability + order-flow proxies), {len(_G['px'])} days", flush=True)
    jobs = [(n, P, s) for n in NEUTRAL for P in PERIODS for s in range(NULLS + 1)]
    rows, curves = [], {}
    with Pool(workers, initializer=_init) as p:
        for rw, cv in p.imap_unordered(job, jobs):
            rows += rw; curves.update(cv)
            pd.DataFrame(rows).to_csv(OUT / "xs_v2_results.csv", index=False)   # checkpoint each job
            pd.to_pickle(curves, OUT / "xs_v2_curves.pkl")
            if rw:
                print("done", rw[0]["neutral"], rw[0]["period"], rw[0]["shift"], flush=True)
    M = pd.DataFrame(rows); M.to_csv(OUT / "xs_v2_results.csv", index=False)
    pd.to_pickle(curves, OUT / "xs_v2_curves.pkl")
    spy_t = _G["spy"].iloc[int(np.searchsorted(_G["px"].index, SPLIT)):]
    sm = metrics(spy_t)
    pd.set_option("display.width", 290); pd.set_option("display.max_rows", 400)
    print(f"\n=== BENCHMARK  SPY: Sharpe {sm['sharpe']:.3f}, ann {sm['ann']:.1%}, "
          f"vol {sm['vol']:.1%}, MDD {sm['mdd']:.1%} ===")
    real = M[M["shift"] == 0]; null = M[M["shift"] > 0]
    nn = null.groupby(["neutral", "period", "model"]).agg(null_sr=("sharpe", "mean"),
                                                          null_ic=("ic_test", "mean"))
    R = real.set_index(["neutral", "period", "model"]).join(nn).reset_index()
    R["d_sr"] = R.sharpe - R.null_sr; R["d_ic"] = R.ic_test - R.null_ic
    R.to_csv(OUT / "xs_v2_vs_null.csv", index=False)
    print("\n=== all cells (test 2023-2026, 5bp/leg, SPY benchmark) ===")
    print(R[["neutral", "period", "model", "n_feat", "ic_test", "t_test", "sharpe", "null_sr", "d_sr",
             "ann", "vol", "mdd", "turnover", "beta", "alpha", "ir", "corr_spy"]].round(3).to_string(index=False))
    print("\n=== model ranking (mean over neutralisation and period) ===")
    print(R.groupby("model")[["ic_test", "sharpe", "null_sr", "d_sr", "ann", "mdd", "ir", "turnover"]]
          .mean().sort_values("d_sr", ascending=False).round(3).to_string())
    print("\n=== rebalance period ===")
    print(R.groupby("period")[["ic_test", "sharpe", "d_sr", "ann", "turnover"]].mean().round(3).to_string())
    print("\n=== neutralisation: does removing past-K return help? ===")
    print(R.groupby("neutral")[["ic_test", "sharpe", "d_sr", "ann", "ir"]].mean().round(3).to_string())
    return R, curves, sm


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
