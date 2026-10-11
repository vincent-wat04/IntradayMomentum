"""Stage 3 (model) + Stage 4 (strategy):  screened factors -> walk-forward models -> tradable strategies.

PIPELINE (everything point-in-time; every estimate uses only labels that matured by the signal close):

  1. features   1155 columns = 172 raw factors x 7 temporal aggregations, each mapped to a trailing-252d
                z-score (the best map from stage 1).  From timing_transform_study.signal_store.
  2. screening  refit every SCREEN_EVERY sessions.  A column passes if its rank IC has the same sign in
                both halves of the available matured history and |IC| >= IC_MIN; survivors are sorted by
                |IC| and added greedily while |corr| with everything already chosen stays < CORR_MAX.
                Top K_FEAT kept.  (Causal version of the repo's screen_factors.)
  3. models     refit every FIT_EVERY sessions on matured labels only:
                  EW     equal-weight dev-signed mean of the selected z            (Rapach-Strauss-Zhou style)
                  VOTE   diffusion index of signs                                   (Neely et al. 2014)
                  RIDGE  ridge on the selected z
                  GBM    LightGBM, shallow
                  RFF    random-Fourier-feature ridge, P >> N                        (Kelly-Malamud-Zhou 2024
                         "virtue of complexity")
                  AVG    mean of the standardized forecasts of the five above        (forecast combination)
                Two targets: RAW = h-day forward log return; VOLSTD = that divided by sigma_t (known at t),
                which stops 2020 and 2022 from dominating the fit.
  4. strategies each signal s_t -> position, h-day smoothed, next_open execution, cost swept 1/5/10 bp:
                  LS        clip(s / 2 sd_252(s), -1, 1)                 pure timing overlay, beta ~ 0
                  LONGCASH  1{s>0}                                       classic market timing
                  TILT      1 + 0.5 LS, floor 0.5, cap 1.5               AQR "sin a little"
                  VOLTGT    TILT x (target vol / trailing vol), cap 1.5  Harvey et al. 2018
                  OVERLAY   1 + k LS with k set so the overlay runs at OVERLAY_VOL of risk
                  GP        partial adjustment w_t = (1-a) w_{t-1} + a aim_t   Garleanu-Pedersen (2013)
                            "aim in front of the target and trade partially towards the aim"

  NULL: the whole feature panel is circularly shifted in time (cross-correlations kept, link to the future
  broken) and the entire pipeline is re-run, so every number can be read net of its own data-mining baseline.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

import timing_transform_study as T
from factor_sharpe import run_book

warnings.filterwarnings("ignore")
try:
    import lightgbm as lgb
    HAVE_LGB = True
except Exception:
    HAVE_LGB = False

OUT = Path("outputs/full/strategy"); OUT.mkdir(parents=True, exist_ok=True)
HS = (1, 2, 5, 20)
TARGETS = ("RAW", "VOLSTD")
NULL_SHIFTS = 2
SCREEN_EVERY, FIT_EVERY = 63, 21
MIN_TRAIN = 504
K_FEAT, IC_MIN, CORR_MAX = 12, 0.03, 0.7
RFF_P, RFF_GAMMA, RFF_ALPHA = 1000, 0.5, 100.0
VOL_TARGET, OVERLAY_VOL = 0.15, 0.05
COSTS = (1.0, 5.0, 10.0)
MODELS = ("EW", "VOTE", "RIDGE", "GBM", "RFF", "AVG")
STRATS = ("LS", "LONGCASH", "TILT", "VOLTGT", "OVERLAY", "GP")
_G = {}


# ---------------------------------------------------------------- screening
def screen(X: np.ndarray, y: np.ndarray, t: int, h: int):
    """Return (indices, signs) chosen with data available at the close of session t."""
    k = t - h + 1
    if k < MIN_TRAIN:
        return np.array([], int), np.array([])
    Xk, yk = X[:k], y[:k]
    fy = np.isfinite(yk)
    if fy.sum() < MIN_TRAIN:
        return np.array([], int), np.array([])
    half = np.flatnonzero(fy); mid = half[len(half) // 2]
    ry = pd.Series(yk).rank().to_numpy()
    RX = pd.DataFrame(Xk).rank().to_numpy()
    ic = np.full(X.shape[1], np.nan); ic_a = np.full(X.shape[1], np.nan); ic_b = np.full(X.shape[1], np.nan)
    for j in range(X.shape[1]):
        m = fy & np.isfinite(Xk[:, j])
        if m.sum() < MIN_TRAIN // 2:
            continue
        a = m.copy(); a[mid:] = False
        b = m.copy(); b[:mid] = False
        if a.sum() < 60 or b.sum() < 60:
            continue
        ic[j] = np.corrcoef(RX[m, j], ry[m])[0, 1]
        ic_a[j] = np.corrcoef(RX[a, j], ry[a])[0, 1]
        ic_b[j] = np.corrcoef(RX[b, j], ry[b])[0, 1]
    ok = np.isfinite(ic) & (np.sign(ic_a) == np.sign(ic_b)) & (np.abs(ic) >= IC_MIN)
    cand = np.flatnonzero(ok)[np.argsort(-np.abs(ic[ok]))]
    chosen = []
    if len(cand):
        cand = cand[:120]                                   # only the strongest 120 need a correlation matrix
        C = np.abs(np.nan_to_num(pd.DataFrame(Xk[:, cand]).corr().to_numpy()))
        for i in range(len(cand)):
            if all(C[i, q] < CORR_MAX for q in chosen):
                chosen.append(i)
            if len(chosen) >= K_FEAT:
                break
    idx = cand[chosen]
    return idx, np.sign(ic[idx])


# ---------------------------------------------------------------- models
def fit_models(Z: np.ndarray, yt: np.ndarray, rows: np.ndarray, rng):
    """Z = selected features (n x k), yt = training target. Returns dict name -> predictor(z_row)."""
    Ztr, ytr = Z[rows], yt[rows]
    mu, sd = Ztr.mean(0), Ztr.std(0) + 1e-9
    Zs = (Ztr - mu) / sd
    out = {}
    out["RIDGE"] = ("lin", Ridge(alpha=10.0).fit(Zs, ytr), mu, sd)
    if HAVE_LGB:
        m = lgb.LGBMRegressor(n_estimators=120, num_leaves=7, max_depth=3, learning_rate=0.03,
                              min_child_samples=40, subsample=0.7, subsample_freq=1, colsample_bytree=0.7,
                              reg_lambda=1.0, verbose=-1, n_jobs=1, force_col_wise=True)
        m.fit(Ztr, ytr)
        out["GBM"] = ("raw", m, None, None)
    W = rng.standard_normal((Z.shape[1], RFF_P // 2)) * np.sqrt(RFF_GAMMA)
    b = rng.uniform(0, 2 * np.pi, RFF_P // 2)
    phi = lambda A: np.c_[np.cos(A @ W + b), np.sin(A @ W + b)] * np.sqrt(2.0 / RFF_P)
    out["RFF"] = ("rff", Ridge(alpha=RFF_ALPHA).fit(phi(Zs), ytr), mu, sd, W, b, phi)
    return out


def predict(model, z: np.ndarray):
    kind = model[0]
    if kind == "lin":
        _, m, mu, sd = model; return float(m.predict(((z - mu) / sd)[None, :])[0])
    if kind == "raw":
        _, m, _, _ = model; return float(m.predict(z[None, :])[0])
    _, m, mu, sd, W, b, phi = model
    return float(m.predict(phi(((z - mu) / sd)[None, :]))[0])


def build_signals(X, y, h, rv, rng):
    n = len(y)
    sig = {k: np.full(n, np.nan) for k in MODELS}
    Xf = np.nan_to_num(X)
    idx, sgn, models = np.array([], int), np.array([]), {}
    nfeat = np.zeros(n)
    for t in range(n):
        if t % SCREEN_EVERY == 0:
            idx, sgn = screen(X, y, t, h)
        nfeat[t] = len(idx)
        if len(idx) == 0:
            continue
        Z = Xf[:, idx]
        if t % FIT_EVERY == 0:
            k = t - h + 1
            rows = np.flatnonzero(np.isfinite(y[:k]))
            rows = rows[rows >= 0]
            models = fit_models(Z, y, rows, rng) if len(rows) >= MIN_TRAIN else {}
        zt = Z[t]
        if not np.isfinite(X[t, idx]).all():
            zt = np.nan_to_num(zt)
        sig["EW"][t] = float(np.mean(zt * sgn))
        sig["VOTE"][t] = float(np.mean(np.sign(zt) * sgn))
        for nm, m in models.items():
            sig[nm][t] = predict(m, zt)
    # AVG = mean of the five standardized signals (standardisation is trailing, PIT)
    comps = []
    for nm in ("EW", "VOTE", "RIDGE", "GBM", "RFF"):
        s = sig[nm]
        sd = pd.Series(s).rolling(252, min_periods=126).std().shift(1).to_numpy()
        comps.append(s / np.where(sd > 0, sd, np.nan))
    sig["AVG"] = np.nanmean(np.vstack(comps), 0)
    return sig, nfeat


# ---------------------------------------------------------------- strategies
def positions(s: np.ndarray, h: int, rv: np.ndarray, vscale: np.ndarray):
    sd = pd.Series(s).rolling(252, min_periods=126).std().shift(1).to_numpy()
    ls = np.clip(T.smooth(s / np.where(sd > 0, sd, np.nan) / 2.0, h), -1, 1)
    vt = np.clip(VOL_TARGET / np.sqrt(252 * rv), 0.3, 1.5)
    out = {"LS": ls, "LONGCASH": np.where(np.isfinite(ls), (ls > 0).astype(float), np.nan),
           "TILT": np.clip(1 + 0.5 * ls, 0.5, 1.5), "VOLTGT": np.clip((1 + 0.5 * ls) * vt, 0.3, 1.5)}
    k = OVERLAY_VOL / np.sqrt(252 * rv)
    out["OVERLAY"] = 1 + np.clip(k, 0, 1.0) * ls
    # Garleanu-Pedersen: partial adjustment towards the aim, a = 1/(1 + h/2) is a decay matched to the signal
    a = 1.0 / (1.0 + h / 2.0)
    aim = np.clip(1 + 0.5 * np.clip(s / np.where(sd > 0, sd, np.nan) / 2.0, -1, 1), 0.5, 1.5)
    w = np.full(len(s), np.nan); prev = 1.0
    for t in range(len(s)):
        if np.isfinite(aim[t]):
            prev = (1 - a) * prev + a * aim[t]
        w[t] = prev
    out["GP"] = w
    return out


def evaluate(panel, w, bh, y, split, fee):
    r = run_book(panel, pd.Series(np.nan_to_num(w), index=panel.dates), fee_bp=fee)
    rt, b = r.loc["2023":], bh.loc["2023":]
    eq = (1 + rt).cumprod(); dd = eq / eq.cummax() - 1
    beta = np.cov(rt, b)[0, 1] / b.var()
    alpha = float((rt - beta * b).mean() * 252)
    ann = float(eq.iloc[-1] ** (252 / len(rt)) - 1)
    bann = float((1 + b).prod() ** (252 / len(b)) - 1)
    return dict(sr_dev=T.sr(r.loc[:"2022"]), sr_test=T.sr(rt), ann=ann, vol=float(rt.std() * np.sqrt(252)),
                mdd=float(dd.min()), excess_ann=ann - bann, alpha=alpha, beta=float(beta),
                ir=T.sr(rt - beta * b), calmar=ann / abs(dd.min()) if dd.min() < 0 else np.nan,
                turnover=float(np.nanmean(np.abs(np.diff(np.nan_to_num(w)))) * 252),
                avg_w=float(np.nanmean(w[split:])), ic_test=T.spearman(w[split:], y[split:])), r


# ---------------------------------------------------------------- driver
def job(args):
    sym, h, tgt, shift = args
    G = _G[sym]; panel, split, bh = G["panel"], G["split"], G["bh"]
    y0 = G["Y"][h]; rv, vscale = G["rv"], G["vscale"]
    X = G["X"].copy()
    if shift:
        k = int(np.random.default_rng(7000 + 13 * shift + h).integers(252, len(X) - 252))
        X = np.roll(X, k, axis=0)
    y = y0 / np.sqrt(h * rv) if tgt == "VOLSTD" else y0
    rng = np.random.default_rng(12345 + h)
    sig, nfeat = build_signals(X, y, h, rv, rng)
    rows, curves = [], {}
    for mname, s in sig.items():
        pos = positions(s, h, rv, vscale)
        for sname, w in pos.items():
            for fee in COSTS:
                m, r = evaluate(panel, w, bh, y0, split, fee)
                rows.append(dict(symbol=sym, h=h, target=tgt, shift=shift, model=mname, strategy=sname,
                                 fee_bp=fee, n_feat=float(np.mean(nfeat[split:])), **m))
                if fee == COSTS[0] and shift == 0:
                    curves[(mname, sname)] = r
    return rows, (curves if shift == 0 else None), args


def _init():
    store = pd.read_pickle("outputs/full/transform/signal_store.pkl")
    for sym in ("SPY", "QQQ"):
        G = T.load_all(sym)
        keys = sorted(k for k in store if k[0] == sym and k[3] == "Z252")
        G["X"] = np.column_stack([store[k].astype(float) for k in keys])
        G["cols"] = [(k[1], k[2]) for k in keys]
        _G[sym] = G


def main(workers=8):
    jobs = [(s, h, t, sh) for s in ("SPY", "QQQ") for h in HS for t in TARGETS for sh in range(NULL_SHIFTS + 1)]
    rows, curves = [], {}
    with Pool(workers, initializer=_init) as pool:
        for r, c, key in pool.imap_unordered(job, jobs):
            rows += r
            if c: curves[key[:3]] = c
            print("done", key, flush=True)
    df = pd.DataFrame(rows); df.to_csv(OUT / "model_strategy_results.csv", index=False)
    pd.to_pickle(curves, OUT / "model_strategy_curves.pkl")
    return df


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8)
