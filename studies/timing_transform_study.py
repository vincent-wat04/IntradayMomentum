"""How should a raw time-series predictor be turned into a timing position?  (SPY / QQQ, 2019-01 .. 2026-10)

Stage 1 - single factors.  Every RAW daily factor (26 intraday/price incl. the noise-band MID_MED / MID_AVG /
MID_END / OUT_AVG / SKEW_DIFF, + ~100 PIT macro) goes through

  temporal aggregation  A in {L1, MA5, MA20, EWM10, MED20, DEV60, CHG20}
  normalisation / map   N in {PCT252, PCTEXP, Z252, ZEXP, RZ252, SIGN, TAIL, VOLZ, REG, REGCT}
  sign                  dev (IC on 2019-22 matured labels)  | xs (expanding, fully OOS)  | prior (economic prior, macro)

All normalisation statistics use PAST values only (value at t compared with t-252..t-1, or expanding).
REG = expanding univariate predictive regression of the h-day next_open log return on Z252(x), using only labels
that matured by the signal close (Goyal-Welch / Campbell-Thompson); REGCT additionally zeroes the forecast if the
slope sign disagrees with the reference sign (Campbell & Thompson 2008 restriction).
Every map is smoothed over h days (overlapping sub-portfolios) and traded with run_book (next_open, 1bp).

Stage 2 - combination over factors (all aggregations as separate predictors, ~880 columns):
  EW_Z (equal-weight dev-signed z), VOTE (diffusion index of signs, Neely et al. 2014), RSZ_MEAN / RSZ_MEDIAN /
  RSZ_CT (Rapach-Strauss-Zhou 2010 forecast combination), PCA3 (Neely et al.), PLS1 (3PRF, Kelly-Pruitt 2015 /
  Huang et al. 2015), CENET (Rapach & Zhou combination-elastic-net).
Evaluation: LS Sharpe (dev / test), test IC, R2_OS vs expanding historical mean (Campbell-Thompson),
mean-variance allocation  w = clip(mu_hat/(gamma*sigma^2), 0, 1.5) vs the historical-mean-only allocation and B&H.
"""
from __future__ import annotations

import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from extended_timing_research import daily_intraday_features
from factor_sharpe import run_book
from index_timing_framework import load_market
from macro_factor_study import load_macro_panel
from multifrequency_models import target
from run_full_extended import config_from

warnings.filterwarnings("ignore")
OUT = Path("outputs/full/transform"); OUT.mkdir(parents=True, exist_ok=True)
SPLIT = "2023-01-03"
HS = (1, 2, 3, 5, 10, 20, 40, 60)
AGGS = ("L1", "MA5", "MA20", "EWM10", "MED20", "DEV60", "CHG20")
INTRA_DAILY = ['LOGVOL_STD', 'LOGVOL_SKEW', 'OPEN_VOL_SHARE', 'OPEN_VOL_SD', 'AM_PM_OPEN_RATIO', 'PM_MID_CLOSE_RATIO',
               'UP_RVOL', 'UP_VOLSD', 'VP_MED_SKEW', 'VP_REVERSAL', 'MOM_RV', 'MOM_SIGN', 'VOL_TARGET', 'RV_RATIO', 'REV_5',
               'AMIHUD_ILLIQ', 'ST_REV_20', 'MAXRET_20', 'VOL_20', 'DOLVOL_20', 'ILLIQ_20']
NOISE = ['MID_MED', 'MID_AVG', 'MID_END', 'OUT_AVG', 'SKEW_DIFF']
MIN_REG = 252
GAMMA = 5.0


# ----------------------------------------------------------------------------------------------- transforms
def aggregate(x: pd.Series, a: str) -> pd.Series:
    if a == "L1": return x
    if a == "MA5": return x.rolling(5, min_periods=3).mean()
    if a == "MA20": return x.rolling(20, min_periods=10).mean()
    if a == "EWM10": return x.ewm(halflife=10, min_periods=5, ignore_na=True).mean().where(x.ffill().notna())
    if a == "MED20": return x.rolling(20, min_periods=10).median()
    if a == "DEV60": return x - x.rolling(60, min_periods=30).mean().shift(1)
    if a == "CHG20":
        m = x.rolling(5, min_periods=3).mean(); return m - m.shift(20)
    raise KeyError(a)


def prior_stats(a: np.ndarray, n=252, minp=126):
    """Stats of a_t relative to its OWN previous n values (a_{t-n}..a_{t-1})."""
    pad = np.r_[np.full(n, np.nan), a]
    W = sliding_window_view(pad, n + 1)[: len(a)]
    prior, cur = W[:, :-1], W[:, -1]
    fin = np.isfinite(prior); cnt = fin.sum(1)
    less = (prior < cur[:, None]).sum(1); eq = (prior == cur[:, None]).sum(1)
    with np.errstate(all="ignore"):
        pct = (less + 0.5 * eq) / cnt
        mu = np.nanmean(prior, 1); sd = np.nanstd(prior, 1, ddof=1)
        med = np.nanmedian(prior, 1); mad = np.nanmedian(np.abs(prior - med[:, None]), 1) * 1.4826
    bad = (cnt < minp) | ~np.isfinite(cur)
    for v in (pct, mu, sd, med, mad):
        v[bad] = np.nan
    return pct, mu, sd, med, mad


def expanding_pct(a: np.ndarray, minp=126):
    out = np.full(len(a), np.nan); hist = []
    for t, v in enumerate(a):
        if np.isfinite(v) and len(hist) >= minp:
            h = np.asarray(hist); out[t] = ((h < v).sum() + 0.5 * (h == v).sum()) / len(h)
        if np.isfinite(v): hist.append(v)
    return out


def expanding_z(a: pd.Series, minp=126):
    mu = a.expanding(minp).mean().shift(1); sd = a.expanding(minp).std().shift(1)
    return ((a - mu) / sd).to_numpy()


def matured_reg(x: np.ndarray, y: np.ndarray, h: int, minn=MIN_REG):
    """Expanding OLS y_s ~ x_s with s+h <= t (labels known at close t). Returns beta_t, xbar_t, ybar_t (all PIT)."""
    n = len(x); ok = np.isfinite(x) & np.isfinite(y)
    xs, ys = np.where(ok, x, 0.0), np.where(ok, y, 0.0)
    c = lambda v: np.r_[0.0, np.cumsum(v)]
    N, SX, SY, SXX, SXY = c(ok.astype(float)), c(xs), c(ys), c(xs * xs), c(xs * ys)
    k = np.clip(np.arange(n) - h + 1, 0, n)     # number of s indices with s <= t-h
    Nk, sx, sy, sxx, sxy = N[k], SX[k], SY[k], SXX[k], SXY[k]
    with np.errstate(all="ignore"):
        xb, yb = sx / Nk, sy / Nk
        beta = (sxy - Nk * xb * yb) / (sxx - Nk * xb * xb)
    beta[Nk < minn] = np.nan
    return beta, xb, yb


def ybar_matured(y: np.ndarray, h: int):
    ok = np.isfinite(y); c = np.r_[0.0, np.cumsum(np.where(ok, y, 0))]; m = np.r_[0.0, np.cumsum(ok)]
    k = np.clip(np.arange(len(y)) - h + 1, 0, len(y))
    with np.errstate(all="ignore"):
        return np.where(m[k] >= 60, c[k] / m[k], np.nan)


def matured_rank_ic(x: np.ndarray, y: np.ndarray, h: int, every=21, minn=MIN_REG):
    """Expanding Spearman IC of (x_s, y_s), s <= t-h, refit every `every` sessions, held in between (PIT)."""
    n = len(x); out = np.full(n, np.nan); cur = np.nan
    for t in range(n):
        k = t - h + 1
        if t % every == 0 and k > 0:
            m = np.isfinite(x[:k]) & np.isfinite(y[:k])
            if m.sum() >= minn:
                xs, ys = x[:k][m], y[:k][m]
                cur = float(pd.Series(xs).rank().corr(pd.Series(ys).rank()))
        out[t] = cur
    return out


def smooth(s: np.ndarray, h: int) -> np.ndarray:
    return pd.Series(s).rolling(h, min_periods=max(1, h // 2)).mean().to_numpy()


# ----------------------------------------------------------------------------------------------- evaluation
def sr(r: pd.Series):
    r = r.dropna(); return float(r.mean() / r.std() * np.sqrt(252)) if len(r) > 20 and r.std() > 0 else np.nan


def spearman(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 50 or np.std(a[m]) == 0: return np.nan
    return float(pd.Series(a[m]).rank().corr(pd.Series(b[m]).rank()))


def evaluate(panel, w: np.ndarray, y, dates, split, bh_test, extra=None):
    r = run_book(panel, pd.Series(np.nan_to_num(w), index=panel.dates))
    rt = r.loc["2023":]
    b = bh_test.reindex(rt.index)
    beta = np.cov(rt, b)[0, 1] / b.var(); resid = rt - beta * b
    out = dict(sr_dev=sr(r.loc[:"2022"]), sr_test=sr(rt), ann_test=float(rt.mean() * 252), vol_test=float(rt.std() * np.sqrt(252)),
               ir_test=sr(resid), corr_bh=float(rt.corr(b)), avg_w_test=float(np.nanmean(w[split:])),
               ic_dev=spearman(w[:split], y[:split]), ic_test=spearman(w[split:], y[split:]),
               turnover=float(np.nanmean(np.abs(np.diff(np.nan_to_num(w))))) * 252)
    if extra: out.update(extra)
    return out, r


def r2_os(y, ybar, d, split, h):
    """Campbell-Thompson R2_OS on test signals whose label is observed; forecast = ybar + d."""
    t = np.arange(len(y)); m = (t >= split) & np.isfinite(y) & np.isfinite(ybar) & np.isfinite(d)
    if m.sum() < 100: return np.nan
    return float(1 - np.sum((y[m] - ybar[m] - d[m]) ** 2) / np.sum((y[m] - ybar[m]) ** 2))


def alloc_book(panel, ybar, d, h, rv):
    """Mean-variance allocation: w = clip((ybar+d)/h / (gamma * daily var), 0, 1.5)."""
    mu = (ybar + np.nan_to_num(d)) / h
    return np.clip(mu / (GAMMA * rv), 0, 1.5)


# ----------------------------------------------------------------------------------------------- data
_G = {}


def load_all(sym):
    cfg = config_from(Path("data"))
    panel = load_market(f"data/{sym}_5min.csv", cfg)
    daily = daily_intraday_features(f"data/{sym}_5min.csv", panel, cfg).reindex(panel.dates)
    F = {f"I:{c}": daily[c] for c in INTRA_DAILY}
    for c in NOISE:
        F[f"N:{c}"] = panel.factors[c].reindex(panel.dates)
    lib = Path(f"outputs/full/cache/{sym}_intraday_lib.pkl")       # realized-measure library
    if lib.exists():
        L = pd.read_pickle(lib).reindex(panel.dates)
        for c in L.columns:
            F[f"R:{c}"] = L[c]
    X, meta, _ = load_macro_panel(Path("../etf_timing"), sym, panel.dates)
    prior = {}
    for c in X.columns:
        F[f"M:{c}"] = X[c]; prior[f"M:{c}"] = meta[c]["prior_sign"]
    F = pd.DataFrame(F, index=panel.dates).replace([np.inf, -np.inf], np.nan)
    split = int(np.searchsorted(panel.dates, SPLIT))
    Y = {h: target(panel, h, "next_open")[0] for h in HS}
    bh = run_book(panel, pd.Series(1.0, index=panel.dates))
    ret = pd.Series(panel.close, index=panel.dates).pct_change()
    rv = ret.rolling(63, min_periods=40).var().to_numpy()       # known at close t
    vol20 = ret.rolling(20, min_periods=15).std()
    vscale = (vol20 / vol20.expanding(126).median().shift(1)).to_numpy()
    return dict(panel=panel, F=F, prior=prior, split=split, Y=Y, bh=bh, bh_test=bh.loc["2023":], rv=rv, vscale=vscale)


def _init():
    for s in ("SPY", "QQQ"):
        _G[s] = load_all(s)


def factor_job(name):
    rows, store = [], {}
    for sym in ("SPY", "QQQ"):
        G = _G[sym]; panel, split = G["panel"], G["split"]
        x0 = G["F"][name]
        if x0.notna().mean() < 0.5 or x0.nunique() < 5:
            continue
        for agg in AGGS:
            a = aggregate(x0, agg)
            av = a.to_numpy(float)
            pct, mu, sd, med, mad = prior_stats(av)
            with np.errstate(all="ignore"):
                z = np.clip((av - mu) / sd, -3, 3) / 3
                rz = np.clip((av - med) / mad, -3, 3) / 3
                ze = np.clip(expanding_z(a), -3, 3) / 3
            pe = expanding_pct(av)
            maps = {"PCT252": 2 * pct - 1, "PCTEXP": 2 * pe - 1, "Z252": z, "ZEXP": ze, "RZ252": rz,
                    "SIGN": np.sign(av - med), "TAIL": np.where(pct > 0.8, 1.0, np.where(pct < 0.2, -1.0, 0.0)) * np.isfinite(pct),
                    "VOLZ": np.clip(z / G["vscale"], -1.5, 1.5)}
            maps["TAIL"][~np.isfinite(pct)] = np.nan
            store[(sym, name, agg, "Z252")] = z.astype(np.float32)
            store[(sym, name, agg, "SIGN")] = maps["SIGN"].astype(np.float32)
            for h in HS:
                y = G["Y"][h]
                lab = (np.arange(len(y)) + h < split) & np.isfinite(y) & np.isfinite(z)
                if lab.sum() < 200 or np.std(z[lab]) == 0:
                    continue
                dev_sign = float(np.sign(spearman(z[lab], y[lab])) or 1.0)
                pri = G["prior"].get(name, 0)
                beta, xb, yb = matured_reg(z, y, h)
                d = beta * (z - xb)
                ybar = ybar_matured(y, h)
                ref = dev_sign
                d_ct = np.where(np.sign(beta) == ref, d, 0.0); d_ct[~np.isfinite(d)] = np.nan
                # vol-standardised target: y_s / (sqrt(h) sigma_s), sigma_s known at s  -> crisis obs not dominant
                sig_s = np.sqrt(h * G["rv"])
                bv, xbv, _ = matured_reg(z, y / sig_s, h)
                dv = bv * (z - xbv) * sig_s                 # back to return units with today's sigma
                ric = matured_rank_ic(z, y, h)              # expanding rank IC (robust sign + magnitude)
                dr = ric * z
                store[(sym, name, agg, f"D{h}")] = d.astype(np.float32)
                store[(sym, name, agg, f"DV{h}")] = dv.astype(np.float32)
                store[(sym, name, agg, f"RIC{h}")] = ric.astype(np.float32)
                store[(sym, name, agg, f"S{h}")] = np.float32(dev_sign)
                dsd = pd.Series(d).expanding(60).std().shift(1).to_numpy()
                dvsd = pd.Series(dv / sig_s).expanding(60).std().shift(1).to_numpy()
                drsd = pd.Series(dr).expanding(60).std().shift(1).to_numpy()
                sig = {k: v * dev_sign for k, v in maps.items()}
                xs = np.sign(beta)                                   # expanding (fully OOS) sign
                sig["Z252_xs"] = z * xs
                sig["PCT252_xs"] = maps["PCT252"] * xs
                if pri != 0:
                    sig["Z252_prior"] = z * pri
                sig["REG"] = np.clip(d / (2 * dsd), -1, 1)
                sig["REGCT"] = np.clip(d_ct / (2 * dsd), -1, 1)
                sig["REGV"] = np.clip(dv / sig_s / (2 * dvsd), -1, 1)
                sig["REGR"] = np.clip(dr / (2 * drsd), -1, 1)
                sig["Z252_xr"] = z * np.sign(ric)
                base = dict(symbol=sym, factor=name, group=name[0], agg=agg, h=h, dev_sign=dev_sign, prior_sign=pri,
                            prior_agrees=(pri == dev_sign) if pri else np.nan)
                r2 = {"REG": r2_os(y, ybar, d, split, h), "REGCT": r2_os(y, ybar, d_ct, split, h),
                      "REGV": r2_os(y, ybar, dv, split, h)}
                for nm, s in sig.items():
                    w = np.clip(smooth(s, h), -1.5, 1.5)
                    met, _ = evaluate(panel, w, y, panel.dates, split, G["bh_test"])
                    rows.append({**base, "map": nm, **met, "r2os_test": r2.get(nm, np.nan)})
    return rows, store


def main(workers=8):
    _init()
    names = list(_G["SPY"]["F"].columns)
    print(f"{len(names)} raw factors; aggs {len(AGGS)}; horizons {HS}", flush=True)
    rows, store = [], {}
    with Pool(workers, initializer=_init) as pool:
        for i, (r, s) in enumerate(pool.imap_unordered(factor_job, names, chunksize=2)):
            rows += r; store.update(s)
            if i % 20 == 0: print(f"  {i}/{len(names)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(OUT / "single_factor_transforms.parquet")
    pd.to_pickle(store, OUT / "signal_store.pkl")
    print("rows", len(df), flush=True)
    return df


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8)
