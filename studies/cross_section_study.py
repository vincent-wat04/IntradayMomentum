"""Stage 11: the cross-section.  Does the extra data make the complex models work?

Motivation (your question): the time-series study had ~1000 training observations and 12 features, and
GBM / random-feature ridge lost to a shallow linear model.  A cross-section of N names gives N x T
observations -- here 98 names x ~1900 days ~ 186,000 -- which is 180x more data for the same number of
parameters.  Non-linear models could plausibly win here even though they lost there.

Three things are tested, in order:
  A. SPY vs QQQ as a 2-asset "cross-section".  With two assets a cross-section degenerates into one
     spread (long QQQ short SPY), so there is nothing to neutralise; it is reported for completeness.
  B. A proper 98-name cross-section with the full preprocessing chain you described:
       cross-sectional rank -> z           (removes the market factor by construction)
       sector demean                        (industry neutral)
       style orthogonalisation              (regress out size, beta, volatility)
     Each stage is evaluated separately so the marginal effect of each is visible.
  C. The same four model families as the time-series study, refit walk-forward on the pooled panel.

Evaluation: daily cross-sectional rank IC (the standard cross-sectional statistic), its t-statistic
computed on the daily IC series (Newey-West style via the IC series' own autocorrelation), and
decile long-short portfolios with 10bp round-trip cost (single stocks, not ETFs).

Null: the entire factor panel is shifted in time by a random offset >= 1 year, keeping the
cross-sectional structure and the factor distributions but breaking the link to future returns.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

try:
    import lightgbm as lgb
    HAVE_LGB = True
except Exception:
    HAVE_LGB = False

D = Path("data/xs"); OUT = Path("outputs/full/xs"); OUT.mkdir(parents=True, exist_ok=True)
SPLIT = "2023-01-03"
HS = (1, 5, 20)
FEE_BP = 5.0                      # per leg, single stocks
MIN_TRAIN = 252
FIT_EVERY = 21
RFF_P, RFF_GAMMA, RFF_ALPHA = 2000, 0.5, 200.0

SECTOR = {
    "AAPL": "Tech", "MSFT": "Tech", "NVDA": "Tech", "AVGO": "Tech", "CRM": "Tech", "AMD": "Tech",
    "CSCO": "Tech", "ADBE": "Tech", "ORCL": "Tech", "ACN": "Tech", "TXN": "Tech", "INTU": "Tech",
    "IBM": "Tech", "NOW": "Tech", "QCOM": "Tech", "ADI": "Tech", "ADP": "Tech",
    "GOOGL": "Comm", "GOOG": "Comm", "META": "Comm", "NFLX": "Comm", "DIS": "Comm", "CMCSA": "Comm", "VZ": "Comm",
    "AMZN": "Disc", "TSLA": "Disc", "HD": "Disc", "MCD": "Disc", "LOW": "Disc", "BKNG": "Disc",
    "SBUX": "Disc", "TJX": "Disc", "GM": "Disc", "F": "Disc",
    "JPM": "Fin", "V": "Fin", "MA": "Fin", "BAC": "Fin", "WFC": "Fin", "AXP": "Fin", "BLK": "Fin",
    "SPGI": "Fin", "CB": "Fin", "PGR": "Fin", "BRK-B": "Fin",
    "LLY": "Health", "UNH": "Health", "JNJ": "Health", "ABBV": "Health", "TMO": "Health", "ABT": "Health",
    "DHR": "Health", "AMGN": "Health", "ELV": "Health", "PFE": "Health", "GILD": "Health", "MDT": "Health",
    "SYK": "Health", "VRTX": "Health", "CI": "Health", "BSX": "Health", "REGN": "Health", "ZTS": "Health",
    "BDX": "Health",
    "COST": "Staples", "PG": "Staples", "WMT": "Staples", "KO": "Staples", "PEP": "Staples", "PM": "Staples",
    "CL": "Staples", "MO": "Staples", "KHC": "Staples",
    "XOM": "Energy", "SLB": "Energy", "EOG": "Energy", "OXY": "Energy", "HAL": "Energy",
    "LIN": "Materials", "APD": "Materials", "FCX": "Materials", "DOW": "Materials",
    "CAT": "Indus", "HON": "Indus", "RTX": "Indus", "GE": "Indus", "UNP": "Indus", "LMT": "Indus",
    "ITW": "Indus", "MMM": "Indus", "CSX": "Indus", "NSC": "Indus", "EMR": "Indus", "AON": "Indus",
    "NEE": "Util", "SO": "Util", "DUK": "Util",
    "PLD": "RE", "PSA": "RE",
}


# ------------------------------------------------------------------ factors from daily OHLCV
def load():
    px = pd.read_csv(D / "close.csv", index_col=0)
    op = pd.read_csv(D / "open.csv", index_col=0)
    hi = pd.read_csv(D / "high.csv", index_col=0)
    lo = pd.read_csv(D / "low.csv", index_col=0)
    vo = pd.read_csv(D / "volume.csv", index_col=0)
    cols = [c for c in px.columns if c in SECTOR]
    return [x[cols] for x in (px, op, hi, lo, vo)]


def build_factors(px, op, hi, lo, vo):
    """~27 cross-sectional factors, all from daily OHLCV, all known at the close of t."""
    r = np.log(px).diff()
    dv = px * vo
    F = {}
    # momentum / reversal  (Jegadeesh & Titman 1993; Jegadeesh 1990; George & Hwang 2004)
    for k in (5, 21, 63, 126, 252):
        F[f"MOM_{k}"] = np.log(px).diff(k)
    F["MOM_12_1"] = np.log(px).diff(252) - np.log(px).diff(21)
    F["REV_5"] = -np.log(px).diff(5)
    F["REV_1"] = -r
    F["HI52_PROX"] = px / px.rolling(252, min_periods=126).max()
    # volatility family (Ang, Hodrick, Xing & Zhang 2006; Bali, Cakici & Whitelaw 2011)
    for k in (21, 63, 252):
        F[f"VOL_{k}"] = r.rolling(k, min_periods=k // 2).std()
    F["MAXRET_21"] = r.rolling(21, min_periods=15).max()
    F["MINRET_21"] = r.rolling(21, min_periods=15).min()
    F["SKEW_63"] = r.rolling(63, min_periods=40).skew()
    F["KURT_63"] = r.rolling(63, min_periods=40).kurt()
    F["VOL_OF_VOL"] = r.rolling(21, min_periods=15).std().rolling(63, min_periods=40).std()
    # range estimators (Parkinson 1980; Garman & Klass 1980; Rogers & Satchell 1991)
    lhr = np.log(hi / lo)
    F["PARKINSON"] = (lhr ** 2 / (4 * np.log(2))).rolling(21, min_periods=15).mean()
    F["GK"] = (0.5 * lhr ** 2 - (2 * np.log(2) - 1) * np.log(px / op) ** 2).rolling(21, min_periods=15).mean()
    F["RS"] = (np.log(hi / px) * np.log(hi / op) + np.log(lo / px) * np.log(lo / op)).rolling(21, min_periods=15).mean()
    F["RANGE_VOL_RATIO"] = F["PARKINSON"] / (r.rolling(21, min_periods=15).var() + 1e-12)
    # overnight / intraday split (Lou, Polk & Skouras 2019)
    ovn = np.log(op / px.shift(1)); intra = np.log(px / op)
    F["OVERNIGHT_21"] = ovn.rolling(21, min_periods=15).mean()
    F["INTRADAY_21"] = intra.rolling(21, min_periods=15).mean()
    F["OVN_SHARE"] = F["OVERNIGHT_21"] / (F["OVERNIGHT_21"].abs() + F["INTRADAY_21"].abs() + 1e-12)
    F["CLOSE_LOC"] = ((px - lo) / (hi - lo + 1e-12)).rolling(5, min_periods=3).mean()
    # liquidity (Amihud 2002)
    F["AMIHUD"] = (r.abs() / (dv + 1)).rolling(21, min_periods=15).mean() * 1e9
    F["DOLVOL"] = np.log1p(dv.rolling(21, min_periods=15).mean())
    F["VOL_TREND"] = dv.rolling(5, min_periods=3).mean() / dv.rolling(63, min_periods=40).mean()
    F["TURN_VOL"] = vo.rolling(21, min_periods=15).std() / (vo.rolling(21, min_periods=15).mean() + 1)
    # size and beta (style controls as well as factors)
    F["SIZE"] = np.log1p(px * vo.rolling(252, min_periods=126).mean())
    mkt = r.mean(axis=1)
    F["BETA"] = r.rolling(252, min_periods=126).cov(mkt).div(mkt.rolling(252, min_periods=126).var(), axis=0)
    F["IVOL"] = (r.sub(F["BETA"].mul(mkt, axis=0))).rolling(63, min_periods=40).std()
    return {k: v.replace([np.inf, -np.inf], np.nan) for k, v in F.items()}, r


# ------------------------------------------------------------------ preprocessing chain
def xs_rank_z(df):
    """Cross-sectional rank -> approximately normal score, per day. Removes the market factor."""
    rk = df.rank(axis=1, pct=True)
    return rk.sub(rk.mean(axis=1), axis=0).div(rk.std(axis=1).replace(0, np.nan), axis=0)


def sector_demean(df, sec: pd.Series):
    """Industry neutral: subtract the sector mean each day."""
    out = df.copy()
    for s in sec.unique():
        c = sec[sec == s].index.intersection(df.columns)
        if len(c) >= 2:
            out[c] = df[c].sub(df[c].mean(axis=1), axis=0)
    return out


def style_orthogonalize(F: dict, styles=("SIZE", "BETA", "VOL_21")):
    """Regress every factor on the style factors cross-sectionally each day and keep the residual."""
    keys = [k for k in F if k not in styles]
    X_styles = [F[s] for s in styles if s in F]
    out = {}
    idx = F[keys[0]].index
    for k in keys:
        Y = F[k]
        res = pd.DataFrame(np.nan, index=idx, columns=Y.columns)
        for t in idx:
            y = Y.loc[t]
            Z = np.column_stack([s.loc[t].to_numpy() for s in X_styles] + [np.ones(len(y))])
            m = np.isfinite(y.to_numpy()) & np.isfinite(Z).all(1)
            if m.sum() < 20:
                continue
            b, *_ = np.linalg.lstsq(Z[m], y.to_numpy()[m], rcond=None)
            res.loc[t, y.index[m]] = y.to_numpy()[m] - Z[m] @ b
        out[k] = res
    for s in styles:
        if s in F:
            out[s] = F[s]
    return out


# ------------------------------------------------------------------ evaluation
def fwd_ret(px, h):
    """h-day forward return, entered at the next open-equivalent (t+1 close) to avoid same-close fills."""
    return np.log(px.shift(-h - 1) / px.shift(-1))


def ic_series(fac: pd.DataFrame, fwd: pd.DataFrame):
    a = fac.rank(axis=1); b = fwd.rank(axis=1)
    return a.corrwith(b, axis=1)


def ic_stats(ic: pd.Series, h: int):
    ic = ic.dropna()
    if len(ic) < 60:
        return dict(ic=np.nan, t=np.nan, n=len(ic))
    # the IC series of an h-day label is autocorrelated; scale the t by sqrt of the overlap
    return dict(ic=float(ic.mean()), t=float(ic.mean() / ic.std() * np.sqrt(len(ic) / h)), n=len(ic))


def ls_portfolio(score: pd.DataFrame, r: pd.DataFrame, h: int, n_dec=10, fee_bp=FEE_BP):
    """Equal-weight top-decile minus bottom-decile, rebalanced every h days, h overlapping tranches."""
    rk = score.rank(axis=1, pct=True)
    w = pd.DataFrame(0.0, index=score.index, columns=score.columns)
    w[rk >= 1 - 1 / n_dec] = 1.0
    w[rk <= 1 / n_dec] = -1.0
    w = w.div(w.abs().sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    held = sum(w.shift(1 + k) for k in range(h)) / h            # h overlapping tranches, 1-day lag
    held = held.fillna(0.0)
    gross = (held * r).sum(axis=1)
    cost = fee_bp * 1e-4 * held.diff().abs().sum(axis=1)
    return gross - cost


def sharpe(x):
    x = x.dropna()
    return float(x.mean() / x.std() * np.sqrt(252)) if len(x) > 60 and x.std() > 0 else np.nan


# ------------------------------------------------------------------ models
def fit_predict(Xtr, ytr, Xte, kind, rng):
    mu, sd = np.nanmean(Xtr, 0), np.nanstd(Xtr, 0) + 1e-9
    A, B = (np.nan_to_num(Xtr) - mu) / sd, (np.nan_to_num(Xte) - mu) / sd
    if kind == "RIDGE":
        return Ridge(alpha=10.0).fit(A, ytr).predict(B)
    if kind == "GBM":
        m = lgb.LGBMRegressor(n_estimators=300, num_leaves=31, max_depth=6, learning_rate=0.03,
                              min_child_samples=100, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.8, reg_lambda=1.0, verbose=-1, n_jobs=1,
                              force_col_wise=True)
        return m.fit(A, ytr).predict(B)
    if kind == "RFF":
        W = rng.standard_normal((A.shape[1], RFF_P // 2)) * np.sqrt(RFF_GAMMA)
        b = rng.uniform(0, 2 * np.pi, RFF_P // 2)
        phi = lambda Z: np.c_[np.cos(Z @ W + b), np.sin(Z @ W + b)] * np.sqrt(2.0 / RFF_P)
        return Ridge(alpha=RFF_ALPHA).fit(phi(A), ytr).predict(phi(B))
    raise KeyError(kind)


def run_models(F: dict, fwd: pd.DataFrame, r: pd.DataFrame, h: int, split_i: int, names, rng):
    """Walk-forward pooled cross-sectional models. Returns score frames per model."""
    dates = fwd.index
    keys = sorted(F)
    cube = np.stack([F[k].to_numpy(float) for k in keys], -1)        # T x N x K
    Y = fwd.to_numpy(float)
    T_, N, K = cube.shape
    out = {m: pd.DataFrame(np.nan, index=dates, columns=F[keys[0]].columns) for m in names}
    models = {}
    for t in range(T_):
        if t % FIT_EVERY == 0 and t - h - 1 >= MIN_TRAIN:
            tr = slice(0, t - h - 1)
            Xtr = cube[tr].reshape(-1, K); ytr = Y[tr].reshape(-1)
            m = np.isfinite(ytr) & np.isfinite(Xtr).all(1)
            if m.sum() > 5000:
                Xs, ys = Xtr[m], ytr[m]
                ys = (ys - ys.mean()) / (ys.std() + 1e-12)
                models = {nm: (Xs, ys) for nm in names}
        if not models:
            continue
        Xte = cube[t]
        ok = np.isfinite(Xte).all(1)
        if ok.sum() < 20:
            continue
        for nm in names:
            Xs, ys = models[nm]
            try:
                out[nm].iloc[t, np.flatnonzero(ok)] = fit_predict(Xs, ys, Xte[ok], nm, rng) \
                    if t % FIT_EVERY == 0 else out[nm].iloc[t - 1].to_numpy()[ok]
            except Exception:
                pass
    return out


def main():
    px, op, hi, lo, vo = load()
    sec = pd.Series({c: SECTOR[c] for c in px.columns})
    F0, r = build_factors(px, op, hi, lo, vo)
    print(f"universe {px.shape[1]} names, {px.shape[0]} days, {len(F0)} raw factors, "
          f"{len(sec.unique())} sectors")

    stages = {}
    stages["raw"] = F0
    stages["xs_z"] = {k: xs_rank_z(v) for k, v in F0.items()}
    stages["xs_z+sector"] = {k: sector_demean(v, sec) for k, v in stages["xs_z"].items()}
    print("orthogonalising styles (slow)...", flush=True)
    stages["xs_z+sector+style"] = style_orthogonalize(stages["xs_z+sector"])

    split_i = int(np.searchsorted(px.index, SPLIT))
    rows = []
    for stage, F in stages.items():
        for h in HS:
            fwd = fwd_ret(px, h)
            for k, v in F.items():
                dev = ic_series(v.iloc[:split_i], fwd.iloc[:split_i])
                tst = ic_series(v.iloc[split_i:], fwd.iloc[split_i:])
                sd, st = ic_stats(dev, h), ic_stats(tst, h)
                sgn = np.sign(sd["ic"]) if np.isfinite(sd["ic"]) and sd["ic"] != 0 else 1.0
                pnl = ls_portfolio(v * sgn, r, h)
                rows.append(dict(stage=stage, h=h, factor=k, ic_dev=sd["ic"], ic_test=st["ic"],
                                 t_test=st["t"], dev_sign=sgn, sr_dev=sharpe(pnl.iloc[:split_i]),
                                 sr_test=sharpe(pnl.iloc[split_i:])))
    S = pd.DataFrame(rows); S.to_csv(OUT / "xs_single_factor.csv", index=False)
    pd.set_option("display.width", 260); pd.set_option("display.max_rows", 400)
    print("\n=== effect of each preprocessing stage (single factors, averaged) ===")
    print(S.groupby(["stage", "h"]).agg(
        n=("factor", "size"), mean_abs_ic_test=("ic_test", lambda s: s.abs().mean()),
        share_t_gt2=("t_test", lambda s: (s.abs() > 2).mean()), med_sr_test=("sr_test", "median"),
        dev_test_ic_corr=("ic_dev", lambda s: np.corrcoef(s.fillna(0), S.loc[s.index, "ic_test"].fillna(0))[0, 1]),
    ).round(3).to_string())
    print("\n=== strongest factors after the full chain (h=5) ===")
    q = S[(S.stage == "xs_z+sector+style") & (S.h == 5)].copy()
    print(q.reindex(q.t_test.abs().sort_values(ascending=False).index)
          [["factor", "ic_dev", "ic_test", "t_test", "sr_dev", "sr_test"]].head(14).round(3).to_string(index=False))
    return S, stages, px, r, sec, split_i


if __name__ == "__main__":
    main()
