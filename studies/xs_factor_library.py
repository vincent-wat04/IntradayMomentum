"""Cross-sectional factor library: 78 factors from daily OHLCV, including the microstructure family.

Why this file exists: the first cross-sectional pass used only ~32 generic factors and left out the
microstructure / liquidity family, which was the single best-performing family in the time-series study
(positive excess IC at all eight horizons).  The family was omitted because we have no intraday bars for
the 98 stocks -- but that was the wrong conclusion: the microstructure literature has a well-developed
set of estimators that recover spreads and price impact from DAILY open/high/low/close/volume.  Those
are implemented here, so the family is now represented.

Families and references
-----------------------
spread estimators (the key addition)
  CS_SPREAD         Corwin & Schultz (2012, JF) high-low spread; the standard daily proxy
  AR_SPREAD         Abdi & Ranaldo (2017, RFS) close-high-low estimator
  ROLL_SPREAD       Roll (1984, JF) serial-covariance spread
  BPW_ILLIQ         Bao, Pan & Wang (2011, JF) negative autocovariance of price changes
  ZERO_RET          Lesmond, Ogden & Trzcinka (1999, RFS) proportion of zero-return days
  HL_RATIO          Hui-Heubel style high-low range over volume
price impact / depth
  AMIHUD            Amihud (2002, JFM) |r| / dollar volume
  KYLE_LAMBDA       Hasbrouck (2009, JF) / Goyenko, Holden & Trzcinka (2009) proxy |r| / sqrt(dollar)
  AMIVEST           Amivest liquidity ratio: volume / |r|
  PS_GAMMA          Pastor & Stambaugh (2003, JPE) signed-volume reversal coefficient
  TURN, TURN_VOL, DOLVOL, VOL_TREND
microstructure noise
  VR_2, VR_5, VR_10 Lo & MacKinlay (1988) variance ratios (<1 mean reverting, >1 trending)
  GK_CC_RATIO       Garman-Klass range variance over close-close variance; noise indicator
  OVN_VAR_SHARE     overnight share of total variance
realized moments / semivariance from daily returns
  RS_UP, RS_DOWN, SJV, RSJ, SEMI_RATIO    Barndorff-Nielsen, Kinnebrock & Shephard (2010);
                                           Patton & Sheppard (2015, REStat)
  SKEW_*, KURT_*, IDIO_SKEW               Amaya, Christoffersen, Jacobs & Vasquez (2015);
                                           Boyer, Mitton & Vorkink (2010)
range volatility estimators
  PARKINSON (1980), GK (Garman & Klass 1980), RS_EST (Rogers & Satchell 1991), YANG_ZHANG (2000)
momentum / reversal
  MOM_*, MOM_12_1 (Jegadeesh & Titman 1993), REV_* (Jegadeesh 1990), HI52_PROX (George & Hwang 2004),
  RESID_MOM (Blitz, Huij & Martens 2011), MOM_CONSISTENCY, TREND_*, ACCEL
risk / beta
  BETA, DOWN_BETA (Ang, Chen & Xing 2006), COSKEW (Harvey & Siddique 2000), IVOL (Ang et al. 2006),
  BETA_STAB, MAXRET_21 (Bali, Cakici & Whitelaw 2011), DD_*, VOL_OF_VOL
seasonality
  SAME_MONTH (Heston & Sadka 2008)

Every value at date t uses only data up to and including t.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FAMILIES = {
    "spread": ["CS_SPREAD", "AR_SPREAD", "ROLL_SPREAD", "BPW_ILLIQ", "ZERO_RET", "HL_RATIO"],
    "impact": ["AMIHUD", "KYLE_LAMBDA", "AMIVEST", "PS_GAMMA", "TURN", "TURN_VOL", "DOLVOL", "VOL_TREND"],
    "noise": ["VR_2", "VR_5", "VR_10", "GK_CC_RATIO", "OVN_VAR_SHARE", "CLOSE_LOC", "OVN_SHARE"],
    "semivar": ["RS_UP", "RS_DOWN", "SJV", "RSJ", "SEMI_RATIO", "UP_DOWN_COUNT", "SIGN_PERSIST"],
    "moments": ["SKEW_21", "SKEW_63", "KURT_63", "IDIO_SKEW", "VOL_OF_VOL"],
    "range": ["PARKINSON", "GK", "RS_EST", "YANG_ZHANG", "RANGE_VOL_RATIO"],
    "vol": ["VOL_5", "VOL_21", "VOL_63", "VOL_252", "IVOL", "MAXRET_21", "MINRET_21", "DD_63", "DD_252"],
    "mom": ["MOM_5", "MOM_21", "MOM_63", "MOM_126", "MOM_252", "MOM_12_1", "REV_1", "REV_5", "REV_21",
            "HI52_PROX", "LO52_PROX", "RESID_MOM", "MOM_CONSISTENCY", "TREND_21_63", "TREND_50_200", "ACCEL"],
    "risk": ["BETA", "DOWN_BETA", "UP_BETA", "COSKEW", "BETA_STAB", "CORR_MKT"],
    "shape": ["OVERNIGHT_21", "INTRADAY_21", "GAP_VOL", "VOL_PRICE_CORR", "AMIHUD_TREND"],
    "style": ["SIZE", "PRICE"],
    "season": ["SAME_MONTH"],
}
NAMES = [n for v in FAMILIES.values() for n in v]
FAM_OF = {n: f for f, ns in FAMILIES.items() for n in ns}


def _roll_cov(a: pd.DataFrame, b: pd.DataFrame, k: int, minp=None):
    return a.rolling(k, min_periods=minp or k // 2).cov(b)


def build(px, op, hi, lo, vo, win=21):
    """Return dict name -> (T x N) DataFrame."""
    F = {}
    r = np.log(px).diff()
    lhi, llo, lpx, lop = np.log(hi), np.log(lo), np.log(px), np.log(op)
    dv = px * vo
    mkt = r.mean(axis=1)

    # ---------------------------------------------------------------- spread estimators
    # Corwin & Schultz (2012): beta from two consecutive single-day ranges, gamma from the 2-day range
    h2 = pd.concat([lhi, lhi.shift(1)]).groupby(level=0).max()
    l2 = pd.concat([llo, llo.shift(1)]).groupby(level=0).min()
    beta = ((lhi - llo) ** 2 + (lhi - llo).shift(1) ** 2)
    gamma = (h2 - l2) ** 2
    den = 3 - 2 * np.sqrt(2.0)
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / den - np.sqrt(gamma / den)
    cs = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    F["CS_SPREAD"] = cs.clip(lower=0).rolling(win, min_periods=win // 2).mean()
    # Abdi & Ranaldo (2017): 4 * (log c - log mid)(log c - log mid_next)
    mid = (lhi + llo) / 2
    ar = 4 * (lpx - mid) * (lpx - mid.shift(-1))
    F["AR_SPREAD"] = np.sqrt(ar.clip(lower=0)).shift(1).rolling(win, min_periods=win // 2).mean()
    # Roll (1984): 2 sqrt(-cov(r_t, r_{t-1}))
    cov1 = _roll_cov(r, r.shift(1), 63, 40)
    F["ROLL_SPREAD"] = 2 * np.sqrt((-cov1).clip(lower=0))
    # Bao, Pan & Wang (2011): -cov of price changes (illiquidity, not a spread)
    F["BPW_ILLIQ"] = -cov1 * 1e4
    F["ZERO_RET"] = (r.abs() < 1e-8).rolling(win, min_periods=win // 2).mean()
    F["HL_RATIO"] = ((hi - lo) / px / (dv + 1) * 1e9).rolling(win, min_periods=win // 2).mean()

    # ---------------------------------------------------------------- price impact / depth
    F["AMIHUD"] = (r.abs() / (dv + 1)).rolling(win, min_periods=win // 2).mean() * 1e9
    F["KYLE_LAMBDA"] = (r.abs() / np.sqrt(dv + 1)).rolling(win, min_periods=win // 2).mean() * 1e3
    F["AMIVEST"] = (dv / (r.abs() + 1e-6)).rolling(win, min_periods=win // 2).mean().pipe(np.log1p)
    # Pastor & Stambaugh (2003): r_{t+1} = a + b r_t + gamma sign(r_t) dv_t ; gamma is the liquidity measure
    sdv = np.sign(r) * dv / 1e9
    num = _roll_cov(r.shift(-1), sdv, 63, 40)
    F["PS_GAMMA"] = (num / sdv.rolling(63, min_periods=40).var()).shift(1)
    F["TURN"] = np.log1p(vo.rolling(win, min_periods=win // 2).mean())
    F["TURN_VOL"] = vo.rolling(win, min_periods=win // 2).std() / (vo.rolling(win, min_periods=win // 2).mean() + 1)
    F["DOLVOL"] = np.log1p(dv.rolling(win, min_periods=win // 2).mean())
    F["VOL_TREND"] = dv.rolling(5, min_periods=3).mean() / (dv.rolling(63, min_periods=40).mean() + 1)
    F["AMIHUD_TREND"] = (F["AMIHUD"] / (r.abs() / (dv + 1)).rolling(126, min_periods=80).mean() / 1e9).replace(
        [np.inf, -np.inf], np.nan)

    # ---------------------------------------------------------------- microstructure noise
    v1 = r.rolling(63, min_periods=40).var()
    for k in (2, 5, 10):
        vk = np.log(px).diff(k).rolling(63, min_periods=40).var()
        F[f"VR_{k}"] = vk / (k * v1)
    gk = 0.5 * (lhi - llo) ** 2 - (2 * np.log(2) - 1) * (lpx - lop) ** 2
    F["GK"] = gk.rolling(win, min_periods=win // 2).mean()
    F["GK_CC_RATIO"] = F["GK"] / (r.rolling(win, min_periods=win // 2).var() + 1e-12)
    ovn = lop - lpx.shift(1); intra = lpx - lop
    F["OVN_VAR_SHARE"] = (ovn.rolling(win, min_periods=win // 2).var() /
                          (ovn.rolling(win, min_periods=win // 2).var() +
                           intra.rolling(win, min_periods=win // 2).var() + 1e-12))
    F["CLOSE_LOC"] = ((px - lo) / (hi - lo + 1e-12)).rolling(5, min_periods=3).mean()
    F["OVERNIGHT_21"] = ovn.rolling(win, min_periods=win // 2).mean()
    F["INTRADAY_21"] = intra.rolling(win, min_periods=win // 2).mean()
    F["OVN_SHARE"] = F["OVERNIGHT_21"] / (F["OVERNIGHT_21"].abs() + F["INTRADAY_21"].abs() + 1e-12)
    F["GAP_VOL"] = ovn.abs().rolling(win, min_periods=win // 2).mean()

    # ---------------------------------------------------------------- semivariance / signed jumps
    rp, rn = r.clip(lower=0), r.clip(upper=0)
    F["RS_UP"] = (rp ** 2).rolling(win, min_periods=win // 2).sum()
    F["RS_DOWN"] = (rn ** 2).rolling(win, min_periods=win // 2).sum()
    F["SJV"] = F["RS_UP"] - F["RS_DOWN"]
    rv = (r ** 2).rolling(win, min_periods=win // 2).sum()
    F["RSJ"] = F["SJV"] / (rv + 1e-12)
    F["SEMI_RATIO"] = F["RS_DOWN"] / (rv + 1e-12)
    F["UP_DOWN_COUNT"] = np.sign(r).rolling(win, min_periods=win // 2).mean()
    F["SIGN_PERSIST"] = (np.sign(r) * np.sign(r).shift(1)).rolling(63, min_periods=40).mean()

    # ---------------------------------------------------------------- moments
    F["SKEW_21"] = r.rolling(21, min_periods=15).skew()
    F["SKEW_63"] = r.rolling(63, min_periods=40).skew()
    F["KURT_63"] = r.rolling(63, min_periods=40).kurt()
    F["VOL_OF_VOL"] = r.rolling(21, min_periods=15).std().rolling(63, min_periods=40).std()

    # ---------------------------------------------------------------- range volatility
    F["PARKINSON"] = ((lhi - llo) ** 2 / (4 * np.log(2))).rolling(win, min_periods=win // 2).mean()
    F["RS_EST"] = ((lhi - lpx) * (lhi - lop) + (llo - lpx) * (llo - lop)).rolling(win, min_periods=win // 2).mean()
    vo_ = ovn.rolling(win, min_periods=win // 2).var()
    vc_ = intra.rolling(win, min_periods=win // 2).var()
    kyz = 0.34 / (1.34 + (win + 1) / (win - 1))
    F["YANG_ZHANG"] = vo_ + kyz * vc_ + (1 - kyz) * F["RS_EST"]
    F["RANGE_VOL_RATIO"] = F["PARKINSON"] / (r.rolling(win, min_periods=win // 2).var() + 1e-12)

    # ---------------------------------------------------------------- volatility and drawdown
    for k in (5, 21, 63, 252):
        F[f"VOL_{k}"] = r.rolling(k, min_periods=max(3, k // 2)).std()
    F["MAXRET_21"] = r.rolling(21, min_periods=15).max()
    F["MINRET_21"] = r.rolling(21, min_periods=15).min()
    for k in (63, 252):
        F[f"DD_{k}"] = px / px.rolling(k, min_periods=k // 2).max() - 1

    # ---------------------------------------------------------------- momentum / reversal
    for k in (5, 21, 63, 126, 252):
        F[f"MOM_{k}"] = lpx.diff(k)
    F["MOM_12_1"] = lpx.diff(252) - lpx.diff(21)
    F["REV_1"] = -r
    F["REV_5"] = -lpx.diff(5)
    F["REV_21"] = -lpx.diff(21)
    F["HI52_PROX"] = px / px.rolling(252, min_periods=126).max()
    F["LO52_PROX"] = px / px.rolling(252, min_periods=126).min()
    F["MOM_CONSISTENCY"] = (r > 0).rolling(126, min_periods=80).mean()
    F["TREND_21_63"] = px.rolling(21, min_periods=15).mean() / px.rolling(63, min_periods=40).mean() - 1
    F["TREND_50_200"] = px.rolling(50, min_periods=30).mean() / px.rolling(200, min_periods=120).mean() - 1
    F["ACCEL"] = lpx.diff(21) - lpx.diff(63) / 3

    # ---------------------------------------------------------------- risk / beta
    mv = mkt.rolling(252, min_periods=126).var()
    F["BETA"] = _roll_cov(r, mkt, 252, 126).div(mv, axis=0)
    F["CORR_MKT"] = r.rolling(252, min_periods=126).corr(mkt)
    dn = mkt < 0
    rd, md = r.where(dn), mkt.where(dn)
    F["DOWN_BETA"] = _roll_cov(rd, md, 252, 60).div(md.rolling(252, min_periods=60).var(), axis=0)
    up = mkt > 0
    ru, mu_ = r.where(up), mkt.where(up)
    F["UP_BETA"] = _roll_cov(ru, mu_, 252, 60).div(mu_.rolling(252, min_periods=60).var(), axis=0)
    m2 = (mkt ** 2)
    F["COSKEW"] = _roll_cov(r, m2, 252, 126).div(m2.rolling(252, min_periods=126).std() * mv ** 0, axis=0)
    resid = r.sub(F["BETA"].mul(mkt, axis=0))
    F["IVOL"] = resid.rolling(63, min_periods=40).std()
    F["RESID_MOM"] = resid.rolling(252, min_periods=126).mean() / (resid.rolling(252, min_periods=126).std() + 1e-12)
    F["IDIO_SKEW"] = resid.rolling(63, min_periods=40).skew()
    F["BETA_STAB"] = F["BETA"].rolling(126, min_periods=80).std()
    F["VOL_PRICE_CORR"] = r.abs().rolling(63, min_periods=40).corr(np.log1p(vo))

    # ---------------------------------------------------------------- size / price / seasonality
    F["SIZE"] = np.log1p(px * vo.rolling(252, min_periods=126).mean())
    F["PRICE"] = np.log(px)
    mo = pd.to_datetime(px.index).month
    sm = pd.DataFrame(np.nan, index=px.index, columns=px.columns)
    for m in range(1, 13):
        sel = mo == m
        hist = r[sel].rolling(12, min_periods=3).mean().shift(1)
        sm.loc[sel] = hist.to_numpy()
    F["SAME_MONTH"] = sm

    F = {k: v.replace([np.inf, -np.inf], np.nan) for k, v in F.items() if k in NAMES}
    missing = [n for n in NAMES if n not in F]
    if missing:
        raise RuntimeError(f"declared but not built: {missing}")
    return F, r


def style_orthogonalize_fast(F: dict, styles=("SIZE", "BETA", "VOL_21")):
    """Same as the slow version but solves all factors at once per date (one lstsq per day)."""
    keys = [k for k in F if k not in styles]
    St = [F[s] for s in styles if s in F]
    idx, cols = F[keys[0]].index, F[keys[0]].columns
    Y = np.stack([F[k].to_numpy(float) for k in keys], -1)       # T x N x K
    Z = np.stack([s.to_numpy(float) for s in St], -1)            # T x N x S
    out = np.full_like(Y, np.nan)
    for t in range(len(idx)):
        z, y = Z[t], Y[t]
        zc = np.c_[z, np.ones(len(z))]
        m = np.isfinite(zc).all(1)
        if m.sum() < 20:
            continue
        A = zc[m]
        yv = y[m]
        good = np.isfinite(yv)
        yf = np.where(good, yv, 0.0)
        b, *_ = np.linalg.lstsq(A, yf, rcond=None)
        res = yf - A @ b
        res[~good] = np.nan
        out[t, np.flatnonzero(m)] = res
    D = {k: pd.DataFrame(out[:, :, j], index=idx, columns=cols) for j, k in enumerate(keys)}
    for s in styles:
        if s in F:
            D[s] = F[s]
    return D
