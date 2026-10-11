"""Noise area (Zarattini, Aziz & Barbon 2024, "Beat the Market: An Effective Intraday
Momentum Strategy for S&P500 ETF (SPY)", SSRN 4824172).

For day t and bar m:
    move_{d,m}   = | close_{d,m} / open_d - 1 |
    sigma_{t,m}  = mean_{d in t-L..t-1} move_{d,m}            (L = 14, PIT: past days only)
    UB_{t,m}     = max(open_t, prev_close_adj_t) * (1 + vm * sigma_{t,m})
    LB_{t,m}     = min(open_t, prev_close_adj_t) * (1 - vm * sigma_{t,m})
The "noise area" is the range of moves that is normal for this time of day; a
close outside it signals a supply/demand imbalance that tends to persist
(price under-reaction / sticky order imbalance, cf. Gao, Han, Li & Zhou 2018).
The max/min with the previous close lets an overnight gap count as part of the move.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from etft.intraday.bars import Bars


def sigma_profile(b: Bars, lookback: int = 14, min_obs: int | None = None) -> pd.DataFrame:
    move = (b.c.div(b.day_open, axis=0) - 1).abs()
    min_obs = min_obs or max(5, int(0.7 * lookback))
    return move.rolling(lookback, min_periods=min_obs).mean().shift(1)  # strictly past days


def noise_area(b: Bars, lookback: int = 14, vm: float = 1.0, signal_bars: Bars | None = None):
    """Bounds computed on `signal_bars` (e.g. the INDEX) and returned on its price scale.
    Returns (ub, lb, ref_close) where ref_close is the close series to compare with."""
    s = signal_bars or b
    sig = sigma_profile(s, lookback)
    hi = np.maximum(s.day_open, s.prev_close_adj.fillna(s.day_open))
    lo = np.minimum(s.day_open, s.prev_close_adj.fillna(s.day_open))
    ub = sig.mul(vm).add(1).mul(hi, axis=0)
    lb = (1 - sig.mul(vm)).mul(lo, axis=0)
    return ub, lb, s.c


def cum_rvol(b: Bars, lookback: int = 14) -> pd.DataFrame:
    """cumulative volume since the open at each bar / the average of the same quantity at the
    same time of day over the PAST `lookback` days (PIT; today's volume only up to bar m)."""
    cv = b.v.cumsum(axis=1)
    norm = cv.rolling(lookback, min_periods=max(5, int(0.7 * lookback))).mean().shift(1)
    return cv / norm.replace(0, np.nan)


def volclock_sigma(b: Bars, lookback: int = 14) -> pd.DataFrame:
    """Noise profile in VOLUME time (Clark 1973; Ane & Geman 2000).
    u_{d,m} = cumvol_{d,m} / ADV_d, ADV_d = mean daily volume of the `lookback` days before d (PIT).
    sigma_{t,m} = mean over past days j of |move_j| evaluated at u_j = u_{t,m} (linear interpolation
    of each past day's move-vs-u path; beyond a past day's final u the end-of-day move is used)."""
    cv = b.v.cumsum(axis=1).values
    dv = b.v.sum(axis=1)
    adv = dv.rolling(lookback, min_periods=max(5, int(0.7 * lookback))).mean().shift(1).values
    U = cv / adv[:, None]
    M = (b.c.div(b.day_open, axis=0) - 1).abs().values
    out = np.full(M.shape, np.nan)
    n = len(M)
    for t in range(lookback + 1, n):
        if not np.isfinite(adv[t]):
            continue
        ut = U[t]
        acc = np.zeros(M.shape[1]); cnt = 0
        for j in range(t - lookback, t):
            uj, mj = U[j], M[j]
            ok = np.isfinite(uj) & np.isfinite(mj)
            if ok.sum() < 30:
                continue
            uj, mj = np.maximum.accumulate(uj[ok]), mj[ok]  # cumvol is non-decreasing
            acc += np.interp(np.nan_to_num(ut, nan=0.0), uj, mj); cnt += 1
        if cnt >= max(5, int(0.7 * lookback)):
            out[t] = acc / cnt
    return pd.DataFrame(out, index=b.c.index, columns=b.c.columns).where(np.isfinite(U))


def noise_area_volume(b: Bars, lookback: int = 14, vm: float = 1.0, signal_bars: Bars | None = None,
                      gamma: float | None = 0.5, mode: str = "power", clip=(0.25, 4.0)):
    """Volume-adjusted noise area.
    mode="power":    sigma_adj = sigma_clock * clip(rvol_cum, .25, 4) ** gamma
    mode="volclock": sigma from volclock_sigma() (non-parametric volume time)."""
    s = signal_bars or b
    if mode == "power":
        sig = sigma_profile(s, lookback) * cum_rvol(s, lookback).clip(*clip) ** gamma
    elif mode == "volclock":
        sig = volclock_sigma(s, lookback)
    else:
        raise ValueError(mode)
    hi = np.maximum(s.day_open, s.prev_close_adj.fillna(s.day_open))
    lo = np.minimum(s.day_open, s.prev_close_adj.fillna(s.day_open))
    return sig.mul(vm).add(1).mul(hi, axis=0), (1 - sig.mul(vm)).mul(lo, axis=0), s.c


def daily_vol(b: Bars, lookback: int = 14) -> pd.Series:
    """std of daily close-to-close (dividend-adjusted) returns over the PAST `lookback` days."""
    r = b.day_close / b.prev_close_adj - 1
    return r.rolling(lookback, min_periods=max(5, lookback // 2)).std().shift(1)
