"""Daily-frequency building blocks for the QQQ core ("B&H leg") and vol-targeting.

The core leg is RULE-BASED, not fitted:  w_t = w_rv(t) * w_trend(t),  both shifted one day (known before the open of t)
    w_rv     = lo + (hi - lo) * (1 - pct_t)          pct_t = causal percentile of 21d realised vol within the last 504 days
               -> 1.5x in the calmest regime, 0.5x in the most volatile one (RV21[0.5,1.5])
    w_trend  = 1 - (1 - e) * 1{close < SMA200}        e = 0.5 -> halve the position in a downtrend
Cost: 2 bp per unit of weight traded (one way).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CORE_COST = 2e-4


def causal_percentile(x: pd.Series, win: int = 504) -> pd.Series:
    """Percentile rank of x_t within the trailing window INCLUDING t (strictly-less count / valid count). NaN until win//2 valid points."""
    minp = win // 2; v = x.to_numpy(float); pad = np.concatenate([np.full(win - 1, np.nan), v])
    W = np.lib.stride_tricks.sliding_window_view(pad, win)
    with np.errstate(invalid="ignore"):
        less = (W < v[:, None]).sum(1)
    fin = np.isfinite(W).sum(1); ok = (fin >= minp) & np.isfinite(v); out = np.full(len(v), np.nan); out[ok] = less[ok] / fin[ok]
    return pd.Series(out, index=x.index)


def rv_weight(close: pd.Series, lo: float = 0.5, hi: float = 1.5, rv_win: int = 21, pct_win: int = 504) -> pd.Series:
    """Volatility-managed weight, already shifted one day. Neutral (1.0) during warm-up."""
    p = causal_percentile(close.pct_change().rolling(rv_win).std(), pct_win)
    return (lo + (hi - lo) * (1.0 - p)).fillna(1.0).shift(1).fillna(1.0)


def trend_down(close: pd.Series, kind: str = "SMA", n: int = 200) -> pd.Series:
    """True when the trend is down. kind='SMA': close < SMA_n; 'MOM': 21d-lagged n-day momentum < 0."""
    if kind == "SMA":
        sma = close.rolling(n).mean(); return (close < sma).where(sma.notna(), False)
    m = close.shift(21) / close.shift(n) - 1; return (m < 0).where(m.notna(), False)


def trend_weight(close: pd.Series, kind: str = "SMA", n: int = 200, e: float = 0.5) -> pd.Series:
    """1 in an uptrend, e in a downtrend; shifted one day."""
    return (1.0 - trend_down(close, kind, n).astype(float) * (1.0 - e)).shift(1).fillna(1.0)


def core_weight(close: pd.Series, lo: float = 0.5, hi: float = 1.5, trend_n: int = 200, e: float = 0.5) -> pd.Series:
    """The project's core leg weight (equals noise_core_dd.core_weights(px)[0]['base'])."""
    return rv_weight(close, lo, hi) * trend_weight(close, "SMA", trend_n, e)


def no_trade_band(w: pd.Series | np.ndarray, band: float = 0.10) -> np.ndarray:
    """Hold the current weight until the target differs from it by at least `band`; then jump to the target."""
    v = np.asarray(w, float); out = np.empty_like(v); cur = v[0]
    for i, x in enumerate(v):
        if abs(x - cur) >= band:
            cur = x
        out[i] = cur
    return out


def leg_return(w: pd.Series, r: pd.Series, cost: float = CORE_COST) -> pd.Series:
    """Net return of a weight series applied to asset returns: w*r - cost*|dw|. w must already be PIT (shifted)."""
    return w * r - cost * w.diff().abs().fillna(0.0)


def vol_target_leverage(sigma: pd.Series, target: float = 0.02, max_lev: float = 4.0) -> pd.Series:
    """e_t = min(max_lev, target / sigma_t) with sigma_t the trailing daily vol known before the open (the Sleeve uses sigma14)."""
    return np.minimum(max_lev, target / sigma)
