"""Minute bars -> day x minute matrices (the only input of the intraday strategies).

    B = load_bars("SPY")            # US ETF, regular session 09:31..16:00 (390 bars)
    B = load_bars("SH.510300")      # A-share, 09:30 auction bar + 09:31..11:30 + 13:01..15:00
    B.o, B.h, B.l, B.c, B.v        # DataFrames: index = day, columns = bar END times "HH:MM"
    B.day_open, B.day_close        # first bar open / last valid bar close of the day
    B.prev_close_adj               # previous close adjusted for dividends/splits ex-date t

Bar convention (moomoo): the label is the bar END time, i.e. bar "10:00" covers
09:59-10:00.  A decision taken on the CLOSE of bar k is filled at the OPEN of bar
k+1 (backtest.py), which is the earliest realistic fill.

Minute bars are UNADJUSTED.  Intraday that is correct; across days the previous
close must be adjusted on ex-dividend days, otherwise max(open, prev_close) is
biased upward by the dividend (Zarattini et al. adjust for this explicitly).
Adjustment factor from daily data: f_t = adj_close_t / raw_close_t,
prev_close_adj_t = raw_close_{t-1} * f_{t-1} / f_t   (known before the open of t:
ex-dates are announced in advance).
"""
from __future__ import annotations

import functools
from dataclasses import dataclass

import numpy as np
import pandas as pd

from etft import config as C

MIN_DIR = {"US": C.RAW / "min1", "CN": C.RAW / "min1_cn"}


def market_of(sym: str) -> str:
    return "CN" if sym.startswith(("SH.", "SZ.")) else "US"


def session_grid(market: str) -> list[str]:
    if market == "US":
        t = pd.date_range("2000-01-01 09:31", "2000-01-01 16:00", freq="1min")
        return [x.strftime("%H:%M") for x in t]
    am = pd.date_range("2000-01-01 09:30", "2000-01-01 11:30", freq="1min")
    pm = pd.date_range("2000-01-01 13:01", "2000-01-01 15:00", freq="1min")
    return [x.strftime("%H:%M") for x in am.append(pm)]


@dataclass
class Bars:
    sym: str
    market: str
    o: pd.DataFrame
    h: pd.DataFrame
    l: pd.DataFrame
    c: pd.DataFrame
    v: pd.DataFrame
    day_open: pd.Series
    day_close: pd.Series
    prev_close_adj: pd.Series
    last_idx: pd.Series        # position of the last valid bar each day (half days)

    @property
    def days(self) -> pd.DatetimeIndex:
        return self.c.index

    @property
    def times(self) -> list[str]:
        return list(self.c.columns)

    def vwap(self) -> pd.DataFrame:
        """cumulative session VWAP at each bar close (typical price x volume)."""
        tp = (self.h + self.l + self.c) / 3
        pv = (tp * self.v).cumsum(axis=1)
        vv = self.v.cumsum(axis=1)
        return pv / vv.replace(0, np.nan)


def _adj_factor(sym: str) -> pd.Series | None:
    """cumulative dividend/split adjustment f_t = adj/raw from daily data."""
    from etft.data.store import raw_yahoo
    ysym = sym
    if market_of(sym) == "CN":
        f = C.RAW / "cn_daily" / f"{sym}.parquet"
        if not f.exists():
            return None
        d = pd.read_parquet(f)
        return (d["close_adj"] / d["close_raw"]).rename("f")
    try:
        d = raw_yahoo(ysym)
    except FileNotFoundError:
        return None
    return (d["close"] / d["close_raw"]).rename("f")


@functools.lru_cache(maxsize=None)
def load_bars(sym: str, min_bars_frac: float = 0.95) -> Bars:
    mkt = market_of(sym)
    df = pd.read_csv(MIN_DIR[mkt] / f"{sym}.csv")
    df["ts"] = pd.to_datetime(df["time_key"])
    df["day"] = df["ts"].dt.normalize()
    df["t"] = df["ts"].dt.strftime("%H:%M")
    grid = session_grid(mkt)
    df = df[df["t"].isin(grid)].drop_duplicates(["day", "t"], keep="last")
    piv = {k: df.pivot(index="day", columns="t", values=k).reindex(columns=grid)
           for k in ("open", "high", "low", "close", "volume")}
    c = piv["close"]
    n = c.notna().sum(axis=1)
    # keep full days and genuine half days (US 13:00 close = 210 bars); drop broken days
    last_pos = c.notna().values[:, ::-1].argmax(axis=1)
    last_idx = pd.Series(c.shape[1] - 1 - last_pos, index=c.index)
    contiguous = n == (last_idx + 1)
    full = n >= min_bars_frac * len(grid)
    half = (mkt == "US") & (n.between(205, 215)) & contiguous
    keep = (full | half) & contiguous
    for k in piv:
        piv[k] = piv[k][keep]
    last_idx = last_idx[keep]
    c = piv["close"]
    # small gaps inside a valid day (missing minutes): carry the last price, zero volume
    for k in ("open", "high", "low"):
        piv[k] = piv[k].where(piv[k].notna(), c.ffill(axis=1))
    piv["close"] = c.ffill(axis=1)
    piv["volume"] = piv["volume"].fillna(0.0)
    # re-blank bars after a half-day close
    after = np.arange(len(grid))[None, :] > last_idx.values[:, None]
    for k in piv:
        piv[k] = piv[k].mask(after)
    day_open = piv["open"].iloc[:, 0]
    day_close = pd.Series(piv["close"].values[np.arange(len(last_idx)), last_idx.values], index=last_idx.index)
    f = _adj_factor(sym)
    if f is not None:
        f = f.reindex(day_close.index).ffill().bfill()
        pca = day_close.shift(1) * f.shift(1) / f
    else:
        pca = day_close.shift(1)
    return Bars(sym, mkt, piv["open"], piv["high"], piv["low"], piv["close"], piv["volume"],
                day_open, day_close, pca, last_idx)


def fix_actions_with_reference(b: Bars, ref: Bars, tol: float = 0.03) -> Bars:
    """Corporate actions (share splits / consolidations, large distributions) that the daily
    adjustment source missed are detected from the ETF's overnight move versus its index's
    overnight move on the same day.  If they differ by more than `tol`, the ETF's previous
    close is rescaled so that its overnight return equals the index's.  Uses only prices of
    day t-1 and the open of day t (available before the first decision)."""
    import copy
    d = b.days.intersection(ref.days)
    on_e = (b.day_open / b.prev_close_adj).reindex(d)
    on_i = (ref.day_open / ref.prev_close_adj).reindex(d)
    bad = (on_e / on_i - 1).abs() > tol
    out = copy.copy(b)
    pca = b.prev_close_adj.copy()
    pca.loc[bad[bad].index] = (b.day_open.reindex(d) / on_i)[bad]
    out.prev_close_adj = pca
    out.n_fixed = int(bad.sum())
    out.fixed_days = list(bad[bad].index.date)
    return out
