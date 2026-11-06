"""Loaders for the raw layout. All return tz-naive, sorted, de-duplicated frames."""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd

from quantkit import paths
from quantkit.data.fetch import yahoo_filename


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df.index = pd.to_datetime(df.index).tz_localize(None) if getattr(df.index, "tz", None) is not None else pd.to_datetime(df.index)
    return df[~df.index.duplicated(keep="last")].sort_index()


def load_daily(ticker: str) -> pd.DataFrame:
    """Yahoo daily frame (open high low close close_raw volume). close = dividend-adjusted."""
    df = _clean(pd.read_parquet(paths.raw("yahoo", yahoo_filename(ticker))))
    for c in ("open", "high", "low"):                              # same repair as etft.data.store.raw_yahoo
        bad = (df[c] <= 0) | df[c].isna(); df.loc[bad, c] = df.loc[bad, "close"]
    return df


def load_close(ticker: str, raw: bool = False) -> pd.Series:
    d = load_daily(ticker); return d["close_raw" if raw else "close"].dropna()


def load_cboe(sym: str) -> pd.Series:
    return _clean(pd.read_parquet(paths.raw("cboe", f"{sym}.parquet")))["close"].rename(sym)


def load_squeeze() -> pd.DataFrame:
    return _clean(pd.read_parquet(paths.raw("squeeze_dix_gex.parquet")))


def load_french(name: str = "F-F_Momentum_Factor_daily") -> pd.DataFrame:
    """Ken French csv -> decimal returns (percent / 100). Daily files have YYYYMMDD dates, monthly YYYYMM (month-end stamped).
    Only the first table in the file is returned (annual tables below it are dropped)."""
    lines = Path(paths.raw("french", f"{name}.csv")).read_text(errors="ignore").splitlines()
    hdr = next(i for i, l in enumerate(lines) if l.split(",")[0].strip() == "" and len(l.split(",")) > 1)
    rows = []
    for l in lines[hdr + 1:]:
        p = [x.strip() for x in l.split(",")]
        if not p or not p[0].isdigit():
            break
        rows.append(p)
    cols = [c.strip() for c in lines[hdr].split(",")][1:]
    df = pd.DataFrame(rows, columns=["date"] + cols)
    d = df["date"]; fmt = "%Y%m%d" if len(d.iloc[0]) == 8 else "%Y%m"
    idx = pd.to_datetime(d, format=fmt)
    if fmt == "%Y%m":
        idx = idx + pd.offsets.MonthEnd(0)
    out = df[cols].apply(pd.to_numeric, errors="coerce").replace([-99.99, -999], np.nan) / 100.0
    out.index = idx; out.index.name = "date"; return out


def load_econ_calendar() -> pd.DataFrame:
    """raw/econ_cal/ec.csv: datetime (ET), currency, event, impact, actual, forecast, previous, surprise."""
    ec = pd.read_csv(paths.raw("econ_cal", "ec.csv"), parse_dates=["datetime"]); return ec.sort_values("datetime").reset_index(drop=True)


def load_min1(sym: str) -> pd.DataFrame:
    """Raw 1-minute csv (time_key = bar END time). For day x minute matrices use quantkit.data.bars.load_bars."""
    df = pd.read_csv(paths.raw("min1", f"{sym}.csv"), parse_dates=["time_key"]); return df.set_index("time_key")
