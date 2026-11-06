"""Readers for the mdrec recorder output (IntradayMomentumLive/src/mdrec): tick + 10-level order book from moomoo.

Layout: <ETFT_DATA>/mdrec/data/<folder date>/<CODE>_{ticker|book}_<HH>.<RUN>.csv.gz   (timestamps are US/Eastern wall time)
Segments that are still being written (truncated gzip) are read up to the last complete row.
"""
from __future__ import annotations

import csv
import glob
import gzip
import zlib

import numpy as np
import pandas as pd

from quantkit import paths

LEVELS = range(1, 11)


def read_segments(code: str, kind: str, root=None) -> pd.DataFrame:
    """kind = 'ticker' | 'book'. Concatenates every segment for the code across all dates (strings, not parsed)."""
    root = root or paths.mdrec_dir(); frames = []
    for f in sorted(glob.glob(str(root / "*" / f"{code}_{kind}_*.csv.gz"))):
        rows, header = [], None
        try:
            with gzip.open(f, "rt", newline="") as fh:
                rd = csv.reader(fh); header = next(rd, None)
                for r in rd:
                    if header and len(r) == len(header):
                        rows.append(r)
        except (EOFError, zlib.error, OSError):
            pass
        if header and rows:
            frames.append(pd.DataFrame(rows, columns=header))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def book_snapshots(code: str = "US.TQQQ", rth_only: bool = True) -> pd.DataFrame:
    """Numeric 10-level snapshots with uncrossed books only (ap1 > bp1). Columns: ts, day, bp1..bp10, bs1..bs10, ap1..ap10, as1..as10."""
    df = read_segments(code, "book")
    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df["t_bid"], format="ISO8601")
    cols = [f"{p}{i}" for p in ("bp", "bs", "ap", "as") for i in LEVELS]
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["bp1", "ap1"]); df = df[df.ap1 > df.bp1]
    sz = [c for c in cols if c[:2] in ("bs", "as")]; df[sz] = df[sz].fillna(0.0)
    if rth_only:
        t = df.ts.dt.time; df = df[(t >= pd.Timestamp("09:30").time()) & (t < pd.Timestamp("16:00").time())]
    df["day"] = df.ts.dt.strftime("%Y-%m-%d")
    return df.sort_values("ts").drop_duplicates(["ts", "bp1", "ap1", "bs1", "as1"]).reset_index(drop=True)


def ticks(code: str = "US.QQQ", rth_only: bool = True) -> pd.DataFrame:
    """Trades with the exchange-reported aggressor side: sgn = +1 BUY, -1 SELL, 0 NEUTRAL/other."""
    df = read_segments(code, "ticker")
    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df["time"], format="ISO8601"); df = df.drop_duplicates("sequence")
    for c in ("price", "volume", "turnover"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["price", "volume"])
    if rth_only:
        t = df.ts.dt.time; df = df[(t >= pd.Timestamp("09:30").time()) & (t < pd.Timestamp("16:00").time())]
    df["sgn"] = np.where(df.direction == "BUY", 1, np.where(df.direction == "SELL", -1, 0)); df["day"] = df.ts.dt.strftime("%Y-%m-%d")
    return df.reset_index(drop=True)
