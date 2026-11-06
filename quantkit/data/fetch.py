"""Download raw data from public sources into <ETFT_DATA>/raw in the layout the research code expects.

    fetch_yahoo("QQQ")            -> raw/yahoo/QQQ.parquet           open high low close(=Adj Close) close_raw(=Close) volume
    fetch_yahoo("^NDX")           -> raw/yahoo/IDX_NDX.parquet       ('^' -> 'IDX_')
    fetch_cboe("VIX")             -> raw/cboe/VIX.parquet            close   (VIX, VIX9D, VXN, VVIX, ...)
    fetch_french("F-F_Momentum_Factor_daily") -> raw/french/*.zip + .csv
    fetch_squeezemetrics()        -> raw/squeeze_dix_gex.parquet     price dix gex
    fetch_moomoo_min1(["QQQ"], start, end)  -> raw/min1/QQQ.csv      needs a running moomoo OpenD (NOT exercised in the sandbox)
    update_all()                  refresh everything that has a public source

Every writer is atomic (write temp file, then rename) and refuses to overwrite a file with a shorter history
unless force=True, so a bad download cannot silently truncate the dataset.
The econ calendar (raw/econ_cal/ec.csv) and the 1-minute bars have no public source here; see the guide.
"""
from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path

import pandas as pd

from quantkit import paths

CBOE_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{sym}_History.csv"
FRENCH_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/{name}_CSV.zip"
SQUEEZE_URL = "https://squeezemetrics.com/monitor/static/DIX.csv"
UA = {"User-Agent": "Mozilla/5.0 (quantkit research; contact: repo owner)"}


def _http_get(url: str, timeout: float = 60) -> bytes:
    import requests
    r = requests.get(url, headers=UA, timeout=timeout, allow_redirects=True)
    r.raise_for_status()
    return r.content


def _atomic_write(path: Path, writer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    writer(tmp)
    os.replace(tmp, path)


def _guard_not_shorter(path: Path, new_len: int, force: bool) -> None:
    if path.exists() and not force:
        old = len(pd.read_parquet(path)) if path.suffix == ".parquet" else None
        if old is not None and new_len < old:
            raise RuntimeError(f"{path.name}: new download has {new_len} rows < existing {old}; pass force=True to overwrite")


def yahoo_filename(ticker: str) -> str:
    return ticker.replace("^", "IDX_").replace("=", "_") + ".parquet"


def fetch_yahoo(ticker: str, start: str = "1990-01-01", end: str | None = None, save: bool = True, force: bool = False) -> pd.DataFrame:
    """Daily bars. open/high/low are unadjusted; close = Adj Close (dividend-adjusted), close_raw = Close.
    end is exclusive (yfinance semantics)."""
    import yfinance as yf
    d = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False, threads=False)
    if d.empty:
        raise RuntimeError(f"yfinance returned no rows for {ticker}")
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    out = pd.DataFrame({"open": d["Open"], "high": d["High"], "low": d["Low"], "close": d["Adj Close"], "close_raw": d["Close"], "volume": d["Volume"]})
    out.index = pd.to_datetime(out.index).tz_localize(None); out.index.name = "date"
    out = out[~out.index.duplicated(keep="last")].sort_index().dropna(subset=["close"])
    if save:
        p = paths.raw("yahoo", yahoo_filename(ticker)); _guard_not_shorter(p, len(out), force)
        _atomic_write(p, lambda t: out.to_parquet(t))
    return out


def fetch_cboe(sym: str, save: bool = True, force: bool = False) -> pd.DataFrame:
    """Cboe index history (VIX, VIX9D, VXN, VVIX, VIX3M, ...). Returns a one-column frame 'close'."""
    raw = pd.read_csv(io.BytesIO(_http_get(CBOE_URL.format(sym=sym))))
    raw.columns = [c.strip().upper() for c in raw.columns]
    col = "CLOSE" if "CLOSE" in raw.columns else sym.upper()
    out = pd.DataFrame({"close": pd.to_numeric(raw[col], errors="coerce").to_numpy()}, index=pd.to_datetime(raw["DATE"], format="%m/%d/%Y"))
    out.index.name = "date"; out = out[~out.index.duplicated(keep="last")].sort_index().dropna()
    if save:
        p = paths.raw("cboe", f"{sym}.parquet"); _guard_not_shorter(p, len(out), force)
        _atomic_write(p, lambda t: out.to_parquet(t))
    return out


def fetch_french(name: str = "F-F_Momentum_Factor_daily", save: bool = True) -> Path:
    """Download a Ken French data-library CSV zip and unpack the csv next to it. Returns the csv path.
    Parse with quantkit.data.load.load_french."""
    blob = _http_get(FRENCH_URL.format(name=name))
    zf = zipfile.ZipFile(io.BytesIO(blob)); member = zf.namelist()[0]
    d = paths.raw("french"); d.mkdir(parents=True, exist_ok=True)
    if save:
        _atomic_write(d / f"{name}_CSV.zip", lambda t: t.write_bytes(blob))
        _atomic_write(d / f"{name}.csv", lambda t: t.write_bytes(zf.read(member)))
    return d / f"{name}.csv"


def fetch_squeezemetrics(save: bool = True, force: bool = False) -> pd.DataFrame:
    """SqueezeMetrics daily DIX / GEX (market-wide scalars; known at the close of the date)."""
    raw = pd.read_csv(io.BytesIO(_http_get(SQUEEZE_URL)))
    out = raw.set_index(pd.to_datetime(raw["date"]))[["price", "dix", "gex"]].astype(float)
    out.index.name = "date"; out = out[~out.index.duplicated(keep="last")].sort_index()
    if save:
        p = paths.raw("squeeze_dix_gex.parquet"); _guard_not_shorter(p, len(out), force)
        _atomic_write(p, lambda t: out.to_parquet(t))
    return out


def fetch_moomoo_min1(symbols: list[str], start: str, end: str, host: str = "127.0.0.1", port: int = 11111, save: bool = True) -> dict[str, pd.DataFrame]:
    """Append 1-minute bars from a running moomoo OpenD (needs the `moomoo` package and history-kline quota).

    Output matches raw/min1/<SYM>.csv: time_key (bar END time, US/Eastern), open, high, low, close, volume, turnover.
    Existing rows are kept; overlapping timestamps are replaced by the new download.
    UNTESTED in the sandbox (no OpenD). Mirrors tools/moomoo_api_probes/probe2.py (request_history_kline, K_1M).
    Each request uses one quota unit per symbol-month; check q.get_history_kl_quota() before bulk pulls.
    """
    from moomoo import KLType, OpenQuoteContext, RET_OK, AuType
    q = OpenQuoteContext(host=host, port=port); res = {}
    try:
        for s in symbols:
            frames, page = [], None
            while True:
                ret, df, page = q.request_history_kline(f"US.{s}", start=start, end=end, ktype=KLType.K_1M, autype=AuType.NONE,
                                                        max_count=1000, page_req_key=page)
                if ret != RET_OK:
                    raise RuntimeError(f"request_history_kline({s}) failed: {df}")
                frames.append(df)
                if page is None:
                    break
            new = pd.concat(frames, ignore_index=True)[["time_key", "open", "high", "low", "close", "volume", "turnover"]]
            p = paths.raw("min1", f"{s}.csv")
            if p.exists():
                old = pd.read_csv(p); new = pd.concat([old[~old.time_key.isin(new.time_key)], new], ignore_index=True)
            new = new.sort_values("time_key").reset_index(drop=True); res[s] = new
            if save:
                _atomic_write(p, lambda t: new.to_csv(t, index=False))
    finally:
        q.close()
    return res


def update_all(yahoo=("QQQ", "SPY", "MTUM", "^NDX", "^GSPC"), cboe=("VIX", "VIX9D", "VXN", "VVIX"),
               french=("F-F_Momentum_Factor_daily", "F-F_Research_Data_Factors")) -> dict[str, str]:
    """Refresh every dataset that has a public source. Returns {name: 'ok' | error text}; one failure does not stop the rest."""
    status = {}
    jobs = [(f"yahoo:{t}", lambda t=t: fetch_yahoo(t)) for t in yahoo] + [(f"cboe:{s}", lambda s=s: fetch_cboe(s)) for s in cboe] + \
           [(f"french:{n}", lambda n=n: fetch_french(n)) for n in french] + [("squeeze", fetch_squeezemetrics)]
    for nm, fn in jobs:
        try:
            fn(); status[nm] = "ok"
        except Exception as e:                                   # noqa: BLE001 - report and continue
            status[nm] = f"{type(e).__name__}: {e}"
    return status
