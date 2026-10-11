"""Point-in-time fundamental factors from SEC XBRL, for the 98-name universe.

WHY THESE FACTORS.  Hou, Xue & Zhang's "Replicating Anomalies" (NBER 23394) re-tested 452 published
anomalies and found most do not survive; the ones that do cluster around INVESTMENT and PROFITABILITY,
which is also what their q-factor model is built from.  So the set below is deliberately small and
limited to the families with replication evidence, rather than a wide sweep of accounting ratios:

  ROE               NetIncome(4q) / StockholdersEquity      Hou, Xue & Zhang (2015) q-factor ROE
  GROSS_PROF        (Revenue - CostOfRevenue)(4q) / Assets  Novy-Marx (2013) gross profitability
  OP_PROF           OperatingIncome(4q) / StockholdersEquity Fama & French (2015) RMW
  ASSET_GROWTH      Assets YoY growth                       Cooper, Gulen & Schill (2008); q-factor I/A
  ACCRUALS          (NetIncome - OperatingCashFlow)(4q)/Assets  Sloan (1996)
  NET_ISSUANCE      shares outstanding YoY change           Pontiff & Woodgate (2008)
  BM                StockholdersEquity / market cap         Fama & French (1992)
  SUE               standardised unexpected earnings        Bernard & Thomas (1989) PEAD
  ROE_CHG           4q change in ROE                        fundamental momentum, Hou et al.

POINT-IN-TIME DISCIPLINE.  Every XBRL fact carries a `filed` date.  A value is only allowed into the
factor panel from its filing date onward -- never from the fiscal period end, which would be a
look-ahead of 30-90 days and is the single most common error in fundamental backtests.  Where several
restatements of the same period exist, the FIRST filed value is used (what was actually knowable).

NOTE ON FREQUENCY.  These are quarterly step functions.  Section "multi-frequency" of the report
discusses how they should and should not be combined with the daily price/volume factors.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

UA = {"User-Agent": "academic-quant-research contact@example.com"}
CACHE = Path("data/xs/sec"); CACHE.mkdir(parents=True, exist_ok=True)
CONCEPTS = {
    "Assets": ["Assets"],
    "Equity": ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "NetIncome": ["NetIncomeLoss"],
    "Revenue": ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet"],
    "CostOfRevenue": ["CostOfRevenue", "CostOfGoodsAndServicesSold"],
    "OpIncome": ["OperatingIncomeLoss"],
    "OCF": ["NetCashProvidedByUsedInOperatingActivities",
            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
    "Shares": ["CommonStockSharesOutstanding", "WeightedAverageNumberOfDilutedSharesOutstanding"],
}


def _get(url, tries=3):
    for i in range(tries):
        try:
            return json.load(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30))
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(1 + i)
        except Exception:
            time.sleep(1 + i)
    return None


def cik_map():
    f = CACHE / "tickers.json"
    if f.exists():
        m = json.loads(f.read_text())
    else:
        m = _get("https://www.sec.gov/files/company_tickers.json")
        f.write_text(json.dumps(m))
    return {v["ticker"]: str(v["cik_str"]).zfill(10) for v in m.values()}


def fetch(ticker, cik):
    """Return {concept: DataFrame[end, val, filed, form]} using the first-filed value per period."""
    f = CACHE / f"{ticker}.json"
    if f.exists():
        raw = json.loads(f.read_text())
    else:
        raw = {}
        for name, tags in CONCEPTS.items():
            for tag in tags:
                d = _get(f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/us-gaap/{tag}.json")
                time.sleep(0.12)
                if d and "units" in d:
                    unit = "USD" if "USD" in d["units"] else ("shares" if "shares" in d["units"] else None)
                    if unit:
                        raw[name] = d["units"][unit]
                        break
        f.write_text(json.dumps(raw))
    out = {}
    for name, recs in raw.items():
        if not recs:
            continue
        df = pd.DataFrame(recs)
        for c in ("end", "filed", "val", "form"):
            if c not in df:
                df[c] = np.nan
        df = df[df.form.astype(str).str.startswith(("10-Q", "10-K"))]
        if "start" in df:                 # flow items: keep the ~quarterly ones
            dur = (pd.to_datetime(df.end) - pd.to_datetime(df.get("start"))).dt.days
            if dur.notna().any():
                df = df[(dur.isna()) | ((dur > 60) & (dur < 115))]
        df = (df.sort_values(["end", "filed"]).groupby("end", as_index=False).first())
        out[name] = df[["end", "val", "filed"]]
    return out


def pit_series(df, dates, how="last"):
    """Step function on the trading calendar: a value appears on its FILED date, not its period end."""
    if df is None or len(df) == 0:
        return pd.Series(np.nan, index=dates)
    d = df.dropna(subset=["filed"]).copy()
    d["filed"] = pd.to_datetime(d.filed).dt.strftime("%Y-%m-%d")
    d = d.sort_values("filed").groupby("filed", as_index=False).last()
    s = pd.Series(d.val.to_numpy(float), index=d.filed)
    return s.reindex(sorted(set(s.index) | set(dates))).ffill().reindex(dates)


def rolling4(df, dates):
    """Trailing-four-quarter sum of a flow item, stamped at each filing date."""
    if df is None or len(df) == 0:
        return pd.Series(np.nan, index=dates)
    d = df.dropna(subset=["filed"]).copy()
    d["filed"] = pd.to_datetime(d.filed).dt.strftime("%Y-%m-%d")
    d = d.sort_values("end")
    d["ttm"] = d.val.astype(float).rolling(4, min_periods=4).sum()
    d = d.dropna(subset=["ttm"]).sort_values("filed").groupby("filed", as_index=False).last()
    s = pd.Series(d.ttm.to_numpy(float), index=d.filed)
    return s.reindex(sorted(set(s.index) | set(dates))).ffill().reindex(dates)


def yoy(df, dates):
    if df is None or len(df) < 5:
        return pd.Series(np.nan, index=dates)
    d = df.dropna(subset=["filed"]).copy()
    d["filed"] = pd.to_datetime(d.filed).dt.strftime("%Y-%m-%d")
    d = d.sort_values("end")
    d["g"] = d.val.astype(float).pct_change(4)
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=["g"])
    d = d.sort_values("filed").groupby("filed", as_index=False).last()
    s = pd.Series(d.g.to_numpy(float), index=d.filed)
    return s.reindex(sorted(set(s.index) | set(dates))).ffill().reindex(dates)


def sue(df, dates):
    """Standardised unexpected earnings: (x_q - x_{q-4}) / std of the last 8 such changes."""
    if df is None or len(df) < 12:
        return pd.Series(np.nan, index=dates)
    d = df.dropna(subset=["filed"]).copy()
    d["filed"] = pd.to_datetime(d.filed).dt.strftime("%Y-%m-%d")
    d = d.sort_values("end")
    ch = d.val.astype(float).diff(4)
    d["sue"] = ch / ch.rolling(8, min_periods=6).std()
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=["sue"])
    d = d.sort_values("filed").groupby("filed", as_index=False).last()
    s = pd.Series(d.sue.to_numpy(float), index=d.filed)
    return s.reindex(sorted(set(s.index) | set(dates))).ffill().reindex(dates)


NAMES = ["ROE", "GROSS_PROF", "OP_PROF", "ASSET_GROWTH", "ACCRUALS", "NET_ISSUANCE", "BM", "SUE", "ROE_CHG"]


def build(px: pd.DataFrame, mktcap: pd.DataFrame):
    dates = list(px.index)
    cm = cik_map()
    cols = {n: {} for n in NAMES}
    miss = []
    for i, tk in enumerate(px.columns):
        cik = cm.get(tk.replace("-", "-")) or cm.get(tk.replace("-", ".")) or cm.get(tk.split("-")[0])
        if cik is None:
            miss.append(tk); continue
        f = fetch(tk, cik)
        A = pit_series(f.get("Assets"), dates)
        E = pit_series(f.get("Equity"), dates)
        NI = rolling4(f.get("NetIncome"), dates)
        REV = rolling4(f.get("Revenue"), dates)
        COGS = rolling4(f.get("CostOfRevenue"), dates)
        OI = rolling4(f.get("OpIncome"), dates)
        OCF = rolling4(f.get("OCF"), dates)
        SH = pit_series(f.get("Shares"), dates)
        with np.errstate(all="ignore"):
            cols["ROE"][tk] = NI / E.where(E > 0)
            cols["GROSS_PROF"][tk] = (REV - COGS) / A.where(A > 0)
            cols["OP_PROF"][tk] = OI / E.where(E > 0)
            cols["ASSET_GROWTH"][tk] = yoy(f.get("Assets"), dates)
            cols["ACCRUALS"][tk] = (NI - OCF) / A.where(A > 0)
            cols["NET_ISSUANCE"][tk] = SH.pct_change(252)
            cols["BM"][tk] = E / mktcap[tk].where(mktcap[tk] > 0)
            cols["SUE"][tk] = sue(f.get("NetIncome"), dates)
            cols["ROE_CHG"][tk] = (NI / E.where(E > 0)).diff(252)
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(px.columns)} tickers", flush=True)
    F = {n: pd.DataFrame(cols[n]).reindex(columns=px.columns).replace([np.inf, -np.inf], np.nan)
         for n in NAMES}
    if miss:
        print("no CIK for:", miss)
    return F


if __name__ == "__main__":
    import cross_section_study as XS
    px, op, hi, lo, vo = XS.load()
    mktcap = px * vo.rolling(252, min_periods=60).mean()      # liquidity-scaled size proxy
    F = build(px, mktcap)
    pd.to_pickle(F, "data/xs/fundamentals.pkl")
    cov = pd.Series({k: v.notna().mean().mean() for k, v in F.items()})
    print("\ncoverage:\n", cov.round(3).to_string())
    print("\ncross-sectional dispersion (last day):")
    print(pd.DataFrame({k: v.iloc[-1].describe()[["count", "mean", "std"]] for k, v in F.items()}).round(3).to_string())
