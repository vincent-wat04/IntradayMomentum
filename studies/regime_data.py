"""宏观/市场环境数据(长历史)。全部为公开来源,缓存到 ETFT_DATA/raw/regime。
Yahoo 日线(复权收盘,另存原始收盘与成交量)、Shiller ie_data.xls(CAPE/10y 利率/盈利)、CFTC 分类期货持仓(汽油 RBOB 管理资金净多头)。
PIT 约定在 regime_features.py 里统一处理(Shiller 月度滞后 2 个月;COT 周二数据周五发布 → 周一起可用)。"""
import io, os, zipfile, sys
from pathlib import Path
import numpy as np, pandas as pd, requests, yfinance as yf

OUT = Path(os.environ["ETFT_DATA"]) / "raw" / "regime"; OUT.mkdir(parents=True, exist_ok=True)
YF = ["SPY", "QQQ", "RSP", "IWM", "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "SMH", "TLT", "IEF", "SHY", "LQD", "HYG", "AGG", "TIP",
      "^TNX", "^TYX", "^FVX", "^IRX", "CL=F", "RB=F", "HG=F", "GC=F", "DX-Y.NYB", "JPY=X", "^GSPC", "^NDX"]


def yahoo():
    f = OUT / "yahoo_adj.parquet"
    if f.exists(): return
    d = yf.download(YF, start="1998-01-01", end="2026-10-12", auto_adjust=True, progress=False, threads=False)
    d["Close"].to_parquet(f); d["Volume"].to_parquet(OUT / "yahoo_vol.parquet")
    print("yahoo", d["Close"].shape, flush=True)


def shiller():
    f = OUT / "shiller.parquet"
    if f.exists(): return
    r = requests.get("http://www.econ.yale.edu/~shiller/data/ie_data.xls", timeout=60)
    x = pd.read_excel(io.BytesIO(r.content), sheet_name="Data", header=7)
    x = x.rename(columns=lambda c: str(c).strip())
    x = x[pd.to_numeric(x.iloc[:, 0], errors="coerce").notna()].copy()
    x["date"] = x.iloc[:, 0].astype(float).round(2).astype(str)
    yr = x.iloc[:, 0].astype(float).astype(int); mo = ((x.iloc[:, 0].astype(float) - yr) * 100).round().astype(int).clip(1, 12)
    x["date"] = pd.to_datetime(dict(year=yr, month=mo, day=1)) + pd.offsets.MonthEnd(0)
    keep = {c: c for c in x.columns if c in ("P", "D", "E", "CPI", "Rate GS10", "CAPE")}
    out = x.set_index("date")[list(keep)].apply(pd.to_numeric, errors="coerce")
    out.to_parquet(f); print("shiller", out.shape, out.index.min(), out.index.max(), list(out.columns), flush=True)


def cot():
    f = OUT / "cot_gasoline.parquet"
    if f.exists(): return
    rows = []
    for y in range(2006, 2027):
        u = f"https://www.cftc.gov/files/dea/history/fut_disagg_txt_{y}.zip"
        r = requests.get(u, timeout=120, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200: print("COT miss", y, r.status_code); continue
        z = zipfile.ZipFile(io.BytesIO(r.content)); d = pd.read_csv(z.open(z.namelist()[0]), low_memory=False)
        m = d[d["CFTC_Contract_Market_Code"].astype(str).str.strip().isin(["111659", "111658"])]       # RBOB / 旧 NYMEX 汽油
        rows.append(m)
    d = pd.concat(rows)
    d["date"] = pd.to_datetime(d["Report_Date_as_YYYY-MM-DD"])
    d = d.sort_values("date").drop_duplicates(["date", "CFTC_Contract_Market_Code"])
    out = pd.DataFrame({"mm_long": d["M_Money_Positions_Long_All"].to_numpy(), "mm_short": d["M_Money_Positions_Short_All"].to_numpy(),
                        "oi": d["Open_Interest_All"].to_numpy(), "code": d["CFTC_Contract_Market_Code"].astype(str).str.strip().to_numpy()}, index=d["date"])
    out["mm_net_pct_oi"] = (out.mm_long - out.mm_short) / out.oi
    out.to_parquet(f); print("cot", out.shape, out.index.min(), out.index.max(), out.code.value_counts().to_dict(), flush=True)


if __name__ == "__main__":
    yahoo(); shiller(); cot()
