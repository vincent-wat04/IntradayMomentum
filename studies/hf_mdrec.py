"""mdrec 逐笔 + 10 档盘口 → 每分钟特征(按美东交易日分组;只取常规时段 09:30–16:00)。
  tick:  买/卖主动成交量(moomoo 的 direction 字段,不需要 Lee-Ready 推断)、成交笔数、大单(≥ 当日 99 分位)净额
  book:  L1 价差(bp)、L1/10 档深度、Cont–Kukanov–Stoikov(2014)L1 OFI、10 档深度失衡
输出 data/_hf_min.pkl:{(ET 日, 代码): DataFrame(index = 分钟收盘时刻 "HH:MM",与 load_bars 的 1 分钟 bar 对齐)}
python3 hf_mdrec.py
"""
from __future__ import annotations

import glob
import gzip
import csv
import os
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("ETFT_DATA", "../data")) / "mdrec" / "data"
OUT = Path(os.environ.get("ETFT_DATA", "../data")) / "_hf_min.pkl"


def read_all(code, kind):
    frames = []
    for f in sorted(glob.glob(str(ROOT / "*" / f"{code}_{kind}_*.csv.gz"))):
        rows, header = [], None
        try:
            with gzip.open(f, "rt", newline="") as fh:
                rd = csv.reader(fh); header = next(rd, None)
                for r in rd:
                    if header and len(r) == len(header): rows.append(r)
        except (EOFError, zlib.error, OSError):
            pass
        if header and rows: frames.append(pd.DataFrame(rows, columns=header))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def minute_key(ts: pd.Series):
    """事件时刻 → 所属 1 分钟 bar 的收盘标签(moomoo 口径:time_key = bar 结束时刻,09:30:xx → 09:31)。"""
    return (ts.dt.floor("min") + pd.Timedelta(minutes=1)).dt.strftime("%H:%M")


def tick_minutes(df):
    df = df.copy(); df["ts"] = pd.to_datetime(df["time"], format="ISO8601"); df = df.drop_duplicates("sequence")
    for c in ("price", "volume", "turnover"): df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["price", "volume"])
    t = df.ts.dt.time; df = df[(t >= pd.Timestamp("09:30").time()) & (t < pd.Timestamp("16:00").time())]
    df["day"] = df.ts.dt.strftime("%Y-%m-%d"); df["m"] = minute_key(df.ts)
    df["sgn"] = np.where(df.direction == "BUY", 1, np.where(df.direction == "SELL", -1, 0))
    out = {}
    for d, g in df.groupby("day"):
        big = g.volume >= g.volume.quantile(0.99)
        G = g.assign(bv=g.volume * (g.sgn > 0), sv=g.volume * (g.sgn < 0), nv=g.volume * (g.sgn == 0),
                     bigflow=g.volume * g.sgn * big, odd=(g["type"] == "ODD_LOT").astype(int)).groupby("m")
        o = pd.DataFrame({"buy_vol": G.bv.sum(), "sell_vol": G.sv.sum(), "neut_vol": G.nv.sum(), "n_trades": G.size(),
                          "big_flow": G.bigflow.sum(), "odd_share": G.odd.mean(), "last_px": G.price.last(), "vol": G.volume.sum()})
        out[d] = o
    return out


def book_minutes(df):
    df = df.copy(); df["ts"] = pd.to_datetime(df["t_bid"], format="ISO8601")
    num = [c for c in df.columns if c[:2] in ("bp", "bs", "ap", "as")]
    for c in num: df[c] = pd.to_numeric(df[c], errors="coerce")
    df[[c for c in num if c[:2] in ("bs", "as")]] = df[[c for c in num if c[:2] in ("bs", "as")]].fillna(0.0)
    df = df.dropna(subset=["bp1", "ap1"]); df = df[(df.ap1 > df.bp1)]
    t = df.ts.dt.time; df = df[(t >= pd.Timestamp("09:30").time()) & (t < pd.Timestamp("16:00").time())].sort_values("ts")
    df["day"] = df.ts.dt.strftime("%Y-%m-%d"); df["m"] = minute_key(df.ts)
    out = {}
    for d, g in df.groupby("day"):
        pb, vb, pa, va = g.bp1.to_numpy(), g.bs1.to_numpy(), g.ap1.to_numpy(), g.as1.to_numpy()
        # Cont–Kukanov–Stoikov(2014)L1 OFI(handbook 第 11 讲同式)
        e_b = np.where(pb[1:] > pb[:-1], vb[1:], np.where(pb[1:] == pb[:-1], vb[1:] - vb[:-1], -vb[:-1]))
        e_a = np.where(pa[1:] < pa[:-1], va[1:], np.where(pa[1:] == pa[:-1], va[1:] - va[:-1], -va[:-1]))
        ofi = np.r_[0.0, e_b - e_a]
        mid = (pb + pa) / 2; spr = (pa - pb) / mid * 1e4
        db10 = g[[f"bs{i}" for i in range(1, 11)]].sum(1).to_numpy(); da10 = g[[f"as{i}" for i in range(1, 11)]].sum(1).to_numpy()
        G = g.assign(ofi=ofi, spr=spr, d1=vb + va, imb10=(db10 - da10) / np.maximum(db10 + da10, 1), mid=mid).groupby("m")
        out[d] = pd.DataFrame({"ofi": G.ofi.sum(), "spread_bp": G.spr.mean(), "depth1": G.d1.mean(), "imb10": G.imb10.mean(),
                               "mid_last": G.mid.last(), "n_book": G.size()})
    return out


def main():
    res = {}
    for code in ("US.QQQ", "US.TQQQ"):
        tk, bk = tick_minutes(read_all(code, "ticker")), book_minutes(read_all(code, "book"))
        for d in sorted(set(tk) | set(bk)):
            res[(d, code)] = pd.concat([tk.get(d, pd.DataFrame()), bk.get(d, pd.DataFrame())], axis=1).sort_index()
    pd.to_pickle(res, OUT)
    rows = []
    for (d, c), x in sorted(res.items()):
        rows.append(dict(日=d, 代码=c, 分钟数=len(x), 首=x.index.min(), 末=x.index.max(), 成交量=x.vol.sum() if "vol" in x else np.nan,
                         主买占比=(x.buy_vol.sum() / (x.buy_vol.sum() + x.sell_vol.sum())) if "buy_vol" in x else np.nan,
                         中性占比=(x.neut_vol.sum() / x.vol.sum()) if "vol" in x else np.nan, 平均价差bp=x.spread_bp.mean() if "spread_bp" in x else np.nan))
    pd.set_option("display.width", 220, "display.float_format", lambda v: f"{v:10.3f}")
    print(pd.DataFrame(rows).to_string(index=False)); print(f"→ {OUT}")


if __name__ == "__main__":
    main()
