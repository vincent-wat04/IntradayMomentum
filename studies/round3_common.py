"""第三轮预注册假设（H1 隐含-已实现波动缺口 / H2 FOMC 日 / H3 隐含相关性）共用的数据构造。

所有信号都在 t−1 收盘后已知，用于解释 t 日 sleeve 的表现：
    signal_t = f(数据截至 t−1)        （shift(1)）
sleeve 当日的 10:00 之后才开始交易，而 VIX/VXN/COR1M 在 t−1 的 16:15 ET 已收盘，没有前视。

响应变量 u_t = r_t / e_t：sleeve 当日净收益除以当日实际使用的敞口倍数。
这样做是吸取了 GEX 那轮的教训 —— 60~85% 的"优势"只是低波动日仓位更大。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import paths as _P
RAW = _P.RAW
CACHE = _HERE / "outputs/full/intraday/sleeve_qqq_spy.pkl"
EVENTS = _HERE / "data_events" / "fomc_calendar_validated.csv"

# 实盘配置（band + 硬尾 0.5 逐 bar + 排除周一）；两个标的用完全相同的参数
BASE = dict(stop_long="band+tail", stop_short="band+tail", tail_k=0.5,
            tail_every_bar=True, skip_dows=(0,))


def chk(off: int):
    base = [(h, m) for h in range(10, 16) for m in (0, 30)][:-1]
    return [(pd.Timestamp(f"2000-01-01 {h:02d}:{m:02d}") +
             pd.Timedelta(minutes=off)).strftime("%H:%M") for h, m in base]


def build_sleeve(sym: str):
    """30 相位均值的 sleeve 日净收益 r，以及当日敞口倍数 e = min(4, 2%/σ_14d)。

    成本用标的自身的每股成本（半价差 0.005 + 佣金 0.0045），**不做 TQQQ 换算**，
    这样 QQQ 与 SPY 的口径完全一致。它和 tearsheet 里的数字略有差别
    （tearsheet 用的是 TQQQ 成本，更贵），不影响本轮的信号-收益关系。
    """
    from audit import reimpl_noise as RI
    from etft.intraday.bars import load_bars
    b = load_bars(sym)
    acc = []
    for off in range(30):
        rr, _ = RI.simulate(b, checks=chk(off), sizing="voltarget", max_lev=4.0,
                            reentry="hold", half_spread=0.005, commission=0.0045, **BASE)
        acc.append(rr)
    r = pd.DataFrame(acc).T.mean(axis=1)
    r.index = pd.to_datetime(r.index)
    sd = RI.daily_sigma(b.day_close, b.prev_close_adj, 14)
    sd.index = pd.to_datetime(sd.index)
    return r, np.minimum(4.0, 0.02 / sd)


def load_sleeves(rebuild: bool = False):
    if CACHE.exists() and not rebuild:
        return pd.read_pickle(CACHE)
    out = {s: build_sleeve(s) for s in ("QQQ", "SPY")}
    pd.to_pickle(out, CACHE)
    return out


def load_close(rel: str) -> pd.Series:
    d = pd.read_parquet(RAW / rel)
    s = d["close"] if "close" in d.columns else d.iloc[:, 0]
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return s[~s.index.duplicated(keep="last")].sort_index()


def causal_z(x: pd.Series, win: int = 504, minp: int = 252) -> pd.Series:
    """因果 z 分数：只用过去 win 个观测的均值与标准差（含当日，随后整体 shift(1)）。"""
    m = x.rolling(win, min_periods=minp).mean()
    s = x.rolling(win, min_periods=minp).std()
    return (x - m) / s


def rv21_ann(px: pd.Series) -> pd.Series:
    """21 日已实现波动，年化，单位 %（与 VIX/VXN 同口径）。"""
    return px.pct_change().rolling(21).std() * np.sqrt(252) * 100


def signals(sym: str) -> pd.DataFrame:
    """返回 t 日可用的回归变量（均已 shift(1)）。

    SPY 用标普 500 指数的 RV21 对比 VIX；QQQ 用 QQQ 的 RV21 对比 VXN。
    """
    if sym == "SPY":
        px, iv = load_close("yahoo/IDX_GSPC.parquet"), load_close("cboe/VIX.parquet")
    else:
        px, iv = load_close("yahoo/QQQ.parquet"), load_close("cboe/VXN.parquet")
    iv = iv.reindex(px.index).ffill()
    rv = rv21_ann(px)
    cor = load_close("cboe/COR1M.parquet").reindex(px.index).ffill()
    S = pd.DataFrame({"z_gap": causal_z(np.log(iv / rv)),
                      "z_lrv": causal_z(np.log(rv)),
                      "z_cor": causal_z(cor)})
    return S.shift(1)


def fomc_calendar():
    c = pd.read_csv(EVENTS, parse_dates=["date"])
    return (set(c[c.kind == "scheduled"].date), set(c[c.kind == "emergency"].date))


def design(sym: str) -> pd.DataFrame:
    """回归用的日度表：u、三个信号、FOMC 哑变量、星期哑变量。周一与紧急会议日已剔除。"""
    r, e = load_sleeves()[sym]
    df = pd.DataFrame({"r": r, "e": e}).dropna()
    df = df[df.index.dayofweek != 0]                      # 策略周一不交易
    sched, emerg = fomc_calendar()
    df = df[~df.index.isin(emerg)]                         # 预先排除 2020-03 紧急行动日
    df["u"] = df.r / df.e
    df = df.join(signals(sym), how="left")
    df["fomc"] = df.index.isin(sched).astype(float)
    for k, nm in ((1, "tue"), (2, "wed"), (3, "thu")):     # 周五为基准
        df[nm] = (df.index.dayofweek == k).astype(float)
    return df
