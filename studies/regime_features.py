"""L0 宏观/市场环境特征 —— 见 docs/PROPOSAL_regime_layer.md §1–§2。所有序列在 t 日收盘形成,只用 ≤t 的数据。
特征符号(+/−)写在 SIGN 里,取自经济含义,不看数据。"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(os.environ["ETFT_DATA"]) / "raw"
SECT = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU"]
CYC, DEF = ["XLY", "XLI", "XLF", "XLB"], ["XLP", "XLU", "XLV"]

SIGN = {"b1": +1, "b2": +1, "b3": +1, "b4": +1, "b5": +1, "b6": +1,
        "m1": -1, "m2": -1, "m3": -1, "m4": +1, "m5": +1, "m6": -1,
        "c1": +1, "c2": +1, "c3": +1, "v1": +1}
GROUPS = {"B": ["b1", "b2", "b3", "b4", "b5", "b6"], "M": ["m1", "m2", "m3", "m4", "m5", "m6"], "C": ["c1", "c2", "c3"], "V": ["v1"]}
CORE = ["b1", "b2", "b4", "b5", "m1", "m2", "m3", "m4", "m5", "m6"]       # 1999 起可得的特征(无 RSP/HYG/LQD/IEF/TLT 之外的晚起序列)


def zroll(x: pd.Series, win: int = 252, minp: int = 126) -> pd.Series:
    m, s = x.rolling(win, min_periods=minp).mean(), x.rolling(win, min_periods=minp).std()
    return (x - m) / s.replace(0, np.nan)


def rsi(x: pd.Series, n: int = 14) -> pd.Series:
    d = x.diff(); up, dn = d.clip(lower=0), (-d).clip(lower=0)
    au, ad = up.ewm(alpha=1 / n, adjust=False, min_periods=n).mean(), dn.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + au / ad.replace(0, np.nan))


def load_P() -> pd.DataFrame:
    P = pd.read_parquet(RAW / "regime" / "yahoo_adj.parquet")
    P.index = pd.to_datetime(P.index).tz_localize(None)
    return P


def ecy_series(P: pd.DataFrame, cal: pd.DatetimeIndex) -> pd.Series:
    """Excess CAPE Yield = 1/CAPE_t − TNX_t/100。CAPE_t = CAPE_m × GSPC_t / GSPC_(m 日),m = 满足『行日期 ≤ t − 31 天』的最近一行(滞后 1 个月,PIT)。
    Shiller 数据止于 2023-09:此后 CAPE_m 固定、价格项继续更新(近似;盈利项越来越旧)。名义超额收益率,未做通胀调整。"""
    s = pd.read_parquet(RAW / "regime" / "shiller.parquet")
    s.index = pd.to_datetime(s.index); cape = s["CAPE"].dropna()
    g = P["^GSPC"].reindex(cal).ffill()
    out = pd.Series(np.nan, index=cal)
    rowdates = cape.index.to_numpy()
    pos = np.searchsorted(rowdates, (cal - pd.Timedelta(days=31)).to_numpy(), side="right") - 1
    ok = pos >= 0
    m_dates = pd.DatetimeIndex(rowdates[pos[ok]])
    g_m = g.reindex(m_dates, method="ffill").to_numpy()
    cape_t = cape.to_numpy()[pos[ok]] * g.to_numpy()[ok] / g_m
    out[ok] = 1.0 / cape_t - P["^TNX"].reindex(cal).ffill().to_numpy()[ok] / 100.0
    return out


def features(P: pd.DataFrame, cal: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    cal = P["QQQ"].dropna().index if cal is None else cal
    Q = P.reindex(cal).ffill()
    lr = lambda a, n=63: np.log(a).diff(n)
    F = pd.DataFrame(index=cal)
    sec = Q[SECT]
    F["b1_raw"] = (sec > sec.rolling(50).mean()).where(sec.rolling(50).mean().notna()).mean(axis=1)
    F["b2_raw"] = (sec.rolling(10).mean() > sec.rolling(20).mean()).where(sec.rolling(20).mean().notna()).mean(axis=1)
    F["b3_raw"] = lr(Q["RSP"] / Q["SPY"])
    F["b4_raw"] = lr(Q[CYC], 63).mean(axis=1) - lr(Q[DEF], 63).mean(axis=1)
    F["b5_raw"] = lr(Q["SMH"] / Q["QQQ"])
    F["b6_raw"] = lr(Q["XLK"] / Q["SPY"])
    F["m1_raw"] = Q["^TYX"].diff(63)
    F["m2_raw"] = Q["^TNX"]
    F["m3_raw"] = lr(Q["CL=F"])
    F["m4_raw"] = Q["^TNX"] - Q["^IRX"]
    F["m5_raw"] = lr(Q["HG=F"] / Q["GC=F"])
    F["m6_raw"] = lr(Q["DX-Y.NYB"])
    F["c1_raw"] = lr(Q["HYG"] / Q["IEF"]); F["c2_raw"] = lr(Q["LQD"] / Q["IEF"]); F["c3_raw"] = lr(Q["TLT"])
    ecy = ecy_series(P, cal); F["v1_raw"] = ecy
    for k in SIGN:
        F[k] = (zroll(F[f"{k}_raw"], 1260, 630) if k == "v1" else zroll(F[f"{k}_raw"])).clip(-2, 2) * SIGN[k]
    for g, mem in GROUPS.items():
        F[f"A_{g}"] = F[mem].mean(axis=1, skipna=True)
    F["A_all"] = F[[f"A_{g}" for g in GROUPS]].mean(axis=1, skipna=True)
    F["A_core"] = F[CORE].mean(axis=1, skipna=True)
    # ---- 符号不明项(只进 walk-forward 岭回归)
    bond = (Q[["TLT", "IEF", "LQD", "AGG"]].pct_change().mean(axis=1, skipna=True)).fillna(0).add(1).cumprod()
    F["bond_rsi_d"] = (rsi(bond, 14) - 50) / 20
    wk = bond.resample("W-FRI").last().dropna()
    F["bond_rsi_w"] = ((rsi(wk, 14) - 50) / 20).reindex(cal, method="ffill")
    return F


def vix_z(cal) -> pd.DataFrame:
    out = {}
    for n in ("VIX", "VIX3M"):
        f = RAW / "cboe" / f"{n}.parquet"
        s = pd.read_parquet(f)["close"]; s.index = pd.to_datetime(s.index).tz_localize(None)
        out[n] = s[~s.index.duplicated()].reindex(cal).ffill()
    D = pd.DataFrame({"vix_z": zroll(np.log(out["VIX"])), "vix_term_z": zroll(np.log(out["VIX3M"] / out["VIX"]))}, index=cal)
    return D


def cot_z(cal) -> pd.Series:
    c = pd.read_parquet(RAW / "regime" / "cot_gasoline.parquet"); c.index = pd.to_datetime(c.index)
    c = c[c.code == "111659"]["mm_net_pct_oi"]
    c = c[c.index.notna()].sort_index()                                              # 1 行报告日缺失(NaT),无法定日,丢弃
    c = c[~c.index.duplicated(keep="last")]
    avail = c.sort_index().copy(); avail.index = avail.index + pd.Timedelta(days=4)               # 周二数据 → 周六发布 → 下一交易日起可用
    z = (avail - avail.rolling(156, min_periods=52).mean()) / avail.rolling(156, min_periods=52).std()
    z = z.sort_index(); z = z[~z.index.duplicated(keep="last")]
    return z.reindex(cal, method="ffill").clip(-3, 3)


# ------------------------------------------------------------------ 第二轮:『指标自身趋势』投票
def votes(P: pd.DataFrame, F: pd.DataFrame | None = None) -> pd.DataFrame:
    """每个特征取水平序列,vote = 1[SIGN·(L − SMA126(L)) > 0];b1/b2 用占比>0.5;v1 用 ECY>1260 日中位数。仅用 ≤t 的数据。"""
    F = features(P) if F is None else F
    cal = F.index; Q = P.reindex(cal).ffill(); lg = np.log
    sec = Q[SECT]
    cyc_def = (Q[CYC].pct_change().mean(axis=1) - Q[DEF].pct_change().mean(axis=1)).fillna(0).add(1).cumprod()
    L = {"b3": lg(Q["RSP"] / Q["SPY"]), "b4": lg(cyc_def), "b5": lg(Q["SMH"] / Q["QQQ"]), "b6": lg(Q["XLK"] / Q["SPY"]),
         "m1": Q["^TYX"], "m2": Q["^TNX"], "m3": lg(Q["CL=F"]), "m4": Q["^TNX"] - Q["^IRX"], "m5": lg(Q["HG=F"] / Q["GC=F"]), "m6": lg(Q["DX-Y.NYB"]),
         "c1": lg(Q["HYG"] / Q["IEF"]), "c2": lg(Q["LQD"] / Q["IEF"]), "c3": lg(Q["TLT"])}
    V = pd.DataFrame(index=cal)
    for k, x in L.items():
        m = x.rolling(126, min_periods=126).mean()
        V[k] = ((SIGN[k] * (x - m)) > 0).astype(float).where(m.notna())
    V["b1"] = (F["b1_raw"] > 0.5).astype(float).where(F["b1_raw"].notna())
    V["b2"] = (F["b2_raw"] > 0.5).astype(float).where(F["b2_raw"].notna())
    ecy = F["v1_raw"]; med = ecy.rolling(1260, min_periods=630).median()
    V["v1"] = (ecy > med).astype(float).where(med.notna())
    out = pd.DataFrame(index=cal)
    for g, mem in GROUPS.items():
        out[f"A2_{g}"] = V[mem].mean(axis=1, skipna=True)
    out["A2_all"] = out[[f"A2_{g}" for g in GROUPS]].mean(axis=1, skipna=True)
    return out.ewm(span=10, adjust=False, min_periods=1).mean().where(out.notna())
