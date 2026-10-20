"""价格持续时间(price duration)事件时钟 —— 1 分钟 close 上的 PDS(price duration sampling)。

来源:Hong–Nolte–Taylor–Zhao (2023) 的阈值事件定义 + Li–Nolte–Nolte–Yu (2025) 的 K-自适应日阈值。
事件定义(论文 Eq.(2)/(3)):相对**上一个事件价**的绝对对数价格变化首次 ≥ δ_d 时记一个事件,
事件价更新为触发 bar 的收盘价;δ_d 在当天开盘前就确定(PIT)。

K-自适应:δ_d = sqrt(E[V_d | F_{d-1}] / K)。这里 E[V_d] 用 σ_14²(= 仓位分母用的同一个量,与基线一致),
于是当天期望事件数 ≈ K,日间波动动态被吸收进阈值,日内事件流只保留日内动态。

每个事件输出:所在 bar 索引、方向(+1/−1)、与上一事件的间隔(分钟)。
特征(全部只用 ≤ 当前 bar 的事件,PIT):
  n_ev            截至 k 的事件数;  ev_rate = n_ev / (k+1) / (K/390)  —— 相对期望强度(1=按预期)
  run_len         末尾同向事件连续个数(带符号:顺入场方向为正);  imb = (上事件数 − 下事件数)/n_ev
  dur_last / dur_mean   最近一个/最近 m 个持续时间(分钟)相对季节期望(同一时点的跨日历史均值)之比的对数
  t_first_exit    第一次出现与入场方向一致的"带宽外"事件的时刻(用于 R2)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def event_clock(c_row: np.ndarray, o_day: float, delta: float, n_valid: int):
    """单日参考实现(显式循环,用于对账)。c_row:该日 1 分钟收盘;o_day:当日开盘价。
    返回 (事件 bar 索引数组, 方向数组)。起始事件价 = 当日开盘价。"""
    ref = float(o_day)
    idx, sg = [], []
    for k in range(n_valid):
        ck = c_row[k]
        if not np.isfinite(ck):
            continue
        ch = np.log(ck / ref)
        if abs(ch) >= delta:
            idx.append(k); sg.append(1 if ch > 0 else -1)
            ref = float(ck)
    return np.array(idx, dtype=int), np.array(sg, dtype=int)


def day_events(b, sd14: np.ndarray, K: int):
    """所有交易日的事件。sd14[d] 是第 d 天 PIT 的 σ_14(日收益标准差)。δ_d = σ14 / sqrt(K)。"""
    c = b.c.to_numpy(); O = b.day_open.to_numpy(); li = b.last_idx.to_numpy()
    out = []
    for i in range(len(c)):
        sdi = sd14[i]
        if not np.isfinite(sdi) or sdi <= 0:
            out.append((np.array([], int), np.array([], int), np.nan)); continue
        delta = sdi / np.sqrt(K)
        idx, sg = event_clock(c[i], O[i], delta, li[i] + 1)
        out.append((idx, sg, delta))
    return out


def seasonal_duration(ev, K: int, n_bins: int = 13, lookback: int = 60):
    """季节期望:同一 30 分钟时段、过去 lookback 个交易日(不含当天)内,事件间隔(分钟)的均值。
    返回 (n_days, n_bins) 数组,第 i 行只用 <i 天的数据(PIT)。"""
    n = len(ev)
    sums = np.zeros((n, n_bins)); cnts = np.zeros((n, n_bins))
    for i, (idx, sg, delta) in enumerate(ev):
        if len(idx) < 2:
            continue
        gaps = np.diff(idx).astype(float)
        bins = np.minimum(idx[1:] // 30, n_bins - 1)
        for g, bn in zip(gaps, bins):
            sums[i, bn] += g; cnts[i, bn] += 1
    S = np.full((n, n_bins), np.nan)
    cs, cc = np.cumsum(sums, axis=0), np.cumsum(cnts, axis=0)
    for i in range(1, n):
        lo = max(0, i - lookback)
        s = cs[i - 1] - (cs[lo - 1] if lo > 0 else 0); c_ = cc[i - 1] - (cc[lo - 1] if lo > 0 else 0)
        S[i] = np.where(c_ >= 5, s / np.maximum(c_, 1), np.nan)
    return S


def trade_features(ev, season, K: int, TR_day_idx: np.ndarray, k_dec: np.ndarray, side: np.ndarray, m: int = 3):
    """对每笔交易(决策 bar = k_dec,方向 side)计算事件流特征。只用 idx <= k_dec 的事件。"""
    n = len(k_dec)
    out = {k: np.full(n, np.nan) for k in ("ev_n", "ev_rate", "ev_run", "ev_imb", "ev_dur_last", "ev_dur_m", "ev_since", "ev_same_frac")}
    for j in range(n):
        idx, sg, delta = ev[TR_day_idx[j]]
        if len(idx) == 0 or not np.isfinite(delta):
            out["ev_n"][j] = 0; out["ev_rate"][j] = 0.0; out["ev_run"][j] = 0; out["ev_imb"][j] = 0.0
            continue
        m_ = idx <= k_dec[j]
        ii, ss = idx[m_], sg[m_] * side[j]                          # ss>0:顺入场方向的事件
        out["ev_n"][j] = len(ii)
        out["ev_rate"][j] = len(ii) / ((k_dec[j] + 1) / 390.0) / K
        if len(ii) == 0:
            out["ev_run"][j] = 0; out["ev_imb"][j] = 0.0; continue
        run = 0
        for v in ss[::-1]:
            if v == ss[-1]: run += 1
            else: break
        out["ev_run"][j] = run * ss[-1]
        out["ev_imb"][j] = ss.sum() / len(ss)
        out["ev_same_frac"][j] = (ss > 0).mean()
        out["ev_since"][j] = (k_dec[j] - ii[-1]) / 30.0
        if len(ii) >= 2:
            gaps = np.diff(ii).astype(float); bins = np.minimum(ii[1:] // 30, season.shape[1] - 1)
            exp = season[TR_day_idx[j], bins]
            r = np.log(np.maximum(gaps, 1.0) / exp)
            out["ev_dur_last"][j] = r[-1]
            out["ev_dur_m"][j] = np.nanmean(r[-m:])
    return pd.DataFrame(out)
