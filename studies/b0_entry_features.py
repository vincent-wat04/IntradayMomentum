"""B0:逐笔入场的条件期望 —— 新特征的构造与增量检验(描述性,不选配置)。

对应 docs/PROPOSAL_duration_vol_entry.md §6 B0。口径:
  * 逐笔表 = trade_filter_study.build_trades(30 相位,29,679 笔,与引擎逐位对账);
  * 基线特征 F1 = trade_filter_study.entry_features 的原函数(14 个,已冻结);
  * 新特征(全部 PIT:只用 bar ≤ k_dec 与 t−1 及更早的数据):
      DUR  事件时钟(duration_events.py):强度 ev_rate、游程 ev_run、不平衡 ev_imb、同向占比、持续时间意外 ev_dur_*;
      MIC  1 分钟 OHLCV 微观结构代理:BVC 净订单流(Φ(ΔP/σ))、Kyle-λ 代理、Amihud、区间价差代理(Corwin–Schultz 简化)、突破 bar 量能占比;
      BRP  Bespoke 风格的"按时点加权"的已实现方差比(Patton–Zhang 的结论:越靠近决策时点的 bar 权重越高):
           近端(最近 30 分钟)RV 占全天 RV 的份额,相对季节期望;
      VOL  波动偏相关族:log VIX/VXN/前日 RV 对 log σ14 取残差(训练窗内回归,系数只用训练期);
      CJ   共同跳跃 E-COS 标记(QQQ–TLT,经验零分布)——见 cojump_cos.py。
  * 目标:y = 方向收益(入场→实际离场,bp),含止损后的实际结局。
  * Walk-forward:每年用截至上年末数据重估;OOS = 2021–2026;dev-OOS = 2021–22,test = 2023–26。

用法: python b0_entry_features.py --build   # 生成特征表并自检
      python b0_entry_features.py --eval    # 增量检验
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import duration_events as DE
import lowvol_avoidance_study as LV
import trade_filter_study as TF
from etft.intraday.bars import load_bars
from event_calendar import event_frame

DATA = Path(os.environ.get("ETFT_DATA", "../data"))
K_EV = 26                       # 事件时钟的 K(平均 ~18 个事件/天,约 22 分钟/事件);**冻结,不扫描**
TAB = Path("../outputs/tables"); TAB.mkdir(parents=True, exist_ok=True)


def _rowwise(fn, X):
    """逐日独立计算。对整个二维数组沿 axis=1 向量化时,NumPy 的分块/SIMD 路径会让某一行的最后一位浮点数
    依赖于其它行的内容(2026-10-20 投毒自检发现:污染第 i 天后,i 之前日子的 nanstd 差 1e-19)。
    这不是前视,但违反"逐位不变"的审计标准,而且 rolling 的在线算法会把误差向后传播。"""
    return np.array([fn(X[i]) for i in range(len(X))])


def _causal_roll_mean(x, w, minp):
    """确定性的因果滚动均值:第 i 个值 = x[i−w:i] 的均值(不含 i,即已 shift(1)),每个窗口独立求和。"""
    out = np.full(len(x), np.nan)
    for i in range(len(x)):
        seg = x[max(0, i - w):i]; seg = seg[np.isfinite(seg)]
        if len(seg) >= minp:
            out[i] = seg.sum() / len(seg)
    return out


def intraday_sd(b):
    """论文口径的日内方差尺度:过去 14 日 1 分钟 RV 均值的平方根(PIT)。"""
    c = b.c.to_numpy(); O = b.day_open.to_numpy()
    r1 = np.full_like(c, np.nan); r1[:, 1:] = np.log(c[:, 1:] / c[:, :-1]); r1[:, 0] = np.log(c[:, 0] / O)
    rv = _rowwise(np.nansum, r1 ** 2)
    return np.sqrt(_causal_roll_mean(rv, 14, 7)), r1


def micro_features(b, TR, r1, sd_in):
    """1 分钟 OHLCV 的微观结构代理。方向性特征按交易方向取符号(顺势为正)。"""
    days = b.days; dpos = {d: i for i, d in enumerate(days)}
    c, o, h, l, v = (b.c.to_numpy(), b.o.to_numpy(), b.h.to_numpy(), b.l.to_numpy(), b.v.to_numpy())
    turnover = v * c
    ii = TR.day.map(dpos).to_numpy(); k = TR.k_dec.to_numpy(); s = TR.side.to_numpy().astype(float); n = len(TR)
    # BVC(Easley–López de Prado–O'Hara):买入量份额 = Φ(ΔP/σ_bar),σ_bar 取过去 14 日 1 分钟收益的标准差(PIT,日级)
    sig_bar = _causal_roll_mean(_rowwise(np.nanstd, r1), 14, 7)           # 逐日独立计算 + 确定性滚动均值(见 _rowwise 的注释)
    dP = r1
    buy_share = norm.cdf(dP / np.where(sig_bar[:, None] > 0, sig_bar[:, None], np.nan))
    sgn_vol = (2 * buy_share - 1) * v                                     # 带符号成交量代理
    # 相对成交量(分钟 × 14 日,PIT):同一 bar 位置过去 14 日的均值
    vs = pd.DataFrame(v, index=days).rolling(14, min_periods=7).mean().shift(1).to_numpy()
    # Parkinson / Corwin–Schultz 简化的区间价差代理:β = (ln(h/l))² 两根 bar 之和,γ = (ln(max h / min l))²
    hl = np.log(h / l) ** 2
    out = {k_: np.full(n, np.nan) for k_ in ("bvc_ofi5", "bvc_ofi15", "bvc_ofi30", "kyle15", "amihud15", "cs_spread15", "bo_vol_share", "turn_rel15")}
    for j in range(n):
        i, kk = ii[j], k[j]
        for w, key in ((5, "bvc_ofi5"), (15, "bvc_ofi15"), (30, "bvc_ofi30")):
            a = max(kk - w + 1, 0); den = np.nansum(v[i, a:kk + 1])
            out[key][j] = s[j] * np.nansum(sgn_vol[i, a:kk + 1]) / den if den > 0 else np.nan
        a = max(kk - 14, 0)
        dv = np.nansum(np.abs(sgn_vol[i, a:kk + 1])); out["kyle15"][j] = s[j] * np.nansum(r1[i, a:kk + 1]) / max(np.nansum(v[i, a:kk + 1]), 1.0) * 1e6
        out["amihud15"][j] = np.log(np.nansum(np.abs(r1[i, a:kk + 1])) / max(np.nansum(turnover[i, a:kk + 1]), 1.0) * 1e9 + 1e-12)
        if kk - a >= 3:
            beta = np.nanmean(hl[i, a + 1:kk + 1] + hl[i, a:kk])
            gam = np.log(np.nanmax(h[i, a:kk + 1]) / np.nanmin(l[i, a:kk + 1])) ** 2
            alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / (3 - 2 * np.sqrt(2)) - np.sqrt(gam / (3 - 2 * np.sqrt(2)))
            out["cs_spread15"][j] = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
        out["bo_vol_share"][j] = v[i, kk] / max(np.nansum(v[i, :kk + 1]), 1.0)
        out["turn_rel15"][j] = np.log(max(np.nansum(v[i, a:kk + 1]), 1.0) / max(np.nansum(vs[i, a:kk + 1]), 1.0)) if np.isfinite(vs[i, a:kk + 1]).all() else np.nan
    return pd.DataFrame(out)


def bespoke_features(b, TR, r1):
    """Patton–Zhang 的结论:预测里 越靠近决策时点的 bar 权重越高。这里取近端 RV 份额,相对季节期望(PIT)。"""
    days = b.days; dpos = {d: i for i, d in enumerate(days)}
    ii = TR.day.map(dpos).to_numpy(); k = TR.k_dec.to_numpy(); n = len(TR)
    r2 = np.nan_to_num(r1 ** 2)
    cum = np.cumsum(r2, axis=1)
    # 季节期望:过去 20 日、同一 k 的"近端 30 分钟 RV 占截至 k 的 RV"的均值(PIT:只用 < i 的日子)
    out = {"rv_near_share": np.full(n, np.nan), "rv_near_ratio": np.full(n, np.nan), "rv_late_w": np.full(n, np.nan)}
    for j in range(n):
        i, kk = ii[j], k[j]
        tot = cum[i, kk]; near = cum[i, kk] - (cum[i, kk - 30] if kk >= 30 else 0.0)
        share = near / tot if tot > 0 else np.nan
        lo = max(0, i - 20)
        if i - lo >= 10:
            t_ = cum[lo:i, kk]; nr = t_ - (cum[lo:i, kk - 30] if kk >= 30 else 0.0)
            exp = np.nanmean(np.where(t_ > 0, nr / t_, np.nan))
            out["rv_near_ratio"][j] = np.log(share / exp) if share > 0 and exp > 0 else np.nan
        out["rv_near_share"][j] = share
        # 时间加权(越新权重越高,线性)的 RV 相对等权 RV 的比值:>1 表示波动集中在最近
        w = np.linspace(0.2, 1.0, kk + 1); out["rv_late_w"][j] = np.log((r2[i, :kk + 1] * w).sum() / max(r2[i, :kk + 1].mean() * w.sum(), 1e-30) + 1e-12)
    return pd.DataFrame(out)


def build(b):
    days = b.days
    TR = pd.read_pickle(DATA / "_TR.pkl").sort_values(["day", "phase", "k_dec"]).reset_index(drop=True)
    Xd, Yd, A, Bc = LV.build(b); E = event_frame(days, b.last_idx.to_numpy())
    ypct = pd.Series(0.5, index=days)                                   # 2021 年前基线预测不存在 → 中性(无泄漏);本环境没有上一轮的 lowvol_study.pkl
    F1 = TF.entry_features(b, TR, Xd, E, ypct)
    sd_in, r1 = intraday_sd(b)
    ev = DE.day_events(b, sd_in, K_EV); season = DE.seasonal_duration(ev, K_EV)
    dpos = {d: i for i, d in enumerate(days)}; ii = TR.day.map(dpos).to_numpy()
    DUR = DE.trade_features(ev, season, K_EV, ii, TR.k_dec.to_numpy(), TR.side.to_numpy().astype(float))
    MIC = micro_features(b, TR, r1, sd_in); BRP = bespoke_features(b, TR, r1)
    return TR, F1, DUR, MIC, BRP, Xd


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--build", action="store_true"); a = ap.parse_args()
    b = load_bars("QQQ")
    TR, F1, DUR, MIC, BRP, Xd = build(b)
    ALL = pd.concat([F1, DUR, MIC, BRP], axis=1); ALL.to_pickle(DATA / "_B0_features.pkl")
    print("特征表", ALL.shape, "缺失率最高的 6 列:", ALL.isna().mean().sort_values(ascending=False).head(6).round(3).to_dict())
    # ---- 自检 1:未来投毒(抽 12 笔:该笔 k_dec 之后的 bar ×7.3,新特征须逐位不变)
    import dataclasses
    rg = np.random.default_rng(0); idx = rg.choice(np.arange(len(TR)), 12, replace=False); worst = 0.0
    dpos = {d: i for i, d in enumerate(b.days)}
    for j in idx:
        i, kd = dpos[TR.day.iloc[j]], int(TR.k_dec.iloc[j])
        def P(df):
            d_ = df.copy(); d_.iloc[i, kd + 1:] = d_.iloc[i, kd + 1:] * 7.3; return d_
        dc = b.day_close.copy(); dc.iloc[i] = dc.iloc[i] * 7.3
        bp = dataclasses.replace(b, o=P(b.o), h=P(b.h), l=P(b.l), c=P(b.c), v=P(b.v), day_close=dc)
        sd_in, r1 = intraday_sd(bp)                                    # σ 取决于 t−1 及更早 → 不应变
        ev = DE.day_events(bp, sd_in, K_EV); season = DE.seasonal_duration(ev, K_EV)
        sub = TR.iloc[[j]].reset_index(drop=True)
        d1 = DE.trade_features(ev, season, K_EV, np.array([i]), sub.k_dec.to_numpy(), sub.side.to_numpy().astype(float))
        m1 = micro_features(bp, sub, r1, sd_in); b1 = bespoke_features(bp, sub, r1)
        new = pd.concat([d1, m1, b1], axis=1).iloc[0]; old = ALL.loc[j, new.index]
        diff = np.nanmax(np.abs(new.to_numpy(float) - old.to_numpy(float))); worst = max(worst, diff)
    print(f"[投毒自检] 12 笔:k_dec 之后 bar ×7.3,新特征最大差 = {worst:.2e}(须为 0)"); assert worst == 0.0
    # ---- 自检 2:阳性对照(加入入场后 5 分钟收益,IC 必须暴增)
    c = b.c.to_numpy(); ii = TR.day.map(dpos).to_numpy(); kd = TR.k_dec.to_numpy()
    fut5 = TR.side.to_numpy() * np.log(c[ii, np.minimum(kd + 6, 389)] / c[ii, kd + 1]) * 1e4
    print(f"[阳性对照] 入场后 5 分钟收益 与 y 的 Spearman = {spearmanr(fut5, TR.y, nan_policy='omit')[0]:.3f}(须≫ 任何诚实特征)")


if __name__ == "__main__":
    main()
