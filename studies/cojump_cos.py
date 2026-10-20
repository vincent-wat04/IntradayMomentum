"""OHLC-COS 共同跳跃检测器(按微信文章文字重建,**不是**论文代码)。

来源:Lantos, Nolte, Nolte & Pham,《Detecting Co-jumps with Cosine Similarity of Candlestick Features》
(Lancaster 工作论文;原文未获取,仅依据中文解读文章。公式在文章里丢失,下面是依据文字与数值例子的还原,
凡是推断之处都在注释里标 [还原]。)

流程(每根 bar 只用截至当前已发生的数据 -> PIT):
  1. 特征 f = (body², body/range, 上比例, 下比例),body = ln(c/o),range = ln(h/l)。
     [还原] 上比例 = |c−o| / (h − min(o,c)),下比例 = |c−o| / (max(o,c) − l):影线消失时都 →1。
  2. 阈值筛选:候选 = 任一资产 r² > 3·min(RV_w, BV_w)/n_w(窗口 w = 前 390 根 bar,**不含当前**;BV = π/2·Σ|r_t||r_{t−1}|)。
     r 取 bar 内收益 ln(c/o)。阈值约等于"单 bar 收益超过约 1.73 倍典型 bar 波动"。
  3. 逐特征 z 分数标准化(窗口 390 根 bar,含当前)。
  4. 余弦相似度 cos(zA, zB);检验统计量 T = (cos+1)/2;原假设下 T ~ Beta((d−1)/2, (d−1)/2)(d=4 ⇒ Beta(1.5,1.5))。
  5. 判定两种临界值:
       nominal  —— 直接用 Beta(1.5,1.5) 的 1−α 分位;
       fitted   —— [还原] 用窗口内**未入选**的 bar 的 T 做矩估计 Beta(a,b) 校准,再取 1−α 分位。
     候选且 T 超过临界值 ⇒ 标记为共同跳跃。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

W = 390          # 阈值与标准化的滚动窗口(根 bar);文章:6.5 小时
FEATS = ("b2", "ratio", "up", "low")


def long_panel(b, valid=None) -> pd.DataFrame:
    """Bars -> 只含有效 bar 的长表(跨日连续),列 o,h,l,c,day,clock。"""
    o, h, l, c = b.o, b.h, b.l, b.c
    m = c.notna() & o.notna() & h.notna() & l.notna()
    if valid is not None:
        m = m & valid
    st = lambda X: X.where(m).stack()
    df = pd.DataFrame({"o": st(o), "h": st(h), "l": st(l), "c": st(c)})
    df.index.names = ["day", "clock"]
    return df.reset_index()


def features(df: pd.DataFrame) -> pd.DataFrame:
    o, h, l, c = (df[k].to_numpy(float) for k in "ohlc")
    body = np.log(c / o)
    rng = np.log(h / l)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(rng > 0, body / rng, 0.0)
        pb = np.abs(c - o)
        up = np.where((h - np.minimum(o, c)) > 0, pb / (h - np.minimum(o, c)), 0.0)
        low = np.where((np.maximum(o, c) - l) > 0, pb / (np.maximum(o, c) - l), 0.0)
    return pd.DataFrame({"r": body, "b2": body ** 2, "ratio": ratio, "up": up, "low": low})


def candidate_threshold(r: pd.Series, w: int = W) -> pd.Series:
    """r² 的阈值:3·min(RV, BV)/w,窗口为 t−1 之前的 w 根 bar(不含当前)。"""
    rv = (r ** 2).rolling(w, min_periods=w).sum()
    bv = (np.pi / 2) * (r.abs() * r.abs().shift(1)).rolling(w, min_periods=w).sum()
    return (3.0 * np.minimum(rv, bv) / w).shift(1)


def zscore(F: pd.DataFrame, w: int = W) -> pd.DataFrame:
    mu = F[list(FEATS)].rolling(w, min_periods=w).mean()
    sd = F[list(FEATS)].rolling(w, min_periods=w).std()
    return (F[list(FEATS)] - mu) / sd.replace(0, np.nan)


def cosine_T(zA: pd.DataFrame, zB: pd.DataFrame) -> pd.Series:
    a, b = zA.to_numpy(), zB.to_numpy()
    num = (a * b).sum(1)
    den = np.sqrt((a * a).sum(1)) * np.sqrt((b * b).sum(1))
    with np.errstate(divide="ignore", invalid="ignore"):
        cs = np.where(den > 0, num / den, np.nan)
    return pd.Series((cs + 1.0) / 2.0, index=zA.index)


def fitted_beta_quantile(T: pd.Series, cand: pd.Series, alpha: float, w: int = W) -> pd.Series:
    """窗口内未入选 bar 的 T 做矩估计 -> Beta(a,b) 的 1−α 分位。窗口不含当前 bar。"""
    non = (~cand) & T.notna()
    x = T.where(non)
    n = non.astype(float).rolling(w, min_periods=1).sum().shift(1)
    m = x.rolling(w, min_periods=1).mean().shift(1)
    s2 = (x ** 2).rolling(w, min_periods=1).mean().shift(1)
    v = (s2 - m ** 2).clip(lower=1e-6)
    k = (m * (1 - m) / v - 1).clip(lower=0.5)
    a, bb = (m * k).to_numpy(), ((1 - m) * k).to_numpy()
    q = np.full(len(T), np.nan)
    ok = (n.to_numpy() >= 100) & np.isfinite(a) & np.isfinite(bb)
    q[ok] = stats.beta.ppf(1 - alpha, a[ok], bb[ok])
    return pd.Series(q, index=T.index)


def detect(dfA: pd.DataFrame, dfB: pd.DataFrame, alpha: float = 0.01, w: int = W) -> pd.DataFrame:
    """dfA/dfB:同一时间网格的长表(行一一对应)。返回逐 bar 的 cand、T、两种判定。"""
    FA, FB = features(dfA), features(dfB)
    candA = FA["r"] ** 2 > candidate_threshold(FA["r"], w)
    candB = FB["r"] ** 2 > candidate_threshold(FB["r"], w)
    cand = (candA | candB).fillna(False)
    T = cosine_T(zscore(FA, w), zscore(FB, w))
    q_nom = float(stats.beta.ppf(1 - alpha, (len(FEATS) - 1) / 2, (len(FEATS) - 1) / 2))
    q_fit = fitted_beta_quantile(T, cand, alpha, w)
    out = pd.DataFrame({"day": dfA["day"].to_numpy(), "clock": dfA["clock"].to_numpy(), "cand": cand.to_numpy(),
                        "candA": candA.fillna(False).to_numpy(), "candB": candB.fillna(False).to_numpy(),
                        "T": T.to_numpy(), "rA": FA["r"].to_numpy(), "rB": FB["r"].to_numpy()})
    out["cj_nom"] = out["cand"] & (out["T"] > q_nom)
    out["cj_fit"] = out["cand"] & (out["T"] > q_fit.to_numpy())
    out["q_nom"] = q_nom
    return out
