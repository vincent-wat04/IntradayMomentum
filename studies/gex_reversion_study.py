"""GEX 的另一面：高 gamma 日跑**反向**策略，以及尾盘反转。

为什么换这个角度（文献驱动）：
  firmtape (2026) 在 1085 个 SPX 交易日上发现：
    - Gao(2018)/Baltussen(2021) 的"末半小时延续全日方向"在 0DTE 时代**消失**
      （全样本斜率 +0.006±0.009, t=+0.6，逐年符号交替）—— 独立印证了我们 PART8
      的 GAO(−1.203) 和 LAST30(−1.554)
    - 延续只活在**做市商短 gamma**的日子，但必须用 **逐笔主动方向签名**的期权账本；
      **公开未平仓量约定（= SqueezeMetrics GEX 的做法）完全看不到**
    - 而且短 gamma 收盘只占 15% 的交易日 -> 这一侧天然数据饥饿

  由此推出一个我们从没测过、而且**功率高 5.7 倍**的方向：
    正 gamma 占 85% 的交易日，而正 gamma 预测的是**均值回归**（做市商逆势对冲）。
    所以应该测"高 GEX 日的反向策略"，而不是"用 GEX 过滤动量策略"。

  另外 EFMA(2024) "End-of-Day Reversal" 报告尾盘反转效应，并明确说它
  **独立于日内动量与 gamma 对冲** —— 也是我们没测过的。

本脚本测：
  1. PART8 的三个反向策略（VWAP_REV / ON_REV / LUNCH_REV）在 GEX 分组下的表现
  2. 尾盘反转：全日涨跌 -> 末 30 分钟**反向**（LAST30 的反面）
  3. 以上都做置换检验（挑日子的零分布很宽，这是 PART11 §2.2 的教训）
全部用 t−1 的 GEX（合法最小滞后；gex_lag_study.py 已证明滞后越长越好是噪声形状）。
"""
from __future__ import annotations

import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from etft.intraday.bars import load_bars

from paths import ETFT
OUT = Path("outputs/full/intraday")
DEV, TEST = ("2019", "2022"), ("2023", "2026")


def sr(x):
    x = pd.Series(x).dropna()
    return float(x.mean() / x.std() * np.sqrt(252)) if len(x) > 30 and x.std() > 0 else np.nan


def perm_p(s, mask, nperm=1000, seed=0):
    s = pd.Series(s).dropna()
    mk = pd.Series(mask).reindex(s.index).fillna(False)
    k = int(mk.sum())
    if k < 50:
        return np.nan, np.nan, k, np.nan
    real = sr(s[mk])
    rg = np.random.default_rng(seed)
    nulls = []
    for _ in range(nperm):
        idx = np.sort(rg.choice(len(s), k, replace=False))
        nulls.append(sr(s.iloc[idx]))
    nulls = np.array([v for v in nulls if np.isfinite(v)])
    return real, float((nulls >= real).mean()), k, float(nulls.mean())


def main():
    b = load_bars("QQQ"); days = b.days
    g = pd.read_parquet(ETFT / "data/raw/squeeze_dix_gex.parquet")
    g.index = pd.to_datetime(g.index)
    GEX = g["gex"].reindex(days).ffill().shift(1)          # t−1，合法最小滞后
    GEXp = GEX.rolling(252, min_periods=120).apply(lambda x: (x[-1] > x[:-1]).mean(), raw=True)

    # ---------- 1. PART8 的反向策略 × GEX 分组
    print("=" * 112)
    print("1. 反向策略 × GEX 分组（PART8 的曲线，0.5bp/边；正 gamma 应有利于反向）")
    print("=" * 112)
    cv = pd.read_pickle(OUT / "intraday_curves.pkl")
    revs = [("VWAP_REV", "偏离VWAP回归"), ("ON_REV", "隔夜跳空反转"),
            ("LUNCH_REV", "午盘反转")]
    for key, nm in revs:
        s = cv.get(("QQQ", key))
        if s is None:
            continue
        print(f"\n--- {nm}（{key}）无条件 test Sharpe {sr(s.loc[TEST[0]:TEST[1]]):+.3f}")
        for per, pl in ((DEV, "dev"), (TEST, "test")):
            ss = s.loc[per[0]:per[1]].dropna()
            q = pd.qcut(GEXp.reindex(ss.index).dropna(), 4,
                        labels=["Q1低GEX", "Q2", "Q3", "Q4高GEX"])
            t = pd.DataFrame({"r": ss.reindex(q.index)}).groupby(q, observed=True).agg(
                天数=("r", "size"), 日均bp=("r", lambda x: x.mean() * 1e4),
                Sharpe=("r", sr))
            print(f"    {pl}:")
            print("      " + t.round(3).to_string().replace("\n", "\n      "))
        # 高 GEX 一半的置换检验
        for per, pl in ((DEV, "dev"), (TEST, "test")):
            ss = s.loc[per[0]:per[1]]
            real, p, k, nm_ = perm_p(ss, GEXp > 0.5, seed=hash(key) % 999)
            print(f"    {pl} 高GEX一半: {k} 天 Sharpe {real:+.3f} "
                  f"(零分布均值 {nm_:+.3f}, p={p:.3f})")

    # ---------- 2. 尾盘反转（EFMA 2024）
    print("\n" + "=" * 112)
    print("2. 尾盘反转：全日涨跌 -> 末 30 分钟**反向**（EFMA 2024 说独立于 gamma）")
    print("=" * 112)
    # 用 1 分钟 bar 直接构造：信号 = 开盘->15:30 的收益，仓位 = −sign(信号)，15:30 入场、收盘平仓
    O, C = b.o, b.c
    cols = list(C.columns)
    i1530 = cols.index("15:30") if "15:30" in cols else None
    if i1530 is None:
        print("   找不到 15:30 的 bar"); return
    px1530 = C["15:30"]
    entry = O[cols[i1530 + 1]] if i1530 + 1 < len(cols) else C["15:30"]
    r_day = px1530 / b.day_open - 1
    r_last = b.day_close / entry - 1
    cost = (0.005 + 0.0045) / b.day_close * 2        # 两条腿，每股口径
    for sgn, nm in ((-1, "反转（末30分钟反向）"), (+1, "延续（末30分钟同向，Gao 口径）")):
        ret = sgn * np.sign(r_day) * r_last - cost
        ret = ret.where(np.abs(r_day) > 1e-6)
        print(f"\n--- {nm}")
        for per, pl in ((DEV, "dev"), (TEST, "test"), (("2019", "2026"), "全样本")):
            x = ret.loc[per[0]:per[1]].dropna()
            print(f"    {pl:6s} {len(x):4d} 天  日均 {x.mean()*1e4:+7.2f}bp  "
                  f"Sharpe {sr(x):+6.3f}  "
                  f"t={x.mean()/x.std()*np.sqrt(len(x)):+5.2f}")
        # 斜率回归（和 firmtape 口径一致：rest-of-day 对 last-30min 回归）
        m = r_day.notna() & r_last.notna()
        A = np.column_stack([np.ones(m.sum()), r_day[m].to_numpy()])
        bta, *_ = np.linalg.lstsq(A, r_last[m].to_numpy(), rcond=None)
        res = r_last[m].to_numpy() - A @ bta
        se = np.sqrt(np.diag(res @ res / (m.sum() - 2) * np.linalg.pinv(A.T @ A)))
        if sgn == -1:
            print(f"    斜率回归 r_last = α + β·r_day：β={bta[1]:+.4f}  t={bta[1]/se[1]:+.2f}")
            print(f"    （firmtape 在 SPX 1085 天上报告 β=+0.006±0.009, t=+0.6）")

        # GEX 条件化
        print(f"    按 GEX 分组（test）：")
        x = ret.loc[TEST[0]:TEST[1]].dropna()
        q = pd.qcut(GEXp.reindex(x.index).dropna(), 4,
                    labels=["Q1低GEX", "Q2", "Q3", "Q4高GEX"])
        t = pd.DataFrame({"r": x.reindex(q.index)}).groupby(q, observed=True).agg(
            天数=("r", "size"), 日均bp=("r", lambda v: v.mean() * 1e4), Sharpe=("r", sr))
        print("      " + t.round(3).to_string().replace("\n", "\n      "))

    print("\n" + "=" * 112)
    print("3. 机制的直接检验：高 GEX 日的**已实现波动**是否真的更低")
    print("=" * 112)
    intra = (b.day_close / b.day_open - 1).abs()
    rng = (b.h.max(axis=1) - b.l.min(axis=1)) / b.day_open
    for nm, v in (("|开盘->收盘|", intra), ("日内振幅", rng)):
        for per, pl in ((DEV, "dev"), (TEST, "test")):
            x = v.loc[per[0]:per[1]]
            q = pd.qcut(GEXp.reindex(x.index).dropna(), 4,
                        labels=["Q1低", "Q2", "Q3", "Q4高"])
            t = x.reindex(q.index).groupby(q, observed=True).mean() * 1e4
            print(f"   {nm:12s} {pl:5s} " +
                  "  ".join(f"{a}:{v2:.0f}bp" for a, v2 in t.items()) +
                  f"   高/低比值 {t.iloc[-1]/t.iloc[0]:.3f}")
    print("\n   机制预测：正 gamma 抑制波动 -> 高 GEX 组的已实现波动应**更低**（比值<1）")


if __name__ == "__main__":
    main()
