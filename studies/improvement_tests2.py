"""改进检验第二轮：NR4 的置换检验、长期信号、fast alpha 执行层、微观结构确认。

第一轮（improvement_tests.py）的结论：
  - 不对称止损：dev 选出 band/vwap -> test +0.079，低于审计的 0.15 噪声护栏，不显著
  - VM：论文说 1.5 最优，我们 dev 说 0.75、test 说 1.0 —— **论文的 1.5 不复现**
  - 不对称 VM（空头要求更大突破）：dev +0.063 / test +0.240，方向一致但 dev 幅度小
  - **NR4 [PIT] 是唯一 dev(+0.31) 和 test(+0.79) 都改善的**，而且前视口径是强负，
    说明论文用的是 PIT 口径；我们的改善比值 1.82 vs 论文 1.88，复现得上
本轮要做的：
  1. NR4 的置换检验（随机挑同样多的天数）—— 它是真信号还是"选日子"的运气？
  2. 排除周一（论文 §4.3 说周一最弱，我们 dev/test 都弱）
  3. 长期信号条件化（200 日趋势、长期动量）
  4. fast alpha 执行层（Zarattini & Pagani 2026）—— 当前 SOTA **没有**用
  5. 微观结构确认（订单流失衡 / 相对成交量 / Amihud）—— 突破是否被资金流确认
"""
from __future__ import annotations

import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from etft.intraday.bars import load_bars
from etft.intraday.strategy import NoiseCfg, run, buy_hold

from audit import reimpl_noise as RI

OUT = Path("outputs/full/intraday")
DEV, TEST = ("2019", "2022"), ("2023", "2026")
NPH = 30


def chk(off):
    base = [(h, m) for h in range(10, 16) for m in (0, 30)][:-1]
    return [(pd.Timestamp(f"2000-01-01 {h:02d}:{m:02d}") +
             pd.Timedelta(minutes=off)).strftime("%H:%M") for h, m in base]


def met(x):
    x = pd.Series(x).dropna()
    if len(x) < 40 or x.std() == 0:
        return dict(sharpe=np.nan, ann=np.nan, n=len(x), 日均bp=np.nan)
    eq = (1 + x).cumprod(); dd = eq / eq.cummax() - 1
    return dict(sharpe=float(x.mean() / x.std() * np.sqrt(252)),
                ann=float(eq.iloc[-1] ** (252 / len(x)) - 1),
                mdd=float(dd.min()), n=int(len(x)), 日均bp=float(x.mean() * 1e4),
                t=float(x.mean() / x.std() * np.sqrt(len(x))))


def main():
    b = load_bars("QQQ"); days = b.days; bh = buy_hold(b)
    pt = yf.download("TQQQ", start="2018-10-01", end="2026-10-05", auto_adjust=False,
                     progress=False, threads=False)["Close"]
    pt = pt.iloc[:, 0] if hasattr(pt, "columns") else pt
    pt.index = pd.to_datetime(pt.index).tz_localize(None)
    sp = yf.Ticker("TQQQ").splits; sp.index = pd.to_datetime(sp.index).tz_localize(None)
    fsp = pd.Series(1.0, index=pt.index)
    for d, r in sp.items():
        fsp[pt.index < d] *= r
    PT = (pt * fsp).reindex(days).ffill()
    scale = (b.day_close / PT) / 3.0

    def run_cfg(**kw):
        acc = []
        for off in range(NPH):
            rr, F = RI.simulate(b, checks=chk(off), sizing="voltarget", max_lev=4.0,
                                reentry="hold", half_spread=0.005, commission=0.0045, **kw)
            if len(F):
                F = F.copy(); F["day"] = pd.to_datetime(F.day)
                c_old = F.groupby("day").commission.sum().reindex(days).fillna(0.0)
                rr = rr + c_old - c_old * scale.reindex(days).fillna(1.0)
            acc.append(rr)
        return pd.DataFrame(acc).T.mean(axis=1)

    base = run_cfg()
    bd, bt = met(base.loc[DEV[0]:DEV[1]]), met(base.loc[TEST[0]:TEST[1]])
    print(f"基线（30相位均值 + TQQQ 成本）：dev {bd['sharpe']:+.3f}  test {bt['sharpe']:+.3f}  "
          f"test 日均 {bt['日均bp']:+.2f}bp，{bt['n']} 天")

    # ================= 1. NR4 的置换检验
    print("\n" + "=" * 100)
    print("1. NR4 置换检验：随机挑同样多的天数 1000 次，看 1.751 是不是选日子的运气")
    print("=" * 100)
    H = b.h.max(axis=1); L = b.l.min(axis=1); rng = H - L
    nr4 = (rng == rng.rolling(4).min()).shift(1).fillna(False)
    nr7 = (rng == rng.rolling(7).min()).shift(1).fillna(False)
    rngm = pd.Series(np.nan, index=days)

    rngmask = {"NR4": nr4, "NR7": nr7, "NR4∪NR7": (nr4 | nr7)}
    rg = np.random.default_rng(42)
    for nm, mk in rngmask.items():
        for per, pl in ((DEV, "dev"), (TEST, "test")):
            s_all = base.loc[per[0]:per[1]].dropna()
            m = mk.reindex(s_all.index).fillna(False)
            real = met(s_all[m])
            k = int(m.sum())
            nulls = []
            for _ in range(1000):
                idx = rg.choice(len(s_all), size=k, replace=False)
                nulls.append(met(s_all.iloc[np.sort(idx)])["sharpe"])
            nulls = np.array(nulls); nulls = nulls[np.isfinite(nulls)]
            p = float((nulls >= real["sharpe"]).mean())
            print(f"  {nm:9s} {pl:5s} 天数 {k:4d}  真实 Sharpe {real['sharpe']:+.3f}  "
                  f"零分布 均值 {nulls.mean():+.3f} std {nulls.std():.3f} "
                  f"p95 {np.percentile(nulls,95):+.3f}  **p={p:.3f}**")

    # ================= 2. 排除周一 / 形态组合
    print("\n" + "=" * 100)
    print("2. 日级筛选的组合（都是事前可算的）")
    print("=" * 100)
    dow = pd.Series(base.index.dayofweek, index=base.index)
    cands = {
        "基线（全部交易日）": pd.Series(True, index=days),
        "排除周一": (dow != 0),
        "NR4": nr4,
        "NR4∪NR7": (nr4 | nr7),
        "NR4 且 非周一": nr4 & (dow != 0),
        "NR4∪NR7 且 非周一": (nr4 | nr7) & (dow != 0),
    }
    for nm, mk in cands.items():
        m = mk.reindex(days).fillna(False)
        s = base.where(m)
        md, mt = met(s.loc[DEV[0]:DEV[1]]), met(s.loc[TEST[0]:TEST[1]])
        # 换算成"同样资金、只在这些天交易"的全期年化（空仓日收益为 0）
        full = base.where(m, 0.0)
        mf = met(full.loc[TEST[0]:TEST[1]])
        print(f"  {nm:20s} test天数 {mt.get('n',0):4d}  dev {md.get('sharpe',np.nan):+.3f}  "
              f"test {mt.get('sharpe',np.nan):+.3f}  "
              f"交易日日均 {mt.get('日均bp',np.nan):+.2f}bp  "
              f"-> 全期摊薄后 Sharpe {mf['sharpe']:+.3f} 年化 {mf['ann']:+.2%}")

    # ================= 3. 长期信号条件化
    print("\n" + "=" * 100)
    print("3. 长期信号条件化（200日趋势 / 长期动量；全部 shift(1) 严格事前）")
    print("=" * 100)
    px = b.day_close
    ma200 = px.rolling(200).mean()
    above = (px > ma200).shift(1).fillna(False)
    mom12 = (px / px.shift(252) - 1).shift(1)
    vol60 = px.pct_change().rolling(60).std().shift(1)
    longs = {
        "价格在 200 日均线上方": above,
        "价格在 200 日均线下方": ~above & px.shift(1).notna(),
        "12 个月动量 > 0": (mom12 > 0).fillna(False),
        "12 个月动量 < 0": (mom12 < 0).fillna(False),
        "60 日波动在上半分位": (vol60 > vol60.rolling(252).median()).fillna(False),
        "60 日波动在下半分位": (vol60 <= vol60.rolling(252).median()).fillna(False),
    }
    for nm, mk in longs.items():
        m = mk.reindex(days).fillna(False)
        s = base.where(m)
        md, mt = met(s.loc[DEV[0]:DEV[1]]), met(s.loc[TEST[0]:TEST[1]])
        print(f"  {nm:24s} test天数 {mt.get('n',0):4d}  dev {md.get('sharpe',np.nan):+.3f}  "
              f"test {mt.get('sharpe',np.nan):+.3f}  test日均 {mt.get('日均bp',np.nan):+.2f}bp")

    # 只做与长期趋势同向的那一侧
    print("\n  只做与长期趋势同向的一侧（趋势上行只做多、下行只做空）：")
    for mode, nm in (("longonly", "只做多"), ("longshort", "多空")):
        pass
    accA = []
    for off in range(NPH):
        rl, Fl = RI.simulate(b, checks=chk(off), sizing="voltarget", max_lev=4.0,
                             reentry="hold", half_spread=0.005, commission=0.0045,
                             vm_short=99.0)     # 空头门槛设到极高 = 实际只做多
        accA.append(rl)
    lo_only = pd.DataFrame(accA).T.mean(axis=1)
    accB = []
    for off in range(NPH):
        rs, _ = RI.simulate(b, checks=chk(off), sizing="voltarget", max_lev=4.0,
                            reentry="hold", half_spread=0.005, commission=0.0045,
                            vm_long=99.0)       # 只做空
        accB.append(rs)
    sh_only = pd.DataFrame(accB).T.mean(axis=1)
    trend_cond = pd.Series(np.where(above.reindex(days).fillna(False),
                                    lo_only.reindex(days), sh_only.reindex(days)),
                           index=days)
    for nm, s in (("只做多", lo_only), ("只做空", sh_only),
                  ("趋势上行只做多 / 下行只做空", trend_cond), ("多空（基线）", base)):
        md, mt = met(s.loc[DEV[0]:DEV[1]]), met(s.loc[TEST[0]:TEST[1]])
        print(f"    {nm:26s} dev {md.get('sharpe',np.nan):+.3f}  "
              f"test {mt.get('sharpe',np.nan):+.3f}  test年化 {mt.get('ann',np.nan):+.2%}")

    pd.to_pickle(dict(base=base, nr4=nr4, nr7=nr7, lo_only=lo_only, sh_only=sh_only),
                 OUT / "improvement2_curves.pkl")
    print(f"\n已写入 {OUT}/improvement2_curves.pkl")


if __name__ == "__main__":
    main()
