"""改进检验第三轮：排除周一的置换检验、fast alpha 执行层、微观结构确认。

前两轮的结论：
  - NR4 置换 p=0.131（test）/ 0.339（dev），**不显著**；而且摊薄回全期后 0.889 < 基线 0.962
  - 论文的 VM=1.5 不复现
  - 不对称止损不显著
  - 存活的候选只有两个：**只做多**（dev 1.007 / test 1.392）和 **排除周一**（dev 0.726 / test 1.264）
本轮：
  1. 排除周一的置换检验 —— 764/940 天的子集，零分布应该窄得多
  2. fast alpha 执行层（Zarattini & Pagani 2026）：触发后等第一根**逆向** bar 再成交
  3. 微观结构确认：相对成交量（订单流强度代理）作为突破的确认条件
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
                ann=float(eq.iloc[-1] ** (252 / len(x)) - 1), mdd=float(dd.min()),
                n=int(len(x)), 日均bp=float(x.mean() * 1e4),
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

    def run_reimpl(**kw):
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

    def run_engine(**kw):
        """引擎版（支持 rvol 微观结构过滤、执行延迟）。成本用 TQQQ 逐日费率。"""
        rate = (0.0095 / PT) / 3.0
        acc = []
        for off in range(NPH):
            r = run(b, NoiseCfg(check_times=chk(off), cost_bp=1.0, stop="vwap",
                                sizing="voltarget", max_lev=4.0, **kw))
            acc.append(r["overlay_gross"] - r["cost"] * 1e4 * rate)
        return pd.DataFrame(acc).T.mean(axis=1)

    base = run_reimpl()
    base_e = run_engine()
    print(f"基线(reimpl, hold 语义)：dev {met(base.loc[DEV[0]:DEV[1]])['sharpe']:+.3f}  "
          f"test {met(base.loc[TEST[0]:TEST[1]])['sharpe']:+.3f}")
    print(f"基线(engine, same_bar 语义)：dev {met(base_e.loc[DEV[0]:DEV[1]])['sharpe']:+.3f}  "
          f"test {met(base_e.loc[TEST[0]:TEST[1]])['sharpe']:+.3f}")

    # ============ 1. 排除周一的置换检验
    print("\n" + "=" * 100)
    print("1. 排除周一：置换检验（随机排除同样多的天数 2000 次）")
    print("=" * 100)
    dow = pd.Series(base.index.dayofweek, index=base.index)
    rg = np.random.default_rng(7)
    for per, pl in ((DEV, "dev"), (TEST, "test")):
        s_all = base.loc[per[0]:per[1]].dropna()
        m = (dow.reindex(s_all.index) != 0)
        real = met(s_all[m]); k = int((~m).sum())
        nulls = []
        for _ in range(2000):
            drop = rg.choice(len(s_all), size=k, replace=False)
            keep = np.setdiff1d(np.arange(len(s_all)), drop)
            nulls.append(met(s_all.iloc[keep])["sharpe"])
        nulls = np.array(nulls); nulls = nulls[np.isfinite(nulls)]
        p = float((nulls >= real["sharpe"]).mean())
        print(f"  {pl:5s} 排除 {k} 天(周一)，保留 {int(m.sum())} 天  真实 {real['sharpe']:+.3f}  "
              f"零分布 均值 {nulls.mean():+.3f} std {nulls.std():.3f} "
              f"p95 {np.percentile(nulls,95):+.3f}  **p={p:.4f}**")
    # 周一本身的收益
    for per, pl in ((DEV, "dev"), (TEST, "test")):
        s = base.loc[per[0]:per[1]].dropna()
        mon = s[dow.reindex(s.index) == 0]
        print(f"  {pl:5s} 周一单独：{len(mon)} 天，日均 {mon.mean()*1e4:+.2f}bp，"
              f"t={mon.mean()/mon.std()*np.sqrt(len(mon)):+.2f}")

    # ============ 2. fast alpha 执行层
    print("\n" + "=" * 100)
    print("2. fast alpha 执行层（Zarattini & Pagani 2026）：触发后等第一根**逆向** bar 再成交")
    print("=" * 100)
    print("   当前 SOTA **没有**用这个（fast_alpha.py 只实现在 ATR 突破基线上，没接到噪声区域）")
    print("   用引擎的执行延迟参数做近似：delay_fixed 固定推迟 N 分钟后成交")
    for dly in (0, 1, 2, 3, 5):
        s = run_engine(delay_fixed=dly)
        md, mt = met(s.loc[DEV[0]:DEV[1]]), met(s.loc[TEST[0]:TEST[1]])
        print(f"   推迟 {dly} 分钟成交：dev {md['sharpe']:+.3f}  test {mt['sharpe']:+.3f}  "
              f"test年化 {mt['ann']:+.2%}")
    print("   （E26 当年的结论是所有延迟都更差；这里复核，顺便看 fast alpha 的方向是否值得实现）")
    # TWAP 拆单
    for n in (1, 3, 5):
        s = run_engine(twap_n=n)
        mt = met(s.loc[TEST[0]:TEST[1]])
        print(f"   拆成 {n} 分钟 TWAP：dev {met(s.loc[DEV[0]:DEV[1]])['sharpe']:+.3f}  "
              f"test {mt['sharpe']:+.3f}")

    # ============ 3. 微观结构确认
    print("\n" + "=" * 100)
    print("3. 微观结构确认：突破必须被成交量/资金流确认")
    print("=" * 100)
    print("   rvol = 当日累计成交量 / 过去 14 天同时点累计成交量均值（严格 PIT）")
    for lo in (None, 0.8, 1.0, 1.2, 1.5):
        kw = {} if lo is None else dict(rvol_min=lo)
        s = run_engine(**kw)
        md, mt = met(s.loc[DEV[0]:DEV[1]]), met(s.loc[TEST[0]:TEST[1]])
        nm = "无过滤" if lo is None else f"rvol >= {lo}"
        print(f"   {nm:14s} dev {md['sharpe']:+.3f}  test {mt['sharpe']:+.3f}  "
              f"test年化 {mt['ann']:+.2%}")
    for hi in (0.8, 1.0, 1.2):
        s = run_engine(rvol_max=hi)
        print(f"   rvol <  {hi:<9} dev {met(s.loc[DEV[0]:DEV[1]])['sharpe']:+.3f}  "
              f"test {met(s.loc[TEST[0]:TEST[1]])['sharpe']:+.3f}   （反向对照）")
    s = run_engine(rvol_size=True)
    print(f"   按 rvol 调仓位    dev {met(s.loc[DEV[0]:DEV[1]])['sharpe']:+.3f}  "
          f"test {met(s.loc[TEST[0]:TEST[1]])['sharpe']:+.3f}")

    # ============ 4. 存活候选的组合
    print("\n" + "=" * 100)
    print("4. 存活候选的组合（只做多 / 排除周一），以及作为 B&H 叠加层的效果")
    print("=" * 100)
    lo_only = run_reimpl(vm_short=99.0)
    combos = {
        "多空（当前实盘）": base,
        "只做多": lo_only,
        "多空 + 排除周一": base.where(dow != 0, 0.0),
        "只做多 + 排除周一": lo_only.where(dow != 0, 0.0),
    }
    m0t = met(bh.loc[TEST[0]:TEST[1]])
    print(f"   纯持有 QQQ test Sharpe {m0t['sharpe']:.3f}  回撤 {m0t['mdd']:.1%}\n")
    for nm, s in combos.items():
        md, mt = met(s.loc[DEV[0]:DEV[1]]), met(s.loc[TEST[0]:TEST[1]])
        ov = met((bh + s.reindex(bh.index).fillna(0)).loc[TEST[0]:TEST[1]])
        print(f"   {nm:18s} 单独 dev {md['sharpe']:+.3f} test {mt['sharpe']:+.3f} "
              f"年化 {mt['ann']:+.2%} 回撤 {mt['mdd']:+.1%}  |  "
              f"+1.0x叠加 test {ov['sharpe']:.3f} ({ov['sharpe']-m0t['sharpe']:+.3f}) "
              f"回撤 {ov['mdd']:.1%}")
    print("\n   2022 熊市（叠加层的意义所在）：")
    for nm, s in combos.items():
        o = met((bh + s.reindex(bh.index).fillna(0)).loc["2022-01-01":"2022-12-31"])
        print(f"   {nm:18s} +1.0x叠加 Sharpe {o['sharpe']:+.3f} 年化 {o['ann']:+.1%} "
              f"回撤 {o['mdd']:.1%}")
    print(f"   {'纯持有':18s} Sharpe {met(bh.loc['2022-01-01':'2022-12-31'])['sharpe']:+.3f} "
          f"年化 {met(bh.loc['2022-01-01':'2022-12-31'])['ann']:+.1%}")

    pd.to_pickle(combos, OUT / "improvement3_curves.pkl")
    print(f"\n已写入 {OUT}/improvement3_curves.pkl")


if __name__ == "__main__":
    main()
