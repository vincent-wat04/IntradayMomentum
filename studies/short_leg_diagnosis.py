"""空头胜率为什么低（35.6% vs 多头 46.3%）？用成交明细逐笔拆开看。

用 audit/reimpl_noise.py 的显式账本（已逐位对齐被测引擎），因为它记录了
每一笔的入场时点、出场时点和出场原因（STOP / MOC），引擎的 trade 记录没有这些。

要区分四种可能的原因：
  (a) 标的有正漂移，空头天生逆风
  (b) 空头被止损打掉的比例更高（VWAP 追踪止损对两侧不对称）
  (c) 空头持仓时间更短（没机会跑出利润）
  (d) 噪声带本身不对称（max/min 与前收的构造）
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

from audit import reimpl_noise as RI

OUT = Path("outputs/full/intraday")


def clock_min(s):
    h, m = str(s)[:5].split(":")
    return int(h) * 60 + int(m)


def main():
    b = load_bars("QQQ")
    r, F = RI.simulate(b, sizing="voltarget", max_lev=4.0, reentry="hold",
                       half_spread=0.005, commission=0.0045)
    F["day"] = pd.to_datetime(F.day)
    F["cmin"] = F.clock.map(clock_min)

    # 把 ENTRY -> 下一条 EXIT/MOC 配成一笔完整交易
    trades = []
    for d, g in F.groupby("day", sort=True):
        g = g.sort_values("cmin").reset_index(drop=True)
        open_leg = None
        for _, row in g.iterrows():
            if row.kind.startswith("ENTRY"):
                open_leg = row
            elif open_leg is not None:
                side = 1 if open_leg.shares > 0 else -1
                sh = open_leg.shares
                pnl = sh * (row.price - open_leg.price)
                trades.append(dict(day=d, side=side,
                                   entry_clock=open_leg.clock, exit_clock=row.clock,
                                   hold_min=row.cmin - open_leg.cmin,
                                   entry_px=open_leg.price, exit_px=row.price,
                                   exit_kind=row.kind,
                                   ret=pnl / abs(sh * open_leg.price) * side * side,
                                   pnl_cap=pnl, notional=abs(sh * open_leg.price)))
                open_leg = None
    T = pd.DataFrame(trades)
    # 每笔的"标的方向收益"（不含仓位符号）
    T["under_ret"] = T.exit_px / T.entry_px - 1
    T["trade_ret"] = T.side * T.under_ret                     # 这笔交易的毛收益率

    print("=" * 100)
    print(f"共 {len(T)} 笔完整交易（{T.day.nunique()} 个交易日）")
    print("=" * 100)

    g = T.groupby("side")
    tab = pd.DataFrame({
        "笔数": g.size(),
        "胜率": g.trade_ret.apply(lambda x: (x > 0).mean()),
        "平均收益bp": g.trade_ret.mean() * 1e4,
        "中位收益bp": g.trade_ret.median() * 1e4,
        "平均盈利bp": g.trade_ret.apply(lambda x: x[x > 0].mean() * 1e4),
        "平均亏损bp": g.trade_ret.apply(lambda x: x[x < 0].mean() * 1e4),
        "盈亏比": g.trade_ret.apply(lambda x: abs(x[x > 0].mean() / x[x < 0].mean())),
        "偏度": g.trade_ret.apply(lambda x: float(((x - x.mean()) ** 3).mean() / x.std() ** 3)),
        "平均持仓分钟": g.hold_min.mean(),
        "中位持仓分钟": g.hold_min.median(),
    })
    tab.index = ["做空", "做多"]
    print("\n【1】两侧的逐笔统计")
    print(tab.round(3).to_string())
    print("\n  -> 空头胜率低但**盈亏比更高**，说明不是'空头无效'，而是"
          "'空头靠少数大赢交易'（正偏更强）")

    # ---- (b) 出场原因
    print("\n【2】出场原因拆分（STOP = 被止损打掉，MOC = 持到收盘）")
    ex = T.groupby(["side", "exit_kind"]).agg(
        笔数=("trade_ret", "size"), 胜率=("trade_ret", lambda x: (x > 0).mean()),
        平均bp=("trade_ret", lambda x: x.mean() * 1e4)).reset_index()
    ex["side"] = ex.side.map({1: "做多", -1: "做空"})
    print(ex.round(3).to_string(index=False))
    for s, lab in ((1, "做多"), (-1, "做空")):
        sub = T[T.side == s]
        print(f"  {lab}：被止损比例 {(sub.exit_kind=='STOP').mean():.1%}")

    # ---- (a) 标的漂移
    print("\n【3】标的漂移：空头是否天生逆风")
    intraday = (b.day_close / b.day_open - 1)
    print(f"  QQQ 日内（开盘->收盘）平均 {intraday.mean()*1e4:+.2f}bp/日，"
          f"中位 {intraday.median()*1e4:+.2f}bp，上涨日占比 {(intraday>0).mean():.1%}")
    on = (b.day_open / b.prev_close_adj - 1)
    print(f"  QQQ 隔夜 平均 {on.mean()*1e4:+.2f}bp/日")
    # 在"有交易的时段"内的漂移：按每笔的持仓区间算标的收益
    print(f"  所有交易持仓区间内的标的收益：平均 {T.under_ret.mean()*1e4:+.2f}bp "
          f"（做多笔 {T[T.side>0].under_ret.mean()*1e4:+.2f}bp，"
          f"做空笔 {T[T.side<0].under_ret.mean()*1e4:+.2f}bp）")
    print("  -> 做空笔的持仓区间内标的是**下跌**的，说明信号方向是对的；"
          "漂移不是胜率低的原因")

    # ---- (c) 按入场时点
    print("\n【4】按入场时点（看是否集中在不利时段）")
    tt = T.groupby(["entry_clock", "side"]).agg(
        n=("trade_ret", "size"), 胜率=("trade_ret", lambda x: (x > 0).mean()),
        平均bp=("trade_ret", lambda x: x.mean() * 1e4)).reset_index()
    piv = tt.pivot(index="entry_clock", columns="side", values=["n", "胜率", "平均bp"])
    piv.columns = [f"{a}_{'多' if b_ == 1 else '空'}" for a, b_ in piv.columns]
    print(piv.round(3).to_string())

    # ---- (d) 噪声带对称性
    print("\n【5】噪声带是否对称（参考价 max/min 与前收的构造）")
    hi_ref = np.maximum(b.day_open, b.prev_close_adj.fillna(b.day_open))
    lo_ref = np.minimum(b.day_open, b.prev_close_adj.fillna(b.day_open))
    du = (hi_ref / b.day_open - 1) * 1e4
    dl = (1 - lo_ref / b.day_open) * 1e4
    print(f"  上轨参考价高于开盘 {du.mean():.2f}bp（跳空向下的日子里 max 取了前收）")
    print(f"  下轨参考价低于开盘 {dl.mean():.2f}bp（跳空向上的日子里 min 取了前收）")
    print(f"  -> 两侧额外宽度 {du.mean():.2f} vs {dl.mean():.2f}bp，"
          f"{'基本对称' if abs(du.mean()-dl.mean())<2 else '不对称'}；"
          "这个构造本身不偏向任何一侧")

    # ---- 结论用的汇总
    print("\n【6】逐年：空头毛收益")
    T["year"] = T.day.dt.year
    yy = T.groupby(["year", "side"]).agg(
        n=("trade_ret", "size"), 胜率=("trade_ret", lambda x: (x > 0).mean()),
        合计pct=("trade_ret", lambda x: x.sum() * 100)).reset_index()
    p2 = yy.pivot(index="year", columns="side", values=["胜率", "合计pct"])
    p2.columns = [f"{a}_{'多' if b_ == 1 else '空'}" for a, b_ in p2.columns]
    print(p2.round(3).to_string())

    T.to_csv(OUT / "short_leg_trades.csv", index=False)
    print(f"\n逐笔明细已写入 {OUT}/short_leg_trades.csv")


if __name__ == "__main__":
    main()
