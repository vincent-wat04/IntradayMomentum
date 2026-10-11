"""多空双向 vs 只做多：空头腿到底贡献了什么。

当前实盘跑的是**多空双向**（突破上轨做多、跌破下轨做空、收盘平仓）。
这个脚本回答三件事：
  1. 多空两侧的交易笔数、持仓时间、盈亏归因
  2. 只做多（longonly）相比多空双向差多少
  3. 空头腿在哪些年份起作用（凸性是不是主要来自空头）

全部用最诚实的口径：TQQQ 成本 + 30 相位均值。
"""
from __future__ import annotations

import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from etft.intraday.bars import load_bars
from etft.intraday.strategy import NoiseCfg, run, buy_hold

OUT = Path("outputs/full/intraday")
PER_SHARE = 0.005 + 0.0045
TEST, ALL = ("2023", "2026"), ("2019", "2026")


def checks_at(off):
    base = [(h, m) for h in range(10, 16) for m in (0, 30)][:-1]
    return [(pd.Timestamp(f"2000-01-01 {h:02d}:{m:02d}") +
             pd.Timedelta(minutes=off)).strftime("%H:%M") for h, m in base]


def met(x, bench=None):
    x = pd.Series(x).dropna()
    if len(x) < 60 or x.std() == 0:
        return dict(sharpe=np.nan, ann=np.nan, mdd=np.nan)
    eq = (1 + x).cumprod(); dd = eq / eq.cummax() - 1
    ann_ = float(eq.iloc[-1] ** (252 / len(x)) - 1)
    o = dict(sharpe=float(x.mean() / x.std() * np.sqrt(252)), ann=ann_,
             vol=float(x.std() * np.sqrt(252)), mdd=float(dd.min()),
             calmar=ann_ / abs(dd.min()) if dd.min() < 0 else np.nan)
    if bench is not None:
        b = pd.Series(bench).reindex(x.index).fillna(0)
        if b.std() > 0:
            o["beta"] = float(np.cov(x, b)[0, 1] / b.var())
            o["alpha"] = float((x - o["beta"] * b).mean() * 252)
    return o


def main():
    b = load_bars("QQQ")
    days = b.days
    bh = buy_hold(b)
    pt = yf.download("TQQQ", start="2018-10-01", end="2026-10-05", auto_adjust=False,
                     progress=False, threads=False)["Close"]
    pt = pt.iloc[:, 0] if hasattr(pt, "columns") else pt
    pt.index = pd.to_datetime(pt.index).tz_localize(None)
    sp = yf.Ticker("TQQQ").splits; sp.index = pd.to_datetime(sp.index).tz_localize(None)
    f = pd.Series(1.0, index=pt.index)
    for d, r in sp.items():
        f[pt.index < d] *= r
    PT = (pt * f).reindex(days).ffill()
    rate = (PER_SHARE / PT) / 3.0            # TQQQ 口径

    SOTA = dict(stop="vwap", sizing="voltarget", max_lev=4.0)

    # ---------- 1. 交易归因（论文相位，带 trade 明细）
    r = run(b, NoiseCfg(cost_bp=1.0, record_trades=True, **SOTA))
    T = r.attrs["trades"]
    T["pnl"] = T["pnl"].astype(float)
    nl, ns = int((T.dir > 0).sum()), int((T.dir < 0).sum())
    print("=" * 92)
    print("1. 多空两侧的交易归因（论文相位，毛收益，2019-2026）")
    print("=" * 92)
    print(f"  交易笔数：做多 {nl}，做空 {ns}（空头占 {ns/(nl+ns):.1%}）")
    for side, lab in ((1, "做多"), (-1, "做空")):
        s = T[np.sign(T.dir) == side]
        print(f"  {lab}：{len(s):4d} 笔  胜率 {(s.pnl>0).mean():.1%}  "
              f"平均 {s.pnl.mean()*1e4:+7.2f}bp  合计 {s.pnl.sum()*100:+7.2f}%  "
              f"最好 {s.pnl.max()*100:+.2f}%  最差 {s.pnl.min()*100:+.2f}%")
    print(f"  毛收益合计：做多 {T[T.dir>0].pnl.sum()*100:+.1f}%，"
          f"做空 {T[T.dir<0].pnl.sum()*100:+.1f}%  "
          f"-> 空头贡献占 {T[T.dir<0].pnl.sum()/T.pnl.sum():.1%}")

    # 持仓时间分布
    lt, st = r["long_trips"].sum(), r["short_trips"].sum()
    print(f"  有仓位的交易日占比 {(r['avg_abs_pos']>0).mean():.1%}；"
          f"日均绝对仓位 {r['avg_abs_pos'].mean():.3f}（时间加权）")
    print(f"  多头开仓次数 {lt:.0f}，空头开仓次数 {st:.0f}")

    # ---------- 2. 多空双向 vs 只做多（30 相位均值 + TQQQ 成本）
    print("\n" + "=" * 92)
    print("2. 多空双向 vs 只做多（30 相位均值，TQQQ 成本）")
    print("=" * 92)
    nets = {}
    for mode in ("longshort", "longonly"):
        acc = []
        for off in range(30):
            rr = run(b, NoiseCfg(check_times=checks_at(off), cost_bp=1.0, mode=mode, **SOTA))
            acc.append(rr["overlay_gross"] - rr["cost"] * 1e4 * rate)
        nets[mode] = pd.DataFrame(acc).T.mean(axis=1)
        # 论文相位单独
        rr0 = run(b, NoiseCfg(cost_bp=1.0, mode=mode, **SOTA))
        nets[mode + "_p0"] = rr0["overlay_gross"] - rr0["cost"] * 1e4 * rate

    rows = []
    for key, lab in (("longshort", "多空双向 / 相位均值"), ("longshort_p0", "多空双向 / 论文相位"),
                     ("longonly", "只做多 / 相位均值"), ("longonly_p0", "只做多 / 论文相位")):
        for per, plab in ((TEST, "test23-26"), (ALL, "all19-26")):
            rows.append(dict(口径=lab, 期间=plab,
                             **met(nets[key].loc[per[0]:per[1]], bh.loc[per[0]:per[1]])))
    R = pd.DataFrame(rows)
    print(R[["口径", "期间", "sharpe", "ann", "vol", "mdd", "beta", "alpha"]].round(3)
          .to_string(index=False))

    # ---------- 3. 逐年
    print("\n" + "=" * 92)
    print("3. 逐年（相位均值 + TQQQ 成本）：空头腿在哪些年份起作用")
    print("=" * 92)
    yr = []
    for y in range(2019, 2027):
        lo, hi = f"{y}-01-01", f"{y}-12-31"
        if len(bh.loc[lo:hi].dropna()) < 60:
            continue
        ls, lo_ = nets["longshort"].loc[lo:hi], nets["longonly"].loc[lo:hi]
        yr.append(dict(年=y, 纯持有QQQ=met(bh.loc[lo:hi])["sharpe"],
                       多空双向=met(ls)["sharpe"], 只做多=met(lo_)["sharpe"],
                       多空年化=met(ls)["ann"], 只做多年化=met(lo_)["ann"],
                       空头增量=met(ls)["sharpe"] - met(lo_)["sharpe"]))
    Y = pd.DataFrame(yr)
    print(Y.round(3).to_string(index=False))

    # ---------- 4. 在指数下跌/上涨日的表现
    print("\n" + "=" * 92)
    print("4. 按指数涨跌分组（test，相位均值+TQQQ 成本，单位 bp/日）")
    print("=" * 92)
    s = nets["longshort"].loc[TEST[0]:TEST[1]].dropna()
    s2 = nets["longonly"].loc[TEST[0]:TEST[1]].reindex(s.index)
    bb = bh.reindex(s.index)
    g = pd.qcut(bb, 5, labels=["Q1 跌最多", "Q2", "Q3", "Q4", "Q5 涨最多"])
    t = pd.DataFrame({"多空": s * 1e4, "只做多": s2 * 1e4, "指数": bb * 100}).groupby(
        g, observed=True).mean()
    t["空头增量"] = t["多空"] - t["只做多"]
    print(t.round(2).to_string())

    R.to_csv(OUT / "long_short_decomposition.csv", index=False)
    Y.to_csv(OUT / "long_short_by_year.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.4))
    ax = axes[0]
    for k, lab in (("longshort", "多空双向"), ("longonly", "只做多")):
        ss = nets[k].loc[TEST[0]:TEST[1]]
        ax.plot((1 + ss).cumprod(), lw=1.4, label=f"{lab}: SR {met(ss)['sharpe']:.2f}")
    ax.set_title("test 期（相位均值 + TQQQ 成本）"); ax.legend(fontsize=9); ax.grid(alpha=.3)
    ax = axes[1]
    for k, lab in (("longshort", "多空双向"), ("longonly", "只做多")):
        ss = nets[k]
        ax.plot((1 + ss).cumprod(), lw=1.4, label=f"{lab}: SR {met(ss)['sharpe']:.2f}")
    ax.set_yscale("log"); ax.set_title("全样本 2019-2026")
    ax.legend(fontsize=9); ax.grid(alpha=.3)
    ax = axes[2]
    x = np.arange(5); w = 0.35
    ax.bar(x - w / 2, t["多空"].to_numpy(), width=w, label="多空双向")
    ax.bar(x + w / 2, t["只做多"].to_numpy(), width=w, label="只做多")
    ax.set_xticks(x); ax.set_xticklabels(t.index, fontsize=8)
    ax.axhline(0, color="k", lw=.8); ax.set_ylabel("日均收益 (bp)")
    ax.set_title("按指数涨跌分组：空头腿在跌的日子救场")
    ax.legend(fontsize=9); ax.grid(alpha=.3, axis="y")
    plt.tight_layout(); plt.savefig(OUT / "fig_long_short.png", dpi=110); plt.close()
    print("\nfigure written")


if __name__ == "__main__":
    main()
