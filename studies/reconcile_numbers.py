"""把所有口径的 Sharpe 对齐成一张账：到底哪个数字是什么意思。

起因：PART9 报告"组合 Sharpe 2.011"，PART10 报告"前瞻 Sharpe 0.3~0.9"，看起来矛盾。
实际上测的是两个不同的东西：
  - 2.011 = 买入持有 QQQ + 1.0x 日内叠加层   <- 里面绝大部分是 QQQ 自己的 beta
  - 0.3~0.9 = 日内叠加层**单独**，且扣掉相位运气、用 TQQQ 成本、考虑近两年衰减

本脚本把两者放进同一张表，并且把每一步修正的影响单独列出来，这样可以看清
"从 1.375 到 0.88"和"从 2.011 到 ?"分别发生了什么。
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

OUT = Path("outputs/full/intraday")
SOTA = dict(stop="vwap", sizing="voltarget", max_lev=4.0)
PER_SHARE = 0.005 + 0.0045
TEST, ALL = ("2023", "2026"), ("2019", "2026")


def checks_at(offset):
    base = [(h, m) for h in range(10, 16) for m in (0, 30)][:-1]
    return [(pd.Timestamp(f"2000-01-01 {h:02d}:{m:02d}") +
             pd.Timedelta(minutes=offset)).strftime("%H:%M") for h, m in base]


def met(x, bench=None):
    x = pd.Series(x).dropna()
    if len(x) < 60 or x.std() == 0:
        return dict(sharpe=np.nan, ann=np.nan, mdd=np.nan, calmar=np.nan)
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
    bh = buy_hold(b)                                    # 买入持有 QQQ 的日收益

    pq = yf.download("QQQ", start="2018-10-01", end="2026-10-05", auto_adjust=False,
                     progress=False, threads=False)["Close"]
    pq = pq.iloc[:, 0] if hasattr(pq, "columns") else pq
    pq.index = pd.to_datetime(pq.index).tz_localize(None)
    t = yf.Ticker("TQQQ")
    pt = yf.download("TQQQ", start="2018-10-01", end="2026-10-05", auto_adjust=False,
                     progress=False, threads=False)["Close"]
    pt = pt.iloc[:, 0] if hasattr(pt, "columns") else pt
    pt.index = pd.to_datetime(pt.index).tz_localize(None)
    sp = t.splits; sp.index = pd.to_datetime(sp.index).tz_localize(None)
    f = pd.Series(1.0, index=pt.index)
    for d, r in sp.items():
        f[pt.index < d] *= r
    pt = pt * f                                          # 真实挂牌价
    PQ, PT = pq.reindex(days).ffill(), pt.reindex(days).ffill()
    rate_qqq = PER_SHARE / PQ                            # 每单位 QQQ 敞口的成本
    rate_tqqq = (PER_SHARE / PT) / 3.0                   # 走 TQQQ：名义金额只要 1/3

    # ---------- 30 个相位的 gross / turnover
    print("跑 30 个相位 ...", flush=True)
    G, TO = {}, {}
    for off in range(30):
        r = run(b, NoiseCfg(check_times=checks_at(off), cost_bp=1.0, **SOTA))
        G[off] = r["overlay_gross"]
        TO[off] = r["cost"] * 1e4
    G, TO = pd.DataFrame(G), pd.DataFrame(TO)

    def sleeve(cost_rate, phase):
        """phase=0 论文相位；phase='mean' 相位均值（30 个相位各自净收益再平均）"""
        if phase == "mean":
            return (G.sub(TO.mul(cost_rate, axis=0))).mean(axis=1)
        return G[phase] - TO[phase] * cost_rate

    rows = []

    def add(name, series, per, w=None):
        lo, hi = per
        m = met(series.loc[lo:hi], bh.loc[lo:hi])
        rows.append(dict(口径=name, 期间=f"{lo}-{hi}", w=w, **m))

    # ========== A 叠加层单独
    print("\n" + "=" * 100)
    print("A. 日内叠加层**单独**（这是 PART10 说 0.3~0.9 的那个东西）")
    print("=" * 100)
    for nm, rr in (("QQQ 保证金成本", rate_qqq), ("TQQQ 成本", rate_tqqq)):
        for ph, plab in ((0, "论文相位"), ("mean", "相位均值")):
            s = sleeve(rr, ph)
            add(f"叠加层单独 / {nm} / {plab}", s, TEST)
            add(f"叠加层单独 / {nm} / {plab}", s, ALL)
    A = pd.DataFrame(rows)
    print(A[A.口径.str.startswith("叠加层")][["口径", "期间", "sharpe", "ann", "vol",
                                              "mdd", "beta", "alpha"]].round(3)
          .to_string(index=False))

    # ========== B 组合
    rows2 = []

    def add2(name, series, per, w):
        lo, hi = per
        m = met(series.loc[lo:hi], bh.loc[lo:hi])
        rows2.append(dict(口径=name, 期间=f"{lo}-{hi}", w=w, **m))

    print("\n" + "=" * 100)
    print("B. 买入持有 QQQ + w × 叠加层（这是 PART9 说 2.011 的那个东西）")
    print("=" * 100)
    add2("纯买入持有 QQQ", bh, TEST, 0.0)
    add2("纯买入持有 QQQ", bh, ALL, 0.0)
    for ph, plab in ((0, "论文相位"), ("mean", "相位均值")):
        for nm, rr in (("QQQ保证金", rate_qqq), ("TQQQ", rate_tqqq)):
            s = sleeve(rr, ph).reindex(bh.index).fillna(0)
            for w in (0.5, 1.0, 1.5):
                add2(f"B&H + {w}x 叠加层 / {nm} / {plab}", bh + w * s, TEST, w)
    B = pd.DataFrame(rows2)
    print(B[["口径", "期间", "sharpe", "ann", "vol", "mdd", "calmar", "beta", "alpha"]]
          .round(3).to_string(index=False))

    # ========== C 关键对照：叠加层给组合**加了**多少
    print("\n" + "=" * 100)
    print("C. 关键问题：叠加层给组合**净加**了多少？（组合 Sharpe − 纯持有 Sharpe）")
    print("=" * 100)
    bh_t = met(bh.loc[TEST[0]:TEST[1]])["sharpe"]
    print(f"   纯买入持有 QQQ（test）Sharpe = {bh_t:.3f}\n")
    for ph, plab in ((0, "论文相位"), ("mean", "相位均值")):
        for nm, rr in (("QQQ保证金", rate_qqq), ("TQQQ", rate_tqqq)):
            s = sleeve(rr, ph).reindex(bh.index).fillna(0)
            out = []
            for w in (0.5, 1.0, 1.5):
                m = met((bh + w * s).loc[TEST[0]:TEST[1]])
                out.append(f"w={w}: {m['sharpe']:.3f} ({m['sharpe']-bh_t:+.3f})")
            print(f"   {plab:6s} / {nm:9s}  " + "   ".join(out))

    # ========== D 逐年（看衰减对两者分别的影响）
    print("\n" + "=" * 100)
    print("D. 逐年：衰减对「叠加层单独」和「组合」的影响不一样")
    print("=" * 100)
    s_lucky = sleeve(rate_tqqq, 0).reindex(bh.index).fillna(0)
    s_mean = sleeve(rate_tqqq, "mean").reindex(bh.index).fillna(0)
    yr = []
    for y in range(2019, 2027):
        lo, hi = f"{y}-01-01", f"{y}-12-31"
        if len(bh.loc[lo:hi].dropna()) < 60:
            continue
        yr.append(dict(年=y,
                       纯持有=met(bh.loc[lo:hi])["sharpe"],
                       叠加层_论文相位=met(s_lucky.loc[lo:hi])["sharpe"],
                       叠加层_相位均值=met(s_mean.loc[lo:hi])["sharpe"],
                       组合w1_论文相位=met((bh + s_lucky).loc[lo:hi])["sharpe"],
                       组合w1_相位均值=met((bh + s_mean).loc[lo:hi])["sharpe"]))
    Y = pd.DataFrame(yr)
    print(Y.round(3).to_string(index=False))

    pd.concat([A, B]).to_csv(OUT / "reconcile_numbers.csv", index=False)
    Y.to_csv(OUT / "reconcile_by_year.csv", index=False)
    print(f"\n已写入 {OUT}/reconcile_numbers.csv 与 reconcile_by_year.csv")


if __name__ == "__main__":
    main()
