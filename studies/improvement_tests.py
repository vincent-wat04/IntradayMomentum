"""四类改进的检验：不对称止损、波动乘数 VM、论文 §4.2 日内形态、§4.3 星期效应。

全部用 30 相位均值 + TQQQ 成本（最诚实的口径），dev(2019-22) / test(2023-26) 分开报。

⚠️ 论文 §4.2 的形态有一个前视陷阱：
   "NR4：当日区间是最近 4 天最小"——**当日区间要到收盘才知道**，
   用它去条件化当日 10:00 开始的交易是前视。论文表 5 没说清用的是哪种口径。
   所以这里两种都测：
     PIT  形态在 t−1 日成立 -> 交易 t 日（合法）
     LOOK 形态在 t 日成立   -> 交易 t 日（前视，仅用于说明论文的数可能是怎么来的）
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
from etft.intraday.strategy import buy_hold

from audit import reimpl_noise as RI

OUT = Path("outputs/full/intraday"); OUT.mkdir(parents=True, exist_ok=True)
PER_SHARE = 0.005 + 0.0045
DEV, TEST = ("2019", "2022"), ("2023", "2026")
NPH = 30


def chk(off):
    base = [(h, m) for h in range(10, 16) for m in (0, 30)][:-1]
    return [(pd.Timestamp(f"2000-01-01 {h:02d}:{m:02d}") +
             pd.Timedelta(minutes=off)).strftime("%H:%M") for h, m in base]


def met(x, bench=None):
    x = pd.Series(x).dropna()
    if len(x) < 60 or x.std() == 0:
        return dict(sharpe=np.nan, ann=np.nan, mdd=np.nan, n=len(x))
    eq = (1 + x).cumprod(); dd = eq / eq.cummax() - 1
    a = float(eq.iloc[-1] ** (252 / len(x)) - 1)
    o = dict(sharpe=float(x.mean() / x.std() * np.sqrt(252)), ann=a,
             vol=float(x.std() * np.sqrt(252)), mdd=float(dd.min()),
             n=int(len(x)), 日均bp=float(x.mean() * 1e4),
             t=float(x.mean() / x.std() * np.sqrt(len(x))))
    if bench is not None:
        b = pd.Series(bench).reindex(x.index).fillna(0)
        if b.std() > 0:
            o["beta"] = float(np.cov(x, b)[0, 1] / b.var())
    return o


def phase_mean(b, rate, **kw):
    """30 个相位各自算净收益再平均。"""
    acc = []
    for off in range(NPH):
        r, _ = RI.simulate(b, checks=chk(off), sizing="voltarget", max_lev=4.0,
                           reentry="hold", half_spread=0.0, commission=0.0, **kw)
        # 成本单独加：reimpl 的成本是每股口径，这里用 TQQQ 的逐日费率重算
        r2, F = RI.simulate(b, checks=chk(off), sizing="voltarget", max_lev=4.0,
                            reentry="hold", half_spread=0.005, commission=0.0045, **kw)
        acc.append((r, r2, F))
    return acc


def main():
    b = load_bars("QQQ"); days = b.days; bh = buy_hold(b)
    pt = yf.download("TQQQ", start="2018-10-01", end="2026-10-05", auto_adjust=False,
                     progress=False, threads=False)["Close"]
    pt = pt.iloc[:, 0] if hasattr(pt, "columns") else pt
    pt.index = pd.to_datetime(pt.index).tz_localize(None)
    sp = yf.Ticker("TQQQ").splits; sp.index = pd.to_datetime(sp.index).tz_localize(None)
    f = pd.Series(1.0, index=pt.index)
    for d, r in sp.items():
        f[pt.index < d] *= r
    PT = (pt * f).reindex(days).ffill()
    # TQQQ 每股成本换算：reimpl 用 (hs+comm) 每股、标的是 QQQ 价格；
    # 要换成 TQQQ 口径，等价于把每股金额乘以 (P_QQQ / P_TQQQ) / 3
    px_qqq = b.day_close
    scale = (px_qqq / PT) / 3.0

    def run_cfg(**kw):
        """返回 30 相位均值的净收益（TQQQ 成本口径）。"""
        acc = []
        for off in range(NPH):
            rr, F = RI.simulate(b, checks=chk(off), sizing="voltarget", max_lev=4.0,
                                reentry="hold", half_spread=0.005 , commission=0.0045, **kw)
            # 重算成本：把每笔佣金按 scale 调整
            if len(F):
                F = F.copy(); F["day"] = pd.to_datetime(F.day)
                c_old = F.groupby("day").commission.sum().reindex(days).fillna(0.0)
                c_new = c_old * scale.reindex(days).fillna(1.0)
                rr = rr + c_old - c_new          # 换成 TQQQ 成本
            acc.append(rr)
        return pd.DataFrame(acc).T.mean(axis=1)

    rows = []

    def add(group, name, s, extra=None):
        for per, pl in ((DEV, "dev19-22"), (TEST, "test23-26")):
            m = met(s.loc[per[0]:per[1]], bh.loc[per[0]:per[1]])
            rows.append(dict(组=group, 配置=name, 期间=pl, **m, **(extra or {})))

    # ================= 1. 不对称止损
    print("=" * 104)
    print("1. 不对称止损（诊断：空头 64.6% 被止损、止损交易胜率仅 10.1%，"
          "而持到收盘的空头 +59.2bp/胜率 77.6%）")
    print("=" * 104)
    base = None
    for sl in ("vwap", "band"):
        for ss in ("vwap", "band"):
            s = run_cfg(stop_long=sl, stop_short=ss)
            nm = f"多头{sl} / 空头{ss}" + ("  <-当前" if (sl, ss) == ("vwap", "vwap") else "")
            add("止损", nm, s)
            if (sl, ss) == ("vwap", "vwap"):
                base = s
            print(f"  {nm:28s} dev {met(s.loc[DEV[0]:DEV[1]])['sharpe']:+.3f}  "
                  f"test {met(s.loc[TEST[0]:TEST[1]])['sharpe']:+.3f}", flush=True)

    # ================= 2. 波动乘数 VM（论文 §4.4 说最优约 1.5）
    print("\n" + "=" * 104)
    print("2. 波动乘数 VM（论文 §4.4：风险调整后最优约 1.5，但他们为简单仍用 1.0）")
    print("=" * 104)
    for vm in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5):
        s = run_cfg(vm=vm)
        add("VM", f"vm={vm}", s, dict(vm=vm))
        print(f"  vm={vm:<5} dev {met(s.loc[DEV[0]:DEV[1]])['sharpe']:+.3f}  "
              f"test {met(s.loc[TEST[0]:TEST[1]])['sharpe']:+.3f}  "
              f"test年化 {met(s.loc[TEST[0]:TEST[1]])['ann']:+.2%}", flush=True)
    # 不对称 VM：空头要求更大的突破
    for vl, vs in ((1.0, 1.5), (1.0, 2.0), (1.5, 1.0)):
        s = run_cfg(vm_long=vl, vm_short=vs)
        add("VM", f"多头vm={vl} / 空头vm={vs}", s)
        print(f"  多头vm={vl} 空头vm={vs}  dev {met(s.loc[DEV[0]:DEV[1]])['sharpe']:+.3f}  "
              f"test {met(s.loc[TEST[0]:TEST[1]])['sharpe']:+.3f}", flush=True)

    # ================= 3/4. 日内形态 与 星期效应（在基线收益上做日级筛选）
    print("\n" + "=" * 104)
    print("3. 论文 §4.2 日内形态（PIT = 形态在 t−1 日成立；LOOK = t 日成立，前视）")
    print("=" * 104)
    H = b.h.max(axis=1); L = b.l.min(axis=1); O = b.day_open; C = b.day_close
    rng = H - L
    pat = {}
    pat["NR4"] = rng == rng.rolling(4).min()
    pat["NR7"] = rng == rng.rolling(7).min()
    pat["ID"] = (H < H.shift(1)) & (L > L.shift(1))
    pat["OD"] = (H > H.shift(1)) & (L < L.shift(1))
    pat["Triangle"] = (H < pd.concat([H.shift(1), H.shift(2)], axis=1).min(axis=1)) & \
                      (L > pd.concat([L.shift(1), L.shift(2)], axis=1).max(axis=1))
    big = rng > rng.rolling(14).mean()
    pat["Trend"] = (((O < L + 0.15 * rng) & (C > L + 0.85 * rng)) |
                    ((O > L + 0.85 * rng) & (C < L + 0.15 * rng))) & big
    pat["BigTail"] = ((O > L + 0.75 * rng) & (C > L + 0.75 * rng)) | \
                     ((O < L + 0.25 * rng) & (C < L + 0.25 * rng))
    pat["Closure"] = (C > L + 0.9 * rng) | (C < L + 0.1 * rng)

    add("形态", "无条件（基线）", base)
    m_all = met(base.loc[TEST[0]:TEST[1]])
    print(f"  {'无条件（基线）':24s} 天数 {m_all['n']:4d}  dev "
          f"{met(base.loc[DEV[0]:DEV[1]])['sharpe']:+.3f}  test {m_all['sharpe']:+.3f}  "
          f"test日均 {m_all['日均bp']:+.2f}bp")
    for nm, mask in pat.items():
        for lag, lab in ((1, "PIT"), (0, "LOOK")):
            m2 = mask.shift(lag).fillna(False).reindex(base.index).fillna(False)
            s = base.where(m2)
            add("形态", f"{nm} [{lab}]", s)
            md, mt = met(s.loc[DEV[0]:DEV[1]]), met(s.loc[TEST[0]:TEST[1]])
            print(f"  {nm+' ['+lab+']':24s} 天数 {mt.get('n',0):4d}  dev "
                  f"{md.get('sharpe',float('nan')):+.3f}  test {mt.get('sharpe',float('nan')):+.3f}  "
                  f"test日均 {mt.get('日均bp',float('nan')):+.2f}bp  t={mt.get('t',float('nan')):+.2f}")

    print("\n" + "=" * 104)
    print("4. 论文 §4.3 星期效应（星期是事前已知的，无前视问题）")
    print("=" * 104)
    dow = pd.Series(base.index.dayofweek, index=base.index)
    for i, nm in enumerate(["周一", "周二", "周三", "周四", "周五"]):
        s = base.where(dow == i)
        add("星期", nm, s)
        md, mt = met(s.loc[DEV[0]:DEV[1]]), met(s.loc[TEST[0]:TEST[1]])
        print(f"  {nm}  天数 {mt.get('n',0):4d}  dev {md.get('sharpe',float('nan')):+.3f}  "
              f"test {mt.get('sharpe',float('nan')):+.3f}  "
              f"test日均 {mt.get('日均bp',float('nan')):+.2f}bp  t={mt.get('t',float('nan')):+.2f}")

    R = pd.DataFrame(rows)
    R.to_csv(OUT / "improvement_tests.csv", index=False)
    print(f"\n已写入 {OUT}/improvement_tests.csv")


if __name__ == "__main__":
    main()
