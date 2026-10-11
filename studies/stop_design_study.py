"""止损设计空间的系统扫描：有没有一种只切左尾、不切正常回撤的止损？

出发点（PART11 §1 + 去止损实验）：
  空头止损=vwap 时逐笔平均 +3.62bp、胜率 34%、最差单笔 −2.00%
  空头止损=none 时逐笔平均 +3.01bp、胜率 48%、最差单笔 −4.63%
  -> **止损没有在赚钱，它只是在削左尾。** 那就应该设计一个"只削左尾"的止损。

刻度的选择很关键：仓位本身是 min(4, 2%/σ_daily)，所以把止损放在 k×σ_daily 处，
单笔最大亏损 ≈ 敞口 × k × σ_daily = **2%·k 的资金**，与波动率环境无关。
k=0.25 -> 单笔最多亏 0.5% 资金；k=0.5 -> 1%；k=1.0 -> 2%。

测的止损族：
  vwap          论文改进版（最紧）
  band          穿过整条噪声带才出（较松）
  none          不止损
  tail          **只切尾部**：不利偏离 > k×σ_daily 才出
  band+tail     band 或 tail 任一触发
  vwap+tail     vwap 或 tail 任一触发
  giveback      从最大浮盈回吐 g 比例出场（跟踪止盈）
  tail+giveback 两者任一

全部 30 相位均值 + TQQQ 成本，dev(2019-22)/test(2023-26) 分开报，并给完整风险口径。
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

OUT = Path("outputs/full/intraday")
DEV, TEST = ("2019", "2022"), ("2023", "2026")
NPH = 30


def chk(off):
    base = [(h, m) for h in range(10, 16) for m in (0, 30)][:-1]
    return [(pd.Timestamp(f"2000-01-01 {h:02d}:{m:02d}") +
             pd.Timedelta(minutes=off)).strftime("%H:%M") for h, m in base]


def risk(x):
    x = pd.Series(x).dropna()
    if len(x) < 40 or x.std() == 0:
        return dict(sharpe=np.nan)
    eq = (1 + x).cumprod(); dd = eq / eq.cummax() - 1
    q5 = x.quantile(0.05)
    return dict(sharpe=float(x.mean() / x.std() * np.sqrt(252)),
                ann=float(eq.iloc[-1] ** (252 / len(x)) - 1),
                vol=float(x.std() * np.sqrt(252)), mdd=float(dd.min()),
                worst=float(x.min()), CVaR5=float(x[x <= q5].mean()),
                skew=float(((x - x.mean()) ** 3).mean() / x.std() ** 3),
                hit=float((x > 0).mean()))


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
    dow = pd.Series(days.dayofweek, index=days)

    def run_cfg(**kw):
        acc, legs = [], []
        for off in range(NPH):
            rr, F = RI.simulate(b, checks=chk(off), sizing="voltarget", max_lev=4.0,
                                reentry="hold", half_spread=0.005, commission=0.0045, **kw)
            if len(F):
                F = F.copy(); F["day"] = pd.to_datetime(F.day)
                c_old = F.groupby("day").commission.sum().reindex(days).fillna(0.0)
                rr = rr + c_old - c_old * scale.reindex(days).fillna(1.0)
                legs.append(len(F) / max(F.day.nunique(), 1))
            acc.append(rr)
        return pd.DataFrame(acc).T.mean(axis=1), (float(np.mean(legs)) if legs else np.nan)

    # ---------- 1. 对称扫描
    cfgs = [("vwap（当前）", dict(stop_long="vwap", stop_short="vwap"))]
    cfgs += [("band", dict(stop_long="band", stop_short="band")),
             ("none", dict(stop_long="none", stop_short="none"))]
    for k in (0.25, 0.5, 1.0):
        cfgs.append((f"tail k={k}", dict(stop_long="tail", stop_short="tail", tail_k=k)))
    for k in (0.25, 0.5, 1.0):
        cfgs.append((f"band+tail k={k}",
                     dict(stop_long="band+tail", stop_short="band+tail", tail_k=k)))
    for k in (0.5, 1.0):
        cfgs.append((f"vwap+tail k={k}",
                     dict(stop_long="vwap+tail", stop_short="vwap+tail", tail_k=k)))
    for g in (0.4, 0.6):
        cfgs.append((f"giveback g={g}",
                     dict(stop_long="giveback", stop_short="giveback", giveback=g)))
    cfgs.append(("tail k=0.5 + giveback 0.5",
                 dict(stop_long="tail+giveback", stop_short="tail+giveback",
                      tail_k=0.5, giveback=0.5)))

    print("=" * 118)
    print("1. 对称止损扫描（两侧用同一种）")
    print("=" * 118)
    print(f"{'止损方式':24s} {'腿/日':>6} | {'dev SR':>7} {'dev年化':>8} {'dev最差日':>9} "
          f"{'dev回撤':>8} | {'test SR':>8} {'test年化':>9} {'test最差日':>10} "
          f"{'test回撤':>9} {'偏度':>6}")
    curves, rows = {}, []
    for nm, kw in cfgs:
        s, lg = run_cfg(**kw)
        curves[nm] = s
        rd, rt = risk(s.loc[DEV[0]:DEV[1]]), risk(s.loc[TEST[0]:TEST[1]])
        rows.append(dict(配置=nm, 腿每日=lg, **{f"dev_{k}": v for k, v in rd.items()},
                         **{f"test_{k}": v for k, v in rt.items()}))
        print(f"{nm:24s} {lg:6.2f} | {rd['sharpe']:+7.3f} {rd['ann']:+8.2%} "
              f"{rd['worst']:+9.2%} {rd['mdd']:+8.1%} | {rt['sharpe']:+8.3f} "
              f"{rt['ann']:+9.2%} {rt['worst']:+10.2%} {rt['mdd']:+9.1%} "
              f"{rt['skew']:+6.2f}", flush=True)

    R = pd.DataFrame(rows)
    R.to_csv(OUT / "stop_design.csv", index=False)

    # ---------- 2. 逐笔分布：止损到底在削什么
    print("\n" + "=" * 118)
    print("2. 逐笔分布：各种止损对左尾和均值分别做了什么（论文相位，便于逐笔统计）")
    print("=" * 118)
    print(f"{'止损方式':24s} {'笔数':>5} {'胜率':>6} {'均值bp':>8} {'中位bp':>8} "
          f"{'最差单笔':>9} {'左尾5%':>8} {'盈亏比':>6} {'中位持仓':>8}")
    for nm, kw in cfgs:
        _, F = RI.simulate(b, sizing="voltarget", max_lev=4.0, reentry="hold",
                           half_spread=0.005, commission=0.0045, **kw)
        if not len(F):
            continue
        F = F.copy(); F["day"] = pd.to_datetime(F.day)
        F["cmin"] = F.clock.map(lambda s: int(str(s)[:2]) * 60 + int(str(s)[3:5]))
        tr = []
        for d, g in F.groupby("day", sort=True):
            g = g.sort_values("cmin").reset_index(drop=True); op = None
            for _, row in g.iterrows():
                if row.kind.startswith("ENTRY"):
                    op = row
                elif op is not None:
                    sd_ = 1 if op.shares > 0 else -1
                    tr.append(dict(side=sd_, ret=sd_ * (row.price / op.price - 1),
                                   hold=row.cmin - op.cmin))
                    op = None
        T = pd.DataFrame(tr)
        pos_, neg_ = T.ret[T.ret > 0], T.ret[T.ret < 0]
        print(f"{nm:24s} {len(T):5d} {(T.ret>0).mean():6.1%} {T.ret.mean()*1e4:+8.2f} "
              f"{T.ret.median()*1e4:+8.2f} {T.ret.min()*100:+9.2f}% "
              f"{T.ret.quantile(.05)*100:+8.2f}% "
              f"{abs(pos_.mean()/neg_.mean()):6.2f} {T.hold.median():8.0f}")

    # ---------- 3. 不对称：按 dev 选出的两侧组合
    print("\n" + "=" * 118)
    print("3. 不对称组合（多头/空头分别用不同止损）")
    print("=" * 118)
    modes = ["vwap", "band", "tail", "band+tail"]
    print(f"{'多头':12s} {'空头':12s} | {'dev SR':>7} | {'test SR':>8} {'test最差日':>10} "
          f"{'test回撤':>9} | {'叠加test':>9} {'Δ':>7} {'2022叠加':>9}")
    m0 = risk(bh.loc[TEST[0]:TEST[1]])
    best = []
    for sl in modes:
        for ss in modes:
            s, lg = run_cfg(stop_long=sl, stop_short=ss, tail_k=0.5)
            key = f"多{sl}/空{ss}"
            curves[key] = s
            rd, rt = risk(s.loc[DEV[0]:DEV[1]]), risk(s.loc[TEST[0]:TEST[1]])
            sx = s.where(dow != 0, 0.0)                      # 已确认采用：排除周一
            ov = risk((bh + sx.reindex(bh.index).fillna(0)).loc[TEST[0]:TEST[1]])
            o22 = risk((bh + sx.reindex(bh.index).fillna(0)).loc["2022-01-01":"2022-12-31"])
            best.append((key, rd["sharpe"], rt["sharpe"], ov["sharpe"], rt["worst"]))
            print(f"{sl:12s} {ss:12s} | {rd['sharpe']:+7.3f} | {rt['sharpe']:+8.3f} "
                  f"{rt['worst']:+10.2%} {rt['mdd']:+9.1%} | {ov['sharpe']:9.3f} "
                  f"{ov['sharpe']-m0['sharpe']:+7.3f} {o22['sharpe']:+9.3f}", flush=True)

    print(f"\n   纯持有 QQQ test Sharpe {m0['sharpe']:.3f}  回撤 {m0['mdd']:.1%}")
    bd = max(best, key=lambda x: x[1])
    print(f"   按 dev 选：{bd[0]}  dev {bd[1]:+.3f} -> test {bd[2]:+.3f}，"
          f"叠加 {bd[3]:.3f}，test 最差单日 {bd[4]:+.2%}")

    pd.to_pickle(curves, OUT / "stop_design_curves.pkl")
    print(f"\n已写入 {OUT}/stop_design.csv 与 stop_design_curves.pkl")


if __name__ == "__main__":
    main()
