"""去掉止损（尤其是空头止损）的收益与风险分析。

诊断（PART11 §1）显示：空头持到收盘 +59.2bp/胜率 77.6%，被止损的 −26.8bp/胜率 10.1%。
但这**不能**推出"去掉止损就能把 −26.8 变成 +59.2" —— 那是生存者偏差：
被止损的交易如果放任不管，有的会反弹，有的会亏更多。必须实测。

另一个必须说清的副作用：**反手是走止损路径实现的**（先平后反向开仓），
所以关掉止损后仓位无法反手，策略退化成"当日第一个信号持到收盘"。
换手会下降（成本降低），但也放弃了日内调整的能力。

风险口径（不能只看 Sharpe）：
  最差单日、左尾 5% CVaR、最大回撤、偏度、以及 4 倍杠杆下的理论单日损失
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
        return {}
    eq = (1 + x).cumprod(); dd = eq / eq.cummax() - 1
    q5 = x.quantile(0.05)
    return dict(sharpe=float(x.mean() / x.std() * np.sqrt(252)),
                ann=float(eq.iloc[-1] ** (252 / len(x)) - 1),
                vol=float(x.std() * np.sqrt(252)), mdd=float(dd.min()),
                最差单日=float(x.min()), 左尾5pct=float(q5),
                CVaR5=float(x[x <= q5].mean()),
                偏度=float(((x - x.mean()) ** 3).mean() / x.std() ** 3),
                峰度=float(((x - x.mean()) ** 4).mean() / x.std() ** 4 - 3),
                胜率=float((x > 0).mean()), n=int(len(x)))


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
        acc, turn = [], []
        for off in range(NPH):
            rr, F = RI.simulate(b, checks=chk(off), sizing="voltarget", max_lev=4.0,
                                reentry="hold", half_spread=0.005, commission=0.0045, **kw)
            if len(F):
                F = F.copy(); F["day"] = pd.to_datetime(F.day)
                c_old = F.groupby("day").commission.sum().reindex(days).fillna(0.0)
                rr = rr + c_old - c_old * scale.reindex(days).fillna(1.0)
                turn.append(len(F) / max(F.day.nunique(), 1))
            acc.append(rr)
        return pd.DataFrame(acc).T.mean(axis=1), float(np.mean(turn)) if turn else np.nan

    # ---------- 1. 3x3 止损网格
    print("=" * 112)
    print("1. 止损组合 3×3（30 相位均值 + TQQQ 成本）。none = 完全不止损（也就无法反手）")
    print("=" * 112)
    rows, curves = [], {}
    for sl in ("vwap", "band", "none"):
        for ss in ("vwap", "band", "none"):
            s, tp = run_cfg(stop_long=sl, stop_short=ss)
            nm = f"多{sl}/空{ss}"
            curves[nm] = s
            for per, pl in ((DEV, "dev"), (TEST, "test")):
                rows.append(dict(配置=nm, 期间=pl, 日均成交腿数=tp,
                                 **risk(s.loc[per[0]:per[1]])))
            print(f"  {nm:16s} 日均成交腿数 {tp:4.2f}  "
                  f"dev {risk(s.loc[DEV[0]:DEV[1]])['sharpe']:+.3f}  "
                  f"test {risk(s.loc[TEST[0]:TEST[1]])['sharpe']:+.3f}", flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "stop_removal.csv", index=False)

    cols = ["配置", "日均成交腿数", "sharpe", "ann", "vol", "mdd", "最差单日",
            "CVaR5", "偏度", "峰度", "胜率"]
    for pl in ("dev", "test"):
        print(f"\n--- {pl} 完整风险口径 ---")
        sub = R[R.期间 == pl][cols].copy()
        for c in ("ann", "vol", "mdd", "最差单日", "CVaR5", "胜率"):
            sub[c] = (sub[c] * 100).round(2)
        print(sub.round(3).to_string(index=False))

    # ---------- 2. 只改空头：三种止损的逐笔对比
    print("\n" + "=" * 112)
    print("2. 只改空头止损（多头保持 vwap）—— 被止损的那些交易放任不管会怎样")
    print("=" * 112)
    for ss in ("vwap", "none"):
        rr, F = RI.simulate(b, sizing="voltarget", max_lev=4.0, reentry="hold",
                            half_spread=0.005, commission=0.0045,
                            stop_long="vwap", stop_short=ss)
        F = F.copy(); F["day"] = pd.to_datetime(F.day)
        F["cmin"] = F.clock.map(lambda s: int(str(s)[:2]) * 60 + int(str(s)[3:5]))
        tr = []
        for d, g in F.groupby("day", sort=True):
            g = g.sort_values("cmin").reset_index(drop=True)
            op = None
            for _, row in g.iterrows():
                if row.kind.startswith("ENTRY"):
                    op = row
                elif op is not None:
                    side = 1 if op.shares > 0 else -1
                    tr.append(dict(side=side, ret=side * (row.price / op.price - 1),
                                   kind=row.kind, hold=row.cmin - op.cmin))
                    op = None
        T = pd.DataFrame(tr)
        sh = T[T.side < 0]
        print(f"  空头止损={ss:5s}: 空头 {len(sh):4d} 笔  胜率 {(sh.ret>0).mean():.1%}  "
              f"平均 {sh.ret.mean()*1e4:+7.2f}bp  中位 {sh.ret.median()*1e4:+7.2f}bp  "
              f"最差单笔 {sh.ret.min()*100:+.2f}%  "
              f"左尾5% {sh.ret.quantile(.05)*100:+.2f}%  "
              f"中位持仓 {sh.hold.median():.0f}分钟")
        if ss == "none":
            print(f"             多头 {len(T[T.side>0]):4d} 笔（未变），"
                  f"空头全部持到收盘所以笔数从 718 降到 {len(sh)}")

    # ---------- 3. 4 倍杠杆下的尾部：最差的几天
    print("\n" + "=" * 112)
    print("3. 尾部风险：最差的 5 天（4 倍敞口上限下，单位 = 资金的百分比）")
    print("=" * 112)
    for nm in ("多vwap/空vwap", "多vwap/空none", "多none/空none"):
        s = curves[nm].loc[TEST[0]:TEST[1]].dropna()
        w = s.nsmallest(5)
        print(f"  {nm:16s} " + "  ".join(f"{d.date()}:{v*100:+.2f}%" for d, v in w.items()))

    # ---------- 4. 叠加到 B&H + 排除周一（用户已同意排除周一）
    print("\n" + "=" * 112)
    print("4. 作为 B&H 叠加层（w=1.0），并叠加「排除周一」（已确认采用）")
    print("=" * 112)
    m0 = risk(bh.loc[TEST[0]:TEST[1]])
    print(f"   纯持有 QQQ test: Sharpe {m0['sharpe']:.3f}  回撤 {m0['mdd']:.1%}  "
          f"最差单日 {m0['最差单日']*100:+.2f}%\n")
    print(f"{'配置':24s} {'排周一':>6} {'单独dev':>8} {'单独test':>9} {'单独年化':>9} "
          f"{'单独回撤':>9} {'叠加test':>9} {'Δ':>7} {'叠加回撤':>9} {'2022叠加':>9}")
    best = []
    for nm in ("多vwap/空vwap", "多band/空vwap", "多vwap/空none", "多band/空none",
               "多none/空none"):
        for ex_mon in (False, True):
            s = curves[nm].where(dow != 0, 0.0) if ex_mon else curves[nm]
            rd, rt = risk(s.loc[DEV[0]:DEV[1]]), risk(s.loc[TEST[0]:TEST[1]])
            ov = risk((bh + s.reindex(bh.index).fillna(0)).loc[TEST[0]:TEST[1]])
            o22 = risk((bh + s.reindex(bh.index).fillna(0)).loc["2022-01-01":"2022-12-31"])
            print(f"{nm:24s} {'是' if ex_mon else '否':>6} {rd['sharpe']:+8.3f} "
                  f"{rt['sharpe']:+9.3f} {rt['ann']:+9.2%} {rt['mdd']:+9.1%} "
                  f"{ov['sharpe']:9.3f} {ov['sharpe']-m0['sharpe']:+7.3f} "
                  f"{ov['mdd']:+9.1%} {o22['sharpe']:+9.3f}")
            best.append((nm, ex_mon, rd["sharpe"], rt["sharpe"], ov["sharpe"]))

    print("\n   按 dev 选（不看 test）：", max(best, key=lambda x: x[2])[:2])
    pd.to_pickle(curves, OUT / "stop_removal_curves.pkl")
    print(f"\n已写入 {OUT}/stop_removal.csv 与 stop_removal_curves.pkl")


if __name__ == "__main__":
    main()
