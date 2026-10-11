"""止盈的前置诊断：到底有没有"逃顶"这回事？

为什么先做诊断而不是直接扫参数：
  止盈能不能赚钱，只取决于一个条件概率 ——
    「已经赚到 +x·σ 之后，剩下这段路的期望收益是正还是负」
  若为正，说明动量还在延续，止盈就是在砍右尾，再怎么调参数都救不回来；
  若为负，说明确实存在日内衰竭/回转，止盈才有讨论价值。
  这个量不依赖任何策略实现，也不需要先决定止盈阈值，所以必须先测它。

重要背景（为什么说"止盈从来没被真正测过"）：
  reimpl_noise.simulate 里 giveback（跟踪止盈）是挂在**止损槽**上的，而
  stop_design_study / stop_significance 全部用 reentry="hold"，而 hold 的语义是
      if hit and reentry == "hold" and side_now == sign(shares): hit = False
  也就是"同方向开仓条件仍成立时，不平仓"。于是 giveback 恰恰在**最该止盈的时候
  被抑制**（趋势还在、浮盈还在涨）。所以 stop_design.csv 里的 giveback 数字
  测的是另一种止损，不是止盈。

口径：
  * 复用已验证的 RI.simulate 产生开仓/离场明细（只借它的**入场逻辑**，不改记账）
  * 对每笔交易，从入场后逐根 bar 算有利偏离 favor = sgn·(close/entry − 1)
  * 首次 favor ≥ x·σ_daily 的那根 bar 记为"止盈触发点"，
    forward = sgn·(实际离场价/该点收盘 − 1) = **止盈放弃掉的那段收益**
  * forward 的符号就是答案：>0 止盈亏钱，<0 止盈赚钱
  * 30 相位汇总，dev(2019-22)/test(2023-26) 分开报，避免相位运气
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from etft.intraday.bars import load_bars

from audit import reimpl_noise as RI

OUT = Path("outputs/full/intraday")
OUT.mkdir(parents=True, exist_ok=True)
DEV, TEST = ("2019", "2022"), ("2023", "2026")
NPH = 30
THRESH = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)


def chk(off):
    base = [(h, m) for h in range(10, 16) for m in (0, 30)][:-1]
    return [(pd.Timestamp(f"2000-01-01 {h:02d}:{m:02d}") +
             pd.Timedelta(minutes=off)).strftime("%H:%M") for h, m in base]


def extract_trades(b, F, sd, times_idx, C, O):
    """把 fills 配对成逐笔交易，并记录入场/离场的 bar 下标与价格。"""
    if not len(F):
        return []
    F = F.copy()
    F["day"] = pd.to_datetime(F.day)
    out = []
    for d, g in F.groupby("day", sort=False):
        g = g.reset_index(drop=True)
        sd_d = float(sd.loc[d]) if d in sd.index and np.isfinite(sd.loc[d]) else np.nan
        if not np.isfinite(sd_d) or sd_d <= 0:
            continue
        cur = None
        for _, r in g.iterrows():
            k = times_idx.get(r.clock)
            if k is None:
                continue
            if str(r.kind).startswith("ENTRY"):
                cur = dict(day=d, sgn=(1.0 if r.shares > 0 else -1.0),
                           entry_k=k + 1, entry_px=float(r.price), sd=sd_d)
            elif cur is not None:
                # 离场：STOP / TAILSTOP / MOC
                cur.update(exit_k=k, exit_px=float(r.price), exit_kind=str(r.kind))
                out.append(cur); cur = None
    return out


def main():
    b = load_bars("QQQ")
    days = b.days
    C, O = b.c, b.o
    times = list(C.columns)
    times_idx = {t: i for i, t in enumerate(times)}
    sd = RI.daily_sigma(b.day_close, b.prev_close_adj, 14)

    # ---------- 收集全部相位的交易
    trades = []
    for off in range(NPH):
        _, F = RI.simulate(b, checks=chk(off), sizing="voltarget", max_lev=4.0,
                           reentry="hold", half_spread=0.005, commission=0.0045,
                           stop_long="vwap", stop_short="vwap",
                           tail_k=1.0, tail_every_bar=True)
        trades += extract_trades(b, F, sd, times_idx, C, O)
    print(f"共收集 {len(trades)} 笔交易（{NPH} 个相位，当前 SOTA 配置）")

    Cv = C.to_numpy()
    day_pos = {d: i for i, d in enumerate(days)}

    # ---------- 逐笔走路径
    rows = []
    for t in trades:
        i = day_pos.get(t["day"])
        if i is None or "exit_k" not in t:
            continue
        a, z = t["entry_k"], t["exit_k"]
        if z <= a:
            continue
        seg = Cv[i, a:z + 1]
        ok = np.isfinite(seg)
        if not ok.any():
            continue
        favor = t["sgn"] * (seg / t["entry_px"] - 1.0)      # 有利偏离（比例）
        fav_s = favor / t["sd"]                              # 以 σ_daily 为单位
        mfe = float(np.nanmax(fav_s))
        rec = dict(day=t["day"], sgn=t["sgn"], exit_kind=t["exit_kind"],
                   mfe_sigma=mfe,
                   final=float(t["sgn"] * (t["exit_px"] / t["entry_px"] - 1.0)) / t["sd"])
        for x in THRESH:
            hit = np.where(np.isfinite(fav_s) & (fav_s >= x))[0]
            if len(hit) == 0:
                rec[f"fwd_{x}"] = np.nan
                rec[f"reach_{x}"] = 0
            else:
                j = int(hit[0])
                px = seg[j]
                # 止盈放弃掉的那段：从触发点收盘 到 实际离场价
                rec[f"fwd_{x}"] = float(t["sgn"] * (t["exit_px"] / px - 1.0))
                rec[f"reach_{x}"] = 1
        rows.append(rec)

    T = pd.DataFrame(rows)
    T["year"] = T.day.dt.year
    T.to_csv(OUT / "takeprofit_diag_trades.csv", index=False)
    print(f"逐笔明细 -> {OUT}/takeprofit_diag_trades.csv  （{len(T)} 笔）")

    def seg_report(T, label):
        print(f"\n{'='*96}\n【{label}】  共 {len(T)} 笔，MFE 中位数 {T.mfe_sigma.median():.3f}σ")
        print(f"{'阈值x':>7} {'达到率':>8} {'n':>7} "
              f"{'放弃段均值(bp)':>15} {'中位(bp)':>11} {'t值':>8} {'放弃段>0占比':>12}")
        out = []
        for x in THRESH:
            m = T[f"reach_{x}"] == 1
            n = int(m.sum())
            if n < 30:
                print(f"{x:>7} {m.mean():>7.1%} {n:>7} {'样本太少':>15}")
                continue
            f = T.loc[m, f"fwd_{x}"].dropna() * 1e4
            tt = f.mean() / f.std() * np.sqrt(len(f)) if f.std() > 0 else np.nan
            print(f"{x:>7} {m.mean():>7.1%} {n:>7} {f.mean():>15.2f} "
                  f"{f.median():>11.2f} {tt:>8.2f} {(f > 0).mean():>11.1%}")
            out.append(dict(seg=label, x=x, reach=m.mean(), n=n, fwd_bp=f.mean(),
                            fwd_med=f.median(), t=tt, frac_pos=(f > 0).mean()))
        return out

    allrep = []
    allrep += seg_report(T, "全样本")
    dv = T[(T.year >= int(DEV[0])) & (T.year <= int(DEV[1]))]
    ts = T[(T.year >= int(TEST[0])) & (T.year <= int(TEST[1]))]
    allrep += seg_report(dv, f"dev {DEV[0]}-{DEV[1]}")
    allrep += seg_report(ts, f"test {TEST[0]}-{TEST[1]}")
    allrep += seg_report(T[T.sgn > 0], "全样本·多头")
    allrep += seg_report(T[T.sgn < 0], "全样本·空头")

    pd.DataFrame(allrep).to_csv(OUT / "takeprofit_diag.csv", index=False)
    print(f"\n汇总 -> {OUT}/takeprofit_diag.csv")

    print("\n" + "=" * 96)
    print("怎么读这张表：")
    print("  『放弃段均值』= 止盈之后到实际离场之间，顺着仓位方向的收益。")
    print("  >0  -> 止盈砍掉的是**还在赚的那段**，止盈亏钱（动量仍在延续）")
    print("  <0  -> 触发后确实会回吐，存在日内衰竭，止盈有价值")
    print("  参考：TQQQ 实测单腿成本 0.1323bp，所以 |放弃段| 要明显超过约 0.3bp 才值得动。")


if __name__ == "__main__":
    main()
