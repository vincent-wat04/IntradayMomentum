"""止盈 × 同向多次开仓：策略层实测。

诊断（takeprofit_diag.py）已经给出方向：
  按天聚类后，**空头**到达 +0.5σ/+0.75σ/+1.0σ 之后的"放弃段"收益
    dev  −12.7 / −17.2 / −30.1 bp  (t=−2.4 / −2.7 / −3.6)
    test −14.8 / −18.1 / −17.6 bp  (t=−2.7 / −2.5 / −2.3)
  同号、同量级、两段都显著 -> 空头确实存在日内衰竭，止盈有理由。
  **多头**反过来：dev 负(t=−3.1) 但 test≈0(t=−0.26)、高阈值还翻正 -> 只做空头侧。

本脚本测的是"放到策略里还剩多少"，因为诊断没算这三件事：
  1) 止盈那条腿要付半价差+佣金（MOC 只付佣金），
  2) 止盈之后能不能、以及什么时候再进场（这决定了止盈是"离场"还是"换手"），
  3) 原本的 vwap 止损 + 硬尾止损已经吃掉了一部分同样的回吐。

再进场语义（tp_rearm）是关键，也是用户问的"同天可多次同向开仓"：
  done       止盈即收工，当天不再开仓
  next_check 下一个检查时点条件仍成立就立刻原向重开 —— 预期会白付 2 条腿
  rearm      必须先等价格回到噪声带内（信号归零）再次突破才允许开仓

口径与之前所有止损实验保持一致：30 相位均值、TQQQ 单股成本、dev/test 分开、
配对区块自助（区块 10 天、4000 次），并对照噪声护栏 0.15。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from etft.intraday.bars import load_bars

from audit import reimpl_noise as RI
from stop_significance import block_boot_pair, sr

OUT = Path("outputs/full/intraday")
OUT.mkdir(parents=True, exist_ok=True)
DEV, TEST = ("2019", "2022"), ("2023", "2026")
NPH = 30
BASE = dict(stop_long="vwap", stop_short="vwap", tail_k=1.0, tail_every_bar=True)


def chk(off):
    base = [(h, m) for h in range(10, 16) for m in (0, 30)][:-1]
    return [(pd.Timestamp(f"2000-01-01 {h:02d}:{m:02d}") +
             pd.Timedelta(minutes=off)).strftime("%H:%M") for h, m in base]


def risk(x):
    x = pd.Series(x).dropna()
    if len(x) < 40 or x.std() == 0:
        return dict(sharpe=np.nan)
    eq = (1 + x).cumprod(); dd = eq / eq.cummax() - 1
    return dict(sharpe=float(x.mean() / x.std() * np.sqrt(252)),
                ann=float(eq.iloc[-1] ** (252 / len(x)) - 1),
                mdd=float(dd.min()), worst=float(x.min()),
                hit=float((x > 0).mean()))


def main():
    b = load_bars("QQQ"); days = b.days
    pt = yf.download("TQQQ", start="2018-10-01", end="2026-10-05", auto_adjust=False,
                     progress=False, threads=False)["Close"]
    pt = pt.iloc[:, 0] if hasattr(pt, "columns") else pt
    pt.index = pd.to_datetime(pt.index).tz_localize(None)
    sp = yf.Ticker("TQQQ").splits; sp.index = pd.to_datetime(sp.index).tz_localize(None)
    fsp = pd.Series(1.0, index=pt.index)
    for d, r in sp.items():
        fsp[pt.index < d] *= r
    PT = (pt * fsp).reindex(days).ffill()
    scale = (b.day_close / PT) / 3.0          # QQQ 单股成本 -> TQQQ 单股成本

    def run_cfg(**kw):
        acc, legs, ent, reent = [], [], [], []
        for off in range(NPH):
            rr, F = RI.simulate(b, checks=chk(off), sizing="voltarget", max_lev=4.0,
                                reentry="hold", half_spread=0.005, commission=0.0045,
                                **BASE, **kw)
            if len(F):
                F = F.copy(); F["day"] = pd.to_datetime(F.day)
                c_old = F.groupby("day").commission.sum().reindex(days).fillna(0.0)
                rr = rr + c_old - c_old * scale.reindex(days).fillna(1.0)
                legs.append(len(F) / max(F.day.nunique(), 1))
                E = F[F.kind.astype(str).str.startswith("ENTRY")]
                per_day = E.groupby("day").size()
                ent.append(per_day.mean())
                reent.append(float((per_day >= 2).mean()))
            acc.append(rr)
        return (pd.DataFrame(acc).T.mean(axis=1), float(np.mean(legs)),
                float(np.mean(ent)), float(np.mean(reent)))

    cfgs = [("基线（无止盈）", {})]
    for k in (0.5, 0.75, 1.0, 1.5):
        for rearm in ("done", "rearm", "next_check"):
            cfgs.append((f"空头止盈 k={k} {rearm}",
                         dict(tp_k=k, tp_side="short", tp_rearm=rearm)))
    for k in (0.75, 1.0):
        cfgs.append((f"两侧止盈 k={k} rearm",
                     dict(tp_k=k, tp_side="both", tp_rearm="rearm")))
        cfgs.append((f"空头止盈 k={k} rearm 逐bar",
                     dict(tp_k=k, tp_side="short", tp_rearm="rearm", tp_every_bar=True)))

    curves, rows = {}, []
    for name, kw in cfgs:
        r, lg, en, re2 = run_cfg(**kw)
        curves[name] = r
        d = risk(r.loc[DEV[0]:DEV[1]]); t = risk(r.loc[TEST[0]:TEST[1]])
        rows.append(dict(配置=name, 腿每日=lg, 开仓每日=en, 两次以上占比=re2,
                         dev_sharpe=d["sharpe"], dev_ann=d["ann"], dev_mdd=d["mdd"],
                         dev_worst=d["worst"], test_sharpe=t["sharpe"],
                         test_ann=t["ann"], test_mdd=t["mdd"], test_worst=t["worst"]))
        print(f"  {name:32s} legs/day={lg:5.3f} entries/day={en:4.2f} "
              f"≥2次={re2:5.1%} dev={d['sharpe']:+.3f} test={t['sharpe']:+.3f}",
              flush=True)

    R = pd.DataFrame(rows)
    R.to_csv(OUT / "takeprofit_study.csv", index=False)
    pd.DataFrame(curves).to_pickle(OUT / "takeprofit_curves.pkl")
    print("\n" + "=" * 118)
    print(R.to_string(index=False))

    # ---------- 配对区块自助 vs 基线
    base = curves["基线（无止盈）"]
    print("\n" + "=" * 118)
    print("配对区块自助（区块 10 天、4000 次）vs 基线；p = 「改进 ≤ 0」的单边概率")
    print("=" * 118)
    print(f"{'配置':32s} | {'dev ΔSR':>9} {'95%区间':>19} {'p':>6} | "
          f"{'test ΔSR':>9} {'95%区间':>19} {'p':>6}")
    srows = []
    for name in curves:
        if name.startswith("基线"):
            continue
        line = f"{name:32s} |"; rec = dict(配置=name)
        for per, pl in ((DEV, "dev"), (TEST, "test")):
            a = curves[name].loc[per[0]:per[1]].dropna()
            bb = base.loc[per[0]:per[1]].dropna()
            real, dist = block_boot_pair(a, bb, seed=abs(hash(name)) % 10000)
            lo, hi = np.percentile(dist, [2.5, 97.5])
            p = float((dist <= 0).mean())
            line += f" {real:+9.3f} [{lo:+7.3f},{hi:+7.3f}] {p:6.3f} |"
            rec[f"{pl}_dSR"] = real; rec[f"{pl}_lo"] = lo
            rec[f"{pl}_hi"] = hi; rec[f"{pl}_p"] = p
        print(line, flush=True)
        srows.append(rec)
    S = pd.DataFrame(srows)
    S.to_csv(OUT / "takeprofit_significance.csv", index=False)

    print("\n" + "=" * 118)
    ok = S[(S.dev_dSR > 0) & (S.test_dSR > 0)]
    print(f"dev 和 test 同时为正的配置：{list(ok.配置) or '（无）'}")
    both = S[(S.dev_p < 0.10) & (S.test_p < 0.10)]
    print(f"dev 和 test 都 p<0.10：{list(both.配置) or '（无）'}")
    print("噪声护栏：随机日子集的 ΔSharpe 标准差 0.15~0.80 —— |ΔSR| 低于 0.15 一律按噪声处理")


if __name__ == "__main__":
    main()
