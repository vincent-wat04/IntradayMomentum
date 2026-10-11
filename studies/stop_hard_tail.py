"""逐根 bar 连续评估的硬尾部止损 —— 「只切极端左尾」的正确实现。

为什么前一轮测得不对：band/vwap 止损按论文只在整半点评估，而 vwap 在整半点
总是比尾部止损更紧，所以 "vwap+tail" 和 "vwap" 逐位相同（已验证最大差 0.00e+00）。
要让"只切极端尾部"真正起作用，硬止损必须**连续生效**（tail_every_bar=True）。

目标很明确：**在不降低逐笔均值的前提下压住最差单日。**
所以主看指标是 (最差单日, CVaR5) 和 (逐笔均值)，Sharpe 只是参考。
显著性一律用配对区块自助法，因为 Sharpe 的非配对标准误有 0.55~0.63。
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


def sr(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return x.mean() / x.std() * np.sqrt(252) if len(x) > 30 and x.std() > 0 else np.nan


def risk(x):
    x = pd.Series(x).dropna()
    if len(x) < 40 or x.std() == 0:
        return dict(sharpe=np.nan)
    eq = (1 + x).cumprod(); dd = eq / eq.cummax() - 1
    q5 = x.quantile(0.05); q1 = x.quantile(0.01)
    return dict(sharpe=sr(x), ann=float(eq.iloc[-1] ** (252 / len(x)) - 1),
                vol=float(x.std() * np.sqrt(252)), mdd=float(dd.min()),
                worst=float(x.min()), q1=float(q1), CVaR5=float(x[x <= q5].mean()),
                skew=float(((x - x.mean()) ** 3).mean() / x.std() ** 3))


def boot_pair(a, b, nb=3000, L=10, seed=0):
    idx = a.index.intersection(b.index)
    A, B = a.reindex(idx).fillna(0).to_numpy(), b.reindex(idx).fillna(0).to_numpy()
    n = len(A); nblk = int(np.ceil(n / L)); rg = np.random.default_rng(seed)
    out = np.empty(nb)
    for i in range(nb):
        st = rg.integers(0, max(n - L, 1), size=nblk)
        sel = np.concatenate([np.arange(s, min(s + L, n)) for s in st])[:n]
        out[i] = sr(A[sel]) - sr(B[sel])
    return sr(A) - sr(B), out


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
        acc, legs, tails = [], [], []
        for off in range(NPH):
            rr, F = RI.simulate(b, checks=chk(off), sizing="voltarget", max_lev=4.0,
                                reentry="hold", half_spread=0.005, commission=0.0045, **kw)
            if len(F):
                F = F.copy(); F["day"] = pd.to_datetime(F.day)
                c_old = F.groupby("day").commission.sum().reindex(days).fillna(0.0)
                rr = rr + c_old - c_old * scale.reindex(days).fillna(1.0)
                legs.append(len(F) / max(F.day.nunique(), 1))
                tails.append((F.kind == "TAILSTOP").sum())
            acc.append(rr)
        return (pd.DataFrame(acc).T.mean(axis=1), float(np.mean(legs)) if legs else np.nan,
                float(np.mean(tails)) if tails else 0.0)

    cfgs = [("vwap（当前基线）", dict(stop_long="vwap", stop_short="vwap"))]
    for k in (0.5, 1.0, 1.5, 2.0):
        cfgs.append((f"vwap + 硬尾{k}σ", dict(stop_long="vwap+tail", stop_short="vwap+tail",
                                            tail_k=k, tail_every_bar=True)))
    for k in (0.5, 1.0, 1.5):
        cfgs.append((f"band + 硬尾{k}σ", dict(stop_long="band+tail", stop_short="band+tail",
                                            tail_k=k, tail_every_bar=True)))
    for k in (0.5, 1.0, 1.5):
        cfgs.append((f"只有硬尾{k}σ", dict(stop_long="tail", stop_short="tail",
                                        tail_k=k, tail_every_bar=True)))
    cfgs.append(("band（对照）", dict(stop_long="band", stop_short="band")))

    print("=" * 122)
    print("1. 逐根 bar 硬尾部止损（k 的单位是 σ_daily；单笔资金风险上限 ≈ 2%·k）")
    print("=" * 122)
    print(f"{'配置':20s} {'腿/日':>6} {'尾损/年':>7} | {'dev SR':>7} {'dev年化':>8} "
          f"{'dev最差日':>9} {'devCVaR5':>9} | {'test SR':>8} {'test年化':>9} "
          f"{'test最差日':>10} {'tCVaR5':>8} {'偏度':>6}")
    curves, rows = {}, []
    for nm, kw in cfgs:
        s, lg, nt = run_cfg(**kw)
        curves[nm] = s
        rd, rt = risk(s.loc[DEV[0]:DEV[1]]), risk(s.loc[TEST[0]:TEST[1]])
        rows.append(dict(配置=nm, 腿每日=lg, 尾损每年=nt / 7.7,
                         **{f"dev_{a}": v for a, v in rd.items()},
                         **{f"test_{a}": v for a, v in rt.items()}))
        print(f"{nm:20s} {lg:6.2f} {nt/7.7:7.1f} | {rd['sharpe']:+7.3f} {rd['ann']:+8.2%} "
              f"{rd['worst']:+9.2%} {rd['CVaR5']:+9.2%} | {rt['sharpe']:+8.3f} "
              f"{rt['ann']:+9.2%} {rt['worst']:+10.2%} {rt['CVaR5']:+8.2%} "
              f"{rt['skew']:+6.2f}", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "stop_hard_tail.csv", index=False)

    # ---------- 2. 配对显著性
    print("\n" + "=" * 122)
    print("2. 配对区块自助（vs 基线 vwap）：ΔSharpe 是否能分辨")
    print("=" * 122)
    base = curves["vwap（当前基线）"]
    print(f"{'配置':20s} | {'dev ΔSR':>9} {'95%区间':>20} {'p':>7} | "
          f"{'test ΔSR':>9} {'95%区间':>20} {'p':>7}")
    for nm in [c[0] for c in cfgs[1:]]:
        line = f"{nm:20s} |"
        for per in (DEV, TEST):
            a = curves[nm].loc[per[0]:per[1]].dropna()
            bb = base.loc[per[0]:per[1]].dropna()
            real, dist = boot_pair(a, bb, seed=abs(hash(nm)) % 9999)
            lo, hi = np.percentile(dist, [2.5, 97.5])
            line += f" {real:+9.3f} [{lo:+7.3f},{hi:+7.3f}] {float((dist<=0).mean()):7.3f} |"
        print(line, flush=True)

    # ---------- 3. 尾部效果：最差的几天对比
    print("\n" + "=" * 122)
    print("3. 硬尾止损到底压住了最差的日子吗（test 期最差 5 天）")
    print("=" * 122)
    for nm in ("vwap（当前基线）", "vwap + 硬尾1.0σ", "vwap + 硬尾0.5σ", "band（对照）"):
        s = curves[nm].loc[TEST[0]:TEST[1]].dropna()
        w = s.nsmallest(5)
        print(f"  {nm:18s} " + "  ".join(f"{v*100:+.2f}%" for v in w))

    # ---------- 4. 叠加到 B&H（含排除周一）
    print("\n" + "=" * 122)
    print("4. 作为 B&H 叠加层（w=1.0，已排除周一）")
    print("=" * 122)
    m0 = risk(bh.loc[TEST[0]:TEST[1]])
    print(f"   纯持有 QQQ: Sharpe {m0['sharpe']:.3f}  回撤 {m0['mdd']:.1%}  "
          f"最差单日 {m0['worst']:+.2%}\n")
    print(f"{'配置':20s} {'叠加test':>9} {'Δ':>7} {'叠加回撤':>9} {'叠加最差日':>10} {'2022':>8}")
    for nm in [c[0] for c in cfgs]:
        sx = curves[nm].where(dow != 0, 0.0).reindex(bh.index).fillna(0)
        ov = risk((bh + sx).loc[TEST[0]:TEST[1]])
        o22 = risk((bh + sx).loc["2022-01-01":"2022-12-31"])
        print(f"{nm:20s} {ov['sharpe']:9.3f} {ov['sharpe']-m0['sharpe']:+7.3f} "
              f"{ov['mdd']:+9.1%} {ov['worst']:+10.2%} {o22['sharpe']:+8.3f}")

    pd.to_pickle(curves, OUT / "stop_hard_tail_curves.pkl")
    print(f"\n已写入 {OUT}/stop_hard_tail.csv")


if __name__ == "__main__":
    main()
