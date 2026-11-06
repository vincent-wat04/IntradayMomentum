"""用 mdrec 的 TQQQ 10 档盘口检查回测成本假设(半价差 $0.005/股)在订单变大时是否还成立。
预注册(运行前写定):
  方法:对每个完整交易日、每个检查时点 T(10:00/10:30/12:00/14:00/15:40,取 [T, T+60s) 内所有无交叉的盘口快照),
        模拟一笔市价单(买、卖各一)逐档吃单,滑点 = 成交均价 − 中间价(买)/ 中间价 − 成交均价(卖);先对快照取均值,再对日取中位数。
        订单规模 Q ∈ {1k, 2k, 5k, 10k, 20k, 50k} 股。静态快照:不含排队补单、不拆单,是"一次性市价单"的上界。
  订单规模分布:Sleeve 单笔名义 = 当日 e_vt · k /3 · CAPITAL_BASE(v4),v5 再乘 ρ·c·f(sleeve_mult);价格取实测 TQQQ 中间价。k 取多/空腿中较大者(上界)。
  判据:若 v4/v5 的订单规模中位数下的额外滑点(相对假设 $0.005)< $0.002/股 → "成本假设在当前规模下成立";
        给出额外滑点 ≥ $0.005/股(即成本翻倍)所对应的订单规模,作为容量上限的粗估。
python3 hf_exec_cost.py   (在 studies/ 下运行)
"""
from __future__ import annotations

import os
import sys
import warnings

warnings.filterwarnings("ignore"); os.environ.setdefault("OMP_NUM_THREADS", "1")
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "audit"))
import hf_mdrec as H

TIMES = ["10:00", "10:30", "12:00", "14:00", "15:40"]; SIZES = [1000, 2000, 5000, 10000, 20000, 50000]; ASSUMED = 0.005
CAPITAL = 500_000.0
TAB = HERE.parent / "outputs" / "tables"


def walk(px, sz, Q):
    """px, sz: (n, 10);返回 (均价, 成交比例)。"""
    cum = np.cumsum(sz, axis=1); prev = cum - sz
    take = np.clip(Q - prev, 0, sz); filled = take.sum(1)
    avg = (take * px).sum(1) / np.maximum(filled, 1e-9)
    return avg, filled / Q


def snapshots():
    df = H.read_all("US.TQQQ", "book"); df["ts"] = pd.to_datetime(df["t_bid"], format="ISO8601")
    cols = [f"{p}{i}" for p in ("bp", "bs", "ap", "as") for i in range(1, 11)]
    for c in cols: df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["bp1", "ap1"]); df = df[df.ap1 > df.bp1]
    t = df.ts.dt.time; df = df[(t >= pd.Timestamp("09:30").time()) & (t < pd.Timestamp("16:00").time())]
    df[[c for c in cols if c[:2] in ("bs", "as")]] = df[[c for c in cols if c[:2] in ("bs", "as")]].fillna(0.0)
    df["day"] = df.ts.dt.strftime("%Y-%m-%d"); df["hm"] = df.ts.dt.strftime("%H:%M")
    return df.drop_duplicates(["ts", "bp1", "ap1", "bs1", "as1"])


def main():
    pd.set_option("display.width", 240, "display.float_format", lambda v: f"{v:9.4f}")
    df = snapshots(); n_day = df.groupby("day").size(); full = n_day[n_day > 5000].index.tolist()
    print(f"[数据] TQQQ 盘口快照 {len(df):,} 条;完整日 {len(full)}:{full}")
    rows = []
    for d in full:
        g = df[df.day == d]
        for T in TIMES:
            hh, mm = map(int, T.split(":")); lo = pd.Timestamp(f"{d} {T}:00"); hi = lo + pd.Timedelta(seconds=60)
            s = g[(g.ts >= lo) & (g.ts < hi)]
            if len(s) < 5: continue
            bp, bs = s[[f"bp{i}" for i in range(1, 11)]].to_numpy(), s[[f"bs{i}" for i in range(1, 11)]].to_numpy()
            ap, as_ = s[[f"ap{i}" for i in range(1, 11)]].to_numpy(), s[[f"as{i}" for i in range(1, 11)]].to_numpy()
            mid = (bp[:, 0] + ap[:, 0]) / 2
            for Q in SIZES:
                pa, fa = walk(ap, as_, Q); pb, fb = walk(bp, bs, Q)
                rows.append(dict(day=d, T=T, Q=Q, mid=mid.mean(), buy_slip=(pa - mid).mean(), sell_slip=(mid - pb).mean(), fill_b=fa.mean(), fill_s=fb.mean(),
                                 L1_ask=as_[:, 0].mean(), L10_ask=as_.sum(1).mean(), L10_bid=bs.sum(1).mean()))
    R = pd.DataFrame(rows); R["slip"] = (R.buy_slip + R.sell_slip) / 2; R["extra"] = R.slip - ASSUMED
    P = R.groupby(["T", "Q"]).agg(中间价=("mid", "median"), 滑点=("slip", "median"), 额外=("extra", "median"), 最差日滑点=("slip", "max"), 成交比例=("fill_b", "min"), L1买盘深度=("L1_ask", "median"), 十档买盘深度=("L10_ask", "median")).reset_index()
    print("\n=== 单笔市价单滑点(相对中间价,$/股;日中位数)==="); print(P.pivot(index="Q", columns="T", values="滑点").to_string())
    print("\n=== 相对假设半价差 $0.005 的额外滑点($/股)==="); print(P.pivot(index="Q", columns="T", values="额外").to_string())
    print("\n=== 最差日滑点 ==="); print(P.pivot(index="Q", columns="T", values="最差日滑点").to_string())
    print("\n=== 最小成交比例(<1 表示 10 档不够)==="); print(P.pivot(index="Q", columns="T", values="成交比例").to_string())
    dep = R.groupby("T")[["L1_ask", "L10_ask", "L10_bid"]].median(); print("\n=== 深度中位数(股)==="); print(dep.to_string())
    P.to_csv(TAB / "hf_exec_cost.csv", index=False)

    # ---------------------------------------------------------------- 订单规模分布
    import review_checks as RC
    import r3_study as R3, noise_core_dd as NC, tearsheet_v5 as T5, round3_common as C, unified_sizing_posthoc as PP
    E = RC.setup(); base = RC.v4(E); days, I, om = E["days"], E["I"], E["om"]
    ev = I["e_vt"].fillna(0).to_numpy(); bhv = I["bh"].to_numpy(); kmax = np.maximum(base["k"].kl.where(om, 1.0).fillna(1.0), base["k"].ks.where(om, 1.0).fillna(1.0)).to_numpy()
    w = NC.core_weights(C.load_close("yahoo/QQQ.parquet"))[0]["base"].reindex(I["bh"].index).ffill().fillna(1.0).reindex(days).fillna(1.0).to_numpy()
    s4 = base["sleeve"].to_numpy(); v5 = T5.build_v5(w, s4, ev, kmax, bhv, base["port"].to_numpy(), rho=1.5)
    px = float(R[R["T"] == "10:00"].mid.median()); rec = (days >= "2025-01-01")
    n4 = ev * kmax / 3 * CAPITAL; n5 = ev * kmax / 3 * v5["sleeve_mult"] * CAPITAL
    # v4 的 sleeve 乘子为 1;v5 的 sleeve_mult = c·ρ·f
    q = lambda a: pd.Series(a[rec & (ev > 0)] / px).describe(percentiles=[.1, .5, .9, .99])[["10%", "50%", "90%", "99%", "max"]]
    Dz = pd.DataFrame({"v4 股数": q(n4), "v5ρ1.5 股数": q(n5)}); print(f"\n=== 订单规模(2025–26,CAPITAL_BASE={CAPITAL:,.0f},TQQQ 价 ${px:.2f})==="); print(Dz.to_string())
    # 插值额外滑点 vs 规模(全时点中位数),并估算对应的资金规模
    cur = P.groupby("Q").额外.median(); Qs = np.array(cur.index, float); ex = cur.to_numpy()
    for nm, col in (("v4", Dz["v4 股数"]), ("v5ρ1.5", Dz["v5ρ1.5 股数"])):
        med = col["50%"]; p90 = col["90%"]
        print(f"  {nm}: 中位订单 {med:,.0f} 股 → 额外滑点 {np.interp(med, Qs, ex):+.4f} $/股;P90 订单 {p90:,.0f} 股 → {np.interp(p90, Qs, ex):+.4f} $/股")
    be = float(np.interp(ASSUMED, ex, Qs)) if ex.max() >= ASSUMED else float("nan")
    print(f"  额外滑点达到 $0.005/股(成本翻倍)的订单规模 ≈ {be:,.0f} 股" + (f" ≈ CAPITAL_BASE 的 {be/ (Dz['v5ρ1.5 股数']['50%']) :.1f}× 当前 v5 中位订单" if be == be else "(本网格内未达到)"))
    # 成本拖累:Sleeve 只在约三分之一的交易日有成交(30 相位里各相位有成交的日占比的均值),所以日均拖累 = 活跃日占比 × 活跃日拖累
    oos_days = days[om.to_numpy()]
    active = float(np.mean([F[pd.to_datetime(F.day).isin(oos_days)].day.nunique() / len(oos_days) for off, (rr, F) in E["PHb"].items() if len(F)]))
    print(f"  Sleeve 有成交的日占比(30 相位均值,OOS 2021–26)≈ {active:.2f}")
    for nm, nn in (("v4", n4), ("v5ρ1.5", n5)):
        sh = nn[rec & (ev > 0)] / px; xs = np.interp(sh, Qs, ex)   # bp(资金)/活跃日 = 2 边 × 额外$/股 × 股数 / 资金
        per_active = float(np.mean(2 * xs * sh / CAPITAL * 1e4))
        print(f"  {nm}: 活跃日额外滑点拖累 ≈ {per_active:.2f} bp(资金);日均(×活跃日占比)≈ {per_active*active:.2f} bp(Sleeve 日均收益(已扣假设成本)≈ 12.8 bp,OOS v4)")

if __name__ == "__main__":
    main()
