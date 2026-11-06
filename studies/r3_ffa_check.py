"""细查:去掉 FFA-P 跳过为什么 Sharpe 更高 —— 是否因为仓位模型(尤其线性)本来就在这些日子给很小的仓位?(2026-10-23,事后诊断;新配置 +1:ffa_rank 连续特征)
对三种仓位设定 {v3 sigmoid, L2 线性+dur_rate, v4 = E1+线性+dur_rate} 各跑"保留 FFA"(F)与"去掉 FFA"(N):
  A. N 模型在 FFA 跳过日 vs 其余日的 k(均值、分位、k=0 比例)
  B. 跳过日的损益:未缩放(k≡1)与 N 模型缩放后,分年
  C. ΔSR(N − F) 精确分解:混合序列 H1 = F 的其余日 + N 的跳过日(只加跳过日)、H2 = N 的其余日 + 跳过日置 0(只换模型)
  D. N 模型的 k 与 ffa_rank 的关系(五分位均值、Spearman)
  E. 对照:去掉硬跳过,把 ffa_rank 作为连续特征放进 v4 的仓位模型
python3 r3_ffa_check.py
"""
from __future__ import annotations

import os
import sys
import time
import warnings

warnings.filterwarnings("ignore"); os.environ.setdefault("OMP_NUM_THREADS", "1")
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import disp_pred_study as DP
import liquidity_inverse_study as LI
import r2_rules_study as RR
import r3_study as R3
import sizing_models_study as M
import sizing_sigma_feed as SF
import sota_unified_stack as SU
import unified_sizing_posthoc as P
from etft.intraday.bars import load_bars


def main():
    t0 = time.time(); pd.set_option("display.width", 280, "display.max_rows", 200, "display.float_format", lambda v: f"{v:8.3f}")
    b = load_bars("QQQ"); days = b.days; yrs = days.year
    I = SF.build_inputs(b); X = I["X"].copy(); keepF = I["keep"]; keep1 = pd.Series(1.0, index=days)
    dn = DP.targets(DP.Arr(b), 30).dn; full = pd.Series(b.last_idx.to_numpy() >= 389, index=days); td = pd.Series(days.dayofweek != 0, index=days)
    sel = (td & full & X.notna().all(1) & dn.notna() & (days >= "2019-01-01")).to_numpy(); om = pd.Series(sel & (yrs >= 2021), index=days)
    X["dur_rate"] = M.family_features(b, I["X"])["dur_rate"]
    chk = float(((X.ffa_rank <= 0.2) != (keepF == 0))[sel].mean()); print(f"[一致性] FFA 跳过日 ≡ ffa_rank ≤ 0.2 的不一致比例(样本日)= {chk:.4f}")
    _, S3m, _ = RR.band_multipliers(b)
    PHb, PHf = LI.run_phases_fills(b), SU.fomc_late_PH(b); PHbE, PHfE = R3.run_PH_entry(b, S3m), R3.run_PH_entry(b, S3m, fomc_only=True)
    LG = {("base", "F"): R3.legs_k(PHb, PHf, b, keepF), ("base", "N"): R3.legs_k(PHb, PHf, b, keep1),
          ("E1", "F"): R3.legs_k(PHbE, PHfE, b, keepF), ("E1", "N"): R3.legs_k(PHbE, PHfE, b, keep1)}
    SPEC = {"v3 sigmoid": ("base", P.CLEAN, "sig"), "L2 线性+dur_rate": ("base", P.CLEAN + ["dur_rate"], "lin"), "v4 E1+线性+dur_rate": ("E1", P.CLEAN + ["dur_rate"], "lin")}
    R = {}
    for nm, (lg, ft, lk) in SPEC.items():
        for f in ("F", "N"):
            L = LG[(lg, f)]; R[(nm, f)] = R3.assemble(R3.sizing(X, ft, L, I, sel, days, "mv", lk), L, I, om)
    L = LG[("E1", "N")]; R[("v4 + ffa_rank 连续特征", "N")] = R3.assemble(R3.sizing(X, P.CLEAN + ["dur_rate", "ffa_rank"], L, I, sel, days, "mv", "lin"), L, I, om)
    print(f"[完成] ({time.time()-t0:.0f}s)", flush=True)
    o = om.to_numpy(); oi = np.where(o)[0]; sk = (keepF == 0).to_numpy() & o; ns = (keepF == 1).to_numpy() & o
    dev = yrs[oi] <= 2022; BI = M.block_idx(len(oi), 10, 2000, 0); srv = lambda v: v.mean() / v.std() * np.sqrt(252)
    print(f"OOS 2021–26:{o.sum()} 个样本日,其中 FFA 跳过日 {sk.sum()}({sk.sum()/o.sum():.1%})")
    # ---- A. k
    rows = []
    for nm in list(SPEC) + ["v4 + ffa_rank 连续特征"]:
        k = R[(nm, "N")]["k"]
        for grp, m in (("FFA 跳过日", sk), ("其余日", ns)):
            kl, ks = k.kl.to_numpy()[m], k.ks.to_numpy()[m]; kk = np.r_[kl, ks]
            rows.append(dict(设定=nm, 日子=grp, k多均值=kl.mean(), k空均值=ks.mean(), k中位=np.median(kk), k_p10=np.quantile(kk, .1), k_p90=np.quantile(kk, .9),
                             k小于0_5比例=float((kk < 0.5).mean()), k等于0比例=float((kk <= 1e-6).mean())))
    TA = pd.DataFrame(rows); print("\n=== A. 去掉 FFA 的模型在两类日子上的仓位乘数 k(OOS)==="); print(TA.to_string(index=False))
    # ---- B. 跳过日损益
    raw = {"base": (LG[("base", "N")]["L2"] + LG[("base", "N")]["S2"]).to_numpy(), "E1": (LG[("E1", "N")]["L2"] + LG[("E1", "N")]["S2"]).to_numpy()}
    rows = []
    for nm, (lg, _, _) in list(SPEC.items()) + [("v4 + ffa_rank 连续特征", ("E1", None, None))]:
        sN = R[(nm, "N")]["sleeve"].to_numpy()
        for y in list(range(2021, 2027)) + ["全部"]:
            m = sk if y == "全部" else sk & (yrs == y); m2 = ns if y == "全部" else ns & (yrs == y)
            rows.append(dict(设定=nm, 年=y, 跳过日数=int(m.sum()), 未缩放均值bp=raw[lg][m].mean() * 1e4, 缩放后均值bp=sN[m].mean() * 1e4,
                             其余日_未缩放bp=raw[lg][m2].mean() * 1e4, 其余日_缩放后bp=sN[m2].mean() * 1e4))
    TB = pd.DataFrame(rows); print("\n=== B. FFA 跳过日的 Sleeve 损益(去掉 FFA 时;bp/天)==="); print(TB.to_string(index=False))
    # ---- C. 分解
    rows = []
    for nm in SPEC:
        pF, pN = R[(nm, "F")]["port"].to_numpy(), R[(nm, "N")]["port"].to_numpy(); legv = I["leg1"].to_numpy()
        H1 = np.where(sk, pN, pF)                                    # 只把跳过日加回来(用 N 模型的仓位),其余日保持 F
        H2 = np.where(sk, legv + I["ov"].to_numpy(), pN)             # 用 N 模型,但跳过日 Sleeve 置 0(只看重训的影响)
        for tag, x in (("合计 N − F", pN), ("① 只加回跳过日", H1), ("② 只换模型(其余日)", H2)):
            xs, fs = x[oi], pF[oi]; db = np.array([srv(xs[q]) - srv(fs[q]) for q in BI])
            rows.append(dict(设定=nm, 部分=tag, ΔSR=srv(xs) - srv(fs), lo=np.percentile(db, 2.5), hi=np.percentile(db, 97.5), dev=srv(xs[dev]) - srv(fs[dev]), test=srv(xs[~dev]) - srv(fs[~dev]),
                             **{str(y): srv(xs[yrs[oi] == y]) - srv(fs[yrs[oi] == y]) for y in range(2021, 2027)}))
        kF, kN = R[(nm, "F")]["k"], R[(nm, "N")]["k"]
        rows.append(dict(设定=nm, 部分=f"其余日 k 变化(N−F):多 {float((kN.kl - kF.kl)[ns].mean()):+.3f} 空 {float((kN.ks - kF.ks)[ns].mean()):+.3f}"))
    TC = pd.DataFrame(rows); print("\n=== C. 去掉 FFA 的 ΔSR 分解(组合层 OOS;相对保留 FFA)==="); print(TC.to_string(index=False))
    # ---- D. k 与 ffa_rank
    fr = X.ffa_rank.to_numpy(); q = pd.qcut(pd.Series(fr[o]).rank(method="first"), 5, labels=["Q1 最低(含跳过日)", "Q2", "Q3", "Q4", "Q5 最高"]).to_numpy()
    rows = []
    for nm in list(SPEC) + ["v4 + ffa_rank 连续特征"]:
        k = R[(nm, "N")]["k"]; kl, ks = k.kl.to_numpy()[o], k.ks.to_numpy()[o]
        d = pd.DataFrame({"q": q, "kl": kl, "ks": ks}).groupby("q", observed=True).mean()
        rows.append(dict(设定=nm, **{f"{i} k多": d.kl.iloc[j] for j, i in enumerate(d.index)}, Spearman_k多=pd.Series(kl).corr(pd.Series(fr[o]), method="spearman"),
                         Spearman_k空=pd.Series(ks).corr(pd.Series(fr[o]), method="spearman")))
    TD = pd.DataFrame(rows); print("\n=== D. 去掉 FFA 的模型:k 与 ffa_rank(FFA 预测振幅的滚动分位)的关系 ==="); print(TD.to_string(index=False))
    rq = pd.DataFrame({"q": q, "raw": raw["base"][o] * 1e4}).groupby("q", observed=True).raw.agg(["mean", "count"]); print("\n未缩放 Sleeve 日损益按 ffa_rank 五分位(bp):"); print(rq.round(2).to_string())
    # ---- E. 总表
    ref = R[("v4 E1+线性+dur_rate", "F")]["port"].to_numpy()[oi]; v3 = R[("v3 sigmoid", "F")]["port"].to_numpy()[oi]; rows = []
    for key, r_ in R.items():
        x_ = r_["port"].to_numpy()[oi]; d4 = np.array([srv(x_[q_]) - srv(ref[q_]) for q_ in BI])
        rows.append(dict(配置=f"{key[0]} | {'保留 FFA' if key[1] == 'F' else '无硬跳过'}", 组合SR_OOS=srv(x_), ΔSR_vs_v3=srv(x_) - srv(v3), ΔSR_vs_v4=srv(x_) - srv(ref),
                         lo_vs_v4=np.percentile(d4, 2.5), hi_vs_v4=np.percentile(d4, 97.5), dev_vs_v4=srv(x_[dev]) - srv(ref[dev]), test_vs_v4=srv(x_[~dev]) - srv(ref[~dev]),
                         **{str(y): srv(x_[yrs[oi] == y]) - srv(ref[yrs[oi] == y]) for y in range(2021, 2027)}))
    TE = pd.DataFrame(rows); print("\n=== E. 总表(组合 OOS 2021–26;分年列相对 v4)==="); print(TE.to_string(index=False))
    for nm, T_ in (("A", TA), ("B", TB), ("C", TC), ("D", TD), ("E", TE)): T_.to_csv(M.TAB / f"r3_ffa_check_{nm}.csv", index=False)
    print(f"总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
