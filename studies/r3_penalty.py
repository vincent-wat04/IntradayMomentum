"""第三轮事后追加 3(2026-10-23;累计 +2):固定 E1 入场带 + CLEAN+dur_rate + 线性链接,只换风险惩罚:
  P-mv   均值–方差(= K1)
  P-semi 组合下行半方差 mean(min(R,0)²)
  P-lpm  只罚 10% 分位以下的亏损 mean(min(R−τ,0)²)
配对比较(相对 P-mv)、分年、分布面板、k 的形状。
python3 r3_penalty.py
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
import tearsheet as TS
import unified_sizing_posthoc as P
from etft.intraday.bars import load_bars


def main():
    t0 = time.time(); pd.set_option("display.width", 280, "display.max_rows", 200, "display.float_format", lambda v: f"{v:8.3f}")
    b = load_bars("QQQ"); days = b.days; yrs = days.year
    I = SF.build_inputs(b); X = I["X"].copy()
    dn = DP.targets(DP.Arr(b), 30).dn; full = pd.Series(b.last_idx.to_numpy() >= 389, index=days); td = pd.Series(days.dayofweek != 0, index=days)
    sel = (td & full & X.notna().all(1) & dn.notna() & (days >= "2019-01-01")).to_numpy(); om = pd.Series(sel & (yrs >= 2021), index=days)
    _, S3m, _ = RR.band_multipliers(b); X["dur_rate"] = M.family_features(b, I["X"])["dur_rate"]
    PHb, PHf = LI.run_phases_fills(b), SU.fomc_late_PH(b); PHbE, PHfE = R3.run_PH_entry(b, S3m), R3.run_PH_entry(b, S3m, fomc_only=True)
    Lb_, LE = R3.legs_k(PHb, PHf, b, I["keep"]), R3.legs_k(PHbE, PHfE, b, I["keep"]); F = P.CLEAN + ["dur_rate"]
    R = {"v3 现行": R3.assemble(R3.sizing(X, P.CLEAN, Lb_, I, sel, days, "mv", "sig"), Lb_, I, om)}
    for nm, obj in (("P-mv 均值–方差(K1)", "mv"), ("P-semi 下行半方差", "semi"), ("P-lpm 只罚 10% 分位以下", "lpm")):
        R[nm] = R3.assemble(R3.sizing(X, F, LE, I, sel, days, obj, "lin"), LE, I, om)
    print(f"[完成] ({time.time()-t0:.0f}s)")
    oi = np.where(om.to_numpy())[0]; dev = yrs[oi] <= 2022; BI = M.block_idx(len(oi), 10, 2000, 0); srv = lambda v: v.mean() / v.std() * np.sqrt(252)
    v3 = R["v3 现行"]["port"].to_numpy()[oi]; mv = R["P-mv 均值–方差(K1)"]["port"].to_numpy()[oi]
    rows = []
    for nm, r_ in R.items():
        x_ = r_["port"].to_numpy()[oi]
        d3 = np.array([srv(x_[q]) - srv(v3[q]) for q in BI]); dm = np.array([srv(x_[q]) - srv(mv[q]) for q in BI])
        k = r_["k"]; kk = np.r_[k.kl[om].to_numpy(), k.ks[om].to_numpy()]
        rows.append(dict(方案=nm, 组合SR_OOS=srv(x_), ΔSR_vs_v3=srv(x_) - srv(v3), v3_lo=np.percentile(d3, 2.5), v3_hi=np.percentile(d3, 97.5),
                         ΔSR_vs_mv=srv(x_) - srv(mv), mv_lo=np.percentile(dm, 2.5), mv_hi=np.percentile(dm, 97.5),
                         dev_vs_mv=srv(x_[dev]) - srv(mv[dev]), test_vs_mv=srv(x_[~dev]) - srv(mv[~dev]),
                         k多均值=float(k.kl[om].mean()), k空均值=float(k.ks[om].mean()), k标准差=float(kk.std()), k在边界=float(np.mean((kk <= 1e-6) | (kk >= 2 - 1e-6))),
                         **{str(y): srv(x_[yrs[oi] == y]) - srv(mv[yrs[oi] == y]) for y in range(2021, 2027)}))
    T = pd.DataFrame(rows); print("\n=== 组合层 OOS 2021–26(分年列为相对 P-mv)==="); print(T.to_string(index=False))
    pan = {nm: SF.panel(pd.Series(r_["port"].to_numpy()[oi])) for nm, r_ in R.items()}
    pan.update({"Sleeve | " + nm: SF.panel(pd.Series(r_["sleeve"].to_numpy()[oi])) for nm, r_ in R.items()})
    print("\n=== 分布面板(OOS 2021–26)==="); print(pd.DataFrame(pan).T.to_string())
    KEYS = ["年化收益 CAGR", "年化波动", "Sharpe（绝对）", "Sortino", "最大回撤 MDD", "Calmar", "最长水下（交易日）", "最差单日", "CVaR95（日）"]
    for pn, a, z in R3.PER:
        print(f"\n===== tearsheet {pn} =====")
        print(pd.DataFrame({nm: {k_: TS.metrics(r_["port"].loc[a:z])[k_] for k_ in KEYS} for nm, r_ in R.items()}).to_string())
    # 风险项差异来自哪里:按组合日损益所在区间,两者 k 的差
    mv_k, se_k = R["P-mv 均值–方差(K1)"]["k"], R["P-semi 下行半方差"]["k"]; base_day = (LE["L2"] + LE["S2"]).to_numpy()[oi] * 1e4
    q = pd.qcut(pd.Series(base_day).rank(method="first"), 5, labels=["最差 20%", "20–40%", "40–60%", "60–80%", "最好 20%"]).to_numpy()
    D = pd.DataFrame({"区间(未缩放 Sleeve 日损益)": q, "Δk多(semi−mv)": (se_k.kl - mv_k.kl).to_numpy()[oi], "Δk空(semi−mv)": (se_k.ks - mv_k.ks).to_numpy()[oi]})
    print("\n=== semi 相对 mv 在哪类日子加/减仓(事后分组,描述性)==="); print(D.groupby("区间(未缩放 Sleeve 日损益)", observed=True).mean().round(3).to_string())
    T.to_csv(M.TAB / "r3_penalty.csv", index=False); print(f"总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
