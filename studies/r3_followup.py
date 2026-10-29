"""第三轮事后追加 2(2026-10-23;累计 +4):
  1) 检验"候选的增量来自同一机制(仓位调得更积极)"这一说法:各配置 OOS 的 k 离散度、相对 v3 的增量日损益之间的相关
  2) 在仓位模型改动(L2、O2-lin)之上叠加 rv_surp:入场带(E1)或整笔过滤(跳过 rv_surp 因果分位最高 20%,同第二轮 C1)
python3 r3_followup.py
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
from b0_econ import causal_rank
from etft.intraday.bars import load_bars
from event_calendar import FOMC_DECISION


def rv_skip_set(b, PHb, PHf, A):
    days = b.days; cols = list(b.c.columns); tix = {t: i for i, t in enumerate(cols)}; dpos = {d: i for i, d in enumerate(days)}
    fm = set(pd.DatetimeIndex(FOMC_DECISION)); T = []
    for src, PH in (("b", PHb), ("f", PHf)):
        for off, (rr, F) in PH.items():
            if len(F): t_ = LI.trades_from_fills(F, off); t_["src"] = src; T.append(t_)
    T = pd.concat(T, ignore_index=True); T["day"] = pd.to_datetime(T.day)
    T = T[((T.src == "b") & ~T.day.isin(fm)) | ((T.src == "f") & T.day.isin(fm))].sort_values(["day", "phase", "t_in"]).reset_index(drop=True)
    ii = T.day.map(dpos).to_numpy(); kk = T.t_in.map(tix).to_numpy(); x = -np.log(A["cRV"][ii, kk] / A["S_rv"][ii, kk])
    rk = causal_rank(x, T.day.to_numpy()); m = np.isfinite(x) & (rk <= 0.20)
    return set(zip(T.phase[m], T.day[m], T.t_in[m]))


def main():
    t0 = time.time(); pd.set_option("display.width", 280, "display.max_rows", 200, "display.float_format", lambda v: f"{v:8.3f}")
    b = load_bars("QQQ"); days = b.days; yrs = days.year
    I = SF.build_inputs(b); X = I["X"].copy(); keepF = I["keep"]
    dn = DP.targets(DP.Arr(b), 30).dn; full = pd.Series(b.last_idx.to_numpy() >= 389, index=days); td = pd.Series(days.dayofweek != 0, index=days)
    sel = (td & full & X.notna().all(1) & dn.notna() & (days >= "2019-01-01")).to_numpy(); om = pd.Series(sel & (yrs >= 2021), index=days)
    _, S3m, A = RR.band_multipliers(b)
    X["rv_surp_1000"] = pd.Series(np.log(A["cRV"][:, 29] / A["S_rv"][:, 29]), index=days).fillna(0.0); X["dur_rate"] = M.family_features(b, I["X"])["dur_rate"]
    CR2 = [("rv_surp_1000" if f == "b30_n" else f) for f in P.CLEAN]
    PHb, PHf = LI.run_phases_fills(b), SU.fomc_late_PH(b); PHbE, PHfE = R3.run_PH_entry(b, S3m), R3.run_PH_entry(b, S3m, fomc_only=True)
    sk = rv_skip_set(b, PHb, PHf, A)
    LG = {"base": R3.legs_k(PHb, PHf, b, keepF), "E1": R3.legs_k(PHbE, PHfE, b, keepF),
          "skip": R3.legs_k(RR.filter_PH(PHb, sk), RR.filter_PH(PHf, sk), b, keepF)}
    SPEC = {"v3 现行": ("base", P.CLEAN, "mv", "sig"), "L1 线性": ("base", P.CLEAN, "mv", "lin"), "L2 线性+dur_rate": ("base", P.CLEAN + ["dur_rate"], "mv", "lin"),
            "O2 半方差": ("base", P.CLEAN, "semi", "sig"), "O2-lin 半方差+线性": ("base", P.CLEAN, "semi", "lin"),
            "T1 O2-lin+dur_rate": ("base", P.CLEAN + ["dur_rate"], "semi", "lin"), "S5 全部叠加": ("E1", CR2 + ["dur_rate"], "semi", "lin"),
            "E1 入场带(sigmoid)": ("E1", P.CLEAN, "mv", "sig"), "C1 rv_surp 跳过(sigmoid)": ("skip", P.CLEAN, "mv", "sig"),
            "K1 L2 + E1 入场带": ("E1", P.CLEAN + ["dur_rate"], "mv", "lin"), "K2 L2 + rv_surp 跳过": ("skip", P.CLEAN + ["dur_rate"], "mv", "lin"),
            "K3 O2-lin + E1 入场带": ("E1", P.CLEAN, "semi", "lin"), "K4 O2-lin + rv_surp 跳过": ("skip", P.CLEAN, "semi", "lin")}
    R = {}
    for nm, (lg, ft, obj, lk) in SPEC.items():
        R[nm] = R3.assemble(R3.sizing(X, ft, LG[lg], I, sel, days, obj, lk), LG[lg], I, om)
    print(f"[完成] {len(R)} 个配置 ({time.time()-t0:.0f}s)", flush=True)
    oi = np.where(om.to_numpy())[0]; dev = yrs[oi] <= 2022; BI = M.block_idx(len(oi), 10, 2000, 0); srv = lambda v: v.mean() / v.std() * np.sqrt(252)
    ref = R["v3 现行"]["port"].to_numpy()[oi]; refb = np.array([srv(ref[q]) for q in BI])
    # ---- 1. 机制:k 的离散度
    rows = []
    for nm, r_ in R.items():
        k = r_["k"]; kl, ks = k.kl[om], k.ks[om]
        rows.append(dict(配置=nm, k多_均值=kl.mean(), k多_标准差=kl.std(), k空_均值=ks.mean(), k空_标准差=ks.std(),
                         平均绝对logk=float(np.mean(np.abs(np.log(np.clip(np.r_[kl.to_numpy(), ks.to_numpy()], 1e-3, None))))),
                         k在边界比例=float(np.mean((np.r_[kl.to_numpy(), ks.to_numpy()] <= 1e-6) | (np.r_[kl.to_numpy(), ks.to_numpy()] >= 2 - 1e-6)))))
    K1t = pd.DataFrame(rows); print("\n=== 1a. 仓位乘数 k 的离散度(OOS 2021–26)==="); print(K1t.to_string(index=False))
    inc = pd.DataFrame({nm: R[nm]["port"].to_numpy()[oi] - ref for nm in R if nm != "v3 现行"})
    print("\n=== 1b. 相对 v3 的增量日损益之间的相关(OOS 2021–26)==="); print(inc[["L1 线性", "L2 线性+dur_rate", "O2 半方差", "O2-lin 半方差+线性", "E1 入场带(sigmoid)", "C1 rv_surp 跳过(sigmoid)"]].corr().round(2).to_string())
    kd = pd.DataFrame({nm: np.r_[R[nm]["k"].kl[om].to_numpy(), R[nm]["k"].ks[om].to_numpy()] - np.r_[R["v3 现行"]["k"].kl[om].to_numpy(), R["v3 现行"]["k"].ks[om].to_numpy()]
                       for nm in ("L1 线性", "L2 线性+dur_rate", "O2 半方差", "O2-lin 半方差+线性")})
    print("\n=== 1c. 相对 v3 的 k 改变量(Δk)之间的相关 ==="); print(kd.corr().round(2).to_string())
    # ---- 2. 叠加 rv_surp
    rows = []
    for nm, r_ in R.items():
        x_ = r_["port"].to_numpy()[oi]; db = np.array([srv(x_[q]) for q in BI]) - refb
        rows.append(dict(方案=nm, 组合SR_OOS=srv(x_), ΔSR_vs_v3=srv(x_) - srv(ref), lo=np.percentile(db, 2.5), hi=np.percentile(db, 97.5),
                         dev=srv(x_[dev]) - srv(ref[dev]), test=srv(x_[~dev]) - srv(ref[~dev]),
                         test期SR=TS.metrics(r_["port"].loc["2023":"2026"])["Sharpe（绝对）"], test期MDD=TS.metrics(r_["port"].loc["2023":"2026"])["最大回撤 MDD"],
                         全期MDD=TS.metrics(r_["port"].loc["2019":"2026"])["最大回撤 MDD"], Sleeve最差日bp=float(r_["sleeve"].to_numpy()[oi].min() * 1e4),
                         **{str(y): srv(x_[yrs[oi] == y]) - srv(ref[yrs[oi] == y]) for y in range(2021, 2027)}))
    T2 = pd.DataFrame(rows); print("\n=== 2. 组合层 OOS 2021–26(ΔSR 相对 v3)==="); print(T2.to_string(index=False))
    rows = []
    for top, base_ in (("K1 L2 + E1 入场带", "L2 线性+dur_rate"), ("K2 L2 + rv_surp 跳过", "L2 线性+dur_rate"),
                       ("K3 O2-lin + E1 入场带", "O2-lin 半方差+线性"), ("K4 O2-lin + rv_surp 跳过", "O2-lin 半方差+线性"),
                       ("E1 入场带(sigmoid)", "v3 现行"), ("C1 rv_surp 跳过(sigmoid)", "v3 现行")):
        a_, b_ = R[base_]["port"].to_numpy()[oi], R[top]["port"].to_numpy()[oi]; db = np.array([srv(b_[q]) - srv(a_[q]) for q in BI])
        rows.append(dict(叠加=top, 在其之上=base_, 本步ΔSR=srv(b_) - srv(a_), lo=np.percentile(db, 2.5), hi=np.percentile(db, 97.5), dev=srv(b_[dev]) - srv(a_[dev]), test=srv(b_[~dev]) - srv(a_[~dev])))
    T3 = pd.DataFrame(rows); print("\n=== 3. rv_surp 在不同仓位模型之上的边际贡献 ==="); print(T3.to_string(index=False))
    K1t.to_csv(M.TAB / "r3_followup_k.csv", index=False); T2.to_csv(M.TAB / "r3_followup.csv", index=False); T3.to_csv(M.TAB / "r3_followup_marginal.csv", index=False)
    print(f"总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
