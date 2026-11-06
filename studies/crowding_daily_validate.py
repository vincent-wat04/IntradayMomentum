"""H5 的验证(看到结果后追加,诊断;新配置 +2:V1a、V1b)。
  V1a 无条件:所有 OOS 日 Sleeve × m,m ∈ {1, 1.25, 1.5} 同样递归选择
  V1b 无条件、杠杆匹配:所有 OOS 日 Sleeve × (H5 的平均杠杆)
  V2  安慰剂:C_TSM_d 状态序列整体圆周平移(≥ 63 个交易日),重做 H5(含递归选 m),1000 次
  V3  其他状态:Sleeve × m 只在 TSM 低 / 中拥挤日
  V4  H5 的高拥挤日 vs 其余日:Sleeve 与核心仓的相关、各自 Sharpe(解释为什么在这些日子加 Sleeve 有效)
python3 crowding_daily_validate.py
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
import crowding_daily as CDY
import crowding_daily_strategy as CS
import disp_pred_study as DP
import liquidity_inverse_study as LI
import r2_rules_study as RR
import r3_study as R3
import sizing_models_study as M
import sizing_sigma_feed as SF
import sota_unified_stack as SU
import unified_sizing_posthoc as P
from etft.intraday.bars import load_bars

MULTS = (1.5, 1.25, 1.0)


def h5(state, oo, leg1, sl4, days, sel, ms=MULTS):
    hi = (state == 2) & oo
    return CS.recursive({m: leg1 + np.where(hi, m * sl4, sl4) for m in ms}, days, sel)


def main():
    t0 = time.time(); pd.set_option("display.width", 260, "display.float_format", lambda v: f"{v:8.3f}")
    b = load_bars("QQQ"); days = b.days; yrs = days.year
    I = SF.build_inputs(b); X = I["X"].copy(); keep = I["keep"]
    dn = DP.targets(DP.Arr(b), 30).dn; full = pd.Series(b.last_idx.to_numpy() >= 389, index=days); td = pd.Series(days.dayofweek != 0, index=days)
    sel = (td & full & X.notna().all(1) & dn.notna() & (days >= "2019-01-01")).to_numpy(); om = pd.Series(sel & (yrs >= 2021), index=days); oo = om.to_numpy()
    X["dur_rate"] = M.family_features(b, I["X"])["dur_rate"]
    _, CD, _ = CDY.build(); full_st = CD["TSM"].state; st = full_st.reindex(days).to_numpy()
    PHbE, PHfE = R3.run_PH_entry(b, RR.band_multipliers(b)[1]), R3.run_PH_entry(b, RR.band_multipliers(b)[1], fomc_only=True); Lv4 = R3.legs_k(PHbE, PHfE, b, keep)
    v4 = R3.assemble(R3.sizing(X, P.CLEAN + ["dur_rate"], Lv4, I, sel, days, "mv", "lin"), Lv4, I, om)
    leg1 = I["leg1"].to_numpy(); sl4 = v4["sleeve"].to_numpy(); base = leg1 + sl4
    oi = np.where(oo)[0]; dev = yrs[oi] <= 2022; srv = lambda v: v.mean() / v.std() * np.sqrt(252); BI = M.block_idx(len(oi), 10, 2000, 0)
    def row(nm, x):
        xs, rs = x[oi], base[oi]; db = np.array([srv(xs[q]) - srv(rs[q]) for q in BI])
        return dict(方案=nm, 组合SR=srv(xs), ΔSR=srv(xs) - srv(rs), lo=np.percentile(db, 2.5), hi=np.percentile(db, 97.5), dev=srv(xs[dev]) - srv(rs[dev]), test=srv(xs[~dev]) - srv(rs[~dev]),
                    MDD=float(((1 + pd.Series(xs)).cumprod() / (1 + pd.Series(xs)).cumprod().cummax() - 1).min()))
    obs, pk = h5(st, oo, leg1, sl4, days, sel); hi = (st == 2) & oo
    mon = days.to_period("M"); mm = np.array([pk.get(str(m_), 1.0) for m_ in mon]); lev = np.where(hi, mm, 1.0)[oo].mean()
    rows = [row("H5(复现)", obs)]
    v1a, pka = CS.recursive({m: leg1 + np.where(oo, m * sl4, sl4) for m in MULTS}, days, sel); rows.append(row(f"V1a 无条件 × m(递归;选择分布 {pd.Series(pka).value_counts().to_dict()})", v1a))
    rows.append(row(f"V1b 无条件 × {lev:.3f}(= H5 平均杠杆)", leg1 + np.where(oo, lev * sl4, sl4)))
    for k_, lab in ((0, "低"), (1, "中")):
        x_, _ = CS.recursive({m: leg1 + np.where((st == k_) & oo, m * sl4, sl4) for m in MULTS}, days, sel); rows.append(row(f"V3 只在 TSM {lab}拥挤日 × m", x_))
    T = pd.DataFrame(rows); print("=== H5 与对照(组合 OOS 2021–26,ΔSR 相对 v4)==="); print(T.to_string(index=False))
    d_obs = srv(obs[oi]) - srv(base[oi]); d_v1b = srv(obs[oi]) - srv((leg1 + np.where(oo, lev * sl4, sl4))[oi])
    sv = full_st.to_numpy(); cal = full_st.index; rg = np.random.default_rng(29); nul = []; nul_rel = []
    v1b = (leg1 + np.where(oo, lev * sl4, sl4))[oi]
    for _ in range(1000):
        k = int(rg.integers(63, len(sv) - 63)); sh = pd.Series(np.roll(sv, k), index=cal).reindex(days).to_numpy()
        x_, _ = h5(sh, oo, leg1, sl4, days, sel); nul.append(srv(x_[oi]) - srv(base[oi])); nul_rel.append(srv(x_[oi]) - srv(v1b))
    nul, nul_rel = np.array(nul), np.array(nul_rel)
    print(f"\n=== V2 安慰剂(C_TSM_d 状态圆周平移 ≥ 63 日,1000 次,含递归选 m)===")
    print(f"相对 v4:观测 {d_obs:+.3f};安慰剂中位 {np.median(nul):+.3f}、95% {np.quantile(nul, .95):+.3f};p = {(np.sum(nul >= d_obs) + 1) / 1001:.3f}")
    print(f"相对杠杆匹配的无条件版本(V1b):观测 {d_v1b:+.3f};安慰剂中位 {np.median(nul_rel):+.3f}、95% {np.quantile(nul_rel, .95):+.3f};p = {(np.sum(nul_rel >= d_v1b) + 1) / 1001:.3f}")
    rows = []
    for lab, m in (("TSM 高拥挤日", hi), ("其余 OOS 日", oo & ~hi)):
        rows.append(dict(日子=lab, 天数=int(m.sum()), Sleeve_bp=sl4[m].mean() * 1e4, Sleeve_SR=srv(sl4[m]), 核心仓_bp=leg1[m].mean() * 1e4, 核心仓_SR=srv(leg1[m]),
                         Sleeve与核心仓相关=float(np.corrcoef(sl4[m], leg1[m])[0, 1]), Sleeve波动年化=float(sl4[m].std() * np.sqrt(252)), 核心仓波动年化=float(leg1[m].std() * np.sqrt(252))))
    print("\n=== V4 为什么在 TSM 高拥挤日加 Sleeve 有效(OOS 2021–26)==="); print(pd.DataFrame(rows).to_string(index=False))
    rows = []
    for y in range(2021, 2027):
        m = hi & (yrs == y); m2 = oo & ~hi & (yrs == y)
        rows.append(dict(年=y, 高拥挤日数=int(m.sum()), 高拥挤日_Sleeve_bp=sl4[m].mean() * 1e4 if m.any() else np.nan, 其余日_Sleeve_bp=sl4[m2].mean() * 1e4, 高拥挤日_相关=float(np.corrcoef(sl4[m], leg1[m])[0, 1]) if m.sum() > 10 else np.nan))
    print(pd.DataFrame(rows).to_string(index=False))
    T.to_csv(M.TAB / "crowding_daily_validate.csv", index=False); print(f"总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
