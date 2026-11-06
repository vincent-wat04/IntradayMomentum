"""第三轮事后追加(看到 r3_study 结果后,2026-10-23;累计 +4):三个最接近的候选叠加 + "要不要去掉 FFA-P 跳过"的专门检验。
叠加:
  T1 = O2-lin ∪ L2 = 半方差目标 + 线性链接 + dur_rate(成交不变,保留 FFA)
  T2 = T1 去掉 FFA
  S5/S6 = 三者全部叠加(E1 入场带 + rv_surp 替换 b30_n + dur_rate + 线性 + O2;S6 去 FFA)—— 与 r3_study 相同,重算作对照
FFA 检验(描述 + 配对):
  1) 同一仓位设定下有/无 FFA 的配对 ΔSR(5 种设定:v3、L2、T1、O2-lin、S5),dev/test/分年
  2) 被 FFA 跳过的日子本身:不跳过时 Sleeve 在这些日子的平均损益(bp)与胜率,分年;与"随机跳过同样多的日子"对比
python3 r3_stack.py
"""
from __future__ import annotations

import os
import sys
import time
import warnings

warnings.filterwarnings("ignore"); os.environ.setdefault("OMP_NUM_THREADS", "1")
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
from cjkfont import use_cjk
from etft.intraday.bars import load_bars


def main():
    t0 = time.time(); pd.set_option("display.width", 280, "display.max_rows", 200, "display.float_format", lambda v: f"{v:8.3f}")
    b = load_bars("QQQ"); days = b.days; yrs = days.year
    I = SF.build_inputs(b); X = I["X"].copy(); keepF = I["keep"]; keep1 = pd.Series(1.0, index=days)
    dn = DP.targets(DP.Arr(b), 30).dn; full = pd.Series(b.last_idx.to_numpy() >= 389, index=days); td = pd.Series(days.dayofweek != 0, index=days)
    sel = (td & full & X.notna().all(1) & dn.notna() & (days >= "2019-01-01")).to_numpy(); om = pd.Series(sel & (yrs >= 2021), index=days)
    _, S3m, A = RR.band_multipliers(b)
    X["rv_surp_1000"] = pd.Series(np.log(A["cRV"][:, 29] / A["S_rv"][:, 29]), index=days).fillna(0.0)
    X["dur_rate"] = M.family_features(b, I["X"])["dur_rate"]
    CR2 = [("rv_surp_1000" if f == "b30_n" else f) for f in P.CLEAN]
    PHb, PHf = LI.run_phases_fills(b), SU.fomc_late_PH(b); PHbE, PHfE = R3.run_PH_entry(b, S3m), R3.run_PH_entry(b, S3m, fomc_only=True)
    LG = {("base", 1): R3.legs_k(PHb, PHf, b, keepF), ("base", 0): R3.legs_k(PHb, PHf, b, keep1),
          ("E1", 1): R3.legs_k(PHbE, PHfE, b, keepF), ("E1", 0): R3.legs_k(PHbE, PHfE, b, keep1)}
    SPEC = {"v3 现行": ("base", P.CLEAN, "mv", "sig"), "L2 线性+dur_rate": ("base", P.CLEAN + ["dur_rate"], "mv", "lin"),
            "O2-lin": ("base", P.CLEAN, "semi", "lin"), "T1 O2-lin+dur_rate": ("base", P.CLEAN + ["dur_rate"], "semi", "lin"),
            "S5 全部叠加": ("E1", CR2 + ["dur_rate"], "semi", "lin")}
    R = {}
    for nm, (lg, ft, obj, lk) in SPEC.items():
        for f in (1, 0):
            L = LG[(lg, f)]; R[(nm, f)] = R3.assemble(R3.sizing(X, ft, L, I, sel, days, obj, lk), L, I, om)
    print(f"[完成] {len(R)} 个组合 ({time.time()-t0:.0f}s)", flush=True)
    oi = np.where(om.to_numpy())[0]; dev = yrs[oi] <= 2022; BI = M.block_idx(len(oi), 10, 2000, 0); srv = lambda v: v.mean() / v.std() * np.sqrt(252)
    ref = R[("v3 现行", 1)]["port"].to_numpy()[oi]; refb = np.array([srv(ref[q]) for q in BI])
    # ---- 1. 叠加结果(相对 v3)
    rows = []
    for (nm, f), r_ in R.items():
        x_ = r_["port"].to_numpy()[oi]; db = np.array([srv(x_[q]) for q in BI]) - refb
        rows.append(dict(方案=nm + ("" if f else " −FFA"), 组合SR_OOS=srv(x_), ΔSR_vs_v3=srv(x_) - srv(ref), lo=np.percentile(db, 2.5), hi=np.percentile(db, 97.5),
                         dev=srv(x_[dev]) - srv(ref[dev]), test=srv(x_[~dev]) - srv(ref[~dev]),
                         **{str(y): srv(x_[yrs[oi] == y]) - srv(ref[yrs[oi] == y]) for y in range(2021, 2027)}))
    T1 = pd.DataFrame(rows); print("\n=== 1. 叠加(组合 OOS 2021–26,相对 v3 = 1.766)==="); print(T1.to_string(index=False))
    # ---- 2. FFA 配对:同一仓位设定,去 FFA 相对保留 FFA
    rows = []
    for nm in SPEC:
        a_, b_ = R[(nm, 1)]["port"].to_numpy()[oi], R[(nm, 0)]["port"].to_numpy()[oi]; db = np.array([srv(b_[q]) - srv(a_[q]) for q in BI])
        rows.append(dict(仓位设定=nm, 去FFA_ΔSR=srv(b_) - srv(a_), lo=np.percentile(db, 2.5), hi=np.percentile(db, 97.5), P_ΔSR大于0=float((db > 0).mean()),
                         dev=srv(b_[dev]) - srv(a_[dev]), test=srv(b_[~dev]) - srv(a_[~dev]), **{str(y): srv(b_[yrs[oi] == y]) - srv(a_[yrs[oi] == y]) for y in range(2021, 2027)}))
    T2 = pd.DataFrame(rows); print("\n=== 2. 去掉 FFA-P 跳过的配对 ΔSR(同一仓位设定,组合 OOS 2021–26)==="); print(T2.to_string(index=False))
    # ---- 3. 被跳过的日子本身
    s_noffa = R[("v3 现行", 0)]["sleeve"]; s_raw = LG[("base", 0)]["L2"] + LG[("base", 0)]["S2"]
    sk = (keepF == 0) & om; ot = om & ~sk & (s_raw != 0)
    print("\n=== 3. FFA-P 跳过的日子(OOS 2021–26,不跳过时的 Sleeve 损益)===")
    rows = []
    for y in list(range(2021, 2027)) + ["全部"]:
        m1 = sk if y == "全部" else sk & (yrs == y); m2 = ot if y == "全部" else ot & (yrs == y)
        rows.append(dict(年=y, 跳过日数=int(m1.sum()), 跳过日_未缩放均值bp=float(s_raw[m1].mean() * 1e4), 跳过日_胜率=float((s_raw[m1] > 0).mean()),
                         跳过日_仓位缩放后均值bp=float(s_noffa[m1].mean() * 1e4), 其余交易日_未缩放均值bp=float(s_raw[m2].mean() * 1e4)))
    T3 = pd.DataFrame(rows); print(T3.to_string(index=False))
    rg = np.random.default_rng(5); cand = np.where((om & (s_raw != 0)).to_numpy())[0]; n_sk = int(sk.sum()); sv = s_raw.to_numpy()
    nul = np.array([sv[rg.choice(cand, n_sk, replace=False)].mean() for _ in range(5000)]) * 1e4
    print(f"跳过日未缩放均值 {s_raw[sk].mean()*1e4:+.2f}bp;随机抽同样多的交易日:中位 {np.median(nul):+.2f}bp,5% 分位 {np.quantile(nul,.05):+.2f}bp;p(随机 ≤ 观测) = {float((nul <= s_raw[sk].mean()*1e4).mean()):.3f}")
    # ---- 4. tearsheet
    bh = I["bh"]; show = {"QQQ 买入持有": bh, "v3 现行": R[("v3 现行", 1)]["port"], "T1 O2-lin+dur_rate": R[("T1 O2-lin+dur_rate", 1)]["port"],
                          "T2 T1 −FFA": R[("T1 O2-lin+dur_rate", 0)]["port"], "S5 全部叠加": R[("S5 全部叠加", 1)]["port"], "S6 全部叠加 −FFA": R[("S5 全部叠加", 0)]["port"]}
    KEYS = ["年化收益 CAGR", "年化波动", "Sharpe（绝对）", "Sortino", "最大回撤 MDD", "Calmar", "最长水下（交易日）", "最差单日", "偏度", "超额峰度", "CVaR95（日）", "α 的 t 值 (HAC)"]
    for pn, a, z in R3.PER:
        Tt = pd.DataFrame({nm: {k_: TS.metrics(x.loc[a:z], bench=None if nm.startswith("QQQ") else bh.loc[a:z]).get(k_, np.nan) for k_ in KEYS} for nm, x in show.items()})
        print(f"\n===== tearsheet {pn} ====="); print(Tt.to_string())
    yr = pd.DataFrame({nm: {y: TS.metrics(x.loc[str(y)])["Sharpe（绝对）"] for y in range(2019, 2027)} for nm, x in show.items()}); print("\n===== 分年 Sharpe ====="); print(yr.round(3).to_string())
    use_cjk(plt); fig, ax = plt.subplots(2, 1, figsize=(15, 10), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    col = {"QQQ 买入持有": "black", "v3 现行": "tab:green", "T1 O2-lin+dur_rate": "tab:blue", "T2 T1 −FFA": "tab:cyan", "S5 全部叠加": "tab:purple", "S6 全部叠加 −FFA": "tab:orange"}
    for nm, x in show.items():
        x = x.loc["2019":]; eq = (1 + x).cumprod(); te = TS.metrics(x.loc["2023":])
        ax[0].plot(eq, color=col[nm], lw=1.4 if nm.startswith("QQQ") else 1.9, ls=":" if nm.startswith("QQQ") else ("--" if "FFA" in nm else "-"),
                   label=f"{nm}(test Sharpe {te['Sharpe（绝对）']:.2f}, test MDD {te['最大回撤 MDD']:.1%})")
        ax[1].plot(eq / eq.cummax() - 1, color=col[nm], lw=1.0, ls=":" if nm.startswith("QQQ") else ("--" if "FFA" in nm else "-"))
    for a_ in ax: a_.axvline(pd.Timestamp("2023-01-01"), color="gray", ls="--", lw=1); a_.grid(alpha=.3)
    ax[0].set_yscale("log"); ax[0].legend(fontsize=9, loc="upper left"); ax[0].set_title("候选叠加 vs v3(虚线 = 去掉 FFA-P 跳过)", loc="left"); ax[1].set_title("回撤", loc="left")
    figp = Path("../outputs/figs/r3_stack.png"); fig.savefig(figp, dpi=120, bbox_inches="tight")
    T1.to_csv(M.TAB / "r3_stack.csv", index=False); T2.to_csv(M.TAB / "r3_ffa_paired.csv", index=False); T3.to_csv(M.TAB / "r3_ffa_skipdays.csv", index=False); yr.to_csv(M.TAB / "r3_stack_yearly.csv")
    print(f"图:{figp};总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
