"""第三轮(docs/PROPOSAL_intraday_vol_entry_r3.md,预注册):入场带自适应、O2 迭代、替换 v3 中重叠的组件、新方法版 v3 的 tearsheet。
python3 r3_study.py
"""
from __future__ import annotations

import inspect
import os
import sys
import textwrap
import time
import warnings

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
warnings.filterwarnings("ignore")
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import disp_pred_study as DP
import ivol_objective_study as AO
import letf_posthoc as LP
import liquidity_inverse_study as LI
import r2_rules_study as RR
import sizing_models_study as M
import sizing_sigma_feed as SF
import sleeve_anatomy as SA
import sota_unified_stack as SU
import tearsheet as TS
import unified_sizing_posthoc as P
import unified_sizing_study as U
from audit import reimpl_noise as RI
from cjkfont import use_cjk
from etft.intraday.bars import load_bars
from event_calendar import FOMC_DECISION

OUT, TAB = SF.OUT, M.TAB
PER = (("dev 2019–22", "2019", "2022"), ("test 2023–26", "2023", "2026"), ("全 2019–26", "2019", "2026"))


# ============================================================ 入场带与止损带分离(源码变换,不改引擎文件)
def make_entry_sim():
    src = textwrap.dedent(inspect.getsource(RI.simulate))
    reps = [("sides=\"both\", skip_dows=(), skip_days=None):", "sides=\"both\", skip_dows=(), skip_days=None, entry_mult=None):"),
            ("        lb = lo_ref.loc[d] * (1 - vm_s * s_row)      # 下轨用空头的 vm\n",
             "        lb = lo_ref.loc[d] * (1 - vm_s * s_row)      # 下轨用空头的 vm\n"
             "        if entry_mult is None:\n            ub_e, lb_e = ub, lb\n        else:\n"
             "            _m = entry_mult.loc[d].to_numpy(); ub_e = hi_ref.loc[d] * (1 + vm_l * s_row * _m); lb_e = lo_ref.loc[d] * (1 - vm_s * s_row * _m)\n"),
            ("            if shares == 0 and not need_rearm and k >= no_entry_before and side_now != 0:\n",
             "            side_e = 1 if px_now > ub_e[k] else (-1 if px_now < lb_e[k] else 0)\n"
             "            if (sides == \"long\" and side_e < 0) or (sides == \"short\" and side_e > 0):\n                side_e = 0\n"
             "            if shares == 0 and not need_rearm and k >= no_entry_before and side_e != 0:\n")]
    for a, b_ in reps:
        assert src.count(a) == 1, f"源码定位失败:{a[:60]}"; src = src.replace(a, b_)
    blk_start = src.index("if shares == 0 and not need_rearm and k >= no_entry_before and side_e != 0:")
    blk_end = src.index("# ---- 收盘 MOC 平仓")
    blk = src[blk_start:blk_end].replace("side_now", "side_e")
    src = src[:blk_start] + blk + src[blk_end:]
    src = src.replace("def simulate(", "def simulate_entry(", 1)
    ns = dict(RI.__dict__); exec(compile(src, "<simulate_entry>", "exec"), ns); return ns["simulate_entry"]


SIM_E = make_entry_sim()


def run_PH_entry(b, mult, fomc_only=False):
    days = b.days; out = {}
    if fomc_only:
        fs = set(d for d in FOMC_DECISION if d in set(days)); nonf = [d for d in days if d not in fs]
    for off in range(30):
        if fomc_only:
            rr, F = SIM_E(b, checks=[t for t in SA.chk(off) if t >= "14:30"], sizing="voltarget", max_lev=4.0, reentry="hold",
                          half_spread=0.005, commission=0.0045, skip_days=nonf, entry_mult=mult, **SA.BASE)
        else:
            rr, F = SIM_E(b, checks=SA.chk(off), sizing="voltarget", max_lev=4.0, reentry="hold", half_spread=LI.HS0, commission=LI.CM0, entry_mult=mult, **SA.BASE)
        F = F.copy()
        if len(F): F["day"] = pd.to_datetime(F.day)
        out[off] = (rr, F)
    return out


def legs_k(PHb, PHf, b, keep):
    days = b.days; scale = (b.day_close / LI.tqqq_traded_px(days)) / 3.0
    Lb0, Sb0 = SU.legs_from_PH(PHb, days, scale); Lf, Sf = SU.legs_from_PH(PHf, days, scale)
    fm = pd.Series(days.isin(pd.DatetimeIndex(FOMC_DECISION)), index=days)
    cols = list(b.c.columns); ob = b.o.to_numpy()
    GLb, GSb = LP.G_series(PHb, days, ob, cols, "15:40"); GLf, GSf = LP.G_series(PHf, days, ob, cols, "15:40")
    return dict(L2=Lb0.where(~fm, Lf) * keep, S2=Sb0.where(~fm, Sf) * keep, GL=GLb.where(~fm, GLf), GS=GSb.where(~fm, GSf), keep=keep)


def sizing(X, feats, legs, I, sel, days, obj="mv", link="sig"):
    Lb, Sb, Eb = (legs["L2"] * 1e4).to_numpy(), (legs["S2"] * 1e4).to_numpy(), ((I["leg1"] + I["ov"]) * 1e4).to_numpy()
    if obj == "mv" and link == "sig":
        return P.wf(X, pd.Series(Lb, index=days), pd.Series(Sb, index=days), pd.Series(Eb, index=days), sel, days, feats)[0]
    Q = pd.DataFrame({"L": Lb, "S": Sb, "E": Eb}, index=days)
    return AO.wf_obj(X, feats, Lb, Sb, Eb, Q, sel, days, obj, link)[0]


def assemble(k, legs, I, om):
    kl = k.kl.where(om, 1.0).fillna(1.0); ks = k.ks.where(om, 1.0).fillna(1.0)
    sl = kl * legs["L2"] + ks * legs["S2"] + I["ov"] - (kl * legs["GL"] + ks * legs["GS"]) * legs["keep"]
    return dict(sleeve=sl, port=I["leg1"] + sl, k=k)


def main():
    t0 = time.time(); pd.set_option("display.width", 280, "display.max_rows", 200, "display.float_format", lambda v: f"{v:8.3f}")
    b = load_bars("QQQ"); days = b.days; yrs = days.year
    I = SF.build_inputs(b); X = I["X"].copy(); keep_ffa = I["keep"]; keep1 = pd.Series(1.0, index=days)
    dn = DP.targets(DP.Arr(b), 30).dn; full = pd.Series(b.last_idx.to_numpy() >= 389, index=days); td = pd.Series(days.dayofweek != 0, index=days)
    sel = (td & full & X.notna().all(1) & dn.notna() & (days >= "2019-01-01")).to_numpy(); om = pd.Series(sel & (yrs >= 2021), index=days)
    _, S3m, A = RR.band_multipliers(b)
    X["rv_surp_1000"] = pd.Series(np.log(A["cRV"][:, 29] / A["S_rv"][:, 29]), index=days).fillna(0.0)
    CLEAN_R2 = [("rv_surp_1000" if f == "b30_n" else f) for f in P.CLEAN]
    X["dur_rate"] = M.family_features(b, I["X"])["dur_rate"]; assert X.loc[sel, "dur_rate"].notna().all()
    # ---- 门槛:变换后的引擎在 entry_mult=None / 全 1 时逐位等于原引擎与缓存
    PHb, PHf = LI.run_phases_fills(b), SU.fomc_late_PH(b)
    one = pd.DataFrame(1.0, index=days, columns=b.c.columns)
    for nm, m_ in (("None", None), ("全 1", one)):
        for off in (0, 13, 29):
            rr, F = SIM_E(b, checks=SA.chk(off), sizing="voltarget", max_lev=4.0, reentry="hold", half_spread=LI.HS0, commission=LI.CM0, entry_mult=m_, **SA.BASE)
            F2 = PHb[off][1].reset_index(drop=True); F = F.reset_index(drop=True)
            same = float((rr - PHb[off][0]).abs().max()) == 0.0 and len(F) == len(F2) and (F.shares.to_numpy() == F2.shares.to_numpy()).all() and (F.clock.to_numpy() == F2.clock.to_numpy()).all()
            assert same, f"门槛失败:entry_mult={nm} 相位 {off}"
    print(f"[门槛] 入场带变换后的引擎(entry_mult=None/全 1,相位 0/13/29)与原引擎缓存逐位一致  ({time.time()-t0:.0f}s)", flush=True)
    # ---- 描述性:重叠程度
    ffa_rank = X["ffa_rank"] if "ffa_rank" in X else None
    print(f"[重叠] Spearman(rv_surp_1000, b30_n) = {X.loc[sel, 'rv_surp_1000'].corr(X.loc[sel, 'b30_n'], method='spearman'):+.3f}", end="")
    if ffa_rank is not None: print(f";Spearman(rv_surp_1000, FFA-P 排名) = {X.loc[sel, 'rv_surp_1000'].corr(ffa_rank[sel], method='spearman'):+.3f}", end="")
    s3_1000 = S3m.iloc[:, 29]; print(f";10:00 时 s3>1 的日子在 FFA-P 跳过日中的比例 {(s3_1000[keep_ffa == 0] > 1).mean():.1%} vs 其余日 {(s3_1000[keep_ffa == 1] > 1).mean():.1%}")
    # ---- 成交
    PHbE, PHfE = run_PH_entry(b, S3m), run_PH_entry(b, S3m, fomc_only=True); print(f"  E1 成交完成 ({time.time()-t0:.0f}s)", flush=True)
    L_base, L_E1 = legs_k(PHb, PHf, b, keep_ffa), legs_k(PHbE, PHfE, b, keep_ffa)
    L_E1_noffa = legs_k(PHbE, PHfE, b, keep1)
    gate = max(float((L_base[k] - I[k]).abs().max()) for k in ("L2", "S2", "GL", "GS")); assert gate == 0.0
    cfg = {"B v3 现行": (L_base, P.CLEAN, "mv", "sig"),
           "O2 组合下行半方差(参照)": (L_base, P.CLEAN, "semi", "sig"),
           "O2-lin 半方差 + 线性链接": (L_base, P.CLEAN, "semi", "lin"),
           "O2-lpm 只罚大亏(10% 分位以下)": (L_base, P.CLEAN, "lpm", "sig"),
           "E1 只放宽入场带(还手)": (L_E1, P.CLEAN, "mv", "sig"),
           "E2 E1 + rv_surp 进仓位": (L_E1, P.CLEAN + ["rv_surp_1000"], "mv", "sig"),
           "R1 去 FFA 跳过 → E1 入场带": (L_E1_noffa, P.CLEAN, "mv", "sig"),
           "R2 b30_n → rv_surp_1000": (L_base, CLEAN_R2, "mv", "sig"),
           "N1 新方法版 v3(E1+R2+O2,保留 FFA)": (L_E1, CLEAN_R2, "semi", "sig"),
           "N2 新方法版 v3(E1+R2+O2,去 FFA)": (L_E1_noffa, CLEAN_R2, "semi", "sig"),
           "L1 线性链接": (L_base, P.CLEAN, "mv", "lin"),
           "L2 线性 + dur_rate": (L_base, P.CLEAN + ["dur_rate"], "mv", "lin"),
           "S2 阶梯:E1 + R2": (L_E1, CLEAN_R2, "mv", "sig"),
           "S3 阶梯:+ dur_rate": (L_E1, CLEAN_R2 + ["dur_rate"], "mv", "sig"),
           "S4 阶梯:+ 线性链接": (L_E1, CLEAN_R2 + ["dur_rate"], "mv", "lin"),
           "S5 阶梯:+ O2 目标": (L_E1, CLEAN_R2 + ["dur_rate"], "semi", "lin"),
           "S6 阶梯:− FFA 跳过": (L_E1_noffa, CLEAN_R2 + ["dur_rate"], "semi", "lin")}
    LADDER = ["B v3 现行", "E1 只放宽入场带(还手)", "S2 阶梯:E1 + R2", "S3 阶梯:+ dur_rate", "S4 阶梯:+ 线性链接", "S5 阶梯:+ O2 目标", "S6 阶梯:− FFA 跳过"]
    R = {}
    for nm, (lg, ft, obj, lk) in cfg.items():
        R[nm] = assemble(sizing(X, ft, lg, I, sel, days, obj, lk), lg, I, om)
    print(f"  全部配置完成 ({time.time()-t0:.0f}s)", flush=True)
    # ---- 评估
    oi = np.where(om.to_numpy())[0]; dev = yrs[oi] <= 2022; BI = M.block_idx(len(oi), 10, 2000, 0); srv = lambda v: v.mean() / v.std() * np.sqrt(252)
    ref = R["B v3 现行"]["port"].to_numpy()[oi]; refb = np.array([srv(ref[q]) for q in BI]); rows = []; pans = {}
    for nm, r_ in R.items():
        x_ = r_["port"].to_numpy()[oi]; db = np.array([srv(x_[q]) for q in BI]) - refb
        row = dict(方案=nm, 组合SR_OOS=srv(x_), ΔSR=srv(x_) - srv(ref), lo=np.percentile(db, 2.5), hi=np.percentile(db, 97.5),
                   ΔSR_dev=srv(x_[dev]) - srv(ref[dev]), ΔSR_test=srv(x_[~dev]) - srv(ref[~dev]), Sleeve最差日bp=float(r_["sleeve"].to_numpy()[oi].min() * 1e4))
        row.update({str(y): srv(x_[yrs[oi] == y]) - srv(ref[yrs[oi] == y]) for y in range(2021, 2027)})
        row["优于B"] = bool(nm != "B v3 现行" and row["ΔSR"] > 0 and row["ΔSR_dev"] > 0 and row["ΔSR_test"] > 0 and row["lo"] > -0.05); rows.append(row)
        pans[nm] = SF.panel(pd.Series(x_))
    Rt = pd.DataFrame(rows); print("\n=== 组合层 OOS 2021–26(ΔSR 相对 v3)==="); print(Rt.to_string(index=False))
    Pn = pd.DataFrame(pans).T; print("\n=== 分布面板(组合,OOS 2021–26)==="); print(Pn.to_string())
    lad = []
    for a_, b_ in zip(LADDER[:-1], LADDER[1:]):
        xa, xb = R[a_]["port"].to_numpy()[oi], R[b_]["port"].to_numpy()[oi]; db = np.array([srv(xb[q]) - srv(xa[q]) for q in BI])
        lad.append(dict(步=b_, 本步ΔSR=srv(xb) - srv(xa), lo=np.percentile(db, 2.5), hi=np.percentile(db, 97.5),
                        本步dev=srv(xb[dev]) - srv(xa[dev]), 本步test=srv(xb[~dev]) - srv(xa[~dev]), 累计ΔSR_vs_v3=srv(xb) - srv(ref)))
    LD = pd.DataFrame(lad); print("\n=== 先验顺序累加阶梯(每步相对上一步;块自助 95%)==="); print(LD.to_string(index=False)); LD.to_csv(TAB / "r3_ladder.csv", index=False)
    # ---- tearsheet:v3 vs N1 vs N2(与 tearsheet_v3 同口径:全部交易日)
    bh = I["bh"]; show = {"QQQ 买入持有": bh, "v3 现行": R["B v3 现行"]["port"], "N1 新方法版(保留 FFA)": R["N1 新方法版 v3(E1+R2+O2,保留 FFA)"]["port"],
                          "N2 新方法版(去 FFA)": R["N2 新方法版 v3(E1+R2+O2,去 FFA)"]["port"],
                          "S5 阶梯终点(保留 FFA)": R["S5 阶梯:+ O2 目标"]["port"], "S6 阶梯终点(去 FFA)": R["S6 阶梯:− FFA 跳过"]["port"]}
    KEYS = ["年化收益 CAGR", "年化波动", "Sharpe（绝对）", "Sortino", "最大回撤 MDD", "Calmar", "最差单日", "偏度", "超额峰度", "CVaR95（日）", "α（年化）", "α 的 t 值 (HAC)"]
    for pn, a, z in PER:
        T_ = pd.DataFrame({nm: {k_: TS.metrics(x.loc[a:z], bench=None if nm.startswith("QQQ") else bh.loc[a:z]).get(k_, np.nan) for k_ in KEYS} for nm, x in show.items()})
        print(f"\n===== tearsheet {pn} ====="); print(T_.to_string())
    yr = pd.DataFrame({nm: {y: TS.metrics(x.loc[str(y)])["Sharpe（绝对）"] for y in range(2019, 2027)} for nm, x in show.items()}); print("\n===== 分年 Sharpe ====="); print(yr.round(3).to_string())
    use_cjk(plt); fig, ax = plt.subplots(2, 1, figsize=(15, 10), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    col = {"QQQ 买入持有": "black", "v3 现行": "tab:green", "N1 新方法版(保留 FFA)": "tab:blue", "N2 新方法版(去 FFA)": "tab:red",
           "S5 阶梯终点(保留 FFA)": "tab:purple", "S6 阶梯终点(去 FFA)": "tab:orange"}
    for nm, x in show.items():
        x = x.loc["2019":]; eq = (1 + x).cumprod(); te = TS.metrics(x.loc["2023":])
        ax[0].plot(eq, color=col[nm], lw=1.4 if nm.startswith("QQQ") else 2, ls=":" if nm.startswith("QQQ") else "-",
                   label=f"{nm}(test Sharpe {te['Sharpe（绝对）']:.2f}, test MDD {te['最大回撤 MDD']:.1%})")
        ax[1].plot(eq / eq.cummax() - 1, color=col[nm], lw=1.1, ls=":" if nm.startswith("QQQ") else "-")
    for a_ in ax: a_.axvline(pd.Timestamp("2023-01-01"), color="gray", ls="--", lw=1); a_.grid(alpha=.3)
    ax[0].set_yscale("log"); ax[0].legend(fontsize=9, loc="upper left"); ax[0].set_title("v3 vs 新方法版 v3(N1/N2;2021 起仓位模型走步生效)", loc="left"); ax[1].set_title("回撤", loc="left")
    figp = Path("../outputs/figs/r3_v3_vs_new.png"); figp.parent.mkdir(parents=True, exist_ok=True); fig.savefig(figp, dpi=120, bbox_inches="tight")
    Rt.to_csv(TAB / "r3_configs.csv", index=False); Pn.to_csv(TAB / "r3_panel.csv"); yr.to_csv(TAB / "r3_yearly.csv")
    print(f"图:{figp};总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
