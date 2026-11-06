"""拥挤度 × Sleeve:关系检验(B1–B5)与纳入策略(C1–C6、D1)—— docs/PROPOSAL_crowding_vol_timing.md §3–4(预注册)。
python3 crowding_study.py
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
import statsmodels.api as sm
from scipy.stats import skew

sys.path.insert(0, str(Path(__file__).resolve().parent))
import crowding_index as CX
import disp_pred_study as DP
import liquidity_inverse_study as LI
import r2_rules_study as RR
import r3_study as R3
import round3_common as C
import sizing_models_study as M
import sizing_sigma_feed as SF
import sota_stack_study as S
import sota_unified_stack as SU
import tearsheet as TS
import unified_sizing_posthoc as P
from etft.intraday.bars import load_bars

TAB = M.TAB; DELTAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def state_table(x, st, name):
    rows = []
    for s_, lab in ((0, "低"), (1, "中"), (2, "高")):
        v = x[st == s_]
        if len(v) < 20: continue
        rows.append(dict(指标=name, 状态=lab, 天数=len(v), 均值bp=v.mean() * 1e4, Sharpe=v.mean() / v.std() * np.sqrt(252), 偏度=skew(v), 最差日bp=v.min() * 1e4, 胜率=(v > 0).mean()))
    return rows


def shift_test(x, st_daily, days, n=2000, seed=0):
    """月度状态序列圆周平移(≥ 6 个月),统计量 = 高 − 低 的日均差。"""
    mon = pd.DatetimeIndex(days).to_period("M"); um = pd.Index(sorted(set(mon))); mi = um.get_indexer(mon)
    sm_ = pd.Series(st_daily, index=mon).groupby(level=0).first().reindex(um).to_numpy()
    obs = np.nanmean(x[st_daily == 2]) - np.nanmean(x[st_daily == 0]); rg = np.random.default_rng(seed); nul = []
    for _ in range(n):
        k = int(rg.integers(6, len(um) - 6)); s2 = np.roll(sm_, k)[mi]
        if (s2 == 2).sum() > 20 and (s2 == 0).sum() > 20: nul.append(np.nanmean(x[s2 == 2]) - np.nanmean(x[s2 == 0]))
    nul = np.array(nul); return obs, float(np.mean(np.abs(nul) >= abs(obs)))


def recursive_delta(cands, days, sel_mask, start="2021-01-01"):
    """cands: {δ: 组合日收益数组}。每月按 2019-01 至上月末(样本日)的扩展窗口夏普选 δ(并列取较大 δ = 更接近原规则)。返回选择后的序列与逐月 δ。"""
    days = pd.DatetimeIndex(days); out = cands[1.0].copy(); mon = days.to_period("M"); picks = {}
    for m in sorted(set(mon[days >= start])):
        hist = sel_mask & (days < m.start_time) & (days >= "2019-01-01")
        sr = {d: cands[d][hist].mean() / cands[d][hist].std() for d in DELTAS}
        best = max(DELTAS, key=lambda d: (round(sr[d], 10), d)); picks[str(m)] = best
        cur = mon == m; out[cur] = cands[best][cur]
    return out, picks


def main():
    t0 = time.time(); pd.set_option("display.width", 280, "display.max_rows", 200, "display.float_format", lambda v: f"{v:8.3f}")
    b = load_bars("QQQ"); days = b.days; yrs = days.year
    I = SF.build_inputs(b); X = I["X"].copy(); bh = I["bh"]; keep = I["keep"]; ev = I["e_vt"].to_numpy()
    dn = DP.targets(DP.Arr(b), 30).dn; full = pd.Series(b.last_idx.to_numpy() >= 389, index=days); td = pd.Series(days.dayofweek != 0, index=days)
    sel = (td & full & X.notna().all(1) & dn.notna() & (days >= "2019-01-01")).to_numpy(); om = pd.Series(sel & (yrs >= 2021), index=days)
    X["dur_rate"] = M.family_features(b, I["X"])["dur_rate"]; F4 = P.CLEAN + ["dur_rate"]
    # ---- 拥挤度(日频映射,PIT)
    Mo, CI = CX.build()
    cr = {"MOM": CX.to_daily(CI["MOM"], days, extra_lag=1), "TSM": CX.to_daily(CI["TSM"], days), "MKT": CX.to_daily(CI["MKT"], days)}
    for k_, v in cr.items(): assert v.loc[sel, ["C", "state"]].notna().all().all(), f"{k_} 在样本日缺失"
    X["c_mom"], X["c_tsm"] = cr["MOM"].C, cr["TSM"].C; X["c_mom_dv"], X["c_tsm_dv"] = X.c_mom * X.dlvix, X.c_tsm * X.dlvix
    # ---- 成交与腿
    PHb, PHf = LI.run_phases_fills(b), SU.fomc_late_PH(b); _, S3m, _ = RR.band_multipliers(b)
    PHbE, PHfE = R3.run_PH_entry(b, S3m), R3.run_PH_entry(b, S3m, fomc_only=True)
    Lv3, Lv4 = R3.legs_k(PHb, PHf, b, keep), R3.legs_k(PHbE, PHfE, b, keep)
    R = {"v4": R3.assemble(R3.sizing(X, F4, Lv4, I, sel, days, "mv", "lin"), Lv4, I, om)}
    print(f"[输入] 完成 ({time.time()-t0:.0f}s);v4 组合 OOS SR {(lambda v: v.mean()/v.std()*np.sqrt(252))(R['v4']['port'].to_numpy()[om.to_numpy()]):.3f}(tearsheet_v4 ① 1.911)", flush=True)
    s = sel
    # ================= B. 关系检验(v3 SOTA Sleeve,未做仓位模型缩放;2019–26)=================
    vt = (Lv3["L2"] + Lv3["S2"] - (Lv3["GL"] + Lv3["GS"]) * keep).to_numpy(); raw = vt + I["ov"].to_numpy()
    rows = []; tests = []
    for nm in ("MOM", "TSM", "MKT"):
        st = cr[nm].state.to_numpy(); rows += state_table(raw[s], st[s], f"C_{nm}")
        o_, p_ = shift_test(raw[s], st[s], days[s]); tests.append(dict(指标=f"C_{nm}", 高减低_bp=o_ * 1e4, p_圆周平移=p_))
    print("\n=== B1. Sleeve(v3 SOTA 腿,未缩放)日损益按拥挤状态(2019–26,周二至周五)==="); print(pd.DataFrame(rows).to_string(index=False)); print(pd.DataFrame(tests).to_string(index=False))
    unit = vt / ev; cfac = np.std(vt[s]) / np.std(unit[s]); inc = vt - cfac * unit; rows = []
    for nm in ("MOM", "TSM", "MKT"):
        st = cr[nm].state.to_numpy()
        for s_, lab in ((0, "低"), (1, "中"), (2, "高")):
            m = s & (st == s_); a_, u_ = vt[m], cfac * unit[m]
            rows.append(dict(指标=f"C_{nm}", 状态=lab, 天数=int(m.sum()), VM增量均值bp=inc[m].mean() * 1e4, 原始SR=u_.mean() / u_.std() * np.sqrt(252), VM后SR=a_.mean() / a_.std() * np.sqrt(252),
                             ΔSR=a_.mean() / a_.std() * np.sqrt(252) - u_.mean() / u_.std() * np.sqrt(252), 原始最差bp=u_.min() * 1e4, VM后最差bp=a_.min() * 1e4))
    print(f"\n=== B2. 文章核心检验的对应物:σ14 波动目标(VM)相对单位仓位(×{cfac:.2f},等波动)的增量,按拥挤状态 ==="); print(pd.DataFrame(rows).to_string(index=False))
    momd = CX._french("F-F_Momentum_Factor_daily.csv", ["MOM"]).MOM.reindex(days).to_numpy(); mm = s & np.isfinite(momd)
    thr = np.quantile(momd[mm], 0.05); crash = mm & (momd <= thr); stm = cr["MOM"].state.to_numpy()
    print(f"\n=== B3. Sleeve 与 MOM 日收益:相关 {np.corrcoef(raw[mm], momd[mm])[0, 1]:+.3f}(多头腿 {np.corrcoef(Lv3['L2'].to_numpy()[mm], momd[mm])[0, 1]:+.3f},空头腿 {np.corrcoef(Lv3['S2'].to_numpy()[mm], momd[mm])[0, 1]:+.3f})")
    print(f"动量崩盘日(MOM 日收益 ≤ {thr*100:.2f}%,{int(crash.sum())} 天):Sleeve 均值 {raw[crash].mean()*1e4:+.1f}bp(其余日 {raw[mm & ~crash].mean()*1e4:+.1f});"
          + ";".join(f"{lab}拥挤 {raw[crash & (stm == k_)].mean()*1e4:+.1f}bp(n={int((crash & (stm == k_)).sum())})" for k_, lab in ((0, "低"), (1, "中"), (2, "高"))))
    vix = pd.read_parquet(CX.RAW / "cboe" / "VIX_long.parquet")["close"]; dv0 = np.log(vix).diff().reindex(days).to_numpy(); dv1 = X.dlvix.to_numpy(); rows = []
    for nm in ("MOM", "TSM"):
        c_ = cr[nm].C.to_numpy()
        for lab, dvv in (("同日 ΔlogVIX(描述性)", dv0), ("前一日 ΔlogVIX(可交易)", dv1)):
            m = s & np.isfinite(dvv); Z = np.column_stack([c_[m], dvv[m] * 10, c_[m] * dvv[m] * 10])
            fit = sm.OLS(raw[m] * 1e4, sm.add_constant(Z)).fit(cov_type="HAC", cov_kwds={"maxlags": 5}); bb, tt = np.asarray(fit.params), np.asarray(fit.tvalues)
            rows.append(dict(指标=f"C_{nm}", ΔVIX=lab, b_C=bb[1], t_C=tt[1], b_ΔVIX=bb[2], t_ΔVIX=tt[2], b_交互=bb[3], t_交互=tt[3], R2=fit.rsquared))
    print("\n=== B4. Sleeve(bp)= a + b₁C + b₂ΔVIX + b₃C×ΔVIX(ΔVIX 单位:10% 对数变化;HAC)==="); print(pd.DataFrame(rows).to_string(index=False))
    sd = pd.Series(raw[s], index=days[s]); dvs = pd.Series(dv0, index=days).reindex(sd.index)
    cs = CX.crowding(sd, dvs, W_RUN=252, W_REG=252, W_STD=252, W_STATE=252); stS = cs.state.shift(1).to_numpy(); xs = sd.to_numpy()
    print("\n=== B5. Sleeve 自身的收益隐含拥挤度(日频改写,前一日状态 → 当日 Sleeve;描述性)===")
    print(pd.DataFrame(state_table(xs[np.isfinite(stS)], stS[np.isfinite(stS)], "C_Sleeve")).to_string(index=False), f"\n  首个可用 {cs.state.dropna().index[0].date()}")
    # ================= C/D. 纳入策略 =================
    for nm, ft in (("C1 +c_mom", F4 + ["c_mom"]), ("C2 +c_mom, c_mom×dlvix", F4 + ["c_mom", "c_mom_dv"]),
                   ("C3 +c_tsm", F4 + ["c_tsm"]), ("C4 +c_tsm, c_tsm×dlvix", F4 + ["c_tsm", "c_tsm_dv"])):
        R[nm] = R3.assemble(R3.sizing(X, ft, Lv4, I, sel, days, "mv", "lin"), Lv4, I, om)
    print(f"  C1–C4 完成 ({time.time()-t0:.0f}s)", flush=True)
    unit4 = ((Lv4["L2"] + Lv4["S2"] - (Lv4["GL"] + Lv4["GS"]) * keep) / I["e_vt"]).to_numpy()
    mon = days.to_period("M"); cbar = np.empty(len(days))
    for m in sorted(set(mon)):
        h = sel & (days < m.start_time) & (days >= "2019-01-01"); cbar[mon == m] = ev[h].mean() if h.sum() > 50 else ev[sel].mean() * 0 + 1.9
    sl4 = R["v4"]["sleeve"].to_numpy(); leg1 = I["leg1"].to_numpy(); ov = I["ov"].to_numpy(); oo = om.to_numpy(); picks_all = {}
    for nm, key in (("C5 CVM-TSM", "TSM"), ("C6 CVM-MOM", "MOM")):
        hi = cr[key].state.to_numpy() == 2
        cands = {d: leg1 + np.where(hi & oo, d * cbar * unit4 + ov, sl4) for d in DELTAS}
        port, picks = recursive_delta(cands, days, sel); picks_all[nm] = picks
        R[nm] = dict(port=pd.Series(port, index=days), sleeve=pd.Series(port - leg1, index=days))
    px = C.load_close("yahoo/QQQ.parquet"); W = S.bh_weights(px, px.pct_change().fillna(0.0), {}, pd.Series(1.0, index=px.index))
    al = lambda w: w.reindex(bh.index).ffill().fillna(1.0).reindex(days).fillna(1.0).to_numpy()
    w_rs, w_s = al(W[("RV21[0.5,1.5]", "SMA200|e=0.5")]), al(W[("静态", "SMA200|e=0.5")]); bhv = bh.to_numpy()
    def legw(w): return w * bhv - 2e-4 * np.abs(np.diff(np.r_[w[0], w]))
    assert np.allclose(legw(w_rs)[sel], leg1[sel], atol=1e-12)
    him = cr["MKT"].state.to_numpy() == 2
    candsD = {d: legw(np.where(him & oo, d * w_s, w_rs)) + sl4 for d in DELTAS}
    portD, picksD = recursive_delta(candsD, days, sel); picks_all["D1 核心仓 CVM-MKT"] = picksD
    R["D1 核心仓 CVM-MKT"] = dict(port=pd.Series(portD, index=days), sleeve=R["v4"]["sleeve"])
    for nm, pk in picks_all.items():
        v = pd.Series(pk); print(f"  [{nm}] 递归 δ 分布:{v.value_counts().sort_index().to_dict()};高拥挤占 OOS 样本日 {float(((cr[{'C5 CVM-TSM': 'TSM', 'C6 CVM-MOM': 'MOM', 'D1 核心仓 CVM-MKT': 'MKT'}[nm]].state == 2).to_numpy() & oo).sum() / oo.sum()):.1%}")
    # ---- 评估
    oi = np.where(oo)[0]; dev = yrs[oi] <= 2022; BI = M.block_idx(len(oi), 10, 2000, 0); srv = lambda v: v.mean() / v.std() * np.sqrt(252)
    ref = R["v4"]["port"].to_numpy()[oi]; refb = np.array([srv(ref[q]) for q in BI]); rows = []; pans = {}
    for nm, r_ in R.items():
        x_ = r_["port"].to_numpy()[oi]; db = np.array([srv(x_[q]) for q in BI]) - refb
        row = dict(方案=nm, 组合SR_OOS=srv(x_), ΔSR=srv(x_) - srv(ref), lo=np.percentile(db, 2.5), hi=np.percentile(db, 97.5), dev=srv(x_[dev]) - srv(ref[dev]), test=srv(x_[~dev]) - srv(ref[~dev]),
                   test期SR=TS.metrics(r_["port"].loc["2023":"2026"])["Sharpe（绝对）"], test期MDD=TS.metrics(r_["port"].loc["2023":"2026"])["最大回撤 MDD"],
                   **{str(y): srv(x_[yrs[oi] == y]) - srv(ref[yrs[oi] == y]) for y in range(2021, 2027)})
        row["优于v4"] = bool(nm != "v4" and row["ΔSR"] > 0 and row["dev"] > 0 and row["test"] > 0 and row["lo"] > -0.05); rows.append(row); pans[nm] = SF.panel(pd.Series(x_))
    T = pd.DataFrame(rows); print("\n=== C/D. 组合层 OOS 2021–26(ΔSR 相对 v4)==="); print(T.to_string(index=False))
    print("\n=== 分布面板(组合,OOS 2021–26)==="); print(pd.DataFrame(pans).T.to_string())
    T.to_csv(TAB / "crowding_configs.csv", index=False); print(f"总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
