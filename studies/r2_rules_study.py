"""第二轮 C 部分(规则进 SOTA v3)+ A 部分(仓位模型)—— docs/PROPOSAL_intraday_vol_entry_r2.md §1–2(预注册)。
python3 r2_rules_study.py
"""
from __future__ import annotations

import os
import sys
import time
import warnings

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import disp_pred_study as DP
import ivol_entry_study as IE
import ivol_objective_study as AO
import letf_posthoc as LP
import liquidity_inverse_study as LI
import sizing_models_study as M
import sizing_sigma_feed as SF
import sleeve_anatomy as SA
import sota_unified_stack as SU
import tearsheet as TS
import unified_sizing_posthoc as P
import unified_sizing_study as U
from audit import reimpl_noise as RI
from b0_econ import causal_rank
from etft.intraday.bars import load_bars
from event_calendar import FOMC_DECISION

OUT, TAB = SF.OUT, M.TAB


# ============================================================ 引擎:只替换噪声带宽(乘以 s)
def run_PH(b, S, fomc_only=False):
    """S: DataFrame(days × times) 带宽乘数;None = 原样。返回 {相位: (日收益, 成交)},与 LI.run_phases_fills / SU.fomc_late_PH 同参数。"""
    days = b.days; orig = RI.sigma_profile
    if S is not None: RI.sigma_profile = lambda C, O, lb=14: orig(C, O, lb) * S
    try:
        out = {}
        if fomc_only:
            fs = set(d for d in FOMC_DECISION if d in set(days)); nonf = [d for d in days if d not in fs]
        for off in range(30):
            if fomc_only:
                rr, F = RI.simulate(b, checks=[t for t in SA.chk(off) if t >= "14:30"], sizing="voltarget", max_lev=4.0, reentry="hold",
                                    half_spread=0.005, commission=0.0045, skip_days=nonf, **SA.BASE)
            else:
                rr, F = RI.simulate(b, checks=SA.chk(off), sizing="voltarget", max_lev=4.0, reentry="hold", half_spread=LI.HS0, commission=LI.CM0, **SA.BASE)
            F = F.copy()
            if len(F): F["day"] = pd.to_datetime(F.day)
            out[off] = (rr, F)
    finally:
        RI.sigma_profile = orig
    return out


def legs_all(PHb, PHf, b, I):
    days = b.days; scale = (b.day_close / LI.tqqq_traded_px(days)) / 3.0
    Lb0, Sb0 = SU.legs_from_PH(PHb, days, scale); Lf, Sf = SU.legs_from_PH(PHf, days, scale)
    fm = pd.Series(days.isin(pd.DatetimeIndex(FOMC_DECISION)), index=days); keep = I["keep"]
    cols = list(b.c.columns); ob = b.o.to_numpy()
    GLb, GSb = LP.G_series(PHb, days, ob, cols, "15:40"); GLf, GSf = LP.G_series(PHf, days, ob, cols, "15:40")
    return dict(L2=Lb0.where(~fm, Lf) * keep, S2=Sb0.where(~fm, Sf) * keep, GL=GLb.where(~fm, GLf), GS=GSb.where(~fm, GSf))


def filter_PH(PH, skip):
    """skip: set of (phase, day, t_in)。删除被跳过的整笔(入场 + 出场两条成交)。"""
    out = {}
    for off, (rr, F) in PH.items():
        if not len(F): out[off] = (rr, F); continue
        F = F.reset_index(drop=True); drop = np.zeros(len(F), bool); cur = None
        for j, r in enumerate(F.itertuples(index=False)):
            if str(r.kind).startswith("ENTRY"): cur = j if (off, pd.Timestamp(r.day), r.clock) in skip else None
            elif cur is not None: drop[cur] = drop[j] = True; cur = None
        out[off] = (rr, F[~drop])
    return out


# ============================================================ 带宽乘数
def band_multipliers(b):
    c = b.c.ffill(axis=1).to_numpy(); O = b.day_open.to_numpy(); A = IE.day_arrays(c, O)
    v = np.ascontiguousarray(np.nan_to_num(b.v.to_numpy()), dtype=float); cV = np.cumsum(v, 1); SV = IE._season(cV)
    rvr = A["cRV"] / A["S_rv"]; vr = cV / SV
    s2 = np.clip(np.sqrt(rvr), 1, 2); s3 = np.clip(np.sqrt(rvr / vr), 1, 2)
    mk = lambda s: pd.DataFrame(np.where(np.isfinite(s), s, 1.0), index=b.days, columns=b.c.columns)
    return mk(s2), mk(s3), A


# ============================================================ 评估
def evaluate(name, legs, I, X, sel, days, om, ref=None):
    leg1, ov, keep = I["leg1"], I["ov"], I["keep"]
    Lb, Sb, Eb = (legs["L2"] * 1e4), (legs["S2"] * 1e4), ((leg1 + ov) * 1e4)
    k = P.wf(X, Lb, Sb, Eb, sel, days, P.CLEAN)[0]
    kl = k.kl.where(om, 1.0).fillna(1.0); ks = k.ks.where(om, 1.0).fillna(1.0)
    sl = kl * legs["L2"] + ks * legs["S2"] + ov - (kl * legs["GL"] + ks * legs["GS"]) * keep
    return dict(name=name, sleeve=sl, port=leg1 + sl, k=k, legs=legs)


def main():
    t0 = time.time(); pd.set_option("display.width", 270, "display.max_rows", 200, "display.float_format", lambda v: f"{v:8.3f}")
    b = load_bars("QQQ"); days = b.days; yrs = days.year
    I = SF.build_inputs(b); X = I["X"]
    dn = DP.targets(DP.Arr(b), 30).dn; full = pd.Series(b.last_idx.to_numpy() >= 389, index=days); td = pd.Series(days.dayofweek != 0, index=days)
    sel = (td & full & X.notna().all(1) & dn.notna() & (days >= "2019-01-01")).to_numpy(); om = pd.Series(sel & (yrs >= 2021), index=days)
    PHb, PHf = LI.run_phases_fills(b), SU.fomc_late_PH(b)
    base_legs = legs_all(PHb, PHf, b, I)
    gate = max(float((base_legs[k] - I[k]).abs().max()) for k in ("L2", "S2", "GL", "GS")); print(f"[门槛1] 重建腿 vs build_inputs 最大差 {gate:.2e}"); assert gate < 1e-12
    S2m, S3m, A = band_multipliers(b)
    one = pd.DataFrame(1.0, index=days, columns=b.c.columns)
    PH1 = run_PH(b, one, fomc_only=False)
    d1 = max(float((PH1[o][0] - PHb[o][0]).abs().max()) for o in range(30)); n1 = sum(len(PH1[o][1]) for o in range(30)) - sum(len(PHb[o][1]) for o in range(30))
    print(f"[门槛2] 带宽 × 1 重跑引擎 vs 缓存成交:日收益最大差 {d1:.2e},成交笔数差 {n1}"); assert d1 == 0.0 and n1 == 0
    print(f"  s 分布(OOS 决策时点 10:00–15:30):C2 中位 {np.median(S2m.loc['2021':].to_numpy()[:, 29:360]):.3f}、>1 比例 {(S2m.loc['2021':].to_numpy()[:, 29:360] > 1).mean():.1%};"
          f" C3 中位 {np.median(S3m.loc['2021':].to_numpy()[:, 29:360]):.3f}、>1 比例 {(S3m.loc['2021':].to_numpy()[:, 29:360] > 1).mean():.1%}  ({time.time()-t0:.0f}s)", flush=True)
    R = {"B v3 现行": evaluate("B", base_legs, I, X, sel, days, om)}
    # ---- C1 rv_surp 跳过
    cols = list(b.c.columns); tix = {t: i for i, t in enumerate(cols)}; dpos = {d: i for i, d in enumerate(days)}
    fm = set(pd.DatetimeIndex(FOMC_DECISION))
    T = []
    for src, PH in (("b", PHb), ("f", PHf)):
        for off, (rr, F) in PH.items():
            if not len(F): continue
            t_ = LI.trades_from_fills(F, off); t_["src"] = src; T.append(t_)
    T = pd.concat(T, ignore_index=True); T["day"] = pd.to_datetime(T.day)
    T = T[((T.src == "b") & ~T.day.isin(fm)) | ((T.src == "f") & T.day.isin(fm))].sort_values(["day", "phase", "t_in"]).reset_index(drop=True)
    ii = T.day.map(dpos).to_numpy(); kk = T.t_in.map(tix).to_numpy()
    rvs = np.log(A["cRV"][ii, kk] / A["S_rv"][ii, kk]); x = -rvs
    rk = causal_rank(x, T.day.to_numpy()); skipm = np.isfinite(x) & (rk <= 0.20)
    print(f"  C1:实际成交 {len(T)} 笔,跳过 {skipm.mean():.1%}(OOS {skipm[T.day.dt.year.to_numpy() >= 2021].mean():.1%})", flush=True)
    sk = set(zip(T.phase[skipm], T.day[skipm], T.t_in[skipm]))
    legs_c1 = legs_all(filter_PH(PHb, {s for s in sk}), filter_PH(PHf, {s for s in sk}), b, I)
    R["C1 rv_surp 跳过 20%"] = evaluate("C1", legs_c1, I, X, sel, days, om)
    # ---- C2 / C3 噪声带放宽
    for nm, Sm in (("C2 带宽按波动放宽", S2m), ("C3 带宽按无量波动放宽(还手)", S3m)):
        PHbx, PHfx = run_PH(b, Sm), run_PH(b, Sm, fomc_only=True)
        R[nm] = evaluate(nm, legs_all(PHbx, PHfx, b, I), I, X, sel, days, om); print(f"  [{nm}] 完成 ({time.time()-t0:.0f}s)", flush=True)
    # ---- A1 腿级下行半方差 / A2 rv_surp 进仓位模型
    leg1, ov, keep = I["leg1"], I["ov"], I["keep"]
    Lb, Sb, Eb = (base_legs["L2"] * 1e4).to_numpy(), (base_legs["S2"] * 1e4).to_numpy(), ((leg1 + ov) * 1e4).to_numpy()
    kA1, lamA1 = AO.wf_obj(X, P.CLEAN, Lb, Sb, Eb, pd.DataFrame({"L": Lb, "S": Sb, "E": Eb}, index=days), sel, days, "legsemi")
    def sized(k, legs):
        kl = k.kl.where(om, 1.0).fillna(1.0); ks = k.ks.where(om, 1.0).fillna(1.0)
        s_ = kl * legs["L2"] + ks * legs["S2"] + ov - (kl * legs["GL"] + ks * legs["GS"]) * keep; return dict(sleeve=s_, port=leg1 + s_, k=k)
    R["A1 腿级下行半方差"] = sized(kA1, base_legs)
    X2 = X.copy(); X2["rv_surp_1000"] = np.log(A["cRV"][:, 29] / A["S_rv"][:, 29])
    nwarm = int((X2.rv_surp_1000.isna().to_numpy() & sel).sum()); X2["rv_surp_1000"] = X2.rv_surp_1000.fillna(0.0)
    print(f"  A2:rv_surp_1000 在 {nwarm} 个样本日(2019-01 预热期,季节基准不足 10 日)置 0(偏离记录)")
    kA2 = P.wf(X2, pd.Series(Lb, index=days), pd.Series(Sb, index=days), pd.Series(Eb, index=days), sel, days, P.CLEAN + ["rv_surp_1000"])[0]
    R["A2 rv_surp 进仓位模型"] = sized(kA2, base_legs); print(f"  A 部分完成 λ(A1) {lamA1} ({time.time()-t0:.0f}s)", flush=True)
    # ---- 评估表
    oi = np.where(om.to_numpy())[0]; dev = yrs[oi] <= 2022; BI = M.block_idx(len(oi), 10, 2000, 0); srv = lambda v: v.mean() / v.std() * np.sqrt(252)
    ref = R["B v3 现行"]["port"].to_numpy()[oi]; refb = np.array([srv(ref[r]) for r in BI]); rows = []; pans = {}
    for nm, r_ in R.items():
        x_ = r_["port"].to_numpy()[oi]; db = np.array([srv(x_[q]) for q in BI]) - refb
        row = dict(方案=nm, 组合SR_OOS=srv(x_), ΔSR=srv(x_) - srv(ref), lo=np.percentile(db, 2.5), hi=np.percentile(db, 97.5),
                   ΔSR_dev=srv(x_[dev]) - srv(ref[dev]), ΔSR_test=srv(x_[~dev]) - srv(ref[~dev]),
                   Sleeve_SR_OOS=srv(r_["sleeve"].to_numpy()[oi]),
                   test期组合SR_全部交易日=TS.metrics(r_["port"].loc["2023":"2026"])["Sharpe（绝对）"], test期MDD=TS.metrics(r_["port"].loc["2023":"2026"])["最大回撤 MDD"],
                   全期MDD=TS.metrics(r_["port"].loc["2019":"2026"])["最大回撤 MDD"])
        row.update({str(y): srv(x_[yrs[oi] == y]) - srv(ref[yrs[oi] == y]) for y in range(2021, 2027)})
        row["优于B"] = bool(nm != "B v3 现行" and row["ΔSR"] > 0 and row["ΔSR_dev"] > 0 and row["ΔSR_test"] > 0 and row["lo"] > -0.05); rows.append(row)
        pans[nm] = SF.panel(pd.Series(x_)); pans["Sleeve | " + nm] = SF.panel(pd.Series(r_["sleeve"].to_numpy()[oi]))
    Rt = pd.DataFrame(rows); print("\n=== 组合层(QQQ ① 腿 + v3 Sleeve,PPP 在新腿上重训),OOS 2021–26;ΔSR 相对 B ==="); print(Rt.to_string(index=False))
    Pn = pd.DataFrame(pans).T; print("\n=== 分布面板(OOS 2021–26)==="); print(Pn.to_string())
    # ---- C1 置换(固定 B 的 k,快速近似:被跳过的笔从腿中减去其净损益与 15:40 增量)
    kB = R["B v3 现行"]["k"]; klB = kB.kl.where(om, 1.0).fillna(1.0).to_numpy(); ksB = kB.ks.where(om, 1.0).fillna(1.0).to_numpy()
    o_ = b.o.to_numpy(); k1540 = cols.index("15:40") + 1
    di = ii; pnl = T.pnl_net.to_numpy() / 30.0; isL = T.side.to_numpy() > 0
    span = (T.t_in.to_numpy() < "15:40") & (T.t_out.to_numpy() > "15:40")
    g = np.where(span, T.shares.to_numpy() * (T.exit_px.to_numpy() - o_[di, k1540]), 0.0) / 30.0
    kp = I["keep"].to_numpy(); ovv = ov.to_numpy(); legv = leg1.to_numpy()
    L0, S0, G0L, G0S = (base_legs[k].to_numpy() for k in ("L2", "S2", "GL", "GS"))
    def stat(mask):
        dL = np.bincount(di[mask & isL], weights=pnl[mask & isL], minlength=len(days)) * kp; dS = np.bincount(di[mask & ~isL], weights=pnl[mask & ~isL], minlength=len(days)) * kp
        gL = np.bincount(di[mask & isL], weights=g[mask & isL], minlength=len(days)); gS = np.bincount(di[mask & ~isL], weights=g[mask & ~isL], minlength=len(days))
        sl = klB * (L0 - dL) + ksB * (S0 - dS) + ovv - (klB * (G0L - gL) + ksB * (G0S - gS)) * kp
        return srv((legv + sl)[oi]) - srv(ref)
    obs = stat(skipm); yrT = T.day.dt.year.to_numpy(); rg = np.random.default_rng(11); nul = []
    for _ in range(500):
        mm = np.zeros(len(T), bool)
        for yv in np.unique(yrT):
            ix = np.where(yrT == yv)[0]; mm[rg.choice(ix, int(skipm[ix].sum()), replace=False)] = True
        nul.append(stat(mm))
    print(f"\n[C1 置换,固定 B 的 k] 观测 ΔSR {obs:+.3f};零分布 中位 {np.median(nul):+.3f}、95% {np.quantile(nul, .95):+.3f};p = {(np.sum(np.array(nul) >= obs) + 1) / 501:.3f}")
    Rt.to_csv(TAB / "r2_rules.csv", index=False); Pn.to_csv(TAB / "r2_rules_panel.csv")
    pd.to_pickle({k: dict(port=v["port"], sleeve=v["sleeve"]) for k, v in R.items()}, OUT / "r2_rules.pkl"); print(f"总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
