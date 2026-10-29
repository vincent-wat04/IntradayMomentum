"""A 部分:仓位模型的风险度量(docs/PROPOSAL_intraday_vol_entry.md §1,预注册)。
D1/D2 诊断 + O2(下行半方差)/O4(组合日内已实现方差)两个目标,与 v3 的 B(均值–方差)对比。
python3 ivol_objective_study.py
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
import statsmodels.api as sm
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import liquidity_inverse_study as LI
import sizing_models_study as M
import sizing_sigma_feed as SF
import sota_stack_study as S
import sota_unified_stack as SU
import unified_sizing_posthoc as P
import unified_sizing_study as U
from audit.reimpl_noise import daily_sigma
from etft.intraday.bars import load_bars
from event_calendar import FOMC_DECISION

TAB = M.TAB


# ============================================================ 分钟损益重建
def leg_minute_pnl(PH, b, days):
    """30 相位成交 → 每日每分钟的多/空腿毛损益(NAV 单位,30 相位均值)。返回 (Lm, Sm, chk) 形状 (n_days, 390)。"""
    cols = list(b.c.columns); tix = {t: i for i, t in enumerate(cols)}; dpos = {d: i for i, d in enumerate(days)}
    c, o = b.c.to_numpy(), b.o.to_numpy(); n = len(days)
    Lm, Sm = np.zeros((n, 390)), np.zeros((n, 390)); gross = np.zeros((n, 2))
    for off, (rr, F) in PH.items():
        if not len(F): continue
        T = LI.trades_from_fills(F, off)
        for r in T.itertuples(index=False):
            i = dpos[pd.Timestamp(r.day)]; ki, ko = tix[r.t_in], tix[r.t_out]; sh = r.shares; arr = Lm if sh > 0 else Sm
            arr[i, ki + 1] += sh * (c[i, ki + 1] - o[i, ki + 1])
            if ko >= ki + 2: arr[i, ki + 2:ko + 1] += sh * (c[i, ki + 2:ko + 1] - c[i, ki + 1:ko])
            arr[i, min(ko + 1, 389)] += sh * (r.exit_px - c[i, ko])     # 止损在 ko+1 开盘成交;MOC 时 exit_px = c[ko],此项为 0
            gross[i, 0 if sh > 0 else 1] += r.pnl_gross
    k = len(PH)
    Lm /= k; Sm /= k; gross /= k
    chk = float(max(np.abs(Lm.sum(1) - gross[:, 0]).max(), np.abs(Sm.sum(1) - gross[:, 1]).max()))
    return Lm, Sm, chk


def build_qv(b, I):
    """组合分钟损益的二次项分解(bp²):QV(k) = Scc + kl²SLL + ks²SSS + 2kl·SEL + 2ks·SES + 2kl·ks·SLS。"""
    days = b.days; full_mask = (b.last_idx.to_numpy() >= 389) & (days >= "2019-01-01")
    PHb, PHf = LI.run_phases_fills(b), SU.fomc_late_PH(b)
    Lb_, Sb_, chk1 = leg_minute_pnl(PHb, b, days); Lf_, Sf_, chk2 = leg_minute_pnl(PHf, b, days)
    fm = days.isin(pd.DatetimeIndex(FOMC_DECISION))[:, None]; keep = I["keep"].to_numpy()[:, None]
    Lm = np.where(fm, Lf_, Lb_) * keep; Sm = np.where(fm, Sf_, Sb_) * keep
    # QQQ 腿 + CPI overlay 的分钟损益:隔夜缺口放在一个额外的"第 0 分钟"
    c = b.c.ffill(axis=1).to_numpy(); O = b.day_open.to_numpy(); pc = b.prev_close_adj.to_numpy()   # 日内缺失 bar 前向填充(分钟差分之和须等于开收差)
    import round3_common as C
    px = C.load_close("yahoo/QQQ.parquet"); W = S.bh_weights(px, px.pct_change().fillna(0.0), {}, pd.Series(1.0, index=px.index))
    w = W[("RV21[0.5,1.5]", "SMA200|e=0.5")].reindex(I["bh"].index).ffill().fillna(1.0).reindex(days).fillna(1.0).to_numpy()   # 与 SF.build_inputs 的 leg1 同一权重
    cprev = np.column_stack([O, c[:, :-1]]); dP = np.nan_to_num(c - cprev); li_ = b.last_idx.to_numpy(); dP[np.arange(390)[None, :] > li_[:, None]] = 0.0
    core = np.column_stack([w * (O / pc - 1), w[:, None] * dP / pc[:, None]])
    ov = I["ov"].to_numpy(); n_ = np.arange(len(days)); oc = c[n_, b.last_idx.to_numpy()] / O - 1
    sg = np.where(ov != 0, np.sign((ov + 2e-4) / np.where(oc != 0, oc, np.nan)), 0.0); sg = np.nan_to_num(sg)   # overlay 方向 = CPI 信号(由 ov = sg·oc − 2bp 反解)
    core[:, 1:] += np.where(ov[:, None] != 0, sg[:, None] * dP / O[:, None], 0.0); core[:, 1] -= np.where(ov != 0, 2e-4, 0.0)   # overlay 成本记在首分钟
    Lz, Sz = np.column_stack([np.zeros(len(days)), Lm]), np.column_stack([np.zeros(len(days)), Sm])
    f = 1e8
    Q = pd.DataFrame({"cc": (core ** 2).sum(1) * f, "LL": (Lz ** 2).sum(1) * f, "SS": (Sz ** 2).sum(1) * f,
                      "EL": (core * Lz).sum(1) * f, "ES": (core * Sz).sum(1) * f, "LS": (Lz * Sz).sum(1) * f}, index=days)
    # 对账:分钟腿损益之和 vs 腿日损益(差 = 成本 + 15:40 无关);核心腿分钟之和 vs 核心腿日收益
    legchk = float(np.nanmax(np.abs(Lm.sum(1) + Sm.sum(1) - (I["L2"] + I["S2"]).to_numpy())))
    corechk = float(np.nanmax(np.abs(core.sum(1) - (w * I["bh"].to_numpy() + ov))[full_mask]))
    return Q, dict(minute_vs_gross=max(chk1, chk2), minute_vs_net_leg_bp=legchk * 1e4, core_vs_daily_bp=corechk * 1e4)


def qv_of(Q, kl, ks):
    return Q.cc + kl ** 2 * Q.LL + ks ** 2 * Q.SS + 2 * kl * Q.EL + 2 * ks * Q.ES + 2 * kl * ks * Q.LS


# ============================================================ 目标函数
def fit_obj(Z, Lb, Sb, Eb, Qd, lam, obj):
    """sigmoid PPP;obj ∈ {'mv','semi','qv'}。返回 (预测函数, 打分函数)。Qd: dict of arrays(仅 qv 用)。"""
    n, p1 = Z.shape; B = Lb + Sb + Eb
    if obj == "mv": gam = B.mean() / B.var()
    elif obj == "semi": gam = B.mean() / np.mean(np.minimum(B, 0) ** 2)
    elif obj == "legsemi": gam = B.mean() / (Eb.var() + np.mean(np.minimum(Lb, 0) ** 2) + np.mean(np.minimum(Sb, 0) ** 2))
    else: gam = B.mean() / np.mean(Qd["cc"] + Qd["LL"] + Qd["SS"] + 2 * Qd["EL"] + 2 * Qd["ES"] + 2 * Qd["LS"])
    def util(R, kl=None, ks=None, Qx=None):
        if obj == "mv": return R.mean() - 0.5 * gam * R.var()
        if obj == "semi": return R.mean() - 0.5 * gam * np.mean(np.minimum(R, 0) ** 2)
        if obj == "legsemi":
            return R.mean() - 0.5 * gam * (Qx["E"].var() + np.mean(np.minimum(kl * Qx["L"], 0) ** 2) + np.mean(np.minimum(ks * Qx["S"], 0) ** 2))
        q = Qx["cc"] + kl ** 2 * Qx["LL"] + ks ** 2 * Qx["SS"] + 2 * kl * Qx["EL"] + 2 * ks * Qx["ES"] + 2 * kl * ks * Qx["LS"]
        return R.mean() - 0.5 * gam * q.mean()
    def f(th):
        kl, dl = M.link(Z @ th[:p1], "sig"); ks, ds = M.link(Z @ th[p1:], "sig")
        R = kl * Lb + ks * Sb + Eb; dRl = (Lb * dl)[:, None] * Z; dRs = (Sb * ds)[:, None] * Z
        if obj == "mv":
            Rc = R - R.mean(); gl = dRl.mean(0) - gam * (Rc[:, None] * dRl).mean(0); gs = dRs.mean(0) - gam * (Rc[:, None] * dRs).mean(0); risk = R.var()
        elif obj == "legsemi":
            nl, ns = np.minimum(Lb, 0), np.minimum(Sb, 0)                 # k>0,故 min(k·L,0) = k·min(L,0)
            gl = dRl.mean(0) - gam * ((kl * nl * nl * dl)[:, None] * Z).mean(0); gs = dRs.mean(0) - gam * ((ks * ns * ns * ds)[:, None] * Z).mean(0)
            risk = Eb.var() + np.mean((kl * nl) ** 2) + np.mean((ks * ns) ** 2)
        elif obj == "semi":
            m_ = np.minimum(R, 0); gl = dRl.mean(0) - gam * (m_[:, None] * dRl).mean(0); gs = dRs.mean(0) - gam * (m_[:, None] * dRs).mean(0); risk = np.mean(m_ ** 2)
        else:
            q = Qd["cc"] + kl ** 2 * Qd["LL"] + ks ** 2 * Qd["SS"] + 2 * kl * Qd["EL"] + 2 * ks * Qd["ES"] + 2 * kl * ks * Qd["LS"]
            dql = 2 * kl * Qd["LL"] + 2 * Qd["EL"] + 2 * ks * Qd["LS"]; dqs = 2 * ks * Qd["SS"] + 2 * Qd["ES"] + 2 * kl * Qd["LS"]
            gl = dRl.mean(0) - 0.5 * gam * ((dql * dl)[:, None] * Z).mean(0); gs = dRs.mean(0) - 0.5 * gam * ((dqs * ds)[:, None] * Z).mean(0); risk = q.mean()
        g = np.r_[gl, gs] - 2 * lam * th
        return -(R.mean() - 0.5 * gam * risk - lam * np.sum(th ** 2)), -g
    r = minimize(f, np.zeros(2 * p1), jac=True, method="L-BFGS-B", options={"maxiter": 500}); th = r.x.reshape(2, p1)
    pred = lambda Zt: (M.link(Zt @ th[0], "sig")[0], M.link(Zt @ th[1], "sig")[0])
    return pred, util


def wf_obj(X, feats, Lb, Sb, Eb, Q, sel, days, obj):
    yrs = days.year; out = pd.DataFrame(np.nan, index=days, columns=["kl", "ks"]); lam_sel = {}
    Qa = {c: Q[c].to_numpy() for c in Q.columns}; sub = lambda idx: {c: v[idx] for c, v in Qa.items()}
    def design(idx, ref):
        mu, sd = X.loc[ref, feats].mean(), X.loc[ref, feats].std().replace(0, 1)
        return np.column_stack([np.ones(int(idx.sum())), ((X.loc[idx, feats] - mu) / sd).to_numpy()])
    for yr in U.OOS_YEARS:
        tr = sel & (yrs < yr); te = sel & (yrs == yr); cv = {}
        for lam in U.LAMS:
            sc = []
            for hy in sorted(set(yrs[tr])):
                a_, h_ = tr & (yrs != hy), tr & (yrs == hy)
                if a_.sum() < 150 or h_.sum() < 50: continue
                pr, util = fit_obj(design(a_, a_), Lb[a_], Sb[a_], Eb[a_], sub(a_), lam, obj); kl, ks = pr(design(h_, a_))
                one = np.ones(int(h_.sum()))
                sc.append(util(kl * Lb[h_] + ks * Sb[h_] + Eb[h_], kl, ks, sub(h_)) - util(Lb[h_] + Sb[h_] + Eb[h_], one, one, sub(h_)))
            cv[lam] = np.mean(sc) if sc else -np.inf
        lam = max(cv, key=cv.get); lam_sel[yr] = lam
        pr, _ = fit_obj(design(tr, tr), Lb[tr], Sb[tr], Eb[tr], sub(tr), lam, obj); out.loc[te, "kl"], out.loc[te, "ks"] = pr(design(te, tr))
    return out, lam_sel


def hac_ols(y, X, names):
    m = sm.OLS(y, sm.add_constant(X)).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    return pd.DataFrame({"系数": np.asarray(m.params)[1:], "t(HAC)": np.asarray(m.tvalues)[1:]}, index=names), m.rsquared


def main():
    t0 = time.time(); pd.set_option("display.width", 260, "display.max_rows", 200, "display.float_format", lambda v: f"{v:9.3f}")
    b = load_bars("QQQ"); days = b.days; yrs = days.year
    import disp_pred_study as DP
    I = SF.build_inputs(b); L2, S2, GL, GS, keep, ov, X, leg1 = (I[k] for k in ("L2", "S2", "GL", "GS", "keep", "ov", "X", "leg1"))
    dn30 = DP.targets(DP.Arr(b), 30).dn; full = pd.Series(b.last_idx.to_numpy() >= 389, index=days); td = pd.Series(days.dayofweek != 0, index=days)
    sel = (td & full & X.notna().all(1) & dn30.notna() & (days >= "2019-01-01")).to_numpy(); om = pd.Series(sel & (yrs >= 2021), index=days)
    Q, chk = build_qv(b, I); print(f"[对账] {chk}  ({time.time()-t0:.0f}s)")
    # ---------------- D1:Sleeve 损益 vs 位移、日内波动
    c = b.c.ffill(axis=1).to_numpy(); O = b.day_open.to_numpy(); li = b.last_idx.to_numpy(); n = np.arange(len(days))
    sd14 = daily_sigma(b.day_close, b.prev_close_adj, 14).reindex(days).to_numpy()
    r1 = np.full_like(c, np.nan); r1[:, 1:] = np.log(c[:, 1:] / c[:, :-1]); r1[:, 0] = np.log(c[:, 0] / O)
    rv = np.nansum(r1 ** 2, 1); disp = np.abs(np.log(c[n, li] / O))
    D = pd.DataFrame({"log_dn": np.log(disp / sd14 + 0.05), "log_nrv": np.log(rv / sd14 ** 2), "log_eff": np.log(np.clip(disp / np.nansum(np.abs(r1), 1), 1e-3, 1))}, index=days)
    m1 = sel & np.isfinite(D).all(1).to_numpy()
    print("\n=== D1. Sleeve(SOTA、未缩放)日损益 bp 对 log 位移、log 日内 RV(均除以 σ14 归一)的联合回归,2019–26 周二至周五 ===")
    print("相关矩阵:"); print(pd.concat([D[m1], (L2 + S2)[m1].rename("sleeve") * 1e4], axis=1).corr().round(3).to_string())
    for nm, yv in (("Sleeve", L2 + S2), ("多头腿", L2), ("空头腿", S2)):
        tab, r2 = hac_ols((yv[m1] * 1e4).to_numpy(), D.loc[m1, ["log_dn", "log_nrv"]].to_numpy(), ["log_dn", "log_nrv"])
        print(f"  [{nm}] R²={r2:.3f}\n{tab.to_string()}")
    # ---------------- 配置
    Lb, Sb, Eb = (L2 * 1e4).to_numpy(), (S2 * 1e4).to_numpy(), ((leg1 + ov) * 1e4).to_numpy()
    K = {}
    for nm, obj in (("B 均值–方差(现行)", "mv"), ("O2 下行半方差", "semi"), ("O4 日内已实现方差", "qv")):
        K[nm] = wf_obj(X, P.CLEAN, Lb, Sb, Eb, Q, sel, days, obj); print(f"  [{nm}] λ* {K[nm][1]}  ({time.time()-t0:.0f}s)", flush=True)
    kref = P.wf(X, pd.Series(Lb, index=days), pd.Series(Sb, index=days), pd.Series(Eb, index=days), sel, days, P.CLEAN)[0]
    print(f"[一致性] B(本脚本)vs repo P.wf:k 最大差 {float(np.nanmax(np.abs(K['B 均值–方差(现行)'][0].to_numpy() - kref.to_numpy()))):.2e}")
    # ---------------- D2:现行模型的 k 与事后位移/日内波动
    k0 = K["B 均值–方差(现行)"][0]; o = om.to_numpy() & np.isfinite(D).all(1).to_numpy()
    print("\n=== D2. 现行模型 OOS 的 log k 与事后变量的 Spearman 相关(2021–26)===")
    for leg in ("kl", "ks"):
        print(f"  {leg}: " + "  ".join(f"{v}: {pd.Series(np.log(k0[leg][o].to_numpy())).corr(pd.Series(D.loc[o, v].to_numpy()), method='spearman'):+.3f}" for v in ("log_dn", "log_nrv", "log_eff")))
    # ---------------- 评估
    oi = np.where(om.to_numpy())[0]; dev = yrs[oi] <= 2022; BI = M.block_idx(len(oi), 10, 2000, 0); srv = lambda x: x.mean() / x.std() * np.sqrt(252)
    def port(k):
        kl = k.kl.where(om, 1.0).fillna(1.0); ks = k.ks.where(om, 1.0).fillna(1.0)
        return (leg1 + kl * L2 + ks * S2 + ov - (kl * GL + ks * GS) * keep).to_numpy()[oi], qv_of(Q, kl, ks).to_numpy()[oi]
    ref, qref = port(k0); refb = np.array([srv(ref[r]) for r in BI]); rows = []; pans = {}
    for nm, (k, lam) in K.items():
        x, q = port(k); db = np.array([srv(x[r]) for r in BI]) - refb
        r_ = dict(方案=nm, 组合SR=srv(x), ΔSR全=srv(x) - srv(ref), lo=np.percentile(db, 2.5), hi=np.percentile(db, 97.5),
                  ΔSR_dev=srv(x[dev]) - srv(ref[dev]), ΔSR_test=srv(x[~dev]) - srv(ref[~dev]),
                  日内已实现波动年化=float(np.sqrt(q.mean() / 1e8 * 252)), 日收益波动年化=float(x.std() * np.sqrt(252)), k多=float(k.kl[om].mean()), k空=float(k.ks[om].mean()))
        r_.update({str(y): srv(x[yrs[oi] == y]) - srv(ref[yrs[oi] == y]) for y in range(2021, 2027)})
        r_["优于B"] = bool(nm != "B 均值–方差(现行)" and r_["ΔSR全"] > 0 and r_["ΔSR_dev"] > 0 and r_["ΔSR_test"] > 0 and r_["lo"] > -0.05); rows.append(r_)
        pans[nm] = SF.panel(pd.Series(x))
    R = pd.DataFrame(rows); print("\n=== 组合层(QQQ ① 腿 + v3 Sleeve),OOS 2021–26,ΔSR 相对 B ==="); print(R.to_string(index=False))
    Pn = pd.DataFrame(pans).T; print("\n=== 分布面板 ==="); print(Pn.to_string())
    R.to_csv(TAB / "ivol_objective.csv", index=False); Pn.to_csv(TAB / "ivol_objective_panel.csv"); print(f"总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
