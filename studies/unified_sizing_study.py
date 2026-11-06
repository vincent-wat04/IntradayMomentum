"""统一的仓位模型:低波动日缩仓 + 恐慌日空头加码放进同一个可学习的 sizing 框架;
附 Reactive Volatility Model(Valeyre et al. 2013)在 QQQ 上的复现。—— docs/PROPOSAL_unified_sizing.md(预注册)。
python3 unified_sizing_study.py"""
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
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.linear_model import LogisticRegression, Ridge

sys.path.insert(0, str(Path(__file__).resolve().parent))
import disp_pred_study as DP
import liquidity_inverse_study as LI
import lowvol_avoidance_study as L
import round3_common as C
import sota_stack_study as S
import trend_rv_sleeve_study as T
from audit import reimpl_noise as RI
from etft.intraday.bars import load_bars
from sens_rv_weights import pct_w
import paths as _P

OUT, TAB = L.OUT, L.TAB
FEATS = ["ffa_rank", "gap_n", "r30_n", "b30_n", "lvix_z", "dlvix", "rvm_panic", "rvm_ratio", "P3", "P4"]
OOS_YEARS = range(2021, 2027)
LAMS = (0.1, 0.3, 1.0, 3.0, 10.0)
NPERM = 2000
RVM = dict(lam_s=0.0241, lam_f=0.1484, ell=8.0, phi=1 / 0.3, lam_sig=1 / 40)


# ======================================================== Reactive Volatility Model(Valeyre et al. 2013,式 1–7)
def rvm(I: pd.Series, p=RVM):
    I = I.dropna(); v = I.to_numpy(float); n = len(v)
    F = lambda z: np.tanh(p["phi"] * z) / p["phi"]
    Ls = np.empty(n); Lf = np.empty(n); Lv = np.empty(n); s2 = np.empty(n)
    Ls[0] = Lf[0] = Lv[0] = v[0]
    r0 = np.diff(v[:41]) / v[:40]; s2[0] = np.var(r0)
    for t in range(1, n):
        Ls[t] = (1 - p["lam_s"]) * Ls[t - 1] + p["lam_s"] * v[t]
        Lf[t] = (1 - p["lam_f"]) * Lf[t - 1] + p["lam_f"] * v[t]
        Ls_hat = v[t] * (1 + F((Ls[t] - v[t]) / v[t]))
        Lv[t] = Ls_hat * (1 + F((Lf[t] / v[t]) ** p["ell"] - 1))
        s2[t] = (1 - p["lam_sig"]) * s2[t - 1] + p["lam_sig"] * ((v[t] - v[t - 1]) / Lv[t]) ** 2
    sig_tilde = np.sqrt(s2); sig_I = sig_tilde * Lv / v
    return pd.DataFrame({"I": v, "Ls": Ls, "Lf": Lf, "L": Lv, "sig_tilde": sig_tilde, "sig_I": sig_I}, index=I.index)


def ewma_vol(r, lam=1 / 40):
    v = r.fillna(0).to_numpy(); s2 = np.empty(len(v)); s2[0] = np.var(v[:40])
    for t in range(1, len(v)): s2[t] = (1 - lam) * s2[t - 1] + lam * v[t] ** 2
    return pd.Series(np.sqrt(s2), index=r.index)


def garch_vol(r):
    x = r.fillna(0).to_numpy() - r.mean()
    def filt(w, a, b_):
        h = np.empty(len(x)); h[0] = np.var(x)
        for t in range(1, len(x)): h[t] = w + a * x[t - 1] ** 2 + b_ * h[t - 1]
        return h
    def nll(z):
        w, a, b_ = np.exp(z[0]), expit(z[1]) * 0.3, expit(z[2])
        if a + b_ >= 0.9999: return 1e12
        h = filt(w, a, b_); return float(np.sum(np.log(h) + x ** 2 / h))
    res = minimize(nll, [np.log(np.var(x) * 0.02), -1.5, 2.5], method="Nelder-Mead", options={"maxiter": 3000})
    w, a, b_ = np.exp(res.x[0]), expit(res.x[1]) * 0.3, expit(res.x[2])
    h = filt(w, a, b_)     # h[t] 用到 t−1 的收益:对 t 日的条件方差
    return pd.Series(np.sqrt(h), index=r.index), (w, a, b_)


# ======================================================== 腿级损益(复用 liquidity_inverse_study 的已验证重建)
def leg_pnl(b, base):
    days = b.days
    PH = LI.run_phases_fills(b); offs = list(PH)
    TR = pd.concat([LI.trades_from_fills(F, off) for off, (rr, F) in PH.items() if len(F)], ignore_index=True)
    cm_day = pd.concat([PH[o][1].groupby("day").commission.sum().reindex(days).fillna(0.0) for o in offs], axis=1).astype(float); cm_day.columns = offs
    scale = (b.day_close / LI.tqqq_traded_px(days)) / 3.0
    cost_adj = (cm_day - cm_day.mul(scale.fillna(1.0), axis=0)).mean(axis=1)
    legs, cms = {}, {}
    for sd_, nm in ((1, "long"), (-1, "short")):
        sub = TR[TR.side == sd_]
        legs[nm] = sub.pivot_table(index="day", columns="phase", values="pnl_net", aggfunc="sum").reindex(index=days, columns=offs).fillna(0.0).mean(axis=1)
        cms[nm] = sub.assign(cm=sub.cm_in + sub.cm_out).pivot_table(index="day", columns="phase", values="cm", aggfunc="sum").reindex(index=days, columns=offs).fillna(0.0).mean(axis=1)
    tot = (cms["long"] + cms["short"]).replace(0, np.nan)
    Lg = legs["long"] + (cost_adj * cms["long"] / tot).fillna(0.0)
    Sg = legs["short"] + (cost_adj * cms["short"] / tot).fillna(0.0)
    err = float((Lg + Sg - base).abs().max()) * 1e4
    return Lg, Sg, err


# ======================================================== 特征
def build_features(b, cand):
    days = b.days; a = DP.Arr(b); c = a.c; h = a.h; l = a.l; O = a.O; sd = pd.Series(a.sd, index=days)
    gap = pd.Series(np.log(O / b.prev_close_adj.to_numpy()), index=days)
    r30 = pd.Series(np.log(c[:, 29] / O), index=days)
    hi30 = np.maximum(h[:, :30].max(1), O); lo30 = np.minimum(l[:, :30].min(1), O)
    p30 = pd.Series((np.log(hi30 / lo30)) ** 2 / (4 * np.log(2)), index=days)
    vix = pd.read_parquet(_P.RAW / "cboe" / "VIX.parquet")["close"]; vix.index = pd.to_datetime(vix.index).tz_localize(None); vix = vix[~vix.index.duplicated()].sort_index()
    lv = np.log(vix); lvz = ((lv - lv.rolling(252, min_periods=126).mean()) / lv.rolling(252, min_periods=126).std())
    # RVM 在日频调整价上;10:00 的即时恐慌因子:把 10:00 的成交价换算到调整价口径
    q = pd.read_parquet(_P.RAW / "yahoo" / "QQQ.parquet"); q.index = pd.to_datetime(q.index).tz_localize(None)
    R = rvm(q.close)
    adjf = (q.close.reindex(days) / b.day_close).shift(1)                   # 前一日:调整价 / 实际价
    p1000_adj = pd.Series(c[:, 29], index=days) * adjf
    Lf_prev = R.Lf.shift(1).reindex(days)
    X = pd.DataFrame({
        "ffa_rank": S.roll_rank(cand["FFA-P"]).reindex(days),
        "gap_n": gap / sd, "r30_n": r30 / (sd * np.sqrt(30 / 390)), "b30_n": np.log((p30 / (sd ** 2 * 30 / 390)).where(p30 > 0)),
        "lvix_z": lvz.shift(1).reindex(days, method="ffill"), "dlvix": lv.diff().shift(1).reindex(days, method="ffill"),
        "rvm_panic": RVM["ell"] * (Lf_prev - p1000_adj) / Lf_prev,
        "rvm_ratio": np.log(R.L / R.I).shift(1).reindex(days),
    }, index=days)
    vixd = vix.shift(1).reindex(days, method="ffill")
    X["P3"] = ((gap < 0) & (vixd > 20)).astype(float); X["P4"] = (X.P3.astype(bool) & (r30 < 0)).astype(float)
    return X, R, q, sd


# ======================================================== 模型
def ppp_fit(Z, Lb, Sb, lam):
    """k_leg = 2σ(θ_legᵀ[1,z]),最大化 mean(R) − γ/2·var(R) − λ‖θ‖²;R、L、S 以 bp 计。返回 θ (2, p+1)。"""
    n, p = Z.shape; Zt = np.column_stack([np.ones(n), Z]); B = Lb + Sb; gam = B.mean() / B.var() if B.var() > 0 else 0.0
    def fobj(th):
        tl, ts = th[:p + 1], th[p + 1:]
        kl, ks = 2 * expit(Zt @ tl), 2 * expit(Zt @ ts); R = kl * Lb + ks * Sb; Rc = R - R.mean()
        U = R.mean() - 0.5 * gam * R.var() - lam * np.sum(th ** 2)
        dRl = (Lb * kl * (1 - kl / 2))[:, None] * Zt; dRs = (Sb * ks * (1 - ks / 2))[:, None] * Zt
        gl = dRl.mean(0) - gam * (Rc[:, None] * dRl).mean(0); gs = dRs.mean(0) - gam * (Rc[:, None] * dRs).mean(0)
        g = np.r_[gl, gs] - 2 * lam * th
        return -U, -g
    res = minimize(fobj, np.zeros(2 * (p + 1)), jac=True, method="L-BFGS-B", options={"maxiter": 500})
    return res.x.reshape(2, p + 1), gam


def ppp_apply(th, Z):
    Zt = np.column_stack([np.ones(len(Z)), Z]); return 2 * expit(Zt @ th[0]), 2 * expit(Zt @ th[1])


def utility(R, gam): return R.mean() - 0.5 * gam * R.var()


def walk_forward(X, Lb, Sb, dn, sel, days):
    out = {k: pd.DataFrame(np.nan, index=days, columns=["kl", "ks"]) for k in ("U2", "U3", "U4a", "U4b")}
    diag = pd.DataFrame(np.nan, index=days, columns=["p_lo", "p_hi", "y_lo", "y_hi", "yhat_L", "yhat_S", "d"])
    coefs = []; lam_sel = {}
    yrs = days.year
    for yr in OOS_YEARS:
        tr = sel & (yrs < yr); te = sel & (yrs == yr)
        if tr.sum() < 200 or te.sum() == 0: continue
        mu, sdv = X[tr].mean(), X[tr].std().replace(0, 1.0)
        Ztr, Zte = ((X[tr] - mu) / sdv).to_numpy(), ((X[te] - mu) / sdv).to_numpy()
        Ltr, Str = Lb[tr].to_numpy(), Sb[tr].to_numpy()
        # ---- U2 双头岭回归 + sigmoid
        kk = {}
        for nm, y_ in (("L", Ltr), ("S", Str)):
            m = Ridge(alpha=10.0).fit(Ztr, y_); fit = m.predict(Ztr); pr = m.predict(Zte)
            kk[nm] = 2 * expit((pr - np.median(fit)) / (fit.std() + 1e-9)); diag.loc[te, f"yhat_{nm}"] = pr
            coefs.append(dict(年=yr, 模型=f"U2-ridge-{nm}", **dict(zip(FEATS, m.coef_))))
        out["U2"].loc[te, "kl"], out["U2"].loc[te, "ks"] = kk["L"], kk["S"]
        # ---- U3 PPP(λ 用训练期留一年交叉验证)
        tr_years = sorted(set(yrs[tr]))
        cv = {}
        for lam in LAMS:
            sc = []
            for hy in tr_years:
                a_ = tr & (yrs != hy); h_ = tr & (yrs == hy)
                if a_.sum() < 150 or h_.sum() < 50: continue
                mu2, sd2 = X[a_].mean(), X[a_].std().replace(0, 1.0)
                th, gam = ppp_fit(((X[a_] - mu2) / sd2).to_numpy(), Lb[a_].to_numpy(), Sb[a_].to_numpy(), lam)
                kl, ks = ppp_apply(th, ((X[h_] - mu2) / sd2).to_numpy())
                sc.append(utility(kl * Lb[h_].to_numpy() + ks * Sb[h_].to_numpy(), gam) - utility(Lb[h_].to_numpy() + Sb[h_].to_numpy(), gam))
            cv[lam] = np.mean(sc) if sc else -np.inf
        lam_b = max(cv, key=cv.get); lam_sel[yr] = (lam_b, {k: round(v, 3) for k, v in cv.items()})
        th, gam = ppp_fit(Ztr, Ltr, Str, lam_b); kl, ks = ppp_apply(th, Zte)
        out["U3"].loc[te, "kl"], out["U3"].loc[te, "ks"] = kl, ks
        coefs.append(dict(年=yr, 模型="U3-PPP-long", 截距=th[0, 0], **dict(zip(FEATS, th[0, 1:]))))
        coefs.append(dict(年=yr, 模型="U3-PPP-short", 截距=th[1, 0], **dict(zip(FEATS, th[1, 1:]))))
        # ---- U4 双分类器(低位移 / 高位移)
        q20, q80 = dn[tr].quantile([0.2, 0.8]); ylo, yhi = (dn[tr] <= q20).astype(int), (dn[tr] >= q80).astype(int)
        clo = LogisticRegression(C=0.5, max_iter=1000).fit(Ztr, ylo); chi = LogisticRegression(C=0.5, max_iter=1000).fit(Ztr, yhi)
        plo_tr, phi_tr = clo.predict_proba(Ztr)[:, 1], chi.predict_proba(Ztr)[:, 1]
        plo, phi = clo.predict_proba(Zte)[:, 1], chi.predict_proba(Zte)[:, 1]
        d_tr, d_te = phi_tr - plo_tr, phi - plo
        k4 = 2 * expit((d_te - np.median(d_tr)) / (d_tr.std() + 1e-9))
        out["U4a"].loc[te, "kl"] = out["U4a"].loc[te, "ks"] = k4
        skip = (plo >= np.quantile(plo_tr, 0.8)).astype(float)
        out["U4b"].loc[te, "kl"] = out["U4b"].loc[te, "ks"] = 1 - skip
        diag.loc[te, "p_lo"], diag.loc[te, "p_hi"], diag.loc[te, "d"] = plo, phi, d_te
        diag.loc[te, "y_lo"], diag.loc[te, "y_hi"] = (dn[te] <= q20).astype(float).to_numpy(), (dn[te] >= q80).astype(float).to_numpy()
        coefs.append(dict(年=yr, 模型="U4-logit-低位移", **dict(zip(FEATS, clo.coef_[0]))))
        coefs.append(dict(年=yr, 模型="U4-logit-高位移", **dict(zip(FEATS, chi.coef_[0]))))
        print(f"  {yr}: 训练 {int(tr.sum())} 天 / 测试 {int(te.sum())} 天;PPP λ*={lam_b}", flush=True)
    return out, diag, pd.DataFrame(coefs), lam_sel


def calib(y, p):
    m = np.isfinite(y) & np.isfinite(p); y, p = y[m], np.clip(p[m], 1e-4, 1 - 1e-4)
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(np.log(p / (1 - p))[:, None], y.astype(int))
    return float(np.mean((p - y) ** 2)), float(lr.coef_[0, 0]), float(lr.intercept_[0])


def main():
    t0 = time.time(); pd.set_option("display.width", 270, "display.max_rows", 300, "display.float_format", lambda x: f"{x:9.3f}")
    b = load_bars("QQQ"); days = b.days; yrs = days.year
    base = pd.read_pickle(OUT / "sleeve_anatomy_curve.pkl"); base.index = pd.to_datetime(base.index); base = base.reindex(days).fillna(0.0)
    Lg, Sg, err = leg_pnl(b, base); print(f"[校验] 腿级(多+空)重建 vs 基线 最大绝对差 {err:.6f} bp")
    cand, _, _ = S.build_candidates(b)
    X, R, q, sd = build_features(b, cand)
    full = pd.Series(b.last_idx.to_numpy() >= 389, index=days); td = pd.Series(days.dayofweek != 0, index=days)
    dn = DP.targets(DP.Arr(b), 30).dn
    sel = (td & full & X.notna().all(1) & dn.notna() & (days >= "2019-01-01")).to_numpy()
    Lb, Sb = Lg * 1e4, Sg * 1e4
    print(f"可用样本 {int(sel.sum())} 天(周二至周五、完整日、特征齐全);OOS {int((sel & (yrs >= 2021)).sum())} 天;特征起点 {X.dropna().index[0].date()}")
    # ======================================================= A. RVM 复现(论文图 4 的 QQQ 版)
    print("\n" + "=" * 110 + "\nA) Reactive Volatility Model 在 QQQ 上的复现(Valeyre et al. 2013,图 4 的设定)\n" + "=" * 110)
    r = q.close.pct_change(); ew = ewma_vol(r); gv, gp = garch_vol(r)
    rows = []
    for ivn in ("VXN", "VIX"):
        iv = pd.read_parquet(_P.RAW / "cboe" / f"{ivn}.parquet")["close"]; iv.index = pd.to_datetime(iv.index).tz_localize(None); iv = iv[~iv.index.duplicated()]
        D = pd.DataFrame({"iv": iv, "rvm": R.sig_I * np.sqrt(252) * 100, "ewma": ew * np.sqrt(252) * 100, "garch": gv.shift(-1) * np.sqrt(252) * 100}).dropna()   # garch 的 h[t] 是对 t 的预测,用 t+1 的值与 t 日收盘对齐
        D = D[D.index >= "2009-09-14"]; dD = D.diff().dropna()
        for m_ in ("rvm", "ewma", "garch"):
            x_, y_ = dD[m_].to_numpy(), dD.iv.to_numpy(); sl, ic = np.polyfit(x_, y_, 1); r2 = np.corrcoef(x_, y_)[0, 1] ** 2
            rows.append(dict(隐含波动=ivn, 估计量=m_, n=len(dD), 日变化斜率=sl, R2=r2, 水平相关=float(D[m_].corr(D.iv))))
    RA = pd.DataFrame(rows); print(RA.to_string(index=False)); RA.to_csv(TAB / "unified_rvm_replication.csv", index=False)
    print(f"  GARCH(1,1) 参数 ω={gp[0]:.2e} α={gp[1]:.3f} β={gp[2]:.3f}(论文 V2X:α≈0.106 β≈0.897)")
    # RVM 作为 Sleeve 仓位(描述性)
    expo0 = np.minimum(4.0, 0.02 / RI.daily_sigma(b.day_close, b.prev_close_adj, 14).reindex(days))
    sig_rvm = R.sig_I.shift(1).reindex(days, method="ffill"); ratio = (np.minimum(4.0, 0.02 / sig_rvm) / expo0).fillna(1.0)
    xr = base * ratio; m_ = sel
    print(f"  RVM 替换 σ₁₄ 做 Sleeve 仓位:Sleeve SR {L.sr(base[m_]):.3f} → {L.sr(xr[m_]):.3f};高位移日(dn≥80%)平均仓位倍数 {float(ratio[m_ & (dn >= dn[m_].quantile(.8)).to_numpy()].mean()):.3f}、低位移日 {float(ratio[m_ & (dn <= dn[m_].quantile(.2)).to_numpy()].mean()):.3f}")
    # ======================================================= B. 走步训练
    print("\n" + "=" * 110 + "\nB) walk-forward(每年初用此前全部年份训练)\n" + "=" * 110)
    outs, diag, coefs, lam_sel = walk_forward(X, Lb, Sb, dn, sel, days)
    print("  PPP 留一年 CV 效用增益(bp):", {y: v for y, v in lam_sel.items()})
    oos = sel & (yrs >= 2021)
    # U1 手工规则(样本内)
    u1l = pd.Series(1.0, index=days); u1s = pd.Series(1.0, index=days)
    skip = (X.ffa_rank <= 0.2); p3 = X.P3.astype(bool)
    u1l[p3] = 0.0; u1s[p3] = 1.5; u1l[skip] = 0.0; u1s[skip] = 0.0
    outs["U1"] = pd.DataFrame({"kl": u1l.where(pd.Series(oos, index=days)), "ks": u1s.where(pd.Series(oos, index=days))})
    # ======================================================= C. 分类诊断:AUC / 校准 / precision-recall / 利润曲线
    print("\n" + "=" * 110 + "\nC) 分类诊断(OOS 2021–2026)\n" + "=" * 110)
    dg = diag[oos]
    for head, yc in (("低位移头 p_lo", "y_lo"), ("高位移头 p_hi", "y_hi")):
        pc = "p_lo" if "低" in head else "p_hi"; y_, p_ = dg[yc].to_numpy(), dg[pc].to_numpy()
        br, slope, icp = calib(y_, p_)
        print(f"  {head}:AUC {L.auc(y_.astype(int), p_):.3f};Brier {br:.4f}(基准率 {y_.mean():.3f} 时的常数预测 Brier {y_.mean()*(1-y_.mean()):.4f});校准斜率 {slope:.2f}(1 = 完美)")
    pnl = (Lb + Sb)[oos]
    hi_mean = float(pnl[dg.y_hi == 1].mean()); lo_mean = float(pnl[dg.y_lo == 1].mean()); mid_mean = float(pnl[(dg.y_hi == 0) & (dg.y_lo == 0)].mean())
    print(f"  OOS Sleeve 日均(bp):低位移日 {lo_mean:+.1f}、中间 {mid_mean:+.1f}、高位移日 {hi_mean:+.1f} —— 两类错误的代价不对称:漏掉高位移日 ≈ {hi_mean:.0f}bp,误交易低位移日 ≈ {-lo_mean:.0f}bp")
    scores = {"U4 p_lo(越高越该跳)": dg.p_lo, "FFA-P 分位(越低越该跳)": -X.ffa_rank[oos], "U2 预测 Sleeve 收益(越低越该跳)": -(dg.yhat_L + dg.yhat_S), "U4 d=p_hi−p_lo(越低越该跳)": -dg.d}
    rows = []
    for nm, sc in scores.items():
        for fr in (0.05, 0.10, 0.20, 0.30, 0.40):
            thr = sc.quantile(1 - fr); flag = (sc >= thr).to_numpy()
            prec = float(dg.y_lo[flag].mean()); rec = float(dg.y_lo[flag].sum() / dg.y_lo.sum()); miss_hi = float(dg.y_hi[flag].mean())
            mu_flag = float(pnl[flag].mean()); mu_nf = float(pnl[flag & (dg.y_lo == 0).to_numpy()].mean())
            p_star = mu_nf / (mu_nf - lo_mean) if (mu_nf - lo_mean) != 0 else np.nan
            new = pnl.copy(); new[flag] = 0.0
            rows.append(dict(分数=nm, 标记率=fr, precision低位移=prec, recall低位移=rec, 被标记中高位移日占比=miss_hi, 被标记日Sleeve均bp=mu_flag,
                             盈亏平衡precision=p_star, 跳过后ΔSR=L.sr(new / 1e4) - L.sr(pnl / 1e4)))
    PR = pd.DataFrame(rows); print("\n  利润曲线 / precision-recall(标记 = 当日跳过;ex-post 排序,仅诊断):"); print(PR.to_string(index=False)); PR.to_csv(TAB / "unified_profit_curve.csv", index=False)
    # ======================================================= D. 经济评价
    print("\n" + "=" * 110 + "\nD) 经济评价(OOS 2021–2026,周二至周五;组合 = L1 RV[0.5,1.5] + Sleeve)\n" + "=" * 110)
    bh = pd.read_pickle(OUT / "strategy_comparison_bh.pkl").dropna(); bh.index = pd.to_datetime(bh.index)
    px_all = C.load_close("yahoo/QQQ.parquet"); p_all = pct_w(px_all.pct_change().rolling(21).std(), 504)
    port = lambda s_: T.build(bh, s_.reindex(bh.index).fillna(0.0), (1.0, 1.0), (("lin", .5, 1.5), None, 1.0), 1.0, px_all, p_all)[0]
    base_p = port(base); om = pd.Series(oos, index=days)
    def sleeve_of(o):
        kl = o.kl.where(om, 1.0).fillna(1.0); ks = o.ks.where(om, 1.0).fillna(1.0); return kl * Lg + ks * Sg, kl, ks
    idx = np.where(oos)[0]; bsr = L.sr(base.iloc[idx]); rg = np.random.default_rng(5); shifts = rg.integers(60, len(idx) - 60, NPERM)
    rows, nulls, obs, pan = [], {}, {}, [S.panel(base[oos], "U0 基线")]
    Lv, Sv = Lg.to_numpy()[idx], Sg.to_numpy()[idx]
    for nm in ("U1", "U2", "U3", "U4a", "U4b"):
        s_, kl, ks = sleeve_of(outs[nm]); xp = port(s_)
        o_ = L.sr(s_.iloc[idx]) - bsr; obs[nm] = o_
        if nm != "U1":
            klv, ksv = kl.to_numpy()[idx], ks.to_numpy()[idx]
            nulls[nm] = np.array([L.sr(pd.Series(np.roll(klv, s) * Lv + np.roll(ksv, s) * Sv)) for s in shifts]) - bsr
        pw = lambda z, a_, z_: z[(z.index >= a_) & (z.index <= z_)]
        row = dict(方案=nm, 平均k多=float(kl[oos].mean()), 平均k空=float(ks[oos].mean()), k多_std=float(kl[oos].std()), k空_std=float(ks[oos].std()),
                   Sleeve_ΔSR_dev=L.sr(s_[oos & (yrs <= 2022)]) - L.sr(base[oos & (yrs <= 2022)]), Sleeve_ΔSR_test=L.sr(s_[oos & (yrs >= 2023)]) - L.sr(base[oos & (yrs >= 2023)]), Sleeve_ΔSR_全=o_,
                   组合ΔSR_dev=L.sr(pw(xp, "2021-01-01", "2022-12-31")) - L.sr(pw(base_p, "2021-01-01", "2022-12-31")),
                   组合ΔSR_test=L.sr(pw(xp, "2023-01-01", "2026-12-31")) - L.sr(pw(base_p, "2023-01-01", "2026-12-31")),
                   组合ΔSR_全=L.sr(pw(xp, "2021-01-01", "2026-12-31")) - L.sr(pw(base_p, "2021-01-01", "2026-12-31")))
        d, dist = T.boot_dsr(pw(xp, "2021-01-01", "2026-12-31"), pw(base_p, "2021-01-01", "2026-12-31"), nb=1500, L=10, seed=0)
        row.update(组合ΔSR_lo=np.percentile(dist, 2.5), 组合ΔSR_hi=np.percentile(dist, 97.5))
        row.update({f"{y}": L.sr(s_[oos & (yrs == y)]) - L.sr(base[oos & (yrs == y)]) for y in OOS_YEARS})
        rows.append(row); pan.append(S.panel(s_[oos], nm))
    fam = np.max(np.vstack(list(nulls.values())), axis=0)
    ED = pd.DataFrame(rows); ED["p单"] = [float(np.mean(nulls[n] >= obs[n])) if n in nulls else np.nan for n in ED.方案]
    ED["p族内(4)"] = [float(np.mean(fam >= obs[n])) if n in nulls else np.nan for n in ED.方案]
    print(ED.to_string(index=False)); ED.to_csv(TAB / "unified_econ.csv", index=False)
    Pn = pd.DataFrame(pan); print("\n  Sleeve 分布面板(OOS 2021–2026):"); print(Pn[["方案", "均值bp", "Sharpe", "Sortino", "CAGR", "MDD", "Calmar", "偏度", "峰度", "CVaR5%bp", "前5%日占利润", "最差日bp"]].to_string(index=False)); Pn.to_csv(TAB / "unified_panel.csv", index=False)
    # 条件仓位:恐慌日与低波日上学到的 k
    print("\n  学到的仓位在关键状态上的平均值(OOS):")
    st = {"P3 跳空跌&VIX>20": X.P3.astype(bool), "P4": X.P4.astype(bool), "FFA 分位≤0.2": X.ffa_rank <= 0.2, "高位移日(事后)": dn >= dn[oos].quantile(.8), "低位移日(事后)": dn <= dn[oos].quantile(.2), "其余": pd.Series(True, index=days)}
    rows = []
    for nm in ("U2", "U3", "U4a"):
        _, kl, ks = sleeve_of(outs[nm])
        for sn, m_ in st.items():
            mm = oos & m_.to_numpy(); rows.append(dict(方案=nm, 状态=sn, 天数=int(mm.sum()), k多=float(kl[mm].mean()), k空=float(ks[mm].mean())))
    KS = pd.DataFrame(rows); print(KS.pivot_table(index="状态", columns="方案", values=["k多", "k空"], sort=False).round(3).to_string()); KS.to_csv(TAB / "unified_k_by_state.csv", index=False)
    print("\n  系数(最后一折 2026 与六折均值):")
    cm_ = coefs.groupby("模型")[FEATS].mean(); print(cm_.round(3).to_string()); coefs.to_csv(TAB / "unified_coefs.csv", index=False)
    pd.to_pickle(dict(outs=outs, diag=diag, X=X, Lg=Lg, Sg=Sg), OUT / "unified_sizing.pkl")
    print(f"\n总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
