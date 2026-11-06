"""σ14 直接喂给仓位模型 vs 在 σ14 波动目标之上叠加模型 —— 预注册(运行前写定,2026-10-20)

问题(用户):与其 `仓位 = min(4, 2%/σ14) × k(z)`(现 v3:PPP 在波动目标之上做乘数),不如把 σ14 作为输入直接喂给仓位模型。

设定:完全沿用 v3 ① 版(tearsheet_v3.py):Sleeve = 实盘 + FOMC≥14:30 + CPI 回补 1× + FFA-P 跳过 20% + 统一仓位(U3p-clean,
按 ① 组合重训)+ 15:40 平仓;QQQ 腿 = RV21[0.5,1.5]×SMA200(e=0.5)。走步、λ 网格、留一年交叉验证、OOS 2021–26、特征 CLEAN(7 个)均不变。
关键事实:引擎的 Sleeve 损益对仓位是线性的(股数、佣金、半价差都按股计),所以 "单位仓位损益" u = 腿损益 / e_vt 是精确的,
任何新仓位 x_t 都等价于在原腿上乘 k_eff = x_t / e_vt(15:40 平仓、跳过、CPI overlay 照原样作用)。

配置(3 个新配置,族内校正按 3 个;B 只是复现,不算新配置):
  B   基线复现:x = e_vt · 2σ(θᵀz)                              (= v3 ①,应复现 test Sharpe 2.46)
  E1  用户原意(替换):x = c · 2σ(θᵀ[z, log σ14]),c = 训练期 e_vt 均值;不再有固定的 1/σ14。θ=0 时为常数仓位。
  E2  替换 + 允许幂律:x = min(4, exp(a + θᵀz + β·log σ14));β、a 不受惩罚(由数据学出波动目标的指数;β=−1 即原规则)。
  E3  我的版本(保留 + 增强):x = e_vt · 2σ(θᵀ[z, log σ14]);σ14 作为额外特征,惩罚把它收缩回原规则。
共同:两条腿各自一套参数;目标 = 组合效用 mean − γ/2·var(γ 取基线一阶条件值,与 U3p 相同);x 上限 4(引擎杠杆上限)。

判定(写在跑之前):
  * 主指标 = 组合层(QQQ 腿 + v3 Sleeve)OOS 2021–26 的 ΔSR(相对 B),块自助 95% 区间;dev(21–22)、test(23–26)分别报。
  * "优于 B" 须同时满足:全 OOS ΔSR > 0、dev 与 test 同号为正、区间下界 > −0.05;否则判 "不优于"。
  * 一律报分布面板(均值/中位/Sharpe/Sortino/CAGR/MDD/Calmar/偏度/超额峰度/CVaR5%/盈亏比/前 5% 日利润占比/最差日)与分年。
  * 机理诊断(非选择):E2 学到的 β;单位仓位 vs 波动目标口径下 "训练效用由高波动日主导" 的程度(前 5% σ14 日占平方损益的比例)。
python3 sizing_sigma_feed.py
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
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import kurtosis, skew

sys.path.insert(0, str(Path(__file__).resolve().parent))
import disp_pred_study as DP
import ffa_vol_study as FV
import letf_posthoc as LP
import liquidity_inverse_study as LI
import lowvol_avoidance_study as L
import round3_common as C
import sota_stack_study as S
import sota_unified_stack as SU
import strategy_comparison as SC
import tearsheet as TS
import trend_rv_sleeve_study as T
import unified_sizing_posthoc as P
import unified_sizing_study as U
from etft.intraday.bars import load_bars
from audit.reimpl_noise import daily_sigma                         # 与 RI.simulate 的 voltarget 完全相同的 σ14
from event_calendar import FOMC_DECISION

OUT, TAB = L.OUT, L.TAB
MAX_LEV, TGT = 4.0, 0.02


# ============================================================ 1. 重建 v3 ① 的全部输入(缓存到 OUT)
def ffa_p_series():
    p = OUT / "ffa_p_only.pkl"
    if p.exists(): return pd.read_pickle(p)
    q, R, r = FV.load(); dates = R.index; y = np.log(R.to_numpy())
    start = int(np.searchsorted(dates, pd.Timestamp("2004-01-01")))
    FP, _ = FV.ffa_paper(y, start); s = pd.Series(FP[1], index=dates); pd.to_pickle(s, p); return s


def build_inputs(b):
    days = b.days
    t0 = time.time()
    scale = (b.day_close / LI.tqqq_traded_px(days)) / 3.0
    PHb = LI.run_phases_fills(b); print(f"  30 相位成交 {time.time()-t0:.0f}s", flush=True)
    PHf = SU.fomc_late_PH(b); print(f"  FOMC≥14:30 相位成交 {time.time()-t0:.0f}s", flush=True)
    Lb0, Sb0 = SU.legs_from_PH(PHb, days, scale); Lf, Sf = SU.legs_from_PH(PHf, days, scale)
    fm = pd.Series(days.isin(pd.DatetimeIndex(FOMC_DECISION)), index=days)
    L1, S1 = Lb0.where(~fm, Lf), Sb0.where(~fm, Sf)
    ffap = ffa_p_series(); print(f"  FFA-P {time.time()-t0:.0f}s", flush=True)
    skip = (S.roll_rank(ffap).reindex(days) <= 0.2).fillna(False); keep = (~skip).astype(float)
    ov = S.cpi_overlay(b)
    L2, S2 = L1 * keep, S1 * keep
    cols = list(b.c.columns); ob = b.o.to_numpy()
    GLb, GSb = LP.G_series(PHb, days, ob, cols, "15:40"); GLf, GSf = LP.G_series(PHf, days, ob, cols, "15:40")
    GL, GS = GLb.where(~fm, GLf), GSb.where(~fm, GSf)
    X, _, _, sd_arr = U.build_features(b, {"FFA-P": ffap})
    bh = SC.buy_hold(b).dropna(); bh.index = pd.to_datetime(bh.index)
    px = C.load_close("yahoo/QQQ.parquet"); r = px.pct_change().fillna(0.0)
    W = S.bh_weights(px, r, {}, pd.Series(1.0, index=px.index))
    leg1 = SU.leg_ret(W[("RV21[0.5,1.5]", "SMA200|e=0.5")], bh).reindex(days).fillna(0.0)
    sd14 = daily_sigma(b.day_close, b.prev_close_adj, 14).reindex(days)
    e_vt = np.minimum(MAX_LEV, TGT / sd14)
    return dict(L2=L2, S2=S2, GL=GL, GS=GS, keep=keep, ov=ov, X=X, leg1=leg1, bh=bh.reindex(days).fillna(0.0), sd14=sd14, e_vt=e_vt, base_live=Lb0 + Sb0)


# ============================================================ 2. 替换型 PPP(E1、E2)
def xmap(a, kind, c):
    """线性指标 a → 仓位 x 及 dx/da。kind='sig':x = c·2σ(a);kind='exp':x = exp(a)。截在 MAX_LEV(截断处梯度为 0)。"""
    if kind == "sig":
        x = c * 2 * expit(a); dx = x * (1 - x / (2 * c))
    else:
        a = np.clip(a, -20, 5); x = np.exp(a); dx = x.copy()
    cap = x >= MAX_LEV
    return np.where(cap, MAX_LEV, x), np.where(cap, 0.0, dx)


def fit_replace(Z, U_l, U_s, Eb, B, lam, kind, c, pen_mask):
    """x_leg = xmap(Zθ_leg)。U_* 为单位仓位损益(bp)。目标 = 组合效用 mean − γ/2·var − λ‖θ‖²(仅惩罚 pen_mask 位置)。"""
    n, p1 = Z.shape; gam = B.mean() / B.var()
    def f(th):
        tl, ts = th[:p1], th[p1:]; xl, dl = xmap(Z @ tl, kind, c); xs, ds = xmap(Z @ ts, kind, c)
        R = xl * U_l + xs * U_s + Eb; Rc = R - R.mean()
        dRl = (U_l * dl)[:, None] * Z; dRs = (U_s * ds)[:, None] * Z
        pen = np.r_[pen_mask, pen_mask]
        g = np.r_[dRl.mean(0) - gam * (Rc[:, None] * dRl).mean(0), dRs.mean(0) - gam * (Rc[:, None] * dRs).mean(0)] - 2 * lam * th * pen
        return -(R.mean() - 0.5 * gam * R.var() - lam * np.sum(pen * th ** 2)), -g
    return f


def init_exp(Z, e_vt):
    """E2 的初值:log e_vt 对 [1, z.., log σ] 的最小二乘(只用截距与 σ 列)→ 起点即原规则。"""
    th = np.zeros(Z.shape[1]); A = Z[:, [0, -1]]; coef = np.linalg.lstsq(A, np.log(e_vt), rcond=None)[0]; th[0], th[-1] = coef; return th


def wf_replace(X, feats, U_l, U_s, Eb, Bfull, e_vt, sel, days, kind):
    yrs = days.year; out = pd.DataFrame(np.nan, index=days, columns=["xl", "xs"]); info = []
    def design(idx, mu, sd):
        Zf = ((X.loc[idx, feats] - mu) / sd).to_numpy(); return np.column_stack([np.ones(int(idx.sum())), Zf])
    def solve(idx, lam):
        mu, sd = X.loc[idx, feats].mean(), X.loc[idx, feats].std().replace(0, 1)
        Z = design(idx, mu, sd); c = float(e_vt[idx].mean())
        pen = np.ones(Z.shape[1])
        if kind == "exp": pen[0] = 0.0; pen[-1] = 0.0                    # 截距与 log σ 不惩罚
        f = fit_replace(Z, U_l[idx], U_s[idx], Eb[idx], Bfull[idx], lam, kind, c, pen)
        th0 = np.r_[init_exp(Z, e_vt[idx]), init_exp(Z, e_vt[idx])] if kind == "exp" else np.zeros(2 * Z.shape[1])
        r = minimize(f, th0, jac=True, method="L-BFGS-B", options={"maxiter": 800})
        return r.x.reshape(2, -1), mu, sd, c
    for yr in U.OOS_YEARS:
        tr = sel & (yrs < yr); te = sel & (yrs == yr); cv = {}
        for lam in U.LAMS:
            sc = []
            for hy in sorted(set(yrs[tr])):
                a_, h_ = tr & (yrs != hy), tr & (yrs == hy)
                if a_.sum() < 150 or h_.sum() < 50: continue
                th, mu, sd, c = solve(a_, lam); Zh = design(h_, mu, sd)
                g = Bfull[a_].mean() / Bfull[a_].var()                   # 与 P.wf 相同:γ 取训练段基线的一阶条件值
                xl = xmap(Zh @ th[0], kind, c)[0]; xs = xmap(Zh @ th[1], kind, c)[0]
                R = xl * U_l[h_] + xs * U_s[h_] + Eb[h_]
                sc.append(U.utility(R, g) - U.utility(Bfull[h_], g))
            cv[lam] = np.mean(sc) if sc else -np.inf
        lam = max(cv, key=cv.get)
        th, mu, sd, c = solve(tr, lam); Zt = design(te, mu, sd)
        out.loc[te, "xl"], out.loc[te, "xs"] = xmap(Zt @ th[0], kind, c)[0], xmap(Zt @ th[1], kind, c)[0]
        sdl = float(sd[feats[-1]])
        info.append(dict(年=yr, λ=lam, c=c, β_long=th[0, -1] / sdl, β_short=th[1, -1] / sdl))
    return out, pd.DataFrame(info)


# ============================================================ 3. 评估
def sr(x): x = pd.Series(x).dropna(); return float(x.mean() / x.std() * np.sqrt(252)) if x.std() > 0 else np.nan


def panel(x):
    x = pd.Series(x).dropna(); eq = (1 + x).cumprod(); dd = float((eq / eq.cummax() - 1).min()); yrs = len(x) / 252
    cagr = float(eq.iloc[-1] ** (1 / yrs) - 1); dn = np.sqrt((np.minimum(x, 0) ** 2).mean()); prof = x[x > 0].sum(); loss = -x[x < 0].sum()
    return {"均值bp": 1e4 * x.mean(), "中位bp": 1e4 * x.median(), "Sharpe": sr(x), "Sortino": x.mean() / dn * np.sqrt(252), "CAGR": cagr, "MDD": dd,
            "Calmar": cagr / abs(dd), "偏度": skew(x), "超额峰度": kurtosis(x), "CVaR5%bp": 1e4 * x[x <= x.quantile(0.05)].mean(),
            "盈亏比": prof / loss if loss > 0 else np.nan, "前5%日占利润": x.nlargest(max(int(.05 * len(x)), 1)).sum() / prof, "最差日bp": 1e4 * x.min()}


def main():
    t0 = time.time(); pd.set_option("display.width", 260, "display.max_rows", 200, "display.float_format", lambda v: f"{v:9.3f}")
    b = load_bars("QQQ"); days = b.days; yrs = days.year
    I = build_inputs(b); print(f"[输入] 完成 {time.time()-t0:.0f}s")
    L2, S2, GL, GS, keep, ov, X, leg1, e_vt, sd14 = (I[k] for k in ("L2", "S2", "GL", "GS", "keep", "ov", "X", "leg1", "e_vt", "sd14"))
    dn = DP.targets(DP.Arr(b), 30).dn; full = pd.Series(b.last_idx.to_numpy() >= 389, index=days); td = pd.Series(days.dayofweek != 0, index=days)
    sel = (td & full & X.notna().all(1) & dn.notna() & (days >= "2019-01-01")).to_numpy(); om = pd.Series(sel & (yrs >= 2021), index=days)
    X = X.copy(); X["lsd14"] = np.log(sd14)
    Lb, Sb, Eb = (L2 * 1e4).to_numpy(), (S2 * 1e4).to_numpy(), ((leg1 + ov) * 1e4).to_numpy()
    ev = e_vt.to_numpy(); Ul, Us = Lb / ev, Sb / ev; Bfull = Lb + Sb + Eb

    def v3_from(kl, ks):
        kl = kl.where(om, 1.0).fillna(1.0); ks = ks.where(om, 1.0).fillna(1.0)
        return kl * L2 + ks * S2 + ov - (kl * GL + ks * GS) * keep

    # ---- B 复现
    kB, coB, lamB = P.wf(X, pd.Series(Lb, index=days), pd.Series(Sb, index=days), pd.Series(Eb, index=days), sel, days, P.CLEAN)
    sl = {"B 基线(v3 ①)": v3_from(kB.kl, kB.ks), "SOTA 无仓位模型": v3_from(pd.Series(1.0, index=days), pd.Series(1.0, index=days))}
    port_B = leg1 + sl["B 基线(v3 ①)"]
    for nm, (a, z) in (("dev 19-22", ("2019", "2022")), ("test 23-26", ("2023", "2026")), ("全 19-26", ("2019", "2026"))):
        print(f"[复现] ① RV21+SMA200+v3Sleeve {nm} Sharpe {TS.metrics(port_B.loc[a:z])['Sharpe（绝对）']:.3f}"
              f"(tearsheet_v3:dev 1.587 / test 2.461 / 全 2.022);λ* {lamB}")
    # ---- E3:保留波动目标,σ14 作为额外特征
    kE3, coE3, lamE3 = P.wf(X, pd.Series(Lb, index=days), pd.Series(Sb, index=days), pd.Series(Eb, index=days), sel, days, P.CLEAN + ["lsd14"])
    sl["E3 保留σ14目标+σ14作特征"] = v3_from(kE3.kl, kE3.ks)
    print(f"[E3] λ* {lamE3};lsd14 系数(六折均值) 多 {coE3[coE3.腿=='long'].lsd14.mean():+.3f} 空 {coE3[coE3.腿=='short'].lsd14.mean():+.3f}", flush=True)
    # ---- E1、E2:替换
    feats = P.CLEAN + ["lsd14"]
    for tag, kind in (("E1 替换:c·2σ(θᵀ[z,logσ14])", "sig"), ("E2 替换:exp(a+θᵀz+β·logσ14)", "exp")):
        xo, info = wf_replace(X, feats, Ul, Us, Eb, Bfull, ev, sel, days, kind)
        kl, ks = xo["xl"] / e_vt, xo["xs"] / e_vt
        sl[tag] = v3_from(kl, ks)
        print(f"[{tag}] 每折:\n{info.round(3).to_string(index=False)}", flush=True)
        I[tag] = (xo, info)
    # ---- 评估
    port = {k: leg1 + v for k, v in sl.items()}; o = om.to_numpy(); ref = port["B 基线(v3 ①)"]
    rows = []; pans = {}
    for k, x in port.items():
        r_ = dict(方案=k)
        for pn, m in (("dev21-22", o & (yrs <= 2022)), ("test23-26", o & (yrs >= 2023)), ("全OOS21-26", o)):
            r_[f"组合SR {pn}"] = sr(x[m]); r_[f"ΔSR vs B {pn}"] = sr(x[m]) - sr(ref[m])
        d, dist = T.boot_dsr(x[o], ref[o], nb=2000, L=10, seed=0); r_["ΔSR 95%lo"], r_["ΔSR 95%hi"] = np.percentile(dist, [2.5, 97.5])
        r_["Sleeve SR 全OOS"] = sr(sl[k][o])
        r_.update({str(y): sr(x[o & (yrs == y)]) - sr(ref[o & (yrs == y)]) for y in range(2021, 2027)})
        rows.append(r_); pans[k] = panel(x[o]); pans["Sleeve | " + k] = panel(sl[k][o])
    R = pd.DataFrame(rows); print("\n=== 组合层(QQQ ① 腿 + v3 Sleeve);OOS 2021–26;ΔSR 相对 B ==="); print(R.to_string(index=False))
    Pn = pd.DataFrame(pans).T; print("\n=== 分布面板(OOS 2021–26)==="); print(Pn.to_string())
    # ---- 机理诊断
    hi = o & (sd14.to_numpy() >= np.nanquantile(sd14.to_numpy()[o], 0.95))
    for nm, (a_, b_) in (("单位仓位口径", (Ul, Us)), ("波动目标口径", (Lb, Sb))):
        s2 = (a_ + b_) ** 2; print(f"[诊断] {nm}:σ14 最高 5% 的日子占 Sleeve 平方损益 {s2[hi].sum() / s2[o].sum():.1%}")
    ek = {k: float((sl[k][o] != 0).mean()) for k in sl}
    for k in ("E1 替换:c·2σ(θᵀ[z,logσ14])", "E2 替换:exp(a+θᵀz+β·logσ14)"):
        xo = I[k][0]; cr = np.corrcoef(np.log(xo["xl"][o]), np.log(e_vt[o]))[0, 1]
        print(f"[诊断] {k}:OOS 多头仓位与原规则 e_vt 的 log 相关 {cr:.3f};平均仓位 多 {xo['xl'][o].mean():.2f} 空 {xo['xs'][o].mean():.2f} vs e_vt {e_vt[o].mean():.2f}")
    R.to_csv(TAB / "sizing_sigma_feed.csv", index=False); Pn.to_csv(TAB / "sizing_sigma_feed_panel.csv")
    print(f"总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
