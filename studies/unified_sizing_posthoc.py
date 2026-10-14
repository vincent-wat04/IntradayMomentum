"""事后(看到 U3 结果后追加,见 PROPOSAL_unified_sizing.md 偏离记录):
1) U3p:PPP 的目标改成**组合**效用(Sleeve + QQQ 腿),因为空头腿是组合的对冲(与 QQQ 腿相关 −0.40);
2) U3-clean / U3p-clean:去掉在 2019–2026 上被发现的三个特征(ffa_rank、P3、P4),检验增益是否来自特征污染。
python3 unified_sizing_posthoc.py"""
import os, sys, warnings
warnings.filterwarnings("ignore"); os.environ.setdefault("OMP_NUM_THREADS", "1"); sys.path.insert(0, ".")
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
import unified_sizing_study as U, lowvol_avoidance_study as L, round3_common as C, trend_rv_sleeve_study as T, sota_stack_study as S
from sens_rv_weights import pct_w

CLEAN = [f for f in U.FEATS if f not in ("ffa_rank", "P3", "P4")]


def ppp_fit(Z, Lb, Sb, Eb, lam):
    n, p = Z.shape; Zt = np.column_stack([np.ones(n), Z]); B = Lb + Sb + Eb; gam = B.mean() / B.var()
    def f(th):
        tl, ts = th[:p + 1], th[p + 1:]; kl, ks = 2 * expit(Zt @ tl), 2 * expit(Zt @ ts)
        R = kl * Lb + ks * Sb + Eb; Rc = R - R.mean()
        dRl = (Lb * kl * (1 - kl / 2))[:, None] * Zt; dRs = (Sb * ks * (1 - ks / 2))[:, None] * Zt
        g = np.r_[dRl.mean(0) - gam * (Rc[:, None] * dRl).mean(0), dRs.mean(0) - gam * (Rc[:, None] * dRs).mean(0)] - 2 * lam * th
        return -(R.mean() - 0.5 * gam * R.var() - lam * np.sum(th ** 2)), -g
    r = minimize(f, np.zeros(2 * (p + 1)), jac=True, method="L-BFGS-B", options={"maxiter": 500})
    return r.x.reshape(2, p + 1), gam


def wf(X, Lb, Sb, Eb, sel, days, feats):
    yrs = days.year; out = pd.DataFrame(np.nan, index=days, columns=["kl", "ks"]); co = []; lams = {}
    for yr in U.OOS_YEARS:
        tr = sel & (yrs < yr); te = sel & (yrs == yr)
        cv = {}
        for lam in U.LAMS:
            sc = []
            for hy in sorted(set(yrs[tr])):
                a_, h_ = tr & (yrs != hy), tr & (yrs == hy)
                if a_.sum() < 150 or h_.sum() < 50: continue
                mu, sd = X.loc[a_, feats].mean(), X.loc[a_, feats].std().replace(0, 1)
                th, g = ppp_fit(((X.loc[a_, feats] - mu) / sd).to_numpy(), Lb[a_].to_numpy(), Sb[a_].to_numpy(), Eb[a_].to_numpy(), lam)
                Zh = np.column_stack([np.ones(h_.sum()), ((X.loc[h_, feats] - mu) / sd).to_numpy()])
                R = 2 * expit(Zh @ th[0]) * Lb[h_].to_numpy() + 2 * expit(Zh @ th[1]) * Sb[h_].to_numpy() + Eb[h_].to_numpy()
                B = Lb[h_].to_numpy() + Sb[h_].to_numpy() + Eb[h_].to_numpy()
                sc.append(U.utility(R, g) - U.utility(B, g))
            cv[lam] = np.mean(sc) if sc else -np.inf
        lam = max(cv, key=cv.get); lams[yr] = lam
        mu, sd = X.loc[tr, feats].mean(), X.loc[tr, feats].std().replace(0, 1)
        th, _ = ppp_fit(((X.loc[tr, feats] - mu) / sd).to_numpy(), Lb[tr].to_numpy(), Sb[tr].to_numpy(), Eb[tr].to_numpy(), lam)
        Zt = np.column_stack([np.ones(te.sum()), ((X.loc[te, feats] - mu) / sd).to_numpy()])
        out.loc[te, "kl"], out.loc[te, "ks"] = 2 * expit(Zt @ th[0]), 2 * expit(Zt @ th[1])
        co.append(dict(年=yr, 腿="long", 截距=th[0, 0], **dict(zip(feats, th[0, 1:])))); co.append(dict(年=yr, 腿="short", 截距=th[1, 0], **dict(zip(feats, th[1, 1:]))))
    return out, pd.DataFrame(co), lams


def main():
    pd.set_option("display.width", 260, "display.float_format", lambda x: f"{x:8.3f}")
    D = pd.read_pickle(U.OUT / "unified_sizing.pkl"); X, Lg, Sg = D["X"], D["Lg"], D["Sg"]; days = Lg.index; yrs = days.year
    from etft.intraday.bars import load_bars
    b = load_bars("QQQ"); import disp_pred_study as DP
    dn = DP.targets(DP.Arr(b), 30).dn; full = pd.Series(b.last_idx.to_numpy() >= 389, index=days); td = pd.Series(days.dayofweek != 0, index=days)
    sel = (td & full & X.notna().all(1) & dn.notna() & (days >= "2019-01-01")).to_numpy()
    bh = pd.read_pickle(U.OUT / "strategy_comparison_bh.pkl").dropna(); bh.index = pd.to_datetime(bh.index)
    px = C.load_close("yahoo/QQQ.parquet"); p = pct_w(px.pct_change().rolling(21).std(), 504)
    port = lambda s_: T.build(bh, s_.reindex(bh.index).fillna(0.0), (1.0, 1.0), (("lin", .5, 1.5), None, 1.0), 1.0, px, p)[0]
    leg = port(pd.Series(0.0, index=days)).reindex(days).fillna(0.0)
    Lb, Sb, Eb = Lg * 1e4, Sg * 1e4, leg * 1e4
    base = Lg + Sg; base_p = port(base); oos = sel & (yrs >= 2021); om = pd.Series(oos, index=days)
    cfg = {"U3(原,Sleeve 目标,全特征)": (D["outs"]["U3"], None),
           "U3p(组合目标,全特征)": wf(X, Lb, Sb, Eb, sel, days, U.FEATS),
           "U3-clean(Sleeve 目标,去污染特征)": wf(X, Lb, Sb, pd.Series(0.0, index=days), sel, days, CLEAN),
           "U3p-clean(组合目标,去污染特征)": wf(X, Lb, Sb, Eb, sel, days, CLEAN)}
    idx = np.where(oos)[0]; rg = np.random.default_rng(9); shifts = rg.integers(60, len(idx) - 60, U.NPERM)
    bpsr = L.sr(base_p.reindex(days).iloc[idx]); rows = []; pan = [S.panel(base_p.reindex(days)[oos], "U0 基线(组合)")]
    Lv, Sv, Ev = Lg.to_numpy()[idx], Sg.to_numpy()[idx], leg.to_numpy()[idx]
    for nm, val in cfg.items():
        o = val[0] if isinstance(val, tuple) else val
        kl = o.kl.where(om, 1.0).fillna(1.0); ks = o.ks.where(om, 1.0).fillna(1.0); s_ = kl * Lg + ks * Sg; xp = port(s_).reindex(days)
        obs = L.sr(xp.iloc[idx]) - bpsr
        null = np.array([L.sr(pd.Series(np.roll(kl.to_numpy()[idx], s) * Lv + np.roll(ks.to_numpy()[idx], s) * Sv + Ev)) for s in shifts]) - bpsr
        d, dist = T.boot_dsr(xp[oos], base_p.reindex(days)[oos], nb=1500, L=10, seed=0)
        r_ = dict(方案=nm, k多=float(kl[oos].mean()), k空=float(ks[oos].mean()),
                  Sleeve_ΔSR=L.sr(s_[oos]) - L.sr(base[oos]), Sleeve与QQQ腿相关=float(np.corrcoef(s_[oos], leg[oos])[0, 1]),
                  组合ΔSR_dev=L.sr(xp[oos & (yrs <= 2022)]) - L.sr(base_p.reindex(days)[oos & (yrs <= 2022)]),
                  组合ΔSR_test=L.sr(xp[oos & (yrs >= 2023)]) - L.sr(base_p.reindex(days)[oos & (yrs >= 2023)]),
                  组合ΔSR_全=obs, lo=np.percentile(dist, 2.5), hi=np.percentile(dist, 97.5), p单_圆周平移=float(np.mean(null >= obs)))
        r_.update({str(y): L.sr(xp[oos & (yrs == y)]) - L.sr(base_p.reindex(days)[oos & (yrs == y)]) for y in U.OOS_YEARS})
        rows.append(r_); pan.append(S.panel(xp[oos], nm))
        if isinstance(val, tuple) and val[1] is not None:
            print(f"\n{nm} 系数(六折均值;λ* {val[2]}):"); print(val[1].groupby("腿").mean(numeric_only=True).drop(columns="年").round(3).to_string())
    R = pd.DataFrame(rows); print("\n事后:组合层(L1 RV[0.5,1.5] + Sleeve;OOS 2021–2026;周二至周五)"); print(R.to_string(index=False)); R.to_csv(U.TAB / "unified_posthoc.csv", index=False)
    P = pd.DataFrame(pan); print("\n组合分布面板:"); print(P[["方案", "均值bp", "Sharpe", "Sortino", "CAGR", "MDD", "Calmar", "CVaR5%bp", "最差日bp", "峰度"]].to_string(index=False)); P.to_csv(U.TAB / "unified_posthoc_panel.csv", index=False)


if __name__ == "__main__":
    main()
