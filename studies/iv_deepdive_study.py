"""与隐含波动对赌的深究:包含回归 + 0DTE 跨式代理 —— docs/PROPOSAL_iv_deepdive.md(预注册)。python3 iv_deepdive_study.py"""
from __future__ import annotations

import os
import sys
import warnings

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lowvol_avoidance_study as L
import paths as _P
from event_calendar import event_frame
from etft.intraday.bars import load_bars

TAB, OUT = L.TAB, L.OUT
F0 = ["log_sd14", "log_vix", "log_vxn", "log_vix9d", "gex_z", "log_rv_prev", "log_rv5", "dow1", "dow2", "dow3", "dow4"]
START = "2022-05-16"


def cboe(name, days, lag=1):
    s = pd.read_parquet(_P.RAW / "cboe" / f"{name}.parquet")["close"]; s.index = pd.to_datetime(s.index).tz_localize(None); s = s[~s.index.duplicated()].sort_index()
    return s.shift(lag).reindex(days, method="ffill")


def ols_fit_pred(Xtr, ytr, Xte):
    mu, sd = Xtr.mean(), Xtr.std().replace(0, 1.0); A = sm.add_constant(((Xtr - mu) / sd).to_numpy()); m = sm.OLS(ytr.to_numpy(), A).fit()
    return m.predict(sm.add_constant(((Xte - mu) / sd).to_numpy(), has_constant="add"))


def mboot(idx, fn, nb=1500, seed=0):
    return L.month_boot(idx, fn, nb=nb, seed=seed)


def trailing(a, b_, n=60, minp=40):
    num = pd.Series(a).rolling(n, min_periods=minp).mean().shift(1); den = pd.Series(b_).rolling(n, min_periods=minp).mean().shift(1)
    return (num / den).to_numpy()


def main():
    pd.set_option("display.width", 230, "display.float_format", lambda x: f"{x:8.3f}")
    b = load_bars("QQQ"); days = b.days; X, Y, A, B = L.build(b); E = event_frame(days, b.last_idx.to_numpy()); EC = list(E.columns); X = pd.concat([X, E], axis=1)
    c = b.c.to_numpy(); O = b.day_open.to_numpy(); r = np.full_like(c, np.nan); r[:, 1:] = np.log(c[:, 1:] / c[:, :-1]); r[:, 0] = np.log(c[:, 0] / O)
    rf = pd.Series(np.nansum(r ** 2, axis=1), index=days); li = b.last_idx.to_numpy(); full = li >= 389
    m_ = pd.Series(np.log(c[np.arange(len(days)), li] / O), index=days); absm = m_.abs(); y = np.log(rf.where(rf > 0 & full))
    Y2 = Y.copy(); Y2["y_rv"] = y; L.OOS_YEARS = range(2021, 2027)
    pred, _, _ = L.walk_forward(X, Y2, {"F0+E": (F0 + EC, "ridge")}); fh = pred[("F0+E", "rv")]
    v1, v9, vx, vn = cboe("VIX1D", days), cboe("VIX9D", days), cboe("VIX", days), cboe("VXN", days); v1b = cboe("VIX1D", days, lag=2)
    ok = (days >= START) & v1.notna().to_numpy() & fh.notna().to_numpy() & y.notna().to_numpy() & full; idx = days[ok]
    print(f"样本 {len(idx)} 天({idx[0].date()} ~ {idx[-1].date()})")
    # ============================================================ A) 包含回归
    D = pd.DataFrame({"y": y, "l1d": np.log(v1), "l9d": np.log(v9), "lxn": np.log(vn), "rv1": X.log_rv_prev, "rv5": X.log_rv5, "sd": X.log_sd14, "gex": X.gex_z, **{c_: X[c_] for c_ in EC}, **{f"dow{d}": X[f"dow{d}"] for d in (1, 2, 3, 4)}, "fh": fh}).loc[idx]
    m1c = ["l1d", "l9d", "lxn", "rv1", "rv5", "sd"]; m2c = m1c + ["gex"] + EC + ["dow1", "dow2", "dow3", "dow4"]
    P1, P2, P0 = pd.Series(np.nan, index=idx), pd.Series(np.nan, index=idx), pd.Series(np.nan, index=idx); base = pd.Series(np.nan, index=idx)
    for yr in (2023, 2024, 2025, 2026):
        tr, te = D.index < f"{yr}-01-01", D.index.year == yr
        if te.sum() == 0 or tr.sum() < 100: continue
        P0[te] = ols_fit_pred(D.loc[tr, ["l1d"]], D.y[tr], D.loc[te, ["l1d"]]); P1[te] = ols_fit_pred(D.loc[tr, m1c], D.y[tr], D.loc[te, m1c]); P2[te] = ols_fit_pred(D.loc[tr, m2c], D.y[tr], D.loc[te, m2c]); base[te] = D.y[tr].mean()
    oo = idx[P2.notna().to_numpy()]; sse0 = pd.Series(((D.y[oo] - base[oo]) ** 2).to_numpy(), index=oo)
    es = {k: pd.Series(((D.y[oo] - p[oo]) ** 2).to_numpy(), index=oo) for k, p in (("仅 VIX1D", P0), ("模型1 全部隐含+持续性", P1), ("模型2 +GEX/事件/星期", P2))}
    print(f"\nA) 包含回归(扩展窗口 OLS;OOS 2023–2026,n={len(oo)})"); rows = []
    for k, e in es.items():
        pt, lo, hi = mboot(oo, lambda ii, e=e: 1 - e.iloc[ii].sum() / sse0.iloc[ii].sum()); rows.append(dict(模型=k, OOS_R2=pt, lo=lo, hi=hi))
    for a_, b_ in (("模型1 全部隐含+持续性", "仅 VIX1D"), ("模型2 +GEX/事件/星期", "模型1 全部隐含+持续性")):
        pt, lo, hi = mboot(oo, lambda ii: (es[b_].iloc[ii].sum() - es[a_].iloc[ii].sum()) / sse0.iloc[ii].sum()); rows.append(dict(模型=f"ΔR²: {a_} − {b_}", OOS_R2=pt, lo=lo, hi=hi))
    RA = pd.DataFrame(rows); print(RA.round(4).to_string(index=False)); RA.to_csv(TAB / "iv_encompassing.csv", index=False)
    Xf = sm.add_constant(D[m1c + ["fh"]]); mm = sm.OLS(D.y, Xf).fit(cov_type="HAC", cov_kwds={"maxlags": 10})
    print(f"在模型1 上加入 f̂(M2 预测)的全样本 HAC:系数 {mm.params['fh']:+.3f},t={mm.tvalues['fh']:+.2f};模型1 各系数 t:{ {k: round(float(v),1) for k,v in mm.tvalues.items() if k in m1c} }")
    # ============================================================ B) 跨式代理
    ix = days.get_indexer(idx); n = len(days)
    ivs = (v1 / 100 / np.sqrt(252) * (vn / vx)); ivs2 = (np.sqrt(v1 * v1b) / 100 / np.sqrt(252) * (vn / vx))
    absm_a = absm.reindex(days).to_numpy(); fh_a = fh.reindex(days).to_numpy()
    E_model = 0.798 * np.sqrt(np.exp(fh_a)); E_naive = 0.798 * np.sqrt(np.exp(X.log_rv_prev.reindex(days).to_numpy()))
    kap_m = trailing(absm_a, E_model); kap_n = trailing(absm_a, E_naive)
    rng = np.random.default_rng(3); rnd = rng.choice([-1.0, 1.0], n)
    res = {}
    for pn, iv in (("P1 原始VIX1D", ivs.to_numpy()), ("P2 降噪(t−1,t−2 几何均)", ivs2.to_numpy())):
        lam = trailing(absm_a, 0.798 * iv); prem = lam * 0.798 * iv
        for sn, Eest in (("模型", E_model * kap_m), ("朴素持续性", E_naive * kap_n)):
            e = Eest / prem
            for h in (0.0, 0.1):
                pos = np.where(e > 1 + h, 1.0, np.where(e < 1 - h, -1.0, 0.0))
                for cst in (0.0, 0.03, 0.05, 0.10):
                    pl = np.where(pos > 0, absm_a / prem - (1 + cst), np.where(pos < 0, (1 - cst) - absm_a / prem, 0.0)); res[(pn, sn, h, cst)] = (pos, pl)
        for cst in (0.05,):
            pos = rnd; pl = np.where(pos > 0, absm_a / prem - (1 + cst), (1 - cst) - absm_a / prem); res[(pn, "随机方向", 0.0, cst)] = (pos, pl)
    def summ(pn, sn, h, cst):
        pos, pl = res[(pn, sn, h, cst)]; ii = np.array([i for i in ix if np.isfinite(pl[i]) and pos[i] != 0]); x = pl[ii]
        if len(x) < 30: return dict(n=len(x))
        mi = days[ii]; pt, lo, hi = mboot(mi, lambda jj: np.nanmean(x[jj]), nb=800)
        return dict(n=len(x), 多头占比=float(np.mean(pos[ii] > 0)), 均损益=pt, lo=lo, hi=hi, 胜率=float(np.mean(x > 0)), Sharpe=float(np.mean(x) / np.std(x) * np.sqrt(252)), 最差=float(x.min()))
    print("\nB) 0DTE 跨式代理(损益单位 = 溢价;c=5%,主结果)"); rows = []
    for pn in ("P1 原始VIX1D", "P2 降噪(t−1,t−2 几何均)"):
        for sn in ("模型", "朴素持续性", "随机方向"):
            for h in ((0.0, 0.1) if sn != "随机方向" else (0.0,)):
                rows.append(dict(溢价=pn, 信号=sn, h=h, **summ(pn, sn, h, 0.05)))
    RB = pd.DataFrame(rows); print(RB.round(3).to_string(index=False)); RB.to_csv(TAB / "iv_straddle.csv", index=False)
    print("\n成本敏感性(h=0;均损益):"); rows = []
    for pn in ("P1 原始VIX1D", "P2 降噪(t−1,t−2 几何均)"):
        for sn in ("模型", "朴素持续性"):
            rows.append({"溢价": pn, "信号": sn, **{f"c={int(c_*100)}%": summ(pn, sn, 0.0, c_).get("均损益", np.nan) for c_ in (0.0, 0.03, 0.05, 0.10)}})
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    # 模型 − 朴素 的逐日差(h=0, c=5%)与分年
    for pn in ("P1 原始VIX1D", "P2 降噪(t−1,t−2 几何均)"):
        pm, plm = res[(pn, "模型", 0.0, 0.05)]; pnv, pln = res[(pn, "朴素持续性", 0.0, 0.05)]
        dd = pd.Series(np.nan_to_num(plm) - np.nan_to_num(pln), index=days).loc[idx]; dd = dd[np.isfinite(plm[ix]) & np.isfinite(pln[ix])]
        pt, lo, hi = mboot(dd.index, lambda jj: dd.iloc[jj].mean(), nb=1000)
        yr = {int(k): round(float(v), 3) for k, v in dd.groupby(dd.index.year).mean().items()}
        print(f"{pn}: 模型 − 朴素 逐日损益差 均值 {pt:+.4f} [{lo:+.4f},{hi:+.4f}];分年 {yr};两信号同向天数占比 {np.mean(np.sign(pm[ix])==np.sign(pnv[ix])):.2f}")
        yrs_m = {int(k): round(float(v), 3) for k, v in pd.Series(np.where(pm[ix] != 0, plm[ix], np.nan), index=idx).groupby(idx.year).mean().items()}
        print(f"   模型(c=5%)分年均损益 {yrs_m}")
    lam1 = trailing(absm_a, 0.798 * ivs.to_numpy()); print(f"\n校准:λ(原始) 中位 {np.nanmedian(lam1[ix]):.2f}(=已实现 |m| 与隐含 0.798σ 的比);mean|m|={np.nanmean(absm_a[ix])*1e4:.1f}bp,0.798σ_iv 均值={np.nanmean(0.798*ivs.to_numpy()[ix])*1e4:.1f}bp")
    # ============================================================ C) 事后分解:模型2 的增量来自哪里?(探索性)
    print("\nC) 事后分解(探索性):在模型1 上逐组加入,OOS ΔR²(月聚类自助 95%)")
    groups = {"GEX(t−1)": ["gex"], "星期": ["dow1", "dow2", "dow3", "dow4"], "FOMC 三天": ["e_fomc", "e_fomc_eve", "e_fomc_post"], "NFP": ["e_nfp"], "OPEX/季度": ["e_opex", "e_quad"], "月初月末": ["e_month_end", "e_month_start"], "节假日前后": ["e_pre_holiday", "e_post_holiday"]}
    rows = []
    for gn, gc in groups.items():
        cols = m1c + gc; Pg = pd.Series(np.nan, index=idx)
        for yr in (2023, 2024, 2025, 2026):
            tr, te = D.index < f"{yr}-01-01", D.index.year == yr
            if te.sum() and tr.sum() >= 100: Pg[te] = ols_fit_pred(D.loc[tr, cols], D.y[tr], D.loc[te, cols])
        eg = pd.Series(((D.y[oo] - Pg[oo]) ** 2).to_numpy(), index=oo); pt, lo, hi = mboot(oo, lambda ii, eg=eg: (es["模型1 全部隐含+持续性"].iloc[ii].sum() - eg.iloc[ii].sum()) / sse0.iloc[ii].sum())
        rows.append(dict(加入=gn, ΔR2=pt, lo=lo, hi=hi))
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print("模型(h=0,c=5%)的跨式损益按事件日分组(均损益 / 天数):")
    for pn in ("P1 原始VIX1D", "P2 降噪(t−1,t−2 几何均)"):
        pm, plm = res[(pn, "模型", 0.0, 0.05)]; sr_ = pd.Series(np.where(pm != 0, plm, np.nan), index=days).loc[idx]
        out = {}
        for ev in ("e_fomc", "e_fomc_eve", "e_fomc_post", "e_nfp", "e_opex", "e_month_end", "e_post_holiday", "e_pre_holiday"):
            mk = (E.loc[idx, ev] == 1).to_numpy(); out[ev] = (round(float(sr_[mk].mean()), 3), int(sr_[mk].notna().sum()))
        none = (E.loc[idx, ["e_fomc", "e_fomc_eve", "e_fomc_post", "e_nfp", "e_opex", "e_month_end", "e_post_holiday", "e_pre_holiday"]].sum(axis=1) == 0).to_numpy(); out["无事件日"] = (round(float(sr_[none].mean()), 3), int(sr_[none].notna().sum()))
        print(f"  {pn}: {out}")
    pd.to_pickle(dict(RA=RA, RB=RB), OUT / "iv_deepdive.pkl")


if __name__ == "__main__":
    main()
