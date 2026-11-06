"""A+B+E 波动模型能否变现 —— docs/PROPOSAL_vol_monetization.md(预注册)。python3 vol_monetization_study.py"""
from __future__ import annotations

import dataclasses
import os
import sys
import warnings

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lowvol_avoidance_study as L
import regime_timing_study as R1
import round3_common as C
import trend_rv_sleeve_study as T
from event_calendar import event_frame
from etft.intraday.bars import load_bars
from sens_rv_weights import pct_w, w_linear
import paths as _P

OUT, TAB = L.OUT, L.TAB
F0 = ["log_sd14", "log_vix", "log_vxn", "log_vix9d", "gex_z", "log_rv_prev", "log_rv5", "dow1", "dow2", "dow3", "dow4"]


def r2_boot(oos, Y, pred, key, sse_b, e_ref=None):
    e = pd.Series(((Y.loc[oos, "y_rv"] - pred.loc[oos]) ** 2).to_numpy(), index=oos)
    if e_ref is None:
        return L.month_boot(oos, lambda ii: 1 - e.iloc[ii].sum() / sse_b.iloc[ii].sum())
    return L.month_boot(oos, lambda ii: (e_ref.iloc[ii].sum() - e.iloc[ii].sum()) / sse_b.iloc[ii].sum())


def poison_m2(b, X0):
    """整日污染(第 i 天全部 bar ×7):F0 在第 i 天的特征必须不变(决策在前一收盘);阳性对照:gap_abs 与 b_ret30 必须变。"""
    rg = np.random.default_rng(1); idx = rg.choice(np.arange(300, len(b.days) - 1), 10, replace=False); worst, lead = 0.0, 0.0
    for i in idx:
        d = b.days[i]
        def P(df, i=i):
            df = df.copy(); df.iloc[i] = df.iloc[i] * 7.0; return df
        dc = b.day_close.copy(); dc.iloc[i] = dc.iloc[i] * 7.0
        bp = dataclasses.replace(b, o=P(b.o), h=P(b.h), l=P(b.l), c=P(b.c), v=P(b.v), day_close=dc)
        Xp, _, _, _ = L.build(bp)
        worst = max(worst, float(np.nanmax(np.abs(X0.loc[d, F0].to_numpy(float) - Xp.loc[d, F0].to_numpy(float)))))
        lead = max(lead, float(np.nanmax(np.abs(X0.loc[d, ["gap_abs", "b_ret30"]].to_numpy(float) - Xp.loc[d, ["gap_abs", "b_ret30"]].to_numpy(float)))))
    return worst, lead


def main():
    b = load_bars("QQQ"); days = b.days
    X, Y, A, B = L.build(b); E = event_frame(days, b.last_idx.to_numpy()); EC = list(E.columns)
    X = pd.concat([X, E], axis=1)
    w_, l_ = poison_m2(b, X)
    print(f"[自检 M2] 第 i 天全部 bar ×7:F0 在第 i 天的特征最大差 {w_:.2e}(须为 0);阳性对照(gap_abs/b_ret30)最大差 {l_:.3f}(须>0)"); assert w_ == 0.0 and l_ > 0
    pd.set_option("display.width", 230)
    # ------------------------------------------------------------------ 第一部分:R² 分解
    sets = {"const": ([], "ridge"), "VIX": (["log_vix"], "ridge"), "VIXall": (["log_vix", "log_vix9d", "log_vxn"], "ridge"), "HAR": (["log_rv_prev", "log_rv5", "log_sd14"], "ridge"),
            "B30": (["b_log_rv30"], "ridge"), "VIXall+B30": (["log_vix", "log_vix9d", "log_vxn", "b_log_rv30"], "ridge"), "A": (A, "ridge"), "A+B": (A + B, "ridge"), "A+B+E": (A + B + EC, "ridge")}
    sets_nc = {k: v for k, v in sets.items() if k != "const"}
    pred, base_rv, thr_s = L.walk_forward(X, Y, sets_nc)
    oos = X.index[(X.index.year >= 2021) & X[A + B + EC].notna().all(1) & Y.notna().all(1)]
    ylow = (Y.loc[oos, "disp"] <= thr_s.loc[oos]).astype(int).to_numpy()
    sse_b = pd.Series(((Y.loc[oos, "y_rv"] - base_rv.loc[oos]) ** 2).to_numpy(), index=oos)
    e_ref = pd.Series(((Y.loc[oos, "y_rv"] - pred[("VIXall", "rv")].loc[oos]) ** 2).to_numpy(), index=oos)
    rows = []
    for nm in sets_nc:
        pt, lo, hi = r2_boot(oos, Y, pred[(nm, "rv")], "rv", sse_b)
        pa = pred[(nm, "low")].loc[oos].to_numpy(); at, alo, ahi = L.month_boot(oos, lambda ii: L.auc(ylow[ii], pa[ii]))
        d = r2_boot(oos, Y, pred[(nm, "rv")], "rv", sse_b, e_ref) if nm != "VIXall" else (0, 0, 0)
        rows.append(dict(模型=nm, R2=pt, R2_lo=lo, R2_hi=hi, AUC低波=at, AUC_lo=alo, AUC_hi=ahi, **{"ΔR2 vs VIXall": d[0], "Δlo": d[1], "Δhi": d[2]}))
    D = pd.DataFrame(rows); print(f"\n1) R² 分解(OOS 2021–2026,{len(oos)} 天)"); print(D.round(4).to_string(index=False)); D.to_csv(TAB / "vol_monet_decomp.csv", index=False)
    # ------------------------------------------------------------------ M1:Sleeve 缩放 + 神谕
    ret = pd.read_pickle(L.OUT / "sleeve_anatomy_curve.pkl"); ret.index = pd.to_datetime(ret.index); ret = ret.reindex(days).fillna(0.0)
    td = pd.Series(days.dayofweek != 0, index=days)
    bh = pd.read_pickle(L.OUT / "strategy_comparison_bh.pkl").dropna(); bh.index = pd.to_datetime(bh.index)
    px_all = C.load_close("yahoo/QQQ.parquet"); p_all = pct_w(px_all.pct_change().rolling(21).std(), 504)
    port = lambda sl: T.build(bh, sl.reindex(bh.index).fillna(0.0), (1.0, 1.0), (("lin", .5, 1.5), None, 1.0), 1.0, px_all, p_all)[0]
    base_p = port(ret)
    oracle = Y.y_rv.where(Y.index.isin(oos)); fc = {"A+B+E": pred[("A+B+E", "rv")], "A": pred[("A", "rv")], "神谕": oracle}
    tr_idx = np.where(td.to_numpy() & (days.year >= 2021))[0]; rg = np.random.default_rng(1); perms = [rg.permutation(tr_idx) for _ in range(L.NPERM)]
    base_sr = L.sr(ret.iloc[tr_idx]); rv_ = ret.to_numpy(); rows = []
    for fn, yh in fc.items():
        for kind, par in (("skip", .2), ("skip", .3), ("cont", .5), ("cont", 1.0)):
            s = L.causal_s(yh, kind, par); sv = s.reindex(days).fillna(1.0).to_numpy(); sser = pd.Series(sv, index=days)
            obs = L.sr(sv[tr_idx] * rv_[tr_idx]) - base_sr; null = np.array([L.sr(sv[pi] * rv_[tr_idx]) for pi in perms]) - base_sr
            xs, xp = ret * sser, port(ret * sser); row = dict(预报=fn, 规则=f"{kind}{par:g}", 跳过比例=float((sv[tr_idx] == 0).mean()))
            for k in ("dev-OOS", "test", "全OOS"):
                row[f"Sleeve ΔSR {k}"] = L.sr(L.period(xs[td], k)) - L.sr(L.period(ret[td], k)); row[f"组合 ΔSR {k}"] = L.sr(L.period(xp, k)) - L.sr(L.period(base_p, k))
            row["置换p(单)"] = float(np.mean(null >= obs)); rows.append(row)
    M1 = pd.DataFrame(rows); print("\n2) M1 Sleeve 缩放:模型 vs 神谕上界(OOS 2021–2026)"); print(M1.round(3).to_string(index=False)); M1.to_csv(TAB / "vol_monet_m1.csv", index=False)
    # ------------------------------------------------------------------ M2:L1 权重 = 预测 vs RV21
    c_ = b.c.to_numpy(); O = b.day_open.to_numpy(); r_ = np.full_like(c_, np.nan); r_[:, 1:] = np.log(c_[:, 1:] / c_[:, :-1]); r_[:, 0] = np.log(c_[:, 0] / O)
    rf = pd.Series(np.nansum(r_ ** 2, axis=1), index=days); Y2 = Y.copy(); Y2["y_rv"] = np.log(rf.where(rf > 0))
    pred2, base2, _ = L.walk_forward(X, Y2, {"F0+E": (F0 + EC, "ridge")})
    f2 = pred2[("F0+E", "rv")]; oos2 = f2.dropna().index
    r2m = 1 - ((Y2.loc[oos2, "y_rv"] - f2.loc[oos2]) ** 2).sum() / ((Y2.loc[oos2, "y_rv"] - base2.loc[oos2]) ** 2).sum()
    rv21 = np.log(px_all.pct_change().rolling(21).std()).reindex(days)
    print(f"\n3) M2 全日 log RV 预测(特征 ≤ t−1 收盘):OOS R² {r2m:.3f}(n={len(oos2)});与 RV21 的相关(OOS):{np.corrcoef(f2.loc[oos2], rv21.loc[oos2].fillna(rv21.mean()))[0,1]:.3f}")
    p_f = pct_w(f2.reindex(px_all.index), 504); w_f = w_linear(p_f, 0.5, 1.5).shift(1); w_r = w_linear(p_all, 0.5, 1.5).shift(1)
    r = px_all.pct_change().fillna(0.0); valid = p_f.dropna().index
    start = valid[0]; print(f"预测版权重有效起点 {start.date()}(504 日窗口 min 252;此前不评估)")
    windows = {"full": (str(start.date()), "2026-10-09"), "dev": (str(start.date()), "2023-12-31"), "test": ("2024-01-01", "2026-10-09")}
    E2 = {"预测版L1": w_f.fillna(1.0), "RV21版L1": w_r.fillna(1.0)}; rows = []
    for nm, e in E2.items():
        for wn, (a, z) in windows.items():
            m = (r.index >= a) & (r.index <= z); x = R1.strat(e, r)[m]; ctrl = float(e[m].mean()) * r[m]; d, dist = T.boot_dsr(x, ctrl, nb=1500, L=40, seed=0)
            rows.append(dict(配置=nm, 期间=wn, 平均权重=float(e[m].mean()), Sharpe=R1.sr(x), CAGR=R1.cagr(x), MDD=R1.mdd(x), dSR=d, lo=np.percentile(dist, 2.5), hi=np.percentile(dist, 97.5)))
    for wn, (a, z) in windows.items():
        m = (r.index >= a) & (r.index <= z); d, dist = T.boot_dsr(R1.strat(E2["预测版L1"], r)[m], R1.strat(E2["RV21版L1"], r)[m], nb=1500, L=40, seed=0)
        rows.append(dict(配置="预测版 − RV21版", 期间=wn, dSR=d, lo=np.percentile(dist, 2.5), hi=np.percentile(dist, 97.5)))
    M2 = pd.DataFrame(rows); print(M2.round(3).to_string(index=False)); M2.to_csv(TAB / "vol_monet_m2.csv", index=False)
    pf, _ = R1.circ_placebo(E2, r, *windows["full"], nperm=2000); print("圆周平移置换 p(族内 2 个版本):", {k: round(v, 3) for k, v in pf.items()})
    # ------------------------------------------------------------------ M3:VIX1D 代理
    v1 = pd.read_parquet(_P.RAW / "cboe" / "VIX1D.parquet")["close"]; v1.index = pd.to_datetime(v1.index).tz_localize(None); v1 = v1[~v1.index.duplicated()].sort_index()
    iv = ((v1.shift(1).reindex(days, method="ffill") / 100) ** 2 / 252); y = np.log(rf.where(rf > 0)); fh = f2
    ok = (days >= "2022-05-16") & iv.notna().to_numpy() & fh.notna().to_numpy() & y.notna().to_numpy(); idx = days[ok]
    liv = np.log(iv[idx]); yy = (y[idx] - liv).to_numpy(); xx = (fh[idx] - liv).to_numpy()
    def slope(ii): X_ = np.column_stack([np.ones(len(ii)), xx[ii]]); return np.linalg.lstsq(X_, yy[ii], rcond=None)[0][1]
    bt, blo, bhi = L.month_boot(idx, slope)
    print(f"\n4) M3 与 VIX1D 对赌(2022-05 起,n={len(idx)}):log rv − log iv 对 (预测 − log iv) 的斜率 b = {bt:.3f} [{blo:.3f},{bhi:.3f}];平均 (log rv − log iv) = {yy.mean():+.3f}(VRP:隐含高于已实现 ⇒ 负)")
    c_t = pd.Series(yy, index=idx).expanding(min_periods=60).mean().shift(1); pos = np.sign(pd.Series(xx, index=idx) - c_t).fillna(0.0)
    pnl = pos * ((rf[idx] - iv[idx]) / iv[idx]); pnl = pnl[pos != 0]
    sh = L.sr(pnl); pt, lo, hi = L.month_boot(pnl.index, lambda ii: L.sr(pnl.iloc[ii]))
    print(f"方差互换代理 P&L(无成本):n={len(pnl)},日均 {pnl.mean():+.3f}(名义的比例),胜率 {np.mean(pnl>0):.2f},Sharpe {sh:+.2f} [{lo:+.2f},{hi:+.2f}];多头方差占比 {np.mean(pos[pos!=0]>0):.2f};最差日 {pnl.min():+.2f}")
    pd.to_pickle(dict(M1=M1, M2=M2, D=D), OUT / "vol_monet.pkl")


if __name__ == "__main__":
    main()
