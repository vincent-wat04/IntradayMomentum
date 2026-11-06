"""Sleeve 逐笔入场过滤 —— 严格按 docs/PROPOSAL_trade_filter.md 执行。"""
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
import lowvol_avoidance_study as LV
import sleeve_anatomy as SA
import trend_rv_sleeve_study as T
import round3_common as C
from event_calendar import event_frame
from etft.intraday.bars import load_bars
from etft.intraday.noise import sigma_profile
from etft.intraday.strategy import rel_volume
from sens_rv_weights import pct_w

OUT, TAB = LV.OUT, LV.TAB
NPH = 30
FEATS_F1 = ["bo_strength", "bo_cross", "vwap_dev", "z_open", "z5", "z15", "z30", "netvol15", "clv30", "relvol", "rv_ratio15", "range15", "pos_dir", "dist_far"]
STATE = ["side", "tod", "gap_dir", "log_sd14", "log_vix", "gex_z", "yA_pct", "e_fomc", "e_nfp", "e_opex", "e_month_start", "e_month_end", "e_half"]
FEATS_F2 = FEATS_F1 + STATE
QS = (0.10, 0.20, 0.30); GAMMAS = (0.5, 1.0)


# ------------------------------------------------------------------ 逐笔配对(含 shares / 佣金)
def pair_with_shares(b, F, off, scale):
    times = list(b.c.columns); tidx = {t: i for i, t in enumerate(times)}; dpos = {d: i for i, d in enumerate(b.days)}
    rows = []
    for d, g in F.groupby("day", sort=False):
        cur = None
        for _, r in g.iterrows():
            kind = str(r.kind); k = tidx.get(r.clock)
            if kind.startswith("ENTRY"):
                cur = dict(day=d, phase=off, side=1 if r.shares > 0 else -1, k_dec=k, entry_px=float(r.price), sh=abs(float(r.shares)), c_in=float(r.commission))
            elif cur is not None:
                gross = cur["side"] * cur["sh"] * (float(r.price) - cur["entry_px"])
                comm = (cur["c_in"] + float(r.commission)) * float(scale.loc[d])
                rows.append({**cur, "exit_px": float(r.price), "exit_kind": kind, "pnl": gross - comm,
                             "y": cur["side"] * (float(r.price) / cur["entry_px"] - 1.0) * 1e4})
                cur = None
    return pd.DataFrame(rows)


def build_trades(b):
    import yfinance as yf
    days = b.days
    pt = yf.download("TQQQ", start="2018-10-01", end="2026-10-12", auto_adjust=False, progress=False, threads=False)["Close"]
    pt = pt.iloc[:, 0] if hasattr(pt, "columns") else pt
    pt.index = pd.to_datetime(pt.index).tz_localize(None)
    sp = yf.Ticker("TQQQ").splits; sp.index = pd.to_datetime(sp.index).tz_localize(None)
    fsp = pd.Series(1.0, index=pt.index)
    for d, r_ in sp.items(): fsp[pt.index < d] *= r_
    scale = ((b.day_close / (pt * fsp).reindex(days).ffill()) / 3.0).fillna(1.0)
    res = SA.run_phases(b, range(NPH))
    Ts = []; chk = []
    for off, rr, F in res:
        F = F.copy(); F["day"] = pd.to_datetime(F.day)
        t = pair_with_shares(b, F, off, scale); Ts.append(t)
        chk.append((rr.reindex(days).fillna(0.0) - t.groupby("day").pnl.sum().reindex(days).fillna(0.0)).abs().max())
    TR = pd.concat(Ts, ignore_index=True)
    ens = pd.concat([r.reindex(days).fillna(0.0) for _, r, _ in res], axis=1).mean(axis=1)
    return TR, ens, max(chk)


# ------------------------------------------------------------------ 入场时刻特征
def entry_features(b, TR, Xd, E, yA_pct):
    days = b.days; dpos = {d: i for i, d in enumerate(days)}
    c, o, hi, lo, v = (b.c.to_numpy(), b.o.to_numpy(), b.h.to_numpy(), b.l.to_numpy(), b.v.to_numpy())
    O = b.day_open.to_numpy(); sd = np.exp(Xd.log_sd14.to_numpy())
    sig = sigma_profile(b, 14).to_numpy(); vw = b.vwap().to_numpy()
    hi_ref = np.maximum(O, b.prev_close_adj.to_numpy()); lo_ref = np.minimum(O, b.prev_close_adj.to_numpy())
    rv = rel_volume(b, 14, "cum").to_numpy()
    r = np.full_like(c, np.nan); r[:, 1:] = np.log(c[:, 1:] / c[:, :-1]); r[:, 0] = np.log(c[:, 0] / O)
    sgn_bar = np.sign(c - o); clv = np.where(hi > lo, ((c - lo) - (hi - c)) / (hi - lo), 0.0)
    hh = np.maximum.accumulate(np.where(np.isfinite(hi), hi, -np.inf), axis=1); ll = np.minimum.accumulate(np.where(np.isfinite(lo), lo, np.inf), axis=1)
    ii = TR.day.map(dpos).to_numpy(); k = TR.k_dec.to_numpy(); s = TR.side.to_numpy().astype(float)
    n = len(TR); ar = np.arange(n)
    ck = c[ii, k]
    ub = hi_ref[ii] * (1 + sig[ii, k]); lb = lo_ref[ii] * (1 - sig[ii, k])
    bo = np.where(s > 0, (ck / ub - 1), (1 - ck / lb)) / np.where(sig[ii, k] > 0, sig[ii, k], np.nan)       # >0 = 已在带外,单位 = 带宽
    # 当日截至 k 已发生的带外次数(同侧)
    outside = np.zeros(n)
    for j in range(n):
        kk = k[j]; i = ii[j]
        cc = c[i, :kk + 1]
        if s[j] > 0: outside[j] = np.sum(cc > hi_ref[i] * (1 + sig[i, :kk + 1]))
        else: outside[j] = np.sum(cc < lo_ref[i] * (1 - sig[i, :kk + 1]))
    f = pd.DataFrame(index=TR.index)
    f["bo_strength"] = bo; f["bo_cross"] = outside
    f["vwap_dev"] = s * np.log(ck / vw[ii, k]) / (sd[ii] * np.sqrt(30 / 390))
    f["z_open"] = s * np.log(ck / O[ii]) / (sd[ii] * np.sqrt((k + 1) / 390))
    for w in (5, 15, 30):
        kb = np.maximum(k - w, 0)
        f[f"z{w}"] = s * np.log(ck / c[ii, kb]) / (sd[ii] * np.sqrt(np.maximum(k - kb, 1) / 390))
    kk15 = np.maximum(k - 14, 0); kk30 = np.maximum(k - 29, 0)
    f["netvol15"] = [s[j] * np.sum(sgn_bar[ii[j], kk15[j]:k[j] + 1] * v[ii[j], kk15[j]:k[j] + 1]) / max(np.sum(v[ii[j], kk15[j]:k[j] + 1]), 1) for j in range(n)]
    f["clv30"] = [s[j] * np.sum(clv[ii[j], kk30[j]:k[j] + 1] * v[ii[j], kk30[j]:k[j] + 1]) / max(np.sum(v[ii[j], kk30[j]:k[j] + 1]), 1) for j in range(n)]
    f["relvol"] = np.log(rv[ii, k])
    rv15 = np.array([np.sum(r[ii[j], kk15[j]:k[j] + 1] ** 2) for j in range(n)]); rvd = np.array([np.sum(r[ii[j], :k[j] + 1] ** 2) for j in range(n)])
    f["rv_ratio15"] = np.log(np.where(rv15 > 0, rv15, np.nan) / np.where(rvd > 0, rvd * 15 / (k + 1), np.nan))
    f["range15"] = np.array([(hi[ii[j], kk15[j]:k[j] + 1].max() - lo[ii[j], kk15[j]:k[j] + 1].min()) / O[ii[j]] for j in range(n)]) / (sd[ii] * np.sqrt(15 / 390))
    pr = (ck - ll[ii, k]) / np.where(hh[ii, k] > ll[ii, k], hh[ii, k] - ll[ii, k], np.nan)
    f["pos_dir"] = np.where(s > 0, pr, 1 - pr)
    f["dist_far"] = np.where(s > 0, np.log(hh[ii, k] / ck), np.log(ck / ll[ii, k])) / sd[ii]
    f["side"] = s; f["tod"] = k / 390.0
    f["gap_dir"] = s * np.log(O[ii] / b.prev_close_adj.to_numpy()[ii]) / sd[ii]
    f["log_sd14"] = Xd.log_sd14.to_numpy()[ii]; f["log_vix"] = Xd.log_vix.to_numpy()[ii]; f["gex_z"] = Xd.gex_z.to_numpy()[ii]
    f["yA_pct"] = yA_pct.reindex(TR.day).fillna(0.5).to_numpy()            # 2021 年前基线预测不存在 → 中性 0.5(无信息、无泄漏)
    for e in ("e_fomc", "e_nfp", "e_opex", "e_month_start", "e_month_end", "e_half"):
        f[e] = E[e].to_numpy()[ii]
    return f.replace([np.inf, -np.inf], np.nan)


def poison_check(b, TR, Xd, E, yA_pct, nsamp=12, seed=0):
    f0 = entry_features(b, TR, Xd, E, yA_pct)
    rg = np.random.default_rng(seed); worst = 0.0
    days = b.days; dpos = {d: i for i, d in enumerate(days)}
    for j in rg.choice(len(TR), nsamp, replace=False):
        d = TR.day.iloc[j]; i = dpos[d]; kd = int(TR.k_dec.iloc[j])
        def P(df):
            df = df.copy(); df.iloc[i, kd + 1:] = df.iloc[i, kd + 1:] * 7.0; return df
        bp = dataclasses.replace(b, o=P(b.o), h=P(b.h), l=P(b.l), c=P(b.c), v=P(b.v))
        sub = TR[(TR.day == d) & (TR.k_dec == kd)]
        fp = entry_features(bp, sub, Xd, E, yA_pct)
        a = f0.loc[sub.index, FEATS_F2].to_numpy(float); c_ = fp[FEATS_F2].to_numpy(float)
        worst = max(worst, float(np.nanmax(np.abs(a - c_))))
    return worst


# ------------------------------------------------------------------ 模型与评估
def fit_predict(Xtr, ytr, Xte, model):
    from sklearn.linear_model import Ridge
    mu, sd = Xtr.mean(), Xtr.std().replace(0, 1.0)
    A, B_ = ((Xtr - mu) / sd).to_numpy(), ((Xte - mu) / sd).to_numpy()
    yt = np.clip(ytr, -150, 150)
    if model == "ridge":
        return Ridge(alpha=100.0).fit(A, yt).predict(B_)
    import lightgbm as lgb
    return lgb.LGBMRegressor(max_depth=3, n_estimators=200, learning_rate=0.03, min_child_samples=500, subsample=0.7, subsample_freq=1,
                             colsample_bytree=0.8, random_state=0, verbose=-1, n_jobs=1).fit(A, yt).predict(B_)


def causal_flags(pred_sorted_days, pred, kind, par):
    """按时间顺序(日期,相位)逐笔因果:q → 因果分位 ≤ q 剔除;cont → s=clip(exp(γ z),0.5,1.5),z=相对此前预测的标准化。返回每笔的权重 s。"""
    n = len(pred); s = np.ones(n); hist = []
    for i in range(n):
        x = pred[i]
        if np.isfinite(x) and len(hist) >= 500:
            h_ = np.asarray(hist)
            if kind == "skip":
                s[i] = 0.0 if np.mean(h_ < x) <= par else 1.0
            else:
                z = (x - h_.mean()) / (h_.std() if h_.std() > 0 else 1.0)
                s[i] = float(np.clip(np.exp(par * z), 0.5, 1.5))
        if np.isfinite(x): hist.append(x)
    return s


def main():
    b = load_bars("QQQ"); days = b.days
    print("构建 30 相位成交与逐笔配对…")
    TR, ens, maxdiff = build_trades(b)
    print(f"逐笔 {len(TR)} 笔;逐相位日收益 − 逐笔净盈亏 的最大绝对误差 = {maxdiff:.2e}(应≈0)")
    assert maxdiff < 1e-9
    Xd, Yd, A, Bc = LV.build(b); E = event_frame(days, b.last_idx.to_numpy())
    S1 = pd.read_pickle(OUT / "lowvol_study.pkl"); yA = S1["pred"][("A·ridge", "rv")].dropna()
    vals = yA.to_numpy(); ypct = pd.Series([(np.mean(vals[:i] < vals[i]) if i >= 60 else np.nan) for i in range(len(vals))], index=yA.index)
    TR = TR.sort_values(["day", "phase", "k_dec"]).reset_index(drop=True)
    F = entry_features(b, TR, Xd, E, ypct)
    w = poison_check(b, TR, Xd, E, ypct)
    print(f"[污染自检] 12 笔:k_dec 之后价格/量×7,入场特征最大差 {w:.2e}(须为 0)"); assert w == 0.0
    yrs = TR.day.dt.year.to_numpy(); y = TR.y.to_numpy()
    # 阳性对照:加入入场后 5 分钟收益作特征
    from scipy.stats import spearmanr
    dpos = {d: i for i, d in enumerate(days)}
    ii = TR.day.map(dpos).to_numpy(); kd = TR.k_dec.to_numpy(); c = b.c.to_numpy()
    fut5 = TR.side.to_numpy() * np.log(c[ii, np.minimum(kd + 6, 389)] / c[ii, kd + 1]) * 1e4
    Fp = F.copy(); Fp["fut5"] = fut5
    okp = Fp[FEATS_F2 + ["fut5"]].notna().all(1).to_numpy()
    trm, tem = okp & (yrs < 2023), okp & (yrs >= 2023)
    pf = fit_predict(Fp.loc[trm, FEATS_F2 + ["fut5"]], y[trm], Fp.loc[tem, FEATS_F2 + ["fut5"]], "ridge")
    print(f"[阳性对照] 加入入场后 5 分钟收益:IC={spearmanr(pf, y[tem])[0]:.3f}(诚实模型 IC 见下)")

    sets = {"F1·ridge": (FEATS_F1, "ridge"), "F2·ridge": (FEATS_F2, "ridge"), "F1·lgbm": (FEATS_F1, "lgbm"), "F2·lgbm": (FEATS_F2, "lgbm")}
    preds = {}
    for nm, (cols, mdl) in sets.items():
        pr = np.full(len(TR), np.nan)
        for yr in range(2021, 2027):
            tr = (yrs < yr) & F[cols].notna().all(1).to_numpy(); te = (yrs == yr) & F[cols].notna().all(1).to_numpy()
            if tr.sum() < 3000 or te.sum() == 0: continue
            pr[te] = fit_predict(F.loc[tr, cols], y[tr], F.loc[te, cols], mdl)
        preds[nm] = pr
    oos = np.isfinite(preds["F1·ridge"]) & np.isfinite(preds["F2·ridge"]) & (yrs >= 2021)
    print("\n" + "=" * 110 + f"\n1) 逐笔预测力(OOS 2021–2026,{oos.sum()} 笔;IC=Spearman(预测 y, 实际 y);区间=按交易日聚类自助)\n" + "=" * 110)
    dayid = TR.day.dt.strftime("%Y-%m-%d").to_numpy()
    ud, inv = np.unique(dayid[oos], return_inverse=True)
    rows = []
    for nm, pr in preds.items():
        yy, pp = y[oos], pr[oos]; ic = spearmanr(pp, yy)[0]
        rg = np.random.default_rng(0); order = np.argsort(inv, kind="stable"); starts = np.searchsorted(inv[order], np.arange(len(ud))); ends = np.append(starts[1:], len(order))
        boots = []
        for _ in range(800):
            pick = rg.integers(0, len(ud), len(ud)); idx = np.concatenate([order[starts[p]:ends[p]] for p in pick]); boots.append(spearmanr(pp[idx], yy[idx])[0])
        lo, hi_ = np.percentile(boots, [2.5, 97.5])
        # 四分位:最差/最好 25% 的平均实际 y
        q = np.quantile(pp, [0.25, 0.75]); worst, best = yy[pp <= q[0]].mean(), yy[pp >= q[1]].mean()
        rows.append(dict(模型=nm, IC=ic, lo=lo, hi=hi_, 预测最差25笔均y_bp=worst, 预测最好25笔均y_bp=best, 全部均y_bp=yy.mean(), IC_dev=spearmanr(pp[yrs[oos] <= 2022], yy[yrs[oos] <= 2022])[0], IC_test=spearmanr(pp[yrs[oos] >= 2023], yy[yrs[oos] >= 2023])[0]))
    R1 = pd.DataFrame(rows); pd.set_option("display.width", 220); print(R1.round(3).to_string(index=False)); R1.to_csv(TAB / "trade_filter_ic.csv", index=False)

    # ---- 单变量:每个特征与 y 的 Spearman(OOS 窗口,描述性)
    uni = pd.DataFrame({c: [spearmanr(F.loc[oos, c], y[oos], nan_policy="omit")[0]] for c in FEATS_F2}).T.rename(columns={0: "rho_y"}).sort_values("rho_y", key=abs, ascending=False)
    print("\n单变量 Spearman(特征, 单笔 y)(OOS 描述性):"); print(uni.round(3).head(10).to_string())

    # ---- 2) 变现
    ret = ens.reindex(days).fillna(0.0)
    td = pd.Series(days.dayofweek != 0, index=days)
    bh = pd.read_pickle(OUT / "strategy_comparison_bh.pkl").dropna(); bh.index = pd.to_datetime(bh.index)
    px_all = C.load_close("yahoo/QQQ.parquet"); p_all = pct_w(px_all.pct_change().rolling(21).std(), 504)
    port = lambda sl: T.build(bh, sl.reindex(bh.index).fillna(0.0), (1.0, 1.0), (("lin", .5, 1.5), None, 1.0), 1.0, px_all, p_all)[0]
    base_p = port(ret)
    dpos_arr = TR.day.map(dpos).to_numpy(); pnl = TR.pnl.to_numpy()
    def curve(s):
        x = np.zeros(len(days)); np.add.at(x, dpos_arr, s * pnl); return pd.Series(x / NPH, index=days)
    assert np.allclose(curve(np.ones(len(TR))).to_numpy(), ret.to_numpy(), atol=1e-9)
    cfgs = {}
    for nm in sets:
        pr = np.where(oos, preds[nm], np.nan)
        for q in QS: cfgs[f"{nm} skip{int(q*100)}%"] = causal_flags(None, pr, "skip", q)
        for g in GAMMAS: cfgs[f"{nm} cont{g:g}"] = causal_flags(None, pr, "cont", g)
    oos_mask = oos & (TR.day.dt.dayofweek.to_numpy() != 0)
    rg = np.random.default_rng(2); nperm = 1500
    oos_idx = np.where(oos)[0]
    base_curve = ret
    rows, nulls = [], {}
    def sr_oos(x): return LV.sr(x.loc["2021":"2026"][td.loc["2021":"2026"]])
    base_sr = sr_oos(ret)
    for nm, s in cfgs.items():
        x = curve(s); xp = port(x); row = dict(配置=nm, 剔除比例=float((s[oos] == 0).mean()), 平均s=float(s[oos].mean()))
        kept, drop = (s > 0) & oos, (s == 0) & oos
        row["保留笔均y_bp"] = float(y[kept].mean()) if kept.any() else np.nan; row["剔除笔均y_bp"] = float(y[drop].mean()) if drop.any() else np.nan
        for k in ("dev-OOS", "test", "全OOS"):
            row[f"Sleeve ΔSR {k}"] = LV.sr(LV.period(x[td], k)) - LV.sr(LV.period(ret[td], k)); row[f"组合 ΔSR {k}"] = LV.sr(LV.period(xp, k)) - LV.sr(LV.period(base_p, k))
        row["_obs"] = sr_oos(x) - base_sr; rows.append(row)
        # 置换:随机把同样多(按 s 分布置换)的笔赋予 s
        nul = np.empty(nperm)
        for p_ in range(nperm):
            ss = s.copy(); ss[oos_idx] = s[rg.permutation(oos_idx)]; nul[p_] = sr_oos(curve(ss)) - base_sr
        nulls[nm] = nul
    fam = np.max(np.vstack(list(nulls.values())), axis=0)
    R2 = pd.DataFrame(rows); R2["置换p(单)"] = [float(np.mean(nulls[c] >= o)) for c, o in zip(R2.配置, R2._obs)]
    R2["置换p(族内max,20配置)"] = [float(np.mean(fam >= o)) for o in R2._obs]; R2 = R2.drop(columns="_obs")
    print("\n" + "=" * 170 + "\n2) 逐笔过滤的变现(OOS;ΔSR 相对不过滤的 Sleeve;置换=随机重新分配 s)\n" + "=" * 170)
    print(R2.round(3).to_string(index=False)); R2.to_csv(TAB / "trade_filter_economic.csv", index=False)
    pd.to_pickle(dict(TR=TR, F=F, preds=preds, cfgs=cfgs), OUT / "trade_filter.pkl")


if __name__ == "__main__":
    main()
