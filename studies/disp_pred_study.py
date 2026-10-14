"""预测归一化位移 / 趋势效率 / 极低位移概率 + 分阶段入场 —— docs/PROPOSAL_disp_prediction.md(预注册)。python3 disp_pred_study.py"""
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
from scipy.stats import kurtosis, skew, spearmanr
from sklearn.linear_model import LogisticRegression, Ridge

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lowvol_avoidance_study as L
import round3_common as C
import sleeve_anatomy as SA
import trend_rv_sleeve_study as T
from audit import reimpl_noise as RI
from event_calendar import event_frame
from etft.intraday.bars import load_bars
from etft.intraday.noise import noise_area
from etft.intraday.strategy import rel_volume
from sens_rv_weights import pct_w

OUT, TAB = L.OUT, L.TAB
KS = (15, 30, 60, 90, 120, 150)
START = {30: "10:00", 60: "10:30", 90: "11:00", 120: "11:30", 150: "12:00"}
YEARS = range(2021, 2027)
ORTHO = ["log_vix", "log_vxn", "log_vix9d", "gex_z", "log_rv_prev", "log_rv5"]
NPERM = 2000


# ---------------------------------------------------------------- 特征与目标
class Arr:
    def __init__(self, b):
        self.b = b; self.days = b.days; self.c = b.c.to_numpy(); self.h = b.h.to_numpy(); self.l = b.l.to_numpy(); self.v = b.v.to_numpy(); self.O = b.day_open.to_numpy()
        self.r = np.full_like(self.c, np.nan); self.r[:, 1:] = np.log(self.c[:, 1:] / self.c[:, :-1]); self.r[:, 0] = np.log(self.c[:, 0] / self.O)
        self.li = b.last_idx.to_numpy(); self.full = self.li >= 389
        from etft.intraday.noise import daily_vol
        self.sd = daily_vol(b, 14).to_numpy(); self.vw = b.vwap().to_numpy(); ub, lb, _ = noise_area(b); self.ub, self.lb = ub.to_numpy(), lb.to_numpy()
        self.rvl = rel_volume(b, 14, "cum").to_numpy(); self.cumabs = np.nancumsum(np.abs(np.nan_to_num(self.r)), axis=1)


def feat_B(a: Arr, K):
    c, h, l, v, r, O, sd = a.c, a.h, a.l, a.v, a.r, a.O, a.sd; sc = sd * np.sqrt(K / 390)
    r0 = r[:, :K]; rv = np.sum(r0 ** 2, axis=1); path = np.sum(np.abs(r0), axis=1); ret = np.abs(np.log(c[:, K - 1] / O)); rng = (h[:, :K].max(1) - l[:, :K].min(1)) / O
    bv = (np.pi / 2) * np.sum(np.abs(r0[:, 1:]) * np.abs(r0[:, :-1]), axis=1); rvk = np.sum(r0[:, 1:] ** 2, axis=1); jump = np.clip(1 - bv / np.where(rvk > 0, rvk, np.nan), 0, 1)
    up = np.sum(np.where(r0 > 0, r0 ** 2, 0), axis=1); dn = np.sum(np.where(r0 < 0, r0 ** 2, 0), axis=1); asym = np.abs(up - dn) / np.where(rv > 0, rv, np.nan)
    sg = np.sign(c[:, :K] - a.vw[:, :K]); cross = np.sum((sg[:, 1:] * sg[:, :-1]) < 0, axis=1) / K
    out_band = np.mean((c[:, :K] > a.ub[:, :K]) | (c[:, :K] < a.lb[:, :K]), axis=1)
    return pd.DataFrame({"b_log_rv_n": np.log(np.where(rv > 0, rv, np.nan) / (sd ** 2 * K / 390)), "b_ret_n": ret / sc, "b_range_n": rng / sc,
                         "b_relvol": np.log(a.rvl[:, K - 1]), "b_eff": ret / np.where(path > 0, path, np.nan), "b_jump": jump, "b_cross": cross, "b_asym": asym, "b_outband": out_band,
                         "b_vwapdev_n": np.abs(np.log(c[:, K - 1] / a.vw[:, K - 1])) / sc}, index=a.days)


def targets(a: Arr, K):
    n = np.arange(len(a.days)); li = a.li; ck = a.c[:, K - 1]; ce = a.c[n, li]; nrem = li + 1 - K
    disp = np.abs(np.log(ce / ck)); sc = a.sd * np.sqrt(np.maximum(nrem, 1) / 390); dn = disp / sc
    path = a.cumabs[n, li] - a.cumabs[:, K - 1]; er = disp / np.where(path > 0, path, np.nan)
    y1 = np.log(dn + 0.05); er_c = np.clip(er, 0.01, 0.99); y2 = np.log(er_c / (1 - er_c))
    ok = a.full & np.isfinite(dn) & np.isfinite(er)
    return pd.DataFrame({"dn": np.where(ok, dn, np.nan), "y1": np.where(ok, y1, np.nan), "y2": np.where(ok, y2, np.nan)}, index=a.days)


# ---------------------------------------------------------------- walk-forward
def wf(F, tg, cols, ortho=False, q=0.25, years=YEARS):
    """逐年重估;ortho=True 时在训练窗内把 ORTHO 中的列对 log_sd14 做 OLS 残差。返回 DataFrame(yhat1,yhat2,p_low,base1,base2,thr)。"""
    res = pd.DataFrame(np.nan, index=F.index, columns=["yhat1", "yhat2", "p_low", "base1", "base2", "thr"])
    for yr in years:
        d0 = pd.Timestamp(f"{yr}-01-01"); tr = (F.index < d0) & tg.y1.notna().to_numpy() & F[cols].notna().all(axis=1).to_numpy(); te = (F.index.year == yr) & F[cols].notna().all(axis=1).to_numpy()
        if tr.sum() < 300 or te.sum() == 0: continue
        Xtr, Xte = F.loc[tr, cols].copy(), F.loc[te, cols].copy()
        if ortho:
            for c_ in [c for c in ORTHO if c in cols]:
                bb = np.polyfit(Xtr["log_sd14"], Xtr[c_], 1); Xtr[c_] = Xtr[c_] - np.polyval(bb, Xtr["log_sd14"]); Xte[c_] = Xte[c_] - np.polyval(bb, Xte["log_sd14"])
        mu, sd = Xtr.mean(), Xtr.std().replace(0, 1.0); A, B_ = ((Xtr - mu) / sd).to_numpy(), ((Xte - mu) / sd).to_numpy()
        thr = tg.loc[tr, "dn"].quantile(q); ylow = (tg.loc[tr, "dn"] <= thr).astype(int).to_numpy()
        res.loc[te, "yhat1"] = Ridge(alpha=L.RIDGE_A).fit(A, tg.loc[tr, "y1"]).predict(B_); res.loc[te, "yhat2"] = Ridge(alpha=L.RIDGE_A).fit(A, tg.loc[tr, "y2"]).predict(B_)
        res.loc[te, "p_low"] = LogisticRegression(C=L.LOGIT_C, max_iter=500).fit(A, ylow).predict_proba(B_)[:, 1]
        res.loc[te, "base1"] = tg.loc[tr, "y1"].mean(); res.loc[te, "base2"] = tg.loc[tr, "y2"].mean(); res.loc[te, "thr"] = thr
    return res


def pred_metrics(res, tg, nb=1000):
    m = res.p_low.notna().to_numpy() & tg.y1.notna().to_numpy(); idx = res.index[m]; out = {}
    for nm, yc, bc in (("R2_dn", "y1", "base1"), ("R2_ER", "y2", "base2")):
        e1 = ((tg.loc[idx, yc] - res.loc[idx, "yhat" + yc[-1]]) ** 2).to_numpy(); e0 = ((tg.loc[idx, yc] - res.loc[idx, bc]) ** 2).to_numpy()
        out[nm], out[nm + "_lo"], out[nm + "_hi"] = L.month_boot(idx, lambda ii: 1 - e1[ii].sum() / e0[ii].sum(), nb=nb)
    low = (tg.loc[idx, "dn"] <= res.loc[idx, "thr"]).astype(int).to_numpy(); p = res.loc[idx, "p_low"].to_numpy()
    out["AUC_low"], out["AUC_lo"], out["AUC_hi"] = L.month_boot(idx, lambda ii: L.auc(low[ii], p[ii]), nb=nb)
    out["IC_dn"] = spearmanr(res.loc[idx, "yhat1"], tg.loc[idx, "dn"])[0]; out["n"] = len(idx)
    return out


# ---------------------------------------------------------------- 分布面板
def panel(x, name=""):
    x = pd.Series(x).dropna(); d = x[x < 0]
    eq = (1 + x).cumprod(); mdd = float((eq / eq.cummax() - 1).min()); yrs = len(x) / 252; cagr = float(eq.iloc[-1] ** (1 / yrs) - 1) if eq.iloc[-1] > 0 else np.nan
    srt = x.mean() / np.sqrt((np.minimum(x, 0) ** 2).mean()) * np.sqrt(252) if (np.minimum(x, 0) ** 2).mean() > 0 else np.nan
    prof = x[x > 0].sum(); top = x.nlargest(max(int(0.05 * len(x)), 1)).sum()
    return {"方案": name, "均值bp": 1e4 * x.mean(), "中位bp": 1e4 * x.median(), "波动bp": 1e4 * x.std(), "Sharpe": x.mean() / x.std() * np.sqrt(252), "Sortino": srt, "CAGR": cagr, "MDD": mdd,
            "Calmar": cagr / abs(mdd) if mdd < 0 else np.nan, "偏度": skew(x), "超额峰度": kurtosis(x), "CVaR5%bp": 1e4 * x[x <= x.quantile(0.05)].mean(), "盈亏比": prof / abs(d.sum()) if len(d) else np.nan,
            "胜率": float((x > 0).mean()), "前5%日占利润": top / prof if prof > 0 else np.nan, "最差日bp": 1e4 * x.min(), "活跃日": int((x != 0).sum())}


# ---------------------------------------------------------------- Sleeve 延迟入场
def tqqq_scale(b):
    import yfinance as yf
    pt = yf.download("TQQQ", start="2018-10-01", end="2026-10-12", auto_adjust=False, progress=False, threads=False)["Close"]
    pt = pt.iloc[:, 0] if hasattr(pt, "columns") else pt; pt.index = pd.to_datetime(pt.index).tz_localize(None)
    sp = yf.Ticker("TQQQ").splits; sp.index = pd.to_datetime(sp.index).tz_localize(None); fsp = pd.Series(1.0, index=pt.index)
    for d, r in sp.items(): fsp[pt.index < d] *= r
    return (b.day_close / (pt * fsp).reindex(b.days).ffill()) / 3.0


def sleeve_start(b, scale, start, phases=range(30)):
    days = b.days; rs = []
    for off in phases:
        checks = [t for t in SA.chk(off) if t >= start]
        rr, F = RI.simulate(b, checks=checks, sizing="voltarget", max_lev=4.0, reentry="hold", half_spread=0.005, commission=0.0045, **SA.BASE)
        if len(F):
            F = F.copy(); F["day"] = pd.to_datetime(F.day); c_old = F.groupby("day").commission.sum().reindex(days).fillna(0.0); rr = rr + c_old - c_old * scale.reindex(days).fillna(1.0)
        rs.append(rr)
    return pd.concat(rs, axis=1).mean(axis=1)


def causal_rank(p, minn=60):
    v = p.to_numpy(float); out = np.full(len(v), 0.5); hist = []
    for i, x in enumerate(v):
        if np.isfinite(x):
            if len(hist) >= minn: out[i] = np.mean(np.array(hist) < x)
            hist.append(x)
    return pd.Series(out, index=p.index)



def selftest(b, a0):
    """第 i 天 bar≥K 全部 ×7(及 day_close):B(K) 特征与 gap_n 在第 i 天必须不变;阳性对照:目标 dn 必须变化。"""
    import dataclasses
    rg = np.random.default_rng(5); idx = rg.choice(np.arange(300, len(b.days) - 1), 8, replace=False); worst, ydiff = 0.0, 0.0
    for K in (30, 90):
        B0 = feat_B(a0, K); t0_ = targets(a0, K)
        for i in idx:
            def P(df, i=i, K=K):
                df = df.copy(); df.iloc[i, K:] = df.iloc[i, K:] * 7.0; return df
            dc = b.day_close.copy(); dc.iloc[i] = dc.iloc[i] * 7.0
            bp = dataclasses.replace(b, o=P(b.o), h=P(b.h), l=P(b.l), c=P(b.c), v=P(b.v), day_close=dc); ap = Arr(bp); d = b.days[i]
            worst = max(worst, float(np.nanmax(np.abs(B0.loc[d].to_numpy(float) - feat_B(ap, K).loc[d].to_numpy(float)))), float(abs(a0.sd[i] - ap.sd[i])))
            ydiff = max(ydiff, float(abs(np.nan_to_num(t0_.dn.iloc[i]) - np.nan_to_num(targets(ap, K).dn.iloc[i]))))
    print(f"[自检] 第 i 天 bar≥K ×7:B(K)/σ₁₄ 在第 i 天最大差 {worst:.2e}(须为 0);阳性对照(目标 dn)最大差 {ydiff:.3f}(须>0)"); assert worst == 0.0 and ydiff > 0


def main():
    t0 = time.time(); pd.set_option("display.width", 250, "display.max_rows", 300, "display.float_format", lambda x: f"{x:8.3f}")
    b = load_bars("QQQ"); days = b.days; a = Arr(b)
    X, Y, A, B_old = L.build(b); E = event_frame(days, b.last_idx.to_numpy()); EC = list(E.columns)
    A_all = A + ["log_sd14"] if "log_sd14" not in A else A
    base = pd.concat([X[A_all], E], axis=1); base["gap_n"] = X["gap_abs"] / a.sd
    A_noGEX = [c for c in A_all if c not in ("gex_z",)]
    A_ortho_cols = [c for c in A_all if c != "gap_abs"] + ["gap_n"]
    print(f"数据就绪 {time.time()-t0:.0f}s;A 列 {A_all}"); selftest(b, a)
    # ======================================================= 1) 预测力:K 敏感性 + 消融
    resK, tgK, FK = {}, {}, {}
    rows = []
    for K in KS:
        BK = feat_B(a, K); F = pd.concat([base, BK], axis=1); tg = targets(a, K); Bc = list(BK.columns); FK[K], tgK[K] = F, tg
        main_cols = A_ortho_cols + Bc + EC; resK[K] = wf(F, tg, main_cols, ortho=True)
        rows.append(dict(K=K, 模型="A_ortho+B+E(主)", **pred_metrics(resK[K], tg)))
        if K == 30:
            for nm, cols, orth in (("A_raw", [c for c in A_all if c != "gap_abs"] + ["gap_abs"], False), ("A_ortho", A_ortho_cols, True), ("A_ortho 去GEX", [c for c in A_ortho_cols if c != "gex_z"], True),
                                   ("B(K) 单独", Bc, False), ("A_ortho+B", A_ortho_cols + Bc, True), ("A_raw+B+E", A_all + Bc + EC, False), ("A_ortho+B+E 去GEX", [c for c in main_cols if c != "gex_z"], True),
                                   ("仅GEX(ortho)+log_sd14", ["log_sd14", "gex_z"], True)):
                rows.append(dict(K=K, 模型=nm, **pred_metrics(wf(F, tg, cols, ortho=orth), tg)))
    P = pd.DataFrame(rows); P.to_csv(TAB / "disp_pred_metrics.csv", index=False)
    print("\n1) 预测力(OOS 2021–2026;R²/AUC 括号为月聚类自助 95%)")
    print(P[["K", "模型", "n", "R2_dn", "R2_dn_lo", "R2_dn_hi", "R2_ER", "AUC_low", "AUC_lo", "AUC_hi", "IC_dn"]].to_string(index=False))
    # ======================================================= 2) 位移 vs Sleeve:分位表(神谕关系)
    ret0 = None
    scale = tqqq_scale(b); R = {}
    for K, st in START.items():
        R[K] = sleeve_start(b, scale, st); print(f"  Sleeve 延迟入场 ≥{st}: 日均 {R[K].mean()*1e4:.2f}bp ({time.time()-t0:.0f}s)")
    base_curve = pd.read_pickle(L.OUT / "sleeve_anatomy_curve.pkl"); base_curve.index = pd.to_datetime(base_curve.index); base_curve = base_curve.reindex(days).fillna(0.0)
    print(f"校验:R(≥10:00) 与 sleeve_anatomy_curve 最大绝对差 {float((R[30]-base_curve).abs().max())*1e4:.5f} bp")
    td = pd.Series(days.dayofweek != 0, index=days); oos = (days.year >= 2021); sel_days = td.to_numpy() & oos
    q5 = lambda s: pd.qcut(s.rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    tg30 = tgK[30]; m = sel_days & tg30.y1.notna().to_numpy()
    tab = pd.DataFrame({"真实归一化位移 dn": R[30][m].groupby(q5(tg30.dn[m])).mean() * 1e4, "真实趋势效率 ER": R[30][m].groupby(q5(np.exp(tg30.y2[m]))).mean() * 1e4,
                        "预测 p(极低位移)": R[30][m & resK[30].p_low.notna().to_numpy()].groupby(q5(resK[30].p_low[m & resK[30].p_low.notna().to_numpy()])).mean() * 1e4,
                        "预测 ŷ1(归一化位移)": R[30][m & resK[30].p_low.notna().to_numpy()].groupby(q5(resK[30].yhat1[m & resK[30].p_low.notna().to_numpy()])).mean() * 1e4})
    print("\n2) Sleeve(10:00 起)日均收益 bp,按五分位(1=最低);OOS 周二至周五"); print(tab.round(1).to_string()); tab.to_csv(TAB / "disp_pred_quintiles.csv")
    # ======================================================= 3) 门控规则
    rk = {K: causal_rank(resK[K].p_low) for K in START}
    ks = list(START)                       # 30,60,90,120,150
    Rm = np.column_stack([R[K].to_numpy() for K in ks] + [np.zeros(len(days))])       # 第 5 列 = 跳过
    def sel_staged(theta, never_skip=False, oracle=False):
        s = np.full(len(days), 5, int)
        for i in range(len(days)):
            for j, K in enumerate(ks):
                if oracle:
                    low = (tgK[K].dn.iloc[i] <= resK[K].thr.iloc[i]) if np.isfinite(tgK[K].dn.iloc[i]) and np.isfinite(resK[K].thr.iloc[i]) else False
                    bad = low
                else: bad = rk[K].iloc[i] > theta
                if not bad: s[i] = j; break
            else:
                if never_skip: s[i] = len(ks) - 1
        return s
    def sel_single(theta, oracle=False):
        s = np.zeros(len(days), int)
        for i in range(len(days)):
            if oracle: bad = bool(tgK[30].dn.iloc[i] <= resK[30].thr.iloc[i]) if np.isfinite(tgK[30].dn.iloc[i]) and np.isfinite(resK[30].thr.iloc[i]) else False
            else: bad = rk[30].iloc[i] > theta
            if bad: s[i] = 5
        return s
    cfg = {}
    for th in (0.7, 0.8):
        cfg[f"ST_{th}"] = ("sel", sel_staged(th)); cfg[f"SS_{th}"] = ("sel", sel_single(th)); cfg[f"W_{th}"] = ("sel", sel_staged(th, never_skip=True))
    yh = resK[30].yhat1; hist = []; med = np.full(len(days), np.nan)
    for i, x in enumerate(yh.to_numpy()):
        if np.isfinite(x):
            if len(hist) >= 60: med[i] = np.median(hist)
            hist.append(x)
    for g in (0.5, 1.0):
        s = np.where(np.isfinite(med), np.clip(np.exp(g * (yh.to_numpy() - med)), 0.5, 1.5), 1.0); cfg[f"CONT_{g}"] = ("scale", s)
    orc = {"神谕ST(真实low_j)": ("sel", sel_staged(0, oracle=True)), "神谕SS(真实low_30)": ("sel", sel_single(0, oracle=True)),
           "神谕CONT(真实y1,γ=1)": ("scale", np.where(tg30.y1.notna(), np.clip(np.exp(1.0 * (tg30.y1.to_numpy() - np.nanmedian(tg30.y1.to_numpy()))), 0.5, 1.5), 1.0))}
    def series_of(kind, v):
        if kind == "sel": return pd.Series(Rm[np.arange(len(days)), v], index=days)
        return pd.Series(R[30].to_numpy() * v, index=days)
    # 置换零分布
    tr_idx = np.where(sel_days)[0]; rg = np.random.default_rng(7); perms = [rg.permutation(len(tr_idx)) for _ in range(NPERM)]
    base_sr = L.sr(R[30].iloc[tr_idx]); nulls, obs = {}, {}
    for nm, (kind, v) in cfg.items():
        vv = v[tr_idx]; ser = []
        for pi in perms:
            vp = vv[pi]
            x = Rm[tr_idx, vp] if kind == "sel" else R[30].to_numpy()[tr_idx] * vp
            ser.append(L.sr(pd.Series(x)))
        nulls[nm] = np.array(ser) - base_sr; x0 = Rm[tr_idx, vv] if kind == "sel" else R[30].to_numpy()[tr_idx] * vv; obs[nm] = L.sr(pd.Series(x0)) - base_sr
    fam = np.max(np.vstack(list(nulls.values())), axis=0)
    # 组合层
    bh = pd.read_pickle(L.OUT / "strategy_comparison_bh.pkl").dropna(); bh.index = pd.to_datetime(bh.index)
    px_all = C.load_close("yahoo/QQQ.parquet"); p_all = pct_w(px_all.pct_change().rolling(21).std(), 504)
    port = lambda sl: T.build(bh, sl.reindex(bh.index).fillna(0.0), (1.0, 1.0), (("lin", .5, 1.5), None, 1.0), 1.0, px_all, p_all)[0]
    full0 = R[30].copy(); base_p = port(full0); rows = []; panels = [panel(L.period(R[30][td], "全OOS"), "基线 Sleeve(≥10:00)"), panel(L.period(base_p, "全OOS"), "基线 组合(L1+Sleeve)")]
    gated = {}
    for nm, (kind, v) in {**cfg, **orc}.items():
        s = series_of(kind, v); g = full0.copy(); g[oos] = s[oos]; gated[nm] = g; xp = port(g)
        row = dict(配置=nm, 入选=nm in cfg, 跳过比例=float((v[tr_idx] == 5).mean()) if kind == "sel" else 0.0, 平均开始时点=float(np.mean([int(START[ks[j]][:2]) + int(START[ks[j]][3:]) / 60 for j in v[tr_idx] if j < 5])) if kind == "sel" else 10.0)
        for k in ("dev-OOS", "test", "全OOS"):
            row[f"Sleeve ΔSR {k}"] = L.sr(L.period(g[td], k)) - L.sr(L.period(R[30][td], k)); row[f"组合 ΔSR {k}"] = L.sr(L.period(xp, k)) - L.sr(L.period(base_p, k))
        if nm in cfg:
            row["置换p(单)"] = float(np.mean(nulls[nm] >= obs[nm])); row["置换p(族内max,8)"] = float(np.mean(fam >= obs[nm]))
            d, dist = T.boot_dsr(L.period(xp, "全OOS"), L.period(base_p, "全OOS"), nb=1500, L=10, seed=0); row["组合ΔSR_lo"], row["组合ΔSR_hi"] = np.percentile(dist, 2.5), np.percentile(dist, 97.5)
        rows.append(row); panels.append(panel(L.period(g[td], "全OOS"), nm + " [Sleeve]")); panels.append(panel(L.period(xp, "全OOS"), nm + " [组合]"))
    Rz = pd.DataFrame(rows); print("\n3) 门控规则(Sleeve/组合 ΔSR;OOS 2021–2026)"); print(Rz.round(3).to_string(index=False)); Rz.to_csv(TAB / "disp_pred_gating.csv", index=False)
    Pn = pd.DataFrame(panels); print("\n4) 收益分布面板(全 OOS,周一排除)"); print(Pn.round(3).to_string(index=False)); Pn.to_csv(TAB / "disp_pred_panel.csv", index=False)
    pd.to_pickle(dict(P=P, Rz=Rz, Pn=Pn, resK=resK), OUT / "disp_pred.pkl"); print(f"总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
