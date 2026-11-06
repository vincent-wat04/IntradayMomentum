"""日内回转 v2 第二轮 —— docs/PROPOSAL_intraday_reversion_v2.md「第二轮」。H1 周期性 / H2 末半小时 / H3 BVC 失衡 / H4 缺口+趋势 / H5 ML v2.1。"""
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
from scipy.stats import norm, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lowvol_avoidance_study as LV
import intraday_reversion_v2_study as V2
import regime_features as RF
from etft.intraday.bars import load_bars
from etft.intraday.strategy import rel_volume

OUT, TAB = LV.OUT, LV.TAB
KS = np.arange(34, 355, 5)
HS = (15, 30, 60, "C")
OOS0 = 2021
COSTS = (0.25, 0.5, 1.0, 2.0)


def sr(x): return V2.sr(x)


def build_panel(b, Xd):
    days = b.days; n = len(days)
    c, o, hi, lo, v = (b.c.to_numpy(), b.o.to_numpy(), b.h.to_numpy(), b.l.to_numpy(), b.v.to_numpy())
    O = b.day_open.to_numpy(); full = (b.last_idx.to_numpy() >= 389); sd = np.exp(Xd.log_sd14.to_numpy())
    vw = b.vwap().to_numpy(); rv = rel_volume(b, 14, "cum").to_numpy()
    r = np.full_like(c, np.nan); r[:, 1:] = np.log(c[:, 1:] / c[:, :-1]); r[:, 0] = np.log(c[:, 0] / O)
    hh = np.maximum.accumulate(np.where(np.isfinite(hi), hi, -np.inf), axis=1); ll = np.minimum.accumulate(np.where(np.isfinite(lo), lo, np.inf), axis=1)
    clv = np.where(hi > lo, ((c - lo) - (hi - c)) / (hi - lo), 0.0); sgn = np.sign(c - o)
    gap = np.log(O / b.prev_close_adj.to_numpy()) / sd
    # ---- BVC:σ_1m 为过去 20 日的 1 分钟收益标准差(因果,shift 1)
    s1 = pd.Series(np.nanstd(r, axis=1), index=days).rolling(20, min_periods=10).mean().shift(1).to_numpy()
    buy = norm.cdf(r / s1[:, None]); sv = (2 * buy - 1) * v
    cs_v, cs_sv = np.nancumsum(v, axis=1), np.nancumsum(np.nan_to_num(sv), axis=1)
    def imb(k, w): lo_ = max(k - w, -1); return (cs_sv[:, k] - (cs_sv[:, lo_] if lo_ >= 0 else 0)) / np.maximum(cs_v[:, k] - (cs_v[:, lo_] if lo_ >= 0 else 0), 1)
    # ---- 窗口收益矩阵 W[h][d,k] = ln(o[k+1+h]/o[k+1]),用于周期性(过去 n 日同窗口均值,shift 1)
    def Wmat(h):
        M = np.full((n, c.shape[1]), np.nan)
        for k in range(0, 389 - h - 1):
            M[:, k] = np.log(o[:, k + 1 + h] / o[:, k + 1])
        return M
    Wm = {h: Wmat(h) for h in (15, 30, 60)}
    per = {}
    for h in Wm:
        D = pd.DataFrame(Wm[h]); per[h] = {nn: D.shift(1).rolling(nn, min_periods=max(3, nn // 2)).mean().to_numpy() for nn in (5, 10, 20)}
        per[(h, "sd")] = D.shift(1).rolling(252, min_periods=60).std().to_numpy()
    # 半小时收益(r1: 首半小时, r12: 15:00–15:30)
    r1 = np.log(c[:, 29] / O); r12 = np.log(c[:, 359] / c[:, 329])
    z = lambda x: (x - pd.Series(x).rolling(252, min_periods=60).mean().shift(1).to_numpy()) / pd.Series(x).rolling(252, min_periods=60).std().shift(1).to_numpy()
    zr1, zr12 = z(r1), z(r12)
    close = c[:, 389]
    rows = []
    for k in KS:
        d = pd.DataFrame({"day": days, "k": k, "di": np.arange(n), "sd": sd})
        w30 = c[:, k - 29:k + 1]; v30 = v[:, k - 29:k + 1]; v15 = v[:, k - 14:k + 1]
        rv15 = np.sum(r[:, k - 14:k + 1] ** 2, axis=1); rvd = np.sum(r[:, :k + 1] ** 2, axis=1)
        F = {"z5": np.log(c[:, k] / c[:, k - 5]) / (sd * np.sqrt(5 / 390)), "z15": np.log(c[:, k] / c[:, k - 15]) / (sd * np.sqrt(15 / 390)),
             "z30": np.log(c[:, k] / c[:, k - 30]) / (sd * np.sqrt(30 / 390)), "z_open": np.log(c[:, k] / O) / (sd * np.sqrt((k + 1) / 390)),
             "vwap_dev": np.log(c[:, k] / vw[:, k]) / (sd * np.sqrt(30 / 390)),
             "pctb30": (c[:, k] - w30.mean(1)) / (2 * np.where(w30.std(1) > 0, w30.std(1), np.nan)),
             "pos_range": (c[:, k] - ll[:, k]) / np.where(hh[:, k] > ll[:, k], hh[:, k] - ll[:, k], np.nan),
             "dist_hi": np.log(hh[:, k] / c[:, k]) / sd, "dist_lo": np.log(c[:, k] / ll[:, k]) / sd,
             "clv30": np.sum(clv[:, k - 29:k + 1] * v30, axis=1) / np.sum(v30, axis=1), "netvol15": np.sum(sgn[:, k - 14:k + 1] * v15, axis=1) / np.sum(v15, axis=1),
             "rv_ratio15": np.log(np.where(rv15 > 0, rv15, np.nan) / np.where(rvd > 0, rvd * 15 / (k + 1), np.nan)),
             "range15": (hi[:, k - 14:k + 1].max(1) - lo[:, k - 14:k + 1].min(1)) / O / (sd * np.sqrt(15 / 390)),
             "relvol": np.log(rv[:, k]), "tod": k / 390.0, "gap": gap, "log_sd14": Xd.log_sd14.to_numpy(), "log_vix": Xd.log_vix.to_numpy(), "gex_z": Xd.gex_z.to_numpy(),
             "imb15": imb(k, 15), "imb30": imb(k, 30), "r1": r1 / (sd * np.sqrt(30 / 390)), "r_prev30": np.log(c[:, k] / c[:, k - 30]) / (sd * np.sqrt(30 / 390))}
        for h in (15, 30, 60):
            s_ = per[(h, "sd")][:, k]
            for nn in (5, 10, 20): F[f"per{nn}_{h}"] = per[h][nn][:, k] / np.where(s_ > 0, s_, np.nan)
        for kk, vv in F.items(): d[kk] = vv
        # 未来收益 (bp) 与持有定义
        for h in (15, 30, 60):
            d[f"y{h}"] = np.log(o[:, k + 1 + h] / o[:, k + 1]) * 1e4 if k + 1 + h <= 389 else np.nan
        d["yC"] = np.log(close / o[:, k + 1]) * 1e4
        d["fut5"] = np.log(o[:, k + 6] / o[:, k + 1]) * 1e4
        rows.append(d[full])
    P = pd.concat(rows, ignore_index=True).replace([np.inf, -np.inf], np.nan)
    return P, dict(zr1=zr1, zr12=zr12, gap=gap, sd=sd, O=O, o=o, c=c, close=close, full=full)


def trades(P, sig, hold):
    """sig: 带符号信号(0=不交易);hold: 持有 bar 数(int)或 'C'(到收盘)。同日不重叠。返回 DataFrame(di, day, k, side, gross_bp)"""
    h = hold
    y = P["yC"].to_numpy() if h == "C" else P[f"y{h}"].to_numpy()
    a = pd.DataFrame({"di": P.di.to_numpy(), "k": P.k.to_numpy(), "sig": sig, "y": y, "day": P.day.to_numpy()})
    a = a[np.isfinite(a.sig) & (a.sig != 0) & np.isfinite(a.y)]
    res = []
    for di, g in a.groupby("di", sort=False):
        free = -1
        for k, s_, y_, d_ in zip(g.k.to_numpy(), g.sig.to_numpy(), g.y.to_numpy(), g.day.to_numpy()):
            if k >= free:
                res.append((di, d_, k, 1 if s_ > 0 else -1, (1 if s_ > 0 else -1) * y_)); free = (10 ** 6) if h == "C" else k + 1 + h
    return pd.DataFrame(res, columns=["di", "day", "k", "side", "gross"])


def summ(T, label, ndays, rows):
    for pk, (a, z) in {"dev-OOS": (2021, 2022), "test": (2023, 2026), "全OOS": (2021, 2026)}.items():
        t = T[(T.day.dt.year >= a) & (T.day.dt.year <= z)]
        row = dict(配置=label, 期间=pk, 笔数=len(t), 每日笔数=len(t) / max(ndays[pk], 1), 毛bp_每笔=t.gross.mean() if len(t) else np.nan, 胜率=(t.gross > 0).mean() if len(t) else np.nan,
                   多侧毛bp=t[t.side > 0].gross.mean() if (t.side > 0).any() else np.nan, 空侧毛bp=t[t.side < 0].gross.mean() if (t.side < 0).any() else np.nan)
        for cst in COSTS:
            net = t.assign(net=t.gross - 2 * cst).groupby("day").net.sum().reindex(ndays["_idx"][pk]).fillna(0.0).to_numpy()
            row[f"净SR@{cst:g}bp"] = sr(net)
        rows.append(row)


def poison(b, Xd, nd=10, seed=0, kmax=250):
    P0, _ = build_panel(b, Xd); rg = np.random.default_rng(seed); full = np.where(b.last_idx.to_numpy() >= 389)[0]
    cols = ["z5", "z15", "z30", "z_open", "vwap_dev", "pctb30", "pos_range", "clv30", "netvol15", "relvol", "imb15", "imb30", "r1", "r_prev30", "per10_30", "per5_15", "per20_60"]
    worst = 0.0
    for i in rg.choice(full[full > 300], nd, replace=False):
        def Pz(df):
            df = df.copy(); df.iloc[i, kmax + 1:] = df.iloc[i, kmax + 1:] * 7.0; return df
        bp = dataclasses.replace(b, o=Pz(b.o), h=Pz(b.h), l=Pz(b.l), c=Pz(b.c), v=Pz(b.v))
        Pp, _ = build_panel(bp, Xd)
        a = P0[(P0.di == i) & (P0.k <= kmax)].set_index("k")[cols].sort_index(); c_ = Pp[(Pp.di == i) & (Pp.k <= kmax)].set_index("k")[cols].sort_index()
        worst = max(worst, float(np.nanmax(np.abs(a.to_numpy(float) - c_.to_numpy(float)))))
    return worst


def main():
    b = load_bars("QQQ"); days = b.days
    Xd, Yd, _, _ = LV.build(b)
    print("构建面板…", flush=True); P, aux = build_panel(b, Xd)
    print(f"面板 {len(P):,} 行;{P.di.nunique()} 个整日;决策网格 {len(KS)}/日(10:05–15:25)")
    w = poison(b, Xd); print(f"[污染自检] 10 个整日 bar>251 ×7:k≤250 的 17 个特征(含 per、imb、r1)最大差 {w:.2e}(须为 0)"); assert w == 0.0
    nd = {}
    for pk, (a, z) in {"dev-OOS": (2021, 2022), "test": (2023, 2026), "全OOS": (2021, 2026)}.items():
        idx = pd.DatetimeIndex([d for d in days if a <= d.year <= z and b.last_idx.loc[d] >= 389]); nd[pk] = len(idx); nd.setdefault("_idx", {})[pk] = idx
    rows, trd = [], {}
    # ---------------- H1 周期性
    for h in (15, 30, 60):
        x = P[f"per10_{h}"].to_numpy()
        for thr in (1.0, 1.5):
            for dname, sgn in (("持续", 1), ("反向", -1)):
                sig = np.where(np.abs(x) >= thr, sgn * np.sign(x), 0.0); T_ = trades(P, sig, h); trd[f"H1 per10 h={h} |z|≥{thr} {dname}"] = T_; summ(T_, f"H1 per10 h={h} |z|≥{thr} {dname}", nd, rows)
    # ---------------- H2 末半小时:k=359 决策 → 开盘成交 → 持有到收盘
    c, o, close = aux["c"], aux["o"], aux["close"]; n = len(days)
    zr1, zr12 = aux["zr1"], aux["zr12"]; y_last = np.log(close / o[:, 360]) * 1e4
    sd14 = np.exp(Xd.log_sd14.to_numpy()); hv = sd14 > pd.Series(sd14).rolling(252, min_periods=60).median().shift(1).to_numpy()
    full = aux["full"]
    for sub, mask in (("全部日", full), ("高波动日", full & hv)):
        sg = np.sign(np.nan_to_num(zr1) + np.nan_to_num(zr12))
        for dname, sgn in (("顺势(文献)", 1), ("反向", -1)):
            ok = mask & np.isfinite(zr1) & np.isfinite(zr12) & (sg != 0)
            T_ = pd.DataFrame({"di": np.where(ok)[0], "day": days[ok], "k": 359, "side": (sgn * sg[ok]).astype(int), "gross": sgn * sg[ok] * y_last[ok]})
            trd[f"H2 末半小时 {sub} {dname}"] = T_; summ(T_, f"H2 末半小时 {sub} {dname}", nd, rows)
    # H2 的 IC 与 R²(文献口径):y_last ~ r1 + r12
    ok = full & np.isfinite(zr1) & np.isfinite(zr12); yr = y_last[ok]; yrs_ = days[ok].year
    for nm, (a, z) in {"dev-OOS": (2021, 2022), "test": (2023, 2026), "2019–2020": (2019, 2020)}.items():
        mm = (yrs_ >= a) & (yrs_ <= z)
        print(f"  H2 IC [{nm}] n={mm.sum()}: Spearman(r1,末30)={spearmanr(zr1[ok][mm], yr[mm])[0]:+.3f}  (r12,末30)={spearmanr(zr12[ok][mm], yr[mm])[0]:+.3f}  (r1+r12,末30)={spearmanr(zr1[ok][mm]+zr12[ok][mm], yr[mm])[0]:+.3f}")
    # ---------------- H3 BVC 失衡
    for w_ in (15, 30):
        col = P[f"imb{w_}"].to_numpy()
        # 因果阈值:同时刻 k、过去 252 日 |imb| 的分位 → 用 pandas 按 k 分组滚动
        Pk = P[["di", "k"]].copy(); Pk["a"] = np.abs(col)
        thr = {}
        for q in (0.90, 0.95):
            thr[q] = Pk.sort_values(["k", "di"]).groupby("k")["a"].transform(lambda s: s.shift(1).rolling(252, min_periods=60).quantile(q)).reindex(P.index).to_numpy()
        for q in (0.90, 0.95):
            for h in (15, 30, 60, "C"):
                for dname, sgn in (("反转", -1), ("顺向", 1)):
                    sig = np.where(np.abs(col) >= thr[q], sgn * np.sign(col), 0.0); T_ = trades(P, sig, h)
                    nm = f"H3 imb{w_} top{int(round((1-q)*100))}% h={h} {dname}"; trd[nm] = T_; summ(T_, nm, nd, rows)
    # ---------------- H4 缺口 + 趋势(09:35 决策 → 开盘成交 → 持有到收盘)
    px = RF.load_P()["QQQ"].dropna(); up = (px > px.rolling(200).mean()).shift(1).reindex(days).fillna(False).to_numpy()
    gap = aux["gap"]; y_day = np.log(close / o[:, 5]) * 1e4
    for th in (1.0, 1.5):
        rules = {"(a)下跌缺口+上升趋势→多": (gap <= -th) & up, "(b)上涨缺口+下降趋势→空": (gap >= th) & (~up), "(c)无条件 fade": np.abs(gap) >= th}
        for rn, mk in rules.items():
            ok = full & mk & np.isfinite(gap)
            side = np.where(rn.startswith("(c)"), -np.sign(gap), np.where(gap < 0, 1, -1))
            T_ = pd.DataFrame({"di": np.where(ok)[0], "day": days[ok], "k": 4, "side": side[ok].astype(int), "gross": side[ok] * y_day[ok]})
            nm = f"H4 θ={th} {rn}"; trd[nm] = T_; summ(T_, nm, nd, rows)
    # ---------------- H5 ML v2.1
    base = V2.FEATS
    extra = {h: [f"per5_{h}", f"per10_{h}", f"per20_{h}"] for h in (15, 30, 60)}
    common = ["imb15", "imb30", "r1", "r_prev30"]
    from sklearn.linear_model import Ridge
    import lightgbm as lgb
    yrs = P.day.dt.year.to_numpy()
    for hh in (15, 30, 60, "C"):
        fe = base + common + extra[30 if hh == "C" else hh]
        y = (P["yC"] if hh == "C" else P[f"y{hh}"]).to_numpy(); sc = P["sd"].to_numpy() * (np.sqrt(((389 - P.k.to_numpy()) / 390.0)) if hh == "C" else np.sqrt(hh / 390.0)) * 1e4
        yn = y / sc; okf = P[fe].notna().all(1).to_numpy()
        for mname in ("ridge", "lgbm"):
            pr = np.full(len(P), np.nan)
            for yr in range(OOS0, 2027):
                tr = (yrs < yr) & np.isfinite(yn) & okf; te = (yrs == yr) & okf
                if tr.sum() < 5000 or te.sum() == 0: continue
                Xtr, Xte = P.loc[tr, fe], P.loc[te, fe]; mu, sd_ = Xtr.mean(), Xtr.std().replace(0, 1.0)
                A, B = ((Xtr - mu) / sd_).to_numpy(), ((Xte - mu) / sd_).to_numpy(); yt = np.clip(yn[tr], -6, 6)
                m = Ridge(alpha=100.0).fit(A, yt) if mname == "ridge" else lgb.LGBMRegressor(max_depth=3, n_estimators=200, learning_rate=0.03, min_child_samples=500, subsample=0.7, subsample_freq=1, colsample_bytree=0.8, random_state=0, verbose=-1, n_jobs=1).fit(A, yt)
                pr[te] = m.predict(B)
            ok = np.isfinite(pr) & np.isfinite(yn)
            ic = spearmanr(pr[ok], yn[ok])[0]; ics = {yy: spearmanr(pr[ok & (yrs == yy)], yn[ok & (yrs == yy)])[0] for yy in range(OOS0, 2027)}
            print(f"  H5 IC {mname} h={hh}: {ic:+.4f}  逐年 " + " ".join(f"{k}:{v:+.3f}" for k, v in ics.items()))
            sig_bp = pr * sc
            for tau in (1.5, 2.5, 4.0):
                T_ = trades(P, np.where(np.abs(sig_bp) >= tau, np.sign(sig_bp), 0.0), hh); nm = f"H5 {mname} h={hh} τ={tau:g}bp"; trd[nm] = T_; summ(T_, nm, nd, rows)
    R = pd.DataFrame(rows); R.to_csv(TAB / "intraday_rev_v2b_trades.csv", index=False)
    # ---------------- 翻转安慰剂(全部 78 个配置;t 统计族内 max)
    rg = np.random.default_rng(1); NP = 2000
    alld = pd.DatetimeIndex(sorted(set(days))); dpos = {d: i for i, d in enumerate(alld)}; U = rg.choice([-1.0, 1.0], size=(NP, len(alld)))
    zs, nulls = {}, {}
    for nm, T_ in trd.items():
        t = T_[T_.day.dt.year >= OOS0]
        if len(t) < 100: continue
        ii = np.array([dpos[d] for d in t.day]); g = t.gross.to_numpy(); zf = lambda a_: a_.mean() / (a_.std() / np.sqrt(len(a_)))
        zs[nm] = zf(g); nulls[nm] = np.array([zf(U[p][ii] * g) for p in range(NP)])
    fam = np.max(np.vstack(list(nulls.values())), axis=0)
    PR = pd.DataFrame([dict(配置=k, 全OOS毛bp=trd[k][trd[k].day.dt.year >= OOS0].gross.mean(), 笔数=int((trd[k].day.dt.year >= OOS0).sum()), t=zs[k], p_单=float(np.mean(nulls[k] >= zs[k])), p_族内max=float(np.mean(fam >= zs[k]))) for k in zs])
    PR.to_csv(TAB / "intraday_rev_v2b_placebo.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
    # 汇总:dev/test 同号为正,毛 bp ≥ 4 或 t 最大
    pv = R.pivot_table(index="配置", columns="期间", values=["毛bp_每笔", "笔数", "净SR@1bp", "净SR@0.25bp"])
    both = pv[(pv[("毛bp_每笔", "dev-OOS")] > 0) & (pv[("毛bp_每笔", "test")] > 0)]
    print(f"\n配置总数 {len(pv)};dev-OOS 与 test 毛收益/笔同为正:{len(both)}")
    show = both.assign(全=pv[("毛bp_每笔", "全OOS")]).sort_values(("毛bp_每笔", "test"), ascending=False)
    cols = [("毛bp_每笔", "dev-OOS"), ("毛bp_每笔", "test"), ("笔数", "dev-OOS"), ("笔数", "test"), ("净SR@1bp", "dev-OOS"), ("净SR@1bp", "test")]
    print(show[cols].round(2).head(25).to_string())
    print("\n翻转安慰剂 p(族内 max, %d 配置) 最小 12 个:" % len(PR)); print(PR.sort_values("p_族内max").head(12).round(3).to_string(index=False))
    pd.to_pickle(dict(R=R, PR=PR), OUT / "intraday_rev_v2b.pkl")


if __name__ == "__main__":
    main()
