"""避开事后低波日 —— 严格按 docs/PROPOSAL_lowvol_avoidance.md 执行。

决策时点 = 10:00 收盘(第 30 根 bar)。引擎最早检查点 10:00,所以 ≤10:00 的信息对 30 个相位全部是事前。
目标变量只含 10:00 之后(Y_rv = log Σ r²;Y_disp = |ln(收盘/10:00 收盘)|)。
缩放为日级、精确:ret_scaled = s × ret(与 gex_scaling_raw.py 同一论证)。

用法(在 studies/ 目录):
  export ETFT_DATA=... ETFT_HOME=... OMP_NUM_THREADS=1
  python lowvol_avoidance_study.py
"""
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
import paths as _P
import trend_rv_sleeve_study as T
from etft.intraday.bars import load_bars
from etft.intraday.noise import daily_vol, noise_area
from etft.intraday.strategy import rel_volume
from sens_rv_weights import pct_w
import round3_common as C

OUT = Path("outputs/full/intraday"); OUT.mkdir(parents=True, exist_ok=True)
TAB = Path("../outputs/tables") if Path("../outputs/tables").exists() else Path("outputs/tables")
K = 30                       # 决策时点:第 30 根 bar(10:00)收盘;列 0..29 为事前
DEV_OOS, TEST = ("2021", "2022"), ("2023", "2026")
OOS_YEARS = range(2021, 2027)
LOW_Q = 0.40
RIDGE_A, LOGIT_C = 10.0, 0.5
NPERM, NBOOT = 2000, 2000


# ----------------------------------------------------------------- 工具
def sr(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return x.mean() / x.std() * np.sqrt(252) if len(x) > 30 and x.std() > 0 else np.nan


def auc(y, s):
    """Mann-Whitney AUC;y∈{0,1}。"""
    from scipy.stats import rankdata
    y = np.asarray(y).astype(bool); s = np.asarray(s, float)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


# ----------------------------------------------------------------- 特征
def cboe_prev(name, days):
    s = pd.read_parquet(_P.RAW / "cboe" / f"{name}.parquet")["close"]
    s.index = pd.to_datetime(s.index).tz_localize(None)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s.shift(1).reindex(days, method="ffill")


def build(b):
    """返回 (X 特征表, Y 目标表)。X 只使用:t 日 bar 0..29;t−1 及更早的完整日;t−1 的指数收盘。"""
    days = b.days; n = len(days)
    O = b.day_open.to_numpy()
    c = b.c.to_numpy(); h = b.h.to_numpy(); l = b.l.to_numpy(); v = b.v.to_numpy()
    r = np.full_like(c, np.nan)
    r[:, 1:] = np.log(c[:, 1:] / c[:, :-1]); r[:, 0] = np.log(c[:, 0] / O)
    sd14 = daily_vol(b, 14).to_numpy()

    rf = np.nansum(r ** 2, axis=1)                                   # 全日 RV(只用于 t−1 之后)
    rv_prev = pd.Series(np.log(np.where(rf > 0, rf, np.nan)), index=days).shift(1)
    rv5 = pd.Series(np.log(pd.Series(rf, index=days).rolling(5).mean().shift(1).where(lambda x: x > 0)), index=days)
    gap = (O / b.prev_close_adj.to_numpy() - 1)

    # ---- B 组:仅 bar 0..K-1
    r0, c0, h0, l0, v0 = r[:, :K], c[:, :K], h[:, :K], l[:, :K], v[:, :K]
    rv30 = np.sum(r0 ** 2, axis=1)
    path = np.sum(np.abs(r0), axis=1)
    ret30 = np.abs(np.log(c0[:, -1] / O))
    rng30 = (h0.max(1) - l0.min(1)) / O
    bv = (np.pi / 2) * np.sum(np.abs(r0[:, 1:]) * np.abs(r0[:, :-1]), axis=1)
    rv_k = np.sum(r0[:, 1:] ** 2, axis=1)
    jump = np.clip(1 - bv / np.where(rv_k > 0, rv_k, np.nan), 0, 1)
    up = np.sum(np.where(r0 > 0, r0 ** 2, 0), axis=1); dn = np.sum(np.where(r0 < 0, r0 ** 2, 0), axis=1)
    asym = np.abs(up - dn) / np.where(rv30 > 0, rv30, np.nan)
    a, bb = r0[:, 1:], r0[:, :-1]
    am, bm = a - a.mean(1, keepdims=True), bb - bb.mean(1, keepdims=True)
    ac1 = (am * bm).sum(1) / np.sqrt((am ** 2).sum(1) * (bm ** 2).sum(1))
    vw = b.vwap().to_numpy()[:, :K]
    sg = np.sign(c0 - vw)
    cross = np.sum((sg[:, 1:] * sg[:, :-1]) < 0, axis=1)
    ub, lb, _ = noise_area(b)
    out_band = np.mean((c0 > ub.to_numpy()[:, :K]) | (c0 < lb.to_numpy()[:, :K]), axis=1)
    rv_rel = rel_volume(b, 14, "cum")["10:00"].to_numpy()

    G = pd.read_parquet(_P.RAW / "squeeze_dix_gex.parquet").gex
    g = G.reindex(days).ffill().shift(1)
    gz = (g - g.rolling(252, min_periods=126).mean()) / g.rolling(252, min_periods=126).std()

    X = pd.DataFrame({
        "log_sd14": np.log(sd14), "log_vix": np.log(cboe_prev("VIX", days)), "log_vxn": np.log(cboe_prev("VXN", days)),
        "log_vix9d": np.log(cboe_prev("VIX9D", days)), "gex_z": gz, "gex_raw": g,
        "log_rv_prev": rv_prev, "log_rv5": rv5, "gap_abs": np.abs(gap),
        **{f"dow{d}": (days.dayofweek == d).astype(float) for d in (1, 2, 3, 4)},
        # B
        "b_log_rv30": np.log(np.where(rv30 > 0, rv30, np.nan)), "b_ret30": ret30, "b_range30_sd": rng30 / sd14,
        "b_rel_vol": np.log(rv_rel), "b_eff": ret30 / np.where(path > 0, path, np.nan), "b_jump": jump,
        "b_cross": cross.astype(float), "b_ac1": ac1, "b_asym": asym, "b_outband": out_band,
    }, index=days)
    A_COLS = ["log_sd14", "log_vix", "log_vxn", "log_vix9d", "gex_z", "log_rv_prev", "log_rv5", "gap_abs", "dow1", "dow2", "dow3", "dow4"]
    B_COLS = [c_ for c_ in X.columns if c_.startswith("b_")]

    li = b.last_idx.to_numpy()
    rv_rem = np.array([np.nansum(r[i, K:li[i] + 1] ** 2) for i in range(n)])
    disp = np.array([abs(np.log(c[i, li[i]] / c[i, K - 1])) for i in range(n)])
    Y = pd.DataFrame({"y_rv": np.log(np.where(rv_rem > 0, rv_rem, np.nan)), "disp": disp}, index=days)
    return X, Y, A_COLS, B_COLS


def poison_check(b, X0, A_COLS, B_COLS, ndays=20, seed=0):
    """把某一天 10:00 之后的 bar(及当日 day_close)×7 污染,该日特征行必须逐位不变。阳性对照:Y 必须变。"""
    rg = np.random.default_rng(seed)
    idx = rg.choice(np.arange(300, len(b.days) - 1), ndays, replace=False)
    worst, ydiff = 0.0, 0.0
    X0b, Y0 = X0, None
    for i in idx:
        d = b.days[i]
        def P(df):
            df = df.copy(); df.iloc[i, K:] = df.iloc[i, K:] * 7.0; return df
        dc = b.day_close.copy(); dc.iloc[i] = dc.iloc[i] * 7.0
        bp = dataclasses.replace(b, o=P(b.o), h=P(b.h), l=P(b.l), c=P(b.c), v=P(b.v), day_close=dc)
        Xp, Yp, _, _ = build(bp)
        a_ = X0.loc[d, A_COLS + B_COLS].to_numpy(float); b_ = Xp.loc[d, A_COLS + B_COLS].to_numpy(float)
        worst = max(worst, float(np.nanmax(np.abs(a_ - b_))))
        ydiff = max(ydiff, abs(float(Yp.loc[d, "y_rv"]) - float(_Y0.loc[d, "y_rv"])))
    return worst, ydiff


# ----------------------------------------------------------------- 模型
def fit_predict(Xtr, ytr, Xte, kind, model):
    from sklearn.linear_model import Ridge, LogisticRegression
    mu, sd = Xtr.mean(0), Xtr.std(0).replace(0, 1.0)
    Xa, Xb = ((Xtr - mu) / sd).to_numpy(), ((Xte - mu) / sd).to_numpy()
    if model == "ridge":
        if kind == "reg":
            return Ridge(alpha=RIDGE_A).fit(Xa, ytr).predict(Xb)
        return LogisticRegression(C=LOGIT_C, max_iter=500).fit(Xa, ytr).predict_proba(Xb)[:, 1]
    import lightgbm as lgb
    kw = dict(max_depth=2, n_estimators=150, learning_rate=0.05, min_child_samples=40, subsample=0.8, subsample_freq=1,
              random_state=0, verbose=-1, n_jobs=1)
    if kind == "reg":
        return lgb.LGBMRegressor(**kw).fit(Xa, ytr).predict(Xb)
    return lgb.LGBMClassifier(**kw).fit(Xa, ytr).predict_proba(Xb)[:, 1]


def tier_dummies(X, train_idx):
    q1, q2 = X.loc[train_idx, "gex_raw"].quantile([1 / 3, 2 / 3])
    return pd.DataFrame({"gex_mid": ((X.gex_raw > q1) & (X.gex_raw <= q2)).astype(float), "gex_hi": (X.gex_raw > q2).astype(float)}, index=X.index)


def walk_forward(X, Y, sets):
    """sets: {name: (cols, model)}。返回每个 (name, 'rv'/'low') 的 OOS 预测 DataFrame 与基线。"""
    pred = {}
    base_rv = pd.Series(np.nan, index=X.index); thr_s = pd.Series(np.nan, index=X.index)
    for yr in OOS_YEARS:
        te = X.index[X.index.year == yr]
        tr = X.index[X.index < f"{yr}-01-01"]
        ok_tr = tr[X.loc[tr].notna().all(1) & Y.loc[tr].notna().all(1)]
        thr = Y.loc[ok_tr, "disp"].quantile(LOW_Q)
        base_rv.loc[te] = Y.loc[ok_tr, "y_rv"].mean(); thr_s.loc[te] = thr
        ylow = (Y.loc[ok_tr, "disp"] <= thr).astype(int).to_numpy()
        for name, (cols, model) in sets.items():
            Xf = X[[c for c in cols if c != "TIER"]].copy()
            if "TIER" in cols:
                Xf = pd.concat([Xf, tier_dummies(X, ok_tr)], axis=1)
            ok_te = te[Xf.loc[te].notna().all(1)]
            for kind, key, yv in (("reg", "rv", Y.loc[ok_tr, "y_rv"].to_numpy()), ("clf", "low", ylow)):
                p = fit_predict(Xf.loc[ok_tr], yv, Xf.loc[ok_te], kind, model)
                pred.setdefault((name, key), pd.Series(np.nan, index=X.index)).loc[ok_te] = p
    return pred, base_rv, thr_s


def month_boot(idx, fn, nb=NBOOT, seed=0):
    """按月聚类自助:fn(行位置数组) -> 标量;返回 (点估计, 2.5%, 97.5%)。"""
    months = pd.Series(np.arange(len(idx)), index=idx).groupby([idx.year, idx.month]).apply(lambda s: s.to_numpy()).tolist()
    rg = np.random.default_rng(seed); out = []
    for _ in range(nb):
        pick = rg.integers(0, len(months), len(months))
        out.append(fn(np.concatenate([months[k] for k in pick])))
    return fn(np.arange(len(idx))), np.nanpercentile(out, 2.5), np.nanpercentile(out, 97.5)


# ----------------------------------------------------------------- 变现
def causal_s(yhat, kind, param):
    """yhat 越大 = 预期位移/波动越大。skip: 因果分位 ≤ q → 0;cont: clip(exp(γ(ŷ−m)),0.5,1.5),m=此前 OOS 预测中位数。"""
    v = yhat.to_numpy(float); s = np.ones(len(v)); hist = []
    for i, x in enumerate(v):
        if np.isfinite(x):
            if len(hist) >= 60:
                if kind == "skip":
                    rank = np.mean(np.array(hist) < x)
                    s[i] = 0.0 if rank <= param else 1.0
                else:
                    s[i] = float(np.clip(np.exp(param * (x - np.median(hist))), 0.5, 1.5))
            hist.append(x)
    return pd.Series(s, index=yhat.index)


def period(x, k):
    if k == "全OOS":
        return x.loc["2021":"2026"]
    a, b_ = (DEV_OOS if k == "dev-OOS" else TEST)
    return x.loc[a:b_]


def main():
    b = load_bars("QQQ"); days = b.days
    X, Y, A_COLS, B_COLS = build(b)
    global _Y0; _Y0 = Y
    print(f"QQQ {len(days)} 天;特征 A={len(A_COLS)} B={len(B_COLS)};决策点=10:00(bar {K});OOS 年 {list(OOS_YEARS)[0]}–{list(OOS_YEARS)[-1]}")

    # ---- 未来污染自检
    worst, ydiff = poison_check(b, X, A_COLS, B_COLS)
    print(f"[污染自检] 20 天 10:00 后价格/量×7:特征最大差 {worst:.2e}(须为 0);阳性对照 Y_rv 最大差 {ydiff:.2f}(须>0)")
    assert worst == 0.0 and ydiff > 0.5, "未来污染自检失败"

    sets = {
        "A·ridge": (A_COLS, "ridge"), "A+B·ridge": (A_COLS + B_COLS, "ridge"), "B·ridge": (B_COLS, "ridge"),
        "A分位GEX·ridge": ([c for c in A_COLS if c not in ("gex_z",)] + ["TIER"], "ridge"),
        "A无GEX·ridge": ([c for c in A_COLS if c != "gex_z"], "ridge"),
        "A·lgbm": (A_COLS, "lgbm"), "A+B·lgbm": (A_COLS + B_COLS, "lgbm"),
    }
    pred, base_rv, thr_s = walk_forward(X, Y, sets)
    oos = X.index[(X.index.year >= 2021) & X.notna().all(1) & Y.notna().all(1)]
    ylow_oos = (Y.loc[oos, "disp"] <= thr_s.loc[oos]).astype(int)
    print(f"OOS 天数 {len(oos)};低波标签占比 {ylow_oos.mean():.3f}(目标≈{LOW_Q})")

    # ================= 1) 预测力
    rows = []
    sse_b = ((Y.loc[oos, "y_rv"] - base_rv.loc[oos]) ** 2).to_numpy()
    for (name, key), p in pred.items():
        p = p.loc[oos]
        if key == "rv":
            e = ((Y.loc[oos, "y_rv"] - p) ** 2).to_numpy()
            f = lambda ii, e=e: 1 - e[ii].sum() / sse_b[ii].sum()
            pt, lo, hi = month_boot(oos, f)
            rows.append(dict(模型=name, 目标="log RV(10:00→收盘)", 指标="OOS R²", 值=pt, lo=lo, hi=hi))
        else:
            yl = ylow_oos.to_numpy(); pp = p.to_numpy()
            f = lambda ii, yl=yl, pp=pp: auc(yl[ii], -pp[ii])           # 预测 P(低波) 越小 → 越不可能低波;AUC 衡量区分「高位移」
            fa = lambda ii, yl=yl, pp=pp: auc(yl[ii], pp[ii])
            pt, lo, hi = month_boot(oos, fa)
            rows.append(dict(模型=name, 目标=f"低波(下{int(LOW_Q*100)}%位移)", 指标="AUC", 值=pt, lo=lo, hi=hi))
    R1 = pd.DataFrame(rows)
    # 增量:A+B vs A
    inc = []
    for (n1, n0) in (("A+B·ridge", "A·ridge"), ("A+B·lgbm", "A·lgbm"), ("A·ridge", "A分位GEX·ridge"), ("A·ridge", "A无GEX·ridge")):
        e1 = ((Y.loc[oos, "y_rv"] - pred[(n1, "rv")].loc[oos]) ** 2).to_numpy(); e0 = ((Y.loc[oos, "y_rv"] - pred[(n0, "rv")].loc[oos]) ** 2).to_numpy()
        pt, lo, hi = month_boot(oos, lambda ii: (e0[ii].sum() - e1[ii].sum()) / sse_b[ii].sum())
        inc.append(dict(比较=f"{n1} − {n0}", 目标="log RV", 指标="ΔR²", 值=pt, lo=lo, hi=hi))
        yl = ylow_oos.to_numpy(); p1, p0 = pred[(n1, "low")].loc[oos].to_numpy(), pred[(n0, "low")].loc[oos].to_numpy()
        pt, lo, hi = month_boot(oos, lambda ii: auc(yl[ii], p1[ii]) - auc(yl[ii], p0[ii]))
        inc.append(dict(比较=f"{n1} − {n0}", 目标="低波标签", 指标="ΔAUC", 值=pt, lo=lo, hi=hi))
    R1i = pd.DataFrame(inc)
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 20)
    print("\n" + "=" * 110 + "\n1) 预测力(OOS 2021–2026;区间=按月聚类自助 95%)\n" + "=" * 110)
    print(R1.round(4).to_string(index=False)); print("\n增量:"); print(R1i.round(4).to_string(index=False))
    R1.to_csv(TAB / "lowvol_prediction.csv", index=False); R1i.to_csv(TAB / "lowvol_prediction_incremental.csv", index=False)

    # ================= 2) 事后低波 vs 高波日的开盘 30 分钟形态(描述性,全样本)
    ok = X.index[X.notna().all(1) & Y.notna().all(1)]
    lo_th, hi_th = Y.loc[ok, "disp"].quantile([0.4, 0.6])
    low = Y.loc[ok, "disp"] <= lo_th; high = Y.loc[ok, "disp"] >= Y.loc[ok, "disp"].quantile(0.6)
    sel = ok[(low | high).to_numpy()]
    yy = low.loc[sel].astype(int).to_numpy()
    rows = []
    for col in A_COLS[:9] + B_COLS:
        xv = X.loc[sel, col].to_numpy(float)
        f = lambda ii, xv=xv, yy=yy: auc(1 - yy[ii], xv[ii])            # AUC>0.5 = 该特征越大越像高位移日
        pt, lo_, hi_ = month_boot(pd.DatetimeIndex(sel), f, nb=1000)
        rows.append(dict(特征=col, 组="B(≤10:00)" if col.startswith("b_") else "A(事前)", AUC_高位移=pt, lo=lo_, hi=hi_,
                         低位移日中位数=np.nanmedian(xv[yy == 1]), 高位移日中位数=np.nanmedian(xv[yy == 0])))
    R2 = pd.DataFrame(rows).sort_values("AUC_高位移", key=lambda s: -(s - 0.5).abs())
    print("\n" + "=" * 110 + "\n2) 事后『低位移 Q1/Q2(下40%)』vs『高位移(上40%)』的单变量区分力(AUC,0.5=无区分;全样本描述性)\n" + "=" * 110)
    print(R2.round(4).to_string(index=False)); R2.to_csv(TAB / "lowvol_feature_auc.csv", index=False)

    # ================= 3) 变现:Sleeve 日级缩放
    cur = pd.read_pickle(OUT / "strategy_comparison_curves.pkl")[T.VER]; cur.index = pd.to_datetime(cur.index)
    ret = pd.read_pickle(OUT / "sleeve_anatomy_curve.pkl"); ret.index = pd.to_datetime(ret.index)
    ret = ret.reindex(days).fillna(0.0)
    assert np.allclose(ret.reindex(cur.index).fillna(0).to_numpy(), cur.fillna(0).to_numpy(), atol=1e-12), "Sleeve 曲线与 combo 口径不一致"
    trade_day = pd.Series(days.dayofweek != 0, index=days)
    bh = pd.read_pickle(OUT / "strategy_comparison_bh.pkl").dropna(); bh.index = pd.to_datetime(bh.index)
    px_all = C.load_close("yahoo/QQQ.parquet"); p_all = pct_w(px_all.pct_change().rolling(21).std(), 504)
    spec = (("lin", 0.5, 1.5), None, 1.0)

    def port(sleeve):
        s_ = sleeve.reindex(bh.index).fillna(0.0)
        return T.build(bh, s_, (1.0, 1.0), spec, 1.0, px_all, p_all)[0]
    base_p = port(ret)

    cfgs = {}
    for mname, key in (("A·ridge", "rv"), ("A+B·ridge", "rv"), ("A·ridge", "low"), ("A+B·ridge", "low")):
        yh = pred[(mname, key)]
        yh = np.log(yh / (1 - yh)) * -1 if key == "low" else yh                    # 越大 = 越可能高位移/高波
        for kind, par in (("skip", 0.2), ("skip", 0.3), ("cont", 0.5), ("cont", 1.0)):
            cfgs[f"{mname}[{key}] {kind}{par:g}"] = causal_s(yh, kind, par)
    gz = X.gex_z
    cfgs["GEX连续 λ=0.3(对照)"] = pd.Series(np.clip(1 - 0.3 * gz.fillna(0), 0.5, 1.5), index=days)
    cfgs["GEX分位 高档→0.5(对照)"] = pd.Series(np.where(X.gex_raw > X.loc[:"2022", "gex_raw"].quantile(2 / 3), 0.5, 1.0), index=days)

    # 置换:仅在 OOS 的可交易日内重排 s(同一重排用于所有配置 → 取 max 得到族内 p)
    tr_idx = np.where(trade_day.to_numpy() & (days.year >= 2021))[0]
    rg = np.random.default_rng(1)
    perms = [rg.permutation(tr_idx) for _ in range(NPERM)]
    base_sr_all = sr(ret.iloc[tr_idx])
    rows, null_max = [], np.full(NPERM, -np.inf)
    null_by = {}
    for nm, s in cfgs.items():
        sv = s.to_numpy(); rv_ = ret.to_numpy()
        obs = sr(sv[tr_idx] * rv_[tr_idx]) - base_sr_all
        nul = np.array([sr(sv[pi] * rv_[tr_idx]) for pi in perms]) - base_sr_all
        null_by[nm] = nul; null_max = np.maximum(null_max, nul)
        xs, xp = ret * s, port(ret * s)
        row = dict(配置=nm, 平均s=float(s.loc[days[tr_idx]].mean()), 跳过比例=float((s.loc[days[tr_idx]] == 0).mean()))
        for k in ("dev-OOS", "test", "全OOS"):
            a0, a1 = period(ret[trade_day], k), period(xs[trade_day], k)
            row[f"Sleeve ΔSR {k}"] = sr(a1) - sr(a0)
            pb, pa = period(base_p, k), period(xp, k)
            row[f"组合 ΔSR {k}"] = sr(pa) - sr(pb)
        row["置换p(单配置)"] = float(np.mean(null_by[nm] >= (sr(sv[tr_idx] * rv_[tr_idx]) - base_sr_all)))
        row["_obs"] = obs
        rows.append(row)
    R3 = pd.DataFrame(rows)
    R3["置换p(族内max)"] = [float(np.mean(null_max >= o)) for o in R3._obs]
    R3 = R3.drop(columns="_obs")
    print("\n" + "=" * 140 + f"\n3) Sleeve 日级缩放(30 相位均值曲线;OOS 可交易日;|ΔSR|<{T.GUARD} 按噪声;置换 {NPERM} 次)\n" + "=" * 140)
    print(f"基线 Sleeve SR: dev-OOS {sr(period(ret[trade_day],'dev-OOS')):.3f}  test {sr(period(ret[trade_day],'test')):.3f}  全OOS {sr(period(ret[trade_day],'全OOS')):.3f}"
          f" | 组合(RV[0.5,1.5]+Sleeve)全OOS {sr(period(base_p,'全OOS')):.3f}")
    print(R3.round(3).to_string(index=False))
    R3.to_csv(TAB / "lowvol_economic.csv", index=False)
    pd.to_pickle(dict(pred=pred, X=X, Y=Y, thr=thr_s, cfgs=cfgs), OUT / "lowvol_study.pkl")
    return R1, R1i, R2, R3


if __name__ == "__main__":
    main()
