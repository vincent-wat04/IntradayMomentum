"""日内回转 v2 —— 严格按 docs/PROPOSAL_intraday_reversion_v2.md 执行。

决策:第 k 根 1 分钟 bar 收盘(k=34,39,…);成交:下一根 bar 开盘;持有 h 分钟后在 bar 开盘价平仓;不跨日;同一时刻一个仓位。
特征仅用 ≤k 的 bar 与 t−1 的日度数据。Walk-forward:每年初重估,OOS 2021–2026。
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
import lowvol_avoidance_study as LV
from etft.intraday.bars import load_bars
from etft.intraday.strategy import rel_volume

OUT = LV.OUT; TAB = LV.TAB
KS = np.arange(34, 329, 5)                      # 决策 bar(0 起),首次 10:05
HS = (15, 30, 60)
TAUS = (1.5, 2.5, 4.0)                          # bp
COSTS = (0.25, 0.5, 1.0, 2.0)                   # 每边 bp
OOS0 = 2021
NPERM = 2000
FEATS = ["z5", "z15", "z30", "z_open", "vwap_dev", "pctb30", "pos_range", "dist_hi", "dist_lo",
         "clv30", "netvol15", "rv_ratio15", "range15", "relvol", "tod", "gap", "log_sd14", "log_vix", "gex_z"]


def sr(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return x.mean() / x.std() * np.sqrt(252) if len(x) > 30 and x.std() > 0 else np.nan


# ------------------------------------------------------------------ 特征
def build_panel(b, Xd, hs=HS):
    """返回长表(day, k, 特征…, y_h…)。仅整日(390 bar)交易日。"""
    days = b.days; n = len(days)
    c, o, hi, lo, v = (b.c.to_numpy(), b.o.to_numpy(), b.h.to_numpy(), b.l.to_numpy(), b.v.to_numpy())
    O = b.day_open.to_numpy()
    full = (b.last_idx.to_numpy() >= 389)
    sd = np.exp(Xd.log_sd14.to_numpy())
    vw = b.vwap().to_numpy()
    rv = rel_volume(b, 14, "cum").to_numpy()
    r = np.full_like(c, np.nan); r[:, 1:] = np.log(c[:, 1:] / c[:, :-1]); r[:, 0] = np.log(c[:, 0] / O)
    hh = np.maximum.accumulate(np.where(np.isfinite(hi), hi, -np.inf), axis=1)
    ll = np.minimum.accumulate(np.where(np.isfinite(lo), lo, np.inf), axis=1)
    clv = np.where(hi > lo, ((c - lo) - (hi - c)) / (hi - lo), 0.0)
    sgn = np.sign(c - o)
    gap = np.log(O / b.prev_close_adj.to_numpy()) / sd
    rows = []
    for k in KS:
        w30 = c[:, k - 29:k + 1]
        rv15 = np.sum(r[:, k - 14:k + 1] ** 2, axis=1); rvd = np.sum(r[:, :k + 1] ** 2, axis=1)
        v30 = v[:, k - 29:k + 1]; v15 = v[:, k - 14:k + 1]
        d = pd.DataFrame({
            "z5": np.log(c[:, k] / c[:, k - 5]) / (sd * np.sqrt(5 / 390)),
            "z15": np.log(c[:, k] / c[:, k - 15]) / (sd * np.sqrt(15 / 390)),
            "z30": np.log(c[:, k] / c[:, k - 30]) / (sd * np.sqrt(30 / 390)),
            "z_open": np.log(c[:, k] / O) / (sd * np.sqrt((k + 1) / 390)),
            "vwap_dev": np.log(c[:, k] / vw[:, k]) / (sd * np.sqrt(30 / 390)),
            "pctb30": (c[:, k] - w30.mean(1)) / (2 * np.where(w30.std(1) > 0, w30.std(1), np.nan)),
            "pos_range": (c[:, k] - ll[:, k]) / np.where(hh[:, k] > ll[:, k], hh[:, k] - ll[:, k], np.nan),
            "dist_hi": np.log(hh[:, k] / c[:, k]) / sd, "dist_lo": np.log(c[:, k] / ll[:, k]) / sd,
            "clv30": np.sum(clv[:, k - 29:k + 1] * v30, axis=1) / np.sum(v30, axis=1),
            "netvol15": np.sum(sgn[:, k - 14:k + 1] * v15, axis=1) / np.sum(v15, axis=1),
            "rv_ratio15": np.log(np.where(rv15 > 0, rv15, np.nan) / np.where(rvd > 0, rvd * 15 / (k + 1), np.nan)),
            "range15": (hi[:, k - 14:k + 1].max(1) - lo[:, k - 14:k + 1].min(1)) / O / (sd * np.sqrt(15 / 390)),
            "relvol": np.log(rv[:, k]), "tod": k / 390.0, "gap": gap,
            "log_sd14": Xd.log_sd14.to_numpy(), "log_vix": Xd.log_vix.to_numpy(), "gex_z": Xd.gex_z.to_numpy(),
        })
        d.insert(0, "k", k); d.insert(0, "day", days); d["di"] = np.arange(n); d["sd"] = sd
        d["px0"] = o[:, k + 1]
        for h in hs:
            d[f"y{h}"] = np.log(o[:, k + 1 + h] / o[:, k + 1]) / (sd * np.sqrt(h / 390)) if k + 1 + h <= 389 else np.nan
        d["fut5"] = np.log(o[:, k + 6] / o[:, k + 1])
        rows.append(d[full])
    P = pd.concat(rows, ignore_index=True)
    return P.replace([np.inf, -np.inf], np.nan)


def poison_check(b, Xd, ndays=12, seed=0, kmax=195):
    P0 = build_panel(b, Xd)
    rg = np.random.default_rng(seed)
    full = np.where(b.last_idx.to_numpy() >= 389)[0]
    idx = rg.choice(full[full > 300], ndays, replace=False)
    worst = 0.0
    for i in idx:
        def Pz(df):
            df = df.copy(); df.iloc[i, kmax + 1:] = df.iloc[i, kmax + 1:] * 7.0; return df
        bp = dataclasses.replace(b, o=Pz(b.o), h=Pz(b.h), l=Pz(b.l), c=Pz(b.c), v=Pz(b.v))
        Pp = build_panel(bp, Xd)
        a = P0[(P0.di == i) & (P0.k <= kmax)].set_index("k")[FEATS].sort_index()
        c_ = Pp[(Pp.di == i) & (Pp.k <= kmax)].set_index("k")[FEATS].sort_index()
        worst = max(worst, float(np.nanmax(np.abs(a.to_numpy(float) - c_.to_numpy(float)))))
    return worst


# ------------------------------------------------------------------ 模型
def walk_forward(P):
    from sklearn.linear_model import Ridge
    import lightgbm as lgb
    out = {}
    yrs = P.day.dt.year.to_numpy()
    for h in HS:
        y = P[f"y{h}"].to_numpy()
        for mname in ("ridge", "lgbm"):
            pr = np.full(len(P), np.nan)
            for yr in range(OOS0, 2027):
                tr = (yrs < yr) & np.isfinite(y) & P[FEATS].notna().all(1).to_numpy()
                # 训练样本的标签不能跨入测试年(同日内,故自然满足);去掉 yr 之前最后一天的任何重叠 — 日内标签不跨日
                te = (yrs == yr) & P[FEATS].notna().all(1).to_numpy()
                if tr.sum() < 5000 or te.sum() == 0:
                    continue
                Xtr, Xte = P.loc[tr, FEATS], P.loc[te, FEATS]
                mu, sd = Xtr.mean(), Xtr.std().replace(0, 1.0)
                A, B = ((Xtr - mu) / sd).to_numpy(), ((Xte - mu) / sd).to_numpy()
                yt = np.clip(y[tr], -6, 6)                                         # 截尾仅用于拟合
                if mname == "ridge":
                    m = Ridge(alpha=100.0).fit(A, yt)
                else:
                    m = lgb.LGBMRegressor(max_depth=3, n_estimators=200, learning_rate=0.03, min_child_samples=500,
                                          subsample=0.7, subsample_freq=1, colsample_bytree=0.8, random_state=0, verbose=-1, n_jobs=1).fit(A, yt)
                pr[te] = m.predict(B)
            out[(mname, h)] = pr
    return out


# ------------------------------------------------------------------ 交易模拟
def trades_from_signal(P, sig_bp, h, tau):
    """sig_bp: 每行预测收益(bp,带符号)。逐日按 k 升序,|sig|≥τ 入场,持仓 h 分钟,期间不再入场。"""
    ret = P[f"y{h}"].to_numpy() * P["sd"].to_numpy() * np.sqrt(h / 390) * 1e4        # 未来 h 分钟对数收益(bp)
    a = pd.DataFrame({"di": P.di.to_numpy(), "k": P.k.to_numpy(), "sig": sig_bp, "ret": ret, "day": P.day.to_numpy()})
    a = a[np.isfinite(a.sig) & np.isfinite(a.ret) & (a.sig.abs() >= tau)]
    res = []
    for di, g in a.groupby("di", sort=False):
        free = -1
        for k, s_, r_, d_ in zip(g.k.to_numpy(), g.sig.to_numpy(), g.ret.to_numpy(), g.day.to_numpy()):
            if k >= free:
                res.append((di, d_, k, 1 if s_ > 0 else -1, (1 if s_ > 0 else -1) * r_)); free = k + 1 + h
    return pd.DataFrame(res, columns=["di", "day", "k", "side", "gross"])


def summarize(T, ndays_all, label, rows, per=None):
    """per: dict 名称→(开始年,结束年)"""
    per = per or {"dev-OOS": (2021, 2022), "test": (2023, 2026), "全OOS": (2021, 2026)}
    for pk, (a, b_) in per.items():
        t = T[(T.day.dt.year >= a) & (T.day.dt.year <= b_)]
        nd = ndays_all[pk]
        row = dict(配置=label, 期间=pk, 笔数=len(t), 每日笔数=len(t) / max(nd, 1),
                   毛bp_每笔=t.gross.mean() if len(t) else np.nan, 胜率=(t.gross > 0).mean() if len(t) else np.nan,
                   多侧毛bp=t[t.side > 0].gross.mean() if (t.side > 0).any() else np.nan,
                   空侧毛bp=t[t.side < 0].gross.mean() if (t.side < 0).any() else np.nan,
                   盈亏平衡成本_每边bp=t.gross.mean() / 2 if len(t) else np.nan)
        dly = t.groupby("day").gross.sum()
        for cst in COSTS:
            net = t.assign(net=t.gross - 2 * cst).groupby("day").net.sum()
            full_idx = ndays_all["_idx"][pk]
            x = net.reindex(full_idx).fillna(0.0).to_numpy()
            row[f"净SR@{cst:g}bp"] = sr(x); row[f"年化净bp@{cst:g}"] = x.mean() * 252
        rows.append(row)


def main():
    b = load_bars("QQQ"); days = b.days
    Xd, Yd, _, _ = LV.build(b)
    print("构建面板…"); P = build_panel(b, Xd)
    print(f"面板 {len(P):,} 行,{P.di.nunique()} 个整日;决策网格 {len(KS)} 个/日")
    w = poison_check(b, Xd)
    print(f"[污染自检] 12 个整日 bar>{196}×7:k≤195 的全部特征最大差 {w:.2e}(须为 0)")
    assert w == 0.0
    ok_rows = P[FEATS].notna().all(1)
    oos_mask = (P.day.dt.year >= OOS0)
    pred = walk_forward(P)

    # --- IC(Spearman)与阳性对照
    from scipy.stats import spearmanr
    print("\n" + "=" * 100 + "\n1) OOS IC(Spearman,预测 vs 未来 h 分钟标准化收益)\n" + "=" * 100)
    ic_rows = []
    for (m, h), pr in pred.items():
        y = P[f"y{h}"].to_numpy(); ok = np.isfinite(pr) & np.isfinite(y)
        row = dict(模型=m, h=h, IC_全=spearmanr(pr[ok], y[ok])[0])
        for yr in range(OOS0, 2027):
            mk = ok & (P.day.dt.year.to_numpy() == yr)
            row[str(yr)] = spearmanr(pr[mk], y[mk])[0] if mk.sum() > 100 else np.nan
        for nm, cond in (("k<14:00", P.k.to_numpy() < 270), ("k≥14:00", P.k.to_numpy() >= 270)):
            mk = ok & cond; row[f"IC {nm}"] = spearmanr(pr[mk], y[mk])[0]
        ic_rows.append(row)
    IC = pd.DataFrame(ic_rows); print(IC.round(4).to_string(index=False)); IC.to_csv(TAB / "intraday_rev_v2_ic.csv", index=False)
    # 阳性对照:把 fut5(未来 5 分钟收益)当特征,IC 必须暴增
    from sklearn.linear_model import Ridge
    ok = P[FEATS + ["fut5", "y30"]].notna().all(1).to_numpy(); tr = ok & (P.day.dt.year.to_numpy() < 2023); te = ok & (P.day.dt.year.to_numpy() >= 2023)
    cols = FEATS + ["fut5"]; mu, sd = P.loc[tr, cols].mean(), P.loc[tr, cols].std()
    m = Ridge(alpha=100).fit(((P.loc[tr, cols] - mu) / sd).to_numpy(), np.clip(P.loc[tr, "y30"], -6, 6))
    icf = spearmanr(m.predict(((P.loc[te, cols] - mu) / sd).to_numpy()), P.loc[te, "y30"])[0]
    print(f"[阳性对照] 加入未来 5 分钟收益作特征:IC={icf:.3f}(须 ≫ 诚实模型 IC)")

    # --- 交易
    di_all = {}
    for pk, (a, b_) in {"dev-OOS": (2021, 2022), "test": (2023, 2026), "全OOS": (2021, 2026)}.items():
        idx = pd.DatetimeIndex([d for d in days if a <= d.year <= b_ and (b.last_idx.loc[d] >= 389)])
        di_all[pk] = len(idx); di_all.setdefault("_idx", {})[pk] = idx
    rows, trades = [], {}
    for (m, h), pr in pred.items():
        sd_ = P["sd"].to_numpy() * np.sqrt(h / 390) * 1e4
        sig_bp = pr * sd_
        for tau in TAUS:
            T_ = trades_from_signal(P, sig_bp, h, tau); trades[(m, h, tau)] = T_
            summarize(T_, di_all, f"{m} h={h} τ={tau:g}bp", rows)
    # 规则对照
    rule_defs = {"VWAP偏离": "vwap_dev", "布林%B": "pctb30", "z30": "z30"}
    for rn, col in rule_defs.items():
        x = P[col].to_numpy(); thr = 2.0 if col != "pctb30" else 1.0
        for h in HS:
            for direction, sgn_ in (("回转", -1), ("延续", 1)):
                sig = np.where(np.abs(x) >= thr, sgn_ * np.sign(x) * 100.0, 0.0)
                T_ = trades_from_signal(P, sig, h, 50.0); trades[(rn + direction, h, 0)] = T_
                summarize(T_, di_all, f"规则 {rn}{direction} h={h}", rows)
    R = pd.DataFrame(rows)
    R.to_csv(TAB / "intraday_rev_v2_trades.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    cols = ["配置", "笔数", "每日笔数", "毛bp_每笔", "胜率", "多侧毛bp", "空侧毛bp", "盈亏平衡成本_每边bp", "净SR@0.25bp", "净SR@1bp", "净SR@2bp"]
    for pk in ("dev-OOS", "test"):
        print("\n" + "=" * 150 + f"\n2) 交易结果 — {pk}(叠加仓 1× 名义;SR 含无交易日;毛 bp/笔 须 ≥ 2×往返成本才满足 §5(i))\n" + "=" * 150)
        print(R[R.期间 == pk][cols].round(3).to_string(index=False))

    # --- 翻转安慰剂(ML 配置;同一日级符号用于所有配置;族内 max 以 t 统计量)
    rg = np.random.default_rng(1)
    all_days = pd.DatetimeIndex(sorted(P.day.unique())); dpos = {d: i for i, d in enumerate(all_days)}
    U = rg.choice([-1.0, 1.0], size=(NPERM, len(all_days)))
    fam, actual = [], {}
    for key, T_ in trades.items():
        if key[0] not in ("ridge", "lgbm"):
            continue
        t = T_[T_.day.dt.year >= OOS0]
        if len(t) < 200:
            continue
        ii = np.array([dpos[d] for d in t.day]); g = t.gross.to_numpy()
        z = lambda a_: a_.mean() / (a_.std() / np.sqrt(len(a_)))
        nulls = np.array([z(U[p][ii] * g) for p in range(NPERM)])
        actual[key] = (z(g), nulls); fam.append(nulls)
    famax = np.max(np.vstack(fam), axis=0) if fam else np.array([])
    print("\n" + "=" * 100 + "\n3) 翻转安慰剂(全 OOS;统计量 = 毛收益/笔的 t;族内 max 置换 p)\n" + "=" * 100)
    prow = []
    for key, (zt, nul) in actual.items():
        prow.append(dict(模型=key[0], h=key[1], τ=key[2], t统计=zt, p_单配置=float(np.mean(nul >= zt)), p_族内max=float(np.mean(famax >= zt))))
    PR = pd.DataFrame(prow); print(PR.round(3).to_string(index=False)); PR.to_csv(TAB / "intraday_rev_v2_placebo.csv", index=False)
    pd.to_pickle(dict(pred=pred, trades=trades), OUT / "intraday_rev_v2.pkl")
    return IC, R, PR


if __name__ == "__main__":
    main()
