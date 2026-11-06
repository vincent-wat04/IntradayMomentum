"""L0 Regime 第三轮 —— docs/PROPOSAL_regime_layer.md「第三轮」:二值子信号库(开源 14 美股类比)+ 行业龙头/落后 + EW/DYN 赋权。
python3 regime_round3_study.py"""
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
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import regime_features as RF
import regime_round2_study as R2
import regime_timing_study as R1
import trend_rv_sleeve_study as T

TAB, FULL, DEV, TEST, SUBP = R1.TAB, R1.FULL, R1.DEV, R1.TEST, R1.SUBP
OUTD = Path("outputs/full/intraday")
SHORT, LONG = 20, 120
MIN_ALL, MIN_GRP = 6, 2
GROUP_OF = {}
GROUPS3 = {
    "MACRO": ["tnx", "curve", "lqd_ief", "hyg_ief", "dxy", "usdjpy", "cu_au", "oil"],
    "SENT": ["vix", "vix_term", "skew"],
    "LEAD": ["rsp_spy", "xlk_spy", "smh_qqq", "cyc_def", "breadth", "xlk_rank", "spread", "def_top3"],
    "VALINF": ["tip_ief", "ecy"],
    "TECH": ["qqq_mom", "qqq_hiamp"],
}
for g, m in GROUPS3.items():
    for k in m: GROUP_OF[k] = g
ALL = [k for m in GROUPS3.values() for k in m]


def load_cboe(names=("VIX", "VIX3M", "SKEW")):
    out = {}
    for n in names:
        s = pd.read_parquet(RF.RAW / "cboe" / f"{n}.parquet")["close"]; s.index = pd.to_datetime(s.index).tz_localize(None)
        out[n] = s[~s.index.duplicated()]
    return out


def load_qqq_ohlc():
    q = pd.read_parquet(RF.RAW / "yahoo" / "QQQ.parquet"); q.index = pd.to_datetime(q.index).tz_localize(None)
    return q


def levels(P, C, Q, cal):
    """返回 {name: (x, sign)};ecy 特殊(见 binary())。全部只用 ≤t 的数据。"""
    Qp = P.reindex(cal).ffill(); lg = np.log; sec = Qp[RF.SECT]
    cb = {k: v.reindex(cal).ffill() for k, v in C.items()}
    L = {}
    L["tnx"] = (Qp["^TNX"], -1); L["curve"] = (Qp["^TNX"] - Qp["^IRX"], +1)
    L["lqd_ief"] = (lg(Qp["LQD"] / Qp["IEF"]), +1); L["hyg_ief"] = (lg(Qp["HYG"] / Qp["IEF"]), +1)
    L["dxy"] = (lg(Qp["DX-Y.NYB"]), -1); L["usdjpy"] = (lg(Qp["JPY=X"]), +1)
    L["cu_au"] = (lg(Qp["HG=F"] / Qp["GC=F"]), +1); L["oil"] = (lg(Qp["CL=F"]), -1)
    L["vix"] = (lg(cb["VIX"]), -1); L["vix_term"] = (lg(cb["VIX3M"] / cb["VIX"]), +1); L["skew"] = (cb["SKEW"], -1)
    L["rsp_spy"] = (lg(Qp["RSP"] / Qp["SPY"]), +1); L["xlk_spy"] = (lg(Qp["XLK"] / Qp["SPY"]), +1); L["smh_qqq"] = (lg(Qp["SMH"] / Qp["QQQ"]), +1)
    cyc_def = (Qp[RF.CYC].pct_change().mean(axis=1) - Qp[RF.DEF].pct_change().mean(axis=1)).fillna(0).add(1).cumprod()
    L["cyc_def"] = (lg(cyc_def), +1)
    ma50 = sec.rolling(50).mean(); L["breadth"] = ((sec > ma50).where(ma50.notna()).mean(axis=1), +1)
    r63 = lg(sec).diff(63); okr = r63.notna().all(axis=1)
    L["xlk_rank"] = (r63.rank(axis=1, pct=True)["XLK"].where(okr), +1)
    srt = np.sort(r63.to_numpy(), axis=1); L["spread"] = (pd.Series(srt[:, -3:].mean(1) - srt[:, :3].mean(1), index=cal).where(okr), -1)
    rk = r63.rank(axis=1, ascending=False); L["def_top3"] = ((rk[RF.DEF] <= 3).sum(axis=1).astype(float).where(okr), -1)
    L["tip_ief"] = (lg(Qp["TIP"] / Qp["IEF"]), -1)
    px = Q["close"].reindex(cal).ffill(); L["qqq_mom"] = (lg(px), +1)
    ret = np.log(px).diff(); amp = ((Q["high"] - Q["low"]) / Q["close_raw"]).reindex(cal)
    hi = amp > amp.rolling(60, min_periods=30).median().shift(1)               # 振幅中位数只用到 t−1(因果)
    L["qqq_hiamp"] = (ret.where(hi, 0.0).fillna(0.0).cumsum().where(amp.notna()), +1)
    return L, Qp


def binary(L, P, cal):
    S = pd.DataFrame(index=cal)
    for k, (x, sg) in L.items():
        ma_s, ma_l = x.rolling(SHORT, min_periods=SHORT).mean(), x.rolling(LONG, min_periods=LONG).mean()
        S[k] = ((sg * (ma_s - ma_l)) > 0).astype(float).where(ma_l.notna() & ma_s.notna())
    ecy = RF.ecy_series(P, cal); med = ecy.rolling(1260, min_periods=630).median()
    S["ecy"] = (ecy > med).astype(float).where(med.notna() & ecy.notna())
    return S[ALL]


def weekly(S, cal):
    s = pd.Series(cal, index=cal); last = s.groupby(cal.to_period("W")).transform("max") == s
    return S.where(pd.DataFrame(np.repeat(last.to_numpy()[:, None], S.shape[1], axis=1), index=S.index, columns=S.columns)).ffill(), last


def dyn_weights(Sw, r, last, win=120):
    """每周最后交易日:过去 win 日最小化 ‖|R| − Σ w s R‖,Σw=1,0.5/n ≤ w ≤ 1.5/n。返回 A_dyn(周更,ffill)与各周权重表。"""
    s_eff = 2 * Sw.shift(1) - 1                                                   # 当日实际持有的信号(t−1 收盘形成)
    idx = Sw.index; A = pd.Series(np.nan, index=idx); W = {}
    rv = r.reindex(idx).fillna(0.0).to_numpy()
    fr = np.flatnonzero(last.to_numpy())
    for i in fr:
        if i < win + LONG: continue
        cols = [c for c in Sw.columns if Sw[c].iloc[i] == Sw[c].iloc[i] and s_eff[c].iloc[i - win + 1:i + 1].notna().all()]
        n = len(cols)
        if n < MIN_ALL: continue
        X = s_eff[cols].iloc[i - win + 1:i + 1].to_numpy() * rv[i - win + 1:i + 1, None] * 100.0
        y = np.abs(rv[i - win + 1:i + 1]) * 100.0
        H, g = X.T @ X, X.T @ y
        w0 = np.full(n, 1.0 / n)
        res = minimize(lambda w: 0.5 * w @ H @ w - g @ w, w0, jac=lambda w: H @ w - g, method="SLSQP",
                       bounds=[(0.5 / n, 1.5 / n)] * n, constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: np.ones_like(w)}],
                       options={"maxiter": 200, "ftol": 1e-12})
        w = res.x if res.success else w0
        W[idx[i]] = pd.Series(w, index=cols)
        A.iloc[i] = float(np.dot(w, Sw[cols].iloc[i].to_numpy()))
    return A.ffill(), pd.DataFrame(W).T


def build(P, C, Q):
    cal = P["QQQ"].dropna().index
    L, _ = levels(P, C, Q, cal); S = binary(L, P, cal); Sw, last = weekly(S, cal)
    px = P["QQQ"].dropna(); r = px.pct_change().fillna(0.0)
    cnt = Sw.notna().sum(axis=1)
    A = {"ALL_EW": Sw.mean(axis=1, skipna=True).where(cnt >= MIN_ALL)}
    for g, m in GROUPS3.items():
        c = Sw[m].notna().sum(axis=1); A[f"{g}_EW"] = Sw[m].mean(axis=1, skipna=True).where(c >= MIN_GRP)
    A["ALL_DYN"], Wt = dyn_weights(Sw, r, last)
    return S, Sw, pd.DataFrame(A), Wt


def poison(P, C, Q):
    cut = pd.Timestamp("2015-06-30"); S0, Sw0, A0, W0 = build(P, C, Q); worst = 0.0; lead = 0.0
    for mult in (7.3, 0.3):
        Pp = P.copy(); Pp.loc[Pp.index > cut] = Pp.loc[Pp.index > cut] * mult
        Cp = {k: v.where(v.index <= cut, v * mult) for k, v in C.items()}
        Qp = Q.copy(); Qp.loc[Qp.index > cut, ["open", "high", "low", "close", "close_raw"]] *= mult
        S1, Sw1, A1, W1 = build(Pp, Cp, Qp)
        for a, b in ((S0, S1), (Sw0, Sw1), (A0, A1)):
            a_, b_ = a.loc[:cut].to_numpy(), b.loc[:cut].to_numpy()
            assert (np.isnan(a_) == np.isnan(b_)).all(), "NaN 模式不一致"
            worst = max(worst, float(np.nanmax(np.abs(a_ - b_))))
        lead = max(lead, float(np.nanmax(np.abs(A0.shift(-2).loc[:cut].to_numpy() - A1.shift(-2).loc[:cut].to_numpy()))))
    return worst, lead


def main():
    P = RF.load_P(); C = load_cboe(); Q = load_qqq_ohlc()
    worst, lead = poison(P, C, Q)
    print(f"[污染自检] cut=2015-06-30 之后 价格/Cboe/QQQ OHLC ×7.3/×0.3:cut 之前 子信号/周频信号/综合分数 最大差 {worst:.2e}(须为 0);阳性对照(前移 2 日)最大差 {lead:.2f}(须>0)")
    assert worst == 0.0 and lead > 0
    S, Sw, A, Wt = build(P, C, Q)
    px = P["QQQ"].dropna(); r = px.pct_change().fillna(0.0); cal = px.index
    print("子信号可用起点:", {c: str(S[c].first_valid_index().date()) for c in S.columns})
    print("可用子信号个数(2004-06 / 2010 / 2020):", [int(Sw.loc[d].notna().sum()) for d in ("2004-06-30", "2010-06-30", "2020-06-30")])
    print("综合分数 A 的均值/标准差(2004-06+):\n", A.loc["2004-06":].agg(["mean", "std", "min", "max"]).round(3).to_string())
    print(f"DYN 权重表 {Wt.shape}:相对 EW 的平均偏离(|w−1/n|·n)={float(np.mean([np.abs(Wt.loc[d].dropna()-1/Wt.loc[d].notna().sum()).mean()*Wt.loc[d].notna().sum() for d in Wt.index])):.3f}(0=等权,0.5=触及边界)")
    print("DYN 平均权重(按信号,2015+ 周):\n", (Wt.loc["2015":].mean()*Wt.loc["2015":].notna().sum(axis=1).mean()).round(2).to_string())
    # ---------------- 配置
    E3 = {}
    for nm in ("ALL_EW", "ALL_DYN"):
        for kind, e_off in (("e0", 0.0), ("e0.5", 0.5)):
            e = R2.exposure2(A[nm], e_off); E3[f"{nm}|{kind}|alone"] = e
            E3[f"{nm}|{kind}|+SMA200"] = (e + R1.trend_exp(px, "SMA", 200, e_off)) / 2
    for g in GROUPS3:
        for kind, e_off in (("e0", 0.0), ("e0.5", 0.5)):
            E3[f"{g}_EW|{kind}|alone"] = R2.exposure2(A[f"{g}_EW"], e_off)
    assert len(E3) == 18
    windows = {"full": FULL, "dev": DEV, "test": TEST}
    R = R1.evaluate(E3, r, px, set(E3), windows)
    prev1 = pd.read_pickle(OUTD / "regime_timing.pkl")["E"]; prev2 = pd.read_pickle(OUTD / "regime_timing_r2.pkl")["E2"]
    fam = {**prev1, **prev2, **E3}; assert len(fam) == 60
    pf, _ = R1.circ_placebo(fam, r, *FULL)
    R["置换p_族内max(60配置,full)"] = [pf.get(c, np.nan) if p == "full" else np.nan for c, p in zip(R.配置, R.期间)]
    R.to_csv(TAB / "regime_timing_r3.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
    cols = ["配置", "平均暴露", "Sharpe", "CAGR", "MDD", "dSR", "dSR_lo", "dSR_hi", "dMDD_pp", "年换手"]
    for wn in ("full", "dev", "test"):
        Sx = R[R.期间 == wn]
        print("\n" + "=" * 140 + f"\n【{wn}】{windows[wn]}\n" + "=" * 140)
        print(Sx[Sx.类型 == "基线"][cols].round(3).to_string(index=False)); print("-" * 140); print(Sx[Sx.类型 == "regime"][cols].round(3).to_string(index=False))
    pv = R[R.类型 == "regime"].pivot_table(index="配置", columns="期间", values=["dSR", "dMDD_pp"])
    ok = pv[(np.sign(pv[("dSR", "dev")]) == np.sign(pv[("dSR", "test")])) & (pv[("dSR", "test")] > 0)]
    print(f"\ndev 与 test ΔSR 同号且为正:{len(ok)}/{len(pv)}"); print(ok.round(3).to_string())
    print("族内 max 置换 p(60 配置)本轮 18 个最小 6 个:", {k: round(v, 3) for k, v in sorted(((k, v) for k, v in pf.items() if k in E3), key=lambda kv: kv[1])[:6]})
    print("族内最小 p(全部 60):", {k: round(v, 3) for k, v in sorted(pf.items(), key=lambda kv: kv[1])[:4]})
    # 对 SMA200 的增量:与 SMA200 的相关与『direct』差
    base_s = R1.trend_exp(px, "SMA", 200, 0.0); m = (r.index >= FULL[0]) & (r.index <= FULL[1])
    print("\n与 SMA200(e=0)暴露的相关:", {k: round(float(np.corrcoef(E3[k][m], base_s[m])[0, 1]), 2) for k in E3 if k.endswith("e0|alone")})
    # 熊市子期
    rows = []
    for sn, (a, b) in SUBP.items():
        mm = (r.index >= a) & (r.index <= b); f = lambda e: f"{100*((1+R1.strat(e,r)[mm]).prod()-1):+6.1f}/{e[mm].mean():.2f}"
        row = {"子期": sn, "B&H": f"{100*((1+r[mm]).prod()-1):+6.1f}/1.00", "SMA200": f(base_s)}
        for k in ("ALL_EW|e0|alone", "ALL_DYN|e0|alone", "MACRO_EW|e0|alone", "SENT_EW|e0|alone", "LEAD_EW|e0|alone", "VALINF_EW|e0|alone", "TECH_EW|e0|alone"): row[k.split("|")[0]] = f(E3[k])
        rows.append(row)
    print("\n熊市子期(区间收益%/平均暴露,e_off=0 alone):"); print(pd.DataFrame(rows).to_string(index=False))
    # ---------------- 描述性:23 个子信号各自作为 {1,0} 暴露(不入族,不用于选择)
    Es = {c: R1.exposure(Sw[c], 0.0) for c in ALL}
    Rs = R1.evaluate(Es, r, px, set(Es), windows); Rs = Rs[Rs.类型 == "regime"]
    pt = Rs.pivot_table(index="配置", columns="期间", values=["dSR", "dMDD_pp", "平均暴露"])
    D = pd.DataFrame({"组": [GROUP_OF[c] for c in pt.index], "平均暴露": pt[("平均暴露", "full")], "dSR_full": pt[("dSR", "full")], "dSR_dev": pt[("dSR", "dev")], "dSR_test": pt[("dSR", "test")],
                      "dMDD_dev": pt[("dMDD_pp", "dev")], "dMDD_test": pt[("dMDD_pp", "test")]}, index=pt.index)
    D = D.loc[ALL]; D.to_csv(TAB / "regime_r3_single_signals.csv")
    print("\n描述性:23 个子信号单独作为 {1,0} 暴露的 ΔSR(相对同平均暴露静态对照;不入族):"); print(D.round(3).to_string())
    print("\n同号为正(dev&test)的单信号:", [c for c in D.index if D.loc[c, "dSR_dev"] > 0 and D.loc[c, "dSR_test"] > 0])
    pd.to_pickle(dict(S=S, Sw=Sw, A=A, Wt=Wt, E3=E3), OUTD / "regime_timing_r3.pkl")


if __name__ == "__main__":
    main()
