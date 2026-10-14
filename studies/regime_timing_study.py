"""L0 宏观/市场环境 Regime 择时 —— 严格按 docs/PROPOSAL_regime_layer.md 执行。
  export ETFT_DATA=... ETFT_HOME=... OMP_NUM_THREADS=1;  python regime_timing_study.py
"""
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import regime_features as RF
import trend_rv_sleeve_study as T

TAB = Path("../outputs/tables") if Path("../outputs/tables").exists() else Path("outputs/tables")
COST = 2.0 / 1e4
FULL, DEV, TEST = ("2004-06-01", "2026-10-09"), ("2004-06-01", "2014-12-31"), ("2015-01-01", "2026-10-09")
SUBP = {"GFC 2007-10~2009-03": ("2007-10-01", "2009-03-09"), "2018Q4": ("2018-10-01", "2018-12-24"),
        "COVID 2020-02~04": ("2020-02-19", "2020-04-30"), "2022": ("2022-01-01", "2022-12-31")}
NPERM, L = 2000, 40


def sr(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return x.mean() / x.std() * np.sqrt(252) if len(x) > 60 and x.std() > 0 else np.nan


def mdd(x):
    e = (1 + pd.Series(x).fillna(0)).cumprod(); return float((e / e.cummax() - 1).min())


def cagr(x):
    x = pd.Series(x).dropna(); e = (1 + x).prod(); return float(e ** (252 / len(x)) - 1)


def exposure(sig, e_off, kind="bin"):
    """sig>0 → 1,否则 e_off;NaN → 1(未预热,中性)。t 收盘形成,返回『t 日实际作用的暴露』(shift 1)。"""
    e = pd.Series(np.where(sig > 0, 1.0, e_off), index=sig.index).where(sig.notna(), 1.0)
    return e.shift(1).fillna(1.0)


def trend_exp(px, kind, n, e_off):
    dn = T.trend_down(px, kind, n).astype(float)
    return (1.0 - dn * (1.0 - e_off)).shift(1).fillna(1.0)


def strat(e, r):
    return e * r - COST * e.diff().abs().fillna(0.0)


def walk_forward_ridge(F, X_cols, r, h=21, start_year=2004):
    """每年初用截至上年末(且目标窗口已结束)的数据重估岭回归,预测未来 h 日累计对数收益。返回逐日预测(t 收盘形成)。"""
    from sklearn.linear_model import Ridge
    y = np.log1p(r).rolling(h).sum().shift(-h)                                # t 日形成的信号对应 t+1..t+h 的收益
    X = F[X_cols].fillna(0.0)
    pred = pd.Series(np.nan, index=F.index)
    for yr in range(start_year, 2027):
        tr_end = pd.Timestamp(f"{yr}-01-01"); idx = X.index
        tr = (idx < tr_end) & y.notna().to_numpy() & (np.arange(len(idx)) < np.searchsorted(idx, tr_end) - h)    # 目标窗口在训练期内结束
        te = idx.year == yr
        if tr.sum() < 500: continue
        mu, sd = X[tr].mean(), X[tr].std().replace(0, 1)
        m = Ridge(alpha=100.0).fit(((X[tr] - mu) / sd).to_numpy(), y[tr].to_numpy())
        pred[te] = m.predict(((X[te] - mu) / sd).to_numpy())
    return pred


def build_all(P):
    F = RF.features(P)
    cal = F.index
    F = F.join(RF.vix_z(cal)); F["cot_z"] = RF.cot_z(cal)
    return F


def configs(F, px, r, with_ridge=True):
    """返回 {名称: 暴露序列(已 shift)}。"""
    out = {}
    for A in ("A_B", "A_M", "A_C", "A_V", "A_all"):
        for kind, e_off in (("bin", 0.0), ("floor", 0.5)):
            e_r = exposure(F[A], e_off)
            out[f"{A}|{kind}|alone"] = e_r
            out[f"{A}|{kind}|+SMA200"] = (e_r + trend_exp(px, "SMA", 200, e_off)) / 2
    if with_ridge:
        cols = list(RF.SIGN) + ["bond_rsi_d", "bond_rsi_w", "vix_z", "vix_term_z", "cot_z"]
        pr = walk_forward_ridge(F, cols, r)
        for kind, e_off in (("bin", 0.0), ("floor", 0.5)):
            out[f"ridge21|{kind}|alone"] = exposure(pr, e_off)
        out["_ridge_pred"] = pr
    return out


def evaluate(E, r, px, name_set, windows):
    rows = []
    base = {"静态100%": pd.Series(1.0, index=r.index)}
    for e_off in (0.0, 0.5):
        for kind, n in (("SMA", 200), ("MOM", 252)):
            base[f"{kind}{n}|e={e_off:g}"] = trend_exp(px, kind, n, e_off)
    allc = {**base, **{k: v for k, v in E.items() if not k.startswith("_")}}
    for nm, e in allc.items():
        for wn, (a, b) in windows.items():
            m = (r.index >= a) & (r.index <= b)
            x = strat(e, r)[m]; ctrl = float(e[m].mean()) * r[m]
            d, dist = T.boot_dsr(x, ctrl, nb=1500, L=L, seed=0)
            lo, hi = np.percentile(dist, [2.5, 97.5])
            rows.append(dict(配置=nm, 类型="基线" if nm in base else "regime", 期间=wn, 平均暴露=float(e[m].mean()), Sharpe=sr(x), CAGR=cagr(x), MDD=mdd(x),
                             对照MDD=mdd(ctrl), dSR=d, dSR_lo=lo, dSR_hi=hi, dMDD_pp=100 * (mdd(x) - mdd(ctrl)), 年换手=float(e[m].diff().abs().sum() / (m.sum() / 252))))
    return pd.DataFrame(rows)


def circ_placebo(E_fam, r, a, b, nperm=NPERM, seed=0):
    """圆周平移暴露(≥252 日),保持自相关与平均暴露;族内 max ΔSR 的 null。"""
    m = (r.index >= a) & (r.index <= b); rv = r[m].to_numpy(); n = len(rv)
    rg = np.random.default_rng(seed); shifts = rg.integers(252, n - 252, size=nperm)
    obs, nulls = {}, []
    for nm, e in E_fam.items():
        ev = e[m].to_numpy(); ctrl = sr(ev.mean() * rv)
        obs[nm] = sr(ev * rv - COST * np.abs(np.diff(ev, prepend=ev[0]))) - ctrl
        nulls.append(np.array([sr(np.roll(ev, s) * rv - COST * np.abs(np.diff(np.roll(ev, s), prepend=np.roll(ev, s)[0]))) - ctrl for s in shifts]))
    famax = np.max(np.vstack(nulls), axis=0)
    return {nm: float(np.mean(famax >= o)) for nm, o in obs.items()}, {nm: float(np.mean(nl >= obs[nm])) for nm, nl in zip(E_fam, nulls)}


def poison_check(P):
    """cut 之后所有价格序列 ×7.3 / ×0.3:cut+1 之前(含)的信号与暴露逐位不变;阳性对照:信号前移 2 日必须被检出。"""
    cut = pd.Timestamp("2015-06-30"); F0 = build_all(P); worst = 0.0; lead = 0.0
    for mult in (7.3, 0.3):
        Pp = P.copy(); Pp.loc[Pp.index > cut] = Pp.loc[Pp.index > cut] * mult
        F1 = build_all(Pp)
        cols = ["A_B", "A_M", "A_C", "A_V", "A_all", "A_core", "bond_rsi_d", "bond_rsi_w"]
        a, b = F0.loc[:cut, cols], F1.loc[:cut, cols]
        worst = max(worst, float(np.nanmax(np.abs(a.to_numpy() - b.to_numpy()))))
        lead = max(lead, float(np.nanmax(np.abs(F0.shift(-2).loc[:cut, cols].to_numpy() - F1.shift(-2).loc[:cut, cols].to_numpy()))))
    return worst, lead


def main():
    P = RF.load_P()
    worst, lead = poison_check(P)
    print(f"[污染自检] cut=2015-06-30 之后全部价格 ×7.3/×0.3:cut 之前的信号最大差 {worst:.2e}(须为 0);阳性对照(前移 2 日)最大差 {lead:.2f}(须>0)")
    assert worst == 0.0 and lead > 0
    F = build_all(P)
    px = P["QQQ"].dropna(); r = px.pct_change().fillna(0.0); F = F.reindex(px.index)
    print(f"QQQ {px.index[0].date()}~{px.index[-1].date()};特征缺失(2004-06 后):", F.loc["2004-06":, RF.SIGN.keys()].isna().mean().round(2).to_dict())
    # 缺失检查:4 组复合分数在 2004-06 后是否都有值
    print("复合分数可用起点:", {c: str(F[c].first_valid_index().date()) for c in ["A_B", "A_M", "A_C", "A_V", "A_all", "A_core"]})
    E = configs(F, px, r)
    fam = {k: v for k, v in E.items() if not k.startswith("_")}
    windows = {"full": FULL, "dev": DEV, "test": TEST}
    R = evaluate(E, r, px, set(fam), windows)
    pf, ps = circ_placebo(fam, r, *FULL)
    R["置换p_族内max(full)"] = [pf.get(c, np.nan) if p == "full" else np.nan for c, p in zip(R.配置, R.期间)]
    R.to_csv(TAB / "regime_timing.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
    cols = ["配置", "平均暴露", "Sharpe", "CAGR", "MDD", "dSR", "dSR_lo", "dSR_hi", "dMDD_pp", "年换手"]
    for wn in ("full", "dev", "test"):
        S = R[R.期间 == wn]
        print("\n" + "=" * 140 + f"\n【{wn}】{windows[wn]} — 相对『同平均暴露静态对照』ΔSR/ΔMDD(块自助 L={L});|ΔSR|<{T.GUARD} 按噪声\n" + "=" * 140)
        print(S[S.类型 == "基线"][cols].round(3).to_string(index=False)); print("-" * 140)
        print(S[S.类型 == "regime"][cols].round(3).to_string(index=False))
    # 判定表:dev/test 同号 + 幅度 + 增量于 SMA200
    pv = R.pivot_table(index="配置", columns="期间", values=["dSR", "dMDD_pp", "Sharpe", "MDD"])
    reg = pv.loc[[c for c in fam]]
    ok = reg[(np.sign(reg[("dSR", "dev")]) == np.sign(reg[("dSR", "test")])) & (reg[("dSR", "test")] > 0)]
    print("\ndev 与 test 同号且为正的 regime 配置:", len(ok), "/", len(reg))
    print(ok[[("dSR", "dev"), ("dSR", "test"), ("dMDD_pp", "dev"), ("dMDD_pp", "test")]].round(3).to_string())
    print("\n族内 max 置换 p(full):", {k: round(v, 3) for k, v in sorted(pf.items(), key=lambda kv: kv[1])[:6]})
    # 子期:熊市里 regime 暴露平均、区间收益 vs B&H vs SMA200
    print("\n" + "=" * 120 + "\n熊市子期:区间收益(%) / 区间平均暴露:B&H | SMA200 e=0 | 各 regime(bin alone / bin +SMA200)\n" + "=" * 120)
    base_s = trend_exp(px, "SMA", 200, 0.0)
    rows = []
    for sn, (a, b) in SUBP.items():
        m = (r.index >= a) & (r.index <= b)
        f = lambda e: f"{100*((1+strat(e,r)[m]).prod()-1):+6.1f}/{e[m].mean():.2f}"
        row = {"子期": sn, "B&H": f"{100*((1+r[m]).prod()-1):+6.1f}/1.00", "SMA200": f(base_s)}
        for A in ("A_B", "A_M", "A_C", "A_V", "A_all"):
            row[f"{A} alone"] = f(fam[f"{A}|bin|alone"])
        row["ridge21"] = f(fam["ridge21|bin|alone"]); rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False))
    pd.to_pickle(dict(F=F, E=fam, pred=E["_ridge_pred"], r=r), Path("outputs/full/intraday") / "regime_timing.pkl")


if __name__ == "__main__":
    main()
