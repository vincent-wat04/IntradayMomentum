"""L0 Regime 第二轮 —— docs/PROPOSAL_regime_layer.md「第二轮」。复用第一轮构件。"""
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
import regime_timing_study as R1
import trend_rv_sleeve_study as T
from scipy.stats import spearmanr

TAB, FULL, DEV, TEST, SUBP = R1.TAB, R1.FULL, R1.DEV, R1.TEST, R1.SUBP


def exposure2(a2, e_off):
    e = e_off + (1 - e_off) * ((a2 - 0.35) / 0.30).clip(0, 1)
    return e.where(a2.notna(), 1.0).shift(1).fillna(1.0)


def poison2(P):
    cut = pd.Timestamp("2015-06-30"); V0 = RF.votes(P); worst = 0.0
    for mult in (7.3, 0.3):
        Pp = P.copy(); Pp.loc[Pp.index > cut] = Pp.loc[Pp.index > cut] * mult
        V1 = RF.votes(Pp); worst = max(worst, float(np.nanmax(np.abs(V0.loc[:cut].to_numpy() - V1.loc[:cut].to_numpy()))))
    return worst


def main():
    P = RF.load_P(); w = poison2(P)
    print(f"[污染自检] 投票分数:cut 之前最大差 {w:.2e}(须为 0)"); assert w == 0.0
    F = R1.build_all(P); px = P["QQQ"].dropna(); r = px.pct_change().fillna(0.0); F = F.reindex(px.index)
    V = RF.votes(P, F).reindex(px.index)
    print("A2 可用起点:", {c: str(V[c].first_valid_index().date()) for c in V.columns}, "\nA2 均值/标准差(2004-06+):", V.loc["2004-06":].agg(["mean", "std"]).round(3).to_dict())
    prev = pd.read_pickle(Path("outputs/full/intraday") / "regime_timing.pkl")["E"]                 # 第一轮 22 个
    E2 = {}
    for A in ("A2_B", "A2_M", "A2_C", "A2_V", "A2_all"):
        for kind, e_off in (("e0", 0.0), ("e0.5", 0.5)):
            e = exposure2(V[A], e_off); E2[f"{A}|{kind}|alone"] = e; E2[f"{A}|{kind}|+SMA200"] = (e + R1.trend_exp(px, "SMA", 200, e_off)) / 2
    windows = {"full": FULL, "dev": DEV, "test": TEST}
    R = R1.evaluate(E2, r, px, set(E2), windows)
    fam = {**prev, **E2}
    pf, _ = R1.circ_placebo(fam, r, *FULL)
    R["置换p_族内max(42配置,full)"] = [pf.get(c, np.nan) if p == "full" else np.nan for c, p in zip(R.配置, R.期间)]
    R.to_csv(TAB / "regime_timing_r2.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
    cols = ["配置", "平均暴露", "Sharpe", "CAGR", "MDD", "dSR", "dSR_lo", "dSR_hi", "dMDD_pp", "年换手"]
    for wn in ("full", "dev", "test"):
        S = R[(R.期间 == wn)]
        print("\n" + "=" * 140 + f"\n【{wn}】{windows[wn]}\n" + "=" * 140)
        print(S[S.类型 == "基线"][cols].round(3).to_string(index=False)); print("-" * 140); print(S[S.类型 == "regime"][cols].round(3).to_string(index=False))
    pv = R[R.类型 == "regime"].pivot_table(index="配置", columns="期间", values=["dSR", "dMDD_pp"])
    ok = pv[(np.sign(pv[("dSR", "dev")]) == np.sign(pv[("dSR", "test")])) & (pv[("dSR", "test")] > 0)]
    print(f"\ndev 与 test ΔSR 同号且为正:{len(ok)}/{len(pv)}"); print(ok.round(3).to_string())
    print("族内 max 置换 p(42 配置)最小 6 个:", {k: round(v, 3) for k, v in sorted(pf.items(), key=lambda kv: kv[1])[:6]})
    # 熊市子期
    base_s = R1.trend_exp(px, "SMA", 200, 0.0); rows = []
    for sn, (a, b) in SUBP.items():
        m = (r.index >= a) & (r.index <= b); f = lambda e: f"{100*((1+R1.strat(e,r)[m]).prod()-1):+6.1f}/{e[m].mean():.2f}"
        row = {"子期": sn, "B&H": f"{100*((1+r[m]).prod()-1):+6.1f}/1.00", "SMA200": f(base_s)}
        for A in ("A2_B", "A2_M", "A2_C", "A2_V", "A2_all"): row[A] = f(E2[f"{A}|e0|alone"])
        rows.append(row)
    print("\n熊市子期(区间收益%/平均暴露,e_off=0 alone):"); print(pd.DataFrame(rows).to_string(index=False))
    # 符号不明单变量:对未来 21 日 QQQ 对数收益的 Spearman
    y = np.log1p(r).rolling(21).sum().shift(-21)
    uni = {"债券日RSI": F["bond_rsi_d"], "债券周RSI": F["bond_rsi_w"], "VIX z": F["vix_z"], "VIX 期限 z": F["vix_term_z"], "汽油COT z": F["cot_z"],
           "RSP/SPY 63d(b3 z)": F["b3"], "行业广度 b1 z": F["b1"], "ECY z": F["v1"]}
    urows = []
    for nm, x in uni.items():
        for pn, (a, b) in {"full": (FULL[0], FULL[1]), "dev": DEV, "test": TEST}.items():
            if nm == "汽油COT z": a = max(a, "2014-06-01")
            m = (x.index >= a) & (x.index <= b) & x.notna().to_numpy() & y.notna().to_numpy()
            xv, yv = x[m].to_numpy(), y[m].to_numpy(); idx = np.arange(len(xv)); rg = np.random.default_rng(0); bs = []
            nb = int(np.ceil(len(xv) / 63))
            for _ in range(800):
                st = rg.integers(0, max(len(xv) - 63, 1), nb); sel = np.concatenate([np.arange(s, min(s + 63, len(xv))) for s in st])[:len(xv)]
                bs.append(spearmanr(xv[sel], yv[sel])[0])
            urows.append(dict(变量=nm, 期间=pn, n=int(m.sum()), rho=spearmanr(xv, yv)[0], lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5)))
    U = pd.DataFrame(urows); print("\n符号不明/单变量 Spearman(信号 vs 未来 21 日 QQQ 收益;块自助 63 日):"); print(U.pivot_table(index="变量", columns="期间", values="rho").round(3).to_string())
    print(U[U.期间 == "full"][["变量", "n", "rho", "lo", "hi"]].round(3).to_string(index=False)); U.to_csv(TAB / "regime_univariate.csv", index=False)
    pd.to_pickle(dict(V=V, E2=E2), Path("outputs/full/intraday") / "regime_timing_r2.pkl")


if __name__ == "__main__":
    main()
