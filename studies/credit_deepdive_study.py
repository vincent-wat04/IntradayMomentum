"""A2_C 类(信用趋势)特征的细致研究 —— docs/PROPOSAL_credit_deepdive.md(预注册)。python3 credit_deepdive_study.py"""
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
import regime_features as RF
import regime_round2_study as R2
import regime_timing_study as R1
import trend_rv_sleeve_study as T

TAB = R1.TAB; OUTD = Path("outputs/full/intraday"); RAW = RF.RAW
SMA_N, EMA_N = 126, 10


def fred(name):
    f = pd.read_csv(RAW / "fred" / f"{name}.csv", index_col=0, parse_dates=True)
    return pd.to_numeric(f.iloc[:, 0], errors="coerce").dropna()


def vote(x, sign=+1, n=SMA_N):
    ma = x.rolling(n, min_periods=n).mean()
    return ((sign * (x - ma)) > 0).astype(float).where(ma.notna() & x.notna())


def smooth(v):
    return v.ewm(span=EMA_N, adjust=False, min_periods=1).mean().where(v.notna())


def credit_votes(baa, vwehx, vfitx, cal, sma=SMA_N):
    """三类投票(均因果)。FRED 序列在交易日历上 ffill 后再 shift(1)(发布滞后再保险一日)。"""
    s = baa.reindex(cal, method="ffill").shift(1)
    vS = vote(s, -1, sma)
    xF = np.log(vwehx.reindex(cal, method="ffill") / vfitx.reindex(cal, method="ffill"))
    vF = vote(xF, +1, sma)
    vSF = pd.concat([vS, vF], axis=1).mean(axis=1, skipna=True).where(vS.notna() | vF.notna())
    return vS, vF, vSF


def episodes(e, px, gap=5):
    off = (e == 0).to_numpy(); idx = e.index; runs = []; i = 0; n = len(off)
    while i < n:
        if off[i]:
            j = i
            while j + 1 < n and off[j + 1]: j += 1
            runs.append([i, j]); i = j + 1
        else: i += 1
    merged = []
    for a, z in runs:
        if merged and a - merged[-1][1] - 1 <= gap: merged[-1][1] = z
        else: merged.append([a, z])
    rows = []; pv = px.to_numpy(); pk = px.rolling(252, min_periods=1).max().to_numpy()      # 起点距前高:252 日滚动高点(历史最高会被 2000 年泡沫污染)
    for a, z in merged:
        seg = pv[a:z + 1]; ddseg = (seg / np.maximum.accumulate(seg) - 1).min()
        rows.append(dict(起=idx[a].date(), 止=idx[z].date(), 天数=z - a + 1, 期间QQQ收益=pv[z] / pv[a] - 1, 期间最大回撤=ddseg, 起点距252日高=pv[a] / pk[a] - 1,
                         后21日=pv[min(a + 21, len(pv) - 1)] / pv[a] - 1, 后63日=pv[min(a + 63, len(pv) - 1)] / pv[a] - 1))
    return pd.DataFrame(rows)


def fwd_mdd(px, h=63):
    v = px.to_numpy(); out = np.full(len(v), np.nan)
    for t in range(len(v) - h):
        seg = v[t:t + h + 1]; out[t] = (seg / np.maximum.accumulate(seg) - 1).min()
    return pd.Series(out, index=px.index)


def hac(y, X, lags=21):
    d = pd.concat([y, X], axis=1).dropna(); m = sm.OLS(d.iloc[:, 0], sm.add_constant(d.iloc[:, 1:])).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return m, len(d)


def main():
    pd.set_option("display.width", 240, "display.max_rows", 200, "display.float_format", lambda x: f"{x:8.3f}")
    P = RF.load_P(); px = P["QQQ"].dropna(); r = px.pct_change().fillna(0.0); yrs = px.index.year
    V = pd.read_pickle(OUTD / "regime_timing_r2.pkl"); E2 = V["E2"]; V = V["V"].reindex(px.index)
    e0, e5 = E2["A2_C|e0|alone"], E2["A2_C|e0.5|alone"]
    # =============================================================== A1 年度表
    print("=" * 120 + "\nA1) 年度表(QQQ,A2_C;e0 = 关闭时空仓,e0.5 = 关闭时半仓)\n" + "=" * 120)
    bh = (1 + r).groupby(yrs).prod() - 1; s0 = (1 + R1.strat(e0, r)).groupby(yrs).prod() - 1; s5 = (1 + R1.strat(e5, r)).groupby(yrs).prod() - 1
    Y = pd.DataFrame({"B&H": bh, "A2_C e0": s0, "差e0(pp)": 100 * (s0 - bh), "平均暴露e0": e0.groupby(yrs).mean(), "A2_C e0.5": s5, "差e0.5(pp)": 100 * (s5 - bh)})
    Y[["B&H", "A2_C e0", "A2_C e0.5"]] *= 100; print(Y.round(1).to_string())
    up, dn = bh > 0, bh < 0
    print(f"保险费(上涨年份 e0 差之和)= {100*(s0-bh)[up].sum():.1f}pp / {int(up.sum())} 年;赔付(下跌年份 e0 差之和)= {100*(s0-bh)[dn].sum():.1f}pp / {int(dn.sum())} 年")
    Y.to_csv(TAB / "credit_annual.csv")
    # =============================================================== A2 关闭期清单
    EP = episodes(e0, px); EPm = EP[EP.天数 >= 10]
    print("\n" + "=" * 120 + f"\nA2) A2_C 关闭期(e0=0,间隔≤5 日合并;共 {len(EP)} 个,≥10 日的 {len(EPm)} 个)\n" + "=" * 120)
    print(EPm.round(3).to_string(index=False))
    print(f"汇总(≥10 日):区间数 {len(EPm)};期间 QQQ 收益<0 的占比 {np.mean(EPm.期间QQQ收益<0):.2f};平均期间收益 {EPm.期间QQQ收益.mean():+.3f}(正 = 空仓期错过的涨幅);"
          f"总关闭天数 {int(EP.天数.sum())}/{len(px)}({EP.天数.sum()/len(px):.1%});起点距252日高均值 {EPm.起点距252日高.mean():+.3f}(越负越晚);起点后 21 日平均 {EPm.后21日.mean():+.3f}、63 日 {EPm.后63日.mean():+.3f}")
    EP.to_csv(TAB / "credit_episodes.csv", index=False)
    # =============================================================== A3 预测回归
    print("\n" + "=" * 120 + "\nA3) 预测回归:A2_C 关闭状态(平滑分数≤0.35,t 收盘已知)对未来收益/波动/最大回撤(Newey–West 21 滞后)\n" + "=" * 120)
    off = (V["A2_C"] <= 0.35).astype(float).where(V["A2_C"].notna()); sma_dn = (px < px.rolling(200).mean()).astype(float).where(px.rolling(200).mean().notna())
    ret21 = px / px.shift(21) - 1
    vix = pd.read_parquet(RF.RAW / "cboe" / "VIX.parquet")["close"]; vix.index = pd.to_datetime(vix.index).tz_localize(None); lvix = np.log(vix[~vix.index.duplicated()].reindex(px.index).ffill())
    Yv = {"fwd21收益(%)": 100 * (np.log(px.shift(-21) / px)), "fwd21波动(日std,%)": 100 * r.rolling(21).std().shift(-21), "fwd63最大回撤(pp)": 100 * fwd_mdd(px, 63)}
    rows = []
    for yn, y in Yv.items():
        for per, m in (("全", slice(None)), ("dev 2004–14", slice("2004", "2014")), ("test 2015+", slice("2015", None))):
            for sp, cols in (("仅关闭", [off]), ("+SMA200", [off, sma_dn]), ("+SMA200+ret21", [off, sma_dn, ret21]), ("+SMA200+ret21+logVIX", [off, sma_dn, ret21, lvix])):
                X = pd.concat([c.rename(n) for c, n in zip(cols, ["off", "sma_dn", "ret21", "lvix"])], axis=1).loc[m]; mm, n = hac(y.loc[m], X)
                rows.append(dict(目标=yn, 期间=per, 设定=sp, 关闭系数=mm.params["off"], t_HAC=mm.tvalues["off"], n=n))
    RG = pd.DataFrame(rows); print(RG.pivot_table(index=["目标", "设定"], columns="期间", values=["关闭系数", "t_HAC"], sort=False).round(2).to_string()); RG.to_csv(TAB / "credit_predictive.csv", index=False)
    # =============================================================== A4 与 VIX/SMA200 的关系
    base_s = R1.trend_exp(px, "SMA", 200, 0.0)
    x_ = pd.DataFrame({"off": off, "sma_dn": sma_dn, "vix20": (np.exp(lvix) > 20).astype(float), "fwd21": Yv["fwd21收益(%)"], "r": r}).dropna(subset=["off", "sma_dn", "fwd21"])
    print("\nA4) 关闭状态与 SMA200/VIX 的重合,以及 A2_C 独有信息")
    print(f"P(关闭)={x_.off.mean():.2f};P(关闭|低于SMA200)={x_[x_.sma_dn==1].off.mean():.2f};P(关闭|高于SMA200)={x_[x_.sma_dn==0].off.mean():.2f};相关(关闭, 低于SMA200)={x_.off.corr(x_.sma_dn):.2f};相关(关闭, VIX>20)={x_.off.corr(x_.vix20):.2f}")
    for nm, m in (("关闭 & 高于SMA200(A2_C 独有)", (x_.off == 1) & (x_.sma_dn == 0)), ("关闭 & 低于SMA200", (x_.off == 1) & (x_.sma_dn == 1)), ("在线 & 低于SMA200(SMA200 独有)", (x_.off == 0) & (x_.sma_dn == 1)), ("在线 & 高于SMA200", (x_.off == 0) & (x_.sma_dn == 0))):
        g = x_[m]; print(f"  {nm:32s} n={len(g):5d}({len(g)/len(x_):.1%}) fwd21 均值 {g.fwd21.mean():+.2f}% 胜率 {np.mean(g.fwd21>0):.2f} 日收益均值 {1e4*g.r.mean():+.1f}bp 日波动 {100*g.r.std():.2f}%")
    # =============================================================== A5 机理:c2(LQD/IEF)到底是不是『信用利差』?
    print("\nA5) LQD/IEF 趋势信号是信用利差吗?(2004-06+)")
    d_ratio = np.log(P["LQD"] / P["IEF"]).diff(); dbaa = fred("BAA10Y").reindex(px.index, method="ffill").diff(); dy10 = fred("DGS10").reindex(px.index, method="ffill").diff()
    dd = pd.concat([d_ratio.rename("y"), dbaa.rename("d_baa10y"), dy10.rename("d_dgs10")], axis=1).loc["2004-06":].dropna()
    for cols in (["d_baa10y"], ["d_dgs10"], ["d_baa10y", "d_dgs10"]):
        mm = sm.OLS(dd.y, sm.add_constant(dd[cols])).fit(cov_type="HAC", cov_kwds={"maxlags": 5}); print(f"  d log(LQD/IEF) ~ {'+'.join(cols):18s} R²={mm.rsquared:.3f} 系数 {dict(zip(cols, mm.params[cols].round(4)))} t={dict(zip(cols, mm.tvalues[cols].round(1)))}")
    vL = vote(np.log(P["LQD"] / P["IEF"]).reindex(px.index), +1); vH = vote(np.log(P["HYG"] / P["IEF"]).reindex(px.index), +1); vT = vote(np.log(P["TLT"]).reindex(px.index), +1)
    vB = vote(fred("BAA10Y").reindex(px.index, method="ffill").shift(1), -1); vD = vote(fred("DGS10").reindex(px.index, method="ffill").shift(1), -1)
    VV = pd.concat([vL.rename("LQD/IEF"), vH.rename("HYG/IEF"), vT.rename("TLT"), vB.rename("BAA10Y利差↓"), vD.rename("10y收益率↓")], axis=1).loc["2004-06":].dropna()
    print("  投票相关矩阵:"); print(VV.corr().round(2).to_string())
    # =============================================================== B 1986–2003 真样本外
    print("\n" + "=" * 120 + "\nB) 1986–2003 真样本外:BAA10Y 利差趋势 / 高收益基金趋势 → ^NDX\n" + "=" * 120)
    LH = pd.read_parquet(RAW / "regime" / "long_history.parquet"); ndx = LH["^NDX"].dropna(); cal = ndx.index; rn = ndx.pct_change().fillna(0.0)
    baa = fred("BAA10Y"); vS, vF, vSF = credit_votes(baa, LH["VWEHX"], LH["VFITX"], cal)
    # --- 自检:cut 之后 ×7.3 → cut 及以前的投票不变;阳性对照(前移 2 日)必须被检出
    cut = pd.Timestamp("1998-06-30"); dmax = 0.0; lead = 0.0
    for mult in (7.3, 0.2):
        b2 = baa.copy(); b2[b2.index > cut] *= mult; l2 = LH.copy(); l2.loc[l2.index > cut, ["VWEHX", "VFITX"]] *= mult
        vS2, vF2, vSF2 = credit_votes(b2, l2["VWEHX"], l2["VFITX"], cal)
        dmax = max(dmax, max(float(np.nanmax(np.abs((a - b).loc[:cut].to_numpy()))) for a, b in ((vS, vS2), (vF, vF2), (vSF, vSF2))))
        lead = max(lead, float(np.nanmax(np.abs((vSF.shift(-20) - vSF2.shift(-20)).loc[:cut].to_numpy()))))
    print(f"[污染自检] 1998-06-30 之后 BAA10Y/基金 ×7.3 与 ×0.2:cut 及以前的投票最大差 {dmax:.2e}(须为 0);阳性对照(前移 20 日)最大差 {lead:.2f}(须>0)"); assert dmax == 0.0 and lead > 0
    A_ = {"A_S": smooth(vS), "A_F": smooth(vF), "A_SF": smooth(vSF)}
    EB = {f"{k}|{kind}|alone": R2.exposure2(a, eo) for k, a in A_.items() for kind, eo in (("e0", 0.0), ("e0.5", 0.5))}
    print("可用起点:", {k: str(a.first_valid_index().date()) for k, a in A_.items()}, "| 平均分数(1986–2003):", {k: round(float(a.loc[:"2003"].mean()), 2) for k, a in A_.items()})
    W = {"OOS-A 1986-09~1994": ("1986-09-01", "1994-12-31"), "OOS-B 1995~2003": ("1995-01-01", "2003-12-31"), "OOS 合并": ("1986-09-01", "2003-12-31"), "ETF时代 2004-06+": ("2004-06-01", "2026-10-09")}
    RB = R1.evaluate(EB, rn, ndx, set(EB), W)
    pf, _ = R1.circ_placebo(EB, rn, *W["OOS 合并"], nperm=2000)
    cols = ["配置", "平均暴露", "Sharpe", "CAGR", "MDD", "对照MDD", "dSR", "dSR_lo", "dSR_hi", "dMDD_pp", "年换手"]
    for wn in W:
        S = RB[RB.期间 == wn]; print(f"\n--- {wn} {W[wn]} ---"); print(S[cols].round(3).to_string(index=False))
    RB["置换p_族内max(6)"] = [pf.get(c, np.nan) if p == "OOS 合并" else np.nan for c, p in zip(RB.配置, RB.期间)]; RB.to_csv(TAB / "credit_oos86.csv", index=False)
    print("\n族内 max 置换 p(OOS 合并,6 个配置):", {k: round(v, 3) for k, v in sorted(pf.items(), key=lambda kv: kv[1])})
    pv = RB[RB.类型 == "regime"].pivot_table(index="配置", columns="期间", values=["dSR", "dMDD_pp"])
    print("\n判定摘要(OOS-A / OOS-B 的 ΔSR 同号为正?;OOS ΔSR、ΔMDD;对 SMA200 的增量):")
    for c in EB:
        sm_ = f"SMA200|e={'0' if 'e0|' in c else '0.5'}"; base = RB[(RB.配置 == sm_) & (RB.期间 == "OOS 合并")].iloc[0]
        a_, b_ = pv.loc[c, ("dSR", "OOS-A 1986-09~1994")], pv.loc[c, ("dSR", "OOS-B 1995~2003")]
        print(f"  {c:18s} A={a_:+.3f} B={b_:+.3f} 同号为正={a_>0 and b_>0} | OOS ΔSR {pv.loc[c,('dSR','OOS 合并')]:+.3f} ΔMDD {pv.loc[c,('dMDD_pp','OOS 合并')]:+.1f}pp | 对 SMA200 的 ΔSR 增量 {pv.loc[c,('dSR','OOS 合并')]-base.dSR:+.3f}, ΔMDD 增量 {pv.loc[c,('dMDD_pp','OOS 合并')]-base.dMDD_pp:+.1f}pp | p={pf[c]:.3f}")
    SUB2 = {"1987 崩盘": ("1987-08-25", "1987-12-04"), "1990": ("1990-07-16", "1990-10-11"), "1998": ("1998-07-17", "1998-10-08"), "互联网泡沫 2000-03~2002-10": ("2000-03-10", "2002-10-09")}
    base0 = R1.trend_exp(ndx, "SMA", 200, 0.0); rows = []
    for sn, (a, z) in SUB2.items():
        m = (rn.index >= a) & (rn.index <= z); f = lambda e: f"{100*((1+R1.strat(e,rn)[m]).prod()-1):+6.1f}/{e[m].mean():.2f}"
        rows.append({"子期": sn, "B&H": f"{100*((1+rn[m]).prod()-1):+6.1f}/1.00", "SMA200": f(base0), **{k.split('|')[0]: f(EB[k]) for k in EB if k.endswith("e0|alone")}})
    print("\n熊市子期(区间收益%/平均暴露,e0):"); print(pd.DataFrame(rows).to_string(index=False))
    # =============================================================== C 参数高原(A_S, OOS 合并)
    print("\n" + "=" * 120 + "\nC) 参数高原(A_S|e0,OOS 合并 1986-09~2003;不用于选参)\n" + "=" * 120)
    m = (rn.index >= W["OOS 合并"][0]) & (rn.index <= W["OOS 合并"][1]); rows = []
    for n in (63, 126, 252):
        for sp in (1, 10):
            v = vote(baa.reindex(cal, method="ffill").shift(1), -1, n); a = v.ewm(span=sp, adjust=False, min_periods=1).mean().where(v.notna()); e = R2.exposure2(a, 0.0)
            x = R1.strat(e, rn)[m]; ctrl = float(e[m].mean()) * rn[m]; d, dist = T.boot_dsr(x, ctrl, nb=1500, L=40, seed=0)
            rows.append(dict(SMA=n, EMA=sp, 平均暴露=float(e[m].mean()), dSR=d, lo=np.percentile(dist, 2.5), hi=np.percentile(dist, 97.5), dMDD_pp=100 * (R1.mdd(x) - R1.mdd(ctrl))))
    print(pd.DataFrame(rows).round(3).to_string(index=False)); pd.DataFrame(rows).to_csv(TAB / "credit_plateau.csv", index=False)
    # =============================================================== D FRED 扩展(QQQ 2004-06+;描述性)
    print("\n" + "=" * 120 + "\nD) FRED 扩展的单信号({1,0} 暴露)ΔSR —— QQQ 2004-06+;探索性,不入族\n" + "=" * 120)
    qcal = px.index; baa_ = fred("BAA10Y"); aaa_ = fred("AAA10Y")
    feats = {"BAA10Y(−)": (baa_, -1), "AAA10Y(−)": (aaa_, -1), "BAA−AAA 质量利差(−)": ((baa_ - aaa_).dropna(), -1), "T10Y3M 曲线(+)": (fred("T10Y3M"), +1), "T10YIE 盈亏平衡通胀(−)": (fred("T10YIE"), -1), "DFII10 实际利率(−)": (fred("DFII10"), -1),
             "LQD/IEF(+,对照 c2)": (np.log(P["LQD"] / P["IEF"]).dropna(), +1)}
    ED = {}
    for nm, (s_, sg) in feats.items():
        x = s_.reindex(qcal, method="ffill"); x = x.shift(1) if "对照" not in nm else x
        ED[nm] = R1.exposure(vote(x, sg), 0.0)
    WD = {"full": R1.FULL, "dev": R1.DEV, "test": R1.TEST}
    RD = R1.evaluate(ED, r, px, set(ED), WD); RD = RD[RD.类型 == "regime"]
    pt = RD.pivot_table(index="配置", columns="期间", values=["dSR", "dMDD_pp", "平均暴露"], sort=False)
    out = pd.DataFrame({"平均暴露": pt[("平均暴露", "full")], "dSR_full": pt[("dSR", "full")], "dSR_dev": pt[("dSR", "dev")], "dSR_test": pt[("dSR", "test")], "dMDD_dev": pt[("dMDD_pp", "dev")], "dMDD_test": pt[("dMDD_pp", "test")]})
    print(out.round(3).to_string()); out.to_csv(TAB / "credit_fred_singles.csv")
    pd.to_pickle(dict(EB=EB, A_=A_, EP=EP, RB=RB), OUTD / "credit_deepdive.pkl")


if __name__ == "__main__":
    main()
