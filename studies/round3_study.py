"""第三轮：H1/H2/H3 主检验（Bonferroni）+ RV21 体制三检验 + 象限分解。

严格按 docs/PROPOSAL_round3.md 执行。偏离项写在该文末尾。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import round3_common as C
from fear_derisk_study import causal_pct, weight_from
from stop_significance import block_boot_pair

ALPHA = 0.05 / 3                 # Bonferroni：三个主假设
Z_MDE = 2.39 + 0.84              # 双侧 α=0.0167 的临界值 + 80% 功效
DOW = ["tue", "wed", "thu"]


def sr(x):
    x = pd.Series(x).dropna()
    return float(x.mean() / x.std() * np.sqrt(252))


def mdd(x):
    eq = (1 + pd.Series(x).dropna()).cumprod()
    return float((eq / eq.cummax() - 1).min())


def ols(y, X, lag):
    X = sm.add_constant(X)
    return sm.OLS(y, X, missing="drop").fit(cov_type="HAC", cov_kwds={"maxlags": lag})


def fmt_p(p):
    return f"{p:.4f}" if p >= 1e-4 else "<1e-4"


# =====================================================================
# H1 · 机制链 M1：VIX 相对 RV21 是否有增量信息（标普 500，独立的 36 年）
# =====================================================================
def h1_m1():
    print("=" * 112)
    print("H1 · M1（计数的主检验）：标普 500 日度，1990-2026。VIX 是否在 RV21 之外预测未来 21 日已实现波动？")
    print("=" * 112)
    px = C.load_close("yahoo/IDX_GSPC.parquet").loc["1990":]
    vix = C.load_close("cboe/VIX.parquet").reindex(px.index).ffill()
    r = px.pct_change()
    rv = C.rv21_ann(px)
    fwd = r[::-1].rolling(21, min_periods=21).std()[::-1].shift(-1) * np.sqrt(252) * 100
    df = pd.DataFrame({"lf": np.log(fwd), "lrv": np.log(rv), "lvix": np.log(vix)})
    df["z_gap"] = C.causal_z(df.lvix - df.lrv)
    df["z_lrv"] = C.causal_z(df.lrv)
    df = df.dropna()
    print(f"样本 {df.index[0].date()} ~ {df.index[-1].date()}，{len(df)} 天；前向窗口重叠 -> HAC lag=25")
    print(f"corr(z_gap, z_lrv) = {df.z_gap.corr(df.z_lrv):+.3f}（缺口与 RV 水平的重合度）")
    print(f"\n  {'样本':14s}{'设定':34s}{'系数':>9}{'HAC t':>8}{'p':>9}")
    res = {}
    for lab, per in (("全 1991-2026", None), ("1991-2008", ("1991", "2008")), ("2009-2026", ("2009", "2026"))):
        d = df if per is None else df.loc[per[0]:per[1]]
        y1 = d.lf - d.lrv
        mA = ols(y1, d[["z_gap"]], 25)
        mB = ols(d.lf, d[["lrv", "lvix"]], 25)
        mC = ols(y1, d[["z_gap", "z_lrv"]], 25)
        rows = [("A 预注册原式 y=lf−lrv ~ z_gap*", mA, "z_gap"),
                ("B 无约束 lf ~ lrv + lvix（主）", mB, "lvix"),
                ("C y ~ z_gap + z_lrv", mC, "z_gap")]
        for nm, m, k in rows:
            print(f"  {lab:14s}{nm:34s}{m.params[k]:>9.4f}{m.tvalues[k]:>8.2f}{fmt_p(m.pvalues[k]):>9}")
        print(f"  {'':14s}{'  B 中 lrv 系数':34s}{mB.params['lrv']:>9.4f}{mB.tvalues['lrv']:>8.2f}{fmt_p(mB.pvalues['lrv']):>9}")
        res[lab] = mB
    print("  * A 的左右两边共享 −log RV21 噪声，机械偏向正值，仅供参考（见偏离记录）。")
    m = res["全 1991-2026"]
    ok = (m.params["lvix"] > 0) and (m.pvalues["lvix"] < ALPHA)
    return ok, m


# =====================================================================
# 经济检验：sleeve 的 u_t 对信号（QQQ / SPY）
# =====================================================================
def econ():
    print("\n" + "=" * 112)
    print("经济检验：u_t = sleeve 净收益 ÷ 当日敞口（bp），回归在 t−1 已知的信号上；HAC lag=5；剔周一与紧急会议日")
    print("=" * 112)
    tests = [
        ("H2 FOMC 日（计数·QQQ）", ["fomc"] + DOW, "fomc"),
        ("H3 z_cor（计数·QQQ）", ["z_cor"] + DOW, "z_cor"),
        ("H3c z_cor | 控制 z_lrv", ["z_cor", "z_lrv"] + DOW, "z_cor"),
        ("H1e z_gap（参考，不计数）", ["z_gap"] + DOW, "z_gap"),
        ("H1c z_gap | 控制 z_lrv", ["z_gap", "z_lrv"] + DOW, "z_gap"),
    ]
    out = {}
    print(f"  {'检验':28s}{'标的':6s}{'n':>6}{'系数(bp)':>10}{'SE':>8}{'t':>7}{'p':>9}{'MDE(bp)':>9}")
    for nm, cols, key in tests:
        for sym in ("QQQ", "SPY"):
            d = C.design(sym).dropna(subset=[c for c in cols if c.startswith("z_")])
            m = ols(d.u * 1e4, d[cols], 5)
            b, se = m.params[key], m.bse[key]
            out[(nm, sym)] = (b, se, m.tvalues[key], m.pvalues[key], len(d))
            print(f"  {nm:28s}{sym:6s}{len(d):>6}{b:>10.2f}{se:>8.2f}{m.tvalues[key]:>7.2f}"
                  f"{fmt_p(m.pvalues[key]):>9}{Z_MDE*se:>9.1f}")
    d = C.design("QQQ")
    f, n = d[d.fomc == 1].u * 1e4, d[d.fomc == 0].u * 1e4
    print(f"\n  描述：QQQ u 均值 FOMC 日 {f.mean():+.2f}bp（n={len(f)}） vs 其他 {n.mean():+.2f}bp（n={len(n)}）；"
          f"FOMC 日 u 标准差 {f.std():.1f}bp vs {n.std():.1f}bp")
    return out


# =====================================================================
# 体制检验
# =====================================================================
def rv21_derisk(px: pd.Series):
    """返回 (次日收益 rr, RV21 降风险后的次日收益 x, 权重 w)。信号 t 作用于 t+1。"""
    r = px.pct_change()
    p = causal_pct(r.rolling(21).std())
    w = weight_from(p, "derisk", 1.0)
    rr = r.shift(-1)
    return rr, w * rr, w


def seg_stats(rr, x, lo, hi, exclude=()):
    a, b = rr.loc[lo:hi].copy(), x.loc[lo:hi].copy()
    for s, e in exclude:
        m = (a.index >= s) & (a.index <= e)
        a, b = a[~m], b[~m]
    a, b = a.dropna(), b.reindex(a.index).dropna()
    a = a.reindex(b.index)
    if len(a) < 250:
        return None
    r, dist = block_boot_pair(b, a, nb=2000, L=10, seed=5)
    return dict(n=len(a), sr_bh=sr(a), sr_rv=sr(b), d=sr(b) - sr(a),
                mdd_bh=mdd(a), mdd_rv=mdd(b), p=float((dist <= 0).mean()))


CRASH = (("2008-09-01", "2009-03-31"), ("2020-02-01", "2020-05-31"))


def regime():
    print("\n" + "=" * 112)
    print("RV21 体制检验（对应 §5 的 R1 / R2 / R3）。ΔSR = RV21 降风险 − 买入持有；p 为配对区块自助的单边概率")
    print("=" * 112)
    cache = {}

    def get(nm, rel):
        if nm not in cache:
            cache[nm] = rv21_derisk(C.load_close(rel).dropna())
        return cache[nm]

    print("\n[R1] 剔除两次崩盘（2008-09~2009-03、2020-02~2020-05）后还剩什么")
    print(f"  {'指数':8s}{'样本':34s}{'n':>6}{'BH SR':>8}{'RV SR':>8}{'ΔSR':>8}{'p':>7}{'BH MDD':>9}{'RV MDD':>9}")
    r1 = {}
    for nm, rel in (("SPX", "yahoo/IDX_GSPC.parquet"), ("NDX", "yahoo/IDX_NDX.parquet")):
        rr, x, _ = get(nm, rel)
        for lab, lo, hi, ex in (
                ("1992-2026 全部", "1992", "2026", ()),
                ("1992-2026 剔两次崩盘", "1992", "2026", CRASH),
                ("2010-2026 全部", "2010", "2026", ()),
                ("2010-2026 剔 COVID", "2010", "2026", (CRASH[1],)),
                ("2010-2019 平静十年", "2010", "2019", ()),
                ("2021-2026 近期", "2021", "2026", ())):
            s = seg_stats(rr, x, lo, hi, ex)
            if s is None:
                continue
            r1[(nm, lab)] = s
            print(f"  {nm:8s}{lab:34s}{s['n']:>6}{s['sr_bh']:>8.3f}{s['sr_rv']:>8.3f}{s['d']:>+8.3f}"
                  f"{s['p']:>7.3f}{s['mdd_bh']:>9.1%}{s['mdd_rv']:>9.1%}")

    print("\n[R2] 其他主要股指：2008 前 / 危机 / 2010 后（各段需 ≥250 天）")
    print(f"  {'指数':8s}{'起点':12s}{'2008前 ΔSR':>12}{'2008-09 ΔSR':>13}{'2010后 ΔSR':>12}{'2010后 MDD(BH→RV)':>22}")
    r2 = {}
    for nm, rel in (("SPX", "yahoo/IDX_GSPC.parquet"), ("NDX", "yahoo/IDX_NDX.parquet"),
                    ("RUT", "yahoo/IDX_RUT.parquet"), ("EFA", "yahoo/EFA.parquet"),
                    ("EEM", "yahoo/EEM.parquet"), ("EWJ", "yahoo/EWJ.parquet"),
                    ("FXI", "yahoo/FXI.parquet")):
        try:
            rr, x, _ = get(nm, rel)
        except Exception as ex:
            print(f"  {nm:8s} 读取失败：{ex}"); continue
        st = rr.dropna().index[0]
        pre = seg_stats(rr, x, str(st.year + 1), "2007")
        cri = seg_stats(rr, x, "2008", "2009")
        pos = seg_stats(rr, x, "2010", "2026")
        f = lambda s: f"{s['d']:+.3f}" if s else "  n/a"
        mm = f"{pos['mdd_bh']:.1%}→{pos['mdd_rv']:.1%}" if pos else "n/a"
        print(f"  {nm:8s}{str(st.date()):12s}{f(pre):>12}{f(cri):>13}{f(pos):>12}{mm:>22}")
        r2[nm] = (pre, cri, pos)

    print("\n[R3] 最近窗口是否衰减：滚动 6 年窗口的 ΔSR（SPX / NDX）")
    print(f"  {'窗口':14s}{'SPX ΔSR':>10}{'SPX BH→RV MDD':>18}{'NDX ΔSR':>10}{'NDX BH→RV MDD':>18}")
    r3 = {}
    for lo, hi in (("1992", "1997"), ("1998", "2003"), ("2004", "2009"), ("2010", "2015"),
                   ("2016", "2021"), ("2021", "2026")):
        row = f"  {lo+'-'+hi:14s}"
        for nm in ("SPX", "NDX"):
            rr, x, _ = cache[nm]
            s = seg_stats(rr, x, lo, hi)
            if s is None:
                row += f"{'n/a':>10}{'':>18}"; continue
            r3[(nm, lo)] = s
            row += f"{s['d']:>+10.3f}{s['mdd_bh']:>8.1%}→{s['mdd_rv']:.1%}".rjust(28)
        print(row)
    return r1, r2, r3


# =====================================================================
# RV21 降风险 与 H1 缺口 / 象限 的关系（描述性）
# =====================================================================
def quadrants():
    print("\n" + "=" * 112)
    print("RV21 降风险 与 隐含-已实现缺口的关系（QQQ + VXN，2011-2026，描述性，不进决定规则）")
    print("=" * 112)
    q = C.load_close("yahoo/QQQ.parquet")
    r = q.pct_change()
    vxn = C.load_close("cboe/VXN.parquet").reindex(q.index).ffill()
    rv = r.rolling(21).std()
    p_rv = causal_pct(rv.loc["2009-09-14":])
    p_iv = causal_pct(vxn.loc["2009-09-14":])
    D = pd.DataFrame({"p_rv": p_rv, "p_iv": p_iv, "rr": r.shift(-1)}).dropna().loc["2011":]
    D["w"] = weight_from(D.p_rv, "derisk", 1.0)
    D["z_gap"] = C.causal_z(np.log(vxn / (rv * np.sqrt(252) * 100))).reindex(D.index)
    D["z_lrv"] = C.causal_z(np.log(rv * np.sqrt(252) * 100)).reindex(D.index)
    print(f"  corr(RV21 降风险权重 w, z_gap) = {D.w.corr(D.z_gap):+.3f}；corr(w, z_lrv) = {D.w.corr(D.z_lrv):+.3f}；"
          f"corr(z_gap, z_lrv) = {D.z_gap.corr(D.z_lrv):+.3f}")
    hi_iv, hi_rv = D.p_iv >= 0.5, D.p_rv >= 0.5
    D["Q"] = np.where(hi_iv & hi_rv, "HH 恐慌中", np.where(~hi_iv & ~hi_rv, "LL 平静",
              np.where(hi_iv & ~hi_rv, "HL 隐含高·实现低", "LH 隐含低·实现高")))
    years = len(D) / 252
    D["dret"] = (D.w - 1.0) * D.rr                 # 降风险相对满仓的收益差
    D["dsq"] = (D.w ** 2 - 1.0) * D.rr ** 2        # 二阶矩变化（<0 = 风险被削）
    tot_ret, tot_sq = D.dret.sum(), D.dsq.sum()
    print(f"\n  {'象限':20s}{'天数占比':>8}{'平均w':>8}{'QQQ次日bp':>11}{'收益贡献(%/年)':>16}{'占总收益差':>11}{'风险削减占比':>13}")
    for k in ("HH 恐慌中", "LL 平静", "HL 隐含高·实现低", "LH 隐含低·实现高"):
        x = D[D.Q == k]
        print(f"  {k:20s}{len(x)/len(D):>8.1%}{x.w.mean():>8.2f}{x.rr.mean()*1e4:>11.1f}"
              f"{x.dret.sum()/years*100:>16.2f}{x.dret.sum()/tot_ret:>11.0%}{x.dsq.sum()/tot_sq:>13.0%}")
    print(f"  {'合计':20s}{'100%':>8}{D.w.mean():>8.2f}{D.rr.mean()*1e4:>11.1f}{tot_ret/years*100:>16.2f}")
    print("  （收益贡献 = Σ(w−1)·r 的年化；为负说明该象限里降风险损失了收益；风险削减占比 = 二阶矩变化的份额）")

    # sleeve 分象限，按敞口归一化（吸取 GEX 一轮的教训）
    r_s, e_s = C.load_sleeves()["QQQ"]
    # 象限由 t 日收盘的 RV21/VXN 算出（已含 t 日行情），所以只能解释 **t+1 日** 的 sleeve：
    # Q 必须 shift(1) 后再与 r_s 对齐。第一版漏了这个 shift，等于用当天信息解释当天收益。
    S = pd.DataFrame({"u": (r_s / e_s) * 1e4, "r": r_s * 1e4, "e": e_s, "Q": D.Q.shift(1)}).dropna()
    S = S[(S.r != 0)]
    print(f"\n  sleeve 分象限（QQQ，2019-2026，剔不交易日）：原始 bp 与 按敞口归一化的 bp")
    print(f"  {'象限':20s}{'n':>6}{'平均敞口':>9}{'原始bp':>9}{'归一化bp(u)':>13}{'u vs 其余':>11}{'t':>7}")
    for k in ("HH 恐慌中", "LL 平静", "HL 隐含高·实现低", "LH 隐含低·实现高"):
        a, b = S[S.Q == k], S[S.Q != k]
        d = a.u.mean() - b.u.mean()
        se = np.sqrt(a.u.var() / len(a) + b.u.var() / len(b))
        print(f"  {k:20s}{len(a):>6}{a.e.mean():>9.2f}{a.r.mean():>9.2f}{a.u.mean():>13.2f}{d:>11.2f}{d/se:>7.2f}")
    return D


def raw_vs_unit():
    print("\n" + "=" * 112)
    print("敞口混淆检验：同一个信号 z_gap，对 **原始 bp** 与 **按敞口归一化的 u** 分别回归（HAC lag=5）")
    print("=" * 112)
    print(f"  {'标的':6s}{'因变量':16s}{'系数(bp/单位z)':>16}{'t':>7}{'p':>9}{'  corr(z_gap, 敞口e)':>22}")
    for sym in ("QQQ", "SPY"):
        d = C.design(sym).dropna(subset=["z_gap"])
        for nm, y in (("原始 r (bp)", d.r * 1e4), ("归一化 u (bp)", d.u * 1e4)):
            m = ols(y, d[["z_gap"] + DOW], 5)
            print(f"  {sym:6s}{nm:16s}{m.params['z_gap']:>16.2f}{m.tvalues['z_gap']:>7.2f}"
                  f"{fmt_p(m.pvalues['z_gap']):>9}{d.z_gap.corr(d.e):>22.3f}")


if __name__ == "__main__":
    ok_m1, mB = h1_m1()
    econ_res = econ()
    r1, r2, r3 = regime()
    quadrants()
    raw_vs_unit()

    print("\n" + "=" * 112)
    print(f"决定规则（Bonferroni α = {ALPHA:.4f}，方向已预先指定）")
    print("=" * 112)
    b = econ_res[("H2 FOMC 日（计数·QQQ）", "QQQ")]
    bs = econ_res[("H2 FOMC 日（计数·QQQ）", "SPY")]
    print(f"  H2 FOMC：QQQ b={b[0]:+.1f}bp p={fmt_p(b[3])} -> {'通过' if (b[0]>0 and b[3]<ALPHA) else '**未通过**'}；"
          f"SPY 同号 {'是' if bs[0]*b[0]>0 else '否'}")
    c = econ_res[("H3 z_cor（计数·QQQ）", "QQQ")]
    cs = econ_res[("H3 z_cor（计数·QQQ）", "SPY")]
    cc = econ_res[("H3c z_cor | 控制 z_lrv", "QQQ")]
    print(f"  H3 COR1M：QQQ b={c[0]:+.2f}bp p={fmt_p(c[3])} -> {'通过' if (c[0]>0 and c[3]<ALPHA) else '**未通过**'}；"
          f"SPY 同号 {'是' if cs[0]*c[0]>0 else '否'}；控制 RV 水平后 b={cc[0]:+.2f} p={fmt_p(cc[3])}")
    h = econ_res[("H1e z_gap（参考，不计数）", "QQQ")]
    hs = econ_res[("H1e z_gap（参考，不计数）", "SPY")]
    print(f"  H1 M1(VIX 增量信息，计数)：{'通过' if ok_m1 else '**未通过**'}；"
          f"经济端 QQQ b={h[0]:+.2f}bp SPY b={hs[0]:+.2f}bp（参考）")
