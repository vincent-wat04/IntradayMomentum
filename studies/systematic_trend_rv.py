"""RV × 趋势 的系统化组合 + 参数证据 + 做空 —— 按 docs/PROPOSAL_systematic.md 冻结的规则。

信号全部只用 t 日收盘及以前；权重 w_t 作用于 t→t+1 的收益。
收益 = w·r_tr + (1−w)·rf − 2bp·|Δw| − 0.5%/年·max(−w,0)/252 ；夏普为超额收益年化。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import round3_common as C
import sens_rv_weights as S

OUTD = Path(__file__).resolve().parent / "outputs" / "full" / "intraday"
# 名称, 路径, 股息率 q, 绝对波动目标, 组别
MARKETS = (("SPX", "yahoo/IDX_GSPC.parquet", 0.018, 0.15, "U1"), ("NDX", "yahoo/IDX_NDX.parquet", 0.009, 0.20, "U1"),
           ("RUT", "yahoo/IDX_RUT.parquet", 0.012, 0.20, "U1"), ("NYA", "yahoo/IDX_NYA.parquet", 0.020, 0.15, "U1"),
           ("EWJ", "yahoo/EWJ.parquet", 0.0, 0.20, "U2"), ("EFA", "yahoo/EFA.parquet", 0.0, 0.15, "U2"),
           ("EEM", "yahoo/EEM.parquet", 0.0, 0.20, "U2"))
CORE5 = ("SPX", "NDX", "EWJ", "EFA", "EEM")
SMA_N, MOM_N, EOFF = (50, 100, 150, 200, 250), (63, 126, 189, 252), (-1.0, -0.5, 0.0, 0.5)
COST, SHORT_FEE = 2e-4, 0.005
SUBP = (("≤2003", None, "2003-12-31"), ("2004-14", "2004-01-01", "2014-12-31"), ("2015-26", "2015-01-01", None))


class Mk:
    def __init__(self, nm, rel, q, sig, grp):
        self.nm, self.grp = nm, grp
        px = C.load_close(rel).dropna()
        self.px = px
        r = px.pct_change()
        self.r = r
        self.rr = r.shift(-1) + q / 252.0                         # t 日权重 -> t+1 总收益
        irx = C.load_close("yahoo/IDX_IRX.parquet").reindex(px.index).ffill()
        self.rfn = irx / 100.0 / 252.0
        self.p = S.pct_w(r.rolling(21).std(), 504)
        self.R = {"none": pd.Series(1.0, index=px.index),
                  "浅蓝": S.w_linear(self.p, 0.5, 1.5),
                  "绝对": (sig / (r.rolling(21).std() * np.sqrt(252))).clip(0.5, 1.5)}
        self.neg = {}
        for n in SMA_N:
            sma = px.rolling(n).mean()
            self.neg[("SMA", n)] = (px < sma).where(sma.notna(), False)
        for n in MOM_N:
            m = px.shift(21) / px.shift(n) - 1
            self.neg[("MOM", n)] = (m < 0).where(m.notna(), False)
        ok = self.p.notna() & px.rolling(250).mean().notna() & px.shift(252).notna()
        self.t0 = ok.index[ok.to_numpy().argmax()]
        self.t0 = max(self.t0, pd.Timestamp("1992-01-01"))
        self.idx = self.rr.loc[self.t0:].dropna().index

    def w(self, R, T=None, e=1.0, mode="prod"):
        base = self.R[R]
        if T is None or e == 1.0:
            return base
        neg = self.neg[T]
        return base * np.where(neg, e, 1.0) if mode == "prod" else pd.Series(np.where(neg, e, base), index=base.index)

    def run(self, w):
        w = pd.Series(w, index=self.px.index)
        turn = w.diff().abs().fillna(0.0)
        tot = w * self.rr + (1 - w) * self.rfn - COST * turn - SHORT_FEE / 252 * (-w).clip(lower=0)
        return tot, tot - self.rfn, turn


def ann_sr(x):
    x = pd.Series(x).dropna()
    return float(x.mean() / x.std() * np.sqrt(252)) if len(x) > 30 and x.std() > 0 else np.nan


def mdd_tot(x):
    eq = (1 + pd.Series(x).dropna()).cumprod()
    return float((eq / eq.cummax() - 1).min())


def cagr(x):
    x = pd.Series(x).dropna()
    return float((1 + x).prod() ** (252 / len(x)) - 1)


class Boot:
    """同一市场所有行共用同一组块自助下标（成对）。"""

    def __init__(self, n, nb=4000, L=10, seed=0):
        rng = np.random.default_rng(seed)
        nblk = int(np.ceil(n / L))
        st = rng.integers(0, n, (nb, nblk))
        self.idx = ((st[:, :, None] + np.arange(L)[None, None, :]).reshape(nb, -1)[:, :n] % n).astype(np.int32)

    def dsr(self, a, b):
        d = np.column_stack([a, b])
        s = d[self.idx]                                              # nb × n × 2
        m, sd = s.mean(1), s.std(1)
        v = (m[:, 0] / sd[:, 0] - m[:, 1] / sd[:, 1]) * np.sqrt(252)
        return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), float((v <= 0).mean())


def row_metrics(M, w, idx):
    tot, ex, turn = M.run(w)
    t, e = tot.loc[idx], ex.loc[idx]
    mw = float(pd.Series(w, index=M.px.index).loc[idx].mean())
    ctrl = (mw * M.rr + (1 - mw) * M.rfn - M.rfn).loc[idx]
    c, m = cagr(t), mdd_tot(t)
    return dict(cagr=c, sr=ann_sr(e), mdd=m, calmar=c / abs(m), mw=mw, turn=float(turn.loc[idx].sum() / (len(idx) / 252)),
                dsr=ann_sr(e) - ann_sr(ctrl), e=e.to_numpy(), ctrl=ctrl.to_numpy(), tot=t, ex=e)


def sub(x, a, b):
    return x.loc[a:b] if a or b else x


# ------------------------------------------------------------------ Q1
def q1(Ms):
    print("=" * 118)
    print("Q1  RV 与趋势是否独立？（t 日信号 -> t+1 收益；趋势强度 = px/SMA200−1，向下 = 收盘<SMA200）")
    print("=" * 118)
    print(f"{'市场':5s}{'Spearman(pRV,趋势强度)':>24}{'P(向下|RV低)':>14}{'P(向下|RV中)':>14}{'P(向下|RV高)':>14}"
          f"{'b3 交互(bp)':>13}{'t':>7}{'  DiD夏普':>10}")
    rows = []
    for M in Ms:
        idx = M.idx
        sma = M.px.rolling(200).mean()
        strength = (M.px / sma - 1).reindex(idx)
        p = M.p.reindex(idx)
        neg = M.neg[("SMA", 200)].reindex(idx).astype(float)
        ok = strength.notna() & p.notna()
        rho = spearmanr(p[ok], strength[ok])[0]
        pl = [float(neg[ok & (p >= a) & (p < b)].mean()) for a, b in ((0, .33), (.33, .67), (.67, 1.01))]
        hi = (p >= .67).astype(float)
        X = pd.DataFrame({"hi": hi, "neg": neg, "int": hi * neg}).assign(const=1.0)
        y = M.rr.reindex(idx) * 1e4
        d = pd.concat([y.rename("y"), X], axis=1).dropna()
        f = sm.OLS(d.y, d[["const", "hi", "neg", "int"]]).fit(cov_type="HAC", cov_kwds={"maxlags": 10})
        bh = row_metrics(M, M.R["none"], idx)["sr"]
        h0 = row_metrics(M, M.w("浅蓝"), idx)["sr"]
        tr = row_metrics(M, M.w("none", ("SMA", 200), 0.5), idx)["sr"]
        cb = row_metrics(M, M.w("浅蓝", ("SMA", 200), 0.5), idx)["sr"]
        did = cb - h0 - tr + bh
        print(f"{M.nm:5s}{rho:>24.3f}{pl[0]:>14.1%}{pl[1]:>14.1%}{pl[2]:>14.1%}{f.params['int']:>13.2f}{f.tvalues['int']:>7.2f}{did:>+10.3f}")
        rows.append(dict(mkt=M.nm, rho=rho, p_low=pl[0], p_mid=pl[1], p_high=pl[2], b3=f.params['int'], t3=f.tvalues['int'], did=did))
    print("  解读：rho<0 ⇒ 波动高时多半处在下跌趋势（杠杆效应）；P(向下|RV高) ≫ P(向下|RV低) ⇒ 两个信号**不独立**；"
          "DiD≈0 ⇒ 夏普贡献近似可加。")
    return rows


# ------------------------------------------------------------------ headline
def headline(M, out):
    idx = M.idx
    B = Boot(len(idx))
    print("\n" + "=" * 130)
    print(f"{M.nm}（{M.grp}） {idx[0].date()} → {idx[-1].date()}  · 乘积式 w=R×g(T)，覆盖式 w=g(T)  ·  ΔSR 对照 = 同平均权重静态组合")
    print("=" * 130)
    print(f"{'方案':30s}{'CAGR':>8}{'夏普':>7}{'MDD':>8}{'Calmar':>8}{'均w':>6}{'换手':>6}{'ΔSR':>8}{'95%CI':>16}{'P(Δ≤0)':>8}")
    specs = [("B&H", dict(R="none")), ("H0 仅RV(浅蓝)", dict(R="浅蓝")), ("仅RV(绝对目标)", dict(R="绝对"))]
    for T in (("SMA", 200), ("MOM", 252)):
        for e in (0.5, 0.0, -0.5, -1.0):
            specs.append((f"{T[0]}{T[1]} 仅趋势 e={e:+.1f}", dict(R="none", T=T, e=e)))
        for e in (0.5, 0.0, -0.5, -1.0):
            specs.append((f"{T[0]}{T[1]} 浅蓝×乘积 e={e:+.1f}", dict(R="浅蓝", T=T, e=e)))
        specs.append((f"{T[0]}{T[1]} 浅蓝·覆盖 e=+0.5", dict(R="浅蓝", T=T, e=0.5, mode="override")))
        for e in (0.5, -0.5):
            specs.append((f"{T[0]}{T[1]} 绝对×乘积 e={e:+.1f}", dict(R="绝对", T=T, e=e)))
    for lab, kw in specs:
        w = M.w(kw["R"], kw.get("T"), kw.get("e", 1.0), kw.get("mode", "prod"))
        m = row_metrics(M, w, idx)
        lo, hi, pz = B.dsr(m["e"], m["ctrl"]) if lab != "B&H" else (0, 0, 1)
        print(f"{lab:30s}{m['cagr']:>8.1%}{m['sr']:>7.2f}{m['mdd']:>8.1%}{m['calmar']:>8.2f}{m['mw']:>6.2f}{m['turn']:>6.1f}"
              f"{m['dsr']:>+8.3f}{f'[{lo:+.2f},{hi:+.2f}]':>16}{pz:>8.2f}")
        out.append(dict(mkt=M.nm, label=lab, **{k: m[k] for k in ("cagr", "sr", "mdd", "calmar", "mw", "turn", "dsr")},
                        ci_lo=lo, ci_hi=hi, **{"R": kw["R"], "T": str(kw.get("T")), "e": kw.get("e", 1.0), "mode": kw.get("mode", "prod")}))


# ------------------------------------------------------------------ Q2(i) plateau
def plateau(Ms):
    print("\n" + "=" * 130)
    print("Q2(i) 平台性：R=浅蓝，乘积式；网格 = 9 个趋势规则；格子'合格' ⇔ 子期夏普 ≥ H0−0.02 且 子期 MDD 优于 H0")
    print("      表中数字 = 合格格子数 / 9。列 = 关闭态 e_off；每个子期一块")
    print("=" * 130)
    rows = []
    trends = [("SMA", n) for n in SMA_N] + [("MOM", n) for n in MOM_N]
    for plab, a, b in SUBP:
        print(f"\n  子期 {plab}")
        print(f"  {'市场':6s}" + "".join(f"{'e=' + format(e, '+.1f'):>9}" for e in EOFF) + "      其中 SMA200 / MOM252 在各 e 下是否合格")
        for M in Ms:
            idx = M.idx
            ii = idx[(idx >= (pd.Timestamp(a) if a else idx[0])) & (idx <= (pd.Timestamp(b) if b else idx[-1]))]
            if len(ii) < 500:
                print(f"  {M.nm:6s}   （样本不足）"); continue
            h0 = row_metrics(M, M.w("浅蓝"), ii)
            line, flags = f"  {M.nm:6s}", ""
            for e in EOFF:
                good = 0
                fl = ""
                for T in trends:
                    m = row_metrics(M, M.w("浅蓝", T, e), ii)
                    g = (m["sr"] >= h0["sr"] - 0.02) and (m["mdd"] > h0["mdd"])
                    good += g
                    if T in (("SMA", 200), ("MOM", 252)):
                        fl += "✓" if g else "✗"
                    rows.append(dict(sub=plab, mkt=M.nm, e=e, T=str(T), good=bool(g), sr=m["sr"], mdd=m["mdd"], h0sr=h0["sr"], h0mdd=h0["mdd"]))
                line += f"{good:>7d}/9"; flags += f" {fl}"
            print(line + "     " + flags)
    return rows


# ------------------------------------------------------------------ Q2(ii) walk-forward
def walk_forward(Ms):
    print("\n" + "=" * 130)
    print("Q2(ii) 向前滚动：每年 12 月底，用截至当时的全部历史（≥5 年训练），在 37 个候选里按训练期夏普选一个，下一年使用")
    print("       候选 = 仅RV(浅蓝) + {9 个趋势规则 × e_off∈{−1,−.5,0,.5}} 的乘积式")
    print("=" * 130)
    rows, picks_all = [], {}
    for M in Ms:
        idx = M.idx
        cand = {"仅RV": M.w("浅蓝")}
        for T in [("SMA", n) for n in SMA_N] + [("MOM", n) for n in MOM_N]:
            for e in EOFF:
                cand[f"{T[0]}{T[1]}|{e:+.1f}"] = M.w("浅蓝", T, e)
        EX = {k: M.run(w)[1].loc[idx] for k, w in cand.items()}
        TOT = {k: M.run(w)[0].loc[idx] for k, w in cand.items()}
        years = sorted(set(idx.year))
        first = [y for y in years if (idx[idx.year < y].size / 252) >= 5.0]
        if not first:
            continue
        oos_t, oos_e, picks = [], [], {}
        for y in first:
            tr = idx[idx.year < y]
            sc = {k: ann_sr(EX[k].loc[tr]) for k in cand}
            best = max(cand, key=lambda k: sc[k] if sc[k] == sc[k] else -9.0)
            ty = idx[idx.year == y]
            oos_t.append(TOT[best].loc[ty]); oos_e.append(EX[best].loc[ty]); picks[y] = best
        oos_t, oos_e = pd.concat(oos_t), pd.concat(oos_e)
        oi = oos_t.index
        ref = {"WF 滚动选择": (oos_t, oos_e)}
        for lab, k in (("固定 H0 仅RV", "仅RV"), ("固定 SMA200|+0.5", "SMA200|+0.5")):
            ref[lab] = (TOT[k].loc[oi], EX[k].loc[oi])
        bt, be, _ = M.run(M.R["none"]); ref["B&H"] = (bt.loc[oi], be.loc[oi])
        print(f"\n  {M.nm}  OOS {oi[0].date()} → {oi[-1].date()}（{len(first)} 年）")
        print(f"    {'':22s}{'CAGR':>8}{'夏普':>7}{'MDD':>8}{'Calmar':>8}")
        for lab, (t, e) in ref.items():
            c, m = cagr(t), mdd_tot(t)
            print(f"    {lab:22s}{c:>8.1%}{ann_sr(e):>7.2f}{m:>8.1%}{c / abs(m):>8.2f}")
            rows.append(dict(mkt=M.nm, strat=lab, cagr=c, sr=ann_sr(e), mdd=m, calmar=c / abs(m)))
        pk = pd.Series(picks)
        print("    每年所选： " + " ".join(f"{y % 100:02d}:{v.replace('|', '/')}" for y, v in list(picks.items())[:40]))
        cnt = pk.value_counts()
        n_short = int(pk.str.contains("|-", regex=False).sum())
        print("    选择频次：" + "；".join(f"{k.replace('|', '/')}×{v}" for k, v in cnt.head(5).items())
              + f"；选中 SMA200 的年数 {int(pk.str.startswith('SMA200').sum())}/{len(pk)}；选中任何趋势规则（非仅RV）{int((pk != '仅RV').sum())}/{len(pk)}；"
              + f"选中 e<0（做空）{n_short}/{len(pk)}")
        picks_all[M.nm] = pk
    return rows


def adoption(Ms, headline_rows):
    """采纳线 (a)(b)(c)：对每个头条格 (T, e_off) 自动判定；(d) 在滚动 OOS 表里看。"""
    MK = {M.nm: M for M in Ms}
    cache = {}

    def st(mk, T, e):
        k = (mk, T, e)
        if k not in cache:
            cache[k] = row_metrics(MK[mk], MK[mk].w("浅蓝", T, e), MK[mk].idx)
        return cache[k]

    h0 = {mk: row_metrics(MK[mk], MK[mk].w("浅蓝"), MK[mk].idx) for mk in CORE5}

    def ab(T, e):
        a = sum(st(mk, T, e)["dsr"] >= 0 for mk in CORE5)
        b = sum(st(mk, T, e)["mdd"] > h0[mk]["mdd"] for mk in CORE5)
        return a, b

    NB = {("SMA", 200): [("SMA", 150), ("SMA", 250)], ("MOM", 252): [("MOM", 189)]}
    print("\n" + "=" * 130)
    print("采纳线判定：头条格 = R=浅蓝 乘积式；CORE5 = SPX/NDX/EWJ/EFA/EEM")
    print("  (a) ΔSR(vs 对照)≥0 的市场数 ≥4/5；(b) MDD 优于 H0 的市场数 =5/5；(c) 相邻回看长度同 e_off 也各自满足 (a)(b)；(d) 见 Q2(ii)")
    print("=" * 130)
    print(f"{'头条格':16s}{'(a)':>6}{'(b)':>6}{'邻近格 (a)/(b)':>34}{'(c)':>6}{'  (a)(b)(c)':>12}")
    res = []
    for T in (("SMA", 200), ("MOM", 252)):
        for e in (0.5, 0.0, -0.5, -1.0):
            a, b = ab(T, e)
            nbs = [(n, *ab(n, e)) for n in NB[T]]
            c = all(x >= 4 and y == 5 for _, x, y in nbs)
            ok = a >= 4 and b == 5 and c
            nbtxt = "  ".join(f"{n[0]}{n[1]}:{x}/{y}" for n, x, y in nbs)
            print(f"{T[0]}{T[1]} e={e:+.1f}".ljust(16) + f"{a:>4d}/5{b:>4d}/5{nbtxt:>34}{('✓' if c else '✗'):>6}{('过' if ok else '否'):>10}")
            res.append(dict(T=str(T), e=e, a=a, b=b, c=c, a_b_c=ok))
    return res


def main():
    Ms = [Mk(*m) for m in MARKETS]
    r1 = q1(Ms)
    out = []
    for M in Ms:
        headline(M, out)
    adopt = adoption(Ms, out)
    pl = plateau(Ms)
    wf = walk_forward(Ms)
    pd.DataFrame(r1).to_csv(OUTD / "systematic_q1.csv", index=False)
    pd.DataFrame(out).to_csv(OUTD / "systematic_headline.csv", index=False)
    pd.DataFrame(pl).to_csv(OUTD / "systematic_plateau.csv", index=False)
    pd.DataFrame(wf).to_csv(OUTD / "systematic_walkforward.csv", index=False)
    pd.DataFrame(adopt).to_csv(OUTD / "systematic_adoption.csv", index=False)


if __name__ == "__main__":
    main()
