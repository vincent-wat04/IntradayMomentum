"""RV21 降风险的权重参数敏感性分析。

动机：上一轮发现 RV21 降风险的全部成本和收益都落在"恐慌进行中(HH)"象限，平均权重被压到 0.40。
但 0.40 是硬编码映射 w = 2·(1−百分位) 的产物（下限 0、上限 2），不是数据给的。
本脚本问：换成别的权重参数，结论会不会变？

事前写死的规则（避免在网格里挑赢家）：
  * **不缩放到均值 1.0**（之前那样做用了全样本信息）。权重参数在事前固定，
    MDD 对照用"同平均敞口的常数权重买入持有"：ΔMDD = MDD(w·r) − MDD(mean(w)·r)。
    Sharpe 对常数缩放不变，所以 ΔSR = SR(w·r) − SR(r)。
  * 成本：每单位 |Δw| 扣 2bp（指数 ETF 的保守值）。
  * 把"降风险"与"加杠杆"拆开：上限 hi=1.0 的那一列 = 只降风险；下限 lo=1.0 的那一行 = 只加杠杆。
  * **不汇报最优格子**。判读只看：曲面形状、正格子占比、默认格子邻居的符号一致性。
  * 这是在已被看过很多次的 SPX/NDX 上做的**描述性**分析，不是确认性检验，p 值不作为门槛。
  * 噪声参照：默认格子的配对区块自助 95% 区间（下文给出），区间之内的差别都不要解读。

参数面板：
  A  线性映射 w = lo + (hi−lo)·(1−p)：lo ∈ {0, .25, .5, .75, 1}，hi ∈ {1, 1.25, 1.5, 2}
  B  阈值映射（纯保险）：p ≥ q 时 w = lo，否则 1：q ∈ {.70,.80,.90,.95}，lo ∈ {0,.25,.5,.75}
  C  信号定义：RV 窗口 ∈ {10,21,42,63} × 百分位窗口 ∈ {252,504,1008}（映射固定为默认 lo=0,hi=2）
样本：SPX、NDX（1992-2026）；每个各三段 —— 全样本 / 2010 后 / 2010 后剔除 COVID。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import round3_common as C
from fear_derisk_study import causal_pct
from stop_significance import block_boot_pair

COST_BP = 2.0
COVID = ("2020-02-01", "2020-05-31")
SAMPLES = [("1992-2026 全部", "1992", "2026", ()),
           ("2010-2026", "2010", "2026", ()),
           ("2010-2026 剔COVID", "2010", "2026", (COVID,))]
LO = (0.0, 0.25, 0.5, 0.75, 1.0)
HI = (1.0, 1.25, 1.5, 2.0)
QS = (0.70, 0.80, 0.90, 0.95)
LOT = (0.0, 0.25, 0.5, 0.75)
RVW = (10, 21, 42, 63)
PW = (252, 504, 1008)
MARKETS = (("SPX", "yahoo/IDX_GSPC.parquet"), ("NDX", "yahoo/IDX_NDX.parquet"))


def sr(x):
    x = pd.Series(x).dropna()
    return float(x.mean() / x.std() * np.sqrt(252))


def mdd(x):
    eq = (1 + pd.Series(x).dropna()).cumprod()
    return float((eq / eq.cummax() - 1).min())


def pct_w(x: pd.Series, win: int) -> pd.Series:
    """因果百分位，窗口可调（含当日，向量化）。与 fear_derisk_study.causal_pct 在 win=504 时逐位一致。"""
    minp = win // 2
    v = x.to_numpy(float)
    pad = np.concatenate([np.full(win - 1, np.nan), v])
    W = np.lib.stride_tricks.sliding_window_view(pad, win)
    with np.errstate(invalid="ignore"):
        less = (W < v[:, None]).sum(1)
    fin = np.isfinite(W).sum(1)
    ok = (fin >= minp) & np.isfinite(v)
    out = np.full(len(v), np.nan)
    out[ok] = less[ok] / fin[ok]
    return pd.Series(out, index=x.index)


class Mkt:
    def __init__(self, nm, rel):
        self.nm = nm
        px = C.load_close(rel).dropna()
        self.r = px.pct_change()
        self.rr = self.r.shift(-1)              # 信号 t 作用于 t+1
        self._c = {}

    def pct(self, rvwin=21, pwin=504):
        k = (rvwin, pwin)
        if k not in self._c:
            self._c[k] = pct_w(self.r.rolling(rvwin).std(), pwin)
        return self._c[k]


def w_linear(p, lo, hi):
    return (lo + (hi - lo) * (1.0 - p)).fillna(1.0)        # 预热期中性


def w_thresh(p, q, lo):
    w = pd.Series(np.where(p >= q, lo, 1.0), index=p.index)
    return w


def evaluate(m: Mkt, w: pd.Series, sample, cost_bp=None):
    cost_bp = COST_BP if cost_bp is None else cost_bp
    _, lo_, hi_, ex = sample
    turn = w.diff().abs().fillna(0.0)
    x = w * m.rr - cost_bp / 1e4 * turn
    a, b, ww = m.rr.loc[lo_:hi_], x.loc[lo_:hi_], w.loc[lo_:hi_]
    tt = turn.loc[lo_:hi_]
    for s, e in ex:
        k = (a.index >= s) & (a.index <= e)
        a, b, ww, tt = a[~k], b[~k], ww[~k], tt[~k]
    ok = a.notna() & b.notna()
    a, b, ww, tt = a[ok], b[ok], ww[ok], tt[ok]
    c = float(ww.mean())
    return dict(dsr=sr(b) - sr(a), dmdd=mdd(b) - mdd(c * a), meanw=c,
                turn=float(tt.sum() / (len(a) / 252)), a=a, b=b)


def table(df, title, fmt, rowlab, collab, mark=None):
    print(f"    {title}")
    print("    " + f"{rowlab}\\{collab}".ljust(12) + "".join(f"{c:>9}" for c in df.columns))
    for i, row in df.iterrows():
        cells = []
        for c, v in row.items():
            s = fmt(v)
            if mark and (i, c) == mark:
                s = "[" + s.strip() + "]"
            cells.append(f"{s:>9}")
        print("    " + f"{i}".ljust(12) + "".join(cells))


def main(args=None):
    panels = (args.panels if args else "ABC").upper()
    mk = {nm: Mkt(nm, rel) for nm, rel in MARKETS}

    # ---------- 自检 1：向量化百分位 == 原实现
    x = mk["SPX"].r.rolling(21).std()
    d = (pct_w(x, 504) - causal_pct(x)).abs().max()
    print(f"[自检1] 向量化百分位 vs causal_pct(504)：最大差 {d:.2e} -> {'一致' if d < 1e-12 else '**不一致**'}")
    assert d < 1e-12

    # ---------- 自检 2：复现上一轮默认格子（成本 0，SPX 1992-2026 ΔSR 应为 −0.093）
    w0 = w_linear(mk["SPX"].pct(21, 504), 0.0, 2.0)
    e0 = evaluate(mk["SPX"], w0, SAMPLES[0], cost_bp=0.0)
    print(f"[自检2] 复现上一轮：SPX 1992-2026 默认(0,2) 成本=0 的 ΔSR = {e0['dsr']:+.3f}（上一轮 −0.093）")
    assert abs(e0["dsr"] - (-0.093)) < 0.005, "无法复现上一轮数字，停止"
    print(f"        去掉均值匹配后的实际平均权重 = {e0['meanw']:.3f}（之前被强制缩放到 1.000）")

    rows = []
    res_a, res_b = {}, {}

    if args is not None and args.cell:
        # 单格子模式：python sens_rv_weights.py --cell linear:0,1  或  --cell thresh:0.9,0.5
        kind, pr = args.cell.split(":"); a_, b_ = [float(x) for x in pr.split(",")]
        print(f"\n单格子 {args.cell}（成本 {COST_BP}bp，RV{args.rvw_one} / 百分位窗 {args.pw_one}）")
        print(f"  {'标的/样本':26s}{'ΔSR':>8}{'ΔMDD(pp)':>10}{'平均w':>8}{'年换手':>8}{'ΔSR 95% 区间':>24}")
        for nm in ("SPX", "NDX"):
            p_ = mk[nm].pct(args.rvw_one, args.pw_one)
            w_ = w_linear(p_, a_, b_) if kind == "linear" else w_thresh(p_, a_, b_)
            for smp in SAMPLES:
                e = evaluate(mk[nm], w_, smp)
                real, dist = block_boot_pair(e["b"], e["a"], nb=1000, L=10, seed=3)
                lo_, hi_ = np.percentile(dist, [2.5, 97.5])
                print(f"  {nm+' '+smp[0]:26s}{e['dsr']:>+8.3f}{e['dmdd']*100:>+10.1f}{e['meanw']:>8.2f}"
                      f"{e['turn']:>8.1f}{f'[{lo_:+.3f}, {hi_:+.3f}]':>24}")
        return

    # ================= 面板 A =================
    print("\n" + "=" * 110)
    print("面板 A · 线性映射 w = lo + (hi−lo)(1−p)。行=下限 lo（最高波动时的权重），列=上限 hi（最低波动时的权重）")
    print("       hi=1 的列 = 只降风险；lo=1 的行 = 只加杠杆；方括号 = 上一轮的默认 (0, 2)")
    print("=" * 110)
    for nm in (("SPX", "NDX") if "A" in panels else ()):
        for smp in SAMPLES:
            dsr = pd.DataFrame(index=LO, columns=HI, dtype=float)
            dmd = dsr.copy(); mw = dsr.copy(); tn = dsr.copy()
            for lo in LO:
                for hi in HI:
                    e = evaluate(mk[nm], w_linear(mk[nm].pct(), lo, hi), smp)
                    dsr.loc[lo, hi], dmd.loc[lo, hi] = e["dsr"], e["dmdd"]
                    mw.loc[lo, hi], tn.loc[lo, hi] = e["meanw"], e["turn"]
                    rows.append(dict(panel="A", mkt=nm, sample=smp[0], p1=lo, p2=hi,
                                     dsr=e["dsr"], dmdd=e["dmdd"], meanw=e["meanw"], turn=e["turn"]))
            res_a[(nm, smp[0])] = (dsr, dmd, mw, tn)
            print(f"\n  ── {nm} · {smp[0]} ──")
            table(dsr, "ΔSharpe（降风险 − 买入持有）", lambda v: f"{v:+.3f}", "lo", "hi", (0.0, 2.0))
            table(dmd * 100, "ΔMDD（百分点，>0 = 回撤更浅；对照 = 同平均敞口的常数权重）",
                  lambda v: f"{v:+.1f}", "lo", "hi", (0.0, 2.0))
    print("\n  ── 面板 A 摘要（排除 lo=1,hi=1 这个恒等格子）──")
    print(f"  {'样本':24s}{'ΔSR>0 占比':>11}{'ΔSR>+.15':>10}{'ΔSR<−.15':>10}{'ΔMDD>0 占比':>13}"
          f"{'只降风险列 均值ΔSR':>18}{'只加杠杆行 均值ΔSR':>18}{'默认邻居同号':>13}")
    for (nm, s_), (dsr, dmd, mw, tn) in res_a.items():
        cells = [(lo, hi) for lo in LO for hi in HI if not (lo == 1.0 and hi == 1.0)]
        v = np.array([dsr.loc[c] for c in cells]); m_ = np.array([dmd.loc[c] for c in cells])
        los = [x for x in LO if x != 1.0]; his = [x for x in HI if x != 1.0]
        dr = dsr.loc[los, 1.0].mean() if (1.0 in HI and los) else float("nan")
        lv = dsr.loc[1.0, his].mean() if (1.0 in LO and his) else float("nan")
        need = [(0.0, 2.0), (0.0, 1.5), (0.25, 2.0), (0.25, 1.5)]
        if all((a in LO and b in HI) for a, b in need):
            same = all(np.sign(dsr.loc[a, b]) == np.sign(dsr.loc[0.0, 2.0]) for a, b in need[1:])
            same = "是" if same else "否"
        else:
            same = "n/a"
        print(f"  {nm+' '+s_:24s}{(v>0).mean():>11.0%}{(v>0.15).mean():>10.0%}{(v<-0.15).mean():>10.0%}"
              f"{(m_>0).mean():>13.0%}{dr:>18.3f}{lv:>18.3f}{same:>13}")

    # ================= 面板 B =================
    print("\n" + "=" * 110)
    print("面板 B · 阈值映射（纯保险）：RV21 百分位 ≥ q 时权重压到 lo，其余满仓 1.0。行=lo，列=q")
    print("=" * 110)
    for nm in (("SPX", "NDX") if "B" in panels else ()):
        for smp in SAMPLES:
            dsr = pd.DataFrame(index=LOT, columns=QS, dtype=float); dmd = dsr.copy(); tn = dsr.copy()
            for lo in LOT:
                for q in QS:
                    e = evaluate(mk[nm], w_thresh(mk[nm].pct(), q, lo), smp)
                    dsr.loc[lo, q], dmd.loc[lo, q], tn.loc[lo, q] = e["dsr"], e["dmdd"], e["turn"]
                    rows.append(dict(panel="B", mkt=nm, sample=smp[0], p1=lo, p2=q,
                                     dsr=e["dsr"], dmdd=e["dmdd"], meanw=e["meanw"], turn=e["turn"]))
            res_b[(nm, smp[0])] = (dsr, dmd, tn)
            print(f"\n  ── {nm} · {smp[0]} ──")
            table(dsr, "ΔSharpe", lambda v: f"{v:+.3f}", "lo", "q")
            table(dmd * 100, "ΔMDD（百分点）", lambda v: f"{v:+.1f}", "lo", "q")
    print("\n  ── 面板 B 摘要 ──")
    print(f"  {'样本':24s}{'ΔSR>0 占比':>11}{'ΔSR>+.15':>10}{'ΔSR<−.15':>10}{'ΔMDD>0 占比':>13}{'年换手(中位)':>13}")
    for (nm, s), (dsr, dmd, tn) in res_b.items():
        v = dsr.to_numpy().ravel(); m_ = dmd.to_numpy().ravel()
        print(f"  {nm+' '+s:24s}{(v>0).mean():>11.0%}{(v>0.15).mean():>10.0%}{(v<-0.15).mean():>10.0%}"
              f"{(m_>0).mean():>13.0%}{np.median(tn.to_numpy()):>13.1f}")

    # ================= 面板 C =================
    print("\n" + "=" * 110)
    print("面板 C · 信号定义：RV 窗口 × 百分位窗口（映射固定为默认 lo=0, hi=2）。单元格 = ΔSharpe / ΔMDD(pp)")
    print("=" * 110)
    short = {"2010-2026": "2010+", "2010-2026 剔COVID": "2010+剔COVID"}
    hdr = "    " + "RV窗 / 百分位窗".ljust(16) + "".join(f"{nm+' '+short[s[0]]:>22}" for nm in ("SPX", "NDX") for s in SAMPLES[1:])
    print(hdr)
    for rvw in (RVW if "C" in panels else ()):
        for pw in PW:
            line = "    " + f"RV{rvw:<3d} / {pw:<5d}".ljust(16)
            for nm in ("SPX", "NDX"):
                for smp in SAMPLES[1:]:
                    e = evaluate(mk[nm], w_linear(mk[nm].pct(rvw, pw), 0.0, 2.0), smp)
                    line += f"{e['dsr']:>+12.3f}/{e['dmdd']*100:+6.1f}".rjust(22)
                    rows.append(dict(panel="C", mkt=nm, sample=smp[0], p1=rvw, p2=pw,
                                     dsr=e["dsr"], dmdd=e["dmdd"], meanw=e["meanw"], turn=e["turn"]))
            print(line)

    # ================= 噪声参照 =================
    print("\n" + "=" * 110)
    print("噪声参照：三个代表格子的 ΔSR 配对区块自助 95% 区间（区块 10 天、1000 次）")
    print("=" * 110)
    cells = (("默认线性 (0,2)", lambda m: w_linear(m.pct(), 0.0, 2.0)),
             ("只降风险 (0,1)", lambda m: w_linear(m.pct(), 0.0, 1.0)),
             ("阈值保险 q=.90,lo=.5", lambda m: w_thresh(m.pct(), 0.90, 0.5)))
    print(f"  {'格子':22s}{'标的/样本':26s}{'ΔSR':>8}{'95% 区间':>22}{'半宽':>7}")
    for lab, fn in cells:
        for nm in ("SPX", "NDX"):
            for smp in SAMPLES:
                e = evaluate(mk[nm], fn(mk[nm]), smp)
                real, dist = block_boot_pair(e["b"], e["a"], nb=1000, L=10, seed=3)
                lo_, hi_ = np.percentile(dist, [2.5, 97.5])
                print(f"  {lab:22s}{nm+' '+smp[0]:26s}{real:>+8.3f}{f'[{lo_:+.3f}, {hi_:+.3f}]':>22}{(hi_-lo_)/2:>7.3f}")

    pd.DataFrame(rows).to_csv("outputs/full/intraday/sens_rv_weights.csv", index=False)
    pd.to_pickle(dict(a=res_a, b=res_b), "outputs/full/intraday/sens_rv_weights.pkl")
    print("\n表 -> outputs/full/intraday/sens_rv_weights.csv")


def _floats(x):
    return tuple(float(t) for t in x.split(","))


def _ints(x):
    return tuple(int(t) for t in x.split(","))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="RV21 降风险权重参数敏感性分析（全部参数可改）")
    ap.add_argument("--lo", type=_floats, default=LO, help="面板A 下限 lo 网格（最高波动时的权重），逗号分隔")
    ap.add_argument("--hi", type=_floats, default=HI, help="面板A 上限 hi 网格（最低波动时的权重）")
    ap.add_argument("--q", type=_floats, default=QS, help="面板B 阈值 q 网格（RV 百分位）")
    ap.add_argument("--lo-thresh", type=_floats, default=LOT, help="面板B 触发后的权重网格")
    ap.add_argument("--rvw", type=_ints, default=RVW, help="面板C RV 窗口网格（天）")
    ap.add_argument("--pw", type=_ints, default=PW, help="面板C 百分位窗口网格（天）")
    ap.add_argument("--cost-bp", type=float, default=COST_BP, help="每单位 |Δw| 的成本（bp）")
    ap.add_argument("--panels", default="ABC", help="要跑哪些面板，例如 A 或 AB")
    ap.add_argument("--cell", default="", help="只算单个格子：linear:lo,hi 或 thresh:q,lo")
    ap.add_argument("--rvw-one", type=int, default=21, help="单格子模式的 RV 窗口")
    ap.add_argument("--pw-one", type=int, default=504, help="单格子模式的百分位窗口")
    a = ap.parse_args()
    LO, HI, QS, LOT, RVW, PW, COST_BP = a.lo, a.hi, a.q, a.lo_thresh, a.rvw, a.pw, a.cost_bp
    main(a)
