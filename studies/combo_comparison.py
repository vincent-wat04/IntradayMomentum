"""静态再配比 × RV 降风险 的组合业绩对比。

组合 = q · w_t · QQQ + sw · Sleeve
  q, sw   静态权重（QQQ、Sleeve）。原始 tearsheet 的"组合"就是 (q, sw) = (1, 1)
  w_t     RV 降风险的 QQQ 腿权重（作用于 QQQ 腿；Sleeve 不动）。信号 t−1 收盘 -> 作用于 t 日
Sleeve = 实盘配置 ⑦（多空 band+尾0.5 无周一，TQQQ 成本），样本 2019-2026。

公平比较：RV 降风险会降低 QQQ 腿的平均敞口，所以
  * 每个组合都报告 QQQ 腿的平均敞口 q·mean(w)；
  * 每个 RV 组合配一个"同平均敞口的纯静态对照"：QQQ 权重 = q·mean(w)、sleeve 权重 = sw。
    真正回答"RV 降风险在静态再配比之上有没有增量"的是和这个对照的差。
成本：每单位 |Δw| 扣 2bp（按 QQQ 腿名义额）。

用法：
  python combo_comparison.py
  python combo_comparison.py --static 1:1,0.6:1.4 --rv none,linear:0.5:1.0,thresh:0.9:0.5
RV 语法：none | linear:lo:hi | thresh:q:lo        静态语法：qqq权重:sleeve权重
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import round3_common as C
from cjkfont import use_cjk
from sens_rv_weights import pct_w, w_linear, w_thresh

OUT = Path("outputs/full/intraday")
FIG = Path("outputs/full/figs")
VER = "多空 band+尾0.5 无周一"
PERIODS = {"全": None, "dev": ("2019", "2022"), "test": ("2023", "2026")}
DEFAULT_STATIC = "1:1,0.6:1.4,0.5:1.5"
DEFAULT_RV = "none,linear:0.5:1.0,linear:0.5:1.5,thresh:0.9:0.5,linear:0:2"
COLORS = ["black", "tab:blue", "tab:cyan", "tab:green", "tab:red", "tab:purple", "tab:orange"]


def sr(x):
    x = pd.Series(x).dropna(); return float(x.mean() / x.std() * np.sqrt(252))


def cagr(x):
    x = pd.Series(x).dropna(); return float((1 + x).prod() ** (252 / len(x)) - 1)


def mdd(x):
    eq = (1 + pd.Series(x).dropna()).cumprod(); return float((eq / eq.cummax() - 1).min())


def stats(x):
    m = mdd(x)
    return dict(sharpe=sr(x), cagr=cagr(x), mdd=m, calmar=cagr(x) / abs(m) if m < 0 else np.nan)


def parse_rv(s):
    if s == "none":
        return ("none",)
    k, a, b = s.split(":")
    return (k, float(a), float(b))


def label(spec):
    if spec[0] == "none":
        return "无 RV 降风险"
    if spec[0] == "linear":
        return f"线性 lo={spec[1]:g}, hi={spec[2]:g}"
    return f"阈值 q={spec[1]:g}, lo={spec[2]:g}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--static", default=DEFAULT_STATIC, help="qqq:sleeve，逗号分隔")
    ap.add_argument("--rv", default=DEFAULT_RV, help="none | linear:lo:hi | thresh:q:lo")
    ap.add_argument("--cost-bp", type=float, default=2.0)
    ap.add_argument("--rvw", type=int, default=21)
    ap.add_argument("--pw", type=int, default=504)
    a = ap.parse_args()
    statics = [tuple(float(t) for t in x.split(":")) for x in a.static.split(",")]
    specs = [parse_rv(x) for x in a.rv.split(",")]

    use_cjk(plt)
    cur = pd.read_pickle(OUT / "strategy_comparison_curves.pkl")
    bh = pd.read_pickle(OUT / "strategy_comparison_bh.pkl").dropna()
    bh.index = pd.to_datetime(bh.index)
    s = cur[VER].copy(); s.index = pd.to_datetime(s.index); s = s.reindex(bh.index).fillna(0.0)

    # RV 信号：QQQ 全历史（1999 起）上的因果百分位，所以 2019 年起点已经有充足的预热
    ry = C.load_close("yahoo/QQQ.parquet").pct_change()
    p = pct_w(ry.rolling(a.rvw).std(), a.pw)
    W, MW = {}, {}
    for sp in specs:
        if sp[0] == "none":
            w = pd.Series(1.0, index=p.index)
        elif sp[0] == "linear":
            w = w_linear(p, sp[1], sp[2])
        else:
            w = w_thresh(p, sp[1], sp[2])
        W[sp] = w.shift(1).reindex(bh.index).fillna(1.0)     # t−1 信号 -> t 日
        MW[sp] = float(W[sp].mean())

    def build(q, sw, sp):
        w = W[sp]; turn = w.diff().abs().fillna(0.0)
        return q * w * bh - a.cost_bp / 1e4 * q * turn + sw * s

    def per(x, k):
        return x if PERIODS[k] is None else x.loc[PERIODS[k][0]:PERIODS[k][1]]

    rows, curves, matched = [], {}, {}
    for (q, sw) in statics:
        for sp in specs:
            x = build(q, sw, sp); curves[(q, sw, sp)] = x
            qeq = q * MW[sp]
            xe = qeq * bh + sw * s; matched[(q, sw, sp)] = xe
            st = stats(x); se = stats(xe)
            rows.append(dict(静态=f"{q:g}:{sw:g}", RV=label(sp), 平均QQQ权重=qeq,
                             Sharpe=st["sharpe"], Sharpe_dev=sr(per(x, "dev")), Sharpe_test=sr(per(x, "test")),
                             CAGR=st["cagr"], MDD=st["mdd"], Calmar=st["calmar"],
                             对照Sharpe=se["sharpe"], 对照MDD=se["mdd"], 对照Calmar=se["calmar"],
                             dSharpe=st["sharpe"] - se["sharpe"], dMDD_pp=(st["mdd"] - se["mdd"]) * 100,
                             dCalmar=st["calmar"] - se["calmar"],
                             换手_年=float(W[sp].diff().abs().sum() / (len(bh) / 252)) * q))
    R = pd.DataFrame(rows); R.to_csv(OUT / "combo_comparison.csv", index=False)

    pd.set_option("display.width", 250)
    print("=" * 128)
    print(f"样本 {bh.index[0].date()} ~ {bh.index[-1].date()}；Sleeve = {VER}；RV 信号 RV{a.rvw}/百分位窗 {a.pw}；成本 {a.cost_bp}bp")
    print("=" * 128)
    print(f"{'静态 q:sw':10s}{'RV 降风险':22s}{'均QQQ权重':>9}{'Sharpe':>8}{'dev':>7}{'test':>7}"
          f"{'CAGR':>8}{'MDD':>8}{'Calmar':>8} | {'对照SR':>7}{'对照MDD':>9}{'对照Calmar':>11} | {'ΔSR':>7}{'ΔMDD(pp)':>9}{'ΔCalmar':>9}")
    for _, r in R.iterrows():
        d = r.RV != "无 RV 降风险"
        print(f"{r.静态:10s}{r.RV:22s}{r.平均QQQ权重:>9.2f}{r.Sharpe:>8.3f}{r.Sharpe_dev:>7.2f}{r.Sharpe_test:>7.2f}"
              f"{r.CAGR:>8.1%}{r.MDD:>8.1%}{r.Calmar:>8.2f} | "
              + (f"{r.对照Sharpe:>7.3f}{r.对照MDD:>9.1%}{r.对照Calmar:>11.2f} | {r.dSharpe:>+7.3f}{r.dMDD_pp:>+9.1f}{r.dCalmar:>+9.2f}"
                 if d else f"{'—':>7}{'—':>9}{'—':>11} | {'—':>7}{'—':>9}{'—':>9}"))
        if r.RV == label(specs[-1]):
            print("-" * 128)

    # ======================= 作图 =======================
    nc = max(2, len(statics))
    fig = plt.figure(figsize=(6.6 * nc, 15.5))
    gs = fig.add_gridspec(3, nc, hspace=.33, wspace=.17, height_ratios=[1.15, 1, 1.05])
    SPLIT = pd.Timestamp("2023-01-01")
    for j, (q, sw) in enumerate(statics):
        A = fig.add_subplot(gs[0, j]); D = fig.add_subplot(gs[1, j])
        A.plot((1 + bh).cumprod(), color="gray", ls=":", lw=1.3, label=f"QQQ 买入持有 (SR {sr(bh):.2f}, MDD {mdd(bh):.0%})")
        for k, sp in enumerate(specs):
            x = curves[(q, sw, sp)]; st = stats(x)
            A.plot((1 + x).cumprod(), color=COLORS[k % len(COLORS)], lw=2.3 if sp[0] == "none" else 1.6,
                   ls="--" if (sp[0] == "linear" and sp[1] == 0.0) else "-",
                   label=f"{label(sp)}  SR {st['sharpe']:.2f} | MDD {st['mdd']:.1%} | 均QQQ {q*MW[sp]:.2f}")
            eq = (1 + x).cumprod(); dd = eq / eq.cummax() - 1
            D.plot(dd * 100, color=COLORS[k % len(COLORS)], lw=1.2, alpha=.9,
                   ls="--" if (sp[0] == "linear" and sp[1] == 0.0) else "-")
        for ax in (A, D):
            ax.axvline(SPLIT, color="gray", lw=1, ls="--"); ax.grid(alpha=.3)
        A.set_yscale("log"); A.set_title(f"QQQ × {q:g} + Sleeve × {sw:g}" + ("   （原始 tearsheet 组合）" if (q, sw) == (1.0, 1.0) else ""),
                                          fontsize=12.5, loc="left")
        A.legend(fontsize=7.6, loc="upper left")
        if j == 0:
            A.set_ylabel("累计净值（对数轴）"); D.set_ylabel("回撤 (%)")
        D.set_title("水下回撤", fontsize=10.5, loc="left")

    # 散点：风险-收益 + 静态前沿
    gb = gs[2, :].subgridspec(1, 2, width_ratios=[1.9, 1.0], wspace=0.42)
    S = fig.add_subplot(gb[0, 0])
    for n_, sw in enumerate(sorted({x[1] for x in statics})):
        qs = np.linspace(0.4, 1.3, 19)       # q<0.4 时回撤由 sleeve 本身主导，曲线会回钩，不画
        pts = [(-mdd(q_ * bh + sw * s) * 100, cagr(q_ * bh + sw * s) * 100) for q_ in qs]
        S.plot([t[0] for t in pts], [t[1] for t in pts], color="gray", lw=1.1, ls="-.", alpha=.85)
        S.annotate(f"静态前沿 sleeve={sw:g}", pts[len(pts) // 2 + 3 * n_ - 3], fontsize=7.5, color="dimgray",
                   xytext=(6, -10 - 9 * n_), textcoords="offset points")
    marks = ["o", "s", "^", "D", "v"]
    for j, (q, sw) in enumerate(statics):
        for k, sp in enumerate(specs):
            st = stats(curves[(q, sw, sp)])
            S.scatter(-st["mdd"] * 100, st["cagr"] * 100, s=95, marker=marks[j % 5], c=COLORS[k % len(COLORS)],
                      edgecolor="k", lw=.6, zorder=3)
    S.scatter(-mdd(bh) * 100, cagr(bh) * 100, marker="*", s=200, c="gray", edgecolor="k", zorder=3)
    S.annotate("QQQ 买入持有", (-mdd(bh) * 100, cagr(bh) * 100), fontsize=8, xytext=(6, -10), textcoords="offset points")
    S.set_xlabel("最大回撤 |MDD| (%)  ← 越小越好"); S.set_ylabel("年化收益 CAGR (%)  ↑ 越高越好")
    S.set_title("风险-收益：点在灰色静态前沿的 左/上 方 = 比同风险的纯静态更好（前沿为 QQQ 权重 0.4→1.3 的纯静态扫描）\n"
                "颜色 = RV 设定（同上图），形状 = 静态配比：圆 1:1，方 0.6:1.4，三角 0.5:1.5", fontsize=10, loc="left")
    S.grid(alpha=.3)

    # Δ 热图：相对"同平均敞口纯静态对照"
    H = fig.add_subplot(gb[0, 1])
    rv_rows = [sp for sp in specs if sp[0] != "none"]
    M = np.full((len(rv_rows), len(statics)), np.nan); txt = {}
    for i, sp in enumerate(rv_rows):
        for j, (q, sw) in enumerate(statics):
            r = R[(R.静态 == f"{q:g}:{sw:g}") & (R.RV == label(sp))].iloc[0]
            M[i, j] = r.dSharpe; txt[(i, j)] = f"ΔSR {r.dSharpe:+.3f}\nΔMDD {r.dMDD_pp:+.1f}pp"
    H.imshow(M, cmap="RdYlGn", vmin=-.15, vmax=.15, aspect="auto")
    H.set_xticks(range(len(statics))); H.set_xticklabels([f"{q:g}:{sw:g}" for q, sw in statics], fontsize=9)
    H.set_yticks(range(len(rv_rows))); H.set_yticklabels([label(sp) for sp in rv_rows], fontsize=8.2)
    for (i, j), t in txt.items():
        H.text(j, i, t, ha="center", va="center", fontsize=8)
    H.set_xlabel("静态配比 QQQ:Sleeve")
    H.set_title("RV 降风险 相对 同平均敞口纯静态对照 的增量\n（色标 ±0.15 = 噪声参照）", fontsize=10.5, loc="left")

    fig.suptitle("静态再配比 × RV 降风险 组合对比（Sleeve = 实盘配置 ⑦，2019–2026）", fontsize=15, y=.995)
    fig.subplots_adjust(left=.055, right=.985, top=.955, bottom=.045)
    fig.savefig(FIG / "combo_comparison.png", dpi=112)
    print(f"\n图 -> {FIG/'combo_comparison.png'}   表 -> {OUT/'combo_comparison.csv'}")


if __name__ == "__main__":
    main()
