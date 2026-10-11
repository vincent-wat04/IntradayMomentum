"""止盈研究的四张图。"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cjkfont import use_cjk

use_cjk(plt)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from stop_significance import sr

OUT = Path("outputs/full/intraday")
FIG = Path("outputs/full/figs"); FIG.mkdir(parents=True, exist_ok=True)
THRESH = (0.25, 0.5, 0.75, 1.0, 1.5)   # 2.0 以上每组只剩 1~3 天，无法估计

fig, ax = plt.subplots(2, 2, figsize=(15.5, 10.5))

# ---------------- (a) 条件延续诊断：按天聚类的"放弃段"
T = pd.read_csv(OUT / "takeprofit_diag_trades.csv", parse_dates=["day"])
T["year"] = T.day.dt.year


def clus(sub, x, min_days=20):
    """按天聚类：同一天的 30 个相位先取均值，再按天估计。
    少于 min_days 天就不报点 —— 高阈值常常只剩 1~3 天，画出来全是噪声。"""
    m = sub[f"reach_{x}"] == 1
    s = sub.loc[m, ["day", f"fwd_{x}"]].dropna()
    if len(s) < 30:
        return np.nan, np.nan
    g = s.groupby("day")[f"fwd_{x}"].mean() * 1e4
    if len(g) < min_days:
        return np.nan, np.nan
    return g.mean(), g.std() / np.sqrt(len(g))


A = ax[0, 0]
series = [("空头 dev19-22", -1, (2019, 2022), "tab:red", "-"),
          ("空头 test23-26", -1, (2023, 2026), "tab:red", "--"),
          ("多头 dev19-22", 1, (2019, 2022), "tab:blue", "-"),
          ("多头 test23-26", 1, (2023, 2026), "tab:blue", "--")]
for lab, sgn, per, col, ls in series:
    sub = T[(T.sgn == sgn) & (T.year >= per[0]) & (T.year <= per[1])]
    mu = [clus(sub, x)[0] for x in THRESH]
    se = [clus(sub, x)[1] for x in THRESH]
    A.errorbar(THRESH, mu, yerr=se, label=lab, color=col, ls=ls,
               marker="o", ms=4, capsize=3, lw=1.8)
A.axhline(0, color="k", lw=1)
A.set_xlabel("止盈阈值 x（有利偏离 / σ_daily）")
A.set_ylabel("放弃段收益（bp）")
A.set_title("(a) 到达 +x·σ 之后还剩多少？按天聚类 ±1SE\n"
            "低于 0 = 会回吐 = 止盈有价值；高于 0 = 还在赚 = 止盈砍右尾")
A.legend(fontsize=9); A.grid(alpha=.3)
A.text(.98, .04, "空头 dev/test 同号、同量级、都显著\n多头只在 dev 成立，test 不成立\n"
                 "（已剔除聚类天数 <20 的点）", transform=A.transAxes,
       ha="right", va="bottom", fontsize=9,
       bbox=dict(fc="lightyellow", ec="gray", alpha=.9))

# ---------------- (b) 独立 Sharpe vs 叠加贡献：减空头的权衡
P = pd.read_csv(OUT / "takeprofit_placebo.csv")
B = ax[0, 1]
pick = {"基线（无止盈）": "基线 多空", "对照B 空头规模×0.9": "空头×0.9",
        "对照B 空头规模×0.75": "空头×0.75", "对照B 空头规模×0.6": "空头×0.6",
        "空头止盈 k=0.75 rearm": "空头止盈 k=0.75"}
xs, ys, labs = [], [], []
for k, lab in pick.items():
    row = P[P.配置 == k]
    if not len(row):
        continue
    xs.append(float(row.test_sharpe.iloc[0]))
    ys.append(float(row.test_叠加贡献.iloc[0]))
    labs.append(lab)
xs.append(1.185); ys.append(0.127); labs.append("long-only（PART9）")
cmap = {"基线 多空": "tab:gray", "空头×0.9": "tab:green", "空头×0.75": "tab:green",
        "空头×0.6": "tab:green", "空头止盈 k=0.75": "tab:orange",
        "long-only（PART9）": "tab:purple"}
offs = {"基线 多空": (-10, 10), "空头×0.9": (6, 8), "空头×0.75": (4, -16),
        "空头×0.6": (8, 6), "空头止盈 k=0.75": (6, 10),
        "long-only（PART9）": (-80, 10)}
B.scatter(xs, ys, s=95, c=[cmap.get(l, "tab:blue") for l in labs], zorder=3)
for x, y, l in zip(xs, ys, labs):
    B.annotate(l, (x, y), textcoords="offset points",
               xytext=offs.get(l, (7, 5)), fontsize=9)
B.annotate("", xy=(1.185, 0.127), xytext=(0.963, 0.268),
           arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.4, alpha=.65))
B.text(1.09, 0.20, "空头腿越弱 →\n叠加价值越低", color="tab:red", fontsize=9, ha="center")
B.axhline(ys[0], color="tab:gray", ls=":", lw=1.5, label="基线叠加贡献")
B.set_xlabel("独立 Sharpe（test 期）")
B.set_ylabel("作为 B&H 叠加层的贡献 ΔSharpe（test 期）")
B.set_title("(b) 为什么『独立 Sharpe 变好』是陷阱\n"
            "一切削弱空头的做法：独立 Sharpe↑ 但叠加贡献↓或不动")
B.legend(fontsize=9); B.grid(alpha=.3)

# ---------------- (c) GEX 分组：机制检验
G = pd.read_csv(OUT / "takeprofit_gex.csv")
C = ax[1, 0]
order = ["止盈·仅低GEX日", "止盈·仅中GEX日", "止盈·仅高GEX日"]
names = ["低 GEX", "中 GEX", "高 GEX"]
base_d = float(G[G.配置 == "基线（无止盈）"].dev_sharpe.iloc[0])
base_t = float(G[G.配置 == "基线（无止盈）"].test_sharpe.iloc[0])
dd = [float(G[G.配置 == o].dev_sharpe.iloc[0]) - base_d for o in order]
tt = [float(G[G.配置 == o].test_sharpe.iloc[0]) - base_t for o in order]
w = .35; xi = np.arange(3)
C.bar(xi - w / 2, dd, w, label="dev 19-22", color="tab:blue")
C.bar(xi + w / 2, tt, w, label="test 23-26", color="tab:cyan")
C.axhline(0, color="k", lw=1)
C.set_xticks(xi); C.set_xticklabels(names)
C.set_ylabel("ΔSharpe vs 基线")
C.set_title("(c) GEX 机制检验：被推翻\n"
            "gamma 理论预测『高 GEX 日止盈最有效』，实测恰好相反")
C.legend(fontsize=9); C.grid(alpha=.3, axis="y")
C.annotate("预测方向", xy=(2, max(dd) * .6), xytext=(0.2, max(dd) * .9),
           arrowprops=dict(arrowstyle="->", color="red"), color="red", fontsize=9)

# ---------------- (d) 逐年
cur = pd.read_pickle(OUT / "takeprofit_placebo_curves.pkl")
D = ax[1, 1]
base = cur["基线（无止盈）"]; tp = cur["空头止盈 k=0.75 rearm"]
yrs, dsr = [], []
for y in range(2019, 2027):
    a = base.loc[str(y)].dropna(); c = tp.loc[str(y)].dropna()
    if len(a) < 40:
        continue
    yrs.append(y); dsr.append(sr(c) - sr(a))
cols = ["tab:green" if v > 0 else "tab:red" for v in dsr]
D.bar([str(y) for y in yrs], dsr, color=cols)
D.axhline(0, color="k", lw=1)
D.set_ylabel("ΔSharpe（止盈 − 基线）")
D.set_title("(d) 逐年并不稳定\n集中在 2021/2026，2025 明显为负")
D.grid(alpha=.3, axis="y")

fig.suptitle("止盈 × 同向再开仓 × GEX：诊断有真信号，但策略层拿不到",
             fontsize=14, y=.995)
fig.tight_layout(rect=[0, 0, 1, .97])
p = FIG / "takeprofit_study.png"
fig.savefig(p, dpi=135)
print("图已写入", p)
