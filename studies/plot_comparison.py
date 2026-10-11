"""12 个版本的完整对照图。"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cjkfont import use_cjk

use_cjk(plt)
sys.path.insert(0, str(Path(__file__).resolve().parent))

OUT = Path("outputs/full/intraday")
FIG = Path("outputs/full/figs"); FIG.mkdir(parents=True, exist_ok=True)

R = pd.read_csv(OUT / "strategy_comparison.csv")
cur = pd.read_pickle(OUT / "strategy_comparison_curves.pkl")
bh = pd.read_pickle(OUT / "strategy_comparison_bh.pkl")

SHORT = {
    "多空 vwap+尾1.0 全周": "①SOTA基线",
    "多空 vwap+尾1.0 全周+空头止盈": "②SOTA+止盈",
    "多空 vwap+尾1.0 无周一": "③SOTA+无周一",
    "多空 vwap+尾1.0 无周一+空头止盈": "④SOTA+无周一+止盈",
    "多空 band+尾0.5 全周": "⑤band",
    "多空 band+尾0.5 全周+空头止盈": "⑥band+止盈",
    "多空 band+尾0.5 无周一": "⑦band+无周一",
    "多空 band+尾0.5 无周一+空头止盈": "⑧band+无周一+止盈",
    "只做多 vwap+尾1.0 全周": "⑨只多·vwap",
    "只做多 vwap+尾1.0 无周一": "⑩只多·vwap无周一",
    "只做多 band+尾0.5 全周": "⑪只多·band",
    "只做多 band+尾0.5 无周一": "⑫只多·band无周一",
}
R["s"] = R.版本.map(SHORT)
LS = [k for k in SHORT if k.startswith("多空")]
LO = [k for k in SHORT if k.startswith("只做多")]

fig = plt.figure(figsize=(21, 12.5))
gs = fig.add_gridspec(2, 3, hspace=.32, wspace=.24)

SPLIT = pd.Timestamp("2023-01-01")


def eqplot(ax, keys, title, extra_base=None):
    cols = plt.cm.tab10(np.linspace(0, 1, 10))
    h = bh.dropna()
    ax.plot((1 + h).cumprod(), color="k", lw=2.2, ls=":", label="QQQ 买入持有", zorder=1)
    kk = list(keys) + ([extra_base] if extra_base else [])
    for i, k in enumerate(kk):
        x = cur[k].dropna()
        lw = 2.6 if k == "多空 vwap+尾1.0 全周" else 1.6
        ax.plot((1 + x).cumprod(), lw=lw, color=cols[i % 10],
                label=SHORT[k], alpha=.9,
                ls="-" if k.startswith("多空") else "--")
    ax.axvline(SPLIT, color="gray", lw=1.2, ls="--")
    ax.set_yscale("log")
    ax.set_ylabel("累计净值（对数轴，初始=1）")
    ax.set_title(title)
    ax.legend(fontsize=8.5, ncol=1, loc="upper left")
    ax.grid(alpha=.3, which="both")
    ax.text(SPLIT, ax.get_ylim()[0] * 1.03, " test 起点 2023",
            color="gray", fontsize=8.5, va="bottom")


# ---- (a) 多空
eqplot(fig.add_subplot(gs[0, 0]), LS,
       "(a) 多空版本的完整净值曲线\n30 相位均值・TQQQ 成本・虚线为 B&H")
# ---- (b) 只做多
eqplot(fig.add_subplot(gs[0, 1]), LO,
       "(b) 只做多版本（粗线为 SOTA 基线作参照）",
       extra_base="多空 vwap+尾1.0 全周")

# ---- (c) dev / test Sharpe
A = fig.add_subplot(gs[0, 2])
o = R.sort_values("test_sharpe")
y = np.arange(len(o)); hh = .38
A.barh(y - hh / 2, o.dev_sharpe, hh, label="dev 2019-22", color="tab:blue")
A.barh(y + hh / 2, o.test_sharpe, hh, label="test 2023-26", color="tab:cyan")
A.set_yticks(y); A.set_yticklabels(o.s, fontsize=9)
A.axvline(o.loc[o.版本 == "多空 vwap+尾1.0 全周", "test_sharpe"].iloc[0],
          color="tab:red", ls=":", lw=1.5, label="基线 test")
A.set_xlabel("独立 Sharpe")
A.set_title("(c) 独立 Sharpe：dev 与 test 经常打架\n（只做多·band 是 dev 最高、test 最低的典型）")
A.legend(fontsize=8.5, loc="lower right"); A.grid(alpha=.3, axis="x")

# ---- (d) 独立 Sharpe vs 叠加贡献
B = fig.add_subplot(gs[1, 0])
for _, r in R.iterrows():
    isls = r.版本.startswith("多空")
    B.scatter(r.test_sharpe, r.test_叠加贡献, s=110,
              c="tab:blue" if isls else "tab:purple",
              marker="o" if isls else "^", zorder=3)
    _off = {"④SOTA+无周一+止盈": (6, -12), "⑧band+无周一+止盈": (6, 4),
            "⑨只多·vwap": (6, 6), "⑩只多·vwap无周一": (-96, -4),
            "⑫只多·band无周一": (6, -4), "①SOTA基线": (6, -12),
            "⑤band": (6, 4), "⑥band+止盈": (6, -11)}
    B.annotate(r.s, (r.test_sharpe, r.test_叠加贡献), fontsize=8,
               textcoords="offset points", xytext=_off.get(r.s, (6, 4)))
b0 = R[R.版本 == "多空 vwap+尾1.0 全周"].iloc[0]
B.axhline(b0.test_叠加贡献, color="tab:red", ls=":", lw=1.5,
          label=f"基线叠加贡献 {b0.test_叠加贡献:+.3f}")
B.set_xlabel("独立 Sharpe（test）"); B.set_ylabel("B&H 叠加层 ΔSharpe（test）")
B.set_title("(d) 决策口径：独立 Sharpe 高 ≠ 叠加价值高\n"
            "只做多（三角）独立最高、叠加最差")
B.legend(fontsize=8.5); B.grid(alpha=.3)

# ---- (e) Calmar / MDD
C = fig.add_subplot(gs[1, 1])
o2 = R.sort_values("全_calmar")
y2 = np.arange(len(o2))
C.barh(y2 - hh / 2, o2.全_calmar, hh, label="Calmar（全样本）", color="tab:green")
C.barh(y2 + hh / 2, -o2.全_mdd * 5, hh, label="|最大回撤| ×5", color="tab:red", alpha=.75)
C.set_yticks(y2); C.set_yticklabels(o2.s, fontsize=9)
C.set_xlabel("Calmar ／ |MDD|×5")
C.set_title("(e) Calmar 与最大回撤（全样本 2019-2026）")
C.legend(fontsize=8.5, loc="lower right"); C.grid(alpha=.3, axis="x")

# ---- (f) 分边胜率 vs 盈亏比
D = fig.add_subplot(gs[1, 2])
for _, r in R.iterrows():
    for side, mk in (("多", "o"), ("空", "s")):
        wc, pc = f"{side}_胜率", f"{side}_盈亏比"
        if wc not in R.columns or not np.isfinite(r.get(wc, np.nan)):
            continue
        isband = "band" in r.版本
        D.scatter(r[wc], r[pc], s=120, marker=mk,
                  c="tab:orange" if isband else "tab:blue",
                  edgecolor="k", lw=.5, zorder=3)
D.axhline(1.0, color="gray", lw=1, ls=":")
D.set_xlabel("胜率"); D.set_ylabel("盈亏比（平均盈利 / |平均亏损|）")
D.set_title("(f) 止损设计改变的是交易画像，不只是收益\n"
            "圆=多头 方=空头；蓝=vwap 橙=band")
D.grid(alpha=.3)
D.set_xlim(.30, .62); D.set_ylim(.80, 2.40)
D.annotate("vwap：低胜率·高盈亏比\n（趋势跟随画像）", xy=(.345, 2.07),
           xytext=(.425, 2.22), fontsize=9, color="tab:blue",
           arrowprops=dict(arrowstyle="->", color="tab:blue"))
D.annotate("band：高胜率·盈亏比≈1\n换手低 19%、单笔亏损有构造上限", xy=(.566, 1.01),
           xytext=(.355, 1.22), fontsize=9, color="tab:orange",
           arrowprops=dict(arrowstyle="->", color="tab:orange"))

fig.suptitle("候选版本全面对照：止损设计 × 方向 × 星期 × 空头止盈（30 相位均值，TQQQ 成本）",
             fontsize=15, y=.975)
p = FIG / "strategy_comparison.png"
fig.savefig(p, dpi=120, bbox_inches="tight")
print("图已写入", p)
