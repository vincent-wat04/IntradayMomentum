"""alpha / beta 分解图。"""
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

OUT = Path("outputs/full/intraday")
FIG = Path("outputs/full/figs")
R = pd.read_csv(OUT / "alpha_beta_decomp.csv")
R["LS"] = R.版本.str.startswith("多空")

fig, ax = plt.subplots(2, 2, figsize=(16, 11))

# ---- (a) β vs 叠加贡献
A = ax[0, 0]
for _, r in R.iterrows():
    A.scatter(r.test_BH_beta, r.test_叠加贡献, s=120,
              c="tab:blue" if r.LS else "tab:purple",
              marker="o" if r.LS else "^", zorder=3)
    A.annotate(r.s, (r.test_BH_beta, r.test_叠加贡献), fontsize=8,
               textcoords="offset points", xytext=(6, 4))
x = R.test_BH_beta.to_numpy(); y = R.test_叠加贡献.to_numpy()
k, c0 = np.polyfit(x, y, 1)
xs = np.linspace(x.min() - .01, x.max() + .02, 50)
A.plot(xs, k * xs + c0, color="tab:red", ls="--", lw=1.5,
       label=f"拟合 corr = {np.corrcoef(x, y)[0,1]:.3f}")
A.set_xlabel("β（对全日 QQQ B&H，test 期）")
A.set_ylabel("B&H 叠加层 ΔSharpe（test）")
A.set_title("(a) 叠加价值几乎完全由 β 决定\nβ 越接近 0 → 对 B&H 的分散化价值越高")
A.legend(fontsize=9); A.grid(alpha=.3)

# ---- (b) 绝对 Sharpe vs alpha Sharpe
B = ax[0, 1]
o = R.sort_values("test_BH_alpha_sharpe")
yy = np.arange(len(o)); h = .38
B.barh(yy - h / 2, o.test_abs_sharpe, h, label="绝对 Sharpe（不对冲）",
       color="tab:gray")
B.barh(yy + h / 2, o.test_BH_alpha_sharpe, h,
       label="alpha Sharpe（β 对冲掉之后）", color="tab:green")
B.set_yticks(yy); B.set_yticklabels(o.s, fontsize=9)
B.set_xlabel("Sharpe（test 期）")
B.set_title("(b) 绝对 Sharpe 里有多少其实是 β\n只做多（下方四个）差 0.52~0.56，多空几乎没差")
B.legend(fontsize=9, loc="lower right"); B.grid(alpha=.3, axis="x")

# ---- (c) α 年化 + HAC t
C = ax[1, 0]
o2 = R.sort_values("test_BH_alpha_ann")
y2 = np.arange(len(o2))
cols = ["tab:blue" if v else "tab:purple" for v in o2.LS]
C.barh(y2, o2.test_BH_alpha_ann * 100, color=cols)
for i, (a_, t_) in enumerate(zip(o2.test_BH_alpha_ann, o2.test_BH_t_alpha)):
    C.text(a_ * 100 + .18, i, f"t={t_:.2f}", va="center", fontsize=8.5,
           color="black" if t_ >= 1.96 else "tab:red")
C.set_yticks(y2); C.set_yticklabels(o2.s, fontsize=9)
C.set_xlabel("α 年化（%，test 期，vs 全日 B&H）")
C.set_title("(c) α 与其 HAC t 值（红色 = t<1.96）\n只做多的 α 只有 3~5%，多空 10~16%")
C.grid(alpha=.3, axis="x")

# ---- (d) alpha Sharpe dev vs test
D = ax[1, 1]
for _, r in R.iterrows():
    D.scatter(r.dev_BH_alpha_sharpe, r.test_BH_alpha_sharpe, s=120,
              c="tab:blue" if r.LS else "tab:purple",
              marker="o" if r.LS else "^", zorder=3)
    D.annotate(r.s, (r.dev_BH_alpha_sharpe, r.test_BH_alpha_sharpe), fontsize=8,
               textcoords="offset points", xytext=(6, 4))
lim = [0.2, 1.35]
D.plot(lim, lim, color="gray", ls=":", lw=1.3, label="dev = test")
D.set_xlim(*lim); D.set_ylim(*lim)
D.set_xlabel("alpha Sharpe（dev 19-22）")
D.set_ylabel("alpha Sharpe（test 23-26）")
D.set_title("(d) alpha 质量的样本一致性\n落在对角线右下 = dev 高 test 低 = 过拟合嫌疑")
D.legend(fontsize=9); D.grid(alpha=.3)

fig.suptitle("alpha / beta 分解：区分『真 alpha』与『只是市场敞口』（30 相位均值，TQQQ 成本）",
             fontsize=14, y=.98)
fig.tight_layout(rect=[0, 0, 1, .955])
p = FIG / "alpha_beta_decomp.png"
fig.savefig(p, dpi=125)
print("图已写入", p)
