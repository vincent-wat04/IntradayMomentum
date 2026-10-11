"""RV21 降风险权重敏感性：热力图（面板 A 的 ΔSharpe 与 ΔMDD）。"""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cjkfont import use_cjk

use_cjk(plt)
A = pd.read_pickle("outputs/full/intraday/sens_rv_weights.pkl")["a"]
SAMP = ["1992-2026 全部", "2010-2026", "2010-2026 剔COVID"]


def heat(ax, df, vmax, fmt, title, cmap="RdYlGn"):
    v = df.to_numpy(float)
    ax.imshow(v, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(df.shape[1])); ax.set_xticklabels([f"{c:g}" for c in df.columns])
    ax.set_yticks(range(df.shape[0])); ax.set_yticklabels([f"{i:g}" for i in df.index])
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            ax.text(j, i, fmt(v[i, j]), ha="center", va="center", fontsize=9.5)
    if 0.0 in list(df.index) and 2.0 in list(df.columns):
        ax.add_patch(Rectangle((list(df.columns).index(2.0) - .5, list(df.index).index(0.0) - .5),
                               1, 1, fill=False, ec="k", lw=2.4))
    ax.set_title(title, fontsize=10.5)


fig, ax = plt.subplots(4, 3, figsize=(15, 15.5))
for r, (nm, kind) in enumerate((("SPX", "dsr"), ("NDX", "dsr"), ("SPX", "dmdd"), ("NDX", "dmdd"))):
    for c, smp in enumerate(SAMP):
        dsr, dmd, mw, tn = A[(nm, smp)]
        if kind == "dsr":
            heat(ax[r, c], dsr, 0.15, lambda x: f"{x:+.3f}", f"{nm} · {smp}\nΔSharpe")
        else:
            heat(ax[r, c], dmd * 100, 15, lambda x: f"{x:+.1f}", f"{nm} · {smp}\nΔMDD（pp，>0=回撤更浅）")
        if c == 0:
            ax[r, c].set_ylabel("下限 lo\n（最高波动时的权重）")
        if r == 3:
            ax[r, c].set_xlabel("上限 hi（最低波动时的权重）")
fig.suptitle("RV21 降风险权重敏感性（粗框 = 上一轮默认 lo=0, hi=2；ΔSharpe 色标 ±0.15 = 噪声参照）\n"
             "lo=0 一行的 ΔSharpe 对 hi 不变：w=hi·(1−p) 是常数倍，Sharpe 对缩放不变", fontsize=12.5, y=.995)
fig.tight_layout(rect=[0, 0, 1, .965])
fig.savefig("outputs/full/figs/sens_rv21_weights.png", dpi=115)
print("图 -> outputs/full/figs/sens_rv21_weights.png")
