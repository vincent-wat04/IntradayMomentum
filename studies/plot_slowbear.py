"""慢熊研究配图：回撤曲线（SPX / NDX，1998-2010 与 2020-2026）。"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cjkfont import use_cjk
import round3_common as C
import slowbear_study as Z

use_cjk(plt)
SHOW = {"B&H": ("black", 1.4), "H0 浅蓝": ("tab:cyan", 1.6), "H1 深蓝hi=1": ("tab:blue", 1.2),
        "H2 趋势200": ("tab:red", 1.8), "H3 动量12-1": ("tab:orange", 1.4)}
fig, ax = plt.subplots(2, 2, figsize=(15, 8.5))
for i, (nm, rel, sig) in enumerate(Z.MARKETS[:2]):
    px, rr, rfn, W, tot, ex, turn = Z.run_market(nm, rel, sig)
    for j, (a, b) in enumerate((("1999-01-01", "2010-12-31"), ("2019-01-01", "2026-10-01"))):
        A = ax[i, j]
        for k, (c, lw) in SHOW.items():
            x = tot[k].loc[a:b].fillna(0)
            eq = (1 + x).cumprod()
            A.plot((eq / eq.cummax() - 1) * 100, color=c, lw=lw, label=k)
        A.set_title(f"{nm} 回撤（自 {a[:4]} 起重新计）%"); A.grid(alpha=.3)
        if i == 0 and j == 0:
            A.legend(loc="lower left", fontsize=9)
fig.suptitle("慢熊研究：趋势类关闭规则 vs RV 降风险 —— 回撤曲线（收益含现金/融资，扣 2bp 成本）", fontsize=13)
fig.tight_layout()
out = Path("outputs/full/figs"); out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / "slowbear_drawdown.png", dpi=130)
print("saved", out / "slowbear_drawdown.png")
