"""系统化研究配图：关闭态权重 e_off 的影响（7 个市场均值 + 个别市场）。读 systematic_headline.csv。"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from cjkfont import use_cjk

use_cjk(plt)
H = pd.read_csv(Path(__file__).resolve().parent / "outputs/full/intraday/systematic_headline.csv")
E = [0.5, 0.0, -0.5, -1.0]
fam = {"SMA200 仅趋势": "SMA200 仅趋势 e={:+.1f}", "SMA200 浅蓝×乘积": "SMA200 浅蓝×乘积 e={:+.1f}",
       "MOM252 仅趋势": "MOM252 仅趋势 e={:+.1f}", "MOM252 浅蓝×乘积": "MOM252 浅蓝×乘积 e={:+.1f}"}
col = {"SMA200 仅趋势": "tab:blue", "SMA200 浅蓝×乘积": "tab:cyan", "MOM252 仅趋势": "tab:red", "MOM252 浅蓝×乘积": "tab:orange"}
fig, ax = plt.subplots(1, 3, figsize=(17, 5))
for k, f in fam.items():
    d = [H[H.label == f.format(e)] for e in E]
    ax[0].plot(E, [x.dsr.mean() for x in d], "o-", color=col[k], label=k)
    ax[1].plot(E, [x.mdd.mean() * 100 for x in d], "o-", color=col[k], label=k)
for mk, c in zip(["SPX", "NDX", "RUT", "EWJ", "EFA", "EEM"], ["k", "gray", "brown", "green", "purple", "olive"]):
    ax[2].plot(E, [H[(H.mkt == mk) & (H.label == f"MOM252 浅蓝×乘积 e={e:+.1f}")].dsr.iloc[0] for e in E], "o-", color=c, label=mk, lw=1.2)
ax[0].set_title("7 个市场平均 ΔSR（vs 同平均权重静态对照）"); ax[1].set_title("7 个市场平均 全样本 MDD (%)")
ax[2].set_title("MOM252×浅蓝 各市场 ΔSR"); [a.axhline(0, color="k", lw=.6) for a in (ax[0], ax[2])]
for a in ax:
    a.set_xlabel("关闭态权重 e_off（负 = 做空）"); a.invert_xaxis(); a.grid(alpha=.3); a.legend(fontsize=8)
fig.suptitle("关闭态权重：0~0.5 内夏普持平、回撤随 e_off 降低；一旦做空 (e<0) 夏普几乎一致下降", fontsize=13)
fig.tight_layout(); out = Path("outputs/full/figs"); fig.savefig(out / "systematic_eoff.png", dpi=130); print("ok")
