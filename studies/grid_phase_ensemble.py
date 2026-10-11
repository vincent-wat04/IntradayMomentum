"""检查时点网格的相位集合：审计暴露出来的优化线索。

审计里有一项反向对照（audit_engine 1.3）：把检查时点整体后移 5 分钟，
累计收益变化 **40.47%**。这说明"10:00/10:30/.../15:00"这个网格的**相位**
对结果影响巨大 —— 而论文选这个相位没有任何理由，纯粹是整半点好看。

含义有两面：
  风险：我们报告的 Sharpe 里含有一份"相位运气"，是一种隐性的过拟合。
  机会：把 30 个相位（偏移 0..29 分钟）等权平均，可以把这份运气分散掉，
        在不改变任何信号的前提下降低方差。这是免费的改进，不是新的参数。

本脚本量化两件事：
  1. 30 个相位各自的 Sharpe 分布 -> 相位运气有多大
  2. 相位集合（等权平均 30 个相位的日收益）vs 论文原相位
成本按每个相位独立计算后再平均（集合的换手就是各相位换手的平均）。
"""
from __future__ import annotations

import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from etft.intraday.bars import load_bars
from etft.intraday.strategy import NoiseCfg, run, buy_hold

OUT = Path("outputs/full/intraday"); OUT.mkdir(parents=True, exist_ok=True)
DEV, TEST = ("2019", "2022"), ("2023", "2026")
SOTA = dict(stop="vwap", sizing="voltarget", max_lev=4.0)
PER_SHARE = 0.005 + 0.0045


def checks_at(offset):
    """把 10:00..15:00 的半点网格整体后移 offset 分钟。"""
    base = [(h, m) for h in range(10, 16) for m in (0, 30)][:-1]
    out = []
    for h, m in base:
        t = pd.Timestamp(f"2000-01-01 {h:02d}:{m:02d}") + pd.Timedelta(minutes=offset)
        out.append(t.strftime("%H:%M"))
    return out


def metrics(x, bench=None):
    x = x.dropna()
    if len(x) < 60 or x.std() == 0:
        return dict(sharpe=np.nan)
    eq = (1 + x).cumprod(); dd = eq / eq.cummax() - 1
    ann_ = float(eq.iloc[-1] ** (252 / len(x)) - 1)
    o = dict(sharpe=float(x.mean() / x.std() * np.sqrt(252)), ann=ann_,
             vol=float(x.std() * np.sqrt(252)), mdd=float(dd.min()),
             calmar=ann_ / abs(dd.min()) if dd.min() < 0 else np.nan,
             t_stat=float(x.mean() / x.std() * np.sqrt(len(x))))
    if bench is not None:
        b = bench.reindex(x.index).fillna(0)
        if b.std() > 0:
            o["beta"] = float(np.cov(x, b)[0, 1] / b.var())
            o["alpha"] = float((x - o["beta"] * b).mean() * 252)
    return o


def main():
    import yfinance as yf
    b = load_bars("QQQ")
    days = b.days
    px = yf.download("QQQ", start="2018-10-01", end="2026-10-05", auto_adjust=False,
                     progress=False, threads=False)["Close"]
    if hasattr(px, "columns"):
        px = px.iloc[:, 0]
    px.index = pd.to_datetime(px.index).tz_localize(None)
    PQ = px.reindex(days).ffill()
    rate = PER_SHARE / PQ
    bh = buy_hold(b)

    nets, rows = {}, []
    for off in range(30):
        r = run(b, NoiseCfg(check_times=checks_at(off), cost_bp=1.0, **SOTA))
        net = r["overlay_gross"] - r["cost"] * 1e4 * rate
        nets[off] = net
        for per, lab in ((TEST, "test23-26"), (("2019", "2026"), "all19-26")):
            rows.append(dict(offset=off, period=lab,
                             turnover=(r["cost"] * 1e4).loc[per[0]:per[1]].mean(),
                             **metrics(net.loc[per[0]:per[1]], bh.loc[per[0]:per[1]])))
        print(f"  offset={off:2d}min  test SR {metrics(net.loc[TEST[0]:TEST[1]])['sharpe']:+.3f}",
              flush=True)

    R = pd.DataFrame(rows)
    ENS = pd.DataFrame(nets).mean(axis=1)
    for per, lab in ((TEST, "test23-26"), (("2019", "2026"), "all19-26")):
        rows.append(dict(offset=-1, period=lab, turnover=np.nan,
                         **metrics(ENS.loc[per[0]:per[1]], bh.loc[per[0]:per[1]])))
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "grid_phase_ensemble.csv", index=False)

    pd.set_option("display.width", 220)
    for lab in ("test23-26", "all19-26"):
        s = R[(R.period == lab) & (R.offset >= 0)]
        e = R[(R.period == lab) & (R.offset == -1)].iloc[0]
        p = s[s.offset == 0].iloc[0]
        print(f"\n=== {lab} ===")
        print(f"  30 个相位的 Sharpe：均值 {s.sharpe.mean():.3f}  std {s.sharpe.std():.3f}  "
              f"最差 {s.sharpe.min():.3f}(offset={int(s.loc[s.sharpe.idxmin(),'offset'])})  "
              f"最好 {s.sharpe.max():.3f}(offset={int(s.loc[s.sharpe.idxmax(),'offset'])})")
        print(f"  论文原相位 offset=0 ：Sharpe {p.sharpe:.3f}  年化 {p.ann:.2%}  "
              f"回撤 {p.mdd:.1%}  Calmar {p.calmar:.2f}  alpha {p.alpha:.2%}")
        print(f"  **30 相位集合**     ：Sharpe {e.sharpe:.3f}  年化 {e.ann:.2%}  "
              f"回撤 {e.mdd:.1%}  Calmar {e.calmar:.2f}  alpha {e.alpha:.2%}")
        print(f"  集合 vs 原相位：ΔSharpe {e.sharpe-p.sharpe:+.3f}；"
              f"集合 vs 相位均值：ΔSharpe {e.sharpe-s.sharpe.mean():+.3f}  "
              f"（后者才是集合的真实贡献：分散掉相位运气）")
        print(f"  原相位在 30 个相位里的分位：{(s.sharpe < p.sharpe).mean():.0%}")

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.4))
    ax = axes[0]
    s = R[(R.period == "test23-26") & (R.offset >= 0)]
    ax.bar(s.offset, s.sharpe)
    ax.axhline(s.sharpe.mean(), color="k", ls=":", label=f"相位均值 {s.sharpe.mean():.3f}")
    e = R[(R.period == "test23-26") & (R.offset == -1)].iloc[0]
    ax.axhline(e.sharpe, color="r", lw=2, label=f"30 相位集合 {e.sharpe:.3f}")
    ax.axhline(s[s.offset == 0].sharpe.iloc[0], color="g", lw=2, ls="--",
               label=f"论文原相位 {s[s.offset==0].sharpe.iloc[0]:.3f}")
    ax.set_xlabel("检查网格偏移（分钟）"); ax.set_ylabel("test Sharpe")
    ax.set_title("相位运气有多大"); ax.legend(fontsize=8); ax.grid(alpha=.3, axis="y")
    ax = axes[1]
    for off in (0, 7, 15, 22):
        ax.plot((1 + nets[off].loc[TEST[0]:TEST[1]]).cumprod(), lw=1, alpha=.65,
                label=f"offset={off}")
    ax.plot((1 + ENS.loc[TEST[0]:TEST[1]]).cumprod(), lw=2.2, color="r", label="30 相位集合")
    ax.set_title("test 期净值"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    ax = axes[2]
    C = pd.DataFrame(nets).loc[TEST[0]:TEST[1]].corr()
    iu = np.triu_indices_from(C, 1)
    ax.hist(C.to_numpy()[iu], bins=30)
    ax.set_title(f"相位两两相关（均值 {C.to_numpy()[iu].mean():.2f}）\n"
                 f"相关越低，集合的分散收益越大")
    ax.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(OUT / "fig_grid_phase.png", dpi=110); plt.close()
    print(f"\n相位两两相关均值 {C.to_numpy()[iu].mean():.3f}（这决定了集合能降多少方差）")
    pd.to_pickle(dict(nets=nets, ens=ENS), OUT / "grid_phase_curves.pkl")
    return R


if __name__ == "__main__":
    main()
