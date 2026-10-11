"""GEX 能不能过滤假突破？—— 这次按正确的假设去测。

之前测过两次都失败，但两次测的都**不是"过滤假突破"**：
  E30   把 GEX 当方向性 alpha（前提就不成立）
  PART9 §2.4  用 GEX 做**仓位缩放**（置换检验 p=0.092），而且是在 vol targeting 之上，
              与 σ_trailing 相关 0.41，部分抵消了 vol targeting
这次的假设更窄、也更贴近做市商 gamma 的机制：

  **做市商净 gamma 为负时，他们必须顺势交易（涨了买、跌了卖），
    所以突破更容易延续；净 gamma 为正时他们逆势交易，突破更容易被打回去（假突破）。**

要测的是"**条件胜率/延续性**"，不是"收益缩放"：
  对每一笔突破，定义"假突破" = 入场后被止损打掉（而不是持到收盘）
  然后看 P(假突破 | GEX 分位) 是否单调。
这是一个比 Sharpe 远更有功效的检验：观测数是**笔数**（1500+）而不是天数，
而且是二元结果，不受厚尾影响。

数据：SqueezeMetrics GEX（data/raw/squeeze_dix_gex.parquet，含 price/dix/gex）
     严格 t−2 滞后（E25 口径：GEX 当日盘后发布，多滞后一天保险）
文献依据见 docs/PART12（本脚本的注释里也标了出处）。
"""
from __future__ import annotations

import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from etft.intraday.bars import load_bars

from audit import reimpl_noise as RI

from paths import ETFT
OUT = Path("outputs/full/intraday"); OUT.mkdir(parents=True, exist_ok=True)
DEV, TEST = ("2019", "2022"), ("2023", "2026")


def load_gex(days):
    g = pd.read_parquet(ETFT / "data/raw/squeeze_dix_gex.parquet")
    g.index = pd.to_datetime(g.index)
    out = {}
    for col in ("gex", "dix"):
        if col in g.columns:
            out[col] = g[col].reindex(days).ffill().shift(2)      # t−2，E25 口径
    return pd.DataFrame(out)


def trades_with_exit(b, **kw):
    """返回逐笔明细，含"是否被止损打掉"（假突破的操作化定义）。"""
    _, F = RI.simulate(b, sizing="voltarget", max_lev=4.0, reentry="hold",
                       half_spread=0.005, commission=0.0045, **kw)
    F = F.copy(); F["day"] = pd.to_datetime(F.day)
    F["cmin"] = F.clock.map(lambda s: int(str(s)[:2]) * 60 + int(str(s)[3:5]))
    tr = []
    for d, g in F.groupby("day", sort=True):
        g = g.sort_values("cmin").reset_index(drop=True); op = None
        for _, row in g.iterrows():
            if row.kind.startswith("ENTRY"):
                op = row
            elif op is not None:
                side = 1 if op.shares > 0 else -1
                tr.append(dict(day=d, side=side, entry_clock=op.clock,
                               ret=side * (row.price / op.price - 1),
                               stopped=int(row.kind in ("STOP", "TAILSTOP")),
                               hold=row.cmin - op.cmin))
                op = None
    return pd.DataFrame(tr)


def main():
    b = load_bars("QQQ"); days = b.days
    G = load_gex(days)
    print(f"GEX 数据：{G.gex.notna().sum()} 天可用（{days[0].date()} .. {days[-1].date()}）")

    T = trades_with_exit(b, stop_long="vwap", stop_short="vwap")
    T["gex"] = G.gex.reindex(T.day).to_numpy()
    T["dix"] = G.dix.reindex(T.day).to_numpy() if "dix" in G.columns else np.nan
    T = T[np.isfinite(T.gex)]
    print(f"可用交易笔数 {len(T)}（{T.day.nunique()} 天）\n")

    # PIT 分位：252 日滚动，只用过去
    gs = G.gex.copy()
    pct = gs.rolling(252, min_periods=120).apply(lambda x: (x[-1] > x[:-1]).mean(), raw=True)
    T["gex_pct"] = pct.reindex(T.day).to_numpy()
    T = T[np.isfinite(T.gex_pct)]

    print("=" * 104)
    print("1. 核心检验：P(被止损 | GEX 分位)。机制预测：**低 GEX（负 gamma）-> 假突破更少**")
    print("=" * 104)
    for per, pl in ((DEV, "dev19-22"), (TEST, "test23-26"), ((None, None), "全样本")):
        sub = T if per[0] is None else T[(T.day >= per[0]) & (T.day <= f"{per[1]}-12-31")]
        if len(sub) < 100:
            continue
        q = pd.qcut(sub.gex_pct, 5, labels=["Q1 最低GEX", "Q2", "Q3", "Q4", "Q5 最高GEX"])
        t = sub.groupby(q, observed=True).agg(
            笔数=("stopped", "size"), 被止损率=("stopped", "mean"),
            胜率=("ret", lambda x: (x > 0).mean()),
            平均bp=("ret", lambda x: x.mean() * 1e4),
            中位持仓=("hold", "median"))
        # 单调性与两端差异的检验
        lo, hi = sub[sub.gex_pct <= 0.2].stopped, sub[sub.gex_pct >= 0.8].stopped
        p1, p2 = lo.mean(), hi.mean()
        n1, n2 = len(lo), len(hi)
        se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
        z = (p2 - p1) / se if se > 0 else np.nan
        print(f"\n--- {pl}（{len(sub)} 笔）")
        print(t.round(4).to_string())
        print(f"    最高 vs 最低 GEX 的被止损率：{p2:.1%} vs {p1:.1%}，"
              f"差 {(p2-p1)*100:+.1f}pp，**z={z:+.2f}**"
              f"（机制预测 z>0，即高 GEX 假突破更多）")

    # ---------- 2. 按方向分开（gamma 的效应可能不对称）
    print("\n" + "=" * 104)
    print("2. 分方向：做多/做空的被止损率 vs GEX")
    print("=" * 104)
    for side, nm in ((1, "做多"), (-1, "做空")):
        sub = T[T.side == side]
        q = pd.qcut(sub.gex_pct, 4, labels=["Q1低", "Q2", "Q3", "Q4高"])
        t = sub.groupby(q, observed=True).agg(
            笔数=("stopped", "size"), 被止损率=("stopped", "mean"),
            平均bp=("ret", lambda x: x.mean() * 1e4))
        print(f"\n--- {nm}")
        print(t.round(4).to_string())

    # ---------- 3. 当成过滤器：只在低 GEX 日交易
    print("\n" + "=" * 104)
    print("3. 当成过滤器用：只在 GEX 低分位的日子交易（dev 定阈值，test 套用）")
    print("=" * 104)
    import importlib
    SD = importlib.import_module("stop_design_study")
    # 直接用已算好的 30 相位均值曲线
    cur = pd.read_pickle(OUT / "stop_hard_tail_curves.pkl")
    base = cur["vwap（当前基线）"]
    gp = pct.reindex(base.index)
    rg = np.random.default_rng(11)
    for thr in (0.2, 0.33, 0.5, 0.67):
        m = (gp <= thr)
        for per, pl in ((DEV, "dev"), (TEST, "test")):
            s = base.loc[per[0]:per[1]]
            mm = m.reindex(s.index).fillna(False)
            real = s[mm]
            if len(real) < 60:
                continue
            k = int(mm.sum())
            nulls = [float(s.iloc[np.sort(rg.choice(len(s), k, replace=False))].pipe(
                lambda x: x.mean() / x.std() * np.sqrt(252))) for _ in range(400)]
            nulls = np.array([x for x in nulls if np.isfinite(x)])
            srr = real.mean() / real.std() * np.sqrt(252)
            p = float((nulls >= srr).mean())
            if pl == "dev":
                line = f"  GEX<={thr:<5} dev {k:4d}天 SR {srr:+.3f} (p={p:.3f})"
            else:
                line += f" | test {k:4d}天 SR {srr:+.3f} (p={p:.3f})"
                print(line)

    T.to_csv(OUT / "gex_false_breakout_trades.csv", index=False)
    print(f"\n逐笔明细已写入 {OUT}/gex_false_breakout_trades.csv")


if __name__ == "__main__":
    main()
