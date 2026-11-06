"""事后(探索性)检查:第一阶段单变量里过 Bonferroni 的 w_mk_GEO / w_mk_BANK 是否真有预测增量。
不改动预注册配置;作为新增 3 个探索配置记账。python3 news_ms_posthoc.py"""
from __future__ import annotations

import os
import sys
import warnings

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lowvol_avoidance_study as L
import news_ms as N
from event_calendar import event_frame
from etft.intraday.bars import load_bars

L.OOS_YEARS = range(2021, 2024)


def main():
    b = load_bars("QQQ"); days = b.days; items = N.load_items()
    X, Y, A, B = L.build(b); E = event_frame(days, b.last_idx.to_numpy()); EC = list(E.columns); W = N.daily_features(items, days)
    X = pd.concat([X, E, W], axis=1)
    G = ["w_mk_GEO", "w_mk_BANK"]
    sets = {"A+B+E": (A + B + EC, "ridge"), "A+B+E+GEO": (A + B + EC + ["w_mk_GEO"], "ridge"), "A+B+E+BANK": (A + B + EC + ["w_mk_BANK"], "ridge"), "A+B+E+GEO+BANK": (A + B + EC + G, "ridge")}
    pred, base_rv, thr_s = L.walk_forward(X, Y, sets)
    oos = X.index[(X.index.year >= 2021) & (X.index.year <= 2023) & X[A + B + EC + list(W.columns)].notna().all(1) & Y.notna().all(1)]
    dev, test = oos[oos.year <= 2022], oos[oos.year == 2023]
    sse_b = pd.Series(((Y.loc[oos, "y_rv"] - base_rv.loc[oos]) ** 2).to_numpy(), index=oos)
    e = {n: pd.Series(((Y.loc[oos, "y_rv"] - pred[(n, "rv")].loc[oos]) ** 2).to_numpy(), index=oos) for n in sets}
    print("事后:在 A+B+E 上加入 GEO / BANK(各自 + 联合)的 ΔR²(月聚类自助 95%)")
    rows = []
    for n in list(sets)[1:]:
        for per, idx in (("全OOS", oos), ("dev", dev), ("test", test)):
            fn = lambda ii, idx=idx, n=n: (e["A+B+E"].loc[idx].iloc[ii].sum() - e[n].loc[idx].iloc[ii].sum()) / sse_b.loc[idx].iloc[ii].sum()
            pt, lo, hi = L.month_boot(idx, fn); rows.append(dict(模型=n, 期间=per, ΔR2=pt, lo=lo, hi=hi))
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    res = (Y.y_rv - pred[("A+B+E", "rv")]).loc[oos]
    print("\n单变量 rho 的月聚类自助区间(特征 vs A+B+E 残差):")
    for c in G:
        x = X.loc[oos, c].to_numpy(); y = res.to_numpy()
        pt, lo, hi = L.month_boot(oos, lambda ii: spearmanr(x[ii], y[ii])[0]); print(f"  {c}: rho {pt:+.3f} [{lo:+.3f},{hi:+.3f}]")
    print("\n逐年 rho:", {c: {int(yv): round(float(spearmanr(X.loc[oos[oos.year == yv], c], res[oos.year == yv])[0]), 3) for yv in (2021, 2022, 2023)} for c in G})
    for c in G:
        raw = X.loc[oos, c]; top = raw.nlargest(15).index
        print(f"{c} 最大 15 天:", [str(d.date()) for d in top[:15]], "| 这 15 天的残差均值 %.3f(全体 %.3f)" % (res.loc[top].mean(), res.mean()))
    nz = {c: float((X.loc[oos, c].abs() > 1e-9).mean()) for c in G}; print("非零天数占比:", nz)


if __name__ == "__main__":
    main()
