"""Backtest the rule-based core leg (RV21 x SMA200) against buy-and-hold and compare with the distribution panel."""
import sys
sys.path.insert(0, ".")
import pandas as pd
from quantkit.data import load
from quantkit.backtest import daily
from quantkit.stats import perf, bootstrap, inference

px = load.load_close("QQQ"); r = px.pct_change().fillna(0.0)
w = daily.core_weight(px)                                   # PIT: already shifted one day
core = daily.leg_return(w, r)                               # 2 bp one-way cost on weight changes
oos = slice("2021", None)
print(perf.compare({"QQQ buy&hold": r, "core RV21xSMA200": core}, start="2021").round(3).T)
print("\nSharpe by year\n", pd.DataFrame({"B&H": perf.by_year(r), "core": perf.by_year(core)}).loc[2019:].round(2).T)
d = bootstrap.paired_delta_sharpe(core.loc[oos], r.loc[oos], nboot=2000, block=10)
print("\ncore - B&H delta Sharpe (paired block bootstrap):", {k: round(v, 3) for k, v in d.items()})
print("HAC test of the daily mean difference:", {k: round(v, 4) for k, v in inference.hac_mean_test((core - r).loc[oos].to_numpy()).items()})
