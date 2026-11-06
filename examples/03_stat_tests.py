"""Overfitting and diagnostics on a toy candidate family built from the core leg (swap in your own return series)."""
import sys
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
from quantkit.data import load
from quantkit.backtest import daily
from quantkit.stats import overfit, diagnostics, perf

px = load.load_close("QQQ"); r = px.pct_change().fillna(0.0)
fam = {f"RV[{lo},{hi}]": daily.leg_return(daily.rv_weight(px, lo, hi) * daily.trend_weight(px), r)
       for lo, hi in ((0.5, 1.5), (0.25, 1.5), (0.5, 1.25), (0.75, 1.25), (0.0, 1.5))}
F = pd.DataFrame(fam).loc["2021":]; base = F["RV[0.5,1.5]"]
inc = pd.DataFrame({k: overfit.vol_matched_increment(F[k], r.loc[F.index]) for k in F})            # increments over B&H at equal risk
var_sr = inc.apply(overfit.sr_period).var(ddof=1)
K = 413                                                                                           # project-wide trial count, docs/TRIAL_REGISTRY.md
print("increment PSR(>0) / DSR(K=%d):" % K)
for k in inc:
    print(f"  {k:12s} ann.SR {overfit.sr_period(inc[k])*np.sqrt(252):6.3f}  PSR {overfit.psr(inc[k]):.3f}  DSR {overfit.dsr(inc[k], var_sr, K):.3f}")
pbo, _ = overfit.cscv_pbo(F.to_numpy(), S=12); print(f"CSCV-PBO over the family: {pbo:.3f}")
# feature drift + monotonicity of a feature against next-day return
feat = r.rolling(21).std(); nxt = r.shift(-1)
print("\nPSI of 21d vol, 2019-22 vs 2025-26:\n", diagnostics.feature_drift(pd.DataFrame({"rv21": feat}).dropna(), ("2019", "2022"), ("2025", "2026")))
qt = diagnostics.quantile_table(feat.loc["2019":], nxt.loc["2019":]); print("\nnext-day return by rv21 quintile (bp):", (qt["bucket_means"] * 1e4).round(1).to_dict(), "| MonoScore", round(qt["mono_score"], 2), "| HAC t", round(qt["hac_t"], 2))
