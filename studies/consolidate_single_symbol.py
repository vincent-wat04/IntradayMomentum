"""Every strategy we have run, reported on SPY and on QQQ separately (no 50/50 blending).

Sources, all already computed:
  outputs/full/strategy/model_strategy_vs_null.csv        6 models x 2 labels x 4 horizons x 6 strategies x 3 costs
  outputs/full/strategy/leveraged_portfolio_metrics.csv   per-horizon sleeves, horizon blend
  outputs/full/strategy/portfolio_construction_metrics.csv beta-hedged sleeves, core+sleeve risk parity
  outputs/full/strategy/leverage_frontier.csv             constant leverage 1x-3x, signal tilt, vol target
  outputs/full/strategy/overlay_check_{SPY,QQQ}.csv       the stage-5 "best" strategy, exposure-matched
  outputs/full/report/factor_sharpe.csv                   single-factor books (reference)

Everything is test period 2023-01..2026-10, next_open execution, 1bp per unit turnover unless stated.
"d_vs_null" is the real result minus the same pipeline on circularly-shifted features.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

S = Path("outputs/full/strategy")
BH = {"SPY": 1.403, "QQQ": 1.496}
BH_ANN = {"SPY": 0.219, "QQQ": 0.324}
BH_MDD = {"SPY": -0.188, "QQQ": -0.228}
rows = []


def add(sym, family, name, sharpe, ann=np.nan, mdd=np.nan, null=np.nan, note=""):
    rows.append(dict(symbol=sym, family=family, strategy=name, sharpe=sharpe, ann_ret=ann, mdd=mdd,
                     null_sharpe=null, d_vs_null=sharpe - null if np.isfinite(null) else np.nan,
                     d_vs_bh=sharpe - BH[sym], note=note))


# ---------------- 1. buy & hold
for s in ("SPY", "QQQ"):
    add(s, "benchmark", "buy & hold", BH[s], BH_ANN[s], BH_MDD[s])

# ---------------- 2. models x strategies (best label, per horizon)
M = pd.read_csv(S / "model_strategy_vs_null.csv")
M = M[M.fee_bp == 1.0].rename(columns={"sr_test": "sharpe", "ann": "ann_ret"})
for (sym, strat, h), g in M[M.target == "VOLSTD"].groupby(["symbol", "strategy", "h"]):
    b = g.loc[g.sharpe.idxmax()]
    add(sym, f"model/{strat}", f"{strat} h={h} best model ({b.model})", b.sharpe, b.ann_ret, b.mdd, b.null_sr,
        f"VOLSTD label")
# model comparison at the single best horizon per symbol
for sym, g in M[(M.target == "VOLSTD") & (M.strategy == "LS")].groupby("symbol"):
    for mdl, gg in g.groupby("model"):
        b = gg.loc[gg.sharpe.idxmax()]
        add(sym, "model/LS by model", f"LS {mdl} (h={int(b.h)})", b.sharpe, b.ann_ret, b.mdd, b.null_sr)

# ---------------- 3. sleeves per horizon + blend
L = pd.read_csv(S / "leveraged_portfolio_metrics.csv")
sl = L[(L.group == "sleeve")].copy()
sl["is_null"] = sl.name.str.contains(r"\[null\]")
sl["key"] = sl.name.str.replace(" [null]", "", regex=False)
piv = sl.pivot_table(index="key", columns="is_null", values=["sharpe", "ann_ret", "mdd"])
for key in piv.index:
    sym = key.split()[0]
    if sym not in BH:
        continue
    add(sym, "sleeve", key.replace(sym + " ", ""), piv.loc[key, ("sharpe", False)],
        piv.loc[key, ("ann_ret", False)], piv.loc[key, ("mdd", False)], piv.loc[key, ("sharpe", True)])

# ---------------- 4. beta-hedged sleeves and core+sleeve
P = pd.read_csv(S / "portfolio_construction_metrics.csv")
pp = P.copy(); pp["is_null"] = pp.name.str.contains(r"\[null\]")
pp["key"] = pp.name.str.replace(" [null]", "", regex=False)
pv = pp.pivot_table(index="key", columns="is_null", values=["sharpe", "ann_ret", "mdd"])
for key in pv.index:
    sym = key.split()[0]
    if sym not in BH or "buy & hold" in key:
        continue
    nullv = pv.loc[key, ("sharpe", True)] if True in pv.columns.get_level_values(1) else np.nan
    fam = "sleeve (beta-hedged)" if "beta-hedged" in key else "core+sleeve risk parity"
    add(sym, fam, key.replace(sym + " ", ""), pv.loc[key, ("sharpe", False)],
        pv.loc[key, ("ann_ret", False)], pv.loc[key, ("mdd", False)], nullv)

# ---------------- 5. leverage frontier
F = pd.read_csv(S / "leverage_frontier.csv")
for _, r in F.iterrows():
    add(r.symbol, "leverage/" + ("constant" if r.kind == "constant E" else "timed"),
        f"{r.kind} (E={r.E:.2f})", r.sharpe, r.ann_ret, r.mdd,
        np.nan, f"vs exposure-matched: dSR {r.d_sharpe:+.3f}" if np.isfinite(r.get("d_sharpe", np.nan)) else "")

# ---------------- 6. the stage-5 "best" strategy, exposure-matched
for sym in ("SPY", "QQQ"):
    f = S / f"overlay_check_{sym}.csv"
    if not f.exists():
        continue
    O = pd.read_csv(f)
    real = O[(O["shift"] == 0) & O.name.str.contains(r"\[")]
    null = O[(O["shift"] == 1)]
    for _, r in real.iterrows():
        nm = r["name"].replace(sym + " ", "")
        nr = null[null.name.str.replace(" [null]", "", regex=False) == r["name"]]
        add(sym, "stage-5 best", nm, r.sharpe, r.ann_ret, r.mdd,
            float(nr.sharpe.iloc[0]) if len(nr) else np.nan,
            f"avgE {r.avg_E:.3f}, vs matched dSR {r.d_sr_vs_ctrl:+.3f}")

T = pd.DataFrame(rows)
T.to_csv(S / "all_strategies_single_symbol.csv", index=False)
pd.set_option("display.width", 300); pd.set_option("display.max_rows", 600)

for sym in ("SPY", "QQQ"):
    q = T[T.symbol == sym].sort_values("sharpe", ascending=False)
    print(f"\n{'=' * 110}\n{sym}  (buy & hold: SR {BH[sym]:.3f}, ann {BH_ANN[sym]:.1%}, MDD {BH_MDD[sym]:.1%})\n{'=' * 110}")
    print(q[["family", "strategy", "sharpe", "ann_ret", "mdd", "null_sharpe", "d_vs_null", "d_vs_bh", "note"]]
          .head(28).round(3).to_string(index=False))

print(f"\n{'=' * 110}\nhow many configurations beat buy & hold, by symbol\n{'=' * 110}")
x = T[T.family != "benchmark"]
print(x.groupby("symbol").apply(lambda g: pd.Series({
    "n_configs": len(g), "beat_bh_sharpe": (g.d_vs_bh > 0).mean(),
    "beat_bh_by>0.1": (g.d_vs_bh > 0.1).mean(), "median_d_vs_bh": g.d_vs_bh.median(),
    "n_with_null": g.d_vs_null.notna().sum(), "median_d_vs_null": g.d_vs_null.median(),
    "beat_null": (g.d_vs_null > 0).mean()})).round(3).to_string())

print(f"\n{'=' * 110}\nbest of each family\n{'=' * 110}")
best = x.loc[x.groupby(["symbol", "family"]).sharpe.idxmax()]
print(best[["symbol", "family", "strategy", "sharpe", "ann_ret", "mdd", "d_vs_bh", "d_vs_null"]]
      .sort_values(["symbol", "sharpe"], ascending=[True, False]).round(3).to_string(index=False))

# ---------------- figure
fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
for ax, sym in zip(axes, ["SPY", "QQQ"]):
    q = T[(T.symbol == sym) & (T.family != "benchmark")].dropna(subset=["mdd", "ann_ret"])
    fams = sorted(q.family.unique())
    cmap = plt.get_cmap("tab20")
    for i, f in enumerate(fams):
        g = q[q.family == f]
        ax.scatter(-g.mdd, g.ann_ret, s=42, color=cmap(i % 20), label=f, alpha=.85)
    ax.scatter([-BH_MDD[sym]], [BH_ANN[sym]], marker="*", s=420, c="k", zorder=6, label=f"{sym} buy & hold")
    for cal in (1.0, 1.4):
        xx = np.linspace(0.05, 0.62, 10); ax.plot(xx, cal * xx, ls=":", lw=.9, c="grey")
        ax.annotate(f"Calmar {cal}", (xx[-1], cal * xx[-1]), fontsize=7, color="grey")
    ax.set_xlabel("max drawdown"); ax.set_title(f"{sym}: every strategy run (test 2023-2026)")
    ax.grid(alpha=.3)
axes[0].set_ylabel("annualised return")
axes[1].legend(fontsize=7, loc="upper left", ncol=2)
plt.tight_layout(); plt.savefig(S / "fig_all_strategies_single_symbol.png", dpi=110); plt.close()
print("\nfigure written:", S / "fig_all_strategies_single_symbol.png")
