"""Figures + tables for timing_transform_study / placebo_transform / timing_combination_study."""
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

D = Path("outputs/full/transform")
MAPS = ["PCT252", "PCTEXP", "Z252", "ZEXP", "RZ252", "SIGN", "TAIL", "VOLZ", "Z252_xs", "Z252_xr", "Z252_prior",
        "REG", "REGCT", "REGV", "REGR"]
AGGS = ["L1", "MA5", "MA20", "EWM10", "MED20", "DEV60", "CHG20"]
M = pd.read_parquet(D / "real_vs_null.parquet")
R = pd.read_parquet(D / "single_factor_transforms.parquet")


def cl_t(g, col):
    f = g.groupby(["factor", "symbol"])[col].mean(); return f.mean() / f.std() * np.sqrt(len(f))


# ---------------- 1. maps
rows = []
for (h, mp), g in M.groupby(["h", "map"]):
    rows.append(dict(h=h, map=mp, real_sr=g.sr_test.mean(), null_sr=g.sr_test_null.mean(), excess_sr=g.d_sr.mean(),
                     t_excess=cl_t(g, "d_sr"), real_ic=g.ic_test.mean(), excess_ic=g.d_ic.mean(), t_ic=cl_t(g, "d_ic"),
                     pos_share=(g.sr_test > 0).mean()))
for mp in ["Z252_prior"]:
    for h, g in R[R["map"] == mp].groupby("h"):
        rows.append(dict(h=h, map=mp, real_sr=g.sr_test.mean(), real_ic=g.ic_test.mean(), pos_share=(g.sr_test > 0).mean()))
maps = pd.DataFrame(rows); maps.to_csv(D / "summary_maps.csv", index=False)

fig, axes = plt.subplots(2, 3, figsize=(20, 9), sharey="row")
for j, h in enumerate([1, 5, 20]):
    t = maps[maps.h == h].set_index("map").reindex(MAPS)
    x = np.arange(len(MAPS))
    ax = axes[0, j]
    ax.bar(x - 0.2, t.real_sr, 0.4, label="real factors"); ax.bar(x + 0.2, t.null_sr, 0.4, label="null (time-shifted factors)", color="grey")
    ax.set_title(f"h={h}: mean single-factor LS Sharpe, test 2023-26"); ax.axhline(0, color="k", lw=.6)
    ax.set_xticks(x); ax.set_xticklabels(MAPS, rotation=60, fontsize=8); ax.legend(fontsize=8); ax.grid(alpha=.3, axis="y")
    ax = axes[1, j]
    ax.bar(x, t.excess_ic, color=["tab:green" if v > 2 else "tab:olive" for v in t.t_ic.fillna(0)])
    for xi, (v, tt) in enumerate(zip(t.excess_ic, t.t_ic)):
        if np.isfinite(v): ax.text(xi, v, f"{tt:.1f}", ha="center", va="bottom", fontsize=7)
    ax.set_title(f"h={h}: mean test IC minus null (label = clustered t)"); ax.axhline(0, color="k", lw=.6)
    ax.set_xticks(x); ax.set_xticklabels(MAPS, rotation=60, fontsize=8); ax.grid(alpha=.3, axis="y")
plt.tight_layout(); plt.savefig(D / "fig1_maps.png", dpi=110); plt.close()

# ---------------- 2. aggregation (Z252 and RZ252)
rows = []
for (h, a, mp), g in M[M["map"].isin(["Z252", "RZ252", "SIGN"])].groupby(["h", "agg", "map"]):
    rows.append(dict(h=h, agg=a, map=mp, real_sr=g.sr_test.mean(), null_sr=g.sr_test_null.mean(), excess_sr=g.d_sr.mean(),
                     t_excess=cl_t(g, "d_sr"), real_ic=g.ic_test.mean(), excess_ic=g.d_ic.mean(), t_ic=cl_t(g, "d_ic")))
aggs = pd.DataFrame(rows); aggs.to_csv(D / "summary_aggs.csv", index=False)
fig, axes = plt.subplots(1, 3, figsize=(18, 4.8), sharey=True)
for ax, h in zip(axes, [1, 5, 20]):
    t = aggs[(aggs.h == h) & (aggs["map"] == "Z252")].set_index("agg").reindex(AGGS)
    x = np.arange(len(AGGS))
    ax.bar(x - 0.2, t.excess_sr, 0.4, label="excess Sharpe vs null"); ax.bar(x + 0.2, t.excess_ic * 10, 0.4, label="excess IC x10", color="tab:orange")
    ax.set_xticks(x); ax.set_xticklabels(AGGS); ax.axhline(0, color="k", lw=.6); ax.grid(alpha=.3, axis="y")
    ax.set_title(f"h={h}: temporal aggregation (map = rolling z)"); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(D / "fig2_aggregation.png", dpi=110); plt.close()

# ---------------- 3. noise-band & skew factors
NF = ["N:MID_MED", "N:MID_AVG", "N:MID_END", "N:OUT_AVG", "N:SKEW_DIFF", "I:VP_MED_SKEW", "I:LOGVOL_SKEW", "I:REV_5", "I:VP_REVERSAL"]
x = M[M.factor.isin(NF) & M["map"].isin(["Z252", "RZ252", "PCT252", "SIGN", "TAIL"])]
nt = x.groupby(["factor", "h"]).agg(sr_dev=("sr_dev", "mean"), sr_test=("sr_test", "mean"), null_sr=("sr_test_null", "mean"),
                                     ic_test=("ic_test", "mean"), pos_share=("sr_test", lambda s: (s > 0).mean())).reset_index()
sg = R[R.factor.isin(NF) & (R["map"] == "Z252")].groupby(["factor", "h"]).dev_sign.agg(lambda s: abs(s.mean())).rename("sign_consistency")
nt = nt.merge(sg.reset_index(), on=["factor", "h"]); nt.to_csv(D / "summary_noise_factors.csv", index=False)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, col, title, cmap in zip(axes, ["sr_test", "ic_test", "sign_consistency"],
                                ["mean test LS Sharpe", "mean test IC", "dev-sign consistency (|mean sign| over agg x symbol)"],
                                ["RdBu", "RdBu", "Greens"]):
    p = nt.pivot(index="factor", columns="h", values=col).reindex(NF)
    v = np.nanmax(np.abs(p.values)) if col != "sign_consistency" else 1
    im = ax.imshow(p.values, cmap=cmap, vmin=-v if col != "sign_consistency" else 0, vmax=v, aspect="auto")
    ax.set_xticks(range(p.shape[1])); ax.set_xticklabels([f"h={c}" for c in p.columns]); ax.set_yticks(range(len(NF))); ax.set_yticklabels(NF)
    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            ax.text(j, i, f"{p.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title(title); plt.colorbar(im, ax=ax, fraction=.04)
plt.tight_layout(); plt.savefig(D / "fig3_noise_band_factors.png", dpi=110); plt.close()

# ---------------- 4. combinations
C = pd.read_csv(D / "combination_results.csv")
real, null = C[C["shift"] == 0], C[C["shift"] > 0]
nm = null.groupby(["symbol", "h", "pool", "method", "book"]).agg(null_sr=("sr_test", "mean"), null_sd=("sr_test", "std"),
                                                                   null_ic=("ic_test", "mean"))
CR = real.set_index(["symbol", "h", "pool", "method", "book"]).join(nm).reset_index()
CR.to_csv(D / "summary_combinations.csv", index=False)
METH = ["EW_DEV", "EW_SEL", "VOTE", "RIC_W", "RSZ_MEAN", "RSZ_MED", "PCA3", "PLS1", "CENET"]
fig, axes = plt.subplots(2, 3, figsize=(20, 9), sharey=True)
for i, sym in enumerate(["SPY", "QQQ"]):
    for j, pool in enumerate(["INTRA", "MACRO", "ALL"]):
        ax = axes[i, j]
        for k, h in enumerate([5, 20]):
            t = CR[(CR.symbol == sym) & (CR.pool == pool) & (CR.h == h) & (CR.book == "LS")].set_index("method").reindex(METH)
            xx = np.arange(len(METH)) + (k - 0.5) * 0.4
            ax.bar(xx, t.sr_test, 0.4, label=f"h={h} real")
            ax.errorbar(xx, t.null_sr, yerr=t.null_sd, fmt="k_", lw=1, capsize=3, label="null mean +/- 1sd" if k == 0 else None)
        ax.set_xticks(range(len(METH))); ax.set_xticklabels(METH, rotation=45, fontsize=8); ax.axhline(0, color="k", lw=.6)
        ax.set_title(f"{sym} {pool}: combined LS timing Sharpe (test)"); ax.grid(alpha=.3, axis="y"); ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig(D / "fig4_combinations.png", dpi=110); plt.close()

# ---------------- 5. equity curves (test period)
curves = pd.read_pickle(D / "combination_curves.pkl")
books = pickle.load(open("outputs/full/cache/factor_books.pkl", "rb"))
SEL = [(("ALL", "EW_DEV", "TILT"), "TILT EW_DEV (ALL)"), (("INTRA", "VOTE", "TILT"), "TILT VOTE (INTRA)"),
       (("INTRA", "CENET", "MV"), "MV C-LASSO (INTRA)"), (("INTRA", "HISTMEAN", "MV"), "MV hist-mean only")]
tab = []
fig, axes = plt.subplots(2, 2, figsize=(17, 9), gridspec_kw=dict(height_ratios=[3, 1.3]))
for j, sym in enumerate(["SPY", "QQQ"]):
    bh = books[(sym, "BH", 0, "BH")].loc["2023":]
    series = {"Buy & hold": bh}
    for (pool, meth, bk), lab in SEL:
        series[lab] = curves[(sym, 20, pool)][(meth, bk)].loc["2023":]
    ls = curves[(sym, 20, "INTRA")][("VOTE", "LS")].loc["2023":]
    series["B&H + 1.0x LS VOTE (INTRA)"] = bh + ls
    for lab, r in series.items():
        eq = (1 + r).cumprod(); dd = eq / eq.cummax() - 1
        srr = r.mean() / r.std() * np.sqrt(252)
        axes[0, j].plot(eq, lw=1.3 if lab == "Buy & hold" else 1.0, color="k" if lab == "Buy & hold" else None,
                        label=f"{lab} (SR {srr:.2f}, MDD {dd.min():.1%})")
        axes[1, j].plot(dd, lw=0.8, color="k" if lab == "Buy & hold" else None)
        ann = eq.iloc[-1] ** (252 / len(r)) - 1
        tab.append(dict(symbol=sym, strategy=lab, sharpe=srr, ann_ret=ann, vol=r.std() * np.sqrt(252), mdd=dd.min(),
                        excess_ann_vs_bh=ann - ((1 + bh).prod() ** (252 / len(bh)) - 1)))
    axes[0, j].set_title(f"{sym} h=20 combined signals, test 2023-01 .. 2026-10"); axes[0, j].legend(fontsize=8); axes[0, j].grid(alpha=.3)
    axes[1, j].set_title("drawdown"); axes[1, j].grid(alpha=.3)
plt.tight_layout(); plt.savefig(D / "fig5_equity.png", dpi=110); plt.close()
tab = pd.DataFrame(tab); tab.to_csv(D / "summary_equity.csv", index=False)

pd.set_option("display.width", 250)
print(maps.round(3).to_string()); print(tab.round(3).to_string()); print(nt.round(3).to_string())
