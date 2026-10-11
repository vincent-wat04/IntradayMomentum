"""Summarise the full-sample (2019-01..2026-10, test from 2023-01-03) factor and model studies.

    python analyze_full.py            -> outputs/full/report/*.csv, *.png

Outputs
  model_summary.csv        every (symbol, horizon, execution, feature_set, model, window): test IC,
                           RankIC, HAC t of the forecast slope (lag = horizon), hit rate, long/cash
                           Sharpe / return / MDD vs buy&hold on the same dates, IC 2023-24 vs 2025-26
  model_ic_heatmap.png     test IC by model x horizon (A vs B, SPY / QQQ, next_open)
  equity_*.png             daily long/cash equity (forecast > 0 -> long) for each model, h=1/5/20
  factor_top.csv           single factors that are stable (same IC sign in 2019-22 and 2023-26 on
                           both SPY and QQQ), ranked by |HAC t|
  selected_factor_corr_*.png / .csv  correlation of the screened features (union over horizons)
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from index_timing_framework import load_market, nw_slope_t  # noqa: E402
from run_full_extended import config_from  # noqa: E402

OUT = Path("outputs/full"); REP = OUT / "report"; REP.mkdir(parents=True, exist_ok=True)
DATA = Path("data")
MODEL_ORDER = ["ridge:60", "ridge:120", "ridge:expanding", "elastic_net:60", "elastic_net:120",
               "elastic_net:expanding", "raven_lite:expanding", "raven_full:expanding"]


def load_models():
    rows = []
    for f in glob.glob(str(OUT / "models_linear_h*.json")) + glob.glob(str(OUT / "models_raven_full_h*.json")):
        rows += json.load(open(f))
    return [r for r in rows if r.get("status") == "ok"]


def daily_long_cash(panel, preds, execution, fee_bp=1.0):
    """Same convention as raven_lite_timing.backtest, but returns the daily series."""
    d = {panel.dates.get_loc(p["signal_day"]): p["forecast"] for p in preds}
    lo, hi = min(d) + 1, min(max(d) + 2, len(panel.dates))
    prev, out = 0.0, []
    for j in range(lo, hi):
        tgt = float(d[j - 1] > 0) if (j - 1) in d else prev
        on = panel.daily_open[j] / panel.close[j - 1] - 1
        intra = panel.close[j] / panel.daily_open[j] - 1
        m = panel.close[j] / panel.close[j - 1] - 1
        g = (1 + prev * on) * (1 + tgt * intra) - 1 if execution == "next_open" else prev * m
        out.append((panel.dates[j], g - fee_bp * 1e-4 * abs(tgt - prev), m))
        prev = tgt
    s = pd.DataFrame(out, columns=["date", "strat", "bh"]).set_index("date")
    s.index = pd.to_datetime(s.index)
    return s


def perf(x):
    eq = (1 + x).cumprod()
    return {"ret_ann": float(eq.iloc[-1] ** (252 / len(x)) - 1), "vol": float(x.std() * np.sqrt(252)),
            "sharpe": float(x.mean() / x.std() * np.sqrt(252)) if x.std() > 0 else np.nan,
            "mdd": float((eq / eq.cummax() - 1).min())}


def model_tables(panels):
    rows, curves = [], {}
    for r in load_models():
        p = pd.DataFrame(r["predictions"])
        x, y = p.forecast.to_numpy(), p.realized.to_numpy()
        yr = p.signal_day.str[:4].astype(int)
        ic = lambda m: float(np.corrcoef(x[m], y[m])[0, 1]) if m.sum() > 20 and np.std(x[m]) > 0 else np.nan  # noqa: E731
        s = daily_long_cash(panels[r["symbol"]], r["predictions"], r["execution"])
        ps, pb = perf(s.strat), perf(s.bh)
        key = f"{r['model']}:{r['training_window']}"
        rows.append({"symbol": r["symbol"], "horizon": r["horizon"], "execution": r["execution"],
                     "feature_set": r["feature_set"], "model": key, "n_test": len(p),
                     "ic": ic(np.ones(len(p), bool)), "rank_ic": r["metric"].get("rank_ic"),
                     "hac_t": nw_slope_t(x, y, r["horizon"]),
                     "hit": float(np.mean(np.sign(x) == np.sign(y))),
                     "ic_2023_24": ic(yr.isin([2023, 2024]).to_numpy()), "ic_2025_26": ic(yr.isin([2025, 2026]).to_numpy()),
                     "time_long": float(np.mean(x > 0)),
                     "strat_sharpe": ps["sharpe"], "strat_ret_ann": ps["ret_ann"], "strat_mdd": ps["mdd"],
                     "bh_sharpe": pb["sharpe"], "bh_ret_ann": pb["ret_ann"], "bh_mdd": pb["mdd"],
                     "features": ",".join(f"{q['name']}:{q['k']}" for q in r["selected"])})
        curves[(r["symbol"], r["horizon"], r["execution"], r["feature_set"], key)] = s
    df = pd.DataFrame(rows).sort_values(["execution", "symbol", "horizon", "feature_set", "model"])
    df.to_csv(REP / "model_summary.csv", index=False)
    return df, curves


def plot_ic_heatmap(df):
    fig, ax = plt.subplots(2, 2, figsize=(16, 9))
    for i, sym in enumerate(["SPY", "QQQ"]):
        for j, fs in enumerate(["A_intraday", "B_intraday+macro"]):
            d = df[(df.symbol == sym) & (df.feature_set == fs) & (df.execution == "next_open")]
            pv = d.pivot(index="model", columns="horizon", values="ic").reindex(MODEL_ORDER)
            tv = d.pivot(index="model", columns="horizon", values="hac_t").reindex(MODEL_ORDER)
            a = ax[i, j]; im = a.imshow(pv.values, cmap="RdBu_r", vmin=-.4, vmax=.4, aspect="auto")
            for (r_, c_), v in np.ndenumerate(pv.values):
                t = tv.values[r_, c_]
                a.text(c_, r_, f"{v:+.2f}\n(t {t:+.1f})" if np.isfinite(v) else "", ha="center", va="center",
                       fontsize=7, fontweight="bold" if np.isfinite(t) and abs(t) > 2 else "normal")
            a.set_xticks(range(pv.shape[1])); a.set_xticklabels([f"h={h}" for h in pv.columns])
            a.set_yticks(range(len(MODEL_ORDER))); a.set_yticklabels(MODEL_ORDER, fontsize=8)
            a.set_title(f"{sym} {fs}: test IC 2023-01..2026-10 (next_open), bold = |HAC t|>2")
    fig.colorbar(im, ax=ax, shrink=.6); plt.savefig(REP / "model_ic_heatmap.png", dpi=105, bbox_inches="tight"); plt.close(fig)


def plot_equity(curves, df):
    for h in (1, 5, 20):
        fig, ax = plt.subplots(2, 2, figsize=(16, 9), sharex=True)
        for i, sym in enumerate(["SPY", "QQQ"]):
            for j, fs in enumerate(["A_intraday", "B_intraday+macro"]):
                a = ax[i, j]; bh = None
                for m in ["ridge:expanding", "elastic_net:expanding", "elastic_net:120", "raven_lite:expanding", "raven_full:expanding"]:
                    s = curves.get((sym, h, "next_open", fs, m))
                    if s is None:
                        continue
                    a.plot((1 + s.strat).cumprod(), lw=1.1, label=f"{m} (SR {perf(s.strat)['sharpe']:.2f})"); bh = s.bh
                if bh is not None:
                    a.plot((1 + bh).cumprod(), "k--", lw=1.4, label=f"buy&hold (SR {perf(bh)['sharpe']:.2f})")
                a.set_title(f"{sym} h={h} {fs}: long/cash if forecast>0, next_open, 1bp"); a.legend(fontsize=7); a.grid(alpha=.3)
        plt.tight_layout(); plt.savefig(REP / f"equity_h{h}.png", dpi=105); plt.close(fig)


def factor_top():
    e = pd.read_csv(OUT / "extended_factor_metrics.csv"); e["family"] = "intraday:" + e.source.astype(str)
    m = pd.read_csv(OUT / "macro_factor_metrics.csv"); m["family"] = "macro:" + m.family.astype(str)
    d = pd.concat([e, m]); d = d[(d.scope == "available") & (d.execution == "next_open") & d.version.isin(["raw", "orthog"])]
    d["ic_pre"] = d[["ic_2019", "ic_2020", "ic_2021", "ic_2022"]].mean(axis=1)
    d["ic_post"] = d[["ic_2023", "ic_2024", "ic_2025", "ic_2026"]].mean(axis=1)
    g = d.groupby(["family", "factor", "aggregation", "version", "horizon"]).agg(
        ic_spy=("pearson_ic", "first"), ic_qqq=("pearson_ic", "last"), t_mean=("hac_slope_t", "mean"),
        pre=("ic_pre", "mean"), post=("ic_post", "mean"),
        spy_pre=("ic_pre", "first"), qqq_pre=("ic_pre", "last"), spy_post=("ic_post", "first"), qqq_post=("ic_post", "last")).reset_index()
    sg = np.sign(g[["spy_pre", "qqq_pre", "spy_post", "qqq_post"]])
    g["stable"] = (sg.nunique(axis=1) == 1) & (sg.abs().sum(axis=1) == 4)
    g["abs_t"] = g.t_mean.abs()
    out = g[g.stable].sort_values("abs_t", ascending=False)
    out.to_csv(REP / "factor_top.csv", index=False)
    summ = g.groupby("horizon").apply(lambda x: pd.Series({"n": len(x), "share_|t|>2": (x.abs_t > 2).mean(),
                                                           "stable_share": x.stable.mean(),
                                                           "stable_and_|t|>2": int((x.stable & (x.abs_t > 2)).sum())}))
    summ.to_csv(REP / "factor_summary_by_horizon.csv")
    fam = g[g.stable & (g.abs_t > 2)].groupby(["horizon", "family"]).size().unstack(0).fillna(0).astype(int)
    fam.to_csv(REP / "factor_stable_by_family.csv")
    return out, summ, fam


def selected_corr(df):
    """Correlation of screened features (union over horizons, next_open), test + pretest days."""
    import pandas as pd  # noqa
    res = {}
    for sym in ["SPY", "QQQ"]:
        A = pd.read_pickle(OUT / "cache" / f"{sym}_A.pkl"); M = pd.read_pickle(OUT / "cache" / f"{sym}_M.pkl")
        allc = pd.concat([A, M], axis=1)
        feats = df[(df.symbol == sym) & (df.execution == "next_open")].groupby("feature_set").features.apply(
            lambda s: sorted({f for row in s for f in row.split(",")})).to_dict()
        for fs, fl in feats.items():
            cols = []
            for f in fl:
                name, k = f.rsplit(":", 1)
                cols.append((name, int(k)))
            X = allc[cols].copy(); X.columns = [f"{c[0].replace('M:', '[M]')}:{c[1]}" for c in cols]
            C = X.corr(min_periods=200)
            C.to_csv(REP / f"selected_factor_corr_{sym}_{fs}.csv")
            n = len(C)
            fig, ax = plt.subplots(figsize=(0.55 * n + 3, 0.55 * n + 2))
            im = ax.imshow(C.values, cmap="RdBu_r", vmin=-1, vmax=1)
            for (i, j), v in np.ndenumerate(C.values):
                if i != j and np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6)
            ax.set_xticks(range(n)); ax.set_xticklabels(C.columns, rotation=90, fontsize=7)
            ax.set_yticks(range(n)); ax.set_yticklabels(C.columns, fontsize=7)
            ax.set_title(f"{sym} {fs}: correlation of screened features (daily, 2019-2026)")
            fig.colorbar(im, shrink=.6); plt.tight_layout(); plt.savefig(REP / f"selected_factor_corr_{sym}_{fs}.png", dpi=105); plt.close(fig)
            off = C.where(~np.eye(n, dtype=bool)).abs().stack()
            res[(sym, fs)] = {"n_features": n, "mean_abs_corr": float(off.mean()), "max_abs_corr": float(off.max()),
                              "pairs_gt_0.5": int((off > .5).sum() // 2),
                              "top_pairs": {f"{a} | {b}": v for (a, b), v in off.sort_values(ascending=False).iloc[::2].head(8).round(3).items()}}
    return res


def main():
    cfg = config_from(DATA)
    panels = {s: load_market(str(DATA / f"{s}_5min.csv"), cfg) for s in ("SPY", "QQQ")}
    df, curves = model_tables(panels)
    plot_ic_heatmap(df); plot_equity(curves, df)
    top, summ, fam = factor_top()
    corr = selected_corr(df)
    pd.set_option("display.width", 250)
    print(summ.round(3).to_string()); print(fam.to_string())
    print(top.head(25)[["family", "factor", "aggregation", "version", "horizon", "ic_spy", "ic_qqq", "t_mean", "pre", "post"]].round(3).to_string())
    g = df.groupby(["feature_set", "model"]).agg(ic=("ic", "mean"), ic_2324=("ic_2023_24", "mean"), ic_2526=("ic_2025_26", "mean"),
                                                 sig_pos=("hac_t", lambda t: int((t > 2).sum())), sig_neg=("hac_t", lambda t: int((t < -2).sum())),
                                                 n=("ic", "size"), sr=("strat_sharpe", "mean"), sr_bh=("bh_sharpe", "mean"),
                                                 beat_bh=("strat_sharpe", lambda s: 0)).reset_index()
    g["beat_bh"] = df.assign(b=df.strat_sharpe > df.bh_sharpe).groupby(["feature_set", "model"]).b.mean().values
    g.to_csv(REP / "model_overview.csv", index=False)
    print(g.round(3).to_string())
    by_h = df.groupby(["feature_set", "horizon"]).agg(ic=("ic", "mean"), sr=("strat_sharpe", "mean"), bh=("bh_sharpe", "mean")).unstack(0)
    by_h.to_csv(REP / "model_by_horizon.csv"); print(by_h.round(3).to_string())
    json.dump({f"{k[0]}|{k[1]}": v for k, v in corr.items()}, open(REP / "selected_factor_corr_summary.json", "w"), indent=1, default=str)
    for k, v in corr.items():
        print(k, {a: b for a, b in v.items() if a != "top_pairs"}); print("   top:", v["top_pairs"])


if __name__ == "__main__":
    main()
