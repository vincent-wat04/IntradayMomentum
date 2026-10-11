"""Stage 6: turn the alpha sleeve into actual strategy return.

Three things are being answered here:
  (4) horizon combination   -- micro sleeves (h=1,2) carry microstructure information, the h=20 sleeve
                               carries macro/risk-premium information.  Are they additive, and with what
                               weights?  Weights are solved on DEV ONLY (2019-2022).
  (5) sizing and vehicle    -- one symbol or both?  How much leverage?  Implemented with REAL leveraged
                               ETF prices (TQQQ/QLD/UPRO/SSO), so the financing cost, expense ratio and
                               daily-rebalance drag are the market's, not a model's.

Exposure E_t is implemented with the cheapest ladder that reaches it:
      E in [0,1]   -> E in the 1x ETF, the rest in cash
      E in (1,2]   -> (E-1) in the 2x ETF, (2-E) in the 1x ETF
      E in (2,3]   -> (E-2) in the 3x ETF, (3-E) in the 2x ETF
This always uses the lowest leverage that reaches E, which minimises the drag measured above
(2x: -4.4pp/yr vs naive 2x, 3x: -8.5pp/yr).

Risk control, all point-in-time:
  * volatility target on the TOTAL portfolio, using a trailing 63d variance known at t
  * drawdown brake (Van Hemert, Ganz, Harvey, Rattray, Sanchez Martin & Yawitch, "Drawdowns"):
    the multiplier falls linearly once the strategy's own realised drawdown, computed from equity up to
    t-1 only, passes DD_SOFT, and reaches DD_FLOOR at DD_HARD.

Every configuration is also run on the circularly-shifted (null) feature panel.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from pathlib import Path

import numpy as np
import pandas as pd

import timing_model_strategy as MS
import timing_transform_study as T
from factor_sharpe import run_book

OUT = Path("outputs/full/strategy"); OUT.mkdir(parents=True, exist_ok=True)
SLEEVE_HS = (1, 2, 5, 20)
MODELS = ("RIDGE", "AVG")
SPLIT = "2023-01-03"
FEE = 1.0
VOL_TARGETS = (0.0, 0.15, 0.20, 0.25)      # 0 = no volatility target
MAX_E = 3.0
DD_SOFT, DD_HARD, DD_FLOOR = 0.12, 0.25, 0.35
LEV = {"SPY": {1: "SPY", 2: "SSO", 3: "UPRO"}, "QQQ": {1: "QQQ", 2: "QLD", 3: "TQQQ"}}


# ------------------------------------------------------------------ sleeves
def sleeve_positions(sym, shift):
    """LS position per horizon, from the stage-3 pipeline (VOLSTD target, RIDGE+AVG)."""
    G = MS._G[sym]; rv = G["rv"]
    X0 = G["X"]
    out = {}
    for h in SLEEVE_HS:
        X = X0.copy()
        if shift:
            k = int(np.random.default_rng(7000 + 13 * shift + h).integers(252, len(X) - 252))
            X = np.roll(X, k, axis=0)
        y = G["Y"][h] / np.sqrt(h * rv)
        sig, _ = MS.build_signals(X, y, h, rv, np.random.default_rng(12345 + h))
        comps = []
        for m in MODELS:
            s = sig[m]; sd = pd.Series(s).rolling(252, min_periods=126).std().shift(1).to_numpy()
            comps.append(s / np.where(sd > 0, sd, np.nan))
        z = np.nanmean(np.vstack(comps), 0)
        sd = pd.Series(z).rolling(252, min_periods=126).std().shift(1).to_numpy()
        out[h] = np.clip(T.smooth(z / np.where(sd > 0, sd, np.nan) / 2.0, h), -1, 1)
    return out


def dev_weights(rets: dict, idx, kind="riskparity"):
    """Weights over sleeves.

    DEFAULT IS RISK PARITY (inverse dev volatility) and that is deliberate: stage 1 showed the
    cross-sectional correlation between dev Sharpe and test Sharpe is only 0.02-0.16, so any weighting
    that needs a RETURN estimate is fitting noise.  Inverse-vol needs only a RISK estimate, which is
    far better behaved.  'sharpe' is kept to document how much damage the return estimate does.
    """
    dev = {k: v.loc[:"2022"] for k, v in rets.items()}
    sr = {k: (v.mean() / v.std() * np.sqrt(252)) for k, v in dev.items()}
    vol = {k: v.std() * np.sqrt(252) for k, v in dev.items()}
    if kind == "equal":
        w = {k: 1.0 for k in rets}
    elif kind == "sharpe":
        w = {k: max(sr[k], 0.0) / vol[k] for k in rets}
    else:
        w = {k: 1.0 / vol[k] for k in rets}
    tot = sum(w.values()) or 1.0
    return {k: v / tot for k, v in w.items()}, sr


# ------------------------------------------------------------------ vehicle
def ladder_returns(sym, E: np.ndarray, dates, lev_px: pd.DataFrame, fee_bp=FEE):
    """Daily return of the cheapest 1x/2x/3x ladder that reaches exposure E (E clipped to [0, MAX_E])."""
    E = np.clip(np.nan_to_num(E, nan=1.0), 0.0, MAX_E)
    r = {k: lev_px[LEV[sym][k]].reindex(dates).pct_change().to_numpy() for k in (1, 2, 3)}
    w = {1: np.zeros(len(E)), 2: np.zeros(len(E)), 3: np.zeros(len(E))}
    lo = E <= 1
    w[1][lo] = E[lo]
    m2 = (E > 1) & (E <= 2)
    w[2][m2] = E[m2] - 1; w[1][m2] = 2 - E[m2]
    m3 = E > 2
    w[3][m3] = E[m3] - 2; w[2][m3] = 3 - E[m3]
    W = np.vstack([w[1], w[2], w[3]])
    R = np.vstack([np.nan_to_num(r[1]), np.nan_to_num(r[2]), np.nan_to_num(r[3])])
    prev = np.c_[np.zeros(3), W[:, :-1]]                      # weight set at t-1 close earns day t
    gross = (prev * R).sum(0)
    cost = fee_bp * 1e-4 * np.abs(np.diff(np.c_[np.zeros(3), W], axis=1)).sum(0)
    return pd.Series(gross - cost, index=pd.to_datetime(dates))


def dd_brake(r: pd.Series) -> np.ndarray:
    """Multiplier from the strategy's own realised drawdown using equity up to t-1 only."""
    eq = (1 + r.fillna(0)).cumprod().shift(1)
    dd = (eq / eq.cummax() - 1).fillna(0).to_numpy()
    x = (np.abs(dd) - DD_SOFT) / (DD_HARD - DD_SOFT)
    return np.clip(1 - np.clip(x, 0, 1) * (1 - DD_FLOOR), DD_FLOOR, 1.0)


def metrics(r, bench=None):
    r = r.dropna(); eq = (1 + r).cumprod(); dd = eq / eq.cummax() - 1
    ann = float(eq.iloc[-1] ** (252 / len(r)) - 1)
    out = dict(sharpe=float(r.mean() / r.std() * np.sqrt(252)), ann_ret=ann, vol=float(r.std() * np.sqrt(252)),
               mdd=float(dd.min()), calmar=ann / abs(dd.min()), worst_day=float(r.min()),
               skew=float(r.skew()), hit=float((r > 0).mean()))
    if bench is not None:
        b = bench.reindex(r.index).fillna(0)
        beta = np.cov(r, b)[0, 1] / b.var(); res = r - beta * b
        out.update(excess_ann=ann - float((1 + b).prod() ** (252 / len(b)) - 1), beta=float(beta),
                   alpha=float(res.mean() * 252), ir=float(res.mean() / res.std() * np.sqrt(252)))
    return out


# ------------------------------------------------------------------ main
def main():
    MS._init()
    lev_px = pd.read_csv("data/lev/levetf_close.csv", index_col=0)
    dates = MS._G["SPY"]["panel"].dates
    rows, curves, diag = [], {}, []

    for shift in (0, 1):
        tag = "" if shift == 0 else " [null]"
        sl_ret, sl_pos, bh = {}, {}, {}
        for sym in ("SPY", "QQQ"):
            panel = MS._G[sym]["panel"]
            pos = sleeve_positions(sym, shift)
            sl_pos[sym] = pos
            sl_ret[sym] = {h: run_book(panel, pd.Series(np.nan_to_num(p), index=panel.dates), fee_bp=FEE)
                           for h, p in pos.items()}
            bh[sym] = run_book(panel, pd.Series(1.0, index=panel.dates), fee_bp=FEE)

        # ---- (4) horizon combination, weights from dev only
        comb_pos, comb_ret = {}, {}
        for sym in ("SPY", "QQQ"):
            w, sr = dev_weights(sl_ret[sym], dates)
            comb_pos[sym] = sum(w[h] * np.nan_to_num(sl_pos[sym][h]) for h in SLEEVE_HS)
            comb_ret[sym] = run_book(MS._G[sym]["panel"], pd.Series(comb_pos[sym], index=dates), fee_bp=FEE)
            if shift == 0:
                diag.append(dict(symbol=sym, **{f"w_h{h}": round(w[h], 3) for h in SLEEVE_HS},
                                 **{f"devSR_h{h}": round(sr[h], 2) for h in SLEEVE_HS}))
            for h in SLEEVE_HS:
                rows.append(dict(group="sleeve", name=f"{sym} LS h={h}" + tag, shift=shift,
                                 **metrics(sl_ret[sym][h].loc["2023":], bh[sym].loc["2023":])))
            rows.append(dict(group="sleeve", name=f"{sym} LS combined" + tag, shift=shift,
                             **metrics(comb_ret[sym].loc["2023":], bh[sym].loc["2023":])))

        # sleeve correlation across horizons (test period, real run only)
        if shift == 0:
            for sym in ("SPY", "QQQ"):
                C = pd.DataFrame({h: sl_ret[sym][h].loc["2023":] for h in SLEEVE_HS}).corr()
                C.to_csv(OUT / f"sleeve_corr_{sym}.csv")
                print(f"\n{sym} sleeve correlation (test):\n", C.round(2).to_string())

        # ---- (5) sizing: core + sleeve, one symbol or both, implemented on the ETF ladder
        for core in ("SPY", "QQQ", "BOTH"):
            syms = ("SPY", "QQQ") if core == "BOTH" else (core,)
            benchmark = sum(bh[s] for s in syms) / len(syms)
            sleeve_pos = sum(comb_pos[s] for s in syms) / len(syms)
            rv = sum(MS._G[s]["rv"] for s in syms) / len(syms)
            split = MS._G[syms[0]]["split"]
            for vt in VOL_TARGETS:
                scale = np.clip(vt / np.sqrt(252 * rv), 0.2, MAX_E) if vt else np.ones(len(dates))
                for lam in (0.0, 0.25, 0.5, 1.0):
                    for brake in (False, True):
                        # exposure = (core 1.0 tilted by the sleeve) x vol-target scale.  lam is FIXED,
                        # not fitted, so nothing here is estimated from dev returns.
                        E = np.clip((1.0 + lam * sleeve_pos) * scale, 0, MAX_E)
                        if brake:    # pass 1 gives the equity path the brake needs, pass 2 trades it
                            r0 = sum(ladder_returns(s, E, dates, lev_px) for s in syms) / len(syms)
                            E = np.clip(E * dd_brake(r0), 0, MAX_E)
                        r = sum(ladder_returns(s, E, dates, lev_px) for s in syms) / len(syms)
                        avgE = float(np.nanmean(E[split:]))
                        # EXPOSURE-MATCHED control: constant exposure equal to this strategy's own
                        # realised average, same ETF ladder, same costs.  Any Sharpe difference is
                        # then attributable to the timing of the leverage, not to its amount.
                        rc = sum(ladder_returns(s, np.full(len(dates), avgE), dates, lev_px) for s in syms) / len(syms)
                        mm = metrics(r.loc["2023":], benchmark.loc["2023":])
                        mc = metrics(rc.loc["2023":], benchmark.loc["2023":])
                        nm = (f"{core} lam={lam:g}, vt={vt:.0%}" if vt else f"{core} lam={lam:g}, no vt") + \
                             (", DD brake" if brake else "")
                        rows.append(dict(group="portfolio", name=nm + tag, shift=shift, core=core, vol_target=vt,
                                         lam=lam, brake=brake, avg_E=avgE, max_E=float(np.nanmax(E)),
                                         sr_matched=mc["sharpe"], ann_matched=mc["ann_ret"], mdd_matched=mc["mdd"],
                                         d_sr_vs_matched=mm["sharpe"] - mc["sharpe"],
                                         d_ann_vs_matched=mm["ann_ret"] - mc["ann_ret"], **mm))
                        if shift == 0:
                            curves[nm] = r
                            curves[f"{core} constant E={avgE:.2f}"] = rc
            if shift == 0:
                curves[f"{core} buy & hold"] = benchmark
                rows.append(dict(group="benchmark", name=f"{core} buy & hold", shift=0, core=core,
                                 **metrics(benchmark.loc["2023":], benchmark.loc["2023":])))

    # raw leveraged ETF buy & hold, for reference
    for c in ("TQQQ", "QLD", "UPRO", "SSO"):
        r = lev_px[c].reindex(dates).pct_change(); r.index = pd.to_datetime(dates)
        rows.append(dict(group="benchmark", name=f"{c} buy & hold", shift=0,
                         **metrics(r.loc["2023":], (lev_px["QQQ"] if "QQ" in c else lev_px["SPY"])
                                   .reindex(dates).pct_change().set_axis(pd.to_datetime(dates)).loc["2023":])))
        curves[f"{c} buy & hold"] = r

    tab = pd.DataFrame(rows); tab.to_csv(OUT / "leveraged_portfolio_metrics.csv", index=False)
    pd.to_pickle(curves, OUT / "leveraged_curves.pkl")
    pd.DataFrame(diag).to_csv(OUT / "sleeve_weights.csv", index=False)
    pd.set_option("display.width", 280); pd.set_option("display.max_rows", 400)
    print("\n=== dev-estimated sleeve weights ==="); print(pd.DataFrame(diag).to_string(index=False))
    print("\n=== sleeves (test) ===")
    print(tab[tab.group == "sleeve"][["name", "sharpe", "ann_ret", "vol", "mdd", "beta", "alpha", "ir"]].round(3).to_string(index=False))
    print("\n=== benchmarks ===")
    bcols = ["name", "sharpe", "ann_ret", "vol", "mdd", "calmar", "excess_ann"]
    print(tab[tab.group == "benchmark"][bcols].round(3).to_string(index=False))
    print("\n=== portfolios: does timing the leverage beat constant leverage at the SAME avg exposure? ===")
    cols = ["name", "sharpe", "ann_ret", "vol", "mdd", "calmar", "avg_E", "sr_matched", "ann_matched",
            "mdd_matched", "d_sr_vs_matched", "d_ann_vs_matched"]
    P = tab[(tab.group == "portfolio") & (tab["shift"] == 0)]
    print(P[cols].round(3).to_string(index=False))
    print("\n=== same, null features ===")
    print(tab[(tab.group == "portfolio") & (tab["shift"] == 1)][
        ["name", "sharpe", "d_sr_vs_matched", "d_ann_vs_matched"]].round(3).to_string(index=False))
    print("\n=== timing gain vs matched constant leverage: mean by core x lambda (real minus null) ===")
    pv = tab[tab.group == "portfolio"].pivot_table(index=["core", "lam"], columns="shift",
                                                   values="d_sr_vs_matched", aggfunc="mean")
    pv.columns = ["real", "null"]; pv["edge"] = pv["real"] - pv["null"]
    print(pv.round(3).to_string())
    return tab, curves


if __name__ == "__main__":
    main()
