"""Keep the style exposure, then attack turnover.  Five reduction methods compared.

The attribution ladder showed the two things that destroyed the composite were
  (a) style + past-K neutralisation   -1.14 Sharpe
  (b) moving from h=20 overlapping tranches to a P=5 period hold   -0.39 Sharpe (turnover 9 -> 22)
You have said style exposure can be kept, so (a) is now a choice rather than a constraint and the
remaining problem is (b).  Methods tested, all point-in-time:

  NONE      set the target weights at every period boundary and hold
  TRANCHE   split capital into h slices, each rebalanced once every h days  (the classic approach;
            this is what the single-factor tests implicitly used)
  GP        w_t = (1-a) w_{t-1} + a * aim_t, a = 1/(1+h/2)
            Garleanu & Pedersen (2013) "aim in front of the target, trade partially towards the aim"
  BUFFER    no-trade band: only move a name's weight if the target differs from the current weight by
            more than BUF of the average gross weight (hysteresis); classic practitioner method
  SHRINK    shrink the target toward the current book by a factor proportional to the estimated cost
            of the trade divided by the expected gain, a one-step approximation to the cost-aware
            optimum
  COMBO     TRANCHE + GP

Reported against SPY.  Both neutralisation levels are run so the cost of keeping/removing style is
visible in the same table.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import ElasticNet, Ridge

import cross_section_study as XS
import xs_factor_library as XL
import xs_pipeline_v2 as V

OUT = Path("outputs/full/xs2")
SPLIT = "2023-01-03"
FEE = 5.0
BUF = 0.75          # no-trade band, in units of the average per-name gross weight
PERIODS = (5, 10, 20)
METHODS = ("NONE", "TRANCHE", "GP", "BUFFER", "SHRINK", "COMBO")
MODELS = ("EW", "RIDGE", "ELASTICNET")
NEUTRAL = ("sector only", "+style+pastK")


# ------------------------------------------------------------------ turnover methods
def apply_method(target: pd.DataFrame, method: str, P: int):
    """target = desired weights each day (already rank-based and normalised). Returns held weights."""
    if method == "NONE":
        return V.period_hold(target, P)
    if method == "TRANCHE":
        return (sum(target.shift(1 + k) for k in range(P)) / P).fillna(0.0)
    T = target.shift(1).fillna(0.0)
    A = T.to_numpy()
    W = np.zeros_like(A)
    prev = np.zeros(A.shape[1])
    if method == "GP":
        a = 1.0 / (1.0 + P / 2.0)
        for t in range(len(A)):
            prev = (1 - a) * prev + a * np.nan_to_num(A[t])
            W[t] = prev
    elif method == "BUFFER":
        thr = BUF * np.nanmean(np.abs(A[np.isfinite(A).all(1)])) if np.isfinite(A).any() else 0.0
        for t in range(len(A)):
            tgt = np.nan_to_num(A[t])
            move = np.abs(tgt - prev) > thr
            prev = np.where(move, tgt, prev)
            W[t] = prev
    elif method == "SHRINK":
        # one-step cost-aware shrink: trade the fraction g/(g+c) of the gap, where c is the round-trip
        # cost of the trade and g the expected gain over the holding period implied by the signal change
        c = 2 * FEE * 1e-4
        for t in range(len(A)):
            tgt = np.nan_to_num(A[t])
            gap = tgt - prev
            g = np.abs(gap) * 0.004 * P          # 40bp/day/unit-weight scale for the expected edge
            frac = np.where(np.abs(gap) > 0, g / (g + c), 0.0)
            prev = prev + frac * gap
            W[t] = prev
    elif method == "COMBO":
        tr = (sum(target.shift(1 + k) for k in range(P)) / P).fillna(0.0).to_numpy()
        a = 1.0 / (1.0 + P / 2.0)
        for t in range(len(tr)):
            prev = (1 - a) * prev + a * np.nan_to_num(tr[t])
            W[t] = prev
    out = pd.DataFrame(W, index=target.index, columns=target.columns)
    if method == "BUFFER":
        return out.fillna(0.0)        # already gross~1 by construction; do NOT renormalise daily
    return out.div(out.abs().sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)


def run(w, r, fee=FEE):
    gross = (w * r).sum(axis=1)
    cost = fee * 1e-4 * w.diff().abs().sum(axis=1)
    return gross - cost, float(w.diff().abs().sum(axis=1).mean() * 252)


def main():
    px, op, hi, lo, vo = XS.load()
    sec = pd.Series({c: XS.SECTOR[c] for c in px.columns})
    F0, r = XL.build(px, op, hi, lo, vo)
    beta = F0["BETA"]
    split = int(np.searchsorted(px.index, SPLIT))
    controls = {f"PASTR_{k}": np.log(px).diff(k) for k in V.PASTK}
    spy = yf.download("SPY", start="2018-11-01", end="2026-10-03", auto_adjust=True,
                      progress=False)["Close"]
    spy.index = spy.index.strftime("%Y-%m-%d")
    spy = spy.reindex(px.index).ffill().pct_change().squeeze()

    Zsec = {k: V.xs_z(V.winsor(v)) for k, v in F0.items()}
    for k in Zsec:
        o = Zsec[k].copy()
        for s in sec.unique():
            c = sec[sec == s].index.intersection(o.columns)
            if len(c) >= 2:
                o[c] = o[c].sub(o[c].mean(axis=1), axis=0)
        Zsec[k] = o
    NEU = {"sector only": Zsec, "+style+pastK": V.neutralise(F0, sec, controls, use_pastk=True)}

    rows, curves = [], {}
    for neu, Z in NEU.items():
        keys = sorted(Z)
        cube = np.stack([Z[k].to_numpy(float) for k in keys], -1)
        for P in PERIODS:
            fwd = XS.fwd_ret(px, P)
            Y = fwd.to_numpy(float)
            # ---- walk-forward screening + three models -> score panels
            scores = {m: np.full((len(px), len(px.columns)), np.nan) for m in MODELS}
            idx, sgn, fitted, mu, sd = np.array([], int), np.array([]), {}, None, None
            need = False
            rng = np.random.default_rng(3 + P)
            for t in range(len(px)):
                if t % 63 == 0:
                    ni, ns = V.screen(cube, Y, t, P)
                    if len(ni) != len(idx) or not np.array_equal(ni, idx):
                        idx, sgn = ni, ns; fitted, mu, sd, need = {}, None, None, True
                if len(idx) == 0:
                    continue
                Zc = cube[:, :, idx]
                if (t % 42 == 0 or need) and t - P - 1 > 252:
                    X = Zc[: t - P - 1].reshape(-1, len(idx)); y = Y[: t - P - 1].reshape(-1)
                    m = np.isfinite(y) & np.isfinite(X).all(1)
                    if m.sum() > 20000:
                        A, yy = X[m], y[m]
                        if len(A) > 40000:
                            s_ = rng.choice(len(A), 40000, replace=False); A, yy = A[s_], yy[s_]
                        mu, sd = A.mean(0), A.std(0) + 1e-9
                        An = (A - mu) / sd; yn = (yy - yy.mean()) / (yy.std() + 1e-12)
                        fitted = {"EW": ("ew", sgn),
                                  "RIDGE": ("lin", Ridge(alpha=10.0).fit(An, yn)),
                                  "ELASTICNET": ("lin", ElasticNet(alpha=1e-3, l1_ratio=0.5,
                                                                   max_iter=20000, random_state=7).fit(An, yn))}
                        need = False
                if not fitted or mu is None:
                    continue
                Xt = Zc[t]; ok = np.isfinite(Xt).all(1)
                if ok.sum() < 20:
                    continue
                B = (Xt[ok] - mu) / sd; jj = np.flatnonzero(ok)
                for nm, md in fitted.items():
                    scores[nm][t, jj] = (B * md[1]).mean(1) if md[0] == "ew" else md[1].predict(B)
            # ---- turnover methods on each score panel
            for nm, arr in scores.items():
                sc = pd.DataFrame(arr, index=px.index, columns=px.columns)
                if sc.notna().sum().sum() == 0:
                    continue
                tgt = V.rank_weights(sc, beta)
                for meth in METHODS:
                    w = apply_method(tgt, meth, P)
                    pnl, turn = run(w, r)
                    md = V.metrics(pnl.iloc[split:], spy.iloc[split:])
                    rows.append(dict(neutral=neu, period=P, model=nm, method=meth, turnover=turn,
                                     cost_pct=turn * FEE * 1e-4, **md))
                    curves[(neu, P, nm, meth)] = pnl
    M = pd.DataFrame(rows); M.to_csv(OUT / "xs_turnover.csv", index=False)
    pd.to_pickle(curves, OUT / "xs_turnover_curves.pkl")
    pd.set_option("display.width", 290); pd.set_option("display.max_rows", 500)
    sm = V.metrics(spy.iloc[split:])
    print(f"=== BENCHMARK SPY: Sharpe {sm['sharpe']:.3f}, ann {sm['ann']:.1%}, MDD {sm['mdd']:.1%} ===")
    print("\n=== keeping style vs neutralising it, by turnover method (mean over models and periods) ===")
    print(M.groupby(["neutral", "method"])[["sharpe", "ann", "vol", "mdd", "ir", "turnover", "cost_pct"]]
          .mean().round(3).to_string())
    print("\n=== sector-only (style KEPT): every cell ===")
    q = M[M.neutral == "sector only"]
    print(q[["period", "model", "method", "sharpe", "ann", "vol", "mdd", "ir", "beta", "corr_spy",
             "turnover", "cost_pct"]].sort_values("sharpe", ascending=False).round(3).to_string(index=False))
    print("\n=== turnover method ranking (sector-only) ===")
    print(q.groupby("method")[["sharpe", "ann", "mdd", "ir", "turnover", "cost_pct"]].mean()
          .sort_values("sharpe", ascending=False).round(3).to_string())

    best = q.loc[q.sharpe.idxmax()]
    fig, axes = plt.subplots(1, 2, figsize=(17, 6))
    ax = axes[0]
    for meth in METHODS:
        k = (best.neutral, best.period, best.model, meth)
        if k in curves:
            p = curves[k].iloc[split:]
            ax.plot((1 + p).cumprod(), lw=1.2,
                    label=f"{meth}: SR {V.metrics(p)['sharpe']:.2f}, turn {M[(M.method == meth) & (M.neutral == best.neutral) & (M.period == best.period) & (M.model == best.model)].turnover.iloc[0]:.0f}x")
    spy_c = (1 + spy.iloc[split:]).cumprod()
    ax.plot(spy_c / spy_c.iloc[0], lw=2, color="k", ls="--",
            label=f"SPY (SR {sm['sharpe']:.2f}) [right scale differs: long-only]")
    ax.set_title(f"turnover methods, {best.neutral}, P={best.period}, {best.model}")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    ax = axes[1]
    for neu, mk in [("sector only", "o"), ("+style+pastK", "x")]:
        s = M[M.neutral == neu]
        ax.scatter(s.turnover, s.sharpe, marker=mk, s=55, label=neu, alpha=.8)
        for _, rr in s.iterrows():
            ax.annotate(rr.method, (rr.turnover, rr.sharpe), fontsize=6, xytext=(3, 2),
                        textcoords="offset points")
    ax.axhline(0, color="k", lw=.6)
    ax.set_xlabel("annual turnover (x)"); ax.set_ylabel("test Sharpe")
    ax.set_title("Sharpe vs turnover"); ax.legend(); ax.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(OUT / "fig_turnover.png", dpi=110); plt.close()
    print("\nfigure written")
    return M


if __name__ == "__main__":
    main()
