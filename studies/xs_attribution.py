"""Why is the single-factor Sharpe ~1.0 but the model composite ~0?  A controlled ladder.

Your hypothesis was that the single-factor numbers were not neutralised.  That is part of it, but not
all of it.  This script changes ONE thing at a time, from the configuration that produced the headline
single-factor numbers in PART3 to the configuration that produced the near-zero composite in PART4,
and reports the Sharpe after each change.  Everything else is held fixed, same universe, same costs.

Step 0 (PART3 configuration)    sector demean only, decile long-short (top/bottom 10%),
                                dev-period sign, h = 20, no beta neutrality, best factors
Step 7 (PART4 configuration)    + style neutral, + past-K neutral, rank weights over all names,
                                beta neutral, walk-forward screening, P = 5, Ridge

Also reported: the selection effect (best of 77 vs median of 77), which is the part that has nothing
to do with neutralisation at all.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

import cross_section_study as XS
import xs_factor_library as XL
import xs_pipeline_v2 as V

OUT = Path("outputs/full/xs2")
SPLIT = "2023-01-03"
FEE = 5.0


def decile_weights(score, frac=0.1):
    rk = score.rank(axis=1, pct=True)
    w = pd.DataFrame(0.0, index=score.index, columns=score.columns)
    w[rk >= 1 - frac] = 1.0
    w[rk <= frac] = -1.0
    return w.div(w.abs().sum(axis=1).replace(0, np.nan), axis=0)


def hold_tranche(w, h):
    """h overlapping tranches, 1-day lag (what the PART3 single-factor test used)."""
    return (sum(w.shift(1 + k) for k in range(h)) / h).fillna(0.0)


def run(w, r, fee=FEE):
    gross = (w * r).sum(axis=1)
    cost = fee * 1e-4 * w.diff().abs().sum(axis=1)
    pnl = gross - cost
    turn = float(w.diff().abs().sum(axis=1).mean() * 252)
    return pnl, turn


def sharpe(x):
    x = x.dropna()
    return float(x.mean() / x.std() * np.sqrt(252)) if len(x) > 60 and x.std() > 0 else np.nan


def beta_neutral(w, beta):
    b = beta.reindex_like(w)
    num = (w * b).sum(axis=1); den = (b * b).sum(axis=1).replace(0, np.nan)
    w2 = w.sub(b.mul(num / den, axis=0))
    w2 = w2.sub(w2.mean(axis=1), axis=0)
    return w2.div(w2.abs().sum(axis=1).replace(0, np.nan), axis=0)


def main():
    px, op, hi, lo, vo = XS.load()
    sec = pd.Series({c: XS.SECTOR[c] for c in px.columns})
    F0, r = XL.build(px, op, hi, lo, vo)
    beta = F0["BETA"]
    split = int(np.searchsorted(px.index, SPLIT))
    controls = {f"PASTR_{k}": np.log(px).diff(k) for k in V.PASTK}

    # three neutralisation levels, same code path
    Zsec = {k: V.xs_z(V.winsor(v)) for k, v in F0.items()}
    for k in Zsec:
        o = Zsec[k].copy()
        for s in sec.unique():
            c = sec[sec == s].index.intersection(o.columns)
            if len(c) >= 2:
                o[c] = o[c].sub(o[c].mean(axis=1), axis=0)
        Zsec[k] = o
    Zsty = V.neutralise(F0, sec, controls, use_pastk=False)
    Zpk = V.neutralise(F0, sec, controls, use_pastk=True)
    NEU = {"sector only": Zsec, "+style": Zsty, "+style+pastK": Zpk}

    # the factor set: top 5 by |dev IC| under sector-only (this is how PART3 picked its headline names)
    h0 = 20
    fwd20 = XS.fwd_ret(px, h0)
    icdev = {}
    for k, v in Zsec.items():
        ic = XS.ic_series(v.iloc[:split], fwd20.iloc[:split])
        icdev[k] = ic.mean()
    ser = pd.Series(icdev).dropna()
    signs = {k: float(np.sign(ser[k])) for k in ser.index}
    rank_all = ser.abs().sort_values(ascending=False)
    # The strongest factors ARE the style factors, which is itself a central finding: once styles are
    # neutralised those columns no longer exist as features.  The ladder therefore has to be built on
    # the strongest NON-style factors, so that every rung is defined at every neutralisation level.
    print(f"top-8 by |dev IC| overall (sector-only, h=20):")
    print(rank_all.head(8).round(4).to_string())
    print(f"  of which style/control columns: {[k for k in rank_all.head(8).index if k in V.STYLES]}")
    non_style = [k for k in rank_all.index if k not in V.STYLES]
    top5 = non_style[:5]
    print(f"ladder factor set (strongest 5 non-style): {top5}")

    def composite(Z, names, use_sign=True):
        return sum((signs[k] if use_sign else 1.0) * Z[k].fillna(0.0) for k in names) / len(names)

    rows = []

    def add(step, desc, pnl, turn, extra=""):
        rows.append(dict(step=step, config=desc, sharpe_test=sharpe(pnl.iloc[split:]),
                         sharpe_dev=sharpe(pnl.iloc[:split]), turnover=turn, note=extra))

    # ---------------- the ladder
    sc = composite(Zsec, top5)
    w = hold_tranche(decile_weights(sc), h0)
    p, t = run(w, r); add(0, "PART3 config: sector-only, decile LS, dev sign, h=20", p, t)

    sc = composite(Zsty, top5)
    w = hold_tranche(decile_weights(sc), h0)
    p, t = run(w, r); add(1, "+ style neutral (SIZE/BETA/VOL)", p, t)

    sc = composite(Zpk, top5)
    w = hold_tranche(decile_weights(sc), h0)
    p, t = run(w, r); add(2, "+ past-K return neutral (K=5,21)", p, t)

    w = hold_tranche(V.rank_weights(sc), h0)
    p, t = run(w, r); add(3, "decile LS -> rank weights over all names", p, t)

    w = hold_tranche(V.rank_weights(sc, beta), h0)
    p, t = run(w, r); add(4, "+ beta neutral", p, t)

    # walk-forward sign instead of the dev-period sign
    cube = np.stack([Zpk[k].to_numpy(float) for k in top5], -1)
    Y20 = fwd20.to_numpy(float)
    sgn_wf = np.zeros((len(px), len(top5)))
    cur = np.ones(len(top5))
    for tt in range(len(px)):
        if tt % 63 == 0:
            idx, sg = V.screen(cube, Y20, tt, h0)
            if len(idx):
                cur = np.zeros(len(top5)); cur[idx] = sg
        sgn_wf[tt] = cur
    sc_wf = sum(pd.DataFrame(sgn_wf[:, j][:, None] * Zpk[k].fillna(0.0).to_numpy(),
                             index=px.index, columns=px.columns) for j, k in enumerate(top5)) / len(top5)
    w = hold_tranche(V.rank_weights(sc_wf, beta), h0)
    p, t = run(w, r); add(5, "dev sign -> walk-forward screened sign (PIT)", p, t)

    # period hold at P=5 instead of 20 overlapping tranches at h=20
    fwd5 = XS.fwd_ret(px, 5)
    w = V.period_hold(V.rank_weights(sc_wf, beta), 5)
    p, t = run(w, r); add(6, "h=20 tranches -> P=5 period hold", p, t)

    # Ridge on the 5 factors, P=5
    Zc = np.stack([Zpk[k].to_numpy(float) for k in top5], -1)
    Y5 = fwd5.to_numpy(float)
    pred = np.full((len(px), len(px.columns)), np.nan)
    mu = sd = None; mdl = None
    for tt in range(len(px)):
        if tt % 42 == 0 and tt - 6 > 252:
            X = Zc[: tt - 6].reshape(-1, len(top5)); y = Y5[: tt - 6].reshape(-1)
            m = np.isfinite(y) & np.isfinite(X).all(1)
            if m.sum() > 20000:
                A, yy = X[m], y[m]
                mu, sd = A.mean(0), A.std(0) + 1e-9
                mdl = Ridge(alpha=10.0).fit((A - mu) / sd, (yy - yy.mean()) / yy.std())
        if mdl is None:
            continue
        Xt = Zc[tt]; ok = np.isfinite(Xt).all(1)
        if ok.sum() > 20:
            pred[tt, np.flatnonzero(ok)] = mdl.predict((Xt[ok] - mu) / sd)
    scr = pd.DataFrame(pred, index=px.index, columns=px.columns)
    w = V.period_hold(V.rank_weights(scr, beta), 5)
    p, t = run(w, r); add(7, "equal-weight composite -> Ridge (PART4 config)", p, t)

    L = pd.DataFrame(rows)
    L["delta"] = L.sharpe_test.diff()
    L.to_csv(OUT / "attribution_ladder.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 100)
    print("\n=== LADDER: one change at a time (test 2023-2026, 5bp/leg) ===")
    print(L[["step", "config", "sharpe_test", "delta", "turnover"]].round(3).to_string(index=False))

    # ---------------- the selection effect, measured separately
    print("\n=== selection effect: best-of-77 vs typical, sector-only, decile LS, h=20 ===")
    srs = {}
    for k, v in Zsec.items():
        w_ = hold_tranche(decile_weights(v * signs.get(k, 1.0)), h0)
        pp, _ = run(w_, r)
        srs[k] = sharpe(pp.iloc[split:])
    S = pd.Series(srs).dropna().sort_values(ascending=False)
    print(f"  best single factor      {S.iloc[0]:.3f}  ({S.index[0]})")
    print(f"  5th best                {S.iloc[4]:.3f}")
    print(f"  MEDIAN of {len(S)} factors   {S.median():.3f}")
    print(f"  share with Sharpe > 0.5 {(S > 0.5).mean():.1%}")
    print(f"  equal-weight composite of the top-5 (step 0 above)  {L.sharpe_test.iloc[0]:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(17, 6))
    ax = axes[0]
    col = ["tab:blue"] + ["tab:red" if d < 0 else "tab:green" for d in L.delta.iloc[1:]]
    ax.bar(L.step, L.sharpe_test, color=col)
    for s_, v_, d_ in zip(L.step, L.sharpe_test, L.delta):
        ax.text(s_, v_, f"{v_:.2f}" + ("" if not np.isfinite(d_) else f"\n({d_:+.2f})"),
                ha="center", va="bottom" if v_ >= 0 else "top", fontsize=8)
    ax.set_xticks(L.step)
    ax.set_xticklabels([f"{i}" for i in L.step])
    ax.axhline(0, color="k", lw=.6); ax.set_xlabel("ladder step"); ax.set_ylabel("test Sharpe")
    ax.set_title("where the single-factor Sharpe goes"); ax.grid(alpha=.3, axis="y")
    ax = axes[1]
    ax.hist(S.values, bins=25, color="tab:grey", edgecolor="k", lw=.4)
    ax.axvline(S.median(), color="tab:blue", lw=2, label=f"median {S.median():.2f}")
    ax.axvline(S.iloc[0], color="tab:red", lw=2, label=f"best of {len(S)} = {S.iloc[0]:.2f}")
    ax.set_xlabel("single-factor decile-LS test Sharpe (sector-only)")
    ax.set_title("the headline numbers were the right tail"); ax.legend(); ax.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(OUT / "fig_attribution.png", dpi=110); plt.close()
    print("\nfigure written")
    return L, S


if __name__ == "__main__":
    main()
