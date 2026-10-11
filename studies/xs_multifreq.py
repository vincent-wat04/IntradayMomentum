"""Fundamentals + how to combine a low-frequency signal with high-frequency ones.

THE PROBLEM you raised.  A quarterly fundamental is a step function: over 2019-2026 it has roughly 30
independent values per name, while a daily price/volume factor has ~1900.  If both are poured into one
pooled regression, the fit is dominated by whichever block has more variation, and the effective sample
size of the fundamental block is overstated by ~60x -- the model cannot tell that a smooth series is
smooth because it has little information, not because it is reliable.

THREE COMBINATION SCHEMES ARE TESTED (the distinction is from Fitzgibbons, Friedman, Pomorski & Serban,
FAJ 2018, "Constructing Long-Only Multifactor Strategies", who find portfolio blending wins at low to
moderate tracking error and signal blending at high tracking error):

  SIGNAL_BLEND     one model on all features at once, price/volume and fundamental together
  PORTFOLIO_EW     fit each block separately at ITS OWN horizon, build one book per block,
                   combine the books at equal risk
  PORTFOLIO_IR     same, but weight the books by their dev-period IR (a return estimate, so it is the
                   scheme most exposed to estimation noise)
  PORTFOLIO_RP     same, inverse dev volatility (needs only a risk estimate -- the robust choice)
  COND_TILT        fundamental used as a conditioner, not an addend: the price/volume score is scaled
                   by (1 + lambda * fundamental_rank), so fundamentals change the SIZE of a bet rather
                   than voting on its direction

Horizons: price/volume block at P=20, fundamental block at P=60 (its natural refresh is quarterly).
Both books are then held on the same 20-day rebalance so they are comparable and combinable.
Neutralisation: sector only (style exposure kept, per your instruction).  Benchmark SPY.
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
import yfinance as yf
from sklearn.linear_model import Ridge

import cross_section_study as XS
import xs_factor_library as XL
import xs_pipeline_v2 as V
import xs_turnover as TU

OUT = Path("outputs/full/xs2")
SPLIT = "2023-01-03"
FEE = 5.0
P_PX, P_FU, P_HOLD = 20, 60, 20
LAM = 0.5


def sector_z(F, sec):
    Z = {k: V.xs_z(V.winsor(v)) for k, v in F.items()}
    for k in Z:
        o = Z[k].copy()
        for s in sec.unique():
            c = sec[sec == s].index.intersection(o.columns)
            if len(c) >= 2:
                o[c] = o[c].sub(o[c].mean(axis=1), axis=0)
        Z[k] = o
    return Z


def score_block(Z, px, P, model="RIDGE", screen_every=63, fit_every=42):
    keys = sorted(Z)
    cube = np.stack([Z[k].to_numpy(float) for k in keys], -1)
    Y = XS.fwd_ret(px, P).to_numpy(float)
    out = np.full((len(px), len(px.columns)), np.nan)
    idx, sgn, mdl, mu, sd = np.array([], int), np.array([]), None, None, None
    need = False
    nsel = []
    for t in range(len(px)):
        if t % screen_every == 0:
            ni, ns = V.screen(cube, Y, t, P)
            if len(ni) != len(idx) or not np.array_equal(ni, idx):
                idx, sgn, mdl, mu, sd, need = ni, ns, None, None, None, True
        nsel.append(len(idx))
        if len(idx) == 0:
            continue
        Zc = cube[:, :, idx]
        if (t % fit_every == 0 or need) and t - P - 1 > 252:
            X = Zc[: t - P - 1].reshape(-1, len(idx)); y = Y[: t - P - 1].reshape(-1)
            m = np.isfinite(y) & np.isfinite(X).all(1)
            if m.sum() > 8000:
                A, yy = X[m], y[m]
                mu, sd = A.mean(0), A.std(0) + 1e-9
                if model == "EW":
                    mdl = ("ew", sgn)
                else:
                    mdl = ("lin", Ridge(alpha=10.0).fit((A - mu) / sd, (yy - yy.mean()) / (yy.std() + 1e-12)))
                need = False
        if mdl is None or mu is None:
            continue
        Xt = Zc[t]; ok = np.isfinite(Xt).all(1)
        if ok.sum() < 20:
            continue
        B = (Xt[ok] - mu) / sd; jj = np.flatnonzero(ok)
        out[t, jj] = (B * mdl[1]).mean(1) if mdl[0] == "ew" else mdl[1].predict(B)
    return pd.DataFrame(out, index=px.index, columns=px.columns), float(np.mean(nsel[-500:]))


def main():
    px, op, hi, lo, vo = XS.load()
    sec = pd.Series({c: XS.SECTOR[c] for c in px.columns})
    F0, r = XL.build(px, op, hi, lo, vo)
    beta = F0["BETA"]
    FU = pd.read_pickle("data/xs/fundamentals.pkl")
    split = int(np.searchsorted(px.index, SPLIT))
    spy = yf.download("SPY", start="2018-11-01", end="2026-10-03", auto_adjust=True,
                      progress=False)["Close"]
    spy.index = spy.index.strftime("%Y-%m-%d")
    spy = spy.reindex(px.index).ffill().pct_change().squeeze()

    import pdf_volume_factors as PV
    stab = {f"{k}_STAB": PV.stability(v, 20) for k, v in F0.items()}
    PXF = {**F0, **stab}
    Zpx = sector_z(PXF, sec)
    Zfu = sector_z(FU, sec)
    print(f"price/volume block: {len(Zpx)} factors | fundamental block: {len(Zfu)} factors", flush=True)

    rows, curves = [], {}

    def record(name, pnl, turn, note=""):
        md = V.metrics(pnl.iloc[split:], spy.iloc[split:])
        rows.append(dict(scheme=name, turnover=turn, cost_pct=turn * FEE * 1e-4, note=note, **md))
        curves[name] = pnl

    def book(score, P=P_HOLD, method="COMBO"):
        tgt = V.rank_weights(score, beta)
        w = TU.apply_method(tgt, method, P)
        return TU.run(w, r)

    # ---- the two standalone blocks
    s_px, n_px = score_block(Zpx, px, P_PX)
    s_fu, n_fu = score_block(Zfu, px, P_FU)
    p_px, t_px = book(s_px); record("price/volume only (P=20)", p_px, t_px, f"{n_px:.0f} feats")
    p_fu, t_fu = book(s_fu); record("fundamentals only (P=60)", p_fu, t_fu, f"{n_fu:.0f} feats")

    # ---- 1. signal blending: everything in one model
    s_all, n_all = score_block({**Zpx, **Zfu}, px, P_PX)
    p, t = book(s_all); record("SIGNAL_BLEND (one model, all features)", p, t, f"{n_all:.0f} feats")

    # ---- 2-4. portfolio blending at three weightings
    dev_px, dev_fu = p_px.iloc[:split], p_fu.iloc[:split]
    v_px, v_fu = dev_px.std(), dev_fu.std()
    ir_px = dev_px.mean() / (v_px + 1e-12)
    ir_fu = dev_fu.mean() / (v_fu + 1e-12)
    for nm, (a, b) in {
        "PORTFOLIO_EW (equal risk)": (0.5 / v_px, 0.5 / v_fu),
        "PORTFOLIO_RP (inverse dev vol)": (1 / v_px, 1 / v_fu),
        "PORTFOLIO_IR (dev IR weighted)": (max(ir_px, 0) / v_px, max(ir_fu, 0) / v_fu),
    }.items():
        tot = a + b
        if tot <= 0:
            continue
        a, b = a / tot, b / tot
        wpx = V.rank_weights(s_px, beta) * a
        wfu = V.rank_weights(s_fu, beta) * b
        comb = (wpx + wfu)
        comb = comb.div(comb.abs().sum(axis=1).replace(0, np.nan), axis=0)
        w = TU.apply_method(comb, "COMBO", P_HOLD)
        pnl, turn = TU.run(w, r)
        record(nm, pnl, turn, f"w_px={a:.2f} w_fu={b:.2f}")

    # ---- 5. conditional tilt: fundamentals scale the size of the price/volume bet
    fr = s_fu.rank(axis=1, pct=True).sub(0.5) * 2
    s_cond = s_px.rank(axis=1, pct=True).sub(0.5) * (1 + LAM * fr.fillna(0.0))
    p, t = book(s_cond); record(f"COND_TILT (lambda={LAM})", p, t)

    M = pd.DataFrame(rows); M.to_csv(OUT / "xs_multifreq.csv", index=False)
    pd.to_pickle(curves, OUT / "xs_multifreq_curves.pkl")
    pd.set_option("display.width", 290)
    sm = V.metrics(spy.iloc[split:])
    print(f"\n=== BENCHMARK SPY: Sharpe {sm['sharpe']:.3f}, ann {sm['ann']:.1%}, MDD {sm['mdd']:.1%} ===")
    print("\n=== combining a low-frequency block with high-frequency blocks (test 2023-2026) ===")
    print(M[["scheme", "sharpe", "ann", "vol", "mdd", "ir", "beta", "corr_spy", "turnover", "note"]]
          .round(3).to_string(index=False))

    # ---- which fundamental factors actually work here
    print("\n=== individual fundamental factors, sector-neutral, P=60 decile LS ===")
    fw = XS.fwd_ret(px, 60)
    fr_rows = []
    for k, v in Zfu.items():
        icd = XS.ic_series(v.iloc[:split], fw.iloc[:split]).mean()
        ict = XS.ic_series(v.iloc[split:], fw.iloc[split:]).mean()
        sg = float(np.sign(icd) or 1.0)
        w = TU.apply_method(V.rank_weights(v * sg, beta), "COMBO", P_HOLD)
        pnl, turn = TU.run(w, r)
        fr_rows.append(dict(factor=k, ic_dev=icd, ic_test=ict, dev_sign=sg,
                            sharpe=V.metrics(pnl.iloc[split:])["sharpe"], turnover=turn))
    FR = pd.DataFrame(fr_rows).sort_values("sharpe", ascending=False)
    FR.to_csv(OUT / "fundamental_factor_detail.csv", index=False)
    print(FR.round(4).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(17, 6))
    ax = axes[0]
    for n, p_ in curves.items():
        pp = p_.iloc[split:]
        ax.plot((1 + pp).cumprod(), lw=2 if "only" in n else 1.2,
                ls="--" if "only" in n else "-",
                label=f"{n}: SR {V.metrics(pp)['sharpe']:.2f}")
    ax.set_title("low-frequency + high-frequency combination schemes (long-short, test)")
    ax.legend(fontsize=7.5); ax.grid(alpha=.3)
    ax = axes[1]
    f2 = FR.set_index("factor").sharpe.sort_values()
    ax.barh(np.arange(len(f2)), f2.values,
            color=["tab:green" if v > 0 else "tab:red" for v in f2.values])
    ax.set_yticks(np.arange(len(f2))); ax.set_yticklabels(f2.index, fontsize=9)
    ax.axvline(0, color="k", lw=.6); ax.set_xlabel("decile long-short test Sharpe")
    ax.set_title("individual PIT fundamental factors"); ax.grid(alpha=.3, axis="x")
    plt.tight_layout(); plt.savefig(OUT / "fig_multifreq.png", dpi=110); plt.close()
    print("\nfigure written")
    return M, FR


if __name__ == "__main__":
    main()
