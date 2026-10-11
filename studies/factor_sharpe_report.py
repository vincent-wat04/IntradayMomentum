"""Figures + tables for single-factor Sharpe (reads factor_sharpe.csv and cache/factor_books.pkl).

Selection is done ONLY on dev (2019-2022): a (factor,h,book) is "dev-selected" if sr_dev>0.5 on BOTH SPY and QQQ.
Composites = equal-weight average of the daily returns of dev-selected books (no test info used).
"""
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path("outputs/full/report")
TEST_START = pd.Timestamp("2023-01-03")
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

df = pd.read_csv(OUT / "factor_sharpe.csv")
books = pickle.load(open("outputs/full/cache/factor_books.pkl", "rb"))


def metrics(r: pd.Series, bh: pd.Series | None = None) -> dict:
    r = r.dropna()
    eq = (1 + r).cumprod()
    ann = eq.iloc[-1] ** (252 / len(r)) - 1
    vol = r.std() * np.sqrt(252)
    sr = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan
    mdd = (eq / eq.cummax() - 1).min()
    d = dict(ann_ret=ann, vol=vol, sharpe=sr, mdd=mdd, calmar=ann / abs(mdd) if mdd < 0 else np.nan)
    if bh is not None:
        b = bh.reindex(r.index).fillna(0)
        beq = (1 + b).cumprod()
        d["excess_ann"] = ann - (beq.iloc[-1] ** (252 / len(b)) - 1)
        d["corr_bh"] = r.corr(b)
        d["exposure_beta"] = np.cov(r, b)[0, 1] / b.var()
    return d


# ---------- 1. dev vs test scatter ----------
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
for ax, bk in zip(axes, ["LS", "LC", "TILT"]):
    s = df[df.book == bk]
    for sym, c in [("SPY", "tab:blue"), ("QQQ", "tab:orange")]:
        t = s[s.symbol == sym]
        ax.scatter(t.sr_dev, t.sr_test, s=6, alpha=0.35, c=c, label=f"{sym} (corr={t.sr_dev.corr(t.sr_test):.2f})")
    if bk != "LS":
        for sym, ls in [("SPY", "--"), ("QQQ", ":")]:
            ax.axhline(s[s.symbol == sym].bh_sr_test.iloc[0], color="k", ls=ls, lw=1, label=f"{sym} B&H test SR")
    ax.axhline(0, color="grey", lw=0.6); ax.axvline(0, color="grey", lw=0.6)
    ax.axvline(0.5, color="red", lw=0.8, ls="--")
    ax.set_title(f"{bk}: dev(2019-22) SR vs test(2023-26) SR")
    ax.set_xlabel("dev Sharpe"); ax.set_ylabel("test Sharpe"); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(OUT / "factor_sharpe_dev_vs_test.png", dpi=120); plt.close()

# ---------- 2. dev-selected composites ----------
piv = df.pivot_table(index=["factor", "h", "book"], columns="symbol", values="sr_dev")
sel = piv[(piv.SPY > 0.5) & (piv.QQQ > 0.5)].reset_index()
sel.to_csv(OUT / "factor_sharpe_dev_selected.csv", index=False)

ROBUST = [("VP_REVERSAL:1", 20), ("[M]vxn_minus_vix:5", 20), ("[M]vxn_minus_vix:1", 20),
          ("[M]tom_next_week:20", 20), ("[M]cape_ey:1", 5), ("[M]breakeven_chg_63:60", 20)]
ROBUST = [(f, h) for f, h in ROBUST if ("SPY", f, h, "LS") in books]

rows = []
for sym in ["SPY", "QQQ"]:
    bh = books[(sym, "BH", 0, "BH")]
    bh_t = bh[bh.index >= TEST_START]
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    # (a) robust LS single factors, test period
    ax = axes[0, 0]
    for f, h in ROBUST:
        r = books[(sym, f, h, "LS")]; r = r[r.index >= TEST_START]
        m = metrics(r, bh_t); rows.append(dict(symbol=sym, series=f"{f} h{h} LS", **m))
        ax.plot((1 + r).cumprod(), lw=1.2, label=f"{f} h{h} (SR {m['sharpe']:.2f})")
    ax.set_title(f"{sym} robust LS single factors (test 2023-26, pure timing L/S)"); ax.legend(fontsize=7); ax.grid(alpha=.3)
    # (b)-(d) composites per book
    for ax, bk in zip([axes[0, 1], axes[1, 0], axes[1, 1]], ["LS", "LC", "TILT"]):
        ax.plot((1 + bh).cumprod(), color="k", lw=1.4, label=f"B&H (test SR {metrics(bh_t)['sharpe']:.2f})")
        rows.append(dict(symbol=sym, series="B&H", **metrics(bh_t, bh_t)))
        for h, c in [(5, "tab:green"), (20, "tab:purple")]:
            ss = sel[(sel.book == bk) & (sel.h == h)]
            if ss.empty:
                continue
            comp = pd.concat([books[(sym, f, h, bk)] for f in ss.factor], axis=1).mean(axis=1)
            mt = metrics(comp[comp.index >= TEST_START], bh_t)
            md = metrics(comp[comp.index < TEST_START], bh[bh.index < TEST_START])
            rows.append(dict(symbol=sym, series=f"EW composite {bk} h{h} (n={len(ss)})", dev_sharpe=md["sharpe"], **mt))
            ax.plot((1 + comp).cumprod(), color=c, lw=1.2,
                    label=f"{bk} h{h} EW n={len(ss)} (dev SR {md['sharpe']:.2f} / test SR {mt['sharpe']:.2f})")
        ax.axvline(TEST_START, color="red", ls="--", lw=1)
        ax.set_title(f"{sym} dev-selected equal-weight composite - {bk}"); ax.legend(fontsize=7); ax.grid(alpha=.3)
        if bk != "LS":
            ax.set_yscale("log")
    plt.tight_layout(); plt.savefig(OUT / f"factor_sharpe_tearsheet_{sym}.png", dpi=120); plt.close()

# ---------- 2b. overlay: B&H + w * k * LS composite (k = dev-period vol match, PIT) ----------
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
for ax, sym in zip(axes, ["SPY", "QQQ"]):
    bh = books[(sym, "BH", 0, "BH")]; bh_t = bh[bh.index >= TEST_START]
    ax.plot((1 + bh_t).cumprod(), color="k", lw=1.4, label=f"B&H (SR {metrics(bh_t)['sharpe']:.2f})")
    for h in [5, 20]:
        ss = sel[(sel.book == "LS") & (sel.h == h)]
        comp = pd.concat([books[(sym, f, h, "LS")] for f in ss.factor], axis=1).mean(axis=1)
        k = bh[bh.index < TEST_START].std() / comp[comp.index < TEST_START].std()
        for w in [0.25, 0.5]:
            o = (bh + w * k * comp)[bh.index >= TEST_START]
            mt = metrics(o, bh_t)
            rows.append(dict(symbol=sym, series=f"B&H + {w}x volmatched LS h{h} (k={k:.1f})", **mt))
            ax.plot((1 + o).cumprod(), lw=1.1, label=f"B&H+{w}xLS h{h} (SR {mt['sharpe']:.2f}, MDD {mt['mdd']:.1%})")
    ax.set_title(f"{sym} B&H + timing overlay (test 2023-26)"); ax.legend(fontsize=8); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig(OUT / "factor_sharpe_overlay.png", dpi=120); plt.close()

tab = pd.DataFrame(rows).drop_duplicates(subset=["symbol", "series"])
tab.to_csv(OUT / "factor_sharpe_composites.csv", index=False)
pd.set_option("display.width", 250)
print(tab.round(3).to_string(index=False))

# ---------- 3. distribution summary ----------
summ = []
for (sym, h, bk), g in df.groupby(["symbol", "h", "book"]):
    s2 = sel[(sel.h == h) & (sel.book == bk)].factor
    gs = g[g.factor.isin(s2)]
    summ.append(dict(symbol=sym, h=h, book=bk, n=len(g), med_test=g.sr_test.median(), p90_test=g.sr_test.quantile(.9),
                     pos_share=(g.sr_test > 0).mean(), beat_bh=(g.sr_test > g.bh_sr_test).mean(),
                     dev_test_corr=g.sr_dev.corr(g.sr_test), n_sel=len(gs), sel_med_test=gs.sr_test.median(),
                     sel_pos=(gs.sr_test > 0).mean(), bh_sr_test=g.bh_sr_test.iloc[0]))
summ = pd.DataFrame(summ); summ.to_csv(OUT / "factor_sharpe_summary.csv", index=False)
print(summ.round(3).to_string(index=False))
