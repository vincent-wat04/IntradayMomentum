"""Order-book execution cost: walk a market order through the recorded 10-level book.

Static-snapshot estimate = upper bound for a one-shot market order (no queue replenishment, no child orders).
For the project's backtest assumption (half-spread $0.005 + $0.0045 commission) compare `slippage_vs_mid` against 0.005.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def walk(prices: np.ndarray, sizes: np.ndarray, qty: float) -> tuple[np.ndarray, np.ndarray]:
    """prices, sizes: (n, L) best-first levels on the side being hit. Returns (average fill price, filled fraction) per snapshot."""
    cum = np.cumsum(sizes, axis=1); prev = cum - sizes; take = np.clip(qty - prev, 0, sizes); filled = take.sum(1)
    return (take * prices).sum(1) / np.maximum(filled, 1e-9), filled / qty


def slippage_vs_mid(snap: pd.DataFrame, qty: float) -> pd.DataFrame:
    """Per-snapshot cost in $/share relative to the mid for a buy and a sell of `qty`, plus filled fractions.
    snap needs bp1..bp10, bs1..bs10, ap1..ap10, as1..as10 (quantkit.data.mdrec.book_snapshots)."""
    L = range(1, 11); bp, bs = snap[[f"bp{i}" for i in L]].to_numpy(), snap[[f"bs{i}" for i in L]].to_numpy()
    ap, as_ = snap[[f"ap{i}" for i in L]].to_numpy(), snap[[f"as{i}" for i in L]].to_numpy(); mid = (bp[:, 0] + ap[:, 0]) / 2
    pa, fa = walk(ap, as_, qty); pb, fb = walk(bp, bs, qty)
    return pd.DataFrame({"buy": pa - mid, "sell": mid - pb, "fill_buy": fa, "fill_sell": fb, "mid": mid}, index=snap.index)


def slippage_table(snaps: pd.DataFrame, times=("10:00", "10:30", "12:00", "14:00", "15:40"), sizes=(1000, 2000, 5000, 10000, 20000), window_s: int = 60) -> pd.DataFrame:
    """Median-across-days slippage ($/share, average of buy and sell) by clock time and order size.
    Within a day the mean over snapshots in [T, T+window_s) is used, then the median across days."""
    rows = []
    for d, g in snaps.groupby("day"):
        for T in times:
            lo = pd.Timestamp(f"{d} {T}:00"); s = g[(g.ts >= lo) & (g.ts < lo + pd.Timedelta(seconds=window_s))]
            if len(s) < 5:
                continue
            for q in sizes:
                r = slippage_vs_mid(s, q); rows.append(dict(day=d, time=T, qty=q, slip=float(((r.buy + r.sell) / 2).mean()), fill=float(min(r.fill_buy.mean(), r.fill_sell.mean()))))
    R = pd.DataFrame(rows); return R.groupby(["qty", "time"]).slip.median().unstack("time")
