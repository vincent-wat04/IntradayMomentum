"""Minute bars as day x minute matrices (adapter to the research engine's loader).

    b = load_bars("QQQ")
    b.o, b.h, b.l, b.c, b.v     DataFrames, index = day, columns = bar END time "HH:MM" (09:31 ... 16:00)
    b.day_open, b.day_close, b.prev_close_adj (dividend-adjusted previous close), b.last_idx (half-day marker)

A decision on the CLOSE of bar k is filled at the OPEN of bar k+1. See studies/etft/intraday/bars.py for details.
"""
from __future__ import annotations

from quantkit import paths


def load_bars(sym: str):
    paths.use_legacy_studies()
    from etft.intraday.bars import load_bars as _lb
    return _lb(sym)


def full_session_mask(b):
    """True on days with a complete 390-bar session (half days and holes excluded)."""
    return b.last_idx.to_numpy() >= 389
