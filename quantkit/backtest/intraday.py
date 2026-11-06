"""Adapters to the minute-bar noise-area engine (studies/audit/reimpl_noise.py is the audited reference implementation).

Strategy (Zarattini-Aziz-Barbon noise area, as implemented here): at each check time compare price with the band around the
open (upper = max(open, prev_close_adj) * (1 + sigma_profile), lower = min(...) * (1 - sigma_profile)); breakout -> enter in
that direction, size = min(max_lev, target_vol / sigma14); exits by band/tail stops and a 15:40 flat. Decision on the close of
bar k, fill at the open of bar k+1. 30 'phases' shift all check times by 0..29 minutes and are averaged to smooth the
sampling grid; the live runner trades ONE phase.
"""
from __future__ import annotations

import pandas as pd

from quantkit import paths

# Settings used for the Sleeve throughout the research (sleeve_anatomy.BASE + execution costs)
SLEEVE_DEFAULTS = dict(stop_long="band+tail", stop_short="band+tail", tail_k=0.5, tail_every_bar=True, skip_dows=(0,),
                       sizing="voltarget", max_lev=4.0, reentry="hold", half_spread=0.005, commission=0.0045)


def check_times(offset_min: int = 0) -> list[str]:
    """Half-hourly decision times 10:00 ... 15:30, shifted by offset_min minutes."""
    base = [(h, m) for h in range(10, 16) for m in (0, 30)][:-1]
    return [(pd.Timestamp(f"2000-01-01 {h:02d}:{m:02d}") + pd.Timedelta(minutes=offset_min)).strftime("%H:%M") for h, m in base]


def simulate(bars, **overrides):
    """Run one phase. Returns (daily return Series, fills DataFrame). Keyword arguments override SLEEVE_DEFAULTS;
    pass checks=check_times(off) to select the phase. See audit/reimpl_noise.simulate for all options."""
    paths.use_legacy_studies()
    from audit.reimpl_noise import simulate as _sim
    kw = {**SLEEVE_DEFAULTS, **overrides}; kw.setdefault("checks", check_times(0)); return _sim(bars, **kw)


def run_phases(bars, offsets=range(30), verbose: bool = False, **overrides) -> dict[int, tuple[pd.Series, pd.DataFrame]]:
    """{offset: (daily returns, fills)} for each phase offset. About 3 seconds per phase on 1,900 days."""
    out = {}
    for off in offsets:
        rr, F = simulate(bars, checks=check_times(off), **overrides)
        F = F.copy()
        if len(F):
            F["day"] = pd.to_datetime(F.day)
        out[off] = (rr, F)
        if verbose:
            print(f"phase {off} done", flush=True)
    return out


def phase_average(phases: dict) -> pd.Series:
    """Equal-weight average of the phases' daily returns (the backtest Sleeve)."""
    return pd.concat([rr for rr, _ in phases.values()], axis=1).mean(axis=1)
