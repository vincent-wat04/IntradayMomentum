"""Mechanical look-ahead / accounting tests (re-exports of studies/audit/harness.py).

    poison_future(bars, cut_day)       multiply every price/volume AFTER cut_day by 7.3; outputs on or before cut_day must not change
    poison_intraday(bars, cut_clock)   multiply the bars in (cut_clock, 15:30] of every day; DECISIONS at or before cut_clock must not change
    cmp_frames / cmp_trades            bit-level comparison of two implementations

Use them on any new feature or signal before it enters a study (docs/HARNESS.md section 2). A test that cannot fail on a known
bug is not a test - see docs/AUDIT.md for the bugs each one catches.
"""
from quantkit import paths

paths.use_legacy_studies()
from audit.harness import cmp_frames, cmp_trades, perf, poison_future, poison_intraday, shift_prices   # noqa: E402,F401
