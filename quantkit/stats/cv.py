"""Time-aware cross-validation helpers (handbook lecture 18: purge / embargo; lecture 22: time isolation).

The research protocol: for each out-of-sample year Y, train on data strictly before Y, choose the regularisation by
leave-one-year-out CV INSIDE the training window, refit, predict year Y. Neighbouring days are not independent
(volatility clusters), so the held-out year's boundary neighbours are embargoed from the inner training fold.
"""
from __future__ import annotations

from typing import Callable, Hashable, Sequence

import numpy as np
import pandas as pd


def embargo_mask(train: np.ndarray, held: np.ndarray, embargo: int) -> np.ndarray:
    """Remove from `train` every position within `embargo` rows before/after the held-out block."""
    if not embargo or not held.any():
        return train
    pos = np.arange(len(train)); lo, hi = pos[held].min() - embargo, pos[held].max() + embargo
    return train & ~((pos >= lo) & (pos <= hi))


def walk_forward_tuned(dates: pd.DatetimeIndex, sample: np.ndarray, grid: Sequence[Hashable], fit: Callable, predict: Callable, score: Callable,
                       first_test_year: int, last_test_year: int, embargo: int = 0, min_train: int = 150, min_held: int = 50):
    """Generic expanding-window walk-forward with inner leave-one-year-out tuning.

    fit(train_mask, param) -> model            (train_mask is a boolean array over `dates`)
    predict(model, test_mask) -> predictions   (anything; returned per year)
    score(model, held_mask) -> float           (higher is better; evaluated on the held-out inner year)
    Returns (dict year -> predictions, dict year -> chosen param). `sample` marks tradable/eligible days.
    """
    yrs = dates.year; preds, chosen = {}, {}
    for yr in range(first_test_year, last_test_year + 1):
        tr = sample & (yrs < yr); te = sample & (yrs == yr)
        if not te.any():
            continue
        cv = {}
        for p in grid:
            sc = []
            for hy in sorted(set(yrs[tr])):
                held = tr & (yrs == hy); inner = embargo_mask(tr & (yrs != hy), held, embargo)
                if inner.sum() < min_train or held.sum() < min_held:
                    continue
                sc.append(score(fit(inner, p), held))
            cv[p] = np.mean(sc) if sc else -np.inf
        best = max(cv, key=cv.get); chosen[yr] = best; preds[yr] = predict(fit(tr, best), te)
    return preds, chosen
