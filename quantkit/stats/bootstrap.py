"""Resampling inference for Sharpe differences between two return series observed on the same days.

Always compare candidate and baseline on the SAME resampled indices (paired); resampling them independently throws away
their strong correlation and makes the interval far too wide.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quantkit.stats.inference import hac_mean_test
from quantkit.stats.perf import sharpe


def fixed_block_indices(n: int, block: int, nboot: int, seed: int = 0) -> np.ndarray:
    """(nboot, n) indices from non-overlapping-start fixed-length blocks. Identical to the scripts' block_idx()."""
    rg = np.random.default_rng(seed); out = np.empty((nboot, n), dtype=int); nblk = int(np.ceil(n / block))
    for i in range(nboot):
        st = rg.integers(0, n - block + 1, nblk); out[i] = (st[:, None] + np.arange(block)[None, :]).ravel()[:n]
    return out


def stationary_indices(n: int, expected_block: float, rng: np.random.Generator) -> np.ndarray:
    """Circular stationary bootstrap (Politis-Romano): geometric block lengths with mean expected_block."""
    if n < 2 or expected_block <= 1:
        raise ValueError("need n >= 2 and expected_block > 1")
    restart = 1.0 / expected_block; out = np.empty(n, dtype=int); out[0] = int(rng.integers(n))
    for i in range(1, n):
        out[i] = int(rng.integers(n)) if rng.random() < restart else (out[i - 1] + 1) % n
    return out


def paired_delta_sharpe(candidate, baseline, nboot: int = 2000, block: int = 10, seed: int = 0, kind: str = "fixed", level: float = 0.95) -> dict[str, float]:
    """Delta annualised Sharpe (candidate - baseline) with a paired block-bootstrap percentile interval.
    kind='fixed' uses fixed blocks, kind='stationary' the stationary bootstrap (block = expected length)."""
    c, b = np.asarray(candidate, float), np.asarray(baseline, float); ok = np.isfinite(c) & np.isfinite(b); c, b = c[ok], b[ok]
    if len(c) < 60:
        return {"delta": np.nan, "lo": np.nan, "hi": np.nan, "n": len(c)}
    if kind == "fixed":
        idx = fixed_block_indices(len(c), block, nboot, seed)
    else:
        rng = np.random.default_rng(seed); idx = np.array([stationary_indices(len(c), block, rng) for _ in range(nboot)])
    d = np.array([sharpe(c[i]) - sharpe(b[i]) for i in idx]); a = (1 - level) / 2
    return {"delta": sharpe(c) - sharpe(b), "lo": float(np.quantile(d, a)), "hi": float(np.quantile(d, 1 - a)), "n": len(c)}


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values; NaN stays NaN."""
    valid = sorted(((k, p) for k, p in pvalues.items() if np.isfinite(p)), key=lambda z: z[1]); out = {k: np.nan for k in pvalues}; run = 0.0; m = len(valid)
    for r, (k, p) in enumerate(valid):
        run = max(run, min(1.0, (m - r) * float(p))); out[k] = run
    return out


def max_t_pvalues(differences: pd.DataFrame, nboot: int = 2000, block: float = 10, seed: int = 0, hac_lags: int = 10) -> dict[str, float]:
    """Single-step family-wise max-t p-values (White/Romano-Wolf style) for H0: mean(candidate - baseline) <= 0, one-sided.
    Columns = candidate-minus-baseline daily returns for ALL candidates you looked at (the family). Synchronous stationary blocks."""
    d = differences.dropna()
    if len(d) < 60 or d.shape[1] == 0:
        return {c: np.nan for c in differences.columns}
    se, obs = {}, {}
    for c in d:
        h = hac_mean_test(d[c].to_numpy(), hac_lags); se[c], obs[c] = h["se"], h["t"]
    good = [c for c in d if np.isfinite(se[c]) and se[c] > 0 and np.isfinite(obs[c])]
    z = d[good].to_numpy() - d[good].to_numpy().mean(axis=0); scale = np.array([se[c] for c in good]); rng = np.random.default_rng(seed)
    mx = np.array([np.max(z[stationary_indices(len(z), block, rng)].mean(axis=0) / scale) for _ in range(nboot)])
    out = {c: np.nan for c in differences.columns}
    for c in good:
        out[c] = float((1 + np.sum(mx >= obs[c])) / (nboot + 1))
    return out
