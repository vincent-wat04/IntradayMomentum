"""Multiple-testing and overfitting diagnostics (Bailey & Lopez de Prado 2012/2014; Bailey, Borwein, Lopez de Prado & Zhu 2014).

Read these with the caveats in docs/HARNESS.md section 5:
  * PSR/DSR of a strategy that is obviously profitable is ~1 and tells you nothing. Apply them to INCREMENTS
    (candidate minus vol-matched baseline) - see vol_matched_increment().
  * DSR's expected-max-SR depends on the cross-sectional SR variance of the candidates, which is understated when the
    candidates are near-duplicates, and on K, which must be the project-wide trial count (docs/TRIAL_REGISTRY.md).
  * CSCV-PBO depends heavily on what you put in the candidate family. A family that varies only a risk-appetite
    dimension (e.g. the Sleeve:core ratio rho) yields a meaningless PBO.
All Sharpe ratios here are per-period (daily) unless the name says annualised.
"""
from __future__ import annotations

import itertools

import numpy as np
from scipy.stats import kurtosis, norm, skew

EULER = 0.5772156649015329


def sr_period(x) -> float:
    x = np.asarray(x, float); return float(x.mean() / x.std(ddof=1))


def psr(x, sr_star: float = 0.0) -> float:
    """Probabilistic Sharpe ratio: P(true SR > sr_star) with skew/kurtosis correction. sr_star is per-period."""
    x = np.asarray(x, float); T = len(x); s = sr_period(x); g3 = skew(x); g4 = kurtosis(x, fisher=False)
    den = np.sqrt(max(1 - g3 * s + (g4 - 1) / 4 * s ** 2, 1e-12)); return float(norm.cdf((s - sr_star) * np.sqrt(T - 1) / den))


def expected_max_sr(var_sr: float, n_trials: int) -> float:
    """Expected maximum per-period SR among n_trials independent zero-skill trials whose SR variance is var_sr."""
    return float(np.sqrt(var_sr) * ((1 - EULER) * norm.ppf(1 - 1 / n_trials) + EULER * norm.ppf(1 - 1 / (n_trials * np.e))))


def dsr(x, var_sr: float, n_trials: int) -> float:
    """Deflated Sharpe ratio = PSR against the expected maximum SR of n_trials trials."""
    return psr(x, expected_max_sr(var_sr, n_trials))


def vol_matched_increment(candidate, baseline):
    """candidate - baseline * (std_c / std_b): the part of the candidate not explained by simply running the baseline at the same risk."""
    c, b = np.asarray(candidate, float), np.asarray(baseline, float); return c - b * (c.std(ddof=1) / b.std(ddof=1))


def cscv_pbo(M: np.ndarray, S: int = 16) -> tuple[float, np.ndarray]:
    """Combinatorially symmetric CV probability of backtest overfitting. M: (T, N) returns of N candidates over the same T days.
    Splits time into S blocks, evaluates all C(S, S/2) train/test splits, picks the in-sample best and records its OOS rank.
    Returns (PBO, logit array). PBO = share of splits where the in-sample winner ranks at or below the OOS median."""
    T, N = M.shape; blocks = np.array_split(np.arange(T), S); lam = []
    for tr in itertools.combinations(range(S), S // 2):
        te = [b for b in range(S) if b not in tr]
        it = np.concatenate([blocks[b] for b in tr]); io = np.concatenate([blocks[b] for b in te])
        sr_in = M[it].mean(0) / M[it].std(0, ddof=1); sr_out = M[io].mean(0) / M[io].std(0, ddof=1)
        n_star = int(np.argmax(sr_in)); rk = (sr_out < sr_out[n_star]).sum() + 0.5 * ((sr_out == sr_out[n_star]).sum() - 1)
        w = (rk + 1) / (N + 1); lam.append(np.log(w / (1 - w)))
    lam = np.array(lam); return float((lam <= 0).mean()), lam
