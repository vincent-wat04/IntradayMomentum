"""HAC (Newey-West) inference. Daily strategy returns and overlapping-horizon targets are serially dependent,
so plain OLS t-statistics are overstated. Use these instead."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


def nw_lag(n: int) -> int:
    """Newey-West rule of thumb floor(4 (n/100)^(2/9))."""
    return int(np.floor(4 * (n / 100.0) ** (2 / 9)))


def hac_mean_test(x, lags: int = 10) -> dict[str, float]:
    """H0: E[x] = 0 with HAC standard error. Returns mean, se, t, two-sided p, one-sided p (E[x] > 0)."""
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) <= max(lags + 2, 20) or np.std(x, ddof=1) <= 1e-14:
        return {"mean": float(np.mean(x)) if len(x) else np.nan, "se": np.nan, "t": np.nan, "p_two": np.nan, "p_one": np.nan}
    fit = sm.OLS(x, np.ones((len(x), 1))).fit(cov_type="HAC", cov_kwds={"maxlags": min(lags, len(x) - 2)}); t = float(fit.tvalues[0])
    return {"mean": float(fit.params[0]), "se": float(fit.bse[0]), "t": t, "p_two": float(2 * stats.norm.sf(abs(t))), "p_one": float(stats.norm.sf(t))}


@dataclass
class HacFit:
    params: pd.Series
    tvalues: pd.Series
    pvalues: pd.Series
    r2: float
    n: int


def hac_ols(y, X, lags: int | None = None, names=None, add_const: bool = True) -> HacFit:
    """OLS with HAC covariance. X may be a Series, DataFrame or 2-D array; rows with NaN are dropped."""
    Xd = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
    if names is not None:
        Xd.columns = names
    Xd.columns = [str(c) for c in Xd.columns]; yv = pd.Series(np.asarray(y, float), index=Xd.index)
    ok = yv.notna() & Xd.notna().all(axis=1); yv, Xd = yv[ok], Xd[ok]
    A = sm.add_constant(Xd, has_constant="add") if add_const else Xd
    fit = sm.OLS(yv.to_numpy(), A.to_numpy()).fit(cov_type="HAC", cov_kwds={"maxlags": lags if lags is not None else nw_lag(len(yv))})
    idx = list(A.columns)
    return HacFit(pd.Series(fit.params, idx), pd.Series(fit.tvalues, idx), pd.Series(fit.pvalues, idx), float(fit.rsquared), int(len(yv)))


def cluster_mean_test(values, groups) -> tuple[float, float]:
    """Mean and t-stat with standard errors clustered by group (e.g. event-level returns clustered by trading day)."""
    v, g = np.asarray(values, float), np.asarray(groups); ok = np.isfinite(v) & pd.notna(g); v, g = v[ok], g[ok]
    if len(v) < 30 or len(np.unique(g)) < 10 or np.std(v, ddof=1) <= 1e-14:
        return np.nan, np.nan
    fit = sm.OLS(v, np.ones((len(v), 1))).fit(cov_type="cluster", cov_kwds={"groups": g}); return float(fit.params[0]), float(fit.tvalues[0])
