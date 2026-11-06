"""Feature and attribution diagnostics (handbook lectures 14-16, 30, 31)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from quantkit.stats.inference import hac_ols


def psi(reference, current, bins: int = 10) -> float:
    """Population stability index with reference deciles. <0.10 stable, 0.10-0.25 watch, >=0.25 critical."""
    ref, cur = np.asarray(reference, float), np.asarray(current, float); ref, cur = ref[np.isfinite(ref)], cur[np.isfinite(cur)]
    q = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1))); q[0], q[-1] = -np.inf, np.inf
    a = np.clip(np.histogram(ref, q)[0] / len(ref), 1e-4, None); c = np.clip(np.histogram(cur, q)[0] / len(cur), 1e-4, None)
    return float(((c - a) * np.log(c / a)).sum())


def psi_status(v: float) -> str:
    return "stable" if v < 0.10 else ("watch" if v < 0.25 else "critical")


def feature_drift(X: pd.DataFrame, ref: tuple[str, str], cur: tuple[str, str], bins: int = 10) -> pd.DataFrame:
    """PSI for every column of X between two date windows, e.g. ref=('2019','2022'), cur=('2025','2026')."""
    r, c = X.loc[ref[0]:ref[1]], X.loc[cur[0]:cur[1]]
    out = pd.DataFrame({"psi": {k: psi(r[k], c[k], bins) for k in X.columns}}); out["status"] = out.psi.map(psi_status); return out


def quantile_table(feature, target, q: int = 5, lags: int = 5) -> dict:
    """Mean target by feature quantile + MonoScore (Spearman of quantile index vs mean; |.|>=0.8 = monotone)
    + HAC t of the slope of target on the z-scored feature (|t|>=3 is the Harvey-Liu-Zhu bar for a NEW single feature)."""
    f, y = np.asarray(feature, float), np.asarray(target, float); m = np.isfinite(f) & np.isfinite(y); f, y = f[m], y[m]
    bucket = pd.qcut(pd.Series(f), q, labels=False, duplicates="drop").to_numpy(); means = pd.Series(y).groupby(bucket).mean()
    mono = float(spearmanr(means.index, means.to_numpy())[0]); z = (f - f.mean()) / f.std(); fit = hac_ols(y, pd.DataFrame({"z": z}), lags=lags)
    return {"bucket_means": means, "mono_score": mono, "slope": float(fit.params["z"]), "hac_t": float(fit.tvalues["z"]), "n": fit.n}


def information_coefficient(signal: pd.Series, forward_return: pd.Series, method: str = "spearman", lags: int = 5) -> dict:
    """Time-series IC (single asset): correlation between the signal and the next-period return, with a HAC t on the
    slope of rank(forward return) on rank(signal). For cross-sections of assets compute IC per date instead."""
    d = pd.concat([signal, forward_return], axis=1).dropna(); d.columns = ["s", "r"]
    if method == "spearman":
        d = d.rank()
    fit = hac_ols(d["r"], d[["s"]], lags=lags); return {"ic": float(d["s"].corr(d["r"])), "t": float(fit.tvalues["s"]), "n": fit.n}


def attribute(increment: pd.Series, factors: pd.DataFrame, lags: int = 5, periods: int = 252) -> pd.DataFrame:
    """Barra-lite: regress an increment series (e.g. vol-matched v5 minus v3) on factor returns/dummies with HAC errors.
    Returns coefficient table; the intercept is annualised alpha. Typical factors: market return, market return^2
    (convexity), high-vol-regime dummy, crisis-year dummy."""
    fit = hac_ols(increment, factors, lags=lags); tab = pd.DataFrame({"coef": fit.params, "t": fit.tvalues})
    tab.loc["const", "coef_annualised"] = fit.params["const"] * periods; tab.attrs["r2"] = fit.r2; tab.attrs["n"] = fit.n; return tab
