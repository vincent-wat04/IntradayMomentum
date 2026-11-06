"""Performance statistics. Inputs are simple daily returns; annualisation = 252.

distribution_panel() is the reporting standard for convex / right-skewed strategies (the Sleeve): never judge a variant
by Sharpe alone. It reports mean/median, Sharpe, Sortino, CAGR, MDD, Calmar, skew, excess kurtosis, CVaR5%, profit
factor, share of profit from the top-5% days, and the worst day.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

PERIODS = 252


def _s(x) -> pd.Series:
    return pd.Series(x).dropna().astype(float)


def sharpe(x, periods: int = PERIODS) -> float:
    """Annualised mean/std (ddof=1), zero risk-free rate."""
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 3 or np.std(x, ddof=1) <= 0:
        return float("nan")
    return float(x.mean() / x.std(ddof=1) * np.sqrt(periods))


def equity(x) -> pd.Series:
    return (1 + _s(x)).cumprod()


def drawdown(x) -> pd.Series:
    eq = equity(x); return eq / eq.cummax() - 1


def max_drawdown(x) -> float:
    return float(drawdown(x).min())


def cagr(x, periods: int = PERIODS) -> float:
    x = _s(x); return float(equity(x).iloc[-1] ** (periods / len(x)) - 1)


def distribution_panel(x, periods: int = PERIODS) -> dict[str, float]:
    x = _s(x); eq = (1 + x).cumprod(); mdd = float((eq / eq.cummax() - 1).min()); g = cagr(x, periods)
    dn = np.sqrt((np.minimum(x, 0) ** 2).mean()); prof = x[x > 0].sum(); loss = -x[x < 0].sum()
    return {"mean_bp": 1e4 * x.mean(), "median_bp": 1e4 * x.median(), "sharpe": sharpe(x, periods), "sortino": x.mean() / dn * np.sqrt(periods),
            "cagr": g, "mdd": mdd, "calmar": g / abs(mdd) if mdd < 0 else np.nan, "skew": float(skew(x)), "excess_kurtosis": float(kurtosis(x)),
            "cvar5_bp": 1e4 * x[x <= x.quantile(0.05)].mean(), "profit_factor": prof / loss if loss > 0 else np.nan,
            "top5pct_days_share_of_profit": x.nlargest(max(int(0.05 * len(x)), 1)).sum() / prof if prof > 0 else np.nan, "worst_day_bp": 1e4 * x.min()}


def compare(series: dict[str, pd.Series], start=None, end=None) -> pd.DataFrame:
    """Panel for several return series side by side (rows = series)."""
    return pd.DataFrame({k: distribution_panel(v.loc[start:end] if (start or end) else v) for k, v in series.items()}).T


def by_year(x: pd.Series, fn=sharpe) -> pd.Series:
    x = _s(x); return x.groupby(x.index.year).apply(lambda v: fn(v.to_numpy()))


def split_periods(x: pd.Series, cuts: dict[str, tuple[str, str]], fn=sharpe) -> pd.Series:
    """e.g. cuts={'dev': ('2019','2022'), 'test': ('2023','2026')}."""
    return pd.Series({k: fn(x.loc[a:b].to_numpy()) for k, (a, b) in cuts.items()})
