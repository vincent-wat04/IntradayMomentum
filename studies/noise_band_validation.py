"""Statistical validation of the paper and current adaptive Noise Band designs.

This is a research orchestrator, not production trading code. It answers two separate
questions:

1. Calibration: does a band describe a stable, point-in-time intraday "noise range"?
2. Economics: after realistic execution and the current v5 portfolio construction,
   does changing only the entry band improve net performance?

The historical 2019-2026 sample has already been inspected extensively. Results from
that sample are diagnostic. Only observations on/after --forward-start are eligible for
confirmatory interpretation.

Run from the repository root:
    python studies/noise_band_validation.py --section all
    python studies/noise_band_validation.py --section calibration
    python studies/noise_band_validation.py --section standalone --symbols SPY --skip-md5
    python studies/noise_band_validation.py --smoke

Raw data are not committed. Set ETFT_DATA as documented in data/README.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
os.environ.setdefault("ETFT_DATA", str(ROOT / "data"))
sys.path.insert(0, str(HERE))

from etft.intraday.bars import Bars, load_bars
from etft.intraday.noise import sigma_profile

TABLES = ROOT / "outputs" / "tables"
TABLES.mkdir(parents=True, exist_ok=True)

CANONICAL_CHECKS = [f"{h:02d}:{m:02d}" for h in range(10, 16) for m in (0, 30)][:-1]
HISTORICAL_END = pd.Timestamp("2026-10-30")
LEGACY_DEV = (pd.Timestamp("2019-01-01"), pd.Timestamp("2022-12-31"))
LEGACY_TEST = (pd.Timestamp("2023-01-01"), HISTORICAL_END)
DEFAULT_FORWARD_START = pd.Timestamp("2026-11-02")
CURRENT_VARIANT = "current_mean14_s3"
Q60 = 0.60
SEED = 20261031


@dataclass(frozen=True)
class Variant:
    name: str
    estimator: str
    lookback: int
    q: float | None
    adaptive_s3: bool
    nominal_coverage: float | None


VARIANTS = (
    Variant("paper_mean14", "rolling_mean_abs_move", 14, None, False, None),
    Variant(CURRENT_VARIANT, "rolling_mean_abs_move", 14, None, True, None),
    Variant("q60_l60", "rolling_quantile_abs_move", 60, Q60, False, Q60),
    # Multiplying a q60 profile by s3 widens it conditionally, so 60% is no longer
    # its nominal unconditional coverage target.
    Variant("q60_l60_s3", "rolling_quantile_abs_move", 60, Q60, True, None),
)


def _sr(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 3 or np.std(x, ddof=1) <= 0:
        return np.nan
    return float(np.mean(x) / np.std(x, ddof=1) * np.sqrt(252.0))


def _hac_mean_test(x: np.ndarray, lags: int = 10) -> dict[str, float]:
    """HAC test of a time-series mean. Returns two- and one-sided p-values."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) <= max(lags + 2, 20) or np.std(x, ddof=1) <= 1e-14:
        return {"mean": float(np.mean(x)) if len(x) else np.nan, "se_hac": np.nan,
                "t_hac": np.nan, "p_two_sided": np.nan, "p_one_sided_gt0": np.nan}
    fit = sm.OLS(x, np.ones((len(x), 1))).fit(cov_type="HAC", cov_kwds={"maxlags": min(lags, len(x) - 2)})
    t = float(fit.tvalues[0])
    return {"mean": float(fit.params[0]), "se_hac": float(fit.bse[0]), "t_hac": t,
            "p_two_sided": float(2 * stats.norm.sf(abs(t))),
            "p_one_sided_gt0": float(stats.norm.sf(t))}


def _cluster_mean_test(values: np.ndarray, groups: np.ndarray) -> tuple[float, float]:
    ok = np.isfinite(values) & pd.notna(groups)
    y, g = np.asarray(values, float)[ok], np.asarray(groups)[ok]
    if len(y) < 30 or len(np.unique(g)) < 10 or np.std(y, ddof=1) <= 1e-14:
        return np.nan, np.nan
    fit = sm.OLS(y, np.ones((len(y), 1))).fit(cov_type="cluster", cov_kwds={"groups": g})
    return float(fit.params[0]), float(fit.tvalues[0])


def _holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm family-wise adjusted p-values; NaNs remain NaN."""
    valid = sorted(((k, p) for k, p in pvalues.items() if np.isfinite(p)), key=lambda z: z[1])
    out = {k: np.nan for k in pvalues}
    running = 0.0
    m = len(valid)
    for rank, (key, p) in enumerate(valid):
        running = max(running, min(1.0, (m - rank) * float(p)))
        out[key] = running
    return out


def _stationary_indices(n: int, expected_block: float, rng: np.random.Generator) -> np.ndarray:
    """Circular stationary bootstrap indices (geometric block lengths)."""
    if n < 2 or expected_block <= 1:
        raise ValueError("stationary bootstrap needs n >= 2 and expected_block > 1")
    restart_p = 1.0 / expected_block
    out = np.empty(n, dtype=int)
    out[0] = int(rng.integers(n))
    for i in range(1, n):
        out[i] = int(rng.integers(n)) if rng.random() < restart_p else (out[i - 1] + 1) % n
    return out


def _paired_bootstrap(candidate: np.ndarray, baseline: np.ndarray, *, nb: int, block: int,
                      seed: int) -> dict[str, float]:
    candidate, baseline = np.asarray(candidate, float), np.asarray(baseline, float)
    ok = np.isfinite(candidate) & np.isfinite(baseline)
    candidate, baseline = candidate[ok], baseline[ok]
    if len(candidate) < 60:
        return {"delta_sr": np.nan, "delta_sr_lo": np.nan, "delta_sr_hi": np.nan,
                "delta_mean_ann": np.nan, "delta_mean_ann_lo": np.nan, "delta_mean_ann_hi": np.nan}
    rng = np.random.default_rng(seed)
    dsr, dmean = np.empty(nb), np.empty(nb)
    for i in range(nb):
        ix = _stationary_indices(len(candidate), block, rng)
        dsr[i] = _sr(candidate[ix]) - _sr(baseline[ix])
        dmean[i] = np.mean(candidate[ix] - baseline[ix]) * 252
    return {"delta_sr": _sr(candidate) - _sr(baseline),
            "delta_sr_lo": float(np.quantile(dsr, 0.025)),
            "delta_sr_hi": float(np.quantile(dsr, 0.975)),
            "delta_mean_ann": float(np.mean(candidate - baseline) * 252),
            "delta_mean_ann_lo": float(np.quantile(dmean, 0.025)),
            "delta_mean_ann_hi": float(np.quantile(dmean, 0.975))}


def _max_t_pvalues(differences: pd.DataFrame, *, nb: int, block: int, seed: int,
                   hac_lags: int = 10) -> dict[str, float]:
    """Single-step family-wise max-t p-values using synchronous stationary blocks.

    Columns are candidate-minus-current daily net returns. The null bootstrap is
    centered column-by-column. Synchronous indices preserve cross-candidate dependence.
    """
    d = differences.dropna()
    if len(d) < 60 or d.shape[1] == 0:
        return {c: np.nan for c in differences.columns}
    se, obs = {}, {}
    for c in d:
        h = _hac_mean_test(d[c].to_numpy(), hac_lags)
        se[c] = h["se_hac"]
        obs[c] = h["t_hac"]
    good = [c for c in d if np.isfinite(se[c]) and se[c] > 0 and np.isfinite(obs[c])]
    if not good:
        return {c: np.nan for c in differences.columns}
    z = d[good].to_numpy() - d[good].to_numpy().mean(axis=0)
    scale = np.array([se[c] for c in good])
    rng = np.random.default_rng(seed)
    maxima = np.empty(nb)
    for i in range(nb):
        ix = _stationary_indices(len(z), block, rng)
        maxima[i] = np.max(z[ix].mean(axis=0) / scale)
    out = {c: np.nan for c in differences.columns}
    for c in good:
        out[c] = float((1 + np.sum(maxima >= obs[c])) / (nb + 1))
    return out


def _rolling_quantile_profile(b: Bars, lookback: int = 60, q: float = Q60) -> pd.DataFrame:
    move = (b.c.div(b.day_open, axis=0) - 1).abs()
    min_obs = max(20, int(0.7 * lookback))
    return move.rolling(lookback, min_periods=min_obs).quantile(q).shift(1)


def _s3_multiplier(b: Bars) -> pd.DataFrame:
    """Exact current C3/E1 multiplier, delegated to the versioned research implementation."""
    import r2_rules_study as r2
    _, s3, _ = r2.band_multipliers(b)
    return s3


def build_profiles(b: Bars) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    mean14 = sigma_profile(b, 14)
    q60 = _rolling_quantile_profile(b, 60, Q60)
    s3 = _s3_multiplier(b)
    profiles = {
        "paper_mean14": mean14,
        CURRENT_VARIANT: mean14 * s3,
        "q60_l60": q60,
        "q60_l60_s3": q60 * s3,
    }
    for name, frame in profiles.items():
        if not frame.index.equals(b.days) or list(frame.columns) != list(b.c.columns):
            raise AssertionError(f"profile alignment failed: {name}")
        if (frame.dropna(how="all") < 0).any().any():
            raise AssertionError(f"negative width: {name}")
    return profiles, s3


def _bounds(b: Bars, profile: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    hi = np.maximum(b.day_open, b.prev_close_adj.fillna(b.day_open))
    lo = np.minimum(b.day_open, b.prev_close_adj.fillna(b.day_open))
    return profile.add(1).mul(hi, axis=0), (1 - profile).mul(lo, axis=0)


def _periods(days: pd.DatetimeIndex, forward_start: pd.Timestamp) -> dict[str, np.ndarray]:
    return {
        "legacy_dev": (days >= LEGACY_DEV[0]) & (days <= LEGACY_DEV[1]),
        "legacy_test": (days >= LEGACY_TEST[0]) & (days <= LEGACY_TEST[1]),
        "historical_frozen": (days >= LEGACY_DEV[0]) & (days <= HISTORICAL_END),
        "forward_confirmatory": days >= forward_start,
    }


def coverage_tables(symbol: str, b: Bars, profiles: dict[str, pd.DataFrame],
                    forward_start: pd.Timestamp, hac_lags: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    checks = [c for c in CANONICAL_CHECKS if c in b.c.columns]
    if not checks:
        raise ValueError(f"{symbol}: canonical checks absent")
    move = (b.c.div(b.day_open, axis=0) - 1).abs()
    periods = _periods(b.days, forward_start)
    summary, by_year, by_clock = [], [], []

    for spec in VARIANTS:
        profile = profiles[spec.name]
        ub, lb = _bounds(b, profile)
        valid = move.notna() & profile.notna() & ub.notna() & lb.notna()
        raw_inside = (move <= profile).where(valid)
        anchored_inside = ((b.c <= ub) & (b.c >= lb)).where(valid)
        actual_width = (ub - lb).div((ub + lb) / 2).abs().where(valid)
        for period, day_mask in periods.items():
            ix = b.days[day_mask]
            raw_block = raw_inside.loc[ix, checks]
            anchored_block = anchored_inside.loc[ix, checks]
            raw_day = raw_block.mean(axis=1).dropna()
            anchored_day = anchored_block.mean(axis=1).dropna()
            width_day = actual_width.loc[ix, checks].mean(axis=1).dropna()
            target_test = _hac_mean_test(raw_day.to_numpy() - spec.nominal_coverage, hac_lags) \
                if spec.nominal_coverage is not None and len(raw_day) else {}
            summary.append({"symbol": symbol, "variant": spec.name, "period": period,
                            "n_days": len(raw_day), "n_day_clock": int(raw_block.notna().sum().sum()),
                            "coverage_weighting": "daily_equal_weighted",
                            "width_timing": "contemporaneous" if spec.adaptive_s3 else "predetermined_before_open",
                            "raw_coverage": float(raw_day.mean()) if len(raw_day) else np.nan,
                            "raw_coverage_pooled": float(raw_block.stack().mean()) if raw_block.notna().any().any() else np.nan,
                            "anchored_coverage": float(anchored_day.mean()) if len(anchored_day) else np.nan,
                            "anchored_coverage_pooled": float(anchored_block.stack().mean()) if anchored_block.notna().any().any() else np.nan,
                            "mean_full_width_bp": float(width_day.mean() * 1e4) if len(width_day) else np.nan,
                            "nominal_coverage": spec.nominal_coverage,
                            "coverage_minus_nominal": target_test.get("mean", np.nan),
                            "coverage_hac_t": target_test.get("t_hac", np.nan),
                            "coverage_hac_p_two_sided": target_test.get("p_two_sided", np.nan)})
        years = sorted(set(b.days[b.days >= LEGACY_DEV[0]].year))
        for year in years:
            ix = b.days.year == year
            raw_day = raw_inside.loc[ix, checks].mean(axis=1).dropna()
            anchored_day = anchored_inside.loc[ix, checks].mean(axis=1).dropna()
            by_year.append({"symbol": symbol, "variant": spec.name, "year": year,
                            "n_days": len(raw_day), "raw_coverage": float(raw_day.mean()),
                            "anchored_coverage": float(anchored_day.mean())})
        for clock in checks:
            x = raw_inside.loc[b.days >= LEGACY_DEV[0], clock].dropna()
            a = anchored_inside.loc[b.days >= LEGACY_DEV[0], clock].dropna()
            by_clock.append({"symbol": symbol, "variant": spec.name, "clock": clock,
                             "n_days": len(x), "raw_coverage": float(x.mean()),
                             "anchored_coverage": float(a.mean())})

    return pd.DataFrame(summary), pd.DataFrame(by_year), pd.DataFrame(by_clock)


def _continuation_panel(b: Bars, profile: pd.DataFrame, s3: pd.DataFrame,
                        horizons: tuple[int, ...] = (5, 30)) -> pd.DataFrame:
    times = list(b.c.columns)
    checks = [c for c in CANONICAL_CHECKS if c in times]
    ub, lb = _bounds(b, profile)
    rows = []
    for day_i, day in enumerate(b.days):
        last = int(b.last_idx.loc[day])
        for clock in checks:
            k = times.index(clock)
            if k >= last or not np.isfinite(b.o.iloc[day_i, k + 1]):
                continue
            px = float(b.c.iloc[day_i, k])
            upper, lower = float(ub.iloc[day_i, k]), float(lb.iloc[day_i, k])
            width = float(profile.iloc[day_i, k])
            if not (np.isfinite(px) and np.isfinite(upper) and np.isfinite(lower) and np.isfinite(width) and width > 0):
                continue
            side = 1 if px > upper else (-1 if px < lower else 0)
            trend_side = int(np.sign(np.log(px / float(b.day_open.iloc[day_i]))))
            fill = float(b.o.iloc[day_i, k + 1])
            base = {"day": day, "clock": clock, "outside": int(side != 0), "outside_side": side,
                    "trend_side": trend_side, "distance": abs(px / float(b.day_open.iloc[day_i]) - 1) / width,
                    "gap_abs_bp": abs(float(b.day_open.iloc[day_i]) / float(b.prev_close_adj.iloc[day_i]) - 1) * 1e4
                    if np.isfinite(b.prev_close_adj.iloc[day_i]) else np.nan,
                    "log_s3": float(np.log(s3.iloc[day_i, k])) if s3.iloc[day_i, k] > 0 else np.nan}
            for h in horizons:
                if k + h <= last:
                    exit_px = float(b.c.iloc[day_i, k + h])
                    base[f"event_y_{h}"] = side * (exit_px / fill - 1) * 1e4 if side else np.nan
                    base[f"trend_y_{h}"] = trend_side * (exit_px / fill - 1) * 1e4 if trend_side else np.nan
                else:
                    base[f"event_y_{h}"] = np.nan
                    base[f"trend_y_{h}"] = np.nan
            exit_px = float(b.day_close.iloc[day_i])
            base["event_y_moc"] = side * (exit_px / fill - 1) * 1e4 if side else np.nan
            base["trend_y_moc"] = trend_side * (exit_px / fill - 1) * 1e4 if trend_side else np.nan
            rows.append(base)
    return pd.DataFrame(rows)


def continuation_tables(symbol: str, b: Bars, profiles: dict[str, pd.DataFrame], s3: pd.DataFrame,
                        forward_start: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    event_rows, regression_rows = [], []
    periods = _periods(b.days, forward_start)
    for spec in VARIANTS:
        panel = _continuation_panel(b, profiles[spec.name], s3)
        for period, day_mask in periods.items():
            allowed = set(b.days[day_mask])
            p = panel[panel.day.isin(allowed)].copy()
            for horizon in ("5", "30", "moc"):
                events = p[p.outside == 1].dropna(subset=[f"event_y_{horizon}"])
                mean, t = _cluster_mean_test(events[f"event_y_{horizon}"].to_numpy(), events.day.to_numpy())
                event_rows.append({"symbol": symbol, "variant": spec.name, "period": period,
                                   "horizon": horizon, "n_events": len(events),
                                   "n_event_days": events.day.nunique(), "mean_continuation_bp": mean,
                                   "cluster_t": t, "win_rate": float((events[f"event_y_{horizon}"] > 0).mean())
                                   if len(events) else np.nan})
                reg = p.dropna(subset=[f"trend_y_{horizon}", "distance", "log_s3", "gap_abs_bp"]).copy()
                if len(reg) < 100 or reg.day.nunique() < 30:
                    regression_rows.append({"symbol": symbol, "variant": spec.name, "period": period,
                                            "horizon": horizon, "n": len(reg), "outside_beta_bp": np.nan,
                                            "outside_cluster_t": np.nan, "outside_p_two_sided": np.nan})
                    continue
                dummies = pd.get_dummies(reg.clock, prefix="clock", drop_first=True, dtype=float)
                x = pd.DataFrame({"outside": reg.outside.astype(float),
                                  "log_distance": np.log1p(reg.distance),
                                  "log_s3": reg.log_s3,
                                  "gap_abs_bp": reg.gap_abs_bp / 100.0}, index=reg.index).join(dummies)
                fit = sm.OLS(reg[f"trend_y_{horizon}"].astype(float), sm.add_constant(x)).fit(
                    cov_type="cluster", cov_kwds={"groups": reg.day.factorize()[0]})
                beta, t = float(fit.params["outside"]), float(fit.tvalues["outside"])
                regression_rows.append({"symbol": symbol, "variant": spec.name, "period": period,
                                        "horizon": horizon, "n": len(reg), "outside_beta_bp": beta,
                                        "outside_cluster_t": t,
                                        "outside_p_two_sided": float(2 * stats.norm.sf(abs(t)))})
    return pd.DataFrame(event_rows), pd.DataFrame(regression_rows)


def _selected_trades(ph_base: dict, ph_fomc: dict) -> pd.DataFrame:
    import liquidity_inverse_study as li
    from event_calendar import FOMC_DECISION
    fm = set(pd.DatetimeIndex(FOMC_DECISION))
    parts = []
    for source, ph in (("base", ph_base), ("fomc", ph_fomc)):
        for phase, (_, fills) in ph.items():
            if fills is None or not len(fills):
                continue
            t = li.trades_from_fills(fills, phase)
            t["source"] = source
            parts.append(t)
    if not parts:
        return pd.DataFrame()
    t = pd.concat(parts, ignore_index=True)
    t["day"] = pd.to_datetime(t.day)
    return t[((t.source == "fomc") & t.day.isin(fm)) | ((t.source == "base") & ~t.day.isin(fm))]


def _execution_stats(ph_base: dict, ph_fomc: dict, b: Bars, keep: pd.Series,
                     k: pd.DataFrame, sleeve_mult: np.ndarray,
                     period_mask: np.ndarray) -> dict[str, float]:
    """Approximate final Sleeve execution after the configured 15:40 early exit.

    Turnover is two-sided TQQQ notional. The 30 phase sleeves are divided by 30;
    QQQ-equivalent exposure is divided by three for TQQQ. This diagnostic does
    not get added to the single-sided core turnover field.
    """
    days = b.days
    period_days = days[np.asarray(period_mask, bool)]
    if not len(period_days):
        return {"sleeve_turnover_ann_tqqq_two_sided": np.nan, "entries_per_year": np.nan,
                "active_market_day_rate": np.nan, "mean_holding_minutes": np.nan}
    trades = _selected_trades(ph_base, ph_fomc)
    if trades.empty:
        return {"sleeve_turnover_ann_tqqq_two_sided": 0.0, "entries_per_year": 0.0,
                "active_market_day_rate": 0.0, "mean_holding_minutes": np.nan}
    trades = trades[trades.day.isin(period_days)].copy()
    trades = trades[trades.day.map(keep).fillna(0).to_numpy() > 0]
    if trades.empty:
        return {"sleeve_turnover_ann_tqqq_two_sided": 0.0, "entries_per_year": 0.0,
                "active_market_day_rate": 0.0, "mean_holding_minutes": np.nan}
    # v4/v5 exits positions still open at the 15:40 decision at the 15:41 open.
    late = (trades.t_in < "15:40") & (trades.t_out > "15:40")
    exit_px = trades.exit_px.astype(float).copy()
    if late.any():
        px1541 = b.o["15:41"]
        exit_px.loc[late] = trades.loc[late, "day"].map(px1541)
    kp = np.where(trades.side.to_numpy() > 0,
                  trades.day.map(k.kl).fillna(1).to_numpy(), trades.day.map(k.ks).fillna(1).to_numpy())
    sm = trades.day.map(pd.Series(sleeve_mult, index=days)).fillna(1).to_numpy()
    turnover = (np.abs(trades.shares.to_numpy() * trades.entry_px.to_numpy()) +
                np.abs(trades.shares.to_numpy() * exit_px.to_numpy())) * kp * sm / (30.0 * 3.0)
    daily = pd.Series(turnover, index=trades.day).groupby(level=0).sum().reindex(period_days).fillna(0)
    years = max(len(period_days) / 252.0, 1 / 252.0)
    corrected_out = trades.t_out.astype(str).copy()
    corrected_out.loc[late] = "15:40"
    tin = pd.to_datetime(trades.t_in, format="%H:%M")
    tout = pd.to_datetime(corrected_out, format="%H:%M")
    return {"sleeve_turnover_ann_tqqq_two_sided": float(daily.sum() / years),
            "entries_per_year": float(len(trades) / 30.0 / years),
            "active_market_day_rate": float((daily > 0).mean()),
            "mean_holding_minutes": float((tout - tin).dt.total_seconds().mean() / 60.0)}


def _early_exit_cost_adjustment(ph_base: dict, ph_fomc: dict, b: Bars,
                                tqqq_scale: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Extra exit cost missing from the historical 15:40 P&L subtraction.

    Existing v4/v5 code replaces post-15:40 gross P&L but leaves the old exit
    commission in place. A 15:41 marketable exit owes half-spread plus commission.
    Return positive daily costs, already converted to the TQQQ implementation.
    """
    trades = _selected_trades(ph_base, ph_fomc)
    zero = pd.Series(0.0, index=b.days)
    if trades.empty:
        return zero.copy(), zero.copy()
    late = (trades.t_in < "15:40") & (trades.t_out > "15:40")
    trades = trades[late].copy()
    if trades.empty:
        return zero.copy(), zero.copy()
    new_px = trades.day.map(b.o["15:41"]).to_numpy(float)
    new_cost = np.abs(trades.shares.to_numpy(float)) * trades.entry_px.to_numpy(float) / new_px * (0.005 + 0.0045)
    extra = (new_cost - trades.cm_out.to_numpy(float)) * trades.day.map(tqqq_scale).fillna(1.0).to_numpy()
    trades["extra"] = extra / 30.0
    out = []
    for side in (1, -1):
        out.append(trades[trades.side == side].groupby("day").extra.sum().reindex(b.days).fillna(0.0))
    return out[0], out[1]


def _download_tqqq_traded_price(days: pd.DatetimeIndex) -> pd.Series:
    """Fetch actual split-restored TQQQ close without forward-filling past source coverage."""
    import yfinance as yf
    end = (days.max() + pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    px = yf.download("TQQQ", start="2018-10-01", end=end, auto_adjust=False,
                     progress=False, threads=False)["Close"]
    px = px.iloc[:, 0] if hasattr(px, "columns") else px
    px.index = pd.to_datetime(px.index).tz_localize(None)
    splits = yf.Ticker("TQQQ").splits
    splits.index = pd.to_datetime(splits.index).tz_localize(None)
    factor = pd.Series(1.0, index=px.index)
    for day, ratio in splits.items():
        factor[px.index < day] *= ratio
    actual = px * factor
    aligned = actual.reindex(days).ffill()
    if actual.dropna().index.max() < days.max() or aligned.isna().any():
        raise RuntimeError("TQQQ actual-price source does not cover the QQQ minute sample")
    return aligned


def _run_entry_phases(b: Bars, multiplier: pd.DataFrame, *, fomc_only: bool = False) -> dict:
    """Fresh 30-phase run with the correct actual-share exit-cost ledger."""
    import r3_study as r3
    from event_calendar import FOMC_DECISION

    out = {}
    if fomc_only:
        fomc = set(pd.DatetimeIndex(FOMC_DECISION).intersection(b.days))
        non_fomc = [day for day in b.days if day not in fomc]
    for phase in range(30):
        kwargs = dict(checks=r3.SA.chk(phase), sizing="voltarget", max_lev=4.0,
                      reentry="hold", half_spread=r3.LI.HS0, commission=r3.LI.CM0,
                      cost_shares="actual", entry_mult=multiplier, **r3.SA.BASE)
        if fomc_only:
            kwargs["checks"] = [clock for clock in r3.SA.chk(phase) if clock >= "14:30"]
            kwargs["half_spread"] = 0.005
            kwargs["commission"] = 0.0045
            kwargs["skip_days"] = non_fomc
        returns, fills = r3.SIM_E(b, **kwargs)
        fills = fills.copy()
        if len(fills):
            fills["day"] = pd.to_datetime(fills.day)
        out[phase] = (returns, fills)
    return out


def _standalone_execution_stats(phases: dict, b: Bars, period_mask: np.ndarray) -> dict[str, float]:
    """Two-sided underlying-notional turnover for an equal-weight 30-phase Sleeve."""
    import liquidity_inverse_study as li

    period_days = b.days[np.asarray(period_mask, bool)]
    parts = []
    for phase, (_, fills) in phases.items():
        if fills is None or not len(fills):
            continue
        trades = li.trades_from_fills(fills, phase)
        if len(trades):
            parts.append(trades)
    if not len(period_days):
        return {"turnover_daily_two_sided": np.nan, "entries_per_year": np.nan,
                "active_market_day_rate": np.nan, "mean_holding_minutes": np.nan}
    if not parts:
        return {"turnover_daily_two_sided": 0.0, "entries_per_year": 0.0,
                "active_market_day_rate": 0.0, "mean_holding_minutes": np.nan}
    trades = pd.concat(parts, ignore_index=True)
    trades["day"] = pd.to_datetime(trades.day)
    trades = trades[trades.day.isin(period_days)]
    if trades.empty:
        return {"turnover_daily_two_sided": 0.0, "entries_per_year": 0.0,
                "active_market_day_rate": 0.0, "mean_holding_minutes": np.nan}
    turnover = (np.abs(trades.shares * trades.entry_px) +
                np.abs(trades.shares * trades.exit_px)) / 30.0
    daily = pd.Series(turnover.to_numpy(), index=trades.day).groupby(level=0).sum().reindex(period_days).fillna(0)
    years = max(len(period_days) / 252.0, 1 / 252.0)
    tin = pd.to_datetime(trades.t_in, format="%H:%M")
    tout = pd.to_datetime(trades.t_out, format="%H:%M")
    return {"turnover_daily_two_sided": float(daily.mean()),
            "entries_per_year": float(len(trades) / 30.0 / years),
            "active_market_day_rate": float((daily > 0).mean()),
            "mean_holding_minutes": float((tout - tin).dt.total_seconds().mean() / 60.0)}


def standalone_strategy_tables(b: Bars, profiles: dict[str, pd.DataFrame], *, nb: int,
                               seed: int, hac_lags: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Band-only external validation using the audited 30-phase Sleeve state machine.

    This deliberately excludes the QQQ core, FFA, PPP and v5 allocation layers. It is
    suitable when an independent source supplies only compatible one-minute OHLCV.
    """
    import sleeve_anatomy as sa

    mean14 = sigma_profile(b, 14)
    curves, phase_runs = {}, {}
    for spec in VARIANTS:
        multiplier = (profiles[spec.name] / mean14).where(mean14 > 0)
        phases = _run_entry_phases(b, multiplier, fomc_only=False)
        phase_runs[spec.name] = phases
        curves[spec.name] = pd.concat(
            {phase: ret.reindex(b.days).fillna(0.0) for phase, (ret, _) in phases.items()}, axis=1
        ).mean(axis=1)

    # All variants must have a valid width at every possible shifted check time.
    # Otherwise q60's longer warm-up would be silently counted as zero return.
    shifted_checks = sorted({clock for phase in range(30)
                             for clock in sa.chk(phase)})
    common_ready = np.logical_and.reduce([
        profiles[spec.name][shifted_checks].notna().all(axis=1).to_numpy()
        for spec in VARIANTS
    ])
    base_periods = {
        "dev": (b.days >= LEGACY_DEV[0]) & (b.days <= LEGACY_DEV[1]),
        "test": (b.days >= LEGACY_TEST[0]) & (b.days <= LEGACY_TEST[1]),
        "all": (b.days >= LEGACY_DEV[0]) & (b.days <= HISTORICAL_END),
    }
    periods = {}
    for label, mask in base_periods.items():
        mask = np.asarray(mask) & common_ready
        if mask.any():
            selected = b.days[mask]
            periods[f"{label}_{selected.min().year}_{selected.max().year}"] = mask
    rows = []
    for spec in VARIANTS:
        for period, mask in periods.items():
            values = curves[spec.name][mask]
            if not len(values):
                continue
            eq = (1 + values).cumprod()
            drawdown = eq / eq.cummax() - 1
            execution = _standalone_execution_stats(phase_runs[spec.name], b, mask)
            rows.append({"symbol": b.sym, "variant": spec.name, "period": period,
                         "n_days": len(values), "mean_bp": float(values.mean() * 1e4),
                         "sharpe": _sr(values.to_numpy()),
                         "cagr": float(eq.iloc[-1] ** (252 / len(values)) - 1),
                         "vol": float(values.std(ddof=1) * np.sqrt(252)),
                         "mdd": float(drawdown.min()), "worst_day": float(values.min()),
                         **execution})

    baseline = curves[CURRENT_VARIANT]
    alternatives = [v.name for v in VARIANTS if v.name != CURRENT_VARIANT]
    comparisons = []
    for period, mask in periods.items():
        if not mask.any():
            continue
        differences = pd.DataFrame({name: (curves[name] - baseline)[mask]
                                    for name in alternatives})
        holm = _holm({name: _hac_mean_test(differences[name].to_numpy(), hac_lags)["p_one_sided_gt0"]
                      for name in alternatives})
        for block in (5, 10, 20):
            max_t = _max_t_pvalues(differences, nb=nb, block=block,
                                   seed=seed + block, hac_lags=hac_lags)
            for name in alternatives:
                h = _hac_mean_test(differences[name].to_numpy(), hac_lags)
                boot = _paired_bootstrap(curves[name][mask].to_numpy(), baseline[mask].to_numpy(),
                                         nb=nb, block=block,
                                         seed=seed + block + sum(map(ord, name)))
                comparisons.append({"symbol": b.sym, "variant": name,
                                    "baseline": CURRENT_VARIANT, "period": period,
                                    "block_length": block, "n_days": int(mask.sum()),
                                    "mean_diff_hac_t": h["t_hac"],
                                    "mean_diff_p_one_sided": h["p_one_sided_gt0"],
                                    "holm_hac_p": holm[name], "max_t_family_p": max_t[name],
                                    **boot})
    return pd.DataFrame(rows), pd.DataFrame(comparisons)


def strategy_tables(b: Bars, profiles: dict[str, pd.DataFrame], *, forward_start: pd.Timestamp,
                    nb: int, seed: int, hac_lags: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rebuild each band through current v5, changing only the entry-width profile."""
    import r3_study as r3
    import round3_common as common
    import sizing_models_study as models
    import sizing_sigma_feed as sf
    import sota_stack_study as stack
    import tearsheet as ts
    import tearsheet_v4 as t4
    import tearsheet_v5 as t5
    import unified_sizing_posthoc as ppp
    import unified_sizing_study as unified
    from audit.reimpl_noise import daily_sigma
    from etft.intraday.strategy import buy_hold
    from event_calendar import FOMC_DECISION

    days, years = b.days, b.days.year
    if days.max().year > int(FOMC_DECISION.max().year):
        raise RuntimeError("FOMC calendar does not cover the minute sample; update and freeze event_calendar.py first")
    # Existing research modules stop their annual walk-forward loop at 2026.
    # Extend it to the data year explicitly instead of silently filling future k with 1.
    unified.OOS_YEARS = range(2021, int(days.max().year) + 1)
    tqqq_price = _download_tqqq_traded_price(days)
    r3.LI.tqqq_traded_px = lambda requested_days: tqqq_price.reindex(requested_days)
    # Build only non-trade context here. Calling sf.build_inputs would consume
    # unversioned phase caches before this experiment's fresh candidate runs.
    ffa_p = sf.ffa_p_series()
    ffa_skip = (stack.roll_rank(ffa_p).reindex(days) <= 0.2).fillna(False)
    keep = (~ffa_skip).astype(float)
    x, _, _, _ = unified.build_features(b, {"FFA-P": ffa_p})
    bh = buy_hold(b).reindex(days).fillna(0.0)
    sd14 = daily_sigma(b.day_close, b.prev_close_adj, 14).reindex(days)
    inputs = {"X": x, "keep": keep, "ov": stack.cpi_overlay(b), "bh": bh,
              "e_vt": np.minimum(4.0, 0.02 / sd14)}
    x = x.copy()
    x["dur_rate"] = models.family_features(b, x)["dur_rate"]
    tue_fri = pd.Series(days.dayofweek != 0, index=days)
    # Do not use end-of-day target availability or final bar count to decide whether
    # the current day enters sizing. Feature availability and the known weekday suffice.
    select = (tue_fri & x.notna().all(axis=1) & (days >= "2019-01-01")).to_numpy()
    oos = pd.Series(select & (years >= 2021), index=days)

    px = common.load_close("yahoo/QQQ.parquet")
    ret = px.pct_change().fillna(0.0)
    weights = stack.bh_weights(px, ret, {"FFA-SD": t4.ffa_sd()}, pd.Series(1.0, index=px.index))
    target_core = weights[("RV21[0.5,1.5]", "SMA200|e=0.5")].reindex(inputs["bh"].index).ffill().fillna(1.0)
    target_core = target_core.reindex(days).fillna(1.0).to_numpy()
    core_leg = pd.Series(target_core * inputs["bh"].to_numpy() -
                         2e-4 * np.abs(np.diff(np.r_[target_core[0], target_core])), index=days)
    i2 = dict(inputs)
    i2["leg1"] = core_leg

    mean14 = sigma_profile(b, 14)
    tqqq_scale = (b.day_close / tqqq_price) / 3.0
    phase_runs, assembled, pre_port = {}, {}, {}
    for spec in VARIANTS:
        multiplier = (profiles[spec.name] / mean14).where(mean14 > 0)
        # Re-run every candidate, including the paper baseline, in this process.
        # This avoids mixing unversioned phase caches with fresh candidate runs.
        phb = _run_entry_phases(b, multiplier, fomc_only=False)
        phf = _run_entry_phases(b, multiplier, fomc_only=True)
        phase_runs[spec.name] = (phb, phf)
        legs = r3.legs_k(phb, phf, b, inputs["keep"])
        extra_l, extra_s = _early_exit_cost_adjustment(phb, phf, b, tqqq_scale)
        legs["L2"] = legs["L2"] - extra_l * inputs["keep"]
        legs["S2"] = legs["S2"] - extra_s * inputs["keep"]
        k = r3.sizing(x, ppp.CLEAN + ["dur_rate"], legs, i2, select, days, "mv", "lin")
        result = r3.assemble(k, legs, i2, oos)
        assembled[spec.name] = (result, k)
        pre_port[spec.name] = core_leg.to_numpy() + result["sleeve"].to_numpy()

    ref_pre = pre_port[CURRENT_VARIANT]
    final, diagnostics = {}, {}
    cap_k_upper = np.full(len(days), 2.0)  # known pre-open bound of the linear sizing link
    for spec in VARIANTS:
        result, _ = assembled[spec.name]
        v5 = t5.build_v5(target_core, result["sleeve"].to_numpy(), inputs["e_vt"].fillna(0).to_numpy(),
                         cap_k_upper, inputs["bh"].to_numpy(), ref_pre,
                         rho=t5.RHO, cap=t5.CAP, band=t5.BAND)
        final[spec.name] = pd.Series(v5["port"], index=days)
        diagnostics[spec.name] = v5

    period_masks = {
        "legacy_dev_2019_2022": (days >= LEGACY_DEV[0]) & (days <= LEGACY_DEV[1]),
        "legacy_test_2023_2026": (days >= LEGACY_TEST[0]) & (days <= LEGACY_TEST[1]),
        # Evaluate the deployed portfolio on every market day. Mondays and other
        # non-Sleeve days still carry the common core leg and must not disappear
        # from portfolio volatility or Sharpe calculations.
        "historical_oos_2021_2026": ((days >= pd.Timestamp("2021-01-01")) &
                                         (days <= HISTORICAL_END)),
        "forward_confirmatory": days >= forward_start,
    }
    rows = []
    for spec in VARIANTS:
        result, k = assembled[spec.name]
        ph_base, ph_fomc = phase_runs[spec.name]
        for period, mask in period_masks.items():
            exe = _execution_stats(ph_base, ph_fomc, b, inputs["keep"], k,
                                   diagnostics[spec.name]["sleeve_mult"], mask)
            series = final[spec.name][mask]
            metrics = ts.metrics(series) if len(series) >= 60 else {}
            rows.append({"variant": spec.name, "period": period, "n_days": len(series),
                         "sharpe": metrics.get("Sharpe（绝对）", np.nan),
                         "cagr": metrics.get("年化收益 CAGR", np.nan),
                         "vol": metrics.get("年化波动", np.nan),
                         "mdd": metrics.get("最大回撤 MDD", np.nan),
                         "cvar5": metrics.get("CVaR95（日）", np.nan),
                         "worst_day": metrics.get("最差单日", np.nan),
                         "mean_sleeve_bp": float(result["sleeve"][mask].mean() * 1e4) if mask.sum() else np.nan,
                         "core_turnover_ann": float(np.abs(np.diff(diagnostics[spec.name]["core_w"][mask])).sum() /
                                                    max(mask.sum() / 252, 1 / 252)) if mask.sum() else np.nan,
                         **exe})

    comparisons = []
    baseline = final[CURRENT_VARIANT]
    blocks = (5, 10, 20)
    family_periods = ("historical_oos_2021_2026", "forward_confirmatory")
    family_max_t: dict[tuple[str, int], dict[str, float]] = {}
    family_holm: dict[str, dict[str, float]] = {}
    for family_period in family_periods:
        mask = period_masks[family_period]
        diffs = {spec.name: (final[spec.name] - baseline)[mask]
                 for spec in VARIANTS if spec.name != CURRENT_VARIANT}
        family_max_t.update({(family_period, block): _max_t_pvalues(
            pd.DataFrame(diffs), nb=nb, block=block, seed=seed + block, hac_lags=hac_lags)
            for block in blocks})
        raw = {name: _hac_mean_test(series.to_numpy(), hac_lags)["p_one_sided_gt0"]
               for name, series in diffs.items()}
        family_holm[family_period] = _holm(raw)

    for spec in VARIANTS:
        if spec.name == CURRENT_VARIANT:
            continue
        for period, mask in period_masks.items():
            cand, base = final[spec.name][mask].to_numpy(), baseline[mask].to_numpy()
            h = _hac_mean_test(cand - base, hac_lags)
            ready = period != "forward_confirmatory" or int(mask.sum()) >= 252
            for block in blocks:
                boot = _paired_bootstrap(cand, base, nb=nb, block=block,
                                         seed=seed + block + sum(map(ord, spec.name)))
                record = {"variant": spec.name, "baseline": CURRENT_VARIANT, "period": period,
                          "block_length": block, "n_days": int(mask.sum()),
                          "confirmatory_ready": ready, **boot,
                          "mean_diff_hac_t": h["t_hac"],
                          "mean_diff_p_one_sided": h["p_one_sided_gt0"],
                          "max_t_family_p": family_max_t[(period, block)].get(spec.name, np.nan)
                          if period in family_periods else np.nan,
                          "holm_hac_p": family_holm[period].get(spec.name, np.nan)
                          if period in family_periods else np.nan}
                if period == "forward_confirmatory" and not ready:
                    record["max_t_family_p"] = np.nan
                    record["holm_hac_p"] = np.nan
                comparisons.append(record)

    daily = pd.DataFrame({name: series for name, series in final.items()})
    daily.index.name = "day"
    daily.to_csv(TABLES / "noise_band_strategy_daily.csv.gz", compression="gzip")
    return pd.DataFrame(rows), pd.DataFrame(comparisons)


def _manifest_check(symbols: list[str], skip: bool, check_all: bool = False) -> dict[str, str]:
    if skip:
        return {"status": "skipped"}
    expected = {}
    manifest = ROOT / "data" / "MANIFEST_sandbox_2026-10.md5"
    if manifest.exists():
        for line in manifest.read_text().splitlines():
            if line.strip():
                digest, rel = line.split(maxsplit=1)
                expected[rel] = digest
    from etft import config
    requested = set(expected) if check_all else {f"raw/min1/{symbol}.csv" for symbol in symbols}
    observed = {}
    for rel in sorted(requested):
        path = config.DATA / rel
        if not path.exists():
            raise FileNotFoundError(f"missing {path}; set ETFT_DATA as documented in data/README.md")
        hasher = hashlib.md5()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                hasher.update(chunk)
        digest = hasher.hexdigest()
        observed[rel] = digest
        if rel in expected and expected[rel] != digest:
            raise RuntimeError(f"data gate failed for {rel}: got {digest}, expected {expected[rel]}")
    return observed


def _smoke() -> None:
    rng = np.random.default_rng(7)
    n, m = 140, 390
    days = pd.bdate_range("2020-01-01", periods=n)
    times = [x.strftime("%H:%M") for x in pd.date_range("2000-01-01 09:31", periods=m, freq="1min")]
    op = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.003, n))), index=days)
    increments = rng.normal(0, 0.0006, (n, m))
    close = pd.DataFrame(op.to_numpy()[:, None] * np.exp(np.cumsum(increments, axis=1)), index=days, columns=times)
    opened = close.shift(axis=1)
    opened.iloc[:, 0] = op
    volume = pd.DataFrame(rng.lognormal(10, 0.5, (n, m)), index=days, columns=times)
    bars = Bars("SYN", "US", opened, close, close, close, volume, op, close.iloc[:, -1],
                close.iloc[:, -1].shift(1), pd.Series(m - 1, index=days))
    profiles, s3 = build_profiles(bars)
    assert profiles["paper_mean14"].iloc[:5].isna().all().all()
    assert profiles["q60_l60"].iloc[:20].isna().all().all()
    cov, yearly, clock = coverage_tables("SYN", bars, profiles, DEFAULT_FORWARD_START, 10)
    events, regressions = continuation_tables("SYN", bars, profiles, s3, DEFAULT_FORWARD_START)
    assert len(cov) == len(VARIANTS) * 4 and not yearly.empty and not clock.empty
    assert not events.empty and not regressions.empty
    ix = _stationary_indices(100, 10, rng)
    assert len(ix) == 100 and ix.min() >= 0 and ix.max() < 100
    h = _hac_mean_test(rng.normal(0.001, 0.01, 300))
    assert np.isfinite(h["t_hac"])
    p = _max_t_pvalues(pd.DataFrame({"a": rng.normal(0.001, 0.01, 300),
                                     "b": rng.normal(0.0, 0.01, 300)}), nb=100, block=10, seed=3)
    assert all(np.isfinite(list(p.values())))
    print("smoke: profiles, coverage, continuation, HAC, stationary bootstrap and max-t passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--section", choices=("calibration", "standalone", "strategy", "all"), default="all")
    parser.add_argument("--symbols", nargs="+", default=["QQQ", "SPY"])
    parser.add_argument("--forward-start", default=str(DEFAULT_FORWARD_START.date()),
                        help="first post-freeze date eligible for confirmatory interpretation")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--hac-lags", type=int, default=10)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--skip-md5", action="store_true")
    parser.add_argument("--data-label", default="canonical-manifest",
                        help="provenance label recorded in experiment metadata")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        _smoke()
        return
    if args.bootstrap < 200:
        raise ValueError("use at least 200 bootstrap draws")
    forward_start = pd.Timestamp(args.forward_start)
    if forward_start <= HISTORICAL_END:
        raise ValueError(f"forward-start must be after frozen historical end {HISTORICAL_END.date()}")
    observed = _manifest_check(args.symbols, args.skip_md5,
                               check_all=args.section in ("strategy", "all"))
    started = time.time()
    coverage, by_year, by_clock, events, regressions = [], [], [], [], []
    built = {}
    if args.section in ("calibration", "all"):
        for symbol in args.symbols:
            bars = load_bars(symbol)
            profiles, s3 = build_profiles(bars)
            built[symbol] = (bars, profiles, s3)
            a, y, c = coverage_tables(symbol, bars, profiles, forward_start, args.hac_lags)
            e, r = continuation_tables(symbol, bars, profiles, s3, forward_start)
            coverage.append(a); by_year.append(y); by_clock.append(c); events.append(e); regressions.append(r)
        pd.concat(coverage, ignore_index=True).to_csv(TABLES / "noise_band_coverage.csv", index=False)
        pd.concat(by_year, ignore_index=True).to_csv(TABLES / "noise_band_coverage_by_year.csv", index=False)
        pd.concat(by_clock, ignore_index=True).to_csv(TABLES / "noise_band_coverage_by_clock.csv", index=False)
        pd.concat(events, ignore_index=True).to_csv(TABLES / "noise_band_continuation.csv", index=False)
        pd.concat(regressions, ignore_index=True).to_csv(TABLES / "noise_band_incremental_regression.csv", index=False)
    if args.section == "standalone":
        standalone_rows, standalone_inference = [], []
        for symbol in args.symbols:
            bars = load_bars(symbol)
            profiles, _ = build_profiles(bars)
            table, inference = standalone_strategy_tables(
                bars, profiles, nb=args.bootstrap, seed=args.seed, hac_lags=args.hac_lags
            )
            standalone_rows.append(table)
            standalone_inference.append(inference)
        pd.concat(standalone_rows, ignore_index=True).to_csv(
            TABLES / "noise_band_standalone_strategy.csv", index=False
        )
        pd.concat(standalone_inference, ignore_index=True).to_csv(
            TABLES / "noise_band_standalone_inference.csv", index=False
        )
    if args.section in ("strategy", "all"):
        if "QQQ" not in built:
            bars = load_bars("QQQ")
            profiles, s3 = build_profiles(bars)
        else:
            bars, profiles, s3 = built["QQQ"]
        strategy, inference = strategy_tables(bars, profiles, forward_start=forward_start,
                                              nb=args.bootstrap, seed=args.seed, hac_lags=args.hac_lags)
        strategy.to_csv(TABLES / "noise_band_strategy.csv", index=False)
        inference.to_csv(TABLES / "noise_band_inference.csv", index=False)
    metadata = {"created_utc": pd.Timestamp.now(tz="UTC").isoformat(), "git_branch": os.popen("git branch --show-current").read().strip(),
                "section": args.section, "symbols": args.symbols,
                "historical_end": str(HISTORICAL_END.date()),
                "forward_start": str(forward_start.date()),
                "confirmatory_min_days": 252,
                "bootstrap": args.bootstrap, "hac_lags": args.hac_lags, "seed": args.seed,
                "variants": [asdict(v) for v in VARIANTS], "data_md5": observed,
                "data_label": args.data_label,
                "interpretation": "history through historical_end is diagnostic; only post-freeze forward data can confirm",
                "elapsed_seconds": time.time() - started}
    payload = json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
    (TABLES / "noise_band_experiment_metadata.json").write_text(payload)
    (TABLES / f"noise_band_experiment_metadata_{args.section}.json").write_text(payload)
    print(f"wrote Noise Band validation tables to {TABLES} in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
