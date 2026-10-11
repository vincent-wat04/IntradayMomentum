"""Intraday noise-area momentum: one configurable state machine for all variants.

    cfg = NoiseCfg(stop="vwap", sizing="voltarget")
    res = run(load_bars("SPY"), cfg)            # DataFrame per day: ret, gross, cost, trades, exposure...

Variants (mapping to the literature / the A-share report):
    stop="band"      basic: long exits when close < LB, short exits when close > UB
    stop="vwap"      Zarattini et al. improved trailing stop: long stop = max(UB, VWAP),
                     short stop = min(LB, VWAP)   (A-share report "改进策略1")
    mode="longshort" 0 overnight, +/-e intraday
    mode="base"      base weight 0.5 held overnight; long signal -> 1.0, short signal -> 0.0,
                     back to 0.5 at stop / close; one round trip per day (A-share T+1 workaround)
    mode="longonly"  0 overnight, +e on long signals only
    sizing="fixed"   exposure 1.0
    sizing="voltarget" exposure = min(max_lev, target_daily_vol / sigma_14d)   (paper: 2%, 4x)
Timing (realistic, stricter than the papers):
    decision on bar k CLOSE -> fill at bar k+1 OPEN, cost on traded notional each side;
    end-of-day flatten at the session close price (market-on-close).
    entries only at `check_times`; exits at check times (paper) or every bar (A-share report).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd

from etft.intraday.bars import Bars
from etft.intraday.noise import noise_area, noise_area_volume, daily_vol


def rel_volume(b: Bars, lookback: int = 14, mode: str = "cum") -> pd.DataFrame:
    """Relative volume at each bar vs the same time of day over the PAST `lookback` days (PIT)."""
    x = b.v.cumsum(axis=1) if mode == "cum" else b.v.T.rolling(30, min_periods=1).sum().T
    norm = x.rolling(lookback, min_periods=max(5, int(0.7 * lookback))).mean().shift(1)
    return x / norm.replace(0, np.nan)

US_CHECKS = [f"{h:02d}:{m:02d}" for h in range(10, 16) for m in (0, 30)][:-1]   # 10:00 .. 15:30
CN_CHECKS = ["10:29", "11:29", "13:59"]


@dataclass
class NoiseCfg:
    lookback: int = 14
    vm: float = 1.0
    stop: str = "vwap"            # band | vwap
    mode: str = "longshort"       # longshort | base | longonly
    sizing: str = "fixed"         # fixed | voltarget
    target_vol: float = 0.02
    max_lev: float = 4.0
    check_times: list = field(default_factory=lambda: None)
    exit_every_bar: bool = False  # False: stops evaluated at check times (paper); True: every bar
    cost_bp: float = 1.0          # per side, on traded notional (used when per_share is None)
    # realistic per-share model (USD or CNY per share): (half_spread, commission+slippage).
    # marketable orders pay both; market-on-close exits pay only the second part.
    per_share: tuple | None = None
    bp_on_top: float = 0.0        # extra bp of notional on EVERY fill when per_share is used (CN commission)
    base_weight: float = 0.5
    one_trip: bool | None = None  # default True for mode=base
    day_filter: object = None     # optional pd.Series(bool) per day: trade only where True
    day_scale: object = None      # optional pd.Series(float) per day: multiplies exposure (NaN -> 1)
    # --- volume (E24): relative volume of the SIGNAL bars at the decision bar
    rvol_min: float | None = None  # enter only if rvol >= rvol_min
    rvol_max: float | None = None  # enter only if rvol <  rvol_max (complement tests)
    rvol_mode: str = "cum"         # cum: volume since open / same-time avg of past days; win30: last 30 bars
    rvol_size: bool = False        # size *= clip(rvol, 0.5, 2.0)
    # --- execution offset (E26): fill at bar k+1+delay_fixed+U{0..delay_max}
    delay_fixed: int = 0
    delay_max: int = 0
    grid_jitter: int = 0          # each day the whole check grid is shifted by U{0..grid_jitter} minutes
    fill_at_signal_close: bool = False  # paper convention: fill at the close of the decision bar (optimistic)
    twap_n: int = 1               # fill = average of the next twap_n bar opens (order split over n minutes)
    seed: int = 0
    record_trades: bool = False
    # --- volume-adjusted noise area (E24b): None = clock-time bands (paper)
    band_volume: str | None = None   # "power" | "volclock"
    band_gamma: float = 0.5

    def label(self) -> str:
        d = asdict(self)
        d.pop("day_filter"); d.pop("day_scale")
        return ",".join(f"{k}={v}" for k, v in d.items() if v is not None)


def run(b: Bars, cfg: NoiseCfg, signal_bars: Bars | None = None) -> pd.DataFrame:
    checks = cfg.check_times or (US_CHECKS if b.market == "US" else CN_CHECKS)
    one_trip = cfg.one_trip if cfg.one_trip is not None else (cfg.mode == "base")
    sb = signal_bars or b
    if cfg.band_volume:
        ub, lb, sc = noise_area_volume(b, cfg.lookback, cfg.vm, signal_bars=sb, gamma=cfg.band_gamma,
                                       mode=cfg.band_volume)
    else:
        ub, lb, sc = noise_area(b, cfg.lookback, cfg.vm, signal_bars=sb)
    vw = sb.vwap()
    days = b.days.intersection(sb.days)
    ub, lb, sc, vw = (x.reindex(days) for x in (ub, lb, sc, vw))
    O = b.o.reindex(days).values
    CL = b.c.reindex(days).values
    times = list(b.c.columns)
    is_check = np.isin(times, checks)
    last = b.last_idx.reindex(days).values
    close_px = b.day_close.reindex(days).values
    prev_adj = b.prev_close_adj.reindex(days).values
    if cfg.sizing == "voltarget":
        sd = daily_vol(sb, cfg.lookback).reindex(days).values
        expo = np.minimum(cfg.max_lev, cfg.target_vol / sd)
    else:
        expo = np.ones(len(days))
    flt = cfg.day_filter.reindex(days).fillna(False).values if cfg.day_filter is not None else np.ones(len(days), bool)
    if cfg.day_scale is not None:
        expo = expo * cfg.day_scale.reindex(days).fillna(1.0).values
    use_rv = cfg.rvol_min is not None or cfg.rvol_max is not None or cfg.rvol_size or cfg.record_trades
    RV = rel_volume(sb, cfg.lookback, cfg.rvol_mode).reindex(days).values if use_rv else None
    rng = np.random.default_rng(cfg.seed)
    trades = []
    UB, LB, SC, VW = ub.values, lb.values, sc.values, vw.values
    cost = cfg.cost_bp / 1e4
    ps = cfg.per_share
    top = cfg.bp_on_top / 1e4
    out = np.zeros((len(days), 7))  # ret, overlay_gross, cost, n_trades, avg_abs_pos, long_trips, short_trips
    for i in range(len(days)):
        e = expo[i]
        base = cfg.base_weight if cfg.mode == "base" else 0.0
        base_ret = base * (close_px[i] / prev_adj[i] - 1) if np.isfinite(prev_adj[i]) else 0.0
        if not np.isfinite(e) or not np.isfinite(UB[i]).any() or not flt[i]:
            out[i, 0] = base_ret if np.isfinite(base_ret) else 0.0
            continue
        pos, entry, gross, c_tot, ntr, held, nl, ns, done = 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0, 0, False
        L = int(last[i])
        active_from, t_rec = -1, None
        if cfg.grid_jitter:
            o_ = int(rng.integers(0, cfg.grid_jitter + 1))
            chk = np.zeros_like(is_check)
            idx_ = np.where(is_check)[0] + o_
            chk[idx_[idx_ < len(chk)]] = True
        else:
            chk = is_check
        for k in range(L):  # decision at close of bar k, fill at open of bar k+1 (+ delay)
            dly = cfg.delay_fixed + (int(rng.integers(0, cfg.delay_max + 1)) if cfg.delay_max else 0)
            fix = min(k + 1 + dly, L)
            if cfg.fill_at_signal_close:
                px_next = CL[i, k]
            elif cfg.twap_n > 1:
                seg = O[i, fix:min(fix + cfg.twap_n, L + 1)]
                px_next = np.nanmean(seg) if np.isfinite(seg).any() else np.nan
            else:
                px_next = O[i, fix]
            if not np.isfinite(px_next) or not np.isfinite(UB[i, k]):
                continue
            s = SC[i, k]
            can_eval = (chk[k] or (cfg.exit_every_bar and pos != 0)) and k >= active_from
            # ---- exits
            if pos != 0 and can_eval:
                if cfg.stop == "vwap":
                    stop_l = max(UB[i, k], VW[i, k]) if np.isfinite(VW[i, k]) else UB[i, k]
                    stop_s = min(LB[i, k], VW[i, k]) if np.isfinite(VW[i, k]) else LB[i, k]
                else:
                    stop_l, stop_s = LB[i, k], UB[i, k]
                if (pos > 0 and s < stop_l) or (pos < 0 and s > stop_s):
                    gross += pos * (px_next / entry - 1)
                    if t_rec is not None:
                        t_rec["pnl"] = pos * (px_next / entry - 1); trades.append(t_rec); t_rec = None
                    c_tot += abs(pos) * (cost if ps is None else (ps[0] + ps[1]) / px_next + top); ntr += 1
                    pos = 0.0
                    if one_trip:
                        done = True
            # ---- entries (only at check times, only if flat)
            if pos == 0 and chk[k] and not done:
                want = 0.0
                if s > UB[i, k]:
                    want = 1.0
                elif s < LB[i, k] and cfg.mode != "longonly":
                    want = -1.0
                rv = RV[i, k] if RV is not None else np.nan
                if want != 0 and RV is not None:
                    if cfg.rvol_min is not None and not (rv >= cfg.rvol_min):
                        want = 0.0
                    if cfg.rvol_max is not None and not (rv < cfg.rvol_max):
                        want = 0.0
                if want != 0:
                    if cfg.mode == "base":
                        size = (1 - base) if want > 0 else base   # 0.5 -> 1.0 or 0.5 -> 0.0
                    else:
                        size = e * (float(np.clip(rv, 0.5, 2.0)) if cfg.rvol_size and np.isfinite(rv) else 1.0)
                    pos = want * size; entry = px_next; active_from = fix + cfg.twap_n - 1
                    if cfg.record_trades:
                        t_rec = {"day": days[i], "bar": times[k], "dir": want, "size": size, "rvol": rv}
                    c_tot += abs(pos) * (cost if ps is None else (ps[0] + ps[1]) / px_next + top); ntr += 1
                    nl += want > 0; ns += want < 0
            held += abs(pos)
        if pos != 0:  # market-on-close
            gross += pos * (close_px[i] / entry - 1)
            if t_rec is not None:
                t_rec["pnl"] = pos * (close_px[i] / entry - 1); trades.append(t_rec)
            c_tot += abs(pos) * (cost if ps is None else ps[1] / close_px[i] + top); ntr += 1
        out[i] = (base_ret + gross - c_tot, gross, c_tot, ntr, held / max(L, 1), nl, ns)
    res = pd.DataFrame(out, index=days, columns=["ret", "overlay_gross", "cost", "trades", "avg_abs_pos",
                                                 "long_trips", "short_trips"])
    res["exposure_cap"] = expo
    if cfg.record_trades:
        res.attrs["trades"] = pd.DataFrame(trades)
    return res


def buy_hold(b: Bars) -> pd.Series:
    return (b.day_close / b.prev_close_adj - 1).fillna(0.0)


def overnight_intraday(b: Bars) -> pd.DataFrame:
    return pd.DataFrame({"overnight": b.day_open / b.prev_close_adj - 1, "intraday": b.day_close / b.day_open - 1})
