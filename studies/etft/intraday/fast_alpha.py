"""Zarattini & Pagani (2026), "Improving Performance with Fast Alphas: A Tactical Overlay for
Intraday Trend Trading", Concretum QuanTips #2 (data/raw/papers/zarattini2026_fast_alphas.pdf).

Three pieces, all on 5-minute regular-session bars built from our 1-minute bars:
  1. fast alpha: streak mean-reversion; N=1 -> position on bar t+1 = -sign(R_t)
  2. baseline intraday trend (Kaufman ATR breakout):
        UB/LB = session open +/- 0.5 * ATR(14, daily, through t-1)
        entries only at HH:00/HH:15/HH:30/HH:45 when the 5-min close is outside a band;
        stop when the price returns to the session open (checked at the same 15-min grid);
        flat at the close; size = AUM * min(4, 2% / sigma_14d) / session open, fixed all day
  3. execution overlay: after an entry (exit) trigger, wait for the first 5-min bar moving
     AGAINST (WITH) the position, then execute at that bar's close
Details the note leaves open are explicit options (documented in docs/10_FAST_ALPHA.md):
  cancel_on_stop  a pending entry is cancelled if the stop condition (price back at the
                  session open) is met before it is filled            (default True)
  fill            "close": at the decision bar close (paper convention)
                  "next_open": at the open of the next 1-minute bar (our realistic convention)
Costs (paper): $0.0035/share on buys, $0.007/share on sells (sell side doubled for SEC fees),
no slippage; optional half-spread per share on every fill.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from etft.intraday.bars import Bars


# ----------------------------------------------------------------------------- 5-min bars
@dataclass
class Bars5:
    days: pd.DatetimeIndex
    times: list
    o: np.ndarray            # (D, B) 5-min bar open
    c: np.ndarray            # 5-min bar close
    next_open: np.ndarray    # open of the first 1-min bar AFTER the 5-min bar (nan for the last bar)
    day_open: np.ndarray
    day_close: np.ndarray
    n_bars: np.ndarray       # valid 5-min bars per day (half days)


def to_5min(b: Bars) -> Bars5:
    times = b.times
    ends = [i for i, t in enumerate(times) if int(t[-2:]) % 5 == 0]
    if b.market == "CN":  # 09:30 auction bar merged into the first bar
        ends = [i for i in ends if times[i] != "09:30"]
    starts = [0] + [e + 1 for e in ends[:-1]]
    O1, C1 = b.o.values, b.c.values
    D = len(b.days)
    o = np.stack([O1[:, s] for s in starts], axis=1)
    c = np.stack([C1[:, e] for e in ends], axis=1)
    nxt = np.stack([O1[:, e + 1] if e + 1 < O1.shape[1] else np.full(D, np.nan) for e in ends], axis=1)
    last = b.last_idx.values
    valid = np.array(ends)[None, :] <= last[:, None]
    c = np.where(valid, c, np.nan); o = np.where(valid, o, np.nan)
    nxt = np.where(np.array(ends)[None, :] + 1 <= last[:, None], nxt, np.nan)
    return Bars5(b.days, [times[e] for e in ends], o, c, nxt, b.day_open.values, b.day_close.values, valid.sum(axis=1))


def daily_atr(b: Bars, n: int = 14, wilder: bool = True) -> pd.Series:
    """ATR(n) on regular-session daily bars, value for day t uses days <= t-1 only."""
    hi = b.h.max(axis=1); lo = b.l.min(axis=1); pc = b.prev_close_adj
    tr = pd.concat([hi - lo, (hi - pc).abs(), (lo - pc).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean() if wilder else tr.rolling(n).mean()
    return atr.shift(1)


def vol_lev(b: Bars, n: int = 14, target: float = 0.02, cap: float = 4.0) -> pd.Series:
    r = b.day_close / b.prev_close_adj - 1
    return (target / r.rolling(n, min_periods=n).std().shift(1)).clip(upper=cap)


# ----------------------------------------------------------------------------- 1. fast alpha stats
def streak_table(b5: Bars5, n_max: int = 5) -> pd.DataFrame:
    """Next-bar return conditional on >= N consecutive same-sign 5-min bars (within the day)."""
    c, o = b5.c, b5.o
    r = np.diff(np.c_[b5.day_open[:, None], c], axis=1) / np.c_[b5.day_open[:, None], c][:, :-1]
    s = np.sign(r)
    run = np.zeros_like(s)
    for j in range(s.shape[1]):
        prev = run[:, j - 1] if j else 0
        same = (s[:, j] == s[:, j - 1]) if j else np.zeros(len(s), bool)
        run[:, j] = np.where(s[:, j] == 0, 0, np.where(same & (s[:, j] != 0), prev + s[:, j], s[:, j]))
    nxt_r = np.c_[r[:, 1:], np.full((len(r), 1), np.nan)]
    nxt_d = np.c_[np.diff(c, axis=1), np.full((len(c), 1), np.nan)]
    rows = []
    D = len(c)
    for N in range(1, n_max + 1):
        for side, m in (("Down", run <= -N), ("Up", run >= N)):
            mm = m & np.isfinite(nxt_r)
            rows.append({"N>=": N, "streak": side, "trades_per_day": mm.sum() / D,
                         "next_bar_bp": np.nanmean(nxt_r[mm]) * 1e4, "next_bar_usd": np.nanmean(nxt_d[mm]),
                         "t": np.nanmean(nxt_r[mm]) / np.nanstd(nxt_r[mm]) * np.sqrt(mm.sum())})
    return pd.DataFrame(rows)


def fast_alpha_standalone(b5: Bars5, buy_cost=0.0035, sell_cost=0.007, half_spread=0.0) -> pd.DataFrame:
    """N=1: hold -sign(R_t) from close t to close t+1, fully invested (shares = AUM/open), flat EOD."""
    c = b5.c
    D, B = c.shape
    ret_g = np.zeros(D); cost = np.zeros(D)
    for i in range(D):
        L = int(b5.n_bars[i]); op = b5.day_open[i]
        if L < 3 or not np.isfinite(op):
            continue
        px = c[i, :L]
        r = np.diff(np.r_[op, px]) / np.r_[op, px][:-1]
        pos = -np.sign(r)[:-1]           # position over bar j+1, decided at close j
        pnl = (pos * np.diff(px)).sum()  # $/share of the fixed share count (= AUM / open)
        pos_path = np.r_[0.0, pos, 0.0]
        dq = np.diff(pos_path)
        cst = (np.clip(dq, 0, None) * (buy_cost + half_spread)).sum() + (np.clip(-dq, 0, None) * (sell_cost + half_spread)).sum()
        ret_g[i] = pnl / op; cost[i] = cst / op
    return pd.DataFrame({"gross": ret_g, "cost": cost, "net": ret_g - cost}, index=b5.days)


# ----------------------------------------------------------------------------- 2+3. trend model
CHECK15 = lambda t: t[-2:] in ("00", "15", "30", "45")  # noqa: E731


def atr_breakout(b: Bars, overlay_entry=False, overlay_exit=False, cancel_on_stop=True, fill="close",
                 buy_cost=0.0035, sell_cost=0.007, half_spread=0.0, band_mult=0.5, atr_n=14,
                 bands: tuple | None = None, stops: tuple | None = None, check_every: int = 15,
                 max_wait: int | None = None, record=False) -> pd.DataFrame:
    """Daily returns of the ATR-breakout trend model with optional fast-alpha execution overlay.

    State machine on 5-min bars, decision at the close of bar j:
      in a position : at a 15-min check, stop if price is back through the session open.
                      overlay_exit -> mark a pending exit and liquidate at the first LATER bar that
                      moves in the position's favour (long: up bar).  Without overlay: exit now.
      flat          : at a 15-min check, breakout trigger if close beyond a band.
                      overlay_entry -> pending entry, filled at the first LATER bar moving against
                      the trade (long: down bar).  cancel_on_stop drops a pending entry once the
                      price is back through the open; a new opposite trigger replaces it.
      max_wait      : optional urgency control (fill anyway after max_wait bars), default None (paper).
    `bands` may override (UB, LB) as (D, B5) arrays on the 5-min grid."""
    b5 = to_5min(b)
    atr = daily_atr(b, atr_n).values
    lev = vol_lev(b).values
    chk = np.array([int(t[-2:]) % check_every == 0 and t != "09:30" for t in b5.times])
    D = len(b5.days)
    out = np.zeros((D, 4)); trades = []
    for i in range(D):
        op = b5.day_open[i]; L = int(b5.n_bars[i]); e = lev[i]
        if not (np.isfinite(op) and np.isfinite(atr[i]) and np.isfinite(e)) or L < 4:
            continue
        if bands is None:
            ub = np.full(L, op + band_mult * atr[i]); lb = np.full(L, op - band_mult * atr[i])
        else:
            ub, lb = bands[0][i, :L], bands[1][i, :L]
        px = b5.c[i, :L]; nx = b5.next_open[i, :L]
        r5 = np.diff(np.r_[op, px])
        sl = np.full(L, op) if stops is None else stops[0][i, :L]   # long exits if price < sl
        ss = np.full(L, op) if stops is None else stops[1][i, :L]   # short exits if price > ss
        pos, entry = 0, 0.0
        pin, pin_j = 0, -1      # pending entry direction / trigger bar
        pout_j = -1             # pending exit trigger bar
        pnl = cst = 0.0; nt = 0
        for j in range(L - 1):
            p = px[j]
            fpx = p if fill == "close" else nx[j]
            if not np.isfinite(fpx):
                continue
            if pos != 0:
                stop = (pos > 0 and p < sl[j]) or (pos < 0 and p > ss[j])
                do_exit = False
                if pout_j < 0 and chk[j] and stop:
                    if overlay_exit:
                        pout_j = j
                    else:
                        do_exit = True
                elif pout_j >= 0 and j > pout_j and (np.sign(r5[j]) == pos or
                                                     (max_wait is not None and j - pout_j >= max_wait)):
                    do_exit = True
                if do_exit:
                    pnl += pos * (fpx - entry)
                    cst += (sell_cost if pos > 0 else buy_cost) + half_spread; nt += 1
                    if record:
                        trades.append((b5.days[i], b5.times[j], pos, entry, fpx))
                    pos, pout_j = 0, -1
                continue
            # flat: 1) try to fill a pending entry, 2) look for a (new) trigger
            if pin != 0 and j > pin_j:
                back = (pin > 0 and p < sl[j]) or (pin < 0 and p > ss[j])
                if cancel_on_stop and back:
                    pin = 0
                elif np.sign(r5[j]) == -pin or (max_wait is not None and j - pin_j >= max_wait):
                    pos, entry, pin = pin, fpx, 0
                    cst += (buy_cost if pos > 0 else sell_cost) + half_spread; nt += 1
                    continue
            if chk[j]:
                want = 1 if p > ub[j] else (-1 if p < lb[j] else 0)
                if want != 0 and want != pin:
                    if overlay_entry:
                        pin, pin_j = want, j
                    else:
                        pos, entry = want, fpx
                        cst += (buy_cost if pos > 0 else sell_cost) + half_spread; nt += 1
        if pos != 0:  # flat at the close (last 5-min close)
            pnl += pos * (px[L - 1] - entry)
            cst += (sell_cost if pos > 0 else buy_cost); nt += 1
            if record:
                trades.append((b5.days[i], "close", pos, entry, px[L - 1]))
        out[i] = (e * pnl / op, e * cst / op, nt, e)
    res = pd.DataFrame(out, index=b5.days, columns=["gross", "cost", "trades", "lev"])
    res["net"] = res["gross"] - res["cost"]
    if record:
        res.attrs["trades"] = pd.DataFrame(trades, columns=["day", "time", "dir", "entry", "exit"])
    return res


def noise_area_grid(b: Bars, lookback: int = 14, vm: float = 1.0):
    """Zarattini-Aziz-Barbon (2024) noise area + VWAP trailing stop sampled on the 5-min grid,
    so the fast-alpha overlay can be applied to the 'Beat the Market' strategy as well.
    Returns (bands, stops) for atr_breakout(..., check_every=30)."""
    from etft.intraday.noise import noise_area
    ub, lb, _ = noise_area(b, lookback, vm)
    vw = b.vwap()
    b5 = to_5min(b)
    cols = b5.times
    UB, LB, VW = ub[cols].values, lb[cols].values, vw[cols].values
    SL = np.fmax(UB, VW); SS = np.fmin(LB, VW)
    return (UB, LB), (SL, SS)
