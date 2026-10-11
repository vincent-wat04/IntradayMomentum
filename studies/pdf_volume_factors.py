"""Intraday volume-distribution factors from 东北证券《日内成交量分布因子及 Logsig-Alpha 因子生成》
(因子选股系列之六, 2023-11-29, 王琦 / 贾英).

Implemented exactly as specified in the report, from 5-minute bars.  Four families plus the report's
key structural idea, the STABILITY operator.

Family 1  日内对数成交量分布 (section 2.1)
  LOGV_STD       (1/tau) sum_s sigma(log V_s)                     report Rank IC 1.55%, ICIR 0.22
  LOGV_SKEW      (1/tau) sum_s skew(log V_s)                      report Rank IC 5.94%, ICIR 0.72
  both negative-signed

Family 2  成交量在时间上的分布 (section 2.2)
  VOL_RATIO_1          first 30min volume / daily volume          Rank IC 4.61%
  VOL_INTRA_STD_1      sigma(first-30min volumes) / daily volume  Rank IC 4.33%
  VOL_RATIO_1_5        first-30min / afternoon-open-30min         Rank IC 5.56%  <- best of family
  VOL_RATIO_8_4        last-30min / pre-midday-close-30min        Rank IC 2.58%
  plus the _STAB version of each (see below)

Family 3  成交量在收益率上的分布 (section 2.3)
  significant-up bars: I0 = { i : r_i > 0 and r_i > mean(r) + sigma(r) }
  SIG_UP_PV_RATIO      sigma(r_I0) * [ mean(V_I0) / mean(V_all) ]  Rank IC 6.30%  <- best single factor
  SIG_UP_PV_STD        sigma(r_I0) * sigma(V_I0) / mean(V_all)     Rank IC 5.93%
  plus _STAB versions

Family 4  成交量在价格上的分布 (section 2.4)
  V_P_SKEWNESS   Pearson median skewness 3(mu-m)/sigma of the volume-over-price-decile distribution
  V_P_REVERSAL   POC-based reversal: 1{r<0}(POC-C+1)r + 1{r>=0}(C-POC+1)r, POC from 3-bin smoothed
                 volume profile                                   Rank IC 5.50%

STABILITY operator (the transferable idea)
  For any daily quantity X_s the report pairs it with
      STAB_tau(X)_t = (1/tau) sum_{s=t-tau+1..t} ( X_s - mean_tau(X) )^2
  i.e. the within-window variance of the daily factor.  In the report the stability versions often have
  HIGHER ICIR than the level versions (e.g. 早盘成交量占比 ICIR 0.54 -> stability 1.06), because they
  select names whose intraday behaviour is persistent rather than episodic.  `stability()` below applies
  it to any daily series, so it can also be applied to factors the report does not cover -- including the
  daily-frequency cross-sectional library, where no intraday data is available.

Order-flow proxies (not in the report; asked for separately)
  True OFI and average order size need L2/tick data.  From 5-minute bars the best available proxies are
  implemented in `order_flow_proxies`: a Lee-Ready-style bar sign using the bar's close location within
  its own range, times bar volume (OFI_PROXY), the buy/sell split of that (BUY_SHARE), and the ratio of
  buy-side to sell-side average bar volume (AVG_SIZE_RATIO), which is the bar-level analogue of
  avg buy order size / avg sell order size.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TAU = 20            # the report's smoothing window
LEVEL_NAMES = ["LOGV_STD", "LOGV_SKEW", "VOL_RATIO_1", "VOL_INTRA_STD_1", "VOL_RATIO_1_5",
               "VOL_RATIO_8_4", "SIG_UP_PV_RATIO", "SIG_UP_PV_STD", "V_P_SKEWNESS", "V_P_REVERSAL",
               "OFI_PROXY", "BUY_SHARE", "AVG_SIZE_RATIO"]
STAB_OF = ["VOL_RATIO_1", "VOL_INTRA_STD_1", "VOL_RATIO_1_5", "VOL_RATIO_8_4",
           "SIG_UP_PV_RATIO", "SIG_UP_PV_STD", "OFI_PROXY", "BUY_SHARE"]
NAMES = LEVEL_NAMES + [f"{n}_STAB" for n in STAB_OF]
# report direction: every factor in sections 2.1-2.4 is negative-signed
REPORT_SIGN = {n: -1 for n in NAMES}


def _s(a, b):
    return float(a / b) if np.isfinite(a) and np.isfinite(b) and abs(b) > 1e-12 else np.nan


def daily_raw(bar_open, bar_close, bar_vol, clock, day_close, seg):
    """Per-day values of every LEVEL factor.  `seg` maps a segment name -> boolean mask over bars."""
    v = np.asarray(bar_vol, float)
    p = np.asarray(bar_open, float)
    c = np.asarray(bar_close, float)
    n = len(v)
    out = {k: np.nan for k in LEVEL_NAMES}
    if n < 10 or not np.isfinite(v).any():
        return out
    tot = float(np.nansum(v))
    mean_v = tot / n if n else np.nan

    # ---- family 1: log volume distribution
    lv = np.log1p(v[v >= 0])
    if len(lv) > 3 and np.nanstd(lv) > 0:
        out["LOGV_STD"] = float(np.nanstd(lv, ddof=1))
        out["LOGV_SKEW"] = float(pd.Series(lv).skew())

    # ---- family 2: volume over time
    px = np.r_[p, day_close]
    r = np.diff(np.log(np.where(px > 0, px, np.nan)))
    o30, pm30, close30, midclose30 = seg["open30"], seg["pm_open30"], seg["close30"], seg["mid_close30"]
    out["VOL_RATIO_1"] = _s(np.nansum(v[o30]), tot)
    out["VOL_INTRA_STD_1"] = _s(np.nanstd(v[o30], ddof=1) if o30.sum() > 1 else np.nan, tot)
    out["VOL_RATIO_1_5"] = _s(np.nansum(v[o30]), np.nansum(v[pm30])) if pm30.any() else np.nan
    out["VOL_RATIO_8_4"] = _s(np.nansum(v[close30]), np.nansum(v[midclose30])) if midclose30.any() else np.nan

    # ---- family 3: volume over returns, on significantly-up bars
    rr = r[: n]
    fin = np.isfinite(rr)
    if fin.sum() > 5:
        mu, sd = np.nanmean(rr[fin]), np.nanstd(rr[fin], ddof=1)
        up = fin & (rr > 0) & (rr > mu + sd)
        if up.sum() >= 2:
            sr = float(np.nanstd(rr[up], ddof=1))
            vu = v[: n][up]
            out["SIG_UP_PV_RATIO"] = _s(sr * np.nanmean(vu), mean_v)
            out["SIG_UP_PV_STD"] = _s(sr * np.nanstd(vu, ddof=1), mean_v)

    # ---- family 4: volume over price
    if tot > 0 and np.isfinite(p).all():
        edges = np.quantile(p, np.linspace(0, 1, 11))
        lab = np.clip(np.searchsorted(edges[1:-1], p, side="right"), 0, 9)
        hist = np.bincount(lab, weights=v, minlength=10)
        prob = hist / tot
        pos = np.arange(10) / 10.0
        mu_ = float(np.sum(prob * pos))
        cdf = np.cumsum(prob)
        med_ = float(np.searchsorted(cdf, 0.5) / 10.0)
        sd_ = float(np.sqrt(np.sum(prob * (pos - mu_) ** 2)))
        out["V_P_SKEWNESS"] = 3 * (mu_ - med_) / sd_ if sd_ > 1e-12 else np.nan
        smooth = np.r_[2 * hist[0] + hist[1], hist[:-2] + hist[2:], hist[-2] + 2 * hist[-1]]
        poc = float(np.argmax(smooth)) / 10.0
        cbin = float(np.clip(np.searchsorted(edges[1:-1], day_close, side="right"), 0, 9)) / 10.0
        dayret = _s(day_close - p[0], p[0])
        if np.isfinite(dayret):
            out["V_P_REVERSAL"] = ((poc - cbin + 1) if dayret < 0 else (cbin - poc + 1)) * dayret

    # ---- order-flow proxies
    rng = np.asarray(seg["hi"], float) - np.asarray(seg["lo"], float)
    loc = np.where(rng > 1e-12, (c - np.asarray(seg["lo"], float)) / rng, 0.5)
    sgn = 2 * loc - 1                                     # +1 close at the high, -1 at the low
    buy = v * np.clip(loc, 0, 1)
    sell = v * np.clip(1 - loc, 0, 1)
    out["OFI_PROXY"] = _s(np.nansum(sgn * v), tot)
    out["BUY_SHARE"] = _s(np.nansum(buy), np.nansum(buy) + np.nansum(sell))
    nb, ns = int((sgn > 0).sum()), int((sgn < 0).sum())
    out["AVG_SIZE_RATIO"] = _s(np.nansum(buy) / max(nb, 1), np.nansum(sell) / max(ns, 1))
    return out


def stability(x: pd.Series, tau=TAU) -> pd.Series:
    """The report's stability operator: within-window variance of the daily factor."""
    return x.rolling(tau, min_periods=max(5, tau // 2)).var(ddof=0)


def smooth(x: pd.Series, tau=TAU) -> pd.Series:
    """The report's smoothing: tau-day mean of the daily factor."""
    return x.rolling(tau, min_periods=max(5, tau // 2)).mean()


def build(path_or_bars, panel, cfg, market="US", tau=TAU):
    """Return a daily DataFrame of smoothed level factors and their stability versions.

    market='US'  : 09:30-16:00, segments open30 = 09:30-10:00, pm_open30 = 13:00-13:30,
                   mid_close30 = 11:30-12:00, close30 = 15:30-16:00
    market='CN'  : 09:30-11:30 + 13:00-15:00, segments open30 = 09:30-10:00,
                   pm_open30 = 13:00-13:30, mid_close30 = 11:00-11:30, close30 = 14:30-15:00
                   (this is the report's own A-share segmentation)
    """
    if isinstance(path_or_bars, (str,)):
        raw = pd.read_csv(path_or_bars, parse_dates=["datetime"])
        raw["day"] = raw.datetime.dt.strftime("%Y-%m-%d")
        raw["clock"] = raw.datetime.dt.strftime("%H:%M")
    else:
        raw = path_or_bars
    close_map = dict(zip(panel.dates, panel.close))
    if market == "CN":
        segs = dict(open30=("09:30", "10:00"), pm_open30=("13:00", "13:30"),
                    mid_close30=("11:00", "11:30"), close30=("14:30", "15:00"))
    else:
        segs = dict(open30=("09:30", "10:00"), pm_open30=("13:00", "13:30"),
                    mid_close30=("11:30", "12:00"), close30=("15:30", "16:00"))
    rows = {}
    early = set(getattr(cfg, "early_closes", ()) or ())
    for day, g in raw.groupby("day", sort=True):
        if day in early or day not in close_map:
            continue
        g = g.sort_values("datetime")
        clock = g.clock.to_numpy()
        seg = {k: (clock >= a) & (clock < b) for k, (a, b) in segs.items()}
        seg["hi"] = g.high.to_numpy(float) if "high" in g else g.close.to_numpy(float)
        seg["lo"] = g.low.to_numpy(float) if "low" in g else g.open.to_numpy(float)
        rows[day] = daily_raw(g.open.to_numpy(float), g.close.to_numpy(float), g.volume.to_numpy(float),
                              clock, float(close_map[day]), seg)
    D = pd.DataFrame(rows).T.reindex(panel.dates).replace([np.inf, -np.inf], np.nan)
    out = {}
    for n in LEVEL_NAMES:
        out[n] = smooth(D[n], tau)
    for n in STAB_OF:
        out[f"{n}_STAB"] = stability(D[n], tau)
    return pd.DataFrame(out, index=panel.dates), D


def order_flow_proxies_note():
    return ("True OFI / average order size need L2 or tick data. The bar-level proxies here use the "
            "close location within each bar's own high-low range as a Lee-Ready substitute; they "
            "capture the sign of pressure but not queue dynamics or true order sizes.")


if __name__ == "__main__":
    from pathlib import Path
    from index_timing_framework import load_market
    from run_full_extended import config_from
    cfg = config_from(Path("data"))
    for sym in ("SPY", "QQQ"):
        panel = load_market(f"data/{sym}_5min.csv", cfg)
        F, D = build(f"data/{sym}_5min.csv", panel, cfg, market="US")
        F.to_pickle(f"outputs/full/cache/{sym}_pdf_vol.pkl")
        print(f"{sym}: {F.shape[1]} factors, mean coverage {F.notna().mean().mean():.3f}")
        print(D[["LOGV_SKEW", "VOL_RATIO_1", "VOL_RATIO_1_5", "SIG_UP_PV_RATIO", "V_P_REVERSAL",
                 "OFI_PROXY", "AVG_SIZE_RATIO"]].describe().loc[["mean", "std"]].round(4).to_string())
