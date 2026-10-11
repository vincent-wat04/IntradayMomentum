"""Stage 9: does the pipeline ADAPT, or are the "good factors" fixed?  A-share vs US.

The question matters because A-share indices are widely described as mean-reverting where US indices
trend.  If the pipeline hard-codes factor identity or sign, it will fail on A-share; if it re-derives
both from data, it should pick a different (possibly sign-flipped) set.

Design of this pipeline, for reference:
  * factor SIGN is never hard-coded.  It is either frozen from the dev-period rank IC, or re-estimated
    on an expanding window (`Z252_xr`).  Nothing uses the registry's economic prior -- stage 1 showed
    that prior agrees with the data only ~50% of the time.
  * factor SELECTION is re-run every 63 sessions on the then-available matured labels.  So the chosen
    set is a function of the market, not a constant.
This script measures how different the resulting choice actually is.

Data: CSI300 / CSI500 / CSI1000 / SSE50 1-minute bars from the research server, 2019-01..2026-09.
A-share sessions are 09:30-11:30 and 13:00-15:00 = 240 minutes = 48 five-minute bars labelled by start.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import types
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

import intraday_factor_library as LIB
import timing_transform_study as T

warnings.filterwarnings("ignore")
SRC = Path("../cn_min1")
OUT = Path("outputs/full/cn"); OUT.mkdir(parents=True, exist_ok=True)
SYMS = {"CSI300": "SH.000300", "CSI500": "SH.000905", "CSI1000": "SH.000852", "SSE50": "SH.000016"}
SPLIT = "2023-01-03"
HS = (1, 2, 5, 20)
AGGS = ("L1", "MA5", "MA20", "EWM10", "MED20")
AM = [f"{9 + (30 + 5 * k) // 60:02d}:{(30 + 5 * k) % 60:02d}" for k in range(24)]   # 09:30..11:25
PM = [f"{13 + (5 * k) // 60:02d}:{(5 * k) % 60:02d}" for k in range(24)]            # 13:00..14:55
BARS = AM + PM


def build_5min(code):
    raw = pd.read_csv(SRC / f"{code}.csv", parse_dates=["time_key"]).sort_values("time_key")
    raw["day"] = raw.time_key.dt.strftime("%Y-%m-%d")
    raw["clock"] = raw.time_key.dt.strftime("%H:%M")
    # the 11:30 and 15:00 prints are session-closing marks; fold them into the preceding bar
    mins = raw.time_key.dt.hour * 60 + raw.time_key.dt.minute
    grp = np.where(mins <= 11 * 60 + 30, (mins - 570).clip(0, None) // 5, 24 + (mins - 780).clip(0, None) // 5)
    raw["bar"] = np.clip(grp, 0, 47)
    g = raw.groupby(["day", "bar"])
    bars = pd.DataFrame(dict(open=g.open.first(), high=g.high.max(), low=g.low.min(),
                             close=g.close.last(), volume=g.volume.sum())).reset_index()
    full = bars.groupby("day").bar.count()
    keep = full[full == 48].index
    bars = bars[bars.day.isin(keep)].copy()
    bars["clock"] = [BARS[b] for b in bars.bar]
    bars["datetime"] = pd.to_datetime(bars.day + " " + bars.clock)
    return bars.sort_values("datetime").reset_index(drop=True)


def panel_from(bars):
    dates = pd.Index(bars.day.drop_duplicates().to_list(), name="day")
    opens = bars[bars.bar == 0].set_index("day").open.reindex(dates).to_numpy(float)
    close = bars[bars.bar == 47].set_index("day").close.reindex(dates).to_numpy(float)
    return types.SimpleNamespace(dates=dates, daily_open=opens, close=close)


def cn_factors(bars, panel):
    """Realized-measure library on A-share bars; open/close windows remapped to the CN session."""
    close_map = dict(zip(panel.dates, panel.close))
    prev = pd.Series(panel.close, index=panel.dates).shift(1)
    out = {}
    for day, g in bars.groupby("day", sort=True):
        g = g.sort_values("bar")
        clock = g.clock.to_numpy()
        # remap the three US windows used inside the library onto CN times
        cl = np.where(np.isin(clock, AM[:6]), "09:30",                     # first 30 min
             np.where(np.isin(clock, PM[-6:]), "15:30",                    # last 30 min
             np.where(np.isin(clock, AM[-6:] + PM[:6]), "12:00", "11:00")))
        out[day] = LIB._day_factors(g.open.to_numpy(float), g.high.to_numpy(float), g.low.to_numpy(float),
                                    g.close.to_numpy(float), g.volume.to_numpy(float), cl,
                                    float(close_map[day]), float(prev.loc[day]))
    return pd.DataFrame(out).T.reindex(panel.dates).replace([np.inf, -np.inf], np.nan)


def target(panel, h):
    """next_open execution: enter at open of t+1, exit at close of t+h."""
    n = len(panel.dates); y = np.full(n, np.nan)
    i = np.arange(n); e, x = i + 1, i + h
    ok = x < n
    y[i[ok]] = np.log(panel.close[x[ok]] / panel.daily_open[e[ok]])
    return y


def run_book_cn(panel, w, fee_bp=1.0):
    o, c = panel.daily_open, panel.close
    w = np.nan_to_num(np.asarray(w, float))
    on = np.r_[np.nan, o[1:] / c[:-1] - 1]; intra = c / o - 1
    prev = np.r_[0.0, w[:-1]]; prev2 = np.r_[0.0, prev[:-1]]
    r = (1 + prev2 * np.nan_to_num(on)) * (1 + prev * intra) - 1 - fee_bp * 1e-4 * np.abs(prev - prev2)
    return pd.Series(r, index=pd.to_datetime(panel.dates))


def study(args):
    name, code = args
    bars = build_5min(code); panel = panel_from(bars)
    F = cn_factors(bars, panel)
    split = int(np.searchsorted(panel.dates, SPLIT))
    rows = []
    for h in HS:
        y = target(panel, h)
        lab = (np.arange(len(y)) + h < split) & np.isfinite(y)
        ytest = np.arange(len(y)) >= split
        for fac in F.columns:
            for agg in AGGS:
                a = T.aggregate(F[fac], agg); av = a.to_numpy(float)
                pct, mu, sd, med, mad = T.prior_stats(av)
                with np.errstate(all="ignore"):
                    z = np.clip((av - mu) / sd, -3, 3) / 3
                m = lab & np.isfinite(z)
                if m.sum() < 200 or np.nanstd(z[m]) == 0:
                    continue
                ic_dev = T.spearman(z[m], y[m])
                sgn = float(np.sign(ic_dev) or 1.0)
                w = np.clip(T.smooth(z * sgn, h), -1.5, 1.5)
                r = run_book_cn(panel, w)
                mt = ytest & np.isfinite(y) & np.isfinite(z)
                rows.append(dict(market="CN", symbol=name, factor=fac, agg=agg, h=h, ic_dev=ic_dev,
                                 dev_sign=sgn, ic_test=T.spearman(z[mt], y[mt]),
                                 sr_dev=T.sr(r.loc[:"2022"]), sr_test=T.sr(r.loc["2023":]),
                                 bh_sr_test=T.sr(run_book_cn(panel, np.ones(len(panel.dates))).loc["2023":])))
    return pd.DataFrame(rows)


def main():
    with Pool(4) as p:
        res = pd.concat(p.map(study, list(SYMS.items())), ignore_index=True)
    res.to_parquet(OUT / "cn_single_factor.parquet")

    # ---- US comparison: same factors, same maps, same horizons
    US = pd.read_parquet("outputs/full/transform/single_factor_transforms.parquet")
    US = US[(US["map"] == "Z252") & US.factor.str.startswith("R:") & US.h.isin(HS) & US["agg"].isin(AGGS)].copy()
    US["factor"] = US.factor.str.replace("R:", "", regex=False)
    us = US.groupby(["factor", "agg", "h"]).agg(us_ic_dev=("ic_dev", "mean"), us_sign=("dev_sign", "mean"),
                                                us_ic_test=("ic_test", "mean"), us_sr_test=("sr_test", "mean"))
    cn = res.groupby(["factor", "agg", "h"]).agg(cn_ic_dev=("ic_dev", "mean"), cn_sign=("dev_sign", "mean"),
                                                 cn_ic_test=("ic_test", "mean"), cn_sr_test=("sr_test", "mean"))
    J = us.join(cn, how="inner").reset_index()
    J["sign_agrees"] = np.sign(J.us_ic_dev) == np.sign(J.cn_ic_dev)
    J.to_csv(OUT / "us_cn_comparison.csv", index=False)

    pd.set_option("display.width", 280); pd.set_option("display.max_rows", 300)
    print("=== A-share buy & hold, test 2023-2026 ===")
    print(res.groupby("symbol").bh_sr_test.first().round(3).to_string())
    print("\n=== dev-period IC sign agreement between US and A-share, by horizon ===")
    print(J.groupby("h").agg(n=("sign_agrees", "size"), agree_share=("sign_agrees", "mean"),
                             ic_corr=("us_ic_dev", lambda s: np.corrcoef(s, J.loc[s.index, "cn_ic_dev"])[0, 1])
                             ).round(3).to_string())
    print("\n=== A-share single-factor performance (dev-signed, test) ===")
    print(res.groupby(["symbol", "h"]).agg(n=("sr_test", "size"), med_sr=("sr_test", "median"),
                                           pos=("sr_test", lambda s: (s > 0).mean()),
                                           med_ic=("ic_test", "median")).round(3).to_string())
    print("\n=== factors whose sign FLIPS between US and A-share (|IC| >= 0.05 in both) ===")
    strong = J[(J.us_ic_dev.abs() >= 0.05) & (J.cn_ic_dev.abs() >= 0.05)]
    fl = strong[~strong.sign_agrees].groupby("factor").size().sort_values(ascending=False)
    ag = strong[strong.sign_agrees].groupby("factor").size().sort_values(ascending=False)
    print("flipped:", dict(fl.head(12)))
    print("agreed :", dict(ag.head(12)))
    print(f"\noverall: {len(strong)} strong cells, {strong.sign_agrees.mean():.1%} sign agreement")
    print("\n=== top A-share factors by test IC (h=5,20) ===")
    t = res[res.h.isin([5, 20])].groupby(["factor", "h"]).agg(
        cn_ic=("ic_test", "mean"), cn_sr=("sr_test", "mean"), nsym=("symbol", "nunique"))
    print(t.reindex(t.cn_ic.abs().sort_values(ascending=False).index).head(15).round(3).to_string())
    return res, J


if __name__ == "__main__":
    main()
