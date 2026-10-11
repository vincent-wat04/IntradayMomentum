"""Minute/5-min bar aggregated daily factors (realized-measure family).

Everything here is computed from the 5-minute bars of day t only (plus the previous close for the
overnight gap), so the value is known at the close of t.  Families and references:

  semivariance / signed jumps   RS+, RS-, SJV = RS+ - RS-, RSJ = SJV/RV
                                Barndorff-Nielsen, Kinnebrock & Shephard (2010); Patton & Sheppard
                                (2015, REStat) "Good volatility, bad volatility"
  jump vs continuous            BPV (bipower variation), JUMP = max(RV-BPV,0), JUMP_SHARE, QUARTICITY
                                Barndorff-Nielsen & Shephard (2004, 2006)
  realized moments              RSKEW, RKURT  -- Amaya, Christoffersen, Jacobs & Vasquez (2015, JFE)
  semibeta-style asymmetry      UP_VOL_SHARE, DOWN_RUN, SIGN_PERSIST
                                Bollerslev, Patton & Quaedvlieg (2022, JFEC) "Realized semibetas"
  range estimators              PARKINSON, GARMAN_KLASS, ROGERS_SATCHELL
                                Parkinson (1980); Garman & Klass (1980); Rogers & Satchell (1991)
  microstructure / liquidity    VARIANCE_RATIO (Lo & MacKinlay 1988), KYLE_LAMBDA (Kyle 1985 proxy,
                                Hasbrouck 2009), AMIHUD_INTRA (Amihud 2002), ROLL_SPREAD (Roll 1984),
                                VWAP_DEV, CLOSE_LOC
  intraday momentum shape       FH_RET, LH_RET, FH_LH_ALIGN, MID_RET, OVERNIGHT, OVN_INTRA_RATIO,
                                TWAP_TREND  -- Gao, Han, Li & Zhou (2018, JFE) "Market intraday momentum";
                                Lou, Polk & Skouras (2019) overnight/intraday tug of war
  volume / activity shape       VOL_ENTROPY, VOL_HHI, VOL_TREND, TRADE_INTENSITY_SKEW, SMART_MONEY
                                (last-30-min volume share), VOL_PRICE_CORR, DolphinDB-style aggregates
  path shape                    EFFICIENCY_RATIO (Kaufman), MAX_DD_INTRA, TIME_ABOVE_VWAP, N_CROSSINGS

Only daily aggregation of intraday data -- no overlapping windows here; the K-day aggregation and
normalisation are done by timing_transform_study.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from index_timing_framework import MarketPanel, StudyConfig

# names grouped so the report can aggregate by family
FAMILIES = {
    "semivariance": ["RS_UP", "RS_DOWN", "SJV", "RSJ", "SEMI_RATIO"],
    "jump": ["BPV", "JUMP", "JUMP_SHARE", "QUARTICITY", "MAX_ABS_RET"],
    "moments": ["RSKEW", "RKURT", "RVOL"],
    "asymmetry": ["UP_VOL_SHARE", "DOWN_RUN", "SIGN_PERSIST", "UP_DOWN_COUNT"],
    "range": ["PARKINSON", "GARMAN_KLASS", "ROGERS_SATCHELL", "RANGE_RV_RATIO"],
    "micro": ["VARIANCE_RATIO", "KYLE_LAMBDA", "AMIHUD_INTRA", "ROLL_SPREAD", "VWAP_DEV", "CLOSE_LOC"],
    "shape": ["FH_RET", "LH_RET", "FH_LH_ALIGN", "MID_RET", "OVERNIGHT", "OVN_INTRA_RATIO", "TWAP_TREND"],
    "activity": ["VOL_ENTROPY", "VOL_HHI", "VOL_TREND", "SMART_MONEY", "VOL_PRICE_CORR", "DOLVOL_INTRA"],
    "path": ["EFFICIENCY_RATIO", "MAX_DD_INTRA", "TIME_ABOVE_VWAP", "N_CROSSINGS"],
}
NAMES = [n for v in FAMILIES.values() for n in v]


def _safe(a, b):
    return float(a / b) if np.isfinite(a) and np.isfinite(b) and abs(b) > 1e-12 else np.nan


def _day_factors(p: np.ndarray, hi: np.ndarray, lo: np.ndarray, cl: np.ndarray, v: np.ndarray,
                 clock: np.ndarray, day_close: float, prev_close: float) -> dict:
    """p = bar opens (78), cl = bar closes, day_close = official close used by the framework."""
    px = np.r_[p, day_close]                       # 79 price points: bar opens + the close
    r = np.diff(np.log(px))                        # 78 intraday log returns
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 20:
        return {k: np.nan for k in NAMES}
    rv = float(np.sum(r ** 2))
    up, dn = r > 0, r < 0
    rs_up, rs_dn = float(np.sum(r[up] ** 2)), float(np.sum(r[dn] ** 2))
    sd = float(np.std(r, ddof=1))
    # bipower variation -> continuous part; the excess of RV over BPV is the jump part
    bpv = float(np.pi / 2 * np.sum(np.abs(r[:-1]) * np.abs(r[1:])))
    jump = max(rv - bpv, 0.0)
    quart = float(n / 3 * np.sum(r ** 4))
    # realized moments (Amaya et al. scale by n to make them comparable across days)
    rskew = _safe(np.sqrt(n) * np.sum(r ** 3), rv ** 1.5)
    rkurt = _safe(n * np.sum(r ** 4), rv ** 2)
    # sign persistence: AR(1) of the sign sequence = run structure
    s = np.sign(r)
    sign_persist = float(np.mean(s[:-1] * s[1:]))
    runs = np.diff(np.r_[0, np.cumsum(s < 0)])
    down_run = 0; best = 0
    for x in s:
        down_run = down_run + 1 if x < 0 else 0
        best = max(best, down_run)
    # variance ratio: 5-bar variance vs 5x 1-bar variance (<1 = mean reverting, >1 = trending)
    k = 5; m = (n // k) * k
    agg = r[:m].reshape(-1, k).sum(1)
    vr = _safe(np.var(agg, ddof=1), k * np.var(r[:m], ddof=1))
    # range estimators on the day
    H, L, O, C = float(np.nanmax(hi)), float(np.nanmin(lo)), float(p[0]), float(day_close)
    park = _safe((np.log(H / L)) ** 2, 4 * np.log(2))
    gk = 0.5 * np.log(H / L) ** 2 - (2 * np.log(2) - 1) * np.log(C / O) ** 2
    rs_est = np.log(H / C) * np.log(H / O) + np.log(L / C) * np.log(L / O)
    # liquidity / impact
    dv = px[:-1] * v
    total_v, total_dv = float(np.sum(v)), float(np.sum(dv))
    kyle = _safe(np.sum(np.abs(r)), np.sum(np.sqrt(dv[dv > 0]))) if np.any(dv > 0) else np.nan
    amihud = _safe(abs(np.log(C / O)), total_dv / 1e9)
    cov = float(np.mean(r[:-1] * r[1:]))
    roll = 2 * np.sqrt(-cov) if cov < 0 else 0.0
    vwap = _safe(total_dv, total_v)
    # intraday shape (clock labels are bar START times, NY local)
    def seg(a, b):
        m_ = (clock >= a) & (clock < b)
        return m_ if m_.any() else None
    fh = seg("09:30", "10:00"); lh = seg("15:30", "16:00"); mid = seg("11:30", "13:30")
    fh_ret = _safe(cl[fh][-1] - p[fh][0], p[fh][0]) if fh is not None else np.nan
    lh_ret = _safe(day_close - p[lh][0], p[lh][0]) if lh is not None else np.nan
    mid_ret = _safe(cl[mid][-1] - p[mid][0], p[mid][0]) if mid is not None else np.nan
    ovn = _safe(O - prev_close, prev_close)
    intra = _safe(C - O, O)
    # volume shape
    prob = v / total_v if total_v > 0 else np.full(len(v), np.nan)
    pos = prob[prob > 0]
    entropy = float(-np.sum(pos * np.log(pos))) if len(pos) else np.nan
    hhi = float(np.sum(prob ** 2))
    tt = np.arange(len(v))
    vol_trend = _safe(np.cov(tt, v)[0, 1], np.var(tt) * np.mean(v)) if np.std(v) > 0 else np.nan
    smart = _safe(np.sum(v[lh]), total_v) if lh is not None else np.nan
    vpc = float(np.corrcoef(np.abs(r), v[:n])[0, 1]) if np.std(v[:n]) > 0 and np.std(np.abs(r)) > 0 else np.nan
    # path shape
    eff = _safe(abs(np.log(C / O)), np.sum(np.abs(r)))
    cum = np.r_[O, px[1:]]
    dd = float(np.min(cum / np.maximum.accumulate(cum) - 1))
    above = float(np.mean(px[:-1] > vwap)) if np.isfinite(vwap) else np.nan
    cross = float(np.sum(np.diff((px[:-1] > vwap).astype(int)) != 0)) if np.isfinite(vwap) else np.nan
    return {
        "RS_UP": rs_up, "RS_DOWN": rs_dn, "SJV": rs_up - rs_dn, "RSJ": _safe(rs_up - rs_dn, rv),
        "SEMI_RATIO": _safe(rs_dn, rv),
        "BPV": bpv, "JUMP": jump, "JUMP_SHARE": _safe(jump, rv), "QUARTICITY": quart,
        "MAX_ABS_RET": float(np.max(np.abs(r))),
        "RSKEW": rskew, "RKURT": rkurt, "RVOL": sd * np.sqrt(252 * n),
        "UP_VOL_SHARE": _safe(np.sum(v[:n][up]), np.sum(v[:n])), "DOWN_RUN": float(best),
        "SIGN_PERSIST": sign_persist, "UP_DOWN_COUNT": _safe(up.sum() - dn.sum(), n),
        "PARKINSON": park, "GARMAN_KLASS": float(gk), "ROGERS_SATCHELL": float(rs_est),
        "RANGE_RV_RATIO": _safe(np.log(H / L) ** 2, rv),
        "VARIANCE_RATIO": vr, "KYLE_LAMBDA": kyle, "AMIHUD_INTRA": amihud, "ROLL_SPREAD": float(roll),
        "VWAP_DEV": _safe(C - vwap, vwap), "CLOSE_LOC": _safe(C - L, H - L),
        "FH_RET": fh_ret, "LH_RET": lh_ret, "FH_LH_ALIGN": (np.sign(fh_ret) * lh_ret
                                                            if np.isfinite(fh_ret) and np.isfinite(lh_ret) else np.nan),
        "MID_RET": mid_ret, "OVERNIGHT": ovn, "OVN_INTRA_RATIO": _safe(ovn, abs(ovn) + abs(intra)),
        "TWAP_TREND": _safe(np.cov(tt[:n], np.log(px[:n]))[0, 1], np.var(tt[:n])) if n > 2 else np.nan,
        "VOL_ENTROPY": entropy, "VOL_HHI": hhi, "VOL_TREND": vol_trend, "SMART_MONEY": smart,
        "VOL_PRICE_CORR": vpc, "DOLVOL_INTRA": float(np.log1p(total_dv)),
        "EFFICIENCY_RATIO": eff, "MAX_DD_INTRA": dd, "TIME_ABOVE_VWAP": above, "N_CROSSINGS": cross,
    }


def build(path: str, panel: MarketPanel, cfg: StudyConfig) -> pd.DataFrame:
    raw = pd.read_csv(path, parse_dates=["datetime"])
    raw["day"] = raw.datetime.dt.strftime("%Y-%m-%d")
    raw["clock"] = raw.datetime.dt.strftime("%H:%M")
    close_map = dict(zip(panel.dates, panel.close))
    prev = pd.Series(panel.close, index=panel.dates).shift(1)
    out = {}
    for day, g in raw.groupby("day", sort=True):
        if day in cfg.early_closes or day not in close_map:
            continue
        g = g.sort_values("datetime")
        out[day] = _day_factors(g.open.to_numpy(float), g.high.to_numpy(float), g.low.to_numpy(float),
                                g.close.to_numpy(float), g.volume.to_numpy(float), g.clock.to_numpy(),
                                float(close_map[day]), float(prev.loc[day]))
    return pd.DataFrame(out).T.reindex(panel.dates).replace([np.inf, -np.inf], np.nan)


if __name__ == "__main__":
    from pathlib import Path
    from index_timing_framework import load_market
    from run_full_extended import config_from
    cfg = config_from(Path("data"))
    for sym in ("SPY", "QQQ"):
        panel = load_market(f"data/{sym}_5min.csv", cfg)
        F = build(f"data/{sym}_5min.csv", panel, cfg)
        F.to_pickle(f"outputs/full/cache/{sym}_intraday_lib.pkl")
        print(sym, F.shape, "coverage", round(float(F.notna().mean().mean()), 3))
        print(F.notna().mean().sort_values().head(5).round(3).to_dict())
