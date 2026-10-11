"""Macro / cross-asset / sentiment factors from the etf_timing project, evaluated with THIS
repository's metrics (same execution conventions, K-aggregation, past-K-return orthogonalisation,
IC / RankIC / HAC t / trailing quintiles / yearly IC).

Source panels: ../etf_timing/data/processed/panel_{SPY,QQQ}.parquet, built by
etft.features.cache_panels.  Every column there is point-in-time (value at date t is known at the
close of t; FRED t+1, CFTC Friday release, ALFRED real-time vintages for revised macro series,
NFCI only after its 2011 launch).  The truncation-invariance tests in etf_timing/tests prove it.

Only market-level families are imported (price/volume factors of the ETF itself already exist in
extended_timing_research.py and are not duplicated).

    python macro_factor_study.py --data data --etf-timing ../etf_timing --output-dir outputs/full
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

import extended_timing_research as E
from index_timing_framework import load_market, orthogonalize_past_k
from run_full_extended import config_from, flatten

MACRO_FAMILIES = ("macro", "macro_surprise", "rates", "credit", "fx", "commodity", "volatility", "correlation",
                  "options", "options_flow", "positioning", "sentiment", "nlp", "valuation", "fundamental",
                  "regime", "global", "risk_appetite", "breadth", "event")
WINDOWS = (1, 5, 20, 60)
VERSIONS = ("raw", "orthog")
MIN_COVERAGE = 0.6       # share of 2019-2026 sessions with a value


def load_macro_panel(etf_timing: Path, symbol: str, dates: pd.Index):
    sys.path.insert(0, str(etf_timing.resolve()))
    from etft.factors import REGISTRY  # noqa: E402
    X = pd.read_parquet(etf_timing / "data" / "processed" / f"panel_{symbol}.parquet")
    X.index = X.index.strftime("%Y-%m-%d")
    cols = [c for c in X.columns if REGISTRY[c].family in MACRO_FAMILIES]
    X = X[cols].reindex(dates)
    cov = X.notna().mean()
    keep = cov[cov >= MIN_COVERAGE].index.tolist()
    meta = {c: {"family": REGISTRY[c].family, "prior_sign": REGISTRY[c].sign, "desc": REGISTRY[c].desc,
                "refs": REGISTRY[c].refs, "coverage": float(cov[c])} for c in cols}
    return X[keep], meta, sorted(set(cols) - set(keep))


_G = {}


def _init(data, etf_timing):
    cfg = config_from(Path(data))
    _G["cfg"] = cfg
    for sym in ("SPY", "QQQ"):
        panel = load_market(str(Path(data) / f"{sym}_5min.csv"), cfg)
        X, _, _ = load_macro_panel(Path(etf_timing), sym, panel.dates)
        _G[sym] = (panel, X)


def _work(job):
    sym, name = job
    panel, X = _G[sym]
    cfg = _G["cfg"]
    rows = []
    for k in WINDOWS:
        raw = X[name].rolling(k, min_periods=k).mean()
        resid = orthogonalize_past_k(raw, panel, k, cfg)
        for v, s in (("raw", raw), ("orthog", resid)):
            E.FACTOR_CATALOG.setdefault(name, ("macro", "mean", ""))
            for r in E.metrics(sym, panel, name, k, v, s, cfg):
                r["source"] = "etf_timing_macro"
                rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data"); ap.add_argument("--etf-timing", default="../etf_timing")
    ap.add_argument("--output-dir", default="outputs/full"); ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    cfg = config_from(Path(a.data))
    panel = load_market(str(Path(a.data) / "SPY_5min.csv"), cfg)
    X, meta, dropped = load_macro_panel(Path(a.etf_timing), "SPY", panel.dates)
    (out / "macro_factor_catalog.json").write_text(json.dumps({"used": list(X.columns), "dropped_low_coverage": dropped,
                                                                "meta": meta}, indent=1, default=str))
    jobs = [(s, c) for s in ("SPY", "QQQ") for c in X.columns]
    print("macro factors", len(X.columns), "dropped", dropped, "jobs", len(jobs), flush=True)
    with Pool(a.workers, initializer=_init, initargs=(a.data, a.etf_timing)) as p:
        rows = [r for part in p.imap_unordered(_work, jobs) for r in part]
    for r in rows:
        r["family"] = meta[r["factor"]]["family"]
    df = flatten(rows)
    df["family"] = [r["family"] for r in rows]
    df.to_csv(out / "macro_factor_metrics.csv", index=False)
    print("rows", len(rows))


if __name__ == "__main__":
    main()
