"""Full-sample driver for extended_timing_research (2019-01 .. 2026-10 moomoo data).

Same factor definitions and metrics as extended_timing_research.run; the only changes are
(1) the early-close calendar is read from data/early_closes.txt instead of the 2024-25
defaults, and (2) results are also flattened to a compact CSV for analysis.

    python run_full_extended.py --data data --output-dir outputs/full
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from index_timing_framework import StudyConfig
import extended_timing_research as E


def config_from(data: Path) -> StudyConfig:
    early = tuple(x.strip() for x in (data / "early_closes.txt").read_text().split() if x.strip())
    return StudyConfig(early_closes=early)


def flatten(rows: list[dict]) -> pd.DataFrame:
    keep = ["symbol", "factor", "source", "aggregation", "version", "execution", "horizon", "scope", "n_ic",
            "pearson_ic", "rank_ic", "hac_slope_t", "mean_return", "bucket_high_low", "bucket_up_steps"]
    df = pd.DataFrame([{k: r.get(k) for k in keep} for r in rows])
    bm = pd.DataFrame([r["bucket_means"] for r in rows], columns=[f"b{i}" for i in range(1, 6)])
    yr = pd.DataFrame([{f"ic_{y}": v["pearson_ic"] for y, v in r["yearly"].items()} for r in rows])
    return pd.concat([df, bm, yr], axis=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data"); ap.add_argument("--output-dir", default="outputs/full")
    a = ap.parse_args()
    data, out = Path(a.data), Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    cfg = config_from(data)
    rows, _, _ = E.run(str(data / "SPY_5min.csv"), str(data / "QQQ_5min.csv"), cfg)
    (out / "extended_factor_metrics.json").write_text(json.dumps(rows, ensure_ascii=False, allow_nan=False))
    flatten(rows).to_csv(out / "extended_factor_metrics.csv", index=False)
    print("rows", len(rows), out)
