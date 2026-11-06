"""Fetch (optional) and load the raw datasets. Run from the repo root:  python examples/01_data.py [--update]"""
import sys
sys.path.insert(0, ".")
from quantkit.data import fetch, load, manifest
from quantkit import paths

if "--update" in sys.argv:                       # downloads QQQ/SPY/MTUM/NDX/SPX, VIX family, French, DIX/GEX; ~5 s
    print(fetch.update_all())

qqq = load.load_daily("QQQ"); print("QQQ daily", qqq.index.min().date(), "->", qqq.index.max().date(), len(qqq), "rows")
print("VIX last", load.load_cboe("VIX").iloc[-1], "| DIX/GEX rows", len(load.load_squeeze()), "| momentum factor rows", len(load.load_french()))
res = manifest.verify_manifest(paths.REPO / "data" / "MANIFEST_sandbox_2026-10.md5")
print("manifest:", {k: v for k, v in res.items() if v != "ok"} or "all files match")
