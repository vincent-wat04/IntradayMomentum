"""Single place for filesystem locations."""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
STUDIES = REPO / "studies"


def data_root() -> Path:
    """$ETFT_DATA, default <repo>/data. Evaluated at call time so tests can override the env var."""
    return Path(os.environ.get("ETFT_DATA", REPO / "data"))


def raw(*parts: str) -> Path:
    return data_root().joinpath("raw", *parts)


def mdrec_dir() -> Path:
    return data_root() / "mdrec" / "data"


def use_legacy_studies() -> None:
    """Make the legacy research modules (etft.*, audit.*, ...) importable. Only the adapters call this."""
    import sys
    os.environ.setdefault("ETFT_DATA", str(data_root()))
    if str(STUDIES) not in sys.path:
        sys.path.insert(0, str(STUDIES))
