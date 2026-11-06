"""MD5 manifests for raw data (same line format as data/MANIFEST_*.md5: '<md5>  <path relative to data root>')."""
from __future__ import annotations

import hashlib
from pathlib import Path

from quantkit import paths


def md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def write_manifest(out: Path, patterns=("raw/**/*.parquet", "raw/**/*.csv")) -> int:
    root = paths.data_root(); files = sorted({p for pat in patterns for p in root.glob(pat) if p.is_file()})
    out.write_text("".join(f"{md5(p)}  {p.relative_to(root)}\n" for p in files)); return len(files)


def verify_manifest(manifest: Path) -> dict[str, str]:
    """Returns {relpath: 'ok' | 'MISMATCH' | 'MISSING'}. Parquet/CSV rebuilt from public sources will legitimately
    mismatch after an update; the 1-minute bars are the file that must match the server copy."""
    root = paths.data_root(); res = {}
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        h, rel = line.split(None, 1); rel = rel.strip(); p = root / rel
        res[rel] = "MISSING" if not p.exists() else ("ok" if md5(p) == h else "MISMATCH")
    return res
