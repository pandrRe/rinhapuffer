#!/usr/bin/env python3
"""Copy resources from the rinha-de-backend-2026 submodule to ./resources/
and gunzip references.json.gz."""

from __future__ import annotations

import gzip
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "rinha-de-backend-2026" / "resources"
DST = ROOT / "resources"


def main() -> None:
    if not SRC.is_dir():
        raise SystemExit(f"source directory not found: {SRC}")

    DST.mkdir(parents=True, exist_ok=True)

    for entry in SRC.iterdir():
        if not entry.is_file():
            continue

        if entry.suffix == ".gz":
            target = DST / entry.stem
            with gzip.open(entry, "rb") as fin, target.open("wb") as fout:
                shutil.copyfileobj(fin, fout)
            print(f"unzipped {entry.name} -> {target.relative_to(ROOT)}")
        else:
            target = DST / entry.name
            shutil.copy2(entry, target)
            print(f"copied   {entry.name} -> {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
