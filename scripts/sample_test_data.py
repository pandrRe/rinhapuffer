#!/usr/bin/env python3
"""Random-sample entries from the rinha test dataset into a smaller fixture.

Reads the full test-data.json produced by the rinha-de-backend-2026 submodule
and writes a deterministic, smaller sample with the same top-level shape so
downstream parsers can consume it identically."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "rinha-de-backend-2026" / "test" / "test-data.json"
DEFAULT_OUTPUT = ROOT / "resources" / "test-data-sample.json"
DEFAULT_N = 200
DEFAULT_SEED = 42


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Random-sample entries from the rinha test dataset."
    )
    parser.add_argument(
        "-n",
        type=int,
        default=DEFAULT_N,
        help=f"number of entries to sample (default {DEFAULT_N})",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="output path (default resources/test-data-sample.json)",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"random seed (default {DEFAULT_SEED})",
    )
    args = parser.parse_args()

    output_path = args.output if args.output.is_absolute() else ROOT / args.output

    if not DEFAULT_INPUT.is_file():
        raise SystemExit(f"input file not found: {DEFAULT_INPUT}")

    with DEFAULT_INPUT.open("r", encoding="utf-8") as fin:
        data = json.load(fin)

    entries = data["entries"]
    source_total = data.get("stats", {}).get("total", len(entries))
    k = min(args.n, len(entries))

    random.seed(args.seed)
    sampled = random.sample(entries, k=k)

    payload = {
        "references_checksum_sha256": data["references_checksum_sha256"],
        "stats": {
            "sampled_total": k,
            "seed": args.seed,
            "source_total": source_total,
        },
        "entries": sampled,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fout:
        json.dump(payload, fout, indent=2)

    print(f"sampled {k}/{source_total} entries -> {output_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
