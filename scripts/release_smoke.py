#!/usr/bin/env python3
"""Validate the public release layout without loading data or model weights."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


REQUIRED_SOURCES = (
    "train.py",
    "generation.py",
    "models/PEGM.py",
    "models/NAELayer.py",
    "models/PBALayer.py",
    "models/SNELayer.py",
    "models/criterions/PocketEnhancedLoss.py",
    "utils/dataset.py",
    "utils/ckpt.py",
)
PRIVATE_PATH = re.compile(r"/(?:data[0-9]+|home/[^/]+|srv)/")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    missing = [name for name in REQUIRED_SOURCES if not (root / name).is_file()]
    if missing:
        raise SystemExit(f"missing release sources: {', '.join(missing)}")

    with args.config.open(encoding="utf-8") as handle:
        cfg = json.load(handle)
    for key in ("train", "model"):
        if key not in cfg:
            raise SystemExit(f"config misses top-level key: {key}")
    for key in ("data_path", "valid_data_path", "max_tokens"):
        if key not in cfg["train"].get("data", {}):
            raise SystemExit(f"config misses train.data.{key}")

    offenders = []
    for path in [args.config, *(root / name for name in REQUIRED_SOURCES)]:
        text = path.read_text(encoding="utf-8")
        if PRIVATE_PATH.search(text):
            offenders.append(str(path.relative_to(root)) if path.is_relative_to(root) else str(path))
    if offenders:
        raise SystemExit("private absolute path found in: " + ", ".join(offenders))

    print("release smoke: PASS")
    print(f"config: {args.config}")
    print(f"required sources: {len(REQUIRED_SOURCES)}")


if __name__ == "__main__":
    main()
