#!/usr/bin/env python3
"""Summarize position-level sequence recovery from EnzyPGM generation JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


def iter_prediction_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
    elif path.is_dir():
        yield from sorted(path.glob("*.json"))
    else:
        raise FileNotFoundError(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    compared = correct = skipped = length_mismatches = 0
    files = list(iter_prediction_files(args.predictions))
    for path in files:
        with path.open(encoding="utf-8") as handle:
            record = json.load(handle)
        truth = record.get("ground_truth", {}).get("protein_seq")
        prediction = record.get("prediction", {}).get("protein_seq")
        if not isinstance(truth, str) or not isinstance(prediction, str):
            skipped += 1
            continue
        if len(truth) != len(prediction):
            length_mismatches += 1
        overlap = min(len(truth), len(prediction))
        correct += sum(a == b for a, b in zip(truth[:overlap], prediction[:overlap]))
        compared += overlap

    summary = {
        "files_seen": len(files),
        "records_scored": len(files) - skipped,
        "records_skipped": skipped,
        "length_mismatches": length_mismatches,
        "positions_compared": compared,
        "positions_correct": correct,
        "position_recovery": (correct / compared) if compared else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
