#!/usr/bin/env python3
"""Validate split experiment registry CSV files."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from split_experiment_results_registry import (
    DEFAULT_OUTPUT_DIR,
    RL_SCHEMA,
    SCOREBOARD_SCHEMA,
    SFT_SCHEMA,
    TOKENIZER_SCHEMA,
)


REGISTRY_SCHEMAS = {
    "tokenizer_registry.csv": TOKENIZER_SCHEMA,
    "sft_registry.csv": SFT_SCHEMA,
    "rl_registry.csv": RL_SCHEMA,
    "downstream_scoreboard.csv": SCOREBOARD_SCHEMA,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate split experiment registry CSV files.")
    parser.add_argument("--registry-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def validate_file(path: Path, expected_header: list[str]) -> tuple[int, list[str]]:
    errors: list[str] = []
    if not path.exists():
        return 0, [f"missing file: {path}"]

    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    if not rows:
        return 0, [f"empty file: {path}"]

    header = rows[0]
    if header != expected_header:
        errors.append(
            f"header mismatch in {path.name}: expected {len(expected_header)} cols, got {len(header)}"
        )

    bad_width = [(idx + 1, len(row)) for idx, row in enumerate(rows) if len(row) != len(header)]
    if bad_width:
        errors.append(f"row width mismatch in {path.name}: {bad_width[:5]}")

    if len(rows) > 1 and "record_id" in header:
        record_idx = header.index("record_id")
        record_ids = [row[record_idx] for row in rows[1:] if len(row) > record_idx]
        missing = [idx + 2 for idx, value in enumerate(record_ids) if not value or value == "-"]
        if missing:
            errors.append(f"missing record_id in {path.name}: rows {missing[:5]}")
        duplicates = [value for value, count in Counter(record_ids).items() if count > 1 and value != "-"]
        if duplicates:
            errors.append(f"duplicate record_id in {path.name}: {duplicates[:5]}")

    return max(len(rows) - 1, 0), errors


def main() -> None:
    args = parse_args()
    all_errors: list[str] = []
    for filename, schema in REGISTRY_SCHEMAS.items():
        expected_header = [name for name, _ in schema]
        row_count, errors = validate_file(args.registry_dir / filename, expected_header)
        print(f"[check] {filename}: rows={row_count} cols={len(expected_header)} errors={len(errors)}")
        all_errors.extend(errors)

    if all_errors:
        print("\n".join(f"[error] {error}" for error in all_errors))
        raise SystemExit(1)

    print("[ok] split experiment registry is valid")


if __name__ == "__main__":
    main()
