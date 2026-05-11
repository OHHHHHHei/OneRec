#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_titles(item_json: Path) -> dict[int, str]:
    if not item_json.exists():
        return {}
    obj = json.loads(item_json.read_text(encoding="utf-8"))
    titles: dict[int, str] = {}
    for key, value in obj.items():
        try:
            item_id = int(key)
        except ValueError:
            continue
        if isinstance(value, dict):
            titles[item_id] = str(value.get("title", f"Item_{item_id}")).replace("\n", " ")
    return titles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Amazon18 .inter train split to the CSV schema used by tokenizer graph builders."
    )
    parser.add_argument("--train-inter", required=True, type=Path)
    parser.add_argument("--item-json", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    titles = load_titles(args.item_json)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    with args.train_inter.open("r", encoding="utf-8") as fin, args.output_csv.open(
        "w", encoding="utf-8", newline=""
    ) as fout:
        reader = csv.DictReader(fin, delimiter="\t")
        writer = csv.DictWriter(
            fout,
            fieldnames=[
                "user_id",
                "history_item_id",
                "item_id",
                "history_item_title",
                "item_title",
            ],
        )
        writer.writeheader()
        rows = 0
        for row in reader:
            user_id = int(row["user_id:token"])
            history_ids = [int(v) for v in str(row["item_id_list:token_seq"]).split() if v]
            item_id = int(row["item_id:token"])
            writer.writerow(
                {
                    "user_id": user_id,
                    "history_item_id": str(history_ids),
                    "item_id": item_id,
                    "history_item_title": str([titles.get(i, f"Item_{i}") for i in history_ids]),
                    "item_title": titles.get(item_id, f"Item_{item_id}"),
                }
            )
            rows += 1

    print(f"[OK] wrote {rows} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
