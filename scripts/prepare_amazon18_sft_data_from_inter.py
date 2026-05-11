#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def semantic_tokens_to_id(tokens: list[str]) -> str:
    return "".join(tokens)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a self-contained OneRec SFT/eval data_experiment subtree from Amazon18 .inter files."
    )
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--index-json", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--split-stem", default=None)
    return parser.parse_args()


def build_info_file(items: dict[str, dict], index_map: dict[str, list[str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fout:
        for item_id, item in items.items():
            tokens = index_map.get(str(item_id))
            if tokens is None:
                continue
            title = str(item.get("title", f"Item_{item_id}")).replace("\n", " ")
            fout.write(f"{semantic_tokens_to_id(tokens)}\t{title}\t{item_id}\n")


def convert_split(
    split_inter: Path,
    output_csv: Path,
    items: dict[str, dict],
    index_map: dict[str, list[str]],
) -> int:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with split_inter.open("r", encoding="utf-8") as fin, output_csv.open(
        "w", encoding="utf-8", newline=""
    ) as fout:
        reader = csv.DictReader(fin, delimiter="\t")
        writer = csv.DictWriter(
            fout,
            fieldnames=[
                "user_id",
                "history_item_title",
                "item_title",
                "history_item_id",
                "item_id",
                "history_item_sid",
                "item_sid",
            ],
        )
        writer.writeheader()
        for row in reader:
            user_id = int(row["user_id:token"])
            history_item_ids = [int(x) for x in row["item_id_list:token_seq"].split() if x]
            target_item_id = int(row["item_id:token"])
            if str(target_item_id) not in index_map:
                continue
            history_sids = []
            for item_id in history_item_ids:
                tokens = index_map.get(str(item_id))
                if tokens is not None:
                    history_sids.append(semantic_tokens_to_id(tokens))
            history_titles = [
                str(items.get(str(item_id), {}).get("title", f"Item_{item_id}")).replace("\n", " ")
                for item_id in history_item_ids
            ]
            target_title = str(
                items.get(str(target_item_id), {}).get("title", f"Item_{target_item_id}")
            ).replace("\n", " ")
            writer.writerow(
                {
                    "user_id": f"A{user_id}",
                    "history_item_title": str(history_titles),
                    "item_title": target_title,
                    "history_item_id": str(history_item_ids),
                    "item_id": target_item_id,
                    "history_item_sid": str(history_sids),
                    "item_sid": semantic_tokens_to_id(index_map[str(target_item_id)]),
                }
            )
            count += 1
    return count


def main() -> None:
    args = parse_args()
    dataset = args.dataset
    split_stem = args.split_stem or f"{dataset}_5_2016-10-2018-11"
    source_root = args.source_root
    output_root = args.output_root

    items = load_json(source_root / f"{dataset}.item.json")
    index_map = load_json(args.index_json)
    if len(items) != len(index_map):
        raise ValueError(f"item count mismatch: items={len(items)} index={len(index_map)}")

    for subdir in ["index", "info", "train", "valid", "test"]:
        (output_root / subdir).mkdir(parents=True, exist_ok=True)

    (output_root / "index" / f"{dataset}.index.json").write_text(
        json.dumps(index_map, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_root / "index" / f"{dataset}.item.json").write_text(
        json.dumps(items, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    build_info_file(items, index_map, output_root / "info" / f"{split_stem}.txt")

    counts: dict[str, int] = {}
    for split in ["train", "valid", "test"]:
        counts[split] = convert_split(
            source_root / f"{dataset}.{split}.inter",
            output_root / split / f"{split_stem}.csv",
            items,
            index_map,
        )

    manifest = {
        "dataset": dataset,
        "split_stem": split_stem,
        "source_root": str(source_root),
        "index_json": str(args.index_json),
        "output_root": str(output_root),
        "split_counts": counts,
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
