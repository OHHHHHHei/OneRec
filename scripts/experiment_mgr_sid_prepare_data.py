#!/usr/bin/env python3
"""Prepare experimental converted datasets under data_experiment/.

This script mirrors the existing OneRec CSV layout while swapping the
item->SID mapping to an experimental index.json. It is intentionally isolated
from the default `convert` stage because the current repository no longer keeps
the original `*.inter` split files that `convert` expects.
"""

from __future__ import annotations

import argparse
import ast
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass
class VariantSpec:
    name: str
    index_json: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare experimental OneRec CSV data with alternative SID indices."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("data/Amazon"),
        help="Existing converted data root containing train/valid/test/index/info.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data_experiment/Amazon"),
        help="Target root. Each variant becomes a self-contained subtree here.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="Industrial_and_Scientific",
        help="Dataset/category stem used in filenames.",
    )
    parser.add_argument(
        "--variant",
        action="append",
        default=[],
        help="Variant in the form name=/path/to/index.json . Repeat for multiple variants.",
    )
    return parser.parse_args()


def parse_variant_specs(raw_specs: Iterable[str]) -> list[VariantSpec]:
    specs: list[VariantSpec] = []
    for raw in raw_specs:
        if "=" not in raw:
            raise ValueError(f"Invalid --variant value: {raw!r}. Expected name=/path/to/index.json")
        name, path_str = raw.split("=", 1)
        index_json = Path(path_str).expanduser().resolve()
        if not index_json.exists():
            raise FileNotFoundError(f"Variant index json not found: {index_json}")
        specs.append(VariantSpec(name=name.strip(), index_json=index_json))
    if not specs:
        raise ValueError("At least one --variant must be provided.")
    return specs


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def semantic_tokens_to_id(tokens: list[str]) -> str:
    return "".join(tokens)


def parse_id_list(raw: object) -> list[int]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [int(x) for x in raw]
    text = str(raw).strip()
    if not text:
        return []
    value = ast.literal_eval(text)
    if isinstance(value, list):
        return [int(x) for x in value]
    raise ValueError(f"Expected list-like history_item_id, got: {raw!r}")


def render_sid_history(item_ids: list[int], index_map: dict[str, list[str]]) -> str:
    history_sids = []
    for item_id in item_ids:
        tokens = index_map.get(str(item_id))
        if tokens is None:
            raise KeyError(f"Missing item_id={item_id} in experimental index map")
        history_sids.append(semantic_tokens_to_id(tokens))
    return str(history_sids)


def render_item_sid(item_id: int, index_map: dict[str, list[str]]) -> str:
    tokens = index_map.get(str(item_id))
    if tokens is None:
        raise KeyError(f"Missing item_id={item_id} in experimental index map")
    return semantic_tokens_to_id(tokens)


def build_info_file(item_meta: dict[str, dict], index_map: dict[str, list[str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for item_id, item_data in item_meta.items():
            tokens = index_map.get(str(item_id))
            if tokens is None:
                continue
            sid = semantic_tokens_to_id(tokens)
            title = item_data.get("title", f"Item_{item_id}")
            f.write(f"{sid}\t{title}\t{item_id}\n")


def convert_split_csv(split_csv: Path, index_map: dict[str, list[str]], output_csv: Path) -> int:
    df = pd.read_csv(split_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    history_sid_col = []
    item_sid_col = []
    for row in df.itertuples(index=False):
        history_item_ids = parse_id_list(getattr(row, "history_item_id"))
        target_item_id = int(getattr(row, "item_id"))
        history_sid_col.append(render_sid_history(history_item_ids, index_map))
        item_sid_col.append(render_item_sid(target_item_id, index_map))

    df = df.copy()
    df["history_item_sid"] = history_sid_col
    df["item_sid"] = item_sid_col
    df.to_csv(output_csv, index=False)
    return len(df)


def copy_optional(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def main() -> None:
    args = parse_args()
    specs = parse_variant_specs(args.variant)

    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    dataset = args.dataset
    csv_name = f"{dataset}_5_2016-10-2018-11.csv"
    info_name = f"{dataset}_5_2016-10-2018-11.txt"

    item_meta_path = source_root / "index" / f"{dataset}.item.json"
    if not item_meta_path.exists():
        raise FileNotFoundError(f"Item meta file not found: {item_meta_path}")
    item_meta = load_json(item_meta_path)

    split_paths = {
        split: source_root / split / csv_name
        for split in ("train", "valid", "test")
    }
    for split, path in split_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing source {split} csv: {path}")

    manifest: dict[str, dict] = {
        "dataset": dataset,
        "source_root": str(source_root),
        "output_root": str(output_root),
        "variants": {},
    }

    for spec in specs:
        variant_root = output_root / spec.name
        index_map = load_json(spec.index_json)
        if len(index_map) != len(item_meta):
            raise ValueError(
                f"Variant {spec.name} has {len(index_map)} indexed items but item meta has {len(item_meta)}"
            )

        for subdir in ("index", "info", "train", "valid", "test"):
            (variant_root / subdir).mkdir(parents=True, exist_ok=True)

        target_index_path = variant_root / "index" / f"{dataset}.index.json"
        shutil.copy2(spec.index_json, target_index_path)
        shutil.copy2(item_meta_path, variant_root / "index" / f"{dataset}.item.json")

        summary_src = spec.index_json.with_suffix(".summary.json")
        summary_dst = variant_root / "index" / f"{dataset}.summary.json"
        copy_optional(summary_src, summary_dst)

        info_path = variant_root / "info" / info_name
        build_info_file(item_meta, index_map, info_path)

        split_counts: dict[str, int] = {}
        for split, src_csv in split_paths.items():
            dst_csv = variant_root / split / csv_name
            split_counts[split] = convert_split_csv(src_csv, index_map, dst_csv)

        manifest["variants"][spec.name] = {
            "index_json": str(spec.index_json),
            "variant_root": str(variant_root),
            "num_items": len(index_map),
            "split_counts": split_counts,
        }
        print(
            f"[done] {spec.name}: items={len(index_map)} "
            + " ".join(f"{split}={count}" for split, count in split_counts.items())
        )

    manifest_path = output_root / f"{dataset}.manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"[manifest] {manifest_path}")


if __name__ == "__main__":
    main()
