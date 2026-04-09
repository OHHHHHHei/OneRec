#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare final SID indices with a focus on local same-l2 ambiguity."
    )
    parser.add_argument("--baseline-index", required=True)
    parser.add_argument("--compare-index", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--item-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


def load_index(path: Path) -> dict[int, tuple[str, str, str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: dict[int, tuple[str, str, str]] = {}
    for item_id, tokens in raw.items():
        if not isinstance(tokens, list) or len(tokens) < 3:
            out[int(item_id)] = ("", "", "")
            continue
        out[int(item_id)] = (str(tokens[0]), str(tokens[1]), str(tokens[2]))
    return out


def load_titles(path: Path) -> dict[int, str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {int(k): str(v.get("title", f"Item_{k}")) for k, v in raw.items()}


def entropy_from_counter(counter: Counter[str]) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    import math

    value = 0.0
    for count in counter.values():
        p = count / total
        value -= p * math.log2(p)
    return value


def build_index_stats(index_map: dict[int, tuple[str, str, str]]) -> dict:
    l2_to_items: dict[tuple[str, str], list[int]] = defaultdict(list)
    l2_to_leaf_counter: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    item_stats: dict[int, dict[str, float | int | str]] = {}

    for item_id, (a, b, c) in index_map.items():
        l2 = (a, b)
        l2_to_items[l2].append(item_id)
        l2_to_leaf_counter[l2][c] += 1

    l2_item_sizes: list[int] = []
    l2_leaf_sizes: list[int] = []
    l3_entropies: list[tuple[float, int]] = []
    crowded_prefixes: list[dict[str, object]] = []

    for l2, items in l2_to_items.items():
        leaf_counter = l2_to_leaf_counter[l2]
        item_count = len(items)
        leaf_count = len(leaf_counter)
        entropy = entropy_from_counter(leaf_counter)
        l2_item_sizes.extend([item_count] * item_count)
        l2_leaf_sizes.extend([leaf_count] * item_count)
        l3_entropies.append((entropy, item_count))

        crowded_prefixes.append(
            {
                "prefix": "".join(l2),
                "item_count": item_count,
                "leaf_count": leaf_count,
                "entropy_bits": entropy,
            }
        )

        for item_id in items:
            item_stats[item_id] = {
                "l1": l2[0],
                "l2_prefix": "".join(l2),
                "l2_item_count": item_count,
                "l2_leaf_count": leaf_count,
                "l3_entropy_bits": entropy,
                "sid": "".join(index_map[item_id]),
            }

    weighted_l3_entropy = (
        sum(ent * w for ent, w in l3_entropies) / sum(w for _, w in l3_entropies)
        if l3_entropies
        else 0.0
    )

    return {
        "item_count": len(index_map),
        "l2_prefix_count": len(l2_to_items),
        "item_weighted_mean_l2_item_count": mean(l2_item_sizes) if l2_item_sizes else 0.0,
        "item_weighted_median_l2_item_count": median(l2_item_sizes) if l2_item_sizes else 0.0,
        "item_weighted_mean_l2_leaf_count": mean(l2_leaf_sizes) if l2_leaf_sizes else 0.0,
        "item_weighted_median_l2_leaf_count": median(l2_leaf_sizes) if l2_leaf_sizes else 0.0,
        "item_fraction_multileaf_l2": (sum(v > 1 for v in l2_leaf_sizes) / len(l2_leaf_sizes)) if l2_leaf_sizes else 0.0,
        "item_fraction_multileaf_l2_ge4": (sum(v >= 4 for v in l2_leaf_sizes) / len(l2_leaf_sizes)) if l2_leaf_sizes else 0.0,
        "item_fraction_multileaf_l2_ge8": (sum(v >= 8 for v in l2_leaf_sizes) / len(l2_leaf_sizes)) if l2_leaf_sizes else 0.0,
        "weighted_l3_entropy_given_l2_bits": weighted_l3_entropy,
        "top_crowded_l2_prefixes": sorted(
            crowded_prefixes,
            key=lambda row: (-int(row["leaf_count"]), -int(row["item_count"]), str(row["prefix"])),
        )[:15],
        "item_stats": item_stats,
    }


def build_test_weighted_stats(item_stats: dict[int, dict[str, float | int | str]], test_df: pd.DataFrame) -> dict:
    l2_item_counts: list[int] = []
    l2_leaf_counts: list[int] = []
    entropies: list[float] = []
    missing = 0

    for item_id in test_df["item_id"].tolist():
        item_id = int(item_id)
        stats = item_stats.get(item_id)
        if stats is None:
            missing += 1
            continue
        l2_item_counts.append(int(stats["l2_item_count"]))
        l2_leaf_counts.append(int(stats["l2_leaf_count"]))
        entropies.append(float(stats["l3_entropy_bits"]))

    total = len(l2_leaf_counts)
    return {
        "example_count": len(test_df),
        "covered_count": total,
        "missing_count": missing,
        "target_weighted_mean_l2_item_count": mean(l2_item_counts) if l2_item_counts else 0.0,
        "target_weighted_median_l2_item_count": median(l2_item_counts) if l2_item_counts else 0.0,
        "target_weighted_mean_l2_leaf_count": mean(l2_leaf_counts) if l2_leaf_counts else 0.0,
        "target_weighted_median_l2_leaf_count": median(l2_leaf_counts) if l2_leaf_counts else 0.0,
        "target_fraction_multileaf_l2": (sum(v > 1 for v in l2_leaf_counts) / total) if total else 0.0,
        "target_fraction_multileaf_l2_ge4": (sum(v >= 4 for v in l2_leaf_counts) / total) if total else 0.0,
        "target_fraction_multileaf_l2_ge8": (sum(v >= 8 for v in l2_leaf_counts) / total) if total else 0.0,
        "target_weighted_mean_l3_entropy_bits": mean(entropies) if entropies else 0.0,
    }


def build_pairwise_comparison(
    baseline_stats: dict[int, dict[str, float | int | str]],
    compare_stats: dict[int, dict[str, float | int | str]],
    test_df: pd.DataFrame,
    titles: dict[int, str],
) -> dict:
    common_items = sorted(set(baseline_stats) & set(compare_stats))
    changed_items = 0
    reduced_leaf_count = 0
    increased_leaf_count = 0
    same_leaf_count = 0
    moved_out_of_same_l2 = 0
    moved_into_same_l2 = 0
    moved_out_of_ge4 = 0
    moved_into_ge4 = 0
    deltas_all: list[int] = []
    deltas_changed: list[int] = []
    best_examples: list[dict[str, object]] = []

    for item_id in common_items:
        base = baseline_stats[item_id]
        comp = compare_stats[item_id]
        base_leaf = int(base["l2_leaf_count"])
        comp_leaf = int(comp["l2_leaf_count"])
        delta = comp_leaf - base_leaf
        deltas_all.append(delta)

        if str(base["sid"]) != str(comp["sid"]):
            changed_items += 1
            deltas_changed.append(delta)

        if comp_leaf < base_leaf:
            reduced_leaf_count += 1
        elif comp_leaf > base_leaf:
            increased_leaf_count += 1
        else:
            same_leaf_count += 1

        if base_leaf > 1 and comp_leaf == 1:
            moved_out_of_same_l2 += 1
        if base_leaf == 1 and comp_leaf > 1:
            moved_into_same_l2 += 1
        if base_leaf >= 4 and comp_leaf < 4:
            moved_out_of_ge4 += 1
        if base_leaf < 4 and comp_leaf >= 4:
            moved_into_ge4 += 1

        if comp_leaf < base_leaf:
            best_examples.append(
                {
                    "item_id": item_id,
                    "title": titles.get(item_id, f"Item_{item_id}"),
                    "baseline_sid": str(base["sid"]),
                    "hierarchy_sid": str(comp["sid"]),
                    "baseline_l2_prefix": str(base["l2_prefix"]),
                    "hierarchy_l2_prefix": str(comp["l2_prefix"]),
                    "baseline_l2_leaf_count": base_leaf,
                    "hierarchy_l2_leaf_count": comp_leaf,
                    "delta_l2_leaf_count": delta,
                }
            )

    test_targets = [int(v) for v in test_df["item_id"].tolist() if int(v) in baseline_stats and int(v) in compare_stats]
    test_reduced = 0
    test_increased = 0
    test_same = 0
    test_out_same_l2 = 0
    test_in_same_l2 = 0
    test_out_ge4 = 0
    test_in_ge4 = 0
    test_deltas: list[int] = []
    for item_id in test_targets:
        base_leaf = int(baseline_stats[item_id]["l2_leaf_count"])
        comp_leaf = int(compare_stats[item_id]["l2_leaf_count"])
        delta = comp_leaf - base_leaf
        test_deltas.append(delta)
        if comp_leaf < base_leaf:
            test_reduced += 1
        elif comp_leaf > base_leaf:
            test_increased += 1
        else:
            test_same += 1
        if base_leaf > 1 and comp_leaf == 1:
            test_out_same_l2 += 1
        if base_leaf == 1 and comp_leaf > 1:
            test_in_same_l2 += 1
        if base_leaf >= 4 and comp_leaf < 4:
            test_out_ge4 += 1
        if base_leaf < 4 and comp_leaf >= 4:
            test_in_ge4 += 1

    top_examples = sorted(
        best_examples,
        key=lambda row: (int(row["delta_l2_leaf_count"]), int(row["hierarchy_l2_leaf_count"]), int(row["item_id"])),
    )[:20]

    total_items = len(common_items)
    total_test = len(test_targets)
    return {
        "item_level": {
            "common_item_count": total_items,
            "changed_sid_count": changed_items,
            "changed_sid_fraction": (changed_items / total_items) if total_items else 0.0,
            "reduced_l2_leaf_count_fraction": (reduced_leaf_count / total_items) if total_items else 0.0,
            "increased_l2_leaf_count_fraction": (increased_leaf_count / total_items) if total_items else 0.0,
            "same_l2_leaf_count_fraction": (same_leaf_count / total_items) if total_items else 0.0,
            "moved_out_of_same_l2_count": moved_out_of_same_l2,
            "moved_into_same_l2_count": moved_into_same_l2,
            "moved_out_of_ge4_count": moved_out_of_ge4,
            "moved_into_ge4_count": moved_into_ge4,
            "mean_delta_l2_leaf_count_all_items": mean(deltas_all) if deltas_all else 0.0,
            "mean_delta_l2_leaf_count_changed_items": mean(deltas_changed) if deltas_changed else 0.0,
        },
        "test_weighted": {
            "target_count": total_test,
            "reduced_l2_leaf_count_fraction": (test_reduced / total_test) if total_test else 0.0,
            "increased_l2_leaf_count_fraction": (test_increased / total_test) if total_test else 0.0,
            "same_l2_leaf_count_fraction": (test_same / total_test) if total_test else 0.0,
            "moved_out_of_same_l2_fraction": (test_out_same_l2 / total_test) if total_test else 0.0,
            "moved_into_same_l2_fraction": (test_in_same_l2 / total_test) if total_test else 0.0,
            "moved_out_of_ge4_fraction": (test_out_ge4 / total_test) if total_test else 0.0,
            "moved_into_ge4_fraction": (test_in_ge4 / total_test) if total_test else 0.0,
            "mean_delta_l2_leaf_count": mean(test_deltas) if test_deltas else 0.0,
        },
        "top_improved_examples": top_examples,
    }


def format_pct(value: float) -> str:
    return f"{value * 100:.2f}\\%"


def render_markdown(summary: dict) -> str:
    base_cat = summary["baseline"]["catalog"]
    comp_cat = summary["hierarchy"]["catalog"]
    base_test = summary["baseline"]["test_weighted"]
    comp_test = summary["hierarchy"]["test_weighted"]
    comp = summary["comparison"]

    lines: list[str] = []
    lines.append("# Baseline vs Hierarchy Final SID Local Ambiguity Analysis")
    lines.append("")
    lines.append("## Conclusion")
    lines.append("")
    if comp_test["target_weighted_mean_l2_leaf_count"] < base_test["target_weighted_mean_l2_leaf_count"]:
        lines.append(
            f"- `same_l2` 有改善：测试目标 item 的平均 `l2` 叶子数从 `{base_test['target_weighted_mean_l2_leaf_count']:.4f}` 降到 `{comp_test['target_weighted_mean_l2_leaf_count']:.4f}`。"
        )
    else:
        lines.append(
            f"- `same_l2` 没改善：测试目标 item 的平均 `l2` 叶子数从 `{base_test['target_weighted_mean_l2_leaf_count']:.4f}` 变到 `{comp_test['target_weighted_mean_l2_leaf_count']:.4f}`。"
        )
    lines.append(
        f"- 测试目标落在多叶 `same_l2` bucket 的比例从 `{base_test['target_fraction_multileaf_l2']:.4f}` 变到 `{comp_test['target_fraction_multileaf_l2']:.4f}`。"
    )
    lines.append(
        f"- 测试目标落在深拥挤 `l2` bucket (`>=4` leaves) 的比例从 `{base_test['target_fraction_multileaf_l2_ge4']:.4f}` 变到 `{comp_test['target_fraction_multileaf_l2_ge4']:.4f}`。"
    )
    lines.append(
        f"- 测试样本里，有 `{comp['test_weighted']['moved_out_of_same_l2_fraction']:.4f}` 的目标 item 从 `same_l2` 多叶 bucket 被移到单叶 bucket；只有 `{comp['test_weighted']['moved_into_same_l2_fraction']:.4f}` 被移入更拥挤的 `same_l2` bucket。"
    )
    lines.append("")
    lines.append("## Catalog-Level")
    lines.append("")
    lines.append("| Metric | Baseline | Hierarchy | Delta |")
    lines.append("|---|---:|---:|---:|")
    for key, label in [
        ("item_weighted_mean_l2_leaf_count", "Item-weighted mean l2 leaf count"),
        ("item_fraction_multileaf_l2", "Fraction items in multi-leaf l2"),
        ("item_fraction_multileaf_l2_ge4", "Fraction items in l2 with >=4 leaves"),
        ("weighted_l3_entropy_given_l2_bits", "Weighted H(level3|level1,level2)"),
    ]:
        b = float(base_cat[key])
        h = float(comp_cat[key])
        lines.append(f"| {label} | {b:.6f} | {h:.6f} | {h - b:+.6f} |")
    lines.append("")
    lines.append("## Test-Weighted")
    lines.append("")
    lines.append("| Metric | Baseline | Hierarchy | Delta |")
    lines.append("|---|---:|---:|---:|")
    for key, label in [
        ("target_weighted_mean_l2_leaf_count", "Mean target l2 leaf count"),
        ("target_fraction_multileaf_l2", "Fraction targets in multi-leaf l2"),
        ("target_fraction_multileaf_l2_ge4", "Fraction targets in l2 with >=4 leaves"),
        ("target_weighted_mean_l3_entropy_bits", "Mean target l3 entropy under l2"),
    ]:
        b = float(base_test[key])
        h = float(comp_test[key])
        lines.append(f"| {label} | {b:.6f} | {h:.6f} | {h - b:+.6f} |")
    lines.append("")
    lines.append("## Movement Summary")
    lines.append("")
    lines.append(f"- `SID` changed on `{format_pct(comp['item_level']['changed_sid_fraction'])}` of catalog items.")
    lines.append(f"- Test-weighted targets with reduced `l2` leaf count: `{format_pct(comp['test_weighted']['reduced_l2_leaf_count_fraction'])}`.")
    lines.append(f"- Test-weighted targets with increased `l2` leaf count: `{format_pct(comp['test_weighted']['increased_l2_leaf_count_fraction'])}`.")
    lines.append(f"- Test-weighted targets moved out of multi-leaf `same_l2`: `{format_pct(comp['test_weighted']['moved_out_of_same_l2_fraction'])}`.")
    lines.append(f"- Test-weighted targets moved into multi-leaf `same_l2`: `{format_pct(comp['test_weighted']['moved_into_same_l2_fraction'])}`.")
    lines.append(f"- Test-weighted mean delta of `l2` leaf count: `{comp['test_weighted']['mean_delta_l2_leaf_count']:+.6f}`.")
    lines.append("")
    lines.append("## Top Improved Examples")
    lines.append("")
    for row in comp["top_improved_examples"][:10]:
        lines.append(
            f"- `{row['item_id']}` | `{row['baseline_l2_leaf_count']} -> {row['hierarchy_l2_leaf_count']}` | {row['title']}"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    baseline_index = Path(args.baseline_index)
    compare_index = Path(args.compare_index)
    test_csv = Path(args.test_csv)
    item_json = Path(args.item_json)
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    baseline = load_index(baseline_index)
    compare = load_index(compare_index)
    titles = load_titles(item_json)
    test_df = pd.read_csv(test_csv)

    baseline_catalog = build_index_stats(baseline)
    compare_catalog = build_index_stats(compare)
    baseline_test = build_test_weighted_stats(baseline_catalog["item_stats"], test_df)
    compare_test = build_test_weighted_stats(compare_catalog["item_stats"], test_df)
    comparison = build_pairwise_comparison(
        baseline_catalog["item_stats"],
        compare_catalog["item_stats"],
        test_df,
        titles,
    )

    summary = {
        "inputs": {
            "baseline_index": str(baseline_index),
            "compare_index": str(compare_index),
            "test_csv": str(test_csv),
            "item_json": str(item_json),
        },
        "baseline": {
            "catalog": {k: v for k, v in baseline_catalog.items() if k != "item_stats"},
            "test_weighted": baseline_test,
        },
        "hierarchy": {
            "catalog": {k: v for k, v in compare_catalog.items() if k != "item_stats"},
            "test_weighted": compare_test,
        },
        "comparison": comparison,
    }

    output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    output_md.write_text(render_markdown(summary), encoding="utf-8")

    print(f"Wrote JSON: {output_json}")
    print(f"Wrote Markdown: {output_md}")


if __name__ == "__main__":
    main()
