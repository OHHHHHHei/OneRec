#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Iterable


def canonicalize_semantic_id(text: object) -> str:
    if text is None:
        return ""
    value = str(text).strip(" \n\r\t\"")
    start = value.find("<a_")
    if start == -1:
        return value
    end = value.find(">", start)
    if end == -1:
        return value
    # The repo currently uses exactly 3-level SIDs like <a_x><b_y><c_z>.
    import re

    match = re.search(r"<a_\d+><b_\d+><c_\d+>", value)
    return match.group(0) if match else value


def parse_sid(sid: str) -> tuple[str, str, str]:
    import re

    parts = re.findall(r"<[abc]_\d+>", canonicalize_semantic_id(sid))
    if len(parts) != 3:
        return ("", "", "")
    return tuple(parts)  # type: ignore[return-value]


def lcp_len(lhs: str, rhs: str) -> int:
    a = parse_sid(lhs)
    b = parse_sid(rhs)
    score = 0
    for x, y in zip(a, b):
        if x == y and x:
            score += 1
        else:
            break
    return score


def entropy_from_counts(counts: Counter[str]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    value = 0.0
    for count in counts.values():
        p = count / total
        value -= p * math.log2(p)
    return value


def load_index(index_path: Path) -> dict[str, list[str]]:
    with open(index_path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    sid_map: dict[str, list[str]] = {}
    for item_id, tokens in raw.items():
        sid_map[str(item_id)] = list(tokens)
    return sid_map


def load_info(info_path: Path) -> tuple[dict[str, str], dict[str, str]]:
    sid_to_title: dict[str, str] = {}
    sid_to_item_id: dict[str, str] = {}
    with open(info_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            sid = canonicalize_semantic_id(parts[0])
            item_id = parts[-1]
            title = "\t".join(parts[1:-1])
            sid_to_title[sid] = title
            sid_to_item_id[sid] = item_id
    return sid_to_title, sid_to_item_id


def weighted_mean(values: Iterable[tuple[float, float]]) -> float:
    total_weight = 0.0
    total_value = 0.0
    for value, weight in values:
        total_value += value * weight
        total_weight += weight
    if total_weight == 0:
        return 0.0
    return total_value / total_weight


def build_catalog_stats(index_map: dict[str, list[str]]) -> dict:
    item_count = len(index_map)
    sid_to_item_ids: defaultdict[str, list[str]] = defaultdict(list)
    l1_to_items: defaultdict[str, list[str]] = defaultdict(list)
    l2_to_items: defaultdict[tuple[str, str], list[str]] = defaultdict(list)
    l1_children: defaultdict[str, Counter[str]] = defaultdict(Counter)
    l2_children: defaultdict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    b_parents: defaultdict[str, set[str]] = defaultdict(set)
    c_parents: defaultdict[str, set[tuple[str, str]]] = defaultdict(set)

    for item_id, tokens in index_map.items():
        sid = "".join(tokens)
        a, b, c = tokens
        sid_to_item_ids[sid].append(item_id)
        l1_to_items[a].append(item_id)
        l2_to_items[(a, b)].append(item_id)
        l1_children[a][b] += 1
        l2_children[(a, b)][c] += 1
        b_parents[b].add(a)
        c_parents[c].add((a, b))

    collision_groups = {sid: ids for sid, ids in sid_to_item_ids.items() if len(ids) > 1}
    collision_count = sum(len(ids) - 1 for ids in collision_groups.values())
    collision_rate = collision_count / item_count if item_count else 0.0

    l1_entropy = weighted_mean(
        (entropy_from_counts(child_counts), sum(child_counts.values()))
        for child_counts in l1_children.values()
    )
    l2_entropy = weighted_mean(
        (entropy_from_counts(child_counts), sum(child_counts.values()))
        for child_counts in l2_children.values()
    )

    layer2_parent_counts = [len(parents) for parents in b_parents.values()]
    layer3_parent_counts = [len(parents) for parents in c_parents.values()]

    return {
        "item_count": item_count,
        "unique_sid_count": len(sid_to_item_ids),
        "collision_group_count": len(collision_groups),
        "collision_item_excess": collision_count,
        "collision_rate": collision_rate,
        "max_collision_group_size": max((len(ids) for ids in collision_groups.values()), default=1),
        "weighted_prefix_conditional_entropy_l2_given_l1_bits": l1_entropy,
        "weighted_prefix_conditional_entropy_l3_given_l1l2_bits": l2_entropy,
        "mean_items_per_l1_prefix": mean(len(items) for items in l1_to_items.values()),
        "mean_items_per_l2_prefix": mean(len(items) for items in l2_to_items.values()),
        "median_layer2_parent_count": sorted(layer2_parent_counts)[len(layer2_parent_counts) // 2] if layer2_parent_counts else 0,
        "median_layer3_parent_count": sorted(layer3_parent_counts)[len(layer3_parent_counts) // 2] if layer3_parent_counts else 0,
        "max_layer2_parent_count": max(layer2_parent_counts, default=0),
        "max_layer3_parent_count": max(layer3_parent_counts, default=0),
        "top_collision_groups": [
            {"sid": sid, "item_ids": ids, "size": len(ids)}
            for sid, ids in sorted(collision_groups.items(), key=lambda item: (-len(item[1]), item[0]))[:10]
        ],
        "top_ambiguous_layer2_codes": [
            {"code": code, "distinct_a_prefixes": len(parents)}
            for code, parents in sorted(b_parents.items(), key=lambda item: (-len(item[1]), item[0]))[:10]
        ],
        "top_ambiguous_layer3_codes": [
            {"code": code, "distinct_ab_prefixes": len(parents)}
            for code, parents in sorted(c_parents.items(), key=lambda item: (-len(item[1]), item[0]))[:10]
        ],
        "sid_to_item_ids": sid_to_item_ids,
        "l1_to_items": l1_to_items,
        "l2_to_items": l2_to_items,
    }


def evaluate_topk(predictions: list[str], target: str, topk: int) -> bool:
    limit = min(topk, len(predictions))
    return target in predictions[:limit]


def build_eval_stats(
    result_data: list[dict],
    sid_to_title: dict[str, str],
    catalog_stats: dict,
) -> tuple[dict, list[dict]]:
    sid_to_item_ids = catalog_stats["sid_to_item_ids"]
    l1_to_items = catalog_stats["l1_to_items"]
    l2_to_items = catalog_stats["l2_to_items"]

    sample_rows: list[dict] = []
    exact_top1 = 0
    topk_hits = {1: 0, 3: 0, 5: 0, 10: 0, 20: 0, 50: 0}
    top1_lcp_total = 0
    top1_error_lcp_total = 0
    top1_error_count = 0
    top1_error_same_l1 = 0
    top1_error_same_l2 = 0
    best_lcp_total = 0
    beam_same_l1 = 0
    beam_same_l2 = 0
    pred1_in_catalog = 0
    collided_target_total = 0
    collided_target_top1_hit = 0
    unique_target_total = 0
    unique_target_top1_hit = 0
    hit_l2_fanouts: list[int] = []
    miss_l2_fanouts: list[int] = []
    hit_l1_fanouts: list[int] = []
    miss_l1_fanouts: list[int] = []

    for row in result_data:
        target_sid = canonicalize_semantic_id(row.get("output", ""))
        predictions = [canonicalize_semantic_id(value) for value in row.get("predict", [])]
        pred1 = predictions[0] if predictions else ""
        target_parts = parse_sid(target_sid)
        pred1_parts = parse_sid(pred1)
        l1_fanout = len(l1_to_items[target_parts[0]]) if target_parts[0] else 0
        l2_fanout = len(l2_to_items[(target_parts[0], target_parts[1])]) if target_parts[1] else 0
        collision_group_size = len(sid_to_item_ids[target_sid]) if target_sid in sid_to_item_ids else 0
        if collision_group_size > 1:
            collided_target_total += 1
        else:
            unique_target_total += 1

        pred1_lcp = lcp_len(pred1, target_sid)
        top1_lcp_total += pred1_lcp
        best_lcp = max((lcp_len(candidate, target_sid) for candidate in predictions), default=0)
        best_lcp_total += best_lcp
        beam_same_l1 += int(best_lcp >= 1)
        beam_same_l2 += int(best_lcp >= 2)
        top1_hit = bool(pred1 and pred1 == target_sid)
        if top1_hit:
            exact_top1 += 1
            if collision_group_size > 1:
                collided_target_top1_hit += 1
            else:
                unique_target_top1_hit += 1
            hit_l1_fanouts.append(l1_fanout)
            hit_l2_fanouts.append(l2_fanout)
        else:
            top1_error_count += 1
            top1_error_lcp_total += pred1_lcp
            if pred1_lcp >= 1:
                top1_error_same_l1 += 1
            if pred1_lcp >= 2:
                top1_error_same_l2 += 1
            miss_l1_fanouts.append(l1_fanout)
            miss_l2_fanouts.append(l2_fanout)

        if pred1 in sid_to_item_ids:
            pred1_in_catalog += 1

        for k in topk_hits:
            topk_hits[k] += int(evaluate_topk(predictions, target_sid, k))

        sample_rows.append(
            {
                "input": row.get("input", ""),
                "target_sid": target_sid,
                "target_title": sid_to_title.get(target_sid, ""),
                "pred1_sid": pred1,
                "pred1_title": sid_to_title.get(pred1, ""),
                "beam_size": len(predictions),
                "exact_top1_hit": int(top1_hit),
                "top3_hit": int(evaluate_topk(predictions, target_sid, 3)),
                "top5_hit": int(evaluate_topk(predictions, target_sid, 5)),
                "top10_hit": int(evaluate_topk(predictions, target_sid, 10)),
                "pred1_in_catalog": int(pred1 in sid_to_item_ids),
                "pred1_lcp": pred1_lcp,
                "best_lcp_in_beam": best_lcp,
                "target_collision_group_size": collision_group_size,
                "target_l1_fanout": l1_fanout,
                "target_l2_fanout": l2_fanout,
            }
        )

    total = len(result_data)
    return (
        {
            "example_count": total,
            "top1_hit_count": exact_top1,
            "top1_hit_rate": exact_top1 / total if total else 0.0,
            "top3_hit_rate": topk_hits[3] / total if total else 0.0,
            "top5_hit_rate": topk_hits[5] / total if total else 0.0,
            "top10_hit_rate": topk_hits[10] / total if total else 0.0,
            "top20_hit_rate": topk_hits[20] / total if total else 0.0,
            "top50_hit_rate": topk_hits[50] / total if total else 0.0,
            "pred1_in_catalog_rate": pred1_in_catalog / total if total else 0.0,
            "avg_top1_lcp": top1_lcp_total / total if total else 0.0,
            "avg_top1_lcp_on_error": top1_error_lcp_total / top1_error_count if top1_error_count else 0.0,
            "avg_best_lcp_in_beam": best_lcp_total / total if total else 0.0,
            "beam_contains_same_l1_rate": beam_same_l1 / total if total else 0.0,
            "beam_contains_same_l2_rate": beam_same_l2 / total if total else 0.0,
            "top1_error_count": top1_error_count,
            "top1_error_same_l1_rate": top1_error_same_l1 / top1_error_count if top1_error_count else 0.0,
            "top1_error_same_l2_rate": top1_error_same_l2 / top1_error_count if top1_error_count else 0.0,
            "top1_error_same_l1_count": top1_error_same_l1,
            "top1_error_same_l2_count": top1_error_same_l2,
            "collided_target_count": collided_target_total,
            "collided_target_fraction": collided_target_total / total if total else 0.0,
            "top1_hit_rate_for_collided_targets": collided_target_top1_hit / collided_target_total if collided_target_total else 0.0,
            "top1_hit_count_for_collided_targets": collided_target_top1_hit,
            "unique_target_count": unique_target_total,
            "top1_hit_rate_for_unique_targets": unique_target_top1_hit / unique_target_total if unique_target_total else 0.0,
            "top1_hit_count_for_unique_targets": unique_target_top1_hit,
            "avg_target_l1_fanout_on_hit": mean(hit_l1_fanouts) if hit_l1_fanouts else 0.0,
            "avg_target_l1_fanout_on_miss": mean(miss_l1_fanouts) if miss_l1_fanouts else 0.0,
            "avg_target_l2_fanout_on_hit": mean(hit_l2_fanouts) if hit_l2_fanouts else 0.0,
            "avg_target_l2_fanout_on_miss": mean(miss_l2_fanouts) if miss_l2_fanouts else 0.0,
        },
        sample_rows,
    )


def summarize_diagnostics(summary: dict) -> list[str]:
    catalog = summary["catalog_level"]
    eval_stats = summary["eval_level"]
    lines = [
        f"items={catalog['item_count']} unique_sids={catalog['unique_sid_count']} collision_rate={catalog['collision_rate']:.4%}",
        f"weighted_H(level2|level1)={catalog['weighted_prefix_conditional_entropy_l2_given_l1_bits']:.4f} bits",
        f"weighted_H(level3|level1,level2)={catalog['weighted_prefix_conditional_entropy_l3_given_l1l2_bits']:.4f} bits",
        f"top1_hit={eval_stats['top1_hit_rate']:.4%} top10_hit={eval_stats['top10_hit_rate']:.4%}",
        f"avg_top1_lcp={eval_stats['avg_top1_lcp']:.4f} avg_top1_lcp_on_error={eval_stats['avg_top1_lcp_on_error']:.4f}",
        f"beam_same_l1={eval_stats['beam_contains_same_l1_rate']:.4%} beam_same_l2={eval_stats['beam_contains_same_l2_rate']:.4%}",
        f"same_l1_among_top1_errors={eval_stats['top1_error_same_l1_rate']:.4%} same_l2_among_top1_errors={eval_stats['top1_error_same_l2_rate']:.4%}",
        f"hit_rate(collided_targets)={eval_stats['top1_hit_rate_for_collided_targets']:.4%} [{eval_stats['top1_hit_count_for_collided_targets']}/{eval_stats['collided_target_count']}] hit_rate(unique_targets)={eval_stats['top1_hit_rate_for_unique_targets']:.4%} [{eval_stats['top1_hit_count_for_unique_targets']}/{eval_stats['unique_target_count']}]",
        f"avg_target_l2_fanout_on_hit={eval_stats['avg_target_l2_fanout_on_hit']:.2f} avg_target_l2_fanout_on_miss={eval_stats['avg_target_l2_fanout_on_miss']:.2f}",
    ]
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose SID issues from catalog SIDs and evaluation results.")
    parser.add_argument("--result-json", required=True, help="Path to final_result_*.json produced by evaluate.")
    parser.add_argument("--index-json", required=True, help="Path to *.index.json mapping item_id to SID tokens.")
    parser.add_argument("--info-txt", required=True, help="Path to *.txt mapping SID to title and item_id.")
    parser.add_argument("--output-json", required=True, help="Where to write the summary JSON.")
    parser.add_argument("--output-csv", required=True, help="Where to write per-example diagnostics CSV.")
    args = parser.parse_args()

    result_path = Path(args.result_json)
    index_path = Path(args.index_json)
    info_path = Path(args.info_txt)
    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(result_path, "r", encoding="utf-8") as handle:
        result_data = json.load(handle)
    if not isinstance(result_data, list):
        raise ValueError(f"{result_path} does not contain a list of evaluation rows")

    index_map = load_index(index_path)
    sid_to_title, sid_to_item_id = load_info(info_path)
    catalog_stats = build_catalog_stats(index_map)
    eval_stats, sample_rows = build_eval_stats(result_data, sid_to_title, catalog_stats)

    summary = {
        "inputs": {
            "result_json": str(result_path),
            "index_json": str(index_path),
            "info_txt": str(info_path),
        },
        "catalog_level": {
            key: value for key, value in catalog_stats.items() if key not in {"sid_to_item_ids", "l1_to_items", "l2_to_items"}
        },
        "eval_level": eval_stats,
        "notes": [
            "collision_rate and prefix conditional entropy are catalog-level diagnostics; they do not by themselves prove downstream errors.",
            "same-prefix miss rates and collided-target hit rates connect SID properties to actual evaluation failures.",
            "This script diagnoses SID/prefix issues more directly than aggregate HR/NDCG, but it does not directly prove the absence of collaborative signal in the tokenizer.",
        ],
        "human_summary": summarize_diagnostics(
            {
                "catalog_level": {key: value for key, value in catalog_stats.items() if key not in {"sid_to_item_ids", "l1_to_items", "l2_to_items"}},
                "eval_level": eval_stats,
            }
        ),
    }

    with open(output_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    fieldnames = [
        "input",
        "target_sid",
        "target_title",
        "pred1_sid",
        "pred1_title",
        "beam_size",
        "exact_top1_hit",
        "top3_hit",
        "top5_hit",
        "top10_hit",
        "pred1_in_catalog",
        "pred1_lcp",
        "best_lcp_in_beam",
        "target_collision_group_size",
        "target_l1_fanout",
        "target_l2_fanout",
    ]
    with open(output_csv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sample_rows)

    print("SID diagnostics written.")
    print(f"Summary JSON: {output_json}")
    print(f"Per-example CSV: {output_csv}")
    for line in summary["human_summary"]:
        print(line)


if __name__ == "__main__":
    main()
