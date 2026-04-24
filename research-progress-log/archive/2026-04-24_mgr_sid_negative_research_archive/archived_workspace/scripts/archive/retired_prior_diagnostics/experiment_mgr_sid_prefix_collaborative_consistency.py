#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from onerec.experiments.mgr_sid.graph_bank import infer_num_items, parse_id_list
from onerec.experiments.mgr_sid.paper_transplants import (
    keep_topk_per_row,
    load_semantic_embeddings,
    symmetrize_matrix,
)
from onerec.experiments.mgr_sid.transplanted_graph_bank import build_transplanted_graph_bank


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose whether multi-leaf l2 prefixes are collaboratively consistent or inconsistent."
    )
    parser.add_argument("--baseline-index", required=True)
    parser.add_argument("--compare-index", required=True)
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--item-json", required=True)
    parser.add_argument("--semantic-embedding-path", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--history-k", type=int, default=10)
    parser.add_argument("--coarse-min-weight", type=float, default=2.0)
    parser.add_argument("--local-min-weight", type=float, default=1.0)
    parser.add_argument("--community-clusters", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--anchor-topk", type=int, default=32)
    parser.add_argument("--semantic-mix", type=float, default=0.35)
    parser.add_argument("--spectral-rank", type=int, default=48)
    parser.add_argument("--band-low", type=float, default=0.25)
    parser.add_argument("--band-high", type=float, default=0.65)
    parser.add_argument("--temporal-mix", type=float, default=0.35)
    parser.add_argument("--fagsp-cascade-high-rank", type=int, default=48)
    parser.add_argument("--fagsp-cascade-low-rank", type=int, default=48)
    parser.add_argument("--fagsp-cascade-support-quantile", type=float, default=0.85)
    parser.add_argument("--fagsp-cascade-boost-alpha", type=float, default=0.35)
    parser.add_argument("--mgdcf-keep-ratio", type=float, default=0.1)
    parser.add_argument("--mgdcf-binarize-edges", action="store_true")
    parser.add_argument("--graph-topk", type=int, default=32)
    parser.add_argument("--graph-weak-quantile", type=float, default=0.25)
    parser.add_argument(
        "--consistent-strong-frac-threshold",
        type=float,
        default=0.5,
        help="Prefix is collaboratively consistent if strong-pair fraction is at least this threshold.",
    )
    parser.add_argument(
        "--mixed-positive-frac-threshold",
        type=float,
        default=0.5,
        help="Prefix is mixed if positive-pair fraction is at least this threshold but strong-pair fraction is below the consistent threshold.",
    )
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


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
    out: dict[int, str] = {}
    for key, value in raw.items():
        if isinstance(value, dict):
            out[int(key)] = str(value.get("title", f"Item_{key}"))
        else:
            out[int(key)] = str(value)
    return out


def load_base_views(args: argparse.Namespace) -> dict[str, sparse.csr_matrix]:
    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)
    views = build_transplanted_graph_bank(
        train_df=train_df,
        test_df=test_df,
        history_k=args.history_k,
        coarse_min_weight=args.coarse_min_weight,
        local_min_weight=args.local_min_weight,
        n_clusters=args.community_clusters,
        seed=args.seed,
        semantic_embedding_path=args.semantic_embedding_path,
        anchor_topk=args.anchor_topk,
        semantic_mix=args.semantic_mix,
        spectral_rank=args.spectral_rank,
        band_low=args.band_low,
        band_high=args.band_high,
        temporal_mix=args.temporal_mix,
        fagsp_cascade_high_rank=args.fagsp_cascade_high_rank,
        fagsp_cascade_low_rank=args.fagsp_cascade_low_rank,
        fagsp_cascade_support_quantile=args.fagsp_cascade_support_quantile,
        fagsp_cascade_boost_alpha=args.fagsp_cascade_boost_alpha,
        mgdcf_keep_ratio=args.mgdcf_keep_ratio,
        mgdcf_binarize_edges=args.mgdcf_binarize_edges,
    )
    selected: dict[str, sparse.csr_matrix] = {}
    for name in ["coarse_purified", "fagsp_mid_base", "local_purified"]:
        selected[name] = keep_topk_per_row(views[name].matrix, topk=args.graph_topk).tocsr().astype(np.float32)
    return selected


def build_combined_graph_affinity(views: dict[str, sparse.csr_matrix]) -> sparse.csr_matrix:
    sym_views = [symmetrize_matrix(matrix) for matrix in views.values()]
    dense = np.stack([view.toarray() for view in sym_views], axis=0)
    combined = np.max(dense, axis=0).astype(np.float32)
    np.fill_diagonal(combined, 0.0)
    return sparse.csr_matrix(combined)


def build_item_user_sets(train_df: pd.DataFrame, n_items: int) -> list[set[str]]:
    item_users: list[set[str]] = [set() for _ in range(n_items)]
    for row in train_df.itertuples(index=False):
        user_id = str(row.user_id)
        target = int(row.item_id)
        if 0 <= target < n_items:
            item_users[target].add(user_id)
        for hist_item in parse_id_list(row.history_item_id):
            if 0 <= hist_item < n_items:
                item_users[hist_item].add(user_id)
    return item_users


def user_overlap(item_users: list[set[str]], a: int, b: int) -> float:
    users_a = item_users[a]
    users_b = item_users[b]
    if not users_a and not users_b:
        return 0.0
    union = users_a | users_b
    if not union:
        return 0.0
    return float(len(users_a & users_b) / len(union))


def weighted_mean(values: list[float], weights: list[float]) -> float:
    if not values or not weights:
        return 0.0
    total = float(sum(weights))
    if total <= 0:
        return 0.0
    return float(sum(v * w for v, w in zip(values, weights, strict=False)) / total)


def classify_prefix(
    leaf_count: int,
    strong_pair_fraction: float,
    positive_pair_fraction: float,
    consistent_threshold: float,
    mixed_threshold: float,
) -> str:
    if leaf_count <= 1:
        return "singleton"
    if strong_pair_fraction >= consistent_threshold:
        return "consistent_crowded"
    if positive_pair_fraction >= mixed_threshold:
        return "mixed_crowded"
    return "inconsistent_crowded"


def build_prefix_rows(
    *,
    system_label: str,
    index_map: dict[int, tuple[str, str, str]],
    combined_graph: sparse.csr_matrix,
    semantic_embeddings: np.ndarray,
    item_users: list[set[str]],
    titles: dict[int, str],
    test_target_counts: dict[int, int],
    weak_threshold: float,
    consistent_threshold: float,
    mixed_threshold: float,
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for item_id, (l1, l2, _) in index_map.items():
        groups[(l1, l2)].append(int(item_id))

    rows: list[dict[str, Any]] = []
    item_stats: dict[int, dict[str, Any]] = {}
    for prefix_tokens, items in groups.items():
        items = sorted(items)
        leaf_ids = sorted({index_map[item_id][2] for item_id in items})
        item_count = len(items)
        leaf_count = len(leaf_ids)
        test_weight = int(sum(test_target_counts.get(item_id, 0) for item_id in items))
        prefix_pairs = list(combinations(items, 2))

        graph_values: list[float] = []
        semantic_values: list[float] = []
        overlap_values: list[float] = []
        coarse_values: list[float] = []
        mid_values: list[float] = []
        local_values: list[float] = []
        positive_pairs = 0
        strong_pairs = 0
        weak_pairs = 0
        non_pairs = 0

        for item_a, item_b in prefix_pairs:
            graph_affinity = float(combined_graph[item_a, item_b])
            emb_a = semantic_embeddings[item_a]
            emb_b = semantic_embeddings[item_b]
            semantic_sim = float(np.dot(emb_a, emb_b))
            overlap = user_overlap(item_users, item_a, item_b)
            graph_values.append(graph_affinity)
            semantic_values.append(semantic_sim)
            overlap_values.append(overlap)
            if graph_affinity > 0.0:
                positive_pairs += 1
                if graph_affinity <= weak_threshold:
                    weak_pairs += 1
                else:
                    strong_pairs += 1
            else:
                non_pairs += 1

        pair_count = len(prefix_pairs)
        positive_pair_fraction = float(positive_pairs / pair_count) if pair_count else 0.0
        strong_pair_fraction = float(strong_pairs / pair_count) if pair_count else 0.0
        weak_pair_fraction = float(weak_pairs / pair_count) if pair_count else 0.0
        non_pair_fraction = float(non_pairs / pair_count) if pair_count else 0.0
        mean_graph_affinity = float(mean(graph_values)) if graph_values else 0.0
        mean_semantic_sim = float(mean(semantic_values)) if semantic_values else 0.0
        mean_user_overlap = float(mean(overlap_values)) if overlap_values else 0.0
        prefix_type = classify_prefix(
            leaf_count=leaf_count,
            strong_pair_fraction=strong_pair_fraction,
            positive_pair_fraction=positive_pair_fraction,
            consistent_threshold=consistent_threshold,
            mixed_threshold=mixed_threshold,
        )
        representative_titles = [titles.get(item_id, f"Item_{item_id}") for item_id in items[:3]]
        row = {
            "system": system_label,
            "prefix": f"{prefix_tokens[0]}{prefix_tokens[1]}",
            "l1": prefix_tokens[0],
            "l2": prefix_tokens[1],
            "item_count": item_count,
            "leaf_count": leaf_count,
            "pair_count": pair_count,
            "test_weight": test_weight,
            "mean_graph_affinity": mean_graph_affinity,
            "mean_semantic_sim": mean_semantic_sim,
            "mean_user_overlap": mean_user_overlap,
            "positive_pair_fraction": positive_pair_fraction,
            "strong_pair_fraction": strong_pair_fraction,
            "weak_pair_fraction": weak_pair_fraction,
            "non_pair_fraction": non_pair_fraction,
            "prefix_type": prefix_type,
            "representative_titles": representative_titles,
        }
        rows.append(row)
        for item_id in items:
            item_stats[item_id] = {
                "prefix": row["prefix"],
                "leaf_count": leaf_count,
                "item_count": item_count,
                "mean_graph_affinity": mean_graph_affinity,
                "mean_semantic_sim": mean_semantic_sim,
                "positive_pair_fraction": positive_pair_fraction,
                "strong_pair_fraction": strong_pair_fraction,
                "prefix_type": prefix_type,
                "test_weight": int(test_target_counts.get(item_id, 0)),
            }
    return rows, item_stats


def summarize_system(prefix_rows: list[dict[str, Any]], item_stats: dict[int, dict[str, Any]], test_df: pd.DataFrame) -> dict[str, Any]:
    if not prefix_rows:
        return {}

    item_weights = [float(row["item_count"]) for row in prefix_rows]
    test_weights = [float(row["test_weight"]) for row in prefix_rows]
    crowded_rows = [row for row in prefix_rows if int(row["leaf_count"]) > 1]
    crowded_item_weights = [float(row["item_count"]) for row in crowded_rows]
    crowded_test_weights = [float(row["test_weight"]) for row in crowded_rows]

    def fraction_for_type(weight_source: str, weights: list[float], target_type: str) -> float:
        if not weights:
            return 0.0
        total = float(sum(weights))
        if total <= 0:
            return 0.0
        subtotal = float(
            sum(float(row[weight_source]) for row in prefix_rows if str(row["prefix_type"]) == target_type)
        )
        return subtotal / total

    summary = {
        "prefix_count": len(prefix_rows),
        "catalog_item_count": len(item_stats),
        "item_weighted_mean_leaf_count": weighted_mean(
            [float(row["leaf_count"]) for row in prefix_rows],
            item_weights,
        ),
        "item_weighted_mean_prefix_graph_affinity": weighted_mean(
            [float(row["mean_graph_affinity"]) for row in prefix_rows],
            item_weights,
        ),
        "item_weighted_mean_prefix_semantic_sim": weighted_mean(
            [float(row["mean_semantic_sim"]) for row in prefix_rows],
            item_weights,
        ),
        "item_fraction_singleton": fraction_for_type("item_count", item_weights, "singleton"),
        "item_fraction_consistent_crowded": fraction_for_type("item_count", item_weights, "consistent_crowded"),
        "item_fraction_mixed_crowded": fraction_for_type("item_count", item_weights, "mixed_crowded"),
        "item_fraction_inconsistent_crowded": fraction_for_type("item_count", item_weights, "inconsistent_crowded"),
        "item_weighted_mean_crowded_graph_affinity": weighted_mean(
            [float(row["mean_graph_affinity"]) for row in crowded_rows],
            crowded_item_weights,
        ),
        "item_weighted_mean_crowded_strong_pair_fraction": weighted_mean(
            [float(row["strong_pair_fraction"]) for row in crowded_rows],
            crowded_item_weights,
        ),
    }

    test_items = [int(item_id) for item_id in test_df["item_id"].tolist() if int(item_id) in item_stats]
    total_test = len(test_items)
    test_leaf_counts: list[int] = []
    test_graph_affinities: list[float] = []
    test_semantic_sims: list[float] = []
    type_counter = defaultdict(int)
    for item_id in test_items:
        stats = item_stats[item_id]
        test_leaf_counts.append(int(stats["leaf_count"]))
        test_graph_affinities.append(float(stats["mean_graph_affinity"]))
        test_semantic_sims.append(float(stats["mean_semantic_sim"]))
        type_counter[str(stats["prefix_type"])] += 1

    summary.update(
        {
            "test_target_count": total_test,
            "test_weighted_mean_leaf_count": float(mean(test_leaf_counts)) if test_leaf_counts else 0.0,
            "test_weighted_median_leaf_count": float(median(test_leaf_counts)) if test_leaf_counts else 0.0,
            "test_weighted_mean_prefix_graph_affinity": float(mean(test_graph_affinities)) if test_graph_affinities else 0.0,
            "test_weighted_mean_prefix_semantic_sim": float(mean(test_semantic_sims)) if test_semantic_sims else 0.0,
            "test_fraction_singleton": float(type_counter["singleton"] / total_test) if total_test else 0.0,
            "test_fraction_consistent_crowded": float(type_counter["consistent_crowded"] / total_test) if total_test else 0.0,
            "test_fraction_mixed_crowded": float(type_counter["mixed_crowded"] / total_test) if total_test else 0.0,
            "test_fraction_inconsistent_crowded": float(type_counter["inconsistent_crowded"] / total_test) if total_test else 0.0,
        }
    )

    summary["top_consistent_crowded_prefixes"] = sorted(
        [row for row in prefix_rows if str(row["prefix_type"]) == "consistent_crowded"],
        key=lambda row: (-int(row["test_weight"]), -float(row["strong_pair_fraction"]), -int(row["item_count"]), str(row["prefix"])),
    )[:10]
    summary["top_inconsistent_crowded_prefixes"] = sorted(
        [row for row in prefix_rows if str(row["prefix_type"]) == "inconsistent_crowded"],
        key=lambda row: (-int(row["test_weight"]), -int(row["leaf_count"]), -int(row["item_count"]), str(row["prefix"])),
    )[:10]
    return summary


def build_comparison(
    baseline_stats: dict[int, dict[str, Any]],
    compare_stats: dict[int, dict[str, Any]],
    test_df: pd.DataFrame,
) -> dict[str, Any]:
    common_items = sorted(set(baseline_stats) & set(compare_stats))
    test_items = [int(item_id) for item_id in test_df["item_id"].tolist() if int(item_id) in baseline_stats and int(item_id) in compare_stats]

    item_moved_to_consistent = 0
    item_moved_to_inconsistent = 0
    item_moved_from_inconsistent_to_consistent = 0
    test_moved_to_consistent = 0
    test_moved_to_inconsistent = 0
    test_moved_from_inconsistent_to_consistent = 0
    delta_leaf_counts_all: list[float] = []
    delta_graph_affinity_all: list[float] = []
    delta_leaf_counts_test: list[float] = []
    delta_graph_affinity_test: list[float] = []

    for item_id in common_items:
        base = baseline_stats[item_id]
        comp = compare_stats[item_id]
        base_type = str(base["prefix_type"])
        comp_type = str(comp["prefix_type"])
        if base_type != "consistent_crowded" and comp_type == "consistent_crowded":
            item_moved_to_consistent += 1
        if base_type != "inconsistent_crowded" and comp_type == "inconsistent_crowded":
            item_moved_to_inconsistent += 1
        if base_type == "inconsistent_crowded" and comp_type == "consistent_crowded":
            item_moved_from_inconsistent_to_consistent += 1
        delta_leaf_counts_all.append(float(comp["leaf_count"]) - float(base["leaf_count"]))
        delta_graph_affinity_all.append(float(comp["mean_graph_affinity"]) - float(base["mean_graph_affinity"]))

    for item_id in test_items:
        base = baseline_stats[item_id]
        comp = compare_stats[item_id]
        base_type = str(base["prefix_type"])
        comp_type = str(comp["prefix_type"])
        if base_type != "consistent_crowded" and comp_type == "consistent_crowded":
            test_moved_to_consistent += 1
        if base_type != "inconsistent_crowded" and comp_type == "inconsistent_crowded":
            test_moved_to_inconsistent += 1
        if base_type == "inconsistent_crowded" and comp_type == "consistent_crowded":
            test_moved_from_inconsistent_to_consistent += 1
        delta_leaf_counts_test.append(float(comp["leaf_count"]) - float(base["leaf_count"]))
        delta_graph_affinity_test.append(float(comp["mean_graph_affinity"]) - float(base["mean_graph_affinity"]))

    total_items = len(common_items)
    total_test = len(test_items)
    return {
        "item_level": {
            "common_item_count": total_items,
            "moved_to_consistent_crowded_fraction": float(item_moved_to_consistent / total_items) if total_items else 0.0,
            "moved_to_inconsistent_crowded_fraction": float(item_moved_to_inconsistent / total_items) if total_items else 0.0,
            "moved_from_inconsistent_to_consistent_fraction": float(item_moved_from_inconsistent_to_consistent / total_items)
            if total_items
            else 0.0,
            "mean_delta_leaf_count": float(mean(delta_leaf_counts_all)) if delta_leaf_counts_all else 0.0,
            "mean_delta_prefix_graph_affinity": float(mean(delta_graph_affinity_all)) if delta_graph_affinity_all else 0.0,
        },
        "test_weighted": {
            "target_count": total_test,
            "moved_to_consistent_crowded_fraction": float(test_moved_to_consistent / total_test) if total_test else 0.0,
            "moved_to_inconsistent_crowded_fraction": float(test_moved_to_inconsistent / total_test) if total_test else 0.0,
            "moved_from_inconsistent_to_consistent_fraction": float(test_moved_from_inconsistent_to_consistent / total_test)
            if total_test
            else 0.0,
            "mean_delta_leaf_count": float(mean(delta_leaf_counts_test)) if delta_leaf_counts_test else 0.0,
            "mean_delta_prefix_graph_affinity": float(mean(delta_graph_affinity_test)) if delta_graph_affinity_test else 0.0,
        },
    }


def format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def render_prefix_rows(rows: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for row in rows:
        lines.append(
            "- "
            f"`{row['prefix']}` | items `{row['item_count']}` | leaves `{row['leaf_count']}` | "
            f"test_weight `{row['test_weight']}` | strong `{row['strong_pair_fraction']:.3f}` | "
            f"positive `{row['positive_pair_fraction']:.3f}` | graph `{row['mean_graph_affinity']:.6f}` | "
            f"semantic `{row['mean_semantic_sim']:.4f}` | "
            + " / ".join(str(title) for title in row["representative_titles"])
        )
    if not lines:
        lines.append("- none")
    return lines


def render_markdown(summary: dict[str, Any]) -> str:
    base = summary["baseline"]
    comp = summary["compare"]
    comparison = summary["comparison"]
    weak_threshold = float(summary["meta"]["graph_weak_threshold"])

    lines: list[str] = []
    lines.append("# Prefix Collaborative Consistency Analysis")
    lines.append("")
    lines.append("## Conclusion")
    lines.append("")
    lines.append(
        f"- This diagnostic treats multi-leaf `l2` prefixes as potentially useful if they are collaboratively consistent, not automatically bad."
    )
    lines.append(
        f"- `graph-weak threshold`（图弱连接阈值） is `{weak_threshold:.6f}`; crowded prefixes are split into `consistent`, `mixed`, and `inconsistent` by intra-prefix graph support."
    )
    lines.append(
        f"- Test-weighted `consistent crowded`（协同一致拥挤前缀） fraction: `{base['test_fraction_consistent_crowded']:.4f} -> {comp['test_fraction_consistent_crowded']:.4f}`."
    )
    lines.append(
        f"- Test-weighted `inconsistent crowded`（协同不一致拥挤前缀） fraction: `{base['test_fraction_inconsistent_crowded']:.4f} -> {comp['test_fraction_inconsistent_crowded']:.4f}`."
    )
    lines.append(
        f"- Test-weighted mean prefix graph affinity（测试加权平均前缀图亲和度）: `{base['test_weighted_mean_prefix_graph_affinity']:.6f} -> {comp['test_weighted_mean_prefix_graph_affinity']:.6f}`."
    )
    lines.append(
        f"- Test-weighted moved to `consistent crowded`（移入协同一致拥挤前缀）: `{comparison['test_weighted']['moved_to_consistent_crowded_fraction']:.4f}`; "
        f"moved to `inconsistent crowded`（移入协同不一致拥挤前缀）: `{comparison['test_weighted']['moved_to_inconsistent_crowded_fraction']:.4f}`."
    )
    lines.append("")
    lines.append("## Test-Weighted Summary")
    lines.append("")
    lines.append("| Metric | Baseline | Compare | Delta |")
    lines.append("|---|---:|---:|---:|")
    for key, label in [
        ("test_weighted_mean_leaf_count", "Mean target leaf count"),
        ("test_weighted_mean_prefix_graph_affinity", "Mean target prefix graph affinity"),
        ("test_weighted_mean_prefix_semantic_sim", "Mean target prefix semantic sim"),
        ("test_fraction_consistent_crowded", "Fraction targets in consistent crowded prefixes"),
        ("test_fraction_mixed_crowded", "Fraction targets in mixed crowded prefixes"),
        ("test_fraction_inconsistent_crowded", "Fraction targets in inconsistent crowded prefixes"),
        ("test_fraction_singleton", "Fraction targets in singleton prefixes"),
    ]:
        b = float(base[key])
        c = float(comp[key])
        lines.append(f"| {label} | {b:.6f} | {c:.6f} | {c - b:+.6f} |")
    lines.append("")
    lines.append("## Movement Summary")
    lines.append("")
    lines.append(
        f"- Test targets moved to `consistent crowded`（协同一致拥挤前缀）: `{format_pct(comparison['test_weighted']['moved_to_consistent_crowded_fraction'])}`."
    )
    lines.append(
        f"- Test targets moved to `inconsistent crowded`（协同不一致拥挤前缀）: `{format_pct(comparison['test_weighted']['moved_to_inconsistent_crowded_fraction'])}`."
    )
    lines.append(
        f"- Test targets moved from `inconsistent` to `consistent`（从协同不一致转为协同一致）: `{format_pct(comparison['test_weighted']['moved_from_inconsistent_to_consistent_fraction'])}`."
    )
    lines.append(
        f"- Test-weighted mean delta of leaf count（测试加权叶子数变化）: `{comparison['test_weighted']['mean_delta_leaf_count']:+.6f}`."
    )
    lines.append(
        f"- Test-weighted mean delta of prefix graph affinity（测试加权前缀图亲和度变化）: `{comparison['test_weighted']['mean_delta_prefix_graph_affinity']:+.6f}`."
    )
    lines.append("")
    lines.append("## Compare Top Consistent Crowded Prefixes")
    lines.append("")
    lines.extend(render_prefix_rows(comp["top_consistent_crowded_prefixes"]))
    lines.append("")
    lines.append("## Compare Top Inconsistent Crowded Prefixes")
    lines.append("")
    lines.extend(render_prefix_rows(comp["top_inconsistent_crowded_prefixes"]))
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    ensure_parent(output_json)
    ensure_parent(output_md)

    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)
    n_items = infer_num_items(train_df, test_df)
    views = load_base_views(args)
    combined_graph = build_combined_graph_affinity(views)
    positive_affinities = combined_graph.data[combined_graph.data > 0.0]
    weak_threshold = (
        float(np.quantile(positive_affinities, args.graph_weak_quantile))
        if positive_affinities.size
        else 0.0
    )

    semantic_embeddings = load_semantic_embeddings(args.semantic_embedding_path)
    if semantic_embeddings is None:
        raise ValueError("semantic embeddings are required")
    semantic_embeddings = semantic_embeddings[:n_items].astype(np.float32)
    norms = np.linalg.norm(semantic_embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    semantic_embeddings = semantic_embeddings / norms

    item_users = build_item_user_sets(train_df, n_items=n_items)
    titles = load_titles(Path(args.item_json))
    test_target_counts = test_df["item_id"].astype(int).value_counts().to_dict()

    baseline_index = load_index(Path(args.baseline_index))
    compare_index = load_index(Path(args.compare_index))

    baseline_rows, baseline_item_stats = build_prefix_rows(
        system_label="baseline",
        index_map=baseline_index,
        combined_graph=combined_graph,
        semantic_embeddings=semantic_embeddings,
        item_users=item_users,
        titles=titles,
        test_target_counts=test_target_counts,
        weak_threshold=weak_threshold,
        consistent_threshold=args.consistent_strong_frac_threshold,
        mixed_threshold=args.mixed_positive_frac_threshold,
    )
    compare_rows, compare_item_stats = build_prefix_rows(
        system_label="compare",
        index_map=compare_index,
        combined_graph=combined_graph,
        semantic_embeddings=semantic_embeddings,
        item_users=item_users,
        titles=titles,
        test_target_counts=test_target_counts,
        weak_threshold=weak_threshold,
        consistent_threshold=args.consistent_strong_frac_threshold,
        mixed_threshold=args.mixed_positive_frac_threshold,
    )

    summary = {
        "meta": {
            "baseline_index": args.baseline_index,
            "compare_index": args.compare_index,
            "train_csv": args.train_csv,
            "test_csv": args.test_csv,
            "item_json": args.item_json,
            "semantic_embedding_path": args.semantic_embedding_path,
            "graph_weak_quantile": float(args.graph_weak_quantile),
            "graph_weak_threshold": weak_threshold,
            "consistent_strong_frac_threshold": float(args.consistent_strong_frac_threshold),
            "mixed_positive_frac_threshold": float(args.mixed_positive_frac_threshold),
        },
        "baseline": summarize_system(baseline_rows, baseline_item_stats, test_df),
        "compare": summarize_system(compare_rows, compare_item_stats, test_df),
        "comparison": build_comparison(baseline_item_stats, compare_item_stats, test_df),
    }
    output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    output_md.write_text(render_markdown(summary), encoding="utf-8")
    print(f"Wrote JSON: {output_json}")
    print(f"Wrote Markdown: {output_md}")


if __name__ == "__main__":
    main()
