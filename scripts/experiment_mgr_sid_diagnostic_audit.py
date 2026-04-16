#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
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


REPO_ROOT = Path("/home/leejt/OneRec")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit whether historical MGR-SID diagnostics align with downstream evaluate results."
    )
    parser.add_argument(
        "--experiment-results-csv",
        default=str(REPO_ROOT / "experiment_results.csv"),
    )
    parser.add_argument(
        "--train-csv",
        default=str(REPO_ROOT / "data/Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.csv"),
    )
    parser.add_argument(
        "--test-csv",
        default=str(REPO_ROOT / "data/Amazon/test/Industrial_and_Scientific_5_2016-10-2018-11.csv"),
    )
    parser.add_argument(
        "--item-json",
        default=str(REPO_ROOT / "data/Amazon/index/Industrial_and_Scientific.item.json"),
    )
    parser.add_argument(
        "--semantic-embedding-path",
        default=str(REPO_ROOT / "data/Amazon/index/Industrial_and_Scientific.emb-qwen-td.npy"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "research-progress-log/experiment_launches/2026-04-16_mgr_sid_diagnostic_audit_industrial"),
    )
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
    parser.add_argument("--pairwise-eps", type=float, default=1e-8)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_index(path: Path) -> dict[int, tuple[str, str, str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: dict[int, tuple[str, str, str]] = {}
    for item_id, tokens in raw.items():
        if not isinstance(tokens, list) or len(tokens) < 3:
            out[int(item_id)] = ("", "", "")
            continue
        out[int(item_id)] = (str(tokens[0]), str(tokens[1]), str(tokens[2]))
    return out


def resolve_index_path(row: dict[str, str]) -> Path | None:
    generated = (row.get("tokenizer_generated_index_path") or "").strip()
    sid_index = (row.get("tokenizer_sid_index_path") or "").strip()
    candidate = generated if generated and generated != "-" else sid_index
    if not candidate or candidate == "-":
        return None
    path = Path(candidate)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path if path.exists() else None


def entropy_from_counter(counter: Counter[str]) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    value = 0.0
    for count in counter.values():
        p = count / total
        value -= p * np.log2(p)
    return float(value)


def build_local_ambiguity_stats(index_map: dict[int, tuple[str, str, str]], test_df: pd.DataFrame) -> dict[str, float]:
    l2_to_items: dict[tuple[str, str], list[int]] = defaultdict(list)
    l2_to_leaf_counter: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    item_stats: dict[int, dict[str, float | int]] = {}

    for item_id, (a, b, c) in index_map.items():
        l2 = (a, b)
        l2_to_items[l2].append(item_id)
        l2_to_leaf_counter[l2][c] += 1

    l2_item_sizes: list[int] = []
    l2_leaf_sizes: list[int] = []
    l3_entropies: list[tuple[float, int]] = []
    for l2, items in l2_to_items.items():
        leaf_counter = l2_to_leaf_counter[l2]
        item_count = len(items)
        leaf_count = len(leaf_counter)
        entropy = entropy_from_counter(leaf_counter)
        l2_item_sizes.extend([item_count] * item_count)
        l2_leaf_sizes.extend([leaf_count] * item_count)
        l3_entropies.append((entropy, item_count))
        for item_id in items:
            item_stats[item_id] = {
                "l2_item_count": item_count,
                "l2_leaf_count": leaf_count,
                "l3_entropy_bits": entropy,
            }

    catalog_weighted_entropy = (
        float(sum(ent * w for ent, w in l3_entropies) / sum(w for _, w in l3_entropies))
        if l3_entropies
        else 0.0
    )

    test_leaf_counts: list[int] = []
    test_item_counts: list[int] = []
    test_entropies: list[float] = []
    for item_id in test_df["item_id"].astype(int).tolist():
        stats = item_stats.get(item_id)
        if stats is None:
            continue
        test_item_counts.append(int(stats["l2_item_count"]))
        test_leaf_counts.append(int(stats["l2_leaf_count"]))
        test_entropies.append(float(stats["l3_entropy_bits"]))

    total_test = len(test_leaf_counts)
    return {
        "catalog_item_weighted_mean_l2_item_count": float(mean(l2_item_sizes)) if l2_item_sizes else 0.0,
        "catalog_item_weighted_mean_l2_leaf_count": float(mean(l2_leaf_sizes)) if l2_leaf_sizes else 0.0,
        "catalog_item_fraction_multileaf_l2": float(sum(v > 1 for v in l2_leaf_sizes) / len(l2_leaf_sizes)) if l2_leaf_sizes else 0.0,
        "catalog_item_fraction_multileaf_l2_ge4": float(sum(v >= 4 for v in l2_leaf_sizes) / len(l2_leaf_sizes)) if l2_leaf_sizes else 0.0,
        "catalog_weighted_l3_entropy_given_l2_bits": catalog_weighted_entropy,
        "test_target_weighted_mean_l2_item_count": float(mean(test_item_counts)) if test_item_counts else 0.0,
        "test_target_weighted_median_l2_item_count": float(median(test_item_counts)) if test_item_counts else 0.0,
        "test_target_weighted_mean_l2_leaf_count": float(mean(test_leaf_counts)) if test_leaf_counts else 0.0,
        "test_target_weighted_median_l2_leaf_count": float(median(test_leaf_counts)) if test_leaf_counts else 0.0,
        "test_target_fraction_multileaf_l2": float(sum(v > 1 for v in test_leaf_counts) / total_test) if total_test else 0.0,
        "test_target_fraction_multileaf_l2_ge4": float(sum(v >= 4 for v in test_leaf_counts) / total_test) if total_test else 0.0,
        "test_target_weighted_mean_l3_entropy_bits": float(mean(test_entropies)) if test_entropies else 0.0,
    }


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


def classify_prefix(
    leaf_count: int,
    strong_pair_fraction: float,
    positive_pair_fraction: float,
    consistent_threshold: float = 0.5,
    mixed_threshold: float = 0.5,
) -> str:
    if leaf_count <= 1:
        return "singleton"
    if strong_pair_fraction >= consistent_threshold:
        return "consistent_crowded"
    if positive_pair_fraction >= mixed_threshold:
        return "mixed_crowded"
    return "inconsistent_crowded"


def build_prefix_consistency_stats(
    index_map: dict[int, tuple[str, str, str]],
    combined_graph: sparse.csr_matrix,
    semantic_embeddings: np.ndarray,
    item_users: list[set[str]],
    test_df: pd.DataFrame,
    weak_threshold: float,
) -> dict[str, float]:
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for item_id, (l1, l2, _) in index_map.items():
        groups[(l1, l2)].append(int(item_id))

    prefix_rows: list[dict[str, Any]] = []
    item_stats: dict[int, dict[str, Any]] = {}
    test_target_counts = test_df["item_id"].astype(int).value_counts().to_dict()

    for prefix_tokens, items in groups.items():
        items = sorted(items)
        leaf_count = len({index_map[item_id][2] for item_id in items})
        pair_count = 0
        positive_pairs = 0
        strong_pairs = 0
        graph_values: list[float] = []
        semantic_values: list[float] = []
        overlap_values: list[float] = []
        for left, right in combinations(items, 2):
            pair_count += 1
            graph_affinity = float(combined_graph[left, right])
            semantic_sim = float(np.dot(semantic_embeddings[left], semantic_embeddings[right]))
            overlap = user_overlap(item_users, left, right)
            graph_values.append(graph_affinity)
            semantic_values.append(semantic_sim)
            overlap_values.append(overlap)
            if graph_affinity > 0.0:
                positive_pairs += 1
                if graph_affinity > weak_threshold:
                    strong_pairs += 1
        positive_pair_fraction = float(positive_pairs / pair_count) if pair_count else 0.0
        strong_pair_fraction = float(strong_pairs / pair_count) if pair_count else 0.0
        prefix_type = classify_prefix(
            leaf_count=leaf_count,
            strong_pair_fraction=strong_pair_fraction,
            positive_pair_fraction=positive_pair_fraction,
        )
        row = {
            "item_count": len(items),
            "leaf_count": leaf_count,
            "test_weight": int(sum(test_target_counts.get(item_id, 0) for item_id in items)),
            "mean_graph_affinity": float(mean(graph_values)) if graph_values else 0.0,
            "mean_semantic_sim": float(mean(semantic_values)) if semantic_values else 0.0,
            "mean_user_overlap": float(mean(overlap_values)) if overlap_values else 0.0,
            "strong_pair_fraction": strong_pair_fraction,
            "positive_pair_fraction": positive_pair_fraction,
            "prefix_type": prefix_type,
        }
        prefix_rows.append(row)
        for item_id in items:
            item_stats[item_id] = {
                "leaf_count": leaf_count,
                "mean_graph_affinity": row["mean_graph_affinity"],
                "mean_semantic_sim": row["mean_semantic_sim"],
                "prefix_type": prefix_type,
            }

    item_weights = [float(row["item_count"]) for row in prefix_rows]
    total_item_weight = float(sum(item_weights)) if item_weights else 0.0
    crowded_rows = [row for row in prefix_rows if int(row["leaf_count"]) > 1]
    crowded_item_weights = [float(row["item_count"]) for row in crowded_rows]
    test_items = [int(item_id) for item_id in test_df["item_id"].astype(int).tolist() if int(item_id) in item_stats]
    total_test = len(test_items)
    type_counter = Counter(str(item_stats[item_id]["prefix_type"]) for item_id in test_items)

    def item_fraction(prefix_type: str) -> float:
        if total_item_weight <= 0:
            return 0.0
        return float(sum(float(row["item_count"]) for row in prefix_rows if str(row["prefix_type"]) == prefix_type) / total_item_weight)

    def weighted_mean(rows: list[dict[str, Any]], key: str, weights: list[float]) -> float:
        if not rows or not weights:
            return 0.0
        denom = float(sum(weights))
        if denom <= 0:
            return 0.0
        numer = sum(float(row[key]) * weight for row, weight in zip(rows, weights, strict=False))
        return float(numer / denom)

    return {
        "prefix_item_fraction_singleton": item_fraction("singleton"),
        "prefix_item_fraction_consistent_crowded": item_fraction("consistent_crowded"),
        "prefix_item_fraction_mixed_crowded": item_fraction("mixed_crowded"),
        "prefix_item_fraction_inconsistent_crowded": item_fraction("inconsistent_crowded"),
        "prefix_item_weighted_mean_leaf_count": weighted_mean(prefix_rows, "leaf_count", item_weights),
        "prefix_item_weighted_mean_graph_affinity": weighted_mean(prefix_rows, "mean_graph_affinity", item_weights),
        "prefix_item_weighted_mean_semantic_sim": weighted_mean(prefix_rows, "mean_semantic_sim", item_weights),
        "prefix_item_weighted_mean_crowded_strong_pair_fraction": weighted_mean(crowded_rows, "strong_pair_fraction", crowded_item_weights),
        "prefix_test_fraction_singleton": float(type_counter["singleton"] / total_test) if total_test else 0.0,
        "prefix_test_fraction_consistent_crowded": float(type_counter["consistent_crowded"] / total_test) if total_test else 0.0,
        "prefix_test_fraction_mixed_crowded": float(type_counter["mixed_crowded"] / total_test) if total_test else 0.0,
        "prefix_test_fraction_inconsistent_crowded": float(type_counter["inconsistent_crowded"] / total_test) if total_test else 0.0,
        "prefix_test_weighted_mean_leaf_count": float(mean(int(item_stats[item_id]["leaf_count"]) for item_id in test_items)) if test_items else 0.0,
        "prefix_test_weighted_mean_graph_affinity": float(mean(float(item_stats[item_id]["mean_graph_affinity"]) for item_id in test_items)) if test_items else 0.0,
        "prefix_test_weighted_mean_semantic_sim": float(mean(float(item_stats[item_id]["mean_semantic_sim"]) for item_id in test_items)) if test_items else 0.0,
    }


def load_experiment_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def aggregate_downstream_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        ndcg10 = (row.get("ndcg_at_10") or "").strip()
        if row.get("dataset_key") != "industrial" or not ndcg10 or ndcg10 == "-":
            continue
        stage = (row.get("stage") or "").strip()
        if stage not in {"sft_eval", "rl_eval"}:
            continue
        key = (
            stage,
            (row.get("title_history2sid_enabled") or "").strip(),
            (row.get("alignment_enabled") or "").strip(),
            (row.get("description_task_probability") or "").strip(),
            (row.get("tokenizer_record_id") or "").strip(),
            (row.get("tokenizer_variant") or "").strip(),
        )
        grouped[key].append(row)

    out: list[dict[str, Any]] = []
    for key, group in grouped.items():
        stage, title_on, align_on, desc_p, tokenizer_record_id, tokenizer_variant = key
        ndcgs = [float(row["ndcg_at_10"]) for row in group]
        hrs = [float(row["hr_at_10"]) for row in group if (row.get("hr_at_10") or "").strip() not in {"", "-"}]
        out.append(
            {
                "stage": stage,
                "title_history2sid_enabled": title_on,
                "alignment_enabled": align_on,
                "description_task_probability": desc_p,
                "tokenizer_record_id": tokenizer_record_id,
                "tokenizer_variant": tokenizer_variant,
                "downstream_run_count": len(group),
                "mean_ndcg_at_10": float(mean(ndcgs)),
                "mean_hr_at_10": float(mean(hrs)) if hrs else 0.0,
                "record_ids": [row["record_id"] for row in group],
            }
        )
    return out


def build_tokenizer_lookup(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["record_id"]: row for row in rows if row.get("stage") == "tokenizer"}


def compare_metric(metric_a: float, metric_b: float, lower_is_better: bool, eps: float) -> int:
    if abs(metric_a - metric_b) <= eps:
        return 0
    if lower_is_better:
        return 1 if metric_a < metric_b else -1
    return 1 if metric_a > metric_b else -1


def audit_metric(
    entries: list[dict[str, Any]],
    metric_name: str,
    score_name: str,
    lower_is_better: bool,
    eps: float,
) -> dict[str, Any]:
    total_pairs = 0
    usable_pairs = 0
    consistent_pairs = 0
    contradiction_rows: list[dict[str, Any]] = []

    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in entries:
        key = (
            row["stage"],
            row["title_history2sid_enabled"],
            row["alignment_enabled"],
            row["description_task_probability"],
        )
        grouped[key].append(row)

    for key, group in grouped.items():
        if len(group) < 2:
            continue
        for left, right in combinations(group, 2):
            total_pairs += 1
            downstream_cmp = compare_metric(
                float(left[score_name]),
                float(right[score_name]),
                lower_is_better=False,
                eps=eps,
            )
            metric_cmp = compare_metric(
                float(left[metric_name]),
                float(right[metric_name]),
                lower_is_better=lower_is_better,
                eps=eps,
            )
            if downstream_cmp == 0 or metric_cmp == 0:
                continue
            usable_pairs += 1
            if downstream_cmp == metric_cmp:
                consistent_pairs += 1
            else:
                contradiction_rows.append(
                    {
                        "group": key,
                        "left_tokenizer": left["tokenizer_record_id"],
                        "right_tokenizer": right["tokenizer_record_id"],
                        "left_downstream": float(left[score_name]),
                        "right_downstream": float(right[score_name]),
                        "left_metric": float(left[metric_name]),
                        "right_metric": float(right[metric_name]),
                    }
                )

    return {
        "metric_name": metric_name,
        "score_name": score_name,
        "lower_is_better": lower_is_better,
        "total_pairs": total_pairs,
        "usable_pairs": usable_pairs,
        "consistent_pairs": consistent_pairs,
        "pairwise_consistency": float(consistent_pairs / usable_pairs) if usable_pairs else 0.0,
        "false_positive_pairs": len(contradiction_rows),
        "false_positive_rate": float(len(contradiction_rows) / usable_pairs) if usable_pairs else 0.0,
        "top_contradictions": contradiction_rows[:10],
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# MGR-SID Diagnostic Audit（MGR-SID 诊断脚本审计）")
    lines.append("")
    lines.append("## Scope（范围）")
    lines.append("")
    lines.append(f"- audit dataset（审计数据集）: `Industrial_and_Scientific`")
    lines.append(f"- tokenizer entries with metrics（带结构指标的分词器数量）: `{payload['tokenizer_metric_count']}`")
    lines.append(f"- downstream comparable entries（可比下游条目数）: `{payload['downstream_entry_count']}`")
    lines.append(f"- comparable groups（可比组数）: `{payload['group_count']}`")
    lines.append("")
    lines.append("## Main Verdicts（主要结论）")
    lines.append("")
    for row in payload["headline_rows"]:
        lines.append(
            f"- `{row['metric_name']}` vs `{row['score_name']}`: "
            f"pairwise consistency（成对一致率）=`{row['pairwise_consistency']:.4f}`, "
            f"usable_pairs（有效成对数）=`{row['usable_pairs']}`"
        )
    lines.append("")
    lines.append("## Metrics Table（指标表）")
    lines.append("")
    lines.append("| Metric | Score | Direction | Usable Pairs | Consistency | False Positive Rate |")
    lines.append("|---|---|---|---:|---:|---:|")
    for row in payload["audit_rows"]:
        direction = "lower_better" if row["lower_is_better"] else "higher_better"
        lines.append(
            f"| `{row['metric_name']}` | `{row['score_name']}` | `{direction}` | "
            f"{row['usable_pairs']} | {row['pairwise_consistency']:.4f} | {row['false_positive_rate']:.4f} |"
        )
    lines.append("")
    lines.append("## Tokenizer Metrics（分词器结构指标）")
    lines.append("")
    lines.append("| Tokenizer | Collision | Test Mean L2 Leaves | Test Multi-Leaf | Test Entropy | Consistent Crowded | Inconsistent Crowded |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in payload["tokenizer_metric_rows"]:
        lines.append(
            f"| `{row['tokenizer_record_id']}` | {row['tokenizer_generated_collision_rate']:.6f} | "
            f"{row['test_target_weighted_mean_l2_leaf_count']:.6f} | "
            f"{row['test_target_fraction_multileaf_l2']:.6f} | "
            f"{row['test_target_weighted_mean_l3_entropy_bits']:.6f} | "
            f"{row['prefix_test_fraction_consistent_crowded']:.6f} | "
            f"{row['prefix_test_fraction_inconsistent_crowded']:.6f} |"
        )
    lines.append("")
    lines.append("## Notes（备注）")
    lines.append("")
    lines.append("- This audit only uses comparable downstream groups with the same stage and recipe（同阶段同配方）.")
    lines.append("- Posterior explainers（后验解释器） such as evaluate error analysis are excluded from this table because they already consume evaluate outputs.")
    lines.append("- Pairwise consistency（成对一致率） is the main decision criterion; low consistency means the diagnostic should not be used as a promotion gate（推进门槛）.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    experiment_rows = load_experiment_rows(Path(args.experiment_results_csv))
    tokenizer_lookup = build_tokenizer_lookup(experiment_rows)
    downstream_entries = aggregate_downstream_rows(experiment_rows)

    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)
    n_items = infer_num_items(train_df, test_df)
    semantic_embeddings = load_semantic_embeddings(args.semantic_embedding_path)
    if semantic_embeddings is None:
        raise ValueError("semantic embeddings are required")
    semantic_embeddings = semantic_embeddings[:n_items].astype(np.float32)
    norms = np.linalg.norm(semantic_embeddings, axis=1, keepdims=True)
    semantic_embeddings = semantic_embeddings / np.clip(norms, 1e-12, None)

    base_views = load_base_views(args)
    combined_graph = build_combined_graph_affinity(base_views)
    positive_affinities = combined_graph.data[combined_graph.data > 0.0]
    weak_threshold = (
        float(np.quantile(positive_affinities, args.graph_weak_quantile))
        if positive_affinities.size
        else 0.0
    )
    item_users = build_item_user_sets(train_df, n_items=n_items)

    relevant_tokenizer_ids = sorted({row["tokenizer_record_id"] for row in downstream_entries})
    tokenizer_metric_rows: list[dict[str, Any]] = []
    metric_lookup: dict[str, dict[str, Any]] = {}
    for tokenizer_id in relevant_tokenizer_ids:
        tokenizer_row = tokenizer_lookup.get(tokenizer_id)
        if tokenizer_row is None:
            continue
        index_path = resolve_index_path(tokenizer_row)
        if index_path is None:
            continue
        index_map = load_index(index_path)
        local_stats = build_local_ambiguity_stats(index_map, test_df)
        prefix_stats = build_prefix_consistency_stats(
            index_map=index_map,
            combined_graph=combined_graph,
            semantic_embeddings=semantic_embeddings,
            item_users=item_users,
            test_df=test_df,
            weak_threshold=weak_threshold,
        )
        merged = {
            "tokenizer_record_id": tokenizer_id,
            "tokenizer_variant": tokenizer_row.get("variant", ""),
            "tokenizer_generated_collision_rate": float(tokenizer_row.get("tokenizer_generated_collision_rate") or 0.0),
            "tokenizer_generated_collision_count": int(float(tokenizer_row.get("tokenizer_generated_collision_count") or 0)),
            "index_path": str(index_path),
            **local_stats,
            **prefix_stats,
        }
        tokenizer_metric_rows.append(merged)
        metric_lookup[tokenizer_id] = merged

    comparable_entries: list[dict[str, Any]] = []
    for row in downstream_entries:
        metric_row = metric_lookup.get(row["tokenizer_record_id"])
        if metric_row is None:
            continue
        comparable_entries.append({**row, **metric_row})

    group_keys = {
        (
            row["stage"],
            row["title_history2sid_enabled"],
            row["alignment_enabled"],
            row["description_task_probability"],
        )
        for row in comparable_entries
    }

    metric_specs = [
        ("tokenizer_generated_collision_rate", True),
        ("test_target_weighted_mean_l2_leaf_count", True),
        ("test_target_fraction_multileaf_l2", True),
        ("test_target_fraction_multileaf_l2_ge4", True),
        ("test_target_weighted_mean_l3_entropy_bits", True),
        ("prefix_test_fraction_consistent_crowded", False),
        ("prefix_test_fraction_inconsistent_crowded", True),
        ("prefix_test_weighted_mean_graph_affinity", False),
    ]

    audit_rows: list[dict[str, Any]] = []
    for metric_name, lower_is_better in metric_specs:
        audit_rows.append(
            audit_metric(
                entries=comparable_entries,
                metric_name=metric_name,
                score_name="mean_ndcg_at_10",
                lower_is_better=lower_is_better,
                eps=args.pairwise_eps,
            )
        )
        audit_rows.append(
            audit_metric(
                entries=comparable_entries,
                metric_name=metric_name,
                score_name="mean_hr_at_10",
                lower_is_better=lower_is_better,
                eps=args.pairwise_eps,
            )
        )

    headline_metric_names = {
        ("tokenizer_generated_collision_rate", "mean_ndcg_at_10"),
        ("test_target_weighted_mean_l2_leaf_count", "mean_ndcg_at_10"),
        ("test_target_weighted_mean_l3_entropy_bits", "mean_ndcg_at_10"),
        ("prefix_test_fraction_consistent_crowded", "mean_ndcg_at_10"),
        ("prefix_test_fraction_inconsistent_crowded", "mean_ndcg_at_10"),
    }
    headline_rows = [
        row for row in audit_rows if (row["metric_name"], row["score_name"]) in headline_metric_names
    ]
    headline_rows = sorted(headline_rows, key=lambda row: (-row["pairwise_consistency"], -row["usable_pairs"]))

    payload = {
        "tokenizer_metric_count": len(tokenizer_metric_rows),
        "downstream_entry_count": len(comparable_entries),
        "group_count": len(group_keys),
        "group_keys": sorted(group_keys),
        "tokenizer_metric_rows": sorted(tokenizer_metric_rows, key=lambda row: row["tokenizer_record_id"]),
        "downstream_rows": comparable_entries,
        "audit_rows": audit_rows,
        "headline_rows": headline_rows,
        "weak_threshold": weak_threshold,
    }

    (output_dir / "diagnostic_audit_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    pd.DataFrame(tokenizer_metric_rows).to_csv(output_dir / "tokenizer_metric_table.csv", index=False)
    pd.DataFrame(comparable_entries).to_csv(output_dir / "downstream_comparable_table.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(output_dir / "audit_metric_table.csv", index=False)
    (output_dir / "SUMMARY.md").write_text(render_markdown(payload), encoding="utf-8")
    print(output_dir / "diagnostic_audit_summary.json")
    print(output_dir / "SUMMARY.md")


if __name__ == "__main__":
    main()
