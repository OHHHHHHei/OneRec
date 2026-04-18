#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from onerec.experiments.mgr_sid.graph_bank import infer_num_items, row_normalize
from onerec.experiments.mgr_sid.paper_transplants import (
    build_semantic_knn_graph,
    keep_topk_per_row,
    load_semantic_embeddings,
)
from onerec.experiments.mgr_sid.transplanted_graph_bank import build_transplanted_graph_bank


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build R700a semantic-collaborative intersection graph sources."
    )
    parser.add_argument(
        "--train-csv",
        default="/home/leejt/OneRec/data/Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.csv",
    )
    parser.add_argument(
        "--test-csv",
        default="/home/leejt/OneRec/data/Amazon/test/Industrial_and_Scientific_5_2016-10-2018-11.csv",
    )
    parser.add_argument(
        "--semantic-embedding-path",
        default="/home/leejt/OneRec/data/Amazon/index/Industrial_and_Scientific.emb-qwen-td.npy",
    )
    parser.add_argument(
        "--output-dir",
        default="/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-18_mgr_sid_r700_semantic_collab_intersection_industrial",
    )
    parser.add_argument("--tag", default="R700a")
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
    parser.add_argument("--local-multihop-alpha", type=float, default=0.35)
    parser.add_argument("--local-multihop-max-hop", type=int, default=2)
    parser.add_argument("--fagsp-cascade-high-rank", type=int, default=16)
    parser.add_argument("--fagsp-cascade-low-rank", type=int, default=32)
    parser.add_argument("--fagsp-cascade-support-quantile", type=float, default=0.8)
    parser.add_argument("--fagsp-cascade-boost-alpha", type=float, default=0.5)
    parser.add_argument("--mgdcf-keep-ratio", type=float, default=0.1)
    parser.add_argument("--mgdcf-binarize-edges", action="store_true")
    parser.add_argument("--coarse-view-name", default="coarse_purified")
    parser.add_argument("--mid-view-name", default="local_multihop")
    parser.add_argument("--semantic-topk", type=int, default=64)
    parser.add_argument("--l1-semantic-topk", type=int, default=32)
    parser.add_argument("--l1-topk", type=int, default=16)
    parser.add_argument("--l2-positive-topk", type=int, default=32)
    parser.add_argument("--l2-negative-topk", type=int, default=32)
    parser.add_argument("--graph-topk", type=int, default=32)
    parser.add_argument("--weak-threshold", type=float, default=1e-8)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_views(args: argparse.Namespace, train_df: pd.DataFrame, test_df: pd.DataFrame):
    return build_transplanted_graph_bank(
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
        local_multihop_alpha=args.local_multihop_alpha,
        local_multihop_max_hop=args.local_multihop_max_hop,
        fagsp_cascade_high_rank=args.fagsp_cascade_high_rank,
        fagsp_cascade_low_rank=args.fagsp_cascade_low_rank,
        fagsp_cascade_support_quantile=args.fagsp_cascade_support_quantile,
        fagsp_cascade_boost_alpha=args.fagsp_cascade_boost_alpha,
        mgdcf_keep_ratio=args.mgdcf_keep_ratio,
        mgdcf_binarize_edges=args.mgdcf_binarize_edges,
    )


def get_row_entries(matrix: sparse.csr_matrix, row: int) -> dict[int, float]:
    start, end = matrix.indptr[row], matrix.indptr[row + 1]
    cols = matrix.indices[start:end]
    vals = matrix.data[start:end]
    return {
        int(col): float(val)
        for col, val in zip(cols, vals, strict=False)
        if int(col) != row and float(val) > 0.0
    }


def intersect_graphs(
    primary: sparse.csr_matrix,
    semantic: sparse.csr_matrix,
    *,
    row_topk: int,
    rule_name: str,
) -> tuple[sparse.csr_matrix, dict[str, float]]:
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    covered_items: set[int] = set()
    row_counts: list[int] = []

    n_items = primary.shape[0]
    for item in range(n_items):
        primary_entries = get_row_entries(primary, item)
        semantic_entries = get_row_entries(semantic, item)
        common = []
        for dst, primary_weight in primary_entries.items():
            sem_weight = semantic_entries.get(dst)
            if sem_weight is None:
                continue
            weight = float(primary_weight * sem_weight)
            if weight <= 0.0:
                continue
            common.append((dst, weight, primary_weight, sem_weight))
        common.sort(key=lambda x: x[1], reverse=True)
        common = common[:row_topk]
        row_counts.append(len(common))
        for dst, weight, _, _ in common:
            rows.append(item)
            cols.append(dst)
            data.append(weight)
            covered_items.add(item)
            covered_items.add(dst)

    graph = sparse.coo_matrix((data, (rows, cols)), shape=primary.shape, dtype=np.float32).tocsr()
    graph = row_normalize(graph).tocsr().astype(np.float32)
    summary = {
        f"{rule_name}_directed_edge_count": int(len(data)),
        f"{rule_name}_item_coverage_rate": float(len(covered_items) / max(n_items, 1)),
        f"{rule_name}_mean_row_degree": float(np.mean(row_counts)) if row_counts else 0.0,
        f"{rule_name}_median_row_degree": float(np.median(row_counts)) if row_counts else 0.0,
        f"{rule_name}_row_topk": int(row_topk),
    }
    return graph, summary


def build_semantic_near_weak_negative_pairs(
    semantic: sparse.csr_matrix,
    mid_graph: sparse.csr_matrix,
    *,
    row_topk: int,
    weak_threshold: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    records: list[dict[str, float | int | str]] = []
    covered_items: set[int] = set()
    n_items = semantic.shape[0]
    mid_graph = mid_graph.tocsr().astype(np.float32)

    for item_a in range(n_items):
        semantic_entries = sorted(
            get_row_entries(semantic, item_a).items(),
            key=lambda x: x[1],
            reverse=True,
        )[:row_topk]
        for item_b, sem_weight in semantic_entries:
            if item_a >= item_b:
                continue
            mid_affinity = float(mid_graph[item_a, item_b])
            if mid_affinity > weak_threshold:
                continue
            reliability = float(sem_weight * (1.0 - min(max(mid_affinity, 0.0), 1.0)))
            if reliability <= 0.0:
                continue
            records.append(
                {
                    "item_a": int(item_a),
                    "item_b": int(item_b),
                    "semantic_affinity": float(sem_weight),
                    "mid_affinity": mid_affinity,
                    "reliability": reliability,
                    "rule": "semantic_near_multihop_weak",
                }
            )
            covered_items.add(item_a)
            covered_items.add(item_b)

    df = pd.DataFrame.from_records(records)
    if not df.empty:
        df = df.sort_values(
            by=["reliability", "semantic_affinity"],
            ascending=[False, False],
        ).reset_index(drop=True)
    summary = {
        "l2_negative_pair_count": int(len(df)),
        "l2_negative_item_coverage_rate": float(len(covered_items) / max(n_items, 1)),
        "l2_negative_row_topk": int(row_topk),
        "l2_negative_weak_threshold": float(weak_threshold),
        "l2_negative_reliability_mean": float(df["reliability"].mean()) if not df.empty else 0.0,
        "l2_negative_semantic_affinity_mean": float(df["semantic_affinity"].mean()) if not df.empty else 0.0,
    }
    return df, summary


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)
    n_items = infer_num_items(train_df, test_df)

    views = build_views(args, train_df, test_df)
    if args.coarse_view_name not in views:
        raise KeyError(f"Unknown coarse view: {args.coarse_view_name}. Available: {sorted(views)}")
    if args.mid_view_name not in views:
        raise KeyError(f"Unknown mid view: {args.mid_view_name}. Available: {sorted(views)}")

    semantic_embeddings = load_semantic_embeddings(args.semantic_embedding_path)
    if semantic_embeddings is None:
        raise ValueError("semantic embeddings are required")
    if semantic_embeddings.shape[0] != n_items:
        semantic_embeddings = semantic_embeddings[:n_items]

    semantic_graph = build_semantic_knn_graph(semantic_embeddings, topk=args.semantic_topk)
    semantic_for_l1 = keep_topk_per_row(semantic_graph, topk=args.l1_semantic_topk).tocsr().astype(np.float32)
    semantic_for_l2 = keep_topk_per_row(semantic_graph, topk=args.semantic_topk).tocsr().astype(np.float32)

    coarse_graph = keep_topk_per_row(
        views[args.coarse_view_name].matrix,
        topk=args.graph_topk,
    ).tocsr().astype(np.float32)
    mid_graph = keep_topk_per_row(
        views[args.mid_view_name].matrix,
        topk=args.graph_topk,
    ).tocsr().astype(np.float32)

    l1_graph, l1_summary = intersect_graphs(
        coarse_graph,
        semantic_for_l1,
        row_topk=args.l1_topk,
        rule_name="l1_semantic_collab_intersection",
    )
    l2_positive_graph, l2_pos_summary = intersect_graphs(
        mid_graph,
        semantic_for_l2,
        row_topk=args.l2_positive_topk,
        rule_name="l2_semantic_multihop_positive",
    )
    negative_df, negative_summary = build_semantic_near_weak_negative_pairs(
        semantic_for_l2,
        mid_graph.maximum(mid_graph.T).tocsr().astype(np.float32),
        row_topk=args.l2_negative_topk,
        weak_threshold=args.weak_threshold,
    )

    tag = args.tag
    l1_path = output_dir / f"{tag}_l1_semantic_collab_intersection_graph.npz"
    l2_pos_path = output_dir / f"{tag}_l2_semantic_multihop_positive_graph.npz"
    l2_neg_path = output_dir / f"{tag}_l2_semantic_near_multihop_weak_pairs.csv"
    l2_neg_top_path = output_dir / f"{tag}_top_l2_semantic_near_multihop_weak_pairs.csv"
    summary_path = output_dir / f"{tag}_graph_source_summary.json"

    sparse.save_npz(l1_path, l1_graph)
    sparse.save_npz(l2_pos_path, l2_positive_graph)
    negative_df.to_csv(l2_neg_path, index=False)
    negative_df.head(200).to_csv(l2_neg_top_path, index=False)

    summary = {
        "tag": tag,
        "n_items": int(n_items),
        "coarse_view_name": args.coarse_view_name,
        "mid_view_name": args.mid_view_name,
        "semantic_topk": int(args.semantic_topk),
        "l1_semantic_topk": int(args.l1_semantic_topk),
        "graph_topk": int(args.graph_topk),
        "l1_graph_path": str(l1_path),
        "l2_positive_graph_path": str(l2_pos_path),
        "l2_negative_pair_csv": str(l2_neg_path),
        "l2_negative_pair_top_csv": str(l2_neg_top_path),
    }
    summary.update(l1_summary)
    summary.update(l2_pos_summary)
    summary.update(negative_summary)

    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
