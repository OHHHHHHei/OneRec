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

from onerec.experiments.mgr_sid.graph_bank import infer_num_items
from onerec.experiments.mgr_sid.paper_transplants import (
    build_semantic_knn_graph,
    keep_topk_per_row,
    load_semantic_embeddings,
)
from onerec.experiments.mgr_sid.transplanted_graph_bank import build_transplanted_graph_bank


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build semantic-near/mid-weak hard negatives for mainline collaborative-ranking SID."
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
        default="/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-18_mgr_sid_r720_l2_ranking_contrastive_industrial",
    )
    parser.add_argument("--tag", default="R720a")
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
    parser.add_argument("--mid-view-name", default="fagsp_mid_base")
    parser.add_argument("--semantic-topk", type=int, default=64)
    parser.add_argument("--graph-topk", type=int, default=32)
    parser.add_argument("--weak-threshold", type=float, default=1e-8)
    parser.add_argument("--save-topn", type=int, default=500)
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


def build_semantic_pairs(semantic_embeddings: np.ndarray, semantic_topk: int) -> list[tuple[int, int, float]]:
    semantic_graph = build_semantic_knn_graph(semantic_embeddings, topk=semantic_topk)
    semantic_graph = keep_topk_per_row(semantic_graph, topk=semantic_topk).tocsr().astype(np.float32)
    pair_scores: dict[tuple[int, int], float] = {}
    for item_a in range(semantic_graph.shape[0]):
        start, end = semantic_graph.indptr[item_a], semantic_graph.indptr[item_a + 1]
        cols = semantic_graph.indices[start:end]
        vals = semantic_graph.data[start:end]
        for item_b, value in zip(cols, vals, strict=False):
            item_b = int(item_b)
            if item_a == item_b:
                continue
            key = tuple(sorted((int(item_a), item_b)))
            score = float(value)
            if score > pair_scores.get(key, -1.0):
                pair_scores[key] = score
    pairs = [(item_a, item_b, score) for (item_a, item_b), score in pair_scores.items()]
    pairs.sort(key=lambda row: row[2], reverse=True)
    return pairs


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)
    n_items = infer_num_items(train_df, test_df)

    semantic_embeddings = load_semantic_embeddings(args.semantic_embedding_path)
    if semantic_embeddings is None:
        raise ValueError("semantic_embedding_path is required")
    semantic_embeddings = semantic_embeddings[:n_items]

    views = build_views(args, train_df=train_df, test_df=test_df)
    if args.mid_view_name not in views:
        raise KeyError(f"Unknown mid view: {args.mid_view_name}. Available views: {sorted(views)}")
    mid_graph = keep_topk_per_row(
        views[args.mid_view_name].matrix,
        topk=args.graph_topk,
    ).tocsr().astype(np.float32)
    mid_graph = mid_graph.maximum(mid_graph.T).tocsr().astype(np.float32)

    semantic_pairs = build_semantic_pairs(semantic_embeddings, semantic_topk=args.semantic_topk)
    records: list[dict[str, float | int | str]] = []
    covered_items: set[int] = set()
    mid_affinities: list[float] = []
    reliabilities: list[float] = []

    weak_threshold = float(args.weak_threshold)
    for item_a, item_b, semantic_affinity in semantic_pairs:
        mid_affinity = float(mid_graph[item_a, item_b])
        if mid_affinity > weak_threshold:
            continue
        if weak_threshold > 0.0:
            weakness = 1.0 - min(max(mid_affinity, 0.0) / weak_threshold, 1.0)
        else:
            weakness = 1.0 if mid_affinity <= 0.0 else 0.0
        reliability = float(semantic_affinity * weakness)
        if reliability <= 0.0:
            continue
        records.append(
            {
                "item_a": int(item_a),
                "item_b": int(item_b),
                "semantic_affinity": float(semantic_affinity),
                "mid_affinity": mid_affinity,
                "reliability": reliability,
                "rule": "semantic_near_mid_weak",
                "weak_threshold": weak_threshold,
            }
        )
        covered_items.add(item_a)
        covered_items.add(item_b)
        mid_affinities.append(mid_affinity)
        reliabilities.append(reliability)

    negative_df = pd.DataFrame.from_records(records)
    if not negative_df.empty:
        negative_df = negative_df.sort_values(
            by=["reliability", "semantic_affinity"],
            ascending=[False, False],
        ).reset_index(drop=True)

    tag = args.tag
    negative_csv = output_dir / f"{tag}_all_semantic_near_mid_weak_pairs.csv"
    top_negative_csv = output_dir / f"{tag}_top_semantic_near_mid_weak_pairs.csv"
    summary_path = output_dir / f"{tag}_ranking_pair_source_summary.json"
    negative_df.to_csv(negative_csv, index=False)
    negative_df.head(max(int(args.save_topn), 1)).to_csv(top_negative_csv, index=False)

    summary = {
        "tag": tag,
        "n_items": int(n_items),
        "mid_view_name": args.mid_view_name,
        "semantic_topk": int(args.semantic_topk),
        "graph_topk": int(args.graph_topk),
        "weak_threshold": weak_threshold,
        "semantic_pair_count": int(len(semantic_pairs)),
        "negative_pair_count": int(len(negative_df)),
        "negative_item_coverage_rate": float(len(covered_items) / max(n_items, 1)),
        "mid_affinity_mean": float(np.mean(mid_affinities)) if mid_affinities else 0.0,
        "reliability_mean": float(np.mean(reliabilities)) if reliabilities else 0.0,
        "negative_pair_csv": str(negative_csv),
        "negative_pair_top_csv": str(top_negative_csv),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
