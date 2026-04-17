#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from onerec.experiments.mgr_sid.graph_bank import infer_num_items
from onerec.experiments.mgr_sid.paper_transplants import (
    build_semantic_knn_graph,
    keep_topk_per_row,
    load_semantic_embeddings,
    symmetrize_matrix,
)
from onerec.experiments.mgr_sid.transplanted_graph_bank import build_transplanted_graph_bank


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build mid-only pull/push pair source for the simplified MGR-SID branch."
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
        default="/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-16_mgr_sid_r630_mid_pull_push_industrial",
    )
    parser.add_argument("--tag", default="R630")
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
    parser.add_argument("--semantic-topk", type=int, default=32)
    parser.add_argument("--graph-topk", type=int, default=32)
    parser.add_argument("--graph-weak-quantile", type=float, default=0.25)
    parser.add_argument("--save-topn", type=int, default=200)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_semantic_pair_list(
    semantic_embeddings: np.ndarray,
    semantic_topk: int,
) -> list[tuple[int, int, float]]:
    graph = build_semantic_knn_graph(semantic_embeddings, topk=semantic_topk)
    graph = keep_topk_per_row(graph, topk=semantic_topk).tocsr()
    pair_scores: dict[tuple[int, int], float] = {}
    for row in range(graph.shape[0]):
        start, end = graph.indptr[row], graph.indptr[row + 1]
        cols = graph.indices[start:end]
        vals = graph.data[start:end]
        for col, val in zip(cols, vals, strict=False):
            a, b = sorted((int(row), int(col)))
            if a == b:
                continue
            key = (a, b)
            if float(val) > pair_scores.get(key, -1.0):
                pair_scores[key] = float(val)
    pairs = [(a, b, score) for (a, b), score in pair_scores.items()]
    pairs.sort(key=lambda item: item[2], reverse=True)
    return pairs


def load_mid_graph(args: argparse.Namespace, train_df: pd.DataFrame, test_df: pd.DataFrame):
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
        local_multihop_alpha=args.local_multihop_alpha,
        local_multihop_max_hop=args.local_multihop_max_hop,
        fagsp_cascade_high_rank=args.fagsp_cascade_high_rank,
        fagsp_cascade_low_rank=args.fagsp_cascade_low_rank,
        fagsp_cascade_support_quantile=args.fagsp_cascade_support_quantile,
        fagsp_cascade_boost_alpha=args.fagsp_cascade_boost_alpha,
        mgdcf_keep_ratio=args.mgdcf_keep_ratio,
        mgdcf_binarize_edges=args.mgdcf_binarize_edges,
    )
    if args.mid_view_name not in views:
        raise KeyError(f"Unknown mid view: {args.mid_view_name}. Available views: {sorted(views.keys())}")
    mid_matrix = keep_topk_per_row(views[args.mid_view_name].matrix, topk=args.graph_topk).tocsr().astype(np.float32)
    return symmetrize_matrix(mid_matrix).tocsr().astype(np.float32)


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

    mid_graph = load_mid_graph(args, train_df=train_df, test_df=test_df)
    semantic_pairs = build_semantic_pair_list(semantic_embeddings, semantic_topk=args.semantic_topk)

    records: list[dict[str, float | int]] = []
    positive_mid_affinities: list[float] = []
    for item_a, item_b, semantic_sim in semantic_pairs:
        mid_affinity = float(mid_graph[item_a, item_b])
        if mid_affinity > 0.0:
            positive_mid_affinities.append(mid_affinity)
        records.append(
            {
                "item_a": int(item_a),
                "item_b": int(item_b),
                "semantic_sim": float(semantic_sim),
                "mid_affinity": mid_affinity,
            }
        )

    all_pairs = pd.DataFrame.from_records(records)
    weak_threshold = (
        float(np.quantile(np.asarray(positive_mid_affinities, dtype=np.float32), args.graph_weak_quantile))
        if positive_mid_affinities
        else 0.0
    )

    if weak_threshold > 0.0:
        weak_df = all_pairs[(all_pairs["mid_affinity"] > 0.0) & (all_pairs["mid_affinity"] <= weak_threshold)].copy()
        weak_df["reliability"] = weak_df["semantic_sim"] * (
            1.0 - np.minimum(weak_df["mid_affinity"] / weak_threshold, 1.0)
        )
    else:
        weak_df = all_pairs.iloc[0:0].copy()
        weak_df["reliability"] = np.asarray([], dtype=np.float32)

    weak_df = weak_df[weak_df["reliability"] > 0.0].copy()
    weak_df["rule"] = "semantic_near_mid_graph_weak"
    weak_df["weak_threshold"] = weak_threshold
    weak_df = weak_df.sort_values(by=["reliability", "semantic_sim"], ascending=[False, False]).reset_index(drop=True)

    save_topn = max(int(args.save_topn), 1)
    tag = args.tag
    all_pairs.sort_values(by=["semantic_sim"], ascending=[False]).to_csv(
        output_dir / f"{tag}_all_semantic_pairs.csv",
        index=False,
    )
    weak_df.to_csv(output_dir / f"{tag}_all_mid_graph_weak_pairs.csv", index=False)
    weak_df.head(save_topn).to_csv(output_dir / f"{tag}_top_mid_graph_weak_pairs.csv", index=False)

    covered_items = set(weak_df["item_a"].tolist()) | set(weak_df["item_b"].tolist()) if not weak_df.empty else set()
    summary = {
        "n_items": int(n_items),
        "mid_view_name": args.mid_view_name,
        "semantic_pair_count": int(len(all_pairs)),
        "weak_pair_count": int(len(weak_df)),
        "weak_pair_item_coverage_rate": float(len(covered_items) / max(n_items, 1)),
        "weak_threshold": weak_threshold,
        "semantic_topk": int(args.semantic_topk),
        "graph_topk": int(args.graph_topk),
        "graph_weak_quantile": float(args.graph_weak_quantile),
        "tag": tag,
        "semantic_sim_mean": float(weak_df["semantic_sim"].mean()) if not weak_df.empty else 0.0,
        "mid_affinity_mean": float(weak_df["mid_affinity"].mean()) if not weak_df.empty else 0.0,
        "reliability_mean": float(weak_df["reliability"].mean()) if not weak_df.empty else 0.0,
        "output_csv": str(output_dir / f"{tag}_all_mid_graph_weak_pairs.csv"),
        "output_top_csv": str(output_dir / f"{tag}_top_mid_graph_weak_pairs.csv"),
    }
    with open(output_dir / f"{tag}_pair_source_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
