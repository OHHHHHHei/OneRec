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
    parser = argparse.ArgumentParser(description="Build a high-confidence collaborative coarse graph for SID L1.")
    parser.add_argument("--train-csv", default="/home/leejt/OneRec/data/Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.csv")
    parser.add_argument("--test-csv", default="/home/leejt/OneRec/data/Amazon/test/Industrial_and_Scientific_5_2016-10-2018-11.csv")
    parser.add_argument("--semantic-embedding-path", default="/home/leejt/OneRec/data/Amazon/index/Industrial_and_Scientific.emb-qwen-td.npy")
    parser.add_argument("--output-dir", default="/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-20_mgr_sid_highconf_l1_collab_ranking_industrial")
    parser.add_argument("--tag", default="highconf_l1_collab_ranking")
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
    parser.add_argument("--seq2g-mix-alpha", type=float, default=0.35)
    parser.add_argument("--seq2g-context-topk", type=int, default=32)
    parser.add_argument("--seq2g-candidate-topm", type=int, default=32)
    parser.add_argument("--seq2g-direct-tau", type=float, default=0.5)
    parser.add_argument("--seq2g-use-reliability", action="store_true", default=True)
    parser.add_argument("--seq2g-use-direct-weak-mask", action="store_true", default=True)
    parser.add_argument("--coarse-view-name", default="coarse_purified")
    parser.add_argument("--graph-topk", type=int, default=32)
    parser.add_argument("--semantic-topk", type=int, default=64)
    parser.add_argument("--l1-topk", type=int, default=16)
    return parser.parse_args()


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
        seq2g_mix_alpha=args.seq2g_mix_alpha,
        seq2g_context_topk=args.seq2g_context_topk,
        seq2g_candidate_topm=args.seq2g_candidate_topm,
        seq2g_direct_tau=args.seq2g_direct_tau,
        seq2g_use_reliability=args.seq2g_use_reliability,
        seq2g_use_direct_weak_mask=args.seq2g_use_direct_weak_mask,
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)
    n_items = infer_num_items(train_df, test_df)
    views = build_views(args, train_df=train_df, test_df=test_df)
    if args.coarse_view_name not in views:
        raise KeyError(f"Unknown coarse view: {args.coarse_view_name}. Available: {sorted(views)}")

    semantic_embeddings = load_semantic_embeddings(args.semantic_embedding_path)
    if semantic_embeddings is None:
        raise ValueError("semantic embeddings are required for the high-confidence L1 graph")
    semantic_embeddings = semantic_embeddings[:n_items]

    coarse = keep_topk_per_row(views[args.coarse_view_name].matrix, topk=args.graph_topk).tocsr().astype(np.float32)
    semantic = build_semantic_knn_graph(semantic_embeddings, topk=args.semantic_topk)
    semantic = keep_topk_per_row(semantic, topk=args.semantic_topk).tocsr().astype(np.float32)
    semantic = semantic.maximum(semantic.T).tocsr().astype(np.float32)

    # Collaborative edges remain the backbone; semantic kNN acts as a gate and confidence multiplier.
    highconf = coarse.multiply(semantic).tocsr().astype(np.float32)
    highconf = keep_topk_per_row(highconf, topk=args.l1_topk)
    highconf = row_normalize(highconf).tocsr().astype(np.float32)

    row_degrees = np.diff(highconf.indptr)
    covered_items = int(np.sum(row_degrees > 0))
    graph_path = output_dir / f"{args.tag}_l1_highconf_coarse_graph.npz"
    summary_path = output_dir / f"{args.tag}_l1_highconf_coarse_graph_summary.json"
    sparse.save_npz(graph_path, highconf)

    summary = {
        "tag": args.tag,
        "n_items": int(n_items),
        "coarse_view_name": args.coarse_view_name,
        "graph_topk": int(args.graph_topk),
        "semantic_topk": int(args.semantic_topk),
        "l1_topk": int(args.l1_topk),
        "l1_graph_path": str(graph_path),
        "l1_graph_nnz": int(highconf.nnz),
        "l1_graph_item_coverage_rate": float(covered_items / max(n_items, 1)),
        "l1_graph_mean_row_degree": float(np.mean(row_degrees)),
        "l1_graph_median_row_degree": float(np.median(row_degrees)),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
