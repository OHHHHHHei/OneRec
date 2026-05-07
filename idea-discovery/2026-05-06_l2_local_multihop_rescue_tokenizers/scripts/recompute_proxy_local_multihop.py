#!/usr/bin/env python
"""Recompute ambiguity prior using local_multihop instead of fagsp_mid_base."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

_archive_root = Path(
    "/home/leejt/OneRec/research-progress-log/archive/"
    "2026-04-24_mgr_sid_negative_research_archive/"
    "archived_workspace/src/"
)
sys.path.insert(0, str(_archive_root))

from onerec.experiments.mgr_sid.paper_transplants import load_semantic_embeddings
from onerec.experiments.mgr_sid.transplanted_graph_bank import build_transplanted_graph_bank


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-csv", default="data/Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.csv")
    p.add_argument("--test-csv", default="data/Amazon/test/Industrial_and_Scientific_5_2016-10-2018-11.csv")
    p.add_argument("--semantic-embedding", default="data/Amazon/index/Industrial_and_Scientific.emb-qwen-td.npy")
    p.add_argument("--output-csv", default="idea-discovery/2026-05-06_l2_local_multihop_rescue_tokenizers/pairs/proxy_item_scores_local_multihop.csv")
    p.add_argument("--semantic-topk", type=int, default=32)
    p.add_argument("--graph-topk", type=int, default=32)
    p.add_argument("--history-k", type=int, default=10)
    p.add_argument("--seed", type=int, default=2024)
    return p.parse_args()


def minmax_normalize(values):
    values = values.astype(np.float32)
    mask = np.isfinite(values)
    if not mask.any():
        return np.zeros_like(values, dtype=np.float32)
    vmin, vmax = float(values[mask].min()), float(values[mask].max())
    if vmax - vmin < 1e-12:
        return np.zeros_like(values, dtype=np.float32)
    out = np.zeros_like(values, dtype=np.float32)
    out[mask] = (values[mask] - vmin) / (vmax - vmin)
    return out


def jaccard_distance(a, b):
    sa, sb = set(int(v) for v in a.tolist()), set(int(v) for v in b.tolist())
    if not sa and not sb:
        return 0.0
    inter, union = len(sa & sb), len(sa | sb)
    return 0.0 if union == 0 else 1.0 - inter / union


def normalized_entropy(weights):
    weights = weights[weights > 0]
    if weights.size <= 1:
        return 0.0
    probs = weights / max(weights.sum(), 1e-12)
    entropy = -float(np.sum(probs * np.log(probs + 1e-12)))
    max_ent = float(np.log(len(probs)))
    return entropy / max_ent if max_ent > 0 else 0.0


def build_semantic_neighbors(embeddings, topk):
    n = embeddings.shape[0]
    nn = NearestNeighbors(n_neighbors=min(topk + 1, n), metric="cosine")
    nn.fit(embeddings)
    distances, indices = nn.kneighbors(embeddings)
    sims = np.maximum(0.0, 1.0 - distances[:, 1:]).astype(np.float32)
    return indices[:, 1:].astype(np.int32), sims


def topk_neighbors_from_sparse(matrix, topk):
    """返回 (weights[n,topk], neighbors_list[n])"""
    topk = max(1, int(topk))
    n = matrix.shape[0]
    weights = np.zeros((n, topk), dtype=np.float32)
    neighbors = []
    matrix = matrix.tocsr()
    for i in range(n):
        s, e = matrix.indptr[i], matrix.indptr[i + 1]
        row_idx, row_data = matrix.indices[s:e], matrix.data[s:e]
        if row_data.size == 0:
            neighbors.append(np.array([], dtype=np.int32))
            continue
        order = np.argsort(row_data)[::-1][:topk]
        neighbors.append(row_idx[order].astype(np.int32))
        weights[i, : len(order)] = row_data[order].astype(np.float32)
    return weights, neighbors


def main():
    args = parse_args()
    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)
    emb = load_semantic_embeddings(args.semantic_embedding)
    n_items = emb.shape[0]

    # Build graph bank, use local_multihop as mid view
    # Other graph params set to v2 defaults (only local_multihop matters for the prior)
    views = build_transplanted_graph_bank(
        train_df=train_df,
        test_df=test_df,
        history_k=args.history_k,
        coarse_min_weight=2.0,
        local_min_weight=1.0,
        n_clusters=64,
        seed=args.seed,
        semantic_embedding_path=args.semantic_embedding,
        anchor_topk=32,
        semantic_mix=0.35,
        spectral_rank=48,
        band_low=0.25,
        band_high=0.65,
        temporal_mix=0.35,
        fagsp_cascade_high_rank=16,
        fagsp_cascade_low_rank=32,
        fagsp_cascade_support_quantile=0.8,
        fagsp_cascade_boost_alpha=0.5,
        local_multihop_alpha=0.35,
        local_multihop_max_hop=2,
    )

    # CHANGE: use local_multihop instead of fagsp_mid_base
    mid_graph = views["local_multihop"].matrix
    print(f"[OK] local_multihop graph built: {mid_graph.shape}")

    # Semantic density
    sem_neighbors, sem_sims = build_semantic_neighbors(emb, topk=args.semantic_topk)
    semantic_density = sem_sims.mean(axis=1).astype(np.float32)

    # Semantic-collaborative disagreement (now vs local_multihop neighbors)
    _, graph_neighbors = topk_neighbors_from_sparse(mid_graph, topk=args.graph_topk)
    sem_collab_disagreement = np.array(
        [jaccard_distance(sem_neighbors[i], graph_neighbors[i]) for i in range(n_items)],
        dtype=np.float32,
    )

    # Graph competition (now based on local_multihop weights)
    graph_weights, _ = topk_neighbors_from_sparse(mid_graph, topk=args.graph_topk)
    graph_competition = np.array(
        [normalized_entropy(graph_weights[i]) for i in range(n_items)],
        dtype=np.float32,
    )

    # Offline combined (same formula as original)
    offline_combined = (
        minmax_normalize(semantic_density)
        + minmax_normalize(sem_collab_disagreement)
        + minmax_normalize(graph_competition)
    ) / 3.0

    df = pd.DataFrame({
        "item_id": np.arange(n_items, dtype=np.int32),
        "semantic_density": semantic_density,
        "semantic_collab_disagreement": sem_collab_disagreement,
        "graph_competition": graph_competition,
        "offline_combined": offline_combined,
    })

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)

    print(f"[OK] Wrote {len(df)} rows to {args.output_csv}")
    print(f"  offline_combined: mean={offline_combined.mean():.4f}, std={offline_combined.std():.4f}")


if __name__ == "__main__":
    main()
