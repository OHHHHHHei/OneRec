from __future__ import annotations

from typing import Any

import pandas as pd

from .graph_bank import (
    CommunityGraphView,
    SparseGraphView,
    build_coarse_graph,
    build_graph_bank,
    build_local_graph,
    build_popularity,
    infer_num_items,
    purify_coarse_graph,
    purify_local_graph,
)
from .paper_transplants import (
    build_fagsp_mid_view,
    build_gsprec_temporal_mid_view,
    load_semantic_embeddings,
    semantic_anchor_purify_graph,
)


def build_transplanted_graph_bank(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    history_k: int,
    coarse_min_weight: float,
    local_min_weight: float,
    n_clusters: int,
    seed: int,
    semantic_embedding_path: str | None,
    anchor_topk: int,
    semantic_mix: float,
    spectral_rank: int,
    band_low: float,
    band_high: float,
    temporal_mix: float,
) -> dict[str, SparseGraphView | CommunityGraphView]:
    views = build_graph_bank(
        train_df=train_df,
        test_df=test_df,
        history_k=history_k,
        coarse_min_weight=coarse_min_weight,
        local_min_weight=local_min_weight,
        n_clusters=n_clusters,
        seed=seed,
    )

    n_items = infer_num_items(train_df, test_df)
    popularity = build_popularity(train_df)
    coarse_raw = build_coarse_graph(train_df, n_items=n_items, history_k=history_k)
    local_raw = build_local_graph(train_df, n_items=n_items, history_k=history_k)
    coarse_purified = purify_coarse_graph(
        coarse_raw,
        popularity=popularity,
        min_weight=coarse_min_weight,
    )
    local_purified = purify_local_graph(
        local_raw,
        popularity=popularity,
        min_weight=local_min_weight,
    )

    semantic_embeddings = load_semantic_embeddings(semantic_embedding_path)
    if semantic_embeddings is not None and semantic_embeddings.shape[0] != n_items:
        semantic_embeddings = semantic_embeddings[:n_items]

    prism_anchor_coarse = semantic_anchor_purify_graph(
        coarse_purified,
        semantic_embeddings=semantic_embeddings,
        min_weight=0.0,
        anchor_topk=anchor_topk,
        semantic_mix=semantic_mix,
    )
    prism_anchor_local = semantic_anchor_purify_graph(
        local_purified,
        semantic_embeddings=semantic_embeddings,
        min_weight=0.0,
        anchor_topk=anchor_topk,
        semantic_mix=semantic_mix,
    )

    views["prism_anchor_coarse"] = SparseGraphView(
        name="prism_anchor_coarse",
        matrix=prism_anchor_coarse,
        metadata={
            "kind": "prism_anchor_coarse",
            "anchor_topk": int(anchor_topk),
            "semantic_mix": float(semantic_mix),
            "semantic_anchor_enabled": semantic_embeddings is not None,
            "nnz": int(prism_anchor_coarse.nnz),
        },
    )
    views["prism_anchor_local"] = SparseGraphView(
        name="prism_anchor_local",
        matrix=prism_anchor_local,
        metadata={
            "kind": "prism_anchor_local",
            "anchor_topk": int(anchor_topk),
            "semantic_mix": float(semantic_mix),
            "semantic_anchor_enabled": semantic_embeddings is not None,
            "nnz": int(prism_anchor_local.nnz),
        },
    )

    views["fagsp_mid_base"] = build_fagsp_mid_view(
        coarse_purified,
        name="fagsp_mid_base",
        rank=spectral_rank,
        eigen_ratio_low=band_low,
        eigen_ratio_high=band_high,
    )
    views["fagsp_mid_prism"] = build_fagsp_mid_view(
        prism_anchor_coarse,
        name="fagsp_mid_prism",
        rank=spectral_rank,
        eigen_ratio_low=band_low,
        eigen_ratio_high=band_high,
    )
    views["gsprec_mid_prism"] = build_gsprec_temporal_mid_view(
        coarse_graph=prism_anchor_coarse,
        local_graph=prism_anchor_local,
        name="gsprec_mid_prism",
        temporal_mix=temporal_mix,
        rank=spectral_rank,
        eigen_ratio_low=band_low,
        eigen_ratio_high=band_high,
    )
    return views
