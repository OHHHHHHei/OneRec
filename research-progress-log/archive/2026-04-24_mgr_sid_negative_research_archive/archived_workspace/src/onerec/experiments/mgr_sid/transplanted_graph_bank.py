from __future__ import annotations

from typing import Any

import pandas as pd

from .graph_bank import (
    CommunityGraphView,
    SparseGraphView,
    build_coarse_graph,
    build_graph_bank,
    build_local_graph,
    build_multi_hop_transition_view,
    build_popularity,
    build_direct_support_graph,
    build_seq2graph_context_matrix,
    build_seq2graph_reliability,
    build_seq2graph_rescue_graph,
    infer_num_items,
    purify_coarse_graph,
    purify_local_graph,
)
from .paper_transplants import (
    build_fagsp_cascade_mid_view,
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
    fagsp_cascade_high_rank: int,
    fagsp_cascade_low_rank: int,
    fagsp_cascade_support_quantile: float,
    fagsp_cascade_boost_alpha: float,
    local_multihop_alpha: float = 0.35,
    local_multihop_max_hop: int = 2,
    local_multihop_base_weight: float = 1.0,
    mgdcf_keep_ratio: float = 0.1,
    mgdcf_binarize_edges: bool = True,
    seq2g_mix_alpha: float = 0.35,
    seq2g_context_topk: int = 32,
    seq2g_candidate_topm: int = 32,
    seq2g_direct_tau: float = 0.5,
    seq2g_use_reliability: bool = True,
    seq2g_use_direct_weak_mask: bool = True,
) -> dict[str, SparseGraphView | CommunityGraphView]:
    views = build_graph_bank(
        train_df=train_df,
        test_df=test_df,
        history_k=history_k,
        coarse_min_weight=coarse_min_weight,
        local_min_weight=local_min_weight,
        n_clusters=n_clusters,
        seed=seed,
        mgdcf_keep_ratio=mgdcf_keep_ratio,
        mgdcf_binarize_edges=mgdcf_binarize_edges,
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
    direct_support = build_direct_support_graph(coarse_raw=coarse_raw, local_raw=local_raw)
    coarse_mgdcf = views["coarse_mgdcf"].matrix
    seq2g_context = build_seq2graph_context_matrix(
        local_raw=local_raw,
        context_topk=seq2g_context_topk,
        candidate_topm=seq2g_candidate_topm,
    )
    seq2g_reliability = build_seq2graph_reliability(
        local_raw=local_raw,
        context_affinity=seq2g_context,
    )
    coarse_seq2g_ctx_only, seq2g_ctx_only_rescue = build_seq2graph_rescue_graph(
        coarse_graph=coarse_purified,
        context_affinity=seq2g_context,
        mix_alpha=seq2g_mix_alpha,
        context_topk=seq2g_context_topk,
        reliability=None,
        direct_support=direct_support,
        direct_tau=seq2g_direct_tau,
        use_reliability=False,
        use_direct_weak_mask=False,
    )
    coarse_seq2g_rel, seq2g_rel_rescue = build_seq2graph_rescue_graph(
        coarse_graph=coarse_purified,
        context_affinity=seq2g_context,
        mix_alpha=seq2g_mix_alpha,
        context_topk=seq2g_context_topk,
        reliability=seq2g_reliability,
        direct_support=direct_support,
        direct_tau=seq2g_direct_tau,
        use_reliability=True,
        use_direct_weak_mask=False,
    )
    coarse_seq2g_rel_masked, seq2g_rel_masked_rescue = build_seq2graph_rescue_graph(
        coarse_graph=coarse_purified,
        context_affinity=seq2g_context,
        mix_alpha=seq2g_mix_alpha,
        context_topk=seq2g_context_topk,
        reliability=seq2g_reliability,
        direct_support=direct_support,
        direct_tau=seq2g_direct_tau,
        use_reliability=True,
        use_direct_weak_mask=True,
    )
    coarse_seq2g_rescue, seq2g_configured_rescue = build_seq2graph_rescue_graph(
        coarse_graph=coarse_purified,
        context_affinity=seq2g_context,
        mix_alpha=seq2g_mix_alpha,
        context_topk=seq2g_context_topk,
        reliability=seq2g_reliability,
        direct_support=direct_support,
        direct_tau=seq2g_direct_tau,
        use_reliability=seq2g_use_reliability,
        use_direct_weak_mask=seq2g_use_direct_weak_mask,
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
    views["coarse_seq2g_ctx_only"] = SparseGraphView(
        name="coarse_seq2g_ctx_only",
        matrix=coarse_seq2g_ctx_only,
        metadata={
            "kind": "seq2graph_lite_coarse",
            "seq2g_variant": "ctx_only",
            "seq2g_mix_alpha": float(seq2g_mix_alpha),
            "seq2g_context_topk": int(seq2g_context_topk),
            "seq2g_candidate_topm": int(seq2g_candidate_topm),
            "seq2g_direct_tau": float(seq2g_direct_tau),
            "rescue_nnz": int(seq2g_ctx_only_rescue.nnz),
            "nnz": int(coarse_seq2g_ctx_only.nnz),
        },
    )
    views["coarse_seq2g_rel"] = SparseGraphView(
        name="coarse_seq2g_rel",
        matrix=coarse_seq2g_rel,
        metadata={
            "kind": "seq2graph_lite_coarse",
            "seq2g_variant": "rel",
            "seq2g_mix_alpha": float(seq2g_mix_alpha),
            "seq2g_context_topk": int(seq2g_context_topk),
            "seq2g_candidate_topm": int(seq2g_candidate_topm),
            "seq2g_direct_tau": float(seq2g_direct_tau),
            "rescue_nnz": int(seq2g_rel_rescue.nnz),
            "nnz": int(coarse_seq2g_rel.nnz),
        },
    )
    views["coarse_seq2g_rel_masked"] = SparseGraphView(
        name="coarse_seq2g_rel_masked",
        matrix=coarse_seq2g_rel_masked,
        metadata={
            "kind": "seq2graph_lite_coarse",
            "seq2g_variant": "rel_masked",
            "seq2g_mix_alpha": float(seq2g_mix_alpha),
            "seq2g_context_topk": int(seq2g_context_topk),
            "seq2g_candidate_topm": int(seq2g_candidate_topm),
            "seq2g_direct_tau": float(seq2g_direct_tau),
            "rescue_nnz": int(seq2g_rel_masked_rescue.nnz),
            "nnz": int(coarse_seq2g_rel_masked.nnz),
        },
    )
    views["coarse_seq2g_rescue"] = SparseGraphView(
        name="coarse_seq2g_rescue",
        matrix=coarse_seq2g_rescue,
        metadata={
            "kind": "seq2graph_lite_coarse",
            "seq2g_variant": "configured",
            "seq2g_mix_alpha": float(seq2g_mix_alpha),
            "seq2g_context_topk": int(seq2g_context_topk),
            "seq2g_candidate_topm": int(seq2g_candidate_topm),
            "seq2g_direct_tau": float(seq2g_direct_tau),
            "seq2g_use_reliability": bool(seq2g_use_reliability),
            "seq2g_use_direct_weak_mask": bool(seq2g_use_direct_weak_mask),
            "rescue_nnz": int(seq2g_configured_rescue.nnz),
            "nnz": int(coarse_seq2g_rescue.nnz),
        },
    )

    views["fagsp_mid_base"] = build_fagsp_mid_view(
        coarse_purified,
        name="fagsp_mid_base",
        rank=spectral_rank,
        eigen_ratio_low=band_low,
        eigen_ratio_high=band_high,
    )
    views["fagsp_mid_mgdcf"] = build_fagsp_mid_view(
        coarse_mgdcf,
        name="fagsp_mid_mgdcf",
        rank=spectral_rank,
        eigen_ratio_low=band_low,
        eigen_ratio_high=band_high,
    )
    views["fagsp_mid_seq2g_ctx_only"] = build_fagsp_mid_view(
        coarse_seq2g_ctx_only,
        name="fagsp_mid_seq2g_ctx_only",
        rank=spectral_rank,
        eigen_ratio_low=band_low,
        eigen_ratio_high=band_high,
    )
    views["fagsp_mid_seq2g_rel"] = build_fagsp_mid_view(
        coarse_seq2g_rel,
        name="fagsp_mid_seq2g_rel",
        rank=spectral_rank,
        eigen_ratio_low=band_low,
        eigen_ratio_high=band_high,
    )
    views["fagsp_mid_seq2g_rel_masked"] = build_fagsp_mid_view(
        coarse_seq2g_rel_masked,
        name="fagsp_mid_seq2g_rel_masked",
        rank=spectral_rank,
        eigen_ratio_low=band_low,
        eigen_ratio_high=band_high,
    )
    views["fagsp_mid_seq2g_rescue"] = build_fagsp_mid_view(
        coarse_seq2g_rescue,
        name="fagsp_mid_seq2g_rescue",
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
    views["fagsp_mid_cascade"] = build_fagsp_cascade_mid_view(
        coarse_purified,
        name="fagsp_mid_cascade",
        high_rank=fagsp_cascade_high_rank,
        low_rank=fagsp_cascade_low_rank,
        support_quantile=fagsp_cascade_support_quantile,
        boost_alpha=fagsp_cascade_boost_alpha,
    )
    views["fagsp_mid_cascade_prism"] = build_fagsp_cascade_mid_view(
        prism_anchor_coarse,
        name="fagsp_mid_cascade_prism",
        high_rank=fagsp_cascade_high_rank,
        low_rank=fagsp_cascade_low_rank,
        support_quantile=fagsp_cascade_support_quantile,
        boost_alpha=fagsp_cascade_boost_alpha,
    )
    views["local_multihop"] = build_multi_hop_transition_view(
        local_purified,
        name="local_multihop",
        alpha=local_multihop_alpha,
        max_hop=local_multihop_max_hop,
        base_weight=local_multihop_base_weight,
    )
    return views
