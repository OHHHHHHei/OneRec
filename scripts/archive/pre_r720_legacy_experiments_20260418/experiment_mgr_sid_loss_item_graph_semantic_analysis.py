#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from onerec.experiments.mgr_sid.paper_transplants import keep_topk_per_row, load_semantic_embeddings, symmetrize_matrix
from onerec.experiments.mgr_sid.transplanted_graph_bank import build_transplanted_graph_bank


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze lost evaluate items from graph and semantic perspectives."
    )
    parser.add_argument(
        "--comparison-csv",
        default="/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-16_mgr_sid_r630c_sft_eval_industrial/TOPK_V2_ON_P05_VS_R630C.csv",
    )
    parser.add_argument("--baseline-label", default="v2_on_p05")
    parser.add_argument("--hierarchy-label", default="R630c")
    parser.add_argument(
        "--item-json",
        default="/home/leejt/OneRec/data/Amazon/index/Industrial_and_Scientific.item.json",
    )
    parser.add_argument(
        "--proxy-csv",
        default="/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-11_mgr_sid_v2_proxy_sanity/proxy_item_scores.csv",
    )
    parser.add_argument(
        "--weak-pair-csv",
        default="/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-16_mgr_sid_r630_mid_pull_push_industrial/R630_all_mid_graph_weak_pairs.csv",
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
    parser.add_argument("--graph-topk", type=int, default=32)
    parser.add_argument("--semantic-topk", type=int, default=10)
    parser.add_argument("--case-topn", type=int, default=5)
    parser.add_argument("--neighbor-topn", type=int, default=6)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


def family_tag(title: str) -> str:
    text = title.lower()
    if any(token in text for token in ["filament", "pla", "abs", "petg", "tpu"]):
        return "3d_filament"
    if any(token in text for token in ["thermometer", "hygrometer", "humidity", "gauge", "monitor"]):
        return "monitor_gauge"
    if any(token in text for token in ["tape", "duct tape"]):
        return "tape"
    if any(token in text for token in ["hose", "pipe", "fitting", "coupler", "plug"]):
        return "hose_fitting"
    if any(token in text for token in ["staple", "staples", "brad"]):
        return "staple_fastener"
    if any(token in text for token in ["strip", "strips", "ph test", "litmus"]):
        return "test_strip"
    return "other"


def weighted_mean(values: list[float], weights: list[float]) -> float:
    if not values or not weights:
        return 0.0
    total = float(sum(weights))
    if total <= 0:
        return 0.0
    return float(sum(v * w for v, w in zip(values, weights, strict=False)) / total)


def safe_mean(values: list[float]) -> float:
    return float(mean(values)) if values else 0.0


def top_neighbors_from_sparse(matrix: sparse.csr_matrix, item_id: int, topn: int) -> list[tuple[int, float]]:
    start, end = matrix.indptr[item_id], matrix.indptr[item_id + 1]
    cols = matrix.indices[start:end]
    vals = matrix.data[start:end]
    pairs = sorted(
        ((int(col), float(val)) for col, val in zip(cols, vals, strict=False) if int(col) != item_id),
        key=lambda item: item[1],
        reverse=True,
    )
    return pairs[:topn]


def build_semantic_neighbors(semantic_embeddings: np.ndarray, topk: int) -> dict[int, list[tuple[int, float]]]:
    nn = NearestNeighbors(n_neighbors=min(topk + 1, semantic_embeddings.shape[0]), metric="cosine")
    nn.fit(semantic_embeddings)
    distances, indices = nn.kneighbors(semantic_embeddings)
    out: dict[int, list[tuple[int, float]]] = {}
    for src in range(semantic_embeddings.shape[0]):
        neighs: list[tuple[int, float]] = []
        for dst, dist in zip(indices[src], distances[src], strict=False):
            dst = int(dst)
            if dst == src:
                continue
            sim = max(0.0, 1.0 - float(dist))
            neighs.append((dst, sim))
        out[src] = neighs[:topk]
    return out


def build_views(args: argparse.Namespace) -> dict[str, sparse.csr_matrix]:
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
        local_multihop_alpha=args.local_multihop_alpha,
        local_multihop_max_hop=args.local_multihop_max_hop,
        fagsp_cascade_high_rank=args.fagsp_cascade_high_rank,
        fagsp_cascade_low_rank=args.fagsp_cascade_low_rank,
        fagsp_cascade_support_quantile=args.fagsp_cascade_support_quantile,
        fagsp_cascade_boost_alpha=args.fagsp_cascade_boost_alpha,
        mgdcf_keep_ratio=args.mgdcf_keep_ratio,
        mgdcf_binarize_edges=args.mgdcf_binarize_edges,
    )
    out: dict[str, sparse.csr_matrix] = {}
    for name in ["coarse_purified", "fagsp_mid_base", "local_purified"]:
        pruned = keep_topk_per_row(views[name].matrix, topk=args.graph_topk).tocsr().astype(np.float32)
        out[name] = symmetrize_matrix(pruned).tocsr().astype(np.float32)
    return out


def build_item_profiles(
    comparison_df: pd.DataFrame,
    item_meta: dict[int, dict[str, str]],
    proxy_df: pd.DataFrame,
    weak_pair_df: pd.DataFrame,
    semantic_neighbors: dict[int, list[tuple[int, float]]],
    weak_threshold: float,
    views: dict[str, sparse.csr_matrix],
) -> pd.DataFrame:
    n_items = max(item_meta) + 1 if item_meta else 0

    weak_endpoint_count = np.zeros(n_items, dtype=np.int32)
    weak_reliability_sum = np.zeros(n_items, dtype=np.float32)
    weak_partner_set: dict[int, set[int]] = defaultdict(set)
    weak_pair_lookup: dict[tuple[int, int], dict[str, float]] = {}
    for row in weak_pair_df.itertuples(index=False):
        a = int(row.item_a)
        b = int(row.item_b)
        reliability = float(row.reliability)
        weak_endpoint_count[a] += 1
        weak_endpoint_count[b] += 1
        weak_reliability_sum[a] += reliability
        weak_reliability_sum[b] += reliability
        weak_partner_set[a].add(b)
        weak_partner_set[b].add(a)
        weak_pair_lookup[(min(a, b), max(a, b))] = {
            "semantic_sim": float(row.semantic_sim),
            "mid_affinity": float(row.mid_affinity),
            "reliability": reliability,
        }

    agg = comparison_df.groupby("item_id").agg(
        eval_count=("item_id", "size"),
        loss_count_top10=("baseline_top10_hit", lambda s: int(((comparison_df.loc[s.index, "baseline_top10_hit"] == 1) & (comparison_df.loc[s.index, "hierarchy_top10_hit"] == 0)).sum())),
        gain_count_top10=("baseline_top10_hit", lambda s: int(((comparison_df.loc[s.index, "baseline_top10_hit"] == 0) & (comparison_df.loc[s.index, "hierarchy_top10_hit"] == 1)).sum())),
        loss_count_top1=("baseline_top1_hit", lambda s: int(((comparison_df.loc[s.index, "baseline_top1_hit"] == 1) & (comparison_df.loc[s.index, "hierarchy_top1_hit"] == 0)).sum())),
        gain_count_top1=("baseline_top1_hit", lambda s: int(((comparison_df.loc[s.index, "baseline_top1_hit"] == 0) & (comparison_df.loc[s.index, "hierarchy_top1_hit"] == 1)).sum())),
        baseline_top10_hit_rate=("baseline_top10_hit", "mean"),
        hierarchy_top10_hit_rate=("hierarchy_top10_hit", "mean"),
        baseline_top1_hit_rate=("baseline_top1_hit", "mean"),
        hierarchy_top1_hit_rate=("hierarchy_top1_hit", "mean"),
    ).reset_index()
    agg["delta_top10_hit_rate"] = agg["hierarchy_top10_hit_rate"] - agg["baseline_top10_hit_rate"]
    agg["delta_top1_hit_rate"] = agg["hierarchy_top1_hit_rate"] - agg["baseline_top1_hit_rate"]

    proxy_use = proxy_df.rename(columns={"item_id": "item_id_proxy"})
    rows: list[dict[str, object]] = []
    for item_id in sorted(item_meta):
        meta = item_meta[item_id]
        row: dict[str, object] = {
            "item_id": int(item_id),
            "title": meta.get("title", ""),
            "brand": meta.get("brand", ""),
            "family": family_tag(meta.get("title", "")),
            "weak_pair_endpoint_count": int(weak_endpoint_count[item_id]) if item_id < len(weak_endpoint_count) else 0,
            "weak_pair_reliability_sum": float(weak_reliability_sum[item_id]) if item_id < len(weak_reliability_sum) else 0.0,
            "weak_pair_partner_count": int(len(weak_partner_set.get(item_id, set()))),
        }
        row_agg = agg[agg["item_id"] == item_id]
        if not row_agg.empty:
            row.update(row_agg.iloc[0].to_dict())
        else:
            row.update(
                {
                    "eval_count": 0,
                    "loss_count_top10": 0,
                    "gain_count_top10": 0,
                    "loss_count_top1": 0,
                    "gain_count_top1": 0,
                    "baseline_top10_hit_rate": 0.0,
                    "hierarchy_top10_hit_rate": 0.0,
                    "baseline_top1_hit_rate": 0.0,
                    "hierarchy_top1_hit_rate": 0.0,
                    "delta_top10_hit_rate": 0.0,
                    "delta_top1_hit_rate": 0.0,
                }
            )

        row_proxy = proxy_use[proxy_use["item_id_proxy"] == item_id]
        for col in [
            "semantic_density",
            "semantic_collab_disagreement",
            "graph_competition",
            "offline_combined",
            "baseline_l2_leaf_count",
            "hierarchy_l2_leaf_count",
            "leaf_reduction",
        ]:
            row[col] = float(row_proxy.iloc[0][col]) if not row_proxy.empty else 0.0

        for view_name, matrix in views.items():
            start, end = matrix.indptr[item_id], matrix.indptr[item_id + 1]
            row[f"{view_name}_degree"] = int(end - start)
            row[f"{view_name}_strength"] = float(matrix.data[start:end].sum()) if end > start else 0.0

        sem_neigh = semantic_neighbors.get(item_id, [])
        row["semantic_neighbor_count"] = int(len(sem_neigh))
        row["semantic_topk_mean_sim"] = safe_mean([sim for _, sim in sem_neigh])
        row["semantic_topk_max_sim"] = max((sim for _, sim in sem_neigh), default=0.0)
        row["semantic_topk_min_sim"] = min((sim for _, sim in sem_neigh), default=0.0)

        mid_affs: list[float] = []
        coarse_affs: list[float] = []
        local_affs: list[float] = []
        weak_count = 0
        zero_count = 0
        overlap_mid = 0
        mid_neighbors = {nid for nid, _ in top_neighbors_from_sparse(views["fagsp_mid_base"], item_id, len(sem_neigh))}
        for neigh_id, _sim in sem_neigh:
            mid_val = float(views["fagsp_mid_base"][item_id, neigh_id])
            coarse_val = float(views["coarse_purified"][item_id, neigh_id])
            local_val = float(views["local_purified"][item_id, neigh_id])
            mid_affs.append(mid_val)
            coarse_affs.append(coarse_val)
            local_affs.append(local_val)
            if mid_val <= 0.0:
                zero_count += 1
            elif mid_val <= weak_threshold:
                weak_count += 1
            if neigh_id in mid_neighbors:
                overlap_mid += 1
        denom = max(len(sem_neigh), 1)
        row["semantic_topk_mean_mid_affinity"] = safe_mean(mid_affs)
        row["semantic_topk_mean_coarse_affinity"] = safe_mean(coarse_affs)
        row["semantic_topk_mean_local_affinity"] = safe_mean(local_affs)
        row["semantic_topk_zero_mid_fraction"] = float(zero_count / denom)
        row["semantic_topk_weak_mid_fraction"] = float(weak_count / denom)
        row["semantic_topk_graph_overlap_fraction"] = float(overlap_mid / denom)
        rows.append(row)

    return pd.DataFrame(rows), weak_pair_lookup


def summarize_segment(df: pd.DataFrame, weight_col: str) -> dict[str, object]:
    if df.empty:
        return {"item_count": 0}
    weights = df[weight_col].astype(float).tolist()
    if sum(weights) <= 0:
        weights = [1.0] * len(df)
    metrics = [
        "semantic_density",
        "semantic_collab_disagreement",
        "graph_competition",
        "offline_combined",
        "weak_pair_endpoint_count",
        "weak_pair_reliability_sum",
        "coarse_purified_degree",
        "fagsp_mid_base_degree",
        "local_purified_degree",
        "semantic_topk_mean_sim",
        "semantic_topk_mean_mid_affinity",
        "semantic_topk_zero_mid_fraction",
        "semantic_topk_weak_mid_fraction",
        "semantic_topk_graph_overlap_fraction",
    ]
    metric_summary = {
        metric: weighted_mean(df[metric].astype(float).tolist(), weights)
        for metric in metrics
    }
    brand_counter = Counter()
    family_counter = Counter()
    for row in df.itertuples(index=False):
        w = float(getattr(row, weight_col))
        if w <= 0:
            continue
        brand = str(getattr(row, "brand")).strip() or "<empty_brand>"
        family = str(getattr(row, "family"))
        brand_counter[brand] += w
        family_counter[family] += w
    return {
        "item_count": int(len(df)),
        "total_weight": float(sum(weights)),
        "metric_summary": metric_summary,
        "top_brands": [
            {"brand": brand, "weight": float(weight), "fraction": float(weight / max(sum(weights), 1.0))}
            for brand, weight in brand_counter.most_common(10)
        ],
        "top_families": [
            {"family": family, "weight": float(weight), "fraction": float(weight / max(sum(weights), 1.0))}
            for family, weight in family_counter.most_common(10)
        ],
    }


def build_case_table(
    item_ids: list[int],
    item_profiles: pd.DataFrame,
    item_meta: dict[int, dict[str, str]],
    semantic_neighbors: dict[int, list[tuple[int, float]]],
    views: dict[str, sparse.csr_matrix],
    weak_pair_lookup: dict[tuple[int, int], dict[str, float]],
    neighbor_topn: int,
) -> list[dict[str, object]]:
    profile_lookup = {int(row["item_id"]): row for row in item_profiles.to_dict("records")}
    cases: list[dict[str, object]] = []
    for item_id in item_ids:
        profile = profile_lookup[item_id]
        semantic_case_rows: list[dict[str, object]] = []
        for neigh_id, sem_sim in semantic_neighbors.get(item_id, [])[:neighbor_topn]:
            pair_key = (min(item_id, neigh_id), max(item_id, neigh_id))
            weak_meta = weak_pair_lookup.get(pair_key)
            semantic_case_rows.append(
                {
                    "neighbor_item_id": int(neigh_id),
                    "neighbor_title": item_meta.get(neigh_id, {}).get("title", ""),
                    "neighbor_brand": item_meta.get(neigh_id, {}).get("brand", ""),
                    "semantic_sim": float(sem_sim),
                    "coarse_affinity": float(views["coarse_purified"][item_id, neigh_id]),
                    "mid_affinity": float(views["fagsp_mid_base"][item_id, neigh_id]),
                    "local_affinity": float(views["local_purified"][item_id, neigh_id]),
                    "is_weak_pair": weak_meta is not None,
                    "weak_pair_reliability": float(weak_meta["reliability"]) if weak_meta else 0.0,
                    "neighbor_loss_count_top10": int(profile_lookup.get(neigh_id, {}).get("loss_count_top10", 0)),
                    "neighbor_gain_count_top10": int(profile_lookup.get(neigh_id, {}).get("gain_count_top10", 0)),
                }
            )

        graph_case_rows: list[dict[str, object]] = []
        for neigh_id, mid_affinity in top_neighbors_from_sparse(views["fagsp_mid_base"], item_id, neighbor_topn):
            pair_key = (min(item_id, neigh_id), max(item_id, neigh_id))
            weak_meta = weak_pair_lookup.get(pair_key)
            sem_sim = 0.0
            for cand_id, cand_sim in semantic_neighbors.get(item_id, []):
                if cand_id == neigh_id:
                    sem_sim = float(cand_sim)
                    break
            graph_case_rows.append(
                {
                    "neighbor_item_id": int(neigh_id),
                    "neighbor_title": item_meta.get(neigh_id, {}).get("title", ""),
                    "neighbor_brand": item_meta.get(neigh_id, {}).get("brand", ""),
                    "mid_affinity": float(mid_affinity),
                    "semantic_sim": sem_sim,
                    "coarse_affinity": float(views["coarse_purified"][item_id, neigh_id]),
                    "local_affinity": float(views["local_purified"][item_id, neigh_id]),
                    "is_weak_pair": weak_meta is not None,
                    "weak_pair_reliability": float(weak_meta["reliability"]) if weak_meta else 0.0,
                }
            )

        cases.append(
            {
                "item_id": int(item_id),
                "title": item_meta.get(item_id, {}).get("title", ""),
                "brand": item_meta.get(item_id, {}).get("brand", ""),
                "family": family_tag(item_meta.get(item_id, {}).get("title", "")),
                "profile": profile,
                "top_semantic_neighbors": semantic_case_rows,
                "top_mid_graph_neighbors": graph_case_rows,
            }
        )
    return cases


def format_float(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def write_markdown(summary: dict[str, object], path: Path) -> None:
    baseline_label = str(summary["baseline_label"])
    hierarchy_label = str(summary["hierarchy_label"])
    overview = dict(summary["overview"])
    loss_summary = dict(summary["loss_items_weighted_by_loss_count_top10"])
    gain_summary = dict(summary["gain_items_weighted_by_gain_count_top10"])
    all_summary = dict(summary["all_eval_items_weighted_by_eval_count"])
    cases = list(summary["loss_case_studies"])

    lines: list[str] = []
    lines.append("# Loss Item Graph/Semantic Analysis（损失物品图与语义分析）\n")
    lines.append("## Scope（范围）\n")
    lines.append(
        f"This note analyzes the `top10` loss items（`top10` 损失物品） where `{baseline_label}` hits but `{hierarchy_label}` misses.\n"
    )

    lines.append("## Overview（概览）\n")
    lines.append(f"- total `top10` loss examples（总 `top10` 损失样本）: `{overview['top10_loss_example_count']}`")
    lines.append(f"- total `top10` gain examples（总 `top10` 增益样本）: `{overview['top10_gain_example_count']}`")
    lines.append(f"- unique loss items（唯一损失物品数）: `{overview['unique_top10_loss_items']}`")
    lines.append(f"- unique gain items（唯一增益物品数）: `{overview['unique_top10_gain_items']}`")
    lines.append(f"- weak pair threshold（弱连接阈值）: `{format_float(float(overview['weak_pair_threshold']), 6)}`\n")

    def metric_table_row(metric: str, label: str) -> str:
        return (
            f"| {label} | "
            f"{format_float(float(all_summary['metric_summary'][metric]))} | "
            f"{format_float(float(loss_summary['metric_summary'][metric]))} | "
            f"{format_float(float(gain_summary['metric_summary'][metric]))} |"
        )

    lines.append("## Aggregate Comparison（聚合对比）\n")
    lines.append("| metric | all_eval_items（全部评测物品） | top10_loss_items（`top10` 损失物品） | top10_gain_items（`top10` 增益物品） |")
    lines.append("|---|---:|---:|---:|")
    for metric, label in [
        ("semantic_density", "semantic_density（语义密度）"),
        ("semantic_collab_disagreement", "semantic_collab_disagreement（语义-协同失配）"),
        ("graph_competition", "graph_competition（图竞争度）"),
        ("offline_combined", "offline_combined（离线合成分数）"),
        ("weak_pair_endpoint_count", "weak_pair_endpoint_count（弱连接对端点数）"),
        ("weak_pair_reliability_sum", "weak_pair_reliability_sum（弱连接可靠性和）"),
        ("semantic_topk_mean_sim", "semantic_topk_mean_sim（语义近邻平均相似度）"),
        ("semantic_topk_mean_mid_affinity", "semantic_topk_mean_mid_affinity（语义近邻中图平均亲和）"),
        ("semantic_topk_zero_mid_fraction", "semantic_topk_zero_mid_fraction（语义近邻零中图占比）"),
        ("semantic_topk_weak_mid_fraction", "semantic_topk_weak_mid_fraction（语义近邻弱中图占比）"),
        ("semantic_topk_graph_overlap_fraction", "semantic_topk_graph_overlap_fraction（语义/中图邻居重叠占比）"),
    ]:
        lines.append(metric_table_row(metric, label))
    lines.append("")

    lines.append("## Common Traits（共同特点）\n")
    lines.append("### Loss Families（损失家族）\n")
    lines.append("| family | fraction | weight |")
    lines.append("|---|---:|---:|")
    for row in loss_summary["top_families"]:
        lines.append(f"| {row['family']} | {float(row['fraction']):.3f} | {float(row['weight']):.1f} |")
    lines.append("")

    lines.append("### Loss Brands（损失品牌）\n")
    lines.append("| brand | fraction | weight |")
    lines.append("|---|---:|---:|")
    for row in loss_summary["top_brands"]:
        lines.append(f"| {row['brand']} | {float(row['fraction']):.3f} | {float(row['weight']):.1f} |")
    lines.append("")

    lines.append("### Gain Families（增益家族）\n")
    lines.append("| family | fraction | weight |")
    lines.append("|---|---:|---:|")
    for row in gain_summary["top_families"]:
        lines.append(f"| {row['family']} | {float(row['fraction']):.3f} | {float(row['weight']):.1f} |")
    lines.append("")

    lines.append("## Case Studies（病例分析）\n")
    for case in cases:
        profile = dict(case["profile"])
        lines.append(f"### {case['item_id']}: {case['title']}\n")
        lines.append(f"- brand（品牌）: `{case['brand']}`")
        lines.append(f"- family（家族）: `{case['family']}`")
        lines.append(
            f"- loss/gain counts（损失/增益次数）: `top10 loss = {int(profile['loss_count_top10'])}`, `top10 gain = {int(profile['gain_count_top10'])}`"
        )
        lines.append(
            f"- proxy scores（代理分数）: "
            f"`semantic_density = {format_float(float(profile['semantic_density']))}`, "
            f"`semantic_collab_disagreement = {format_float(float(profile['semantic_collab_disagreement']))}`, "
            f"`graph_competition = {format_float(float(profile['graph_competition']))}`"
        )
        lines.append(
            f"- weak-pair exposure（弱连接对暴露）: "
            f"`endpoint_count = {int(profile['weak_pair_endpoint_count'])}`, "
            f"`reliability_sum = {format_float(float(profile['weak_pair_reliability_sum']))}`"
        )
        lines.append(
            f"- graph stats（图统计）: "
            f"`mid_degree = {int(profile['fagsp_mid_base_degree'])}`, "
            f"`mid_strength = {format_float(float(profile['fagsp_mid_base_strength']))}`, "
            f"`semantic_topk_mean_mid_affinity = {format_float(float(profile['semantic_topk_mean_mid_affinity']))}`, "
            f"`semantic_topk_weak_mid_fraction = {format_float(float(profile['semantic_topk_weak_mid_fraction']))}`"
        )
        lines.append("")

        lines.append("Top semantic neighbors（顶部语义近邻）:")
        lines.append("| neighbor | brand | semantic_sim | coarse | mid | local | weak_pair | reliability | neighbor_top10_loss | neighbor_top10_gain |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in case["top_semantic_neighbors"]:
            lines.append(
                f"| {row['neighbor_item_id']}: {row['neighbor_title']} | {row['neighbor_brand']} | "
                f"{format_float(float(row['semantic_sim']))} | {format_float(float(row['coarse_affinity']))} | "
                f"{format_float(float(row['mid_affinity']))} | {format_float(float(row['local_affinity']))} | "
                f"{int(bool(row['is_weak_pair']))} | {format_float(float(row['weak_pair_reliability']))} | "
                f"{int(row['neighbor_loss_count_top10'])} | {int(row['neighbor_gain_count_top10'])} |"
            )
        lines.append("")

        lines.append("Top mid-graph neighbors（顶部中图近邻）:")
        lines.append("| neighbor | brand | mid | semantic_sim | coarse | local | weak_pair | reliability |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
        for row in case["top_mid_graph_neighbors"]:
            lines.append(
                f"| {row['neighbor_item_id']}: {row['neighbor_title']} | {row['neighbor_brand']} | "
                f"{format_float(float(row['mid_affinity']))} | {format_float(float(row['semantic_sim']))} | "
                f"{format_float(float(row['coarse_affinity']))} | {format_float(float(row['local_affinity']))} | "
                f"{int(bool(row['is_weak_pair']))} | {format_float(float(row['weak_pair_reliability']))} |"
            )
        lines.append("")

    lines.append("## Reading（解读）\n")
    lines.append(
        "- If loss items have higher `weak_pair_endpoint_count`（弱连接对端点数） and higher `semantic_collab_disagreement`（语义-协同失配）, "
        "then the push term is concentrating on exactly those semantically dense but collaboratively weak neighborhoods."
    )
    lines.append(
        "- If their `semantic_topk_mean_mid_affinity`（语义近邻中图平均亲和） is low while semantic similarity stays high, "
        "then the method is facing semantic-near / graph-weak tension rather than a simple sparse-item problem."
    )
    lines.append(
        "- The case tables show whether the lost items are surrounded by same-family variants（同家族变体） that are semantically near but weakly supported by `G_mid`（中尺度图）."
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    comparison_df = pd.read_csv(args.comparison_csv)
    item_json_raw = json.loads(Path(args.item_json).read_text(encoding="utf-8"))
    item_meta = {int(k): dict(v) for k, v in item_json_raw.items()}
    proxy_df = pd.read_csv(args.proxy_csv)
    weak_pair_df = pd.read_csv(args.weak_pair_csv)
    weak_threshold = float(weak_pair_df["weak_threshold"].iloc[0]) if not weak_pair_df.empty else 0.0

    views = build_views(args)
    semantic_embeddings = load_semantic_embeddings(args.semantic_embedding_path)
    if semantic_embeddings is None:
        raise ValueError("semantic embeddings are required")
    semantic_neighbors = build_semantic_neighbors(semantic_embeddings, topk=args.semantic_topk)

    item_profiles, weak_pair_lookup = build_item_profiles(
        comparison_df=comparison_df,
        item_meta=item_meta,
        proxy_df=proxy_df,
        weak_pair_df=weak_pair_df,
        semantic_neighbors=semantic_neighbors,
        weak_threshold=weak_threshold,
        views=views,
    )

    loss_items = item_profiles[item_profiles["loss_count_top10"] > 0].copy()
    gain_items = item_profiles[item_profiles["gain_count_top10"] > 0].copy()
    all_eval_items = item_profiles[item_profiles["eval_count"] > 0].copy()

    loss_cases_df = loss_items.sort_values(
        by=["loss_count_top10", "delta_top10_hit_rate", "weak_pair_endpoint_count"],
        ascending=[False, True, False],
    ).head(args.case_topn)
    loss_case_ids = [int(v) for v in loss_cases_df["item_id"].tolist()]

    summary = {
        "baseline_label": args.baseline_label,
        "hierarchy_label": args.hierarchy_label,
        "overview": {
            "top10_loss_example_count": int(
                ((comparison_df["baseline_top10_hit"] == 1) & (comparison_df["hierarchy_top10_hit"] == 0)).sum()
            ),
            "top10_gain_example_count": int(
                ((comparison_df["baseline_top10_hit"] == 0) & (comparison_df["hierarchy_top10_hit"] == 1)).sum()
            ),
            "unique_top10_loss_items": int(len(loss_items)),
            "unique_top10_gain_items": int(len(gain_items)),
            "weak_pair_threshold": weak_threshold,
        },
        "all_eval_items_weighted_by_eval_count": summarize_segment(all_eval_items, "eval_count"),
        "loss_items_weighted_by_loss_count_top10": summarize_segment(loss_items, "loss_count_top10"),
        "gain_items_weighted_by_gain_count_top10": summarize_segment(gain_items, "gain_count_top10"),
        "loss_case_studies": build_case_table(
            item_ids=loss_case_ids,
            item_profiles=item_profiles,
            item_meta=item_meta,
            semantic_neighbors=semantic_neighbors,
            views=views,
            weak_pair_lookup=weak_pair_lookup,
            neighbor_topn=args.neighbor_topn,
        ),
    }

    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(summary, output_md)
    print(f"Wrote JSON: {output_json}")
    print(f"Wrote MD: {output_md}")


if __name__ == "__main__":
    main()
