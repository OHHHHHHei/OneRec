from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from onerec.experiments.mgr_sid.graph_bank import infer_num_items, parse_id_list
from onerec.experiments.mgr_sid.paper_transplants import (
    build_semantic_knn_graph,
    keep_topk_per_row,
    load_semantic_embeddings,
    symmetrize_matrix,
)
from onerec.experiments.mgr_sid.transplanted_graph_bank import build_transplanted_graph_bank


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run D600 offline diagnostics for selective-separation pair construction."
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
        default="/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-16_mgr_sid_selective_separation_pair_diagnostics_industrial",
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
    parser.add_argument("--semantic-topk", type=int, default=32)
    parser.add_argument("--graph-topk", type=int, default=32)
    parser.add_argument("--graph-weak-quantile", type=float, default=0.25)
    parser.add_argument("--save-topn", type=int, default=200)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


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


def build_combined_graph_affinity(views: dict[str, sparse.csr_matrix]) -> sparse.csr_matrix:
    sym_views = [symmetrize_matrix(matrix) for matrix in views.values()]
    dense = np.stack([view.toarray() for view in sym_views], axis=0)
    combined = np.max(dense, axis=0).astype(np.float32)
    np.fill_diagonal(combined, 0.0)
    return sparse.csr_matrix(combined)


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
            if val > pair_scores.get(key, -1.0):
                pair_scores[key] = float(val)
    pairs = [(a, b, score) for (a, b), score in pair_scores.items()]
    pairs.sort(key=lambda item: item[2], reverse=True)
    return pairs


def user_overlap(item_users: list[set[str]], a: int, b: int) -> float:
    users_a = item_users[a]
    users_b = item_users[b]
    if not users_a and not users_b:
        return 0.0
    union = users_a | users_b
    if not union:
        return 0.0
    return float(len(users_a & users_b) / len(union))


def describe_values(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "count": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "q90": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
    arr = np.asarray(values, dtype=np.float32)
    return {
        "count": float(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "q90": float(np.quantile(arr, 0.9)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def summarize_pairs(df: pd.DataFrame, n_items: int) -> dict[str, float]:
    if df.empty:
        return {
            "pair_count": 0.0,
            "pair_count_ratio": 0.0,
            "item_coverage_rate": 0.0,
            "semantic_sim_mean": 0.0,
            "graph_affinity_mean": 0.0,
            "user_overlap_mean": 0.0,
            "reliability_mean": 0.0,
        }
    covered_items = set(df["item_a"].tolist()) | set(df["item_b"].tolist())
    summary = {
        "pair_count": float(len(df)),
        "item_coverage_rate": float(len(covered_items) / max(n_items, 1)),
        "semantic_sim_mean": float(df["semantic_sim"].mean()),
        "graph_affinity_mean": float(df["graph_affinity"].mean()),
        "user_overlap_mean": float(df["user_overlap"].mean()),
        "reliability_mean": float(df["reliability"].mean()),
    }
    for prefix in ["semantic_sim", "graph_affinity", "user_overlap", "reliability"]:
        stats = describe_values(df[prefix].tolist())
        for key, value in stats.items():
            summary[f"{prefix}_{key}"] = value
    return summary


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

    views = load_base_views(args)
    combined_graph = build_combined_graph_affinity(views)
    semantic_pairs = build_semantic_pair_list(semantic_embeddings, semantic_topk=args.semantic_topk)
    item_users = build_item_user_sets(train_df, n_items=n_items)

    records: list[dict[str, Any]] = []
    positive_graph_affinities: list[float] = []
    for item_a, item_b, semantic_sim in semantic_pairs:
        graph_affinity = float(combined_graph[item_a, item_b])
        if graph_affinity > 0.0:
            positive_graph_affinities.append(graph_affinity)
        overlap = user_overlap(item_users, item_a, item_b)
        reliability = float(semantic_sim * (1.0 - min(max(graph_affinity, 0.0), 1.0)) * (1.0 - overlap))
        records.append(
            {
                "item_a": int(item_a),
                "item_b": int(item_b),
                "semantic_sim": float(semantic_sim),
                "graph_affinity": graph_affinity,
                "user_overlap": overlap,
                "reliability": reliability,
                "coarse_affinity": float(views["coarse_purified"][item_a, item_b]),
                "mid_affinity": float(views["fagsp_mid_base"][item_a, item_b]),
                "local_affinity": float(views["local_purified"][item_a, item_b]),
            }
        )

    all_pairs = pd.DataFrame.from_records(records)
    positive_graph_affinities = [value for value in positive_graph_affinities if value > 0.0]
    weak_threshold = (
        float(np.quantile(np.asarray(positive_graph_affinities, dtype=np.float32), args.graph_weak_quantile))
        if positive_graph_affinities
        else 0.0
    )

    non_neighbor_df = all_pairs[all_pairs["graph_affinity"] <= 0.0].copy()
    non_neighbor_df["rule"] = "semantic_near_graph_non_neighbor"

    if weak_threshold > 0.0:
        weak_df = all_pairs[(all_pairs["graph_affinity"] > 0.0) & (all_pairs["graph_affinity"] <= weak_threshold)].copy()
    else:
        weak_df = all_pairs.iloc[0:0].copy()
    weak_df["rule"] = "semantic_near_graph_weak"

    non_neighbor_df = non_neighbor_df.sort_values(
        by=["reliability", "semantic_sim"], ascending=[False, False]
    ).reset_index(drop=True)
    weak_df = weak_df.sort_values(
        by=["reliability", "semantic_sim"], ascending=[False, False]
    ).reset_index(drop=True)

    save_topn = int(max(args.save_topn, 1))
    non_neighbor_df.to_csv(output_dir / "D600_all_non_neighbor_pairs.csv", index=False)
    weak_df.to_csv(output_dir / "D600_all_graph_weak_pairs.csv", index=False)
    non_neighbor_df.head(save_topn).to_csv(output_dir / "D600_top_non_neighbor_pairs.csv", index=False)
    weak_df.head(save_topn).to_csv(output_dir / "D600_top_graph_weak_pairs.csv", index=False)
    all_pairs.sort_values(by=["semantic_sim"], ascending=[False]).head(save_topn).to_csv(
        output_dir / "D600_top_semantic_pairs.csv", index=False
    )

    summary = {
        "n_items": int(n_items),
        "semantic_pair_count": int(len(all_pairs)),
        "graph_weak_quantile": float(args.graph_weak_quantile),
        "graph_weak_threshold": float(weak_threshold),
        "all_semantic_pairs": summarize_pairs(all_pairs, n_items=n_items),
        "semantic_near_graph_non_neighbor": summarize_pairs(non_neighbor_df, n_items=n_items),
        "semantic_near_graph_weak": summarize_pairs(weak_df, n_items=n_items),
        "base_graph_nnz": {
            name: int(matrix.nnz) for name, matrix in views.items()
        },
    }
    semantic_pair_count = max(float(summary["semantic_pair_count"]), 1.0)
    summary["semantic_near_graph_non_neighbor"]["pair_count_ratio"] = (
        summary["semantic_near_graph_non_neighbor"]["pair_count"] / semantic_pair_count
    )
    summary["semantic_near_graph_weak"]["pair_count_ratio"] = (
        summary["semantic_near_graph_weak"]["pair_count"] / semantic_pair_count
    )

    non_neighbor_too_broad = (
        summary["semantic_near_graph_non_neighbor"]["pair_count_ratio"] > 0.5
        or summary["semantic_near_graph_non_neighbor"]["item_coverage_rate"] > 0.8
    )
    weak_rule_reasonable = (
        summary["semantic_near_graph_weak"]["pair_count"] > 0
        and 0.1 <= summary["semantic_near_graph_weak"]["item_coverage_rate"] <= 0.8
    )

    recommendation = {
        "preferred_first_pair_rule": "semantic_near_graph_weak"
        if non_neighbor_too_broad and weak_rule_reasonable
        else (
            "semantic_near_graph_non_neighbor"
            if summary["semantic_near_graph_non_neighbor"]["pair_count"] > 0
            else "semantic_near_graph_weak"
        ),
        "reason": (
            "graph-non-neighbor pairs are too broad for a first training pass, so the better initial rule is semantic-near + graph-weak, which stays closer to the current collaborative support boundary"
            if non_neighbor_too_broad and weak_rule_reasonable
            else (
                "graph-non-neighbor pairs are the most conservative first-pass negative candidates because they are semantically close yet unsupported by any current base graph"
                if summary["semantic_near_graph_non_neighbor"]["pair_count"] > 0
                else "graph-weak pairs are the fallback first-pass candidates because positive graph affinities are too scarce for a stronger non-neighbor filter"
            )
        ),
        "non_neighbor_too_broad": bool(non_neighbor_too_broad),
        "weak_rule_reasonable": bool(weak_rule_reasonable),
    }
    summary["recommendation"] = recommendation

    (output_dir / "D600_pair_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps({"D600": summary}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summary_md = [
        "# D600 Pair Diagnostics Summary（D600 物品对诊断摘要）",
        "",
        f"- semantic pair count（语义近邻物品对数量）: `{summary['semantic_pair_count']}`",
        f"- graph-weak threshold（图弱连接阈值）: `{summary['graph_weak_threshold']:.6f}`",
        "",
        "## semantic-near + graph-non-neighbor（语义接近 + 图上无邻接）",
        f"- pair count（物品对数量）: `{int(summary['semantic_near_graph_non_neighbor']['pair_count'])}`",
        f"- pair ratio（物品对比例）: `{summary['semantic_near_graph_non_neighbor']['pair_count_ratio']:.4f}`",
        f"- item coverage rate（物品覆盖率）: `{summary['semantic_near_graph_non_neighbor']['item_coverage_rate']:.4f}`",
        f"- mean semantic sim（平均语义相似度）: `{summary['semantic_near_graph_non_neighbor']['semantic_sim_mean']:.4f}`",
        f"- mean user overlap（平均用户重叠）: `{summary['semantic_near_graph_non_neighbor']['user_overlap_mean']:.4f}`",
        f"- mean reliability（平均可靠性）: `{summary['semantic_near_graph_non_neighbor']['reliability_mean']:.4f}`",
        "",
        "## semantic-near + graph-weak（语义接近 + 图弱连接）",
        f"- pair count（物品对数量）: `{int(summary['semantic_near_graph_weak']['pair_count'])}`",
        f"- pair ratio（物品对比例）: `{summary['semantic_near_graph_weak']['pair_count_ratio']:.4f}`",
        f"- item coverage rate（物品覆盖率）: `{summary['semantic_near_graph_weak']['item_coverage_rate']:.4f}`",
        f"- mean semantic sim（平均语义相似度）: `{summary['semantic_near_graph_weak']['semantic_sim_mean']:.4f}`",
        f"- mean graph affinity（平均图亲和度）: `{summary['semantic_near_graph_weak']['graph_affinity_mean']:.6f}`",
        f"- mean reliability（平均可靠性）: `{summary['semantic_near_graph_weak']['reliability_mean']:.4f}`",
        "",
        "## Recommendation（推荐）",
        f"- preferred first pair rule（优先物品对规则）: `{recommendation['preferred_first_pair_rule']}`",
        f"- reason（原因）: {recommendation['reason']}",
        "",
        "## Files（文件）",
        "- `D600_all_non_neighbor_pairs.csv`",
        "- `D600_all_graph_weak_pairs.csv`",
        "- `D600_pair_summary.json`",
        "- `D600_top_non_neighbor_pairs.csv`",
        "- `D600_top_graph_weak_pairs.csv`",
        "- `D600_top_semantic_pairs.csv`",
    ]
    (output_dir / "SUMMARY.md").write_text("\n".join(summary_md) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
