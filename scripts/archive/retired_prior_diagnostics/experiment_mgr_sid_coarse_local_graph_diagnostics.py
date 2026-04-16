from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import csgraph
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD

from onerec.experiments.mgr_sid.graph_bank import (
    build_coarse_graph,
    build_mgdcf_item_graph,
    build_local_graph,
    build_popularity,
    infer_num_items,
    parse_id_list,
    purify_coarse_graph,
    purify_local_graph,
    row_normalize,
)
from onerec.experiments.mgr_sid.transplanted_graph_bank import build_transplanted_graph_bank


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run D530/D540/D541 offline diagnostics for coarse/local graph-carrier candidates."
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
        "--output-dir",
        default="/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-15_mgr_sid_coarse_local_graph_diagnostics_industrial",
    )
    parser.add_argument("--history-k", type=int, default=10)
    parser.add_argument("--coarse-min-weight", type=float, default=2.0)
    parser.add_argument("--local-min-weight", type=float, default=1.0)
    parser.add_argument("--community-clusters", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--graph-topk", type=int, default=32)
    parser.add_argument("--segment-k-values", nargs="+", type=int, default=[4, 8])
    parser.add_argument("--local-alpha-values", nargs="+", type=float, default=[0.35, 0.50])
    parser.add_argument("--max-local-hop", type=int, default=3)
    parser.add_argument("--cir-mix", type=float, default=0.5)
    parser.add_argument("--mgdcf-keep-ratios", nargs="+", type=float, default=[0.05, 0.1, 0.2])
    parser.add_argument("--mgdcf-binarize-edges", action="store_true")
    parser.add_argument("--semantic-embedding-path", default="/home/leejt/OneRec/data/Amazon/index/Industrial_and_Scientific.emb-qwen-td.npy")
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
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def keep_topk_per_row(matrix: sparse.csr_matrix, topk: int) -> sparse.csr_matrix:
    matrix = matrix.tocsr(copy=True)
    data: list[float] = []
    indices: list[int] = []
    indptr = [0]
    for row in range(matrix.shape[0]):
        start, end = matrix.indptr[row], matrix.indptr[row + 1]
        row_indices = matrix.indices[start:end]
        row_data = matrix.data[start:end]
        if row_data.size > topk:
            keep_idx = np.argpartition(row_data, -topk)[-topk:]
            keep_idx = keep_idx[np.argsort(row_data[keep_idx])[::-1]]
            row_indices = row_indices[keep_idx]
            row_data = row_data[keep_idx]
        data.extend(row_data.tolist())
        indices.extend(row_indices.tolist())
        indptr.append(len(data))
    pruned = sparse.csr_matrix(
        (np.asarray(data, dtype=np.float32), np.asarray(indices, dtype=np.int32), np.asarray(indptr, dtype=np.int32)),
        shape=matrix.shape,
    )
    pruned.eliminate_zeros()
    return pruned


def zero_diagonal(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    matrix = matrix.tocsr(copy=True)
    matrix.setdiag(0.0)
    matrix.eliminate_zeros()
    return matrix


def topk_neighbors(matrix: sparse.csr_matrix, row: int, topk: int) -> set[int]:
    start, end = matrix.indptr[row], matrix.indptr[row + 1]
    row_indices = matrix.indices[start:end]
    row_data = matrix.data[start:end]
    if row_data.size == 0:
        return set()
    if row_data.size > topk:
        keep_idx = np.argpartition(row_data, -topk)[-topk:]
        row_indices = row_indices[keep_idx]
    return {int(idx) for idx in row_indices}


def mean_neighbor_overlap(graph_a: sparse.csr_matrix, graph_b: sparse.csr_matrix, topk: int) -> float:
    overlaps: list[float] = []
    for row in range(graph_a.shape[0]):
        neigh_a = topk_neighbors(graph_a, row, topk)
        neigh_b = topk_neighbors(graph_b, row, topk)
        union = neigh_a | neigh_b
        if not union:
            overlaps.append(0.0)
            continue
        overlaps.append(len(neigh_a & neigh_b) / len(union))
    return float(np.mean(overlaps)) if overlaps else 0.0


def expansion_stats(candidate: sparse.csr_matrix, baseline: sparse.csr_matrix, topk: int) -> dict[str, float]:
    added_neighbor_counts: list[int] = []
    retained_neighbor_counts: list[int] = []
    baseline_sizes: list[int] = []
    candidate_sizes: list[int] = []
    for row in range(candidate.shape[0]):
        cand = topk_neighbors(candidate, row, topk)
        base = topk_neighbors(baseline, row, topk)
        added_neighbor_counts.append(len(cand - base))
        retained_neighbor_counts.append(len(cand & base))
        baseline_sizes.append(len(base))
        candidate_sizes.append(len(cand))
    baseline_total = float(np.sum(baseline_sizes))
    return {
        "mean_added_neighbors_topk": float(np.mean(added_neighbor_counts)) if added_neighbor_counts else 0.0,
        "mean_retained_neighbors_topk": float(np.mean(retained_neighbor_counts)) if retained_neighbor_counts else 0.0,
        "mean_candidate_neighbors_topk": float(np.mean(candidate_sizes)) if candidate_sizes else 0.0,
        "topk_expansion_ratio": float(np.sum(added_neighbor_counts) / baseline_total) if baseline_total > 0 else 0.0,
    }


def graph_metrics(
    graph: sparse.csr_matrix,
    baseline_graph: sparse.csr_matrix | None,
    topk: int,
) -> dict[str, float]:
    graph = graph.tocsr()
    n_items = graph.shape[0]
    out_degree = np.diff(graph.indptr)
    in_degree = np.bincount(graph.indices, minlength=n_items) if graph.nnz else np.zeros(n_items, dtype=np.int64)
    undirected = ((graph + graph.T) > 0).astype(np.int32)
    n_components, labels = csgraph.connected_components(undirected, directed=False, return_labels=True)
    largest_ratio = 0.0
    if labels.size > 0:
        largest_ratio = float(np.max(np.bincount(labels)) / len(labels))
    metrics: dict[str, float] = {
        "n_items": float(n_items),
        "graph_nnz": float(graph.nnz),
        "graph_density": float(graph.nnz / max(n_items * n_items, 1)),
        "avg_out_degree": float(np.mean(out_degree)) if out_degree.size else 0.0,
        "avg_in_degree": float(np.mean(in_degree)) if in_degree.size else 0.0,
        "outgoing_coverage_rate": float(np.mean(out_degree > 0)) if out_degree.size else 0.0,
        "incoming_coverage_rate": float(np.mean(in_degree > 0)) if in_degree.size else 0.0,
        "connected_item_rate": float(np.mean((out_degree + in_degree) > 0)) if out_degree.size else 0.0,
        "orphan_item_rate": float(np.mean((out_degree + in_degree) == 0)) if out_degree.size else 0.0,
        "largest_component_ratio": largest_ratio,
        "connected_component_count": float(n_components),
    }
    if baseline_graph is not None:
        metrics["mean_neighbor_overlap_with_baseline"] = mean_neighbor_overlap(graph, baseline_graph, topk=topk)
        metrics.update(expansion_stats(graph, baseline_graph, topk=topk))
    return metrics


def build_multi_hop_local(base_graph: sparse.csr_matrix, alpha: float, max_hop: int) -> sparse.csr_matrix:
    normalized = row_normalize(base_graph)
    accum = normalized.copy().astype(np.float32)
    power = normalized.copy().astype(np.float32)
    for hop in range(2, max_hop + 1):
        power = (power @ normalized).tocsr().astype(np.float32)
        accum = (accum + (alpha ** (hop - 1)) * power).tocsr()
    accum = zero_diagonal(accum)
    return row_normalize(accum)


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


def build_cir_reweighted_coarse(
    coarse_graph: sparse.csr_matrix,
    item_users: list[set[str]],
    mix: float,
) -> tuple[sparse.csr_matrix, dict[str, float]]:
    coo = coarse_graph.tocoo(copy=True)
    cir_values = np.zeros_like(coo.data, dtype=np.float32)
    for idx, (row, col) in enumerate(zip(coo.row, coo.col, strict=False)):
        users_a = item_users[int(row)]
        users_b = item_users[int(col)]
        if not users_a and not users_b:
            cir = 0.0
        else:
            union = users_a | users_b
            cir = (len(users_a & users_b) / len(union)) if union else 0.0
        cir_values[idx] = float(cir)
    blend = mix + (1.0 - mix) * cir_values
    coo.data = coo.data * blend
    matrix = row_normalize(coo.tocsr())
    metadata = {
        "cir_mix": float(mix),
        "cir_mean": float(np.mean(cir_values)) if cir_values.size else 0.0,
        "cir_median": float(np.median(cir_values)) if cir_values.size else 0.0,
        "cir_q90": float(np.quantile(cir_values, 0.9)) if cir_values.size else 0.0,
        "cir_nonzero_rate": float(np.mean(cir_values > 0)) if cir_values.size else 0.0,
    }
    return matrix, metadata


def build_user_item_matrix(train_df: pd.DataFrame, n_items: int) -> tuple[sparse.csr_matrix, list[str]]:
    user_to_idx: dict[str, int] = {}
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for row in train_df.itertuples(index=False):
        user_id = str(row.user_id)
        user_idx = user_to_idx.setdefault(user_id, len(user_to_idx))
        counter: Counter[int] = Counter()
        target = int(row.item_id)
        if 0 <= target < n_items:
            counter[target] += 1
        for hist_item in parse_id_list(row.history_item_id):
            if 0 <= hist_item < n_items:
                counter[hist_item] += 1
        for item_id, count in counter.items():
            rows.append(user_idx)
            cols.append(item_id)
            data.append(float(count))
    matrix = sparse.coo_matrix(
        (np.asarray(data, dtype=np.float32), (np.asarray(rows), np.asarray(cols))),
        shape=(len(user_to_idx), n_items),
        dtype=np.float32,
    ).tocsr()
    users = [""] * len(user_to_idx)
    for user_id, idx in user_to_idx.items():
        users[idx] = user_id
    return matrix, users


def build_user_segment_coarse(
    train_df: pd.DataFrame,
    base_graph: sparse.csr_matrix,
    n_items: int,
    n_segments: int,
    seed: int,
    history_k: int,
) -> tuple[sparse.csr_matrix, pd.DataFrame, dict[str, float]]:
    user_item_matrix, users = build_user_item_matrix(train_df, n_items=n_items)
    n_users = user_item_matrix.shape[0]
    n_segments = max(2, min(int(n_segments), n_users))
    n_components = max(2, min(32, min(user_item_matrix.shape) - 1))
    if n_components >= min(user_item_matrix.shape):
        user_embed = user_item_matrix.toarray()
    else:
        svd = TruncatedSVD(n_components=n_components, random_state=seed)
        user_embed = svd.fit_transform(user_item_matrix)
    kmeans = KMeans(n_clusters=n_segments, random_state=seed, n_init=10)
    segment_ids = kmeans.fit_predict(user_embed)
    user_segment_df = pd.DataFrame({"user_id": users, "segment_id": segment_ids.astype(int)})

    support_counter: defaultdict[tuple[int, int], set[int]] = defaultdict(set)
    train_with_segment = train_df.merge(user_segment_df, on="user_id", how="left")
    for row in train_with_segment.itertuples(index=False):
        segment_id = int(row.segment_id)
        history = parse_id_list(row.history_item_id)
        target = int(row.item_id)
        seq = history[-history_k:] + [target]
        unique_seq: list[int] = []
        seen: set[int] = set()
        for item in seq:
            if item < 0 or item >= n_items or item in seen:
                continue
            seen.add(item)
            unique_seq.append(item)
        for i, src in enumerate(unique_seq):
            for j in range(i + 1, len(unique_seq)):
                dst = unique_seq[j]
                support_counter[(src, dst)].add(segment_id)
                support_counter[(dst, src)].add(segment_id)

    coo = base_graph.tocoo(copy=True)
    support_ratio = np.zeros_like(coo.data, dtype=np.float32)
    for idx, (row, col) in enumerate(zip(coo.row, coo.col, strict=False)):
        support_ratio[idx] = len(support_counter[(int(row), int(col))]) / max(n_segments, 1)
    coo.data = coo.data * support_ratio
    candidate = row_normalize(coo.tocsr())
    metadata = {
        "n_segments": float(n_segments),
        "segment_support_mean": float(np.mean(support_ratio)) if support_ratio.size else 0.0,
        "segment_support_median": float(np.median(support_ratio)) if support_ratio.size else 0.0,
        "segment_support_q90": float(np.quantile(support_ratio, 0.9)) if support_ratio.size else 0.0,
        "edge_supported_by_ge_2_segments_rate": float(np.mean(support_ratio >= (2.0 / max(n_segments, 1))))
        if support_ratio.size
        else 0.0,
    }
    return candidate, user_segment_df, metadata


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def render_summary_md(payload: dict[str, Any]) -> str:
    lines = [
        "# Coarse / Local Graph Diagnostics（粗图 / 局部图诊断）",
        "",
        f"- Dataset（数据集）: `{payload['dataset']}`",
        f"- Baseline tokenizer line（基线分词器线）: `{payload['baseline_line']}`",
        "",
        "## D530: `G_local`（局部图）多跳扩散",
        "",
        "| Variant（变体） | graph_nnz | outgoing_cov | connected_rate | overlap | topk_expansion | |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["d530"]["rows"]:
        lines.append(
            f"| `{row['name']}` | {row['graph_nnz']:.0f} | {row['outgoing_coverage_rate']:.4f} | "
            f"{row['connected_item_rate']:.4f} | {row['mean_neighbor_overlap_with_baseline']:.4f} | "
            f"{row['topk_expansion_ratio']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## D540: `G_coarse`（粗图）用户分群条件化",
            "",
            "| Variant（变体） | graph_nnz | connected_rate | overlap | seg_mean | seg_ge2_rate | |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["d540"]["rows"]:
        lines.append(
            f"| `{row['name']}` | {row['graph_nnz']:.0f} | {row['connected_item_rate']:.4f} | "
            f"{row['mean_neighbor_overlap_with_baseline']:.4f} | {row['segment_support_mean']:.4f} | "
            f"{row['edge_supported_by_ge_2_segments_rate']:.4f} |"
        )

    cir = payload["d541"]["metrics"]
    lines.extend(
        [
            "",
            "## D541: `G_coarse`（粗图）`CIR`（边可靠性）重加权",
            "",
            f"- `graph_nnz`: `{cir['graph_nnz']:.0f}`",
            f"- `connected_item_rate`（连通物品比例）: `{cir['connected_item_rate']:.4f}`",
            f"- `mean_neighbor_overlap_with_baseline`（与基线邻域重叠）: `{cir['mean_neighbor_overlap_with_baseline']:.4f}`",
            f"- `cir_mean`（平均 CIR）: `{cir['cir_mean']:.4f}`",
            f"- `cir_nonzero_rate`（非零 CIR 比例）: `{cir['cir_nonzero_rate']:.4f}`",
            "",
            "## D542: `G_coarse`（粗图）`MGDCF`（全局同构物品图）重构",
            "",
            "| Variant（变体） | graph_nnz | connected_rate | overlap | topk_expansion | |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in payload["d542"]["rows"]:
        lines.append(
            f"| `{row['name']}` | {row['graph_nnz']:.0f} | {row['connected_item_rate']:.4f} | "
            f"{row['mean_neighbor_overlap_with_baseline']:.4f} | {row['topk_expansion_ratio']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Quick Read（快速结论）",
            "",
            f"- `G_local`（局部图）最值得继续推进的候选：`{payload['recommendation']['local_pick']}`",
            f"- `G_coarse`（粗图）同源重加权分支最值得继续推进的候选：`{payload['recommendation']['coarse_pick']}`",
            f"- `G_coarse`（粗图）低风险对照是否值得推进：`{payload['recommendation']['cir_pick']}`",
            f"- `G_coarse`（粗图）重构分支最值得继续推进的候选：`{payload['recommendation']['mgdcf_pick']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)
    n_items = infer_num_items(train_df, test_df)
    popularity = build_popularity(train_df)

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
        mgdcf_keep_ratio=min(args.mgdcf_keep_ratios),
        mgdcf_binarize_edges=args.mgdcf_binarize_edges,
    )
    coarse_base = views["coarse_purified"].matrix.tocsr()  # type: ignore[attr-defined]
    local_base = views["local_purified"].matrix.tocsr()  # type: ignore[attr-defined]

    raw_coarse = build_coarse_graph(train_df, n_items=n_items, history_k=args.history_k)
    coarse_purified = purify_coarse_graph(raw_coarse, popularity=popularity, min_weight=args.coarse_min_weight)
    raw_local = build_local_graph(train_df, n_items=n_items, history_k=args.history_k)
    local_purified = purify_local_graph(raw_local, popularity=popularity, min_weight=args.local_min_weight)
    assert coarse_base.shape == coarse_purified.shape
    assert local_base.shape == local_purified.shape

    # D530
    d530_rows: list[dict[str, float | str]] = []
    d530_payload: dict[str, Any] = {"rows": d530_rows}
    for alpha in args.local_alpha_values:
        for max_hop in [2, args.max_local_hop]:
            if max_hop == 2 and alpha != args.local_alpha_values[0]:
                continue
            graph = build_multi_hop_local(local_purified, alpha=float(alpha), max_hop=int(max_hop))
            metrics = graph_metrics(graph, baseline_graph=local_base, topk=args.graph_topk)
            metrics["alpha"] = float(alpha)
            metrics["max_hop"] = float(max_hop)
            metrics["name"] = f"local_multihop_a{alpha:.2f}_h{max_hop}"
            d530_rows.append(metrics)
            sparse.save_npz(output_dir / f"{metrics['name']}.npz", keep_topk_per_row(graph, args.graph_topk))
    write_json(output_dir / "D530_local_multihop_summary.json", d530_payload)

    # D540
    d540_rows: list[dict[str, float | str]] = []
    d540_payload: dict[str, Any] = {"rows": d540_rows}
    user_segment_frames: list[pd.DataFrame] = []
    for k in args.segment_k_values:
        graph, user_segment_df, meta = build_user_segment_coarse(
            train_df=train_df,
            base_graph=coarse_purified,
            n_items=n_items,
            n_segments=int(k),
            seed=args.seed,
            history_k=args.history_k,
        )
        metrics = graph_metrics(graph, baseline_graph=coarse_base, topk=args.graph_topk)
        metrics.update(meta)
        metrics["name"] = f"coarse_user_segment_k{k}"
        d540_rows.append(metrics)
        user_segment_df = user_segment_df.copy()
        user_segment_df["k"] = int(k)
        user_segment_frames.append(user_segment_df)
        sparse.save_npz(output_dir / f"{metrics['name']}.npz", keep_topk_per_row(graph, args.graph_topk))
    if user_segment_frames:
        pd.concat(user_segment_frames, ignore_index=True).to_csv(output_dir / "D540_user_segment_assignments.csv", index=False)
    write_json(output_dir / "D540_user_segment_summary.json", d540_payload)

    # D541
    item_users = build_item_user_sets(train_df, n_items=n_items)
    cir_graph, cir_meta = build_cir_reweighted_coarse(coarse_purified, item_users=item_users, mix=args.cir_mix)
    d541_metrics = graph_metrics(cir_graph, baseline_graph=coarse_base, topk=args.graph_topk)
    d541_metrics.update(cir_meta)
    sparse.save_npz(output_dir / "D541_coarse_cir_reweighted.npz", keep_topk_per_row(cir_graph, args.graph_topk))
    write_json(output_dir / "D541_cir_summary.json", {"metrics": d541_metrics})

    # D542
    d542_rows: list[dict[str, float | str]] = []
    d542_payload: dict[str, Any] = {"rows": d542_rows}
    for keep_ratio in args.mgdcf_keep_ratios:
        graph = build_mgdcf_item_graph(
            train_df=train_df,
            n_items=n_items,
            keep_ratio=float(keep_ratio),
            binarize_edges=bool(args.mgdcf_binarize_edges),
        )
        metrics = graph_metrics(graph, baseline_graph=coarse_base, topk=args.graph_topk)
        metrics["name"] = f"coarse_mgdcf_r{keep_ratio:.4f}"
        metrics["mgdcf_keep_ratio"] = float(keep_ratio)
        metrics["mgdcf_binarize_edges"] = float(bool(args.mgdcf_binarize_edges))
        d542_rows.append(metrics)
        sparse.save_npz(output_dir / f"{metrics['name']}.npz", keep_topk_per_row(graph, args.graph_topk))
    write_json(output_dir / "D542_mgdcf_summary.json", d542_payload)

    local_pick = "cut"
    if d530_rows:
        local_pick = max(
            d530_rows,
            key=lambda row: (
                row["outgoing_coverage_rate"],
                -abs(row["mean_neighbor_overlap_with_baseline"] - 0.5),
                -row["orphan_item_rate"],
            ),
        )["name"]
    coarse_pick = "cut"
    if d540_rows:
        coarse_pick = max(
            d540_rows,
            key=lambda row: (
                row["segment_support_mean"],
                row["edge_supported_by_ge_2_segments_rate"],
                row["connected_item_rate"],
            ),
        )["name"]
    cir_pick = "promote" if d541_metrics["connected_item_rate"] >= max(0.95, float(coarse_base.nnz > 0) * 0.0) else "review"
    mgdcf_pick = "cut"
    if d542_rows:
        mgdcf_pick = max(
            d542_rows,
            key=lambda row: (
                row["topk_expansion_ratio"],
                1.0 - abs(row["mean_neighbor_overlap_with_baseline"] - 0.5),
                row["connected_item_rate"],
            ),
        )["name"]

    summary = {
        "dataset": "Industrial_and_Scientific",
        "baseline_line": "v2_on_p05 tokenizer graph bank",
        "d530": d530_payload,
        "d540": d540_payload,
        "d541": {"metrics": d541_metrics},
        "d542": d542_payload,
        "recommendation": {
            "local_pick": local_pick,
            "coarse_pick": coarse_pick,
            "cir_pick": cir_pick,
            "mgdcf_pick": mgdcf_pick,
        },
        "notes": [
            "These diagnostics are offline graph checks, not downstream verdicts.",
            "A candidate should be promoted only if diagnostics and tokenizer-side evidence agree.",
        ],
    }
    write_json(output_dir / "summary.json", summary)
    (output_dir / "SUMMARY.md").write_text(render_summary_md(summary))
    print(f"Diagnostics written to {output_dir}")


if __name__ == "__main__":
    main()
