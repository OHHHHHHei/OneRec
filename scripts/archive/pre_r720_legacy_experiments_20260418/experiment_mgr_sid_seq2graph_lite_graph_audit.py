from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import csgraph

from onerec.experiments.mgr_sid.graph_bank import (
    build_coarse_graph,
    build_direct_support_graph,
    build_local_graph,
    build_popularity,
    build_seq2graph_context_matrix,
    build_seq2graph_reliability,
    build_seq2graph_rescue_graph,
    infer_num_items,
    keep_topk_per_row,
    purify_coarse_graph,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run D640 Seq2Graph-lite offline graph audit on Industrial."
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
        "--item-meta-json",
        default="/home/leejt/OneRec/data/Amazon/index/Industrial_and_Scientific.item.json",
    )
    parser.add_argument(
        "--hotspot-loss-analysis-json",
        default="/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-16_mgr_sid_r630c_sft_eval_industrial/LOSS_ITEM_GRAPH_SEMANTIC_V2_ON_P05_VS_R630C.json",
    )
    parser.add_argument(
        "--output-dir",
        default="/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-16_mgr_sid_d640_seq2graph_lite_graph_audit_industrial",
    )
    parser.add_argument("--history-k", type=int, default=10)
    parser.add_argument("--coarse-min-weight", type=float, default=2.0)
    parser.add_argument("--local-min-weight", type=float, default=1.0)
    parser.add_argument("--graph-topk", type=int, default=32)
    parser.add_argument("--seq2g-mix-alpha", type=float, default=0.35)
    parser.add_argument("--seq2g-context-topk", type=int, default=32)
    parser.add_argument("--seq2g-candidate-topm", type=int, default=32)
    parser.add_argument("--seq2g-direct-tau", type=float, default=0.5)
    parser.add_argument("--report-top-pairs", type=int, default=12)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_item_meta(path: Path) -> dict[int, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    result: dict[int, dict[str, Any]] = {}
    for key, value in payload.items():
        try:
            item_id = int(key)
        except (TypeError, ValueError):
            continue
        result[item_id] = {
            "title": str(value.get("title", "")),
            "brand": str(value.get("brand", "")),
            "categories": str(value.get("categories", "")),
        }
    return result


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
    baseline_graph: sparse.csr_matrix,
    direct_support: sparse.csr_matrix,
    direct_tau: float,
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

    graph_coo = graph.tocoo(copy=False)
    baseline_values = np.asarray(baseline_graph[graph_coo.row, graph_coo.col]).reshape(-1) if graph.nnz else np.asarray([], dtype=np.float32)
    direct_values = np.asarray(direct_support[graph_coo.row, graph_coo.col]).reshape(-1) if graph.nnz else np.asarray([], dtype=np.float32)
    novel_mask = baseline_values <= 0.0
    weak_mask = direct_values < float(direct_tau)
    rescue_mask = novel_mask & weak_mask

    metrics: dict[str, float] = {
        "graph_nnz": float(graph.nnz),
        "graph_density": float(graph.nnz / max(n_items * n_items, 1)),
        "avg_out_degree": float(np.mean(out_degree)) if out_degree.size else 0.0,
        "avg_in_degree": float(np.mean(in_degree)) if in_degree.size else 0.0,
        "connected_item_rate": float(np.mean((out_degree + in_degree) > 0)) if out_degree.size else 0.0,
        "largest_component_ratio": largest_ratio,
        "connected_component_count": float(n_components),
        "mean_neighbor_overlap_with_baseline": mean_neighbor_overlap(graph, baseline_graph, topk=topk),
        "novel_edge_ratio_vs_baseline": float(np.mean(novel_mask)) if novel_mask.size else 0.0,
        "direct_weak_edge_ratio": float(np.mean(weak_mask)) if weak_mask.size else 0.0,
        "rescue_edge_ratio": float(np.mean(rescue_mask)) if rescue_mask.size else 0.0,
    }
    metrics.update(expansion_stats(graph, baseline_graph, topk=topk))
    return metrics


def load_hotspot_pairs(path: Path, item_meta: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    pairs: list[dict[str, Any]] = []
    for case in payload.get("loss_case_studies", []):
        anchor = int(case["item_id"])
        anchor_meta = item_meta.get(anchor, {})
        for neighbor in case.get("top_semantic_neighbors", []):
            neighbor_id = int(neighbor["neighbor_item_id"])
            neighbor_meta = item_meta.get(neighbor_id, {})
            pairs.append(
                {
                    "anchor_item_id": anchor,
                    "anchor_title": case.get("title", anchor_meta.get("title", "")),
                    "anchor_brand": case.get("brand", anchor_meta.get("brand", "")),
                    "anchor_family": case.get("family", ""),
                    "neighbor_item_id": neighbor_id,
                    "neighbor_title": neighbor.get("neighbor_title", neighbor_meta.get("title", "")),
                    "neighbor_brand": neighbor.get("neighbor_brand", neighbor_meta.get("brand", "")),
                    "semantic_sim": float(neighbor.get("semantic_sim", 0.0)),
                }
            )
    return pairs


def pair_value(matrix: sparse.csr_matrix, row: int, col: int) -> float:
    return float(matrix[row, col])


def build_hotspot_pair_frame(
    hotspot_pairs: list[dict[str, Any]],
    context_affinity: sparse.csr_matrix,
    direct_support: sparse.csr_matrix,
    graphs: dict[str, sparse.csr_matrix],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for pair in hotspot_pairs:
        row = dict(pair)
        anchor = int(pair["anchor_item_id"])
        neighbor = int(pair["neighbor_item_id"])
        row["context_affinity"] = pair_value(context_affinity, anchor, neighbor)
        row["direct_support"] = pair_value(direct_support, anchor, neighbor)
        row["direct_zero"] = int(row["direct_support"] <= 0.0)
        for name, graph in graphs.items():
            row[f"{name}_affinity"] = pair_value(graph, anchor, neighbor)
            row[f"{name}_visible"] = int(row[f"{name}_affinity"] > 0.0)
        row["coarse_delta_ctx_only"] = row["coarse_seq2g_ctx_only_affinity"] - row["coarse_purified_affinity"]
        row["coarse_delta_rel"] = row["coarse_seq2g_rel_affinity"] - row["coarse_purified_affinity"]
        row["coarse_delta_rel_masked"] = row["coarse_seq2g_rel_masked_affinity"] - row["coarse_purified_affinity"]
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_hotspot_visibility(
    hotspot_df: pd.DataFrame,
    direct_tau: float,
    graph_names: list[str],
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    direct_weak = hotspot_df["direct_support"] < float(direct_tau)
    predecessor_sharing = hotspot_df["context_affinity"] > 0.0
    direct_zero = hotspot_df["direct_support"] <= 0.0
    for name in graph_names:
        visible = hotspot_df[f"{name}_visible"] > 0
        summaries.append(
            {
                "graph": name,
                "pair_count": int(len(hotspot_df)),
                "visible_fraction": float(visible.mean()) if len(hotspot_df) else 0.0,
                "mean_affinity": float(hotspot_df[f"{name}_affinity"].mean()) if len(hotspot_df) else 0.0,
                "direct_weak_visible_fraction": float((visible & direct_weak).sum() / max(int(direct_weak.sum()), 1)),
                "direct_zero_visible_fraction": float((visible & direct_zero).sum() / max(int(direct_zero.sum()), 1)),
                "predecessor_sharing_visible_fraction": float((visible & predecessor_sharing).sum() / max(int(predecessor_sharing.sum()), 1)),
                "predecessor_sharing_direct_zero_visible_fraction": float(
                    (visible & predecessor_sharing & direct_zero).sum() / max(int((predecessor_sharing & direct_zero).sum()), 1)
                ),
            }
        )
    return summaries


def top_rescued_pairs(hotspot_df: pd.DataFrame, delta_column: str, limit: int) -> list[dict[str, Any]]:
    if hotspot_df.empty:
        return []
    subset = hotspot_df.sort_values(by=[delta_column, "context_affinity", "semantic_sim"], ascending=[False, False, False]).head(limit)
    columns = [
        "anchor_item_id",
        "anchor_title",
        "neighbor_item_id",
        "neighbor_title",
        "semantic_sim",
        "context_affinity",
        "direct_support",
        "coarse_purified_affinity",
        delta_column,
    ]
    return subset[columns].to_dict(orient="records")


def write_summary_md(
    output_path: Path,
    args: argparse.Namespace,
    graph_summary: dict[str, dict[str, float]],
    hotspot_summary: list[dict[str, Any]],
    top_pairs: dict[str, list[dict[str, Any]]],
) -> None:
    lines: list[str] = []
    lines.append("# D640 Seq2Graph-lite Graph Audit（图审计）\n")
    lines.append("## Scope（范围）\n")
    lines.append("- dataset（数据集）: `Industrial_and_Scientific`")
    lines.append("- status（状态）: `completed（已完成）`")
    lines.append("- role（角色）: engineering filter（工程过滤）, not scientific verdict（不是科学裁决）\n")

    lines.append("## Settings（设置）\n")
    lines.append(f"- `seq2g_mix_alpha = {args.seq2g_mix_alpha}`")
    lines.append(f"- `seq2g_context_topk = {args.seq2g_context_topk}`")
    lines.append(f"- `seq2g_candidate_topm = {args.seq2g_candidate_topm}`")
    lines.append(f"- `seq2g_direct_tau = {args.seq2g_direct_tau}`\n")

    lines.append("## Graph Summary（图摘要）\n")
    lines.append("| graph | nnz | connected_rate | overlap | novel_edge_ratio | rescue_edge_ratio | topk_expansion | |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
    for name, metrics in graph_summary.items():
        lines.append(
            f"| `{name}` | {int(metrics['graph_nnz'])} | {metrics['connected_item_rate']:.4f} | {metrics['mean_neighbor_overlap_with_baseline']:.4f} | {metrics['novel_edge_ratio_vs_baseline']:.4f} | {metrics['rescue_edge_ratio']:.4f} | {metrics['topk_expansion_ratio']:.4f} | |"
        )

    lines.append("\n## Hotspot Visibility（热点可见性）\n")
    lines.append("| graph | visible_fraction | direct_weak_visible | direct_zero_visible | predecessor_visible | predecessor_direct_zero_visible | |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for row in hotspot_summary:
        lines.append(
            f"| `{row['graph']}` | {row['visible_fraction']:.4f} | {row['direct_weak_visible_fraction']:.4f} | {row['direct_zero_visible_fraction']:.4f} | {row['predecessor_sharing_visible_fraction']:.4f} | {row['predecessor_sharing_direct_zero_visible_fraction']:.4f} | |"
        )

    lines.append("\n## Top Rescued Hotspot Pairs（最典型补盲热点对）\n")
    for name, rows in top_pairs.items():
        lines.append(f"### `{name}`\n")
        if not rows:
            lines.append("- no rescued pairs（没有补盲物品对）\n")
            continue
        lines.append("| anchor | neighbor | semantic_sim | context_affinity | direct_support | baseline | delta | |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---|")
        delta_column = {
            "coarse_seq2g_ctx_only": "coarse_delta_ctx_only",
            "coarse_seq2g_rel": "coarse_delta_rel",
            "coarse_seq2g_rel_masked": "coarse_delta_rel_masked",
        }[name]
        for row in rows:
            lines.append(
                f"| {row['anchor_item_id']}: {row['anchor_title']} | {row['neighbor_item_id']}: {row['neighbor_title']} | {row['semantic_sim']:.4f} | {row['context_affinity']:.4f} | {row['direct_support']:.4f} | {row['coarse_purified_affinity']:.4f} | {row[delta_column]:.4f} | |"
            )
        lines.append("")

    lines.append("## Quick Read（快速结论）\n")
    lines.append("- 如果 `coarse_seq2g_rel_masked`（带掩码的可靠性感知补盲粗图）在 `direct_zero_visible_fraction`（直接零连接可见率）和 `predecessor_sharing_direct_zero_visible_fraction`（前驱共享且直接零连接可见率）上明显更高，就说明它确实在补 blind spot（盲区），而不是只做全局加边。")
    lines.append("- 如果 `coarse_seq2g_ctx_only`（仅上下文补盲粗图）有更高的 `topk_expansion_ratio`（邻域扩张率）但更低的 `rescue_edge_ratio`（补盲边比例），就说明 reliability（可靠性）和 mask（掩码）是必要的。")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)
    item_meta = load_item_meta(Path(args.item_meta_json))

    n_items = infer_num_items(train_df, test_df)
    popularity = build_popularity(train_df)
    coarse_raw = build_coarse_graph(train_df, n_items=n_items, history_k=args.history_k)
    local_raw = build_local_graph(train_df, n_items=n_items, history_k=args.history_k)
    coarse_purified = purify_coarse_graph(coarse_raw, popularity=popularity, min_weight=args.coarse_min_weight)
    direct_support = build_direct_support_graph(coarse_raw=coarse_raw, local_raw=local_raw)

    context_affinity = build_seq2graph_context_matrix(
        local_raw=local_raw,
        context_topk=args.seq2g_context_topk,
        candidate_topm=args.seq2g_candidate_topm,
    )
    reliability = build_seq2graph_reliability(
        local_raw=local_raw,
        context_affinity=context_affinity,
    )
    coarse_seq2g_ctx_only, rescue_ctx_only = build_seq2graph_rescue_graph(
        coarse_graph=coarse_purified,
        context_affinity=context_affinity,
        mix_alpha=args.seq2g_mix_alpha,
        context_topk=args.seq2g_context_topk,
        reliability=None,
        direct_support=direct_support,
        direct_tau=args.seq2g_direct_tau,
        use_reliability=False,
        use_direct_weak_mask=False,
    )
    coarse_seq2g_rel, rescue_rel = build_seq2graph_rescue_graph(
        coarse_graph=coarse_purified,
        context_affinity=context_affinity,
        mix_alpha=args.seq2g_mix_alpha,
        context_topk=args.seq2g_context_topk,
        reliability=reliability,
        direct_support=direct_support,
        direct_tau=args.seq2g_direct_tau,
        use_reliability=True,
        use_direct_weak_mask=False,
    )
    coarse_seq2g_rel_masked, rescue_rel_masked = build_seq2graph_rescue_graph(
        coarse_graph=coarse_purified,
        context_affinity=context_affinity,
        mix_alpha=args.seq2g_mix_alpha,
        context_topk=args.seq2g_context_topk,
        reliability=reliability,
        direct_support=direct_support,
        direct_tau=args.seq2g_direct_tau,
        use_reliability=True,
        use_direct_weak_mask=True,
    )

    graphs = {
        "coarse_purified": keep_topk_per_row(coarse_purified, topk=args.graph_topk),
        "coarse_seq2g_ctx_only": keep_topk_per_row(coarse_seq2g_ctx_only, topk=args.graph_topk),
        "coarse_seq2g_rel": keep_topk_per_row(coarse_seq2g_rel, topk=args.graph_topk),
        "coarse_seq2g_rel_masked": keep_topk_per_row(coarse_seq2g_rel_masked, topk=args.graph_topk),
    }

    graph_summary = {
        name: graph_metrics(
            graph=graph,
            baseline_graph=graphs["coarse_purified"],
            direct_support=direct_support,
            direct_tau=args.seq2g_direct_tau,
            topk=args.graph_topk,
        )
        for name, graph in graphs.items()
    }

    hotspot_pairs = load_hotspot_pairs(Path(args.hotspot_loss_analysis_json), item_meta=item_meta)
    hotspot_df = build_hotspot_pair_frame(
        hotspot_pairs=hotspot_pairs,
        context_affinity=context_affinity,
        direct_support=direct_support,
        graphs=graphs,
    )
    hotspot_summary = summarize_hotspot_visibility(
        hotspot_df=hotspot_df,
        direct_tau=args.seq2g_direct_tau,
        graph_names=list(graphs.keys()),
    )

    top_pairs = {
        "coarse_seq2g_ctx_only": top_rescued_pairs(hotspot_df, delta_column="coarse_delta_ctx_only", limit=args.report_top_pairs),
        "coarse_seq2g_rel": top_rescued_pairs(hotspot_df, delta_column="coarse_delta_rel", limit=args.report_top_pairs),
        "coarse_seq2g_rel_masked": top_rescued_pairs(hotspot_df, delta_column="coarse_delta_rel_masked", limit=args.report_top_pairs),
    }

    sparse.save_npz(output_dir / "coarse_seq2g_ctx_only.npz", coarse_seq2g_ctx_only)
    sparse.save_npz(output_dir / "coarse_seq2g_rel.npz", coarse_seq2g_rel)
    sparse.save_npz(output_dir / "coarse_seq2g_rel_masked.npz", coarse_seq2g_rel_masked)
    sparse.save_npz(output_dir / "seq2g_context_affinity.npz", context_affinity)
    sparse.save_npz(output_dir / "seq2g_reliability.npz", reliability)
    sparse.save_npz(output_dir / "seq2g_rescue_ctx_only.npz", rescue_ctx_only)
    sparse.save_npz(output_dir / "seq2g_rescue_rel.npz", rescue_rel)
    sparse.save_npz(output_dir / "seq2g_rescue_rel_masked.npz", rescue_rel_masked)

    hotspot_df.to_csv(output_dir / "D640_hotspot_semantic_pairs.csv", index=False)
    summary = {
        "args": vars(args),
        "graph_summary": graph_summary,
        "hotspot_visibility": hotspot_summary,
        "top_rescued_pairs": top_pairs,
        "hotspot_pair_count": int(len(hotspot_df)),
    }
    (output_dir / "D640_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_summary_md(
        output_path=output_dir / "SUMMARY.md",
        args=args,
        graph_summary=graph_summary,
        hotspot_summary=hotspot_summary,
        top_pairs=top_pairs,
    )


if __name__ == "__main__":
    main()
