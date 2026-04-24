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
from onerec.experiments.mgr_sid.paper_transplants import keep_topk_per_row
from onerec.experiments.mgr_sid.transplanted_graph_bank import build_transplanted_graph_bank


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build graph sources for the hierarchical collaboration-only MGR-SID branch."
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
        default="/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-18_mgr_sid_r693_hier_collab_only_multihop_industrial",
    )
    parser.add_argument("--tag", default="R693a")
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
    parser.add_argument("--coarse-view-name", default="coarse_purified")
    parser.add_argument("--mid-view-name", default="local_multihop")
    parser.add_argument("--l1-topk", type=int, default=8)
    parser.add_argument("--l1-quantile", type=float, default=0.75)
    parser.add_argument("--negative-candidate-topk", type=int, default=16)
    parser.add_argument("--negative-mid-weak-quantile", type=float, default=0.25)
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


def get_topk_row_entries(matrix: sparse.csr_matrix, row: int, topk: int) -> list[tuple[int, float]]:
    start, end = matrix.indptr[row], matrix.indptr[row + 1]
    cols = matrix.indices[start:end]
    vals = matrix.data[start:end]
    if len(cols) == 0:
        return []
    order = np.argsort(vals)[::-1]
    if topk > 0:
        order = order[:topk]
    return [(int(cols[idx]), float(vals[idx])) for idx in order if int(cols[idx]) != row and float(vals[idx]) > 0.0]


def build_l1_highconf_graph(
    coarse_directed: sparse.csr_matrix,
    l1_topk: int,
    l1_quantile: float,
) -> tuple[sparse.csr_matrix, dict[str, float]]:
    n_items = coarse_directed.shape[0]
    selected: list[dict[int, float]] = []
    for row in range(n_items):
        candidates = get_topk_row_entries(coarse_directed, row, l1_topk)
        if not candidates:
            selected.append({})
            continue
        vals = np.asarray([val for _, val in candidates], dtype=np.float32)
        threshold = float(np.quantile(vals, l1_quantile))
        selected.append({col: val for col, val in candidates if val >= threshold})

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    undirected_edges = 0
    covered_items: set[int] = set()
    for i in range(n_items):
        for j, val_ij in selected[i].items():
            if i >= j:
                continue
            val_ji = selected[j].get(i)
            if val_ji is None:
                continue
            weight = float(np.sqrt(max(val_ij, 0.0) * max(val_ji, 0.0)))
            if weight <= 0.0:
                continue
            rows.extend([i, j])
            cols.extend([j, i])
            data.extend([weight, weight])
            undirected_edges += 1
            covered_items.add(i)
            covered_items.add(j)

    graph = sparse.coo_matrix((data, (rows, cols)), shape=coarse_directed.shape, dtype=np.float32).tocsr()
    graph = row_normalize(graph).tocsr().astype(np.float32)
    summary = {
        "l1_graph_undirected_edge_count": int(undirected_edges),
        "l1_graph_item_coverage_rate": float(len(covered_items) / max(n_items, 1)),
        "l1_topk": int(l1_topk),
        "l1_quantile": float(l1_quantile),
    }
    return graph, summary


def build_negative_pair_csv(
    coarse_sym: sparse.csr_matrix,
    mid_sym: sparse.csr_matrix,
    candidate_topk: int,
    weak_quantile: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    n_items = coarse_sym.shape[0]
    records: list[dict[str, float | int | str]] = []
    covered_items: set[int] = set()
    weak_thresholds: list[float] = []
    mid_affinities_kept: list[float] = []
    coarse_affinities_kept: list[float] = []
    reliability_kept: list[float] = []

    for item_a in range(n_items):
        candidates = get_topk_row_entries(coarse_sym, item_a, candidate_topk)
        if not candidates:
            continue
        candidate_mid = np.asarray([float(mid_sym[item_a, item_b]) for item_b, _ in candidates], dtype=np.float32)
        weak_threshold = float(np.quantile(candidate_mid, weak_quantile))
        weak_thresholds.append(weak_threshold)
        for item_b, coarse_affinity in candidates:
            if item_a >= item_b:
                continue
            mid_affinity = float(mid_sym[item_a, item_b])
            if mid_affinity > weak_threshold:
                continue
            if weak_threshold > 0.0:
                weakness = 1.0 - min(mid_affinity / weak_threshold, 1.0)
            else:
                weakness = 1.0 if mid_affinity <= 0.0 else 0.0
            reliability = float(coarse_affinity * weakness)
            if reliability <= 0.0:
                continue
            records.append(
                {
                    "item_a": int(item_a),
                    "item_b": int(item_b),
                    "coarse_affinity": float(coarse_affinity),
                    "mid_affinity": float(mid_affinity),
                    "reliability": reliability,
                    "rule": "coarse_candidate_mid_graph_weak",
                    "weak_threshold": weak_threshold,
                }
            )
            covered_items.add(item_a)
            covered_items.add(item_b)
            mid_affinities_kept.append(mid_affinity)
            coarse_affinities_kept.append(float(coarse_affinity))
            reliability_kept.append(reliability)

    df = pd.DataFrame.from_records(records)
    if not df.empty:
        df = df.sort_values(by=["reliability", "coarse_affinity"], ascending=[False, False]).reset_index(drop=True)

    summary = {
        "negative_candidate_topk": int(candidate_topk),
        "negative_mid_weak_quantile": float(weak_quantile),
        "weak_pair_count": int(len(df)),
        "weak_pair_item_coverage_rate": float(len(covered_items) / max(n_items, 1)),
        "weak_threshold_mean": float(np.mean(weak_thresholds)) if weak_thresholds else 0.0,
        "mid_affinity_mean": float(np.mean(mid_affinities_kept)) if mid_affinities_kept else 0.0,
        "coarse_affinity_mean": float(np.mean(coarse_affinities_kept)) if coarse_affinities_kept else 0.0,
        "reliability_mean": float(np.mean(reliability_kept)) if reliability_kept else 0.0,
    }
    return df, summary


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)
    n_items = infer_num_items(train_df, test_df)
    views = build_views(args, train_df, test_df)

    if args.coarse_view_name not in views:
        raise KeyError(f"Unknown coarse view: {args.coarse_view_name}. Available views: {sorted(views.keys())}")
    if args.mid_view_name not in views:
        raise KeyError(f"Unknown mid view: {args.mid_view_name}. Available views: {sorted(views.keys())}")

    coarse_directed = keep_topk_per_row(views[args.coarse_view_name].matrix, topk=args.graph_topk).tocsr().astype(np.float32)
    coarse_sym = coarse_directed.maximum(coarse_directed.T).tocsr().astype(np.float32)
    mid_directed = keep_topk_per_row(views[args.mid_view_name].matrix, topk=args.graph_topk).tocsr().astype(np.float32)
    mid_sym = mid_directed.maximum(mid_directed.T).tocsr().astype(np.float32)

    l1_graph, l1_summary = build_l1_highconf_graph(
        coarse_directed=coarse_directed,
        l1_topk=args.l1_topk,
        l1_quantile=args.l1_quantile,
    )
    negative_df, negative_summary = build_negative_pair_csv(
        coarse_sym=coarse_sym,
        mid_sym=mid_sym,
        candidate_topk=args.negative_candidate_topk,
        weak_quantile=args.negative_mid_weak_quantile,
    )

    tag = args.tag
    l1_graph_path = output_dir / f"{tag}_l1_coarse_highconf_graph.npz"
    negative_csv_path = output_dir / f"{tag}_all_mid_graph_weak_pairs.csv"
    top_negative_csv_path = output_dir / f"{tag}_top_mid_graph_weak_pairs.csv"
    summary_path = output_dir / f"{tag}_graph_source_summary.json"

    sparse.save_npz(l1_graph_path, l1_graph)
    negative_df.to_csv(negative_csv_path, index=False)
    negative_df.head(200).to_csv(top_negative_csv_path, index=False)

    summary = {
        "tag": tag,
        "n_items": int(n_items),
        "coarse_view_name": args.coarse_view_name,
        "mid_view_name": args.mid_view_name,
        "graph_topk": int(args.graph_topk),
        "l1_graph_path": str(l1_graph_path),
        "negative_pair_csv": str(negative_csv_path),
        "negative_pair_top_csv": str(top_negative_csv_path),
    }
    summary.update(l1_summary)
    summary.update(negative_summary)

    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
