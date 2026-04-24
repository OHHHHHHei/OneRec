#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


DEFAULT_METADATA_CSV = (
    "/home/leejt/OneRec/research-progress-log/experiment_launches/"
    "2026-04-17_mgr_sid_r650a_sft_eval_industrial/R650A_L1_CODEBOOK_ITEM_DETAIL.csv"
)
DEFAULT_EMBEDDING_PATH = "/home/leejt/OneRec/data/Amazon/index/Industrial_and_Scientific.emb-qwen-td.npy"
DEFAULT_OUTPUT_DIR = (
    "/home/leejt/OneRec/research-progress-log/experiment_launches/"
    "2026-04-18_mgr_sid_r670_clean_l1_semantic_l2_push_pull_industrial"
)

GENERIC_BRANDS = {
    "a",
    "an",
    "and",
    "for",
    "industrial",
    "scientific",
    "the",
    "with",
}
MULTI_WORD_BRANDS = (
    "3d solutech",
    "small parts",
    "micro swiss",
    "blue demon",
    "black diamond",
    "easy wood",
    "eisco labs",
    "eisco scientific",
    "nilight",
    "uxcell",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the R670a high-confidence L1 semantic graph.")
    parser.add_argument("--metadata-csv", default=DEFAULT_METADATA_CSV)
    parser.add_argument("--semantic-embedding-path", default=DEFAULT_EMBEDDING_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tag", default="R670a")
    parser.add_argument("--semantic-neighbor-topk", type=int, default=64)
    parser.add_argument("--strong-rank-topk", type=int, default=16)
    parser.add_argument("--l1-topk", type=int, default=16)
    parser.add_argument("--brand-bonus", type=float, default=1.25)
    return parser.parse_args()


def normalize_title(value: object) -> str:
    text = "" if not isinstance(value, str) else value.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_brand(title: object) -> str:
    text = normalize_title(title)
    if not text:
        return ""
    for brand in MULTI_WORD_BRANDS:
        if text.startswith(brand):
            return brand
    token = text.split(" ", 1)[0]
    if len(token) < 3 or token in GENERIC_BRANDS:
        return ""
    return token


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = matrix.astype(np.float32, copy=False)
    denom = np.linalg.norm(matrix, axis=1, keepdims=True)
    denom = np.maximum(denom, 1e-12)
    return matrix / denom


def row_normalize(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    matrix = matrix.tocsr().astype(np.float32)
    row_sum = np.asarray(matrix.sum(axis=1)).reshape(-1)
    inv = np.zeros_like(row_sum, dtype=np.float32)
    nz = row_sum > 0.0
    inv[nz] = 1.0 / row_sum[nz]
    return sparse.diags(inv).dot(matrix).tocsr().astype(np.float32)


def sorted_top_neighbors(similarity: np.ndarray, row: int, topk: int) -> list[tuple[int, float, int]]:
    scores = similarity[row].copy()
    scores[row] = -np.inf
    topk = min(topk, scores.shape[0] - 1)
    if topk <= 0:
        return []
    candidates = np.argpartition(-scores, topk - 1)[:topk]
    candidates = candidates[np.argsort(-scores[candidates])]
    return [(int(col), float(scores[col]), int(rank + 1)) for rank, col in enumerate(candidates)]


def build_graph(args: argparse.Namespace) -> tuple[sparse.csr_matrix, pd.DataFrame, dict[str, object]]:
    metadata = pd.read_csv(args.metadata_csv).sort_values("item_id").reset_index(drop=True)
    if metadata["item_id"].tolist() != list(range(len(metadata))):
        raise ValueError("metadata_csv must contain contiguous item_id values sorted from 0")

    embeddings = np.load(args.semantic_embedding_path)
    n_items = len(metadata)
    embeddings = l2_normalize(embeddings[:n_items])
    similarity = embeddings @ embeddings.T

    families = metadata["family"].fillna("other").astype(str).tolist()
    titles = metadata["title"].fillna("").astype(str).tolist()
    brands = [extract_brand(title) for title in titles]

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    pair_records: list[dict[str, object]] = []
    self_loop_rows = 0
    fallback_rows = 0
    positive_rows = 0

    for item_id in range(n_items):
        neighbors = sorted_top_neighbors(similarity, item_id, args.semantic_neighbor_topk)
        selected: list[dict[str, object]] = []
        seen: set[int] = set()
        item_family = families[item_id]
        item_brand = brands[item_id]

        for neighbor, sim, rank in neighbors:
            neighbor_family = families[neighbor]
            neighbor_brand = brands[neighbor]
            same_family = item_family == neighbor_family and item_family not in {"", "other"}
            same_brand = bool(item_brand) and item_brand == neighbor_brand
            strong_semantic = rank <= args.strong_rank_topk
            if not ((same_family and (same_brand or strong_semantic)) or same_brand):
                continue
            if neighbor in seen:
                continue
            seen.add(neighbor)
            selected.append(
                {
                    "item_b": neighbor,
                    "semantic_sim": sim,
                    "rank": rank,
                    "same_family": same_family,
                    "same_brand": same_brand,
                    "rule": "same_brand_or_family_strong_semantic",
                }
            )

        if not selected:
            for neighbor, sim, rank in neighbors:
                same_family = item_family == families[neighbor] and item_family not in {"", "other"}
                if same_family:
                    selected.append(
                        {
                            "item_b": neighbor,
                            "semantic_sim": sim,
                            "rank": rank,
                            "same_family": True,
                            "same_brand": bool(item_brand) and item_brand == brands[neighbor],
                            "rule": "fallback_same_family",
                        }
                    )
                    fallback_rows += 1
                    break

        selected = sorted(selected, key=lambda item: float(item["semantic_sim"]), reverse=True)[: args.l1_topk]

        if not selected:
            rows.append(item_id)
            cols.append(item_id)
            vals.append(1.0)
            self_loop_rows += 1
            continue

        positive_rows += 1
        for record in selected:
            neighbor = int(record["item_b"])
            sim = max(float(record["semantic_sim"]), 0.0)
            same_brand = bool(record["same_brand"])
            weight = max(sim, 1e-6) * (float(args.brand_bonus) if same_brand else 1.0)
            rows.append(item_id)
            cols.append(neighbor)
            vals.append(weight)
            pair_records.append(
                {
                    "item_a": item_id,
                    "item_b": neighbor,
                    "semantic_sim": float(record["semantic_sim"]),
                    "rank": int(record["rank"]),
                    "family_a": item_family,
                    "family_b": families[neighbor],
                    "brand_a": item_brand,
                    "brand_b": brands[neighbor],
                    "same_family": bool(record["same_family"]),
                    "same_brand": same_brand,
                    "weight": float(weight),
                    "rule": str(record["rule"]),
                }
            )

    graph = sparse.csr_matrix((vals, (rows, cols)), shape=(n_items, n_items), dtype=np.float32)
    graph = graph.maximum(graph.T).tocsr()
    graph = row_normalize(graph)

    pairs = pd.DataFrame.from_records(pair_records)
    summary = {
        "tag": args.tag,
        "n_items": int(n_items),
        "semantic_neighbor_topk": int(args.semantic_neighbor_topk),
        "strong_rank_topk": int(args.strong_rank_topk),
        "l1_topk": int(args.l1_topk),
        "brand_bonus": float(args.brand_bonus),
        "nnz_after_symmetry": int(graph.nnz),
        "positive_rows_before_symmetry": int(positive_rows),
        "fallback_same_family_rows": int(fallback_rows),
        "self_loop_rows_before_symmetry": int(self_loop_rows),
        "non_self_pair_rows": int(len(pairs)),
        "unique_non_self_pairs": int(len({tuple(sorted((int(a), int(b)))) for a, b in zip(pairs["item_a"], pairs["item_b"], strict=False)}))
        if not pairs.empty
        else 0,
        "family_pair_counts_top10": pairs["family_a"].value_counts().head(10).to_dict() if not pairs.empty else {},
    }
    return graph, pairs, summary


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    graph, pairs, summary = build_graph(args)
    graph_path = output_dir / f"{args.tag}_l1_high_conf_semantic_graph.npz"
    pair_path = output_dir / f"{args.tag}_l1_high_conf_semantic_pairs.csv"
    summary_path = output_dir / f"{args.tag}_l1_high_conf_semantic_graph_summary.json"

    sparse.save_npz(graph_path, graph)
    pairs.to_csv(pair_path, index=False)
    summary["output_graph"] = str(graph_path)
    summary["output_pairs"] = str(pair_path)

    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
