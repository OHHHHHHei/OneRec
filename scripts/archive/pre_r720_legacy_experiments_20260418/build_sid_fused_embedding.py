#!/usr/bin/env python
from __future__ import annotations

import argparse
import ast
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD


def parse_id_list(value: Any) -> list[int]:
    if isinstance(value, list):
        return [int(v) for v in value]
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return []
    if not isinstance(parsed, list):
        return []
    result: list[int] = []
    for item in parsed:
        try:
            result.append(int(item))
        except (TypeError, ValueError):
            continue
    return result


def l2_normalize_rows(array: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    return array / norms


def build_transition_matrix(train_csv: Path, n_items: int, history_k: int) -> tuple[sparse.csr_matrix, np.ndarray]:
    df = pd.read_csv(train_csv)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    pop_counter: Counter = Counter()
    for _, row in df.iterrows():
        history = parse_id_list(row.get("history_item_id"))
        try:
            target = int(row["item_id"])
        except (TypeError, ValueError):
            continue
        if target < 0 or target >= n_items:
            continue
        pop_counter[target] += 1
        if not history:
            continue
        recent = list(reversed(history[-history_k:]))
        for rank, hist_item in enumerate(recent, start=1):
            if hist_item < 0 or hist_item >= n_items:
                continue
            rows.append(hist_item)
            cols.append(target)
            data.append(1.0 / rank)
    matrix = sparse.coo_matrix((data, (rows, cols)), shape=(n_items, n_items), dtype=np.float32).tocsr()
    matrix.sum_duplicates()
    pop = np.zeros(n_items, dtype=np.float32)
    for item_id, count in pop_counter.items():
        pop[item_id] = count
    return matrix, pop


def build_collaborative_embedding(matrix: sparse.csr_matrix, rank: int, seed: int) -> np.ndarray:
    if matrix.nnz == 0:
        return np.zeros((matrix.shape[0], rank * 2), dtype=np.float32)
    transformed = matrix.copy().astype(np.float32)
    transformed.data = np.log1p(transformed.data)
    svd = TruncatedSVD(n_components=rank, random_state=seed)
    history_role = svd.fit_transform(transformed)
    target_role = svd.components_.T * svd.singular_values_
    cf = np.concatenate([history_role, target_role], axis=1).astype(np.float32)
    return l2_normalize_rows(cf)


def zscore(values: np.ndarray) -> np.ndarray:
    std = values.std()
    if std <= 1e-12:
        return np.zeros_like(values)
    return (values - values.mean()) / std


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build fused SID embeddings for V0.5 experiments.")
    parser.add_argument("--base_embedding_path", required=True)
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--history_k", type=int, default=10)
    parser.add_argument("--svd_rank", type=int, default=32)
    parser.add_argument("--cf_weight", type=float, default=0.35)
    parser.add_argument("--include_popularity", action="store_true")
    parser.add_argument("--pop_weight", type=float, default=0.1)
    parser.add_argument("--control", choices=["none", "shuffled_cf", "popularity_only"], default="none")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--summary_path", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    base_embedding_path = Path(args.base_embedding_path)
    train_csv = Path(args.train_csv)
    output_path = Path(args.output_path)
    summary_path = Path(args.summary_path) if args.summary_path else output_path.with_suffix(".summary.json")

    base_embeddings = np.load(base_embedding_path).astype(np.float32)
    n_items = base_embeddings.shape[0]

    matrix, popularity = build_transition_matrix(train_csv, n_items, args.history_k)
    cf = build_collaborative_embedding(matrix, args.svd_rank, args.seed)

    control = args.control
    include_popularity = args.include_popularity or control == "popularity_only"
    if control == "shuffled_cf":
        permutation = rng.permutation(n_items)
        cf = cf[permutation]
    elif control == "popularity_only":
        cf = np.zeros_like(cf)

    base_norm = l2_normalize_rows(base_embeddings)
    features = [base_norm]
    if cf.shape[1] > 0:
        features.append((args.cf_weight * cf).astype(np.float32))
    if include_popularity:
        pop_feature = zscore(np.log1p(popularity)).astype(np.float32).reshape(-1, 1)
        features.append((args.pop_weight * pop_feature).astype(np.float32))

    fused = np.concatenate(features, axis=1).astype(np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, fused)

    summary = {
        "base_embedding_path": str(base_embedding_path),
        "train_csv": str(train_csv),
        "output_path": str(output_path),
        "control": control,
        "history_k": args.history_k,
        "svd_rank": args.svd_rank,
        "cf_weight": args.cf_weight,
        "include_popularity": include_popularity,
        "pop_weight": args.pop_weight if include_popularity else 0.0,
        "seed": args.seed,
        "n_items": int(n_items),
        "base_dim": int(base_embeddings.shape[1]),
        "cf_dim": int(cf.shape[1]),
        "fused_dim": int(fused.shape[1]),
        "transition_nnz": int(matrix.nnz),
        "pop_nonzero_items": int((popularity > 0).sum()),
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(output_path)
    print(summary_path)


if __name__ == "__main__":
    main()
