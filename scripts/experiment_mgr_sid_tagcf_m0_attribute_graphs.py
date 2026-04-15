#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import csgraph
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

from onerec.experiments.mgr_sid.transplanted_graph_bank import build_transplanted_graph_bank


STOPWORDS = set(ENGLISH_STOP_WORDS) | {
    "inch",
    "inches",
    "pack",
    "count",
    "steel",
    "tool",
    "tools",
    "product",
    "products",
    "use",
    "used",
    "using",
    "ideal",
    "includes",
    "include",
    "set",
    "kit",
    "item",
    "items",
    "new",
    "old",
    "high",
    "low",
    "heavy",
    "duty",
    "black",
    "white",
    "blue",
    "red",
    "green",
    "yellow",
    "gray",
    "grey",
    "brown",
    "silver",
    "gold",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TAGCF-inspired M0 attribute-graph experiments.")
    parser.add_argument(
        "--item-json",
        default="/home/leejt/OneRec/data/Amazon/index/Industrial_and_Scientific.item.json",
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
        "--output-root",
        default="/home/leejt/OneRec/results/tagcf_branch/20260415_m0_attribute_graphs",
    )
    parser.add_argument("--dataset-key", default="industrial")
    parser.add_argument("--history-k", type=int, default=10)
    parser.add_argument("--coarse-min-weight", type=float, default=2.0)
    parser.add_argument("--local-min-weight", type=float, default=1.0)
    parser.add_argument("--community-clusters", type=int, default=64)
    parser.add_argument("--anchor-topk", type=int, default=32)
    parser.add_argument("--semantic-mix", type=float, default=0.35)
    parser.add_argument("--spectral-rank", type=int, default=48)
    parser.add_argument("--band-low", type=float, default=0.25)
    parser.add_argument("--band-high", type=float, default=0.65)
    parser.add_argument("--temporal-mix", type=float, default=0.35)
    parser.add_argument("--graph-topk", type=int, default=32)
    parser.add_argument("--neighbor-topk", type=int, default=20)
    parser.add_argument("--raw-max-attrs", type=int, default=6)
    parser.add_argument("--fused-max-attrs", type=int, default=5)
    parser.add_argument("--heuristic-max-attrs", type=int, default=4)
    parser.add_argument("--min-fused-df", type=int, default=2)
    parser.add_argument("--sample-preview-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def _safe_literal_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if str(v).strip()]
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except Exception:
        return [text]
    if isinstance(parsed, list):
        return [str(v) for v in parsed if str(v).strip()]
    return [str(parsed)]


def load_item_meta(path: Path) -> list[dict[str, object]]:
    raw = json.loads(path.read_text())
    items: list[dict[str, object]] = []
    for key, value in sorted(raw.items(), key=lambda kv: int(kv[0])):
        item_id = int(key)
        if isinstance(value, dict):
            title = str(value.get("title", f"Item_{item_id}")).strip()
            brand = str(value.get("brand", "")).strip()
            desc_list = _safe_literal_list(value.get("description", []))
        else:
            title = str(value).strip()
            brand = ""
            desc_list = []
        description = " ".join(desc_list).strip()
        items.append(
            {
                "item_id": item_id,
                "title": title,
                "brand": brand,
                "description": description,
            }
        )
    return items


def build_text_docs(items: list[dict[str, object]]) -> tuple[list[str], list[str]]:
    raw_docs: list[str] = []
    title_docs: list[str] = []
    for item in items:
        title = str(item["title"])
        desc = str(item["description"])
        brand = str(item["brand"])
        raw_docs.append(" ".join(part for part in [title, brand, desc] if part).strip())
        title_docs.append(" ".join(part for part in [title, brand] if part).strip())
    return raw_docs, title_docs


def clean_phrase_tokens(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9][a-z0-9/-]*", text.lower())
    clean: list[str] = []
    for token in tokens:
        token = token.strip("-/ ")
        if not token:
            continue
        if token in STOPWORDS:
            continue
        if token.isdigit():
            continue
        if len(token) <= 2 and not any(ch.isdigit() for ch in token):
            continue
        clean.append(token)
    return clean


def singularize_token(token: str) -> str:
    if len(token) <= 4:
        return token
    if token.endswith("ies") and len(token) > 5:
        return token[:-3] + "y"
    if token.endswith("es") and len(token) > 5 and not token.endswith(("ses", "xes", "zes")):
        return token[:-2]
    if token.endswith("s") and not token.endswith(("ss", "ics")):
        return token[:-1]
    return token


def phrase_is_bad(tokens: list[str]) -> bool:
    if not tokens:
        return True
    if all(token.isdigit() for token in tokens):
        return True
    letter_tokens = sum(any(ch.isalpha() for ch in token) for token in tokens)
    if letter_tokens == 0:
        return True
    return False


def build_vectorizer(raw_docs: list[str], title_docs: list[str]) -> tuple[TfidfVectorizer, sparse.csr_matrix, TfidfVectorizer, sparse.csr_matrix]:
    raw_vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 3),
        max_features=60000,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9/-]+\b",
    )
    title_vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=30000,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9/-]+\b",
    )
    raw_matrix = raw_vectorizer.fit_transform(raw_docs)
    title_matrix = title_vectorizer.fit_transform(title_docs)
    return raw_vectorizer, raw_matrix.tocsr(), title_vectorizer, title_matrix.tocsr()


def select_doc_phrases(
    matrix: sparse.csr_matrix,
    feature_names: np.ndarray,
    max_attrs: int,
) -> tuple[list[list[str]], list[list[float]]]:
    all_phrases: list[list[str]] = []
    all_scores: list[list[float]] = []
    for row in range(matrix.shape[0]):
        start, end = matrix.indptr[row], matrix.indptr[row + 1]
        indices = matrix.indices[start:end]
        values = matrix.data[start:end]
        if values.size == 0:
            all_phrases.append([])
            all_scores.append([])
            continue
        order = np.argsort(values)[::-1]
        selected_phrases: list[str] = []
        selected_scores: list[float] = []
        selected_sets: list[set[str]] = []
        for idx in order:
            phrase = str(feature_names[indices[idx]])
            score = float(values[idx])
            tokens = clean_phrase_tokens(phrase)
            if phrase_is_bad(tokens):
                continue
            token_set = set(tokens)
            if any(token_set <= prev or prev <= token_set for prev in selected_sets):
                continue
            selected_phrases.append(" ".join(tokens))
            selected_scores.append(score)
            selected_sets.append(token_set)
            if len(selected_phrases) >= max_attrs:
                break
        all_phrases.append(selected_phrases)
        all_scores.append(selected_scores)
    return all_phrases, all_scores


def canonicalize_phrase(phrase: str) -> str:
    tokens = [singularize_token(tok) for tok in clean_phrase_tokens(phrase)]
    tokens = sorted(dict.fromkeys(tokens))
    if not tokens:
        return ""
    return " ".join(tokens)


def fuse_attributes(
    raw_attrs: list[list[str]],
    raw_scores: list[list[float]],
    max_attrs: int,
    min_df: int,
) -> tuple[list[list[str]], list[list[float]], dict[str, str], dict[str, int]]:
    canonical_counts: Counter[str] = Counter()
    canonical_surface_counts: dict[str, Counter[str]] = defaultdict(Counter)
    item_canonical: list[list[str]] = []
    item_canonical_scores: list[list[float]] = []

    for attrs, scores in zip(raw_attrs, raw_scores, strict=False):
        bucket: dict[str, float] = defaultdict(float)
        for attr, score in zip(attrs, scores, strict=False):
            canon = canonicalize_phrase(attr)
            if not canon:
                continue
            bucket[canon] += float(score)
            canonical_surface_counts[canon][attr] += 1
        item_canonical.append(list(bucket.keys()))
        item_canonical_scores.append(list(bucket.values()))
        for canon in bucket.keys():
            canonical_counts[canon] += 1

    keep = {canon for canon, df in canonical_counts.items() if df >= min_df}
    canonical_labels: dict[str, str] = {}
    for canon in keep:
        surface_counter = canonical_surface_counts[canon]
        best_surface = sorted(surface_counter.items(), key=lambda kv: (-kv[1], len(kv[0]), kv[0]))[0][0]
        canonical_labels[canon] = best_surface

    fused_attrs: list[list[str]] = []
    fused_scores: list[list[float]] = []
    for attrs, scores in zip(item_canonical, item_canonical_scores, strict=False):
        pairs = [(canon, score) for canon, score in zip(attrs, scores, strict=False) if canon in keep]
        pairs.sort(key=lambda kv: kv[1], reverse=True)
        pairs = pairs[:max_attrs]
        fused_attrs.append([canonical_labels[canon] for canon, _ in pairs])
        fused_scores.append([float(score) for _, score in pairs])

    keep_df = {canonical_labels[canon]: canonical_counts[canon] for canon in keep}
    return fused_attrs, fused_scores, canonical_labels, keep_df


def build_attr_graph(
    item_attrs: list[list[str]],
    item_scores: list[list[float]],
    topk: int,
) -> tuple[sparse.csr_matrix, dict[str, int], np.ndarray]:
    attr_to_id: dict[str, int] = {}
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    df_counter: Counter[str] = Counter()

    for attrs in item_attrs:
        for attr in set(attrs):
            df_counter[attr] += 1

    n_items = len(item_attrs)
    for item_id, (attrs, scores) in enumerate(zip(item_attrs, item_scores, strict=False)):
        for attr, score in zip(attrs, scores, strict=False):
            if attr not in attr_to_id:
                attr_to_id[attr] = len(attr_to_id)
            attr_id = attr_to_id[attr]
            df = df_counter[attr]
            idf = math.log((1.0 + n_items) / (1.0 + df)) + 1.0
            rows.append(item_id)
            cols.append(attr_id)
            data.append(float(score) * float(idf))

    incidence = sparse.coo_matrix((data, (rows, cols)), shape=(n_items, len(attr_to_id)), dtype=np.float32).tocsr()
    graph = (incidence @ incidence.T).tocsr().astype(np.float32)
    graph.setdiag(0.0)
    graph.eliminate_zeros()
    graph = keep_topk_per_row(graph, topk=topk)
    graph = row_normalize_csr(graph)
    attr_df = np.zeros(len(attr_to_id), dtype=np.int32)
    for attr, idx in attr_to_id.items():
        attr_df[idx] = int(df_counter[attr])
    return graph, attr_to_id, attr_df


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


def row_normalize_csr(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    matrix = matrix.tocsr(copy=True).astype(np.float32)
    row_sums = np.asarray(matrix.sum(axis=1)).reshape(-1)
    inv = np.zeros_like(row_sums, dtype=np.float32)
    mask = row_sums > 0
    inv[mask] = 1.0 / row_sums[mask]
    return sparse.diags(inv).dot(matrix).tocsr()


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


def graph_health_metrics(
    graph: sparse.csr_matrix,
    item_attrs: list[list[str]],
    train_df: pd.DataFrame,
    baseline_graph: sparse.csr_matrix,
    topk: int,
) -> dict[str, float]:
    n_items = graph.shape[0]
    covered_items = sum(1 for attrs in item_attrs if attrs)
    attr_counts = [len(attrs) for attrs in item_attrs]
    degree = np.diff(graph.indptr)
    undirected = ((graph + graph.T) > 0).astype(np.int32)
    n_components, labels = csgraph.connected_components(undirected, directed=False, return_labels=True)
    largest_ratio = 0.0
    if labels.size > 0:
        largest_ratio = float(np.max(np.bincount(labels)) / len(labels))

    popularity = np.zeros(n_items, dtype=np.int32)
    for item_id in train_df["item_id"].astype(int).tolist():
        if 0 <= item_id < n_items:
            popularity[item_id] += 1
    cold_mask = popularity <= 1
    cold_connected = float(np.mean(degree[cold_mask] > 0)) if np.any(cold_mask) else 0.0

    metrics = {
        "n_items": int(n_items),
        "covered_items": int(covered_items),
        "coverage_rate": float(covered_items / max(n_items, 1)),
        "avg_attrs_per_item": float(np.mean(attr_counts)) if attr_counts else 0.0,
        "avg_attrs_per_covered_item": float(np.mean([c for c in attr_counts if c > 0])) if covered_items else 0.0,
        "median_attrs_per_item": float(np.median(attr_counts)) if attr_counts else 0.0,
        "graph_nnz": int(graph.nnz),
        "graph_density": float(graph.nnz / max(n_items * n_items, 1)),
        "avg_out_degree": float(np.mean(degree)) if degree.size else 0.0,
        "connected_item_rate": float(np.mean(degree > 0)) if degree.size else 0.0,
        "largest_component_ratio": largest_ratio,
        "cold_item_count": int(np.sum(cold_mask)),
        "cold_item_connected_rate": cold_connected,
        "mean_neighbor_overlap_with_fagsp_mid_base": mean_neighbor_overlap(graph, baseline_graph, topk=topk),
    }
    return metrics


def write_item_attribute_jsonl(
    path: Path,
    items: list[dict[str, object]],
    attrs: list[list[str]],
    scores: list[list[float]],
) -> None:
    with path.open("w") as f:
        for item, item_attrs, item_scores in zip(items, attrs, scores, strict=False):
            payload = {
                "item_id": int(item["item_id"]),
                "title": str(item["title"]),
                "attributes": item_attrs,
                "scores": [round(float(score), 6) for score in item_scores],
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_attr_vocab(path: Path, attr_to_id: dict[str, int], attr_df: np.ndarray) -> None:
    rows = [
        {"attribute_id": int(idx), "attribute": attr, "df": int(attr_df[idx])}
        for attr, idx in sorted(attr_to_id.items(), key=lambda kv: kv[1])
    ]
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2))


def write_preview_csv(
    path: Path,
    items: list[dict[str, object]],
    attrs: list[list[str]],
    scores: list[list[float]],
    sample_size: int,
    seed: int,
) -> None:
    rng = random.Random(seed)
    indices = list(range(len(items)))
    rng.shuffle(indices)
    chosen = indices[: min(sample_size, len(indices))]
    rows = []
    for idx in chosen:
        item = items[idx]
        rows.append(
            {
                "item_id": int(item["item_id"]),
                "title": str(item["title"]),
                "attributes": " | ".join(attrs[idx]),
                "scores": " | ".join(f"{score:.4f}" for score in scores[idx]),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def save_variant_outputs(
    variant_dir: Path,
    items: list[dict[str, object]],
    attrs: list[list[str]],
    scores: list[list[float]],
    graph: sparse.csr_matrix,
    attr_to_id: dict[str, int],
    attr_df: np.ndarray,
    metrics: dict[str, float],
    sample_preview_size: int,
    seed: int,
) -> None:
    variant_dir.mkdir(parents=True, exist_ok=True)
    write_item_attribute_jsonl(variant_dir / "item_attributes.jsonl", items, attrs, scores)
    write_attr_vocab(variant_dir / "attribute_vocab.json", attr_to_id, attr_df)
    sparse.save_npz(variant_dir / "item_attribute_graph.npz", graph)
    write_preview_csv(variant_dir / "attribute_preview.csv", items, attrs, scores, sample_preview_size, seed)
    (variant_dir / "summary.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))


def run_variant(
    name: str,
    items: list[dict[str, object]],
    attrs: list[list[str]],
    scores: list[list[float]],
    output_root: Path,
    train_df: pd.DataFrame,
    baseline_graph: sparse.csr_matrix,
    graph_topk: int,
    neighbor_topk: int,
    sample_preview_size: int,
    seed: int,
) -> dict[str, float]:
    graph, attr_to_id, attr_df = build_attr_graph(attrs, scores, topk=graph_topk)
    metrics = graph_health_metrics(
        graph=graph,
        item_attrs=attrs,
        train_df=train_df,
        baseline_graph=baseline_graph,
        topk=neighbor_topk,
    )
    metrics["unique_attributes"] = int(len(attr_to_id))
    metrics["variant"] = name
    save_variant_outputs(
        variant_dir=output_root / name,
        items=items,
        attrs=attrs,
        scores=scores,
        graph=graph,
        attr_to_id=attr_to_id,
        attr_df=attr_df,
        metrics=metrics,
        sample_preview_size=sample_preview_size,
        seed=seed,
    )
    return metrics


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    items = load_item_meta(Path(args.item_json))
    raw_docs, title_docs = build_text_docs(items)
    raw_vectorizer, raw_matrix, title_vectorizer, title_matrix = build_vectorizer(raw_docs, title_docs)

    raw_attrs, raw_scores = select_doc_phrases(
        matrix=raw_matrix,
        feature_names=np.asarray(raw_vectorizer.get_feature_names_out()),
        max_attrs=args.raw_max_attrs,
    )
    heuristic_attrs, heuristic_scores = select_doc_phrases(
        matrix=title_matrix,
        feature_names=np.asarray(title_vectorizer.get_feature_names_out()),
        max_attrs=args.heuristic_max_attrs,
    )
    fused_attrs, fused_scores, _, _ = fuse_attributes(
        raw_attrs=raw_attrs,
        raw_scores=raw_scores,
        max_attrs=args.fused_max_attrs,
        min_df=args.min_fused_df,
    )

    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)
    graph_bank = build_transplanted_graph_bank(
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
    )
    baseline_graph = graph_bank["fagsp_mid_base"].matrix  # type: ignore[attr-defined]

    summaries: list[dict[str, float]] = []
    summaries.append(
        run_variant(
            name="R500_attr_raw_textphrase",
            items=items,
            attrs=raw_attrs,
            scores=raw_scores,
            output_root=output_root,
            train_df=train_df,
            baseline_graph=baseline_graph,
            graph_topk=args.graph_topk,
            neighbor_topk=args.neighbor_topk,
            sample_preview_size=args.sample_preview_size,
            seed=args.seed,
        )
    )
    summaries.append(
        run_variant(
            name="R501_attr_fused_textphrase",
            items=items,
            attrs=fused_attrs,
            scores=fused_scores,
            output_root=output_root,
            train_df=train_df,
            baseline_graph=baseline_graph,
            graph_topk=args.graph_topk,
            neighbor_topk=args.neighbor_topk,
            sample_preview_size=args.sample_preview_size,
            seed=args.seed,
        )
    )
    summaries.append(
        run_variant(
            name="R502_attr_heuristic_title",
            items=items,
            attrs=heuristic_attrs,
            scores=heuristic_scores,
            output_root=output_root,
            train_df=train_df,
            baseline_graph=baseline_graph,
            graph_topk=args.graph_topk,
            neighbor_topk=args.neighbor_topk,
            sample_preview_size=args.sample_preview_size,
            seed=args.seed,
        )
    )

    pd.DataFrame(summaries).to_csv(output_root / "summary_table.csv", index=False)


if __name__ == "__main__":
    main()
