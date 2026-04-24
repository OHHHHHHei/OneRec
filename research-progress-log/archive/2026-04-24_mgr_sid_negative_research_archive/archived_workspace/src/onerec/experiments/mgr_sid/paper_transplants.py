from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from sklearn.neighbors import NearestNeighbors

from .graph_bank import (
    SparseGraphView,
    positive_part,
    row_normalize,
    sparse_density,
    support_prune,
)


def load_semantic_embeddings(path: str | None) -> np.ndarray | None:
    if not path:
        return None
    array = np.load(path).astype(np.float32)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    return array / norms


def build_semantic_knn_graph(
    semantic_embeddings: np.ndarray,
    topk: int,
) -> sparse.csr_matrix:
    n_items = semantic_embeddings.shape[0]
    topk = max(2, min(topk, n_items))
    nn = NearestNeighbors(n_neighbors=topk, metric="cosine")
    nn.fit(semantic_embeddings)
    distances, indices = nn.kneighbors(semantic_embeddings)

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for src in range(n_items):
        for dst, dist in zip(indices[src], distances[src], strict=False):
            if src == int(dst):
                continue
            similarity = max(0.0, 1.0 - float(dist))
            if similarity <= 0.0:
                continue
            rows.append(src)
            cols.append(int(dst))
            data.append(similarity)
    graph = sparse.coo_matrix((data, (rows, cols)), shape=(n_items, n_items), dtype=np.float32).tocsr()
    graph.sum_duplicates()
    return row_normalize(graph)


def symmetrically_normalize(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    matrix = matrix.tocsr(copy=True).astype(np.float32)
    degrees = np.asarray(matrix.sum(axis=1)).reshape(-1)
    inv_sqrt = np.zeros_like(degrees, dtype=np.float32)
    mask = degrees > 0
    inv_sqrt[mask] = 1.0 / np.sqrt(degrees[mask])
    d = sparse.diags(inv_sqrt)
    return (d @ matrix @ d).tocsr()


def symmetrize_matrix(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    matrix = matrix.tocsr(copy=True).astype(np.float32)
    matrix = (matrix + matrix.T).multiply(0.5).tocsr()
    matrix.eliminate_zeros()
    return matrix


def keep_topk_per_row(matrix: sparse.csr_matrix, topk: int) -> sparse.csr_matrix:
    matrix = matrix.tocsr(copy=True).astype(np.float32)
    topk = max(1, int(topk))
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


def semantic_anchor_purify_graph(
    matrix: sparse.csr_matrix,
    semantic_embeddings: np.ndarray | None,
    min_weight: float,
    anchor_topk: int,
    semantic_mix: float,
) -> sparse.csr_matrix:
    purified = support_prune(matrix, min_weight=min_weight)
    purified = row_normalize(purified)
    if semantic_embeddings is None:
        return purified

    semantic_graph = build_semantic_knn_graph(semantic_embeddings, topk=anchor_topk)
    overlap = purified.multiply(semantic_graph)
    mixed = ((1.0 - semantic_mix) * purified + semantic_mix * overlap).tocsr()
    mixed = keep_topk_per_row(mixed, topk=anchor_topk)
    return row_normalize(mixed)


def _spectral_reconstruct(
    matrix: sparse.csr_matrix,
    eigen_ratio_low: float,
    eigen_ratio_high: float,
    rank: int,
) -> sparse.csr_matrix:
    normalized = symmetrically_normalize(matrix)
    n_items = normalized.shape[0]
    if n_items <= 2:
        return row_normalize(normalized)
    max_rank = max(2, min(rank, n_items - 1))
    try:
        values, vectors = eigsh(normalized, k=max_rank, which="LA")
    except Exception:
        dense = normalized.toarray()
        values, vectors = np.linalg.eigh(dense)
        order = np.argsort(values)[::-1][:max_rank]
        values = values[order]
        vectors = vectors[:, order]

    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order]
    lo = int(np.floor(max_rank * eigen_ratio_low))
    hi = int(np.ceil(max_rank * eigen_ratio_high))
    hi = max(lo + 1, min(hi, max_rank))
    selected_values = values[lo:hi]
    selected_vectors = vectors[:, lo:hi]
    reconstructed = selected_vectors @ np.diag(np.maximum(selected_values, 0.0)) @ selected_vectors.T
    reconstructed = np.maximum(reconstructed, 0.0).astype(np.float32)
    np.fill_diagonal(reconstructed, 0.0)
    sparse_view = sparse.csr_matrix(reconstructed)
    sparse_view.eliminate_zeros()
    return row_normalize(sparse_view)


def _eigendecompose_symmetric(
    matrix: sparse.csr_matrix,
    rank: int,
    which: str,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = matrix.tocsr().astype(np.float32)
    n_items = matrix.shape[0]
    if n_items <= 2:
        dense = matrix.toarray()
        values, vectors = np.linalg.eigh(dense)
        if which == "LA":
            order = np.argsort(values)[::-1]
        else:
            order = np.argsort(values)
        return values[order], vectors[:, order]

    k = max(1, min(rank, n_items - 1))
    try:
        values, vectors = eigsh(matrix, k=k, which=which)
    except Exception:
        dense = matrix.toarray()
        values, vectors = np.linalg.eigh(dense)
        if which == "LA":
            order = np.argsort(values)[::-1][:k]
        else:
            order = np.argsort(values)[:k]
        values = values[order]
        vectors = vectors[:, order]
        return values, vectors

    if which == "LA":
        order = np.argsort(values)[::-1]
    else:
        order = np.argsort(values)
    return values[order], vectors[:, order]


def _support_quantile_mask(
    support_scores: sparse.csr_matrix,
    base_graph: sparse.csr_matrix,
    quantile: float,
) -> sparse.csr_matrix:
    support_scores = support_scores.tocsc()
    base_graph = base_graph.tocsc()
    n_items = support_scores.shape[1]
    quantile = float(np.clip(quantile, 0.0, 1.0))

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for col in range(n_items):
        base_start, base_end = base_graph.indptr[col], base_graph.indptr[col + 1]
        base_rows = base_graph.indices[base_start:base_end]
        if base_rows.size == 0:
            continue

        score_start, score_end = support_scores.indptr[col], support_scores.indptr[col + 1]
        score_rows = support_scores.indices[score_start:score_end]
        score_vals = support_scores.data[score_start:score_end]
        score_map = {int(r): float(v) for r, v in zip(score_rows, score_vals, strict=False)}
        filtered_scores = np.asarray(
            [max(0.0, score_map.get(int(r), 0.0)) for r in base_rows],
            dtype=np.float32,
        )
        positive = filtered_scores[filtered_scores > 0.0]
        if positive.size == 0:
            continue
        threshold = float(np.quantile(positive, quantile))
        keep_mask = filtered_scores >= threshold
        kept_rows = base_rows[keep_mask]
        kept_vals = filtered_scores[keep_mask]
        rows.extend(int(r) for r in kept_rows.tolist())
        cols.extend([col] * len(kept_rows))
        data.extend(float(v) for v in kept_vals.tolist())

    mask = sparse.coo_matrix(
        (np.asarray(data, dtype=np.float32), (np.asarray(rows), np.asarray(cols))),
        shape=base_graph.shape,
        dtype=np.float32,
    ).tocsr()
    mask.sum_duplicates()
    return mask


def build_fagsp_mid_view(
    base_graph: sparse.csr_matrix,
    name: str,
    rank: int,
    eigen_ratio_low: float,
    eigen_ratio_high: float,
) -> SparseGraphView:
    view = _spectral_reconstruct(
        matrix=base_graph,
        eigen_ratio_low=eigen_ratio_low,
        eigen_ratio_high=eigen_ratio_high,
        rank=rank,
    )
    return SparseGraphView(
        name=name,
        matrix=view,
        metadata={
            "kind": "fagsp_band",
            "rank": int(rank),
            "eigen_ratio_low": float(eigen_ratio_low),
            "eigen_ratio_high": float(eigen_ratio_high),
            "nnz": int(view.nnz),
            "density": sparse_density(view),
        },
    )


def build_fagsp_cascade_mid_view(
    base_graph: sparse.csr_matrix,
    name: str,
    high_rank: int,
    low_rank: int,
    support_quantile: float,
    boost_alpha: float,
) -> SparseGraphView:
    base_graph = row_normalize(base_graph)
    spectral_base = symmetrically_normalize(symmetrize_matrix(base_graph))

    high_values, high_vectors = _eigendecompose_symmetric(
        spectral_base,
        rank=high_rank,
        which="SA",
    )
    high_projection = high_vectors @ high_vectors.T
    high_projection = np.maximum(high_projection, 0.0).astype(np.float32)
    np.fill_diagonal(high_projection, 0.0)
    high_support = sparse.csr_matrix(high_projection)
    high_support = _support_quantile_mask(
        support_scores=high_support,
        base_graph=symmetrize_matrix(base_graph),
        quantile=support_quantile,
    )

    boosted_edges = symmetrize_matrix(base_graph).multiply(high_support.sign())
    enhanced_graph = symmetrize_matrix(base_graph) + float(boost_alpha) * boosted_edges
    enhanced_graph = row_normalize(enhanced_graph)
    spectral_enhanced = symmetrically_normalize(symmetrize_matrix(enhanced_graph))

    low_values, low_vectors = _eigendecompose_symmetric(
        spectral_enhanced,
        rank=low_rank,
        which="LA",
    )
    low_values = np.maximum(low_values, 0.0)
    low_reconstruct = low_vectors @ np.diag(low_values) @ low_vectors.T
    low_reconstruct = np.maximum(low_reconstruct, 0.0).astype(np.float32)
    np.fill_diagonal(low_reconstruct, 0.0)
    view = row_normalize(sparse.csr_matrix(low_reconstruct))
    view.eliminate_zeros()

    return SparseGraphView(
        name=name,
        matrix=view,
        metadata={
            "kind": "fagsp_cascade",
            "high_rank": int(high_rank),
            "low_rank": int(low_rank),
            "support_quantile": float(support_quantile),
            "boost_alpha": float(boost_alpha),
            "high_value_min": float(np.min(high_values)) if high_values.size else 0.0,
            "high_value_max": float(np.max(high_values)) if high_values.size else 0.0,
            "low_value_min": float(np.min(low_values)) if low_values.size else 0.0,
            "low_value_max": float(np.max(low_values)) if low_values.size else 0.0,
            "support_nnz": int(high_support.nnz),
            "nnz": int(view.nnz),
            "density": sparse_density(view),
        },
    )


def build_gsprec_temporal_mid_view(
    coarse_graph: sparse.csr_matrix,
    local_graph: sparse.csr_matrix,
    name: str,
    temporal_mix: float,
    rank: int,
    eigen_ratio_low: float,
    eigen_ratio_high: float,
) -> SparseGraphView:
    temporal_mix = float(np.clip(temporal_mix, 0.0, 1.0))
    mixed = ((1.0 - temporal_mix) * row_normalize(coarse_graph) + temporal_mix * row_normalize(local_graph)).tocsr()
    mixed = positive_part(mixed)
    view = _spectral_reconstruct(
        matrix=mixed,
        eigen_ratio_low=eigen_ratio_low,
        eigen_ratio_high=eigen_ratio_high,
        rank=rank,
    )
    return SparseGraphView(
        name=name,
        matrix=view,
        metadata={
            "kind": "gsprec_temporal_band",
            "temporal_mix": temporal_mix,
            "rank": int(rank),
            "eigen_ratio_low": float(eigen_ratio_low),
            "eigen_ratio_high": float(eigen_ratio_high),
            "nnz": int(view.nnz),
            "density": sparse_density(view),
        },
    )
