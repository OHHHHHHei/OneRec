from __future__ import annotations

import ast
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD


def parse_id_list(value: Any) -> list[int]:
    if isinstance(value, list):
        return [int(v) for v in value]
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
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


def row_normalize(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    matrix = matrix.tocsr(copy=True).astype(np.float32)
    row_sums = np.asarray(matrix.sum(axis=1)).reshape(-1)
    inv = np.zeros_like(row_sums, dtype=np.float32)
    mask = row_sums > 0
    inv[mask] = 1.0 / row_sums[mask]
    return sparse.diags(inv).dot(matrix).tocsr()


def positive_part(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    matrix = matrix.tocsr(copy=True).astype(np.float32)
    matrix.data = np.maximum(matrix.data, 0.0)
    matrix.eliminate_zeros()
    return matrix


def support_prune(matrix: sparse.csr_matrix, min_weight: float) -> sparse.csr_matrix:
    matrix = matrix.tocsr(copy=True).astype(np.float32)
    keep = matrix.data >= float(min_weight)
    matrix.data = matrix.data[keep]
    matrix.indices = matrix.indices[keep]
    indptr = [0]
    cursor = 0
    for row in range(matrix.shape[0]):
        start = matrix.indptr[row]
        end = matrix.indptr[row + 1]
        row_keep = keep[start:end]
        cursor += int(row_keep.sum())
        indptr.append(cursor)
    matrix.indptr = np.asarray(indptr, dtype=np.int32)
    matrix.eliminate_zeros()
    return matrix


def debias_by_popularity(
    matrix: sparse.csr_matrix,
    popularity: np.ndarray,
    alpha: float = 0.5,
) -> sparse.csr_matrix:
    coo = matrix.tocoo(copy=True).astype(np.float32)
    pop = popularity.astype(np.float32)
    row_scale = np.power(np.maximum(pop[coo.row], 1.0), alpha)
    col_scale = np.power(np.maximum(pop[coo.col], 1.0), alpha)
    denom = row_scale * col_scale
    coo.data = coo.data / np.maximum(denom, 1e-12)
    return coo.tocsr()


def sparse_density(matrix: sparse.csr_matrix) -> float:
    total = matrix.shape[0] * matrix.shape[1]
    if total == 0:
        return 0.0
    return float(matrix.nnz / total)


@dataclass
class SparseGraphView:
    name: str
    matrix: sparse.csr_matrix
    metadata: dict[str, Any]

    def score(self, history: list[int], candidate: int, history_k: int) -> float:
        if candidate < 0 or candidate >= self.matrix.shape[1]:
            return 0.0
        if not history:
            return 0.0
        score = 0.0
        recent = list(reversed(history[-history_k:]))
        for rank, hist_item in enumerate(recent, start=1):
            if hist_item < 0 or hist_item >= self.matrix.shape[0]:
                continue
            score += float(self.matrix[hist_item, candidate]) / rank
        return score


@dataclass
class CommunityGraphView:
    name: str
    cluster_ids: np.ndarray
    cluster_affinity: np.ndarray
    metadata: dict[str, Any]

    def score(self, history: list[int], candidate: int, history_k: int) -> float:
        if candidate < 0 or candidate >= len(self.cluster_ids):
            return 0.0
        if not history:
            return 0.0
        cand_cluster = int(self.cluster_ids[candidate])
        score = 0.0
        recent = list(reversed(history[-history_k:]))
        for rank, hist_item in enumerate(recent, start=1):
            if hist_item < 0 or hist_item >= len(self.cluster_ids):
                continue
            hist_cluster = int(self.cluster_ids[hist_item])
            score += float(self.cluster_affinity[hist_cluster, cand_cluster]) / rank
        return score


def infer_num_items(train_df: pd.DataFrame, test_df: pd.DataFrame) -> int:
    max_item = -1
    for df in (train_df, test_df):
        max_item = max(max_item, int(df["item_id"].max()))
        for value in df["history_item_id"]:
            history = parse_id_list(value)
            if history:
                max_item = max(max_item, max(history))
    return max_item + 1


def build_popularity(train_df: pd.DataFrame) -> np.ndarray:
    n_items = infer_num_items(train_df, train_df)
    pop = np.zeros(n_items, dtype=np.float32)
    for _, row in train_df.iterrows():
        target = int(row["item_id"])
        pop[target] += 1.0
        for hist_item in parse_id_list(row["history_item_id"]):
            if 0 <= hist_item < n_items:
                pop[hist_item] += 1.0
    return pop


def build_coarse_graph(train_df: pd.DataFrame, n_items: int, history_k: int) -> sparse.csr_matrix:
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for _, row in train_df.iterrows():
        history = parse_id_list(row["history_item_id"])
        target = int(row["item_id"])
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
                weight = 1.0 / (j - i)
                rows.extend((src, dst))
                cols.extend((dst, src))
                data.extend((weight, weight))
    matrix = sparse.coo_matrix((data, (rows, cols)), shape=(n_items, n_items), dtype=np.float32).tocsr()
    matrix.sum_duplicates()
    return matrix


def build_local_graph(train_df: pd.DataFrame, n_items: int, history_k: int) -> sparse.csr_matrix:
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for _, row in train_df.iterrows():
        history = parse_id_list(row["history_item_id"])
        if not history:
            continue
        target = int(row["item_id"])
        for rank, hist_item in enumerate(reversed(history[-history_k:]), start=1):
            if hist_item < 0 or hist_item >= n_items or target < 0 or target >= n_items:
                continue
            rows.append(hist_item)
            cols.append(target)
            data.append(1.0 / rank)
    matrix = sparse.coo_matrix((data, (rows, cols)), shape=(n_items, n_items), dtype=np.float32).tocsr()
    matrix.sum_duplicates()
    return matrix


def build_user_item_binary_matrix(train_df: pd.DataFrame, n_items: int) -> sparse.csr_matrix:
    user_to_idx: dict[str, int] = {}
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for row in train_df.itertuples(index=False):
        user_id = str(row.user_id)
        user_idx = user_to_idx.setdefault(user_id, len(user_to_idx))
        items: set[int] = set()
        target = int(row.item_id)
        if 0 <= target < n_items:
            items.add(target)
        for hist_item in parse_id_list(row.history_item_id):
            if 0 <= hist_item < n_items:
                items.add(hist_item)
        for item_id in items:
            rows.append(user_idx)
            cols.append(item_id)
            data.append(1.0)
    matrix = sparse.coo_matrix(
        (
            np.asarray(data, dtype=np.float32),
            (np.asarray(rows, dtype=np.int32), np.asarray(cols, dtype=np.int32)),
        ),
        shape=(len(user_to_idx), n_items),
        dtype=np.float32,
    ).tocsr()
    matrix.sum_duplicates()
    if matrix.nnz:
        matrix.data[:] = 1.0
    return matrix


def _symmetric_global_top_ratio_prune(
    matrix: sparse.csr_matrix,
    keep_ratio: float,
    binarize: bool,
) -> sparse.csr_matrix:
    matrix = matrix.tocsr(copy=True).astype(np.float32)
    matrix.setdiag(0.0)
    matrix.eliminate_zeros()
    if matrix.nnz == 0:
        return matrix

    upper = sparse.triu(matrix, k=1).tocoo()
    if upper.nnz == 0:
        return sparse.csr_matrix(matrix.shape, dtype=np.float32)

    keep_ratio = float(np.clip(keep_ratio, 0.0, 1.0))
    if keep_ratio <= 0.0:
        return sparse.csr_matrix(matrix.shape, dtype=np.float32)

    keep_edges = max(1, int(np.ceil(upper.nnz * keep_ratio)))
    if keep_edges >= upper.nnz:
        selected_rows = upper.row
        selected_cols = upper.col
        selected_data = upper.data
    else:
        selected_idx = np.argpartition(upper.data, -keep_edges)[-keep_edges:]
        selected_rows = upper.row[selected_idx]
        selected_cols = upper.col[selected_idx]
        selected_data = upper.data[selected_idx]

    if binarize:
        selected_data = np.ones_like(selected_data, dtype=np.float32)

    rows = np.concatenate([selected_rows, selected_cols]).astype(np.int32, copy=False)
    cols = np.concatenate([selected_cols, selected_rows]).astype(np.int32, copy=False)
    data = np.concatenate([selected_data, selected_data]).astype(np.float32, copy=False)
    pruned = sparse.coo_matrix((data, (rows, cols)), shape=matrix.shape, dtype=np.float32).tocsr()
    pruned.sum_duplicates()
    pruned.setdiag(0.0)
    pruned.eliminate_zeros()
    return pruned


def build_mgdcf_item_graph(
    train_df: pd.DataFrame,
    n_items: int,
    keep_ratio: float,
    binarize_edges: bool,
) -> sparse.csr_matrix:
    user_item = build_user_item_binary_matrix(train_df, n_items=n_items)
    user_to_item = row_normalize(user_item)
    item_to_user = row_normalize(user_item.T.tocsr())
    item_to_item = (item_to_user @ user_to_item).tocsr().astype(np.float32)
    item_to_item.setdiag(0.0)
    item_to_item.eliminate_zeros()

    affinity = item_to_item.multiply(item_to_item.T).tocsr().astype(np.float32)
    if affinity.nnz:
        affinity.data = np.sqrt(np.maximum(affinity.data, 0.0))
    affinity.setdiag(0.0)
    affinity.eliminate_zeros()

    sparsified = _symmetric_global_top_ratio_prune(
        affinity,
        keep_ratio=keep_ratio,
        binarize=binarize_edges,
    )
    return row_normalize(sparsified)


def build_multi_hop_transition_view(
    base_graph: sparse.csr_matrix,
    name: str,
    alpha: float,
    max_hop: int,
) -> SparseGraphView:
    normalized = row_normalize(base_graph)
    accum = normalized.copy().astype(np.float32)
    power = normalized.copy().astype(np.float32)
    for hop in range(2, max_hop + 1):
        power = (power @ normalized).tocsr().astype(np.float32)
        accum = (accum + (alpha ** (hop - 1)) * power).tocsr()
    accum = accum.tocsr(copy=True)
    accum.setdiag(0.0)
    accum.eliminate_zeros()
    view = row_normalize(accum)
    return SparseGraphView(
        name=name,
        matrix=view,
        metadata={
            "kind": "multi_hop_transition",
            "alpha": float(alpha),
            "max_hop": int(max_hop),
            "nnz": int(view.nnz),
            "density": sparse_density(view),
        },
    )


def purify_coarse_graph(
    matrix: sparse.csr_matrix,
    popularity: np.ndarray,
    min_weight: float,
) -> sparse.csr_matrix:
    pruned = support_prune(matrix, min_weight=min_weight)
    debiased = debias_by_popularity(pruned, popularity, alpha=0.5)
    return row_normalize(debiased)


def purify_local_graph(
    matrix: sparse.csr_matrix,
    popularity: np.ndarray,
    min_weight: float,
) -> sparse.csr_matrix:
    pruned = support_prune(matrix, min_weight=min_weight)
    target_pop = np.maximum(popularity, 1.0).astype(np.float32)
    coo = pruned.tocoo(copy=True)
    coo.data = coo.data / np.sqrt(target_pop[coo.col])
    return row_normalize(coo.tocsr())


def build_diffusion_residual_view(
    base_graph: sparse.csr_matrix,
    name: str,
) -> SparseGraphView:
    a1 = row_normalize(base_graph)
    a2 = row_normalize((a1 @ a1).tocsr())
    residual = positive_part((a2 - a1).tocsr())
    view = row_normalize(residual)
    return SparseGraphView(
        name=name,
        matrix=view,
        metadata={"kind": "diffusion_residual", "nnz": int(view.nnz), "density": sparse_density(view)},
    )


def build_band_pass_view(
    base_graph: sparse.csr_matrix,
    name: str,
) -> SparseGraphView:
    a1 = row_normalize(base_graph)
    a2 = row_normalize((a1 @ a1).tocsr())
    a3 = row_normalize((a2 @ a1).tocsr())
    band = positive_part((2.0 * a2 - a1 - a3).tocsr())
    view = row_normalize(band)
    return SparseGraphView(
        name=name,
        matrix=view,
        metadata={"kind": "band_pass_proxy", "nnz": int(view.nnz), "density": sparse_density(view)},
    )


def build_community_view(
    base_graph: sparse.csr_matrix,
    n_clusters: int,
    seed: int,
    name: str,
) -> CommunityGraphView:
    normalized = row_normalize(base_graph)
    n_items = normalized.shape[0]
    n_components = max(2, min(32, n_items - 1))
    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    embed = l2_normalize_rows(svd.fit_transform(normalized))
    n_clusters = max(2, min(n_clusters, n_items))
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    cluster_ids = kmeans.fit_predict(embed)

    cluster_affinity = np.zeros((n_clusters, n_clusters), dtype=np.float32)
    coo = normalized.tocoo()
    for row, col, value in zip(coo.row, coo.col, coo.data, strict=False):
        cluster_affinity[cluster_ids[row], cluster_ids[col]] += float(value)
    row_sums = cluster_affinity.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    cluster_affinity = cluster_affinity / row_sums

    counts = Counter(int(cid) for cid in cluster_ids)
    metadata = {
        "kind": "community_aware",
        "n_clusters": int(n_clusters),
        "largest_cluster": int(max(counts.values()) if counts else 0),
        "active_cluster_pairs": int(np.count_nonzero(cluster_affinity)),
    }
    return CommunityGraphView(
        name=name,
        cluster_ids=cluster_ids.astype(np.int32),
        cluster_affinity=cluster_affinity,
        metadata=metadata,
    )


def build_graph_bank(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    history_k: int,
    coarse_min_weight: float,
    local_min_weight: float,
    n_clusters: int,
    seed: int,
    mgdcf_keep_ratio: float = 0.1,
    mgdcf_binarize_edges: bool = True,
) -> dict[str, SparseGraphView | CommunityGraphView]:
    n_items = infer_num_items(train_df, test_df)
    popularity = build_popularity(train_df)
    coarse_raw = build_coarse_graph(train_df, n_items=n_items, history_k=history_k)
    local_raw = build_local_graph(train_df, n_items=n_items, history_k=history_k)
    coarse_purified = purify_coarse_graph(coarse_raw, popularity=popularity, min_weight=coarse_min_weight)
    local_purified = purify_local_graph(local_raw, popularity=popularity, min_weight=local_min_weight)
    coarse_mgdcf = build_mgdcf_item_graph(
        train_df,
        n_items=n_items,
        keep_ratio=mgdcf_keep_ratio,
        binarize_edges=mgdcf_binarize_edges,
    )

    views: dict[str, SparseGraphView | CommunityGraphView] = {
        "coarse_raw": SparseGraphView(
            name="coarse_raw",
            matrix=row_normalize(coarse_raw),
            metadata={"kind": "coarse_raw", "nnz": int(coarse_raw.nnz), "density": sparse_density(coarse_raw)},
        ),
        "coarse_purified": SparseGraphView(
            name="coarse_purified",
            matrix=coarse_purified,
            metadata={"kind": "coarse_purified", "nnz": int(coarse_purified.nnz), "density": sparse_density(coarse_purified)},
        ),
        "coarse_mgdcf": SparseGraphView(
            name="coarse_mgdcf",
            matrix=coarse_mgdcf,
            metadata={
                "kind": "coarse_mgdcf",
                "nnz": int(coarse_mgdcf.nnz),
                "density": sparse_density(coarse_mgdcf),
                "mgdcf_keep_ratio": float(mgdcf_keep_ratio),
                "mgdcf_binarize_edges": bool(mgdcf_binarize_edges),
            },
        ),
        "local_raw": SparseGraphView(
            name="local_raw",
            matrix=row_normalize(local_raw),
            metadata={"kind": "local_raw", "nnz": int(local_raw.nnz), "density": sparse_density(local_raw)},
        ),
        "local_purified": SparseGraphView(
            name="local_purified",
            matrix=local_purified,
            metadata={"kind": "local_purified", "nnz": int(local_purified.nnz), "density": sparse_density(local_purified)},
        ),
        "mid_diffusion_raw": build_diffusion_residual_view(row_normalize(coarse_raw), "mid_diffusion_raw"),
        "mid_diffusion_purified": build_diffusion_residual_view(coarse_purified, "mid_diffusion_purified"),
        "mid_band_pass_raw": build_band_pass_view(row_normalize(coarse_raw), "mid_band_pass_raw"),
        "mid_band_pass_purified": build_band_pass_view(coarse_purified, "mid_band_pass_purified"),
        "mid_community_raw": build_community_view(row_normalize(coarse_raw), n_clusters=n_clusters, seed=seed, name="mid_community_raw"),
        "mid_community_purified": build_community_view(coarse_purified, n_clusters=n_clusters, seed=seed, name="mid_community_purified"),
    }
    return views
