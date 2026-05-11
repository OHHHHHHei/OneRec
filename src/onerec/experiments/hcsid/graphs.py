from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

from .config import HcsidTrainConfig
from .data import parse_id_list


@dataclass
class HcsidGraphTensors:
    l1_semantic: torch.Tensor
    l2_local_multihop: torch.Tensor
    l3_local: torch.Tensor


def row_normalize(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    matrix = matrix.tocsr(copy=True).astype(np.float32)
    row_sums = np.asarray(matrix.sum(axis=1)).reshape(-1)
    inv = np.zeros_like(row_sums, dtype=np.float32)
    mask = row_sums > 0
    inv[mask] = 1.0 / row_sums[mask]
    return sparse.diags(inv).dot(matrix).tocsr()


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
        (
            np.asarray(data, dtype=np.float32),
            np.asarray(indices, dtype=np.int32),
            np.asarray(indptr, dtype=np.int32),
        ),
        shape=matrix.shape,
    )
    pruned.eliminate_zeros()
    return pruned


def infer_num_items(train_df: pd.DataFrame) -> int:
    max_item = int(train_df["item_id"].max()) if len(train_df) else -1
    for value in train_df["history_item_id"]:
        history = parse_id_list(value)
        if history:
            max_item = max(max_item, max(history))
    return max_item + 1


def build_popularity(train_df: pd.DataFrame, n_items: int) -> np.ndarray:
    popularity = np.zeros(n_items, dtype=np.float32)
    for row in train_df.itertuples(index=False):
        target = int(row.item_id)
        if 0 <= target < n_items:
            popularity[target] += 1.0
        for hist_item in parse_id_list(row.history_item_id):
            if 0 <= hist_item < n_items:
                popularity[hist_item] += 1.0
    return popularity


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
        cursor += int(keep[start:end].sum())
        indptr.append(cursor)
    matrix.indptr = np.asarray(indptr, dtype=np.int32)
    matrix.eliminate_zeros()
    return matrix


def build_local_graph(train_df: pd.DataFrame, n_items: int, history_k: int) -> sparse.csr_matrix:
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for row in train_df.itertuples(index=False):
        history = parse_id_list(row.history_item_id)
        if not history:
            continue
        target = int(row.item_id)
        if target < 0 or target >= n_items:
            continue
        for rank, hist_item in enumerate(reversed(history[-history_k:]), start=1):
            if 0 <= hist_item < n_items:
                rows.append(hist_item)
                cols.append(target)
                data.append(1.0 / rank)
    matrix = sparse.coo_matrix((data, (rows, cols)), shape=(n_items, n_items), dtype=np.float32).tocsr()
    matrix.sum_duplicates()
    return matrix


def purify_local_graph(matrix: sparse.csr_matrix, popularity: np.ndarray, min_weight: float) -> sparse.csr_matrix:
    pruned = support_prune(matrix, min_weight=min_weight)
    target_pop = np.maximum(popularity, 1.0).astype(np.float32)
    coo = pruned.tocoo(copy=True)
    coo.data = coo.data / np.sqrt(target_pop[coo.col])
    return row_normalize(coo.tocsr())


def build_local_multihop_graph(
    base_graph: sparse.csr_matrix,
    alpha: float,
    max_hop: int,
    base_weight: float,
) -> sparse.csr_matrix:
    normalized = row_normalize(base_graph)
    accum = (float(base_weight) * normalized).tocsr().astype(np.float32)
    power = normalized.copy().astype(np.float32)
    for hop in range(2, int(max_hop) + 1):
        power = (power @ normalized).tocsr().astype(np.float32)
        accum = (accum + (float(alpha) ** (hop - 1)) * power).tocsr()
    accum = accum.tocsr(copy=True)
    accum.setdiag(0.0)
    accum.eliminate_zeros()
    return row_normalize(accum)


def load_semantic_embeddings(path: str | None) -> np.ndarray:
    if not path:
        raise ValueError("semantic_embedding_path is required for HCSID")
    array = np.load(path).astype(np.float32)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    return array / norms


def build_semantic_knn_graph(semantic_embeddings: np.ndarray, topk: int) -> sparse.csr_matrix:
    n_items = semantic_embeddings.shape[0]
    topk = max(2, min(int(topk), n_items))
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
            if similarity > 0.0:
                rows.append(src)
                cols.append(int(dst))
                data.append(similarity)
    graph = sparse.coo_matrix((data, (rows, cols)), shape=(n_items, n_items), dtype=np.float32).tocsr()
    graph.sum_duplicates()
    return row_normalize(graph)


def to_torch_dense(matrix: sparse.csr_matrix, device: torch.device) -> torch.Tensor:
    dense = matrix.astype(np.float32).toarray()
    return torch.tensor(dense, dtype=torch.float32, device=device)


def build_hcsid_graphs(cfg: HcsidTrainConfig, device: torch.device, n_items: int) -> HcsidGraphTensors:
    train_df = pd.read_csv(cfg.train_csv)
    graph_n_items = infer_num_items(train_df)
    if graph_n_items < n_items:
        graph_n_items = n_items

    popularity = build_popularity(train_df, graph_n_items)
    local_raw = build_local_graph(train_df, n_items=graph_n_items, history_k=cfg.history_k)
    local_purified = purify_local_graph(local_raw, popularity=popularity, min_weight=cfg.local_min_weight)
    local_multihop = build_local_multihop_graph(
        local_purified,
        alpha=cfg.local_multihop_alpha,
        max_hop=cfg.local_multihop_max_hop,
        base_weight=cfg.local_multihop_base_weight,
    )

    semantic_embeddings = load_semantic_embeddings(cfg.semantic_embedding_path)
    semantic_embeddings = semantic_embeddings[:graph_n_items]
    semantic_graph = build_semantic_knn_graph(semantic_embeddings, topk=cfg.semantic_graph_topk)

    l1_semantic = keep_topk_per_row(semantic_graph, topk=cfg.semantic_graph_topk)
    l2_local_multihop = keep_topk_per_row(local_multihop, topk=cfg.graph_topk)
    l3_local = keep_topk_per_row(local_purified, topk=cfg.graph_topk)

    return HcsidGraphTensors(
        l1_semantic=to_torch_dense(l1_semantic[:n_items, :n_items], device=device),
        l2_local_multihop=to_torch_dense(l2_local_multihop[:n_items, :n_items], device=device),
        l3_local=to_torch_dense(l3_local[:n_items, :n_items], device=device),
    )


def select_subgraph(graph: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    subgraph = graph.index_select(0, indices)
    return subgraph.index_select(1, indices)
