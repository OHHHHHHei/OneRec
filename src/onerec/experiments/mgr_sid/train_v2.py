from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import sparse
from tqdm import tqdm

from onerec.sid.models.rqvae import RQVAE
from onerec.sid.utils import ensure_dir, get_local_time
from onerec.utils.io import read_yaml

from .graph_bank import row_normalize
from .paper_transplants import build_semantic_knn_graph, keep_topk_per_row, load_semantic_embeddings
from .train_v1 import (
    IndexedEmbDataset,
    _build_optimizer,
    _build_scheduler,
    _collision_rate,
    _forward_hierarchy,
    _select_subgraph,
    _to_torch_dense,
    set_seed,
)
from .transplanted_graph_bank import build_transplanted_graph_bank


@dataclass
class MgrSidV2TrainConfig:
    mode: str
    data_path: str
    train_csv: str
    semantic_embedding_path: str | None
    ambiguity_csv: str
    ambiguity_column: str
    ckpt_dir: str
    device: str = "cuda:0"
    seed: int = 2024
    epochs: int = 10000
    batch_size: int = 20480
    num_workers: int = 4
    lr: float = 1e-3
    weight_decay: float = 0.0
    eval_step: int = 50
    learner: str = "AdamW"
    lr_scheduler_type: str = "constant"
    warmup_epochs: int = 50
    save_limit: int = 5
    num_emb_list: list[int] | None = None
    e_dim: int = 32
    layers: list[int] | None = None
    dropout_prob: float = 0.0
    bn: bool = False
    loss_type: str = "mse"
    quant_loss_weight: float = 1.0
    beta: float = 0.25
    kmeans_init: bool = True
    kmeans_iters: int = 100
    sk_epsilons: list[float] | None = None
    sk_iters: int = 50
    history_k: int = 10
    coarse_min_weight: float = 2.0
    local_min_weight: float = 1.0
    community_clusters: int = 64
    anchor_topk: int = 32
    semantic_mix: float = 0.35
    spectral_rank: int = 48
    band_low: float = 0.25
    band_high: float = 0.65
    temporal_mix: float = 0.35
    local_multihop_alpha: float = 0.35
    local_multihop_max_hop: int = 2
    fagsp_cascade_high_rank: int = 16
    fagsp_cascade_low_rank: int = 32
    fagsp_cascade_support_quantile: float = 0.8
    fagsp_cascade_boost_alpha: float = 0.5
    mgdcf_keep_ratio: float = 0.1
    mgdcf_binarize_edges: bool = True
    seq2g_mix_alpha: float = 0.35
    seq2g_context_topk: int = 32
    seq2g_candidate_topm: int = 32
    seq2g_direct_tau: float = 0.5
    seq2g_use_reliability: bool = True
    seq2g_use_direct_weak_mask: bool = True
    graph_topk: int = 32
    semantic_graph_topk: int = 32
    semantic_external_graph_path: str | None = None
    coarse_weight: float = 0.05
    mid_weight: float = 0.15
    local_weight: float = 0.05
    l2_contrastive_pull_weight: float = 0.0
    semantic_coarse_weight: float = 0.05
    semantic_mid_weight: float = 0.025
    graph_scale_min: float = 0.5
    graph_scale_max: float = 1.5
    semantic_scale_min: float = 0.5
    semantic_scale_max: float = 1.5
    coarse_view_name: str = "coarse_purified"
    mid_view_name: str = "fagsp_mid_base"
    local_view_name: str = "local_purified"
    mid_external_graph_path: str | None = None
    mid_external_graph_mix_base_weight: float = 1.0
    hierarchy_stopgrad_previous_levels: bool = False
    semantic_retention_mode: str = "smoothness"
    semantic_retention_temperature: float = 0.1
    warm_start_ckpt_path: str | None = None
    teacher_ckpt_path: str | None = None
    prefix_retention_l1_weight: float = 0.0
    prefix_retention_l2_weight: float = 0.0
    prefix_retention_scale_min: float = 1.0
    prefix_retention_scale_max: float = 1.0
    prefix_retention_use_inverse_ambiguity: bool = False
    prefix_retention_teacher_use_sk: bool = False
    codebook_anchor_l1_weight: float = 0.0
    codebook_anchor_l2_weight: float = 0.0
    selective_separation_weight: float = 0.0
    selective_separation_margin: float = 0.2
    selective_separation_pair_csv: str | None = None
    selective_separation_pair_rule: str | None = None
    selective_separation_levels: list[int] | None = None
    selective_separation_use_pair_reliability: bool = True
    selective_separation_use_ambiguity_scaling: bool = True
    selective_separation_scale_min: float = 0.5
    selective_separation_scale_max: float = 1.5


def load_train_config(config_path: str, overrides: dict[str, Any] | None = None) -> MgrSidV2TrainConfig:
    payload = read_yaml(config_path)
    payload = dict(payload)
    for key, value in (overrides or {}).items():
        if value is not None:
            payload[key] = value
    if payload.get("num_emb_list") is None:
        payload["num_emb_list"] = [256, 256, 256]
    if payload.get("layers") is None:
        payload["layers"] = [2048, 1024, 512, 256, 128, 64]
    if payload.get("sk_epsilons") is None:
        payload["sk_epsilons"] = [0.0, 0.0, 0.0]
    return MgrSidV2TrainConfig(**payload)


def _load_ambiguity_prior(path: str, column: str, n_items: int, device: torch.device) -> torch.Tensor:
    df = pd.read_csv(path)
    if "item_id" not in df.columns:
        raise ValueError(f"Ambiguity CSV missing item_id column: {path}")
    if column not in df.columns:
        raise ValueError(f"Ambiguity CSV missing target column `{column}`: {path}")
    values = np.zeros(n_items, dtype=np.float32)
    for _, row in df.iterrows():
        item_id = int(row["item_id"])
        if 0 <= item_id < n_items:
            value = float(row[column])
            if not np.isfinite(value):
                value = 0.0
            values[item_id] = np.clip(value, 0.0, 1.0)
    return torch.tensor(values, dtype=torch.float32, device=device)

# 图平滑损失，输入是表示矩阵、图邻接矩阵和每个item的权重，输出是一个标量损失值
def _weighted_graph_smoothness_loss(
    representations: torch.Tensor,
    graph: torch.Tensor,
    item_weights: torch.Tensor,
) -> torch.Tensor:
    propagated = graph @ representations
    per_item = torch.mean((representations - propagated) ** 2, dim=1)
    item_weights = item_weights.float()
    denom = item_weights.sum().clamp(min=1e-6)
    return torch.sum(per_item * item_weights) / denom


def _weighted_batch_local_neighbor_kl_loss(
    teacher_repr: torch.Tensor,
    student_repr: torch.Tensor,
    item_weights: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if teacher_repr.size(0) <= 1:
        return teacher_repr.new_tensor(0.0)

    if temperature <= 0:
        raise ValueError(f"semantic_retention_temperature must be positive, got {temperature}")

    teacher_repr = F.normalize(teacher_repr.float(), p=2, dim=1)
    student_repr = F.normalize(student_repr.float(), p=2, dim=1)

    teacher_logits = teacher_repr @ teacher_repr.T
    student_logits = student_repr @ student_repr.T
    teacher_logits = teacher_logits / temperature
    student_logits = student_logits / temperature

    mask = torch.eye(teacher_logits.size(0), dtype=torch.bool, device=teacher_logits.device)
    teacher_logits = teacher_logits.masked_fill(mask, -1e9)
    student_logits = student_logits.masked_fill(mask, -1e9)

    teacher_probs = F.softmax(teacher_logits, dim=1)
    student_log_probs = F.log_softmax(student_logits, dim=1)
    per_item = torch.sum(teacher_probs * (torch.log(teacher_probs.clamp_min(1e-12)) - student_log_probs), dim=1)

    item_weights = item_weights.float()
    denom = item_weights.sum().clamp(min=1e-6)
    return torch.sum(per_item * item_weights) / denom


def _load_selective_separation_pair_matrix(
    path: str,
    n_items: int,
    device: torch.device,
    rule: str | None = None,
) -> torch.Tensor:
    df = pd.read_csv(path)
    required = {"item_a", "item_b"}
    if not required.issubset(df.columns):
        missing = sorted(required - set(df.columns))
        raise ValueError(f"Selective-separation CSV missing required columns {missing}: {path}")
    if rule and "rule" in df.columns:
        df = df[df["rule"] == rule].copy()

    values = torch.zeros((n_items, n_items), dtype=torch.float32, device=device)
    has_reliability = "reliability" in df.columns
    for row in df.itertuples(index=False):
        item_a = int(row.item_a)
        item_b = int(row.item_b)
        if not (0 <= item_a < n_items and 0 <= item_b < n_items) or item_a == item_b:
            continue
        value = 1.0
        if has_reliability:
            value = float(getattr(row, "reliability"))
            if not np.isfinite(value) or value <= 0.0:
                continue
        if value > float(values[item_a, item_b].item()):
            values[item_a, item_b] = value
            values[item_b, item_a] = value
    return values


def _weighted_selective_separation_loss(
    representations: torch.Tensor,
    pair_weights: torch.Tensor,
    item_scales: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    if representations.size(0) <= 1:
        return representations.new_tensor(0.0)
    if not (-1.0 < margin < 1.0):
        raise ValueError(f"selective_separation_margin must be in (-1, 1), got {margin}")

    representations = F.normalize(representations.float(), p=2, dim=1)
    similarity = representations @ representations.T

    pair_weights = pair_weights.float()
    if item_scales.numel() > 0:
        pair_scale = torch.sqrt(torch.outer(item_scales.float(), item_scales.float()).clamp_min(0.0))
        pair_weights = pair_weights * pair_scale

    pair_weights = torch.triu(pair_weights, diagonal=1)
    active_mask = pair_weights > 0.0
    if not torch.any(active_mask):
        return representations.new_tensor(0.0)

    penalties = F.relu(similarity - margin).pow(2)
    denom = pair_weights[active_mask].sum().clamp(min=1e-6)
    return torch.sum(penalties * pair_weights) / denom


def _weighted_pairwise_pull_loss(
    representations: torch.Tensor,
    pair_weights: torch.Tensor,
    item_scales: torch.Tensor,
) -> torch.Tensor:
    if representations.size(0) <= 1:
        return representations.new_tensor(0.0)

    representations = F.normalize(representations.float(), p=2, dim=1)
    similarity = representations @ representations.T

    pair_weights = torch.maximum(pair_weights.float(), pair_weights.T.float())
    if item_scales.numel() > 0:
        pair_scale = torch.sqrt(torch.outer(item_scales.float(), item_scales.float()).clamp_min(0.0))
        pair_weights = pair_weights * pair_scale

    pair_weights = torch.triu(pair_weights, diagonal=1)
    active_mask = pair_weights > 0.0
    if not torch.any(active_mask):
        return representations.new_tensor(0.0)

    penalties = (1.0 - similarity).clamp_min(0.0)
    denom = pair_weights[active_mask].sum().clamp(min=1e-6)
    return torch.sum(penalties * pair_weights) / denom


def _scale_from_prior(prior: torch.Tensor, low: float, high: float) -> torch.Tensor:
    return low + (high - low) * prior


def _build_model_from_cfg(cfg: MgrSidV2TrainConfig, in_dim: int) -> RQVAE:
    return RQVAE(
        in_dim=in_dim,
        num_emb_list=cfg.num_emb_list,
        e_dim=cfg.e_dim,
        layers=cfg.layers,
        dropout_prob=cfg.dropout_prob,
        bn=cfg.bn,
        loss_type=cfg.loss_type,
        quant_loss_weight=cfg.quant_loss_weight,
        beta=cfg.beta,
        kmeans_init=cfg.kmeans_init,
        kmeans_iters=cfg.kmeans_iters,
        sk_epsilons=cfg.sk_epsilons,
        sk_iters=cfg.sk_iters,
    )


def _load_checkpoint_state(model: RQVAE, ckpt_path: str, device: torch.device) -> None:
    ckpt = torch.load(ckpt_path, map_location=torch.device("cpu"), weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)


def _resolve_retention_item_scale(cfg: MgrSidV2TrainConfig, prior_batch: torch.Tensor) -> torch.Tensor:
    if not cfg.prefix_retention_use_inverse_ambiguity:
        return torch.ones_like(prior_batch, dtype=torch.float32)
    return _scale_from_prior(
        1.0 - prior_batch,
        cfg.prefix_retention_scale_min,
        cfg.prefix_retention_scale_max,
    )


def _resolve_selective_separation_levels(cfg: MgrSidV2TrainConfig) -> list[int]:
    if cfg.selective_separation_weight <= 0:
        return []
    levels = cfg.selective_separation_levels or [3]
    normalized = sorted({int(level) for level in levels if int(level) in (1, 2, 3)})
    if not normalized:
        raise ValueError("selective_separation_levels must contain at least one of [1, 2, 3]")
    return normalized


def _build_level_representations(
    cumulative_outputs: list[torch.Tensor],
    level_outputs: list[torch.Tensor],
    stopgrad_previous_levels: bool,
) -> list[torch.Tensor]:
    if not stopgrad_previous_levels:
        return cumulative_outputs

    level_reps: list[torch.Tensor] = []
    detached_prefix = torch.zeros_like(level_outputs[0])
    for level_q in level_outputs:
        level_reps.append(detached_prefix + level_q)
        detached_prefix = detached_prefix + level_q.detach()
    return level_reps


def _build_graph_tensors(cfg: MgrSidV2TrainConfig, device: torch.device, n_items: int) -> dict[str, torch.Tensor]:
    train_df = pd.read_csv(cfg.train_csv)
    views = build_transplanted_graph_bank(
        train_df=train_df,
        test_df=train_df,
        history_k=cfg.history_k,
        coarse_min_weight=cfg.coarse_min_weight,
        local_min_weight=cfg.local_min_weight,
        n_clusters=cfg.community_clusters,
        seed=cfg.seed,
        semantic_embedding_path=cfg.semantic_embedding_path,
        anchor_topk=cfg.anchor_topk,
        semantic_mix=cfg.semantic_mix,
        spectral_rank=cfg.spectral_rank,
        band_low=cfg.band_low,
        band_high=cfg.band_high,
        temporal_mix=cfg.temporal_mix,
        local_multihop_alpha=cfg.local_multihop_alpha,
        local_multihop_max_hop=cfg.local_multihop_max_hop,
        fagsp_cascade_high_rank=cfg.fagsp_cascade_high_rank,
        fagsp_cascade_low_rank=cfg.fagsp_cascade_low_rank,
        fagsp_cascade_support_quantile=cfg.fagsp_cascade_support_quantile,
        fagsp_cascade_boost_alpha=cfg.fagsp_cascade_boost_alpha,
        mgdcf_keep_ratio=cfg.mgdcf_keep_ratio,
        mgdcf_binarize_edges=cfg.mgdcf_binarize_edges,
        seq2g_mix_alpha=cfg.seq2g_mix_alpha,
        seq2g_context_topk=cfg.seq2g_context_topk,
        seq2g_candidate_topm=cfg.seq2g_candidate_topm,
        seq2g_direct_tau=cfg.seq2g_direct_tau,
        seq2g_use_reliability=cfg.seq2g_use_reliability,
        seq2g_use_direct_weak_mask=cfg.seq2g_use_direct_weak_mask,
    )

    def _view_matrix(view_name: str):
        if view_name not in views:
            raise KeyError(f"Unknown graph view: {view_name}. Available views: {sorted(views.keys())}")
        view = views[view_name]
        matrix = getattr(view, "matrix", None)
        if matrix is None:
            raise TypeError(f"Graph view {view_name} does not expose a sparse matrix")
        return matrix

    selected = {
        "coarse": _view_matrix(cfg.coarse_view_name),
        "mid": _view_matrix(cfg.mid_view_name),
        "local": _view_matrix(cfg.local_view_name),
    }

    if cfg.mid_external_graph_path:
        external_mid = sparse.load_npz(cfg.mid_external_graph_path).tocsr().astype(np.float32)
        if external_mid.shape[0] != n_items:
            external_mid = external_mid[:n_items, :n_items]
        external_mid = row_normalize(external_mid)
        base_weight = float(np.clip(cfg.mid_external_graph_mix_base_weight, 0.0, 1.0))
        if base_weight <= 0.0:
            selected["mid"] = external_mid
        elif base_weight < 1.0:
            selected["mid"] = row_normalize(
                (base_weight * row_normalize(selected["mid"]) + (1.0 - base_weight) * external_mid).tocsr()
            )

    graph_tensors: dict[str, torch.Tensor] = {}
    for name, matrix in selected.items():
        matrix = keep_topk_per_row(matrix, topk=cfg.graph_topk)
        graph_tensors[name] = _to_torch_dense(matrix, device=device)

    if cfg.semantic_external_graph_path:
        semantic_graph = sparse.load_npz(cfg.semantic_external_graph_path).tocsr().astype(np.float32)
        if semantic_graph.shape[0] != n_items:
            semantic_graph = semantic_graph[:n_items, :n_items]
        semantic_graph = row_normalize(semantic_graph)
    else:
        semantic_embeddings = load_semantic_embeddings(cfg.semantic_embedding_path)
        if semantic_embeddings is None:
            raise ValueError("semantic_embedding_path is required for tokenizer v2")
        if semantic_embeddings.shape[0] != n_items:
            semantic_embeddings = semantic_embeddings[:n_items]
        semantic_graph = build_semantic_knn_graph(semantic_embeddings, topk=cfg.semantic_graph_topk)
    semantic_graph = keep_topk_per_row(semantic_graph, topk=cfg.semantic_graph_topk)
    graph_tensors["semantic"] = _to_torch_dense(semantic_graph, device=device)

    for name, tensor in list(graph_tensors.items()):
        if tensor.shape[0] != n_items:
            graph_tensors[name] = tensor[:n_items, :n_items]
    return graph_tensors


def run_training(cfg: MgrSidV2TrainConfig) -> dict[str, Any]:
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    ensure_dir(cfg.ckpt_dir)
    run_dir = Path(cfg.ckpt_dir) / get_local_time()
    ensure_dir(str(run_dir))

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("mgr_sid_v2")
    logger.info("Starting MGR-SID tokenizer v2 training")

    dataset = IndexedEmbDataset(cfg.data_path)
    train_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=True,
        pin_memory=True,
    )
    eval_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=False,
        pin_memory=True,
    )

    graph_tensors = _build_graph_tensors(cfg, device=device, n_items=len(dataset))
    ambiguity_prior = _load_ambiguity_prior(
        path=cfg.ambiguity_csv,
        column=cfg.ambiguity_column,
        n_items=len(dataset),
        device=device,
    )
    selective_separation_levels = _resolve_selective_separation_levels(cfg)
    selective_separation_pair_matrix = None
    if cfg.selective_separation_weight > 0:
        if not cfg.selective_separation_pair_csv:
            raise ValueError("selective_separation_pair_csv is required when selective_separation_weight > 0")
        selective_separation_pair_matrix = _load_selective_separation_pair_matrix(
            path=cfg.selective_separation_pair_csv,
            n_items=len(dataset),
            device=device,
            rule=cfg.selective_separation_pair_rule,
        )
        if not cfg.selective_separation_use_pair_reliability:
            selective_separation_pair_matrix = (selective_separation_pair_matrix > 0).float()

    model = _build_model_from_cfg(cfg, in_dim=dataset.dim).to(device)
    if cfg.warm_start_ckpt_path:
        _load_checkpoint_state(model, cfg.warm_start_ckpt_path, device=device)
        logger.info("Warm-started student model from %s", cfg.warm_start_ckpt_path)

    teacher_model: RQVAE | None = None
    if cfg.prefix_retention_l1_weight > 0 or cfg.prefix_retention_l2_weight > 0:
        teacher_ckpt_path = cfg.teacher_ckpt_path or cfg.warm_start_ckpt_path
        if not teacher_ckpt_path:
            raise ValueError(
                "prefix retention requires teacher_ckpt_path or warm_start_ckpt_path"
            )
        teacher_model = _build_model_from_cfg(cfg, in_dim=dataset.dim).to(device)
        _load_checkpoint_state(teacher_model, teacher_ckpt_path, device=device)
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False
        logger.info("Loaded frozen teacher model from %s", teacher_ckpt_path)

    optimizer = _build_optimizer(cfg, model)
    scheduler = _build_scheduler(cfg, optimizer, steps_per_epoch=len(train_loader))
    codebook_init: dict[int, torch.Tensor] = {
        level: model.rq.vq_layers[level].embedding.weight.detach().clone()
        for level in range(3)
    }

    best = {
        "loss": float("inf"),
        "collision_rate": float("inf"),
        "epoch": -1,
        "best_loss_ckpt": "",
        "best_collision_ckpt": "",
    }
    history: list[dict[str, Any]] = []
    effective_eval_step = min(cfg.eval_step, cfg.epochs)

    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0.0
        total_recon_loss = 0.0
        total_rq_loss = 0.0
        total_coarse_graph_loss = 0.0
        total_mid_graph_loss = 0.0
        total_local_graph_loss = 0.0
        total_l2_contrastive_pull_loss = 0.0
        total_semantic_coarse_loss = 0.0
        total_semantic_mid_loss = 0.0
        total_prefix_retain_l1_loss = 0.0
        total_prefix_retain_l2_loss = 0.0
        total_codebook_anchor_l1_loss = 0.0
        total_codebook_anchor_l2_loss = 0.0
        total_selective_sep_l1_loss = 0.0
        total_selective_sep_l2_loss = 0.0
        total_selective_sep_l3_loss = 0.0

        iter_data = tqdm(
            train_loader,
            total=len(train_loader),
            ncols=100,
            desc=f"MGR-V2-Train {epoch}",
        )
        for batch_indices, batch_embeddings in iter_data:
            batch_indices = batch_indices.to(device, non_blocking=True)
            batch_embeddings = batch_embeddings.to(device, non_blocking=True)

            optimizer.zero_grad()
            pack = _forward_hierarchy(model, batch_embeddings, use_sk=True)
            loss_total, loss_recon = model.compute_loss(pack["recon"], pack["rq_loss"], xs=batch_embeddings)
            level_representations = _build_level_representations(
                cumulative_outputs=pack["cumulative_outputs"],
                level_outputs=pack["level_outputs"],
                stopgrad_previous_levels=cfg.hierarchy_stopgrad_previous_levels,
            )

            prior_batch = ambiguity_prior.index_select(0, batch_indices)
            graph_item_scale = _scale_from_prior(prior_batch, cfg.graph_scale_min, cfg.graph_scale_max)
            semantic_item_scale = _scale_from_prior(
                1.0 - prior_batch,
                cfg.semantic_scale_min,
                cfg.semantic_scale_max,
            )
            retention_item_scale = _resolve_retention_item_scale(cfg, prior_batch)
            if cfg.selective_separation_use_ambiguity_scaling:
                selective_item_scale = _scale_from_prior(
                    prior_batch,
                    cfg.selective_separation_scale_min,
                    cfg.selective_separation_scale_max,
                )
            else:
                selective_item_scale = torch.ones_like(prior_batch, dtype=torch.float32)

            coarse_subgraph = _select_subgraph(graph_tensors["coarse"], batch_indices)
            mid_subgraph = _select_subgraph(graph_tensors["mid"], batch_indices)
            local_subgraph = _select_subgraph(graph_tensors["local"], batch_indices)
            semantic_subgraph = _select_subgraph(graph_tensors["semantic"], batch_indices)

            graph_losses = {
                "coarse": _weighted_graph_smoothness_loss(level_representations[0], coarse_subgraph, graph_item_scale),
                "mid": _weighted_graph_smoothness_loss(level_representations[1], mid_subgraph, graph_item_scale),
                "local": _weighted_graph_smoothness_loss(level_representations[2], local_subgraph, graph_item_scale),
            }
            l2_contrastive_pull = batch_embeddings.new_tensor(0.0)
            if cfg.l2_contrastive_pull_weight > 0:
                l2_contrastive_pull = _weighted_pairwise_pull_loss(
                    level_representations[1],
                    mid_subgraph,
                    graph_item_scale,
                )
            if cfg.semantic_retention_mode == "smoothness":
                graph_losses["semantic_coarse"] = _weighted_graph_smoothness_loss(
                    level_representations[0], semantic_subgraph, semantic_item_scale
                )
                graph_losses["semantic_mid"] = _weighted_graph_smoothness_loss(
                    level_representations[1], semantic_subgraph, semantic_item_scale
                )
            elif cfg.semantic_retention_mode == "batch_local_kl":
                graph_losses["semantic_coarse"] = _weighted_batch_local_neighbor_kl_loss(
                    teacher_repr=batch_embeddings.detach(),
                    student_repr=level_representations[0],
                    item_weights=semantic_item_scale,
                    temperature=cfg.semantic_retention_temperature,
                )
                graph_losses["semantic_mid"] = _weighted_batch_local_neighbor_kl_loss(
                    teacher_repr=batch_embeddings.detach(),
                    student_repr=level_representations[1],
                    item_weights=semantic_item_scale,
                    temperature=cfg.semantic_retention_temperature,
                )
            else:
                raise ValueError(
                    f"Unsupported semantic_retention_mode: {cfg.semantic_retention_mode}"
                )

            prefix_retain_l1 = batch_embeddings.new_tensor(0.0)
            prefix_retain_l2 = batch_embeddings.new_tensor(0.0)
            codebook_anchor_l1 = batch_embeddings.new_tensor(0.0)
            codebook_anchor_l2 = batch_embeddings.new_tensor(0.0)
            if teacher_model is not None:
                with torch.no_grad():
                    teacher_pack = _forward_hierarchy(
                        teacher_model,
                        batch_embeddings,
                        use_sk=cfg.prefix_retention_teacher_use_sk,
                    )
                if cfg.prefix_retention_l1_weight > 0:
                    per_item_l1 = torch.mean(
                        (level_representations[0] - teacher_pack["cumulative_outputs"][0]) ** 2,
                        dim=1,
                    )
                    denom_l1 = retention_item_scale.sum().clamp(min=1e-6)
                    prefix_retain_l1 = torch.sum(per_item_l1 * retention_item_scale) / denom_l1
                if cfg.prefix_retention_l2_weight > 0:
                    per_item_l2 = torch.mean(
                        (level_representations[1] - teacher_pack["cumulative_outputs"][1]) ** 2,
                        dim=1,
                    )
                    denom_l2 = retention_item_scale.sum().clamp(min=1e-6)
                    prefix_retain_l2 = torch.sum(per_item_l2 * retention_item_scale) / denom_l2

            if cfg.codebook_anchor_l1_weight > 0:
                current = model.rq.vq_layers[0].embedding.weight
                codebook_anchor_l1 = torch.mean((current - codebook_init[0]) ** 2)
            if cfg.codebook_anchor_l2_weight > 0:
                current = model.rq.vq_layers[1].embedding.weight
                codebook_anchor_l2 = torch.mean((current - codebook_init[1]) ** 2)

            selective_sep_l1 = batch_embeddings.new_tensor(0.0)
            selective_sep_l2 = batch_embeddings.new_tensor(0.0)
            selective_sep_l3 = batch_embeddings.new_tensor(0.0)
            selective_sep_mean = batch_embeddings.new_tensor(0.0)
            if selective_separation_pair_matrix is not None:
                pair_subgraph = _select_subgraph(selective_separation_pair_matrix, batch_indices)
                selective_losses: list[torch.Tensor] = []
                if 1 in selective_separation_levels:
                    selective_sep_l1 = _weighted_selective_separation_loss(
                        level_representations[0],
                        pair_subgraph,
                        selective_item_scale,
                        cfg.selective_separation_margin,
                    )
                    selective_losses.append(selective_sep_l1)
                if 2 in selective_separation_levels:
                    selective_sep_l2 = _weighted_selective_separation_loss(
                        level_representations[1],
                        pair_subgraph,
                        selective_item_scale,
                        cfg.selective_separation_margin,
                    )
                    selective_losses.append(selective_sep_l2)
                if 3 in selective_separation_levels:
                    selective_sep_l3 = _weighted_selective_separation_loss(
                        level_representations[2],
                        pair_subgraph,
                        selective_item_scale,
                        cfg.selective_separation_margin,
                    )
                    selective_losses.append(selective_sep_l3)
                if selective_losses:
                    selective_sep_mean = torch.stack(selective_losses).mean()

            loss_total = (
                loss_total
                + cfg.coarse_weight * graph_losses["coarse"]
                + cfg.mid_weight * graph_losses["mid"]
                + cfg.local_weight * graph_losses["local"]
                + cfg.l2_contrastive_pull_weight * l2_contrastive_pull
                + cfg.semantic_coarse_weight * graph_losses["semantic_coarse"]
                + cfg.semantic_mid_weight * graph_losses["semantic_mid"]
                + cfg.prefix_retention_l1_weight * prefix_retain_l1
                + cfg.prefix_retention_l2_weight * prefix_retain_l2
                + cfg.codebook_anchor_l1_weight * codebook_anchor_l1
                + cfg.codebook_anchor_l2_weight * codebook_anchor_l2
                + cfg.selective_separation_weight * selective_sep_mean
            )

            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += float(loss_total.item())
            total_recon_loss += float(loss_recon.item())
            total_rq_loss += float(pack["rq_loss"].item())
            total_coarse_graph_loss += float(graph_losses["coarse"].item())
            total_mid_graph_loss += float(graph_losses["mid"].item())
            total_local_graph_loss += float(graph_losses["local"].item())
            total_l2_contrastive_pull_loss += float(l2_contrastive_pull.item())
            total_semantic_coarse_loss += float(graph_losses["semantic_coarse"].item())
            total_semantic_mid_loss += float(graph_losses["semantic_mid"].item())
            total_prefix_retain_l1_loss += float(prefix_retain_l1.item())
            total_prefix_retain_l2_loss += float(prefix_retain_l2.item())
            total_codebook_anchor_l1_loss += float(codebook_anchor_l1.item())
            total_codebook_anchor_l2_loss += float(codebook_anchor_l2.item())
            total_selective_sep_l1_loss += float(selective_sep_l1.item())
            total_selective_sep_l2_loss += float(selective_sep_l2.item())
            total_selective_sep_l3_loss += float(selective_sep_l3.item())

        record = {
            "epoch": epoch,
            "total_loss": total_loss,
            "recon_loss": total_recon_loss,
            "rq_loss": total_rq_loss,
            "coarse_graph_loss": total_coarse_graph_loss,
            "mid_graph_loss": total_mid_graph_loss,
            "local_graph_loss": total_local_graph_loss,
            "l2_contrastive_pull_loss": total_l2_contrastive_pull_loss,
            "semantic_coarse_loss": total_semantic_coarse_loss,
            "semantic_mid_loss": total_semantic_mid_loss,
            "prefix_retain_l1_loss": total_prefix_retain_l1_loss,
            "prefix_retain_l2_loss": total_prefix_retain_l2_loss,
            "codebook_anchor_l1_loss": total_codebook_anchor_l1_loss,
            "codebook_anchor_l2_loss": total_codebook_anchor_l2_loss,
            "selective_sep_l1_loss": total_selective_sep_l1_loss,
            "selective_sep_l2_loss": total_selective_sep_l2_loss,
            "selective_sep_l3_loss": total_selective_sep_l3_loss,
        }

        if (epoch + 1) % effective_eval_step == 0:
            model.eval()
            indices_set: list[torch.Tensor] = []
            with torch.no_grad():
                for _, batch_embeddings in tqdm(
                    eval_loader,
                    total=len(eval_loader),
                    ncols=100,
                    desc="MGR-V2-Eval",
                ):
                    batch_embeddings = batch_embeddings.to(device, non_blocking=True)
                    indices_set.append(model.get_indices(batch_embeddings, use_sk=False))
            indices = torch.cat(indices_set, dim=0)
            collision_rate = _collision_rate(indices)
            record["collision_rate"] = collision_rate

            if record["total_loss"] < best["loss"]:
                best["loss"] = record["total_loss"]
                best["epoch"] = epoch
                ckpt = run_dir / "best_loss_model.pth"
                torch.save({"config": cfg.__dict__, "state_dict": model.state_dict(), "epoch": epoch}, ckpt)
                best["best_loss_ckpt"] = str(ckpt)

            if collision_rate < best["collision_rate"]:
                best["collision_rate"] = collision_rate
                ckpt = run_dir / "best_collision_model.pth"
                torch.save({"config": cfg.__dict__, "state_dict": model.state_dict(), "epoch": epoch}, ckpt)
                best["best_collision_ckpt"] = str(ckpt)

        logger.info(
            "epoch=%d total=%.6f recon=%.6f rq=%.6f coarse=%.6f mid=%.6f local=%.6f l2_pull=%.6f sem_coarse=%.6f sem_mid=%.6f retain_l1=%.6f retain_l2=%.6f anchor_l1=%.6f anchor_l2=%.6f sep_l1=%.6f sep_l2=%.6f sep_l3=%.6f collision=%s",
            epoch,
            record["total_loss"],
            record["recon_loss"],
            record["rq_loss"],
            record["coarse_graph_loss"],
            record["mid_graph_loss"],
            record["local_graph_loss"],
            record["l2_contrastive_pull_loss"],
            record["semantic_coarse_loss"],
            record["semantic_mid_loss"],
            record["prefix_retain_l1_loss"],
            record["prefix_retain_l2_loss"],
            record["codebook_anchor_l1_loss"],
            record["codebook_anchor_l2_loss"],
            record["selective_sep_l1_loss"],
            record["selective_sep_l2_loss"],
            record["selective_sep_l3_loss"],
            f"{record.get('collision_rate', float('nan')):.6f}" if "collision_rate" in record else "na",
        )
        history.append(record)

    summary = {
        "config": cfg.__dict__,
        "run_dir": str(run_dir),
        "best": best,
        "history": history,
    }
    summary_path = run_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    logger.info("Summary written to %s", summary_path)
    return summary
