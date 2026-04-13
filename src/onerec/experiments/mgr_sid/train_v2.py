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
from tqdm import tqdm

from onerec.sid.models.rqvae import RQVAE
from onerec.sid.utils import ensure_dir, get_local_time
from onerec.utils.io import read_yaml

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
    graph_topk: int = 32
    semantic_graph_topk: int = 32
    coarse_weight: float = 0.05
    mid_weight: float = 0.15
    local_weight: float = 0.05
    semantic_coarse_weight: float = 0.05
    semantic_mid_weight: float = 0.025
    graph_scale_min: float = 0.5
    graph_scale_max: float = 1.5
    semantic_scale_min: float = 0.5
    semantic_scale_max: float = 1.5
    hierarchy_stopgrad_previous_levels: bool = False
    semantic_retention_mode: str = "smoothness"
    semantic_retention_temperature: float = 0.1


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


def _scale_from_prior(prior: torch.Tensor, low: float, high: float) -> torch.Tensor:
    return low + (high - low) * prior


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
    )

    selected = {
        "coarse": views["coarse_purified"].matrix,
        "mid": views["fagsp_mid_base"].matrix,
        "local": views["local_purified"].matrix,
    }

    graph_tensors: dict[str, torch.Tensor] = {}
    for name, matrix in selected.items():
        matrix = keep_topk_per_row(matrix, topk=cfg.graph_topk)
        graph_tensors[name] = _to_torch_dense(matrix, device=device)

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

    model = RQVAE(
        in_dim=dataset.dim,
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
    ).to(device)
    optimizer = _build_optimizer(cfg, model)
    scheduler = _build_scheduler(cfg, optimizer, steps_per_epoch=len(train_loader))

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
        total_semantic_coarse_loss = 0.0
        total_semantic_mid_loss = 0.0

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

            coarse_subgraph = _select_subgraph(graph_tensors["coarse"], batch_indices)
            mid_subgraph = _select_subgraph(graph_tensors["mid"], batch_indices)
            local_subgraph = _select_subgraph(graph_tensors["local"], batch_indices)
            semantic_subgraph = _select_subgraph(graph_tensors["semantic"], batch_indices)

            graph_losses = {
                "coarse": _weighted_graph_smoothness_loss(level_representations[0], coarse_subgraph, graph_item_scale),
                "mid": _weighted_graph_smoothness_loss(level_representations[1], mid_subgraph, graph_item_scale),
                "local": _weighted_graph_smoothness_loss(level_representations[2], local_subgraph, graph_item_scale),
            }
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

            loss_total = (
                loss_total
                + cfg.coarse_weight * graph_losses["coarse"]
                + cfg.mid_weight * graph_losses["mid"]
                + cfg.local_weight * graph_losses["local"]
                + cfg.semantic_coarse_weight * graph_losses["semantic_coarse"]
                + cfg.semantic_mid_weight * graph_losses["semantic_mid"]
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
            total_semantic_coarse_loss += float(graph_losses["semantic_coarse"].item())
            total_semantic_mid_loss += float(graph_losses["semantic_mid"].item())

        record = {
            "epoch": epoch,
            "total_loss": total_loss,
            "recon_loss": total_recon_loss,
            "rq_loss": total_rq_loss,
            "coarse_graph_loss": total_coarse_graph_loss,
            "mid_graph_loss": total_mid_graph_loss,
            "local_graph_loss": total_local_graph_loss,
            "semantic_coarse_loss": total_semantic_coarse_loss,
            "semantic_mid_loss": total_semantic_mid_loss,
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
            "epoch=%d total=%.6f recon=%.6f rq=%.6f coarse=%.6f mid=%.6f local=%.6f sem_coarse=%.6f sem_mid=%.6f collision=%s",
            epoch,
            record["total_loss"],
            record["recon_loss"],
            record["rq_loss"],
            record["coarse_graph_loss"],
            record["mid_graph_loss"],
            record["local_graph_loss"],
            record["semantic_coarse_loss"],
            record["semantic_mid_loss"],
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
