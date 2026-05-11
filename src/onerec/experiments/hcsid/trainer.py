from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import optim
from tqdm import tqdm
from transformers import get_constant_schedule_with_warmup, get_linear_schedule_with_warmup

from onerec.sid.models.rqvae import RQVAE
from onerec.sid.utils import ensure_dir, get_local_time

from .config import HcsidTrainConfig
from .data import IndexedEmbDataset, set_seed
from .graphs import build_hcsid_graphs, select_subgraph
from .losses import (
    graph_guided_infonce_loss,
    load_ambiguity_prior,
    load_pair_matrix,
    pairwise_pull_loss,
    scale_from_prior,
)
from .model_factory import build_model


def _build_optimizer(cfg: HcsidTrainConfig, model: RQVAE) -> optim.Optimizer:
    params = model.parameters()
    learner = cfg.learner.lower()
    if learner == "adam":
        return optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if learner == "sgd":
        return optim.SGD(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if learner == "adagrad":
        return optim.Adagrad(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if learner == "rmsprop":
        return optim.RMSprop(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if learner == "adamw":
        return optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    raise ValueError(f"Unsupported learner: {cfg.learner}")


def _build_scheduler(cfg: HcsidTrainConfig, optimizer: optim.Optimizer, steps_per_epoch: int):
    warmup_steps = cfg.warmup_epochs * steps_per_epoch
    max_steps = cfg.epochs * steps_per_epoch
    if cfg.lr_scheduler_type.lower() == "linear":
        return get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
        )
    return get_constant_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=warmup_steps)


def _forward_hierarchy(model: RQVAE, x: torch.Tensor, use_sk: bool = True) -> dict[str, Any]:
    encoded = model.encoder(x)
    residual = encoded
    cumulative_outputs: list[torch.Tensor] = []
    level_outputs: list[torch.Tensor] = []
    losses: list[torch.Tensor] = []
    all_indices: list[torch.Tensor] = []
    quantized = torch.zeros_like(encoded)
    for quantizer in model.rq.vq_layers:
        level_q, level_loss, level_idx = quantizer(residual, use_sk=use_sk)
        residual = residual - level_q
        quantized = quantized + level_q
        level_outputs.append(level_q)
        cumulative_outputs.append(quantized.clone())
        losses.append(level_loss)
        all_indices.append(level_idx)
    rq_loss = torch.stack(losses).mean()
    indices = torch.stack(all_indices, dim=-1)
    recon = model.decoder(quantized)
    return {
        "recon": recon,
        "rq_loss": rq_loss,
        "indices": indices,
        "cumulative_outputs": cumulative_outputs,
        "level_outputs": level_outputs,
    }


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


def _collision_rate(indices: torch.Tensor) -> float:
    flat = indices.detach().cpu().numpy()
    seen = {"-".join(str(int(v)) for v in row) for row in flat}
    return float((len(flat) - len(seen)) / len(flat))


def run_training(cfg: HcsidTrainConfig) -> dict[str, Any]:
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    ensure_dir(cfg.ckpt_dir)
    run_dir = Path(cfg.ckpt_dir) / get_local_time()
    ensure_dir(str(run_dir))

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("hcsid")
    logger.info("Starting LMH-HCSID tokenizer training")

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

    graphs = build_hcsid_graphs(cfg, device=device, n_items=len(dataset))
    ambiguity_prior = load_ambiguity_prior(
        path=cfg.ambiguity_csv,
        column=cfg.ambiguity_column,
        n_items=len(dataset),
        device=device,
    )
    if cfg.l2_graph_infonce_weight > 0:
        if not cfg.l2_negative_pair_csv:
            raise ValueError("l2_negative_pair_csv is required when l2_graph_infonce_weight > 0")
        l2_negative_pairs = load_pair_matrix(
            path=cfg.l2_negative_pair_csv,
            n_items=len(dataset),
            device=device,
            rule=cfg.l2_negative_pair_rule,
            use_reliability=cfg.l2_negative_pair_use_reliability,
        )
    else:
        l2_negative_pairs = torch.zeros((len(dataset), len(dataset)), dtype=torch.float32, device=device)

    model = build_model(cfg, in_dim=dataset.dim).to(device)
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
        totals = {
            "total_loss": 0.0,
            "recon_loss": 0.0,
            "rq_loss": 0.0,
            "l1_semantic_loss": 0.0,
            "l2_graph_infonce_loss": 0.0,
            "l3_local_loss": 0.0,
        }

        iter_data = tqdm(train_loader, total=len(train_loader), ncols=100, desc=f"HCSID-Train {epoch}")
        for batch_indices, batch_embeddings in iter_data:
            batch_indices = batch_indices.to(device, non_blocking=True)
            batch_embeddings = batch_embeddings.to(device, non_blocking=True)

            optimizer.zero_grad()
            pack = _forward_hierarchy(model, batch_embeddings, use_sk=True)
            loss_total, loss_recon = model.compute_loss(pack["recon"], pack["rq_loss"], xs=batch_embeddings)
            level_reps = _build_level_representations(
                cumulative_outputs=pack["cumulative_outputs"],
                level_outputs=pack["level_outputs"],
                stopgrad_previous_levels=cfg.hierarchy_stopgrad_previous_levels,
            )

            prior_batch = ambiguity_prior.index_select(0, batch_indices)
            graph_item_scale = scale_from_prior(prior_batch, cfg.graph_scale_min, cfg.graph_scale_max)
            semantic_item_scale = scale_from_prior(1.0 - prior_batch, cfg.semantic_scale_min, cfg.semantic_scale_max)

            l1_subgraph = select_subgraph(graphs.l1_semantic, batch_indices)
            l2_positive_subgraph = select_subgraph(graphs.l2_local_multihop, batch_indices)
            l2_negative_subgraph = select_subgraph(l2_negative_pairs, batch_indices)
            l3_subgraph = select_subgraph(graphs.l3_local, batch_indices)

            l1_loss = pairwise_pull_loss(level_reps[0], l1_subgraph, semantic_item_scale)
            l2_loss = graph_guided_infonce_loss(
                level_reps[1],
                positive_pair_weights=l2_positive_subgraph,
                negative_pair_weights=l2_negative_subgraph,
                item_scales=graph_item_scale,
                temperature=cfg.l2_temperature,
            )
            l3_loss = pairwise_pull_loss(level_reps[2], l3_subgraph, graph_item_scale)

            loss_total = (
                loss_total
                + cfg.l1_semantic_weight * l1_loss
                + cfg.l2_graph_infonce_weight * l2_loss
                + cfg.l3_local_weight * l3_loss
            )

            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            totals["total_loss"] += float(loss_total.item())
            totals["recon_loss"] += float(loss_recon.item())
            totals["rq_loss"] += float(pack["rq_loss"].item())
            totals["l1_semantic_loss"] += float(l1_loss.item())
            totals["l2_graph_infonce_loss"] += float(l2_loss.item())
            totals["l3_local_loss"] += float(l3_loss.item())

        record = {"epoch": epoch, **totals}
        if (epoch + 1) % effective_eval_step == 0:
            model.eval()
            indices_set: list[torch.Tensor] = []
            with torch.no_grad():
                for _, batch_embeddings in tqdm(eval_loader, total=len(eval_loader), ncols=100, desc="HCSID-Eval"):
                    batch_embeddings = batch_embeddings.to(device, non_blocking=True)
                    indices_set.append(model.get_indices(batch_embeddings, use_sk=False))
            indices = torch.cat(indices_set, dim=0)
            collision_rate = _collision_rate(indices)
            record["collision_rate"] = collision_rate

            if record["total_loss"] < best["loss"]:
                best["loss"] = record["total_loss"]
                best["epoch"] = epoch
                ckpt = run_dir / "best_loss_model.pth"
                torch.save({"config": asdict(cfg), "state_dict": model.state_dict(), "epoch": epoch}, ckpt)
                best["best_loss_ckpt"] = str(ckpt)

            if collision_rate < best["collision_rate"]:
                best["collision_rate"] = collision_rate
                ckpt = run_dir / "best_collision_model.pth"
                torch.save({"config": asdict(cfg), "state_dict": model.state_dict(), "epoch": epoch}, ckpt)
                best["best_collision_ckpt"] = str(ckpt)

        logger.info(
            "epoch=%d total=%.6f recon=%.6f rq=%.6f l1_sem=%.6f l2_nce=%.6f l3_local=%.6f collision=%s",
            epoch,
            record["total_loss"],
            record["recon_loss"],
            record["rq_loss"],
            record["l1_semantic_loss"],
            record["l2_graph_infonce_loss"],
            record["l3_local_loss"],
            f"{record.get('collision_rate', float('nan')):.6f}" if "collision_rate" in record else "na",
        )
        history.append(record)

    summary = {"config": asdict(cfg), "run_dir": str(run_dir), "best": best, "history": history}
    summary_path = run_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    logger.info("Summary written to %s", summary_path)
    return summary
