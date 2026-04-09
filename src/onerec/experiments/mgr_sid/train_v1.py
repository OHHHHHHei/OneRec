from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import get_constant_schedule_with_warmup, get_linear_schedule_with_warmup

from onerec.sid.models.rqvae import RQVAE
from onerec.sid.utils import ensure_dir, get_local_time
from onerec.utils.io import read_yaml

from .paper_transplants import keep_topk_per_row
from .transplanted_graph_bank import build_transplanted_graph_bank


@dataclass
class MgrSidTrainConfig:
    mode: str
    data_path: str
    train_csv: str
    semantic_embedding_path: str | None
    ckpt_dir: str
    device: str = "cuda:0"
    seed: int = 2024
    epochs: int = 10
    batch_size: int = 256
    num_workers: int = 0
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
    coarse_weight: float = 0.05
    mid_weight: float = 0.15
    local_weight: float = 0.05


def load_train_config(config_path: str, overrides: dict[str, Any] | None = None) -> MgrSidTrainConfig:
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
    return MgrSidTrainConfig(**payload)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class IndexedEmbDataset(Dataset):
    def __init__(self, data_path: str):
        embeddings = np.load(data_path).astype(np.float32)
        nan_mask = np.isnan(embeddings)
        if nan_mask.any():
            embeddings[nan_mask] = 0.0
        inf_mask = np.isinf(embeddings)
        if inf_mask.any():
            embeddings[inf_mask] = 0.0
        self.embeddings = embeddings
        self.dim = embeddings.shape[-1]

    def __getitem__(self, index: int) -> tuple[int, torch.Tensor]:
        return index, torch.tensor(self.embeddings[index], dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.embeddings)


def _to_torch_dense(matrix, device: torch.device) -> torch.Tensor:
    dense = matrix.astype(np.float32).toarray()
    return torch.tensor(dense, dtype=torch.float32, device=device)


def _collision_rate(indices: torch.Tensor) -> float:
    flat = indices.detach().cpu().numpy()
    seen = {"-".join(str(int(v)) for v in row) for row in flat}
    return float((len(flat) - len(seen)) / len(flat))


def _forward_hierarchy(model: RQVAE, x: torch.Tensor, use_sk: bool = True) -> dict[str, Any]:
    encoded = model.encoder(x)
    residual = encoded
    cumulative_outputs: list[torch.Tensor] = []
    losses: list[torch.Tensor] = []
    all_indices: list[torch.Tensor] = []
    quantized = torch.zeros_like(encoded)
    for quantizer in model.rq.vq_layers:
        level_q, level_loss, level_idx = quantizer(residual, use_sk=use_sk)
        residual = residual - level_q
        quantized = quantized + level_q
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
    }


def _graph_smoothness_loss(representations: torch.Tensor, graph: torch.Tensor) -> torch.Tensor:
    propagated = graph @ representations
    return F.mse_loss(representations, propagated)


def _select_subgraph(graph: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    subgraph = graph.index_select(0, indices)
    subgraph = subgraph.index_select(1, indices)
    return subgraph


def _build_optimizer(cfg: MgrSidTrainConfig, model: RQVAE) -> optim.Optimizer:
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


def _build_scheduler(cfg: MgrSidTrainConfig, optimizer: optim.Optimizer, steps_per_epoch: int):
    warmup_steps = cfg.warmup_epochs * steps_per_epoch
    max_steps = cfg.epochs * steps_per_epoch
    if cfg.lr_scheduler_type.lower() == "linear":
        return get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
        )
    return get_constant_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
    )


def _build_graph_views(cfg: MgrSidTrainConfig, device: torch.device, n_items: int) -> dict[str, torch.Tensor]:
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
        "mid_alt": views["gsprec_mid_prism"].matrix,
    }

    graph_tensors: dict[str, torch.Tensor] = {}
    for name, matrix in selected.items():
        matrix = keep_topk_per_row(matrix, topk=cfg.graph_topk)
        graph_tensors[name] = _to_torch_dense(matrix, device=device)
    graph_tensors["uniform_avg"] = (
        graph_tensors["coarse"] + graph_tensors["mid"] + graph_tensors["local"]
    ) / 3.0

    for name, tensor in list(graph_tensors.items()):
        if tensor.shape[0] != n_items:
            graph_tensors[name] = tensor[:n_items, :n_items]
    return graph_tensors


def run_training(cfg: MgrSidTrainConfig) -> dict[str, Any]:
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    ensure_dir(cfg.ckpt_dir)
    run_dir = Path(cfg.ckpt_dir) / get_local_time()
    ensure_dir(str(run_dir))

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("mgr_sid_v1")
    logger.info("Starting aligned MGR-SID v1 experimental training: mode=%s", cfg.mode)

    dataset = IndexedEmbDataset(cfg.data_path)
    train_loader = DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=True,
        pin_memory=True,
    )
    eval_loader = DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=False,
        pin_memory=True,
    )

    graph_tensors = _build_graph_views(cfg, device=device, n_items=len(dataset))

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
        total_uniform_graph_loss = 0.0

        iter_data = tqdm(
            train_loader,
            total=len(train_loader),
            ncols=100,
            desc=f"MGR-Train {epoch}",
        )
        for batch_indices, batch_embeddings in iter_data:
            batch_indices = batch_indices.to(device, non_blocking=True)
            batch_embeddings = batch_embeddings.to(device, non_blocking=True)

            optimizer.zero_grad()
            pack = _forward_hierarchy(model, batch_embeddings, use_sk=True)
            loss_total, loss_recon = model.compute_loss(pack["recon"], pack["rq_loss"], xs=batch_embeddings)

            graph_losses = {
                "coarse": torch.tensor(0.0, device=device),
                "mid": torch.tensor(0.0, device=device),
                "local": torch.tensor(0.0, device=device),
                "uniform": torch.tensor(0.0, device=device),
            }
            if cfg.mode == "uniform_reg":
                uniform_subgraph = _select_subgraph(graph_tensors["uniform_avg"], batch_indices)
                graph_losses["uniform"] = _graph_smoothness_loss(pack["cumulative_outputs"][-1], uniform_subgraph)
                loss_total = loss_total + (cfg.coarse_weight + cfg.mid_weight + cfg.local_weight) * graph_losses["uniform"]
            elif cfg.mode == "hierarchy_reg":
                coarse_subgraph = _select_subgraph(graph_tensors["coarse"], batch_indices)
                mid_subgraph = _select_subgraph(graph_tensors["mid"], batch_indices)
                local_subgraph = _select_subgraph(graph_tensors["local"], batch_indices)
                graph_losses["coarse"] = _graph_smoothness_loss(pack["cumulative_outputs"][0], coarse_subgraph)
                graph_losses["mid"] = _graph_smoothness_loss(pack["cumulative_outputs"][1], mid_subgraph)
                graph_losses["local"] = _graph_smoothness_loss(pack["cumulative_outputs"][2], local_subgraph)
                loss_total = (
                    loss_total
                    + cfg.coarse_weight * graph_losses["coarse"]
                    + cfg.mid_weight * graph_losses["mid"]
                    + cfg.local_weight * graph_losses["local"]
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
            total_uniform_graph_loss += float(graph_losses["uniform"].item())

        record = {
            "epoch": epoch,
            "total_loss": total_loss,
            "recon_loss": total_recon_loss,
            "rq_loss": total_rq_loss,
            "coarse_graph_loss": total_coarse_graph_loss,
            "mid_graph_loss": total_mid_graph_loss,
            "local_graph_loss": total_local_graph_loss,
            "uniform_graph_loss": total_uniform_graph_loss,
        }

        if (epoch + 1) % effective_eval_step == 0:
            model.eval()
            indices_set: list[torch.Tensor] = []
            with torch.no_grad():
                for _, batch_embeddings in tqdm(
                    eval_loader,
                    total=len(eval_loader),
                    ncols=100,
                    desc="MGR-Eval",
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
            "epoch=%d total=%.6f recon=%.6f rq=%.6f coarse=%.6f mid=%.6f local=%.6f uniform=%.6f collision=%s",
            epoch,
            record["total_loss"],
            record["recon_loss"],
            record["rq_loss"],
            record["coarse_graph_loss"],
            record["mid_graph_loss"],
            record["local_graph_loss"],
            record["uniform_graph_loss"],
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
