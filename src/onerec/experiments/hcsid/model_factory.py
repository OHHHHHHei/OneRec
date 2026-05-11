from __future__ import annotations

import torch

from onerec.sid.models.rqvae import RQVAE

from .config import HcsidTrainConfig


def build_model(cfg: HcsidTrainConfig, in_dim: int) -> RQVAE:
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


def load_checkpoint_state(model: RQVAE, ckpt_path: str, device: torch.device) -> None:
    ckpt = torch.load(ckpt_path, map_location=torch.device("cpu"), weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
