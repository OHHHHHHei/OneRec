from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

from onerec.utils.io import read_yaml


@dataclass
class HcsidTrainConfig:
    mode: str = "lmh_hcsid"
    data_path: str = ""
    train_csv: str = ""
    semantic_embedding_path: str | None = None
    ambiguity_csv: str = ""
    ambiguity_column: str = "offline_combined"
    l2_negative_pair_csv: str | None = None
    l2_negative_pair_rule: str | None = None
    l2_negative_pair_use_reliability: bool = True
    ckpt_dir: str = ""
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
    local_min_weight: float = 1.0
    graph_topk: int = 32
    semantic_graph_topk: int = 32
    local_multihop_alpha: float = 0.35
    local_multihop_max_hop: int = 2
    local_multihop_base_weight: float = 1.0
    graph_scale_min: float = 1.0
    graph_scale_max: float = 1.0
    semantic_scale_min: float = 1.0
    semantic_scale_max: float = 1.0
    l1_semantic_weight: float = 0.03
    l2_graph_infonce_weight: float = 0.01
    l2_temperature: float = 0.1
    l3_local_weight: float = 0.02
    hierarchy_stopgrad_previous_levels: bool = True


_ALIASES = {
    "l1_contrastive_pull_weight": "l1_semantic_weight",
    "l2_contrastive_pull_weight": "l2_graph_infonce_weight",
    "l2_infonce_temperature": "l2_temperature",
    "l2_infonce_negative_pair_csv": "l2_negative_pair_csv",
    "l2_infonce_negative_pair_rule": "l2_negative_pair_rule",
    "l2_infonce_use_pair_reliability": "l2_negative_pair_use_reliability",
    "l3_contrastive_pull_weight": "l3_local_weight",
}


def load_train_config(config_path: str, overrides: dict[str, Any] | None = None) -> HcsidTrainConfig:
    payload = dict(read_yaml(config_path))
    for old_key, new_key in _ALIASES.items():
        if old_key in payload and new_key not in payload:
            payload[new_key] = payload[old_key]
    for key, value in (overrides or {}).items():
        if value is not None:
            payload[key] = value

    payload.setdefault("num_emb_list", [256, 256, 256])
    payload.setdefault("layers", [2048, 1024, 512, 256, 128, 64])
    payload.setdefault("sk_epsilons", [0.0, 0.0, 0.0])

    allowed = {field.name for field in fields(HcsidTrainConfig)}
    filtered = {key: value for key, value in payload.items() if key in allowed}
    return HcsidTrainConfig(**filtered)
