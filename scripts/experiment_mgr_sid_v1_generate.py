#!/usr/bin/env python
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from onerec.sid.datasets import EmbDataset
from onerec.sid.models.rqvae import RQVAE


def check_collision(indices_as_strings: np.ndarray) -> bool:
    return len(indices_as_strings) == len(set(indices_as_strings.tolist()))


def get_indices_count(indices_as_strings: np.ndarray) -> dict[str, int]:
    counts: dict[str, int] = collections.defaultdict(int)
    for value in indices_as_strings:
        counts[value] += 1
    return counts


def get_collision_item(indices_as_strings: np.ndarray) -> list[list[int]]:
    index2items: dict[str, list[int]] = {}
    for item_id, index in enumerate(indices_as_strings):
        index2items.setdefault(index, []).append(item_id)
    return [items for items in index2items.values() if len(items) > 1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate indices for experimental MGR-SID v1 checkpoints.")
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--summary_file", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_collision_rounds", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    ckpt = torch.load(args.ckpt_path, map_location=torch.device("cpu"), weights_only=False)
    cfg = ckpt["config"]
    state_dict = ckpt["state_dict"]

    data = EmbDataset(cfg["data_path"])
    model = RQVAE(
        in_dim=data.dim,
        num_emb_list=cfg["num_emb_list"],
        e_dim=cfg["e_dim"],
        layers=cfg["layers"],
        dropout_prob=cfg["dropout_prob"],
        bn=cfg["bn"],
        loss_type=cfg["loss_type"],
        quant_loss_weight=cfg["quant_loss_weight"],
        beta=cfg["beta"],
        kmeans_init=cfg["kmeans_init"],
        kmeans_iters=cfg["kmeans_iters"],
        sk_epsilons=cfg["sk_epsilons"],
        sk_iters=cfg["sk_iters"],
    )
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    loader = DataLoader(
        data,
        num_workers=int(cfg.get("num_workers", 0)),
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
    )

    all_indices: list[list[str]] = []
    all_indices_str: list[str] = []
    prefix = ["<a_{}>", "<b_{}>", "<c_{}>", "<d_{}>", "<e_{}>"]

    for batch in tqdm(loader, desc="Initial encoding"):
        batch = batch.to(device)
        indices = model.get_indices(batch, use_sk=False).view(-1, model.rq.num_quantizers).cpu().numpy()
        for index in indices:
            code = [prefix[i].format(int(ind)) for i, ind in enumerate(index)]
            all_indices.append(code)
            all_indices_str.append(str(code))

    all_indices_np = np.array(all_indices)
    all_indices_str_np = np.array(all_indices_str)

    for vq in model.rq.vq_layers[:-1]:
        vq.sk_epsilon = 0.0
    if model.rq.vq_layers[-1].sk_epsilon == 0.0:
        model.rq.vq_layers[-1].sk_epsilon = 0.003

    collision_round = 0
    while collision_round < args.max_collision_rounds and not check_collision(all_indices_str_np):
        for collision_items in get_collision_item(all_indices_str_np):
            batch = data[collision_items].to(device)
            indices = model.get_indices(batch, use_sk=True).view(-1, model.rq.num_quantizers).cpu().numpy()
            for item, index in zip(collision_items, indices):
                code = [prefix[i].format(int(ind)) for i, ind in enumerate(index)]
                all_indices_np[item] = code
                all_indices_str_np[item] = str(code)
        collision_round += 1

    counts = get_indices_count(all_indices_str_np)
    collision_rate = (len(all_indices_str_np) - len(set(all_indices_str_np.tolist()))) / len(all_indices_str_np)
    max_conflict = max(counts.values())

    payload = {str(item): list(indices) for item, indices in enumerate(all_indices_np.tolist())}
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    summary = {
        "ckpt_path": args.ckpt_path,
        "output_file": args.output_file,
        "collision_rate": collision_rate,
        "max_conflict": int(max_conflict),
        "collision_rounds_used": collision_round,
        "num_items": int(len(all_indices_str_np)),
    }
    if args.summary_file:
        os.makedirs(os.path.dirname(args.summary_file), exist_ok=True)
        with open(args.summary_file, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
