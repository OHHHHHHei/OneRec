from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from onerec.sid.datasets import EmbDataset
from onerec.sid.models.rqvae import RQVAE


def build_model(ckpt_args, data_dim: int) -> RQVAE:
    return RQVAE(
        in_dim=data_dim,
        num_emb_list=ckpt_args.num_emb_list,
        e_dim=ckpt_args.e_dim,
        layers=ckpt_args.layers,
        dropout_prob=ckpt_args.dropout_prob,
        bn=ckpt_args.bn,
        loss_type=ckpt_args.loss_type,
        quant_loss_weight=ckpt_args.quant_loss_weight,
        beta=getattr(ckpt_args, "beta", 0.25),
        kmeans_init=ckpt_args.kmeans_init,
        kmeans_iters=ckpt_args.kmeans_iters,
        sk_epsilons=ckpt_args.sk_epsilons,
        sk_iters=ckpt_args.sk_iters,
        attn_residual_enable=getattr(ckpt_args, "attn_residual_enable", False),
        attn_residual_mode=getattr(ckpt_args, "attn_residual_mode", "dynamic"),
        attn_residual_reg_weight=getattr(ckpt_args, "attn_residual_reg_weight", 0.0),
        attn_residual_use_rmsnorm=getattr(ckpt_args, "attn_residual_use_rmsnorm", True),
        attn_residual_temperature=getattr(ckpt_args, "attn_residual_temperature", 1.0),
    )


def summarize_gamma(gamma: np.ndarray) -> dict:
    summary = {}
    for level in range(gamma.shape[1]):
        values = gamma[:, level]
        summary[f"level_{level + 1}"] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "p05": float(np.quantile(values, 0.05)),
            "p25": float(np.quantile(values, 0.25)),
            "p50": float(np.quantile(values, 0.50)),
            "p75": float(np.quantile(values, 0.75)),
            "p95": float(np.quantile(values, 0.95)),
            "max": float(values.max()),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump AttnRQ residual weight diagnostics")
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_csv", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    device = torch.device(args.device)
    ckpt = torch.load(args.ckpt_path, map_location=torch.device("cpu"), weights_only=False)
    ckpt_args = ckpt["args"]
    if not getattr(ckpt_args, "attn_residual_enable", False):
        raise ValueError("Checkpoint was not trained with attn_residual_enable=true")

    data = EmbDataset(ckpt_args.data_path)
    model = build_model(ckpt_args, data.dim)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device)
    model.eval()

    loader = DataLoader(data, num_workers=getattr(ckpt_args, "num_workers", 0), batch_size=args.batch_size, shuffle=False)
    all_gamma = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Gamma diagnostics"):
            batch = batch.to(device)
            model(batch, use_sk=False)
            gamma = model.get_last_attn_residual_gamma()
            if gamma is None:
                raise RuntimeError("AttnRQ gamma was not produced by the model forward pass")
            all_gamma.append(gamma.cpu().numpy())

    gamma = np.concatenate(all_gamma, axis=0)
    result = {
        "ckpt_path": args.ckpt_path,
        "num_items": int(gamma.shape[0]),
        "num_levels": int(gamma.shape[1]),
        "attn_residual_mode": getattr(ckpt_args, "attn_residual_mode", "dynamic"),
        "attn_residual_reg_weight": getattr(ckpt_args, "attn_residual_reg_weight", 0.0),
        "attn_residual_use_rmsnorm": getattr(ckpt_args, "attn_residual_use_rmsnorm", True),
        "summary": summarize_gamma(gamma),
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))

    if args.output_csv:
        output_csv = Path(args.output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["item_id", *[f"gamma_l{idx + 1}" for idx in range(gamma.shape[1])]])
            for item_id, row in enumerate(gamma.tolist()):
                writer.writerow([item_id, *row])


if __name__ == "__main__":
    main()
