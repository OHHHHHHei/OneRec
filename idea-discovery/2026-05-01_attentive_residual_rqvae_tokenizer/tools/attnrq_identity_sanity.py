from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from onerec.sid.models.rqvae import RQVAE


def parse_int_list(raw: str) -> list[int]:
    return [int(part) for part in raw.split(",") if part]


def load_inputs(args: argparse.Namespace) -> torch.Tensor:
    if args.data_path:
        embeddings = np.load(args.data_path)
        embeddings = embeddings[: args.num_samples]
        return torch.as_tensor(embeddings, dtype=torch.float32)

    generator = torch.Generator().manual_seed(args.seed)
    return torch.randn(args.num_samples, args.in_dim, generator=generator)


def build_model(args: argparse.Namespace, *, attn: bool, in_dim: int) -> RQVAE:
    return RQVAE(
        in_dim=in_dim,
        num_emb_list=parse_int_list(args.num_emb_list),
        e_dim=args.e_dim,
        layers=parse_int_list(args.layers),
        dropout_prob=0.0,
        bn=False,
        loss_type="mse",
        quant_loss_weight=1.0,
        beta=0.25,
        kmeans_init=False,
        kmeans_iters=1,
        sk_epsilons=[0.0] * len(parse_int_list(args.num_emb_list)),
        sk_iters=1,
        attn_residual_enable=attn,
        attn_residual_mode=args.attn_residual_mode,
        attn_residual_reg_weight=args.attn_residual_reg_weight,
        attn_residual_use_rmsnorm=args.attn_residual_use_rmsnorm,
        attn_residual_temperature=args.attn_residual_temperature,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanity check for AttnRQ-Identity")
    parser.add_argument("--data_path", default="")
    parser.add_argument("--output_json", default="")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--in_dim", type=int, default=32)
    parser.add_argument("--layers", default="64,32")
    parser.add_argument("--e_dim", type=int, default=16)
    parser.add_argument("--num_emb_list", default="16,16,16")
    parser.add_argument("--attn_residual_mode", choices=["dynamic", "static"], default="dynamic")
    parser.add_argument("--attn_residual_reg_weight", type=float, default=0.001)
    parser.add_argument("--attn_residual_use_rmsnorm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--attn_residual_temperature", type=float, default=1.0)
    parser.add_argument("--tolerance", type=float, default=1e-7)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    inputs = load_inputs(args)
    in_dim = inputs.shape[-1]

    torch.manual_seed(args.seed)
    baseline = build_model(args, attn=False, in_dim=in_dim)
    attnrq = build_model(args, attn=True, in_dim=in_dim)
    load_result = attnrq.load_state_dict(baseline.state_dict(), strict=False)

    baseline.eval()
    attnrq.eval()

    with torch.no_grad():
        baseline_out, baseline_rq_loss, baseline_indices = baseline(inputs, use_sk=False)
        attnrq_out, attnrq_rq_loss, attnrq_indices = attnrq(inputs, use_sk=False)
        baseline_loss, baseline_recon = baseline.compute_loss(baseline_out, baseline_rq_loss, xs=inputs)
        attnrq_loss, attnrq_recon = attnrq.compute_loss(attnrq_out, attnrq_rq_loss, xs=inputs)
        gamma = attnrq.get_last_attn_residual_gamma()

    max_output_abs_diff = (baseline_out - attnrq_out).abs().max().item()
    max_rq_loss_abs_diff = abs(baseline_rq_loss.item() - attnrq_rq_loss.item())
    max_total_loss_abs_diff = abs(baseline_loss.item() - attnrq_loss.item())
    indices_equal = torch.equal(baseline_indices, attnrq_indices)

    result = {
        "passed": bool(
            max_output_abs_diff < args.tolerance
            and max_rq_loss_abs_diff < args.tolerance
            and max_total_loss_abs_diff < args.tolerance
            and indices_equal
        ),
        "tolerance": args.tolerance,
        "max_output_abs_diff": max_output_abs_diff,
        "max_rq_loss_abs_diff": max_rq_loss_abs_diff,
        "max_total_loss_abs_diff": max_total_loss_abs_diff,
        "indices_equal": indices_equal,
        "baseline_loss": baseline_loss.item(),
        "attnrq_loss": attnrq_loss.item(),
        "baseline_recon_loss": baseline_recon.item(),
        "attnrq_recon_loss": attnrq_recon.item(),
        "gamma_mean_by_level": gamma.mean(dim=0).cpu().tolist(),
        "gamma_std_by_level": gamma.std(dim=0).cpu().tolist(),
        "missing_keys_after_copy": list(load_result.missing_keys),
        "unexpected_keys_after_copy": list(load_result.unexpected_keys),
    }

    payload = json.dumps(result, indent=2)
    print(payload)
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(payload + "\n", encoding="utf-8")

    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
