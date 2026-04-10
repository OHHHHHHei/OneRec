#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

from onerec.experiments.mgr_sid.graph_bank import parse_id_list
from onerec.experiments.mgr_sid.paper_transplants import load_semantic_embeddings
from onerec.experiments.mgr_sid.train_v1 import MgrSidTrainConfig
from onerec.experiments.mgr_sid.transplanted_graph_bank import build_transplanted_graph_bank
from onerec.sid.models.rqvae import RQVAE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="R001-R002 minimal proxy sanity for MGR-SID v2.")
    parser.add_argument(
        "--train-csv",
        default="/home/leejt/OneRec/data/Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.csv",
    )
    parser.add_argument(
        "--test-csv",
        default="/home/leejt/OneRec/data/Amazon/test/Industrial_and_Scientific_5_2016-10-2018-11.csv",
    )
    parser.add_argument(
        "--item-json",
        default="/home/leejt/OneRec/data/Amazon/index/Industrial_and_Scientific.item.json",
    )
    parser.add_argument(
        "--semantic-embedding",
        default="/home/leejt/OneRec/data/Amazon/index/Industrial_and_Scientific.emb-qwen-td.npy",
    )
    parser.add_argument(
        "--baseline-index",
        default="/home/leejt/OneRec/output/experiments/mgr_sid_v1_upstream/generated_indices/Industrial_and_Scientific.mgr_upstream_baseline.index.json",
    )
    parser.add_argument(
        "--hierarchy-index",
        default="/home/leejt/OneRec/output/experiments/mgr_sid_v1_upstream/generated_indices/Industrial_and_Scientific.mgr_upstream_hierarchy.index.json",
    )
    parser.add_argument(
        "--hierarchy-ckpt",
        default="/home/leejt/OneRec/output/experiments/mgr_sid_v1_upstream/industrial_hierarchy_reg/Apr-09-2026_23-09-22/best_collision_model.pth",
    )
    parser.add_argument(
        "--topk-rows-csv",
        default="/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-10_mgr_sid_sft_eval_industrial/topk_structural_aligned_rows.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-11_mgr_sid_v2_proxy_sanity",
    )
    parser.add_argument("--semantic-topk", type=int, default=32)
    parser.add_argument("--graph-topk", type=int, default=32)
    parser.add_argument("--history-k", type=int, default=10)
    parser.add_argument("--coarse-min-weight", type=float, default=2.0)
    parser.add_argument("--local-min-weight", type=float, default=1.0)
    parser.add_argument("--community-clusters", type=int, default=64)
    parser.add_argument("--anchor-topk", type=int, default=32)
    parser.add_argument("--semantic-mix", type=float, default=0.35)
    parser.add_argument("--spectral-rank", type=int, default=48)
    parser.add_argument("--band-low", type=float, default=0.25)
    parser.add_argument("--band-high", type=float, default=0.65)
    parser.add_argument("--temporal-mix", type=float, default=0.35)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_index(path: Path) -> dict[int, tuple[str, str, str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: dict[int, tuple[str, str, str]] = {}
    for item_id, tokens in raw.items():
        if not isinstance(tokens, list) or len(tokens) < 3:
            continue
        out[int(item_id)] = (str(tokens[0]), str(tokens[1]), str(tokens[2]))
    return out


def build_l2_leaf_counts(index_map: dict[int, tuple[str, str, str]]) -> tuple[dict[int, int], dict[int, str]]:
    groups: dict[tuple[str, str], set[str]] = {}
    prefixes: dict[int, str] = {}
    for item_id, (a, b, c) in index_map.items():
        groups.setdefault((a, b), set()).add(c)
        prefixes[item_id] = f"{a}{b}"
    leaf_counts = {item_id: len(groups[(a, b)]) for item_id, (a, b, _) in index_map.items()}
    return leaf_counts, prefixes


def minmax_normalize(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    finite_mask = np.isfinite(values)
    if not finite_mask.any():
        return np.zeros_like(values, dtype=np.float32)
    vmin = float(values[finite_mask].min())
    vmax = float(values[finite_mask].max())
    if vmax - vmin < 1e-12:
        return np.zeros_like(values, dtype=np.float32)
    out = np.zeros_like(values, dtype=np.float32)
    out[finite_mask] = (values[finite_mask] - vmin) / (vmax - vmin)
    return out


def load_titles(path: Path) -> dict[int, str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    titles: dict[int, str] = {}
    for key, value in raw.items():
        item_id = int(key)
        if isinstance(value, dict):
            titles[item_id] = str(value.get("title", f"Item_{item_id}"))
        else:
            titles[item_id] = str(value)
    return titles


def build_semantic_neighbors(embeddings: np.ndarray, topk: int) -> tuple[np.ndarray, np.ndarray]:
    n_items = embeddings.shape[0]
    topk = max(2, min(topk + 1, n_items))
    nn = NearestNeighbors(n_neighbors=topk, metric="cosine")
    nn.fit(embeddings)
    distances, indices = nn.kneighbors(embeddings)
    similarities = np.maximum(0.0, 1.0 - distances[:, 1:]).astype(np.float32)
    neighbors = indices[:, 1:].astype(np.int32)
    return neighbors, similarities


def topk_neighbors_from_sparse(matrix, topk: int) -> tuple[np.ndarray, list[np.ndarray]]:
    topk = max(1, int(topk))
    n_items = matrix.shape[0]
    weights = np.zeros((n_items, topk), dtype=np.float32)
    neighbors: list[np.ndarray] = []
    matrix = matrix.tocsr()
    for row in range(n_items):
        start, end = matrix.indptr[row], matrix.indptr[row + 1]
        row_indices = matrix.indices[start:end]
        row_data = matrix.data[start:end]
        if row_data.size == 0:
            neighbors.append(np.asarray([], dtype=np.int32))
            continue
        order = np.argsort(row_data)[::-1][:topk]
        top_idx = row_indices[order].astype(np.int32)
        top_w = row_data[order].astype(np.float32)
        neighbors.append(top_idx)
        weights[row, : len(top_w)] = top_w
    return weights, neighbors


def jaccard_distance(a: np.ndarray, b: np.ndarray) -> float:
    set_a = set(int(v) for v in a.tolist())
    set_b = set(int(v) for v in b.tolist())
    if not set_a and not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    if union == 0:
        return 0.0
    return 1.0 - (inter / union)


def normalized_entropy(weights: np.ndarray) -> float:
    weights = weights[weights > 0]
    if weights.size <= 1:
        return 0.0
    probs = weights / np.maximum(weights.sum(), 1e-12)
    entropy = -float(np.sum(probs * np.log(probs + 1e-12)))
    max_entropy = float(np.log(len(probs)))
    if max_entropy <= 0:
        return 0.0
    return entropy / max_entropy


def aggregate_topk_effects(topk_rows: pd.DataFrame, n_items: int) -> pd.DataFrame:
    frames = []
    metrics = [1, 3, 5, 10, 20, 50]
    grouped = topk_rows.groupby("item_id")
    rows = []
    for item_id, group in grouped:
        row: dict[str, Any] = {"item_id": int(item_id), "count": int(len(group))}
        for k in metrics:
            row[f"improved_at_{k}_rate"] = float(group[f"improved_at_{k}"].mean())
            row[f"worsened_at_{k}_rate"] = float(group[f"worsened_at_{k}"].mean())
        row["baseline_target_l2_fanout_mean"] = float(group["baseline_target_l2_fanout"].mean())
        row["hierarchy_target_l2_fanout_mean"] = float(group["hierarchy_target_l2_fanout"].mean())
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame({"item_id": np.arange(n_items, dtype=np.int32)})
    full = pd.DataFrame({"item_id": np.arange(n_items, dtype=np.int32)})
    merged = full.merge(df, on="item_id", how="left")
    return merged.fillna(0.0)


def instantiate_model_from_ckpt(ckpt_path: Path, device: torch.device) -> tuple[RQVAE, dict[str, Any]]:
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = dict(ckpt["config"])
    model = RQVAE(
        in_dim=int(np.load(cfg["data_path"]).shape[1]),
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
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, cfg


def compute_online_uncertainty(
    model: RQVAE,
    embeddings: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    all_level_inv_margins: list[list[np.ndarray]] = []
    all_level_residual_norms: list[list[np.ndarray]] = []
    n_levels = len(model.rq.vq_layers)
    all_level_inv_margins = [[] for _ in range(n_levels)]
    all_level_residual_norms = [[] for _ in range(n_levels)]

    with torch.no_grad():
        for start in range(0, len(embeddings), batch_size):
            batch = torch.tensor(embeddings[start : start + batch_size], dtype=torch.float32, device=device)
            encoded = model.encoder(batch)
            residual = encoded
            for level, quantizer in enumerate(model.rq.vq_layers):
                latent = residual.view(-1, quantizer.e_dim)
                d = (
                    torch.sum(latent**2, dim=1, keepdim=True)
                    + torch.sum(quantizer.embedding.weight**2, dim=1, keepdim=True).t()
                    - 2 * torch.matmul(latent, quantizer.embedding.weight.t())
                )
                top2 = torch.topk(d, k=2, largest=False, dim=1).values
                margin = (top2[:, 1] - top2[:, 0]).float()
                inv_margin = (1.0 / (margin + 1e-6)).detach().cpu().numpy()
                all_level_inv_margins[level].append(inv_margin)

                indices = torch.argmin(d, dim=-1)
                level_q = quantizer.embedding(indices).view(residual.shape)
                residual = residual - level_q
                residual_norm = torch.norm(residual, dim=1).detach().cpu().numpy()
                all_level_residual_norms[level].append(residual_norm)

    level_scores: dict[str, np.ndarray] = {}
    combined_parts: list[np.ndarray] = []
    for level in range(n_levels):
        inv_margin = np.concatenate(all_level_inv_margins[level], axis=0)
        residual_norm = np.concatenate(all_level_residual_norms[level], axis=0)
        score = 0.6 * minmax_normalize(inv_margin) + 0.4 * minmax_normalize(residual_norm)
        level_scores[f"level_{level + 1}_uncertainty"] = score.astype(np.float32)
        combined_parts.append(score.astype(np.float32))
    level_scores["online_uncertainty"] = np.mean(np.stack(combined_parts, axis=0), axis=0).astype(np.float32)
    return level_scores


def summarize_proxy(
    name: str,
    scores: np.ndarray,
    hard_mask: np.ndarray,
    easy_mask: np.ndarray,
    improved_rate: np.ndarray,
    worsened_rate: np.ndarray,
    leaf_reduction: np.ndarray,
    titles: dict[int, str],
) -> dict[str, Any]:
    hard_scores = scores[hard_mask]
    easy_scores = scores[easy_mask]
    auc = None
    if hard_mask.sum() > 0 and easy_mask.sum() > 0:
        labels = np.concatenate([np.ones(hard_mask.sum()), np.zeros(easy_mask.sum())])
        values = np.concatenate([hard_scores, easy_scores])
        auc = float(roc_auc_score(labels, values))

    order = np.argsort(scores)[::-1]
    top10_n = max(1, int(0.1 * len(scores)))
    top20_n = max(1, int(0.2 * len(scores)))
    top10_idx = order[:top10_n]
    top20_idx = order[:top20_n]

    result = {
        "name": name,
        "hard_count": int(hard_mask.sum()),
        "easy_count": int(easy_mask.sum()),
        "hard_mean": float(hard_scores.mean()) if hard_scores.size else 0.0,
        "easy_mean": float(easy_scores.mean()) if easy_scores.size else 0.0,
        "hard_easy_gap": float(hard_scores.mean() - easy_scores.mean()) if hard_scores.size and easy_scores.size else 0.0,
        "hard_vs_easy_auc": auc,
        "hard_base_rate": float(hard_mask.mean()),
        "hard_rate_top10pct": float(hard_mask[top10_idx].mean()) if top10_idx.size else 0.0,
        "hard_rate_top20pct": float(hard_mask[top20_idx].mean()) if top20_idx.size else 0.0,
        "improved_at_3_rate_top10pct": float(improved_rate[top10_idx].mean()) if top10_idx.size else 0.0,
        "worsened_at_3_rate_top10pct": float(worsened_rate[top10_idx].mean()) if top10_idx.size else 0.0,
        "mean_leaf_reduction_top10pct": float(leaf_reduction[top10_idx].mean()) if top10_idx.size else 0.0,
        "top_examples": [
            {"item_id": int(idx), "title": titles.get(int(idx), f"Item_{int(idx)}"), "score": float(scores[idx])}
            for idx in top10_idx[:10]
        ],
    }
    result["usable"] = bool(
        (result["hard_easy_gap"] > 0.05)
        and ((result["hard_vs_easy_auc"] or 0.0) >= 0.60)
        and (result["hard_rate_top10pct"] > result["hard_base_rate"])
    )
    return result


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# MGR-SID v2 Proxy Sanity",
        "",
        f"- Date: {summary['date']}",
        f"- Item count: {summary['item_count']}",
        f"- Hard-item definition: baseline `l2_leaf_count >= 4`",
        f"- Easy-item definition: baseline `l2_leaf_count == 1`",
        "",
        "## Main Takeaways",
        "",
    ]
    for key in ("offline_combined", "offline_plus_online"):
        row = summary["proxy_summaries"][key]
        lines.extend(
            [
                f"### {key}",
                f"- usable: `{row['usable']}`",
                f"- hard-vs-easy AUC: `{row['hard_vs_easy_auc']}`",
                f"- hard/easy mean gap: `{row['hard_easy_gap']:.4f}`",
                f"- hard base rate: `{row['hard_base_rate']:.4f}`",
                f"- hard rate in top 10% proxy items: `{row['hard_rate_top10pct']:.4f}`",
                f"- improved@3 rate in top 10% proxy items: `{row['improved_at_3_rate_top10pct']:.4f}`",
                f"- worsened@3 rate in top 10% proxy items: `{row['worsened_at_3_rate_top10pct']:.4f}`",
                f"- mean `l2` leaf-count reduction in top 10% proxy items: `{row['mean_leaf_reduction_top10pct']:.4f}`",
                "",
            ]
        )

    lines.extend(
        [
            "## Proxy Components",
            "",
            f"- semantic density mean: `{summary['component_means']['semantic_density']:.4f}`",
            f"- semantic-collab disagreement mean: `{summary['component_means']['semantic_collab_disagreement']:.4f}`",
            f"- graph competition mean: `{summary['component_means']['graph_competition']:.4f}`",
            f"- online uncertainty mean: `{summary['component_means']['online_uncertainty']:.4f}`",
            "",
            "## Recommendation",
            "",
        ]
    )
    if summary["proxy_summaries"]["offline_plus_online"]["usable"]:
        lines.append("- `R002` looks usable as the first `v2` ambiguity setting. Prefer the combined offline + online proxy for the first tokenizer run.")
    elif summary["proxy_summaries"]["offline_combined"]["usable"]:
        lines.append("- `R001` already looks usable. Start `v2` with the offline combined prior and keep the online term optional.")
    else:
        lines.append("- Neither minimal proxy cleared the usability threshold. Revisit proxy design before launching `v2` training.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)
    titles = load_titles(Path(args.item_json))

    baseline_index = load_index(Path(args.baseline_index))
    hierarchy_index = load_index(Path(args.hierarchy_index))
    baseline_l2_leaf_counts, baseline_l2_prefixes = build_l2_leaf_counts(baseline_index)
    hierarchy_l2_leaf_counts, _ = build_l2_leaf_counts(hierarchy_index)

    semantic_embeddings = load_semantic_embeddings(args.semantic_embedding)
    assert semantic_embeddings is not None
    n_items = semantic_embeddings.shape[0]

    views = build_transplanted_graph_bank(
        train_df=train_df,
        test_df=test_df,
        history_k=args.history_k,
        coarse_min_weight=args.coarse_min_weight,
        local_min_weight=args.local_min_weight,
        n_clusters=args.community_clusters,
        seed=2024,
        semantic_embedding_path=args.semantic_embedding,
        anchor_topk=args.anchor_topk,
        semantic_mix=args.semantic_mix,
        spectral_rank=args.spectral_rank,
        band_low=args.band_low,
        band_high=args.band_high,
        temporal_mix=args.temporal_mix,
    )
    mid_graph = views["fagsp_mid_base"].matrix

    sem_neighbors, sem_sims = build_semantic_neighbors(semantic_embeddings, topk=args.semantic_topk)
    semantic_density = sem_sims.mean(axis=1).astype(np.float32)

    graph_weights, graph_neighbors = topk_neighbors_from_sparse(mid_graph, topk=args.graph_topk)
    semantic_collab_disagreement = np.asarray(
        [jaccard_distance(sem_neighbors[i], graph_neighbors[i]) for i in range(n_items)],
        dtype=np.float32,
    )
    graph_competition = np.asarray(
        [normalized_entropy(graph_weights[i]) for i in range(n_items)],
        dtype=np.float32,
    )

    offline_combined = (
        minmax_normalize(semantic_density)
        + minmax_normalize(semantic_collab_disagreement)
        + minmax_normalize(graph_competition)
    ) / 3.0

    device = torch.device(args.device)
    model, _ = instantiate_model_from_ckpt(Path(args.hierarchy_ckpt), device=device)
    online_scores = compute_online_uncertainty(
        model=model,
        embeddings=semantic_embeddings.astype(np.float32),
        batch_size=args.batch_size,
        device=device,
    )
    offline_plus_online = (
        minmax_normalize(offline_combined) + minmax_normalize(online_scores["online_uncertainty"])
    ) / 2.0

    topk_rows = pd.read_csv(args.topk_rows_csv)
    effects = aggregate_topk_effects(topk_rows, n_items=n_items)

    baseline_leaf = np.asarray([baseline_l2_leaf_counts.get(i, 0) for i in range(n_items)], dtype=np.int32)
    hierarchy_leaf = np.asarray([hierarchy_l2_leaf_counts.get(i, 0) for i in range(n_items)], dtype=np.int32)
    leaf_reduction = (baseline_leaf - hierarchy_leaf).astype(np.float32)
    hard_mask = baseline_leaf >= 4
    easy_mask = baseline_leaf == 1

    improved_rate = effects["improved_at_3_rate"].to_numpy(dtype=np.float32)
    worsened_rate = effects["worsened_at_3_rate"].to_numpy(dtype=np.float32)

    proxy_summaries = {
        "offline_combined": summarize_proxy(
            name="offline_combined",
            scores=offline_combined,
            hard_mask=hard_mask,
            easy_mask=easy_mask,
            improved_rate=improved_rate,
            worsened_rate=worsened_rate,
            leaf_reduction=leaf_reduction,
            titles=titles,
        ),
        "offline_plus_online": summarize_proxy(
            name="offline_plus_online",
            scores=offline_plus_online,
            hard_mask=hard_mask,
            easy_mask=easy_mask,
            improved_rate=improved_rate,
            worsened_rate=worsened_rate,
            leaf_reduction=leaf_reduction,
            titles=titles,
        ),
    }

    per_item = pd.DataFrame(
        {
            "item_id": np.arange(n_items, dtype=np.int32),
            "title": [titles.get(i, f"Item_{i}") for i in range(n_items)],
            "baseline_l2_leaf_count": baseline_leaf,
            "hierarchy_l2_leaf_count": hierarchy_leaf,
            "baseline_l2_prefix": [baseline_l2_prefixes.get(i, "") for i in range(n_items)],
            "semantic_density": semantic_density,
            "semantic_collab_disagreement": semantic_collab_disagreement,
            "graph_competition": graph_competition,
            "offline_combined": offline_combined,
            "online_uncertainty": online_scores["online_uncertainty"],
            "offline_plus_online": offline_plus_online,
            "leaf_reduction": leaf_reduction,
        }
    ).merge(effects, on="item_id", how="left")
    per_item = per_item.fillna(0.0)

    summary = {
        "date": "2026-04-11",
        "item_count": n_items,
        "proxy_summaries": proxy_summaries,
        "component_means": {
            "semantic_density": float(semantic_density.mean()),
            "semantic_collab_disagreement": float(semantic_collab_disagreement.mean()),
            "graph_competition": float(graph_competition.mean()),
            "online_uncertainty": float(online_scores["online_uncertainty"].mean()),
        },
        "inputs": {
            "train_csv": args.train_csv,
            "test_csv": args.test_csv,
            "baseline_index": args.baseline_index,
            "hierarchy_index": args.hierarchy_index,
            "hierarchy_ckpt": args.hierarchy_ckpt,
            "topk_rows_csv": args.topk_rows_csv,
        },
    }

    (output_dir / "proxy_item_scores.csv").write_text(per_item.to_csv(index=False), encoding="utf-8")
    (output_dir / "proxy_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "README.md").write_text(render_markdown(summary), encoding="utf-8")

    print(json.dumps(summary["proxy_summaries"], indent=2, ensure_ascii=False))
    print(f"Wrote proxy sanity results to {output_dir}")


if __name__ == "__main__":
    main()
