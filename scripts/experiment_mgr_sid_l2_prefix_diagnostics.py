#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from onerec.experiments.mgr_sid.graph_bank import (  # noqa: E402
    build_local_graph,
    build_multi_hop_transition_view,
    build_popularity,
    infer_num_items,
    keep_topk_per_row,
    purify_local_graph,
)
from onerec.experiments.mgr_sid.paper_transplants import (  # noqa: E402
    build_semantic_knn_graph,
    load_semantic_embeddings,
)

SID_RE = re.compile(r"<([abc])_(\d+)>")


@dataclass(frozen=True)
class RunSpec:
    name: str
    label: str
    result_json: Path
    index_json: Path
    recipe: str


RUNS = [
    RunSpec(
        name="recipe_original",
        label="recipe-aligned original SFT",
        result_json=ROOT / "results/recovered_legacy/final_result_sft_Industrial_and_Scientific_refactor_align.json",
        index_json=ROOT / "data/Amazon/index/Industrial_and_Scientific.index.json",
        recipe="title_history2sid_on+desc_align_p05",
    ),
    RunSpec(
        name="v2_on_p05",
        label="v2_on_p05 SFT",
        result_json=ROOT
        / "results/experiments/mgr_sid_v2_recipe_isolation_industrial_20260411/final_result_sft_mgr_tokenizer_v2_title_on_desc_p05_Industrial_and_Scientific.json",
        index_json=ROOT / "data_experiment/Amazon/mgr_tokenizer_v2_offline/index/Industrial_and_Scientific.index.json",
        recipe="title_history2sid_on+desc_align_p05",
    ),
    RunSpec(
        name="original_l2",
        label="original_l2_multihop_ranking",
        result_json=ROOT
        / "results/experiments/mgr_sid_original_l2_multihop_ranking_sft_eval_20260421/final_result_sft_mgr_original_l2_multihop_ranking_title_on_desc_p05_4gpu_Industrial_and_Scientific.json",
        index_json=ROOT
        / "data_experiment/Amazon/original_l2_multihop_ranking/index/Industrial_and_Scientific.index.json",
        recipe="title_history2sid_on+desc_align_p05",
    ),
    RunSpec(
        name="original_l3",
        label="original_l3_collab_local",
        result_json=ROOT
        / "results/experiments/mgr_sid_original_l3_collab_local_sft_eval_20260421/final_result_sft_mgr_original_l3_collab_local_title_on_desc_p05_4gpu_Industrial_and_Scientific.json",
        index_json=ROOT / "data_experiment/Amazon/original_l3_collab_local/index/Industrial_and_Scientific.index.json",
        recipe="title_history2sid_on+desc_align_p05",
    ),
    RunSpec(
        name="r720e",
        label="R720e",
        result_json=ROOT
        / "results/experiments/mgr_sid_collab_ranking_local_multihop_mid_l1_inverse_ambiguity_sft_eval_20260419/final_result_sft_mgr_collab_ranking_local_multihop_mid_l1_inverse_ambiguity_title_on_desc_p05_Industrial_and_Scientific.json",
        index_json=ROOT
        / "data_experiment/Amazon/collab_ranking_local_multihop_mid_l1_inverse_ambiguity/index/Industrial_and_Scientific.index.json",
        recipe="title_history2sid_on+desc_align_p05",
    ),
    RunSpec(
        name="strongest_original_sft",
        label="strongest original SFT",
        result_json=ROOT / "results/final_result_sft_Industrial_and_Scientific_title_history2sid_off__desc_align_p05_20260325_192249.json",
        index_json=ROOT / "data/Amazon/index/Industrial_and_Scientific.index.json",
        recipe="title_history2sid_off+desc_align_p05",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose L2 prefix/beam proxy behavior for MGR-SID SFT outputs.")
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "research-progress-log/experiment_launches/2026-04-21_mgr_sid_original_l2_multihop_ranking_industrial"),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260421)
    parser.add_argument(
        "--primary-only",
        action="store_true",
        help="Report only primary cutoffs @1/@3/@5/@10 instead of including @50 diagnostics.",
    )
    parser.add_argument("--history-k", type=int, default=10)
    parser.add_argument("--local-min-weight", type=float, default=1.0)
    parser.add_argument("--local-multihop-alpha", type=float, default=0.35)
    parser.add_argument("--local-multihop-max-hop", type=int, default=2)
    parser.add_argument("--graph-topk", type=int, default=32)
    parser.add_argument("--semantic-topk", type=int, default=64)
    parser.add_argument(
        "--train-csv",
        default=str(ROOT / "data/Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.csv"),
    )
    parser.add_argument(
        "--test-csv",
        default=str(ROOT / "data/Amazon/test/Industrial_and_Scientific_5_2016-10-2018-11.csv"),
    )
    parser.add_argument(
        "--semantic-embedding-path",
        default=str(ROOT / "data/Amazon/index/Industrial_and_Scientific.emb-qwen-td.npy"),
    )
    return parser.parse_args()


def sid_tokens_to_string(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "".join(str(token).strip() for token in value)
    return str(value).strip()


def sid_prefix(sid: str, level: int) -> tuple[str, ...]:
    tokens = tuple(f"<{name}_{idx}>" for name, idx in SID_RE.findall(sid))
    return tokens[:level]


def entropy(values: Iterable[tuple[str, ...]]) -> float:
    counter = Counter(values)
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    probs = np.asarray([count / total for count in counter.values()], dtype=np.float64)
    return float(-np.sum(probs * np.log2(probs)))


def load_index(path: Path) -> tuple[dict[int, str], dict[str, list[int]]]:
    with path.open() as f:
        raw = json.load(f)
    item_to_sid: dict[int, str] = {}
    sid_to_items: dict[str, list[int]] = defaultdict(list)
    for key, value in raw.items():
        item_id = int(key)
        sid = sid_tokens_to_string(value)
        item_to_sid[item_id] = sid
        sid_to_items[sid].append(item_id)
    return item_to_sid, dict(sid_to_items)


def load_result(path: Path) -> list[dict[str, object]]:
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError(f"{path} is not a list result JSON")
    return data


def ranks_for_result(entries: list[dict[str, object]]) -> np.ndarray:
    ranks = np.full(len(entries), fill_value=-1, dtype=np.int32)
    for idx, row in enumerate(entries):
        target = sid_tokens_to_string(row["output"])
        preds = [sid_tokens_to_string(pred) for pred in row.get("predict", [])]
        try:
            ranks[idx] = preds.index(target) + 1
        except ValueError:
            ranks[idx] = -1
    return ranks


def metric_arrays(ranks: np.ndarray, ks: list[int]) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for k in ks:
        hit = ((ranks > 0) & (ranks <= k)).astype(np.float64)
        ndcg = np.zeros_like(hit, dtype=np.float64)
        mask = hit > 0
        ndcg[mask] = 1.0 / np.log2(ranks[mask].astype(np.float64) + 1.0)
        result[f"hr@{k}"] = hit
        result[f"ndcg@{k}"] = ndcg
    return result


def bootstrap_diff(
    a: np.ndarray,
    b: np.ndarray,
    *,
    samples: int,
    seed: int,
    chunk_size: int = 1000,
) -> dict[str, float]:
    diff = a.astype(np.float64) - b.astype(np.float64)
    rng = np.random.default_rng(seed)
    boot: list[np.ndarray] = []
    n = diff.shape[0]
    remaining = samples
    while remaining > 0:
        batch = min(chunk_size, remaining)
        indices = rng.integers(0, n, size=(batch, n), endpoint=False)
        boot.append(diff[indices].mean(axis=1))
        remaining -= batch
    boot_arr = np.concatenate(boot, axis=0)
    mean = float(diff.mean())
    ci_low, ci_high = np.percentile(boot_arr, [2.5, 97.5])
    p_two_sided = 2.0 * min(float(np.mean(boot_arr <= 0.0)), float(np.mean(boot_arr >= 0.0)))
    return {
        "mean_diff": mean,
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "bootstrap_p_two_sided_approx": min(1.0, p_two_sided),
    }


def build_neighbor_sets(args: argparse.Namespace) -> tuple[list[set[int]], list[set[int]], dict[str, object]]:
    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)
    n_items = infer_num_items(train_df, test_df)

    popularity = build_popularity(train_df)
    local_raw = build_local_graph(train_df, n_items=n_items, history_k=args.history_k)
    local_purified = purify_local_graph(local_raw, popularity=popularity, min_weight=args.local_min_weight)
    local_multihop = build_multi_hop_transition_view(
        local_purified,
        name="local_multihop",
        alpha=args.local_multihop_alpha,
        max_hop=args.local_multihop_max_hop,
    ).matrix
    local_multihop = keep_topk_per_row(local_multihop, topk=args.graph_topk).tocsr()

    semantic_embeddings = load_semantic_embeddings(args.semantic_embedding_path)
    if semantic_embeddings is None:
        raise ValueError("semantic embeddings are required")
    semantic_embeddings = semantic_embeddings[:n_items]
    semantic_graph = build_semantic_knn_graph(semantic_embeddings, topk=args.semantic_topk).tocsr()

    def to_sets(matrix) -> list[set[int]]:
        matrix = matrix.tocsr()
        out: list[set[int]] = []
        for row in range(matrix.shape[0]):
            start, end = matrix.indptr[row], matrix.indptr[row + 1]
            out.append(set(int(v) for v in matrix.indices[start:end]))
        return out

    metadata = {
        "n_items": int(n_items),
        "local_multihop_nnz": int(local_multihop.nnz),
        "semantic_graph_nnz": int(semantic_graph.nnz),
        "graph_topk": int(args.graph_topk),
        "semantic_topk": int(args.semantic_topk),
    }
    return to_sets(local_multihop), to_sets(semantic_graph), metadata


def prefix_overlap_for_neighbors(
    item_to_sid: dict[int, str],
    neighbor_sets: list[set[int]],
    *,
    level: int,
) -> dict[str, float]:
    values: list[float] = []
    covered = 0
    for item_id, neighbors in enumerate(neighbor_sets):
        if not neighbors or item_id not in item_to_sid:
            continue
        src_prefix = sid_prefix(item_to_sid[item_id], level)
        denom = 0
        match = 0
        for neigh in neighbors:
            if neigh not in item_to_sid:
                continue
            denom += 1
            if sid_prefix(item_to_sid[neigh], level) == src_prefix:
                match += 1
        if denom > 0:
            covered += 1
            values.append(match / denom)
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()) if arr.size else 0.0,
        "median": float(np.median(arr)) if arr.size else 0.0,
        "covered_items": int(covered),
    }


def summarize_run_proxy(
    entries: list[dict[str, object]],
    sid_to_items: dict[str, list[int]],
    graph_neighbors: list[set[int]],
    semantic_neighbors: list[set[int]],
    *,
    topks: list[int],
) -> dict[str, object]:
    max_topk = max(topks)
    accum: dict[str, list[float]] = defaultdict(list)
    target_ambiguous = 0
    target_unmapped = 0
    pred_unmapped = 0
    hit_ranks: list[int] = []

    for row in entries:
        target = sid_tokens_to_string(row["output"])
        preds = [sid_tokens_to_string(pred) for pred in row.get("predict", [])][:max_topk]
        target_items = sid_to_items.get(target, [])
        if not target_items:
            target_unmapped += 1
            target_item = None
        else:
            if len(target_items) > 1:
                target_ambiguous += 1
            target_item = target_items[0]

        target_l1 = sid_prefix(target, 1)
        target_l2 = sid_prefix(target, 2)
        pred_l1 = [sid_prefix(pred, 1) for pred in preds]
        pred_l2 = [sid_prefix(pred, 2) for pred in preds]

        try:
            hit_ranks.append(preds.index(target) + 1)
        except ValueError:
            pass

        for k in topks:
            pk = preds[:k]
            l1k = pred_l1[:k]
            l2k = pred_l2[:k]
            accum[f"gt_l1_prefix_covered@{k}"].append(float(any(prefix == target_l1 for prefix in l1k)))
            accum[f"gt_l2_prefix_covered@{k}"].append(float(any(prefix == target_l2 for prefix in l2k)))
            accum[f"same_l1_fraction@{k}"].append(float(np.mean([prefix == target_l1 for prefix in l1k])) if l1k else 0.0)
            accum[f"same_l2_fraction@{k}"].append(float(np.mean([prefix == target_l2 for prefix in l2k])) if l2k else 0.0)
            accum[f"l1_entropy@{k}"].append(entropy(l1k))
            accum[f"l2_entropy@{k}"].append(entropy(l2k))

            graph_count = 0
            semantic_count = 0
            mapped_count = 0
            if target_item is not None:
                graph_set = graph_neighbors[target_item] if target_item < len(graph_neighbors) else set()
                semantic_set = semantic_neighbors[target_item] if target_item < len(semantic_neighbors) else set()
                for pred in pk:
                    pred_items = sid_to_items.get(pred, [])
                    if not pred_items:
                        pred_unmapped += 1
                        continue
                    pred_item = pred_items[0]
                    mapped_count += 1
                    if pred_item in graph_set:
                        graph_count += 1
                    if pred_item in semantic_set:
                        semantic_count += 1
            denom = max(mapped_count, 1)
            accum[f"graph_neighbor_fraction@{k}"].append(graph_count / denom)
            accum[f"semantic_neighbor_fraction@{k}"].append(semantic_count / denom)

        first_l1 = next((rank + 1 for rank, prefix in enumerate(pred_l1) if prefix == target_l1), -1)
        first_l2 = next((rank + 1 for rank, prefix in enumerate(pred_l2) if prefix == target_l2), -1)
        accum[f"first_gt_l1_prefix_rank_top{max_topk}"].append(float(first_l1))
        accum[f"first_gt_l2_prefix_rank_top{max_topk}"].append(float(first_l2))

    summary: dict[str, object] = {}
    for key, values in sorted(accum.items()):
        arr = np.asarray(values, dtype=np.float64)
        valid = arr[arr > 0] if key.startswith("first_gt_") else arr
        summary[key] = {
            "mean": float(arr.mean()) if arr.size else 0.0,
            "median_nonzero": float(np.median(valid)) if valid.size else -1.0,
        }
    summary["hit_rank"] = {
        f"hit_count_top{max_topk}": int(len(hit_ranks)),
        "mean_rank_when_hit": float(np.mean(hit_ranks)) if hit_ranks else -1.0,
        "median_rank_when_hit": float(np.median(hit_ranks)) if hit_ranks else -1.0,
    }
    summary["mapping"] = {
        "target_unmapped": int(target_unmapped),
        "target_ambiguous_sid": int(target_ambiguous),
        "pred_unmapped_events": int(pred_unmapped),
    }
    return summary


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    out = ["| " + " | ".join(headers) + " |"]
    out.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(out)


def fmt(value: float) -> str:
    return f"{value:.8f}"


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    graph_neighbors, semantic_neighbors, neighbor_metadata = build_neighbor_sets(args)

    loaded: dict[str, dict[str, object]] = {}
    primary_ks = [1, 3, 5, 10]
    ks = primary_ks if args.primary_only else [1, 3, 5, 10, 20, 50]
    proxy_topks = primary_ks if args.primary_only else [10, 50]
    for spec in RUNS:
        item_to_sid, sid_to_items = load_index(spec.index_json)
        entries = load_result(spec.result_json)
        ranks = ranks_for_result(entries)
        metrics = metric_arrays(ranks, ks)
        loaded[spec.name] = {
            "spec": spec,
            "entries": entries,
            "ranks": ranks,
            "metrics": metrics,
            "item_to_sid": item_to_sid,
            "sid_to_items": sid_to_items,
            "target_item_ids": np.asarray(
                [
                    sid_to_items.get(sid_tokens_to_string(row["output"]), [-1])[0]
                    for row in entries
                ],
                dtype=np.int32,
            ),
            "proxy": summarize_run_proxy(
                entries,
                sid_to_items,
                graph_neighbors,
                semantic_neighbors,
                topks=proxy_topks,
            ),
            "graph_prefix_overlap_l1": prefix_overlap_for_neighbors(item_to_sid, graph_neighbors, level=1),
            "graph_prefix_overlap_l2": prefix_overlap_for_neighbors(item_to_sid, graph_neighbors, level=2),
            "semantic_prefix_overlap_l1": prefix_overlap_for_neighbors(item_to_sid, semantic_neighbors, level=1),
            "semantic_prefix_overlap_l2": prefix_overlap_for_neighbors(item_to_sid, semantic_neighbors, level=2),
        }

    n_examples = {name: len(obj["entries"]) for name, obj in loaded.items()}  # type: ignore[arg-type]
    reference_targets = loaded["recipe_original"]["target_item_ids"]
    alignment: dict[str, object] = {}
    for name, obj in loaded.items():
        targets = obj["target_item_ids"]  # type: ignore[assignment]
        if len(targets) != len(reference_targets):
            mismatches = None
        else:
            mismatches = int(np.sum(targets != reference_targets))
        alignment[name] = {"target_item_mismatches_vs_recipe_original": mismatches}

    metric_summary: dict[str, dict[str, float]] = {}
    for name, obj in loaded.items():
        metrics = obj["metrics"]  # type: ignore[assignment]
        metric_summary[name] = {key: float(arr.mean()) for key, arr in metrics.items()}

    pairs = [
        ("original_l2", "recipe_original"),
        ("original_l3", "recipe_original"),
        ("r720e", "recipe_original"),
        ("v2_on_p05", "recipe_original"),
        ("original_l2", "original_l3"),
        ("original_l2", "r720e"),
        ("original_l2", "v2_on_p05"),
        ("original_l2", "strongest_original_sft"),
    ]
    bootstrap: list[dict[str, object]] = []
    for left, right in pairs:
        bootstrap_metrics = [f"{kind}@{k}" for k in primary_ks for kind in ("ndcg", "hr")]
        if not args.primary_only:
            bootstrap_metrics.append("hr@50")
        for metric_name in bootstrap_metrics:
            left_arr = loaded[left]["metrics"][metric_name]  # type: ignore[index]
            right_arr = loaded[right]["metrics"][metric_name]  # type: ignore[index]
            stat = bootstrap_diff(
                left_arr,
                right_arr,
                samples=args.bootstrap_samples,
                seed=args.bootstrap_seed + len(bootstrap),
            )
            bootstrap.append(
                {
                    "left": left,
                    "right": right,
                    "metric": metric_name,
                    **stat,
                }
            )

    payload = {
        "metadata": {
            "bootstrap_samples": int(args.bootstrap_samples),
            "bootstrap_seed": int(args.bootstrap_seed),
            "neighbor_metadata": neighbor_metadata,
            "n_examples": n_examples,
            "note": "Paired bootstrap captures test-sample uncertainty for existing outputs, not training-seed variance.",
        },
        "alignment": alignment,
        "metrics": metric_summary,
        "bootstrap": bootstrap,
        "proxy_diagnostics": {
            name: loaded[name]["proxy"]
            for name in loaded
        },
        "tokenizer_prefix_overlap": {
            name: {
                "graph_l1": loaded[name]["graph_prefix_overlap_l1"],
                "graph_l2": loaded[name]["graph_prefix_overlap_l2"],
                "semantic_l1": loaded[name]["semantic_prefix_overlap_l1"],
                "semantic_l2": loaded[name]["semantic_prefix_overlap_l2"],
            }
            for name in loaded
        },
        "code_path_audit": {
            "finding": "original_l2_multihop_ranking already uses hierarchy_stopgrad_previous_levels=true; in train_v2._build_level_representations, level_representations[1] is detach(q1)+q2, so L2 ranking gradients do not flow into q1 through the auxiliary ranking representation.",
            "remaining_caveat": "L2 ranking still updates shared encoder/codebooks through q2 and the base reconstruction/RQ losses still train all levels. This is route-preserving for the L2 auxiliary path, not a fully frozen L1 tokenizer.",
            "files": [
                "config/experiments/sid_train_industrial_mgr_sid_original_l2_multihop_ranking.yaml",
                "src/onerec/experiments/mgr_sid/train_v2.py",
            ],
        },
    }

    json_name = (
        "l2_prefix_primary_cutoff_diagnostics_20260421.json"
        if args.primary_only
        else "l2_prefix_bootstrap_proxy_diagnostics_20260421.json"
    )
    json_path = output_dir / json_name
    with json_path.open("w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    metric_rows = []
    for name in ["recipe_original", "v2_on_p05", "original_l2", "original_l3", "r720e", "strongest_original_sft"]:
        m = metric_summary[name]
        row: list[object] = [name]
        for k in primary_ks:
            row.extend([fmt(m[f"ndcg@{k}"]), fmt(m[f"hr@{k}"])])
        if not args.primary_only:
            row.append(fmt(m["hr@50"]))
        metric_rows.append(row)

    boot_rows = []
    for row in bootstrap:
        if row["left"] == "original_l2" or (row["right"] == "recipe_original" and row["left"] in {"original_l3", "r720e", "v2_on_p05"}):
            boot_rows.append(
                [
                    f"{row['left']} - {row['right']}",
                    row["metric"],
                    fmt(float(row["mean_diff"])),
                    f"[{fmt(float(row['ci95_low']))}, {fmt(float(row['ci95_high']))}]",
                    f"{float(row['bootstrap_p_two_sided_approx']):.4f}",
                ]
            )

    proxy_rows = []
    for name in ["recipe_original", "v2_on_p05", "original_l2", "original_l3", "r720e"]:
        proxy = loaded[name]["proxy"]  # type: ignore[assignment]
        row = [name]
        for k in proxy_topks:
            row.append(fmt(proxy[f"gt_l2_prefix_covered@{k}"]["mean"]))  # type: ignore[index]
        row.extend(
            [
                fmt(proxy[f"same_l2_fraction@{proxy_topks[-1]}"]["mean"]),  # type: ignore[index]
                fmt(proxy[f"graph_neighbor_fraction@{proxy_topks[-1]}"]["mean"]),  # type: ignore[index]
                fmt(proxy[f"semantic_neighbor_fraction@{proxy_topks[-1]}"]["mean"]),  # type: ignore[index]
                fmt(proxy["hit_rank"]["mean_rank_when_hit"]),  # type: ignore[index]
            ]
        )
        proxy_rows.append(row)

    overlap_rows = []
    for name in ["recipe_original", "v2_on_p05", "original_l2", "original_l3", "r720e"]:
        overlap = payload["tokenizer_prefix_overlap"][name]
        overlap_rows.append(
            [
                name,
                fmt(overlap["graph_l1"]["mean"]),
                fmt(overlap["graph_l2"]["mean"]),
                fmt(overlap["semantic_l1"]["mean"]),
                fmt(overlap["semantic_l2"]["mean"]),
            ]
        )

    md = "\n".join(
        [
            (
                "# L2 Prefix Primary-Cutoff Diagnostics（第二层前缀主要截断诊断）"
                if args.primary_only
                else "# L2 Prefix Bootstrap / Proxy Diagnostics（第二层前缀自助法与代理诊断）"
            ),
            "",
            "Status（状态）: `diagnostic_snapshot（诊断快照）`",
            "",
            "This diagnostic uses existing evaluate outputs only. Paired bootstrap（配对自助法） measures test-sample uncertainty（测试样本不确定性）, not training-seed variance（训练随机性方差）.",
            "",
            "## Metric Summary（指标摘要）",
            "",
            markdown_table(
                ["run（运行）"]
                + [f"{metric.upper()}@{k}" for k in primary_ks for metric in ("ndcg", "hr")]
                + ([] if args.primary_only else ["HR@50"]),
                metric_rows,
            ),
            "",
            "## Paired Bootstrap（配对自助法）",
            "",
            markdown_table(["comparison（对比）", "metric（指标）", "mean diff（均值差）", "95% CI（置信区间）", "approx p"], boot_rows),
            "",
            (
                "## Final Top10 Proxy Diagnostics（最终前 10 代理诊断）"
                if args.primary_only
                else "## Final Top50 Proxy Diagnostics（最终前 50 代理诊断）"
            ),
            "",
            "These are final-output proxies（最终输出代理）, not exact per-step beam survival（逐步束搜索存活）.",
            "",
            markdown_table(
                ["run（运行）"]
                + [f"GT L2 covered@{k}" for k in proxy_topks]
                + [
                    f"same L2 frac@{proxy_topks[-1]}",
                    f"graph-neighbor frac@{proxy_topks[-1]}",
                    f"semantic-neighbor frac@{proxy_topks[-1]}",
                    f"mean hit rank@{proxy_topks[-1]}",
                ],
                proxy_rows,
            ),
            "",
            "## Tokenizer-Level Neighbor Prefix Overlap（分词器级邻居前缀重叠）",
            "",
            markdown_table(
                [
                    "run（运行）",
                    "graph L1 overlap",
                    "graph L2 overlap",
                    "semantic L1 overlap",
                    "semantic L2 overlap",
                ],
                overlap_rows,
            ),
            "",
            "## Code Path Audit（代码路径审查）",
            "",
            "- `original_l2_multihop_ranking` already sets `hierarchy_stopgrad_previous_levels=true`（前层停梯度为真）.",
            "- In `train_v2._build_level_representations`, `level_representations[1] = detach(q1) + q2`, so the L2 ranking loss（第二层排序损失） does not backpropagate into `q1` through that auxiliary representation.",
            "- Caveat（注意）: this protects the auxiliary L2 path（辅助第二层路径）, but base reconstruction/RQ losses（重建/量化损失） still train all levels.",
            "",
            f"Structured JSON（结构化结果）: `{json_path}`",
            "",
        ]
    )
    md_name = (
        "L2_PREFIX_PRIMARY_CUTOFF_DIAGNOSTICS_20260421.md"
        if args.primary_only
        else "L2_PREFIX_BOOTSTRAP_PROXY_DIAGNOSTICS_20260421.md"
    )
    md_path = output_dir / md_name
    md_path.write_text(md)

    print(f"[ok] wrote {json_path}")
    print(f"[ok] wrote {md_path}")


if __name__ == "__main__":
    main()
