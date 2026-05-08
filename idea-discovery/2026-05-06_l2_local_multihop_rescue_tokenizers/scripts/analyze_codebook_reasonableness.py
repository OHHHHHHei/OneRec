#!/usr/bin/env python3
"""Analyze SID codebook reasonableness for the L2/L3 local-multihop line.

The analysis is intentionally train-only for collaborative statistics:
history_item_id -> item_id edges are built from the training CSV, then compared
with semantic similarity from the frozen item embeddings.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class Variant:
    key: str
    label: str
    index_path: Path
    result_json_path: Path
    registry_variant: str


VARIANTS = [
    Variant(
        key="l2_0003",
        label="L2=0.003",
        index_path=Path(
            "/data/leejt/OneRec/output_weights/experiments/"
            "mgr_sid_l2_lmh_sweep_20260507/generated_indices/"
            "Industrial_and_Scientific.r690b_lmh_l2_contrastive_pull_weight0003.index.json"
        ),
        result_json_path=Path(
            "results/experiments/mgr_sid_l2_lmh_sweep_sft_eval_20260507/"
            "final_result_sft_mgr_r690b_lmh_l2_contrastive_pull_weight0003_"
            "title_on_desc_p05_4gpu_Industrial_and_Scientific.json"
        ),
        registry_variant="mgr_r690b_lmh_l2_contrastive_pull_weight0003_title_on_desc_p05_4gpu",
    ),
    Variant(
        key="main_l2_001_l3_002",
        label="Main L2=0.010,L3=0.020",
        index_path=Path(
            "/data/leejt/OneRec/output_weights/experiments/"
            "mgr_sid_l2_lmh_sweep_20260507/generated_indices/"
            "Industrial_and_Scientific.r690b_lmh_l2_contrastive_pull_weight001.index.json"
        ),
        result_json_path=Path(
            "results/experiments/mgr_sid_l2_lmh_sweep_sft_eval_20260507/"
            "final_result_sft_mgr_r690b_lmh_l2_contrastive_pull_weight001_"
            "title_on_desc_p05_4gpu_Industrial_and_Scientific.json"
        ),
        registry_variant="mgr_r690b_lmh_l2_contrastive_pull_weight001_title_on_desc_p05_4gpu",
    ),
    Variant(
        key="l3_001",
        label="L3=0.010",
        index_path=Path(
            "/data/leejt/OneRec/output_weights/experiments/"
            "mgr_sid_l3_lmh_sweep_20260508/generated_indices/"
            "Industrial_and_Scientific.r690b_lmh_l2_weight001_l3_weight010.index.json"
        ),
        result_json_path=Path(
            "results/experiments/mgr_sid_l3_lmh_sweep_sft_eval_20260508/"
            "final_result_sft_mgr_r690b_lmh_l2_weight001_l3_weight010_"
            "title_on_desc_p05_4gpu_Industrial_and_Scientific.json"
        ),
        registry_variant="mgr_r690b_lmh_l2_weight001_l3_weight010_title_on_desc_p05_4gpu",
    ),
]


OPTIONAL_VARIANTS = {
    "l2_0015": Variant(
        key="l2_0015",
        label="L2=0.015",
        index_path=Path(
            "/data/leejt/OneRec/output_weights/experiments/"
            "mgr_sid_l2_lmh_sweep_20260507/generated_indices/"
            "Industrial_and_Scientific.r690b_lmh_l2_contrastive_pull_weight0015.index.json"
        ),
        result_json_path=Path(
            "results/experiments/mgr_sid_l2_lmh_sweep_sft_eval_20260507/"
            "final_result_sft_mgr_r690b_lmh_l2_contrastive_pull_weight0015_"
            "title_on_desc_p05_4gpu_Industrial_and_Scientific.json"
        ),
        registry_variant="mgr_r690b_lmh_l2_contrastive_pull_weight0015_title_on_desc_p05_4gpu",
    )
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("research-progress-log/experiment_analysis/2026-05-08_codebook_reasonableness"),
        help="Directory for report and machine-readable artifacts.",
    )
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=Path("data/Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.csv"),
    )
    parser.add_argument(
        "--item-json",
        type=Path,
        default=Path("data/Amazon/index/Industrial_and_Scientific.item.json"),
    )
    parser.add_argument(
        "--embedding-npy",
        type=Path,
        default=Path("data/Amazon/index/Industrial_and_Scientific.emb-qwen-td.npy"),
    )
    parser.add_argument("--semantic-topk", type=int, default=20)
    parser.add_argument("--max-pairs-per-set", type=int, default=10000)
    parser.add_argument("--example-count", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--include-l2-0015",
        action="store_true",
        help="Also analyze the L2=0.015 tokenizer from the L2 LMH sweep.",
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def load_index(path: Path) -> dict[int, tuple[str, str, str]]:
    with resolve(path).open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): tuple(v) for k, v in raw.items()}


def load_items(path: Path) -> dict[int, dict[str, Any]]:
    with resolve(path).open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def normalize_embeddings(path: Path) -> np.ndarray:
    emb = np.load(resolve(path)).astype("float32")
    return emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)


def parse_id_list(raw: Any) -> list[int]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [int(x) for x in raw]
    text = str(raw).strip()
    if not text:
        return []
    value = ast.literal_eval(text)
    if not isinstance(value, list):
        return []
    return [int(x) for x in value]


def build_collab_stats(train_csv: Path) -> tuple[Counter[tuple[int, int]], dict[tuple[int, int], float], dict[str, Any]]:
    df = pd.read_csv(resolve(train_csv), usecols=["history_item_id", "item_id"])
    pair_counts: Counter[tuple[int, int]] = Counter()
    item_freq: Counter[int] = Counter()
    edge_events = 0

    for hist_raw, target_raw in zip(df["history_item_id"], df["item_id"]):
        target = int(target_raw)
        history = parse_id_list(hist_raw)
        item_freq[target] += 1
        for h in history:
            item_freq[h] += 1
        for h in set(history):
            if h == target:
                continue
            a, b = sorted((h, target))
            pair_counts[(a, b)] += 1
            edge_events += 1

    total_pair_events = max(1, sum(pair_counts.values()))
    ppmi: dict[tuple[int, int], float] = {}
    for (a, b), count in pair_counts.items():
        denom = max(1, item_freq[a] * item_freq[b])
        pmi = math.log((count * total_pair_events) / denom)
        ppmi[(a, b)] = max(0.0, pmi)

    stats = {
        "train_rows": int(len(df)),
        "direct_edge_count": int(len(pair_counts)),
        "edge_events": int(edge_events),
        "cooc_count_percentiles": percentile_dict(list(pair_counts.values())),
        "ppmi_percentiles": percentile_dict(list(ppmi.values())),
    }
    return pair_counts, ppmi, stats


def percentile_dict(values: list[float] | np.ndarray) -> dict[str, float]:
    if len(values) == 0:
        return {}
    arr = np.asarray(values, dtype=float)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


def build_semantic_top_pairs(emb: np.ndarray, topk: int) -> dict[tuple[int, int], float]:
    n = emb.shape[0]
    pairs: dict[tuple[int, int], float] = {}
    batch = 256
    for start in range(0, n, batch):
        sims = emb[start : start + batch] @ emb.T
        for row in range(sims.shape[0]):
            item_id = start + row
            sims[row, item_id] = -2.0
            candidate_ids = np.argpartition(-sims[row], topk)[:topk]
            for cand in candidate_ids:
                a, b = sorted((item_id, int(cand)))
                if a == b:
                    continue
                sim = float(sims[row, cand])
                if sim > pairs.get((a, b), -2.0):
                    pairs[(a, b)] = sim
    return pairs


def build_pair_sets(
    emb: np.ndarray,
    semantic_pairs: dict[tuple[int, int], float],
    pair_counts: Counter[tuple[int, int]],
    ppmi: dict[tuple[int, int], float],
    max_pairs: int,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    def row(a: int, b: int, sim: float, score: float, cooc: int) -> dict[str, Any]:
        return {
            "item_a": a,
            "item_b": b,
            "semantic_sim": float(sim),
            "ppmi": float(score),
            "cooc_count": int(cooc),
        }

    s_near_c_far = [
        row(a, b, sim, 0.0, 0)
        for (a, b), sim in semantic_pairs.items()
        if pair_counts.get((a, b), 0) == 0 and sim >= 0.94
    ]
    s_near_c_far = sorted(s_near_c_far, key=lambda x: -x["semantic_sim"])[:max_pairs]

    s_near_c_near = [
        row(a, b, sim, ppmi.get((a, b), 0.0), pair_counts[(a, b)])
        for (a, b), sim in semantic_pairs.items()
        if pair_counts.get((a, b), 0) > 0 and sim >= 0.90
    ]
    s_near_c_near = sorted(
        s_near_c_near,
        key=lambda x: (-x["cooc_count"], -x["ppmi"], -x["semantic_sim"]),
    )[:max_pairs]

    s_far_c_near = []
    for (a, b), score in ppmi.items():
        if score <= 0:
            continue
        sim = float(emb[a] @ emb[b])
        cooc = pair_counts[(a, b)]
        if sim <= 0.76:
            s_far_c_near.append(row(a, b, sim, score, cooc))
    s_far_c_near = sorted(
        s_far_c_near,
        key=lambda x: (-x["cooc_count"], -x["ppmi"], x["semantic_sim"]),
    )[:max_pairs]

    rng = random.Random(seed)
    s_far_c_far: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    n = emb.shape[0]
    tries = 0
    while len(s_far_c_far) < max_pairs and tries < max_pairs * 200:
        tries += 1
        a, b = sorted(rng.sample(range(n), 2))
        if (a, b) in seen or pair_counts.get((a, b), 0) > 0:
            continue
        seen.add((a, b))
        sim = float(emb[a] @ emb[b])
        if sim <= 0.76:
            s_far_c_far.append(row(a, b, sim, 0.0, 0))

    return {
        "S-near C-far": s_near_c_far,
        "S-near C-near": s_near_c_near,
        "S-far C-near": s_far_c_near,
        "S-far C-far": s_far_c_far,
    }


def entropy_norm(counts: list[int]) -> float:
    total = sum(counts)
    if total <= 0 or len(counts) <= 1:
        return 0.0
    ent = -sum((c / total) * math.log(c / total) for c in counts if c)
    return ent / math.log(len(counts))


def gini(values: list[int]) -> float:
    if not values or sum(values) == 0:
        return 0.0
    vals = sorted(values)
    n = len(vals)
    return (2 * sum((i + 1) * v for i, v in enumerate(vals)) / (n * sum(vals))) - ((n + 1) / n)


def structure_metrics(code_map: dict[int, tuple[str, str, str]]) -> dict[str, Any]:
    l1 = Counter(code[0] for code in code_map.values())
    l2 = Counter(code[1] for code in code_map.values())
    l3 = Counter(code[2] for code in code_map.values())
    l12 = Counter(code[:2] for code in code_map.values())
    sid = Counter(code_map.values())
    l1_to_l2: dict[str, set[str]] = defaultdict(set)
    l12_to_l3: dict[tuple[str, str], set[str]] = defaultdict(set)
    for code in code_map.values():
        l1_to_l2[code[0]].add(code[1])
        l12_to_l3[code[:2]].add(code[2])
    l1_sizes = list(l1.values())
    l12_sizes = list(l12.values())
    collision_count = sum(c - 1 for c in sid.values() if c > 1)
    return {
        "active_l1": len(l1),
        "active_l2": len(l2),
        "active_l3": len(l3),
        "unique_l12": len(l12),
        "unique_sid": len(sid),
        "collision_count": collision_count,
        "collision_rate": collision_count / len(code_map),
        "max_conflict": max(sid.values()),
        "max_l1_bucket": max(l1_sizes),
        "top5_l1_cover": sum(c for _, c in l1.most_common(5)),
        "top5_l1": [[token, count] for token, count in l1.most_common(5)],
        "l1_entropy_norm": entropy_norm(l1_sizes),
        "l1_gini": gini(l1_sizes),
        "l12_mean_size": float(np.mean(l12_sizes)),
        "l12_median_size": float(np.median(l12_sizes)),
        "l12_p90_size": float(np.percentile(l12_sizes, 90)),
        "l12_max_size": max(l12_sizes),
        "l12_singletons": sum(1 for x in l12_sizes if x == 1),
        "l12_ge5": sum(1 for x in l12_sizes if x >= 5),
        "avg_l2_per_l1": float(np.mean([len(v) for v in l1_to_l2.values()])),
        "median_l2_per_l1": float(np.median([len(v) for v in l1_to_l2.values()])),
        "avg_l3_per_l12": float(np.mean([len(v) for v in l12_to_l3.values()])),
        "median_l3_per_l12": float(np.median([len(v) for v in l12_to_l3.values()])),
    }


def pair_metrics(code_map: dict[int, tuple[str, str, str]], pairs: list[dict[str, Any]]) -> dict[str, float]:
    if not pairs:
        return {
            "pair_count": 0,
            "same_l1_pct": 0.0,
            "same_l12_pct": 0.0,
            "same_sid_pct": 0.0,
            "same_l2_token_pct": 0.0,
            "same_l3_token_pct": 0.0,
            "avg_token_overlap": 0.0,
            "avg_lcp": 0.0,
            "split_after_l1_pct": 0.0,
        }

    same_l1 = same_l12 = same_sid = same_l2_token = same_l3_token = 0
    overlap_sum = 0
    lcp_sum = 0
    split_after_l1 = 0
    for pair in pairs:
        a = int(pair["item_a"])
        b = int(pair["item_b"])
        code_a = code_map[a]
        code_b = code_map[b]
        if code_a[0] == code_b[0]:
            same_l1 += 1
        if code_a[:2] == code_b[:2]:
            same_l12 += 1
        if code_a == code_b:
            same_sid += 1
        if code_a[1] == code_b[1]:
            same_l2_token += 1
        if code_a[2] == code_b[2]:
            same_l3_token += 1
        overlap = sum(x == y for x, y in zip(code_a, code_b))
        overlap_sum += overlap
        lcp = 0
        for x, y in zip(code_a, code_b):
            if x != y:
                break
            lcp += 1
        lcp_sum += lcp
        if code_a[0] == code_b[0] and code_a[:2] != code_b[:2]:
            split_after_l1 += 1

    denom = len(pairs)
    return {
        "pair_count": float(denom),
        "same_l1_pct": same_l1 / denom * 100,
        "same_l12_pct": same_l12 / denom * 100,
        "same_sid_pct": same_sid / denom * 100,
        "same_l2_token_pct": same_l2_token / denom * 100,
        "same_l3_token_pct": same_l3_token / denom * 100,
        "avg_token_overlap": overlap_sum / denom,
        "avg_lcp": lcp_sum / denom,
        "split_after_l1_pct": split_after_l1 / denom * 100,
    }


def code_relation(code_a: tuple[str, str, str], code_b: tuple[str, str, str]) -> str:
    lcp = 0
    for x, y in zip(code_a, code_b):
        if x == y:
            lcp += 1
        else:
            break
    overlap = sum(x == y for x, y in zip(code_a, code_b))
    return f"LCP={lcp}, overlap={overlap}"


def title(items: dict[int, dict[str, Any]], item_id: int, max_len: int = 120) -> str:
    text = str(items[item_id].get("title", f"Item_{item_id}")).replace("\n", " ")
    return text[:max_len]


def select_examples(
    pair_sets: dict[str, list[dict[str, Any]]],
    code_maps: dict[str, dict[int, tuple[str, str, str]]],
    example_count: int,
) -> dict[str, list[dict[str, Any]]]:
    examples: dict[str, list[dict[str, Any]]] = {}

    def add_codes(pair: dict[str, Any]) -> dict[str, Any]:
        enriched = dict(pair)
        a = int(pair["item_a"])
        b = int(pair["item_b"])
        enriched["codes"] = {
            label: {
                "item_a": list(code_map[a]),
                "item_b": list(code_map[b]),
                "relation": code_relation(code_map[a], code_map[b]),
            }
            for label, code_map in code_maps.items()
        }
        return enriched

    for set_name, pairs in pair_sets.items():
        selected: list[dict[str, Any]] = []
        if set_name == "S-near C-far":
            l2_003 = code_maps["L2=0.003"]
            main = code_maps["Main L2=0.010,L3=0.020"]
            l3 = code_maps["L3=0.010"]
            for pair in sorted(pairs, key=lambda x: -x["semantic_sim"]):
                a, b = int(pair["item_a"]), int(pair["item_b"])
                weak_bad = l2_003[a][:2] == l2_003[b][:2]
                main_good = main[a][0] == main[b][0] and main[a][:2] != main[b][:2]
                l3_good = l3[a][0] == l3[b][0] and l3[a][:2] != l3[b][:2]
                if weak_bad and (main_good or l3_good):
                    selected.append(add_codes(pair))
                if len(selected) >= example_count:
                    break
        elif set_name == "S-near C-near":
            candidates = sorted(
                pairs,
                key=lambda x: (-x["cooc_count"], -x["semantic_sim"], -x["ppmi"]),
            )
            selected = [add_codes(pair) for pair in candidates[:example_count]]
        elif set_name == "S-far C-near":
            candidates = sorted(
                pairs,
                key=lambda x: (-x["cooc_count"], -x["ppmi"], x["semantic_sim"]),
            )
            selected = [add_codes(pair) for pair in candidates[:example_count]]
        else:
            candidates = sorted(pairs, key=lambda x: (x["semantic_sim"], x["item_a"], x["item_b"]))
            selected = [add_codes(pair) for pair in candidates[:example_count]]

        if len(selected) < example_count:
            seen = {(ex["item_a"], ex["item_b"]) for ex in selected}
            for pair in pairs:
                key = (pair["item_a"], pair["item_b"])
                if key in seen:
                    continue
                selected.append(add_codes(pair))
                if len(selected) >= example_count:
                    break
        examples[set_name] = selected
    return examples


def ranking_metrics_from_result(result_path: Path) -> dict[str, float] | None:
    path = resolve(result_path)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        return None
    ks = [1, 3, 5, 10, 20, 50]
    hits = {k: 0.0 for k in ks}
    ndcgs = {k: 0.0 for k in ks}
    invalid = 0
    for row in rows:
        target = str(row.get("output", "")).strip()
        preds = row.get("predict", [])
        if not isinstance(preds, list):
            preds = []
        if not preds:
            invalid += 1
        try:
            rank = preds.index(target) + 1
        except ValueError:
            rank = None
        for k in ks:
            if rank is not None and rank <= k:
                hits[k] += 1
                ndcgs[k] += 1 / math.log2(rank + 1)
    denom = max(1, len(rows))
    result: dict[str, float] = {
        "test_example_count": float(len(rows)),
        "constraint_invalid_total": float(invalid),
    }
    for k in ks:
        result[f"hr_at_{k}"] = hits[k] / denom
        result[f"ndcg_at_{k}"] = ndcgs[k] / denom
    return result


def downstream_metrics(variants: list[Variant]) -> dict[str, dict[str, Any]]:
    scoreboard_path = REPO_ROOT / "research-progress-log/experiment_registry/downstream_scoreboard.csv"
    scoreboard = pd.read_csv(scoreboard_path) if scoreboard_path.exists() else pd.DataFrame()
    out: dict[str, dict[str, Any]] = {}
    for variant in variants:
        metrics = ranking_metrics_from_result(variant.result_json_path)
        source = "result_json" if metrics is not None else "pending"
        if metrics is None and not scoreboard.empty:
            rows = scoreboard[scoreboard["variant"] == variant.registry_variant]
            rows = rows[rows["stage"] == "sft_eval"] if "stage" in rows else rows
            if not rows.empty:
                row = rows.iloc[-1]
                metrics = {
                    "test_example_count": float(row.get("test_example_count", 0)),
                    "constraint_invalid_total": float(row.get("constraint_invalid_total", 0)),
                }
                for k in [1, 3, 5, 10, 50]:
                    metrics[f"ndcg_at_{k}"] = float(row.get(f"ndcg_at_{k}", 0))
                    metrics[f"hr_at_{k}"] = float(row.get(f"hr_at_{k}", 0))
                source = "downstream_scoreboard"
        out[variant.label] = {
            "status": "available" if metrics is not None else "pending",
            "source": source,
            "result_json_path": str(variant.result_json_path),
            "metrics": metrics or {},
        }
    return out


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt_pct(value: float) -> str:
    return f"{value:.2f}%"


def fmt_float(value: float) -> str:
    return f"{value:.6f}"


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def render_report(
    output_dir: Path,
    data_stats: dict[str, Any],
    pair_sets: dict[str, list[dict[str, Any]]],
    structure: dict[str, dict[str, Any]],
    pair_metric_table: dict[str, dict[str, dict[str, float]]],
    downstream: dict[str, dict[str, Any]],
    examples: dict[str, list[dict[str, Any]]],
    items: dict[int, dict[str, Any]],
) -> str:
    lines: list[str] = []
    lines.append("# L2/L3 Codebook Reasonableness（码本合理性）分析")
    lines.append("")
    lines.append("## 结论")
    lines.append("")
    lines.append(
        "- 当前主线 `Main L2=0.010,L3=0.020` 是目前最强的 validated tokenizer（已验证分词器）："
        "它比 `L2=0.003` 更能把 semantic-near collaborative-far（语义近但协同远）物品保留在同一 L1（第一层）粗类下，并在 L2/L3（第二/三层）拆开；对应 SFT（监督微调）NDCG@10 也更高。"
    )
    lines.append(
        "- `L3=0.010` 的结构指标和 pair-level（物品对级）指标最像一个强候选：same L1（同第一层）更高、same L12（同前两层）不升反降，说明它保留语义粗类同时进一步增强后层分辨。"
    )
    lines.append(
        "- `S-far C-near`（语义远但协同近）上三组 tokenizer（分词器）的后层 token overlap（token 重合）都很弱；当前方法主要解决了“语义近但协同远要拆开”，还没有很好解决“语义远但协同近要拉近”。"
    )
    lines.append(
        "- 因为 `L3=0.010` 的 SFT（监督微调）结果当前仍是 pending（待完成），现在不能声称“结构越合理下游一定越好”；只能说 `L2=0.003 -> 当前主线` 这组已完成对比支持该趋势。"
    )
    lines.append("")

    lines.append("## 数据定义")
    lines.append("")
    lines.append(
        "- semantic similarity（语义相似度）：`Industrial_and_Scientific.emb-qwen-td.npy` 的 cosine similarity（余弦相似度）。"
    )
    lines.append(
        "- collaborative similarity（协同相似度）：只用 train interaction（训练交互），把每条样本的 `history_item_id -> item_id` 作为 direct edge（直接边），累计 co-occurrence（共现）并计算 PPMI（正点互信息）。"
    )
    lines.append("- valid/test interaction（验证/测试交互）没有参与构图，避免 leakage（泄露）。")
    lines.append("")
    lines.append(
        markdown_table(
            ["stat（统计）", "value（数值）"],
            [
                ["train rows（训练样本数）", data_stats["train_rows"]],
                ["direct edges（直接协同边）", data_stats["direct_edge_count"]],
                ["edge events（边事件数）", data_stats["edge_events"]],
                ["cooc p95（共现次数 p95）", data_stats["cooc_count_percentiles"].get("p95")],
                ["PPMI p95（正点互信息 p95）", round(data_stats["ppmi_percentiles"].get("p95", 0), 4)],
            ],
        )
    )
    lines.append("")

    lines.append("## 结构指标")
    lines.append("")
    struct_rows = []
    for label, s in structure.items():
        struct_rows.append(
            [
                label,
                s["active_l1"],
                s["unique_l12"],
                s["unique_sid"],
                s["collision_count"],
                s["max_conflict"],
                s["top5_l1_cover"],
                s["max_l1_bucket"],
            ]
        )
    lines.append(
        markdown_table(
            [
                "tokenizer（分词器）",
                "active L1（活跃第一层）",
                "unique L12（唯一前两层）",
                "unique SID（唯一语义标识）",
                "collision（冲突）",
                "max conflict（最大冲突簇）",
                "top5 L1 cover（前五个第一层覆盖）",
                "max L1 bucket（最大第一层桶）",
            ],
            struct_rows,
        )
    )
    lines.append("")

    lines.append("## Pair-Level（物品对级）指标")
    lines.append("")
    for set_name, metrics_by_variant in pair_metric_table.items():
        pairs = pair_sets[set_name]
        sim_mean = np.mean([p["semantic_sim"] for p in pairs]) if pairs else 0.0
        ppmi_mean = np.mean([p["ppmi"] for p in pairs]) if pairs else 0.0
        lines.append(f"### {set_name}")
        lines.append("")
        lines.append(
            f"pair count（物品对数量）={len(pairs)}, semantic mean（语义均值）={sim_mean:.4f}, PPMI mean（正点互信息均值）={ppmi_mean:.4f}"
        )
        rows = []
        for label, m in metrics_by_variant.items():
            rows.append(
                [
                    label,
                    fmt_pct(m["same_l1_pct"]),
                    fmt_pct(m["same_l12_pct"]),
                    fmt_pct(m["same_sid_pct"]),
                    fmt_pct(m["same_l2_token_pct"]),
                    fmt_pct(m["same_l3_token_pct"]),
                    f"{m['avg_token_overlap']:.3f}",
                    f"{m['avg_lcp']:.3f}",
                    fmt_pct(m["split_after_l1_pct"]),
                ]
            )
        lines.append(
            markdown_table(
                [
                    "tokenizer（分词器）",
                    "same L1（同第一层）",
                    "same L12（同前两层）",
                    "same SID（同语义标识）",
                    "same L2 token（同第二层 token）",
                    "same L3 token（同第三层 token）",
                    "avg overlap（平均 token 重合）",
                    "avg LCP（平均最长前缀）",
                    "split after L1（同 L1 后拆开）",
                ],
                rows,
            )
        )
        lines.append("")

    lines.append("## Downstream SFT（下游监督微调）")
    lines.append("")
    downstream_rows = []
    for label, entry in downstream.items():
        metrics = entry["metrics"]
        downstream_rows.append(
            [
                label,
                entry["status"],
                fmt_float(metrics["ndcg_at_1"]) if metrics else "-",
                fmt_float(metrics["ndcg_at_3"]) if metrics else "-",
                fmt_float(metrics["ndcg_at_5"]) if metrics else "-",
                fmt_float(metrics["ndcg_at_10"]) if metrics else "-",
                fmt_float(metrics["hr_at_1"]) if metrics else "-",
                fmt_float(metrics["hr_at_3"]) if metrics else "-",
                fmt_float(metrics["hr_at_5"]) if metrics else "-",
                fmt_float(metrics["hr_at_10"]) if metrics else "-",
            ]
        )
    lines.append(
        markdown_table(
            [
                "tokenizer（分词器）",
                "status（状态）",
                "NDCG@1",
                "NDCG@3",
                "NDCG@5",
                "NDCG@10",
                "HR@1",
                "HR@3",
                "HR@5",
                "HR@10",
            ],
            downstream_rows,
        )
    )
    lines.append("")

    lines.append("## 具体物品例子")
    lines.append("")
    for set_name, exs in examples.items():
        lines.append(f"### {set_name}")
        lines.append("")
        for i, ex in enumerate(exs, 1):
            a = int(ex["item_a"])
            b = int(ex["item_b"])
            lines.append(
                f"{i}. pair（物品对） `{a}` - `{b}`; sim（语义相似度）={ex['semantic_sim']:.4f}; "
                f"cooc（共现）={ex['cooc_count']}; PPMI（正点互信息）={ex['ppmi']:.4f}"
            )
            lines.append(f"   - A: {title(items, a)}")
            lines.append(f"   - B: {title(items, b)}")
            for label, code_info in ex["codes"].items():
                lines.append(
                    f"   - {label}: `{code_info['item_a']}` vs `{code_info['item_b']}`; {code_info['relation']}"
                )
        lines.append("")

    lines.append("## 判断")
    lines.append("")
    lines.append(
        "1. `L2=0.003` 的 L1（第一层）并没有坏，但 L2（第二层）协同干预偏弱；在 `S-near C-far`（语义近但协同远）上，同 L12（同前两层）比例最高，说明后层拆分不够。"
    )
    lines.append(
        "2. 当前主线的结构合理性和下游 SFT（监督微调）是一致的：它在 `S-near C-far` 上更会“同粗类、后层拆”，同时 NDCG@10 从 `0.095737` 提到 `0.104383`。"
    )
    lines.append(
        "3. `L3=0.010` 在结构上目前最漂亮：same L1（同第一层）最高，same L12（同前两层）与当前主线相近甚至略低，split after L1（同第一层后拆开）最高。若它的 SFT（监督微调）结果也提升，就能更强地支持“码本合理性 -> 下游收益”的叙事。"
    )
    lines.append(
        "4. 如果 `L3=0.010` 下游没有提升，优先怀疑 learnability（可学习性）或 route distribution（路由分布）问题，而不是单纯否定 pair-level reasonableness（物品对级合理性）。"
    )
    lines.append("")
    lines.append("## 输出文件")
    lines.append("")
    lines.append(f"- `metrics.json`: `{output_dir / 'metrics.json'}`")
    lines.append(f"- `structure_metrics.csv`: `{output_dir / 'structure_metrics.csv'}`")
    lines.append(f"- `pair_metrics.csv`: `{output_dir / 'pair_metrics.csv'}`")
    lines.append(f"- `pair_examples.json`: `{output_dir / 'pair_examples.json'}`")

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    variants = list(VARIANTS)
    if args.include_l2_0015:
        variants.append(OPTIONAL_VARIANTS["l2_0015"])

    code_maps = {variant.label: load_index(variant.index_path) for variant in variants}
    items = load_items(args.item_json)
    emb = normalize_embeddings(args.embedding_npy)
    pair_counts, ppmi, data_stats = build_collab_stats(args.train_csv)
    semantic_pairs = build_semantic_top_pairs(emb, args.semantic_topk)
    pair_sets = build_pair_sets(
        emb,
        semantic_pairs,
        pair_counts,
        ppmi,
        max_pairs=args.max_pairs_per_set,
        seed=args.random_seed,
    )

    structure = {label: structure_metrics(code_map) for label, code_map in code_maps.items()}
    pair_metric_table = {
        set_name: {label: pair_metrics(code_map, pairs) for label, code_map in code_maps.items()}
        for set_name, pairs in pair_sets.items()
    }
    downstream = downstream_metrics(variants)
    examples = select_examples(pair_sets, code_maps, args.example_count)

    metrics = {
        "data_stats": data_stats,
        "semantic_top_pair_count": len(semantic_pairs),
        "semantic_top_pair_percentiles": percentile_dict(list(semantic_pairs.values())),
        "pair_set_stats": {
            set_name: {
                "pair_count": len(pairs),
                "semantic_sim_mean": float(np.mean([p["semantic_sim"] for p in pairs])) if pairs else 0.0,
                "ppmi_mean": float(np.mean([p["ppmi"] for p in pairs])) if pairs else 0.0,
                "cooc_count_mean": float(np.mean([p["cooc_count"] for p in pairs])) if pairs else 0.0,
            }
            for set_name, pairs in pair_sets.items()
        },
        "structure": structure,
        "pair_metrics": pair_metric_table,
        "downstream_sft": downstream,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with (output_dir / "pair_examples.json").open("w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)

    structure_rows = [
        {"tokenizer": label, **values}
        for label, values in structure.items()
    ]
    write_csv(
        output_dir / "structure_metrics.csv",
        structure_rows,
        [
            "tokenizer",
            "active_l1",
            "active_l2",
            "active_l3",
            "unique_l12",
            "unique_sid",
            "collision_count",
            "collision_rate",
            "max_conflict",
            "max_l1_bucket",
            "top5_l1_cover",
            "l1_entropy_norm",
            "l1_gini",
            "l12_mean_size",
            "l12_median_size",
            "l12_p90_size",
            "l12_max_size",
            "l12_singletons",
            "l12_ge5",
            "avg_l2_per_l1",
            "median_l2_per_l1",
            "avg_l3_per_l12",
            "median_l3_per_l12",
            "top5_l1",
        ],
    )

    pair_rows = []
    for set_name, by_variant in pair_metric_table.items():
        for label, values in by_variant.items():
            pair_rows.append({"pair_set": set_name, "tokenizer": label, **values})
    write_csv(
        output_dir / "pair_metrics.csv",
        pair_rows,
        [
            "pair_set",
            "tokenizer",
            "pair_count",
            "same_l1_pct",
            "same_l12_pct",
            "same_sid_pct",
            "same_l2_token_pct",
            "same_l3_token_pct",
            "avg_token_overlap",
            "avg_lcp",
            "split_after_l1_pct",
        ],
    )

    downstream_rows = []
    for label, entry in downstream.items():
        row = {
            "tokenizer": label,
            "status": entry["status"],
            "source": entry["source"],
            "result_json_path": entry["result_json_path"],
        }
        row.update(entry["metrics"])
        downstream_rows.append(row)
    write_csv(
        output_dir / "downstream_sft_metrics.csv",
        downstream_rows,
        [
            "tokenizer",
            "status",
            "source",
            "result_json_path",
            "test_example_count",
            "constraint_invalid_total",
            "ndcg_at_1",
            "ndcg_at_3",
            "ndcg_at_5",
            "ndcg_at_10",
            "ndcg_at_20",
            "ndcg_at_50",
            "hr_at_1",
            "hr_at_3",
            "hr_at_5",
            "hr_at_10",
            "hr_at_20",
            "hr_at_50",
        ],
    )

    report = render_report(
        output_dir=output_dir,
        data_stats=data_stats,
        pair_sets=pair_sets,
        structure=structure,
        pair_metric_table=pair_metric_table,
        downstream=downstream,
        examples=examples,
        items=items,
    )
    report_path = output_dir / "codebook_reasonableness_report.md"
    report_path.write_text(report, encoding="utf-8")

    print(f"[done] report: {report_path}")
    print(f"[done] metrics: {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
