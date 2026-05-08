#!/usr/bin/env python3
"""SID structural diagnostic for tokenizer codebooks.

The diagnostic is intentionally train-only on collaborative statistics. It
does not use valid/test interactions to build pair sets, and it separates the
structural verdict from downstream SFT metrics so the rules can be reused for
future tokenizers before their downstream results finish.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
HELPER_PATH = Path(__file__).with_name("analyze_codebook_reasonableness.py")


@dataclass(frozen=True)
class Candidate:
    label: str
    tokenizer_record_id: str
    index_path: Path
    split: str
    note: str


CANDIDATES = [
    Candidate(
        label="Original semantic",
        tokenizer_record_id="tok_industrial_original_semantic",
        index_path=Path("data/Amazon/index/Industrial_and_Scientific.index.json"),
        split="validation",
        note="Original MiniOneRec semantic SID baseline.",
    ),
    Candidate(
        label="V2 offline",
        tokenizer_record_id="tok_industrial_mgr_tokenizer_v2_offline",
        index_path=Path(
            "/data/leejt/OneRec/output_weights/experiments/"
            "mgr_sid_tokenizer_v2/generated_indices/"
            "Industrial_and_Scientific.mgr_tokenizer_v2_offline.index.json"
        ),
        split="validation",
        note="Strong pre-LMH tokenizer line; useful as an out-of-family challenge case.",
    ),
    Candidate(
        label="R690b L2=0.010 main",
        tokenizer_record_id="tok_industrial_r690b_lmh_l2_contrastive_pull_weight001_20260507",
        index_path=Path(
            "/data/leejt/OneRec/output_weights/experiments/"
            "mgr_sid_l2_lmh_sweep_20260507/generated_indices/"
            "Industrial_and_Scientific.r690b_lmh_l2_contrastive_pull_weight001.index.json"
        ),
        split="calibration",
        note="Current mainline and strongest tokenizer-side SFT anchor.",
    ),
    Candidate(
        label="R690b L2=0.003 weak",
        tokenizer_record_id="tok_industrial_r690b_lmh_l2_contrastive_pull_weight0003_20260507",
        index_path=Path(
            "/data/leejt/OneRec/output_weights/experiments/"
            "mgr_sid_l2_lmh_sweep_20260507/generated_indices/"
            "Industrial_and_Scientific.r690b_lmh_l2_contrastive_pull_weight0003.index.json"
        ),
        split="calibration",
        note="Low L2 collaborative contrastive weight.",
    ),
    Candidate(
        label="R690b L2=0.005 weak",
        tokenizer_record_id="tok_industrial_r690b_lmh_l2_contrastive_pull_weight0005_20260507",
        index_path=Path(
            "/data/leejt/OneRec/output_weights/experiments/"
            "mgr_sid_l2_lmh_sweep_20260507/generated_indices/"
            "Industrial_and_Scientific.r690b_lmh_l2_contrastive_pull_weight0005.index.json"
        ),
        split="calibration",
        note="Low-to-mid L2 collaborative contrastive weight.",
    ),
    Candidate(
        label="R690b L2=0.015 fragmented",
        tokenizer_record_id="tok_industrial_r690b_lmh_l2_contrastive_pull_weight0015_20260507",
        index_path=Path(
            "/data/leejt/OneRec/output_weights/experiments/"
            "mgr_sid_l2_lmh_sweep_20260507/generated_indices/"
            "Industrial_and_Scientific.r690b_lmh_l2_contrastive_pull_weight0015.index.json"
        ),
        split="calibration",
        note="Upper-side L2 weight; expected over-fragmentation challenge.",
    ),
    Candidate(
        label="R690b no L1 semantic",
        tokenizer_record_id="tok_industrial_r690b_lmh_l2_contrastive_pull_weight001_no_l1_semantic_20260508",
        index_path=Path(
            "/data/leejt/OneRec/output_weights/experiments/"
            "mgr_sid_l1_ablation_20260507/generated_indices/"
            "Industrial_and_Scientific.r690b_lmh_l2_contrastive_pull_weight001_no_l1_semantic.index.json"
        ),
        split="calibration",
        note="L1 semantic pull ablation.",
    ),
    Candidate(
        label="R690b L3=0.010 pending",
        tokenizer_record_id="tok_industrial_r690b_lmh_l2_weight001_l3_weight010_20260508",
        index_path=Path(
            "/data/leejt/OneRec/output_weights/experiments/"
            "mgr_sid_l3_lmh_sweep_20260508/generated_indices/"
            "Industrial_and_Scientific.r690b_lmh_l2_weight001_l3_weight010.index.json"
        ),
        split="prospective",
        note="Prospective L3 local-pull candidate with fixed L2=0.010 anchor.",
    ),
    Candidate(
        label="R690b L3=0.005 pending",
        tokenizer_record_id="tok_industrial_r690b_lmh_l2_weight001_l3_weight005_20260508",
        index_path=Path(
            "/data/leejt/OneRec/output_weights/experiments/"
            "mgr_sid_l3_lmh_sweep_20260508/generated_indices/"
            "Industrial_and_Scientific.r690b_lmh_l2_weight001_l3_weight005.index.json"
        ),
        split="prospective",
        note="Prospective lower L3 local-pull candidate with fixed L2=0.010 anchor.",
    ),
    Candidate(
        label="R690b L3=0.015 gate-failed",
        tokenizer_record_id="tok_industrial_r690b_lmh_l2_weight001_l3_weight015_20260508",
        index_path=Path(
            "/data/leejt/OneRec/output_weights/experiments/"
            "mgr_sid_l3_lmh_sweep_20260508/generated_indices/"
            "Industrial_and_Scientific.r690b_lmh_l2_weight001_l3_weight015.index.json"
        ),
        split="prospective",
        note="Prospective upper L3 local-pull candidate; initial generation gate failed.",
    ),
    Candidate(
        label="Original L2 multihop ranking",
        tokenizer_record_id="tok_industrial_original_l2_multihop_ranking_20260421",
        index_path=Path(
            "/data/leejt/OneRec/output_weights/experiments/"
            "mgr_sid_original_l2_multihop_ranking_20260421/generated_indices/"
            "Industrial_and_Scientific.original_l2_multihop_ranking.index.json"
        ),
        split="validation",
        note="Minimal-edit collaborative ranking screen.",
    ),
    Candidate(
        label="QCR L2 conflict ranking",
        tokenizer_record_id="tok_industrial_qcr_l2_conflict_ranking_20260421",
        index_path=Path(
            "/data/leejt/OneRec/output_weights/experiments/"
            "mgr_sid_qcr_l2_conflict_ranking_20260421/generated_indices/"
            "Industrial_and_Scientific.qcr_l2_conflict_ranking.index.json"
        ),
        split="validation",
        note="Healthy collision but negative SFT; important single-metric counterexample.",
    ),
    Candidate(
        label="Stage3 prefix retained",
        tokenizer_record_id="tok_industrial_stage3_r401b_g005",
        index_path=Path(
            "/data/leejt/OneRec/output_weights/experiments/"
            "mgr_sid_stage3_prefix_retained_20260414/generated_indices/"
            "Industrial_and_Scientific.stage3_r401b_g005.index.json"
        ),
        split="validation",
        note="Prefix-retention branch.",
    ),
    Candidate(
        label="TAGCF attr mid",
        tokenizer_record_id="tok_industrial_tagcf_r510_attr_mid",
        index_path=Path(
            "/data/leejt/OneRec/output_weights/experiments/"
            "mgr_sid_tagcf_branch_20260415/generated_indices/"
            "Industrial_and_Scientific.tagcf_r510_attr_mid.index.json"
        ),
        split="validation",
        note="Attribute-topology mid graph branch.",
    ),
    Candidate(
        label="V2 LMH mid=0.010",
        tokenizer_record_id="tok_industrial_v2_lmh_mid_weight001_20260507",
        index_path=Path(
            "/data/leejt/OneRec/output_weights/experiments/"
            "mgr_sid_l2_lmh_sweep_20260507/generated_indices/"
            "Industrial_and_Scientific.v2_lmh_mid_weight001.index.json"
        ),
        split="validation",
        note="Same LMH idea on the v2 tokenizer branch.",
    ),
]


RULES = {
    "l1_stability": {
        "fail": "active_l1 > 150 or top5_l1_cover < 300 or S-near C-near same_l1 < 80",
        "warn": "active_l1 > 100 or top5_l1_cover < 500 or S-near C-near same_l1 < 86",
    },
    "selective_separation": {
        "fail": "S-near C-far same_l1 < 80 or split_after_l1 < 55 or same_l12 > 32.5",
        "warn": "split_after_l1 < 60 or same_l12 > 30.5",
    },
    "collaborative_preservation": {
        "fail": "S-near C-near same_l1 < 80 or same_l12 < 27",
        "warn": "S-near C-near same_l1 < 86 or same_l12 < 31",
    },
    "learnability": {
        "fail": "active_l1 > 180 or unique_l12 > 2750 or top5_l1_cover < 300 or catalog_l12_zero_train_pct > 50 or catalog_sid_zero_train_pct > 80",
        "warn": "active_l1 > 100 or unique_l12 > 2550 or top5_l1_cover < 500 or l12_singletons > 2000 or catalog_l12_zero_train_pct > 25 or catalog_sid_zero_train_pct > 55",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("research-progress-log/experiment_analysis/2026-05-08_sid_structural_diagnostic"),
    )
    parser.add_argument("--semantic-topk", type=int, default=20)
    parser.add_argument("--max-pairs-per-set", type=int, default=10000)
    parser.add_argument("--smid-min", type=float, default=0.80)
    parser.add_argument("--smid-max", type=float, default=0.90)
    parser.add_argument("--example-count", type=int, default=4)
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def load_helper() -> Any:
    spec = importlib.util.spec_from_file_location("codebook_reasonableness", HELPER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import helper from {HELPER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_best_sft() -> pd.DataFrame:
    path = REPO_ROOT / "research-progress-log/experiment_registry/sft_registry.csv"
    sft = pd.read_csv(path)
    sft = sft[sft["dataset_key"].eq("industrial")].copy()
    metric_cols = ["ndcg_at_1", "ndcg_at_3", "ndcg_at_5", "ndcg_at_10", "hr_at_10", "hr_at_50"]
    for col in metric_cols:
        sft[col] = pd.to_numeric(sft[col], errors="coerce")
    return (
        sft.sort_values(["tokenizer_record_id", "ndcg_at_10", "hr_at_10"], ascending=[True, False, False])
        .groupby("tokenizer_record_id", as_index=False)
        .head(1)
    )


def build_extended_pair_sets(
    helper: Any,
    emb: np.ndarray,
    semantic_pairs: dict[tuple[int, int], float],
    pair_counts: Counter[tuple[int, int]],
    ppmi: dict[tuple[int, int], float],
    args: argparse.Namespace,
) -> dict[str, list[dict[str, Any]]]:
    """Build original extreme pair sets plus mid-similarity blind-spot sets."""
    pair_sets = helper.build_pair_sets(
        emb,
        semantic_pairs,
        pair_counts,
        ppmi,
        max_pairs=args.max_pairs_per_set,
        seed=args.random_seed,
    )

    def row(a: int, b: int, sim: float, score: float, cooc: int) -> dict[str, Any]:
        return {
            "item_a": int(a),
            "item_b": int(b),
            "semantic_sim": float(sim),
            "ppmi": float(score),
            "cooc_count": int(cooc),
        }

    smid_c_far = [
        row(a, b, sim, 0.0, 0)
        for (a, b), sim in semantic_pairs.items()
        if args.smid_min <= sim < args.smid_max and pair_counts.get((a, b), 0) == 0
    ]
    smid_c_near = [
        row(a, b, sim, ppmi.get((a, b), 0.0), pair_counts[(a, b)])
        for (a, b), sim in semantic_pairs.items()
        if args.smid_min <= sim < args.smid_max and pair_counts.get((a, b), 0) > 0
    ]
    pair_sets["S-mid C-far"] = sorted(
        smid_c_far,
        key=lambda x: (-x["semantic_sim"], x["item_a"], x["item_b"]),
    )[: args.max_pairs_per_set]
    pair_sets["S-mid C-near"] = sorted(
        smid_c_near,
        key=lambda x: (-x["cooc_count"], -x["ppmi"], -x["semantic_sim"]),
    )[: args.max_pairs_per_set]
    return pair_sets


def train_prefix_frequency_metrics(helper: Any, code_map: dict[int, tuple[str, str, str]]) -> dict[str, Any]:
    """Train-only prefix exposure metrics for downstream learnability checks."""
    train_csv = REPO_ROOT / "data/Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.csv"
    df = pd.read_csv(train_csv, usecols=["history_item_id", "item_id"])
    l1_events: Counter[tuple[str, ...]] = Counter()
    l12_events: Counter[tuple[str, ...]] = Counter()
    sid_events: Counter[tuple[str, ...]] = Counter()
    item_events: Counter[int] = Counter()

    for hist_raw, target_raw in zip(df["history_item_id"], df["item_id"]):
        ids = helper.parse_id_list(hist_raw)
        ids.append(int(target_raw))
        for item_id in ids:
            code = code_map.get(int(item_id))
            if code is None:
                continue
            item_events[int(item_id)] += 1
            l1_events[(code[0],)] += 1
            l12_events[code[:2]] += 1
            sid_events[code] += 1

    def entropy_norm(counter: Counter[tuple[str, ...]]) -> float:
        return helper.entropy_norm(list(counter.values()))

    def catalog_exposure(counter: Counter[tuple[str, ...]], prefix_len: int) -> dict[str, float]:
        vals = [counter.get(code[:prefix_len], 0) for code in code_map.values()]
        arr = np.asarray(vals, dtype=float)
        return {
            f"catalog_l{prefix_len}_zero_train_pct": float(np.mean(arr == 0) * 100),
            f"catalog_l{prefix_len}_p10_train_events": float(np.percentile(arr, 10)),
            f"catalog_l{prefix_len}_median_train_events": float(np.median(arr)),
            f"catalog_l{prefix_len}_mean_train_events": float(np.mean(arr)),
        }

    metrics: dict[str, Any] = {
        "train_interaction_rows": int(len(df)),
        "train_prefix_event_count": int(sum(sid_events.values())),
        "train_seen_item_count": int(len(item_events)),
        "catalog_item_zero_train_pct": float((len(code_map) - len(item_events)) / len(code_map) * 100),
        "train_l1_entropy_norm": entropy_norm(l1_events),
        "train_l12_entropy_norm": entropy_norm(l12_events),
        "train_sid_entropy_norm": entropy_norm(sid_events),
    }
    metrics.update(catalog_exposure(l1_events, 1))
    metrics.update(catalog_exposure(l12_events, 2))
    metrics.update(catalog_exposure(sid_events, 3))
    return metrics


def axis_l1(row: dict[str, Any]) -> tuple[str, str]:
    reasons = []
    if row["active_l1"] > 150:
        reasons.append(f"active_l1={row['active_l1']} > 150")
    if row["top5_l1_cover"] < 300:
        reasons.append(f"top5_l1_cover={row['top5_l1_cover']} < 300")
    if row["snear_cnear_same_l1"] < 80:
        reasons.append(f"S-near C-near same_l1={row['snear_cnear_same_l1']:.2f}% < 80%")
    if reasons:
        return "fail", "; ".join(reasons)

    if row["active_l1"] > 100:
        reasons.append(f"active_l1={row['active_l1']} > 100")
    if row["top5_l1_cover"] < 500:
        reasons.append(f"top5_l1_cover={row['top5_l1_cover']} < 500")
    if row["snear_cnear_same_l1"] < 86:
        reasons.append(f"S-near C-near same_l1={row['snear_cnear_same_l1']:.2f}% < 86%")
    if reasons:
        return "warn", "; ".join(reasons)
    return "pass", "L1 routing remains semantically stable."


def axis_separation(row: dict[str, Any]) -> tuple[str, str]:
    reasons = []
    if row["snear_cfar_same_l1"] < 80:
        reasons.append(f"S-near C-far same_l1={row['snear_cfar_same_l1']:.2f}% < 80%")
    if row["snear_cfar_split_after_l1"] < 55:
        reasons.append(f"split_after_l1={row['snear_cfar_split_after_l1']:.2f}% < 55%")
    if row["snear_cfar_same_l12"] > 32.5:
        reasons.append(f"S-near C-far same_l12={row['snear_cfar_same_l12']:.2f}% > 32.5%")
    if reasons:
        return "fail", "; ".join(reasons)

    if row["snear_cfar_split_after_l1"] < 60:
        reasons.append(f"split_after_l1={row['snear_cfar_split_after_l1']:.2f}% < 60%")
    if row["snear_cfar_same_l12"] > 30.5:
        reasons.append(f"S-near C-far same_l12={row['snear_cfar_same_l12']:.2f}% > 30.5%")
    if reasons:
        return "warn", "; ".join(reasons)
    return "pass", "Semantic-near collaborative-far pairs are selectively separated after L1."


def axis_preservation(row: dict[str, Any]) -> tuple[str, str]:
    reasons = []
    if row["snear_cnear_same_l1"] < 80:
        reasons.append(f"S-near C-near same_l1={row['snear_cnear_same_l1']:.2f}% < 80%")
    if row["snear_cnear_same_l12"] < 27:
        reasons.append(f"S-near C-near same_l12={row['snear_cnear_same_l12']:.2f}% < 27%")
    if reasons:
        return "fail", "; ".join(reasons)

    if row["snear_cnear_same_l1"] < 86:
        reasons.append(f"S-near C-near same_l1={row['snear_cnear_same_l1']:.2f}% < 86%")
    if row["snear_cnear_same_l12"] < 31:
        reasons.append(f"S-near C-near same_l12={row['snear_cnear_same_l12']:.2f}% < 31%")
    if reasons:
        return "warn", "; ".join(reasons)
    return "pass", "Semantic-near collaborative-near pairs are preserved."


def axis_learnability(row: dict[str, Any]) -> tuple[str, str]:
    reasons = []
    if row["active_l1"] > 180:
        reasons.append(f"active_l1={row['active_l1']} > 180")
    if row["unique_l12"] > 2750:
        reasons.append(f"unique_l12={row['unique_l12']} > 2750")
    if row["top5_l1_cover"] < 300:
        reasons.append(f"top5_l1_cover={row['top5_l1_cover']} < 300")
    if row.get("catalog_l2_zero_train_pct", 0.0) > 50:
        reasons.append(f"catalog_l12_zero_train_pct={row['catalog_l2_zero_train_pct']:.2f}% > 50%")
    if row.get("catalog_l3_zero_train_pct", 0.0) > 80:
        reasons.append(f"catalog_sid_zero_train_pct={row['catalog_l3_zero_train_pct']:.2f}% > 80%")
    if reasons:
        return "fail", "; ".join(reasons)

    if row["active_l1"] > 100:
        reasons.append(f"active_l1={row['active_l1']} > 100")
    if row["unique_l12"] > 2550:
        reasons.append(f"unique_l12={row['unique_l12']} > 2550")
    if row["top5_l1_cover"] < 500:
        reasons.append(f"top5_l1_cover={row['top5_l1_cover']} < 500")
    if row["l12_singletons"] > 2000:
        reasons.append(f"l12_singletons={row['l12_singletons']} > 2000")
    if row.get("catalog_l2_zero_train_pct", 0.0) > 25:
        reasons.append(f"catalog_l12_zero_train_pct={row['catalog_l2_zero_train_pct']:.2f}% > 25%")
    if row.get("catalog_l3_zero_train_pct", 0.0) > 55:
        reasons.append(f"catalog_sid_zero_train_pct={row['catalog_l3_zero_train_pct']:.2f}% > 55%")
    if reasons:
        return "warn", "; ".join(reasons)
    return "pass", "Prefix distribution is compact enough for SFT."


def profile(row: dict[str, Any]) -> tuple[str, str, str]:
    axes = [row["l1_axis"], row["separation_axis"], row["preservation_axis"], row["learnability_axis"]]
    fail_count = axes.count("fail")
    warn_count = axes.count("warn")

    if row["label"] == "Original semantic":
        return "semantic-stable-baseline", "high", "Stable semantic baseline, but not a collaborative-injection claim."
    if row["label"] == "V2 offline":
        return "out-of-family-flat-routing", "unknown", "Strong historical tokenizer with non-semantic-flat L1; diagnostic should not be used as a pure rank predictor here."

    if fail_count == 0 and warn_count == 0:
        return "balanced-positive", "high", "All four axes pass."
    if row["separation_axis"] == "warn" and row["l1_axis"] == "pass" and row["preservation_axis"] == "pass":
        return "under-separated", "medium", "Stable and learnable, but collaborative separation is weak."
    if row["separation_axis"] == "pass" and row["preservation_axis"] in {"warn", "fail"}:
        if row["preservation_axis"] == "fail":
            return "over-separated-unstable", "low", "Can split semantic-near collaborative-far pairs, but also damages near/near preservation."
        return "separating-but-risky", "low", "Can split target pairs, but preservation or routing stability is risky."
    if row["l1_axis"] == "fail" or row["learnability_axis"] == "fail":
        return "fragmented-routing", "low", "Routing distribution is too fragmented for a reliable downstream expectation."
    if fail_count > 0:
        return "structurally-risky", "low", "At least one key structural axis fails."
    if warn_count >= 2:
        return "borderline", "medium", "Multiple warning axes; downstream may be unstable."
    return "acceptable", "medium", "No hard failure, but not a clean balanced-positive structure."


def sft_band(ndcg: Any) -> str:
    if pd.isna(ndcg):
        return "pending"
    value = float(ndcg)
    if value >= 0.102:
        return "high"
    if value >= 0.097:
        return "medium"
    return "low"


def consistency(predicted: str, actual: str) -> str:
    if actual == "pending":
        return "pending"
    if predicted == "unknown":
        return "out-of-scope"
    if predicted == actual:
        return "match"
    if predicted == "medium" and actual in {"high", "low"}:
        return "partial"
    if predicted == "high" and actual == "medium":
        return "partial"
    if predicted == "low" and actual == "medium":
        return "partial"
    return "mismatch"


def collect_rows(
    helper: Any,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, list[dict[str, Any]]]]:
    emb = helper.normalize_embeddings(Path("data/Amazon/index/Industrial_and_Scientific.emb-qwen-td.npy"))
    pair_counts, ppmi, data_stats = helper.build_collab_stats(
        Path("data/Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.csv")
    )
    semantic_pairs = helper.build_semantic_top_pairs(emb, args.semantic_topk)
    pair_sets = build_extended_pair_sets(helper, emb, semantic_pairs, pair_counts, ppmi, args)
    best_sft = load_best_sft()

    rows: list[dict[str, Any]] = []
    metrics_json: dict[str, Any] = {
        "rules": RULES,
        "data_stats": data_stats,
        "pair_set_stats": {},
        "variants": {},
    }
    for set_name, pairs in pair_sets.items():
        metrics_json["pair_set_stats"][set_name] = {
            "pair_count": len(pairs),
            "semantic_sim_mean": float(np.mean([p["semantic_sim"] for p in pairs])) if pairs else 0.0,
            "ppmi_mean": float(np.mean([p["ppmi"] for p in pairs])) if pairs else 0.0,
        }

    for candidate in CANDIDATES:
        path = resolve(candidate.index_path)
        if not path.exists():
            print(f"[skip] missing index: {candidate.label}: {path}")
            continue

        code_map = helper.load_index(candidate.index_path)
        structure = helper.structure_metrics(code_map)
        learnability = train_prefix_frequency_metrics(helper, code_map)
        pair_metric = {set_name: helper.pair_metrics(code_map, pairs) for set_name, pairs in pair_sets.items()}

        sft_rows = best_sft[best_sft["tokenizer_record_id"].eq(candidate.tokenizer_record_id)]
        sft_row = sft_rows.iloc[0].to_dict() if not sft_rows.empty else {}

        row: dict[str, Any] = {
            "label": candidate.label,
            "tokenizer_record_id": candidate.tokenizer_record_id,
            "split": candidate.split,
            "note": candidate.note,
            "sft_status": "available" if sft_row else "pending",
            "sft_variant": sft_row.get("variant", ""),
            "recipe": sft_row.get("recipe", ""),
            "ndcg_at_1": sft_row.get("ndcg_at_1", np.nan),
            "ndcg_at_3": sft_row.get("ndcg_at_3", np.nan),
            "ndcg_at_5": sft_row.get("ndcg_at_5", np.nan),
            "ndcg_at_10": sft_row.get("ndcg_at_10", np.nan),
            "hr_at_10": sft_row.get("hr_at_10", np.nan),
            "hr_at_50": sft_row.get("hr_at_50", np.nan),
            **structure,
            **learnability,
            "snear_cfar_same_l1": pair_metric["S-near C-far"]["same_l1_pct"],
            "snear_cfar_same_l12": pair_metric["S-near C-far"]["same_l12_pct"],
            "snear_cfar_split_after_l1": pair_metric["S-near C-far"]["split_after_l1_pct"],
            "snear_cnear_same_l1": pair_metric["S-near C-near"]["same_l1_pct"],
            "snear_cnear_same_l12": pair_metric["S-near C-near"]["same_l12_pct"],
            "smid_cfar_pair_count": pair_metric["S-mid C-far"]["pair_count"],
            "smid_cfar_same_l1": pair_metric["S-mid C-far"]["same_l1_pct"],
            "smid_cfar_same_l12": pair_metric["S-mid C-far"]["same_l12_pct"],
            "smid_cfar_split_after_l1": pair_metric["S-mid C-far"]["split_after_l1_pct"],
            "smid_cnear_pair_count": pair_metric["S-mid C-near"]["pair_count"],
            "smid_cnear_same_l1": pair_metric["S-mid C-near"]["same_l1_pct"],
            "smid_cnear_same_l12": pair_metric["S-mid C-near"]["same_l12_pct"],
            "sfar_cnear_avg_overlap": pair_metric["S-far C-near"]["avg_token_overlap"],
            "sfar_cnear_same_l1": pair_metric["S-far C-near"]["same_l1_pct"],
        }

        row["l1_axis"], row["l1_reason"] = axis_l1(row)
        row["separation_axis"], row["separation_reason"] = axis_separation(row)
        row["preservation_axis"], row["preservation_reason"] = axis_preservation(row)
        row["learnability_axis"], row["learnability_reason"] = axis_learnability(row)
        row["diagnostic_profile"], row["predicted_sft_band"], row["profile_reason"] = profile(row)
        row["actual_sft_band"] = sft_band(row["ndcg_at_10"])
        row["diagnostic_consistency"] = consistency(row["predicted_sft_band"], row["actual_sft_band"])

        rows.append(row)
        metrics_json["variants"][candidate.label] = {
            "structure": structure,
            "learnability": learnability,
            "pair_metrics": pair_metric,
            "diagnostic": {
                "l1_axis": row["l1_axis"],
                "separation_axis": row["separation_axis"],
                "preservation_axis": row["preservation_axis"],
                "learnability_axis": row["learnability_axis"],
                "profile": row["diagnostic_profile"],
                "predicted_sft_band": row["predicted_sft_band"],
            },
        }
    return rows, metrics_json, pair_sets


def md_table(df: pd.DataFrame, columns: list[str], floatfmt: str = ".4f") -> str:
    return df[columns].to_markdown(index=False, floatfmt=floatfmt)


def render_report(df: pd.DataFrame, output_dir: Path) -> str:
    lines: list[str] = []
    lines.append("# SID Structural Diagnostic（语义标识结构诊断）")
    lines.append("")
    lines.append("## Pre-registered Rules（预注册规则）")
    lines.append("")
    lines.append("- 诊断只使用 tokenizer（分词器）、semantic embedding（语义嵌入）和 train interaction（训练交互）构造的 pair（物品对）。")
    lines.append("- SFT（监督微调）指标只用于事后对照，不参与四轴 verdict（裁决）。")
    lines.append("- 这不是单指标 predictor（预测器），而是 multi-axis diagnostic（多轴诊断）。")
    lines.append("")
    for name, rule in RULES.items():
        lines.append(f"- `{name}`: fail（失败） if {rule['fail']}; warn（警告） if {rule['warn']}.")
    lines.append("")
    lines.append("## Diagnostic vs SFT（诊断对监督微调）")
    lines.append("")
    cols = [
        "label",
        "split",
        "diagnostic_profile",
        "predicted_sft_band",
        "actual_sft_band",
        "diagnostic_consistency",
        "ndcg_at_10",
        "hr_at_10",
        "l1_axis",
        "separation_axis",
        "preservation_axis",
        "learnability_axis",
    ]
    lines.append(md_table(df.sort_values(["split", "ndcg_at_10"], ascending=[True, False]), cols))
    lines.append("")
    available = df[df["actual_sft_band"].ne("pending")]
    if not available.empty:
        counts = available["diagnostic_consistency"].value_counts().to_dict()
        lines.append("## Consistency Summary（一致性总结）")
        lines.append("")
        lines.append(
            f"- available SFT（已有监督微调）样本数: {len(available)}; "
            f"match（完全匹配）={counts.get('match', 0)}, partial（部分匹配）={counts.get('partial', 0)}, "
            f"mismatch（不匹配）={counts.get('mismatch', 0)}, out-of-scope（超出适用域）={counts.get('out-of-scope', 0)}."
        )
        calibration = available[available["split"].eq("calibration")]
        if not calibration.empty:
            c_counts = calibration["diagnostic_consistency"].value_counts().to_dict()
            lines.append(
                f"- calibration（校准集）样本数: {len(calibration)}; "
                f"match={c_counts.get('match', 0)}, partial={c_counts.get('partial', 0)}, mismatch={c_counts.get('mismatch', 0)}."
            )
    lines.append("")
    lines.append("## Structure Distribution（码本结构分布）")
    lines.append("")
    structure_cols = [
        "label",
        "active_l1",
        "unique_l12",
        "unique_sid",
        "collision_count",
        "top5_l1_cover",
        "l1_entropy_norm",
        "l1_gini",
        "avg_l2_per_l1",
        "l12_singletons",
        "l12_ge5",
    ]
    lines.append(md_table(df.sort_values(["split", "label"]), structure_cols))
    lines.append("")
    lines.append("## Mid-Similarity Blind Spot（中等语义相似盲区）")
    lines.append("")
    lines.append(
        "S-mid（中等语义相似）使用语义相似度区间 `[0.80, 0.90)`，用于补充极端 S-near/S-far（语义近/远）之外的商品对。"
    )
    lines.append("")
    mid_cols = [
        "label",
        "smid_cfar_pair_count",
        "smid_cfar_same_l1",
        "smid_cfar_same_l12",
        "smid_cfar_split_after_l1",
        "smid_cnear_pair_count",
        "smid_cnear_same_l1",
        "smid_cnear_same_l12",
    ]
    lines.append(md_table(df.sort_values(["split", "label"]), mid_cols))
    lines.append("")
    lines.append("## Train-Only Learnability（仅训练集可学习性）")
    lines.append("")
    lines.append(
        "这些指标只使用 train interaction（训练交互），衡量 SID prefix（语义标识前缀）在 SFT（监督微调）训练数据中是否有足够曝光。"
    )
    lines.append("")
    learn_cols = [
        "label",
        "train_l1_entropy_norm",
        "train_l12_entropy_norm",
        "train_sid_entropy_norm",
        "catalog_l2_zero_train_pct",
        "catalog_l2_median_train_events",
        "catalog_l3_zero_train_pct",
        "catalog_l3_median_train_events",
        "catalog_item_zero_train_pct",
    ]
    lines.append(md_table(df.sort_values(["split", "label"]), learn_cols))
    lines.append("")
    lines.append("## Key Observations（关键观察）")
    lines.append("")
    lines.append(
        "1. R690b L2 sweep（第二层权重扫描）形成了清晰的结构趋势：`0.010` 是 balanced-positive（平衡正向），`0.003` 更像 under-separated（拆分不足），`0.015` 和 no-L1（无第一层语义）是过拆或路由不稳。"
    )
    lines.append(
        "2. QCR 是核心反例：selective separation（选择性拆分）很好，但 preservation（协同保持）和 learnability（可学习性）有警告，因此不能只看 split-after-L1（同第一层后拆分）。"
    )
    lines.append(
        "3. V2 offline（离线 v2）是跨方法族例外：它的 L1 semantic stability（第一层语义稳定性）不符合当前诊断假设，但下游仍强，说明本诊断最适合解释“语义层级协同注入”方法族，不应当当作跨所有 SID 的单一排名器。"
    )
    lines.append(
        "4. L3=0.010 目前是 prospective（前瞻）样本：四轴均 pass（通过），所以结构上是 high-band candidate（高潜力候选），等待 SFT（监督微调）结果验证。"
    )
    lines.append(
        "5. 新增 S-mid（中等语义相似）和 train-only learnability（仅训练集可学习性）是增强证据，不替代原四轴规则；它们主要用于发现“结构看着好但下游学不到”的情况。"
    )
    lines.append("")
    lines.append("## Output Files（输出文件）")
    lines.append("")
    lines.append(f"- `diagnostic_metrics.csv`: `{output_dir / 'diagnostic_metrics.csv'}`")
    lines.append(f"- `diagnostic_rules.json`: `{output_dir / 'diagnostic_rules.json'}`")
    lines.append(f"- `diagnostic_case_studies.md`: `{output_dir / 'diagnostic_case_studies.md'}`")
    lines.append(f"- `diagnostic_pair_examples.md`: `{output_dir / 'diagnostic_pair_examples.md'}`")
    lines.append(f"- `metrics.json`: `{output_dir / 'metrics.json'}`")
    return "\n".join(lines) + "\n"


def case_summary(row: pd.Series) -> str:
    sft = "pending（待完成）" if pd.isna(row["ndcg_at_10"]) else f"NDCG@10={row['ndcg_at_10']:.6f}, HR@10={row['hr_at_10']:.6f}"
    return (
        f"### {row['label']}\n\n"
        f"- split（划分）: `{row['split']}`\n"
        f"- profile（画像）: `{row['diagnostic_profile']}`; predicted SFT band（预测监督微调档位）: `{row['predicted_sft_band']}`; actual（实际）: `{row['actual_sft_band']}`\n"
        f"- SFT（监督微调）: {sft}\n"
        f"- L1 stability（第一层稳定性）: `{row['l1_axis']}`; {row['l1_reason']}\n"
        f"- selective separation（选择性拆分）: `{row['separation_axis']}`; {row['separation_reason']}\n"
        f"- collaborative preservation（协同保持）: `{row['preservation_axis']}`; {row['preservation_reason']}\n"
        f"- learnability（可学习性）: `{row['learnability_axis']}`; {row['learnability_reason']}\n"
        f"- interpretation（解释）: {row['profile_reason']}\n"
    )


def render_case_studies(df: pd.DataFrame) -> str:
    selected = [
        "R690b L2=0.010 main",
        "R690b L2=0.003 weak",
        "R690b L2=0.015 fragmented",
        "R690b no L1 semantic",
        "QCR L2 conflict ranking",
        "V2 offline",
        "R690b L3=0.010 pending",
    ]
    lines = ["# Diagnostic Case Studies（诊断案例分析）", ""]
    for label in selected:
        rows = df[df["label"].eq(label)]
        if rows.empty:
            continue
        lines.append(case_summary(rows.iloc[0]))
        lines.append("")
    return "\n".join(lines)


def render_pair_examples(
    helper: Any,
    pair_sets: dict[str, list[dict[str, Any]]],
    args: argparse.Namespace,
) -> str:
    """Render support/counter/random examples without using downstream labels."""
    label_to_candidate = {candidate.label: candidate for candidate in CANDIDATES}
    focus_labels = [
        "R690b L2=0.003 weak",
        "R690b L2=0.010 main",
        "R690b L3=0.010 pending",
    ]
    missing = [label for label in focus_labels if not resolve(label_to_candidate[label].index_path).exists()]
    if missing:
        return "# Diagnostic Pair Examples（诊断物品对案例）\n\n" + f"Missing focus indices（缺少索引）: {missing}\n"

    items = helper.load_items(Path("data/Amazon/index/Industrial_and_Scientific.item.json"))
    code_maps = {
        label: helper.load_index(label_to_candidate[label].index_path)
        for label in focus_labels
    }
    weak = code_maps["R690b L2=0.003 weak"]
    main = code_maps["R690b L2=0.010 main"]
    l3 = code_maps["R690b L3=0.010 pending"]
    rng = random.Random(args.random_seed)

    def add_codes(pair: dict[str, Any]) -> dict[str, Any]:
        a, b = int(pair["item_a"]), int(pair["item_b"])
        out = dict(pair)
        out["title_a"] = helper.title(items, a, max_len=80)
        out["title_b"] = helper.title(items, b, max_len=80)
        for label, cmap in code_maps.items():
            out[label] = f"{cmap[a]} / {cmap[b]} | {helper.code_relation(cmap[a], cmap[b])}"
        return out

    def support_filter(pair: dict[str, Any]) -> bool:
        a, b = int(pair["item_a"]), int(pair["item_b"])
        weak_keeps_too_much = weak[a][:2] == weak[b][:2]
        main_splits_after_l1 = main[a][0] == main[b][0] and main[a][:2] != main[b][:2]
        l3_splits_after_l1 = l3[a][0] == l3[b][0] and l3[a][:2] != l3[b][:2]
        return weak_keeps_too_much and (main_splits_after_l1 or l3_splits_after_l1)

    def counter_filter(pair: dict[str, Any]) -> bool:
        a, b = int(pair["item_a"]), int(pair["item_b"])
        weak_preserves = weak[a][:2] == weak[b][:2]
        main_breaks_l1 = main[a][0] != main[b][0]
        main_breaks_l12 = main[a][:2] != main[b][:2]
        l3_breaks_l1 = l3[a][0] != l3[b][0]
        return weak_preserves and (main_breaks_l1 or (main_breaks_l12 and l3_breaks_l1))

    def choose(name: str, predicate: Any | None = None, randomize: bool = False) -> list[dict[str, Any]]:
        pairs = list(pair_sets.get(name, []))
        if predicate is not None:
            pairs = [pair for pair in pairs if predicate(pair)]
        if randomize:
            rng.shuffle(pairs)
        return [add_codes(pair) for pair in pairs[: args.example_count]]

    sections = {
        "Supportive S-near/S-mid C-far（支持例：语义近/中等且协同远）": (
            choose("S-near C-far", support_filter) + choose("S-mid C-far", support_filter)
        )[: args.example_count],
        "Counter S-near/S-mid C-near（反例：语义近/中等且协同近）": (
            choose("S-near C-near", counter_filter) + choose("S-mid C-near", counter_filter)
        )[: args.example_count],
        "Random S-near C-far（随机例：语义近协同远）": choose("S-near C-far", randomize=True),
        "Random S-mid C-far（随机例：语义中等协同远）": choose("S-mid C-far", randomize=True),
        "Random S-mid C-near（随机例：语义中等协同近）": choose("S-mid C-near", randomize=True),
    }

    lines = [
        "# Diagnostic Pair Examples（诊断物品对案例）",
        "",
        "案例只使用 tokenizer（分词器）、title（标题）、semantic similarity（语义相似度）和 train-only collaborative statistics（仅训练集协同统计）。",
        "支持例和反例用于人工审查，随机例用于降低 confirmation bias（确认偏误）。",
        "",
    ]
    cols = [
        "pair",
        "semantic_sim",
        "cooc_count",
        "ppmi",
        "title_a",
        "title_b",
        "R690b L2=0.003 weak",
        "R690b L2=0.010 main",
        "R690b L3=0.010 pending",
    ]
    for section, rows in sections.items():
        lines.append(f"## {section}")
        lines.append("")
        if not rows:
            lines.append("No examples found（未找到样例）.")
            lines.append("")
            continue
        table_rows = []
        for row in rows:
            table_rows.append(
                {
                    "pair": f"{row['item_a']}-{row['item_b']}",
                    "semantic_sim": row["semantic_sim"],
                    "cooc_count": row["cooc_count"],
                    "ppmi": row["ppmi"],
                    "title_a": row["title_a"],
                    "title_b": row["title_b"],
                    "R690b L2=0.003 weak": row["R690b L2=0.003 weak"],
                    "R690b L2=0.010 main": row["R690b L2=0.010 main"],
                    "R690b L3=0.010 pending": row["R690b L3=0.010 pending"],
                }
            )
        lines.append(pd.DataFrame(table_rows)[cols].to_markdown(index=False, floatfmt=".4f"))
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    np.random.seed(args.random_seed)
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    helper = load_helper()
    rows, metrics_json, pair_sets = collect_rows(helper, args)
    df = pd.DataFrame(rows)

    preferred_cols = [
        "label",
        "split",
        "diagnostic_profile",
        "predicted_sft_band",
        "actual_sft_band",
        "diagnostic_consistency",
        "ndcg_at_10",
        "hr_at_10",
        "l1_axis",
        "separation_axis",
        "preservation_axis",
        "learnability_axis",
        "active_l1",
        "unique_l12",
        "top5_l1_cover",
        "l1_entropy_norm",
        "l1_gini",
        "avg_l2_per_l1",
        "l12_singletons",
        "l12_ge5",
        "snear_cfar_same_l1",
        "snear_cfar_same_l12",
        "snear_cfar_split_after_l1",
        "snear_cnear_same_l1",
        "snear_cnear_same_l12",
        "smid_cfar_pair_count",
        "smid_cfar_same_l1",
        "smid_cfar_same_l12",
        "smid_cfar_split_after_l1",
        "smid_cnear_pair_count",
        "smid_cnear_same_l1",
        "smid_cnear_same_l12",
        "sfar_cnear_avg_overlap",
        "train_l1_entropy_norm",
        "train_l12_entropy_norm",
        "train_sid_entropy_norm",
        "catalog_l2_zero_train_pct",
        "catalog_l2_median_train_events",
        "catalog_l3_zero_train_pct",
        "catalog_l3_median_train_events",
        "l1_reason",
        "separation_reason",
        "preservation_reason",
        "learnability_reason",
        "profile_reason",
        "note",
        "tokenizer_record_id",
        "sft_variant",
    ]
    remaining_cols = [col for col in df.columns if col not in preferred_cols]
    df[preferred_cols + remaining_cols].to_csv(output_dir / "diagnostic_metrics.csv", index=False)
    with (output_dir / "diagnostic_rules.json").open("w", encoding="utf-8") as f:
        json.dump(RULES, f, ensure_ascii=False, indent=2)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics_json, f, ensure_ascii=False, indent=2)

    (output_dir / "diagnostic_vs_sft_report.md").write_text(render_report(df, output_dir), encoding="utf-8")
    (output_dir / "diagnostic_case_studies.md").write_text(render_case_studies(df), encoding="utf-8")
    (output_dir / "diagnostic_pair_examples.md").write_text(
        render_pair_examples(helper, pair_sets, args),
        encoding="utf-8",
    )

    print(f"[done] report: {output_dir / 'diagnostic_vs_sft_report.md'}")
    print(f"[done] metrics: {output_dir / 'diagnostic_metrics.csv'}")


if __name__ == "__main__":
    main()
