#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from onerec.evaluate.collaborative_rerank import CollaborativeReranker, parse_id_list, parse_prefix_parts
from onerec.evaluate.semantic_id import canonicalize_semantic_id

TOPK_LIST = [1, 3, 5, 10, 20, 50]


def compute_metrics(samples: list[dict[str, Any]]) -> dict[str, Any]:
    topk_list = TOPK_LIST
    if not samples:
        return {"count": 0, "hr": {}, "ndcg": {}}
    max_beam = min(len(samples[0]["predict"]), max(topk_list))
    valid_topk = [k for k in topk_list if k <= max_beam]
    hr = {k: 0.0 for k in valid_topk}
    ndcg = {k: 0.0 for k in valid_topk}
    same_l1_errors = 0
    same_l2_errors = 0
    misses = 0
    beam_has_same_l1 = 0
    beam_has_same_l2 = 0
    for sample in samples:
        target = canonicalize_semantic_id(sample["output"])
        predicts = [canonicalize_semantic_id(pred) for pred in sample["predict"]]
        target_a, target_b, _ = parse_prefix_parts(target)
        rank = None
        for idx, pred in enumerate(predicts):
            if pred == target:
                rank = idx
                break
        if any(parse_prefix_parts(pred)[0] == target_a for pred in predicts):
            beam_has_same_l1 += 1
        if any(parse_prefix_parts(pred)[:2] == (target_a, target_b) for pred in predicts):
            beam_has_same_l2 += 1
        if rank is None:
            misses += 1
            pred_a, pred_b, _ = parse_prefix_parts(predicts[0] if predicts else "")
            if pred_a == target_a:
                same_l1_errors += 1
            if (pred_a, pred_b) == (target_a, target_b):
                same_l2_errors += 1
            continue
        for k in valid_topk:
            if rank < k:
                hr[k] += 1.0
                ndcg[k] += 1.0 / math.log2(rank + 2)
    count = len(samples)
    return {
        "count": count,
        "hr": {f"HR@{k}": hr[k] / count for k in valid_topk},
        "ndcg": {f"NDCG@{k}": ndcg[k] / count for k in valid_topk},
        "top1_hit_rate": hr[1] / count if 1 in hr else None,
        "top1_error_same_l1_rate": (same_l1_errors / misses) if misses else 0.0,
        "top1_error_same_l2_rate": (same_l2_errors / misses) if misses else 0.0,
        "beam_has_same_l1_rate": beam_has_same_l1 / count,
        "beam_has_same_l2_rate": beam_has_same_l2 / count,
        "miss_count": misses,
    }


def load_and_validate_alignment(test_csv: Path, result_json: Path) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    test_df = pd.read_csv(test_csv)
    with open(result_json, "r", encoding="utf-8") as handle:
        result_data = json.load(handle)
    if len(test_df) != len(result_data):
        raise ValueError(f"Length mismatch: test={len(test_df)} result={len(result_data)}")
    mismatches = 0
    for idx, (_, row) in enumerate(test_df.iterrows()):
        expected = canonicalize_semantic_id(row["item_sid"])
        observed = canonicalize_semantic_id(result_data[idx]["output"])
        if expected != observed:
            mismatches += 1
            if mismatches <= 3:
                print(f"[ALIGN_MISMATCH] row={idx} test={expected} result={observed}")
    if mismatches:
        raise ValueError(f"Detected {mismatches} row-order mismatches between test CSV and result JSON")
    return test_df, result_data


def run_modes(
    test_df: pd.DataFrame,
    result_data: list[dict[str, Any]],
    modes: list[str],
    reranker: CollaborativeReranker,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    summaries: dict[str, Any] = {}
    reranked_payloads: dict[str, list[dict[str, Any]]] = {}
    for mode in modes:
        payload: list[dict[str, Any]] = []
        activation_count = 0
        for idx, (_, row) in enumerate(test_df.iterrows()):
            history = parse_id_list(row["history_item_id"])
            predicts = result_data[idx]["predict"]
            reranked_predicts, activated = reranker.rerank_predict_list(predicts, history, mode)
            if activated:
                activation_count += 1
            payload.append(
                {
                    "input": result_data[idx]["input"],
                    "output": result_data[idx]["output"],
                    "predict": reranked_predicts,
                }
            )
        summary = compute_metrics(payload)
        summary["activated_count"] = activation_count
        summaries[mode] = summary
        reranked_payloads[mode] = payload
    return summaries, reranked_payloads


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline collaborative rerank / ACLR-lite baseline for V0.5")
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--result_json", required=True)
    parser.add_argument("--index_json", required=True)
    parser.add_argument("--modes", nargs="+", default=["baseline", "global", "same_l1", "same_l2", "ambiguity_l2"])
    parser.add_argument("--history_k", type=int, default=10)
    parser.add_argument("--sid_score_mode", choices=["best", "avg", "worst"], default="best")
    parser.add_argument("--ambiguity_leaf_threshold", type=int, default=8)
    parser.add_argument("--output_dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_csv = Path(args.train_csv)
    test_csv = Path(args.test_csv)
    result_json = Path(args.result_json)
    index_json = Path(args.index_json)
    output_dir = Path(args.output_dir) if args.output_dir else None

    reranker = CollaborativeReranker.from_files(
        train_csv=train_csv,
        index_json=index_json,
        history_k=args.history_k,
        sid_score_mode=args.sid_score_mode,
        ambiguity_leaf_threshold=args.ambiguity_leaf_threshold,
    )
    test_df, result_data = load_and_validate_alignment(test_csv, result_json)

    summaries, reranked_payloads = run_modes(
        test_df,
        result_data,
        args.modes,
        reranker,
    )

    final_summary = {
        "train_csv": str(train_csv),
        "test_csv": str(test_csv),
        "result_json": str(result_json),
        "index_json": str(index_json),
        "history_k": args.history_k,
        "sid_score_mode": args.sid_score_mode,
        "ambiguity_leaf_threshold": args.ambiguity_leaf_threshold,
        "ambiguous_prefix_count": len(reranker.ambiguous_prefixes or set()),
        "modes": summaries,
    }

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = output_dir / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(final_summary, handle, indent=2, ensure_ascii=False)
        for mode, payload in reranked_payloads.items():
            with open(output_dir / f"{mode}.json", "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, ensure_ascii=False)
        print(summary_path)
    else:
        print(json.dumps(final_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
