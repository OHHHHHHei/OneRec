#!/usr/bin/env python3
"""
Quantify the gap between train-derived collaborative compatibility and model predictions.

This script aligns:
1. train.csv to build a simple train-only collaborative score
2. test.csv to recover history/target item_ids
3. final_result_*.json to recover top1 predicted SID

It then measures how often the true target is more collaborative-compatible with
the user's history than the top1 prediction.
"""

from __future__ import annotations

import argparse
import ast
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Iterable

import numpy as np
import pandas as pd

from onerec.evaluate.semantic_id import canonicalize_semantic_id


def parse_item_list(value) -> list[int]:
    if isinstance(value, list):
        return [int(x) for x in value]
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return []
    if isinstance(parsed, list):
        return [int(x) for x in parsed]
    return []


def load_index(index_json: Path) -> tuple[dict[str, list[int]], dict[int, str]]:
    with index_json.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    sid_to_items: dict[str, list[int]] = defaultdict(list)
    item_to_sid: dict[int, str] = {}
    for item_id, tokens in raw.items():
        sid = "".join(tokens)
        iid = int(item_id)
        sid_to_items[sid].append(iid)
        item_to_sid[iid] = sid
    return dict(sid_to_items), item_to_sid


def load_titles(item_json: Path) -> dict[int, str]:
    with item_json.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v.get("title", f"Item_{k}") for k, v in raw.items()}


def build_pair_statistics(train_df: pd.DataFrame, history_k: int) -> tuple[Counter, Counter]:
    pair_count: Counter[tuple[int, int]] = Counter()
    hist_count: Counter[int] = Counter()

    for _, row in train_df.iterrows():
        history = parse_item_list(row["history_item_id"])
        if not history:
            continue
        target = int(row["item_id"])
        tail = history[-history_k:]
        for hist_item in tail:
            pair_count[(hist_item, target)] += 1
            hist_count[hist_item] += 1

    return pair_count, hist_count


def collaborative_score(history: list[int], candidate: int, pair_count: Counter, hist_count: Counter, history_k: int) -> float:
    if not history:
        return 0.0

    score = 0.0
    tail = history[-history_k:]
    for rank, hist_item in enumerate(reversed(tail), start=1):
        denom = hist_count.get(hist_item, 0)
        if denom == 0:
            continue
        prob = pair_count.get((hist_item, candidate), 0) / denom
        score += prob / rank
    return score


def parse_sid_levels(sid: str) -> tuple[str, str, str]:
    sid = sid.strip()
    parts: list[str] = []
    token = []
    inside = False
    for ch in sid:
        if ch == "<":
            inside = True
            token = ["<"]
        elif ch == ">" and inside:
            token.append(">")
            parts.append("".join(token))
            inside = False
        elif inside:
            token.append(ch)
    while len(parts) < 3:
        parts.append("")
    return tuple(parts[:3])  # type: ignore[return-value]


def cosine_similarity(emb: np.ndarray | None, a: int, b: int) -> float | None:
    if emb is None:
        return None
    if a < 0 or b < 0 or a >= len(emb) or b >= len(emb):
        return None
    va = emb[a]
    vb = emb[b]
    na = np.linalg.norm(va)
    nb = np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return None
    return float(np.dot(va, vb) / (na * nb))


def safe_mean(values: Iterable[float]) -> float | None:
    values = list(values)
    if not values:
        return None
    return float(mean(values))


def safe_median(values: Iterable[float]) -> float | None:
    values = list(values)
    if not values:
        return None
    return float(median(values))


def analyze(
    train_csv: Path,
    test_csv: Path,
    result_json: Path,
    index_json: Path,
    item_json: Path,
    output_json: Path,
    output_csv: Path,
    emb_npy: Path | None,
    history_k: int,
) -> None:
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    with result_json.open("r", encoding="utf-8") as f:
        results = json.load(f)

    if len(test_df) != len(results):
        raise ValueError(f"Length mismatch: test={len(test_df)} results={len(results)}")

    sid_to_items, item_to_sid = load_index(index_json)
    item_to_title = load_titles(item_json)
    emb = np.load(emb_npy) if emb_npy else None

    pair_count, hist_count = build_pair_statistics(train_df, history_k)

    total_examples = len(test_df)
    top1_hit_count = 0
    top1_error_count = 0
    analyzable_error_count = 0
    unique_pred_analyzable_error_count = 0
    ambiguous_pred_count = 0
    pred_sid_missing_count = 0

    cf_target_better_unique_count = 0
    cf_pred_better_unique_count = 0
    cf_tie_unique_count = 0
    both_zero_unique_count = 0
    cf_target_better_bestcase_count = 0
    cf_target_better_avgcase_count = 0
    cf_target_better_worstcase_count = 0
    cf_pred_better_bestcase_count = 0
    cf_pred_better_avgcase_count = 0
    cf_pred_better_worstcase_count = 0
    cf_tie_bestcase_count = 0
    cf_tie_avgcase_count = 0
    cf_tie_worstcase_count = 0
    both_zero_bestcase_count = 0
    both_zero_avgcase_count = 0
    both_zero_worstcase_count = 0

    same_l1_error_count = 0
    same_l2_error_count = 0
    same_l1_analyzable_unique_count = 0
    same_l2_analyzable_unique_count = 0
    same_l1_analyzable_all_count = 0
    same_l2_analyzable_all_count = 0
    same_l1_target_better_unique_count = 0
    same_l2_target_better_unique_count = 0
    same_l1_target_better_bestcase_count = 0
    same_l2_target_better_bestcase_count = 0
    same_l1_target_better_avgcase_count = 0
    same_l2_target_better_avgcase_count = 0
    same_l1_target_better_worstcase_count = 0
    same_l2_target_better_worstcase_count = 0

    cf_gaps_unique: list[float] = []
    cf_gaps_bestcase: list[float] = []
    cf_gaps_avgcase: list[float] = []
    cf_gaps_worstcase: list[float] = []
    text_sims_error: list[float] = []
    text_sims_target_better_unique: list[float] = []
    text_sims_target_better_bestcase: list[float] = []
    text_sims_same_l1: list[float] = []
    text_sims_same_l2: list[float] = []

    example_rows: list[dict] = []

    for idx, (_, row) in enumerate(test_df.iterrows()):
        result = results[idx]
        predicts = result.get("predict", [])
        pred_sid = canonicalize_semantic_id(predicts[0] if predicts else "")
        target_sid = canonicalize_semantic_id(row["item_sid"])
        history = parse_item_list(row["history_item_id"])
        target_item = int(row["item_id"])

        if pred_sid == target_sid:
            top1_hit_count += 1
            continue

        top1_error_count += 1

        pred_candidates = sid_to_items.get(pred_sid, [])
        if not pred_candidates:
            pred_sid_missing_count += 1
            continue

        analyzable_error_count += 1
        if len(pred_candidates) == 1:
            unique_pred_analyzable_error_count += 1
            pred_item = pred_candidates[0]
        else:
            ambiguous_pred_count += 1
            pred_item = None

        target_score = collaborative_score(history, target_item, pair_count, hist_count, history_k)
        pred_scores = [collaborative_score(history, candidate, pair_count, hist_count, history_k) for candidate in pred_candidates]
        pred_best_score = max(pred_scores)
        pred_worst_score = min(pred_scores)
        pred_avg_score = float(mean(pred_scores))
        gap_bestcase = target_score - pred_best_score
        gap_avgcase = target_score - pred_avg_score
        gap_worstcase = target_score - pred_worst_score
        cf_gaps_bestcase.append(gap_bestcase)
        cf_gaps_avgcase.append(gap_avgcase)
        cf_gaps_worstcase.append(gap_worstcase)

        a_t, b_t, _ = parse_sid_levels(target_sid)
        a_p, b_p, _ = parse_sid_levels(pred_sid)
        same_l1 = int(a_t == a_p and a_t != "")
        same_l2 = int(same_l1 and b_t == b_p and b_t != "")
        if same_l1:
            same_l1_error_count += 1
            same_l1_analyzable_all_count += 1
        if same_l2:
            same_l2_error_count += 1
            same_l2_analyzable_all_count += 1

        text_sims = [sim for sim in (cosine_similarity(emb, target_item, candidate) for candidate in pred_candidates) if sim is not None]
        text_sim_best = max(text_sims) if text_sims else None
        text_sim_avg = safe_mean(text_sims)
        if text_sim_best is not None:
            text_sims_error.append(text_sim_best)
            if same_l1:
                text_sims_same_l1.append(text_sim_best)
            if same_l2:
                text_sims_same_l2.append(text_sim_best)

        if gap_bestcase > 0:
            cf_target_better_bestcase_count += 1
            if same_l1:
                same_l1_target_better_bestcase_count += 1
            if same_l2:
                same_l2_target_better_bestcase_count += 1
            if text_sim_best is not None:
                text_sims_target_better_bestcase.append(text_sim_best)
        elif gap_bestcase < 0:
            cf_pred_better_bestcase_count += 1
        else:
            cf_tie_bestcase_count += 1

        if gap_avgcase > 0:
            cf_target_better_avgcase_count += 1
            if same_l1:
                same_l1_target_better_avgcase_count += 1
            if same_l2:
                same_l2_target_better_avgcase_count += 1
        elif gap_avgcase < 0:
            cf_pred_better_avgcase_count += 1
        else:
            cf_tie_avgcase_count += 1

        if gap_worstcase > 0:
            cf_target_better_worstcase_count += 1
            if same_l1:
                same_l1_target_better_worstcase_count += 1
            if same_l2:
                same_l2_target_better_worstcase_count += 1
        elif gap_worstcase < 0:
            cf_pred_better_worstcase_count += 1
        else:
            cf_tie_worstcase_count += 1

        if target_score == 0.0 and pred_best_score == 0.0:
            both_zero_bestcase_count += 1
        if target_score == 0.0 and pred_avg_score == 0.0:
            both_zero_avgcase_count += 1
        if target_score == 0.0 and pred_worst_score == 0.0:
            both_zero_worstcase_count += 1

        if pred_item is not None:
            pred_score = pred_scores[0]
            gap_unique = target_score - pred_score
            cf_gaps_unique.append(gap_unique)
            if same_l1:
                same_l1_analyzable_unique_count += 1
            if same_l2:
                same_l2_analyzable_unique_count += 1

            text_sim = text_sim_best
            if gap_unique > 0:
                cf_target_better_unique_count += 1
                if same_l1:
                    same_l1_target_better_unique_count += 1
                if same_l2:
                    same_l2_target_better_unique_count += 1
                if text_sim is not None:
                    text_sims_target_better_unique.append(text_sim)
            elif gap_unique < 0:
                cf_pred_better_unique_count += 1
            else:
                cf_tie_unique_count += 1

            if target_score == 0.0 and pred_score == 0.0:
                both_zero_unique_count += 1
            representative_item = pred_item
            representative_title = item_to_title.get(pred_item, f"Item_{pred_item}")
        else:
            best_idx = max(range(len(pred_candidates)), key=lambda i: pred_scores[i])
            representative_item = pred_candidates[best_idx]
            representative_title = item_to_title.get(representative_item, f"Item_{representative_item}")

        example_rows.append(
            {
                "row_idx": idx,
                "history_len": len(history),
                "target_item_id": target_item,
                "target_title": item_to_title.get(target_item, f"Item_{target_item}"),
                "target_sid": target_sid,
                "pred_candidate_count": len(pred_candidates),
                "pred_item_id_representative": representative_item,
                "pred_title_representative": representative_title,
                "pred_sid": pred_sid,
                "same_l1": same_l1,
                "same_l2": same_l2,
                "target_cf_score": target_score,
                "pred_cf_score_bestcase": pred_best_score,
                "pred_cf_score_avgcase": pred_avg_score,
                "pred_cf_score_worstcase": pred_worst_score,
                "cf_gap_target_minus_pred_bestcase": gap_bestcase,
                "cf_gap_target_minus_pred_avgcase": gap_avgcase,
                "cf_gap_target_minus_pred_worstcase": gap_worstcase,
                "text_cosine_target_pred_best": text_sim_best,
                "text_cosine_target_pred_avg": text_sim_avg,
                "history_item_id": history,
            }
        )

    summary = {
        "train_pair_stats": {
            "history_k": history_k,
            "distinct_history_items": len(hist_count),
            "distinct_history_target_pairs": len(pair_count),
        },
        "evaluation_alignment": {
            "diagnostic_scope": "correlation_diagnostic_not_causal",
            "total_examples": total_examples,
            "top1_hit_count": top1_hit_count,
            "top1_hit_rate": top1_hit_count / total_examples if total_examples else 0.0,
            "top1_error_count": top1_error_count,
            "analyzable_error_count": analyzable_error_count,
            "unique_pred_analyzable_error_count": unique_pred_analyzable_error_count,
            "ambiguous_pred_count": ambiguous_pred_count,
            "pred_sid_missing_count": pred_sid_missing_count,
        },
        "collaborative_gap_unique_pred_only": {
            "cf_target_better_count": cf_target_better_unique_count,
            "cf_target_better_rate_over_unique_analyzable_errors": cf_target_better_unique_count / unique_pred_analyzable_error_count if unique_pred_analyzable_error_count else 0.0,
            "cf_target_better_rate_over_all_examples": cf_target_better_unique_count / total_examples if total_examples else 0.0,
            "cf_pred_better_count": cf_pred_better_unique_count,
            "cf_tie_count": cf_tie_unique_count,
            "both_zero_count": both_zero_unique_count,
            "mean_cf_gap_target_minus_pred": safe_mean(cf_gaps_unique),
            "median_cf_gap_target_minus_pred": safe_median(cf_gaps_unique),
        },
        "collaborative_gap_with_ambiguous_pred": {
            "interpretation": {
                "bestcase": "compare target CF against the strongest CF candidate under the predicted SID; this is the most conservative estimate against the target",
                "avgcase": "compare target CF against the average CF across all candidate items under the predicted SID",
                "worstcase": "compare target CF against the weakest CF candidate under the predicted SID; this is the loosest estimate",
            },
            "bestcase": {
                "cf_target_better_count": cf_target_better_bestcase_count,
                "cf_target_better_rate_over_analyzable_errors": cf_target_better_bestcase_count / analyzable_error_count if analyzable_error_count else 0.0,
                "cf_target_better_rate_over_all_examples": cf_target_better_bestcase_count / total_examples if total_examples else 0.0,
                "cf_pred_better_count": cf_pred_better_bestcase_count,
                "cf_tie_count": cf_tie_bestcase_count,
                "both_zero_count": both_zero_bestcase_count,
                "mean_cf_gap_target_minus_pred": safe_mean(cf_gaps_bestcase),
                "median_cf_gap_target_minus_pred": safe_median(cf_gaps_bestcase),
            },
            "avgcase": {
                "cf_target_better_count": cf_target_better_avgcase_count,
                "cf_target_better_rate_over_analyzable_errors": cf_target_better_avgcase_count / analyzable_error_count if analyzable_error_count else 0.0,
                "cf_target_better_rate_over_all_examples": cf_target_better_avgcase_count / total_examples if total_examples else 0.0,
                "cf_pred_better_count": cf_pred_better_avgcase_count,
                "cf_tie_count": cf_tie_avgcase_count,
                "both_zero_count": both_zero_avgcase_count,
                "mean_cf_gap_target_minus_pred": safe_mean(cf_gaps_avgcase),
                "median_cf_gap_target_minus_pred": safe_median(cf_gaps_avgcase),
            },
            "worstcase": {
                "cf_target_better_count": cf_target_better_worstcase_count,
                "cf_target_better_rate_over_analyzable_errors": cf_target_better_worstcase_count / analyzable_error_count if analyzable_error_count else 0.0,
                "cf_target_better_rate_over_all_examples": cf_target_better_worstcase_count / total_examples if total_examples else 0.0,
                "cf_pred_better_count": cf_pred_better_worstcase_count,
                "cf_tie_count": cf_tie_worstcase_count,
                "both_zero_count": both_zero_worstcase_count,
                "mean_cf_gap_target_minus_pred": safe_mean(cf_gaps_worstcase),
                "median_cf_gap_target_minus_pred": safe_median(cf_gaps_worstcase),
            },
        },
        "same_prefix_errors": {
            "same_l1_error_count": same_l1_error_count,
            "same_l2_error_count": same_l2_error_count,
            "same_l1_target_better_unique_count": same_l1_target_better_unique_count,
            "same_l2_target_better_unique_count": same_l2_target_better_unique_count,
            "same_l1_target_better_rate_over_same_l1_unique_errors": same_l1_target_better_unique_count / same_l1_analyzable_unique_count if same_l1_analyzable_unique_count else 0.0,
            "same_l2_target_better_rate_over_same_l2_unique_errors": same_l2_target_better_unique_count / same_l2_analyzable_unique_count if same_l2_analyzable_unique_count else 0.0,
            "same_l1_target_better_bestcase_count": same_l1_target_better_bestcase_count,
            "same_l2_target_better_bestcase_count": same_l2_target_better_bestcase_count,
            "same_l1_target_better_rate_over_same_l1_all_analyzable_errors_bestcase": same_l1_target_better_bestcase_count / same_l1_analyzable_all_count if same_l1_analyzable_all_count else 0.0,
            "same_l2_target_better_rate_over_same_l2_all_analyzable_errors_bestcase": same_l2_target_better_bestcase_count / same_l2_analyzable_all_count if same_l2_analyzable_all_count else 0.0,
            "same_l1_target_better_rate_over_same_l1_all_analyzable_errors_avgcase": same_l1_target_better_avgcase_count / same_l1_analyzable_all_count if same_l1_analyzable_all_count else 0.0,
            "same_l2_target_better_rate_over_same_l2_all_analyzable_errors_avgcase": same_l2_target_better_avgcase_count / same_l2_analyzable_all_count if same_l2_analyzable_all_count else 0.0,
            "same_l1_target_better_rate_over_same_l1_all_analyzable_errors_worstcase": same_l1_target_better_worstcase_count / same_l1_analyzable_all_count if same_l1_analyzable_all_count else 0.0,
            "same_l2_target_better_rate_over_same_l2_all_analyzable_errors_worstcase": same_l2_target_better_worstcase_count / same_l2_analyzable_all_count if same_l2_analyzable_all_count else 0.0,
        },
        "text_similarity": {
            "mean_text_cosine_on_error": safe_mean(text_sims_error),
            "mean_text_cosine_when_cf_target_better_unique_only": safe_mean(text_sims_target_better_unique),
            "mean_text_cosine_when_cf_target_better_bestcase": safe_mean(text_sims_target_better_bestcase),
            "mean_text_cosine_on_same_l1_error": safe_mean(text_sims_same_l1),
            "mean_text_cosine_on_same_l2_error": safe_mean(text_sims_same_l2),
        },
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    examples_df = pd.DataFrame(example_rows)
    if not examples_df.empty:
        examples_df = examples_df.sort_values(
            by=["cf_gap_target_minus_pred_bestcase", "same_l2", "same_l1"],
            ascending=[False, False, False],
        )
    examples_df.to_csv(output_csv, index=False)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote summary to: {output_json}")
    print(f"Wrote examples to: {output_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose collaborative-gap errors from train/test/result files.")
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--result-json", required=True)
    parser.add_argument("--index-json", required=True)
    parser.add_argument("--item-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--emb-npy", default=None)
    parser.add_argument("--history-k", type=int, default=10)
    args = parser.parse_args()

    analyze(
        train_csv=Path(args.train_csv),
        test_csv=Path(args.test_csv),
        result_json=Path(args.result_json),
        index_json=Path(args.index_json),
        item_json=Path(args.item_json),
        output_json=Path(args.output_json),
        output_csv=Path(args.output_csv),
        emb_npy=Path(args.emb_npy) if args.emb_npy else None,
        history_k=args.history_k,
    )


if __name__ == "__main__":
    main()
