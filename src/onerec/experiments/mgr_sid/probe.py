from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from onerec.evaluate.collaborative_rerank import parse_prefix_parts
from onerec.evaluate.semantic_id import canonicalize_semantic_id
from onerec.experiments.mgr_sid.graph_bank import CommunityGraphView, SparseGraphView, parse_id_list

GraphView = SparseGraphView | CommunityGraphView


def load_sid_to_item_ids(index_json: Path) -> dict[str, list[int]]:
    raw = json.loads(index_json.read_text(encoding="utf-8"))
    sid_to_items: dict[str, list[int]] = {}
    for item_id, sid_tokens in raw.items():
        sid = canonicalize_semantic_id("".join(sid_tokens))
        sid_to_items.setdefault(sid, []).append(int(item_id))
    return sid_to_items


@dataclass
class BucketStats:
    count: int = 0
    coverage_count: int = 0
    target_better_count: int = 0

    def to_dict(self) -> dict[str, float | int]:
        rate = self.target_better_count / self.count if self.count else 0.0
        coverage = self.coverage_count / self.count if self.count else 0.0
        return {
            "count": self.count,
            "target_better_count": self.target_better_count,
            "target_better_rate": rate,
            "coverage_count": self.coverage_count,
            "coverage": coverage,
        }


def evaluate_view(
    test_df: pd.DataFrame,
    result_data: list[dict[str, Any]],
    sid_to_items: dict[str, list[int]],
    view: GraphView,
    history_k: int,
    max_examples: int | None = None,
) -> dict[str, Any]:
    buckets = {
        "all": BucketStats(),
        "same_l1": BucketStats(),
        "same_l2": BucketStats(),
    }
    top1_hit_count = 0
    missing_sid_count = 0
    evaluated = 0

    for idx, (_, row) in enumerate(test_df.iterrows()):
        if max_examples is not None and evaluated >= max_examples:
            break
        result = result_data[idx]
        predicts = result.get("predict", [])
        pred_sid = canonicalize_semantic_id(predicts[0] if predicts else "")
        target_sid = canonicalize_semantic_id(row["item_sid"])
        if pred_sid == target_sid:
            top1_hit_count += 1
            continue
        pred_candidates = sid_to_items.get(pred_sid, [])
        if not pred_candidates:
            missing_sid_count += 1
            continue

        evaluated += 1
        history = parse_id_list(row["history_item_id"])
        target_item = int(row["item_id"])
        target_score = view.score(history, target_item, history_k=history_k)
        pred_best_score = max(view.score(history, candidate, history_k=history_k) for candidate in pred_candidates)
        has_signal = (target_score > 0.0) or (pred_best_score > 0.0)

        bucket_names = ["all"]
        pred_l1, pred_l2, _ = parse_prefix_parts(pred_sid)
        tgt_l1, tgt_l2, _ = parse_prefix_parts(target_sid)
        if pred_l1 and pred_l1 == tgt_l1:
            bucket_names.append("same_l1")
        if pred_l1 == tgt_l1 and pred_l2 == tgt_l2:
            bucket_names.append("same_l2")

        for bucket_name in bucket_names:
            bucket = buckets[bucket_name]
            bucket.count += 1
            if has_signal:
                bucket.coverage_count += 1
            if target_score > pred_best_score:
                bucket.target_better_count += 1

    return {
        "view_name": view.name,
        "view_metadata": view.metadata,
        "evaluated_top1_errors": evaluated,
        "top1_hit_count": top1_hit_count,
        "pred_sid_missing_count": missing_sid_count,
        "buckets": {name: stats.to_dict() for name, stats in buckets.items()},
    }
