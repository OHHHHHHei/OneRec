from __future__ import annotations

import ast
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from onerec.evaluate.semantic_id import canonicalize_semantic_id


def parse_id_list(value: Any) -> list[int]:
    if isinstance(value, list):
        return [int(v) for v in value]
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return []
    if not isinstance(parsed, list):
        return []
    result: list[int] = []
    for item in parsed:
        try:
            result.append(int(item))
        except (TypeError, ValueError):
            continue
    return result


def parse_prefix_parts(sid: str) -> tuple[str, str, str]:
    canonical = canonicalize_semantic_id(sid)
    if not canonical:
        return ("", "", "")
    parts = canonical.replace("><", "> <").split()
    if len(parts) != 3:
        return ("", "", "")
    return parts[0], parts[1], parts[2]


def build_pair_statistics(train_csv: Path, history_k: int) -> tuple[Counter, Counter]:
    df = pd.read_csv(train_csv)
    pair_count: Counter = Counter()
    hist_count: Counter = Counter()
    for _, row in df.iterrows():
        history = parse_id_list(row.get("history_item_id"))
        try:
            target = int(row["item_id"])
        except (TypeError, ValueError):
            continue
        if not history:
            continue
        recent = list(reversed(history[-history_k:]))
        for rank, hist_item in enumerate(recent, start=1):
            weight = 1.0 / rank
            pair_count[(hist_item, target)] += weight
            hist_count[hist_item] += weight
    return pair_count, hist_count


def collaborative_score(
    history: list[int],
    candidate: int,
    pair_count: Counter,
    hist_count: Counter,
    history_k: int,
) -> float:
    if not history:
        return 0.0
    score = 0.0
    recent = list(reversed(history[-history_k:]))
    for rank, hist_item in enumerate(recent, start=1):
        denom = hist_count.get(hist_item, 0.0)
        if denom <= 0.0:
            continue
        score += (pair_count.get((hist_item, candidate), 0.0) / denom) / rank
    return score


def load_sid_to_item_ids(index_json: Path) -> dict[str, list[int]]:
    import json

    with open(index_json, "r", encoding="utf-8") as handle:
        index_map = json.load(handle)
    sid_to_item_ids: dict[str, list[int]] = defaultdict(list)
    for item_id, sid_tokens in index_map.items():
        sid = canonicalize_semantic_id("".join(sid_tokens))
        sid_to_item_ids[sid].append(int(item_id))
    return dict(sid_to_item_ids)


def build_l2_leaf_counts(index_json: Path) -> dict[tuple[str, str], int]:
    import json

    with open(index_json, "r", encoding="utf-8") as handle:
        index_map = json.load(handle)
    prefix_to_leafs: dict[tuple[str, str], set[str]] = defaultdict(set)
    for sid_tokens in index_map.values():
        sid = canonicalize_semantic_id("".join(sid_tokens))
        a, b, c = parse_prefix_parts(sid)
        if a and b and c:
            prefix_to_leafs[(a, b)].add(c)
    return {prefix: len(leafs) for prefix, leafs in prefix_to_leafs.items()}


def build_ambiguity_prefixes(leaf_counts: dict[tuple[str, str], int], leaf_threshold: int) -> set[tuple[str, str]]:
    return {prefix for prefix, count in leaf_counts.items() if count >= leaf_threshold}


def sid_candidate_score(
    history: list[int],
    sid: str,
    sid_to_item_ids: dict[str, list[int]],
    pair_count: Counter,
    hist_count: Counter,
    history_k: int,
    sid_score_mode: str,
) -> float | None:
    item_ids = sid_to_item_ids.get(canonicalize_semantic_id(sid), [])
    if not item_ids:
        return None
    scores = [collaborative_score(history, item_id, pair_count, hist_count, history_k) for item_id in item_ids]
    if sid_score_mode == "best":
        return max(scores)
    if sid_score_mode == "worst":
        return min(scores)
    return sum(scores) / len(scores)


def reorder_subset_positions(
    predicts: list[str],
    history: list[int],
    positions: list[int],
    sid_to_item_ids: dict[str, list[int]],
    pair_count: Counter,
    hist_count: Counter,
    history_k: int,
    sid_score_mode: str,
) -> list[str]:
    if len(positions) <= 1:
        return list(predicts)
    scored: list[tuple[float, int, str]] = []
    for rank_in_subset, position in enumerate(positions):
        sid = predicts[position]
        score = sid_candidate_score(
            history,
            sid,
            sid_to_item_ids,
            pair_count,
            hist_count,
            history_k,
            sid_score_mode,
        )
        scored.append((float("-inf") if score is None else score, rank_in_subset, sid))
    scored.sort(key=lambda item: (-item[0], item[1]))
    reranked = list(predicts)
    for position, (_, _, sid) in zip(positions, scored):
        reranked[position] = sid
    return reranked


@dataclass
class CollaborativeReranker:
    pair_count: Counter
    hist_count: Counter
    sid_to_item_ids: dict[str, list[int]]
    history_k: int = 10
    sid_score_mode: str = "best"
    ambiguous_prefixes: set[tuple[str, str]] | None = None

    @classmethod
    def from_files(
        cls,
        train_csv: str | Path,
        index_json: str | Path,
        history_k: int = 10,
        sid_score_mode: str = "best",
        ambiguity_leaf_threshold: int = 8,
    ) -> "CollaborativeReranker":
        train_csv = Path(train_csv)
        index_json = Path(index_json)
        pair_count, hist_count = build_pair_statistics(train_csv, history_k)
        sid_to_item_ids = load_sid_to_item_ids(index_json)
        leaf_counts = build_l2_leaf_counts(index_json)
        ambiguous_prefixes = build_ambiguity_prefixes(leaf_counts, ambiguity_leaf_threshold)
        return cls(
            pair_count=pair_count,
            hist_count=hist_count,
            sid_to_item_ids=sid_to_item_ids,
            history_k=history_k,
            sid_score_mode=sid_score_mode,
            ambiguous_prefixes=ambiguous_prefixes,
        )

    def rerank_predict_list(self, predicts: list[str], history: list[int], mode: str) -> tuple[list[str], bool]:
        canonical_predicts = [canonicalize_semantic_id(pred) for pred in predicts]
        if not canonical_predicts:
            return canonical_predicts, False
        top1 = canonical_predicts[0]
        top1_a, top1_b, _ = parse_prefix_parts(top1)
        if mode == "baseline":
            return canonical_predicts, False
        if mode == "global":
            positions = list(range(len(canonical_predicts)))
            return (
                reorder_subset_positions(
                    canonical_predicts,
                    history,
                    positions,
                    self.sid_to_item_ids,
                    self.pair_count,
                    self.hist_count,
                    self.history_k,
                    self.sid_score_mode,
                ),
                True,
            )
        if mode == "same_l1":
            positions = [idx for idx, sid in enumerate(canonical_predicts) if parse_prefix_parts(sid)[0] == top1_a]
            reranked = reorder_subset_positions(
                canonical_predicts,
                history,
                positions,
                self.sid_to_item_ids,
                self.pair_count,
                self.hist_count,
                self.history_k,
                self.sid_score_mode,
            )
            return reranked, len(positions) > 1
        if mode == "same_l2":
            positions = [
                idx
                for idx, sid in enumerate(canonical_predicts)
                if parse_prefix_parts(sid)[:2] == (top1_a, top1_b)
            ]
            reranked = reorder_subset_positions(
                canonical_predicts,
                history,
                positions,
                self.sid_to_item_ids,
                self.pair_count,
                self.hist_count,
                self.history_k,
                self.sid_score_mode,
            )
            return reranked, len(positions) > 1
        if mode == "ambiguity_l2":
            prefix = (top1_a, top1_b)
            if prefix not in (self.ambiguous_prefixes or set()):
                return canonical_predicts, False
            positions = [
                idx
                for idx, sid in enumerate(canonical_predicts)
                if parse_prefix_parts(sid)[:2] == prefix
            ]
            reranked = reorder_subset_positions(
                canonical_predicts,
                history,
                positions,
                self.sid_to_item_ids,
                self.pair_count,
                self.hist_count,
                self.history_k,
                self.sid_score_mode,
            )
            return reranked, len(positions) > 1
        raise ValueError(f"Unsupported mode: {mode}")
