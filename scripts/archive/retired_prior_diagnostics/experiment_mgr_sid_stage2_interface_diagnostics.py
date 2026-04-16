#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score


def canonicalize_sid(text: object) -> str:
    if text is None:
        return ""
    value = str(text).strip(" \n\r\t\"")
    start = value.find("<a_")
    if start == -1:
        return value
    end = value.find(">", start)
    if end == -1:
        return value
    import re

    match = re.search(r"<a_\d+><b_\d+><c_\d+>", value)
    return match.group(0) if match else value


def parse_sid(sid: str) -> tuple[str, str, str]:
    import re

    parts = re.findall(r"<[abc]_\d+>", canonicalize_sid(sid))
    if len(parts) != 3:
        return ("", "", "")
    return tuple(parts)  # type: ignore[return-value]


def parse_sequence(text: str) -> list[str]:
    value = text
    if isinstance(text, float) and math.isnan(text):
        return []
    try:
        parsed = ast.literal_eval(str(value))
    except Exception:
        return []
    if not isinstance(parsed, list):
        return []
    return [canonicalize_sid(x) for x in parsed]


def load_index(path: Path) -> dict[int, str]:
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return {int(k): "".join(v) for k, v in raw.items()}


def invert_prefix_map(sid_map: dict[int, str], level: int) -> dict[str, set[int]]:
    groups: dict[str, set[int]] = defaultdict(set)
    for item_id, sid in sid_map.items():
        a, b, c = parse_sid(sid)
        if level == 1:
            key = a
        elif level == 2:
            key = f"{a}|{b}"
        else:
            key = f"{a}|{b}|{c}"
        groups[key].add(item_id)
    return groups


def prefix_sets_for_items(groups: dict[str, set[int]]) -> dict[int, set[int]]:
    out: dict[int, set[int]] = {}
    for items in groups.values():
        for item_id in items:
            out[item_id] = items
    return out


def safe_jaccard(lhs: set[int], rhs: set[int]) -> float:
    lhs = set(lhs)
    rhs = set(rhs)
    lhs.discard(next(iter(lhs)) if False else -1)
    union = lhs | rhs
    if not union:
        return 1.0
    return len(lhs & rhs) / len(union)


def per_item_neighbor_metrics(
    baseline_groups: dict[str, set[int]],
    variant_groups: dict[str, set[int]],
) -> dict[int, dict[str, float]]:
    baseline_membership = prefix_sets_for_items(baseline_groups)
    variant_membership = prefix_sets_for_items(variant_groups)
    item_ids = sorted(set(baseline_membership) & set(variant_membership))
    metrics: dict[int, dict[str, float]] = {}
    for item_id in item_ids:
        baseline_neighbors = set(baseline_membership[item_id])
        variant_neighbors = set(variant_membership[item_id])
        baseline_neighbors.discard(item_id)
        variant_neighbors.discard(item_id)
        inter = len(baseline_neighbors & variant_neighbors)
        union = len(baseline_neighbors | variant_neighbors)
        recall = inter / len(baseline_neighbors) if baseline_neighbors else 1.0
        precision = inter / len(variant_neighbors) if variant_neighbors else 1.0
        jaccard = inter / union if union else 1.0
        metrics[item_id] = {
            "baseline_size": float(len(baseline_neighbors)),
            "variant_size": float(len(variant_neighbors)),
            "recall": float(recall),
            "precision": float(precision),
            "jaccard": float(jaccard),
        }
    return metrics


def prefix_pair_retention(
    baseline_groups: dict[str, set[int]],
    variant_sid_map: dict[int, str],
    level: int,
) -> float:
    retained = 0
    total = 0
    for items in baseline_groups.values():
        item_list = sorted(items)
        for i, left in enumerate(item_list):
            left_prefix = parse_sid(variant_sid_map[left])[:level]
            for right in item_list[i + 1 :]:
                total += 1
                if left_prefix == parse_sid(variant_sid_map[right])[:level]:
                    retained += 1
    if total == 0:
        return 1.0
    return retained / total


def run_r301(
    baseline_name: str,
    baseline_sid_map: dict[int, str],
    variant_maps: dict[str, dict[int, str]],
) -> dict[str, Any]:
    baseline_l1 = invert_prefix_map(baseline_sid_map, level=1)
    baseline_l2 = invert_prefix_map(baseline_sid_map, level=2)

    rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    for variant_name, variant_sid_map in variant_maps.items():
        item_ids = sorted(set(baseline_sid_map) & set(variant_sid_map))
        changed_l1 = 0
        changed_l2 = 0
        changed_full = 0
        for item_id in item_ids:
            base_parts = parse_sid(baseline_sid_map[item_id])
            var_parts = parse_sid(variant_sid_map[item_id])
            if base_parts[:1] != var_parts[:1]:
                changed_l1 += 1
            if base_parts[:2] != var_parts[:2]:
                changed_l2 += 1
            if base_parts != var_parts:
                changed_full += 1

        variant_l1 = invert_prefix_map(variant_sid_map, level=1)
        variant_l2 = invert_prefix_map(variant_sid_map, level=2)
        item_l1 = per_item_neighbor_metrics(baseline_l1, variant_l1)
        item_l2 = per_item_neighbor_metrics(baseline_l2, variant_l2)

        row = {
            "baseline": baseline_name,
            "variant": variant_name,
            "n_items": len(item_ids),
            "changed_l1_rate": changed_l1 / len(item_ids),
            "changed_l2_rate": changed_l2 / len(item_ids),
            "changed_full_sid_rate": changed_full / len(item_ids),
            "l1_pair_retention": prefix_pair_retention(baseline_l1, variant_sid_map, level=1),
            "l2_pair_retention": prefix_pair_retention(baseline_l2, variant_sid_map, level=2),
            "mean_l1_neighbor_jaccard": float(np.mean([m["jaccard"] for m in item_l1.values()])),
            "mean_l2_neighbor_jaccard": float(np.mean([m["jaccard"] for m in item_l2.values()])),
            "mean_l1_neighbor_recall": float(np.mean([m["recall"] for m in item_l1.values()])),
            "mean_l2_neighbor_recall": float(np.mean([m["recall"] for m in item_l2.values()])),
        }
        rows.append(row)
        details[variant_name] = {
            "per_item_l1": item_l1,
            "per_item_l2": item_l2,
            "summary": row,
        }
    return {"summary_rows": rows, "details": details}


def cosine_spread(embeddings: np.ndarray) -> float:
    if embeddings.shape[0] <= 1:
        return 0.0
    center = embeddings.mean(axis=0, keepdims=True)
    center_norm = np.linalg.norm(center, axis=1, keepdims=True)
    center = center / np.clip(center_norm, 1e-12, None)
    sims = embeddings @ center.T
    return float(np.mean(1.0 - sims.reshape(-1)))


def pairwise_centroid_drift(centers: list[np.ndarray]) -> float:
    if len(centers) <= 1:
        return 0.0
    values: list[float] = []
    for i, left in enumerate(centers):
        for right in centers[i + 1 :]:
            values.append(float(1.0 - np.dot(left, right)))
    return float(np.mean(values)) if values else 0.0


def run_r302(
    embeddings: np.ndarray,
    variant_maps: dict[str, dict[int, str]],
) -> dict[str, Any]:
    summary_rows: list[dict[str, Any]] = []
    examples: dict[str, Any] = {}

    for variant_name, sid_map in variant_maps.items():
        token_items: dict[str, dict[str, list[int]]] = {
            "a": defaultdict(list),
            "b": defaultdict(list),
            "c": defaultdict(list),
        }
        b_parent_items: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
        c_parent_items: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))

        for item_id, sid in sid_map.items():
            a, b, c = parse_sid(sid)
            token_items["a"][a].append(item_id)
            token_items["b"][b].append(item_id)
            token_items["c"][c].append(item_id)
            b_parent_items[b][a].append(item_id)
            c_parent_items[c][f"{a}|{b}"].append(item_id)

        for level_name in ["a", "b", "c"]:
            rows_for_level = []
            for token, items in token_items[level_name].items():
                token_emb = embeddings[np.asarray(items, dtype=np.int32)]
                rows_for_level.append(
                    {
                        "token": token,
                        "count": len(items),
                        "semantic_spread": cosine_spread(token_emb),
                    }
                )
            level_df = pd.DataFrame(rows_for_level).sort_values(["count", "semantic_spread"], ascending=[False, False])
            weighted_mean = float(np.average(level_df["semantic_spread"], weights=level_df["count"]))
            summary_rows.append(
                {
                    "variant": variant_name,
                    "level": level_name,
                    "token_count": int(len(level_df)),
                    "weighted_mean_semantic_spread": weighted_mean,
                    "median_semantic_spread": float(level_df["semantic_spread"].median()),
                    "p90_semantic_spread": float(level_df["semantic_spread"].quantile(0.9)),
                    "mean_reuse_count": float(level_df["count"].mean()),
                }
            )
            examples[f"{variant_name}_{level_name}_top_spread"] = level_df.head(10).to_dict(orient="records")

        drift_rows = []
        for token, parent_map in b_parent_items.items():
            if len(parent_map) < 2:
                continue
            centers = []
            total = 0
            for items in parent_map.values():
                token_emb = embeddings[np.asarray(items, dtype=np.int32)]
                center = token_emb.mean(axis=0)
                center = center / np.clip(np.linalg.norm(center), 1e-12, None)
                centers.append(center)
                total += len(items)
            drift_rows.append(
                {
                    "token": token,
                    "parent_count": len(parent_map),
                    "item_count": total,
                    "prefix_conditioned_drift": pairwise_centroid_drift(centers),
                }
            )
        b_drift_df = pd.DataFrame(drift_rows)

        drift_rows = []
        for token, parent_map in c_parent_items.items():
            if len(parent_map) < 2:
                continue
            centers = []
            total = 0
            for items in parent_map.values():
                token_emb = embeddings[np.asarray(items, dtype=np.int32)]
                center = token_emb.mean(axis=0)
                center = center / np.clip(np.linalg.norm(center), 1e-12, None)
                centers.append(center)
                total += len(items)
            drift_rows.append(
                {
                    "token": token,
                    "parent_count": len(parent_map),
                    "item_count": total,
                    "prefix_conditioned_drift": pairwise_centroid_drift(centers),
                }
            )
        c_drift_df = pd.DataFrame(drift_rows)

        summary_rows.extend(
            [
                {
                    "variant": variant_name,
                    "level": "b_prefix_drift",
                    "token_count": int(len(b_drift_df)),
                    "weighted_mean_semantic_spread": float(np.average(b_drift_df["prefix_conditioned_drift"], weights=b_drift_df["item_count"]))
                    if len(b_drift_df)
                    else 0.0,
                    "median_semantic_spread": float(b_drift_df["prefix_conditioned_drift"].median()) if len(b_drift_df) else 0.0,
                    "p90_semantic_spread": float(b_drift_df["prefix_conditioned_drift"].quantile(0.9)) if len(b_drift_df) else 0.0,
                    "mean_reuse_count": float(b_drift_df["item_count"].mean()) if len(b_drift_df) else 0.0,
                },
                {
                    "variant": variant_name,
                    "level": "c_prefix_drift",
                    "token_count": int(len(c_drift_df)),
                    "weighted_mean_semantic_spread": float(np.average(c_drift_df["prefix_conditioned_drift"], weights=c_drift_df["item_count"]))
                    if len(c_drift_df)
                    else 0.0,
                    "median_semantic_spread": float(c_drift_df["prefix_conditioned_drift"].median()) if len(c_drift_df) else 0.0,
                    "p90_semantic_spread": float(c_drift_df["prefix_conditioned_drift"].quantile(0.9)) if len(c_drift_df) else 0.0,
                    "mean_reuse_count": float(c_drift_df["item_count"].mean()) if len(c_drift_df) else 0.0,
                },
            ]
        )
        examples[f"{variant_name}_b_drift_top"] = (
            b_drift_df.sort_values(["prefix_conditioned_drift", "item_count"], ascending=[False, False]).head(10).to_dict(orient="records")
            if len(b_drift_df)
            else []
        )
        examples[f"{variant_name}_c_drift_top"] = (
            c_drift_df.sort_values(["prefix_conditioned_drift", "item_count"], ascending=[False, False]).head(10).to_dict(orient="records")
            if len(c_drift_df)
            else []
        )
    return {"summary_rows": summary_rows, "examples": examples}


def run_r303(
    topk_csv: Path,
    baseline_sid_map: dict[int, str],
    variant_sid_map: dict[int, str],
    prefix_detail: dict[str, Any],
) -> dict[str, Any]:
    df = pd.read_csv(topk_csv)
    item_l1 = {
        int(k): v for k, v in prefix_detail["per_item_l1"].items()
    }
    item_l2 = {
        int(k): v for k, v in prefix_detail["per_item_l2"].items()
    }

    changed_rows = []
    for item_id in sorted(set(df["item_id"])):
        item_id = int(item_id)
        base = parse_sid(baseline_sid_map[item_id])
        var = parse_sid(variant_sid_map[item_id])
        changed_rows.append(
            {
                "item_id": item_id,
                "changed_l1": int(base[:1] != var[:1]),
                "changed_l2": int(base[:2] != var[:2]),
                "changed_full_sid": int(base != var),
                "l1_neighbor_jaccard": item_l1[item_id]["jaccard"],
                "l2_neighbor_jaccard": item_l2[item_id]["jaccard"],
                "l1_neighbor_recall": item_l1[item_id]["recall"],
                "l2_neighbor_recall": item_l2[item_id]["recall"],
            }
        )
    change_df = pd.DataFrame(changed_rows)
    joined = df.merge(change_df, on="item_id", how="left")

    summary_rows: list[dict[str, Any]] = []
    for cutoff in [1, 3, 5, 10, 20]:
        improved = joined[joined[f"improved_at_{cutoff}"] == 1]
        worsened = joined[joined[f"worsened_at_{cutoff}"] == 1]
        unchanged = joined[(joined[f"improved_at_{cutoff}"] == 0) & (joined[f"worsened_at_{cutoff}"] == 0)]
        for label, subdf in [("improved", improved), ("worsened", worsened), ("unchanged", unchanged)]:
            if len(subdf) == 0:
                summary_rows.append(
                    {
                        "cutoff": cutoff,
                        "group": label,
                        "count": 0,
                        "changed_l1_rate": 0.0,
                        "changed_l2_rate": 0.0,
                        "changed_full_sid_rate": 0.0,
                        "mean_l1_neighbor_jaccard": 0.0,
                        "mean_l2_neighbor_jaccard": 0.0,
                        "mean_baseline_target_l2_fanout": 0.0,
                    }
                )
                continue
            summary_rows.append(
                {
                    "cutoff": cutoff,
                    "group": label,
                    "count": int(len(subdf)),
                    "changed_l1_rate": float(subdf["changed_l1"].mean()),
                    "changed_l2_rate": float(subdf["changed_l2"].mean()),
                    "changed_full_sid_rate": float(subdf["changed_full_sid"].mean()),
                    "mean_l1_neighbor_jaccard": float(subdf["l1_neighbor_jaccard"].mean()),
                    "mean_l2_neighbor_jaccard": float(subdf["l2_neighbor_jaccard"].mean()),
                    "mean_baseline_target_l2_fanout": float(subdf["baseline_target_l2_fanout"].mean()),
                }
            )
    return {"summary_rows": summary_rows}


@dataclass
class ProbeResult:
    variant: str
    target: str
    accuracy: float
    hard_accuracy: float
    stable_accuracy: float
    n_train: int
    n_valid: int


def sid_feature_dict(history_sids: list[str], condition_tokens: list[str] | None = None) -> dict[str, float]:
    features: dict[str, float] = {}
    for sid in history_sids:
        a, b, c = parse_sid(sid)
        if a:
            features[f"tok:{a}"] = features.get(f"tok:{a}", 0.0) + 1.0
        if b:
            features[f"tok:{b}"] = features.get(f"tok:{b}", 0.0) + 1.0
        if c:
            features[f"tok:{c}"] = features.get(f"tok:{c}", 0.0) + 1.0
        if a and b:
            features[f"pair:{a}|{b}"] = features.get(f"pair:{a}|{b}", 0.0) + 1.0
    for token in condition_tokens or []:
        features[f"cond:{token}"] = 1.0
    return features


def fit_probe(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    target_kind: str,
    fanout_map: dict[str, int],
) -> tuple[float, float, float]:
    train_features = []
    train_targets = []
    valid_features = []
    valid_targets = []

    for _, row in train_df.iterrows():
        target_sid = canonicalize_sid(row["item_sid"])
        a, b, c = parse_sid(target_sid)
        history_sids = parse_sequence(row["history_item_sid"])
        if target_kind == "a":
            target = a
            cond = []
        elif target_kind == "b_given_a":
            target = b
            cond = [a]
        elif target_kind == "c_given_ab":
            target = c
            cond = [a, b]
        else:
            raise ValueError(target_kind)
        train_features.append(sid_feature_dict(history_sids, cond))
        train_targets.append(target)

    for _, row in valid_df.iterrows():
        target_sid = canonicalize_sid(row["item_sid"])
        a, b, c = parse_sid(target_sid)
        history_sids = parse_sequence(row["history_item_sid"])
        if target_kind == "a":
            target = a
            cond = []
        elif target_kind == "b_given_a":
            target = b
            cond = [a]
        elif target_kind == "c_given_ab":
            target = c
            cond = [a, b]
        else:
            raise ValueError(target_kind)
        valid_features.append(sid_feature_dict(history_sids, cond))
        valid_targets.append(target)

    vectorizer = DictVectorizer()
    x_train = vectorizer.fit_transform(train_features)
    x_valid = vectorizer.transform(valid_features)

    clf = SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-5, max_iter=120, tol=1e-4, random_state=42)
    clf.fit(x_train, train_targets)
    pred = clf.predict(x_valid)

    valid_target_sids = [canonicalize_sid(x) for x in valid_df["item_sid"].tolist()]
    valid_fanout = [fanout_map.get("|".join(parse_sid(sid)[:2]), 0) for sid in valid_target_sids]
    hard_mask = np.asarray([v >= 4 for v in valid_fanout], dtype=bool)
    stable_mask = np.asarray([v <= 2 for v in valid_fanout], dtype=bool)

    overall = accuracy_score(valid_targets, pred)
    hard = accuracy_score(np.asarray(valid_targets)[hard_mask], np.asarray(pred)[hard_mask]) if hard_mask.any() else 0.0
    stable = accuracy_score(np.asarray(valid_targets)[stable_mask], np.asarray(pred)[stable_mask]) if stable_mask.any() else 0.0
    return float(overall), float(hard), float(stable)


def run_r304(
    variant_train_valid: dict[str, tuple[Path, Path]],
    sid_maps: dict[str, dict[int, str]],
) -> dict[str, Any]:
    summary_rows: list[dict[str, Any]] = []
    for variant_name, (train_path, valid_path) in variant_train_valid.items():
        train_df = pd.read_csv(train_path)
        valid_df = pd.read_csv(valid_path)
        fanout_counter = Counter()
        for sid in sid_maps[variant_name].values():
            fanout_counter["|".join(parse_sid(sid)[:2])] += 1
        for target_kind in ["a", "b_given_a", "c_given_ab"]:
            acc, hard_acc, stable_acc = fit_probe(train_df, valid_df, target_kind, fanout_counter)
            summary_rows.append(
                {
                    "variant": variant_name,
                    "target": target_kind,
                    "accuracy": acc,
                    "hard_accuracy_l2_ge_4": hard_acc,
                    "stable_accuracy_l2_le_2": stable_acc,
                    "n_train": int(len(train_df)),
                    "n_valid": int(len(valid_df)),
                }
            )
    return {"summary_rows": summary_rows}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-index", required=True)
    parser.add_argument("--r202a-index", required=True)
    parser.add_argument("--r202b-index", required=True)
    parser.add_argument("--r205-index", required=True)
    parser.add_argument("--strongest-index", default="")
    parser.add_argument("--semantic-embedding", required=True)
    parser.add_argument("--r208-topk-csv", required=True)
    parser.add_argument("--baseline-train", required=True)
    parser.add_argument("--baseline-valid", required=True)
    parser.add_argument("--r202a-train", required=True)
    parser.add_argument("--r202a-valid", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sid_maps = {
        "current_v2": load_index(Path(args.baseline_index)),
        "r202a": load_index(Path(args.r202a_index)),
        "r202b_r075": load_index(Path(args.r202b_index)),
        "r205": load_index(Path(args.r205_index)),
    }
    if args.strongest_index:
        sid_maps["strongest_original"] = load_index(Path(args.strongest_index))

    r301 = run_r301(
        baseline_name="current_v2",
        baseline_sid_map=sid_maps["current_v2"],
        variant_maps={k: v for k, v in sid_maps.items() if k != "current_v2"},
    )
    pd.DataFrame(r301["summary_rows"]).to_csv(out_dir / "R301_prefix_stability.csv", index=False)
    write_json(out_dir / "R301_prefix_stability.json", r301)

    semantic_embeddings = np.load(args.semantic_embedding).astype(np.float32)
    norms = np.linalg.norm(semantic_embeddings, axis=1, keepdims=True)
    semantic_embeddings = semantic_embeddings / np.clip(norms, 1e-12, None)
    r302 = run_r302(
        embeddings=semantic_embeddings,
        variant_maps={k: v for k, v in sid_maps.items() if k in {"current_v2", "r202a", "strongest_original"}},
    )
    pd.DataFrame(r302["summary_rows"]).to_csv(out_dir / "R302_code_polysemy.csv", index=False)
    write_json(out_dir / "R302_code_polysemy.json", r302)

    r303 = run_r303(
        topk_csv=Path(args.r208_topk_csv),
        baseline_sid_map=sid_maps["current_v2"],
        variant_sid_map=sid_maps["r202a"],
        prefix_detail=r301["details"]["r202a"],
    )
    pd.DataFrame(r303["summary_rows"]).to_csv(out_dir / "R303_transfer_attribution.csv", index=False)
    write_json(out_dir / "R303_transfer_attribution.json", r303)

    r304 = run_r304(
        variant_train_valid={
            "current_v2": (Path(args.baseline_train), Path(args.baseline_valid)),
            "r202a": (Path(args.r202a_train), Path(args.r202a_valid)),
        },
        sid_maps={k: sid_maps[k] for k in ["current_v2", "r202a"]},
    )
    pd.DataFrame(r304["summary_rows"]).to_csv(out_dir / "R304_learnability_probe.csv", index=False)
    write_json(out_dir / "R304_learnability_probe.json", r304)


if __name__ == "__main__":
    main()
