#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

import experiment_mgr_sid_stage2_interface_diagnostics as diag


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--baseline-index", required=True)
    parser.add_argument("--r202a-index", required=True)
    parser.add_argument("--candidate-index", required=True)
    parser.add_argument("--semantic-embedding", required=True)
    parser.add_argument("--baseline-train", required=True)
    parser.add_argument("--baseline-valid", required=True)
    parser.add_argument("--r202a-train", required=True)
    parser.add_argument("--r202a-valid", required=True)
    parser.add_argument("--candidate-train", required=True)
    parser.add_argument("--candidate-valid", required=True)
    parser.add_argument("--ambiguity-csv", required=True)
    parser.add_argument("--ambiguity-column", required=True)
    parser.add_argument("--base-ckpt", required=True)
    parser.add_argument("--candidate-ckpt", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def load_ambiguity_buckets(path: Path, column: str) -> dict[int, str]:
    df = pd.read_csv(path)
    df = df[["item_id", column]].copy()
    df[column] = df[column].fillna(0.0).clip(0.0, 1.0)
    q1 = float(df[column].quantile(1.0 / 3.0))
    q2 = float(df[column].quantile(2.0 / 3.0))
    buckets: dict[int, str] = {}
    for row in df.itertuples(index=False):
        value = float(getattr(row, column))
        if value <= q1:
            bucket = "easy"
        elif value <= q2:
            bucket = "medium"
        else:
            bucket = "hard"
        buckets[int(row.item_id)] = bucket
    return buckets


def bucket_pair_retention(
    baseline_groups: dict[str, set[int]],
    variant_sid_map: dict[int, str],
    level: int,
    bucket_map: dict[int, str],
) -> list[dict[str, Any]]:
    stats: dict[str, dict[str, int]] = {
        "easy": {"retained": 0, "total": 0},
        "medium": {"retained": 0, "total": 0},
        "hard": {"retained": 0, "total": 0},
    }
    for items in baseline_groups.values():
        item_list = sorted(items)
        for i, left in enumerate(item_list):
            left_bucket = bucket_map.get(left)
            if left_bucket not in stats:
                continue
            left_prefix = diag.parse_sid(variant_sid_map[left])[:level]
            for right in item_list[i + 1 :]:
                right_bucket = bucket_map.get(right)
                if right_bucket != left_bucket:
                    continue
                stats[left_bucket]["total"] += 1
                if left_prefix == diag.parse_sid(variant_sid_map[right])[:level]:
                    stats[left_bucket]["retained"] += 1
    rows: list[dict[str, Any]] = []
    for bucket, values in stats.items():
        total = values["total"]
        rows.append(
            {
                "bucket": bucket,
                "pair_count": total,
                "retained_count": values["retained"],
                "pair_retention": (values["retained"] / total) if total else 0.0,
            }
        )
    return rows


def fit_probe_with_seed(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    target_kind: str,
    fanout_map: dict[str, int],
    seed: int,
) -> tuple[float, float, float]:
    train_features = []
    train_targets = []
    valid_features = []
    valid_targets = []

    for _, row in train_df.iterrows():
        target_sid = diag.canonicalize_sid(row["item_sid"])
        a, b, c = diag.parse_sid(target_sid)
        history_sids = diag.parse_sequence(row["history_item_sid"])
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
        train_features.append(diag.sid_feature_dict(history_sids, cond))
        train_targets.append(target)

    for _, row in valid_df.iterrows():
        target_sid = diag.canonicalize_sid(row["item_sid"])
        a, b, c = diag.parse_sid(target_sid)
        history_sids = diag.parse_sequence(row["history_item_sid"])
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
        valid_features.append(diag.sid_feature_dict(history_sids, cond))
        valid_targets.append(target)

    vectorizer = DictVectorizer()
    x_train = vectorizer.fit_transform(train_features)
    x_valid = vectorizer.transform(valid_features)

    clf = SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-5, max_iter=120, tol=1e-4, random_state=seed)
    clf.fit(x_train, train_targets)
    pred = clf.predict(x_valid)

    valid_target_sids = [diag.canonicalize_sid(x) for x in valid_df["item_sid"].tolist()]
    valid_fanout = [fanout_map.get("|".join(diag.parse_sid(sid)[:2]), 0) for sid in valid_target_sids]
    hard_mask = np.asarray([v >= 4 for v in valid_fanout], dtype=bool)
    stable_mask = np.asarray([v <= 2 for v in valid_fanout], dtype=bool)

    overall = accuracy_score(valid_targets, pred)
    hard = accuracy_score(np.asarray(valid_targets)[hard_mask], np.asarray(pred)[hard_mask]) if hard_mask.any() else 0.0
    stable = accuracy_score(np.asarray(valid_targets)[stable_mask], np.asarray(pred)[stable_mask]) if stable_mask.any() else 0.0
    return float(overall), float(hard), float(stable)


def run_r304_multi_seed(
    variant_train_valid: dict[str, tuple[Path, Path]],
    sid_maps: dict[str, dict[int, str]],
    seeds: list[int],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for variant_name, (train_path, valid_path) in variant_train_valid.items():
        train_df = pd.read_csv(train_path)
        valid_df = pd.read_csv(valid_path)
        fanout_counter: dict[str, int] = {}
        for sid in sid_maps[variant_name].values():
            key = "|".join(diag.parse_sid(sid)[:2])
            fanout_counter[key] = fanout_counter.get(key, 0) + 1
        for target_kind in ["a", "b_given_a", "c_given_ab"]:
            values = [
                fit_probe_with_seed(train_df, valid_df, target_kind, fanout_counter, seed)
                for seed in seeds
            ]
            accs = [v[0] for v in values]
            hard_accs = [v[1] for v in values]
            stable_accs = [v[2] for v in values]
            rows.append(
                {
                    "variant": variant_name,
                    "target": target_kind,
                    "seed_list": seeds,
                    "accuracy_mean": float(np.mean(accs)),
                    "accuracy_std": float(np.std(accs)),
                    "hard_accuracy_mean": float(np.mean(hard_accs)),
                    "hard_accuracy_std": float(np.std(hard_accs)),
                    "stable_accuracy_mean": float(np.mean(stable_accs)),
                    "stable_accuracy_std": float(np.std(stable_accs)),
                    "n_train": int(len(train_df)),
                    "n_valid": int(len(valid_df)),
                }
            )
    return {"summary_rows": rows}


def run_r305_codebook_drift(base_ckpt: Path, candidate_ckpt: Path) -> dict[str, Any]:
    base = torch.load(base_ckpt, map_location="cpu", weights_only=False)["state_dict"]
    cand = torch.load(candidate_ckpt, map_location="cpu", weights_only=False)["state_dict"]
    rows: list[dict[str, Any]] = []
    for level in range(3):
        key = f"rq.vq_layers.{level}.embedding.weight"
        base_weight = base[key].float()
        cand_weight = cand[key].float()
        delta = cand_weight - base_weight
        rows.append(
            {
                "level": level + 1,
                "param": key,
                "mean_l2_sq_drift": float(torch.mean(delta.pow(2)).item()),
                "rms_drift": float(torch.sqrt(torch.mean(delta.pow(2))).item()),
                "relative_rms_drift": float(
                    torch.sqrt(torch.mean(delta.pow(2))).item()
                    / max(torch.sqrt(torch.mean(base_weight.pow(2))).item(), 1e-12)
                ),
            }
        )
    return {"summary_rows": rows}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sid_maps = {
        "current_v2": diag.load_index(Path(args.baseline_index)),
        "r202a": diag.load_index(Path(args.r202a_index)),
        args.candidate_name: diag.load_index(Path(args.candidate_index)),
    }

    r301 = diag.run_r301(
        baseline_name="current_v2",
        baseline_sid_map=sid_maps["current_v2"],
        variant_maps={k: v for k, v in sid_maps.items() if k != "current_v2"},
    )
    pd.DataFrame(r301["summary_rows"]).to_csv(out_dir / "R301_prefix_stability.csv", index=False)
    write_json(out_dir / "R301_prefix_stability.json", r301)

    bucket_map = load_ambiguity_buckets(Path(args.ambiguity_csv), args.ambiguity_column)
    baseline_l1 = diag.invert_prefix_map(sid_maps["current_v2"], level=1)
    baseline_l2 = diag.invert_prefix_map(sid_maps["current_v2"], level=2)
    bucket_rows = []
    for variant_name in ["r202a", args.candidate_name]:
        bucket_rows.extend(
            [
                {"variant": variant_name, "level": "l1", **row}
                for row in bucket_pair_retention(baseline_l1, sid_maps[variant_name], level=1, bucket_map=bucket_map)
            ]
        )
        bucket_rows.extend(
            [
                {"variant": variant_name, "level": "l2", **row}
                for row in bucket_pair_retention(baseline_l2, sid_maps[variant_name], level=2, bucket_map=bucket_map)
            ]
        )
    bucket_payload = {"summary_rows": bucket_rows}
    pd.DataFrame(bucket_rows).to_csv(out_dir / "R301b_prefix_stability_by_ambiguity.csv", index=False)
    write_json(out_dir / "R301b_prefix_stability_by_ambiguity.json", bucket_payload)

    semantic_embeddings = np.load(args.semantic_embedding).astype(np.float32)
    norms = np.linalg.norm(semantic_embeddings, axis=1, keepdims=True)
    semantic_embeddings = semantic_embeddings / np.clip(norms, 1e-12, None)
    r302 = diag.run_r302(
        embeddings=semantic_embeddings,
        variant_maps=sid_maps,
    )
    pd.DataFrame(r302["summary_rows"]).to_csv(out_dir / "R302_code_polysemy.csv", index=False)
    write_json(out_dir / "R302_code_polysemy.json", r302)

    r304 = run_r304_multi_seed(
        variant_train_valid={
            "current_v2": (Path(args.baseline_train), Path(args.baseline_valid)),
            "r202a": (Path(args.r202a_train), Path(args.r202a_valid)),
            args.candidate_name: (Path(args.candidate_train), Path(args.candidate_valid)),
        },
        sid_maps=sid_maps,
        seeds=[42, 43, 44],
    )
    pd.DataFrame(r304["summary_rows"]).to_csv(out_dir / "R304_learnability_probe.csv", index=False)
    write_json(out_dir / "R304_learnability_probe.json", r304)

    r305 = run_r305_codebook_drift(Path(args.base_ckpt), Path(args.candidate_ckpt))
    pd.DataFrame(r305["summary_rows"]).to_csv(out_dir / "R305_codebook_drift.csv", index=False)
    write_json(out_dir / "R305_codebook_drift.json", r305)


if __name__ == "__main__":
    main()
