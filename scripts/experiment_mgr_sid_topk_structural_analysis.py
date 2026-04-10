#!/usr/bin/env python
from __future__ import annotations

import argparse
import ast
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

import pandas as pd


TOPKS = [1, 3, 5, 10, 20, 50]
MAIN_TOPKS = [1, 3, 5, 10]


def canonicalize_sid(text: object) -> str:
    if text is None:
        return ""
    value = str(text).strip(" \n\r\t\"")
    import re

    match = re.search(r"<a_\d+><b_\d+><c_\d+>", value)
    return match.group(0) if match else value


def parse_sid(sid: str) -> tuple[str, str, str]:
    import re

    parts = re.findall(r"<[abc]_\d+>", canonicalize_sid(sid))
    if len(parts) != 3:
        return ("", "", "")
    return tuple(parts)  # type: ignore[return-value]


def lcp_len(lhs: str, rhs: str) -> int:
    left = parse_sid(lhs)
    right = parse_sid(rhs)
    score = 0
    for x, y in zip(left, right):
        if x == y and x:
            score += 1
        else:
            break
    return score


def topk_hit(predictions: list[str], target_sid: str, k: int) -> bool:
    return target_sid in predictions[: min(k, len(predictions))]


def best_lcp_in_topk(predictions: list[str], target_sid: str, k: int) -> int:
    subset = predictions[: min(k, len(predictions))]
    return max((lcp_len(candidate, target_sid) for candidate in subset), default=0)


def target_rank(predictions: list[str], target_sid: str) -> int | None:
    try:
        return predictions.index(target_sid) + 1
    except ValueError:
        return None


def rank_bucket(rank: int | None) -> str:
    if rank is None:
        return ">50"
    if rank == 1:
        return "1"
    if rank <= 3:
        return "2-3"
    if rank <= 5:
        return "4-5"
    if rank <= 10:
        return "6-10"
    if rank <= 20:
        return "11-20"
    if rank <= 50:
        return "21-50"
    return ">50"


def fanout_bucket(value: int) -> str:
    if value <= 2:
        return "l2<=2"
    if value == 3:
        return "l2=3"
    return "l2>=4"


def load_index(path: Path) -> dict[int, tuple[str, str, str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {int(k): tuple(v[:3]) for k, v in raw.items()}


def build_index_stats(index_map: dict[int, tuple[str, str, str]]) -> dict[int, dict[str, int | str]]:
    sid_counts: Counter[str] = Counter()
    l1_counts: Counter[str] = Counter()
    l2_counts: Counter[tuple[str, str]] = Counter()
    for tokens in index_map.values():
        sid = "".join(tokens)
        a, b, _ = tokens
        sid_counts[sid] += 1
        l1_counts[a] += 1
        l2_counts[(a, b)] += 1

    stats: dict[int, dict[str, int | str]] = {}
    for item_id, tokens in index_map.items():
        sid = "".join(tokens)
        a, b, _ = tokens
        stats[item_id] = {
            "sid": sid,
            "l1": a,
            "l2_prefix": f"{a}{b}",
            "collision_group_size": sid_counts[sid],
            "l1_fanout": l1_counts[a],
            "l2_fanout": l2_counts[(a, b)],
        }
    return stats


def load_results(path: Path) -> list[dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: list[dict] = []
    for row in raw:
        out.append(
            {
                "input": row.get("input", ""),
                "output": canonicalize_sid(row.get("output", "")),
                "predict": [canonicalize_sid(v) for v in row.get("predict", [])],
            }
        )
    return out


def parse_history_ids(text: str) -> list[int]:
    try:
        values = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return []
    if not isinstance(values, list):
        return []
    return [int(v) for v in values]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Top-k structural comparison for baseline vs hierarchy evaluate results.")
    parser.add_argument("--baseline-result", required=True)
    parser.add_argument("--hierarchy-result", required=True)
    parser.add_argument("--baseline-test-csv", required=True)
    parser.add_argument("--hierarchy-test-csv", required=True)
    parser.add_argument("--baseline-index", required=True)
    parser.add_argument("--hierarchy-index", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-csv", required=True)
    return parser.parse_args()


def build_rows(args: argparse.Namespace) -> list[dict]:
    baseline_result = load_results(Path(args.baseline_result))
    hierarchy_result = load_results(Path(args.hierarchy_result))
    baseline_test = pd.read_csv(args.baseline_test_csv)
    hierarchy_test = pd.read_csv(args.hierarchy_test_csv)
    baseline_stats = build_index_stats(load_index(Path(args.baseline_index)))
    hierarchy_stats = build_index_stats(load_index(Path(args.hierarchy_index)))

    if not (len(baseline_result) == len(hierarchy_result) == len(baseline_test) == len(hierarchy_test)):
        raise ValueError("Result/test lengths do not align.")

    rows: list[dict] = []
    for idx in range(len(baseline_result)):
        b_res = baseline_result[idx]
        h_res = hierarchy_result[idx]
        b_test = baseline_test.iloc[idx]
        h_test = hierarchy_test.iloc[idx]

        item_id = int(b_test["item_id"])
        if item_id != int(h_test["item_id"]):
            raise ValueError(f"Mismatched item_id at row {idx}: {item_id} vs {int(h_test['item_id'])}")

        if str(b_test["item_title"]) != str(h_test["item_title"]):
            raise ValueError(f"Mismatched item_title at row {idx}")

        b_target_sid = baseline_stats[item_id]["sid"]
        h_target_sid = hierarchy_stats[item_id]["sid"]
        if b_target_sid != b_res["output"]:
            raise ValueError(f"Baseline target SID mismatch at row {idx}")
        if h_target_sid != h_res["output"]:
            raise ValueError(f"Hierarchy target SID mismatch at row {idx}")

        b_preds = b_res["predict"]
        h_preds = h_res["predict"]
        b_rank = target_rank(b_preds, b_target_sid)
        h_rank = target_rank(h_preds, h_target_sid)

        row = {
            "row_id": idx,
            "item_id": item_id,
            "item_title": str(b_test["item_title"]),
            "user_id": str(b_test["user_id"]),
            "history_len": len(parse_history_ids(str(b_test["history_item_id"]))),
            "baseline_target_sid": b_target_sid,
            "hierarchy_target_sid": h_target_sid,
            "baseline_rank": b_rank if b_rank is not None else 999,
            "hierarchy_rank": h_rank if h_rank is not None else 999,
            "baseline_rank_bucket": rank_bucket(b_rank),
            "hierarchy_rank_bucket": rank_bucket(h_rank),
            "baseline_target_l2_fanout": int(baseline_stats[item_id]["l2_fanout"]),
            "hierarchy_target_l2_fanout": int(hierarchy_stats[item_id]["l2_fanout"]),
            "baseline_target_l1_fanout": int(baseline_stats[item_id]["l1_fanout"]),
            "hierarchy_target_l1_fanout": int(hierarchy_stats[item_id]["l1_fanout"]),
            "baseline_collision_group_size": int(baseline_stats[item_id]["collision_group_size"]),
            "hierarchy_collision_group_size": int(hierarchy_stats[item_id]["collision_group_size"]),
            "baseline_fanout_bucket": fanout_bucket(int(baseline_stats[item_id]["l2_fanout"])),
            "baseline_pred1_lcp": lcp_len(b_preds[0] if b_preds else "", b_target_sid),
            "hierarchy_pred1_lcp": lcp_len(h_preds[0] if h_preds else "", h_target_sid),
        }

        for k in TOPKS:
            row[f"baseline_top{k}_hit"] = int(topk_hit(b_preds, b_target_sid, k))
            row[f"hierarchy_top{k}_hit"] = int(topk_hit(h_preds, h_target_sid, k))
            row[f"baseline_best_lcp_top{k}"] = best_lcp_in_topk(b_preds, b_target_sid, k)
            row[f"hierarchy_best_lcp_top{k}"] = best_lcp_in_topk(h_preds, h_target_sid, k)
            row[f"baseline_top{k}_same_l1"] = int(row[f"baseline_best_lcp_top{k}"] >= 1)
            row[f"baseline_top{k}_same_l2"] = int(row[f"baseline_best_lcp_top{k}"] >= 2)
            row[f"hierarchy_top{k}_same_l1"] = int(row[f"hierarchy_best_lcp_top{k}"] >= 1)
            row[f"hierarchy_top{k}_same_l2"] = int(row[f"hierarchy_best_lcp_top{k}"] >= 2)
            row[f"improved_at_{k}"] = int(row[f"baseline_top{k}_hit"] == 0 and row[f"hierarchy_top{k}_hit"] == 1)
            row[f"worsened_at_{k}"] = int(row[f"baseline_top{k}_hit"] == 1 and row[f"hierarchy_top{k}_hit"] == 0)

        rows.append(row)
    return rows


def mean_of(rows: list[dict], key: str) -> float:
    if not rows:
        return 0.0
    return float(mean(float(row[key]) for row in rows))


def summarize(rows: list[dict]) -> dict:
    summary: dict[str, object] = {"example_count": len(rows)}

    topk_summary: dict[str, dict[str, float | int]] = {}
    for k in TOPKS:
        topk_summary[f"top{k}"] = {
            "baseline_hit_rate": mean_of(rows, f"baseline_top{k}_hit"),
            "hierarchy_hit_rate": mean_of(rows, f"hierarchy_top{k}_hit"),
            "delta_hit_rate": mean_of(rows, f"hierarchy_top{k}_hit") - mean_of(rows, f"baseline_top{k}_hit"),
            "baseline_same_l1_rate": mean_of(rows, f"baseline_top{k}_same_l1"),
            "hierarchy_same_l1_rate": mean_of(rows, f"hierarchy_top{k}_same_l1"),
            "baseline_same_l2_rate": mean_of(rows, f"baseline_top{k}_same_l2"),
            "hierarchy_same_l2_rate": mean_of(rows, f"hierarchy_top{k}_same_l2"),
            "baseline_best_lcp": mean_of(rows, f"baseline_best_lcp_top{k}"),
            "hierarchy_best_lcp": mean_of(rows, f"hierarchy_best_lcp_top{k}"),
            "improved_count": int(sum(int(row[f"improved_at_{k}"]) for row in rows)),
            "worsened_count": int(sum(int(row[f"worsened_at_{k}"]) for row in rows)),
        }
    summary["topk"] = topk_summary

    rank_transition: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        rank_transition[str(row["baseline_rank_bucket"])][str(row["hierarchy_rank_bucket"])] += 1
    summary["rank_transition"] = {
        bucket: dict(sorted(counter.items()))
        for bucket, counter in sorted(rank_transition.items())
    }

    bucket_summary: dict[str, dict[str, dict[str, float | int]]] = {}
    for bucket in ["l2<=2", "l2=3", "l2>=4"]:
        bucket_rows = [row for row in rows if row["baseline_fanout_bucket"] == bucket]
        per_topk: dict[str, dict[str, float | int]] = {}
        for k in MAIN_TOPKS:
            per_topk[f"top{k}"] = {
                "count": len(bucket_rows),
                "baseline_hit_rate": mean_of(bucket_rows, f"baseline_top{k}_hit"),
                "hierarchy_hit_rate": mean_of(bucket_rows, f"hierarchy_top{k}_hit"),
                "delta_hit_rate": mean_of(bucket_rows, f"hierarchy_top{k}_hit") - mean_of(bucket_rows, f"baseline_top{k}_hit"),
                "improved_count": int(sum(int(row[f"improved_at_{k}"]) for row in bucket_rows)),
                "worsened_count": int(sum(int(row[f"worsened_at_{k}"]) for row in bucket_rows)),
            }
        bucket_summary[bucket] = per_topk
    summary["fanout_bucket"] = bucket_summary

    baseline_rank_bucket_summary: dict[str, dict[str, dict[str, float | int]]] = {}
    for bucket in ["1", "2-3", "4-5", "6-10", "11-20", "21-50", ">50"]:
        bucket_rows = [row for row in rows if row["baseline_rank_bucket"] == bucket]
        if not bucket_rows:
            continue
        per_topk = {}
        for k in MAIN_TOPKS:
            per_topk[f"top{k}"] = {
                "count": len(bucket_rows),
                "hierarchy_hit_rate": mean_of(bucket_rows, f"hierarchy_top{k}_hit"),
                "improved_count": int(sum(int(row[f"improved_at_{k}"]) for row in bucket_rows)),
                "worsened_count": int(sum(int(row[f"worsened_at_{k}"]) for row in bucket_rows)),
            }
        baseline_rank_bucket_summary[bucket] = per_topk
    summary["baseline_rank_bucket"] = baseline_rank_bucket_summary

    improved_vs_worsened: dict[str, dict[str, dict[str, float | int]]] = {}
    for k in MAIN_TOPKS:
        improved_rows = [row for row in rows if int(row[f"improved_at_{k}"]) == 1]
        worsened_rows = [row for row in rows if int(row[f"worsened_at_{k}"]) == 1]
        improved_vs_worsened[f"top{k}"] = {
            "improved": {
                "count": len(improved_rows),
                "baseline_same_l1_rate": mean_of(improved_rows, f"baseline_top{k}_same_l1"),
                "baseline_same_l2_rate": mean_of(improved_rows, f"baseline_top{k}_same_l2"),
                "baseline_mean_l2_fanout": mean_of(improved_rows, "baseline_target_l2_fanout"),
            },
            "worsened": {
                "count": len(worsened_rows),
                "baseline_same_l1_rate": mean_of(worsened_rows, f"baseline_top{k}_same_l1"),
                "baseline_same_l2_rate": mean_of(worsened_rows, f"baseline_top{k}_same_l2"),
                "baseline_mean_l2_fanout": mean_of(worsened_rows, "baseline_target_l2_fanout"),
            },
        }
    summary["improved_vs_worsened"] = improved_vs_worsened

    example_tables: dict[str, list[dict[str, object]]] = {}
    for k in MAIN_TOPKS:
        improved_rows = [row for row in rows if int(row[f"improved_at_{k}"]) == 1]
        worsened_rows = [row for row in rows if int(row[f"worsened_at_{k}"]) == 1]
        improved_rows = sorted(
            improved_rows,
            key=lambda row: (
                -int(row["baseline_target_l2_fanout"]),
                int(row["baseline_rank"]),
                int(row["item_id"]),
            ),
        )[:15]
        worsened_rows = sorted(
            worsened_rows,
            key=lambda row: (
                int(row["baseline_rank"]),
                -int(row["baseline_target_l2_fanout"]),
                int(row["item_id"]),
            ),
        )[:15]
        example_tables[f"top{k}"] = {
            "improved_examples": [
                {
                    "item_id": int(row["item_id"]),
                    "item_title": row["item_title"],
                    "baseline_rank": int(row["baseline_rank"]) if int(row["baseline_rank"]) != 999 else None,
                    "hierarchy_rank": int(row["hierarchy_rank"]) if int(row["hierarchy_rank"]) != 999 else None,
                    "baseline_l2_fanout": int(row["baseline_target_l2_fanout"]),
                    "baseline_best_lcp": int(row[f"baseline_best_lcp_top{k}"]),
                    "hierarchy_best_lcp": int(row[f"hierarchy_best_lcp_top{k}"]),
                }
                for row in improved_rows
            ],
            "worsened_examples": [
                {
                    "item_id": int(row["item_id"]),
                    "item_title": row["item_title"],
                    "baseline_rank": int(row["baseline_rank"]) if int(row["baseline_rank"]) != 999 else None,
                    "hierarchy_rank": int(row["hierarchy_rank"]) if int(row["hierarchy_rank"]) != 999 else None,
                    "baseline_l2_fanout": int(row["baseline_target_l2_fanout"]),
                    "baseline_best_lcp": int(row[f"baseline_best_lcp_top{k}"]),
                    "hierarchy_best_lcp": int(row[f"hierarchy_best_lcp_top{k}"]),
                }
                for row in worsened_rows
            ],
        }
    summary["example_tables"] = example_tables

    return summary


def format_rate(value: float) -> str:
    return f"{value:.5f}"


def write_markdown(summary: dict, path: Path) -> None:
    topk = summary["topk"]
    fanout_bucket = summary["fanout_bucket"]
    rank_transition = summary["rank_transition"]
    improved_vs_worsened = summary["improved_vs_worsened"]

    lines: list[str] = []
    lines.append("# Top-k Structural Error Analysis\n")
    lines.append("## Scope\n")
    lines.append("This note compares the current MiniOneRec baseline against the hierarchy-aware SID SFT run on Industrial.\n")
    lines.append("The focus is no longer only `top1`, but the full `top-k` structural behavior: hit rates, same-prefix retention, rank migration, and fanout-sensitive gains.\n")

    lines.append("## Top-k Summary\n")
    lines.append("| k | baseline hit | hierarchy hit | delta | baseline same `l1` in top-k | hierarchy same `l1` in top-k | baseline same `l2` in top-k | hierarchy same `l2` in top-k | improved | worsened |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for k in TOPKS:
        row = topk[f"top{k}"]
        lines.append(
            f"| {k} | {format_rate(row['baseline_hit_rate'])} | {format_rate(row['hierarchy_hit_rate'])} | "
            f"{row['delta_hit_rate']:+.5f} | {format_rate(row['baseline_same_l1_rate'])} | "
            f"{format_rate(row['hierarchy_same_l1_rate'])} | {format_rate(row['baseline_same_l2_rate'])} | "
            f"{format_rate(row['hierarchy_same_l2_rate'])} | {int(row['improved_count'])} | {int(row['worsened_count'])} |"
        )

    lines.append("\n## Reading\n")
    lines.append("- `hierarchy` is strongest on short-range metrics: `top3` improves most clearly, and `top5` remains slightly positive.")
    lines.append("- `hierarchy` reduces same-prefix presence in the beam at almost every `k`, which means its gain does not come from retaining more nearby candidates; it comes from cleaning up some hard local neighborhoods while losing others.")
    lines.append("- This already suggests the core tradeoff: better local disambiguation on difficult cases, but weaker neighborhood retention on some examples that the baseline already handled well.\n")

    lines.append("## Rank Transition Matrix\n")
    lines.append("Rows are baseline target-rank buckets, columns are hierarchy target-rank buckets.\n")
    buckets = ["1", "2-3", "4-5", "6-10", "11-20", "21-50", ">50"]
    lines.append("| baseline \\\\ hierarchy | " + " | ".join(buckets) + " |")
    lines.append("|---|" + "|".join(["---:"] * len(buckets)) + "|")
    for base_bucket in buckets:
        counter = rank_transition.get(base_bucket, {})
        lines.append("| " + base_bucket + " | " + " | ".join(str(counter.get(col, 0)) for col in buckets) + " |")

    lines.append("\n## Fanout Bucket Analysis\n")
    lines.append("Buckets are defined by the **baseline** target `l2` fanout, so every row refers to the same subset of examples before the SID swap.\n")
    for bucket in ["l2<=2", "l2=3", "l2>=4"]:
        lines.append(f"### {bucket}\n")
        lines.append("| k | count | baseline hit | hierarchy hit | delta | improved | worsened |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for k in MAIN_TOPKS:
            row = fanout_bucket[bucket][f"top{k}"]
            lines.append(
                f"| {k} | {int(row['count'])} | {format_rate(row['baseline_hit_rate'])} | "
                f"{format_rate(row['hierarchy_hit_rate'])} | {row['delta_hit_rate']:+.5f} | "
                f"{int(row['improved_count'])} | {int(row['worsened_count'])} |"
            )
        lines.append("")

    lines.append("## Improved vs Worsened Sets\n")
    for k in MAIN_TOPKS:
        block = improved_vs_worsened[f"top{k}"]
        imp = block["improved"]
        wor = block["worsened"]
        lines.append(f"### top{k}\n")
        lines.append("| set | count | baseline same `l1` in top-k | baseline same `l2` in top-k | baseline mean target `l2` fanout |")
        lines.append("|---|---:|---:|---:|---:|")
        lines.append(
            f"| improved by hierarchy | {int(imp['count'])} | {format_rate(imp['baseline_same_l1_rate'])} | "
            f"{format_rate(imp['baseline_same_l2_rate'])} | {format_rate(imp['baseline_mean_l2_fanout'])} |"
        )
        lines.append(
            f"| worsened by hierarchy | {int(wor['count'])} | {format_rate(wor['baseline_same_l1_rate'])} | "
            f"{format_rate(wor['baseline_same_l2_rate'])} | {format_rate(wor['baseline_mean_l2_fanout'])} |"
        )
        lines.append("")

    lines.append("## Main Takeaways\n")
    lines.append("- The positive effect of `hierarchy` is not a uniform top-k gain. It is concentrated in short-range ranking and crowded local structures.")
    lines.append("- The strongest positive evidence remains on high-fanout `same_l2`-like cases, where moving the target upward inside a hard local neighborhood matters most.")
    lines.append("- The strongest negative evidence is rank-drop on examples that the baseline already kept in the correct local neighborhood. This is the clearest sign that the current tokenizer-to-SFT transfer is still imperfect.")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = build_rows(args)
    summary = summarize(rows)

    output_csv = Path(args.output_csv)
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(rows).to_csv(output_csv, index=False)
    output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(summary, output_md)

    print(f"Wrote CSV: {output_csv}")
    print(f"Wrote JSON: {output_json}")
    print(f"Wrote MD: {output_md}")


if __name__ == "__main__":
    main()
