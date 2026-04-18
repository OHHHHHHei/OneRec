#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detailed pairwise output diagnosis from an aligned top-k comparison CSV."
    )
    parser.add_argument("--comparison-csv", required=True)
    parser.add_argument("--baseline-label", required=True)
    parser.add_argument("--hierarchy-label", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


def rank_bucket_group(bucket: str) -> str:
    if bucket == "1":
        return "1"
    if bucket in {"2-3", "4-5", "6-10"}:
        return "2-10"
    if bucket in {"11-20", "21-50"}:
        return "11-50"
    return ">50"


def lcp_bucket(value: int) -> str:
    if value >= 3:
        return "exact_target_present"
    if value == 2:
        return "same_l2_only"
    if value == 1:
        return "same_l1_only"
    return "cross_prefix_only"


def mean_of(rows: list[dict], key: str) -> float:
    if not rows:
        return 0.0
    return float(mean(float(row[key]) for row in rows))


def fanout_delta_stats(rows: list[dict]) -> dict[str, float | int]:
    if not rows:
        return {
            "count": 0,
            "baseline_mean_l2_fanout": 0.0,
            "hierarchy_mean_l2_fanout": 0.0,
            "delta_mean_l2_fanout": 0.0,
            "hierarchy_l2_decreased_count": 0,
            "hierarchy_l2_equal_count": 0,
            "hierarchy_l2_increased_count": 0,
            "hierarchy_l2_not_increased_fraction": 0.0,
            "mean_history_len": 0.0,
        }

    decreased = sum(
        int(row["hierarchy_target_l2_fanout"]) < int(row["baseline_target_l2_fanout"]) for row in rows
    )
    equal = sum(
        int(row["hierarchy_target_l2_fanout"]) == int(row["baseline_target_l2_fanout"]) for row in rows
    )
    increased = len(rows) - decreased - equal
    baseline_mean = mean_of(rows, "baseline_target_l2_fanout")
    hierarchy_mean = mean_of(rows, "hierarchy_target_l2_fanout")
    return {
        "count": len(rows),
        "baseline_mean_l2_fanout": baseline_mean,
        "hierarchy_mean_l2_fanout": hierarchy_mean,
        "delta_mean_l2_fanout": hierarchy_mean - baseline_mean,
        "hierarchy_l2_decreased_count": decreased,
        "hierarchy_l2_equal_count": equal,
        "hierarchy_l2_increased_count": increased,
        "hierarchy_l2_not_increased_fraction": (decreased + equal) / len(rows),
        "mean_history_len": mean_of(rows, "history_len"),
    }


def history_bucket(value: int) -> str:
    if value <= 3:
        return "1-3"
    if value <= 7:
        return "4-7"
    return "8+"


def counter_fraction(counter: Counter[str], total: int) -> dict[str, dict[str, float | int]]:
    out: dict[str, dict[str, float | int]] = {}
    for key, count in counter.items():
        out[key] = {
            "count": int(count),
            "fraction": count / total if total else 0.0,
        }
    return out


def analyze_transition(rows: list[dict], k: int) -> dict[str, object]:
    losses = [row for row in rows if int(row[f"baseline_top{k}_hit"]) == 1 and int(row[f"hierarchy_top{k}_hit"]) == 0]
    gains = [row for row in rows if int(row[f"baseline_top{k}_hit"]) == 0 and int(row[f"hierarchy_top{k}_hit"]) == 1]

    if k == 1:
        loss_retention_lcp = Counter(lcp_bucket(int(row["hierarchy_best_lcp_top10"])) for row in losses)
        gain_source_lcp = Counter(lcp_bucket(int(row["baseline_best_lcp_top10"])) for row in gains)
    else:
        loss_retention_lcp = Counter(lcp_bucket(int(row["hierarchy_best_lcp_top50"])) for row in losses)
        gain_source_lcp = Counter(lcp_bucket(int(row["baseline_best_lcp_top50"])) for row in gains)

    loss_rank_bucket = Counter(str(row["hierarchy_rank_bucket"]) for row in losses)
    gain_rank_bucket = Counter(str(row["baseline_rank_bucket"]) for row in gains)
    loss_rank_group = Counter(rank_bucket_group(str(row["hierarchy_rank_bucket"])) for row in losses)
    gain_rank_group = Counter(rank_bucket_group(str(row["baseline_rank_bucket"])) for row in gains)
    loss_history = Counter(history_bucket(int(row["history_len"])) for row in losses)
    gain_history = Counter(history_bucket(int(row["history_len"])) for row in gains)

    return {
        "loss_count": len(losses),
        "gain_count": len(gains),
        "loss_rank_bucket": counter_fraction(loss_rank_bucket, len(losses)),
        "gain_rank_bucket": counter_fraction(gain_rank_bucket, len(gains)),
        "loss_rank_group": counter_fraction(loss_rank_group, len(losses)),
        "gain_rank_group": counter_fraction(gain_rank_group, len(gains)),
        "loss_retention_lcp": counter_fraction(loss_retention_lcp, len(losses)),
        "gain_source_lcp": counter_fraction(gain_source_lcp, len(gains)),
        "loss_history_bucket": counter_fraction(loss_history, len(losses)),
        "gain_history_bucket": counter_fraction(gain_history, len(gains)),
        "loss_fanout_delta": fanout_delta_stats(losses),
        "gain_fanout_delta": fanout_delta_stats(gains),
    }


def analyze_items(rows: list[dict], k: int, top_n: int = 10, min_count: int = 4) -> dict[str, list[dict[str, object]]]:
    item_stats: dict[str, dict[str, object]] = defaultdict(
        lambda: {
            "item_title": "",
            "count": 0,
            "baseline_top1_hits": 0,
            "hierarchy_top1_hits": 0,
            "baseline_top10_hits": 0,
            "hierarchy_top10_hits": 0,
            "baseline_l2_sum": 0,
            "hierarchy_l2_sum": 0,
        }
    )
    for row in rows:
        item_id = str(row["item_id"])
        entry = item_stats[item_id]
        entry["item_title"] = row["item_title"]
        entry["count"] = int(entry["count"]) + 1
        entry["baseline_top1_hits"] = int(entry["baseline_top1_hits"]) + int(row["baseline_top1_hit"])
        entry["hierarchy_top1_hits"] = int(entry["hierarchy_top1_hits"]) + int(row["hierarchy_top1_hit"])
        entry["baseline_top10_hits"] = int(entry["baseline_top10_hits"]) + int(row["baseline_top10_hit"])
        entry["hierarchy_top10_hits"] = int(entry["hierarchy_top10_hits"]) + int(row["hierarchy_top10_hit"])
        entry["baseline_l2_sum"] = int(entry["baseline_l2_sum"]) + int(row["baseline_target_l2_fanout"])
        entry["hierarchy_l2_sum"] = int(entry["hierarchy_l2_sum"]) + int(row["hierarchy_target_l2_fanout"])

    item_rows: list[dict[str, object]] = []
    for item_id, entry in item_stats.items():
        count = int(entry["count"])
        if count < min_count:
            continue
        baseline_top1 = int(entry["baseline_top1_hits"]) / count
        hierarchy_top1 = int(entry["hierarchy_top1_hits"]) / count
        baseline_top10 = int(entry["baseline_top10_hits"]) / count
        hierarchy_top10 = int(entry["hierarchy_top10_hits"]) / count
        item_rows.append(
            {
                "item_id": int(item_id),
                "item_title": str(entry["item_title"]),
                "count": count,
                "baseline_top1_hit_rate": baseline_top1,
                "hierarchy_top1_hit_rate": hierarchy_top1,
                "delta_top1_hit_rate": hierarchy_top1 - baseline_top1,
                "baseline_top10_hit_rate": baseline_top10,
                "hierarchy_top10_hit_rate": hierarchy_top10,
                "delta_top10_hit_rate": hierarchy_top10 - baseline_top10,
                "baseline_mean_l2_fanout": int(entry["baseline_l2_sum"]) / count,
                "hierarchy_mean_l2_fanout": int(entry["hierarchy_l2_sum"]) / count,
            }
        )

    worst = sorted(
        item_rows,
        key=lambda row: (
            float(row[f"delta_top{k}_hit_rate"]),
            -int(row["count"]),
            int(row["item_id"]),
        ),
    )[:top_n]
    best = sorted(
        item_rows,
        key=lambda row: (
            -float(row[f"delta_top{k}_hit_rate"]),
            -int(row["count"]),
            int(row["item_id"]),
        ),
    )[:top_n]
    return {
        "worst_by_topk_delta": worst,
        "best_by_topk_delta": best,
    }


def build_summary(rows: list[dict], baseline_label: str, hierarchy_label: str) -> dict[str, object]:
    return {
        "baseline_label": baseline_label,
        "hierarchy_label": hierarchy_label,
        "example_count": len(rows),
        "top1": analyze_transition(rows, 1),
        "top10": analyze_transition(rows, 10),
        "item_hotspots_top10": analyze_items(rows, 10),
    }


def format_percent(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def fmt(value: float) -> str:
    return f"{value:.5f}"


def write_markdown(summary: dict[str, object], path: Path) -> None:
    baseline_label = str(summary["baseline_label"])
    hierarchy_label = str(summary["hierarchy_label"])
    top1 = dict(summary["top1"])
    top10 = dict(summary["top10"])
    hotspots = dict(summary["item_hotspots_top10"])

    lines: list[str] = []
    lines.append("# Output Pair Diagnosis（输出成对诊断）\n")
    lines.append("## Scope（范围）\n")
    lines.append(
        f"This note compares `{baseline_label}` against `{hierarchy_label}` using the aligned per-example top-k comparison rows（逐样本对齐 top-k 对比行）.\n"
    )

    lines.append("## Headline（摘要）\n")
    lines.append(
        f"- `top1`: `{hierarchy_label}` gains on `{int(top1['gain_count'])}` examples but loses on `{int(top1['loss_count'])}` examples."
    )
    lines.append(
        f"- `top10`: `{hierarchy_label}` gains on `{int(top10['gain_count'])}` examples but loses on `{int(top10['loss_count'])}` examples."
    )

    top1_loss_rank_group = dict(top1["loss_rank_group"])
    top10_loss_rank_group = dict(top10["loss_rank_group"])
    lines.append(
        f"- Among `top1` losses, `{format_percent(float(top1_loss_rank_group.get('2-10', {}).get('fraction', 0.0)))}` stay within `2-10`, "
        f"`{format_percent(float(top1_loss_rank_group.get('11-50', {}).get('fraction', 0.0)))}` fall to `11-50`, and "
        f"`{format_percent(float(top1_loss_rank_group.get('>50', {}).get('fraction', 0.0)))}` collapse beyond `50`."
    )
    lines.append(
        f"- Among `top10` losses, `{format_percent(float(top10_loss_rank_group.get('11-50', {}).get('fraction', 0.0)))}` only fall behind `10` but stay in `11-50`, "
        f"while `{format_percent(float(top10_loss_rank_group.get('>50', {}).get('fraction', 0.0)))}` disappear beyond `50`.\n"
    )

    lines.append("## Retention Diagnosis（保留诊断）\n")
    top1_retention = dict(top1["loss_retention_lcp"])
    top10_retention = dict(top10["loss_retention_lcp"])
    lines.append(
        f"- For `top1` losses, the exact target is still inside `{hierarchy_label}` `top10` on "
        f"`{format_percent(float(top1_retention.get('exact_target_present', {}).get('fraction', 0.0)))}` of lost examples."
    )
    lines.append(
        f"- For `top10` losses, the exact target is still inside `{hierarchy_label}` `top50` on "
        f"`{format_percent(float(top10_retention.get('exact_target_present', {}).get('fraction', 0.0)))}` of lost examples."
    )
    lines.append(
        f"- On the remaining `top10` losses, `{format_percent(float(top10_retention.get('same_l2_only', {}).get('fraction', 0.0)))}` keep only a same-`l2` neighbor（同 `l2` 邻居）, "
        f"`{format_percent(float(top10_retention.get('same_l1_only', {}).get('fraction', 0.0)))}` keep only a same-`l1` neighbor（同 `l1` 邻居）, and "
        f"`{format_percent(float(top10_retention.get('cross_prefix_only', {}).get('fraction', 0.0)))}` lose the whole local neighborhood（局部邻域）."
    )
    lines.append(
        "- This separates rank-drop（名次下掉） from neighborhood-collapse（邻域坍塌）: if the exact target is still in `top50`, the main problem is beam retention（候选束保留） rather than total routing failure（整体路由失败）.\n"
    )

    lines.append("## Structure vs Output（结构与输出）\n")
    top1_loss_fanout = dict(top1["loss_fanout_delta"])
    top10_loss_fanout = dict(top10["loss_fanout_delta"])
    top10_gain_fanout = dict(top10["gain_fanout_delta"])
    lines.append(
        f"- On `top1` losses, baseline mean target `l2` fanout is `{fmt(float(top1_loss_fanout['baseline_mean_l2_fanout']))}` "
        f"vs hierarchy `{fmt(float(top1_loss_fanout['hierarchy_mean_l2_fanout']))}`."
    )
    lines.append(
        f"- On `top10` losses, baseline mean target `l2` fanout is `{fmt(float(top10_loss_fanout['baseline_mean_l2_fanout']))}` "
        f"vs hierarchy `{fmt(float(top10_loss_fanout['hierarchy_mean_l2_fanout']))}`."
    )
    lines.append(
        f"- On `top10` losses, hierarchy `l2` fanout does not increase on "
        f"`{format_percent(float(top10_loss_fanout['hierarchy_l2_not_increased_fraction']))}` of examples."
    )
    lines.append(
        f"- On `top10` gains, hierarchy `l2` fanout does not increase on "
        f"`{format_percent(float(top10_gain_fanout['hierarchy_l2_not_increased_fraction']))}` of examples."
    )
    lines.append(
        "- So a cleaner local structure（更干净的局部结构） can appear on both gains and losses; tokenizer-side crowding reduction（分词器侧拥挤度降低） alone is not sufficient to explain downstream behavior（下游行为）.\n"
    )

    lines.append("## History Length（历史长度）\n")
    top10_loss_history = dict(top10["loss_history_bucket"])
    top10_gain_history = dict(top10["gain_history_bucket"])
    lines.append(
        f"- `top10` losses by history bucket: `1-3={format_percent(float(top10_loss_history.get('1-3', {}).get('fraction', 0.0)))}`, "
        f"`4-7={format_percent(float(top10_loss_history.get('4-7', {}).get('fraction', 0.0)))}`, "
        f"`8+={format_percent(float(top10_loss_history.get('8+', {}).get('fraction', 0.0)))}`."
    )
    lines.append(
        f"- `top10` gains by history bucket: `1-3={format_percent(float(top10_gain_history.get('1-3', {}).get('fraction', 0.0)))}`, "
        f"`4-7={format_percent(float(top10_gain_history.get('4-7', {}).get('fraction', 0.0)))}`, "
        f"`8+={format_percent(float(top10_gain_history.get('8+', {}).get('fraction', 0.0)))}`.\n"
    )

    lines.append("## Item Hotspots（物品热点）\n")
    lines.append("### Worst `top10` deltas（最差 `top10` 差值）\n")
    lines.append("| count | delta_top10 | delta_top1 | baseline_top10 | hierarchy_top10 | baseline_mean_l2 | hierarchy_mean_l2 | item |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---|")
    for row in list(hotspots["worst_by_topk_delta"]):
        lines.append(
            f"| {int(row['count'])} | {float(row['delta_top10_hit_rate']):+.3f} | {float(row['delta_top1_hit_rate']):+.3f} | "
            f"{float(row['baseline_top10_hit_rate']):.3f} | {float(row['hierarchy_top10_hit_rate']):.3f} | "
            f"{float(row['baseline_mean_l2_fanout']):.1f} | {float(row['hierarchy_mean_l2_fanout']):.1f} | "
            f"{int(row['item_id'])}: {str(row['item_title'])} |"
        )
    lines.append("")

    lines.append("### Best `top10` deltas（最好 `top10` 差值）\n")
    lines.append("| count | delta_top10 | delta_top1 | baseline_top10 | hierarchy_top10 | baseline_mean_l2 | hierarchy_mean_l2 | item |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---|")
    for row in list(hotspots["best_by_topk_delta"]):
        lines.append(
            f"| {int(row['count'])} | {float(row['delta_top10_hit_rate']):+.3f} | {float(row['delta_top1_hit_rate']):+.3f} | "
            f"{float(row['baseline_top10_hit_rate']):.3f} | {float(row['hierarchy_top10_hit_rate']):.3f} | "
            f"{float(row['baseline_mean_l2_fanout']):.1f} | {float(row['hierarchy_mean_l2_fanout']):.1f} | "
            f"{int(row['item_id'])}: {str(row['item_title'])} |"
        )
    lines.append("")

    lines.append("## Takeaways（结论）\n")
    lines.append(
        f"- If `{hierarchy_label}` loses mostly by dropping exact targets from `top10` to `11-50` or `>50`, the core failure is beam retention（候选束保留） rather than simple local reranking（局部重排）."
    )
    lines.append(
        "- If many losses happen even when hierarchy-side `l2` fanout does not increase, tokenizer-side structural cleanup（分词器侧结构清理） is not a reliable explanation for downstream improvement（下游提升）."
    )
    lines.append(
        "- The item hotspot table（物品热点表） helps distinguish systematic category failures（系统性类别失败） from random per-example noise（逐样本随机噪声）."
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = list(csv.DictReader(open(args.comparison_csv, "r", encoding="utf-8")))
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    summary = build_summary(rows, args.baseline_label, args.hierarchy_label)
    output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(summary, output_md)
    print(f"Wrote JSON: {output_json}")
    print(f"Wrote MD: {output_md}")


if __name__ == "__main__":
    main()
