#!/usr/bin/env python3
"""Item-level SFT error analysis for the R690b L3 sweep.

The analysis compares the current downstream anchor (L2=0.010,L3=0.020)
against the L3=0.010 tokenizer on the same Industrial test rows.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
SID_RE = re.compile(r"<a_\d+><b_\d+><c_\d+>")
TOKEN_RE = re.compile(r"<[abc]_\d+>")


VARIANTS = {
    "main_l3_020": {
        "label": "Current main L3=0.020",
        "test_csv": Path(
            "data_experiment/Amazon/r690b_lmh_l2_contrastive_pull_weight001/test/"
            "Industrial_and_Scientific_5_2016-10-2018-11.csv"
        ),
        "index": Path(
            "data_experiment/Amazon/r690b_lmh_l2_contrastive_pull_weight001/index/"
            "Industrial_and_Scientific.index.json"
        ),
        "result": Path(
            "results/experiments/mgr_sid_l2_lmh_sweep_sft_eval_20260507/"
            "final_result_sft_mgr_r690b_lmh_l2_contrastive_pull_weight001_"
            "title_on_desc_p05_4gpu_Industrial_and_Scientific.json"
        ),
    },
    "l3_010": {
        "label": "L3=0.010",
        "test_csv": Path(
            "data_experiment/Amazon/r690b_lmh_l2_weight001_l3_weight010/test/"
            "Industrial_and_Scientific_5_2016-10-2018-11.csv"
        ),
        "index": Path(
            "data_experiment/Amazon/r690b_lmh_l2_weight001_l3_weight010/index/"
            "Industrial_and_Scientific.index.json"
        ),
        "result": Path(
            "results/experiments/mgr_sid_l3_lmh_sweep_sft_eval_20260508/"
            "final_result_sft_mgr_r690b_lmh_l2_weight001_l3_weight010_"
            "title_on_desc_p05_4gpu_Industrial_and_Scientific.json"
        ),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("research-progress-log/experiment_analysis/2026-05-08_l3_010_sft_error_analysis"),
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def parse_id_list(raw: Any) -> list[int]:
    if isinstance(raw, list):
        return [int(x) for x in raw]
    text = str(raw).strip()
    if not text:
        return []
    value = ast.literal_eval(text)
    return [int(x) for x in value] if isinstance(value, list) else []


def sid_tokens(sid: str) -> tuple[str, str, str]:
    tokens = TOKEN_RE.findall(str(sid).strip())
    if len(tokens) != 3:
        raise ValueError(f"Bad SID: {sid!r}")
    return tuple(tokens)  # type: ignore[return-value]


def clean_sid(sid: Any) -> str:
    text = str(sid).strip()
    match = SID_RE.search(text)
    return match.group(0) if match else text


def sid_string(code: tuple[str, str, str]) -> str:
    return "".join(code)


def rank_of(target: str, preds: list[str]) -> int | None:
    target = clean_sid(target)
    clean_preds = [clean_sid(x) for x in preds]
    try:
        return clean_preds.index(target) + 1
    except ValueError:
        return None


def prefix_rank(target: str, preds: list[str], level: int) -> int | None:
    target_prefix = sid_tokens(target)[:level]
    for idx, pred in enumerate(preds, start=1):
        try:
            if sid_tokens(pred)[:level] == target_prefix:
                return idx
        except ValueError:
            continue
    return None


def hit(rank: int | None, k: int) -> int:
    return int(rank is not None and rank <= k)


def ndcg(rank: int | None, k: int) -> float:
    if rank is None or rank > k:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def load_json(path: Path) -> Any:
    return json.loads(resolve(path).read_text(encoding="utf-8"))


def load_index(path: Path) -> dict[int, tuple[str, str, str]]:
    raw = load_json(path)
    return {int(k): tuple(v[:3]) for k, v in raw.items()}


def code_stats(code_map: dict[int, tuple[str, str, str]]) -> dict[str, Counter[Any]]:
    return {
        "l1": Counter(code[0] for code in code_map.values()),
        "l12": Counter(code[:2] for code in code_map.values()),
        "sid": Counter(code_map.values()),
    }


def train_event_stats(
    code_maps: dict[str, dict[int, tuple[str, str, str]]],
) -> tuple[Counter[int], dict[str, dict[str, Counter[Any]]]]:
    train = pd.read_csv(
        REPO_ROOT / "data/Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.csv",
        usecols=["history_item_id", "item_id"],
    )
    item_events: Counter[int] = Counter()
    prefix_events: dict[str, dict[str, Counter[Any]]] = {
        key: {"l1": Counter(), "l12": Counter(), "sid": Counter()}
        for key in code_maps
    }
    for hist_raw, target_raw in zip(train["history_item_id"], train["item_id"]):
        ids = parse_id_list(hist_raw)
        ids.append(int(target_raw))
        for item_id in ids:
            item_events[item_id] += 1
            for key, cmap in code_maps.items():
                code = cmap.get(item_id)
                if code is None:
                    continue
                prefix_events[key]["l1"][code[0]] += 1
                prefix_events[key]["l12"][code[:2]] += 1
                prefix_events[key]["sid"][code] += 1
    return item_events, prefix_events


def sid_to_title(
    sid: str,
    sid_to_items: dict[str, list[int]],
    items: dict[int, dict[str, Any]],
) -> str:
    ids = sid_to_items.get(clean_sid(sid), [])
    if not ids:
        return "-"
    title = str(items[ids[0]].get("title", f"Item_{ids[0]}")).replace("\n", " ")
    suffix = "" if len(ids) == 1 else f" (+{len(ids)-1})"
    return title[:90] + suffix


def build_sample_rows() -> tuple[pd.DataFrame, dict[str, Any]]:
    items = {int(k): v for k, v in load_json(Path("data/Amazon/index/Industrial_and_Scientific.item.json")).items()}
    code_maps = {key: load_index(meta["index"]) for key, meta in VARIANTS.items()}
    stats = {key: code_stats(cmap) for key, cmap in code_maps.items()}
    item_events, prefix_events = train_event_stats(code_maps)
    test_main = pd.read_csv(resolve(VARIANTS["main_l3_020"]["test_csv"]))
    test_l3 = pd.read_csv(resolve(VARIANTS["l3_010"]["test_csv"]))
    result_main = load_json(VARIANTS["main_l3_020"]["result"])
    result_l3 = load_json(VARIANTS["l3_010"]["result"])
    if not (len(test_main) == len(test_l3) == len(result_main) == len(result_l3)):
        raise RuntimeError("Mismatched test/result lengths.")

    sid_to_items_by_variant: dict[str, dict[str, list[int]]] = {}
    for key, cmap in code_maps.items():
        sid_to_items: dict[str, list[int]] = defaultdict(list)
        for item_id, code in cmap.items():
            sid_to_items[sid_string(code)].append(item_id)
        sid_to_items_by_variant[key] = sid_to_items

    rows: list[dict[str, Any]] = []
    for idx in range(len(test_main)):
        item_id = int(test_main.iloc[idx]["item_id"])
        if item_id != int(test_l3.iloc[idx]["item_id"]):
            raise RuntimeError(f"Mismatched item_id at row {idx}")
        history = parse_id_list(test_main.iloc[idx]["history_item_id"])
        title = str(items[item_id].get("title", test_main.iloc[idx]["item_title"])).replace("\n", " ")
        row: dict[str, Any] = {
            "row_id": idx,
            "user_id": test_main.iloc[idx]["user_id"],
            "item_id": item_id,
            "item_title": title,
            "brand": str(items[item_id].get("brand", "")),
            "history_len": len(history),
            "train_item_events": item_events[item_id],
        }
        for key, result in [("main_l3_020", result_main[idx]), ("l3_010", result_l3[idx])]:
            target = clean_sid(result["output"])
            preds = result.get("predict", []) or []
            code = code_maps[key][item_id]
            exact_rank = rank_of(target, preds)
            row[f"{key}_sid"] = target
            row[f"{key}_rank"] = exact_rank if exact_rank is not None else 999
            row[f"{key}_hit10"] = hit(exact_rank, 10)
            row[f"{key}_ndcg10"] = ndcg(exact_rank, 10)
            row[f"{key}_l1_rank"] = prefix_rank(target, preds, 1) or 999
            row[f"{key}_l12_rank"] = prefix_rank(target, preds, 2) or 999
            row[f"{key}_l1_hit1"] = hit(prefix_rank(target, preds, 1), 1)
            row[f"{key}_l12_hit10"] = hit(prefix_rank(target, preds, 2), 10)
            row[f"{key}_top1"] = clean_sid(preds[0]) if preds else ""
            row[f"{key}_top1_title"] = sid_to_title(row[f"{key}_top1"], sid_to_items_by_variant[key], items)
            row[f"{key}_top1_in_history"] = int(row[f"{key}_top1"] in {sid_string(code_maps[key][h]) for h in history if h in code_maps[key]})
            row[f"{key}_same_l1_hist_count"] = sum(
                1 for h in history if h in code_maps[key] and code_maps[key][h][0] == code[0]
            )
            row[f"{key}_same_l12_hist_count"] = sum(
                1 for h in history if h in code_maps[key] and code_maps[key][h][:2] == code[:2]
            )
            row[f"{key}_l1_bucket_size"] = stats[key]["l1"][code[0]]
            row[f"{key}_l12_bucket_size"] = stats[key]["l12"][code[:2]]
            row[f"{key}_l1_train_events"] = prefix_events[key]["l1"][code[0]]
            row[f"{key}_l12_train_events"] = prefix_events[key]["l12"][code[:2]]
            row[f"{key}_sid_train_events"] = prefix_events[key]["sid"][code]
        row["ndcg10_delta_l3_minus_main"] = row["l3_010_ndcg10"] - row["main_l3_020_ndcg10"]
        row["hit10_transition"] = (
            f"{row['main_l3_020_hit10']}->{row['l3_010_hit10']}"
        )
        rows.append(row)
    df = pd.DataFrame(rows)
    aux = {
        "items": items,
        "code_maps": code_maps,
        "stats": stats,
        "item_events": item_events,
        "prefix_events": prefix_events,
        "sid_to_items_by_variant": sid_to_items_by_variant,
    }
    return df, aux


def metric_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key in ["main_l3_020", "l3_010"]:
        rows.append(
            {
                "variant": VARIANTS[key]["label"],
                "hr@1": float((df[f"{key}_rank"] <= 1).mean()),
                "hr@3": float((df[f"{key}_rank"] <= 3).mean()),
                "hr@5": float((df[f"{key}_rank"] <= 5).mean()),
                "hr@10": float((df[f"{key}_rank"] <= 10).mean()),
                "ndcg@10": float(df[f"{key}_ndcg10"].mean()),
                "l1_prefix_hit@1": float((df[f"{key}_l1_rank"] <= 1).mean()),
                "l12_prefix_hit@10": float((df[f"{key}_l12_rank"] <= 10).mean()),
                "top1_history_copy": float(df[f"{key}_top1_in_history"].mean()),
                "target_has_history_l12": float((df[f"{key}_same_l12_hist_count"] > 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def conditional_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    conditions = [
        ("all", lambda key: np.ones(len(df), dtype=bool)),
        ("L1 prefix @1", lambda key: df[f"{key}_l1_rank"] <= 1),
        ("L12 prefix @10", lambda key: df[f"{key}_l12_rank"] <= 10),
        ("target has history L12", lambda key: df[f"{key}_same_l12_hist_count"] > 0),
        ("target no history L12", lambda key: df[f"{key}_same_l12_hist_count"] == 0),
    ]
    for key in ["main_l3_020", "l3_010"]:
        for name, predicate in conditions:
            mask = predicate(key)
            sub = df[mask]
            if sub.empty:
                continue
            rows.append(
                {
                    "variant": VARIANTS[key]["label"],
                    "condition": name,
                    "count": len(sub),
                    "exact_hr@10": float(sub[f"{key}_hit10"].mean()),
                    "ndcg@10": float(sub[f"{key}_ndcg10"].mean()),
                    "median_exact_rank": float(sub[f"{key}_rank"].median()),
                    "top1_history_copy": float(sub[f"{key}_top1_in_history"].mean()),
                }
            )
    return pd.DataFrame(rows)


def l12_prefix_exact_miss_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key in ["main_l3_020", "l3_010"]:
        sub = df[(df[f"{key}_l12_rank"] <= 10) & (df[f"{key}_hit10"] == 0)]
        rows.append(
            {
                "variant": VARIANTS[key]["label"],
                "l12_prefix_but_exact_miss_count": len(sub),
                "fraction_of_test": len(sub) / len(df),
                "mean_l12_prefix_rank": float(sub[f"{key}_l12_rank"].mean()) if len(sub) else np.nan,
                "top1_history_copy": float(sub[f"{key}_top1_in_history"].mean()) if len(sub) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def item_summary(df: pd.DataFrame) -> pd.DataFrame:
    agg = df.groupby(["item_id", "item_title", "brand"], as_index=False).agg(
        test_count=("row_id", "count"),
        train_item_events=("train_item_events", "first"),
        main_hit10=("main_l3_020_hit10", "sum"),
        l3_hit10=("l3_010_hit10", "sum"),
        main_mean_rank=("main_l3_020_rank", "mean"),
        l3_mean_rank=("l3_010_rank", "mean"),
        main_mean_ndcg10=("main_l3_020_ndcg10", "mean"),
        l3_mean_ndcg10=("l3_010_ndcg10", "mean"),
        main_same_l12_hist_mean=("main_l3_020_same_l12_hist_count", "mean"),
        l3_same_l12_hist_mean=("l3_010_same_l12_hist_count", "mean"),
        main_l12_bucket_size=("main_l3_020_l12_bucket_size", "first"),
        l3_l12_bucket_size=("l3_010_l12_bucket_size", "first"),
        main_sid=("main_l3_020_sid", "first"),
        l3_sid=("l3_010_sid", "first"),
    )
    agg["hit10_delta_l3_minus_main"] = agg["l3_hit10"] - agg["main_hit10"]
    agg["ndcg10_delta_l3_minus_main"] = agg["l3_mean_ndcg10"] - agg["main_mean_ndcg10"]
    return agg.sort_values(["ndcg10_delta_l3_minus_main", "test_count"], ascending=[True, False])


def route_summary(df: pd.DataFrame, key: str, min_count: int = 20) -> pd.DataFrame:
    route_col = f"{key}_sid"
    # Use prefixes from target SID string so the table is directly readable.
    tmp = df.copy()
    tmp[f"{key}_l1"] = tmp[route_col].map(lambda s: sid_tokens(s)[0])
    rows = tmp.groupby(f"{key}_l1", as_index=False).agg(
        test_count=("row_id", "count"),
        hit10=(f"{key}_hit10", "mean"),
        ndcg10=(f"{key}_ndcg10", "mean"),
        same_l12_hist_mean=(f"{key}_same_l12_hist_count", "mean"),
        l1_bucket_size=(f"{key}_l1_bucket_size", "first"),
        l1_train_events=(f"{key}_l1_train_events", "first"),
    )
    return rows[rows["test_count"] >= min_count].sort_values("ndcg10")


def examples_table(df: pd.DataFrame, transition: str, n: int = 8) -> pd.DataFrame:
    cols = [
        "row_id",
        "item_id",
        "item_title",
        "train_item_events",
        "history_len",
        "main_l3_020_sid",
        "l3_010_sid",
        "main_l3_020_rank",
        "l3_010_rank",
        "main_l3_020_same_l12_hist_count",
        "l3_010_same_l12_hist_count",
        "main_l3_020_top1",
        "main_l3_020_top1_title",
        "l3_010_top1",
        "l3_010_top1_title",
    ]
    if transition == "1->0":
        sub = df[df["hit10_transition"].eq("1->0")].sort_values(
            ["main_l3_020_rank", "l3_010_rank"], ascending=[True, False]
        )
    elif transition == "0->1":
        sub = df[df["hit10_transition"].eq("0->1")].sort_values(
            ["l3_010_rank", "main_l3_020_rank"], ascending=[True, False]
        )
    else:
        sub = df[df["hit10_transition"].eq(transition)].sort_values("ndcg10_delta_l3_minus_main")
    out = sub[cols].head(n).copy()
    for col in ["item_title", "main_l3_020_top1_title", "l3_010_top1_title"]:
        out[col] = out[col].map(lambda x: str(x)[:90])
    return out


def render_report(df: pd.DataFrame, out_dir: Path) -> str:
    metrics = metric_summary(df)
    conditional = conditional_summary(df)
    prefix_miss = l12_prefix_exact_miss_summary(df)
    transitions = df["hit10_transition"].value_counts().rename_axis("main_hit10->l3_hit10").reset_index(name="count")
    transitions["pct"] = transitions["count"] / len(df) * 100

    item = item_summary(df)
    lost_items = item[item["hit10_delta_l3_minus_main"] < 0].head(12).copy()
    gained_items = item[item["hit10_delta_l3_minus_main"] > 0].sort_values(
        ["hit10_delta_l3_minus_main", "test_count"], ascending=[False, False]
    ).head(12).copy()

    main_routes = route_summary(df, "main_l3_020").head(12)
    l3_routes = route_summary(df, "l3_010").head(12)

    lines: list[str] = []
    lines.append("# L3=0.010 SFT Error Analysis（监督微调错误分析）")
    lines.append("")
    lines.append("## Summary（摘要）")
    lines.append("")
    lines.append(
        "This compares the same 4533 test rows for the current mainline L3=0.020（当前主线第三层权重 0.020） and L3=0.010（第三层权重 0.010）."
    )
    lines.append("")
    lines.append(metrics.to_markdown(index=False, floatfmt=".6f"))
    lines.append("")
    lines.append("## Hit Transition（命中迁移）")
    lines.append("")
    lines.append(transitions.to_markdown(index=False, floatfmt=".2f"))
    lines.append("")
    lines.append("## Conditional Exact Accuracy（条件精确命中）")
    lines.append("")
    lines.append(
        "This table checks whether prefix routing（前缀路由） actually converts into exact SID（精确语义标识） ranking."
    )
    lines.append("")
    lines.append(conditional.to_markdown(index=False, floatfmt=".6f"))
    lines.append("")
    lines.append("## Prefix-Correct But Exact-Wrong（前缀对但精确错）")
    lines.append("")
    lines.append(prefix_miss.to_markdown(index=False, floatfmt=".6f"))
    lines.append("")
    lines.append("## Item-Level Lost Targets（物品级退化目标）")
    lines.append("")
    item_cols = [
        "item_id",
        "test_count",
        "train_item_events",
        "main_hit10",
        "l3_hit10",
        "hit10_delta_l3_minus_main",
        "main_mean_rank",
        "l3_mean_rank",
        "main_same_l12_hist_mean",
        "l3_same_l12_hist_mean",
        "main_l12_bucket_size",
        "l3_l12_bucket_size",
        "item_title",
    ]
    lines.append(lost_items[item_cols].to_markdown(index=False, floatfmt=".3f"))
    lines.append("")
    lines.append("## Item-Level Gained Targets（物品级改善目标）")
    lines.append("")
    lines.append(gained_items[item_cols].to_markdown(index=False, floatfmt=".3f"))
    lines.append("")
    lines.append("## Weakest L1 Routes（最弱第一层路由）")
    lines.append("")
    lines.append("### Current main L3=0.020（当前主线）")
    lines.append(main_routes.to_markdown(index=False, floatfmt=".4f"))
    lines.append("")
    lines.append("### L3=0.010（第三层权重 0.010）")
    lines.append(l3_routes.to_markdown(index=False, floatfmt=".4f"))
    lines.append("")
    lines.append("## Main Hit / L3 Miss Examples（主线命中但 L3=0.010 失败样例）")
    lines.append("")
    lines.append(examples_table(df, "1->0").to_markdown(index=False, floatfmt=".3f"))
    lines.append("")
    lines.append("## L3 Hit / Main Miss Examples（L3=0.010 命中但主线失败样例）")
    lines.append("")
    lines.append(examples_table(df, "0->1").to_markdown(index=False, floatfmt=".3f"))
    lines.append("")
    lines.append("## Output Files（输出文件）")
    lines.append("")
    lines.append(f"- per_sample（逐样本）: `{out_dir / 'per_sample.csv'}`")
    lines.append(f"- per_item（逐物品）: `{out_dir / 'per_item.csv'}`")
    lines.append(f"- report（报告）: `{out_dir / 'report.md'}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    out_dir = resolve(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df, _ = build_sample_rows()
    per_item = item_summary(df)
    df.to_csv(out_dir / "per_sample.csv", index=False)
    per_item.to_csv(out_dir / "per_item.csv", index=False)
    (out_dir / "report.md").write_text(render_report(df, out_dir), encoding="utf-8")
    print(f"[done] {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
