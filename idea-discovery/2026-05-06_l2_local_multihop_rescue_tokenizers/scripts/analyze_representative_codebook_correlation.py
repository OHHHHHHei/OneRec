#!/usr/bin/env python3
"""Compare representative tokenizer codebook metrics against SFT results.

This is a thin batch wrapper around analyze_codebook_reasonableness.py. It uses
the same train-only collaborative pair definitions, then joins structure/pair
metrics with finalized SFT scores from the split registry.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
HELPER_PATH = Path(__file__).with_name("analyze_codebook_reasonableness.py")


def load_helper() -> Any:
    spec = importlib.util.spec_from_file_location("codebook_reasonableness", HELPER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import helper from {HELPER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class Representative:
    label: str
    tokenizer_record_id: str
    index_path: Path
    note: str


REPRESENTATIVES = [
    Representative(
        label="Original semantic",
        tokenizer_record_id="tok_industrial_original_semantic",
        index_path=Path("data/Amazon/index/Industrial_and_Scientific.index.json"),
        note="original MiniOneRec semantic SID baseline",
    ),
    Representative(
        label="V2 offline",
        tokenizer_record_id="tok_industrial_mgr_tokenizer_v2_offline",
        index_path=Path(
            "/data/leejt/OneRec/output_weights/experiments/"
            "mgr_sid_tokenizer_v2/generated_indices/"
            "Industrial_and_Scientific.mgr_tokenizer_v2_offline.index.json"
        ),
        note="strong pre-LMH tokenizer line",
    ),
    Representative(
        label="R690b L2=0.010 main",
        tokenizer_record_id="tok_industrial_r690b_lmh_l2_contrastive_pull_weight001_20260507",
        index_path=Path(
            "/data/leejt/OneRec/output_weights/experiments/"
            "mgr_sid_l2_lmh_sweep_20260507/generated_indices/"
            "Industrial_and_Scientific.r690b_lmh_l2_contrastive_pull_weight001.index.json"
        ),
        note="current mainline and best tokenizer-side SFT",
    ),
    Representative(
        label="R690b L2=0.003 weak",
        tokenizer_record_id="tok_industrial_r690b_lmh_l2_contrastive_pull_weight0003_20260507",
        index_path=Path(
            "/data/leejt/OneRec/output_weights/experiments/"
            "mgr_sid_l2_lmh_sweep_20260507/generated_indices/"
            "Industrial_and_Scientific.r690b_lmh_l2_contrastive_pull_weight0003.index.json"
        ),
        note="low L2 collaborative weight",
    ),
    Representative(
        label="R690b L2=0.005 weak",
        tokenizer_record_id="tok_industrial_r690b_lmh_l2_contrastive_pull_weight0005_20260507",
        index_path=Path(
            "/data/leejt/OneRec/output_weights/experiments/"
            "mgr_sid_l2_lmh_sweep_20260507/generated_indices/"
            "Industrial_and_Scientific.r690b_lmh_l2_contrastive_pull_weight0005.index.json"
        ),
        note="low-to-mid L2 collaborative weight",
    ),
    Representative(
        label="R690b L2=0.015 fragmented",
        tokenizer_record_id="tok_industrial_r690b_lmh_l2_contrastive_pull_weight0015_20260507",
        index_path=Path(
            "/data/leejt/OneRec/output_weights/experiments/"
            "mgr_sid_l2_lmh_sweep_20260507/generated_indices/"
            "Industrial_and_Scientific.r690b_lmh_l2_contrastive_pull_weight0015.index.json"
        ),
        note="upper-side L2 weight with fragmented L1 routing",
    ),
    Representative(
        label="R690b no L1 semantic",
        tokenizer_record_id="tok_industrial_r690b_lmh_l2_contrastive_pull_weight001_no_l1_semantic_20260508",
        index_path=Path(
            "/data/leejt/OneRec/output_weights/experiments/"
            "mgr_sid_l1_ablation_20260507/generated_indices/"
            "Industrial_and_Scientific.r690b_lmh_l2_contrastive_pull_weight001_no_l1_semantic.index.json"
        ),
        note="L1 semantic pull ablation",
    ),
    Representative(
        label="V2 LMH mid=0.010",
        tokenizer_record_id="tok_industrial_v2_lmh_mid_weight001_20260507",
        index_path=Path(
            "/data/leejt/OneRec/output_weights/experiments/"
            "mgr_sid_l2_lmh_sweep_20260507/generated_indices/"
            "Industrial_and_Scientific.v2_lmh_mid_weight001.index.json"
        ),
        note="same LMH idea on weaker v2 branch",
    ),
    Representative(
        label="Original L2 multihop ranking",
        tokenizer_record_id="tok_industrial_original_l2_multihop_ranking_20260421",
        index_path=Path(
            "/data/leejt/OneRec/output_weights/experiments/"
            "mgr_sid_original_l2_multihop_ranking_20260421/generated_indices/"
            "Industrial_and_Scientific.original_l2_multihop_ranking.index.json"
        ),
        note="minimal-edit collaborative ranking screen",
    ),
    Representative(
        label="QCR L2 conflict ranking",
        tokenizer_record_id="tok_industrial_qcr_l2_conflict_ranking_20260421",
        index_path=Path(
            "/data/leejt/OneRec/output_weights/experiments/"
            "mgr_sid_qcr_l2_conflict_ranking_20260421/generated_indices/"
            "Industrial_and_Scientific.qcr_l2_conflict_ranking.index.json"
        ),
        note="QCR branch with healthy collision but negative SFT",
    ),
    Representative(
        label="Stage3 prefix retained",
        tokenizer_record_id="tok_industrial_stage3_r401b_g005",
        index_path=Path(
            "/data/leejt/OneRec/output_weights/experiments/"
            "mgr_sid_stage3_prefix_retained_20260414/generated_indices/"
            "Industrial_and_Scientific.stage3_r401b_g005.index.json"
        ),
        note="prefix-retention branch",
    ),
    Representative(
        label="TAGCF attr mid",
        tokenizer_record_id="tok_industrial_tagcf_r510_attr_mid",
        index_path=Path(
            "/data/leejt/OneRec/output_weights/experiments/"
            "mgr_sid_tagcf_branch_20260415/generated_indices/"
            "Industrial_and_Scientific.tagcf_r510_attr_mid.index.json"
        ),
        note="attribute-topology mid graph branch",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("research-progress-log/experiment_analysis/2026-05-08_representative_codebook_correlation"),
    )
    parser.add_argument("--semantic-topk", type=int, default=20)
    parser.add_argument("--max-pairs-per-set", type=int, default=10000)
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def load_best_sft() -> pd.DataFrame:
    sft = pd.read_csv(REPO_ROOT / "research-progress-log/experiment_registry/sft_registry.csv")
    sft = sft[sft["dataset_key"].eq("industrial")].copy()
    for col in ["ndcg_at_1", "ndcg_at_3", "ndcg_at_5", "ndcg_at_10", "hr_at_10", "hr_at_50"]:
        sft[col] = pd.to_numeric(sft[col], errors="coerce")
    return (
        sft.sort_values(["tokenizer_record_id", "ndcg_at_10", "hr_at_10"], ascending=[True, False, False])
        .groupby("tokenizer_record_id", as_index=False)
        .head(1)
    )


def zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if std == 0 or np.isnan(std):
        return series * 0
    return (series - series.mean()) / std


def render_report(summary: pd.DataFrame, corr: pd.DataFrame, corr_same_recipe: pd.DataFrame, output_dir: Path) -> str:
    lines: list[str] = []
    lines.append("# Representative Codebook-SFT Correlation（代表性码本-监督微调相关性）")
    lines.append("")
    lines.append("## 结论")
    lines.append("")
    lines.append(
        "- 这批代表 tokenizer（分词器）显示，codebook reasonableness（码本合理性）确实能解释一部分 SFT（监督微调）趋势，尤其是当前 `R690b L2=0.010 main`、`L2=0.003 weak`、`L2=0.015 fragmented`、`no L1 semantic` 这一组。"
    )
    lines.append(
        "- 但它不是单变量充分条件：original semantic（原版语义）和 v2 offline（离线 v2）仍然受 recipe（训练配方）和 SID learnability（语义标识可学习性）影响，不能只看某一个结构指标。"
    )
    lines.append(
        "- 最有解释力的模式是：L1 routing（第一层路由）不能过碎，`S-near C-far`（语义近但协同远）需要在保持较高 same L1（同第一层）的同时降低 same L12（同前两层）；过强约束会破坏 `S-near C-near`（语义近且协同近）。"
    )
    lines.append("")
    lines.append("## 代表样本")
    lines.append("")
    cols = [
        "label",
        "ndcg_at_10",
        "hr_at_10",
        "active_l1",
        "unique_l12",
        "top5_l1_cover",
        "snear_cfar_same_l1",
        "snear_cfar_same_l12",
        "snear_cfar_split_after_l1",
        "snear_cnear_same_l1",
        "sfar_cnear_avg_overlap",
    ]
    lines.append(summary[cols].to_markdown(index=False, floatfmt=".4f"))
    lines.append("")
    lines.append("## Correlation（相关性）")
    lines.append("")
    lines.append("### All Representative Tokenizers（全部代表分词器）")
    lines.append("")
    lines.append(corr.to_markdown(index=False, floatfmt=".4f"))
    lines.append("")
    lines.append("### Same SFT Recipe Subset（同监督微调配方子集）")
    lines.append("")
    lines.append(corr_same_recipe.to_markdown(index=False, floatfmt=".4f"))
    lines.append("")
    lines.append("## 输出文件")
    lines.append("")
    lines.append(f"- `summary.csv`: `{output_dir / 'summary.csv'}`")
    lines.append(f"- `correlation.csv`: `{output_dir / 'correlation.csv'}`")
    lines.append(f"- `correlation_same_recipe.csv`: `{output_dir / 'correlation_same_recipe.csv'}`")
    lines.append(f"- `metrics.json`: `{output_dir / 'metrics.json'}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    helper = load_helper()
    np.random.seed(args.random_seed)

    emb = helper.normalize_embeddings(Path("data/Amazon/index/Industrial_and_Scientific.emb-qwen-td.npy"))
    pair_counts, ppmi, data_stats = helper.build_collab_stats(Path("data/Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.csv"))
    semantic_pairs = helper.build_semantic_top_pairs(emb, args.semantic_topk)
    pair_sets = helper.build_pair_sets(
        emb,
        semantic_pairs,
        pair_counts,
        ppmi,
        max_pairs=args.max_pairs_per_set,
        seed=args.random_seed,
    )

    best_sft = load_best_sft()
    rows: list[dict[str, Any]] = []
    metrics_json: dict[str, Any] = {"data_stats": data_stats, "pair_set_stats": {}, "variants": {}}
    for set_name, pairs in pair_sets.items():
        metrics_json["pair_set_stats"][set_name] = {
            "pair_count": len(pairs),
            "semantic_sim_mean": float(np.mean([p["semantic_sim"] for p in pairs])) if pairs else 0.0,
            "ppmi_mean": float(np.mean([p["ppmi"] for p in pairs])) if pairs else 0.0,
        }

    for rep in REPRESENTATIVES:
        index_path = resolve(rep.index_path)
        if not index_path.exists():
            print(f"[skip] missing index: {rep.label} {index_path}")
            continue
        sft_rows = best_sft[best_sft["tokenizer_record_id"].eq(rep.tokenizer_record_id)]
        if sft_rows.empty:
            print(f"[skip] missing SFT: {rep.label} {rep.tokenizer_record_id}")
            continue
        sft_row = sft_rows.iloc[0].to_dict()
        code_map = helper.load_index(rep.index_path)
        structure = helper.structure_metrics(code_map)
        pair_metric = {set_name: helper.pair_metrics(code_map, pairs) for set_name, pairs in pair_sets.items()}

        row: dict[str, Any] = {
            "label": rep.label,
            "tokenizer_record_id": rep.tokenizer_record_id,
            "sft_variant": sft_row.get("variant"),
            "note": rep.note,
            "recipe": sft_row.get("recipe"),
            "ndcg_at_1": sft_row.get("ndcg_at_1"),
            "ndcg_at_3": sft_row.get("ndcg_at_3"),
            "ndcg_at_5": sft_row.get("ndcg_at_5"),
            "ndcg_at_10": sft_row.get("ndcg_at_10"),
            "hr_at_10": sft_row.get("hr_at_10"),
            "hr_at_50": sft_row.get("hr_at_50"),
            **structure,
            "snear_cfar_same_l1": pair_metric["S-near C-far"]["same_l1_pct"],
            "snear_cfar_same_l12": pair_metric["S-near C-far"]["same_l12_pct"],
            "snear_cfar_split_after_l1": pair_metric["S-near C-far"]["split_after_l1_pct"],
            "snear_cnear_same_l1": pair_metric["S-near C-near"]["same_l1_pct"],
            "snear_cnear_same_l12": pair_metric["S-near C-near"]["same_l12_pct"],
            "sfar_cnear_avg_overlap": pair_metric["S-far C-near"]["avg_token_overlap"],
            "sfar_cnear_same_l1": pair_metric["S-far C-near"]["same_l1_pct"],
            "sfar_cfar_avg_overlap": pair_metric["S-far C-far"]["avg_token_overlap"],
        }
        rows.append(row)
        metrics_json["variants"][rep.label] = {"structure": structure, "pair_metrics": pair_metric}

    summary = pd.DataFrame(rows).sort_values("ndcg_at_10", ascending=False)
    feature_cols = [
        "active_l1",
        "unique_l12",
        "top5_l1_cover",
        "l1_entropy_norm",
        "l1_gini",
        "snear_cfar_same_l1",
        "snear_cfar_same_l12",
        "snear_cfar_split_after_l1",
        "snear_cnear_same_l1",
        "snear_cnear_same_l12",
        "sfar_cnear_avg_overlap",
        "sfar_cnear_same_l1",
    ]
    def compute_corr(frame: pd.DataFrame) -> pd.DataFrame:
        corr_rows = []
        for col in feature_cols:
            corr_rows.append(
                {
                    "metric": col,
                    "pearson_with_ndcg10": frame[col].corr(frame["ndcg_at_10"], method="pearson"),
                    "spearman_with_ndcg10": frame[col].corr(frame["ndcg_at_10"], method="spearman"),
                    "pearson_with_hr10": frame[col].corr(frame["hr_at_10"], method="pearson"),
                }
            )
        return pd.DataFrame(corr_rows).sort_values("spearman_with_ndcg10", ascending=False)

    corr = compute_corr(summary)
    same_recipe = summary[summary["recipe"].eq("title_history2sid_on+desc_align_p05")].copy()
    corr_same_recipe = compute_corr(same_recipe)

    # A compact diagnostic score for the current hypothesis, not a trained predictor.
    summary["routing_fragment_z"] = zscore(summary["active_l1"]) - zscore(summary["top5_l1_cover"])
    summary["semantic_split_balance"] = summary["snear_cfar_split_after_l1"] - (100 - summary["snear_cnear_same_l1"])
    summary["codebook_reasonableness_proxy"] = (
        zscore(summary["snear_cfar_split_after_l1"])
        + zscore(summary["snear_cnear_same_l1"])
        - zscore(summary["routing_fragment_z"])
        - zscore(summary["snear_cfar_same_l12"])
    )

    summary.to_csv(output_dir / "summary.csv", index=False)
    corr.to_csv(output_dir / "correlation.csv", index=False)
    corr_same_recipe.to_csv(output_dir / "correlation_same_recipe.csv", index=False)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics_json, f, ensure_ascii=False, indent=2)
    report = render_report(summary, corr, corr_same_recipe, output_dir)
    (output_dir / "report.md").write_text(report, encoding="utf-8")

    print(f"[done] summary: {output_dir / 'summary.csv'}")
    print(f"[done] report: {output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
