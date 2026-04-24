#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from onerec.experiments.mgr_sid.probe import evaluate_view, load_sid_to_item_ids
from onerec.experiments.mgr_sid.transplanted_graph_bank import build_transplanted_graph_bank


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experimental paper-transplant graph-bank probe for MGR-SID.")
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--result_json", required=True)
    parser.add_argument("--index_json", required=True)
    parser.add_argument("--semantic_embedding_path", default=None)
    parser.add_argument("--history_k", type=int, default=10)
    parser.add_argument("--coarse_min_weight", type=float, default=2.0)
    parser.add_argument("--local_min_weight", type=float, default=1.0)
    parser.add_argument("--community_clusters", type=int, default=64)
    parser.add_argument("--anchor_topk", type=int, default=32)
    parser.add_argument("--semantic_mix", type=float, default=0.35)
    parser.add_argument("--spectral_rank", type=int, default=48)
    parser.add_argument("--band_low", type=float, default=0.25)
    parser.add_argument("--band_high", type=float, default=0.65)
    parser.add_argument("--temporal_mix", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--output_dir", default=None)
    return parser.parse_args()


def write_summary_markdown(summary: dict[str, Any], output_path: Path) -> None:
    lines = [
        "# Experimental MGR-SID Paper-Transplant Probe",
        "",
        f"- generated_at: `{summary['generated_at']}`",
        f"- train_csv: `{summary['train_csv']}`",
        f"- test_csv: `{summary['test_csv']}`",
        f"- result_json: `{summary['result_json']}`",
        f"- index_json: `{summary['index_json']}`",
        f"- semantic_embedding_path: `{summary['semantic_embedding_path']}`",
        "",
        "## View Summary",
        "",
        "| View | All | Same-l1 | Same-l2 | Coverage(all) |",
        "|------|-----|---------|---------|---------------|",
    ]
    for view_name, payload in summary["views"].items():
        buckets = payload["buckets"]
        lines.append(
            f"| `{view_name}` | "
            f"{buckets['all']['target_better_rate']:.6f} | "
            f"{buckets['same_l1']['target_better_rate']:.6f} | "
            f"{buckets['same_l2']['target_better_rate']:.6f} | "
            f"{buckets['all']['coverage']:.6f} |"
        )
    lines.append("")
    lines.append("## Transplanted Mid-View Ranking")
    lines.append("")
    for view_name in summary["transplanted_mid_view_ranking"]:
        payload = summary["views"][view_name]
        same_l2 = payload["buckets"]["same_l2"]["target_better_rate"]
        coverage = payload["buckets"]["all"]["coverage"]
        lines.append(f"- `{view_name}`: same-l2={same_l2:.6f}, all-coverage={coverage:.6f}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()

    train_csv = Path(args.train_csv)
    test_csv = Path(args.test_csv)
    result_json = Path(args.result_json)
    index_json = Path(args.index_json)

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    result_data = json.loads(result_json.read_text(encoding="utf-8"))
    if len(test_df) != len(result_data):
        raise ValueError(f"Length mismatch: test={len(test_df)} results={len(result_data)}")

    sid_to_items = load_sid_to_item_ids(index_json)
    views = build_transplanted_graph_bank(
        train_df=train_df,
        test_df=test_df,
        history_k=args.history_k,
        coarse_min_weight=args.coarse_min_weight,
        local_min_weight=args.local_min_weight,
        n_clusters=args.community_clusters,
        seed=args.seed,
        semantic_embedding_path=args.semantic_embedding_path,
        anchor_topk=args.anchor_topk,
        semantic_mix=args.semantic_mix,
        spectral_rank=args.spectral_rank,
        band_low=args.band_low,
        band_high=args.band_high,
        temporal_mix=args.temporal_mix,
    )

    view_summaries: dict[str, Any] = {}
    for view_name, view in views.items():
        view_summaries[view_name] = evaluate_view(
            test_df=test_df,
            result_data=result_data,
            sid_to_items=sid_to_items,
            view=view,
            history_k=args.history_k,
            max_examples=args.max_examples,
        )

    transplanted_mid_views = [
        "mid_diffusion_purified",
        "mid_band_pass_purified",
        "mid_community_purified",
        "fagsp_mid_base",
        "fagsp_mid_prism",
        "gsprec_mid_prism",
    ]
    ranking = sorted(
        transplanted_mid_views,
        key=lambda name: (
            view_summaries[name]["buckets"]["same_l2"]["target_better_rate"],
            view_summaries[name]["buckets"]["all"]["coverage"],
        ),
        reverse=True,
    )

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "train_csv": str(train_csv),
        "test_csv": str(test_csv),
        "result_json": str(result_json),
        "index_json": str(index_json),
        "semantic_embedding_path": str(args.semantic_embedding_path) if args.semantic_embedding_path else None,
        "history_k": args.history_k,
        "coarse_min_weight": args.coarse_min_weight,
        "local_min_weight": args.local_min_weight,
        "community_clusters": args.community_clusters,
        "anchor_topk": args.anchor_topk,
        "semantic_mix": args.semantic_mix,
        "spectral_rank": args.spectral_rank,
        "band_low": args.band_low,
        "band_high": args.band_high,
        "temporal_mix": args.temporal_mix,
        "seed": args.seed,
        "max_examples": args.max_examples,
        "views": view_summaries,
        "transplanted_mid_view_ranking": ranking,
    }

    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_json = output_dir / "summary.json"
        summary_md = output_dir / "summary.md"
        summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        write_summary_markdown(summary, summary_md)
        print(summary_json)
        print(summary_md)
    else:
        print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
