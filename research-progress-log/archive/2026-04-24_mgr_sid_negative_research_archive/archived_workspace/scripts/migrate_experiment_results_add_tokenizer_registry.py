from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path("/home/leejt/OneRec")
CSV_PATH = ROOT / "experiment_results.csv"


TOKENIZER_COLUMNS = [
    "tokenizer_record_id",
    "tokenizer_variant",
    "tokenizer_family",
    "tokenizer_research_stage",
    "tokenizer_source_kind",
    "tokenizer_train_ckpt_path",
    "tokenizer_train_summary_path",
    "tokenizer_generated_index_path",
    "tokenizer_sid_index_path",
    "tokenizer_data_root",
    "tokenizer_launch_readme",
    "tokenizer_status",
    "tokenizer_generated_collision_rate",
    "tokenizer_generated_collision_count",
    "tokenizer_downstream_sft_count",
    "tokenizer_downstream_rl_count",
    "tokenizer_notes",
]


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return reader.fieldnames or [], rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def default_row(fieldnames: list[str]) -> dict[str, str]:
    return {k: "-" for k in fieldnames}


def ensure_tokenizer_columns(fieldnames: list[str]) -> list[str]:
    existing = set(fieldnames)
    new_cols = [c for c in TOKENIZER_COLUMNS if c not in existing]
    if not new_cols:
        return fieldnames
    insert_after = fieldnames.index("variant") + 1 if "variant" in fieldnames else len(fieldnames)
    return fieldnames[:insert_after] + new_cols + fieldnames[insert_after:]


def load_json(path_str: str) -> dict | list:
    path = Path(path_str)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def collision_stats_for_index(path_str: str) -> tuple[str, str]:
    if path_str in {"", "-"}:
        return "-", "-"
    path = Path(path_str)
    if not path.exists():
        return "-", "-"
    raw = load_json(path_str)
    if not isinstance(raw, dict):
        return "-", "-"
    values = [str(v) for v in raw.values()]
    total = len(values)
    unique = len(set(values))
    collision_count = total - unique
    rate = (collision_count / total) if total else 0.0
    return f"{rate:.10f}", str(collision_count)


def build_tokenizer_specs() -> dict[str, dict[str, str]]:
    specs: dict[str, dict[str, str]] = {
        "tok_industrial_original_semantic": {
            "recorded_at": "2026-04-14",
            "run_finished_at": "-",
            "dataset_key": "industrial",
            "category": "Industrial_and_Scientific",
            "split_stem": "Industrial_and_Scientific_5_2016-10-2018-11",
            "stage": "tokenizer",
            "variant": "original_semantic_industrial",
            "tokenizer_record_id": "tok_industrial_original_semantic",
            "tokenizer_variant": "original_semantic_industrial",
            "tokenizer_family": "original_semantic",
            "tokenizer_research_stage": "baseline",
            "tokenizer_source_kind": "repo_index",
            "tokenizer_train_ckpt_path": "-",
            "tokenizer_train_summary_path": "-",
            "tokenizer_generated_index_path": "-",
            "tokenizer_sid_index_path": "./data/Amazon/index/Industrial_and_Scientific.index.json",
            "tokenizer_data_root": "./data/Amazon",
            "tokenizer_launch_readme": "-",
            "tokenizer_notes": "Repo default semantic SID space used by the original MiniOneRec SFT/RL baselines.",
        },
        "tok_office_original_semantic": {
            "recorded_at": "2026-04-14",
            "run_finished_at": "-",
            "dataset_key": "office",
            "category": "Office_Products",
            "split_stem": "Office_Products_5_2016-10-2018-11",
            "stage": "tokenizer",
            "variant": "original_semantic_office",
            "tokenizer_record_id": "tok_office_original_semantic",
            "tokenizer_variant": "original_semantic_office",
            "tokenizer_family": "original_semantic",
            "tokenizer_research_stage": "baseline",
            "tokenizer_source_kind": "repo_index",
            "tokenizer_train_ckpt_path": "-",
            "tokenizer_train_summary_path": "-",
            "tokenizer_generated_index_path": "-",
            "tokenizer_sid_index_path": "./data/Amazon/index/Office_Products.index.json",
            "tokenizer_data_root": "./data/Amazon",
            "tokenizer_launch_readme": "-",
            "tokenizer_notes": "Repo default semantic SID space used by the Office legacy baselines.",
        },
        "tok_industrial_mgr_upstream_baseline": {
            "recorded_at": "2026-04-09",
            "run_finished_at": "-",
            "dataset_key": "industrial",
            "category": "Industrial_and_Scientific",
            "split_stem": "Industrial_and_Scientific_5_2016-10-2018-11",
            "stage": "tokenizer",
            "variant": "mgr_upstream_baseline",
            "tokenizer_record_id": "tok_industrial_mgr_upstream_baseline",
            "tokenizer_variant": "mgr_upstream_baseline",
            "tokenizer_family": "mgr_v1_upstream",
            "tokenizer_research_stage": "v1_upstream",
            "tokenizer_source_kind": "generated",
            "tokenizer_train_ckpt_path": "/data/leejt/OneRec/output_weights/experiments/mgr_sid_v1_upstream/industrial_baseline/Apr-09-2026_23-09-22/best_collision_model.pth",
            "tokenizer_train_summary_path": "/data/leejt/OneRec/output_weights/experiments/mgr_sid_v1_upstream/industrial_baseline/Apr-09-2026_23-09-22/summary.json",
            "tokenizer_generated_index_path": "/data/leejt/OneRec/output_weights/experiments/mgr_sid_v1_upstream/generated_indices/Industrial_and_Scientific.mgr_upstream_baseline.index.json",
            "tokenizer_sid_index_path": "./data_experiment/Amazon/mgr_upstream_baseline/index/Industrial_and_Scientific.index.json",
            "tokenizer_data_root": "./data_experiment/Amazon/mgr_upstream_baseline",
            "tokenizer_launch_readme": "./research-progress-log/experiment_launches/2026-04-09_mgr_sid_v1_upstream/README.md",
            "tokenizer_notes": "First upstream-aligned MGR-SID baseline tokenizer; downstream SFT already completed.",
        },
        "tok_industrial_mgr_upstream_hierarchy": {
            "recorded_at": "2026-04-09",
            "run_finished_at": "-",
            "dataset_key": "industrial",
            "category": "Industrial_and_Scientific",
            "split_stem": "Industrial_and_Scientific_5_2016-10-2018-11",
            "stage": "tokenizer",
            "variant": "mgr_upstream_hierarchy",
            "tokenizer_record_id": "tok_industrial_mgr_upstream_hierarchy",
            "tokenizer_variant": "mgr_upstream_hierarchy",
            "tokenizer_family": "mgr_v1_upstream",
            "tokenizer_research_stage": "v1_upstream",
            "tokenizer_source_kind": "generated",
            "tokenizer_train_ckpt_path": "/data/leejt/OneRec/output_weights/experiments/mgr_sid_v1_upstream/industrial_hierarchy_reg/Apr-09-2026_23-09-22/best_collision_model.pth",
            "tokenizer_train_summary_path": "/data/leejt/OneRec/output_weights/experiments/mgr_sid_v1_upstream/industrial_hierarchy_reg/Apr-09-2026_23-09-22/summary.json",
            "tokenizer_generated_index_path": "/data/leejt/OneRec/output_weights/experiments/mgr_sid_v1_upstream/generated_indices/Industrial_and_Scientific.mgr_upstream_hierarchy.index.json",
            "tokenizer_sid_index_path": "./data_experiment/Amazon/mgr_upstream_hierarchy/index/Industrial_and_Scientific.index.json",
            "tokenizer_data_root": "./data_experiment/Amazon/mgr_upstream_hierarchy",
            "tokenizer_launch_readme": "./research-progress-log/experiment_launches/2026-04-09_mgr_sid_v1_upstream/README.md",
            "tokenizer_notes": "First hierarchy-regularized upstream tokenizer; downstream SFT already completed.",
        },
        "tok_industrial_mgr_upstream_uniform": {
            "recorded_at": "2026-04-09",
            "run_finished_at": "-",
            "dataset_key": "industrial",
            "category": "Industrial_and_Scientific",
            "split_stem": "Industrial_and_Scientific_5_2016-10-2018-11",
            "stage": "tokenizer",
            "variant": "mgr_upstream_uniform",
            "tokenizer_record_id": "tok_industrial_mgr_upstream_uniform",
            "tokenizer_variant": "mgr_upstream_uniform",
            "tokenizer_family": "mgr_v1_upstream",
            "tokenizer_research_stage": "v1_upstream",
            "tokenizer_source_kind": "generated",
            "tokenizer_train_ckpt_path": "/data/leejt/OneRec/output_weights/experiments/mgr_sid_v1_upstream/industrial_uniform_reg/Apr-09-2026_23-09-22/best_collision_model.pth",
            "tokenizer_train_summary_path": "/data/leejt/OneRec/output_weights/experiments/mgr_sid_v1_upstream/industrial_uniform_reg/Apr-09-2026_23-09-22/summary.json",
            "tokenizer_generated_index_path": "/data/leejt/OneRec/output_weights/experiments/mgr_sid_v1_upstream/generated_indices/Industrial_and_Scientific.mgr_upstream_uniform.index.json",
            "tokenizer_sid_index_path": "/data/leejt/OneRec/output_weights/experiments/mgr_sid_v1_upstream/generated_indices/Industrial_and_Scientific.mgr_upstream_uniform.index.json",
            "tokenizer_data_root": "-",
            "tokenizer_launch_readme": "./research-progress-log/experiment_launches/2026-04-09_mgr_sid_v1_upstream/README.md",
            "tokenizer_notes": "Uniform graph-regularized upstream tokenizer; generated SID kept, but not pushed downstream.",
        },
        "tok_industrial_mgr_tokenizer_v2_offline": {
            "recorded_at": "2026-04-11",
            "run_finished_at": "-",
            "dataset_key": "industrial",
            "category": "Industrial_and_Scientific",
            "split_stem": "Industrial_and_Scientific_5_2016-10-2018-11",
            "stage": "tokenizer",
            "variant": "mgr_tokenizer_v2_offline",
            "tokenizer_record_id": "tok_industrial_mgr_tokenizer_v2_offline",
            "tokenizer_variant": "mgr_tokenizer_v2_offline",
            "tokenizer_family": "mgr_v2",
            "tokenizer_research_stage": "v2",
            "tokenizer_source_kind": "generated",
            "tokenizer_train_ckpt_path": "/data/leejt/OneRec/output_weights/experiments/mgr_sid_tokenizer_v2/industrial_offline_combined/Apr-11-2026_01-36-05/best_collision_model.pth",
            "tokenizer_train_summary_path": "/data/leejt/OneRec/output_weights/experiments/mgr_sid_tokenizer_v2/industrial_offline_combined/Apr-11-2026_01-36-05/summary.json",
            "tokenizer_generated_index_path": "/data/leejt/OneRec/output_weights/experiments/mgr_sid_tokenizer_v2/generated_indices/Industrial_and_Scientific.mgr_tokenizer_v2_offline.index.json",
            "tokenizer_sid_index_path": "./data_experiment/Amazon/mgr_tokenizer_v2_offline/index/Industrial_and_Scientific.index.json",
            "tokenizer_data_root": "./data_experiment/Amazon/mgr_tokenizer_v2_offline",
            "tokenizer_launch_readme": "./research-progress-log/experiment_launches/2026-04-11_mgr_sid_tokenizer_v2_r005/README.md",
            "tokenizer_notes": "Current strongest tokenizer line before stage-2/3 refinements; already pushed to SFT and RL.",
        },
        "tok_industrial_stage2_r202a_stopgrad": {
            "recorded_at": "2026-04-13",
            "run_finished_at": "-",
            "dataset_key": "industrial",
            "category": "Industrial_and_Scientific",
            "split_stem": "Industrial_and_Scientific_5_2016-10-2018-11",
            "stage": "tokenizer",
            "variant": "stage2_r202a_stopgrad",
            "tokenizer_record_id": "tok_industrial_stage2_r202a_stopgrad",
            "tokenizer_variant": "stage2_r202a_stopgrad",
            "tokenizer_family": "stage2_retention",
            "tokenizer_research_stage": "stage2",
            "tokenizer_source_kind": "generated",
            "tokenizer_train_ckpt_path": "/data/leejt/OneRec/output_weights/experiments/mgr_sid_stage2_retention_20260413/industrial_r202a_stopgrad/Apr-13-2026_00-11-11/best_collision_model.pth",
            "tokenizer_train_summary_path": "/data/leejt/OneRec/output_weights/experiments/mgr_sid_stage2_retention_20260413/industrial_r202a_stopgrad/Apr-13-2026_00-11-11/summary.json",
            "tokenizer_generated_index_path": "/data/leejt/OneRec/output_weights/experiments/mgr_sid_stage2_retention_20260413/generated_indices/Industrial_and_Scientific.stage2_r202a_stopgrad.index.json",
            "tokenizer_sid_index_path": "./data_experiment/Amazon/stage2_r202a_stopgrad/index/Industrial_and_Scientific.index.json",
            "tokenizer_data_root": "./data_experiment/Amazon/stage2_r202a_stopgrad",
            "tokenizer_launch_readme": "./research-progress-log/experiment_launches/2026-04-13_mgr_sid_stage2_stopgrad_industrial/README.md",
            "tokenizer_notes": "Best stage-2 tokenizer-side branch; already pushed to SFT (`R208`).",
        },
        "tok_industrial_stage2_r202b_retry075": {
            "recorded_at": "2026-04-13",
            "run_finished_at": "-",
            "dataset_key": "industrial",
            "category": "Industrial_and_Scientific",
            "split_stem": "Industrial_and_Scientific_5_2016-10-2018-11",
            "stage": "tokenizer",
            "variant": "stage2_r202b_retry075",
            "tokenizer_record_id": "tok_industrial_stage2_r202b_retry075",
            "tokenizer_variant": "stage2_r202b_retry075",
            "tokenizer_family": "stage2_retention",
            "tokenizer_research_stage": "stage2",
            "tokenizer_source_kind": "generated",
            "tokenizer_train_ckpt_path": "/data/leejt/OneRec/output_weights/experiments/mgr_sid_stage2_retention_20260413/industrial_r202b_retry075/Apr-13-2026_01-48-27/best_collision_model.pth",
            "tokenizer_train_summary_path": "/data/leejt/OneRec/output_weights/experiments/mgr_sid_stage2_retention_20260413/industrial_r202b_retry075/Apr-13-2026_01-48-27/summary.json",
            "tokenizer_generated_index_path": "/data/leejt/OneRec/output_weights/experiments/mgr_sid_stage2_retention_20260413/generated_indices/Industrial_and_Scientific.stage2_r202b_retry075.index.json",
            "tokenizer_sid_index_path": "/data/leejt/OneRec/output_weights/experiments/mgr_sid_stage2_retention_20260413/generated_indices/Industrial_and_Scientific.stage2_r202b_retry075.index.json",
            "tokenizer_data_root": "-",
            "tokenizer_launch_readme": "./research-progress-log/experiment_launches/2026-04-13_mgr_sid_stage2_stopgrad_industrial/README.md",
            "tokenizer_notes": "Stage-2 retry branch with generated SID kept for reference; not pushed downstream.",
        },
        "tok_industrial_stage2_r205_stopgrad_kl": {
            "recorded_at": "2026-04-13",
            "run_finished_at": "-",
            "dataset_key": "industrial",
            "category": "Industrial_and_Scientific",
            "split_stem": "Industrial_and_Scientific_5_2016-10-2018-11",
            "stage": "tokenizer",
            "variant": "stage2_r205_stopgrad_kl",
            "tokenizer_record_id": "tok_industrial_stage2_r205_stopgrad_kl",
            "tokenizer_variant": "stage2_r205_stopgrad_kl",
            "tokenizer_family": "stage2_retention",
            "tokenizer_research_stage": "stage2",
            "tokenizer_source_kind": "generated",
            "tokenizer_train_ckpt_path": "/data/leejt/OneRec/output_weights/experiments/mgr_sid_stage2_retention_20260413/industrial_r205_stopgrad_kl/Apr-13-2026_02-15-01/best_collision_model.pth",
            "tokenizer_train_summary_path": "/data/leejt/OneRec/output_weights/experiments/mgr_sid_stage2_retention_20260413/industrial_r205_stopgrad_kl/Apr-13-2026_02-15-01/summary.json",
            "tokenizer_generated_index_path": "/data/leejt/OneRec/output_weights/experiments/mgr_sid_stage2_retention_20260413/generated_indices/Industrial_and_Scientific.stage2_r205_stopgrad_kl.index.json",
            "tokenizer_sid_index_path": "/data/leejt/OneRec/output_weights/experiments/mgr_sid_stage2_retention_20260413/generated_indices/Industrial_and_Scientific.stage2_r205_stopgrad_kl.index.json",
            "tokenizer_data_root": "-",
            "tokenizer_launch_readme": "./research-progress-log/experiment_launches/2026-04-13_mgr_sid_stage2_semantic_retention_industrial/README.md",
            "tokenizer_notes": "Stage-2 semantic-retention branch with generated SID kept for reference; not pushed downstream.",
        },
        "tok_industrial_stage3_r401b_g005": {
            "recorded_at": "2026-04-14",
            "run_finished_at": "-",
            "dataset_key": "industrial",
            "category": "Industrial_and_Scientific",
            "split_stem": "Industrial_and_Scientific_5_2016-10-2018-11",
            "stage": "tokenizer",
            "variant": "stage3_r401b_g005",
            "tokenizer_record_id": "tok_industrial_stage3_r401b_g005",
            "tokenizer_variant": "stage3_r401b_g005",
            "tokenizer_family": "stage3_prefix_retained",
            "tokenizer_research_stage": "stage3",
            "tokenizer_source_kind": "generated",
            "tokenizer_train_ckpt_path": "/data/leejt/OneRec/output_weights/experiments/mgr_sid_stage3_prefix_retained_20260414/industrial_r401b_g005/Apr-14-2026_00-47-31/best_collision_model.pth",
            "tokenizer_train_summary_path": "/data/leejt/OneRec/output_weights/experiments/mgr_sid_stage3_prefix_retained_20260414/industrial_r401b_g005/Apr-14-2026_00-47-31/summary.json",
            "tokenizer_generated_index_path": "/data/leejt/OneRec/output_weights/experiments/mgr_sid_stage3_prefix_retained_20260414/generated_indices/Industrial_and_Scientific.stage3_r401b_g005.index.json",
            "tokenizer_sid_index_path": "./data_experiment/Amazon/stage3_r401b_g005/index/Industrial_and_Scientific.index.json",
            "tokenizer_data_root": "./data_experiment/Amazon/stage3_r401b_g005",
            "tokenizer_launch_readme": "./research-progress-log/experiment_launches/2026-04-14_mgr_sid_stage3_prefix_retained_industrial/README.md",
            "tokenizer_notes": "First stage-3 candidate codebook space; full downstream SFT/RL not yet launched.",
        },
        "tok_industrial_stage3_r401d_g005_a005": {
            "recorded_at": "2026-04-14",
            "run_finished_at": "-",
            "dataset_key": "industrial",
            "category": "Industrial_and_Scientific",
            "split_stem": "Industrial_and_Scientific_5_2016-10-2018-11",
            "stage": "tokenizer",
            "variant": "stage3_r401d_g005_a005",
            "tokenizer_record_id": "tok_industrial_stage3_r401d_g005_a005",
            "tokenizer_variant": "stage3_r401d_g005_a005",
            "tokenizer_family": "stage3_prefix_retained",
            "tokenizer_research_stage": "stage3",
            "tokenizer_source_kind": "generated",
            "tokenizer_train_ckpt_path": "/data/leejt/OneRec/output_weights/experiments/mgr_sid_stage3_prefix_retained_20260414/industrial_r401d_g005_a005/Apr-14-2026_17-03-32/best_collision_model.pth",
            "tokenizer_train_summary_path": "/data/leejt/OneRec/output_weights/experiments/mgr_sid_stage3_prefix_retained_20260414/industrial_r401d_g005_a005/Apr-14-2026_17-03-32/summary.json",
            "tokenizer_generated_index_path": "/data/leejt/OneRec/output_weights/experiments/mgr_sid_stage3_prefix_retained_20260414/generated_indices/Industrial_and_Scientific.stage3_r401d_g005_a005.index.json",
            "tokenizer_sid_index_path": "./data_experiment/Amazon/stage3_r401d_g005_a005/index/Industrial_and_Scientific.index.json",
            "tokenizer_data_root": "./data_experiment/Amazon/stage3_r401d_g005_a005",
            "tokenizer_launch_readme": "./research-progress-log/experiment_launches/2026-04-14_mgr_sid_stage3_prefix_retained_industrial/README.md",
            "tokenizer_notes": "Second stage-3 candidate codebook space with codebook anchor; full downstream SFT/RL not yet launched.",
        },
    }
    for spec in specs.values():
        rate, count = collision_stats_for_index(
            spec.get("tokenizer_generated_index_path", "-")
            if spec.get("tokenizer_generated_index_path", "-") not in {"", "-"}
            else spec.get("tokenizer_sid_index_path", "-")
        )
        spec["tokenizer_generated_collision_rate"] = rate
        spec["tokenizer_generated_collision_count"] = count
    return specs


def infer_tokenizer_record_id(row: dict[str, str]) -> str:
    if row.get("stage") == "tokenizer":
        return row.get("tokenizer_record_id", row.get("record_id", "-"))

    sid_index = row.get("sft_sid_index_path", "-")
    record_id = row.get("record_id", "")
    dataset_key = row.get("dataset_key", "")
    category = row.get("category", "")

    if "mgr_upstream_baseline" in sid_index or "mgr_upstream_baseline" in record_id:
        return "tok_industrial_mgr_upstream_baseline"
    if "mgr_upstream_hierarchy" in sid_index or "mgr_upstream_hierarchy" in record_id:
        return "tok_industrial_mgr_upstream_hierarchy"
    if "mgr_tokenizer_v2_offline" in sid_index or "mgr_tokenizer_v2" in record_id:
        return "tok_industrial_mgr_tokenizer_v2_offline"
    if "stage2_r202a_stopgrad" in sid_index or "mgr_stage2_r202a" in record_id:
        return "tok_industrial_stage2_r202a_stopgrad"
    if "stage3_r401b_g005" in sid_index or "stage3_r401b" in record_id:
        return "tok_industrial_stage3_r401b_g005"
    if "stage3_r401d_g005_a005" in sid_index or "stage3_r401d" in record_id:
        return "tok_industrial_stage3_r401d_g005_a005"

    if dataset_key == "office" or category == "Office_Products":
        return "tok_office_original_semantic"
    if dataset_key == "industrial" or category == "Industrial_and_Scientific":
        return "tok_industrial_original_semantic"
    return "-"


def backfill_row_tokenizer_fields(row: dict[str, str], spec: dict[str, str]) -> None:
    for key in TOKENIZER_COLUMNS:
        if key in spec:
            row[key] = spec[key]


def main() -> None:
    fieldnames, rows = read_csv(CSV_PATH)
    fieldnames = ensure_tokenizer_columns(fieldnames)
    specs = build_tokenizer_specs()

    non_tokenizer_rows = [row for row in rows if row.get("stage") != "tokenizer"]

    sft_count: dict[str, int] = {}
    rl_count: dict[str, int] = {}
    for row in non_tokenizer_rows:
        tok_id = infer_tokenizer_record_id(row)
        row["tokenizer_record_id"] = tok_id
        spec = specs.get(tok_id)
        if spec:
            backfill_row_tokenizer_fields(row, spec)
        if row.get("stage") == "sft_eval" and tok_id != "-":
            sft_count[tok_id] = sft_count.get(tok_id, 0) + 1
        if row.get("stage") == "rl_eval" and tok_id != "-":
            rl_count[tok_id] = rl_count.get(tok_id, 0) + 1

    tokenizer_rows: list[dict[str, str]] = []
    for tok_id, spec in specs.items():
        row = default_row(fieldnames)
        row.update(spec)
        row["record_id"] = tok_id
        row["tokenizer_record_id"] = tok_id
        row["tokenizer_downstream_sft_count"] = str(sft_count.get(tok_id, 0))
        row["tokenizer_downstream_rl_count"] = str(rl_count.get(tok_id, 0))
        if rl_count.get(tok_id, 0) > 0:
            row["tokenizer_status"] = "rl_evaluated"
        elif sft_count.get(tok_id, 0) > 0:
            row["tokenizer_status"] = "sft_evaluated"
        elif spec.get("tokenizer_data_root", "-") not in {"", "-"}:
            row["tokenizer_status"] = "sft_ready_not_run"
        elif spec.get("tokenizer_generated_index_path", "-") not in {"", "-"}:
            row["tokenizer_status"] = "generated_only"
        else:
            row["tokenizer_status"] = "baseline_available"
        tokenizer_rows.append(row)

    merged_rows = non_tokenizer_rows + tokenizer_rows
    write_csv(CSV_PATH, fieldnames, merged_rows)
    print(f"Updated {CSV_PATH}")
    print(f"Non-tokenizer rows: {len(non_tokenizer_rows)}")
    print(f"Tokenizer rows appended: {len(tokenizer_rows)}")


if __name__ == "__main__":
    main()
