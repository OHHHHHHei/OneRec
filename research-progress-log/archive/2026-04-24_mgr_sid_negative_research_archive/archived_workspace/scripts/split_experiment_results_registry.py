#!/usr/bin/env python3
"""Split the legacy wide experiment registry into task-specific narrow tables."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Callable


def _find_repo_root() -> Path:
    candidates = [Path.cwd(), *Path(__file__).resolve().parents]
    for candidate in candidates:
        if (candidate / "pyproject.toml").exists() and (candidate / "research-progress-log").exists():
            return candidate
    return Path(__file__).resolve().parents[1]


ROOT = _find_repo_root()
DEFAULT_SOURCE = ROOT / "experiment_results.csv"
DEFAULT_OUTPUT_DIR = ROOT / "research-progress-log" / "experiment_registry"

ValueGetter = str | Callable[[dict[str, str]], str]


TOKENIZER_SCHEMA: list[tuple[str, ValueGetter]] = [
    ("record_id", "record_id"),
    ("recorded_at", "recorded_at"),
    ("run_finished_at", "run_finished_at"),
    ("dataset_key", "dataset_key"),
    ("category", "category"),
    ("split_stem", "split_stem"),
    ("variant", "variant"),
    ("tokenizer_family", "tokenizer_family"),
    ("tokenizer_research_stage", "tokenizer_research_stage"),
    ("tokenizer_source_kind", "tokenizer_source_kind"),
    ("tokenizer_status", "tokenizer_status"),
    ("generated_collision_rate", "tokenizer_generated_collision_rate"),
    ("generated_collision_count", "tokenizer_generated_collision_count"),
    ("downstream_sft_count", "tokenizer_downstream_sft_count"),
    ("downstream_rl_count", "tokenizer_downstream_rl_count"),
    ("train_ckpt_path", "tokenizer_train_ckpt_path"),
    ("train_summary_path", "tokenizer_train_summary_path"),
    ("generated_index_path", "tokenizer_generated_index_path"),
    ("sid_index_path", "tokenizer_sid_index_path"),
    ("data_root", "tokenizer_data_root"),
    ("launch_readme", "tokenizer_launch_readme"),
    ("notes", "tokenizer_notes"),
]


SFT_SCHEMA: list[tuple[str, ValueGetter]] = [
    ("record_id", "record_id"),
    ("recorded_at", "recorded_at"),
    ("run_finished_at", "run_finished_at"),
    ("dataset_key", "dataset_key"),
    ("category", "category"),
    ("split_stem", "split_stem"),
    ("variant", "variant"),
    ("tokenizer_record_id", "tokenizer_record_id"),
    ("tokenizer_variant", "tokenizer_variant"),
    ("tokenizer_family", "tokenizer_family"),
    ("recipe", lambda row: recipe_label(row)),
    ("title_history2sid_enabled", "title_history2sid_enabled"),
    ("alignment_enabled", "alignment_enabled"),
    ("description_task_probability", "description_task_probability"),
    ("alignment_mode", "alignment_mode"),
    ("base_model", "base_model"),
    ("sft_config_path", "sft_config_path"),
    ("sft_output_dir", "sft_output_dir"),
    ("sft_model_path", "sft_model_path"),
    ("result_json_path", "result_json_path"),
    ("wandb_run_name", "wandb_run_name"),
    ("wandb_run_id", "sft_wandb_run_id"),
    ("runtime_gpus", "sft_runtime_gpus"),
    ("batch_size", "sft_batch_size"),
    ("micro_batch_size", "sft_micro_batch_size"),
    ("world_size", "sft_world_size"),
    ("grad_accum_steps", "sft_grad_accum_steps"),
    ("effective_global_batch", "sft_effective_global_batch"),
    ("num_epochs", "sft_num_epochs"),
    ("learning_rate", "sft_learning_rate"),
    ("final_eval_loss", "sft_final_eval_loss"),
    ("final_train_loss", "sft_final_train_loss"),
    ("stop_epoch", "sft_stop_epoch"),
    ("eval_runtime_gpus", "eval_runtime_gpus"),
    ("eval_batch_size", "eval_batch_size"),
    ("eval_num_beams", "eval_num_beams"),
    ("test_example_count", "test_example_count"),
    ("constraint_invalid_total", "constraint_invalid_total"),
    ("ndcg_at_1", "ndcg_at_1"),
    ("ndcg_at_3", "ndcg_at_3"),
    ("ndcg_at_5", "ndcg_at_5"),
    ("ndcg_at_10", "ndcg_at_10"),
    ("ndcg_at_20", "ndcg_at_20"),
    ("ndcg_at_50", "ndcg_at_50"),
    ("hr_at_1", "hr_at_1"),
    ("hr_at_3", "hr_at_3"),
    ("hr_at_5", "hr_at_5"),
    ("hr_at_10", "hr_at_10"),
    ("hr_at_20", "hr_at_20"),
    ("hr_at_50", "hr_at_50"),
    ("tokenizer_collision_rate", "tokenizer_generated_collision_rate"),
    ("tokenizer_collision_count", "tokenizer_generated_collision_count"),
    ("tokenizer_launch_readme", "tokenizer_launch_readme"),
    ("notes", "notes"),
]


RL_SCHEMA: list[tuple[str, ValueGetter]] = [
    ("record_id", "record_id"),
    ("recorded_at", "recorded_at"),
    ("run_finished_at", "run_finished_at"),
    ("dataset_key", "dataset_key"),
    ("category", "category"),
    ("split_stem", "split_stem"),
    ("variant", "variant"),
    ("tokenizer_record_id", "tokenizer_record_id"),
    ("tokenizer_variant", "tokenizer_variant"),
    ("tokenizer_family", "tokenizer_family"),
    ("recipe", lambda row: recipe_label(row)),
    ("rl_output_dir", "rl_output_dir"),
    ("rl_model_path", "rl_model_path"),
    ("rl_source_sft_model_path", "rl_source_sft_model_path"),
    ("result_json_path", "result_json_path"),
    ("wandb_run_name", "wandb_run_name"),
    ("rl_wandb_run_name", "rl_wandb_run_name"),
    ("recovery_status", "rl_recovery_status"),
    ("reference_policy", "rl_reference_policy"),
    ("reward_type", "rl_reward_type"),
    ("beam_search", "rl_beam_search"),
    ("num_generations", "rl_num_generations"),
    ("learning_rate", "rl_learning_rate"),
    ("num_epochs", "rl_num_epochs"),
    ("eval_runtime_gpus", "eval_runtime_gpus"),
    ("eval_batch_size", "eval_batch_size"),
    ("eval_num_beams", "eval_num_beams"),
    ("test_example_count", "test_example_count"),
    ("constraint_invalid_total", "constraint_invalid_total"),
    ("ndcg_at_1", "ndcg_at_1"),
    ("ndcg_at_3", "ndcg_at_3"),
    ("ndcg_at_5", "ndcg_at_5"),
    ("ndcg_at_10", "ndcg_at_10"),
    ("ndcg_at_20", "ndcg_at_20"),
    ("ndcg_at_50", "ndcg_at_50"),
    ("hr_at_1", "hr_at_1"),
    ("hr_at_3", "hr_at_3"),
    ("hr_at_5", "hr_at_5"),
    ("hr_at_10", "hr_at_10"),
    ("hr_at_20", "hr_at_20"),
    ("hr_at_50", "hr_at_50"),
    ("tokenizer_collision_rate", "tokenizer_generated_collision_rate"),
    ("tokenizer_collision_count", "tokenizer_generated_collision_count"),
    ("tokenizer_launch_readme", "tokenizer_launch_readme"),
    ("notes", "notes"),
]


SCOREBOARD_SCHEMA: list[tuple[str, ValueGetter]] = [
    ("record_id", "record_id"),
    ("stage", "stage"),
    ("run_finished_at", "run_finished_at"),
    ("dataset_key", "dataset_key"),
    ("category", "category"),
    ("variant", "variant"),
    ("tokenizer_record_id", "tokenizer_record_id"),
    ("tokenizer_variant", "tokenizer_variant"),
    ("tokenizer_family", "tokenizer_family"),
    ("recipe", lambda row: recipe_label(row)),
    ("ndcg_at_1", "ndcg_at_1"),
    ("ndcg_at_3", "ndcg_at_3"),
    ("ndcg_at_5", "ndcg_at_5"),
    ("ndcg_at_10", "ndcg_at_10"),
    ("hr_at_1", "hr_at_1"),
    ("hr_at_3", "hr_at_3"),
    ("hr_at_5", "hr_at_5"),
    ("hr_at_10", "hr_at_10"),
    ("hr_at_50", "hr_at_50"),
    ("test_example_count", "test_example_count"),
    ("constraint_invalid_total", "constraint_invalid_total"),
    ("result_json_path", "result_json_path"),
    ("model_path", lambda row: first_present(row, "rl_model_path", "sft_model_path")),
    ("notes", "notes"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split experiment_results.csv into tokenizer/SFT/RL registries."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing split registry files. Use only for migration refreshes.",
    )
    return parser.parse_args()


def clean(value: object) -> str:
    if value is None:
        return "-"
    text = str(value).strip()
    return text if text else "-"


def truthy_label(value: str, true_label: str, false_label: str, unknown_label: str = "unknown") -> str:
    lowered = clean(value).lower()
    if lowered == "true":
        return true_label
    if lowered == "false":
        return false_label
    return unknown_label


def desc_probability_label(value: str) -> str:
    text = clean(value)
    if text == "-":
        return "p_unknown"
    try:
        number = float(text)
    except ValueError:
        return f"p_{text}"
    if abs(number - 0.5) < 1e-9:
        return "p05"
    if abs(number) < 1e-9:
        return "p00"
    return f"p{str(number).replace('.', '')}"


def recipe_label(row: dict[str, str]) -> str:
    title = truthy_label(
        row.get("title_history2sid_enabled", ""),
        "title_history2sid_on",
        "title_history2sid_off",
    )
    align_enabled = clean(row.get("alignment_enabled", "")).lower()
    if align_enabled == "true":
        desc = f"desc_align_{desc_probability_label(row.get('description_task_probability', ''))}"
    elif align_enabled == "false":
        desc = "desc_align_off"
    else:
        desc = "desc_align_unknown"
    return f"{title}+{desc}"


def first_present(row: dict[str, str], *keys: str) -> str:
    for key in keys:
        value = clean(row.get(key, "-"))
        if value != "-":
            return value
    return "-"


def project_row(row: dict[str, str], schema: list[tuple[str, ValueGetter]]) -> dict[str, str]:
    projected: dict[str, str] = {}
    for output_name, getter in schema:
        if callable(getter):
            projected[output_name] = clean(getter(row))
        else:
            projected[output_name] = clean(row.get(getter, "-"))
    return projected


def float_key(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "-"))
    except ValueError:
        return float("-inf")


def write_csv(path: Path, rows: list[dict[str, str]], schema: list[tuple[str, ValueGetter]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [name for name, _ in schema]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    with args.source.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        source_rows = list(reader)

    tokenizer_rows = [
        project_row(row, TOKENIZER_SCHEMA) for row in source_rows if row.get("stage") == "tokenizer"
    ]
    sft_rows = [
        project_row(row, SFT_SCHEMA) for row in source_rows if row.get("stage") == "sft_eval"
    ]
    rl_rows = [
        project_row(row, RL_SCHEMA) for row in source_rows if row.get("stage") == "rl_eval"
    ]
    scoreboard_rows = [
        project_row(row, SCOREBOARD_SCHEMA)
        for row in source_rows
        if row.get("stage") in {"sft_eval", "rl_eval"}
    ]
    scoreboard_rows.sort(
        key=lambda row: (
            row.get("dataset_key", ""),
            -float_key(row, "ndcg_at_10"),
            -float_key(row, "hr_at_10"),
            row.get("stage", ""),
            row.get("record_id", ""),
        )
    )

    outputs = [
        ("tokenizer_registry.csv", tokenizer_rows, TOKENIZER_SCHEMA),
        ("sft_registry.csv", sft_rows, SFT_SCHEMA),
        ("rl_registry.csv", rl_rows, RL_SCHEMA),
        ("downstream_scoreboard.csv", scoreboard_rows, SCOREBOARD_SCHEMA),
    ]
    existing = [args.output_dir / filename for filename, _, _ in outputs if (args.output_dir / filename).exists()]
    if existing and not args.overwrite:
        paths = "\n".join(f"  - {path}" for path in existing)
        raise SystemExit(
            "Refusing to overwrite existing split registry files without --overwrite:\n"
            f"{paths}\n"
            "This guard prevents losing new rows that may have been added directly to the split registries."
        )
    for filename, rows, schema in outputs:
        write_csv(args.output_dir / filename, rows, schema)
        print(f"[write] {filename}: rows={len(rows)} cols={len(schema)}")


if __name__ == "__main__":
    main()
