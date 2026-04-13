from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml


ROOT = Path("/home/leejt/OneRec")
MAIN_CSV = ROOT / "experiment_results.csv"
LEGACY_CSV = ROOT / "legacy_experiment_results.csv"


SFT_COLUMNS = [
    "sft_config_path",
    "sft_train_file",
    "sft_eval_file",
    "sft_sid_index_path",
    "sft_item_meta_path",
    "sft_seed",
    "sft_warmup_steps",
    "sft_freeze_llm",
    "sft_group_by_length",
    "sft_load_best_model_at_end",
    "sft_early_stopping_patience",
    "sft_train_from_scratch",
    "sft_launcher",
    "sft_cuda_visible_devices",
    "sft_nproc_per_node",
    "sft_wandb_project",
    "sft_wandb_run_id",
    "sft_report_to",
]


RL_COLUMNS = [
    "rl_output_dir",
    "rl_model_path",
    "rl_source_sft_model_path",
    "rl_wandb_run_name",
    "rl_recovery_status",
    "rl_reference_policy",
    "rl_source_code_window",
    "rl_base_model_source",
    "rl_train_batch_size",
    "rl_gradient_accum_steps",
    "rl_world_size",
    "rl_effective_global_batch",
    "rl_num_epochs",
    "rl_learning_rate",
    "rl_eval_steps",
    "rl_save_steps",
    "rl_bf16",
    "rl_fp16",
    "rl_reward_type",
    "rl_beam_search",
    "rl_num_generations",
    "rl_max_completion_length",
    "rl_resume_from_checkpoint",
]


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return reader.fieldnames or [], rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def default_row(fieldnames: list[str]) -> dict[str, str]:
    return {k: "-" for k in fieldnames}


def ensure_registry_columns(fieldnames: list[str]) -> list[str]:
    if "eval_runtime_gpus" in fieldnames:
        eval_idx = fieldnames.index("eval_runtime_gpus")
    else:
        eval_idx = len(fieldnames)

    existing = set(fieldnames)
    sft_new = [c for c in SFT_COLUMNS if c not in existing]
    with_sft = fieldnames[:eval_idx] + sft_new + fieldnames[eval_idx:]

    if "notes" not in with_sft:
        existing2 = set(with_sft)
        return with_sft + [c for c in RL_COLUMNS if c not in existing2]

    notes_idx = with_sft.index("notes")
    existing2 = set(with_sft)
    rl_new = [c for c in RL_COLUMNS if c not in existing2]
    return with_sft[:notes_idx] + rl_new + with_sft[notes_idx:]


def enrich_sft_row(row: dict[str, str]) -> None:
    rid = row.get("record_id")
    if rid == "sft_industrial_mgr_tokenizer_v2_offline_20260411_024035":
        row.update(
            {
                "sft_config_path": "./config/experiments/sft_industrial_mgr_tokenizer_v2_offline.yaml",
                "sft_train_file": "./data_experiment/Amazon/mgr_tokenizer_v2_offline/train/Industrial_and_Scientific_5_2016-10-2018-11.csv",
                "sft_eval_file": "./data_experiment/Amazon/mgr_tokenizer_v2_offline/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv",
                "sft_sid_index_path": "./data_experiment/Amazon/mgr_tokenizer_v2_offline/index/Industrial_and_Scientific.index.json",
                "sft_item_meta_path": "./data_experiment/Amazon/mgr_tokenizer_v2_offline/index/Industrial_and_Scientific.item.json",
                "sft_seed": "42",
                "sft_warmup_steps": "20",
                "sft_freeze_llm": "false",
                "sft_group_by_length": "false",
                "sft_load_best_model_at_end": "true",
                "sft_early_stopping_patience": "3",
                "sft_train_from_scratch": "false",
                "sft_launcher": "torchrun",
                "sft_cuda_visible_devices": "2,3,4,5",
                "sft_nproc_per_node": "4",
                "sft_wandb_project": "OneRec",
                "sft_wandb_run_id": "dsr1j9md",
                "sft_report_to": "wandb",
            }
        )
        return

    if row.get("stage") == "sft_eval":
        # Conservative generic backfill for standard SFT rows:
        # only populate values that are directly implied by existing path/split fields.
        split_stem = row.get("split_stem", "-")
        category = row.get("category", "-")
        if split_stem not in {"", "-"}:
            if row.get("sft_train_file", "-") == "-":
                row["sft_train_file"] = f"./data/Amazon/train/{split_stem}.csv"
            if row.get("sft_eval_file", "-") == "-":
                row["sft_eval_file"] = f"./data/Amazon/valid/{split_stem}.csv"
        if category not in {"", "-"} and row.get("sft_sid_index_path", "-") == "-" and "data_experiment" not in row.get("sft_output_dir", ""):
            row["sft_sid_index_path"] = f"./data/Amazon/index/{category}.index.json"
            row["sft_item_meta_path"] = f"./data/Amazon/index/{category}.item.json"


def split_stem_for_category(category: str) -> str:
    mapping = {
        "Industrial_and_Scientific": "Industrial_and_Scientific_5_2016-10-2018-11",
        "Office_Products": "Office_Products_5_2016-10-2018-11",
    }
    return mapping.get(category, "-")


def metrics_from_result_json(result_json: str) -> dict[str, str]:
    path = ROOT / result_json.lstrip("./")
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    if isinstance(payload, dict) and "NDCG" in payload and "HR" in payload:
        ndcg = payload["NDCG"]
        hr = payload["HR"]
        return {
            "ndcg_at_1": f"{ndcg[0]:.8f}",
            "ndcg_at_3": f"{ndcg[1]:.8f}",
            "ndcg_at_5": f"{ndcg[2]:.8f}",
            "ndcg_at_10": f"{ndcg[3]:.8f}",
            "ndcg_at_20": f"{ndcg[4]:.8f}",
            "ndcg_at_50": f"{ndcg[5]:.8f}",
            "hr_at_1": f"{hr[0]:.8f}",
            "hr_at_3": f"{hr[1]:.8f}",
            "hr_at_5": f"{hr[2]:.8f}",
            "hr_at_10": f"{hr[3]:.8f}",
            "hr_at_20": f"{hr[4]:.8f}",
            "hr_at_50": f"{hr[5]:.8f}",
            "test_example_count": str(payload.get("example_num", "-")),
            "constraint_invalid_total": str(payload.get("constraint_invalid_total", "-")),
        }
    if isinstance(payload, list):
        ks = [1, 3, 5, 10, 20, 50]
        hr_hits = {k: 0 for k in ks}
        ndcg_sum = {k: 0.0 for k in ks}
        total = len(payload)
        for row in payload:
            target = row["output"].strip()
            preds = [p.strip() for p in row.get("predict", [])]
            try:
                rank = preds.index(target) + 1
            except ValueError:
                rank = None
            for k in ks:
                if rank is not None and rank <= k:
                    hr_hits[k] += 1
                    ndcg_sum[k] += 1.0 / ((rank + 1).bit_length() if False else 1.0)
        # Use explicit log2 to avoid cleverness.
        import math

        ndcg = {}
        for k in ks:
            if total == 0:
                ndcg[k] = 0.0
            else:
                s = 0.0
                for row in payload:
                    target = row["output"].strip()
                    preds = [p.strip() for p in row.get("predict", [])]
                    try:
                        rank = preds.index(target) + 1
                    except ValueError:
                        rank = None
                    if rank is not None and rank <= k:
                        s += 1.0 / math.log2(rank + 1)
                ndcg[k] = s / total
        return {
            "ndcg_at_1": f"{ndcg[1]:.8f}",
            "ndcg_at_3": f"{ndcg[3]:.8f}",
            "ndcg_at_5": f"{ndcg[5]:.8f}",
            "ndcg_at_10": f"{ndcg[10]:.8f}",
            "ndcg_at_20": f"{ndcg[20]:.8f}",
            "ndcg_at_50": f"{ndcg[50]:.8f}",
            "hr_at_1": f"{hr_hits[1] / total:.8f}" if total else "0.00000000",
            "hr_at_3": f"{hr_hits[3] / total:.8f}" if total else "0.00000000",
            "hr_at_5": f"{hr_hits[5] / total:.8f}" if total else "0.00000000",
            "hr_at_10": f"{hr_hits[10] / total:.8f}" if total else "0.00000000",
            "hr_at_20": f"{hr_hits[20] / total:.8f}" if total else "0.00000000",
            "hr_at_50": f"{hr_hits[50] / total:.8f}" if total else "0.00000000",
            "test_example_count": str(total),
            "constraint_invalid_total": "-",
        }
    return {}


def convert_legacy_sft_row(main_fields: list[str], legacy: dict[str, str]) -> dict[str, str]:
    row = default_row(main_fields)
    category = legacy["category"]
    split_stem = split_stem_for_category(category)
    row.update(
        {
            "record_id": legacy["record_id"],
            "recorded_at": "-",
            "run_finished_at": "-",
            "dataset_key": legacy["dataset_key"],
            "category": category,
            "split_stem": split_stem,
            "stage": legacy["stage"],
            "variant": legacy["variant"],
            "title_history2sid_enabled": legacy["title_history2sid_enabled"] or "-",
            "alignment_enabled": legacy["alignment_enabled"] or "-",
            "description_task_probability": legacy["description_task_probability"] or "-",
            "alignment_mode": legacy["alignment_mode"] or "-",
            "base_model": legacy["base_model_source"] or "-",
            "git_head": legacy["source_code_window"] or "-",
            "git_dirty": "-",
            "sft_output_dir": legacy["source_output_dir"] or "-",
            "sft_model_path": legacy["source_model_path"] or "-",
            "result_json_path": legacy["reeval_result_json"] or "-",
            "temp_result_dir": "-",
            "wandb_run_name": legacy["matched_wandb_run"] or "-",
            "sft_runtime_gpus": legacy["world_size"] or "-",
            "sft_batch_size": legacy["effective_global_batch"] or "-",
            "sft_micro_batch_size": legacy["per_device_batch_size"] or "-",
            "sft_world_size": legacy["world_size"] or "-",
            "sft_grad_accum_steps": legacy["grad_accum_steps"] or "-",
            "sft_effective_global_batch": legacy["effective_global_batch"] or "-",
            "sft_num_epochs": legacy["num_epochs"] or "-",
            "sft_learning_rate": legacy["learning_rate"] or "-",
            "sft_cutoff_len": "512",
            "sft_eval_step": legacy["eval_steps"] or "-",
            "sft_final_eval_loss": "-",
            "sft_final_train_loss": "-",
            "sft_stop_epoch": "-",
            "sft_config_path": "-",
            "sft_train_file": f"./data/Amazon/train/{split_stem}.csv" if split_stem != "-" else "-",
            "sft_eval_file": f"./data/Amazon/valid/{split_stem}.csv" if split_stem != "-" else "-",
            "sft_sid_index_path": f"./data/Amazon/index/{category}.index.json",
            "sft_item_meta_path": f"./data/Amazon/index/{category}.item.json",
            "sft_seed": "-",
            "sft_warmup_steps": "-",
            "sft_freeze_llm": "-",
            "sft_group_by_length": "-",
            "sft_load_best_model_at_end": "-",
            "sft_early_stopping_patience": "-",
            "sft_train_from_scratch": "false",
            "sft_launcher": "-",
            "sft_cuda_visible_devices": "-",
            "sft_nproc_per_node": "-",
            "sft_wandb_project": "OneRec",
            "sft_wandb_run_id": "-",
            "sft_report_to": "wandb",
            "eval_runtime_gpus": "-",
            "eval_batch_size": legacy["eval_batch_size"] or "-",
            "eval_num_beams": legacy["eval_num_beams"] or "-",
            "eval_max_new_tokens": legacy["eval_max_new_tokens"] or "-",
            "eval_length_penalty": legacy["eval_length_penalty"] or "-",
            "eval_temperature": legacy["eval_temperature"] or "-",
            "test_example_count": legacy["test_example_count"] or "-",
            "constraint_invalid_total": legacy["constraint_invalid_total"] or "-",
            "ndcg_at_1": legacy["ndcg_at_1"] or "-",
            "ndcg_at_3": legacy["ndcg_at_3"] or "-",
            "ndcg_at_5": legacy["ndcg_at_5"] or "-",
            "ndcg_at_10": legacy["ndcg_at_10"] or "-",
            "ndcg_at_20": legacy["ndcg_at_20"] or "-",
            "ndcg_at_50": legacy["ndcg_at_50"] or "-",
            "hr_at_1": legacy["hr_at_1"] or "-",
            "hr_at_3": legacy["hr_at_3"] or "-",
            "hr_at_5": legacy["hr_at_5"] or "-",
            "hr_at_10": legacy["hr_at_10"] or "-",
            "hr_at_20": legacy["hr_at_20"] or "-",
            "hr_at_50": legacy["hr_at_50"] or "-",
            "notes": legacy.get("evidence") or "-",
        }
    )
    if legacy.get("ambiguity_notes"):
        row["notes"] = (
            legacy["ambiguity_notes"]
            if row["notes"] in {"", "-", None}
            else f"{row['notes']}; {legacy['ambiguity_notes']}"
        )
    return row
def convert_legacy_rl_row(main_fields: list[str], legacy: dict[str, str]) -> dict[str, str]:
    row = default_row(main_fields)
    row.update(
        {
            "record_id": legacy["record_id"],
            "recorded_at": "-",
            "run_finished_at": "-",
            "dataset_key": legacy["dataset_key"],
            "category": legacy["category"],
            "split_stem": "-" if legacy["category"] == "Office_Products" else "Industrial_and_Scientific_5_2016-10-2018-11",
            "stage": legacy["stage"],
            "variant": legacy["variant"],
            "title_history2sid_enabled": legacy["title_history2sid_enabled"] or "-",
            "alignment_enabled": legacy["alignment_enabled"] or "-",
            "description_task_probability": legacy["description_task_probability"] or "-",
            "alignment_mode": legacy["alignment_mode"] or "-",
            "base_model": "-",
            "git_head": "-",
            "git_dirty": "-",
            "sft_output_dir": "-",
            "sft_model_path": "-",
            "result_json_path": legacy["reeval_result_json"] or "-",
            "temp_result_dir": "-",
            "wandb_run_name": legacy["matched_wandb_run"] or "-",
            "sft_runtime_gpus": "-",
            "sft_batch_size": "-",
            "sft_micro_batch_size": "-",
            "sft_world_size": "-",
            "sft_grad_accum_steps": "-",
            "sft_effective_global_batch": "-",
            "sft_num_epochs": "-",
            "sft_learning_rate": "-",
            "sft_cutoff_len": "-",
            "sft_eval_step": "-",
            "sft_final_eval_loss": "-",
            "sft_final_train_loss": "-",
            "sft_stop_epoch": "-",
            "eval_runtime_gpus": "-",
            "eval_batch_size": legacy["eval_batch_size"] or "-",
            "eval_num_beams": legacy["eval_num_beams"] or "-",
            "eval_max_new_tokens": legacy["eval_max_new_tokens"] or "-",
            "eval_length_penalty": legacy["eval_length_penalty"] or "-",
            "eval_temperature": legacy["eval_temperature"] or "-",
            "test_example_count": legacy["test_example_count"] or "-",
            "constraint_invalid_total": legacy["constraint_invalid_total"] or "-",
            "ndcg_at_1": legacy["ndcg_at_1"] or "-",
            "ndcg_at_3": legacy["ndcg_at_3"] or "-",
            "ndcg_at_5": legacy["ndcg_at_5"] or "-",
            "ndcg_at_10": legacy["ndcg_at_10"] or "-",
            "ndcg_at_20": legacy["ndcg_at_20"] or "-",
            "ndcg_at_50": legacy["ndcg_at_50"] or "-",
            "hr_at_1": legacy["hr_at_1"] or "-",
            "hr_at_3": legacy["hr_at_3"] or "-",
            "hr_at_5": legacy["hr_at_5"] or "-",
            "hr_at_10": legacy["hr_at_10"] or "-",
            "hr_at_20": legacy["hr_at_20"] or "-",
            "hr_at_50": legacy["hr_at_50"] or "-",
            "rl_output_dir": legacy["source_output_dir"] or "-",
            "rl_model_path": legacy["source_model_path"] or "-",
            "rl_source_sft_model_path": legacy["source_sft_model_path"] or "-",
            "rl_wandb_run_name": legacy["matched_wandb_run"] or "-",
            "rl_recovery_status": legacy["recovery_status"] or "-",
            "rl_reference_policy": legacy["reference_policy"] or "-",
            "rl_source_code_window": legacy["source_code_window"] or "-",
            "rl_base_model_source": legacy["base_model_source"] or "-",
            "rl_train_batch_size": legacy["per_device_batch_size"] or "-",
            "rl_gradient_accum_steps": legacy["grad_accum_steps"] or "-",
            "rl_world_size": legacy["world_size"] or "-",
            "rl_effective_global_batch": legacy["effective_global_batch"] or "-",
            "rl_num_epochs": legacy["num_epochs"] or "-",
            "rl_learning_rate": legacy["learning_rate"] or "-",
            "rl_eval_steps": legacy["eval_steps"] or "-",
            "rl_save_steps": legacy["save_steps"] or "-",
            "rl_bf16": legacy["bf16"] or "-",
            "rl_fp16": legacy["fp16"] or "-",
            "rl_reward_type": legacy["reward_type"] or "-",
            "rl_beam_search": legacy["beam_search"] or "-",
            "rl_num_generations": legacy["num_generations"] or "-",
            "rl_max_completion_length": legacy["max_completion_length"] or "-",
            "rl_resume_from_checkpoint": legacy["resume_from_checkpoint"] or "-",
            "notes": legacy.get("evidence") or legacy.get("ambiguity_notes") or "-",
        }
    )
    if row["notes"] == "-" and legacy.get("ambiguity_notes"):
        row["notes"] = legacy["ambiguity_notes"]
    elif row["notes"] != "-" and legacy.get("ambiguity_notes"):
        row["notes"] = f"{row['notes']}; {legacy['ambiguity_notes']}"
    return row


def enrich_strongest_industrial_rl(row: dict[str, str]) -> None:
    if row.get("record_id") != "rl_industrial_title_history2sid_off__desc_align_p05_batch256_20260329_152417":
        return
    row.update(
        {
            "rl_output_dir": "./output/rl_Industrial_and_Scientific_title_history2sid_off__desc_align_p05_batch256_20260329_152417",
            "rl_model_path": "./output/rl_Industrial_and_Scientific_title_history2sid_off__desc_align_p05_batch256_20260329_152417/final_checkpoint",
            "rl_source_sft_model_path": "-",
            "rl_wandb_run_name": "-",
            "rl_recovery_status": "partial_from_reports",
            "rl_reference_policy": "report_backfill",
            "rl_source_code_window": "-",
            "rl_base_model_source": "-",
            "rl_train_batch_size": "-",
            "rl_gradient_accum_steps": "-",
            "rl_world_size": "-",
            "rl_effective_global_batch": "256",
            "rl_num_epochs": "-",
            "rl_learning_rate": "-",
            "rl_eval_steps": "-",
            "rl_save_steps": "-",
            "rl_bf16": "-",
            "rl_fp16": "-",
            "rl_reward_type": "-",
            "rl_beam_search": "-",
            "rl_num_generations": "16",
            "rl_max_completion_length": "-",
            "rl_resume_from_checkpoint": "-",
        }
    )
    note = row.get("notes", "-")
    backfill = (
        "RL hyperparameters partially recovered from internal reports: "
        "effective_global_batch=256 and num_generations=16; other RL train hyperparameters remain unknown."
    )
    row["notes"] = backfill if note in {"", "-", None} else f"{note}; {backfill}"


def maybe_fix_legacy_path_alias(row: dict[str, str]) -> None:
    if row.get("record_id") != "legacy_rl_industrial_refactor":
        return
    model_path = row.get("rl_model_path", "-")
    if model_path in {"", "-"}:
        return
    if (ROOT / model_path.lstrip("./")).exists():
        return
    alias_output = "./output/rl_Industrial_and_Scientific_refactor__weights_moved_to_data"
    alias_model = f"{alias_output}/final_checkpoint"
    if (ROOT / alias_model.lstrip("./")).exists():
        row["rl_output_dir"] = alias_output
        row["rl_model_path"] = alias_model
        note = row.get("notes", "-")
        alias_note = "rl_model_path remapped to __weights_moved_to_data alias"
        row["notes"] = alias_note if note in {"", "-", None} else f"{note}; {alias_note}"


def convert_mgr_upstream_sft_row(main_fields: list[str], config_path: str, result_json_path: str, record_id: str) -> dict[str, str]:
    cfg = yaml.safe_load((ROOT / config_path.lstrip("./")).read_text())
    metrics = metrics_from_result_json(result_json_path)
    category = cfg["data"]["category"]
    output_dir = cfg["output"]["output_dir"]
    row = default_row(main_fields)
    row.update(
        {
            "record_id": record_id,
            "recorded_at": "-",
            "run_finished_at": "-",
            "dataset_key": "industrial",
            "category": category,
            "split_stem": split_stem_for_category(category),
            "stage": "sft_eval",
            "variant": record_id.replace("sft_industrial_", "").replace("_20260410", ""),
            "title_history2sid_enabled": str(cfg["training"]["enable_title_history2sid_dataset"]).lower(),
            "alignment_enabled": str(cfg["training"]["enable_title_description_alignment"]).lower(),
            "description_task_probability": str(cfg["training"]["description_task_probability"]),
            "alignment_mode": "title_only",
            "base_model": cfg["model"]["base_model"],
            "git_head": "-",
            "git_dirty": "-",
            "sft_output_dir": output_dir,
            "sft_model_path": f"{output_dir}/final_checkpoint",
            "result_json_path": result_json_path,
            "temp_result_dir": "-",
            "wandb_run_name": cfg["logging"]["wandb_run_name"],
            "sft_runtime_gpus": str(cfg["runtime"]["nproc_per_node"]),
            "sft_batch_size": str(cfg["training"]["batch_size"]),
            "sft_micro_batch_size": str(cfg["training"]["micro_batch_size"]),
            "sft_world_size": str(cfg["runtime"]["nproc_per_node"]),
            "sft_grad_accum_steps": str(
                cfg["training"]["batch_size"]
                // (cfg["training"]["micro_batch_size"] * cfg["runtime"]["nproc_per_node"])
            ),
            "sft_effective_global_batch": str(cfg["training"]["batch_size"]),
            "sft_num_epochs": str(cfg["training"]["num_epochs"]),
            "sft_learning_rate": str(cfg["training"]["learning_rate"]),
            "sft_cutoff_len": str(cfg["training"]["cutoff_len"]),
            "sft_eval_step": str(cfg["training"]["eval_step"]),
            "sft_final_eval_loss": "-",
            "sft_final_train_loss": "-",
            "sft_stop_epoch": "-",
            "sft_config_path": config_path,
            "sft_train_file": cfg["data"]["train_file"],
            "sft_eval_file": cfg["data"]["eval_file"],
            "sft_sid_index_path": cfg["data"]["sid_index_path"],
            "sft_item_meta_path": cfg["data"]["item_meta_path"],
            "sft_seed": str(cfg["training"]["seed"]),
            "sft_warmup_steps": str(cfg["training"]["warmup_steps"]),
            "sft_freeze_llm": str(cfg["training"]["freeze_llm"]).lower(),
            "sft_group_by_length": str(cfg["training"]["group_by_length"]).lower(),
            "sft_load_best_model_at_end": str(cfg["training"]["load_best_model_at_end"]).lower(),
            "sft_early_stopping_patience": str(cfg["training"]["early_stopping_patience"]),
            "sft_train_from_scratch": str(cfg["model"]["train_from_scratch"]).lower(),
            "sft_launcher": cfg["runtime"]["launcher"],
            "sft_cuda_visible_devices": str(cfg["runtime"]["cuda_visible_devices"]),
            "sft_nproc_per_node": str(cfg["runtime"]["nproc_per_node"]),
            "sft_wandb_project": cfg["logging"]["wandb_project"],
            "sft_wandb_run_id": "s2hzz5ds" if "baseline" in record_id else "giwhyo3h",
            "sft_report_to": cfg["logging"]["report_to"],
            "eval_runtime_gpus": str(cfg["runtime"]["nproc_per_node"]),
            "eval_batch_size": "8",
            "eval_num_beams": "50",
            "eval_max_new_tokens": "256",
            "eval_length_penalty": "0.0",
            "eval_temperature": "1.0",
            "notes": "backfilled from config + evaluate result + training log",
        }
    )
    row.update(metrics)
    return row


def convert_partial_rl_row(main_fields: list[str], spec: dict[str, str]) -> dict[str, str]:
    row = default_row(main_fields)
    row.update(
        {
            "record_id": spec["record_id"],
            "recorded_at": "-",
            "run_finished_at": "-",
            "dataset_key": "industrial",
            "category": "Industrial_and_Scientific",
            "split_stem": split_stem_for_category("Industrial_and_Scientific"),
            "stage": "rl_eval",
            "variant": spec["variant"],
            "title_history2sid_enabled": spec["title_history2sid_enabled"],
            "alignment_enabled": spec["alignment_enabled"],
            "description_task_probability": spec["description_task_probability"],
            "alignment_mode": spec["alignment_mode"],
            "base_model": "-",
            "git_head": "-",
            "git_dirty": "-",
            "sft_output_dir": "-",
            "sft_model_path": "-",
            "result_json_path": spec["result_json_path"],
            "temp_result_dir": "-",
            "wandb_run_name": spec["wandb_run_name"],
            "sft_runtime_gpus": "-",
            "sft_batch_size": "-",
            "sft_micro_batch_size": "-",
            "sft_world_size": "-",
            "sft_grad_accum_steps": "-",
            "sft_effective_global_batch": "-",
            "sft_num_epochs": "-",
            "sft_learning_rate": "-",
            "sft_cutoff_len": "-",
            "sft_eval_step": "-",
            "sft_final_eval_loss": "-",
            "sft_final_train_loss": "-",
            "sft_stop_epoch": "-",
            "eval_runtime_gpus": "4",
            "eval_batch_size": "8",
            "eval_num_beams": "50",
            "eval_max_new_tokens": "256",
            "eval_length_penalty": "0.0",
            "eval_temperature": "1.0",
            "rl_output_dir": spec["rl_output_dir"],
            "rl_model_path": spec["rl_model_path"],
            "rl_source_sft_model_path": spec["rl_source_sft_model_path"],
            "rl_wandb_run_name": spec["wandb_run_name"],
            "rl_recovery_status": spec["rl_recovery_status"],
            "rl_reference_policy": spec["rl_reference_policy"],
            "rl_source_code_window": "-",
            "rl_base_model_source": spec["rl_source_sft_model_path"],
            "rl_train_batch_size": spec["rl_train_batch_size"],
            "rl_gradient_accum_steps": spec["rl_gradient_accum_steps"],
            "rl_world_size": "4",
            "rl_effective_global_batch": spec["rl_effective_global_batch"],
            "rl_num_epochs": "2",
            "rl_learning_rate": "1e-05",
            "rl_eval_steps": "0.05",
            "rl_save_steps": "-",
            "rl_bf16": "true",
            "rl_fp16": "false",
            "rl_reward_type": "ranking",
            "rl_beam_search": "true",
            "rl_num_generations": "16",
            "rl_max_completion_length": "128",
            "rl_resume_from_checkpoint": "-",
            "notes": spec["notes"],
        }
    )
    row.update(metrics_from_result_json(spec["result_json_path"]))
    return row


def main() -> None:
    main_fields, main_rows = read_csv(MAIN_CSV)
    legacy_fields, legacy_rows = read_csv(LEGACY_CSV)
    new_fields = ensure_registry_columns(main_fields)

    normalized_rows: list[dict[str, str]] = []
    for row in main_rows:
        normalized = default_row(new_fields)
        normalized.update({k: v if v != "" else "-" for k, v in row.items()})
        for col in SFT_COLUMNS:
            if normalized[col] in {"", None}:
                normalized[col] = "-"
        for col in RL_COLUMNS:
            if normalized[col] in {"", None}:
                normalized[col] = "-"
        enrich_sft_row(normalized)
        enrich_strongest_industrial_rl(normalized)
        maybe_fix_legacy_path_alias(normalized)
        normalized_rows.append(normalized)

    existing_ids = {row["record_id"] for row in normalized_rows}
    for legacy in legacy_rows:
        if legacy["record_id"] in existing_ids:
            continue
        if legacy.get("stage") == "rl_eval":
            normalized_rows.append(convert_legacy_rl_row(new_fields, legacy))
        elif legacy.get("stage") == "sft_eval":
            normalized_rows.append(convert_legacy_sft_row(new_fields, legacy))
        else:
            continue
        existing_ids.add(legacy["record_id"])

    for record_id, config_path, result_json_path in [
        (
            "sft_industrial_mgr_upstream_baseline_20260410",
            "./config/experiments/sft_industrial_mgr_upstream_baseline.yaml",
            "./results/experiments/mgr_sid_sft_eval_industrial_20260410/final_result_sft_mgr_upstream_baseline_Industrial_and_Scientific.json",
        ),
        (
            "sft_industrial_mgr_upstream_hierarchy_20260410",
            "./config/experiments/sft_industrial_mgr_upstream_hierarchy.yaml",
            "./results/experiments/mgr_sid_sft_eval_industrial_20260410/final_result_sft_mgr_upstream_hierarchy_Industrial_and_Scientific.json",
        ),
    ]:
        if record_id not in existing_ids:
            normalized_rows.append(
                convert_mgr_upstream_sft_row(new_fields, config_path, result_json_path, record_id)
            )
            existing_ids.add(record_id)

    partial_rl_specs = [
        {
            "record_id": "rl_industrial_title_history2sid_off__desc_align_off_mb4_20260326_130556",
            "variant": "title_history2sid_off__desc_align_off_mb4",
            "title_history2sid_enabled": "false",
            "alignment_enabled": "false",
            "description_task_probability": "-",
            "alignment_mode": "title_only",
            "wandb_run_name": "rl_industrial_title_history2sid_off__desc_align_off_mb4_20260326_130556",
            "rl_output_dir": "./output/rl_Industrial_and_Scientific_title_history2sid_off__desc_align_off_mb4_20260326_130556__weights_moved_to_data",
            "rl_model_path": "./output/rl_Industrial_and_Scientific_title_history2sid_off__desc_align_off_mb4_20260326_130556__weights_moved_to_data/final_checkpoint",
            "rl_source_sft_model_path": "./output/sft_Industrial_and_Scientific_title_history2sid_off__desc_align_off_mb4_20260326_003015/final_checkpoint",
            "rl_recovery_status": "partial_from_logs",
            "rl_reference_policy": "single_log_plus_result_json",
            "rl_train_batch_size": "4",
            "rl_gradient_accum_steps": "8",
            "rl_effective_global_batch": "128",
            "result_json_path": "./results/final_result_rl_Industrial_and_Scientific_title_history2sid_off__desc_align_off_mb4_20260326_130556.json",
            "notes": "RL hyperparameters backfilled from launch log; train_batch_size/effective_global_batch inferred from variant suffix mb4 plus logged DeepSpeed grad_accum=8.",
        },
        {
            "record_id": "rl_industrial_title_history2sid_off__desc_align_p05_20260327_125414",
            "variant": "title_history2sid_off__desc_align_p05",
            "title_history2sid_enabled": "false",
            "alignment_enabled": "true",
            "description_task_probability": "0.5",
            "alignment_mode": "description_p05",
            "wandb_run_name": "rl_industrial_title_history2sid_off__desc_align_p05_20260327_125414",
            "rl_output_dir": "./output/rl_Industrial_and_Scientific_title_history2sid_off__desc_align_p05_20260327_125414__weights_moved_to_data",
            "rl_model_path": "./output/rl_Industrial_and_Scientific_title_history2sid_off__desc_align_p05_20260327_125414__weights_moved_to_data/final_checkpoint",
            "rl_source_sft_model_path": "./output/sft_Industrial_and_Scientific_title_history2sid_off__desc_align_p05_20260325_192249/final_checkpoint",
            "rl_recovery_status": "partial_from_logs",
            "rl_reference_policy": "single_log_plus_result_json",
            "rl_train_batch_size": "-",
            "rl_gradient_accum_steps": "8",
            "rl_effective_global_batch": "-",
            "result_json_path": "./results/final_result_rl_Industrial_and_Scientific_title_history2sid_off__desc_align_p05_20260327_125414.json",
            "notes": "RL hyperparameters backfilled from launch log; grad_accum=8, world_size=4, bf16, num_generations=16 are explicit. Per-device RL batch and effective_global_batch remain unknown.",
        },
    ]
    for spec in partial_rl_specs:
        if spec["record_id"] not in existing_ids:
            normalized_rows.append(convert_partial_rl_row(new_fields, spec))
            existing_ids.add(spec["record_id"])

    write_csv(MAIN_CSV, new_fields, normalized_rows)


if __name__ == "__main__":
    main()
