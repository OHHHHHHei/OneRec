from __future__ import annotations

import importlib
import warnings


def _legacy_to_overrides(kwargs: dict[str, object], mapping_map: dict[str, str]) -> list[str]:
    overrides = []
    for key, value in kwargs.items():
        if value is None:
            continue
        target = mapping_map.get(key)
        if target is None:
            continue
        overrides.append(f"{target}={value}")
    return overrides


def run_legacy_sft(**kwargs):
    warnings.warn("Legacy sft.py entrypoint is deprecated. Use `python -m minionerec.cli.main sft --config ...`.", DeprecationWarning, stacklevel=2)
    mapping_map = {
        "base_model": "model.base_model",
        "train_file": "data.train_file",
        "eval_file": "data.eval_file",
        "output_dir": "output.output_dir",
        "seed": "training.seed",
        "batch_size": "training.batch_size",
        "micro_batch_size": "training.micro_batch_size",
        "num_epochs": "training.num_epochs",
        "learning_rate": "training.learning_rate",
        "freeze_LLM": "training.freeze_llm",
        "wandb_project": "logging.wandb_project",
        "wandb_run_name": "logging.wandb_run_name",
        "resume_from_checkpoint": "output.resume_from_checkpoint",
        "category": "data.category",
        "train_from_scratch": "model.train_from_scratch",
        "sid_index_path": "data.sid_index_path",
        "item_meta_path": "data.item_meta_path",
        "save_total_limit": "output.save_total_limit",
        "report_to": "logging.report_to",
    }
    return importlib.import_module("minionerec.cli.sft").run_sft_cli(config_path=None, overrides=_legacy_to_overrides(kwargs, mapping_map))


def run_legacy_rl(**kwargs):
    warnings.warn("Legacy rl.py entrypoint is deprecated. Use `python -m minionerec.cli.main rl --config ...`.", DeprecationWarning, stacklevel=2)
    mapping_map = {
        "model_path": "model.base_model",
        "train_file": "data.train_file",
        "eval_file": "data.eval_file",
        "info_file": "data.info_file",
        "category": "data.category",
        "wandb_project": "logging.wandb_project",
        "wandb_run_name": "logging.wandb_run_name",
        "output_dir": "output.output_dir",
        "train_batch_size": "training.train_batch_size",
        "eval_batch_size": "training.eval_batch_size",
        "gradient_accumulation_steps": "training.gradient_accumulation_steps",
        "temperature": "training.temperature",
        "num_generations": "training.num_generations",
        "num_train_epochs": "training.num_epochs",
        "learning_rate": "training.learning_rate",
        "beta": "training.beta",
        "reward_type": "training.reward_type",
        "sid_index_path": "data.sid_index_path",
        "item_meta_path": "data.item_meta_path",
        "resume_from_checkpoint": "output.resume_from_checkpoint",
        "save_total_limit": "output.save_total_limit",
        "seed": "training.seed",
        "ada_path": "ada_path",
    }
    return importlib.import_module("minionerec.cli.rl").run_rl_cli(config_path=None, overrides=_legacy_to_overrides(kwargs, mapping_map))


def run_legacy_evaluate(**kwargs):
    warnings.warn("Legacy evaluate.py entrypoint is deprecated. Use `python -m minionerec.cli.main evaluate --config ...`.", DeprecationWarning, stacklevel=2)
    mapping_map = {
        "base_model": "model.base_model",
        "info_file": "data.info_file",
        "category": "data.category",
        "test_data_path": "data.test_file",
        "result_json_data": "output.output_dir",
        "seed": "training.seed",
        "num_beams": "num_beams",
        "max_new_tokens": "max_new_tokens",
    }
    return importlib.import_module("minionerec.cli.evaluate").run_evaluate_cli(config_path=None, overrides=_legacy_to_overrides(kwargs, mapping_map))


def run_legacy_convert(**kwargs):
    warnings.warn("Legacy convert_dataset.py entrypoint is deprecated. Use `python -m minionerec.cli.main convert --config ...`.", DeprecationWarning, stacklevel=2)
    mapping_map = {
        "data_dir": "data.data_dir",
        "dataset_name": "data.dataset_name",
        "output_dir": "data.output_dir",
        "category": "data.category",
        "seed": "training.seed",
        "max_valid_samples": "max_valid_samples",
        "max_test_samples": "max_test_samples",
        "keep_longest_only": "keep_longest_only",
        "info_path": "info_path",
    }
    return importlib.import_module("minionerec.cli.convert").run_convert_cli(config_path=None, overrides=_legacy_to_overrides(kwargs, mapping_map))
