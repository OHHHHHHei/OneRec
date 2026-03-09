from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import Any

from onerec.utils.io import read_yaml, write_yaml


PLACEHOLDER_PATTERN = re.compile(r"%\{([a-zA-Z0-9_]+)\}")
DEFAULT_DATASET_KEY = "industrial"


def load_dataset_mapping(datasets_path: str | Path) -> dict[str, dict[str, Any]]:
    payload = read_yaml(datasets_path)
    return {str(key).strip().lower(): value for key, value in payload.items()}


def build_template_context(
    datasets_path: str | Path,
    dataset_key: str | None = None,
    eval_model_stage: str = "sft",
) -> dict[str, str]:
    mapping = load_dataset_mapping(datasets_path)
    key = (dataset_key or DEFAULT_DATASET_KEY).strip().lower()
    if key not in mapping:
        valid = ", ".join(sorted(mapping))
        raise KeyError(f"Unknown dataset key {key!r}. Available: {valid}")
    if eval_model_stage not in {"sft", "rl"}:
        raise ValueError(f"Unsupported evaluate model stage: {eval_model_stage!r}")

    entry = mapping[key]
    category = str(entry["category"]).strip()
    split_stem = str(entry["split_stem"]).strip()
    artifact_stem = str(entry.get("artifact_stem", category)).strip()
    return {
        "dataset_key": key,
        "category": category,
        "split_stem": split_stem,
        "artifact_stem": artifact_stem,
        "eval_model_stage": eval_model_stage,
        "eval_result_suffix": "" if eval_model_stage == "sft" else f"_{eval_model_stage}",
    }


def render_template_string(value: str, context: dict[str, str]) -> str:
    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in context:
            available = ", ".join(sorted(context))
            raise KeyError(f"Missing template key {key!r}. Available: {available}")
        return str(context[key])

    return PLACEHOLDER_PATTERN.sub(replace, value)


def render_template_payload(payload: Any, context: dict[str, str]) -> Any:
    if isinstance(payload, dict):
        return {key: render_template_payload(value, context) for key, value in payload.items()}
    if isinstance(payload, list):
        return [render_template_payload(item, context) for item in payload]
    if isinstance(payload, str):
        return render_template_string(payload, context)
    return payload


def render_config_payload(
    config_path: str | Path,
    datasets_path: str | Path,
    dataset_key: str | None = None,
    eval_model_stage: str = "sft",
) -> dict[str, Any]:
    payload = read_yaml(config_path)
    context = build_template_context(
        datasets_path=datasets_path,
        dataset_key=dataset_key,
        eval_model_stage=eval_model_stage,
    )
    return render_template_payload(payload, context)


def render_config_file(
    config_path: str | Path,
    datasets_path: str | Path,
    dataset_key: str | None = None,
    eval_model_stage: str = "sft",
    output_path: str | Path | None = None,
) -> Path:
    rendered = render_config_payload(
        config_path=config_path,
        datasets_path=datasets_path,
        dataset_key=dataset_key,
        eval_model_stage=eval_model_stage,
    )
    if output_path is None:
        temp_dir = Path(tempfile.gettempdir()) / "onerec-rendered"
        temp_dir.mkdir(parents=True, exist_ok=True)
        key = (dataset_key or DEFAULT_DATASET_KEY).strip().lower()
        output_path = temp_dir / f"{Path(config_path).stem}-{key}-{eval_model_stage}.yaml"
    target = Path(output_path)
    write_yaml(target, rendered)
    return target
