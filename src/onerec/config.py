from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, get_type_hints

from onerec.utils.io import read_yaml

T = TypeVar("T")


@dataclass
class StageConfig:
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfig:
    base_model: str = ""
    train_from_scratch: bool = False


@dataclass
class DataConfig:
    train_file: str = ""
    eval_file: str = ""
    test_file: str = ""
    item_meta_path: str = ""
    sid_index_path: str = ""
    info_file: str = ""
    data_dir: str = ""
    dataset_name: str = ""
    output_dir: str = ""
    category: str = ""


@dataclass
class CommonTrainConfig:
    seed: int = 42


@dataclass
class SFTTrainConfig(CommonTrainConfig):
    batch_size: int = 32
    micro_batch_size: int = 4
    cutoff_len: int = 512
    enable_title_description_alignment: bool = True
    description_task_probability: float = 0.5
    group_by_length: bool = False
    warmup_steps: int = 20
    load_best_model_at_end: bool = True
    early_stopping_patience: int = 3
    num_epochs: int = 1
    learning_rate: float = 1e-4
    freeze_llm: bool = False
    eval_step: float = 0.1


@dataclass
class RLTrainConfig(CommonTrainConfig):
    gradient_accumulation_steps: int = 1
    eval_step: float = 0.1
    train_batch_size: int = 32
    eval_batch_size: int = 32
    num_generations: int = 8
    temperature: float = 1.0
    beta: float = 1e-3
    num_epochs: int = 1
    learning_rate: float = 1e-4
    reward_type: str = "rule"
    add_gt: bool = False
    beam_search: bool = False
    test_during_training: bool = True
    dynamic_sampling: bool = False
    sync_ref_model: bool = False
    test_beam: int = 20
    sample_train: bool = False
    dapo: bool = False
    gspo: bool = False
    mask_all_zero: bool = False
    ada_path: str = ""
    cf_path: str = ""


@dataclass
class EvaluateTrainConfig(CommonTrainConfig):
    pass


@dataclass
class LoggingConfig:
    wandb_project: str = ""
    wandb_run_name: str = ""
    report_to: str = "wandb"


@dataclass
class OutputConfig:
    output_dir: str = ""
    save_total_limit: int = 2
    resume_from_checkpoint: str | None = None


@dataclass
class PreprocessConfig(StageConfig):
    data: DataConfig = field(default_factory=DataConfig)
    training: CommonTrainConfig = field(default_factory=CommonTrainConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


@dataclass
class EmbedConfig(StageConfig):
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: CommonTrainConfig = field(default_factory=CommonTrainConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


@dataclass
class SidTrainConfig(StageConfig):
    data: DataConfig = field(default_factory=DataConfig)
    training: CommonTrainConfig = field(default_factory=CommonTrainConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


@dataclass
class SidGenerateConfig(StageConfig):
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


@dataclass
class ConvertConfig(StageConfig):
    data: DataConfig = field(default_factory=DataConfig)
    training: CommonTrainConfig = field(default_factory=CommonTrainConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


@dataclass
class SFTConfig(StageConfig):
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: SFTTrainConfig = field(default_factory=SFTTrainConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


@dataclass
class RLConfig(StageConfig):
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: RLTrainConfig = field(default_factory=RLTrainConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


@dataclass
class EvaluateConfig(StageConfig):
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: EvaluateTrainConfig = field(default_factory=EvaluateTrainConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    batch_size: int = 4
    K: int = 0
    length_penalty: float = 0.0
    max_new_tokens: int = 256
    num_beams: int = 50
    temperature: float = 1.0
    guidance_scale: float | None = 1.0


DEFAULT_CONFIGS = {
    "preprocess": "config/preprocess_amazon18.yaml",
    "embed": "config/embed.yaml",
    "sid-train": "config/sid_train.yaml",
    "sid-generate": "config/sid_generate.yaml",
    "convert": "config/convert.yaml",
    "sft": "config/sft.yaml",
    "rl": "config/rl.yaml",
    "evaluate": "config/evaluate.yaml",
}


def _deep_set(target: dict[str, Any], dotted_key: str, value: Any) -> None:
    cursor = target
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = value


def _coerce_value(raw: str) -> Any:
    lowered = raw.lower()
    if lowered in {"none", "null"}:
        return None
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def apply_overrides(payload: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override: {override}")
        key, raw_value = override.split("=", 1)
        _deep_set(payload, key, _coerce_value(raw_value))
    return payload


def _construct(cls: type[T], payload: Any) -> T:
    if not is_dataclass(cls):
        return payload
    type_hints = get_type_hints(cls)
    kwargs = {}
    consumed = set()
    for field_def in fields(cls):
        if not isinstance(payload, dict) or field_def.name not in payload:
            continue
        value = payload[field_def.name]
        field_type = type_hints.get(field_def.name, field_def.type)
        if value is None:
            kwargs[field_def.name] = None
            consumed.add(field_def.name)
            continue
        try:
            kwargs[field_def.name] = _construct(field_type, value)
        except TypeError:
            kwargs[field_def.name] = value
        consumed.add(field_def.name)
    if "extras" in {field_def.name for field_def in fields(cls)} and isinstance(payload, dict):
        kwargs["extras"] = {key: value for key, value in payload.items() if key not in consumed}
    return cls(**kwargs)


def resolve_config_path(stage: str, config_path: str | None) -> str:
    if config_path:
        return config_path
    resolved = DEFAULT_CONFIGS.get(stage)
    if not resolved:
        raise KeyError(f"No default config path registered for stage: {stage}")
    if not Path(resolved).exists():
        raise FileNotFoundError(f"Default config not found for stage `{stage}`: {resolved}")
    return resolved


def load_config(config_cls: type[T], config_path: str, overrides: list[str] | None = None) -> T:
    payload = read_yaml(config_path)
    payload = apply_overrides(payload, overrides or [])
    return _construct(config_cls, payload)
