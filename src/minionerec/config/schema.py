from dataclasses import dataclass, field

from .base import StageConfig


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

