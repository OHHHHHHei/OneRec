from minionerec.cli._shared import build_config
from minionerec.config.schema import SFTConfig
from minionerec.flows.sft.pipeline import run_sft


def run_sft_cli(config_path: str | None, overrides: list[str] | None):
    config = build_config(SFTConfig, config_path, overrides, stage="sft")
    return run_sft(config)
