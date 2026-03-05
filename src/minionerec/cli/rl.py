from minionerec.cli._shared import build_config
from minionerec.config.schema import RLConfig
from minionerec.flows.rl.pipeline import run_rl


def run_rl_cli(config_path: str | None, overrides: list[str] | None):
    config = build_config(RLConfig, config_path, overrides, stage="rl")
    return run_rl(config)
