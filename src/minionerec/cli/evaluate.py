from minionerec.cli._shared import build_config
from minionerec.config.schema import EvaluateConfig
from minionerec.flows.evaluate.pipeline import run_evaluate


def run_evaluate_cli(config_path: str | None, overrides: list[str] | None):
    config = build_config(EvaluateConfig, config_path, overrides, stage="evaluate")
    return run_evaluate(config)
