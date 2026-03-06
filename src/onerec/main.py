from __future__ import annotations

import argparse
import runpy
import sys

from onerec.config import (
    ConvertConfig,
    EmbedConfig,
    EvaluateConfig,
    PreprocessConfig,
    RLConfig,
    SFTConfig,
    SidGenerateConfig,
    SidTrainConfig,
    load_config,
    resolve_config_path,
)
from onerec.utils.logging import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OneRec stage runner")
    parser.add_argument(
        "stage",
        choices=[
            "preprocess",
            "embed",
            "sid-train",
            "sid-generate",
            "convert",
            "sft",
            "rl",
            "evaluate",
            "split",
            "merge",
            "metrics",
        ],
    )
    parser.add_argument("--config", dest="config_path", default=None)
    return parser


def _run_module(module_name: str, argv: list[str]) -> None:
    old_argv = sys.argv
    try:
        sys.argv = [module_name, *argv]
        runpy.run_module(module_name, run_name="__main__")
    finally:
        sys.argv = old_argv


def _run_preprocess(config_path: str | None, overrides: list[str]) -> None:
    config = load_config(PreprocessConfig, resolve_config_path("preprocess", config_path), overrides)
    target = "onerec.preprocess.amazon18" if config.extras.get("source", "amazon18") == "amazon18" else "onerec.preprocess.amazon23"
    argv = []
    for key, value in config.extras.items():
        if key == "source":
            continue
        argv.extend([f"--{key}", str(value)])
    _run_module(target, argv)


def _run_embed(config_path: str | None, overrides: list[str]) -> None:
    config = load_config(EmbedConfig, resolve_config_path("embed", config_path), overrides)
    _run_module(
        "onerec.sid.embed",
        [
            "--dataset",
            config.data.dataset_name,
            "--root",
            config.data.data_dir,
            "--plm_name",
            config.extras.get("plm_name", "qwen"),
            "--plm_checkpoint",
            config.model.base_model,
        ],
    )


def _run_sid_train(config_path: str | None, overrides: list[str]) -> None:
    config = load_config(SidTrainConfig, resolve_config_path("sid-train", config_path), overrides)
    kind = config.extras.get("kind", "rqvae")
    module_name = {
        "rqvae": "onerec.sid.quantizers.rqvae",
        "rqkmeans_faiss": "onerec.sid.quantizers.rqkmeans_faiss",
        "rqkmeans_constrained": "onerec.sid.quantizers.rqkmeans_constrained",
        "rqkmeans_plus": "onerec.sid.quantizers.rqkmeans_plus",
    }[kind]
    argv = []
    for key, value in config.extras.items():
        if key == "kind":
            continue
        argv.extend([f"--{key}", str(value)])
    _run_module(module_name, argv)


def _run_sid_generate(config_path: str | None, overrides: list[str]) -> None:
    config = load_config(SidGenerateConfig, resolve_config_path("sid-generate", config_path), overrides)
    kind = config.extras.get("kind", "rqvae")
    module_name = "onerec.sid.generate.rqvae_indices" if kind == "rqvae" else "onerec.sid.generate.rqkmeans_plus_indices"
    argv = []
    for key, value in config.extras.items():
        if key == "kind":
            continue
        argv.extend([f"--{key}", str(value)])
    _run_module(module_name, argv)


def _run_convert(config_path: str | None, overrides: list[str]) -> str:
    from onerec.convert.pipeline import run_convert

    config = load_config(ConvertConfig, resolve_config_path("convert", config_path), overrides)
    return run_convert(config)


def _run_sft(config_path: str | None, overrides: list[str]) -> str:
    from onerec.sft.pipeline import run_sft

    config = load_config(SFTConfig, resolve_config_path("sft", config_path), overrides)
    return run_sft(config)


def _run_rl(config_path: str | None, overrides: list[str]) -> str:
    from onerec.rl.pipeline import run_rl

    config = load_config(RLConfig, resolve_config_path("rl", config_path), overrides)
    return run_rl(config)


def _run_evaluate(config_path: str | None, overrides: list[str]) -> str:
    from onerec.evaluate.pipeline import run_evaluate

    config = load_config(EvaluateConfig, resolve_config_path("evaluate", config_path), overrides)
    return run_evaluate(config)


def _build_internal_parser(name: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=f"python -m onerec.main {name}")
    return parser


def _run_split(argv: list[str]) -> str:
    from onerec.evaluate.split_merge import split

    parser = _build_internal_parser("split")
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--cuda_list", required=True)
    args = parser.parse_args(argv)
    return split(args.input_path, args.output_path, args.cuda_list)


def _run_merge(argv: list[str]) -> None:
    from onerec.evaluate.merge import merge

    parser = _build_internal_parser("merge")
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--cuda_list", required=True)
    args = parser.parse_args(argv)
    return merge(args.input_path, args.output_path, args.cuda_list)


def _run_metrics(argv: list[str]) -> None:
    from onerec.evaluate.metrics import gao

    parser = _build_internal_parser("metrics")
    parser.add_argument("--path", required=True)
    parser.add_argument("--item_path", required=True)
    args = parser.parse_args(argv)
    return gao(args.path, args.item_path)


def main() -> None:
    configure_logging()
    parser = build_parser()
    args, rest = parser.parse_known_args()
    dispatch = {
        "preprocess": lambda: _run_preprocess(args.config_path, rest),
        "embed": lambda: _run_embed(args.config_path, rest),
        "sid-train": lambda: _run_sid_train(args.config_path, rest),
        "sid-generate": lambda: _run_sid_generate(args.config_path, rest),
        "convert": lambda: _run_convert(args.config_path, rest),
        "sft": lambda: _run_sft(args.config_path, rest),
        "rl": lambda: _run_rl(args.config_path, rest),
        "evaluate": lambda: _run_evaluate(args.config_path, rest),
        "split": lambda: _run_split(rest),
        "merge": lambda: _run_merge(rest),
        "metrics": lambda: _run_metrics(rest),
    }
    result = dispatch[args.stage]()
    if result is not None:
        print(result)


if __name__ == "__main__":
    main()
