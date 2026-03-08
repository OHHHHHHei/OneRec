from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent


def _read_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be dict: {path}")
    return data


def _lookup(payload: dict, dotted: str):
    cur = payload
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(dotted)
        cur = cur[part]
    return cur


def _check_defaults(name: str, payload: dict, expected: dict[str, object]) -> list[str]:
    errors: list[str] = []
    for key, value in expected.items():
        try:
            actual = _lookup(payload, key)
        except KeyError:
            errors.append(f"[{name}] missing key: {key}")
            continue
        if actual != value:
            errors.append(f"[{name}] mismatch {key}: expected={value!r} actual={actual!r}")
    return errors


def main() -> int:
    sft_cfg = _read_yaml(ROOT / "config" / "sft.yaml")
    rl_cfg = _read_yaml(ROOT / "config" / "rl.yaml")
    eval_cfg = _read_yaml(ROOT / "config" / "evaluate.yaml")

    errors: list[str] = []
    errors += _check_defaults(
        "sft",
        sft_cfg,
        {
            "training.batch_size": 1024,
            "training.micro_batch_size": 4,
            "training.warmup_steps": 20,
            "training.group_by_length": False,
            "training.load_best_model_at_end": True,
            "training.freeze_llm": False,
            "training.eval_step": 0.05,
            "logging.wandb_project": "OneRec",
        },
    )
    errors += _check_defaults(
        "rl",
        rl_cfg,
        {
            "training.num_generations": 4,
            "training.temperature": 1.0,
            "training.eval_step": 0.0999,
            "training.beam_search": True,
            "training.test_during_training": False,
            "training.reward_type": "ranking",
            "training.sync_ref_model": True,
            "logging.wandb_project": "OneRec",
        },
    )
    errors += _check_defaults(
        "evaluate",
        eval_cfg,
        {
            "batch_size": 8,
            "num_beams": 50,
            "max_new_tokens": 256,
            "length_penalty": 0.0,
            "temperature": 1.0,
            "guidance_scale": None,
        },
    )

    if errors:
        print("Parity check FAILED:")
        for err in errors:
            print(f"- {err}")
        return 1
    print("Parity check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
