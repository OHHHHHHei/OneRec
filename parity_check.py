from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from onerec.utils.config_templates import render_config_payload


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
    datasets_path = ROOT / "config" / "datasets.yaml"
    sft_cfg = render_config_payload(ROOT / "config" / "sft.yaml", datasets_path, dataset_key="industrial", eval_model_stage="sft")
    rl_cfg = render_config_payload(ROOT / "config" / "rl.yaml", datasets_path, dataset_key="industrial", eval_model_stage="sft")
    eval_cfg = render_config_payload(ROOT / "config" / "evaluate.yaml", datasets_path, dataset_key="industrial", eval_model_stage="sft")

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
            "training.eval_step": 0.05,
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
