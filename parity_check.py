from __future__ import annotations

import ast
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


def _extract_mapping_keys(legacy_text: str, fn_name: str) -> set[str]:
    marker = f"def {fn_name}(**kwargs):"
    start = legacy_text.find(marker)
    if start < 0:
        return set()
    tail = legacy_text[start:]
    map_pos = tail.find("mapping_map = {")
    if map_pos < 0:
        return set()
    snippet = tail[map_pos + len("mapping_map = ") :]
    depth = 0
    end = -1
    for idx, ch in enumerate(snippet):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = idx + 1
                break
    if end < 0:
        return set()
    mapping_obj = ast.literal_eval(snippet[:end])
    return set(mapping_obj.keys()) if isinstance(mapping_obj, dict) else set()


def _check_legacy_mapping() -> list[str]:
    errors: list[str] = []
    legacy_path = ROOT / "src" / "minionerec" / "compat" / "legacy_cli.py"
    text = legacy_path.read_text(encoding="utf-8")
    required = {
        "run_legacy_sft": {"freeze_LLM"},
        "run_legacy_rl": {"beam_search", "test_during_training", "eval_step", "sync_ref_model"},
        "run_legacy_evaluate": {"num_beams", "max_new_tokens", "length_penalty", "temperature", "guidance_scale"},
    }
    for fn_name, keys in required.items():
        existing = _extract_mapping_keys(text, fn_name)
        missing = sorted(keys - existing)
        if missing:
            errors.append(f"[legacy_map] {fn_name} missing keys: {missing}")
    return errors


def main() -> int:
    sft_cfg = _read_yaml(ROOT / "flows" / "sft" / "default.yaml")
    rl_cfg = _read_yaml(ROOT / "flows" / "rl" / "default.yaml")
    eval_cfg = _read_yaml(ROOT / "flows" / "evaluate" / "default.yaml")

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
        },
    )
    errors += _check_defaults(
        "rl",
        rl_cfg,
        {
            "training.num_generations": 8,
            "training.temperature": 1.0,
            "training.eval_step": 0.0999,
            "training.beam_search": True,
            "training.test_during_training": False,
            "training.reward_type": "ranking",
            "training.sync_ref_model": True,
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
    errors += _check_legacy_mapping()

    if errors:
        print("Parity check FAILED:")
        for err in errors:
            print(f"- {err}")
        return 1
    print("Parity check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

