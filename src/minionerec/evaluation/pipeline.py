import json
import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, LogitsProcessorList

from minionerec.common.seed import set_global_seed
from minionerec.data.datasets.eval import EvalSidDataset
from minionerec.evaluation.constrained_decoding import ConstrainedLogitsProcessor


logger = logging.getLogger(__name__)


def _get_hash(tokens) -> str:
    return "-".join(str(token) for token in tokens)


def build_prefix_allowed_tokens(tokenizer, info_file: str, base_model: str):
    with open(info_file, "r", encoding="utf-8") as handle:
        semantic_ids = [line.split("\t")[0].strip() + "\n" for line in handle]
    info_semantic = [f"### Response:\n{item}" for item in semantic_ids]
    prefix_ids = [tokenizer(entry).input_ids[1:] if "llama" in base_model.lower() else tokenizer(entry).input_ids for entry in info_semantic]
    prefix_index = 4 if "gpt2" in base_model.lower() else 3
    hash_dict: dict[str, list[int]] = {}
    for input_ids in prefix_ids:
        ids = list(input_ids) + [tokenizer.eos_token_id]
        for i in range(prefix_index, len(ids)):
            hash_key = _get_hash(ids[:i] if i == prefix_index else ids[prefix_index:i])
            hash_dict.setdefault(hash_key, set()).add(ids[i])
    return {key: list(values) for key, values in hash_dict.items()}


def _resolve_eval_param(config, key: str, default):
    if hasattr(config, key):
        value = getattr(config, key)
        if value is not None:
            return value
    return config.extras.get(key, default)


def run_evaluate(config) -> str:
    set_global_seed(config.training.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    model_dtype = torch.bfloat16 if use_bf16 else torch.float16 if torch.cuda.is_available() else torch.float32
    batch_size = int(_resolve_eval_param(config, "batch_size", 4))
    num_beams = int(_resolve_eval_param(config, "num_beams", 50))
    max_new_tokens = int(_resolve_eval_param(config, "max_new_tokens", 256))
    length_penalty = float(_resolve_eval_param(config, "length_penalty", 0.0))
    # Keep this for legacy config compatibility; deterministic beam decoding below does not use temperature.
    _temperature = float(_resolve_eval_param(config, "temperature", 1.0))
    guidance_scale = _resolve_eval_param(config, "guidance_scale", 1.0)
    if isinstance(guidance_scale, str) and guidance_scale.lower() in {"none", "null"}:
        guidance_scale = None
    logger.info("Evaluate precision config: dtype=%s", model_dtype)
    model_kwargs = {"torch_dtype": model_dtype}
    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(config.model.base_model, **model_kwargs)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(config.model.base_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    hash_dict = build_prefix_allowed_tokens(tokenizer, config.data.info_file, config.model.base_model)

    def prefix_allowed_tokens_fn(batch_id, input_ids):
        return hash_dict.get(_get_hash(input_ids), [])

    val_dataset = EvalSidDataset(
        config.data.test_file,
        tokenizer=tokenizer,
        max_len=2560,
        category=config.data.category,
        test=True,
        K=int(_resolve_eval_param(config, "K", 0)),
        seed=config.training.seed,
    )
    encodings = [val_dataset[i] for i in range(len(val_dataset))]
    test_data = val_dataset.get_all()

    model.config.pad_token_id = tokenizer.eos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model = model.to(device)

    outputs = []
    total = len(encodings)
    blocks = (total + batch_size - 1) // batch_size
    for block_idx in range(blocks):
        batch_encodings = encodings[block_idx * batch_size : (block_idx + 1) * batch_size]
        max_len = max(len(item["input_ids"]) for item in batch_encodings)
        input_ids = []
        attention_mask = []
        for item in batch_encodings:
            item_len = len(item["input_ids"])
            input_ids.append([tokenizer.pad_token_id] * (max_len - item_len) + item["input_ids"])
            attention_mask.append([0] * (max_len - item_len) + [1] * item_len)

        clp = ConstrainedLogitsProcessor(
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            num_beams=num_beams,
            base_model=config.model.base_model,
            eos_token_id=tokenizer.eos_token_id,
        )
        generation_kwargs = {}
        if guidance_scale is not None:
            generation_kwargs["guidance_scale"] = guidance_scale
        generation_output = model.generate(
            torch.tensor(input_ids).to(device),
            attention_mask=torch.tensor(attention_mask).to(device),
            generation_config=GenerationConfig(
                num_beams=num_beams,
                num_return_sequences=num_beams,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                top_k=None,
                top_p=None,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                length_penalty=length_penalty,
            ),
            logits_processor=LogitsProcessorList([clp]),
            return_dict_in_generate=True,
            output_scores=True,
            **generation_kwargs,
        )
        completions = generation_output.sequences[:, max_len:]
        if "llama" in config.model.base_model.lower():
            decoded = tokenizer.batch_decode(completions, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        else:
            decoded = tokenizer.batch_decode(completions, skip_special_tokens=True)
        decoded = [entry.split("Response:\n")[-1].strip() for entry in decoded]
        grouped = [decoded[i : i + num_beams] for i in range(0, len(decoded), num_beams)]
        outputs.extend(grouped)

    for idx, row in enumerate(test_data):
        row["predict"] = outputs[idx]
        row.pop("dedup", None)
    with open(config.output.output_dir, "w", encoding="utf-8") as handle:
        json.dump(test_data, handle, indent=4)
    return config.output.output_dir
