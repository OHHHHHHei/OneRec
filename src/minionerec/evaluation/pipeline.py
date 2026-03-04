import json
import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, LogitsProcessorList

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


def run_evaluate(config) -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    model_dtype = torch.bfloat16 if use_bf16 else torch.float16 if torch.cuda.is_available() else torch.float32
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

    val_dataset = EvalSidDataset(config.data.test_file, tokenizer=tokenizer, max_len=2560, category=config.data.category, test=True, seed=config.training.seed)
    encodings = [val_dataset[i] for i in range(len(val_dataset))]
    test_data = val_dataset.get_all()

    max_len = max(len(item["input_ids"]) for item in encodings)
    input_ids = []
    attention_mask = []
    for item in encodings:
        item_len = len(item["input_ids"])
        input_ids.append([tokenizer.pad_token_id] * (max_len - item_len) + item["input_ids"])
        attention_mask.append([0] * (max_len - item_len) + [1] * item_len)

    clp = ConstrainedLogitsProcessor(prefix_allowed_tokens_fn=prefix_allowed_tokens_fn, num_beams=config.extras.get("num_beams", 20), base_model=config.model.base_model, eos_token_id=tokenizer.eos_token_id)
    generation_output = model.generate(
        torch.tensor(input_ids).to(device),
        attention_mask=torch.tensor(attention_mask).to(device),
        generation_config=GenerationConfig(
            num_beams=config.extras.get("num_beams", 20),
            num_return_sequences=config.extras.get("num_beams", 20),
            max_new_tokens=config.extras.get("max_new_tokens", 256),
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            length_penalty=0.0,
        ),
        logits_processor=LogitsProcessorList([clp]),
        return_dict_in_generate=True,
    )
    completions = generation_output.sequences[:, max_len:]
    decoded = tokenizer.batch_decode(completions, skip_special_tokens=True)
    grouped = [decoded[i : i + config.extras.get("num_beams", 20)] for i in range(0, len(decoded), config.extras.get("num_beams", 20))]
    for idx, row in enumerate(test_data):
        row["predict"] = [entry.split("Response:\n")[-1].strip() for entry in grouped[idx]]
        row.pop("dedup", None)
    with open(config.output.output_dir, "w", encoding="utf-8") as handle:
        json.dump(test_data, handle, indent=2)
    return config.output.output_dir
