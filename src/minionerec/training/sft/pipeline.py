import logging
import math
import os
from pathlib import Path

import torch
import transformers
from torch.utils.data import ConcatDataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from minionerec.common.seed import set_global_seed
from minionerec.data.datasets.sft import FusionSeqRecDataset, SidItemFeatDataset, SidSFTDataset
from minionerec.training.sft.token_extension import TokenExtender
from minionerec.training.sft.trainer import concat_dataset_to_hf


logger = logging.getLogger(__name__)


def _resolve_precision() -> tuple[torch.dtype, bool, bool]:
    has_cuda = torch.cuda.is_available()
    use_bf16 = bool(has_cuda and torch.cuda.is_bf16_supported())
    use_fp16 = bool(has_cuda and not use_bf16)
    model_dtype = torch.bfloat16 if use_bf16 else torch.float16 if use_fp16 else torch.float32
    return model_dtype, use_bf16, use_fp16


def _resolve_grad_accum_steps(config) -> tuple[int, int, int]:
    world_size = max(1, int(os.environ.get("WORLD_SIZE", "1")))
    per_device_batch = max(1, int(config.training.micro_batch_size))
    target_global_batch = max(1, int(config.training.batch_size))
    denominator = per_device_batch * world_size
    gradient_accumulation_steps = max(1, math.ceil(target_global_batch / denominator))
    effective_global_batch = denominator * gradient_accumulation_steps
    if effective_global_batch != target_global_batch:
        logger.warning(
            "SFT global batch mismatch: requested=%s, effective=%s (world_size=%s, per_device=%s, grad_accum=%s).",
            target_global_batch,
            effective_global_batch,
            world_size,
            per_device_batch,
            gradient_accumulation_steps,
        )
    logger.info(
        "SFT launch config: world_size=%s, per_device_batch=%s, grad_accum=%s, effective_global_batch=%s",
        world_size,
        per_device_batch,
        gradient_accumulation_steps,
        effective_global_batch,
    )
    return gradient_accumulation_steps, world_size, effective_global_batch


def run_sft(config) -> str:
    set_global_seed(config.training.seed)
    model_dtype, use_bf16, use_fp16 = _resolve_precision()
    if config.model.train_from_scratch:
        model_config = AutoConfig.from_pretrained(config.model.base_model)
        model = AutoModelForCausalLM.from_config(model_config)
    else:
        model = AutoModelForCausalLM.from_pretrained(config.model.base_model, torch_dtype=model_dtype)
    tokenizer = AutoTokenizer.from_pretrained(config.model.base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    original_vocab_size = len(tokenizer)
    if config.data.sid_index_path and Path(config.data.sid_index_path).exists():
        extender = TokenExtender(config.data.sid_index_path)
        new_tokens = extender.get_new_tokens()
        if new_tokens:
            tokenizer.add_tokens(new_tokens)
            model.resize_token_embeddings(len(tokenizer))
    else:
        new_tokens = []

    if config.training.freeze_llm:
        for param in model.parameters():
            param.requires_grad = False
        embedding_layer = model.get_input_embeddings()
        embedding_layer.weight.requires_grad = True

        def mask_grad(grad):
            grad[:original_vocab_size].zero_()
            return grad

        embedding_layer.weight.register_hook(mask_grad)

    train_datasets = [
        SidSFTDataset(config.data.train_file, tokenizer=tokenizer, max_len=512, category=config.data.category, seed=config.training.seed),
        SidItemFeatDataset(config.data.item_meta_path, config.data.sid_index_path, tokenizer=tokenizer, max_len=512, category=config.data.category, seed=config.training.seed),
        FusionSeqRecDataset(config.data.train_file, config.data.item_meta_path, config.data.sid_index_path, tokenizer=tokenizer, max_len=512, category=config.data.category, seed=config.training.seed),
    ]
    train_dataset = ConcatDataset(train_datasets)
    eval_dataset = SidSFTDataset(config.data.eval_file, tokenizer=tokenizer, max_len=512, category=config.data.category, seed=config.training.seed)
    hf_train = concat_dataset_to_hf(train_dataset)
    hf_eval = concat_dataset_to_hf(eval_dataset)

    gradient_accumulation_steps, _, _ = _resolve_grad_accum_steps(config)
    args = transformers.TrainingArguments(
        output_dir=config.output.output_dir,
        run_name=config.logging.wandb_run_name,
        per_device_train_batch_size=config.training.micro_batch_size,
        per_device_eval_batch_size=config.training.micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=config.training.num_epochs,
        learning_rate=config.training.learning_rate,
        bf16=use_bf16,
        fp16=use_fp16,
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=0.05,
        save_strategy="steps",
        save_steps=0.05,
        save_total_limit=config.output.save_total_limit,
        load_best_model_at_end=False,
        report_to=config.logging.report_to,
    )
    trainer = transformers.Trainer(
        model=model,
        train_dataset=hf_train,
        eval_dataset=hf_eval,
        args=args,
        data_collator=transformers.DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True),
    )
    model.config.use_cache = False
    trainer.train(resume_from_checkpoint=config.output.resume_from_checkpoint)
    trainer.save_model(config.output.output_dir)
    final_checkpoint = os.path.join(config.output.output_dir, "final_checkpoint")
    trainer.model.save_pretrained(final_checkpoint)
    tokenizer.save_pretrained(final_checkpoint)
    return final_checkpoint
