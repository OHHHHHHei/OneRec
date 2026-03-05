import logging
import os
from pathlib import Path

import torch
import transformers
from torch.utils.data import ConcatDataset
from transformers import EarlyStoppingCallback
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from minionerec.common.seed import set_global_seed
from minionerec.flows.sft.datasets import FusionSeqRecDataset, SidItemFeatDataset, SidSFTDataset, TitleHistory2SidSFTDataset
from minionerec.flows.sft.token_extension import TokenExtender
from minionerec.flows.sft.trainer_runtime import concat_dataset_to_hf


logger = logging.getLogger(__name__)


def _resolve_precision() -> tuple[torch.dtype, bool, bool]:
    has_cuda = torch.cuda.is_available()
    use_bf16 = bool(has_cuda and torch.cuda.is_bf16_supported())
    use_fp16 = bool(has_cuda and not use_bf16)
    model_dtype = torch.bfloat16 if use_bf16 else torch.float16 if use_fp16 else torch.float32
    return model_dtype, use_bf16, use_fp16


def _resolve_grad_accum_steps(config) -> tuple[int, int, bool]:
    micro_batch_size = max(1, int(config.training.micro_batch_size))
    batch_size = max(1, int(config.training.batch_size))
    world_size = max(1, int(os.environ.get("WORLD_SIZE", "1")))
    ddp = world_size != 1
    gradient_accumulation_steps = max(1, batch_size // micro_batch_size)
    if ddp:
        gradient_accumulation_steps = max(1, gradient_accumulation_steps // world_size)
    logger.info(
        "SFT launch config: world_size=%s, batch_size=%s, micro_batch_size=%s, grad_accum=%s",
        world_size,
        batch_size,
        micro_batch_size,
        gradient_accumulation_steps,
    )
    return gradient_accumulation_steps, world_size, ddp


def run_sft(config) -> str:
    set_global_seed(config.training.seed)
    if config.logging.wandb_project:
        os.environ["WANDB_PROJECT"] = config.logging.wandb_project
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

    if int(os.environ.get("WORLD_SIZE", "1")) == 1 and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    cutoff_len = max(1, int(config.training.cutoff_len))
    train_datasets = [
        SidSFTDataset(config.data.train_file, tokenizer=tokenizer, max_len=cutoff_len, category=config.data.category, seed=config.training.seed),
        SidItemFeatDataset(config.data.item_meta_path, config.data.sid_index_path, tokenizer=tokenizer, max_len=cutoff_len, category=config.data.category, seed=config.training.seed),
        FusionSeqRecDataset(config.data.train_file, config.data.item_meta_path, config.data.sid_index_path, tokenizer=tokenizer, max_len=cutoff_len, category=config.data.category, seed=config.training.seed),
        TitleHistory2SidSFTDataset(config.data.train_file, config.data.item_meta_path, config.data.sid_index_path, tokenizer=tokenizer, max_len=cutoff_len, category=config.data.category, seed=config.training.seed),
    ]
    train_dataset = ConcatDataset(train_datasets)
    eval_dataset = SidSFTDataset(config.data.eval_file, tokenizer=tokenizer, max_len=cutoff_len, category=config.data.category, seed=config.training.seed)
    hf_train = concat_dataset_to_hf(train_dataset).shuffle(seed=42)
    hf_eval = concat_dataset_to_hf(eval_dataset).shuffle(seed=config.training.seed).shuffle(seed=42)

    gradient_accumulation_steps, _, ddp = _resolve_grad_accum_steps(config)
    eval_step = float(config.training.eval_step)
    callbacks = []
    if int(config.training.early_stopping_patience) > 0 and bool(config.training.load_best_model_at_end):
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=int(config.training.early_stopping_patience)))
    args = transformers.TrainingArguments(
        output_dir=config.output.output_dir,
        run_name=config.logging.wandb_run_name,
        per_device_train_batch_size=config.training.micro_batch_size,
        per_device_eval_batch_size=config.training.micro_batch_size,
        ddp_find_unused_parameters=False if ddp else None,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=int(config.training.warmup_steps),
        num_train_epochs=config.training.num_epochs,
        learning_rate=config.training.learning_rate,
        bf16=use_bf16,
        fp16=use_fp16,
        optim="adamw_torch",
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=eval_step,
        save_strategy="steps",
        save_steps=eval_step,
        save_total_limit=config.output.save_total_limit,
        load_best_model_at_end=bool(config.training.load_best_model_at_end),
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        group_by_length=bool(config.training.group_by_length),
        report_to=config.logging.report_to,
    )
    trainer = transformers.Trainer(
        model=model,
        train_dataset=hf_train,
        eval_dataset=hf_eval,
        args=args,
        data_collator=transformers.DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True),
        callbacks=callbacks,
    )
    model.config.use_cache = False
    trainer.train(resume_from_checkpoint=config.output.resume_from_checkpoint)
    trainer.save_model(config.output.output_dir)
    final_checkpoint = os.path.join(config.output.output_dir, "final_checkpoint")
    trainer.model.save_pretrained(final_checkpoint)
    tokenizer.save_pretrained(final_checkpoint)
    return final_checkpoint
