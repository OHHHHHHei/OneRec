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


def run_sft(config) -> str:
    set_global_seed(config.training.seed)
    if config.model.train_from_scratch:
        model_config = AutoConfig.from_pretrained(config.model.base_model)
        model = AutoModelForCausalLM.from_config(model_config)
    else:
        model = AutoModelForCausalLM.from_pretrained(config.model.base_model, torch_dtype=torch.bfloat16)
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

    gradient_accumulation_steps = max(1, config.training.batch_size // max(1, config.training.micro_batch_size))
    args = transformers.TrainingArguments(
        output_dir=config.output.output_dir,
        run_name=config.logging.wandb_run_name,
        per_device_train_batch_size=config.training.micro_batch_size,
        per_device_eval_batch_size=config.training.micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=config.training.num_epochs,
        learning_rate=config.training.learning_rate,
        bf16=True,
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
