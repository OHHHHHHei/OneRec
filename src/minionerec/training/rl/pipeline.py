import json
import logging
import os
import pickle

import torch
from datasets import Dataset
from torch.utils.data import ConcatDataset
from transformers import AutoTokenizer
from trl import GRPOConfig

from minionerec.data.datasets.rl import RLSeqTitle2SidDataset, RLTitle2SidDataset, SidDataset
from minionerec.training.rl.rewards import build_ranking_reward, build_rule_reward, build_semantic_reward
from minionerec.training.rl.trainer import ReReTrainer


logger = logging.getLogger(__name__)


def _resolve_precision() -> tuple[torch.dtype, bool, bool]:
    has_cuda = torch.cuda.is_available()
    use_bf16 = bool(has_cuda and torch.cuda.is_bf16_supported())
    use_fp16 = bool(has_cuda and not use_bf16)
    model_dtype = torch.bfloat16 if use_bf16 else torch.float16 if use_fp16 else torch.float32
    return model_dtype, use_bf16, use_fp16


def run_rl(config) -> str:
    model_dtype, use_bf16, use_fp16 = _resolve_precision()
    logger.info("RL precision config: bf16=%s fp16=%s dtype=%s", use_bf16, use_fp16, model_dtype)

    with open(config.data.info_file, "r", encoding="utf-8") as handle:
        lines = handle.readlines()
    item_names = [line.split("\t")[0].strip() for line in lines]
    item2id = {name: idx for idx, name in enumerate(item_names)}

    train_parts = [
        SidDataset(config.data.train_file, category=config.data.category),
        RLTitle2SidDataset(config.data.item_meta_path, config.data.sid_index_path, category=config.data.category),
        RLSeqTitle2SidDataset(config.data.train_file, category=config.data.category, sample=10000),
    ]
    eval_data = SidDataset(config.data.eval_file, category=config.data.category)
    train_data = ConcatDataset(train_parts)
    train_dataset = Dataset.from_dict({key: [row[key] for row in train_data] for key in train_data[0].keys()})
    eval_dataset = Dataset.from_dict({key: [row[key] for row in eval_data] for key in eval_data[0].keys()})

    prompt2history = {}
    history2target = {}
    for dataset in train_parts + [eval_data]:
        if hasattr(dataset, "prompt2history"):
            prompt2history.update(dataset.prompt2history)
        if hasattr(dataset, "history2target"):
            history2target.update(dataset.history2target)

    reward_fun = build_rule_reward(prompt2history, history2target)
    if config.training.reward_type == "ranking":
        reward_fun = [build_rule_reward(prompt2history, history2target), build_ranking_reward(prompt2history, history2target, config.training.num_generations)]
    elif config.training.reward_type == "semantic":
        with open(config.extras["ada_path"], "rb") as handle:
            item_embeddings = torch.tensor(pickle.load(handle))
        reward_fun = build_semantic_reward(prompt2history, history2target, item2id, item_embeddings)

    training_args = GRPOConfig(
        output_dir=config.output.output_dir,
        model_init_kwargs={"torch_dtype": model_dtype},
        save_steps=0.1,
        save_total_limit=config.output.save_total_limit,
        eval_strategy="steps",
        max_completion_length=128,
        num_generations=config.training.num_generations,
        temperature=config.training.temperature,
        per_device_eval_batch_size=config.training.eval_batch_size,
        per_device_train_batch_size=config.training.train_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        eval_steps=0.1,
        logging_steps=1,
        learning_rate=config.training.learning_rate,
        beta=config.training.beta,
        warmup_ratio=0.03,
        max_grad_norm=0.3,
        num_train_epochs=config.training.num_epochs,
        bf16=use_bf16,
        fp16=use_fp16,
        optim="paged_adamw_32bit",
        lr_scheduler_type="cosine",
        save_strategy="steps",
        report_to=config.logging.report_to,
        run_name=config.logging.wandb_run_name,
    )
    trainer = ReReTrainer(
        model=config.model.base_model,
        base_model=config.model.base_model,
        reward_funcs=reward_fun,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        info_file=config.data.info_file,
        prompt2history=prompt2history,
        history2target=history2target,
        beam_search=True,
        test_during_training=False,
        test_beam=20,
    )
    trainer.train(resume_from_checkpoint=config.output.resume_from_checkpoint)
    trainer.save_model(config.output.output_dir)
    final_checkpoint = os.path.join(config.output.output_dir, "final_checkpoint")
    trainer.model.save_pretrained(final_checkpoint)
    AutoTokenizer.from_pretrained(config.model.base_model).save_pretrained(final_checkpoint)
    return final_checkpoint
