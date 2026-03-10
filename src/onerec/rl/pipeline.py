import gc
import json
import logging
import os
import pickle

import torch
from datasets import Dataset
from torch.utils.data import ConcatDataset
from transformers import AutoTokenizer
from trl import GRPOConfig

from onerec.rl.deepspeed_compat import patch_bf16_optimizer_destroy
from onerec.utils.seed import set_global_seed
from onerec.rl.datasets import RLSeqTitle2SidDataset, RLTitle2SidDataset, SidDataset
from onerec.rl.rewards import build_ranking_reward, build_rule_reward, build_semantic_reward
from onerec.rl.trainer import ReReTrainer


logger = logging.getLogger(__name__)


def _resolve_precision() -> tuple[torch.dtype, bool, bool]:
    has_cuda = torch.cuda.is_available()
    # Keep mainline aligned with legacy MiniOneRec: CUDA runs default to BF16.
    use_bf16 = bool(has_cuda)
    use_fp16 = False
    model_dtype = torch.bfloat16 if use_bf16 else torch.float32
    return model_dtype, use_bf16, use_fp16


def _wait_for_everyone(trainer) -> None:
    accelerator = getattr(trainer, "accelerator", None)
    if accelerator is not None and hasattr(accelerator, "wait_for_everyone"):
        accelerator.wait_for_everyone()


def _is_main_process(trainer) -> bool:
    accelerator = getattr(trainer, "accelerator", None)
    if accelerator is None:
        return True
    return bool(getattr(accelerator, "is_main_process", True))


def _cleanup_rl_runtime(trainer) -> None:
    if trainer is None:
        return

    accelerator = getattr(trainer, "accelerator", None)
    try:
        _wait_for_everyone(trainer)
    except Exception:
        logger.warning("RL cleanup: wait_for_everyone failed before shutdown.", exc_info=True)

    if accelerator is not None and hasattr(accelerator, "end_training"):
        try:
            accelerator.end_training()
        except Exception:
            logger.warning("RL cleanup: accelerator.end_training() failed.", exc_info=True)

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        try:
            torch.distributed.destroy_process_group()
        except Exception:
            logger.warning("RL cleanup: destroy_process_group() failed.", exc_info=True)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_rl(config) -> str | None:
    set_global_seed(config.training.seed)
    patch_bf16_optimizer_destroy()
    if config.logging.wandb_project:
        os.environ["WANDB_PROJECT"] = config.logging.wandb_project
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)

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
    train_dataset = Dataset.from_dict({key: [row[key] for row in train_data] for key in train_data[0].keys()}).shuffle(seed=config.training.seed)
    if config.training.sample_train and "sft" in config.model.base_model.lower():
        start = int(0.2 * len(train_dataset))
        train_dataset = train_dataset.select(range(start, len(train_dataset)))
    eval_dataset = Dataset.from_dict({key: [row[key] for row in eval_data] for key in eval_data[0].keys()}).shuffle(seed=config.training.seed)

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
    elif config.training.reward_type == "ranking_only":
        reward_fun = build_ranking_reward(prompt2history, history2target, config.training.num_generations)
    elif config.training.reward_type == "semantic":
        ada_path = config.training.ada_path or config.extras.get("ada_path")
        if not ada_path:
            raise ValueError("reward_type=semantic requires training.ada_path (or ada_path override).")
        with open(ada_path, "rb") as handle:
            item_embeddings = torch.tensor(pickle.load(handle))
        reward_fun = build_semantic_reward(prompt2history, history2target, item2id, item_embeddings)
    elif config.training.reward_type == "sasrec":
        cf_path = config.training.cf_path or config.extras.get("cf_path")
        if not cf_path:
            raise ValueError("reward_type=sasrec requires training.cf_path (or cf_path override).")
        raise NotImplementedError("reward_type=sasrec is reserved for legacy CF reward path and is not wired in mainline RL pipeline yet.")

    training_args = GRPOConfig(
        output_dir=config.output.output_dir,
        model_init_kwargs={"torch_dtype": model_dtype},
        save_steps=0.1,
        save_total_limit=config.output.save_total_limit,
        eval_strategy="steps",
        max_completion_length=128,
        num_generations=config.training.num_generations,
        temperature=config.training.temperature,
        sync_ref_model=bool(config.training.sync_ref_model),
        per_device_eval_batch_size=config.training.eval_batch_size,
        per_device_train_batch_size=config.training.train_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        eval_steps=float(config.training.eval_step),
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
    trainer = None
    try:
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
            add_gt=bool(config.training.add_gt),
            dynamic_sampling=bool(config.training.dynamic_sampling),
            beam_search=bool(config.training.beam_search),
            test_during_training=bool(config.training.test_during_training),
            test_beam=int(config.training.test_beam),
            dapo=bool(config.training.dapo),
            gspo=bool(config.training.gspo),
        )
        trainer.train(resume_from_checkpoint=config.output.resume_from_checkpoint)
        _wait_for_everyone(trainer)

        final_checkpoint = os.path.join(config.output.output_dir, "final_checkpoint")
        if _is_main_process(trainer):
            trainer.model.save_pretrained(final_checkpoint)
            AutoTokenizer.from_pretrained(config.model.base_model).save_pretrained(final_checkpoint)

        _wait_for_everyone(trainer)
        return final_checkpoint if _is_main_process(trainer) else None
    finally:
        _cleanup_rl_runtime(trainer)
        trainer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
