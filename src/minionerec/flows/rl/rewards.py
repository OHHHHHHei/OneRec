from __future__ import annotations

import logging
import math
import random

import torch

logger = logging.getLogger(__name__)


def build_rule_reward(prompt2history, history2target):
    def rule_reward(prompts, completions):
        history = [prompt2history[prompt] for prompt in prompts]
        targets = [history2target[item] for item in history]
        return [1.0 if completion.strip("\n\" ") == targets[i].strip("\n\" ") else 0.0 for i, completion in enumerate(completions)]

    return rule_reward


def build_ranking_reward(prompt2history, history2target, num_generations: int):
    ndcg_rewards = [-1.0 / math.log2(i + 2) for i in range(num_generations)]
    ndcg_rewards = [-value / sum(ndcg_rewards) for value in ndcg_rewards]

    def ranking_reward(prompts, completions):
        history = [prompt2history[prompt] for prompt in prompts]
        targets = [history2target[item] for item in history]
        rewards = []
        hit = False
        group_values = []
        for i, completion in enumerate(completions):
            if completion.strip("\n\" ") == targets[i].strip("\n\" "):
                hit = True
                group_values.append(0.0)
            else:
                group_values.append(ndcg_rewards[i % num_generations])
            if (i + 1) % num_generations == 0:
                rewards.extend(group_values if hit else [0.0] * num_generations)
                hit = False
                group_values = []
        return rewards

    return ranking_reward


def build_semantic_reward(prompt2history, history2target, item2id, item_embeddings):
    def semantic_reward(prompts, completions):
        history = [prompt2history[prompt] for prompt in prompts]
        targets = [history2target[item] for item in history]
        target_ids = [item2id[target.strip("\"\n")] for target in targets]
        resolved_completion_ids = []
        invalid_count = 0
        for idx, completion in enumerate([entry.strip("\"\n") for entry in completions]):
            if completion in item2id:
                resolved_completion_ids.append(item2id[completion])
            else:
                invalid_count += 1
                logger.warning("semantic_reward fallback for invalid completion at index=%s value=%r", idx, completion)
                resolved_completion_ids.append(random.choice(target_ids))
        if invalid_count > 0:
            logger.warning("semantic_reward encountered %s/%s invalid completions; used target-based fallback ids.", invalid_count, len(completions))
        rewards = torch.cosine_similarity(item_embeddings[target_ids], item_embeddings[resolved_completion_ids], dim=-1)
        return rewards.tolist() if isinstance(rewards, torch.Tensor) else rewards

    return semantic_reward
