from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


def scale_from_prior(prior: torch.Tensor, low: float, high: float) -> torch.Tensor:
    return low + (high - low) * prior


def load_ambiguity_prior(path: str, column: str, n_items: int, device: torch.device) -> torch.Tensor:
    if not path:
        return torch.zeros(n_items, dtype=torch.float32, device=device)
    df = pd.read_csv(path)
    if "item_id" not in df.columns:
        raise ValueError(f"Ambiguity CSV missing item_id column: {path}")
    if column not in df.columns:
        raise ValueError(f"Ambiguity CSV missing target column `{column}`: {path}")
    values = np.zeros(n_items, dtype=np.float32)
    for row in df.itertuples(index=False):
        item_id = int(getattr(row, "item_id"))
        if 0 <= item_id < n_items:
            value = float(getattr(row, column))
            values[item_id] = np.clip(value if np.isfinite(value) else 0.0, 0.0, 1.0)
    return torch.tensor(values, dtype=torch.float32, device=device)


def load_pair_matrix(
    path: str,
    n_items: int,
    device: torch.device,
    rule: str | None = None,
    use_reliability: bool = True,
) -> torch.Tensor:
    df = pd.read_csv(path)
    required = {"item_a", "item_b"}
    if not required.issubset(df.columns):
        missing = sorted(required - set(df.columns))
        raise ValueError(f"Pair CSV missing required columns {missing}: {path}")
    if rule and "rule" in df.columns:
        df = df[df["rule"] == rule].copy()

    values = torch.zeros((n_items, n_items), dtype=torch.float32, device=device)
    has_reliability = use_reliability and "reliability" in df.columns
    for row in df.itertuples(index=False):
        item_a = int(row.item_a)
        item_b = int(row.item_b)
        if not (0 <= item_a < n_items and 0 <= item_b < n_items) or item_a == item_b:
            continue
        value = 1.0
        if has_reliability:
            value = float(getattr(row, "reliability"))
            if not np.isfinite(value) or value <= 0.0:
                continue
        if value > float(values[item_a, item_b].item()):
            values[item_a, item_b] = value
            values[item_b, item_a] = value
    return values


def pairwise_pull_loss(
    representations: torch.Tensor,
    pair_weights: torch.Tensor,
    item_scales: torch.Tensor,
) -> torch.Tensor:
    if representations.size(0) <= 1:
        return representations.new_tensor(0.0)
    representations = F.normalize(representations.float(), p=2, dim=1)
    similarity = representations @ representations.T
    pair_weights = torch.maximum(pair_weights.float(), pair_weights.T.float())
    if item_scales.numel() > 0:
        pair_scale = torch.sqrt(torch.outer(item_scales.float(), item_scales.float()).clamp_min(0.0))
        pair_weights = pair_weights * pair_scale
    pair_weights = torch.triu(pair_weights, diagonal=1)
    active_mask = pair_weights > 0.0
    if not torch.any(active_mask):
        return representations.new_tensor(0.0)
    penalties = (1.0 - similarity).clamp_min(0.0)
    denom = pair_weights[active_mask].sum().clamp(min=1e-6)
    return torch.sum(penalties * pair_weights) / denom


def graph_guided_infonce_loss(
    representations: torch.Tensor,
    positive_pair_weights: torch.Tensor,
    negative_pair_weights: torch.Tensor,
    item_scales: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if representations.size(0) <= 1:
        return representations.new_tensor(0.0)
    if temperature <= 0:
        raise ValueError(f"graph InfoNCE temperature must be positive, got {temperature}")

    representations = F.normalize(representations.float(), p=2, dim=1)
    similarity = (representations @ representations.T) / temperature
    positive_pair_weights = torch.maximum(positive_pair_weights.float(), positive_pair_weights.T.float())
    negative_pair_weights = torch.maximum(negative_pair_weights.float(), negative_pair_weights.T.float())
    if item_scales.numel() > 0:
        pair_scale = torch.sqrt(torch.outer(item_scales.float(), item_scales.float()).clamp_min(0.0))
        positive_pair_weights = positive_pair_weights * pair_scale
        negative_pair_weights = negative_pair_weights * pair_scale

    eye_mask = torch.eye(similarity.size(0), dtype=torch.bool, device=similarity.device)
    positive_pair_weights = positive_pair_weights.masked_fill(eye_mask, 0.0)
    negative_pair_weights = negative_pair_weights.masked_fill(eye_mask, 0.0)
    active_pair_mask = (positive_pair_weights > 0.0) | (negative_pair_weights > 0.0)
    if not torch.any(positive_pair_weights.sum(dim=1) > 0.0):
        return representations.new_tensor(0.0)

    masked_similarity = similarity.masked_fill(~active_pair_mask, float("-inf"))
    row_has_active = active_pair_mask.any(dim=1)
    row_max = torch.where(row_has_active, masked_similarity.max(dim=1).values, torch.zeros_like(similarity[:, 0]))
    stable_logits = (masked_similarity - row_max.unsqueeze(1)).masked_fill(~active_pair_mask, float("-inf"))
    stabilized_exp = torch.where(active_pair_mask, torch.exp(stable_logits), torch.zeros_like(stable_logits))

    numerator = torch.sum(stabilized_exp * positive_pair_weights, dim=1)
    denominator = numerator + torch.sum(stabilized_exp * negative_pair_weights, dim=1)
    valid_rows = numerator > 0.0
    if not torch.any(valid_rows):
        return representations.new_tensor(0.0)

    per_item = torch.zeros_like(numerator)
    per_item[valid_rows] = -torch.log(
        numerator[valid_rows].clamp_min(1e-12) / denominator[valid_rows].clamp_min(1e-12)
    )
    row_weights = item_scales.float() * valid_rows.float() if item_scales.numel() > 0 else valid_rows.float()
    denom = row_weights.sum().clamp(min=1e-6)
    return torch.sum(per_item * row_weights) / denom
