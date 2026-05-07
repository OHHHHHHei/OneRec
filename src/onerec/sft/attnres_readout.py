from __future__ import annotations

import json
import types
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast


READOUT_CONFIG_NAME = "attnres_readout_config.json"
READOUT_WEIGHTS_NAME = "attnres_readout.pt"


class RMSNormNoAffine(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_float = x.float()
        inv_rms = torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x_float * inv_rms).to(dtype=x.dtype)


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _resolve_source_indices(num_hidden_layers: int, spec: str) -> list[int]:
    hidden_state_count = num_hidden_layers + 1
    spec = str(spec or "last8").strip().lower()
    if spec.startswith("last"):
        count = int(spec.removeprefix("last"))
        start = max(1, hidden_state_count - count)
        return list(range(start, hidden_state_count))
    if spec == "all":
        return list(range(1, hidden_state_count))
    indices = []
    for raw in spec.split(","):
        raw = raw.strip()
        if not raw:
            continue
        idx = int(raw)
        if idx < 0:
            idx = hidden_state_count + idx
        if idx < 0 or idx >= hidden_state_count:
            raise ValueError(f"attnres source layer index out of range: {raw} for {num_hidden_layers} layers")
        indices.append(idx)
    if not indices:
        raise ValueError(f"Empty attnres source layer spec: {spec}")
    return indices


def _build_level_lookup(tokenizer) -> torch.Tensor:
    vocab = tokenizer.get_vocab()
    lookup = torch.full((len(tokenizer),), -1, dtype=torch.long)
    for token, token_id in vocab.items():
        if token.startswith("<a_") and token.endswith(">"):
            lookup[token_id] = 0
        elif token.startswith("<b_") and token.endswith(">"):
            lookup[token_id] = 1
        elif token.startswith("<c_") and token.endswith(">"):
            lookup[token_id] = 2
    return lookup


class HierarchyAwareAttnResReadout(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        source_indices: list[int],
        mode: str = "sid_only",
        init_mode: str = "final_layer_biased",
        bias_strength: float = 10.0,
        use_rmsnorm: bool = True,
        level_lookup: torch.Tensor | None = None,
    ):
        super().__init__()
        mode = str(mode).strip().lower()
        if mode not in {"global", "sid_only", "level_aware"}:
            raise ValueError(f"Unsupported attnres readout mode: {mode}")
        self.hidden_size = int(hidden_size)
        self.vocab_size = int(vocab_size)
        self.source_indices = [int(idx) for idx in source_indices]
        self.mode = mode
        self.init_mode = str(init_mode or "final_layer_biased")
        self.bias_strength = float(bias_strength)
        self.use_rmsnorm = bool(use_rmsnorm)
        self.num_routes = 3 if mode == "level_aware" else 1

        self.query = nn.Parameter(torch.zeros(self.num_routes, self.hidden_size))
        self.layer_bias = nn.Parameter(torch.zeros(self.num_routes, len(self.source_indices)))
        if self.init_mode == "final_layer_biased":
            with torch.no_grad():
                self.layer_bias.fill_(-self.bias_strength)
                self.layer_bias[:, -1] = 0.0
        elif self.init_mode != "uniform":
            raise ValueError(f"Unsupported attnres init mode: {self.init_mode}")

        self.norm = RMSNormNoAffine() if self.use_rmsnorm else nn.Identity()
        if level_lookup is None:
            level_lookup = torch.full((self.vocab_size,), -1, dtype=torch.long)
        self.register_buffer("level_lookup", level_lookup.long(), persistent=False)

    def export_config(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "source_indices": self.source_indices,
            "init_mode": self.init_mode,
            "bias_strength": self.bias_strength,
            "use_rmsnorm": self.use_rmsnorm,
            "hidden_size": self.hidden_size,
            "vocab_size": self.vocab_size,
        }

    def refresh_tokenizer(self, tokenizer) -> None:
        self.vocab_size = len(tokenizer)
        self.level_lookup = _build_level_lookup(tokenizer).to(device=self.level_lookup.device)

    def _route_ids_from_labels(self, labels: torch.Tensor, seq_len: int) -> torch.Tensor:
        route_ids = torch.full(labels.shape, -1, dtype=torch.long, device=labels.device)
        if seq_len <= 1:
            return route_ids
        target_ids = labels[:, 1:]
        valid = target_ids >= 0
        safe_target_ids = target_ids.clamp_min(0)
        lookup = self.level_lookup.to(device=labels.device)
        token_routes = lookup[safe_target_ids]
        token_routes = token_routes.masked_fill(~valid, -1)
        if self.mode == "sid_only":
            token_routes = torch.where(token_routes >= 0, torch.zeros_like(token_routes), token_routes)
        route_ids[:, : seq_len - 1] = token_routes[:, : seq_len - 1]
        return route_ids[:, :seq_len]

    def _next_route_from_input_ids(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        route_ids = torch.full(input_ids.shape, -1, dtype=torch.long, device=input_ids.device)
        lookup = self.level_lookup.to(device=input_ids.device)
        for row_idx in range(input_ids.shape[0]):
            if attention_mask is not None and attention_mask.shape == input_ids.shape:
                valid_positions = attention_mask[row_idx].nonzero(as_tuple=False).flatten()
                pos = int(valid_positions[-1].item()) if len(valid_positions) else input_ids.shape[1] - 1
            else:
                pos = input_ids.shape[1] - 1
            last_token = int(input_ids[row_idx, pos].item())
            last_level = int(lookup[last_token].item()) if 0 <= last_token < lookup.numel() else -1
            if self.mode == "global":
                next_route = 0
            elif last_level == 0:
                next_route = 1 if self.mode == "level_aware" else 0
            elif last_level == 1:
                next_route = 2 if self.mode == "level_aware" else 0
            elif last_level == 2:
                next_route = -1
            else:
                next_route = 0
            route_ids[row_idx, pos] = next_route
        return route_ids

    def route_ids(
        self,
        input_ids: torch.Tensor,
        seq_len: int,
        labels: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.mode == "global" and labels is not None:
            return torch.zeros(labels.shape[0], seq_len, dtype=torch.long, device=labels.device)
        if labels is not None:
            return self._route_ids_from_labels(labels, seq_len)
        return self._next_route_from_input_ids(input_ids, attention_mask)

    def _mix_route(self, hidden_states: tuple[torch.Tensor, ...], route_idx: int) -> torch.Tensor:
        values = torch.stack([hidden_states[idx] for idx in self.source_indices], dim=0)
        keys = self.norm(values)
        query = self.query[route_idx].to(dtype=keys.dtype)
        logits = torch.einsum("d,sbtd->sbt", query, keys)
        logits = logits + self.layer_bias[route_idx].to(dtype=logits.dtype).view(-1, 1, 1)
        weights = torch.softmax(logits.float(), dim=0).to(dtype=values.dtype)
        return torch.einsum("sbt,sbtd->btd", weights, values)

    def mix(self, hidden_states: tuple[torch.Tensor, ...], route_ids: torch.Tensor) -> torch.Tensor:
        mixed = hidden_states[-1].clone()
        for route_idx in range(self.num_routes):
            mask = route_ids == route_idx
            if not torch.any(mask):
                continue
            route_hidden = self._mix_route(hidden_states, route_idx)
            mixed[mask] = route_hidden[mask]
        return mixed

    def _mix_route_flat(self, hidden_states: tuple[torch.Tensor, ...], flat_indices: torch.Tensor, route_idx: int) -> torch.Tensor:
        hidden_size = hidden_states[-1].shape[-1]
        values = torch.stack(
            [hidden_states[idx].reshape(-1, hidden_size).index_select(0, flat_indices) for idx in self.source_indices],
            dim=0,
        )
        keys = self.norm(values)
        query = self.query[route_idx].to(dtype=keys.dtype)
        logits = torch.einsum("d,snd->sn", query, keys)
        logits = logits + self.layer_bias[route_idx].to(dtype=logits.dtype).view(-1, 1)
        weights = torch.softmax(logits.float(), dim=0).to(dtype=values.dtype)
        return torch.einsum("sn,snd->nd", weights, values)

    def mix_selected(self, hidden_states: tuple[torch.Tensor, ...], route_ids: torch.Tensor, selected_mask: torch.Tensor) -> torch.Tensor:
        hidden_size = hidden_states[-1].shape[-1]
        flat_selected = selected_mask.reshape(-1).nonzero(as_tuple=False).flatten()
        final_flat = hidden_states[-1].reshape(-1, hidden_size)
        mixed = final_flat.index_select(0, flat_selected).clone()
        selected_routes = route_ids.reshape(-1).index_select(0, flat_selected)
        for route_idx in range(self.num_routes):
            route_mask = selected_routes == route_idx
            if not torch.any(route_mask):
                continue
            route_flat = flat_selected.index_select(0, route_mask.nonzero(as_tuple=False).flatten())
            mixed[route_mask] = self._mix_route_flat(hidden_states, route_flat, route_idx)
        return mixed


def _compute_shifted_ce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1).to(shift_logits.device),
        ignore_index=-100,
    )


def _compute_selected_shifted_ce_loss(
    mixed_shift_hidden: torch.Tensor,
    shift_labels: torch.Tensor,
    lm_head: nn.Module,
) -> torch.Tensor:
    selected_labels = shift_labels[shift_labels != -100].to(mixed_shift_hidden.device)
    if selected_labels.numel() == 0:
        return mixed_shift_hidden.sum() * 0.0
    logits = lm_head(mixed_shift_hidden)
    return F.cross_entropy(logits.float(), selected_labels)


def _zero_parameter_anchor(module: nn.Module) -> torch.Tensor:
    anchor = None
    for param in module.parameters():
        term = param.sum() * 0.0
        anchor = term if anchor is None else anchor + term
    if anchor is None:
        raise ValueError("AttnRes readout has no parameters")
    return anchor


def apply_attnres_readout(
    model,
    tokenizer,
    *,
    mode: str,
    source_layers: str,
    init_mode: str,
    bias_strength: float,
    use_rmsnorm: bool,
):
    if getattr(model, "_onerec_attnres_patched", False):
        model.attnres_readout.refresh_tokenizer(tokenizer)
        return model

    num_hidden_layers = int(getattr(model.config, "num_hidden_layers", 0))
    hidden_size = int(getattr(model.config, "hidden_size", 0))
    if num_hidden_layers <= 0 or hidden_size <= 0:
        raise ValueError("AttnRes readout requires model.config.num_hidden_layers and hidden_size")
    source_indices = _resolve_source_indices(num_hidden_layers, source_layers)
    readout = HierarchyAwareAttnResReadout(
        hidden_size=hidden_size,
        vocab_size=len(tokenizer),
        source_indices=source_indices,
        mode=mode,
        init_mode=init_mode,
        bias_strength=bias_strength,
        use_rmsnorm=use_rmsnorm,
        level_lookup=_build_level_lookup(tokenizer),
    )
    model.add_module("attnres_readout", readout)
    original_forward = model.forward

    def forward_with_attnres(self, *args, **kwargs):
        labels = kwargs.pop("labels", None)
        input_ids = kwargs.get("input_ids", args[0] if args else None)
        attention_mask = kwargs.get("attention_mask", None)
        kwargs["output_hidden_states"] = True
        kwargs["return_dict"] = True
        kwargs.setdefault("logits_to_keep", 1)
        outputs = original_forward(*args, labels=None, **kwargs)
        if input_ids is None:
            logits = outputs.logits
            loss = _compute_shifted_ce_loss(logits, labels) if labels is not None else None
            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
        route_ids = self.attnres_readout.route_ids(
            input_ids=input_ids,
            seq_len=outputs.hidden_states[-1].shape[1],
            labels=labels,
            attention_mask=attention_mask,
        )
        if labels is not None:
            shift_labels = labels[:, 1:].contiguous()
            selected_mask = shift_labels != -100
            shift_hidden_states = tuple(hidden[:, :-1, :] for hidden in outputs.hidden_states)
            mixed_shift_hidden = self.attnres_readout.mix_selected(
                shift_hidden_states,
                route_ids[:, :-1],
                selected_mask,
            )
            loss = _compute_selected_shifted_ce_loss(mixed_shift_hidden, shift_labels, self.get_output_embeddings())
            loss = loss + _zero_parameter_anchor(self.attnres_readout)
            logits = outputs.logits
        else:
            selected_mask = torch.zeros(route_ids.shape, dtype=torch.bool, device=route_ids.device)
            selected_mask[:, -1] = True
            mixed_last_hidden = self.attnres_readout.mix_selected(outputs.hidden_states, route_ids, selected_mask)
            logits = self.get_output_embeddings()(mixed_last_hidden.unsqueeze(1))
            loss = None
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    model.forward = types.MethodType(forward_with_attnres, model)
    model._onerec_attnres_patched = True
    # The patched forward uses **kwargs only to preserve the base model API. It
    # does not consume Trainer's num_items_in_batch loss kwarg, so opt out to
    # keep gradient-accumulation loss normalization correct.
    model.accepts_loss_kwargs = False
    model.config.attnres_readout_enable = True
    model.config.attnres_readout_mode = mode
    model.config.attnres_source_layers = source_layers
    model.config.attnres_init_mode = init_mode
    model.config.attnres_bias_strength = float(bias_strength)
    model.config.attnres_use_rmsnorm = bool(use_rmsnorm)
    return model


def save_attnres_readout(model, checkpoint_dir: str | Path) -> None:
    if not hasattr(model, "attnres_readout"):
        return
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.attnres_readout.state_dict(), checkpoint_dir / READOUT_WEIGHTS_NAME)
    with open(checkpoint_dir / READOUT_CONFIG_NAME, "w", encoding="utf-8") as handle:
        json.dump(model.attnres_readout.export_config(), handle, indent=2)


def maybe_apply_saved_attnres_readout(model, tokenizer, checkpoint_dir: str | Path):
    checkpoint_dir = Path(checkpoint_dir)
    config_path = checkpoint_dir / READOUT_CONFIG_NAME
    weights_path = checkpoint_dir / READOUT_WEIGHTS_NAME
    if not config_path.exists() or not weights_path.exists():
        return model
    with open(config_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    source_indices = payload.get("source_indices")
    if source_indices:
        num_hidden_layers = int(getattr(model.config, "num_hidden_layers", 0))
        hidden_state_count = num_hidden_layers + 1
        source_layers = ",".join(str(idx if idx >= 0 else hidden_state_count + idx) for idx in source_indices)
    else:
        source_layers = payload.get("source_layers", "last8")
    model = apply_attnres_readout(
        model,
        tokenizer,
        mode=payload["mode"],
        source_layers=source_layers,
        init_mode=payload.get("init_mode", "final_layer_biased"),
        bias_strength=float(payload.get("bias_strength", 10.0)),
        use_rmsnorm=_parse_bool(payload.get("use_rmsnorm", True)),
    )
    state = torch.load(weights_path, map_location="cpu")
    model.attnres_readout.load_state_dict(state)
    return model
