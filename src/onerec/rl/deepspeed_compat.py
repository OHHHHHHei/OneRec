from __future__ import annotations

import logging
from typing import Any


logger = logging.getLogger(__name__)


def _is_missing_backend_error(exc: BaseException) -> bool:
    return isinstance(exc, AssertionError) and "DeepSpeed backend not set" in str(exc)


def _distributed_backend_ready() -> bool:
    try:
        import torch
    except Exception:
        return False

    if not torch.distributed.is_available():
        return False
    if not torch.distributed.is_initialized():
        return False

    try:
        from deepspeed.comm import comm as ds_comm
    except Exception:
        return True

    cdb = getattr(ds_comm, "cdb", None)
    if cdb is None:
        return False
    is_initialized = getattr(cdb, "is_initialized", None)
    if callable(is_initialized):
        try:
            return bool(is_initialized())
        except Exception:
            return False
    return True


def _patch_bf16_optimizer_destroy_cls(cls: type[Any]) -> bool:
    destroy = getattr(cls, "destroy", None)
    if destroy is None:
        return False
    if getattr(destroy, "_onerec_safe_destroy_patch", False):
        return True

    original_destroy = destroy

    def safe_destroy(self, *args, **kwargs):
        if not getattr(self, "using_real_optimizer", True):
            return None

        optimizer = getattr(self, "optimizer", None)
        param_groups = getattr(optimizer, "param_groups", ()) if optimizer is not None else ()
        bf16_groups = getattr(self, "bf16_groups", None)
        if bf16_groups is not None and len(bf16_groups) < len(param_groups):
            return None

        return original_destroy(self, *args, **kwargs)

    safe_destroy._onerec_safe_destroy_patch = True
    safe_destroy._onerec_original_destroy = original_destroy
    cls.destroy = safe_destroy
    return True


def _patch_zero_optimizer_destroy_cls(cls: type[Any]) -> bool:
    destroy = getattr(cls, "destroy", None)
    if destroy is None:
        return False
    if getattr(destroy, "_onerec_safe_destroy_patch", False):
        return True

    original_destroy = destroy

    def safe_destroy(self, *args, **kwargs):
        if not _distributed_backend_ready():
            return None
        try:
            return original_destroy(self, *args, **kwargs)
        except BaseException as exc:
            if _is_missing_backend_error(exc):
                return None
            raise

    safe_destroy._onerec_safe_destroy_patch = True
    safe_destroy._onerec_original_destroy = original_destroy
    cls.destroy = safe_destroy
    return True


def _patch_engine_destroy_cls(cls: type[Any]) -> bool:
    destroy = getattr(cls, "destroy", None)
    if destroy is None:
        return False
    if getattr(destroy, "_onerec_safe_destroy_patch", False):
        return True

    original_destroy = destroy

    def safe_destroy(self, *args, **kwargs):
        try:
            return original_destroy(self, *args, **kwargs)
        except BaseException as exc:
            if _is_missing_backend_error(exc):
                return None
            raise

    safe_destroy._onerec_safe_destroy_patch = True
    safe_destroy._onerec_original_destroy = original_destroy
    cls.destroy = safe_destroy
    return True


def patch_bf16_optimizer_destroy() -> bool:
    try:
        from deepspeed.runtime.bf16_optimizer import BF16_Optimizer
    except Exception:
        return False

    patched = _patch_bf16_optimizer_destroy_cls(BF16_Optimizer)
    if patched:
        logger.info(
            "Applied DeepSpeed BF16 destroy compatibility patch "
            "for DummyOptim cleanup safety."
        )
    return patched


def patch_deepspeed_cleanup() -> bool:
    patched_any = patch_bf16_optimizer_destroy()

    try:
        from deepspeed.runtime.engine import DeepSpeedEngine
    except Exception:
        DeepSpeedEngine = None
    if DeepSpeedEngine is not None:
        patched_any = _patch_engine_destroy_cls(DeepSpeedEngine) or patched_any

    try:
        import deepspeed.runtime.zero.stage_1_and_2 as zero_stage_1_and_2
    except Exception:
        zero_stage_1_and_2 = None
    if zero_stage_1_and_2 is not None:
        for name in dir(zero_stage_1_and_2):
            candidate = getattr(zero_stage_1_and_2, name, None)
            if not isinstance(candidate, type):
                continue
            if getattr(candidate, "__module__", "") != zero_stage_1_and_2.__name__:
                continue
            if not hasattr(candidate, "destroy") or not hasattr(candidate, "print_rank_0"):
                continue
            patched_any = _patch_zero_optimizer_destroy_cls(candidate) or patched_any

    if patched_any:
        logger.info("Applied DeepSpeed cleanup compatibility patches for RL shutdown.")
    return patched_any
