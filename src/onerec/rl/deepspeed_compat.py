from __future__ import annotations

import logging
from typing import Any


logger = logging.getLogger(__name__)


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
