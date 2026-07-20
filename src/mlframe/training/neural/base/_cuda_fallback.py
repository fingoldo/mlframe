"""Shared CUDA-illegal-memory-access CPU-retry fallback for fit/predict.

Some driver/CUDA-toolkit combos (or GPU contention from another process) leave the in-process
CUDA context invalidated mid-session -- the NEXT CUDA-touching call (a kernel launch, a
cache-clear, an accelerator setup) then raises "CUDA error: an illegal memory access was
encountered" even though model + data were fine moments earlier. Retrying once on CPU lets the
call complete with equivalent numeric results (losing GPU acceleration for that one call only); a
second CUDA-side failure on the CPU retry (the model still resident on the dead GPU) means the
context is permanently dead for this process, so CUDA is hard-disabled at the torch module level
so every subsequent estimator in this process skips straight to CPU instead of repeating the same
dead-context failure.

Originally written for the predict path (``_base_predict.py``); extracted here so ``fit`` can
share the identical retry/disable logic instead of leaving fit-time CUDA failures uncaught.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, TypeVar

import torch

logger = logging.getLogger("mlframe.training.neural.base")

T = TypeVar("T")

CUDA_ERROR_FINGERPRINTS = (
    "CUDA",
    "cuda runtime error",
    "illegal memory access",
    "device-side assert",
    "out of memory",
    "CUBLAS_STATUS_",
    "CUDNN_STATUS_",
    # Device-placement mismatch (model on cuda:0 but a batch/buffer left on cpu) surfaces as a
    # RuntimeError carrying only the lowercase device tag, not the uppercase "CUDA" fingerprint.
    "Expected all tensors to be on the same device",
)


def is_cuda_runtime_error(exc: BaseException, accelerator: str) -> bool:
    """True iff ``exc`` looks like a CUDA-runtime failure AND the caller was actually using CUDA."""
    if accelerator not in ("cuda", "gpu", "auto"):
        return False
    msg = str(exc)
    return any(fp in msg for fp in CUDA_ERROR_FINGERPRINTS)


def _disable_cuda_globally() -> None:
    """Hide CUDA from torch for the rest of this process so subsequent estimators skip straight to CPU."""
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        torch.cuda.is_available = lambda: False
    except Exception as _e_disable:  # nosec B110 - best-effort path
        logger.debug("Failed to hard-disable CUDA globally: %s", _e_disable)


def _best_effort_move_to_cpu(model: Any) -> None:
    """Move the model (and its ``_orig_mod`` if torch-compiled) to CPU; logs but never raises."""
    try:
        model.to("cpu")
        if hasattr(model, "_orig_mod"):
            model._orig_mod.to("cpu")
    except Exception as _e_move:
        logger.error(
            "Failed to move model parameters off the invalidated GPU context (%s); the CPU retry below will likely re-raise the CUDA error.",
            _e_move,
        )


def _best_effort_reset_cuda_state() -> None:
    """Synchronize, empty the cache, and release IPC handles; each step is best-effort/independent."""
    try:
        if not torch.cuda.is_available():
            return
    except Exception as _e_avail:  # nosec B110 - best-effort path
        logger.debug("torch.cuda.is_available() itself failed during CUDA-state reset: %s", _e_avail)
        return
    try:
        torch.cuda.synchronize()
    except Exception as _e_sync:  # nosec B110 - best-effort path
        logger.debug("torch.cuda.synchronize() failed during CUDA-state reset: %s", _e_sync)
    try:
        torch.cuda.empty_cache()
    except Exception as _e_empty:  # nosec B110 - best-effort path
        logger.debug("torch.cuda.empty_cache() failed during CUDA-state reset: %s", _e_empty)
    try:
        torch.cuda.ipc_collect()
    except Exception as _e_ipc:  # nosec B110 - best-effort path
        logger.debug("torch.cuda.ipc_collect() failed during CUDA-state reset: %s", _e_ipc)


def run_with_cuda_cpu_fallback(
    *,
    action: str,
    primary_trainer: Any,
    model: Any,
    accelerator: str,
    run_fn: "Callable[[Any], T]",
    build_cpu_trainer: "Callable[[], Any]",
) -> "tuple[T, Any]":
    """Run ``run_fn(primary_trainer)`` with a CPU retry on CUDA-runtime failure.

    ``run_fn`` is a closure invoking the desired trainer method (``lambda t: t.fit(model=model,
    datamodule=dm)`` / ``lambda t: t.predict(model=model, datamodule=datamodule)``) on whatever
    trainer it's given, returning that call's result. ``build_cpu_trainer`` constructs a FRESH
    CPU-only trainer with the caller's real config -- fit needs its callbacks/max_epochs
    preserved, predict needs only a bare inference trainer -- and is called again for the second
    retry after the hard CUDA-disable below.

    Returns ``(result, trainer_used)`` so the caller can rebind its own ``trainer`` reference
    (e.g. fit reads ``trainer.callbacks`` afterward) to whichever trainer actually produced the
    result.
    """
    try:
        result = run_fn(primary_trainer)
        return result, primary_trainer
    except RuntimeError as e:
        if not is_cuda_runtime_error(e, accelerator):
            logger.error("%s failed: %s", action.capitalize(), e)
            raise
        logger.warning(
            "%s on accelerator=%r failed with CUDA-side error (%s); retrying on CPU. Common cause: "
            "another process holds the GPU or the in-process CUDA context was invalidated by an "
            "earlier failure. The CPU fallback produces equivalent numeric results but loses GPU "
            "acceleration for this single %s.",
            action, accelerator, e, action,
        )
        _best_effort_move_to_cpu(model)
        _best_effort_reset_cuda_state()
        try:
            cpu_trainer = build_cpu_trainer()
            result = run_fn(cpu_trainer)
            return result, cpu_trainer
        except Exception as e_cpu:
            if not is_cuda_runtime_error(e_cpu, accelerator):
                logger.error(
                    "CPU fallback after CUDA %s failure ALSO failed with a non-CUDA error: %s. Original CUDA error: %s",
                    action, e_cpu, e,
                )
                raise
            logger.error(
                "CPU fallback after CUDA %s failure ALSO failed with a CUDA-side error: %s. The CUDA "
                "context is permanently invalidated for this process. Disabling CUDA at the torch "
                "module level so subsequent estimators skip GPU and run on CPU; GPU acceleration will "
                "resume on the next process restart. Original CUDA error: %s",
                action, e_cpu, e,
            )
            _disable_cuda_globally()
            _best_effort_move_to_cpu(model)
            try:
                cpu_trainer2 = build_cpu_trainer()
                result = run_fn(cpu_trainer2)
                return result, cpu_trainer2
            except Exception as e_cpu2:
                logger.error("Even with CUDA hidden the %s failed: %s. Re-raising original CUDA error.", action, e_cpu2)
                raise
