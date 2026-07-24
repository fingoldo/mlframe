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
import threading
from typing import Any, Callable, TypeVar

import torch

logger = logging.getLogger("mlframe.training.neural.base")

T = TypeVar("T")

# Serializes every Trainer-construction-through-predict/fit call process-wide. Lightning's
# ``_replace_dunder_methods`` (lightning/fabric/utilities/data.py) monkeypatches the ``DataLoader`` CLASS
# itself (not per-instance) for the duration of each Trainer call, then restores the pre-patch method on exit.
# If two threads enter concurrently -- exactly what sklearn's ``permutation_importance(..., n_jobs=-1)`` under
# joblib's threading backend does, calling ``estimator.predict()`` on ONE shared model from many worker threads
# -- thread B wraps thread A's already-wrapped method, and their restores interleave inconsistently: the wrap
# depth grows monotonically and PERSISTS process-wide (it's class-level state), eventually blowing Python's
# recursion limit in a totally unrelated LATER Trainer call (caught live: a fuzz combo's permutation-importance
# loop corrupted the DataLoader class enough that the SAME combo's later, single-threaded PDP/ICE predict loop
# hit "RecursionError: maximum recursion depth exceeded" inside Lightning's own wrapper). The identical
# concurrent-Trainer-access root cause also underlies the CUDA device-churn race documented below (one thread's
# CUDA-error recovery mutates the shared model's device placement while a sibling thread is mid-forward-pass on
# the same CUDA tensors). A single process-wide lock around the whole Trainer call eliminates both: no two
# threads can ever be inside Lightning's Trainer/DataLoader machinery simultaneously.
_TRAINER_CALL_LOCK = threading.RLock()
# Reentrancy depth of ``run_with_cuda_cpu_fallback`` on the thread currently holding ``_TRAINER_CALL_LOCK``
# (an RLock, so a nested call -- e.g. fit triggering a nested predict -- re-enters on the same thread).
# Only guarded while the lock is held, so a plain int suffices. Used to run the leaked-patch repair below
# ONLY on the outermost call's exit -- repairing mid-nesting would strip an outer call's still-active wrap.
_trainer_call_depth = 0


def _repair_leaked_lightning_dunder_patch() -> None:
    """Undo Lightning's ``DataLoader``/``BatchSampler`` monkeypatch if a Trainer call left it stuck.

    ``lightning.fabric.utilities.data._replace_dunder_methods`` patches ``__init__``/``__setattr__``/
    ``__delattr__`` on ``DataLoader``/``BatchSampler`` (+ subclasses) for the duration of its ``with`` block,
    restoring them in the code AFTER its ``yield`` -- but that restore loop is NOT inside a try/finally. Any
    exception escaping the wrapped block (a CUDA error, a Lightning validation error, anything) skips the
    restore entirely, leaving the patch stuck on the class. The NEXT unrelated Trainer call re-wraps on top of
    the still-wrapped method, and depth grows monotonically across purely SEQUENTIAL calls (no concurrency
    needed) until Python's recursion limit trips -- caught live via a fuzz combo whose 5th sequential model fit
    in one suite run hit "RecursionError: maximum recursion depth exceeded" inside Lightning's own wrapper, with
    every fit in that combo running one at a time (this module's ``_TRAINER_CALL_LOCK`` only guards the
    CONCURRENT-race variant of this same underlying bug; it does nothing for a single thread that leaks on its
    own exception). Mirrors Lightning's own restore loop exactly, made unconditional by running after every
    outermost locked call regardless of success/failure -- a no-op when Lightning's own restore already ran.
    """
    try:
        from torch.utils.data import BatchSampler, DataLoader
    except ImportError:
        return

    def _all_subclasses(cls: type) -> set:
        """Every subclass, recursively -- avoids a direct dependency on lightning_utilities (a transitive
        dep) purely to reuse its identical ``get_all_subclasses`` helper for this one-off walk."""
        direct = cls.__subclasses__()
        return set(direct).union(*(_all_subclasses(c) for c in direct)) if direct else set()

    for base_cls in (DataLoader, BatchSampler):
        for cls in _all_subclasses(base_cls) | {base_cls}:
            for patched_name in ("__setattr__", "__delattr__", "__init__"):
                saved_name = f"__old{patched_name}"
                if saved_name in cls.__dict__:
                    setattr(cls, patched_name, getattr(cls, saved_name))
                    delattr(cls, saved_name)

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

    The ENTIRE call (including the happy path) runs under ``_TRAINER_CALL_LOCK`` -- see its module-level
    comment for why concurrent Trainer construction/predict is unsafe even without a CUDA error in the mix.
    """
    with _TRAINER_CALL_LOCK:
        global _trainer_call_depth
        _trainer_call_depth += 1
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
        finally:
            _trainer_call_depth -= 1
            if _trainer_call_depth == 0:
                _repair_leaked_lightning_dunder_patch()
