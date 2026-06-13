"""Regression: the numba warmup must call _max_abs_pct_error_kernel_par with its full
3-arg arity (y_true, y_pred, nthr). The kernel sizes a per-thread accumulator from nthr
and indexes it by numba.get_thread_id(), so a missing nthr both raises TypeError (silently
swallowed by the warmup try/except, aborting every later kernel warmup in the same block)
and would index out of bounds if it ran. This pins the call arity behaviourally.
"""
import numpy as np


def test_warmup_calls_mape_par_kernel_with_nthr(monkeypatch):
    import mlframe.metrics.core as core
    from mlframe.metrics import _core_numba_warmup as warmup

    seen = {"nargs": None, "called": False}

    def _spy(*args):
        seen["called"] = True
        seen["nargs"] = len(args)
        return (0.0, 0, 0)

    # The warmup body does `from .core import ... _max_abs_pct_error_kernel_par`, so the
    # name resolves from the core module at call time -> patching core catches it.
    monkeypatch.setattr(core, "_max_abs_pct_error_kernel_par", _spy)

    warmup._prewarm_numba_cache_body()

    assert seen["called"], "warmup never reached the mape par kernel (earlier kernel aborted the block?)"
    assert seen["nargs"] == 3, f"warmup must pass (y_true, y_pred, nthr); got {seen['nargs']} args"
