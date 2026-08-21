"""Regression: the numba warmup must call _max_abs_pct_error_kernel_par with its full
3-arg arity (y_true, y_pred, nthr). The kernel sizes a per-thread accumulator from nthr
and indexes it by numba.get_thread_id(), so a missing nthr both raises TypeError (silently
swallowed by the warmup try/except, aborting every later kernel warmup in the same block)
and would index out of bounds if it ran. This pins the call arity behaviourally.
"""


def test_warmup_calls_mape_par_kernel_with_nthr():
    """Warmup calls mape par kernel with nthr."""
    from mlframe.metrics._core_precision_mape import _max_abs_pct_error_kernel_par
    from mlframe.metrics import _core_numba_warmup as warmup

    # Behavioral (not monkeypatch-spy) check (2026-08-21): a monkeypatch.setattr(core,
    # "_max_abs_pct_error_kernel_par", spy) here reliably fails to be observed by the warmup body
    # on CI -- confirmed via a diagnostic showing the warmup body still calling the REAL
    # njit-compiled kernel (CPUDispatcher) instead of the test's spy (a plain function), for a
    # reason that remains unconfirmed despite extensive investigation this session (also affected
    # an unrelated facade-attribute monkeypatch elsewhere, see
    # test_pipeline_json_disk_cache_roundtrip's history). Sidesteps the whole mystery: numba
    # records one compiled signature per DISTINCT arg-type tuple it's called with, so a 3-arg
    # signature can only exist if something called this kernel with exactly 3 positional args --
    # verifying the real, observable side effect instead of intercepting the call.
    warmup.prewarm_numba_cache()

    sigs = _max_abs_pct_error_kernel_par.nopython_signatures
    assert sigs, "warmup never reached the mape par kernel (earlier kernel aborted the block?)"
    assert any(
        len(sig.args) == 3 for sig in sigs
    ), f"warmup must pass (y_true, y_pred, nthr) -- no compiled signature has 3 args; got {[len(sig.args) for sig in sigs]}"
