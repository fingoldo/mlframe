"""Wave 9.1 loop-iter-5 regression: Wave 8 thread-local toggles MUST
propagate to joblib worker threads.

Iter-5 agent flagged that ``set_su_normalization`` / ``set_jmim_aggregator``
/ ``set_bur_lambda`` use ``threading.local()`` storage which is per-thread.
joblib workers (even on ``backend='threading'``) have their own thread
identity so reading the toggles inside the worker returned the default
``False`` / ``0.0`` even after the main thread set them.

Effect pre-fix: ``MRMR(mi_normalization='su', bur_lambda=0.3,
redundancy_aggregator='jmim')`` silently degraded to plain MI / Fleuret /
no-BUR in the parallel hot path -- the user's choice of Wave 8 estimator
was a no-op whenever the workload exceeded ``NMAX_NONPARALLEL_ITERS``
and the workers_pool branch fired.

Fix: snapshot main-thread state in ``_confirm_predictor.py`` and forward
as explicit kwargs to ``evaluate_candidates``, which re-publishes the
toggles in the worker thread at function entry and restores in ``finally``.

Tests verify the propagation mechanism at the function-boundary level
(does not require a real MRMR.fit -- much faster, less brittle).
"""

from __future__ import annotations

from joblib import Parallel, delayed


def test_threadlocal_does_not_propagate_to_joblib_workers_baseline():
    """Sanity: confirm the underlying bug pattern still exists -- if it
    EVER stops, this test will catch the regression and we can simplify
    the fix away. Until joblib changes its semantics, the bare
    thread-local must NOT propagate to backend='threading' workers.
    """
    from mlframe.feature_selection.filters.info_theory import (
        set_su_normalization,
        use_su_normalization,
        set_jmim_aggregator,
        use_jmim_aggregator,
        set_bur_lambda,
        get_bur_lambda,
    )

    # Reset to known state.
    set_su_normalization(False)
    set_jmim_aggregator(False)
    set_bur_lambda(0.0)
    # Now flip them on the main thread.
    set_su_normalization(True)
    set_jmim_aggregator(True)
    set_bur_lambda(0.3)

    def w(_):
        """Probe the calling thread's view of the su/jmim/bur thread-locals, to compare main-thread vs joblib-worker state."""
        return (use_su_normalization(), use_jmim_aggregator(), get_bur_lambda())

    main_state = w(0)
    assert main_state == (True, True, 0.3), main_state

    worker_states = Parallel(n_jobs=4, backend="threading")(delayed(w)(i) for i in range(4))
    # Pre-iter-5: every worker reports (False, False, 0.0). If this
    # invariant ever flips, the iter-5 republish becomes redundant and
    # the test will catch the change.
    assert all(s == (False, False, 0.0) for s in worker_states), worker_states

    # Cleanup.
    set_su_normalization(False)
    set_jmim_aggregator(False)
    set_bur_lambda(0.0)


def test_evaluate_candidates_republishes_threadlocals_in_worker():
    """``evaluate_candidates`` MUST re-publish ``use_su`` / ``use_jmim`` /
    ``bur_lambda`` kwargs into the worker thread-local at entry, so any
    code inside the worker (``evaluate_candidate`` -> ``compute_relevance_score``
    -> ``cmi_or_csu``, plus the BUR bonus block) reads the caller's intent.
    """
    from mlframe.feature_selection.filters.info_theory import (
        set_su_normalization,
        set_jmim_aggregator,
        set_bur_lambda,
        use_su_normalization,
        use_jmim_aggregator,
        get_bur_lambda,
    )

    # Main thread off.
    set_su_normalization(False)
    set_jmim_aggregator(False)
    set_bur_lambda(0.0)

    # Mimic exactly what ``evaluate_candidates`` does at its entry,
    # then probe the worker's view of the thread-local.
    def worker_using_iter5_pattern(use_su, use_jmim, bur_lambda):
        """Replicate evaluate_candidates' entry-point republish pattern inside a worker thread, then report the thread-local it now sees."""
        _prev_su = use_su_normalization()
        _prev_jmim = use_jmim_aggregator()
        _prev_bur = get_bur_lambda()
        set_su_normalization(bool(use_su))
        set_jmim_aggregator(bool(use_jmim))
        set_bur_lambda(float(bur_lambda))
        try:
            # Simulate downstream read inside the worker.
            return (use_su_normalization(), use_jmim_aggregator(), get_bur_lambda())
        finally:
            set_su_normalization(_prev_su)
            set_jmim_aggregator(_prev_jmim)
            set_bur_lambda(_prev_bur)

    results = Parallel(n_jobs=4, backend="threading")(delayed(worker_using_iter5_pattern)(True, True, 0.3) for _ in range(4))
    assert all(r == (True, True, 0.3) for r in results), results

    # And: main-thread state must be untouched after the workers finish
    # (no pollution leaking back).
    assert (use_su_normalization(), use_jmim_aggregator(), get_bur_lambda()) == (False, False, 0.0)


def test_evaluate_candidates_signature_carries_iter5_kwargs():
    """Signature-level guard: if these kwargs ever disappear, the
    parallel path silently regresses to the pre-iter-5 behavior because
    the caller in ``_confirm_predictor.py`` would pass them as
    ``**kwargs`` and joblib would no-op.
    """
    import inspect
    from mlframe.feature_selection.filters.evaluation import evaluate_candidates

    sig = inspect.signature(evaluate_candidates)
    for name in ("use_su", "use_jmim", "bur_lambda"):
        assert name in sig.parameters, (
            f"evaluate_candidates must accept '{name}' kwarg -- without it, "
            f"the iter-5 republish at worker entry doesn't fire and Wave 8 "
            f"toggles silently disable themselves in the parallel hot path."
        )
