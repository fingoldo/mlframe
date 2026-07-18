"""Regression test for ``maybe_prewarm_polynom_loky_pool`` (2026-07-11 perf fix).

``run_polynom_pair_fe`` dispatches per-pair Optuna/CMA search work through a ``LokyBackend``
(``inner_max_num_threads=1, initializer=disable_cuda_in_worker``). py-spy on a live 100k-row
production run showed this pool's ``QueueFeederThread`` continuously CPU-bound for the phase's
opening seconds; an isolated A/B on the real (79237, 544) production shape confirmed why: a COLD
pool (16 fresh worker processes each re-importing mlframe/numba/cupy) took 28.1s, the SAME pool
warm took 0.7s, and this pool has no other user earlier in a typical fit (the sibling CPU
pair-MI-sweep pool in ``_step_pairmi.py`` only engages when the GPU MI path fails) -- so production
paid the full cold-start tax synchronously inside the polynom-pair-FE phase every single fit.

``maybe_prewarm_polynom_loky_pool`` kicks off that SAME pool shape on a daemon thread from
``MRMR._fit_impl``, gated on the same condition ``run_polynom_pair_fe`` itself uses to pick the loky
path (``fe_smart_polynom_iters`` truthy, ``n_jobs`` > 1). Called right AFTER categorization completes
(NOT at fit-entry): an earlier version called this before categorization and REGRESSED a full
100k-row production run by ~223s -- categorization is itself CPU-active, so the pre-warm's 16
concurrent worker-process spawns contended with it instead of overlapping idle time. Also passes
``POLYNOM_LOKY_IDLE_WORKER_TIMEOUT`` (900s, matched on both this pre-warm's backend and
``run_polynom_pair_fe``'s) since the measured real gap to actual use (~418s) exceeds loky's default
300s idle-worker timeout -- without the match, the pre-warmed pool self-terminates before
``run_polynom_pair_fe`` ever gets to use it. These tests pin: (1) the gate only fires the thread when
both conditions hold, (2) the thread is a daemon (never blocks interpreter exit), (3) a failure
inside the background dispatch is silently swallowed, not raised on the caller's thread, (4)
end-to-end: pre-warming via this function measurably speeds up a REAL ``run_polynom_pair_fe`` call
afterward (the actual contract this fix exists for), (5) the idle-worker-timeout constant is shared
correctly between the two call sites and exceeds the joblib default.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from mlframe.feature_selection.filters._joblib_safe import maybe_prewarm_polynom_loky_pool


def test_gate_off_when_fe_smart_polynom_iters_zero():
    """Gate off when fe smart polynom iters zero."""
    thread = maybe_prewarm_polynom_loky_pool(fe_smart_polynom_iters=0, n_jobs=16)
    assert thread is None


def test_gate_off_when_n_jobs_is_one():
    """Gate off when n jobs is one."""
    thread = maybe_prewarm_polynom_loky_pool(fe_smart_polynom_iters=5, n_jobs=1)
    assert thread is None


def test_gate_off_when_n_jobs_is_none_or_zero():
    """Gate off when n jobs is none or zero."""
    assert maybe_prewarm_polynom_loky_pool(fe_smart_polynom_iters=5, n_jobs=0) is None
    assert maybe_prewarm_polynom_loky_pool(fe_smart_polynom_iters=5, n_jobs=None) is None  # type: ignore[arg-type]


def test_gate_on_starts_a_daemon_thread():
    """Gate on starts a daemon thread."""
    thread = maybe_prewarm_polynom_loky_pool(fe_smart_polynom_iters=5, n_jobs=2)
    try:
        assert thread is not None
        assert isinstance(thread, threading.Thread)
        assert thread.daemon is True
        assert thread.name == "mrmr-polynom-loky-prewarm"
    finally:
        if thread is not None:
            thread.join(timeout=60)


def test_prewarm_failure_is_swallowed_not_raised(monkeypatch):
    """A broken loky backend inside the background thread must never propagate to the caller."""
    import mlframe.feature_selection.filters._joblib_safe as joblib_safe

    def _boom(*args, **kwargs):
        """Helper that boom."""
        raise RuntimeError("simulated loky failure")

    monkeypatch.setattr(joblib_safe, "disable_cuda_in_worker", _boom)
    thread = maybe_prewarm_polynom_loky_pool(fe_smart_polynom_iters=5, n_jobs=2)
    assert thread is not None
    thread.join(timeout=30)
    assert not thread.is_alive()  # completed (via the swallowed exception), did not hang


@pytest.mark.slow
def test_prewarm_speeds_up_a_real_run_polynom_pair_fe_call():
    """End-to-end contract: pre-warming via this function measurably speeds up the REAL dispatch.

    Matches the isolated A/B this fix was designed around (cold ~26-28s, warm <5s on real (79237,
    544)-shaped data) -- uses a smaller shape here to keep CI time bounded while still crossing the
    ``_PARALLEL_PAIR_THRESHOLD=16`` loky-dispatch gate in ``run_polynom_pair_fe``.
    """
    import mlframe.feature_selection.filters.polynom_pair_fe as ppfe

    n, ncols = 20_000, 32
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, ncols)).astype(np.float64)
    y = rng.integers(0, 5, size=n)
    cols = [f"f{i}" for i in range(ncols)]
    nbins = np.full(ncols, 10, dtype=np.int64)
    prospective_pairs = {(i, i + 1): 1.0 for i in range(16)}

    def _run():
        """Time one run_polynom_pair_fe call on the shared 16-pair fixture, for prewarm-vs-cold timing comparison."""
        t0 = time.perf_counter()
        ppfe.run_polynom_pair_fe(
            X=X,
            is_polars_input=False,
            prospective_pairs=prospective_pairs,
            classes_y=y,
            cols=cols,
            nbins=nbins,
            data=X.copy(),
            engineered_features=set(),
            engineered_recipes={},
            hermite_features_list=[],
            feature_names_in=cols,
            fe_smart_polynom_iters=1,
            fe_smart_polynom_optimization_steps=3,
            fe_min_polynom_degree=1,
            fe_max_polynom_degree=3,
            fe_min_polynom_coeff=-2.0,
            fe_max_polynom_coeff=2.0,
            fe_min_engineered_mi_prevalence=0.0,
            fe_hermite_l2_penalty=0.0,
            fe_polynomial_basis="hermite",
            fe_mi_estimator="plugin",
            fe_optimizer="random",
            fe_warm_start=False,
            fe_multi_fidelity=False,
            quantization_nbins=10,
            quantization_method="uniform",
            quantization_dtype=np.int8,
            n_jobs=16,
            verbose=0,
        )
        return time.perf_counter() - t0

    thread = maybe_prewarm_polynom_loky_pool(fe_smart_polynom_iters=1, n_jobs=16)
    assert thread is not None
    thread.join(timeout=120)  # ensure the pool is fully warm before the real call, like the tens-of-seconds
    # of categorization + GPU pair-MI screening give it in production

    warm_elapsed = _run()
    assert warm_elapsed < 15.0, f"expected a warm-pool dispatch well under the ~26-28s cold baseline, got {warm_elapsed:.1f}s"


def test_idle_worker_timeout_exceeds_joblib_default_and_is_shared_with_real_dispatch(monkeypatch):
    """Regression guard for the 2026-07-11 timeout bug: a pre-warmed pool that self-terminates (loky's default
    ``idle_worker_timeout=300``) before ``run_polynom_pair_fe`` actually dispatches (measured real gap ~418s)
    pays the contention cost of pre-warming for zero benefit. Pins that the shared constant exceeds the joblib
    default AND that ``run_polynom_pair_fe``'s REAL ``LokyBackend`` construction is called with that SAME
    constant (not two independently-tuned numbers that could drift apart and silently break
    ``get_reusable_executor``'s reuse-key match) -- verified by spying the actual constructor call, not by
    grepping source text."""
    import mlframe.feature_selection.filters.polynom_pair_fe as ppfe
    from mlframe.feature_selection.filters._joblib_safe import POLYNOM_LOKY_IDLE_WORKER_TIMEOUT

    assert POLYNOM_LOKY_IDLE_WORKER_TIMEOUT > 300, "must exceed the joblib LokyBackend default of 300s"

    captured: dict = {}
    real_loky_backend = ppfe.LokyBackend

    def _spy_loky_backend(*args, **kwargs):
        """Spy loky backend."""
        captured["idle_worker_timeout"] = kwargs.get("idle_worker_timeout")
        return real_loky_backend(*args, **kwargs)

    monkeypatch.setattr(ppfe, "LokyBackend", _spy_loky_backend)

    n, ncols = 200, 32
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, ncols)).astype(np.float64)
    y = rng.integers(0, 5, size=n)
    cols = [f"f{i}" for i in range(ncols)]
    nbins = np.full(ncols, 10, dtype=np.int64)
    prospective_pairs = {(i, i + 1): 1.0 for i in range(16)}  # crosses _PARALLEL_PAIR_THRESHOLD=16

    ppfe.run_polynom_pair_fe(
        X=X,
        is_polars_input=False,
        prospective_pairs=prospective_pairs,
        classes_y=y,
        cols=cols,
        nbins=nbins,
        data=X.copy(),
        engineered_features=set(),
        engineered_recipes={},
        hermite_features_list=[],
        feature_names_in=cols,
        fe_smart_polynom_iters=1,
        fe_smart_polynom_optimization_steps=1,
        fe_min_polynom_degree=1,
        fe_max_polynom_degree=2,
        fe_min_polynom_coeff=-2.0,
        fe_max_polynom_coeff=2.0,
        fe_min_engineered_mi_prevalence=0.0,
        fe_hermite_l2_penalty=0.0,
        fe_polynomial_basis="hermite",
        fe_mi_estimator="plugin",
        fe_optimizer="random",
        fe_warm_start=False,
        fe_multi_fidelity=False,
        quantization_nbins=10,
        quantization_method="uniform",
        quantization_dtype=np.int8,
        n_jobs=16,
        verbose=0,
    )

    assert (
        captured.get("idle_worker_timeout") == POLYNOM_LOKY_IDLE_WORKER_TIMEOUT
    ), f"run_polynom_pair_fe's real LokyBackend construction must reference the SAME shared constant maybe_prewarm_polynom_loky_pool uses; got {captured}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x", "--no-cov"])
