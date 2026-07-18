"""Regression test for MRMR audit follow-on fix (2026-07-10): when the loky pair-MI pool fails or times
out, the fallback must retry via the FAST batched CPU dispatcher before ever reaching the slow legacy
per-pair sweep.

Reproduced live at n=3M rows / p=423 columns (n_pairs=89,253): the loky pool timed out after 300s, and
the pre-fix code fell straight to ``compute_pairs_mis``'s serial per-pair ``mi_direct`` permutation-test
loop -- a single CPU core grinding through up to 89,253 pairs, each potentially costing a meaningful
fraction of a second at 3M rows. Caught live via a real training run: 2+ hours elapsed with near-zero
aggregate CPU/GPU utilization, reproducing the exact "10h44m fit, weak CPU/GPU utilization" pathology
the whole MRMR audit started from.

This test forces the loky dispatch to fail and asserts the batched CPU retry is what actually populates
``cached_MIs`` -- not the legacy per-pair sweep (which would still technically "work" but is the thing
being fixed away from).

``fe_npermutations=25`` (not the module's own default): the 2026-07-19 joblib-audit fix added
``_LOKY_POOL_MIN_FE_NPERMUTATIONS=20`` (measured: the loky pool never wins at the realistic
``fe_npermutations=3`` production budget -- 0.03-0.38x across n_pairs=190..20000 -- so calls below the floor
now skip the pool entirely and never reach the failure/retry path this test exercises). This test is
specifically about loky-failure recovery, which requires actually reaching the pool branch first.
"""

from __future__ import annotations


import numpy as np
import pytest

import mlframe.feature_selection.filters._mrmr_fe_step._step_pairmi as step_pairmi_mod


def test_loky_failure_retries_batched_cpu_before_legacy_sweep(monkeypatch, caplog):
    """A loky pool failure must be recovered via ``dispatch_batch_pair_mi_chunked(force_backend=
    'njit_parallel')``, not the slow legacy per-pair sweep, whenever the batched retry can cover the pool."""
    import logging

    n = 500
    k = 10  # C(10,2) = 45 pairs, comfortably above _MRMR_BATCH_PRECOMPUTE_MIN_PAIRS
    rng = np.random.default_rng(0)
    data = rng.integers(0, 4, size=(n, k + 1)).astype(np.int32)
    nbins = np.array([4] * (k + 1), dtype=np.int32)
    target_indices = (k,)
    classes_y = data[:, k].astype(np.int32)
    freqs_y = np.bincount(classes_y).astype(np.float64)
    numeric_vars_to_consider = set(range(k))
    cols = [f"f{i}" for i in range(k)] + ["y"]

    # Force the PRIMARY batch precompute to no-op (simulate it having already failed/been skipped),
    # so the pool actually reaches the loky-dispatch branch with an empty cache -- matching the
    # production scenario where the loky pool is reached BECAUSE the primary precompute didn't cover
    # the full pool.
    class _Fake:
        """Groups tests covering Fake."""
        fe_max_engineered_operands = -1
        fe_escalation_feedforward_enable = True
        _fe_synergy_exhaustive_active_ = False
        feature_names_in_ = [f"f{i}" for i in range(k)]

    call_log = {"batch_precompute_calls": 0, "retry_calls": 0}

    from mlframe.feature_selection.filters.batch_pair_mi_gpu import dispatch_batch_pair_mi_chunked as _real_dispatch

    def _spy_dispatch_batch_pair_mi_chunked(**kwargs):
        # First call = the primary precompute (force it to fail so the loky branch is reached).
        # Second call = the retry-after-loky-failure this test targets (must succeed).
        """Spy dispatch batch pair mi chunked."""
        if call_log["batch_precompute_calls"] == 0:
            call_log["batch_precompute_calls"] += 1
            raise RuntimeError("simulated primary batch-precompute failure")
        call_log["retry_calls"] += 1
        # Delegate to the REAL (pre-patch) implementation for the retry so the test exercises real
        # behavior -- captured BEFORE patching to avoid recursing into this same spy.
        return _real_dispatch(**kwargs)

    monkeypatch.setattr(
        "mlframe.feature_selection.filters.batch_pair_mi_gpu.dispatch_batch_pair_mi_chunked",
        _spy_dispatch_batch_pair_mi_chunked,
    )

    # Force the loky pool itself to fail immediately (simulating the 300s timeout / pool-spawn hang):
    # patch the module-level ``Parallel`` that ``_run_loky_pair_mi_pool`` calls.
    def _boom_parallel(*a, **kw):
        """Boom parallel."""
        raise RuntimeError("simulated loky pool spawn failure")

    monkeypatch.setattr(step_pairmi_mod, "Parallel", _boom_parallel)

    with caplog.at_level(logging.WARNING):
        result = step_pairmi_mod.compute_pair_mis_and_floor(
            _Fake(),
            data=data,
            cols=cols,
            nbins=nbins,
            X=None,
            classes_y=classes_y,
            classes_y_safe=classes_y,
            freqs_y=freqs_y,
            target_indices=target_indices,
            cached_MIs={},
            cached_confident_MIs={},
            numeric_vars_to_consider=numeric_vars_to_consider,
            _prevalence_debias_auto=False,
            n_jobs=16,
            prefetch_factor=2,
            parallel_kwargs={"backend": "threading"},
            fe_min_nonzero_confidence=0.99,
            fe_npermutations=25,
            fe_min_pair_mi=0.001,
            fe_min_pair_mi_prevalence=1.05,
            verbose=0,
        )

    assert call_log["batch_precompute_calls"] == 1, "primary batch precompute must have been attempted (and forced to fail)"
    assert call_log["retry_calls"] == 1, "the batched-CPU retry must fire after the loky pool fails"
    assert any(
        "batched CPU retry covered" in rec.message for rec in caplog.records
    ), "the retry's coverage must be logged so a real production run is diagnosable"
    assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x", "--no-cov"])
