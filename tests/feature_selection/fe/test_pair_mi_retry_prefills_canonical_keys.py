"""The batched-CPU pair-MI retry must write canonical ``(low, high)`` keys (mrmr_audit_2026-09-14 FE_STEP-1).

When the loky pair-MI pool fails, ``compute_pair_mis_and_floor`` retries through the batched CPU dispatcher and prefills
the fit-persistent ``cached_MIs``. The primary prefill canonicalises every key with ``tuple(sorted(...))`` -- with a
comment explaining why -- but the retry path did not. The dispatcher takes pairs by POSITION in its ``ids`` array, and
that array comes from iterating ``numeric_vars_to_consider``, a set, whose order is not ascending once a large id
collides into a small id's hash slot. The retry then stores ``(40, 8)`` where the primary path stores ``(8, 40)``: one
logical pair under two keys, ranked twice, and a de-dup that misses it on the next FE step.

The pre-existing loky-failure test asserts only call counts and uses ``set(range(10))``, whose iteration IS ascending,
so it could never observe this. This fixture forces a non-ascending order and asserts that precondition explicitly.
"""

from __future__ import annotations

import numpy as np

import mlframe.feature_selection.filters._mrmr_fe_step._step_pairmi as step_pairmi_mod


def _non_ascending_var_set(big: int) -> set:
    """Column ids whose CPython set iteration is NOT ascending: ``big`` is inserted before the id it collides with."""
    ids = set(range(8))
    ids.add(big)
    ids.update([8, 9])
    return ids


def test_pair_mi_retry_prefills_canonical_keys(monkeypatch):
    """After a loky failure, every pair key the retry writes must be (low, high), with no pair stored both ways."""
    n, big = 500, 40
    k = big + 1  # feature columns 0..40 must exist for id 40 to be addressable
    rng = np.random.default_rng(0)
    data = rng.integers(0, 4, size=(n, k + 1)).astype(np.int32)
    nbins = np.array([4] * (k + 1), dtype=np.int32)
    target_indices = (k,)
    classes_y = data[:, k].astype(np.int32)
    freqs_y = np.bincount(classes_y).astype(np.float64)
    cols = [f"f{i}" for i in range(k)] + ["y"]

    numeric_vars = _non_ascending_var_set(big)
    order = list(numeric_vars)
    assert order != sorted(order), f"fixture precondition: set iteration must be non-ascending to reach the bug, got {order}"

    class _Fake:
        """The estimator attributes the pair-MI step reads."""

        fe_max_engineered_operands = -1
        fe_escalation_feedforward_enable = True
        _fe_synergy_exhaustive_active_ = False
        feature_names_in_ = [f"f{i}" for i in range(k)]

    call_log = {"primary": 0, "retry": 0}
    from mlframe.feature_selection.filters.batch_pair_mi_gpu import dispatch_batch_pair_mi_chunked as _real_dispatch

    def _spy_dispatch(**kwargs):
        """Fail the primary precompute so the loky branch is reached; let the post-loky retry run for real."""
        if call_log["primary"] == 0:
            call_log["primary"] += 1
            raise RuntimeError("simulated primary batch-precompute failure")
        call_log["retry"] += 1
        return _real_dispatch(**kwargs)

    def _boom_parallel(*args, **kwargs):
        """Simulate the loky pool failing to spawn."""
        raise RuntimeError("simulated loky pool spawn failure")

    monkeypatch.setattr("mlframe.feature_selection.filters.batch_pair_mi_gpu.dispatch_batch_pair_mi_chunked", _spy_dispatch)
    monkeypatch.setattr(step_pairmi_mod, "Parallel", _boom_parallel)

    cached_MIs: dict = {}
    step_pairmi_mod.compute_pair_mis_and_floor(
        _Fake(),
        data=data,
        cols=cols,
        nbins=nbins,
        X=None,
        classes_y=classes_y,
        classes_y_safe=classes_y,
        freqs_y=freqs_y,
        target_indices=target_indices,
        cached_MIs=cached_MIs,
        cached_confident_MIs={},
        numeric_vars_to_consider=numeric_vars,
        _prevalence_debias_auto=False,
        n_jobs=16,
        prefetch_factor=2,
        parallel_kwargs={"backend": "threading"},
        fe_min_nonzero_confidence=0.99,
        fe_npermutations=25,  # >= the loky-pool floor, so the failure/retry branch is actually reached
        fe_min_pair_mi=0.001,
        fe_min_pair_mi_prevalence=1.05,
        verbose=0,
    )

    assert call_log["retry"] == 1, "the batched-CPU retry must have run, or this test observes nothing"
    pair_keys = [key for key in cached_MIs if isinstance(key, tuple) and len(key) == 2]
    assert pair_keys, "the retry prefilled no pair keys; the assertions below would be vacuous"
    non_canonical = [key for key in pair_keys if key[0] > key[1]]
    assert not non_canonical, f"retry wrote non-canonical pair keys: {non_canonical[:5]}"
    stored_both_ways = [key for key in pair_keys if key[0] != key[1] and (key[1], key[0]) in cached_MIs]
    assert not stored_both_ways, f"a pair is cached under both orientations: {stored_both_ways[:5]}"
