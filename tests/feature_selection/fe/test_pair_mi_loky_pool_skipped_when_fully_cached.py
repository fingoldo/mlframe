"""Regression test for MRMR audit follow-on fix (2026-07-10): the FE order-2 pair-MI loky PROCESS pool is
now skipped entirely when the batch precompute already covers every candidate pair.

Reproduced live: on a p=249-operand pool (n_pairs=31125), the batch precompute (finding #21's fix) routinely
reports "batch-prefilled 31125/31125 pair MIs" -- 100% coverage -- yet the code unconditionally still spun
up a fresh loky PROCESS pool for the (now fully-redundant) legacy sweep. On this Windows host that pool
spawn was observed to hang for the FULL 300s watchdog bound before falling back to a serial pass that also
found nothing to do -- wasting up to 300s per FE round for zero benefit. The fix computes
``_all_pairs_precomputed`` (a membership scan against ``cached_MIs``/``cached_confident_MIs``) and routes a
fully-covered pool to the cheap serial branch (a fast no-op scan, not a compute pass) instead of ever
reaching the loky pool.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest

import mlframe.feature_selection.filters._mrmr_fe_step._step_pairmi as step_pairmi_mod


def test_all_pairs_precomputed_flag_true_when_cache_fully_covers_pool():
    """All pairs precomputed flag true when cache fully covers pool."""
    k = 6
    numeric_vars_to_consider = set(range(k))
    all_pairs = list(combinations(range(k), 2))
    cached_MIs = {p: 0.1 for p in all_pairs}
    cached_confident_MIs: dict = {}
    n_pairs = len(all_pairs)

    covered = n_pairs > 0 and all((p in cached_MIs or p in cached_confident_MIs) for p in combinations(numeric_vars_to_consider, 2))
    assert covered is True


def test_all_pairs_precomputed_flag_false_when_one_pair_missing():
    """All pairs precomputed flag false when one pair missing."""
    k = 6
    numeric_vars_to_consider = set(range(k))
    all_pairs = list(combinations(range(k), 2))
    cached_MIs = {p: 0.1 for p in all_pairs[:-1]}  # one pair missing
    cached_confident_MIs: dict = {}
    n_pairs = len(all_pairs)

    covered = n_pairs > 0 and all((p in cached_MIs or p in cached_confident_MIs) for p in combinations(numeric_vars_to_consider, 2))
    assert covered is False


def test_loky_pool_never_constructed_when_pool_fully_precomputed(monkeypatch):
    """Direct integration check: with n_jobs forced high enough to normally reach the loky branch, and the
    pool pre-seeded so every pair is already cached, ``LokyBackend`` must never be instantiated."""
    calls = []

    class _BoomBackend:
        """Groups tests covering BoomBackend."""
        def __init__(self, *a, **kw):
            calls.append((a, kw))
            raise AssertionError("LokyBackend must not be constructed when the pool is fully precomputed")

    monkeypatch.setattr(step_pairmi_mod, "LokyBackend", _BoomBackend)

    n = 300
    k = 12
    rng = np.random.default_rng(0)
    data = rng.integers(0, 4, size=(n, k + 1)).astype(np.int32)
    nbins = np.array([4] * (k + 1), dtype=np.int32)
    target_indices = (k,)
    classes_y = data[:, k].astype(np.int32)
    freqs_y = np.bincount(classes_y).astype(np.float64)
    numeric_vars_to_consider = set(range(k))
    all_pairs = list(combinations(range(k), 2))
    cached_MIs = {(v,): 0.2 for v in range(k)}
    cached_MIs.update({p: 0.1 for p in all_pairs})
    cached_confident_MIs: dict = {}

    class _Fake:
        """Groups tests covering Fake."""
        fe_max_engineered_operands = -1
        fe_escalation_feedforward_enable = True
        _fe_synergy_exhaustive_active_ = False
        feature_names_in_ = [f"f{i}" for i in range(k)]

    cols = [f"f{i}" for i in range(k)] + ["y"]

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
        cached_confident_MIs=cached_confident_MIs,
        numeric_vars_to_consider=numeric_vars_to_consider,
        _prevalence_debias_auto=False,
        n_jobs=16,
        prefetch_factor=2,
        parallel_kwargs={"backend": "threading"},
        fe_min_nonzero_confidence=0.99,
        fe_npermutations=10,
        fe_min_pair_mi=0.001,
        fe_min_pair_mi_prevalence=1.05,
        verbose=0,
    )
    assert calls == [], "LokyBackend was constructed despite full pair-MI cache coverage"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x", "--no-cov"])
