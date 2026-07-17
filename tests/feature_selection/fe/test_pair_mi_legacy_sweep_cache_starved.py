"""Regression test for MRMR audit finding #22 (2026-07-09).

Finding #22 flagged that ``fe_min_pair_mi``'s near-zero default (0.001 nats) barely prunes the O(p^2)
candidate space before ``compute_pairs_mis`` pays the expensive per-pair ``mi_direct`` permutation test.
That concern is now MOOT for the common case as a direct side effect of the finding-#21 fix (removing
``_MRMR_BATCH_PRECOMPUTE_MAX_K``): ``dispatch_batch_pair_mi_chunked`` unconditionally pre-fills
``cached_MIs`` for EVERY C(k,2) pair via a fast batched kernel before the legacy sweep ever runs, and
``compute_pairs_mis`` skips its expensive ``mi_direct`` call whenever the pair is already in
``cached_MIs`` (or ``cached_confident_MIs``) -- so the legacy per-pair permutation-test path, which is
what ``fe_min_pair_mi`` gates, is starved of work regardless of how weak the floor is.

This test pins that starvation mechanism directly: with ``cached_MIs`` pre-populated for every pair (as
the batch precompute now always does at n_pairs>=8), ``compute_pairs_mis`` must not invoke the
underlying ``mi_direct`` kernel at all -- proving the O(p^2) *expensive-compute* cost, not just the
O(p^2) *iteration* cost, is eliminated.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest

import mlframe.feature_selection.filters.feature_engineering as fe_mod
from mlframe.feature_selection.filters.feature_engineering import compute_pairs_mis


def _make_inputs(n=200, k=6, seed=0):
    """Make inputs."""
    rng = np.random.default_rng(seed)
    data = rng.integers(0, 4, size=(n, k + 1)).astype(np.int32)  # last col = target
    nbins = np.array([4] * (k + 1), dtype=np.int32)
    target_indices = (k,)
    classes_y = data[:, k].astype(np.int32)
    classes_y_safe = classes_y.copy()
    freqs_y = np.bincount(classes_y).astype(np.float64)
    return data, nbins, target_indices, classes_y, classes_y_safe, freqs_y


def test_legacy_sweep_skips_mi_direct_when_cache_fully_prepopulated(monkeypatch):
    """The exact scenario the batch precompute (finding #21 fix) now always produces: every singleton
    AND every pair MI already sits in ``cached_MIs`` before ``compute_pairs_mis`` runs."""
    data, nbins, target_indices, classes_y, classes_y_safe, freqs_y = _make_inputs(k=6)
    k = 6
    all_pairs = list(combinations(range(k), 2))

    cached_MIs = {(v,): 0.5 for v in range(k)}
    for p in all_pairs:
        cached_MIs[p] = 0.1
    cached_confident_MIs: dict = {}

    calls = []

    def _spy_mi_direct(*args, **kwargs):
        """Spy mi direct."""
        calls.append((args, kwargs))
        return 0.0, 1.0

    monkeypatch.setattr(fe_mod, "mi_direct", _spy_mi_direct)

    compute_pairs_mis(
        all_pairs=all_pairs,
        data=data,
        target_indices=target_indices,
        nbins=nbins,
        classes_y=classes_y,
        classes_y_safe=classes_y_safe,
        freqs_y=freqs_y,
        fe_min_nonzero_confidence=0.0,
        fe_npermutations=5,
        cached_confident_MIs=cached_confident_MIs,
        cached_MIs=cached_MIs,
        fe_min_pair_mi=0.001,  # the weak default finding #22 flagged -- irrelevant here, cache wins first
        fe_min_pair_mi_prevalence=1.05,
    )

    assert len(calls) == 0, f"expected 0 mi_direct calls (every pair pre-cached by the batch precompute), got {len(calls)}"


def test_legacy_sweep_only_computes_missing_pairs(monkeypatch):
    """Sanity check on the spy itself: with a PARTIALLY-populated cache (simulating a batch-precompute
    failure fallback for a few pairs), exactly the missing pairs get computed -- not more, not fewer."""
    data, nbins, target_indices, classes_y, classes_y_safe, freqs_y = _make_inputs(k=5, seed=1)
    k = 5
    all_pairs = list(combinations(range(k), 2))
    assert len(all_pairs) == 10

    cached_MIs = {(v,): 0.5 for v in range(k)}
    missing = {all_pairs[0], all_pairs[3], all_pairs[7]}
    for p in all_pairs:
        if p not in missing:
            cached_MIs[p] = 0.1
    cached_confident_MIs: dict = {}

    computed_pairs = []

    def _spy_mi_direct(data, x, y, **kwargs):
        """Spy mi direct."""
        computed_pairs.append(tuple(x))
        return 0.05, 1.0

    monkeypatch.setattr(fe_mod, "mi_direct", _spy_mi_direct)

    compute_pairs_mis(
        all_pairs=all_pairs,
        data=data,
        target_indices=target_indices,
        nbins=nbins,
        classes_y=classes_y,
        classes_y_safe=classes_y_safe,
        freqs_y=freqs_y,
        fe_min_nonzero_confidence=0.0,
        fe_npermutations=5,
        cached_confident_MIs=cached_confident_MIs,
        cached_MIs=cached_MIs,
        fe_min_pair_mi=0.001,
        fe_min_pair_mi_prevalence=1.05,
    )

    assert set(computed_pairs) == missing


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x", "--no-cov"])
