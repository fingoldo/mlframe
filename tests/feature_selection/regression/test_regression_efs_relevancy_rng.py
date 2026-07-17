"""Regression sensor for S28 (A1#2): estimate_features_relevancy used the global numpy legacy RNG (np.random.shuffle) which produced non-deterministic permuted baselines under parallel suite calls. Two suite calls running concurrently could interleave shuffle state.

Pre-fix code path: ``arr = bins.to_numpy(allow_copy=True).copy(); np.random.shuffle(arr[:, idx])``.
Post-fix code path: per-call ``np.random.default_rng(random_state)`` plus zero-copy ``to_numpy(allow_copy=True)``.

This sensor pins the new reproducibility contract: the same ``(bins, target_columns, random_state)`` MUST produce a bit-identical permuted MI dict across two calls even when the global RNG state differs between them.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from mlframe.feature_selection.general import estimate_features_relevancy
from mlframe.feature_selection.mi import grok_compute_mutual_information


def _toy_bins(n: int = 300, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    target = rng.integers(0, 8, size=n, dtype=np.int8)
    signal = (target + rng.integers(0, 2, size=n, dtype=np.int8)) % 8
    noise = rng.integers(0, 8, size=n, dtype=np.int8)
    arr = np.column_stack([target, signal, noise]).astype(np.int8)
    return pl.DataFrame(arr, schema=["target", "signal", "noise"])


def test_estimate_features_relevancy_reproducible_under_random_state():
    """Two calls with the same ``random_state`` must produce bit-identical ``all_permuted_mis`` even after disturbing the global RNG between them."""
    bins = _toy_bins(n=400, seed=42)

    # Disturb the legacy global RNG so a pre-fix run would see different shuffle sequences across the two calls.
    np.random.seed(0)
    np.random.bytes(64)
    cols1, mi1, perm1, _ = estimate_features_relevancy(
        bins=bins,
        target_columns=["target"],
        mi_algorithms_ranking=[grok_compute_mutual_information],
        benchmark_mi_algorithms=False,
        min_randomized_permutations=4,
        min_permuted_mi_evaluations=8,
        random_state=12345,
        verbose=0,
    )

    np.random.seed(999)
    np.random.bytes(1024)
    cols2, mi2, perm2, _ = estimate_features_relevancy(
        bins=bins,
        target_columns=["target"],
        mi_algorithms_ranking=[grok_compute_mutual_information],
        benchmark_mi_algorithms=False,
        min_randomized_permutations=4,
        min_permuted_mi_evaluations=8,
        random_state=12345,
        verbose=0,
    )

    assert cols1 == cols2, f"drop list must be deterministic under fixed random_state; got {cols1!r} vs {cols2!r}"
    np.testing.assert_allclose(mi1, mi2, rtol=0, atol=0)
    assert set(perm1) == set(perm2)
    for k in perm1:
        np.testing.assert_allclose(perm1[k], perm2[k], rtol=0, atol=0)


def test_estimate_features_relevancy_distinct_seeds_diverge():
    """Distinct ``random_state`` values must produce DIFFERENT permuted MI streams; sensor for "seed is actually consumed"."""
    bins = _toy_bins(n=400, seed=42)
    _, _, perm_a, _ = estimate_features_relevancy(
        bins=bins,
        target_columns=["target"],
        mi_algorithms_ranking=[grok_compute_mutual_information],
        benchmark_mi_algorithms=False,
        min_randomized_permutations=4,
        min_permuted_mi_evaluations=8,
        random_state=1,
        verbose=0,
    )
    _, _, perm_b, _ = estimate_features_relevancy(
        bins=bins,
        target_columns=["target"],
        mi_algorithms_ranking=[grok_compute_mutual_information],
        benchmark_mi_algorithms=False,
        min_randomized_permutations=4,
        min_permuted_mi_evaluations=8,
        random_state=2,
        verbose=0,
    )
    # At least one cell of the permuted MI matrix must differ between the two seeds.
    a = perm_a["target"]
    b = perm_b["target"]
    assert a.shape == b.shape
    assert not np.array_equal(a, b), "distinct random_state must yield distinct permuted MI samples"
