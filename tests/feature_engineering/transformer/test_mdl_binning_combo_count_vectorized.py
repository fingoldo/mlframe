"""Regression test for the MDL-binning pairwise combo-count vectorization.

``compute_mdl_binning_pairwise_features._process`` counts, per query row, how many train rows
share the same (feature-0, feature-1) bin combo. The OLD code built a ``collections.Counter`` over
the train combo codes and looked each query row up with a per-row Python ``counts.get(int(c), 0)``
list comprehension (plus ``len(set(query_combo))`` for the unique-combo feature). The optimization
replaces both with ``np.unique(return_counts=True)`` + ``np.searchsorted``.

Counts are integers, so the vectorised result is BIT-IDENTICAL to the dict path. This test pins:

1. The vectorised lookup equals the ``Counter.get`` reference for random combo arrays, including
   query combos absent from train (count 0) and the unique-combo scalar.
2. The full feature function produces finite, correctly-shaped, deterministic output.

bench: src/mlframe/feature_engineering/_benchmarks/bench_mdl_binning_combo_count_vectorized.py
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest


def _old_combo(train_combo: np.ndarray, query_combo: np.ndarray):
    """Helper: Old combo."""
    combo_counts = Counter(train_combo)
    out = np.array([combo_counts.get(int(c), 0) for c in query_combo], dtype=np.float32)
    return out, float(len(set(query_combo)))


def _new_combo(train_combo: np.ndarray, query_combo: np.ndarray):
    """Helper: New combo."""
    uniq_combo, uniq_counts = np.unique(train_combo, return_counts=True)
    pos = np.searchsorted(uniq_combo, query_combo)
    pos_clipped = np.clip(pos, 0, uniq_combo.shape[0] - 1)
    matched = uniq_combo[pos_clipped] == query_combo
    out = np.where(matched, uniq_counts[pos_clipped], 0).astype(np.float32)
    return out, float(np.unique(query_combo).shape[0])


def test_vectorized_combo_count_bit_identical_to_counter():
    """Vectorized combo count bit identical to counter."""
    rng = np.random.default_rng(42)
    for _ in range(50):
        n_train = int(rng.integers(100, 3000))
        n_query = int(rng.integers(100, 3000))
        hi = int(rng.integers(2, 12))
        # Intentionally let some query combos be absent from train (count 0).
        train_combo = rng.integers(0, hi, n_train) * 100 + rng.integers(0, hi, n_train)
        query_combo = rng.integers(0, hi + 2, n_query) * 100 + rng.integers(0, hi + 2, n_query)
        out_old, uc_old = _old_combo(train_combo, query_combo)
        out_new, uc_new = _new_combo(train_combo, query_combo)
        assert np.array_equal(out_old, out_new)
        assert uc_old == uc_new


def test_full_feature_function_runs_and_deterministic():
    """Full feature function runs and deterministic."""
    pytest.importorskip("sklearn")
    from sklearn.model_selection import KFold

    from mlframe.feature_engineering.transformer.mdl_binning_pairwise import (
        compute_mdl_binning_pairwise_features,
    )

    rng = np.random.default_rng(7)
    n, d = 600, 6
    X = rng.standard_normal((n, d)).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1] + 0.3 * rng.standard_normal(n)).astype(np.float32)
    splitter = KFold(n_splits=3, shuffle=True, random_state=0)

    df1 = compute_mdl_binning_pairwise_features(X, y, None, splitter, seed=0, task="regression")
    df2 = compute_mdl_binning_pairwise_features(X, y, None, splitter, seed=0, task="regression")
    assert df1.shape == (n, 5)
    arr = df1.to_numpy()
    assert np.isfinite(arr).all()
    assert np.allclose(arr, df2.to_numpy())
