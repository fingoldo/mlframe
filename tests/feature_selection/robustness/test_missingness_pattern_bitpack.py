"""Regression sensor for the fused njit bit-pack signature + vectorised label
lookup in ``missingness_pattern_fit`` / ``apply_missingness_pattern`` (perf iter122).

The row-pattern signature must stay bit-identical to the numpy reference
``(arr.astype(int64) * (1 << arange(k))).sum(axis=1)`` while avoiding the two
(n, k) int64 broadcast temporaries; ``apply`` must label rows identically to
``fit`` without a per-row Python loop. A spy pins that the signature routes
through the njit kernel -- a revert to the broadcast-numpy form trips it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters import _missingness_fe as mf


def _numpy_reference_signature(arr: np.ndarray) -> np.ndarray:
    _n, k = arr.shape
    weights = 1 << np.arange(k, dtype=np.int64)
    return (arr.astype(np.int64) * weights[None, :]).sum(axis=1)


def _frame(n: int = 6000, k: int = 9, seed: int = 11) -> tuple[pd.DataFrame, list[str]]:
    rng = np.random.default_rng(seed)
    data = {f"c{j}": np.where(rng.random(n) < 0.3, np.nan, rng.random(n)) for j in range(k)}
    X = pd.DataFrame(data)
    return X, list(X.columns)


def test_signature_bit_identical_to_numpy_reference():
    X, _ = _frame()
    block = X.isna().to_numpy()
    assert np.array_equal(mf._row_pattern_signature(block), _numpy_reference_signature(block))


def test_signature_routes_through_njit_kernel(monkeypatch):
    X, _ = _frame()
    block = X.isna().to_numpy()
    called = {"n": 0}
    real = mf._bitpack_rows_njit

    def spy(arr):
        called["n"] += 1
        return real(arr)

    monkeypatch.setattr(mf, "_bitpack_rows_njit", spy)
    mf._row_pattern_signature(block)
    assert called["n"] == 1, "k<=63 signature must route through the njit bit-pack kernel"


def test_fit_apply_labels_identical_and_vectorised():
    X, cols = _frame()
    labels, recipe = mf.missingness_pattern_fit(X, cols, top_k=5)
    applied = mf.apply_missingness_pattern(X, recipe)
    assert np.array_equal(labels, applied)
    # Both top-k buckets and the "other" sink must be populated for the test to be meaningful.
    assert len(np.unique(labels)) >= 2


def test_apply_unseen_pattern_maps_to_other():
    X, cols = _frame()
    _, recipe = mf.missingness_pattern_fit(X, cols, top_k=2)
    applied = mf.apply_missingness_pattern(X, recipe)
    other = int(recipe["other_label"])
    # With only top_k=2 retained, the rare patterns must collapse to the other sink.
    assert (applied == other).any()
    assert set(np.unique(applied)).issubset(set(recipe["pattern_to_label"].values()) | {other})
