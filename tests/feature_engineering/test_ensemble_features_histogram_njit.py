"""Regression sensors for the per-row bin-histogram fast path in ensemble disagreement features.

``predictor_consensus_entropy`` / ``predictor_top2_mode_gap`` replaced a per-predictor ``np.add.at`` scatter loop
with the fused ``_row_bin_histogram_njit`` prange kernel (~27x on the scatter at 10M rows, bit-identical integer counts).
"""
from __future__ import annotations

import numpy as np

from mlframe.feature_engineering import ensemble_features as ef


def _reference_consensus_entropy(arr: np.ndarray, n_bins: int = 5) -> np.ndarray:
    """The exact pre-fix np.add.at implementation, kept here as an independent oracle."""
    n, k = arr.shape
    lo = arr.min(axis=1, keepdims=True) - 1e-9
    hi = arr.max(axis=1, keepdims=True) + 1e-9
    span = (hi - lo) + 1e-12
    binned = np.clip(((arr - lo) / span * n_bins).astype(np.int32), 0, n_bins - 1)
    counts = np.zeros((n, n_bins), dtype=np.float64)
    for j in range(k):
        np.add.at(counts, (np.arange(n), binned[:, j]), 1.0)
    probs = counts / counts.sum(axis=1, keepdims=True)
    return -np.sum(probs * np.log(probs + 1e-12), axis=1)


def test_row_bin_histogram_njit_matches_add_at_scatter():
    rng = np.random.default_rng(7)
    binned = rng.integers(0, 5, size=(2000, 9)).astype(np.int32)
    n, k = binned.shape
    expected = np.zeros((n, 5), dtype=np.float64)
    for j in range(k):
        np.add.at(expected, (np.arange(n), binned[:, j]), 1.0)
    got = ef._row_bin_histogram_njit(np.ascontiguousarray(binned), 5)
    assert np.array_equal(got, expected)


def test_consensus_entropy_bit_identical_to_reference():
    rng = np.random.default_rng(11)
    arr = rng.standard_normal((3000, 8))
    assert np.array_equal(ef.predictor_consensus_entropy(arr), _reference_consensus_entropy(arr))


def test_top2_mode_gap_bit_identical_to_reference():
    rng = np.random.default_rng(13)
    arr = rng.standard_normal((3000, 8))
    n, k, n_bins = arr.shape[0], arr.shape[1], 5
    lo = arr.min(axis=1, keepdims=True) - 1e-9
    hi = arr.max(axis=1, keepdims=True) + 1e-9
    span = (hi - lo) + 1e-12
    binned = np.clip(((arr - lo) / span * n_bins).astype(np.int32), 0, n_bins - 1)
    counts = np.zeros((n, n_bins), dtype=np.float64)
    for j in range(k):
        np.add.at(counts, (np.arange(n), binned[:, j]), 1.0)
    sc = -np.sort(-counts, axis=1)
    expected = (sc[:, 0] - sc[:, 1]) / float(k)
    assert np.array_equal(ef.predictor_top2_mode_gap(arr), expected)


def test_functions_no_longer_use_add_at_scatter(monkeypatch):
    """Behavioral pin: make ``np.add.at`` explode, then confirm both functions still run.

    The pre-fix scatter loop called ``np.add.at`` per predictor; on post-fix code the fused njit histogram replaces it,
    so the sabotaged ufunc is never touched. A revert to the scatter loop trips the sabotage and fails this sensor.
    """
    def _boom(*_a, **_k):
        raise AssertionError("np.add.at must not be used on the histogram fast path")

    monkeypatch.setattr(np.add, "at", _boom)
    rng = np.random.default_rng(17)
    arr = rng.standard_normal((500, 6))
    ef.predictor_consensus_entropy(arr)
    ef.predictor_top2_mode_gap(arr)
