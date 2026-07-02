"""Edge-case coverage for ``mlframe.calibration.policy._ece_score``.

Covers empty / all-NaN / mixed-NaN probabilities, boundary probabilities (0/1),
single-class labels (raises), rare-1pct imbalance, float32 input, and the 2D
probability-matrix dispatch (positive-class column selection).
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("numba")

from mlframe.calibration.policy import _ece_score


def test_ece_empty_returns_nan():
    assert np.isnan(_ece_score(np.array([]), np.array([])))


def test_ece_all_nan_probs_returns_nan():
    # No finite (p, y) rows -> n_finite == 0 -> NaN (not a crash, not a spurious 0).
    y = np.array([0, 1, 0, 1])
    p = np.full(4, np.nan)
    assert np.isnan(_ece_score(y, p))


def test_ece_mixed_nan_scores_finite_rows_only():
    # Non-finite rows are skipped inside the kernel; the ECE is computed from the survivors.
    y = np.array([0, 1, 0, 1, 0, 1])
    p = np.array([0.1, 0.9, np.nan, 0.8, 0.2, np.nan])
    res = _ece_score(y, p)
    assert np.isfinite(res)
    assert 0.0 <= res <= 1.0


def test_ece_boundary_probs_perfectly_calibrated_is_zero():
    # p in {0, 1} exactly matching y -> perfect calibration -> ECE 0.
    y = np.array([0, 1, 0, 1])
    p = np.array([0.0, 1.0, 0.0, 1.0])
    assert _ece_score(y, p) == pytest.approx(0.0, abs=1e-12)


def test_ece_single_class_labels_raise():
    # The per-bin accuracy mean(y in bin) is only valid for a 2-class label set; a single
    # distinct finite label has no valid binary encoding -> explicit ValueError.
    with pytest.raises(ValueError, match="2 distinct"):
        _ece_score(np.zeros(4), np.array([0.1, 0.2, 0.3, 0.4]))


def test_ece_non_01_labels_are_remapped():
    # {1, 2} labels are remapped (larger->1) rather than treated as raw magnitudes.
    y12 = np.array([1, 2, 1, 2])
    y01 = np.array([0, 1, 0, 1])
    p = np.array([0.2, 0.8, 0.3, 0.7])
    assert _ece_score(y12, p) == pytest.approx(_ece_score(y01, p))


def test_ece_rare_1pct_imbalance_finite():
    # 1 positive in 100 rows: still >=2 distinct finite labels -> a finite ECE.
    y = np.zeros(100, dtype=int)
    y[0] = 1
    p = np.full(100, 0.01)
    p[0] = 0.9
    res = _ece_score(y, p)
    assert np.isfinite(res)
    assert 0.0 <= res <= 1.0


def test_ece_float32_input_finite():
    # float32 falls out of the float64 fast path into the coercion branch; must still work.
    rng = np.random.default_rng(0)
    y = (rng.random(200) > 0.5).astype(np.int64)
    p = rng.random(200).astype(np.float32)
    res = _ece_score(y, p)
    assert np.isfinite(res)
    assert 0.0 <= res <= 1.0


def test_ece_2d_prob_matrix_uses_positive_column():
    # A 2D (n, 2) matrix must be reduced to its positive-class column p[:, 1], matching the
    # 1D call on that column exactly.
    rng = np.random.default_rng(1)
    y = (rng.random(150) > 0.5).astype(np.int64)
    pos = rng.random(150)
    p2d = np.column_stack([1.0 - pos, pos])
    assert _ece_score(y, p2d) == pytest.approx(_ece_score(y, pos))
