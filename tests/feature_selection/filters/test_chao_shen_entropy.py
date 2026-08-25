"""Coverage for filters._chao_shen (Chao-Shen entropy/MI estimator), previously untested."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._chao_shen import (
    chao_shen_entropy,
    chao_shen_entropy_from_counts,
    chao_shen_mi,
)

pytestmark = pytest.mark.fast


def test_chao_shen_entropy_from_counts_uniform_close_to_log_k():
    """A large, well-sampled uniform categorical's Chao-Shen entropy should be close to ln(K) (no singleton bias)."""
    rng = np.random.default_rng(0)
    K = 8
    x = rng.integers(0, K, size=20_000)
    counts = np.bincount(x, minlength=K).astype(np.int64)
    h = chao_shen_entropy_from_counts(counts)
    assert h == pytest.approx(np.log(K), abs=0.02)


def test_chao_shen_entropy_from_counts_degenerate_single_category():
    """A single-category (zero-entropy) sample yields exactly 0."""
    counts = np.array([100], dtype=np.int64)
    assert chao_shen_entropy_from_counts(counts) == 0.0


def test_chao_shen_entropy_from_counts_all_singletons_falls_back_to_plugin():
    """All-singleton counts (coverage estimate collapses to 0) fall back to the plug-in entropy, not NaN/crash."""
    counts = np.ones(10, dtype=np.int64)
    h = chao_shen_entropy_from_counts(counts)
    assert np.isfinite(h)
    assert h >= 0.0
    # Plug-in entropy on 10 uniform singletons is exactly ln(10).
    assert h == pytest.approx(np.log(10), abs=1e-9)


def test_chao_shen_entropy_from_counts_empty_or_zero_returns_zero():
    """All-zero counts (N=0) return 0.0 rather than raising."""
    assert chao_shen_entropy_from_counts(np.zeros(5, dtype=np.int64)) == 0.0


def test_chao_shen_entropy_from_counts_shared_coverage_basis():
    """Passing an explicit external coverage overrides the internally-estimated one."""
    counts = np.array([5, 5, 5, 5], dtype=np.int64)
    h_default = chao_shen_entropy_from_counts(counts)
    h_forced = chao_shen_entropy_from_counts(counts, coverage=0.5)
    assert h_default != pytest.approx(h_forced)
    assert np.isfinite(h_forced)


def test_chao_shen_entropy_wrapper_int_input():
    """chao_shen_entropy bincounts integer input and matches the from_counts primitive directly."""
    rng = np.random.default_rng(1)
    x = rng.integers(0, 5, size=500).astype(np.int64)
    counts = np.bincount(x).astype(np.int64)
    assert chao_shen_entropy(x) == pytest.approx(chao_shen_entropy_from_counts(counts))


def test_chao_shen_entropy_wrapper_float_input_quantile_binned():
    """chao_shen_entropy quantile-bins float input before scoring; finite, non-negative result."""
    rng = np.random.default_rng(2)
    x = rng.standard_normal(500)
    h = chao_shen_entropy(x)
    assert np.isfinite(h)
    assert h >= 0.0


def test_chao_shen_entropy_wrapper_empty_input():
    """Empty array returns 0.0."""
    assert chao_shen_entropy(np.array([])) == 0.0


def test_chao_shen_mi_independent_variables_near_zero():
    """MI between two independent uniform categoricals should be small (near-zero, positive-biased)."""
    rng = np.random.default_rng(3)
    n = 5000
    x = rng.integers(0, 4, size=n).astype(np.int64)
    y = rng.integers(0, 4, size=n).astype(np.int64)
    mi = chao_shen_mi(x, y)
    assert mi >= 0.0
    assert mi < 0.05  # near-zero for genuinely independent variables at this sample size


def test_chao_shen_mi_deterministic_relationship_high():
    """MI between x and a deterministic function of x (y = x) should be close to ln(K)."""
    rng = np.random.default_rng(4)
    K = 4
    x = rng.integers(0, K, size=5000).astype(np.int64)
    y = x.copy()
    mi = chao_shen_mi(x, y)
    assert mi == pytest.approx(np.log(K), abs=0.05)


def test_chao_shen_mi_negative_codes_dropped_not_crashed():
    """Negative sentinel codes (e.g. NaN markers) in x are dropped rather than crashing bincount."""
    x = np.array([0, 1, -1, 2, 1, -1, 0], dtype=np.int64)
    y = np.array([0, 1, 0, 2, 1, 1, 0], dtype=np.int64)
    mi = chao_shen_mi(x, y)
    assert np.isfinite(mi)
    assert mi >= 0.0


def test_chao_shen_mi_empty_input():
    """Empty x returns 0.0."""
    assert chao_shen_mi(np.array([], dtype=np.int64), np.array([], dtype=np.int64)) == 0.0
