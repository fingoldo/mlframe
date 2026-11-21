"""AUC entry points validate their inputs and the bootstrap resampler is safe under threads (audit 2026-11-20, findings 25-F8..F11)."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from mlframe.metrics._auc_labels import _binary_labels_01
from mlframe.metrics._auc_per_group import fast_aucs_per_group_optimized
from mlframe.metrics._core_auc_brier import bootstrap_auc_distribution_parallel, make_bootstrap_auc_resampler


def _data(n=200, seed=0):
    """Binary labels and noisy scores correlated with them."""
    rng = np.random.default_rng(seed)
    y = (rng.uniform(size=n) > 0.5).astype(np.float64)
    s = rng.standard_normal(n) + y
    return y, s


def test_bootstrap_distribution_returns_nan_for_nan_scores():
    """NaN scores used to rank as the highest score and gave a finite wrong distribution (25-F8)."""
    y, s = _data()
    s[3] = np.nan
    out = bootstrap_auc_distribution_parallel(y, s, n_bootstrap=5, random_state=0)
    assert out is not None and np.isnan(out).all()


def test_resampler_returns_nan_for_nan_scores():
    """The factory's resampler gives NaN for NaN scores instead of a finite value (25-F8)."""
    y, s = _data()
    s[3] = np.nan
    resampler = make_bootstrap_auc_resampler(y, s)
    assert np.isnan(resampler(np.arange(len(y))))


def test_resampler_length_mismatch_raises():
    """Mismatched lengths must raise, not reach the unchecked kernel (25-F8)."""
    y, s = _data()
    with pytest.raises(ValueError):
        make_bootstrap_auc_resampler(y[:-1], s)


def test_resampler_is_correct_when_called_from_many_threads():
    """Shared scratch buffers used to be written by concurrent resamples; each thread now has its own (25-F9)."""
    y, s = _data(n=400, seed=2)
    resampler = make_bootstrap_auc_resampler(y, s)
    rng = np.random.default_rng(5)
    idx_sets = [rng.integers(0, len(y), size=len(y)) for _ in range(200)]
    serial = [resampler(idx) for idx in idx_sets]
    with ThreadPoolExecutor(max_workers=8) as pool:
        threaded = list(pool.map(resampler, idx_sets))
    np.testing.assert_allclose(threaded, serial)


def test_per_group_auc_rejects_length_mismatch():
    """fast_aucs_per_group_optimized validates length like fast_aucs (25-F10)."""
    y, s = _data()
    g = np.zeros(len(y) - 1, dtype=np.int64)
    with pytest.raises(ValueError):
        fast_aucs_per_group_optimized(y[:-1], s, g)


def test_multiclass_labels_are_refused_for_binary_auc():
    """Three distinct labels used to reduce silently to class 1 vs rest (25-F11)."""
    with pytest.raises(ValueError):
        _binary_labels_01(np.array([0, 1, 2, 1, 0]))
