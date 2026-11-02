"""The RFECV tolerance band is the sample standard error of the fold scores (tracker 2026-11-02 wave 12, STAT12-1, STAT12-2)."""

import numpy as np
import pytest

from mlframe.feature_selection.wrappers.rfecv._one_se_band import band_half_width


def test_se_band_equals_ddof1_std_over_sqrt_k():
    """Folds [0.80 ... 0.88], k=5: the population std 0.02828 over sqrt(5) gave 0.01265; the standard error is 0.01414."""
    folds = np.array([0.80, 0.82, 0.84, 0.86, 0.88])
    got = band_half_width(np.array([np.std(folds)]), np.array([5]), "se")[0]
    assert got == pytest.approx(np.std(folds, ddof=1) / np.sqrt(5))
    assert got == pytest.approx(0.0141421356)


def test_foldstd_band_is_unchanged():
    """The legacy band stays the raw across-fold std."""
    assert band_half_width(np.array([0.03]), np.array([5]), "foldstd")[0] == pytest.approx(0.03)
