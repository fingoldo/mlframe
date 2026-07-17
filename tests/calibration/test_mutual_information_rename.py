"""Tests for the mutual_information_score rename + deprecated hyvarinen_score alias (P1-2)."""

from __future__ import annotations

import warnings

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")


def _data(seed: int = 0):
    """Helper that data."""
    rng = np.random.default_rng(seed)
    y_pred = rng.random(500)
    y_true = (rng.random(500) < y_pred).astype(np.float64)
    return y_true, y_pred


@pytest.mark.fast
def test_mutual_information_score_present_and_returns_float():
    """Mutual information score present and returns float."""
    from mlframe.calibration.quality import mutual_information_score

    y_true, y_pred = _data()
    val = mutual_information_score(y_true, y_pred)
    assert np.isscalar(val) or np.ndim(val) == 0
    assert np.isfinite(float(val))
    assert float(val) >= 0.0  # MI is non-negative


@pytest.mark.fast
def test_hyvarinen_score_alias_warns_deprecation():
    """Hyvarinen score alias warns deprecation."""
    from mlframe.calibration.quality import hyvarinen_score

    y_true, y_pred = _data()
    with pytest.warns(DeprecationWarning):
        hyvarinen_score(y_true, y_pred)


@pytest.mark.fast
def test_alias_delegates_identically():
    """Alias delegates identically."""
    from mlframe.calibration.quality import hyvarinen_score, mutual_information_score

    y_true, y_pred = _data(seed=7)
    new_val = mutual_information_score(y_true, y_pred)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        old_val = hyvarinen_score(y_true, y_pred)
    assert float(new_val) == float(old_val)
