"""Tests for DeLong AUC confidence intervals (charts/calibration.py + multiclass ROC).

Covers: DeLong AUC matches sklearn roc_auc_score exactly (incl. ties via midranks),
the CI brackets the point estimate and clips to [0,1], empty-class -> NaN, the
multiclass ROC panel legend carries the [lo, hi] CI when enabled, and biz_value --
the CI brackets a known-AUC synthetic and NARROWS as n grows (the 1/sqrt(n) law).
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from mlframe.reporting.charts.calibration import (
    _midrank,
    delong_auc_ci,
    delong_auc_variance,
)
from mlframe.reporting.charts.multiclass import _roc_panel


def _binary(n=5000, sep=1.5, seed=0):
    """Helper: Binary."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    raw = rng.standard_normal(n) + sep * y
    return y, raw


# ----------------------------------------------------------------------------
# Unit: DeLong AUC point estimate + variance
# ----------------------------------------------------------------------------


def test_midrank_handles_ties():
    # [10, 20, 20, 40] -> ranks 1, 2.5, 2.5, 4.
    """Midrank handles ties."""
    r = _midrank(np.array([10.0, 20.0, 20.0, 40.0]))
    assert np.allclose(r, [1.0, 2.5, 2.5, 4.0])


@pytest.mark.parametrize("sep", [0.3, 1.0, 2.5])
def test_delong_auc_matches_sklearn(sep):
    """Delong auc matches sklearn."""
    y, s = _binary(sep=sep, seed=1)
    auc, var = delong_auc_variance(y, s)
    assert auc == pytest.approx(roc_auc_score(y, s), abs=1e-9)
    assert var > 0


def test_delong_auc_matches_sklearn_with_ties():
    """Delong auc matches sklearn with ties."""
    y = np.array([0, 0, 1, 1, 0, 1, 1, 0])
    s = np.array([0.2, 0.2, 0.8, 0.8, 0.5, 0.5, 0.8, 0.2])
    auc, _var = delong_auc_variance(y, s)
    assert auc == pytest.approx(roc_auc_score(y, s), abs=1e-9)


def test_ci_brackets_point_estimate_and_clips():
    """Ci brackets point estimate and clips."""
    y, s = _binary(sep=2.0, seed=2)
    auc, lo, hi = delong_auc_ci(y, s)
    assert lo <= auc <= hi
    assert 0.0 <= lo and hi <= 1.0


def test_empty_class_returns_nan():
    """Empty class returns nan."""
    y = np.zeros(50, dtype=int)  # no positives
    s = np.random.default_rng(0).random(50)
    auc, lo, hi = delong_auc_ci(y, s)
    assert np.isnan(auc) and np.isnan(lo) and np.isnan(hi)


# ----------------------------------------------------------------------------
# Unit: multiclass ROC panel wiring
# ----------------------------------------------------------------------------


def test_roc_panel_legend_carries_ci_when_enabled():
    """Roc panel legend carries ci when enabled."""
    rng = np.random.default_rng(3)
    n, K = 2000, 3
    yt = rng.integers(0, K, n)
    proba = rng.dirichlet([1] * K, size=n)
    for i, t in enumerate(yt):
        proba[i, t] += 0.6
        proba[i] /= proba[i].sum()
    panel = _roc_panel(yt, proba, list(range(K)), show_auc_ci=True)
    # At least one class label has the "[lo, hi]" CI bracket.
    assert any("[" in lab and "]" in lab for lab in panel.series_labels)
    panel_off = _roc_panel(yt, proba, list(range(K)), show_auc_ci=False)
    assert all("[" not in lab for lab in panel_off.series_labels if "AUC=" in lab)


# ----------------------------------------------------------------------------
# biz_value: CI brackets known AUC and narrows with n
# ----------------------------------------------------------------------------


def test_biz_val_delong_ci_brackets_known_auc():
    """On a synthetic with a known population AUC (sep=1.0 gaussians -> AUC=Phi(1/sqrt2)
    ~0.760), the 95% DeLong CI must bracket the truth. Measured: estimate ~0.76, CI half-width
    ~0.013 at n=10000. Pins that the CI is centred on and contains the true AUC."""
    from scipy.stats import norm

    true_auc = float(norm.cdf(1.0 / np.sqrt(2.0)))  # ~0.7602 for unit-variance gaussians shifted by 1
    y, s = _binary(n=10000, sep=1.0, seed=5)
    auc, lo, hi = delong_auc_ci(y, s)
    assert lo <= true_auc <= hi, (lo, true_auc, hi)
    assert abs(auc - true_auc) < 0.03


def test_biz_val_delong_ci_narrows_with_n():
    """The DeLong CI width must shrink ~1/sqrt(n): going from n=2000 to n=32000 (16x) should
    cut the half-width by roughly sqrt(16)=4x. Floor: width(2000)/width(32000) >= 3.0 (below the
    ~4x theoretical, above noise). A regression that drops the n-dependence trips this."""

    def _half_width(n):
        """Helper: Half width."""
        y, s = _binary(n=n, sep=1.0, seed=9)
        _auc, lo, hi = delong_auc_ci(y, s)
        return (hi - lo) / 2.0

    w_small = _half_width(2000)
    w_large = _half_width(32000)
    assert w_small / w_large >= 3.0, (w_small, w_large)
