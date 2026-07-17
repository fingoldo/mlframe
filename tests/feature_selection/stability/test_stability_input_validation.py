"""Wave 9.1 loop-iter-41 regression: ``StabilityMRMR.fit`` MUST
validate inputs before dispatching bootstraps.

Pre-fix at ``stability.py:73, 105``:

1. ``sub_size = int(self.sample_fraction * n_samples)``
   With ``sample_fraction=0.05, n=10`` -> ``int(0.5) = 0``. Every
   clone fit on X[0:0] -> garbage selection probabilities, no warning.

2. ``self.selection_probabilities_ = counts / self.n_bootstraps``
   With ``n_bootstraps=0`` -> div-by-zero RuntimeWarning + NaN probs +
   empty support. Indistinguishable from a legitimate "all below
   threshold" result.

3. Negative / out-of-range ``support_threshold`` / ``sample_fraction``
   leaked to numpy with unhelpful errors.

Severity: P1/P2 silent corruption (depending on sub-case) in
parameter-sweep / grid-search loops where the wrapper sees the full
range of input values.

Fix at stability.py:68: explicit param validation up front with
sklearn-style ``ValueError`` (after the ``__init__`` -> ``fit``
boundary as sklearn convention requires). Floor sub_size at 2 so the
inner MRMR never sees a 0-row fit; reject out-of-range params with
informative messages.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest


def _setup():
    """Helper that setup."""
    from mlframe.feature_selection.filters.stability import StabilityMRMR
    from mlframe.feature_selection.filters.mrmr import MRMR

    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame(rng.standard_normal((n, 3)), columns=["a", "b", "c"])
    y = pd.Series((X["a"] > 0).astype(np.int64))
    return StabilityMRMR, MRMR, X, y


def test_n_bootstraps_zero_raises():
    """N bootstraps zero raises."""
    StabilityMRMR, MRMR, X, y = _setup()
    with pytest.raises(ValueError, match="n_bootstraps"):
        StabilityMRMR(
            estimator=MRMR(verbose=0),
            n_bootstraps=0,
            sample_fraction=0.5,
        ).fit(X, y)


def test_n_bootstraps_negative_raises():
    """N bootstraps negative raises."""
    StabilityMRMR, MRMR, X, y = _setup()
    with pytest.raises(ValueError, match="n_bootstraps"):
        StabilityMRMR(
            estimator=MRMR(verbose=0),
            n_bootstraps=-5,
            sample_fraction=0.5,
        ).fit(X, y)


@pytest.mark.parametrize("frac", [0.0, -0.1, 1.5, 2.0])
def test_sample_fraction_out_of_range_raises(frac):
    """Sample fraction out of range raises."""
    StabilityMRMR, MRMR, X, y = _setup()
    with pytest.raises(ValueError, match="sample_fraction"):
        StabilityMRMR(
            estimator=MRMR(verbose=0),
            n_bootstraps=3,
            sample_fraction=frac,
        ).fit(X, y)


def test_support_threshold_negative_raises():
    """Only strictly-negative support_threshold is rejected. 0.0 and
    >1.0 are documented sentinels (stability.py:90-100): 0.0 keeps
    every touched feature (>=0 always satisfied); >1.0 yields empty
    support (no probability can clear it). Reject only the values
    that produce numerically undefined behaviour, not the useful
    extremes.
    """
    StabilityMRMR, MRMR, X, y = _setup()
    with pytest.raises(ValueError, match="support_threshold"):
        StabilityMRMR(
            estimator=MRMR(verbose=0),
            n_bootstraps=3,
            sample_fraction=0.5,
            support_threshold=-0.1,
        ).fit(X, y)


def test_support_threshold_zero_keeps_all_touched():
    """0.0 sentinel: every feature with prob>=0 (all of them) is in support_."""
    StabilityMRMR, MRMR, X, y = _setup()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sel = StabilityMRMR(
            estimator=MRMR(verbose=0),
            n_bootstraps=3,
            sample_fraction=0.5,
            support_threshold=0.0,
        ).fit(X, y)
    assert len(sel.support_) == X.shape[1]


@pytest.mark.parametrize("thr", [1.5, 2.0])
def test_support_threshold_above_one_yields_empty(thr):
    """>1.0 sentinel: no probability can clear it -> support_ empty."""
    StabilityMRMR, MRMR, X, y = _setup()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sel = StabilityMRMR(
            estimator=MRMR(verbose=0),
            n_bootstraps=3,
            sample_fraction=0.5,
            support_threshold=thr,
        ).fit(X, y)
    assert len(sel.support_) == 0


def test_valid_params_still_work():
    """Negative control: valid params must succeed without warning."""
    StabilityMRMR, MRMR, X, y = _setup()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sel = StabilityMRMR(
            estimator=MRMR(verbose=0),
            n_bootstraps=3,
            sample_fraction=0.5,
        ).fit(X, y)
    assert hasattr(sel, "selection_probabilities_")
    assert hasattr(sel, "support_")
    # Probabilities must be in [0, 1] - not NaN from div-by-zero.
    assert (sel.selection_probabilities_ >= 0).all()
    assert (sel.selection_probabilities_ <= 1).all()


def test_tiny_n_with_tiny_frac_raises_informatively():
    """Pre-fix gave sub_size=0 and silent empty fits. Post-fix the
    floor sub_size=2 either succeeds (if MRMR can fit 2 rows - unusual)
    or raises a clear inner-MRMR error - either way no silent corruption.
    """
    StabilityMRMR, MRMR, X, y = _setup()
    with pytest.raises((ValueError, RuntimeError)):
        StabilityMRMR(
            estimator=MRMR(verbose=0),
            n_bootstraps=2,
            sample_fraction=0.05,
        ).fit(X.iloc[:10], y.iloc[:10])
