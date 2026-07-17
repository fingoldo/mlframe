"""Regression: ``CompositeQuantileEstimator`` quantile labels must stay correct
under a monotone-DECREASING transform inverse (``reciprocal_residual``).

``reciprocal_residual`` maps ``y = 1/(T + 1/base)``, which DECREASES in T, so the
head trained for the tau-quantile of T produces the (1-tau)-quantile of y. Before
the fit-time decreasing-inverse detection + complementary-head re-labelling, the
raw columns (``enforce_non_crossing=False``) were silently SWAPPED: the column
labelled tau=0.1 carried the 0.9 y-quantile and vice versa. The non-crossing sort
papered over it in the default path but any consumer of the raw columns, or any
caller diagnosing the crossing rate, read mislabelled quantiles.

The pre-fix code (no ``_inverse_decreasing_`` flag, direct head lookup) FAILS the
swap assertion below; the post-fix code serves each column from the complementary
head so labels are correct in BOTH sort modes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.training.composite import CompositeQuantileEstimator
from mlframe.training.composite.quantile import _transform_inverse_decreasing


def _make_reciprocal_data(n: int = 4000, seed: int = 3):
    """Make reciprocal data."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(2.0, 5.0, size=n)
    inv_y = 1.0 / base + rng.normal(scale=0.05, size=n)
    inv_y = np.clip(inv_y, 0.01, None)
    y = 1.0 / inv_y
    X = pd.DataFrame({"base": base, "f0": rng.normal(size=n)})
    return X, y


def _inner():
    """Inner."""
    from sklearn.ensemble import GradientBoostingRegressor

    return GradientBoostingRegressor(n_estimators=80, max_depth=2, random_state=0)


def test_reciprocal_inverse_detected_decreasing():
    """Reciprocal inverse detected decreasing."""
    assert _transform_inverse_decreasing("reciprocal_residual") is True
    # Increasing / additive inverses must NOT be flagged (no false-positive flip).
    for name in ("linear_residual", "diff", "logratio"):
        assert _transform_inverse_decreasing(name) is False


@pytest.mark.parametrize("enforce_non_crossing", [True, False])
def test_reciprocal_quantile_labels_not_swapped(enforce_non_crossing):
    """The tau=0.1 column must cover ~10% of y and tau=0.9 ~90%, in BOTH modes.

    Pre-fix (no complementary-head re-labelling) the raw-mode (``False``) columns
    are swapped, so the 0.1 column covers ~90% -> this assertion fails.
    """
    X, y = _make_reciprocal_data()
    est = CompositeQuantileEstimator(
        base_estimator=_inner(),
        transform_name="reciprocal_residual",
        base_column="base",
        quantiles=(0.1, 0.5, 0.9),
        enforce_non_crossing=enforce_non_crossing,
    )
    est.fit(X, y)
    qp = est.predict_quantile(X)  # columns labelled [0.1, 0.5, 0.9]
    cov_lo = float((y <= qp[:, 0]).mean())
    cov_mid = float((y <= qp[:, 1]).mean())
    cov_hi = float((y <= qp[:, 2]).mean())
    # Each labelled column must cover close to its nominal level.
    assert abs(cov_lo - 0.1) < 0.05, f"tau=0.1 column covers {cov_lo:.3f}, expected ~0.10 (swapped?)"
    assert abs(cov_mid - 0.5) < 0.05, f"tau=0.5 column covers {cov_mid:.3f}"
    assert abs(cov_hi - 0.9) < 0.05, f"tau=0.9 column covers {cov_hi:.3f}, expected ~0.90 (swapped?)"
    # And the low column must lie below the high column (correct ordering).
    assert cov_lo < cov_hi


def test_reciprocal_raw_columns_ascending_in_value():
    """With the fix, even raw (unsorted) columns are ascending in value per row,
    because complementary-head re-labelling already orders them correctly."""
    X, y = _make_reciprocal_data()
    est = CompositeQuantileEstimator(
        base_estimator=_inner(),
        transform_name="reciprocal_residual",
        base_column="base",
        quantiles=(0.1, 0.25, 0.5, 0.75, 0.9),
        enforce_non_crossing=False,
    )
    est.fit(X, y)
    qp = est.predict_quantile(X)
    # The median of each column must be monotone-increasing across levels.
    col_medians = np.median(qp, axis=0)
    assert np.all(np.diff(col_medians) > 0), f"column medians not ascending: {col_medians}"
