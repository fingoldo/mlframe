"""Cluster 23 regression: ``is_risky`` needs a real gap in fold stability, not just the largest z-score of near-identical variants (audit 2026-11-20, 23-F10)."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

from mlframe.evaluation.imputation_sensitivity_check import _clears_std_floor, imputation_sensitivity_check


class _FoldNoise(BaseEstimator, RegressorMixin):
    """A regressor whose error amplitude is set by a column of X: the per-fold score spread then differs between variants on purpose."""

    def fit(self, X, y):
        """Nothing to learn."""
        return self

    def predict(self, X):
        """Predict the target plus an offset read from the ``noise`` column."""
        return X["base"].to_numpy() + X["noise"].to_numpy()


def _variants(noise_levels):
    """One variant per noise level over the same rows and target."""
    n = 200
    rng = np.random.default_rng(0)
    y = rng.normal(size=n)
    jitter = rng.normal(size=n)
    return {f"v{i}": pd.DataFrame({"base": y, "noise": lvl * jitter}) for i, lvl in enumerate(noise_levels)}, y


def _neg_mse(y_true, y_pred):
    """Higher is better."""
    return -float(np.mean((y_true - y_pred) ** 2))


def test_floor_helper_compares_with_the_median_of_the_others():
    """A std clears the floor only above ratio x median of the others; with no others, or others at zero, any positive std clears it (23-F10)."""
    stds = np.array([0.10, 0.101, 0.102])
    z = (stds - stds.mean()) / stds.std(ddof=1)
    assert z.max() > 0.5, "the pure z-score rule would flag the least stable of these near-identical variants"
    assert not _clears_std_floor(stds, 1.25).any()
    assert _clears_std_floor(np.array([0.1, 0.1, 0.3]), 1.25).tolist() == [False, False, True]
    assert _clears_std_floor(np.array([0.0, 0.0, 0.2]), 1.25).tolist() == [False, False, True]
    assert _clears_std_floor(np.array([0.5]), 1.25).tolist() == [True]


def test_near_identical_variants_are_not_flagged_but_a_clearly_worse_one_is():
    """Equal-noise variants give no risky flag, and a much noisier variant is the one flagged, with the default floor (23-F10)."""
    same, y = _variants([1.0, 1.0, 1.0])
    kw = dict(estimator=_FoldNoise(), y=y, scoring=_neg_mse)
    floor_on = imputation_sensitivity_check(X_variants=same, **kw)
    assert not floor_on["is_risky"].any()
    worse, y2 = _variants([1.0, 1.0, 6.0])
    flagged = imputation_sensitivity_check(X_variants=worse, estimator=_FoldNoise(), y=y2, scoring=_neg_mse)
    assert flagged.loc["v2", "is_risky"] and not flagged.loc[["v0", "v1"], "is_risky"].any()
