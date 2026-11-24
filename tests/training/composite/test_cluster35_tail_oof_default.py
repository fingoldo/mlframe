"""Cluster 35 regression: the tail residuals default to out-of-fold body predictions (audit 2026-11-20, finding 35-F33).

With no residual_X, the default cross-fits the body so the tail threshold and GPD are fitted on held-out residuals. Setting
residual_oof_folds below 2 restores the in-sample path, which logs a WARNING.
"""

import logging

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from mlframe.training.composite.extremes import TailCompositeEstimator


def _data(seed: int = 0):
    """Return a small frame where the target is a noisy function of one base column."""
    rng = np.random.default_rng(seed)
    n = 400
    b = rng.normal(size=n)
    f = rng.normal(size=n)
    y = b + 0.5 * f + rng.standard_t(df=3, size=n)
    return pd.DataFrame({"b": b, "f": f}), pd.Series(y, name="y")


def test_default_residuals_are_out_of_fold():
    """The default path leaves fitted residuals that differ from the in-sample ones (35-F33)."""
    X, y = _data()
    oof = TailCompositeEstimator(base_estimator=LinearRegression(), base_column="b", transform_name="diff").fit(X, y)
    ins = TailCompositeEstimator(base_estimator=LinearRegression(), base_column="b", transform_name="diff", residual_oof_folds=0).fit(X, y)
    assert oof.residual_oof_folds == 5
    assert not np.allclose(oof.residuals_sorted_, ins.residuals_sorted_)


def test_in_sample_path_logs_warning(caplog):
    """residual_oof_folds below 2 falls back to in-sample residuals with a WARNING (35-F33)."""
    X, y = _data(seed=1)
    with caplog.at_level(logging.WARNING):
        TailCompositeEstimator(base_estimator=LinearRegression(), base_column="b", transform_name="diff", residual_oof_folds=0).fit(X, y)
    assert any("IN-SAMPLE" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("folds", [5, 3])
def test_oof_fit_sets_a_threshold(folds):
    """The cross-fitted path produces a finite threshold for each fold count (35-F33)."""
    X, y = _data(seed=2)
    est = TailCompositeEstimator(base_estimator=LinearRegression(), base_column="b", transform_name="diff", residual_oof_folds=folds).fit(X, y)
    assert np.isfinite(est.threshold_)
