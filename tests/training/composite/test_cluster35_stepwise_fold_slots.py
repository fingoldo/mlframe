"""Audit 2026-11-20 finding 35-F5: the stepwise fold scorer keeps one slot per fold, so paired tests zip by fold, not by position.

A fold with no finite prediction used to be dropped, which shifted every later fold's pair. Now it keeps a NaN slot.
"""

import numpy as np

from mlframe.training.composite.discovery.forward_stepwise import _stepwise_cv_rmse_with_folds


def test_a_fully_unusable_fold_keeps_its_nan_slot():
    """With time-ordered 4 folds over 80 rows, the last validation block (rows 64..79) is all NaN: the list still has 4 entries,
    and its last entry is NaN instead of the list shrinking to 3."""
    rng = np.random.default_rng(0)
    n = 80
    x = rng.standard_normal(n)
    y = 2.0 * x + rng.standard_normal(n) * 0.1
    y[64:] = np.nan
    _, fold_rmses = _stepwise_cv_rmse_with_folds(
        y, {"x": x}, ["x"], groups=None, cv_splitter=None, time_aware=True, cv_folds=4, random_state=0,
        cv_selector_mode="mean", cv_selector_alpha=0.05, cv_selector_confidence=0.95, cv_selector_quantile_level=0.5,
    )
    assert len(fold_rmses) == 4
    assert np.isnan(fold_rmses[-1])
    assert np.all(np.isfinite(fold_rmses[:-1]))
