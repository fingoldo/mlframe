"""Hosmer-Lemeshow refuses NaN scores, and CV(RMSE) is NaN for a non-positive mean as its docstring promises."""

from __future__ import annotations

import numpy as np

from mlframe.metrics.classification._classification_calibration import hosmer_lemeshow_test
from mlframe.metrics.regression._regression_extras import fast_cv_rmse


def test_nan_score_gives_nan_statistic_instead_of_a_finite_h_on_fewer_groups():
    """One NaN among 200 scores used to drop a group silently and return a finite H."""
    rng = np.random.default_rng(0)
    p = rng.random(200)
    y = (rng.random(200) < p).astype(int)
    p[7] = np.nan
    h, pval, dof = hosmer_lemeshow_test(y, p)
    assert np.isnan(h) and np.isnan(pval) and dof == 0


def test_negative_mean_target_gives_nan_cv_rmse():
    """y_true=[-10, -10] has mean -10: the ratio is meaningless, NaN (it returned 0.1)."""
    assert np.isnan(fast_cv_rmse(np.array([-10.0, -10.0]), np.array([-9.0, -11.0])))


def test_positive_mean_keeps_its_value():
    """Negative control: mean 10, rmse 1 gives 0.1."""
    assert abs(fast_cv_rmse(np.array([10.0, 10.0]), np.array([9.0, 11.0])) - 0.1) < 1e-12
