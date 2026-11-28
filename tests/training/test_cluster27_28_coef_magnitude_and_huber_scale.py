"""Regressions: coef_ importances are magnitudes on every path, and the Huber delta follows the target scale (audit 2026-11-20, 27-F8 and 28-F38)."""

import numpy as np
from sklearn.linear_model import LinearRegression

from mlframe.training._feature_importances import _collapse_coef
from mlframe.training.loss_recommendation import huber_delta_for_scale


def test_collapse_coef_returns_magnitudes_for_every_shape():
    """1-D, single-row 2-D and multi-row 2-D coefficient arrays all give non-negative importances."""
    one_d = np.array([-2.0, 1.0])
    np.testing.assert_array_equal(_collapse_coef(one_d), [2.0, 1.0])
    np.testing.assert_array_equal(_collapse_coef(one_d[None, :]), [2.0, 1.0])
    np.testing.assert_array_equal(_collapse_coef(np.array([[-2.0, 1.0], [2.0, -3.0]])), [2.0, 2.0])


def test_single_model_and_its_multioutput_wrapper_agree_in_sign_convention():
    """A bare linear model with a negative coefficient reports a positive importance, as the MultiOutput branch does."""
    from mlframe.training._feature_importances import get_model_feature_importances

    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 2))
    y = -3.0 * X[:, 0] + 0.5 * X[:, 1]
    fi = get_model_feature_importances(LinearRegression().fit(X, y), ["a", "b"])
    assert (np.asarray(fi) >= 0).all() and fi[0] > fi[1]


def test_huber_delta_has_no_raw_unit_floor():
    """A small-scale target gets a delta of 1.345 * scale; an unknown scale still falls back to 1.0."""
    assert abs(huber_delta_for_scale(0.01) - 0.01345) < 1e-12
    assert huber_delta_for_scale(0.0) == 1.0 and huber_delta_for_scale(float("nan")) == 1.0
    assert abs(huber_delta_for_scale(10.0) - 13.45) < 1e-12
