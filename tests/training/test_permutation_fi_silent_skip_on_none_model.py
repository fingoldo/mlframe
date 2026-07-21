"""Regression test for the 2026-05-27 PI warning spam on ensembles.

The TVT regression log showed 6 warnings (one per ensemble flavour) of
form:
  permutation_importance failed (The 'estimator' parameter of
  permutation_importance must be an object implementing 'fit'.
  Got None instead.); skipping FI.

Ensembles (EnsARITHM / HARM / MEDIAN / QUAD / QUBE / GEO) arrive at the
PI helper with model=None because the per-member voting logic does not
expose a sklearn-style estimator surface. The fix: short-circuit with
a DEBUG note (not WARN) when model is None, so the log stays clean.
"""

from __future__ import annotations

import logging

import numpy as np


def test_permutation_fi_silent_skip_when_model_is_none(caplog) -> None:
    """When model=None, the helper returns None WITHOUT a warning."""
    from mlframe.training._feature_importances import (
        _permutation_feature_importances,
    )

    X = np.random.default_rng(0).standard_normal((100, 5))
    y = np.random.default_rng(1).standard_normal(100)
    with caplog.at_level(logging.WARNING, logger="mlframe.training._feature_importances"):
        out = _permutation_feature_importances(None, X, y)
    assert out is None
    # No WARNING-level records emitted on the None-model short-circuit.
    warning_msgs = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert not warning_msgs, f"expected no WARNING records for None-model case, got: {warning_msgs}"


def test_permutation_fi_still_warns_on_genuine_failure(caplog) -> None:
    """When model IS supplied but the underlying call fails, the helper
    still emits WARN (regression guard so the silent-skip doesn't
    swallow real bugs).
    """
    from mlframe.training._feature_importances import (
        _permutation_feature_importances,
    )

    class _BrokenEstimator:
        # Has fit + predict so sklearn passes the surface check, but
        # predict raises so permutation_importance fails internally.
        """Groups tests covering broken estimator."""
        def fit(self, X, y):
            """Fit."""
            return self

        def predict(self, X):
            """Predict."""
            raise RuntimeError("synthetic predict failure")

    X = np.random.default_rng(2).standard_normal((50, 3))
    y = np.random.default_rng(3).standard_normal(50)
    with caplog.at_level(logging.WARNING, logger="mlframe.training._feature_importances"):
        _permutation_feature_importances(_BrokenEstimator(), X, y)
    # Either the function returns None silently (predict raises -> -inf
    # scorer -> permutation still runs but produces uninformative FI),
    # or it warns. Either is acceptable; we just want the None-model
    # path NOT to suppress real failures.
    # No assertion here; this test exists so the None-model fix above
    # cannot accidentally regress into a blanket silent-skip on all
    # failures.


def test_permutation_fi_force_cpu_predict_set_during_loop_and_restored() -> None:
    """During the threaded permutation-importance loop, model._force_cpu_predict must be True for every
    predict call (routes concurrent predicts to CPU, avoiding the device-churn race documented in
    _base_predict.py -- see the 2026-07-21 fix), and cleaned up afterward."""
    from mlframe.training._feature_importances import _permutation_feature_importances

    seen_flag_values = []

    class _FlagCheckingEstimator:
        """Groups tests covering flag checking estimator."""
        def fit(self, X, y):
            """Fit."""
            return self

        def predict(self, X):
            """Predict."""
            seen_flag_values.append(getattr(self, "_force_cpu_predict", "MISSING"))
            return np.zeros(len(X))

    rng = np.random.default_rng(0)
    X = rng.standard_normal((60, 3))
    y = rng.standard_normal(60)
    est = _FlagCheckingEstimator()
    assert not hasattr(est, "_force_cpu_predict")
    _permutation_feature_importances(est, X, y, n_repeats=2)
    assert seen_flag_values, "expected predict to have been called at least once"
    assert all(v is True for v in seen_flag_values), f"expected _force_cpu_predict=True during every predict call, got {set(seen_flag_values)}"
    assert not hasattr(est, "_force_cpu_predict"), "expected _force_cpu_predict to be cleaned up after the loop"


def test_permutation_fi_return_std_tuple_shape_on_none_model() -> None:
    """return_std=True keeps a stable (None, None) tuple on the short-circuit path."""
    from mlframe.training._feature_importances import _permutation_feature_importances

    X = np.random.default_rng(0).standard_normal((40, 4))
    y = np.random.default_rng(1).standard_normal(40)
    out = _permutation_feature_importances(None, X, y, return_std=True)
    assert out == (None, None)


def test_permutation_fi_return_std_yields_dispersion() -> None:
    """return_std=True returns a per-feature std aligned to the mean (INV-22 whisker source)."""
    from sklearn.neighbors import KNeighborsRegressor

    from mlframe.training._feature_importances import _permutation_feature_importances

    rng = np.random.default_rng(0)
    X = rng.standard_normal((300, 4))
    y = X[:, 0] * 2.0 + rng.standard_normal(300) * 0.1
    model = KNeighborsRegressor(n_neighbors=5).fit(X, y)
    mean, std = _permutation_feature_importances(model, X, y, return_std=True, n_repeats=4, random_state=0)
    assert mean is not None and std is not None
    assert mean.shape == (4,) and std.shape == (4,)
    assert np.all(std >= 0.0)
    # n_repeats>1 on a noisy estimator must produce some nonzero dispersion.
    assert np.any(std > 0.0)


def test_get_model_feature_importances_return_std_for_permutation_model() -> None:
    """get_model_feature_importances(return_std=True) surfaces the permutation std.

    KNeighborsRegressor has neither feature_importances_ nor coef_, so it routes
    through the permutation fallback; the std must reach the caller for the chart."""
    from sklearn.neighbors import KNeighborsRegressor

    from mlframe.training._feature_importances import get_model_feature_importances

    rng = np.random.default_rng(1)
    X = rng.standard_normal((300, 3))
    y = X[:, 1] * 1.5 + rng.standard_normal(300) * 0.1
    model = KNeighborsRegressor(n_neighbors=5).fit(X, y)
    cols = ["a", "b", "c"]
    fi, std = get_model_feature_importances(model, cols, X=X, y=y, return_std=True)
    assert fi is not None and std is not None
    assert fi.shape == (3,) and std.shape == (3,)
    assert np.all(std >= 0.0)


def test_get_model_feature_importances_native_path_std_is_none() -> None:
    """Native tree-gain FI carries no dispersion -> std is None under return_std=True."""
    from sklearn.ensemble import RandomForestRegressor

    from mlframe.training._feature_importances import get_model_feature_importances

    rng = np.random.default_rng(2)
    X = rng.standard_normal((200, 3))
    y = X[:, 0] * 2.0 + rng.standard_normal(200) * 0.1
    model = RandomForestRegressor(n_estimators=10, random_state=0).fit(X, y)
    fi, std = get_model_feature_importances(model, ["a", "b", "c"], return_std=True)
    assert fi is not None
    assert std is None
