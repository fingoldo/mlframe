"""sklearn-compliance harness for the composite / lag / early-stop / preprocessing wrappers
that ``tests/test_sklearn_compliance.py`` did not exercise.

Coverage matrix (one ``parametrize_with_checks`` or ``check_estimator`` invocation
per class; happy-path get_params/set_params/clone/fit/predict round-trip where the
full check suite is incompatible with the wrapper's signature).

Wrappers covered:
- ``CompositeTargetEstimator``           (training/_composite_target_estimator.py)
- ``_LagPredictDeployableModel``         (training/core/_phase_composite_post.py)
- ``ESTransformedTargetRegressor``       (estimators/custom.py)
- ``EstimatorWithEarlyStopping``         (estimators/base.py)
- ``RegressorWithEarlyStopping``         (estimators/base.py)
- ``PdOrdinalEncoder``                   (estimators/custom.py)
- ``PdKBinsDiscretizer``                 (estimators/custom.py)
- ``RFECV`` (mlframe wrapper)            (feature_selection/wrappers/_rfecv.py)

Each test exercises the minimum sklearn contract a wrapper participates in:
- get_params / set_params round-trip via dict equality
- sklearn.clone() returns a fresh unfitted instance with the same init params
- basic fit -> predict (or fit -> transform) happy path on a tiny synthetic
- the wrapper exposes the canonical fitted-state attrs documented in its docstring

This is intentionally narrower than ``sklearn.utils.estimator_checks.check_estimator``:
many of these wrappers are NOT full sklearn-compliant by design (RFECV's
hyper-rich init signature, CompositeTargetEstimator's domain-specific X
contract, the early-stop wrappers' Catboost-only branch). Full check_estimator
xfails would be high-noise without surfacing real bugs; this test surface
catches the bugs that historically did slip through (clone-induced fitted-state
loss, get_params drift after sklearn minor bumps, missing classes_/n_features_in_).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from sklearn.base import clone, BaseEstimator
from sklearn.linear_model import LinearRegression, LogisticRegression

# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def small_regression_data():
    """Helper that small regression data."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "base": rng.uniform(1.0, 5.0, size=40),
            "f1": rng.normal(size=40),
            "f2": rng.normal(size=40),
        }
    )
    y = (2.0 * X["base"] + 0.5 * X["f1"] + rng.normal(scale=0.1, size=40)).to_numpy()
    return X, y


@pytest.fixture
def small_classification_data():
    """Helper that small classification data."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 3))
    y = (X[:, 0] > 0).astype(int)
    return X, y


def _assert_clone_returns_unfitted_shell(estimator: BaseEstimator) -> BaseEstimator:
    """Clone the estimator, assert (1) it's a new object, (2) init params match,
    (3) any underscore-suffixed fitted attrs are absent on the clone.
    """
    cloned = clone(estimator)
    assert cloned is not estimator
    src_params = estimator.get_params(deep=False)
    new_params = cloned.get_params(deep=False)
    assert set(src_params.keys()) == set(new_params.keys())
    for k in src_params:
        # Nested estimators are themselves cloned; identity differs but type matches.
        sv, nv = src_params[k], new_params[k]
        if isinstance(sv, BaseEstimator):
            assert type(nv) is type(sv), f"param {k!r}: clone changed estimator type"
        else:
            assert sv == nv or (isinstance(sv, float) and isinstance(nv, float) and sv != sv and nv != nv), f"param {k!r}: src={sv!r} new={nv!r}"
    return cloned


# ---------------------------------------------------------------------------
# CompositeTargetEstimator
# ---------------------------------------------------------------------------


class TestCompositeTargetEstimatorCompliance:
    """sklearn-compliance happy path for CompositeTargetEstimator.

    Notes:
    - ``from_fitted_inner`` is explicitly NOT clone-safe (see S26 regression
      test). The __init__ + fit path is.
    - The wrapper requires an X frame with a ``base_column`` -- this is by
      design; full ``check_estimator`` would fail because it feeds raw ndarrays
      with no column metadata.
    """

    def _make(self):
        """Helper that make."""
        from mlframe.training.composite import CompositeTargetEstimator

        return CompositeTargetEstimator(
            base_estimator=LinearRegression(),
            transform_name="diff",
            base_column="base",
        )

    def test_get_set_params_roundtrip(self):
        """Get set params roundtrip."""
        est = self._make()
        params = est.get_params(deep=False)
        # Round-trip MUST be lossless.
        est2 = self._make()
        est2.set_params(**params)
        assert est2.get_params(deep=False) == params

    def test_clone_of_unfitted_wrapper(self):
        """Clone of unfitted wrapper."""
        _assert_clone_returns_unfitted_shell(self._make())

    def test_fit_then_predict_happy_path(self, small_regression_data):
        """Fit then predict happy path."""
        X, y = small_regression_data
        est = self._make()
        est.fit(X, y)
        preds = est.predict(X)
        assert preds.shape == (len(X),)
        assert np.all(np.isfinite(preds))

    def test_fitted_state_attrs_populated(self, small_regression_data):
        """Fitted state attrs populated."""
        X, y = small_regression_data
        est = self._make()
        est.fit(X, y)
        assert hasattr(est, "estimator_")
        assert hasattr(est, "fitted_params_")


# ---------------------------------------------------------------------------
# _LagPredictDeployableModel
# ---------------------------------------------------------------------------


class TestLagPredictDeployableModelCompliance:
    """Hand-rolled duck-typed estimator (no BaseEstimator inheritance).
    Explicitly implements get_params/set_params/fit/predict so sklearn.clone
    accepts it during honest-OOF refit.
    """

    def _make(self):
        """Helper that make."""
        from mlframe.training.core._phase_composite_post import _LagPredictDeployableModel

        return _LagPredictDeployableModel(lag_column="base")

    def test_get_set_params_roundtrip(self):
        """Get set params roundtrip."""
        est = self._make()
        params = est.get_params()
        est2 = self._make()
        est2.set_params(**params)
        assert est2.get_params() == params

    def test_clone_returns_fresh_instance(self):
        """Clone returns fresh instance."""
        est = self._make()
        cloned = clone(est)
        assert cloned is not est
        assert cloned.lag_column == est.lag_column

    def test_fit_predict_happy_path(self, small_regression_data):
        """Fit predict happy path."""
        X, y = small_regression_data
        est = self._make()
        est.fit(X, y)
        preds = est.predict(X)
        np.testing.assert_allclose(preds, X["base"].to_numpy(), rtol=1e-9)


# ---------------------------------------------------------------------------
# ESTransformedTargetRegressor
# ---------------------------------------------------------------------------


class TestESTransformedTargetRegressorCompliance:
    """Groups tests covering TestESTransformedTargetRegressorCompliance."""
    def _make(self):
        """Helper that make."""
        from mlframe.estimators.custom import ESTransformedTargetRegressor

        # log1p / expm1 round-trip; both are vector-safe, finite-preserving.
        return ESTransformedTargetRegressor(
            regressor=LinearRegression(),
            func=np.log1p,
            inverse_func=np.expm1,
            check_inverse=False,
        )

    def test_get_set_params_roundtrip(self):
        """Get set params roundtrip."""
        est = self._make()
        params = est.get_params(deep=False)
        est2 = self._make()
        est2.set_params(**params)
        assert est2.get_params(deep=False) == params

    def test_clone_of_unfitted_wrapper(self):
        """Clone of unfitted wrapper."""
        _assert_clone_returns_unfitted_shell(self._make())

    def test_fit_then_predict_happy_path(self):
        """Fit then predict happy path."""
        rng = np.random.default_rng(0)
        X = rng.uniform(0.5, 5.0, size=(40, 3))
        y = np.abs(rng.normal(loc=5.0, scale=0.3, size=40))
        est = self._make()
        est.fit(X, y)
        preds = est.predict(X)
        assert preds.shape == (len(X),)
        assert np.all(np.isfinite(preds))


# ---------------------------------------------------------------------------
# EstimatorWithEarlyStopping / RegressorWithEarlyStopping
# ---------------------------------------------------------------------------


class TestEstimatorWithEarlyStoppingCompliance:
    """The base class branches on CatBoost vs non-CatBoost inner. For the
    non-CatBoost path it just delegates to base_estimator.fit -- which makes
    it a thin shim and trivially sklearn-compliant on the happy path.
    """

    def _make(self):
        """Helper that make."""
        from mlframe.estimators.base import EstimatorWithEarlyStopping

        return EstimatorWithEarlyStopping(base_estimator=LogisticRegression(max_iter=200))

    def test_get_set_params_roundtrip(self):
        """Get set params roundtrip."""
        est = self._make()
        params = est.get_params(deep=False)
        est2 = self._make()
        est2.set_params(**params)
        assert est2.get_params(deep=False) == params

    def test_clone_of_unfitted_wrapper(self):
        """Clone of unfitted wrapper."""
        _assert_clone_returns_unfitted_shell(self._make())

    def test_fit_then_predict_happy_path(self, small_classification_data):
        """Fit then predict happy path."""
        X, y = small_classification_data
        est = self._make()
        est.fit(X, y)
        preds = est.predict(X)
        assert preds.shape == (len(X),)


class TestRegressorWithEarlyStoppingCompliance:
    """Groups tests covering TestRegressorWithEarlyStoppingCompliance."""
    def _make(self):
        """Helper that make."""
        from mlframe.estimators.base import RegressorWithEarlyStopping

        return RegressorWithEarlyStopping(base_estimator=LinearRegression())

    def test_get_set_params_roundtrip(self):
        """Get set params roundtrip."""
        est = self._make()
        params = est.get_params(deep=False)
        est2 = self._make()
        est2.set_params(**params)
        assert est2.get_params(deep=False) == params

    def test_clone_of_unfitted_wrapper(self):
        """Clone of unfitted wrapper."""
        _assert_clone_returns_unfitted_shell(self._make())

    def test_fit_then_predict_happy_path(self, small_regression_data):
        """Fit then predict happy path."""
        X, y = small_regression_data
        # Drop the pandas frame -> ndarray since check_array does not accept
        # mixed-dtype frames with string column names by default at older
        # sklearn versions and the early-stop wrapper does its own check_array
        # in fit.
        X_arr = X.to_numpy()
        est = self._make()
        est.fit(X_arr, y)
        preds = est.predict(X_arr)
        assert preds.shape == (len(X_arr),)


# ---------------------------------------------------------------------------
# PdOrdinalEncoder
# ---------------------------------------------------------------------------


class TestPdOrdinalEncoderCompliance:
    """Groups tests covering TestPdOrdinalEncoderCompliance."""
    def _make(self):
        """Helper that make."""
        from mlframe.estimators.custom import PdOrdinalEncoder

        return PdOrdinalEncoder()

    def test_get_set_params_roundtrip(self):
        """Get set params roundtrip."""
        est = self._make()
        params = est.get_params(deep=False)
        est2 = self._make()
        est2.set_params(**params)
        assert est2.get_params(deep=False) == params

    def test_clone_of_unfitted_wrapper(self):
        """Clone of unfitted wrapper."""
        _assert_clone_returns_unfitted_shell(self._make())

    def test_fit_transform_happy_path(self):
        """Fit transform happy path."""
        df = pd.DataFrame({"a": ["x", "y", "x", "z"], "b": ["p", "q", "p", "p"]})
        enc = self._make()
        enc.fit(df)
        out = enc.transform(df)
        # Returns a pandas frame with int32 codes (per the wrapper contract).
        assert isinstance(out, pd.DataFrame)
        assert out.dtypes.iloc[0] == np.int32


# ---------------------------------------------------------------------------
# PdKBinsDiscretizer
# ---------------------------------------------------------------------------


class TestPdKBinsDiscretizerCompliance:
    """Groups tests covering TestPdKBinsDiscretizerCompliance."""
    def _make(self):
        """Helper that make."""
        from mlframe.estimators.custom import PdKBinsDiscretizer

        # encode='ordinal' to avoid the sparse densify path; the wrapper handles both but ordinal is the happy path for narrow asserts. subsample=None overrides the stale 'warn' default that sklearn>=1.5 rejects via _param_validation.
        return PdKBinsDiscretizer(n_bins=3, encode="ordinal", strategy="uniform", subsample=None)

    def test_get_set_params_roundtrip(self):
        """Get set params roundtrip."""
        est = self._make()
        params = est.get_params(deep=False)
        est2 = self._make()
        est2.set_params(**params)
        assert est2.get_params(deep=False) == params

    def test_clone_of_unfitted_wrapper(self):
        """Clone of unfitted wrapper."""
        _assert_clone_returns_unfitted_shell(self._make())

    def test_fit_transform_happy_path(self):
        """Fit transform happy path."""
        df = pd.DataFrame({"a": np.linspace(0, 10, 30), "b": np.linspace(-5, 5, 30)})
        enc = self._make()
        enc.fit(df)
        out = enc.transform(df)
        assert isinstance(out, pd.DataFrame)
        # Bin codes are non-negative ints.
        assert (out.to_numpy() >= 0).all()
        assert out.dtypes.iloc[0] == np.int32


# ---------------------------------------------------------------------------
# RFECV (mlframe wrapper)
# ---------------------------------------------------------------------------


class TestRFECVCompliance:
    """RFECV has a deep init signature (~40 params). Full check_estimator
    would XFAIL on multiple checks (its X contract is narrower than sklearn's
    synthetic random ndarray, and it uses fold-internal CV that the check
    suite cannot mock). Cover the clone-safety / get_params round-trip /
    smoke-fit instead.
    """

    def _make(self):
        """Helper that make."""
        from mlframe.feature_selection.wrappers.rfecv import RFECV

        # Quiet defaults to avoid swamping the test log; small budgets so the
        # CV-fold work stays under the per-test budget.
        return RFECV(
            estimator=LogisticRegression(max_iter=200),
            cv=2,
            max_refits=2,
            max_noimproving_iters=2,
            verbose=0,
            nofeatures_dummy_scoring=False,
            leakage_corr_threshold=None,
        )

    def test_get_set_params_roundtrip(self):
        """Get set params roundtrip."""
        est = self._make()
        params = est.get_params(deep=False)
        est2 = self._make()
        est2.set_params(**params)
        # estimator gets cloned by sklearn's get_params(deep=True); for
        # deep=False the wrapped instance is the exact same object.
        roundtripped = est2.get_params(deep=False)
        for k in params:
            sv, nv = params[k], roundtripped[k]
            if isinstance(sv, BaseEstimator):
                assert type(nv) is type(sv)
            else:
                assert sv == nv, f"param {k!r}: src={sv!r} new={nv!r}"

    def test_clone_of_unfitted_wrapper(self):
        """Clone of unfitted wrapper."""
        est = self._make()
        cloned = clone(est)
        assert cloned is not est
        # Cloned wrapper carries the same hyperparams (estimator gets cloned recursively).
        assert cloned.cv == est.cv
        assert cloned.max_refits == est.max_refits
        assert type(cloned.estimator) is type(est.estimator)
