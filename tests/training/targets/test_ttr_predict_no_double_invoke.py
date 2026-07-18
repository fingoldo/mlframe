"""Regression test for iter191: ``_TTRWithEvalSetScaling.predict`` must invoke
the inner regressor ONCE per call, not twice.

Bug: prior form called ``self.regressor_.predict(X)`` for the
extrapolation-sensor probe AND then ``super().predict(X)`` (which calls
``self.regressor_.predict(X)`` AGAIN) for the actual return. c0115 iter191
profile attributed 1.222s / 2 calls (611ms each on 200k rows) to the
double-invocation -- on a 200k MLP that's ~600ms of waste per predict call.

Fix: inline the parent's ``inverse_transform`` path after the single predict
call so the sensor probe and the return value reuse the same t_hat.
"""

import numpy as np


class _CountingRegressor:
    """Mock regressor that counts predict calls so the test can assert
    the wrapper invokes the inner exactly once per call."""

    def __init__(self):
        self.predict_call_count = 0
        self.fit_calls = 0

    # sklearn clone needs get_params / set_params.
    def get_params(self, deep=True):
        """Get params."""
        return {}

    def set_params(self, **kwargs):
        """Set params."""
        return self

    def fit(self, X, y, **fit_params):
        """Fit."""
        self.fit_calls += 1
        self._n_features = X.shape[1]
        return self

    def predict(self, X, **predict_params):
        """Predict."""
        self.predict_call_count += 1
        # Return zeros in scaled space; TTR.inverse_transform maps back to mean(y).
        return np.zeros(len(X), dtype=np.float64)


def _make_data(n=300, seed=0):
    """Make data."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    y = 0.95 * X[:, 0] * 100.0 + 11500.0 + rng.normal(scale=10.0, size=n)
    return X, y


def test_ttr_predict_invokes_inner_exactly_once():
    """Single call to ``ttr.predict(X)`` MUST hit the inner regressor's
    predict exactly once. Pre-fix it hit it TWICE (sensor + super.predict).

    NOTE: sklearn's TransformedTargetRegressor.fit clones the regressor into
    ``self.regressor_`` (NOT the original ``inner``). Counter lives on the
    fitted clone, accessed via ``ttr.regressor_``.
    """
    from sklearn.preprocessing import StandardScaler
    from mlframe.training.targets._ttr_eval_set_scaling import _TTRWithEvalSetScaling

    X, y = _make_data(n=300)
    ttr = _TTRWithEvalSetScaling(regressor=_CountingRegressor(), transformer=StandardScaler())
    ttr.fit(X, y)
    fitted_inner = ttr.regressor_  # cloned + fitted instance
    # Reset counter after fit -- some sklearn versions call predict during
    # CV / scoring at fit-time.
    fitted_inner.predict_call_count = 0

    preds = ttr.predict(X[:50])

    assert fitted_inner.predict_call_count == 1, (
        f"_TTRWithEvalSetScaling.predict called fitted inner regressor "
        f"{fitted_inner.predict_call_count} times; expected exactly 1. "
        f"The previous form double-invoked (sensor probe + super().predict). "
        f"Each duplicate predict on 200k rows costs ~600ms of pure waste."
    )
    # Sanity: result is still the inverse-transformed prediction.
    assert preds.shape == (50,)
    assert np.allclose(preds, np.full(50, float(np.mean(y))), rtol=1e-6)


def test_ttr_predict_passthrough_when_no_transformer():
    """The no-transformer branch (transformer_ is None) MUST still work and
    still only invoke the inner once."""
    from mlframe.training.targets._ttr_eval_set_scaling import _TTRWithEvalSetScaling

    X, y = _make_data(n=200)
    ttr = _TTRWithEvalSetScaling(regressor=_CountingRegressor(), transformer=None)
    ttr.fit(X, y)
    fitted_inner = ttr.regressor_
    fitted_inner.predict_call_count = 0
    preds = ttr.predict(X[:10])
    # super().predict() in the no-transformer branch goes to TransformedTargetRegressor
    # which still calls inner.predict once.
    assert fitted_inner.predict_call_count == 1
    assert preds.shape == (10,)


def test_ttr_predict_2d_output_squeeze():
    """2-D inner-predict output of shape (n, 1) must be squeezed back to 1-D
    (mirrors sklearn's TransformedTargetRegressor.predict squeeze behaviour)."""
    from sklearn.preprocessing import StandardScaler
    from mlframe.training.targets._ttr_eval_set_scaling import _TTRWithEvalSetScaling

    class _2DOutputRegressor(_CountingRegressor):
        """Groups tests covering 2 d output regressor."""
        def predict(self, X, **predict_params):
            """Predict."""
            self.predict_call_count += 1
            return np.zeros((len(X), 1), dtype=np.float64)

    X, y = _make_data(n=200)
    ttr = _TTRWithEvalSetScaling(regressor=_2DOutputRegressor(), transformer=StandardScaler())
    ttr.fit(X, y)
    fitted_inner = ttr.regressor_
    fitted_inner.predict_call_count = 0
    preds = ttr.predict(X[:10])
    assert preds.ndim == 1
    assert preds.shape == (10,)
    assert fitted_inner.predict_call_count == 1
