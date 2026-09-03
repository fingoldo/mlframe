"""TRAINING_COMPOSITE_ENSEMBLE_ESTIMATOR_TRANSFORMS-1 (2026-08-05 audit): predict_quantile() never
called _soft_shrink.compute, so the default-ON out-of-range-base soft-shrink/smart-fallback protection
that predict() applies was silently absent from every quantile/interval prediction -- an out-of-range
base at predict-quantile time hit the raw (unshrunk) additive inverse and could explode, while predict()
on the SAME row would have smoothly shrunk it. Fixed by wiring _soft_shrink.compute/apply_smart_fallback/
record_info into predict_quantile, mirroring _predict_unclipped exactly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

from mlframe.training.composite import CompositeTargetEstimator


class _ConstQuantileInner(BaseEstimator, RegressorMixin):
    """Inner predicting a fixed in-envelope T plus a matching quantile head."""

    def fit(self, X, y, **kw):
        """Fit."""
        self.n_features_in_ = X.shape[1]
        self._mean_t = float(np.mean(np.asarray(y, dtype=np.float64)))
        return self

    def predict(self, X):
        """Predict."""
        return np.full(X.shape[0], self._mean_t, dtype=np.float64)

    def predict_quantile(self, X, alpha=0.5):
        """Predict quantile."""
        n = X.shape[0]
        if np.isscalar(alpha):
            return np.full(n, self._mean_t, dtype=np.float64)
        return np.column_stack([np.full(n, self._mean_t) for _ in alpha])


def _fit_linear_residual_quantile(seed=0, n=400, slope=3.0, base_lo=0.0, base_hi=10.0):
    """Fit a CompositeTargetEstimator (linear_residual transform, in-range base) with a quantile-capable inner."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(base_lo, base_hi, n)
    y = slope * base + rng.normal(0.0, 0.3, n)
    X = pd.DataFrame({"base": base, "f": rng.normal(0.0, 1.0, n)})
    est = CompositeTargetEstimator(
        base_estimator=_ConstQuantileInner(),
        transform_name="linear_residual",
        base_column="base",
    )
    est.fit(X, y)
    return est


def test_predict_quantile_out_of_range_base_shrinks_same_as_predict():
    """An out-of-range base row must be soft-shrunk identically whether reached via predict() or
    predict_quantile(alpha=0.5) -- before the fix, predict_quantile skipped the shrink entirely and
    used the raw (unshrunk) base, diverging from predict()'s value on the exact same row."""
    est = _fit_linear_residual_quantile()
    assert est.soft_base_shrink is True, "soft_base_shrink must default ON for this test to be meaningful"

    Xoor = pd.DataFrame({"base": [5.0, 200.0], "f": [0.0, 0.0]})
    p_predict = est.predict(Xoor)
    p_quantile = est.predict_quantile(Xoor, alpha=0.5)

    assert p_predict[0] == p_quantile[0], "in-range row must match"
    assert abs(p_predict[1] - p_quantile[1]) < 1e-6, (
        f"out-of-range row must be shrunk identically via predict() ({p_predict[1]}) and "
        f"predict_quantile() ({p_quantile[1]}), not diverge because predict_quantile skipped the shrink"
    )


def test_predict_quantile_disabled_soft_shrink_matches_raw_inverse():
    """With soft_base_shrink=False, predict_quantile must still match the raw (unshrunk) inverse --
    confirms the fix doesn't accidentally force shrinking on."""
    est = _fit_linear_residual_quantile()
    Xoor = pd.DataFrame({"base": [5.0, 200.0], "f": [0.0, 0.0]})

    est.soft_base_shrink = False
    p_off = est.predict_quantile(Xoor, alpha=0.5)
    est.soft_base_shrink = True
    p_on = est.predict_quantile(Xoor, alpha=0.5)

    assert p_off[0] == p_on[0], "in-range row unaffected by the flag"
    assert p_off[1] != p_on[1], "out-of-range row must differ between shrink on/off (mechanism is active)"


def test_predict_quantile_vector_alpha_also_gets_soft_shrink():
    """The vector-alpha (multi-quantile) path must apply the same soft-shrink as the scalar path."""
    est = _fit_linear_residual_quantile()
    Xoor = pd.DataFrame({"base": [200.0], "f": [0.0]})

    p_scalar = est.predict_quantile(Xoor, alpha=0.5)
    p_vector = est.predict_quantile(Xoor, alpha=[0.1, 0.5, 0.9])

    assert p_vector.shape == (1, 3)
    assert abs(p_scalar[0] - p_vector[0, 1]) < 1e-6, "the vector path's median column must match the scalar path"
