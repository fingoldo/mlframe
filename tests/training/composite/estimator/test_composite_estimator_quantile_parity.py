"""Regression + biz_value sensors for the composite predict_quantile parity pass.

- E14: predict_quantile must reuse the domain-mask/fallback + T-scale clip that
  predict() applies. A NaN / out-of-domain base at predict-time must route to
  the y_train_median fallback (not a silent NaN quantile), and an inner that
  blows the T-quantile far outside the fitted T-train envelope must be clipped
  before the inverse (so the y-scale quantile does not extrapolate wildly).
- E17: the T-clip hit counts must surface in runtime_stats_ / the callback
  payload (predict AND predict_quantile) instead of being logged-then-discarded.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sklearn.base import BaseEstimator, RegressorMixin

from mlframe.training.composite import CompositeTargetEstimator


class _ConstQuantileInner(BaseEstimator, RegressorMixin):
    """Inner predicting a fixed in-envelope T plus a quantile head."""

    def __init__(self, t_value: float = 0.0):
        self.t_value = t_value

    def fit(self, X, y, **kw):
        self.n_features_in_ = X.shape[1]
        self._mean_t = float(np.mean(np.asarray(y, dtype=np.float64)))
        return self

    def predict(self, X):
        return np.full(X.shape[0], self._mean_t, dtype=np.float64)

    def predict_quantile(self, X, alpha=0.5):
        n = X.shape[0]
        if np.isscalar(alpha):
            return np.full(n, self._mean_t, dtype=np.float64)
        return np.column_stack([np.full(n, self._mean_t + float(a) - 0.5) for a in alpha])


class _BlowupQuantileInner(BaseEstimator, RegressorMixin):
    """Inner whose quantile head returns a T far outside the train envelope.

    Used to force the T-clip to fire so the hit counts are observable.
    """

    def __init__(self, blow: float = 1.0e6):
        self.blow = blow

    def fit(self, X, y, **kw):
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        return np.full(X.shape[0], self.blow, dtype=np.float64)

    def predict_quantile(self, X, alpha=0.5):
        n = X.shape[0]
        if np.isscalar(alpha):
            return np.full(n, self.blow, dtype=np.float64)
        return np.column_stack([np.full(n, self.blow) for _ in alpha])


def _diff_frame(n: int = 300, seed: int = 0):
    rng = np.random.default_rng(seed)
    b = rng.normal(10.0, 2.0, size=n)
    feat = rng.normal(0.0, 1.0, size=n)
    y = b + 0.5 + 0.3 * feat + rng.normal(0.0, 0.05, size=n)
    X = pd.DataFrame({"b": b, "feat": feat})
    return X, y


class TestE14QuantileDomainFallback:
    """A NaN / out-of-domain base at predict must route to the fallback, not
    leak a silent NaN quantile (pre-fix predict_quantile had no domain mask)."""

    @pytest.mark.parametrize("alpha", [0.5, [0.1, 0.5, 0.9]])
    def test_nan_base_routes_to_fallback_not_nan(self, alpha) -> None:
        X, y = _diff_frame()
        est = CompositeTargetEstimator(
            base_estimator=_ConstQuantileInner(),
            transform_name="diff",
            base_column="b",
            fallback_predict="y_train_median",
        )
        est.fit(X, y)
        med = est.fitted_params_["y_train_median"]
        X_pred = X.copy()
        bad = X_pred.index[:7]
        X_pred.loc[bad, "b"] = np.nan  # out-of-domain base
        out = est.predict_quantile(X_pred, alpha)
        # Pre-fix: the NaN base flows through ``T + base`` -> NaN quantiles.
        assert np.all(np.isfinite(out)), "domain-violating rows must not be NaN"
        col0 = out if np.isscalar(alpha) else out[:, 0]
        np.testing.assert_allclose(col0[:7], med, rtol=0, atol=1e-9)

    def test_nan_fallback_mode_keeps_nan(self) -> None:
        """fallback_predict='nan' is honoured: violating rows stay NaN, the
        rest are finite (so the gate did run, it just opted into NaN)."""
        X, y = _diff_frame()
        est = CompositeTargetEstimator(
            base_estimator=_ConstQuantileInner(),
            transform_name="diff",
            base_column="b",
            fallback_predict="nan",
        )
        est.fit(X, y)
        X_pred = X.copy()
        X_pred.loc[X_pred.index[:7], "b"] = np.nan
        out = est.predict_quantile(X_pred, 0.5)
        assert np.all(np.isnan(out[:7]))
        assert np.all(np.isfinite(out[7:]))


class TestE14QuantileTClip:
    """The inner T-quantile must be T-clipped before the inverse so a blown-out
    quantile does not extrapolate the y-scale interval far past the envelope."""

    @pytest.mark.parametrize("alpha", [0.5, [0.1, 0.9]])
    def test_blown_quantile_is_t_clipped_before_inverse(self, alpha) -> None:
        X, y = _diff_frame()
        est = CompositeTargetEstimator(
            base_estimator=_BlowupQuantileInner(blow=1.0e6),
            transform_name="diff",
            base_column="b",
        )
        est.fit(X, y)
        t_high = est.fitted_params_["t_clip_high"]
        assert np.isfinite(t_high), "fitted T-clip bound must be finite for the test"
        out = est.predict_quantile(X, alpha)
        # Without the T-clip, y = T + base ~ 1e6; with it, T is bounded to the
        # envelope so the y-scale quantile stays near the train range.
        assert np.max(np.abs(out)) < 1.0e5, "blown T-quantile must be clipped"
        # And the clip must register hits in runtime_stats_.
        assert est.runtime_stats_["t_clip_high_hits"] > 0


class TestE17TClipCountsObservable:
    """T-clip hit counts surface in runtime_stats_ / the callback (E17)."""

    def test_point_predict_records_t_clip_hits(self) -> None:
        X, y = _diff_frame()
        payloads: list[dict] = []
        est = CompositeTargetEstimator(
            base_estimator=_BlowupQuantileInner(blow=1.0e6),
            transform_name="diff",
            base_column="b",
            runtime_stats_callback=payloads.append,
        )
        est.fit(X, y)
        est.predict(X)  # the blow-up inner pushes every row past the T-envelope
        rs = est.runtime_stats_
        assert "t_clip_low_hits" in rs and "t_clip_high_hits" in rs
        assert rs["t_clip_high_hits"] == len(X)
        assert payloads, "callback must fire on predict"
        last = payloads[-1]
        assert last["batch_t_clip_high_hits"] == len(X)
        assert last["cumulative_t_clip_high_hits"] == len(X)

    def test_quantile_predict_records_t_clip_hits_in_stats_and_callback(self) -> None:
        X, y = _diff_frame()
        payloads: list[dict] = []
        est = CompositeTargetEstimator(
            base_estimator=_BlowupQuantileInner(blow=1.0e6),
            transform_name="diff",
            base_column="b",
            runtime_stats_callback=payloads.append,
        )
        est.fit(X, y)
        est.predict_quantile(X, [0.1, 0.9])
        rs = est.runtime_stats_
        # Two quantile columns each clip every row -> hits accumulate across cols.
        assert rs["t_clip_high_hits"] >= 2 * len(X)
        assert rs["predict_calls"] == 1
        assert payloads, "callback must fire on predict_quantile"
        last = payloads[-1]
        assert last["batch_t_clip_high_hits"] >= 2 * len(X)
        assert last["cumulative_t_clip_high_hits"] == rs["t_clip_high_hits"]

    def test_quantile_predict_records_domain_violations(self) -> None:
        X, y = _diff_frame()
        est = CompositeTargetEstimator(
            base_estimator=_ConstQuantileInner(),
            transform_name="diff",
            base_column="b",
        )
        est.fit(X, y)
        X_pred = X.copy()
        X_pred.loc[X_pred.index[:5], "b"] = np.nan
        est.predict_quantile(X_pred, 0.5)
        assert est.runtime_stats_["domain_violation_rows"] == 5
        assert est.runtime_stats_["predict_rows_total"] == len(X_pred)
