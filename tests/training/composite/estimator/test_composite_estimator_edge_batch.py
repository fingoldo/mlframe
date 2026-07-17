"""Estimator-side FUTURE batch: E13 (update envelope refresh), E23 (robust
allowlist), E6/DX3 (grouped predict_quantile)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression

from mlframe.training.composite import CompositeTargetEstimator


class TestE13EnvelopeRefresh:
    """Groups tests covering e13 envelope refresh."""
    def test_drift_refit_refreshes_y_clip_envelope(self) -> None:
        """Drift refit refreshes y clip envelope."""
        rng = np.random.default_rng(0)
        n = 600
        b = rng.normal(0.0, 1.0, n)
        y = 2.0 * b + rng.normal(0.0, 0.1, n)
        X = pd.DataFrame({"b": b})
        est = CompositeTargetEstimator(
            base_estimator=LinearRegression(),
            transform_name="linear_residual",
            base_column="b",
            online_refit_enabled=True,
            online_refit_min_buffer_n=100,
            online_refit_z_threshold=1.0,
        ).fit(X, y)
        hi0 = est.fitted_params_["y_clip_high"]
        bd = rng.normal(5.0, 1.0, 300)
        yd = 5.0 * bd + rng.normal(0.0, 0.1, 300)
        info = est.update(yd, bd)
        assert info.get("refit") is True
        # The drifted regime (y ~ 25) must widen the clip envelope so the
        # corrected prediction is not pulled back toward the dead regime.
        assert est.fitted_params_["y_clip_high"] > hi0 * 1.2


class TestE23RobustAllowlist:
    """Groups tests covering e23 robust allowlist."""
    def test_linear_residual_robust_update_allowed(self) -> None:
        """Linear residual robust update allowed."""
        rng = np.random.default_rng(1)
        n = 400
        b = rng.normal(0.0, 1.0, n)
        y = 2.0 * b + rng.normal(0.0, 0.1, n)
        X = pd.DataFrame({"b": b})
        est = CompositeTargetEstimator(
            base_estimator=LinearRegression(),
            transform_name="linear_residual_robust",
            base_column="b",
            online_refit_enabled=True,
        ).fit(X, y)
        # Must NOT raise NotImplementedError (robust shares the alpha/beta shape).
        est.update(y[:50], b[:50])

    def test_unsupported_transform_still_rejected(self) -> None:
        """Unsupported transform still rejected."""
        rng = np.random.default_rng(2)
        b = np.abs(rng.normal(2.0, 0.5, 300)) + 0.5
        y = b * 1.5 + 0.1
        X = pd.DataFrame({"b": b})
        est = CompositeTargetEstimator(
            base_estimator=LinearRegression(),
            transform_name="ratio",
            base_column="b",
            online_refit_enabled=True,
        ).fit(X, y)
        with pytest.raises(NotImplementedError):
            est.update(y[:50], b[:50])


class _GroupedQuantileInner(BaseEstimator, RegressorMixin):
    """Inner exposing predict + predict_quantile; records the columns it sees
    so the test can assert the group column was stripped."""

    seen_cols = None

    def fit(self, X, y, **kw):
        """Fit."""
        type(self).seen_cols = list(X.columns) if hasattr(X, "columns") else None
        self.n_features_in_ = X.shape[1]
        self._m = float(np.mean(np.asarray(y, dtype=np.float64)))
        return self

    def predict(self, X):
        """Predict."""
        return np.full(X.shape[0], self._m, dtype=np.float64)

    def predict_quantile(self, X, alpha=0.5):
        """Predict quantile."""
        type(self).seen_cols = list(X.columns) if hasattr(X, "columns") else None
        n = X.shape[0]
        if np.isscalar(alpha):
            return np.full(n, self._m, dtype=np.float64)
        return np.column_stack([np.full(n, self._m + float(a) - 0.5) for a in alpha])


class TestE6GroupedPredictQuantile:
    """Groups tests covering e6 grouped predict quantile."""
    @pytest.mark.parametrize("alpha", [0.5, [0.1, 0.5, 0.9]])
    def test_grouped_quantile_does_not_crash_and_strips_group_col(self, alpha) -> None:
        """Grouped quantile does not crash and strips group col."""
        rng = np.random.default_rng(3)
        n = 600
        g = rng.integers(0, 4, n).astype(str)
        b = rng.normal(0.0, 1.0, n)
        feat = rng.normal(0.0, 1.0, n)
        y = b + 0.3 * feat + rng.normal(0.0, 0.1, n)
        X = pd.DataFrame({"b": b, "feat": feat, "grp": g})
        est = CompositeTargetEstimator(
            base_estimator=_GroupedQuantileInner(),
            transform_name="linear_residual_grouped",
            base_column="b",
            group_column="grp",
        )
        est.fit(X, y)
        out = est.predict_quantile(X, alpha)  # pre-fix: groups kwarg missing -> raise
        assert np.all(np.isfinite(out))
        # The inner must not have seen the (string) group column.
        assert "grp" not in (_GroupedQuantileInner.seen_cols or [])
