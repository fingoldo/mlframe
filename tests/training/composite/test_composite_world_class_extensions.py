"""Beyond-audit world-class extensions: M7 multiclass + calibration, M8 CQR
(adaptive-width conformal)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scipy.stats import norm
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from mlframe.training.composite import (
    CompositeClassificationEstimator,
    CompositeTargetEstimator,
)

lgb = pytest.importorskip("lightgbm")


# ---- M7 multiclass + calibration -----------------------------------------
def _three_class_data(seed=0, n=6000):
    rng = np.random.default_rng(seed)
    s = rng.normal(0.0, 1.0, (n, 2))
    a, b = rng.normal(0, 1, n), rng.normal(0, 1, n)
    score = np.column_stack([2 * s[:, 0], 2 * s[:, 1], 1.5 * np.sign(a * b)])
    p = np.exp(score - score.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    y = np.array([rng.choice(3, p=p[i]) for i in range(n)])
    X = pd.DataFrame({"s0": s[:, 0], "s1": s[:, 1], "a": a, "b": b})
    return X, y


class TestM7Multiclass:
    def test_biz_multiclass_residual_beats_base(self) -> None:
        X, y = _three_class_data()
        tr, te = slice(0, 4000), slice(4000, None)
        base = LogisticRegression(max_iter=1000).fit(X.iloc[tr], y[tr])
        acc_b = accuracy_score(y[te], base.predict(X.iloc[te]))
        est = CompositeClassificationEstimator(
            base_estimator=lgb.LGBMClassifier(n_estimators=150, verbose=-1),
        ).fit(X.iloc[tr], y[tr])
        acc_c = accuracy_score(y[te], est.predict(X.iloc[te]))
        assert acc_c >= acc_b + 0.02, f"composite {acc_c:.3f} vs base {acc_b:.3f}"

    def test_multiclass_proba_shape_and_normalisation(self) -> None:
        X, y = _three_class_data(n=2000)
        est = CompositeClassificationEstimator(
            base_estimator=lgb.LGBMClassifier(n_estimators=50, verbose=-1),
        ).fit(X, y)
        proba = est.predict_proba(X)
        assert proba.shape == (len(X), 3)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)
        assert set(np.unique(est.predict(X))).issubset({0, 1, 2})

    def test_calibration_report_ece(self) -> None:
        X, y = _three_class_data(n=3000)
        est = CompositeClassificationEstimator(
            base_estimator=lgb.LGBMClassifier(n_estimators=80, verbose=-1),
        ).fit(X, y)
        rep = est.calibration_report(X, y, n_bins=10)
        assert 0.0 <= rep["ece"] <= 1.0
        assert rep["bin_count"].sum() == len(X)
        assert rep["bin_confidence"].shape == (10,)


# ---- M8 CQR (adaptive-width conformal) -----------------------------------
class _HeteroQuantileInner(BaseEstimator, RegressorMixin):
    """Quantile inner with input-dependent spread, for CQR testing."""

    def fit(self, X, y, **kw):
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        return np.asarray(X)[:, 0]

    def predict_quantile(self, X, alpha):
        Xa = np.asarray(X)
        mu, sig = Xa[:, 0], 0.3 + np.abs(Xa[:, 1])
        al = np.atleast_1d(alpha)
        cols = [mu + sig * norm.ppf(a) for a in al]
        return np.column_stack(cols) if np.size(al) > 1 else cols[0]


def _hetero_data(seed=1, n=4000):
    rng = np.random.default_rng(seed)
    x0, x1 = rng.normal(0, 1, n), rng.normal(0, 1, n)
    y = x0 + (0.3 + np.abs(x1)) * rng.normal(0, 1, n)
    return pd.DataFrame({"x0": x0, "x1": x1}), y, (0.3 + np.abs(x1))


class TestM8CQR:
    def test_cqr_width_is_adaptive_unlike_constant(self) -> None:
        X, y, spread = _hetero_data()
        est = CompositeTargetEstimator(
            base_estimator=_HeteroQuantileInner(),
            transform_name="additive_residual",
            base_column="x0",
        ).fit(X.iloc[:2000], y[:2000])
        est.calibrate_conformal(X.iloc[2000:3000], y[2000:3000], 0.1)
        lo0, hi0 = est.predict_interval(X.iloc[3000:], 0.1)
        est.calibrate_conformal_cqr(X.iloc[2000:3000], y[2000:3000], 0.1)
        lo, hi = est.predict_interval_cqr(X.iloc[3000:], 0.1)
        sp = spread[3000:]
        corr_const = np.corrcoef(hi0 - lo0, sp)[0, 1]
        corr_cqr = np.corrcoef(hi - lo, sp)[0, 1]
        assert abs(corr_const) < 0.2, "constant-width band must not track spread"
        assert corr_cqr > 0.8, f"CQR width must track spread (corr={corr_cqr:.2f})"

    def test_cqr_marginal_coverage(self) -> None:
        X, y, _ = _hetero_data(seed=2)
        est = CompositeTargetEstimator(
            base_estimator=_HeteroQuantileInner(),
            transform_name="additive_residual",
            base_column="x0",
        ).fit(X.iloc[:2000], y[:2000])
        est.calibrate_conformal_cqr(X.iloc[2000:3000], y[2000:3000], 0.1)
        lo, hi = est.predict_interval_cqr(X.iloc[3000:], 0.1)
        cov = float(np.mean((y[3000:] >= lo) & (y[3000:] <= hi)))
        assert cov >= 0.86, f"CQR under-covered: {cov:.3f}"

    def test_cqr_uncalibrated_raises(self) -> None:
        X, y, _ = _hetero_data(n=500)
        est = CompositeTargetEstimator(
            base_estimator=_HeteroQuantileInner(),
            transform_name="additive_residual",
            base_column="x0",
        ).fit(X, y)
        with pytest.raises(RuntimeError, match="no CQR radius"):
            est.predict_interval_cqr(X, 0.1)
