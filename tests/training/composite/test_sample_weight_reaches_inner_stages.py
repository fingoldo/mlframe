"""sample_weight must reach every inner fit of the staged composite estimators, not only the final stage."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression

from mlframe.training.composite._heteroscedastic import HeteroscedasticCompositeEstimator
from mlframe.training.composite.chained_window_forecast import ChainedWindowForecaster
from mlframe.training.composite.stacking_multi_stage import MultiStageMetaFeatureStacker


class _WeightRecorder(BaseEstimator, RegressorMixin):
    """Mean regressor that records the sample_weight of every fit call (None when unweighted)."""

    calls: list = []

    def fit(self, X, y, sample_weight=None):
        """Record the weight vector length and sum, then fit a constant."""
        type(self).calls.append(None if sample_weight is None else (len(sample_weight), float(np.sum(sample_weight))))
        self.mean_ = float(np.average(y, weights=sample_weight))
        return self

    def predict(self, X):
        """Predict the weighted training mean."""
        return np.full(len(X), self.mean_)


def _frame(n: int = 60) -> pd.DataFrame:
    """Small numeric frame."""
    rng = np.random.default_rng(0)
    return pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})


def test_chained_stage1_receives_sample_weight_with_extra_rows_padded() -> None:
    """Stage 1 is fit with the caller's weights (ones for transductive extra rows), not unweighted."""
    _WeightRecorder.calls = []
    X = _frame()
    extra = _frame(20)
    w = np.full(len(X), 2.0)
    m = ChainedWindowForecaster(stage1_estimator=_WeightRecorder(), stage2_estimator=_WeightRecorder())
    m.fit(X, X, np.arange(60.0), np.arange(60.0), sample_weight=w, X_prev_extra=extra, y_curr_extra=np.arange(20.0))
    assert _WeightRecorder.calls[0] == (80, 120.0 + 20.0)
    assert _WeightRecorder.calls[1] == (60, 120.0)


def test_multi_stage_stacker_stage1_receives_sample_weight() -> None:
    """Stage-1 OOF folds and the full refit are weighted; before, only stage 2 saw the weights."""
    _WeightRecorder.calls = []
    X = _frame(50)
    w = np.linspace(1.0, 2.0, 50)
    st = MultiStageMetaFeatureStacker(
        stage1_estimator_factories={"aux": _WeightRecorder}, stage2_estimator=_WeightRecorder(), n_splits=5, quantile_transform=False,
    )
    st.fit(X, np.arange(50.0), {"aux": np.arange(50.0)}, sample_weight=w)
    assert _WeightRecorder.calls, "no fits recorded"
    assert all(c is not None for c in _WeightRecorder.calls)
    assert (50, float(w.sum())) in _WeightRecorder.calls


def test_heteroscedastic_variance_head_receives_sample_weight() -> None:
    """The second-model variance head is fit with the (finite-row) sample weights and calibration is weighted."""
    _WeightRecorder.calls = []
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"a": rng.normal(size=80), "b": rng.normal(size=80)})
    y = np.exp(0.3 * X["a"].to_numpy() + 0.3 * rng.normal(size=80))
    w = np.linspace(1.0, 3.0, 80)
    est = HeteroscedasticCompositeEstimator(
        base_estimator=LinearRegression(), variance_estimator=_WeightRecorder(), transform_name="log_y", prefer_ngboost=False,
    )
    est.fit(X, y, sample_weight=w)
    assert _WeightRecorder.calls == [(80, float(w.sum()))]


def test_heteroscedastic_calibration_is_weight_sensitive() -> None:
    """The global sigma calibration factor changes when the weights concentrate on high-residual rows."""
    resid = np.array([1.0, 1.0, 1.0, 5.0])
    sigma = np.ones(4)
    plain = HeteroscedasticCompositeEstimator._fit_calibration(resid, sigma)
    heavy = HeteroscedasticCompositeEstimator._fit_calibration(resid, sigma, np.array([1.0, 1.0, 1.0, 100.0]))
    assert heavy > plain * 1.5
    assert HeteroscedasticCompositeEstimator._fit_calibration(resid, sigma, np.ones(4)) == plain


def test_dual_direction_oof_scale_score_is_weighted() -> None:
    """oof_scale_score_ is the sample-weighted R^2 of the stored OOF scale predictions, not the unweighted one."""
    from sklearn.linear_model import Ridge

    from mlframe.metrics.regression._regression_metrics import fast_r2_score
    from mlframe.training.composite.dual_direction import DualDirectionCompositeEstimator

    rng = np.random.default_rng(6)
    n = 200
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    scale_y = 5.0 + 2.0 * X["a"].to_numpy() + np.where(X["b"].to_numpy() > 0, 3.0 * rng.normal(size=n), 0.05 * rng.normal(size=n))
    y = scale_y * (1.0 + 0.1 * rng.normal(size=n)) + 0.0
    w_noisy = np.where(X["b"].to_numpy() > 0, 30.0, 1.0)
    plain = DualDirectionCompositeEstimator(scale_estimator=Ridge(), shape_estimator=Ridge(), random_state=0).fit(X, y, scale_y)
    weighted = DualDirectionCompositeEstimator(scale_estimator=Ridge(), shape_estimator=Ridge(), random_state=0).fit(X, y, scale_y, sample_weight=w_noisy)
    expected = fast_r2_score(scale_y, weighted.oof_scale_predictions_, sample_weight=w_noisy)
    assert weighted.oof_scale_score_ == expected
    assert expected != fast_r2_score(scale_y, weighted.oof_scale_predictions_)
    assert plain.oof_scale_score_ == fast_r2_score(scale_y, plain.oof_scale_predictions_)
