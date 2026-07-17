"""Tests for the polish round 2 features:

- Prometheus hook (``runtime_stats_callback``) on
  ``CompositeTargetEstimator``: fires per predict, receives correct
  per-batch + cumulative counters, swallows exceptions silently.
- Sample weights propagation through ``linear_residual.fit``: weighted
  alpha differs from unweighted alpha when weights bias toward a
  subset.
- Stratified MI sampling option (``mi_sample_strategy``) -- bin
  coverage is guaranteed even when y is heavy-tail.
- ``CompositeTargetEstimator.predict_quantile`` for monotonic
  transforms (``diff`` / ``linear_residual``); raises for ``ratio``
  with mixed-sign base.
- Plot helpers smoke-test (figure objects produced + no crash).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# B1 sklearn matrix marker convention -- this file runs in the multi-sklearn-version CI matrix.
pytestmark = pytest.mark.sklearn_matrix


from mlframe.training.composite import (
    CompositeTargetEstimator,
    _linear_residual_fit,
    _sample_indices,
)


# ----------------------------------------------------------------------
# Prometheus hook
# ----------------------------------------------------------------------


class TestRuntimeStatsCallback:
    @pytest.fixture
    def fitted_wrapper(self):
        rng = np.random.default_rng(0)
        n = 400
        df = pd.DataFrame(
            {
                "base": rng.normal(loc=10, scale=3, size=n),
                "x1": rng.normal(size=n),
                "x2": rng.normal(size=n),
            }
        )
        y = 0.95 * df["base"].to_numpy() + 0.5 * df["x1"].to_numpy() + rng.normal(scale=0.3, size=n)

        captured: list = []

        def callback(stats: dict) -> None:
            captured.append(stats)

        from sklearn.linear_model import Ridge

        wrapper = CompositeTargetEstimator(
            base_estimator=Ridge(alpha=1.0),
            transform_name="linear_residual",
            base_column="base",
            runtime_stats_callback=callback,
        )
        wrapper.fit(df, y)
        return wrapper, captured

    def test_callback_fires_per_predict(self, fitted_wrapper):
        wrapper, captured = fitted_wrapper
        df_test = pd.DataFrame(
            {
                "base": [10.0, 11.0, 12.0],
                "x1": [0.0, 0.0, 0.0],
                "x2": [0.0, 0.0, 0.0],
            }
        )
        captured.clear()
        wrapper.predict(df_test)
        assert len(captured) == 1
        stats = captured[0]
        assert stats["batch_n"] == 3
        assert "transform_name" in stats and stats["transform_name"] == "linear_residual"
        assert "base_column" in stats and stats["base_column"] == "base"
        assert "cumulative_predict_calls" in stats

    def test_callback_cumulative_counts(self, fitted_wrapper):
        wrapper, captured = fitted_wrapper
        captured.clear()
        df_test = pd.DataFrame(
            {
                "base": [10.0] * 5,
                "x1": [0.0] * 5,
                "x2": [0.0] * 5,
            }
        )
        wrapper.predict(df_test)
        wrapper.predict(df_test)
        wrapper.predict(df_test)
        assert len(captured) == 3
        # cumulative_predict_calls counts post-predict, so by the 3rd
        # call it should be at least 3.
        assert captured[-1]["cumulative_predict_calls"] >= 3
        assert captured[-1]["cumulative_predict_rows_total"] >= 15

    def test_callback_failure_swallowed(self, fitted_wrapper):
        wrapper, _ = fitted_wrapper
        # Replace callback with one that raises -- predict must not
        # propagate the exception.
        wrapper.runtime_stats_callback = lambda stats: (_ for _ in ()).throw(
            RuntimeError("monitoring system down"),
        )
        df_test = pd.DataFrame(
            {
                "base": [10.0],
                "x1": [0.0],
                "x2": [0.0],
            }
        )
        # Predict should still succeed; callback failure is logged
        # at DEBUG and swallowed.
        result = wrapper.predict(df_test)
        assert np.isfinite(result).all()


# ----------------------------------------------------------------------
# Sample weights through linear_residual
# ----------------------------------------------------------------------


class TestSampleWeights:
    def test_weighted_alpha_differs_from_unweighted(self) -> None:
        """Heavy weight on rows where alpha would be different
        produces a different fitted alpha than the unweighted version."""
        rng = np.random.default_rng(0)
        n = 200
        # First 100 rows: y = 1.0 * base; next 100: y = 2.0 * base.
        # Unweighted alpha ≈ 1.5 (midpoint).
        # Weighted with w=10 on first half: alpha pulls toward 1.0.
        base = rng.normal(loc=5, scale=2, size=n)
        y = np.empty(n)
        y[:100] = 1.0 * base[:100] + rng.normal(scale=0.2, size=100)
        y[100:] = 2.0 * base[100:] + rng.normal(scale=0.2, size=100)

        params_unw = _linear_residual_fit(y, base)
        weights = np.where(np.arange(n) < 100, 10.0, 1.0)
        params_w = _linear_residual_fit(y, base, sample_weight=weights)

        # Unweighted alpha midpoint-ish.
        assert 1.3 < params_unw["alpha"] < 1.7
        # Weighted alpha pulled toward 1.0 (first half dominates).
        assert params_w["alpha"] < params_unw["alpha"]
        assert 0.9 < params_w["alpha"] < 1.3

    def test_zero_weights_fallback(self) -> None:
        params = _linear_residual_fit(
            np.array([1.0, 2.0, 3.0]),
            np.array([0.5, 1.0, 1.5]),
            sample_weight=np.array([0.0, 0.0, 0.0]),
        )
        # All-zero weights -> alpha=0, beta=mean(y).
        assert params["alpha"] == 0.0
        assert abs(params["beta"] - 2.0) < 1e-9

    def test_weight_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="length"):
            _linear_residual_fit(
                np.array([1.0, 2.0]),
                np.array([0.5, 1.0]),
                sample_weight=np.array([1.0, 1.0, 1.0]),
            )


# ----------------------------------------------------------------------
# Stratified MI sampling
# ----------------------------------------------------------------------


class TestStratifiedSampling:
    def test_stratified_gives_per_bin_coverage(self) -> None:
        """Stratified-quantile sampling guarantees ~equal rows per
        quantile bin even when y is heavy-tail. Random sampling has
        proportional coverage which can starve the tail."""
        rng = np.random.default_rng(0)
        n_total = 10_000
        # 95% bulk, 5% tail.
        y = np.concatenate(
            [
                rng.normal(scale=1.0, size=int(n_total * 0.95)),
                rng.normal(scale=20.0, size=int(n_total * 0.05)),
            ]
        )
        sample_n = 1000

        idx_random = _sample_indices(
            n_total,
            sample_n,
            random_state=0,
            strategy="random",
        )
        idx_strat = _sample_indices(
            n_total,
            sample_n,
            random_state=0,
            strategy="stratified_quantile",
            y=y,
            n_strata=10,
        )
        # Both should produce arrays of approximately the requested
        # size (stratified can slightly over- or under-shoot due to
        # ceiling division per bin).
        assert abs(len(idx_random) - sample_n) <= 50
        assert abs(len(idx_strat) - sample_n) <= 100

        # Per-bin coverage on stratified must be ~equal across the 10
        # quantile bins of y.
        cuts = np.quantile(y, np.linspace(0, 1, 11)[1:-1])
        random_per_bin = np.bincount(
            np.searchsorted(cuts, y[idx_random], side="right"),
            minlength=10,
        )
        strat_per_bin = np.bincount(
            np.searchsorted(cuts, y[idx_strat], side="right"),
            minlength=10,
        )
        # Stratified per-bin counts much more uniform than random.
        assert strat_per_bin.std() < random_per_bin.std() * 0.5, f"stratified {strat_per_bin} vs random {random_per_bin}"

    def test_unknown_strategy_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown strategy"):
            _sample_indices(
                100,
                50,
                random_state=0,
                strategy="quantum_supremacy",
                y=np.zeros(100),
            )

    def test_stratified_falls_back_when_y_missing(self) -> None:
        # No y supplied -> falls back to random; check it doesn't crash.
        idx = _sample_indices(
            1000,
            100,
            random_state=0,
            strategy="stratified_quantile",
            y=None,
        )
        assert len(idx) == 100


# ----------------------------------------------------------------------
# predict_quantile
# ----------------------------------------------------------------------


from sklearn.base import BaseEstimator
from typing import Optional


class _StubQuantileRegressor(BaseEstimator):
    """Minimal sklearn-clone-friendly quantile regressor: predicts
    a constant per-quantile value. ``q_to_value`` is the per-alpha
    constant. Inherits from ``BaseEstimator`` so ``sklearn.clone``
    works (the wrapper clones inner at fit time)."""

    def __init__(self, q_to_value: Optional[dict] = None) -> None:
        self.q_to_value = q_to_value if q_to_value is not None else {0.5: 0.0}

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.zeros(len(X))

    def predict_quantile(self, X, alpha=0.5):
        return np.full(len(X), self.q_to_value.get(alpha, 0.0))


class TestPredictQuantile:
    def test_diff_quantile_inverts_correctly(self) -> None:
        """For diff: y_q = T_q + base. With stub T_q=0.5 and
        base=[10, 20, 30] expect y_q=[10.5, 20.5, 30.5]."""
        rng = np.random.default_rng(0)
        n = 200
        df = pd.DataFrame(
            {
                "base": rng.normal(loc=10, scale=3, size=n),
                "x1": rng.normal(size=n),
            }
        )
        y = df["base"].to_numpy() + rng.normal(scale=0.3, size=n)

        wrapper = CompositeTargetEstimator(
            base_estimator=_StubQuantileRegressor({0.9: 0.5}),
            transform_name="diff",
            base_column="base",
        )
        wrapper.fit(df, y)
        df_test = pd.DataFrame(
            {
                "base": [10.0, 20.0, 30.0],
                "x1": [0.0, 0.0, 0.0],
            }
        )
        y_q = wrapper.predict_quantile(df_test, alpha=0.9)
        # T_q (stub) = 0.5; inverse for diff: y = T + base.
        np.testing.assert_allclose(y_q, [10.5, 20.5, 30.5])

    def test_ratio_with_negative_base_raises(self) -> None:
        """ratio inverse y = T * base flips sign when base < 0;
        quantile preservation breaks. Must raise NotImplementedError."""
        rng = np.random.default_rng(0)
        n = 200
        df = pd.DataFrame(
            {
                "base": rng.uniform(0.5, 10.0, size=n),
                "x1": rng.normal(size=n),
            }
        )
        y = df["base"].to_numpy() * rng.uniform(0.8, 1.2, size=n)

        wrapper = CompositeTargetEstimator(
            base_estimator=_StubQuantileRegressor({0.5: 1.0}),
            transform_name="ratio",
            base_column="base",
        )
        wrapper.fit(df, y)
        df_test = pd.DataFrame(
            {
                "base": [-1.0, 1.0, 2.0],  # contains negative
                "x1": [0.0, 0.0, 0.0],
            }
        )
        with pytest.raises(NotImplementedError, match="ratio"):
            wrapper.predict_quantile(df_test, alpha=0.5)

    def test_no_predict_quantile_on_inner_raises(self) -> None:
        """Inner without predict_quantile -> NotImplementedError with
        helpful message pointing the user at compatible regressors."""
        from sklearn.linear_model import Ridge

        rng = np.random.default_rng(0)
        n = 100
        df = pd.DataFrame(
            {
                "base": rng.normal(size=n),
                "x1": rng.normal(size=n),
            }
        )
        y = rng.normal(size=n)
        wrapper = CompositeTargetEstimator(
            base_estimator=Ridge(),
            transform_name="diff",
            base_column="base",
        )
        wrapper.fit(df, y)
        with pytest.raises(NotImplementedError, match="predict_quantile"):
            wrapper.predict_quantile(df, alpha=0.5)


# ----------------------------------------------------------------------
# Plot helpers (smoke tests; figures produced, no crash)
# ----------------------------------------------------------------------


class TestPlotHelpers:
    def test_plot_target_distribution(self) -> None:
        from mlframe.training.composite.diagnostics import plot_target_distribution

        rng = np.random.default_rng(0)
        y = rng.normal(loc=10, scale=3, size=500)
        t = y - 9.5  # diff residual
        fig = plot_target_distribution(y, t)
        assert fig is not None and hasattr(fig, "savefig") and len(getattr(fig, "axes", [])) >= 1
        assert hasattr(fig, "savefig")  # matplotlib Figure

    def test_plot_qq(self) -> None:
        from mlframe.training.composite.diagnostics import plot_qq

        rng = np.random.default_rng(0)
        t = rng.normal(size=500)
        fig = plot_qq(t)
        assert fig is not None and hasattr(fig, "savefig") and len(getattr(fig, "axes", [])) >= 1

    def test_plot_linear_fit(self) -> None:
        from mlframe.training.composite.diagnostics import plot_linear_fit

        rng = np.random.default_rng(0)
        base = rng.normal(loc=10, scale=3, size=500)
        y = 0.95 * base + rng.normal(scale=0.3, size=500)
        fig = plot_linear_fit(y, base, alpha=0.95, beta=0.0)
        assert fig is not None and hasattr(fig, "savefig") and len(getattr(fig, "axes", [])) >= 1

    def test_plot_mi_gain_with_ci(self) -> None:
        from mlframe.training.composite.diagnostics import plot_mi_gain_with_ci

        specs = [
            {"name": "TVT__diff__base", "mi_gain": 0.5},
            {"name": "TVT__linear_residual__base", "mi_gain": 0.6},
            {"name": "TVT__ratio__base", "mi_gain": 0.4},
        ]
        fig = plot_mi_gain_with_ci(specs, n_bootstrap=20)
        assert fig is not None and hasattr(fig, "savefig") and len(getattr(fig, "axes", [])) >= 1

    def test_plot_per_fold_tiny_rmse(self) -> None:
        from mlframe.training.composite.diagnostics import plot_per_fold_tiny_rmse

        per_fold = {
            "spec_a": [1.2, 1.3, 1.1, 1.25],
            "spec_b": [0.9, 1.0, 0.95, 0.92],
        }
        fig = plot_per_fold_tiny_rmse(per_fold, raw_baseline=1.5)
        assert fig is not None and hasattr(fig, "savefig") and len(getattr(fig, "axes", [])) >= 1

    def test_plot_per_fold_tiny_rmse_empty(self) -> None:
        from mlframe.training.composite.diagnostics import plot_per_fold_tiny_rmse

        fig = plot_per_fold_tiny_rmse({})
        # Empty input -> graceful empty-state Figure (may have no axes); we only require a real
        # matplotlib Figure object back so callers don't AttributeError on .savefig().
        assert fig is not None and hasattr(fig, "savefig")

    def test_plot_per_family_disagreement(self) -> None:
        from mlframe.training.composite.diagnostics import plot_per_family_disagreement

        # 3 families, 4 specs.
        per_family = {
            "lightgbm": [1.0, 0.9, 1.2, 1.1],
            "xgboost": [1.1, 0.85, 1.3, 1.2],
            "catboost": [0.95, 0.9, 1.25, 1.05],
        }
        fig = plot_per_family_disagreement(per_family, spec_names=["s1", "s2", "s3", "s4"])
        assert fig is not None and hasattr(fig, "savefig") and len(getattr(fig, "axes", [])) >= 1

    def test_plot_per_family_disagreement_single_family(self) -> None:
        """One-family input -> graceful "need >= 2 families" placeholder."""
        from mlframe.training.composite.diagnostics import plot_per_family_disagreement

        fig = plot_per_family_disagreement(
            {"lightgbm": [1.0, 0.9, 1.2]},
            spec_names=["s1", "s2", "s3"],
        )
        assert fig is not None and hasattr(fig, "savefig") and len(getattr(fig, "axes", [])) >= 1

    def test_plot_alpha_stability(self) -> None:
        from mlframe.training.composite.diagnostics import plot_alpha_stability

        alphas = [0.95, 0.97, 0.93, 0.96, 0.98, 0.94, 0.95]
        fig = plot_alpha_stability(alphas, expected_alpha=0.95)
        assert fig is not None and hasattr(fig, "savefig") and len(getattr(fig, "axes", [])) >= 1

    def test_plot_alpha_stability_empty(self) -> None:
        from mlframe.training.composite.diagnostics import plot_alpha_stability

        fig = plot_alpha_stability([])
        assert fig is not None and hasattr(fig, "savefig") and len(getattr(fig, "axes", [])) >= 1

    def test_plot_predictions_vs_actual(self) -> None:
        from mlframe.training.composite.diagnostics import plot_predictions_vs_actual

        rng = np.random.default_rng(0)
        y_true = rng.normal(size=500)
        y_preds = {
            "spec_a": y_true + rng.normal(scale=0.1, size=500),
            "spec_b": y_true + rng.normal(scale=0.5, size=500),
        }
        fig = plot_predictions_vs_actual(y_true, y_preds, sample_n=500)
        assert fig is not None and hasattr(fig, "savefig") and len(getattr(fig, "axes", [])) >= 1

    def test_plot_predictions_vs_actual_size_mismatch_handled(self) -> None:
        """Mismatched y_pred size for a spec doesn't crash; that
        sub-plot just shows a placeholder error message."""
        from mlframe.training.composite.diagnostics import plot_predictions_vs_actual

        rng = np.random.default_rng(0)
        y_true = rng.normal(size=500)
        y_preds = {
            "spec_a": y_true + rng.normal(scale=0.1, size=500),
            "spec_b": rng.normal(size=300),  # wrong size
        }
        fig = plot_predictions_vs_actual(y_true, y_preds, sample_n=500)
        assert fig is not None and hasattr(fig, "savefig") and len(getattr(fig, "axes", [])) >= 1
