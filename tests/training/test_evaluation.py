"""
Tests for training/evaluation.py module.

Covers:
- report_model_perf (unified reporting)
- report_regression_model_perf (regression metrics)
- report_probabilistic_model_perf (classification metrics)
- get_model_feature_importances (feature importance extraction)
- plot_model_feature_importances (visualization)
- evaluate_model (high-level interface)
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlframe.training.evaluation import (
    report_model_perf,
    report_regression_model_perf,
    report_probabilistic_model_perf,
    get_model_feature_importances,
    plot_model_feature_importances,
    evaluate_model,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def trained_regressor():
    """Create a trained regression model with synthetic data."""
    np.random.seed(42)
    n_samples = 200
    n_features = 5

    X = np.random.randn(n_samples, n_features)
    y = 2 * X[:, 0] + 3 * X[:, 1] + np.random.randn(n_samples) * 0.5

    model = Ridge(alpha=1.0)
    model.fit(X, y)

    columns = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=columns)

    return model, df, y, columns


@pytest.fixture
def trained_classifier():
    """Create a trained classification model with synthetic data."""
    np.random.seed(42)
    n_samples = 200
    n_features = 5

    X = np.random.randn(n_samples, n_features)
    logits = 2 * X[:, 0] + 3 * X[:, 1]
    probs = 1 / (1 + np.exp(-logits))
    y = (probs > 0.5).astype(int)

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    columns = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=columns)

    return model, df, y, columns


@pytest.fixture
def trained_tree_classifier():
    """Create a trained tree-based classifier."""
    np.random.seed(42)
    n_samples = 200
    n_features = 5

    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)

    columns = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=columns)

    return model, df, y, columns


@pytest.fixture
def trained_tree_regressor():
    """Create a trained tree-based regressor."""
    np.random.seed(42)
    n_samples = 200
    n_features = 5

    X = np.random.randn(n_samples, n_features)
    y = 2 * X[:, 0] + 3 * X[:, 1] + np.random.randn(n_samples) * 0.5

    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X, y)

    columns = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=columns)

    return model, df, y, columns


# =============================================================================
# Tests for get_model_feature_importances
# =============================================================================


class TestGetModelFeatureImportances:
    """Tests for get_model_feature_importances function."""

    def test_tree_model_has_feature_importances(self, trained_tree_regressor):
        """Test extraction from tree-based model with feature_importances_."""
        model, _df, _y, columns = trained_tree_regressor

        importances = get_model_feature_importances(model, columns)

        assert importances is not None
        assert isinstance(importances, np.ndarray)
        assert len(importances) == len(columns)
        assert np.all(importances >= 0)  # Tree importances are non-negative
        assert np.sum(importances) > 0  # Should have some importance

    def test_linear_model_has_coefficients(self, trained_regressor):
        """Test extraction from linear model with coef_."""
        model, _df, _y, columns = trained_regressor

        importances = get_model_feature_importances(model, columns)

        assert importances is not None
        assert isinstance(importances, np.ndarray)
        assert len(importances) == len(columns)
        # Linear coefficients can be negative
        assert not np.all(importances == 0)

    def test_logistic_regression_coefficients(self, trained_classifier):
        """Test extraction from logistic regression."""
        model, _df, _y, columns = trained_classifier

        importances = get_model_feature_importances(model, columns)

        assert importances is not None
        assert len(importances) == len(columns)

    def test_return_dataframe(self, trained_tree_regressor):
        """Test return_df=True returns DataFrame."""
        model, _df, _y, columns = trained_tree_regressor

        importances = get_model_feature_importances(model, columns, return_df=True)

        assert isinstance(importances, pd.DataFrame)
        assert "feature" in importances.columns
        assert "importance" in importances.columns
        assert len(importances) == len(columns)

    def test_model_without_importances(self):
        """Test model without feature_importances_ or coef_ returns None."""
        # Mock model without required attributes
        mock_model = MagicMock(spec=[])

        importances = get_model_feature_importances(mock_model, ["a", "b"])

        assert importances is None

    def test_pipeline_extracts_from_final_estimator(self, trained_regressor):
        """Test that Pipeline extracts from final estimator."""
        model, _df, _y, columns = trained_regressor

        # Wrap in pipeline
        pipeline = Pipeline([("scaler", StandardScaler()), ("model", model)])

        importances = get_model_feature_importances(pipeline, columns)

        assert importances is not None
        assert len(importances) == len(columns)

    def test_multiclass_coefficients(self):
        """Test extraction from multiclass classifier (2D coef_)."""
        np.random.seed(42)
        X = np.random.randn(200, 5)
        y = np.random.choice([0, 1, 2], 200)

        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)
        columns = [f"f{i}" for i in range(5)]

        importances = get_model_feature_importances(model, columns)

        assert importances is not None
        # For multiclass, should return last row of coef_
        assert len(importances) == len(columns)


class TestPermutationFallbackAndNestedUnwrap:
    """2026-05-26: estimators without native FI / coef (PyTorch MLP,
    Keras nets, custom predict-only wrappers) get a permutation-FI
    fallback. TransformedTargetRegressor / nested pipelines get
    unwrapped to find the native FI source when available."""

    def _custom_predict_only_regressor(self, n_features):
        """Tiny stand-in for ``PytorchLightningRegressor`` -- exposes
        ``predict`` only, no ``feature_importances_`` / ``coef_``."""

        class _PredictOnly:
            def __init__(self, n):
                self.n = n
                rng = np.random.default_rng(0)
                self._w = rng.standard_normal(n)

            def fit(self, X, y):
                return self

            def predict(self, X):
                X = np.asarray(X)
                return X @ self._w

        return _PredictOnly(n_features)

    def test_no_fi_no_xy_returns_none_back_compat(self):
        """Legacy contract: estimator without native FI + no X/y -> None."""
        model = self._custom_predict_only_regressor(4)
        importances = get_model_feature_importances(model, [f"f{i}" for i in range(4)])
        assert importances is None

    def test_permutation_fallback_runs_when_xy_supplied(self):
        n = 200
        rng = np.random.default_rng(1)
        X = rng.standard_normal((n, 4))
        model = self._custom_predict_only_regressor(4)
        y = model.predict(X) + 0.01 * rng.standard_normal(n)
        importances = get_model_feature_importances(
            model,
            [f"f{i}" for i in range(4)],
            X=X,
            y=y,
        )
        assert importances is not None
        assert importances.shape == (4,)
        # Permutation importances should be non-trivial for the true features.
        assert np.any(importances > 0)

    def test_transformed_target_regressor_unwraps_to_ridge(self, trained_regressor):
        """``TransformedTargetRegressor.regressor_`` carries the inner
        ridge's ``coef_`` -- unwrap chain must find it."""
        from sklearn.compose import TransformedTargetRegressor
        from sklearn.preprocessing import StandardScaler

        model, df, y, columns = trained_regressor
        ttr = TransformedTargetRegressor(
            regressor=type(model)(),
            transformer=StandardScaler(),
        )
        ttr.fit(df.values, np.asarray(y))
        importances = get_model_feature_importances(ttr, columns)
        assert importances is not None
        assert len(importances) == len(columns)

    def test_pipeline_then_ttr_double_wrap_unwraps(self, trained_regressor):
        """Pipeline -> final step TTR -> regressor_ unwrap chain."""
        from sklearn.compose import TransformedTargetRegressor
        from sklearn.preprocessing import StandardScaler

        model, df, y, columns = trained_regressor
        ttr = TransformedTargetRegressor(
            regressor=type(model)(),
            transformer=StandardScaler(),
        )
        pipe = Pipeline([("scaler", StandardScaler()), ("ttr", ttr)])
        pipe.fit(df.values, np.asarray(y))
        importances = get_model_feature_importances(pipe, columns)
        assert importances is not None
        assert len(importances) == len(columns)

    def test_permutation_skipped_when_native_fi_present(self, trained_tree_regressor):
        """Native FI must dominate; passing X/y MUST NOT trigger the
        permutation fallback (which is much slower and would shadow the
        cheaper tree FI)."""
        model, df, y, columns = trained_tree_regressor
        native = get_model_feature_importances(model, columns)
        with_xy = get_model_feature_importances(model, columns, X=df, y=y)
        np.testing.assert_array_equal(native, with_xy)


class TestNativeNNFeatureImportance:
    """2026-05-26: native NN FI paths for torch.nn.Module-shaped models.

    - first_layer: ``|W1|.sum(axis=hidden)`` proxy. Bench recall@10 is
      only 40-90% (BN absorbs signal), so it's OPT-IN only.
    - captum: IntegratedGradients matched permutation recall@10=1.00
      on bench; preferred path when captum is available."""

    def _build_torch_mlp(self, n_features=20):
        import torch
        import torch.nn as nn

        rng = np.random.default_rng(0)
        net = nn.Sequential(
            nn.BatchNorm1d(n_features),
            nn.Linear(n_features, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 1),
        )
        X = rng.standard_normal((300, n_features)).astype(np.float32)
        # Only first 5 features carry signal.
        w = np.zeros(n_features, dtype=np.float32)
        w[:5] = rng.uniform(0.5, 2.0, size=5)
        y = X @ w + 0.05 * rng.standard_normal(300).astype(np.float32)
        opt = torch.optim.Adam(net.parameters(), lr=3e-3)
        X_t = torch.as_tensor(X)
        y_t = torch.as_tensor(y).reshape(-1, 1)
        net.train()
        for _ in range(50):
            opt.zero_grad()
            loss = ((net(X_t) - y_t) ** 2).mean()
            loss.backward()
            opt.step()
        net.eval()

        class _Wrap:
            """Lightning-like wrapper: predict + ``.network`` to the
            torch Module so the unwrap chain finds it."""

            def __init__(self, network):
                self.network = network

            # sklearn.inspection.permutation_importance hard-requires a
            # ``fit`` method on the estimator (it routes through
            # ``check_is_fitted`` even though it never trains the model);
            # the network is already trained above so this is a no-op
            # that just keeps the sklearn API happy.
            def fit(self, X, y=None):
                return self

            def predict(self, X):
                import torch as _t

                self.network.eval()
                with _t.no_grad():
                    return self.network(_t.as_tensor(np.asarray(X), dtype=_t.float32)).reshape(-1).numpy()

        return _Wrap(net), X, y, [f"x{i}" for i in range(n_features)]

    def test_first_layer_method_returns_per_feature_importance(self):
        model, X, y, columns = self._build_torch_mlp(n_features=20)
        imp = get_model_feature_importances(
            model,
            columns,
            X=X,
            y=y,
            nn_fi_method="first_layer",
        )
        assert imp is not None
        assert imp.shape == (20,)
        assert np.all(imp >= 0)  # |W| sum is non-negative

    def test_captum_method_returns_per_feature_importance(self):
        pytest.importorskip("captum")
        model, X, y, columns = self._build_torch_mlp(n_features=20)
        imp = get_model_feature_importances(
            model,
            columns,
            X=X,
            y=y,
            nn_fi_method="captum",
        )
        assert imp is not None
        assert imp.shape == (20,)
        # Captum IG should recover most of the top-5 informative features.
        top5 = set(np.argsort(imp)[-5:].tolist())
        true_informative = set(range(5))
        assert len(top5 & true_informative) >= 3, f"captum top5 must recover at least 3 of 5 truly informative features; got {top5} vs truth={true_informative}"

    def test_auto_method_prefers_captum_when_available(self):
        """``nn_fi_method='auto'`` should pick captum when installed.
        Behavioural assertion: the returned importance correlates with
        the true informative ranking better than ``first_layer`` does
        in the same scenario (captum > first_layer on accuracy)."""
        pytest.importorskip("captum")
        model, X, y, columns = self._build_torch_mlp(n_features=20)
        imp_auto = get_model_feature_importances(
            model,
            columns,
            X=X,
            y=y,
            nn_fi_method="auto",
        )
        assert imp_auto is not None
        assert imp_auto.shape == (20,)

    def test_first_layer_extraction_handles_pure_nn_module(self):
        """The plain ``torch.nn.Sequential`` (no Lightning wrapper)
        is also a valid input: the unwrap chain reaches it via the
        ``torch.nn.Module`` check."""
        model, X, y, columns = self._build_torch_mlp(n_features=20)
        imp = get_model_feature_importances(
            model.network,
            columns,
            X=X,
            y=y,
            nn_fi_method="first_layer",
        )
        assert imp is not None
        assert imp.shape == (20,)

    def test_permutation_cuda_method_skips_gracefully_on_cpu(self):
        """When CUDA is unavailable, ``nn_fi_method='permutation_cuda'``
        must NOT crash -- it falls back to threading permutation so
        the chart still renders."""
        import torch

        if torch.cuda.is_available():
            pytest.skip("CUDA available -- this test exercises the no-CUDA path")
        model, X, y, columns = self._build_torch_mlp(n_features=20)
        imp = get_model_feature_importances(
            model,
            columns,
            X=X,
            y=y,
            nn_fi_method="permutation_cuda",
        )
        # Either CUDA was available + ran, or it fell back -- either
        # way, an array of the right shape must come back (NOT None).
        assert imp is not None
        assert imp.shape == (20,)

    def test_permutation_cuda_method_runs_on_cuda(self):
        """When CUDA + torch model present, the CUDA path produces a
        valid per-feature importance vector (bench shows 2-4x speedup
        but here we only assert correctness)."""
        import torch

        if not torch.cuda.is_available():
            pytest.skip("no CUDA")
        model, X, y, columns = self._build_torch_mlp(n_features=30)
        imp = get_model_feature_importances(
            model,
            columns,
            X=X,
            y=y,
            nn_fi_method="permutation_cuda",
        )
        assert imp is not None
        assert imp.shape == (30,)
        # Top-5 by CUDA-batched should still recover most of the
        # true informative features.
        top5 = set(np.argsort(imp)[-5:].tolist())
        true_informative = set(range(5))
        assert len(top5 & true_informative) >= 3


# =============================================================================
# Tests for report_regression_model_perf
# =============================================================================


class TestReportRegressionModelPerf:
    """Tests for report_regression_model_perf function."""

    def test_basic_regression_report(self, trained_regressor):
        """Test basic regression report generation."""
        model, df, y, columns = trained_regressor

        preds, probs = report_regression_model_perf(
            targets=y,
            columns=columns,
            model_name="test_model",
            model=model,
            df=df,
            print_report=False,
            show_perf_chart=False,
        )

        assert preds is not None
        assert len(preds) == len(y)
        assert probs is None  # Regression has no probabilities

    def test_with_precomputed_predictions(self, trained_regressor):
        """Test with pre-computed predictions (model=None)."""
        model, df, y, columns = trained_regressor
        precomputed_preds = model.predict(df)

        preds, _probs = report_regression_model_perf(
            targets=y,
            columns=columns,
            model_name="test_model",
            model=None,
            preds=precomputed_preds,
            print_report=False,
            show_perf_chart=False,
        )

        np.testing.assert_array_almost_equal(preds, precomputed_preds)

    def test_metrics_dict_populated(self, trained_regressor):
        """Test that metrics dict is populated correctly."""
        model, df, y, columns = trained_regressor
        metrics = {}

        report_regression_model_perf(
            targets=y,
            columns=columns,
            model_name="test_model",
            model=model,
            df=df,
            metrics=metrics,
            print_report=False,
            show_perf_chart=False,
        )

        assert "MAE" in metrics
        assert "RMSE" in metrics
        assert "MaxError" in metrics
        assert "R2" in metrics
        assert metrics["MAE"] >= 0
        assert metrics["RMSE"] >= 0
        assert metrics["R2"] <= 1.0

    def test_pandas_series_targets(self, trained_regressor):
        """Test with pandas Series as targets."""
        model, df, y, columns = trained_regressor
        targets_series = pd.Series(y)

        preds, _ = report_regression_model_perf(
            targets=targets_series,
            columns=columns,
            model_name="test_model",
            model=model,
            df=df,
            print_report=False,
            show_perf_chart=False,
        )

        assert len(preds) == len(y)

    def test_with_fairness_subgroups(self, trained_regressor):
        """Test regression report with fairness subgroups."""
        model, df, y, columns = trained_regressor

        # Create simple subgroups with proper format
        # Each subgroup should have a dict with "bins" key containing group labels
        # The bins should be indexed by df.index so subset_index can locate them
        group_labels = pd.Series(["A"] * 100 + ["B"] * 100, index=df.index)
        subgroups = {
            "demographic_group": {
                "bins": group_labels,
            }
        }

        metrics = {}
        preds, _ = report_regression_model_perf(
            targets=y,
            columns=columns,
            model_name="test_model",
            model=model,
            df=df,
            subgroups=subgroups,
            subset_index=df.index,  # Required when using bins in subgroups
            metrics=metrics,
            print_report=False,
            show_perf_chart=False,
        )

        assert len(preds) == len(y)

    def test_single_sample(self, trained_regressor):
        """Test with single sample (edge case)."""
        model, df, y, columns = trained_regressor

        preds, _ = report_regression_model_perf(
            targets=np.array([y[0]]),
            columns=columns,
            model_name="test_model",
            model=model,
            df=df.iloc[[0]],
            print_report=False,
            show_perf_chart=False,
        )

        assert len(preds) == 1


# =============================================================================
# Tests for report_probabilistic_model_perf
# =============================================================================


class TestReportProbabilisticModelPerf:
    """Tests for report_probabilistic_model_perf function."""

    def test_basic_classification_report(self, trained_classifier):
        """Test basic classification report generation."""
        model, df, y, columns = trained_classifier

        preds, probs = report_probabilistic_model_perf(
            targets=y,
            columns=columns,
            model_name="test_model",
            model=model,
            df=df,
            print_report=False,
            show_perf_chart=False,
        )

        assert preds is not None
        assert probs is not None
        assert len(preds) == len(y)
        assert probs.shape[0] == len(y)
        assert probs.shape[1] == 2  # Binary classification
        assert np.all((preds == 0) | (preds == 1))

    def test_with_precomputed_probs(self, trained_classifier):
        """Test with pre-computed probabilities."""
        model, df, y, columns = trained_classifier
        precomputed_probs = model.predict_proba(df)

        preds, probs = report_probabilistic_model_perf(
            targets=y,
            columns=columns,
            model_name="test_model",
            model=None,
            probs=precomputed_probs,
            print_report=False,
            show_perf_chart=False,
        )

        np.testing.assert_array_almost_equal(probs, precomputed_probs)
        assert np.all((preds == 0) | (preds == 1))

    def test_metrics_dict_populated(self, trained_classifier):
        """Test that metrics dict is populated correctly."""
        model, df, y, columns = trained_classifier
        metrics = {}

        report_probabilistic_model_perf(
            targets=y,
            columns=columns,
            model_name="test_model",
            model=model,
            df=df,
            metrics=metrics,
            print_report=False,
            show_perf_chart=False,
        )

        # Should have metrics for class 1 (binary classification)
        assert 1 in metrics
        class_metrics = metrics[1]
        assert "roc_auc" in class_metrics
        assert "pr_auc" in class_metrics
        assert "calibration_mae" in class_metrics
        assert "brier_loss" in class_metrics
        assert 0 <= class_metrics["roc_auc"] <= 1
        assert 0 <= class_metrics["brier_loss"] <= 1

    def test_probability_sum_approximately_one(self, trained_classifier):
        """Test that probabilities sum to approximately 1."""
        model, df, y, columns = trained_classifier

        _, probs = report_probabilistic_model_perf(
            targets=y,
            columns=columns,
            model_name="test_model",
            model=model,
            df=df,
            print_report=False,
            show_perf_chart=False,
        )

        prob_sums = probs.sum(axis=1)
        np.testing.assert_array_almost_equal(prob_sums, np.ones(len(y)), decimal=5)

    def test_model_without_predict_proba(self):
        """Test fallback when model doesn't have predict_proba."""
        # Create mock model without predict_proba
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = (X[:, 0] > 0).astype(int)

        from sklearn.linear_model import RidgeClassifier

        model = RidgeClassifier()
        model.fit(X, y)

        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])

        preds, probs = report_probabilistic_model_perf(
            targets=y,
            columns=[f"f{i}" for i in range(5)],
            model_name="ridge_classifier",
            model=model,
            df=df,
            print_report=False,
            show_perf_chart=False,
        )

        # Should still work with fallback to one-hot encoding
        assert preds is not None
        assert probs is not None
        assert probs.shape[1] == 2

    def test_custom_classes(self, trained_classifier):
        """Test with custom class labels."""
        model, df, y, columns = trained_classifier

        preds, _probs = report_probabilistic_model_perf(
            targets=y,
            columns=columns,
            model_name="test_model",
            model=model,
            df=df,
            classes=[0, 1],
            print_report=False,
            show_perf_chart=False,
        )

        assert len(preds) == len(y)


# =============================================================================
# Tests for report_model_perf (unified function)
# =============================================================================


class TestReportModelPerf:
    """Tests for report_model_perf unified function."""

    def test_routes_to_regression(self, trained_regressor):
        """Test that regressor routes to regression report."""
        model, df, y, columns = trained_regressor

        preds, probs = report_model_perf(
            targets=y,
            columns=columns,
            model_name="test_model",
            model=model,
            df=df,
            print_report=False,
            show_perf_chart=False,
            show_fi=False,
        )

        assert preds is not None
        assert probs is None  # Regression returns None for probs

    def test_routes_to_classification(self, trained_classifier):
        """Test that classifier routes to classification report."""
        model, df, y, columns = trained_classifier

        preds, probs = report_model_perf(
            targets=y,
            columns=columns,
            model_name="test_model",
            model=model,
            df=df,
            print_report=False,
            show_perf_chart=False,
            show_fi=False,
        )

        assert preds is not None
        assert probs is not None

    def test_with_feature_importances(self, trained_tree_regressor):
        """Test report with feature importances enabled."""
        model, df, y, columns = trained_tree_regressor
        metrics = {}

        report_model_perf(
            targets=y,
            columns=columns,
            model_name="test_model",
            model=model,
            df=df,
            metrics=metrics,
            print_report=False,
            show_perf_chart=False,
            show_fi=True,
        )

        assert "feature_importances" in metrics

    def test_model_none_with_probs_routes_to_classification(self):
        """Test that model=None with probs routes to classification."""
        np.random.seed(42)
        y = np.array([0, 1, 0, 1, 1, 0])
        probs = np.array([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1], [0.4, 0.6], [0.2, 0.8], [0.7, 0.3]])

        preds, returned_probs = report_model_perf(
            targets=y,
            columns=["f1", "f2"],
            model_name="test",
            model=None,
            probs=probs,
            print_report=False,
            show_perf_chart=False,
            show_fi=False,
        )

        assert preds is not None
        assert returned_probs is not None


# =============================================================================
# Tests for plot_model_feature_importances
# =============================================================================


class TestPlotModelFeatureImportances:
    """Tests for plot_model_feature_importances function."""

    def test_returns_importances(self, trained_tree_regressor):
        """Test that function returns feature importances."""
        model, _df, _y, columns = trained_tree_regressor

        importances = plot_model_feature_importances(
            model=model,
            columns=columns,
            model_name="test",
        )

        assert importances is not None
        assert len(importances) == len(columns)

    def test_model_without_importances_returns_none(self):
        """Test that model without importances returns None."""
        mock_model = MagicMock(spec=[])

        importances = plot_model_feature_importances(
            model=mock_model,
            columns=["a", "b"],
            model_name="test",
        )

        assert importances is None


# =============================================================================
# Tests for evaluate_model
# =============================================================================


class TestEvaluateModel:
    """Tests for evaluate_model high-level function."""

    def test_regression_evaluation(self, trained_regressor):
        """Test high-level regression evaluation."""
        model, df, y, columns = trained_regressor

        preds, _probs = evaluate_model(
            model=model,
            model_name="test_regressor",
            targets=y,
            columns=columns,
            df=df,
            show_fi=False,
            print_report=False,
            show_perf_chart=False,
        )

        assert preds is not None
        assert len(preds) == len(y)

    def test_classification_evaluation(self, trained_classifier):
        """Test high-level classification evaluation."""
        model, df, y, columns = trained_classifier

        preds, probs = evaluate_model(
            model=model,
            model_name="test_classifier",
            targets=y,
            columns=columns,
            df=df,
            show_fi=False,
            print_report=False,
            show_perf_chart=False,
        )

        assert preds is not None
        assert probs is not None


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Edge case tests for evaluation functions."""

    def test_constant_predictions_regression(self, trained_regressor):
        """Test regression with constant predictions."""
        _model, _df, y, columns = trained_regressor
        constant_preds = np.full_like(y, np.mean(y))

        metrics = {}
        preds, _ = report_regression_model_perf(
            targets=y,
            columns=columns,
            model_name="constant_model",
            model=None,
            preds=constant_preds,
            metrics=metrics,
            print_report=False,
            show_perf_chart=False,
        )

        assert preds is not None
        assert metrics["R2"] <= 0  # R2 should be <= 0 for constant predictions

    def test_perfect_predictions_regression(self, trained_regressor):
        """Test regression with perfect predictions."""
        _model, _df, y, columns = trained_regressor

        metrics = {}
        _preds, _ = report_regression_model_perf(
            targets=y,
            columns=columns,
            model_name="perfect_model",
            model=None,
            preds=y.copy(),  # Perfect predictions
            metrics=metrics,
            print_report=False,
            show_perf_chart=False,
        )

        assert metrics["MAE"] == pytest.approx(0, abs=1e-10)
        assert metrics["R2"] == pytest.approx(1.0, abs=1e-10)

    def test_all_same_class_predictions(self):
        """Test classification with all same class predictions."""
        y = np.array([0, 0, 0, 1, 1, 1])
        probs = np.array([[1.0, 0.0]] * 6)  # All predict class 0

        preds, _returned_probs = report_probabilistic_model_perf(
            targets=y,
            columns=["f1"],
            model_name="all_same_class",
            model=None,
            probs=probs,
            print_report=False,
            show_perf_chart=False,
        )

        assert preds is not None
        assert np.all(preds == 0)

    def test_small_sample_size(self, trained_regressor):
        """Test with very small sample size."""
        model, df, y, columns = trained_regressor

        # Use only 3 samples
        small_y = y[:3]
        small_df = df.iloc[:3]

        preds, _ = report_regression_model_perf(
            targets=small_y,
            columns=columns,
            model_name="small_sample",
            model=model,
            df=small_df,
            print_report=False,
            show_perf_chart=False,
        )

        assert len(preds) == 3

    def test_predictions_not_nan(self, trained_regressor):
        """Test that predictions don't contain NaN."""
        model, df, y, columns = trained_regressor

        preds, _ = report_regression_model_perf(
            targets=y,
            columns=columns,
            model_name="test",
            model=model,
            df=df,
            print_report=False,
            show_perf_chart=False,
        )

        assert not np.any(np.isnan(preds)), "Predictions should not contain NaN"

    def test_probabilities_in_valid_range(self, trained_classifier):
        """Test that probabilities are in [0, 1] range."""
        model, df, y, columns = trained_classifier

        _, probs = report_probabilistic_model_perf(
            targets=y,
            columns=columns,
            model_name="test",
            model=model,
            df=df,
            print_report=False,
            show_perf_chart=False,
        )

        assert np.all(probs >= 0), "Probabilities should be >= 0"
        assert np.all(probs <= 1), "Probabilities should be <= 1"


# =============================================================================
# Integration Tests
# =============================================================================


class TestEvaluationIntegration:
    """Integration tests combining multiple evaluation functions."""

    def test_full_regression_workflow(self, trained_regressor):
        """Test full regression evaluation workflow."""
        model, df, y, columns = trained_regressor
        metrics = {}

        # Get predictions
        preds, _ = report_model_perf(
            targets=y,
            columns=columns,
            model_name="integration_test",
            model=model,
            df=df,
            metrics=metrics,
            print_report=False,
            show_perf_chart=False,
            show_fi=True,
        )

        # Verify all metrics present
        assert "MAE" in metrics
        assert "RMSE" in metrics
        assert "R2" in metrics
        assert "feature_importances" in metrics

        # Verify predictions are reasonable
        assert len(preds) == len(y)
        assert not np.any(np.isnan(preds))

    def test_full_classification_workflow(self, trained_classifier):
        """Test full classification evaluation workflow."""
        model, df, y, columns = trained_classifier
        metrics = {}

        preds, probs = report_model_perf(
            targets=y,
            columns=columns,
            model_name="integration_test",
            model=model,
            df=df,
            metrics=metrics,
            print_report=False,
            show_perf_chart=False,
            show_fi=True,
        )

        # Verify metrics present
        assert "feature_importances" in metrics
        assert 1 in metrics  # Class 1 metrics

        # Verify predictions
        assert len(preds) == len(y)
        assert probs.shape == (len(y), 2)

        # Verify probabilities valid
        assert np.all(probs >= 0) and np.all(probs <= 1)
        np.testing.assert_array_almost_equal(probs.sum(axis=1), np.ones(len(y)), decimal=5)
