"""Comprehensive tests for mlframe.ensembling module using hypothesis."""

import pytest
import numpy as np
import pandas as pd
from hypothesis import given, strategies as st, settings, assume, HealthCheck
from hypothesis.extra.numpy import arrays

from mlframe.ensembling import (
    batch_numaggs,
    enrich_ensemble_preds_with_numaggs,
    ensemble_probabilistic_predictions,
    build_predictive_kwargs,
    compare_ensembles,
    score_ensemble,
    SIMPLE_ENSEMBLING_METHODS,
)
from unittest.mock import MagicMock


# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Strategies
# -----------------------------------------------------------------------------------------------------------------------------------------------------

prob_arrays = arrays(
    dtype=np.float32,
    shape=st.tuples(st.integers(1, 50), st.integers(2, 10)),
    elements=st.floats(0.01, 0.99, allow_nan=False, allow_infinity=False)
)

small_prob_arrays = arrays(
    dtype=np.float32,
    shape=st.tuples(st.integers(1, 20), st.integers(2, 5)),
    elements=st.floats(0.01, 0.99, allow_nan=False, allow_infinity=False)
)


# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Tests for ensemble_probabilistic_predictions
# -----------------------------------------------------------------------------------------------------------------------------------------------------

@given(preds=small_prob_arrays)
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow], deadline=None)
def test_ensemble_methods_return_valid_shape(preds):
    """All ensemble methods should return same shape as input."""
    for method in SIMPLE_ENSEMBLING_METHODS:
        result, _, _ = ensemble_probabilistic_predictions(
            preds, ensemble_method=method, verbose=False
        )
        assert result.shape == preds.shape, f"Method {method} changed shape"


@given(method=st.sampled_from(SIMPLE_ENSEMBLING_METHODS))
@settings(max_examples=20)
def test_ensemble_multiple_predictions_same_shape(method):
    """Ensembling multiple predictions with same shape should work."""
    shape = (20, 5)
    preds = [np.random.rand(*shape).astype(np.float32) * 0.98 + 0.01 for _ in range(3)]

    result, _, _ = ensemble_probabilistic_predictions(
        *preds, ensemble_method=method, verbose=False
    )
    assert result.shape == shape
    assert np.all(np.isfinite(result))


@given(preds=small_prob_arrays)
@settings(max_examples=30)
def test_ensure_prob_limits_clips_results(preds):
    """Results should be clipped to [0, 1] when ensure_prob_limits=True."""
    result, _, _ = ensemble_probabilistic_predictions(
        preds, ensemble_method="arithm", ensure_prob_limits=True, verbose=False
    )
    assert np.all((result >= 0) & (result <= 1))


def test_empty_predictions_returns_none():
    """Empty prediction list should return None."""
    result, uncertainty, confident = ensemble_probabilistic_predictions(verbose=False)
    assert result is None
    assert uncertainty is None
    assert confident is None


def test_none_predictions_filtered():
    """None predictions in list should be filtered out."""
    pred = np.random.rand(10, 3).astype(np.float32)
    result, _, _ = ensemble_probabilistic_predictions(
        pred, None, pred, ensemble_method="arithm", verbose=False
    )
    assert result is not None
    assert result.shape == pred.shape


def test_nan_handling_replaces_with_mean():
    """NaN values should be replaced with arithmetic mean."""
    # Harmonic mean will produce NaN when there's a zero
    preds = np.array([[0.0, 0.5], [0.5, 0.5]], dtype=np.float32)
    result, _, _ = ensemble_probabilistic_predictions(
        preds, ensemble_method="harm", verbose=False
    )
    assert not np.any(np.isnan(result))


@given(n_cols=st.integers(1, 10))
@settings(max_examples=20)
def test_single_prediction_returns_same_values(n_cols):
    """Single prediction should return approximately itself."""
    pred = np.random.rand(10, n_cols).astype(np.float32)
    for method in ["arithm", "median"]:
        result, _, _ = ensemble_probabilistic_predictions(
            pred, ensemble_method=method, verbose=False
        )
        np.testing.assert_array_almost_equal(result, pred, decimal=5)


def test_confidence_indices_with_uncertainty_quantile():
    """uncertainty_quantile should produce valid confident_indices."""
    preds = [np.random.rand(100, 5).astype(np.float32) for _ in range(3)]
    _, uncertainty, confident = ensemble_probabilistic_predictions(
        *preds, uncertainty_quantile=0.2, verbose=False
    )
    assert uncertainty is not None
    assert confident is not None
    assert len(confident) <= 20  # 20% of 100


def test_confidence_indices_disabled():
    """uncertainty_quantile=0 should disable confidence calculation."""
    preds = [np.random.rand(10, 3).astype(np.float32) for _ in range(2)]
    _, uncertainty, confident = ensemble_probabilistic_predictions(
        *preds, uncertainty_quantile=0, verbose=False
    )
    assert uncertainty is None
    assert confident is None


@pytest.mark.parametrize("method", SIMPLE_ENSEMBLING_METHODS)
def test_all_ensemble_methods_produce_finite_results(method):
    """All ensemble methods should produce finite results for valid input."""
    preds = [np.random.rand(20, 4).astype(np.float32) * 0.8 + 0.1 for _ in range(4)]
    result, _, _ = ensemble_probabilistic_predictions(
        *preds, ensemble_method=method, verbose=False
    )
    assert np.all(np.isfinite(result))


def test_outlier_prediction_excluded():
    """Predictions deviating too much from median should be excluded."""
    # Create 3 similar predictions and 1 outlier
    base = np.random.rand(10, 3).astype(np.float32) * 0.5 + 0.25
    preds = [
        base.copy(),
        base.copy() + 0.01,
        base.copy() - 0.01,
        np.ones_like(base)  # Outlier
    ]
    result, _, _ = ensemble_probabilistic_predictions(
        *preds, ensemble_method="arithm", max_mae=0.1, verbose=False
    )
    # Result should be closer to base than to outlier
    assert np.mean(np.abs(result - base)) < np.mean(np.abs(result - np.ones_like(base)))


# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Tests for build_predictive_kwargs
# -----------------------------------------------------------------------------------------------------------------------------------------------------

@given(is_regression=st.booleans())
def test_build_predictive_kwargs_with_none(is_regression):
    """build_predictive_kwargs should handle None inputs."""
    result = build_predictive_kwargs(None, None, None, is_regression)
    assert isinstance(result, dict)
    if is_regression:
        assert "train_preds" in result
        assert result["train_preds"] is None
    else:
        assert "train_probs" in result
        assert result["train_probs"] is None


def test_build_predictive_kwargs_classification():
    """Classification should return train_probs, test_probs, val_probs."""
    data = np.random.rand(10, 3)
    result = build_predictive_kwargs(data, data, data, is_regression=False)
    assert "train_probs" in result
    assert "test_probs" in result
    assert "val_probs" in result
    assert result["train_probs"].shape == data.shape


def test_build_predictive_kwargs_regression():
    """Regression should return flattened train_preds, test_preds, val_preds."""
    data = np.random.rand(10, 1)
    result = build_predictive_kwargs(data, data, data, is_regression=True)
    assert "train_preds" in result
    assert "test_preds" in result
    assert "val_preds" in result
    assert result["train_preds"].ndim == 1


def test_build_predictive_kwargs_with_tuple_indices():
    """build_predictive_kwargs should handle (predictions, indices) tuples."""
    preds = np.random.rand(20, 3)
    indices = np.array([0, 5, 10, 15])
    result = build_predictive_kwargs((preds, indices), None, None, is_regression=False)
    assert result["train_probs"].shape == (4, 3)


def test_build_predictive_kwargs_with_none_indices():
    """Tuple with None indices should return full predictions."""
    preds = np.random.rand(10, 3)
    result = build_predictive_kwargs((preds, None), None, None, is_regression=False)
    assert result["train_probs"].shape == preds.shape


# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Tests for enrich_ensemble_preds_with_numaggs
# -----------------------------------------------------------------------------------------------------------------------------------------------------

@given(preds=small_prob_arrays)
@settings(max_examples=20, deadline=None)
def test_enrich_ensemble_basic_returns_dataframe(preds):
    """enrich_ensemble_preds_with_numaggs should return DataFrame with correct rows."""
    result = enrich_ensemble_preds_with_numaggs(
        preds, means_only=True, keep_probs=False, n_jobs=1
    )
    assert isinstance(result, pd.DataFrame)
    assert len(result) == preds.shape[0]


def test_enrich_ensemble_keeps_probs():
    """keep_probs=True should include original probabilities."""
    preds = np.random.rand(10, 3).astype(np.float32)
    result = enrich_ensemble_preds_with_numaggs(
        preds, means_only=True, keep_probs=True, n_jobs=1
    )
    assert len(result.columns) > 5  # At least probs + aggregates
    # First columns should be probabilities
    for i in range(3):
        col = f"p{i}"
        assert col in result.columns


def test_enrich_ensemble_custom_model_names():
    """Custom model names should be used as column names."""
    preds = np.random.rand(10, 3).astype(np.float32)
    names = ["model_a", "model_b", "model_c"]
    result = enrich_ensemble_preds_with_numaggs(
        preds, models_names=names, means_only=True, keep_probs=True, n_jobs=1
    )
    for name in names:
        assert name in result.columns


def test_enrich_ensemble_means_only_columns():
    """means_only=True should return specific aggregation columns."""
    preds = np.random.rand(10, 5).astype(np.float32)
    result = enrich_ensemble_preds_with_numaggs(
        preds, means_only=True, keep_probs=False, n_jobs=1
    )
    expected_cols = ["arimean", "quadmean", "qubmean", "geomean", "harmmean"]
    for col in expected_cols:
        assert col in result.columns


def test_enrich_ensemble_mutable_default_not_shared():
    """Mutable default arguments should not be shared between calls."""
    preds1 = np.random.rand(5, 3).astype(np.float32)
    preds2 = np.random.rand(5, 3).astype(np.float32)

    # First call
    result1 = enrich_ensemble_preds_with_numaggs(
        preds1, means_only=True, keep_probs=False, n_jobs=1
    )

    # Second call should not be affected by first
    result2 = enrich_ensemble_preds_with_numaggs(
        preds2, means_only=True, keep_probs=False, n_jobs=1
    )

    assert len(result1) == 5
    assert len(result2) == 5


# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Tests for compare_ensembles
# -----------------------------------------------------------------------------------------------------------------------------------------------------

def test_compare_ensembles_empty_dict():
    """compare_ensembles should handle empty dict."""
    result = compare_ensembles({}, show_plot=False)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0


# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Edge case tests
# -----------------------------------------------------------------------------------------------------------------------------------------------------

def test_ensemble_with_two_predictions():
    """Ensembling exactly 2 predictions should skip outlier detection."""
    preds = [np.random.rand(10, 3).astype(np.float32) for _ in range(2)]
    result, _, _ = ensemble_probabilistic_predictions(
        *preds, ensemble_method="arithm", verbose=False
    )
    assert result is not None


def test_ensemble_all_same_predictions():
    """All identical predictions should return the same values."""
    pred = np.random.rand(10, 3).astype(np.float32)
    preds = [pred.copy() for _ in range(4)]

    result, _, _ = ensemble_probabilistic_predictions(
        *preds, ensemble_method="arithm", verbose=False
    )
    np.testing.assert_array_almost_equal(result, pred)


def test_geometric_mean_positive_values():
    """Geometric mean should work correctly for positive values."""
    preds = [np.random.rand(10, 3).astype(np.float32) * 0.8 + 0.1 for _ in range(3)]
    result, _, _ = ensemble_probabilistic_predictions(
        *preds, ensemble_method="geo", verbose=False
    )
    assert np.all(result > 0)
    assert np.all(np.isfinite(result))


@pytest.mark.parametrize("n_preds", [3, 5, 10])
def test_ensemble_various_prediction_counts(n_preds):
    """Ensembling should work with various numbers of predictions."""
    preds = [np.random.rand(20, 4).astype(np.float32) * 0.8 + 0.1 for _ in range(n_preds)]
    result, _, _ = ensemble_probabilistic_predictions(
        *preds, ensemble_method="arithm", verbose=False
    )
    assert result.shape == (20, 4)


def test_quadratic_mean_larger_than_arithmetic():
    """Quadratic mean should generally be >= arithmetic mean (RMS inequality)."""
    preds = [np.random.rand(100, 5).astype(np.float32) * 0.8 + 0.1 for _ in range(5)]

    quad_result, _, _ = ensemble_probabilistic_predictions(
        *preds, ensemble_method="quad", verbose=False
    )
    arith_result, _, _ = ensemble_probabilistic_predictions(
        *preds, ensemble_method="arithm", verbose=False
    )

    # On average, quadratic mean should be >= arithmetic mean
    assert np.mean(quad_result) >= np.mean(arith_result) - 0.01


def test_harmonic_mean_smaller_than_arithmetic():
    """Harmonic mean should generally be <= arithmetic mean."""
    preds = [np.random.rand(100, 5).astype(np.float32) * 0.8 + 0.1 for _ in range(5)]

    harm_result, _, _ = ensemble_probabilistic_predictions(
        *preds, ensemble_method="harm", verbose=False
    )
    arith_result, _, _ = ensemble_probabilistic_predictions(
        *preds, ensemble_method="arithm", verbose=False
    )

    # On average, harmonic mean should be <= arithmetic mean
    assert np.mean(harm_result) <= np.mean(arith_result) + 0.01


# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Tests for score_ensemble with ensembling_level
# -----------------------------------------------------------------------------------------------------------------------------------------------------

def _create_mock_model_result(n_samples=50, n_classes=3):
    """Create a mock model result object for testing score_ensemble."""
    mock = MagicMock()
    mock.train_probs = np.random.rand(n_samples, n_classes).astype(np.float32)
    mock.test_probs = np.random.rand(n_samples, n_classes).astype(np.float32)
    mock.val_probs = np.random.rand(n_samples, n_classes).astype(np.float32)
    mock.train_preds = None
    mock.test_preds = None
    mock.val_preds = None
    return mock


def _create_mock_regression_result(n_samples=50):
    """Create a mock regression result object for testing score_ensemble."""
    mock = MagicMock()
    mock.train_probs = None
    mock.test_probs = None
    mock.val_probs = None
    mock.train_preds = np.random.rand(n_samples).astype(np.float32)
    mock.test_preds = np.random.rand(n_samples).astype(np.float32)
    mock.val_preds = np.random.rand(n_samples).astype(np.float32)
    return mock


@pytest.mark.parametrize("max_level", [1, 2])
def test_score_ensemble_ensembling_levels(max_level):
    """score_ensemble should process different max_ensembling_level values."""
    from unittest.mock import patch

    # Create mock models
    models = [_create_mock_model_result() for _ in range(3)]

    # Create targets
    val_target = pd.Series(np.random.randint(0, 3, 50))
    test_target = pd.Series(np.random.randint(0, 3, 50))
    train_target = pd.Series(np.random.randint(0, 3, 50))

    # Mock train_and_evaluate_model to return a simple result
    mock_result = MagicMock()
    mock_result.metrics = {"test": {"integral_error": 0.1}, "val": {"integral_error": 0.15}}
    mock_result.train_probs = np.random.rand(50, 3).astype(np.float32)
    mock_result.test_probs = np.random.rand(50, 3).astype(np.float32)
    mock_result.val_probs = np.random.rand(50, 3).astype(np.float32)
    mock_result.train_preds = None
    mock_result.test_preds = None
    mock_result.val_preds = None

    with patch('mlframe.training.train_and_evaluate_model', return_value=mock_result):
        result = score_ensemble(
            models_and_predictions=models,
            ensemble_name="test_ensemble",
            train_target=train_target,
            test_target=test_target,
            val_target=val_target,
            max_ensembling_level=max_level,
            ensembling_methods=["arithm", "harm"],
            uncertainty_quantile=0,  # Disable confidence for simplicity
            verbose=False,
        )

    assert isinstance(result, dict)
    # Should have entries for each method and level
    expected_keys = max_level * 2  # 2 methods * max_level levels
    if max_level > 1:
        # Higher levels add " L1", " L2" suffixes
        assert any("L" in key for key in result.keys()) or len(result) >= 2


@pytest.mark.parametrize("max_level", [1, 2, 3])
def test_score_ensemble_regression_ensembling_levels(max_level):
    """score_ensemble should handle regression with different ensembling levels."""
    from unittest.mock import patch

    # Create mock regression models
    models = [_create_mock_regression_result() for _ in range(3)]

    # Create targets
    val_target = pd.Series(np.random.rand(50))
    test_target = pd.Series(np.random.rand(50))
    train_target = pd.Series(np.random.rand(50))

    # Mock train_and_evaluate_model
    mock_result = MagicMock()
    mock_result.metrics = {"test": {"mse": 0.1}, "val": {"mse": 0.15}}
    mock_result.train_probs = None
    mock_result.test_probs = None
    mock_result.val_probs = None
    mock_result.train_preds = np.random.rand(50).astype(np.float32)
    mock_result.test_preds = np.random.rand(50).astype(np.float32)
    mock_result.val_preds = np.random.rand(50).astype(np.float32)

    with patch('mlframe.training.train_and_evaluate_model', return_value=mock_result):
        result = score_ensemble(
            models_and_predictions=models,
            ensemble_name="test_regression",
            train_target=train_target,
            test_target=test_target,
            val_target=val_target,
            max_ensembling_level=max_level,
            ensembling_methods=["arithm"],  # Single method for simplicity
            uncertainty_quantile=0,
            verbose=False,
        )

    assert isinstance(result, dict)
    assert len(result) >= max_level


def test_score_ensemble_level_labeling():
    """Ensembling level > 0 should add 'L{level}' to method names."""
    from unittest.mock import patch

    models = [_create_mock_model_result() for _ in range(3)]
    val_target = pd.Series(np.random.randint(0, 3, 50))
    test_target = pd.Series(np.random.randint(0, 3, 50))
    train_target = pd.Series(np.random.randint(0, 3, 50))

    mock_result = MagicMock()
    mock_result.metrics = {"test": {}, "val": {}}
    mock_result.train_probs = np.random.rand(50, 3).astype(np.float32)
    mock_result.test_probs = np.random.rand(50, 3).astype(np.float32)
    mock_result.val_probs = np.random.rand(50, 3).astype(np.float32)
    mock_result.train_preds = None
    mock_result.test_preds = None
    mock_result.val_preds = None

    with patch('mlframe.training.train_and_evaluate_model', return_value=mock_result):
        result = score_ensemble(
            models_and_predictions=models,
            ensemble_name="test",
            train_target=train_target,
            test_target=test_target,
            val_target=val_target,
            max_ensembling_level=2,
            ensembling_methods=["arithm"],
            uncertainty_quantile=0,
            verbose=False,
        )

    # Level 0: "arithm", Level 1: "arithm L1"
    assert "arithm" in result
    assert "arithm L1" in result


@pytest.mark.parametrize("uncertainty_quantile", [0, 0.1, 0.2, 0.5])
def test_score_ensemble_uncertainty_quantile_values(uncertainty_quantile):
    """score_ensemble should handle different uncertainty_quantile values."""
    from unittest.mock import patch

    models = [_create_mock_model_result() for _ in range(3)]
    val_target = pd.Series(np.random.randint(0, 3, 50))
    test_target = pd.Series(np.random.randint(0, 3, 50))
    train_target = pd.Series(np.random.randint(0, 3, 50))

    mock_result = MagicMock()
    mock_result.metrics = {"test": {}, "val": {}}
    mock_result.train_probs = np.random.rand(50, 3).astype(np.float32)
    mock_result.test_probs = np.random.rand(50, 3).astype(np.float32)
    mock_result.val_probs = np.random.rand(50, 3).astype(np.float32)
    mock_result.train_preds = None
    mock_result.test_preds = None
    mock_result.val_preds = None

    with patch('mlframe.training.train_and_evaluate_model', return_value=mock_result):
        result = score_ensemble(
            models_and_predictions=models,
            ensemble_name="test",
            train_target=train_target,
            test_target=test_target,
            val_target=val_target,
            max_ensembling_level=1,
            ensembling_methods=["arithm"],
            uncertainty_quantile=uncertainty_quantile,
            verbose=False,
        )

    assert isinstance(result, dict)
    # With uncertainty_quantile > 0, we get additional "conf" entries
    if uncertainty_quantile > 0:
        assert any("conf" in key for key in result.keys())
    else:
        assert not any("conf" in key for key in result.keys())


@pytest.mark.parametrize("max_mae,max_std", [(0.01, 0.01), (0.1, 0.1), (0.5, 0.5)])
def test_score_ensemble_mae_std_thresholds(max_mae, max_std):
    """score_ensemble should handle different max_mae and max_std thresholds."""
    from unittest.mock import patch

    models = [_create_mock_model_result() for _ in range(4)]
    val_target = pd.Series(np.random.randint(0, 3, 50))
    test_target = pd.Series(np.random.randint(0, 3, 50))
    train_target = pd.Series(np.random.randint(0, 3, 50))

    mock_result = MagicMock()
    mock_result.metrics = {"test": {}, "val": {}}
    mock_result.train_probs = np.random.rand(50, 3).astype(np.float32)
    mock_result.test_probs = np.random.rand(50, 3).astype(np.float32)
    mock_result.val_probs = np.random.rand(50, 3).astype(np.float32)
    mock_result.train_preds = None
    mock_result.test_preds = None
    mock_result.val_preds = None

    with patch('mlframe.training.train_and_evaluate_model', return_value=mock_result):
        result = score_ensemble(
            models_and_predictions=models,
            ensemble_name="test",
            train_target=train_target,
            test_target=test_target,
            val_target=val_target,
            max_ensembling_level=1,
            ensembling_methods=["arithm"],
            max_mae=max_mae,
            max_std=max_std,
            uncertainty_quantile=0,
            verbose=False,
        )

    assert isinstance(result, dict)


def test_score_ensemble_ensure_prob_limits():
    """score_ensemble should respect ensure_prob_limits parameter."""
    from unittest.mock import patch

    models = [_create_mock_model_result() for _ in range(3)]
    val_target = pd.Series(np.random.randint(0, 3, 50))
    test_target = pd.Series(np.random.randint(0, 3, 50))
    train_target = pd.Series(np.random.randint(0, 3, 50))

    mock_result = MagicMock()
    mock_result.metrics = {"test": {}, "val": {}}
    mock_result.train_probs = np.random.rand(50, 3).astype(np.float32)
    mock_result.test_probs = np.random.rand(50, 3).astype(np.float32)
    mock_result.val_probs = np.random.rand(50, 3).astype(np.float32)
    mock_result.train_preds = None
    mock_result.test_preds = None
    mock_result.val_preds = None

    for ensure_limits in [True, False]:
        with patch('mlframe.training.train_and_evaluate_model', return_value=mock_result):
            result = score_ensemble(
                models_and_predictions=models,
                ensemble_name="test",
                train_target=train_target,
                test_target=test_target,
                val_target=val_target,
                max_ensembling_level=1,
                ensembling_methods=["arithm"],
                ensure_prob_limits=ensure_limits,
                uncertainty_quantile=0,
                verbose=False,
            )

        assert isinstance(result, dict)


def test_score_ensemble_normalize_stds_by_mean_preds():
    """score_ensemble should handle normalize_stds_by_mean_preds parameter."""
    from unittest.mock import patch

    models = [_create_mock_model_result() for _ in range(3)]
    val_target = pd.Series(np.random.randint(0, 3, 50))
    test_target = pd.Series(np.random.randint(0, 3, 50))
    train_target = pd.Series(np.random.randint(0, 3, 50))

    mock_result = MagicMock()
    mock_result.metrics = {"test": {}, "val": {}}
    mock_result.train_probs = np.random.rand(50, 3).astype(np.float32)
    mock_result.test_probs = np.random.rand(50, 3).astype(np.float32)
    mock_result.val_probs = np.random.rand(50, 3).astype(np.float32)
    mock_result.train_preds = None
    mock_result.test_preds = None
    mock_result.val_preds = None

    with patch('mlframe.training.train_and_evaluate_model', return_value=mock_result):
        result = score_ensemble(
            models_and_predictions=models,
            ensemble_name="test",
            train_target=train_target,
            test_target=test_target,
            val_target=val_target,
            max_ensembling_level=1,
            ensembling_methods=["arithm"],
            normalize_stds_by_mean_preds=True,
            uncertainty_quantile=0.2,
            verbose=False,
        )

    assert isinstance(result, dict)
