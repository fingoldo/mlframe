"""Import-only smoke test pinning the public symbols referenced by the README
quickstart, the feature examples, and the composite-targets tutorial notebook.

If a doc snippet's import path drifts (module rename, symbol moved/removed),
this test fails fast so the documentation cannot silently rot. Import-only:
no training, no fitting -- keep it fast.
"""

import importlib

import pytest


def _assert_importable(module_path: str, symbol: str) -> None:
    """Test helper: mod = importlib.import_module(module_path); assert hasattr(mod, symbol), f'{module_path!r} is missing...."""
    mod = importlib.import_module(module_path)
    assert hasattr(mod, symbol), f"{module_path!r} is missing {symbol!r}"


def test_readme_training_quickstart_symbols():
    """Readme training quickstart symbols."""
    _assert_importable("mlframe.training.core", "train_mlframe_models_suite")
    _assert_importable("mlframe.training.extractors", "SimpleFeaturesAndTargetsExtractor")


def test_readme_composite_estimator_symbol():
    """Readme composite estimator symbol."""
    _assert_importable("mlframe.training.composite", "CompositeTargetEstimator")


def test_readme_calibration_symbols():
    """Readme calibration symbols."""
    _assert_importable("mlframe.metrics", "fast_calibration_report")
    _assert_importable("mlframe.calibration.policy", "pick_best_calibrator")


def test_readme_feature_selection_symbols():
    """Readme feature selection symbols."""
    _assert_importable("mlframe.feature_selection.filters.mrmr", "MRMR")
    _assert_importable("mlframe.feature_selection.wrappers", "RFECV")


def test_readme_feature_engineering_symbols():
    """Readme feature engineering symbols."""
    _assert_importable("mlframe.feature_engineering.timeseries", "create_aggregated_features")
    # financial FE is polars-native; skip if polars is unavailable.
    pytest.importorskip("polars")
    _assert_importable("mlframe.feature_engineering.financial", "create_ohlcv_wholemarket_features")


def test_readme_inference_symbols():
    """Readme inference symbols."""
    _assert_importable("mlframe.inference.predict", "read_trained_models")
    _assert_importable("mlframe.inference.predict", "get_models_raw_predictions")
