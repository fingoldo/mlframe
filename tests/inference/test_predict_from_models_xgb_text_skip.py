"""Regression test for iter#52:

predict_from_models passed the full input frame (including text_col) to
each model's predict(). XGB / LGB sklearn wrappers reject object-dtype
columns at predict time:
    DataFrame.dtypes for data must be int, float, bool or category.
    When categorical type is supplied, the experimental DMatrix parameter
    `enable_categorical` must be set to `True`. Invalid columns: text_col: object

CatBoost accepts the same frame because trained CB models retain
text_features metadata and auto-build a Pool. The asymmetry between
frameworks meant predict_from_models() silently caught the exception per
model (logger.error + continue), so an XGB-only suite returned a
predictions dict that LOOKED ok but was actually empty.

The fix subsets input_for_model to the model's own feature_names_in_
(sklearn API used by XGB/LGB) or feature_names_ (CatBoost) before
calling predict, dropping framework-incompatible columns symmetrically
with the training-side per-strategy column selection.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.training.core import train_mlframe_models_suite
from mlframe.training.core.predict import predict_from_models
from mlframe.training.configs import (
    BaselineDiagnosticsConfig,
    CompositeTargetDiscoveryConfig,
    DummyBaselinesConfig,
    OutputConfig,
    ReportingConfig,
)
from mlframe.training.extractors import SimpleFeaturesAndTargetsExtractor


def _build_frame_with_text(n: int = 3_000, seed: int = 0):
    """Frame with numeric + low-card cat + 4-word text col + y."""
    rng = np.random.default_rng(seed)
    _vocab = np.array("alpha beta gamma delta epsilon zeta".split(), dtype=object)
    _idx = rng.integers(0, len(_vocab), (n, 4))
    df = pd.DataFrame(
        {
            "x0": rng.normal(size=n).astype("float32"),
            "x1": rng.normal(size=n).astype("float32"),
            "cat_low": np.array(["A", "B", "C"], dtype=object)[rng.integers(0, 3, n)],
            "text_col": np.array([" ".join(_vocab[r]) for r in _idx], dtype=object),
        }
    )
    df["y"] = (1.5 * df["x0"] - 1.0 * df["x1"] + rng.normal(0, 0.3, n)).astype("float32")
    return df


def _run_suite(df: pd.DataFrame, models_list):
    """Returns ``train_mlframe_models_suite(df=df, target_name='y', model_name='prof', features_and_targ...`` (after 1 setup step)."""
    fte = SimpleFeaturesAndTargetsExtractor(regression_targets=["y"])
    return train_mlframe_models_suite(
        df=df,
        target_name="y",
        model_name="prof",
        features_and_targets_extractor=fte,
        mlframe_models=list(models_list),
        verbose=0,
        output_config=OutputConfig(data_dir="", models_dir="", save_charts=False),
        composite_target_discovery_config=CompositeTargetDiscoveryConfig(enabled=False),
        baseline_diagnostics_config=BaselineDiagnosticsConfig(enabled=False),
        dummy_baselines_config=DummyBaselinesConfig(enabled=False),
        reporting_config=ReportingConfig(plot_outputs="matplotlib[png]", plot_inline_display=False),
    )


def test_predict_from_models_xgb_with_text_col():
    """Training+predict pipeline must complete without text_col-related
    framework errors. predict_from_models must return ``models_used``
    containing the XGB regressor (it did not pre-fix, because XGB.predict
    raised on object text_col and the silent except dropped it)."""
    pytest.importorskip("xgboost")
    df = _build_frame_with_text()
    fte = SimpleFeaturesAndTargetsExtractor(regression_targets=["y"])

    models, metadata = _run_suite(df, ["xgb"])
    assert models, "training returned empty models dict"

    results = predict_from_models(
        df=df,
        models=models,
        metadata=metadata,
        features_and_targets_extractor=fte,
        return_probabilities=False,
        verbose=0,
    )
    assert results["models_used"], "predict_from_models returned no successful models — XGB likely crashed on text_col (the iter#52 bug)"
    assert results["ensemble_predictions"] is not None, "ensemble_predictions missing; expected XGB single-model fallback"


def test_predict_from_models_lgb_with_text_col():
    """Same contract for LGB. LGB sklearn rejects object dtypes the same
    way as XGB."""
    pytest.importorskip("lightgbm")
    df = _build_frame_with_text()
    fte = SimpleFeaturesAndTargetsExtractor(regression_targets=["y"])

    models, metadata = _run_suite(df, ["lgb"])
    assert models, "training returned empty models dict"

    results = predict_from_models(
        df=df,
        models=models,
        metadata=metadata,
        features_and_targets_extractor=fte,
        return_probabilities=False,
        verbose=0,
    )
    assert results["models_used"], "predict_from_models returned no successful models — LGB likely crashed on text_col"
