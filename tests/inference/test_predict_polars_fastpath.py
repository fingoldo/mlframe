"""Wave-2 predict-path parity Fix 1: polars fastpath at predict entry.

When every loaded model is CB / XGB sklearn-API (polars-native) the predict entry must NOT eagerly call
``to_pandas`` / ``get_pandas_view_of_polars_df`` on the input. Pre-fix ``predict_mlframe_models_suite`` and
``predict_from_models`` always converted polars -> pandas before pipeline.transform, which paid the conversion
even when the saved models could consume polars directly.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import polars as pl
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


def _build_polars_frame(n: int = 3_000, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    return pl.DataFrame(
        {
            "x0": rng.normal(size=n).astype("float32"),
            "x1": rng.normal(size=n).astype("float32"),
            "x2": rng.normal(size=n).astype("float32"),
            "y": (1.5 * rng.normal(size=n) + rng.normal(0, 0.3, n)).astype("float32"),
        }
    )


def _run_suite(df: pl.DataFrame, models_list: list[str]):
    fte = SimpleFeaturesAndTargetsExtractor(regression_targets=["y"])
    return train_mlframe_models_suite(
        df=df,
        target_name="y",
        model_name="fastpath",
        features_and_targets_extractor=fte,
        mlframe_models=list(models_list),
        verbose=0,
        output_config=OutputConfig(data_dir="", models_dir="", save_charts=False),
        composite_target_discovery_config=CompositeTargetDiscoveryConfig(enabled=False),
        baseline_diagnostics_config=BaselineDiagnosticsConfig(enabled=False),
        dummy_baselines_config=DummyBaselinesConfig(enabled=False),
        reporting_config=ReportingConfig(plot_outputs="matplotlib[png]", plot_inline_display=False),
    )


def test_predict_from_models_polars_fastpath_cb_keeps_polars():
    """All-CB suite + polars input: predict_from_models must NOT materialise the input via
    ``get_pandas_view_of_polars_df`` at entry. Mock the helper and assert call count == 0 on the predict path.
    Pre-fix the helper was called unconditionally before pipeline.transform; the fastpath now keeps polars when
    every in-memory model is CB / XGB sklearn-API."""
    pytest.importorskip("catboost")
    df = _build_polars_frame()
    models, metadata = _run_suite(df, ["cb"])
    assert models, "training returned empty models dict"

    fte = SimpleFeaturesAndTargetsExtractor(regression_targets=["y"])
    import mlframe.training.core.predict as predict_mod

    call_counter = {"n": 0}
    orig_helper = predict_mod.get_pandas_view_of_polars_df

    def _spy(_df, *a, **kw):
        call_counter["n"] += 1
        return orig_helper(_df, *a, **kw)

    with patch.object(predict_mod, "get_pandas_view_of_polars_df", side_effect=_spy):
        results = predict_from_models(
            df=df,
            models=models,
            metadata=metadata,
            features_and_targets_extractor=fte,
            return_probabilities=False,
            verbose=0,
        )
    assert results["models_used"], "no models produced predictions"
    assert call_counter["n"] == 0, (
        f"polars fastpath broken: get_pandas_view_of_polars_df was invoked {call_counter['n']} time(s) even though every loaded model is CB-native."
    )


def test_predict_from_models_polars_fastpath_xgb_keeps_polars():
    """Same as the CB test but with XGBoost (also polars-native via the sklearn wrapper)."""
    xgb = pytest.importorskip("xgboost")
    # XGBoost's QuantileDMatrix data-iterator on the 2.1.x line still
    # rejects ``polars.dataframe.frame.DataFrame`` with ``TypeError:
    # Value type is not supported for data iterator`` (verified xgb
    # 2.1.4 on Python 3.9 CI 2026-05-24). The polars-iterator path
    # only became reliable on the 3.0+ line. The fastpath sensor
    # only makes sense when XGB itself can consume polars natively;
    # skip on older XGB so the assertion target reflects production
    # rather than the version constraint.
    _xgb_ver = tuple(int(x) for x in xgb.__version__.split(".")[:2])
    if _xgb_ver < (3, 0):
        pytest.skip(
            f"xgboost {xgb.__version__} QuantileDMatrix iterator does "
            f"not accept polars frames (verified failure on 2.1.4); "
            f"native polars support stabilised in 3.0+."
        )
    df = _build_polars_frame()
    models, metadata = _run_suite(df, ["xgb"])
    assert models

    fte = SimpleFeaturesAndTargetsExtractor(regression_targets=["y"])
    import mlframe.training.core.predict as predict_mod

    call_counter = {"n": 0}
    orig_helper = predict_mod.get_pandas_view_of_polars_df

    def _spy(_df, *a, **kw):
        call_counter["n"] += 1
        return orig_helper(_df, *a, **kw)

    with patch.object(predict_mod, "get_pandas_view_of_polars_df", side_effect=_spy):
        results = predict_from_models(
            df=df,
            models=models,
            metadata=metadata,
            features_and_targets_extractor=fte,
            return_probabilities=False,
            verbose=0,
        )
    assert results["models_used"]
    assert call_counter["n"] == 0, f"polars fastpath broken for XGB: get_pandas_view_of_polars_df invoked {call_counter['n']} time(s)."


def test_predict_from_models_non_native_falls_back_to_pandas():
    """Sanity: an lgb-only suite is NOT polars-native (the LightGBM sklearn wrapper does not consume polars
    directly); the lazy-conversion path must still produce predictions."""
    pytest.importorskip("lightgbm")
    df = _build_polars_frame()
    models, metadata = _run_suite(df, ["lgb"])
    assert models

    fte = SimpleFeaturesAndTargetsExtractor(regression_targets=["y"])
    results = predict_from_models(
        df=df,
        models=models,
        metadata=metadata,
        features_and_targets_extractor=fte,
        return_probabilities=False,
        verbose=0,
    )
    assert results["models_used"]
    # Predictions still produced -- the fallback path materialises the polars view as needed.
    preds = results["ensemble_predictions"]
    assert preds is not None
    arr = np.asarray(preds)
    assert arr.shape[0] == df.height, f"prediction length {arr.shape[0]} != input rows {df.height}"
    assert np.all(np.isfinite(arr)), "non-native fallback emitted NaN/inf predictions"
