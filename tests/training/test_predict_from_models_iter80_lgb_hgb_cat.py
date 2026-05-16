"""Regression test for iter#80: lgb+hgb mixed-model on Polars+cat input.

The iter#55 global cat-cast to pandas ``category`` made LGB happy but
broke sklearn HGB whose ``pre_pipeline`` carries a category_encoders
CatBoostEncoder step. The CatBoostEncoder + sklearn check_array chain
calls isnan on the cat column; categorical dtype trips::

    ufunc 'isnan' not supported for the input types, and the inputs
    could not be safely coerced to any supported types according to
    the casting rule 'safe'

Pre-iter#55 (no cast) cat_low arrives as pandas object/string after
the polars-to-pandas conversion. LGB then trips::

    train and valid dataset categorical_feature do not match.

Neither dtype satisfies both models. The fix is per-model dispatch:
keep the shared df with cat_low as object so HGB's CatBoostEncoder
sees what it was fitted on; in the LGB branch of the per-model loop
cast cat_features to ``category`` only for LGB models.

Sensor test: a single suite call with ``mlframe_models=['lgb', 'hgb']``
on a Polars frame containing cat_low must produce predictions for
BOTH models. Pre-fix this returns an empty models_used list (both
crashed and were caught by the outer try/except).
"""
from __future__ import annotations

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


def _build_polars_frame_with_cat(n: int = 3_000, seed: int = 0):
    rng = np.random.default_rng(seed)
    df = pl.DataFrame({
        "x0": rng.normal(size=n).astype("float32"),
        "x1": rng.normal(size=n).astype("float32"),
        "cat_low": np.array(["A", "B", "C", "D", "E"], dtype=object)[rng.integers(0, 5, n)],
        "y": (rng.uniform(0, 1, n) < 0.5).astype("int32"),
    })
    return df


def _run_suite_binary(df, models_list):
    fte = SimpleFeaturesAndTargetsExtractor(
        classification_targets=["y"],
        classification_exact_values={"y": 1},
    )
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


def test_iter80_lgb_hgb_polars_cat_both_predict_succeed():
    """Both LGB and HGB must produce predictions on Polars+cat input.

    Pre-fix LGB tripped ``categorical_feature do not match`` (cat_low
    as object) AND HGB tripped ``isnan not supported`` (cat_low as
    pandas category after iter#55 global cast). Per-model dispatch
    feeds each the dtype it was fit on.
    """
    pytest.importorskip("lightgbm")
    pytest.importorskip("polars_ds")
    df = _build_polars_frame_with_cat()
    fte = SimpleFeaturesAndTargetsExtractor(
        classification_targets=["y"],
        classification_exact_values={"y": 1},
    )

    models, metadata = _run_suite_binary(df, ["lgb", "hgb"])
    assert models, "training returned empty models dict"

    results = predict_from_models(
        df=df,
        models=models,
        metadata=metadata,
        features_and_targets_extractor=fte,
        return_probabilities=True,
        verbose=0,
    )
    assert len(results["models_used"]) >= 2, (
        f"Expected both LGB and HGB to predict; got {results['models_used']}. "
        f"Pre-fix one or both crashed silently (the iter#80 bug)."
    )
    # ``model_name`` format is ``<target_type>_<target_name>[_<pre_pipeline_cls>]``.
    # LGB has no pre_pipeline -> bare base name; HGB has Pipeline pre_pipeline
    # -> ``..._Pipeline`` suffix. Both must yield predictions plus a non-None
    # ensemble (the strong cross-model signal).
    assert results["ensemble_predictions"] is not None, (
        f"ensemble_predictions missing despite models_used={results['models_used']}"
    )
    bare = [k for k in results["probabilities"] if k != "ensemble" and not k.endswith("_Pipeline")]
    suffixed = [k for k in results["probabilities"] if k.endswith("_Pipeline")]
    assert bare and suffixed, (
        f"Expected one LGB-shaped key (no _Pipeline suffix) AND one HGB-shaped "
        f"key (with _Pipeline suffix); got bare={bare} suffixed={suffixed} "
        f"from {list(results['probabilities'].keys())}"
    )


def test_iter80_lgb_standalone_polars_cat_still_works():
    """Sanity: the iter#55 fix path (LGB standalone) still works."""
    pytest.importorskip("lightgbm")
    pytest.importorskip("polars_ds")
    df = _build_polars_frame_with_cat()
    fte = SimpleFeaturesAndTargetsExtractor(
        classification_targets=["y"],
        classification_exact_values={"y": 1},
    )

    models, metadata = _run_suite_binary(df, ["lgb"])
    assert models

    results = predict_from_models(
        df=df,
        models=models,
        metadata=metadata,
        features_and_targets_extractor=fte,
        return_probabilities=True,
        verbose=0,
    )
    assert results["models_used"], "LGB standalone predict should succeed"


def test_iter80_hgb_standalone_polars_cat_still_works():
    """Sanity: HGB standalone on the same input must keep working."""
    pytest.importorskip("polars_ds")
    df = _build_polars_frame_with_cat()
    fte = SimpleFeaturesAndTargetsExtractor(
        classification_targets=["y"],
        classification_exact_values={"y": 1},
    )

    models, metadata = _run_suite_binary(df, ["hgb"])
    assert models

    results = predict_from_models(
        df=df,
        models=models,
        metadata=metadata,
        features_and_targets_extractor=fte,
        return_probabilities=True,
        verbose=0,
    )
    assert results["models_used"], "HGB standalone predict should succeed"
