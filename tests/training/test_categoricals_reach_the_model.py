"""Categorical columns reach the model, and composite columns replayed at predict survive input validation.

The default-on row-wise extension steps run the numeric-only sklearn bridge, whose filter used to DROP every non-numeric
column, so a default suite trained CatBoost without its categorical features (they stayed listed in ``cat_features``). And
the predict-time input validator allowed only raw input columns, so a composite column replayed from metadata (categorical
concat, target encoding) was stripped as "extra" before reaching the model that was trained on it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _frame(n=1200, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({k: rng.integers(0, 6, n).astype(str) for k in "abc"})
    df["num"] = rng.normal(size=n)
    table = {(i, j): rng.integers(0, 2) for i in map(str, range(6)) for j in map(str, range(6))}
    df["y"] = [table[(x, z)] for x, z in zip(df["a"], df["b"])]
    return df


def _train(tmp_path, **kwargs):
    from mlframe.training.configs import OutputConfig
    from mlframe.training.core import train_mlframe_models_suite
    from mlframe.training.extractors import SimpleFeaturesAndTargetsExtractor

    df = _frame()
    models, meta = train_mlframe_models_suite(
        df=df, target_name="t", model_name="m", features_and_targets_extractor=SimpleFeaturesAndTargetsExtractor(classification_targets=["y"], use_recency_weighting=False),
        mlframe_models=["cb"], hyperparams_config={"iterations": 5, "cb_kwargs": {"task_type": "CPU", "verbose": 0}},
        use_ordinary_models=True, use_mlframe_ensembles=False, verbose=0,
        output_config=OutputConfig(data_dir=str(tmp_path), models_dir="models", save_charts=False, run_diagnostics=[]), **kwargs,
    )
    feats = [set(getattr(e.model, "feature_names_", []) or []) for by in models.values() for es in by.values() for e in es]
    return df, models, meta, feats


def test_default_suite_trains_catboost_on_its_categorical_columns(tmp_path):
    pytest.importorskip("catboost")
    _, _, meta, feats = _train(tmp_path)
    assert feats and all({"a", "b", "c"} <= f for f in feats), feats
    assert {"a", "b", "c"} <= set(meta["cat_features"])


def test_replayed_composite_column_reaches_the_model_at_predict(tmp_path):
    pytest.importorskip("catboost")
    from mlframe.training.configs import PreprocessingExtensionsConfig
    from mlframe.training.core import predict_mlframe_models_suite

    df, _, meta, feats = _train(tmp_path, preprocessing_extensions=PreprocessingExtensionsConfig(categorical_group_concat_auto_enabled=True))
    composite = [c for c in meta.get("composite_fe_emitted_columns") or [] if c.startswith("concat_group__")]
    assert composite and all(set(composite) <= f for f in feats)
    out = predict_mlframe_models_suite(df.drop(columns=["y"]).iloc[:100], models_path=str(tmp_path / "models" / "t" / "m"), verbose=0)
    assert out is not None
