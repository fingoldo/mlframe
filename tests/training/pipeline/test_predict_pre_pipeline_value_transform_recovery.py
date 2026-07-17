"""Regression: predict-time pre_pipeline recovery must NOT silently serve raw
features when the pre_pipeline VALUE-TRANSFORMS them.

``_apply_pre_pipeline_with_passthrough`` (mlframe.training.core.predict) catches
a failed ``pre_pipeline.transform`` and recovers by subsetting the raw predict
frame to the inner model's ``feature_names_in_``. That recovery is value-correct
ONLY when the pre_pipeline is a pure name-keyed column selector (MRMR / RFECV):
its transform == "select these columns by name", so the raw subset equals the
fit-time input.

When the pre_pipeline instead alters values (StandardScaler / imputer / encoder
/ PCA), the raw subset is NOT what the model trained on. The pre-fix code served
it anyway with only a WARNING -> silently-wrong predictions shipped as if
correct. The fix gates the subset-recovery on ``_pre_pipeline_is_pure_selector``
and RE-RAISES for value-transformers, dropping the model (clean degrade) instead
of shipping nonsense.

This test FAILS on pre-fix code: a value-scaling pre_pipeline whose transform is
broken produces a prediction (the model is kept) that is WRONG (computed on
unscaled raw input). Post-fix: the model is dropped (empty predictions) -- a
clean, announced degrade.
"""

from __future__ import annotations

import logging
import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd

RANDOM_SEED = 23


def _build_value_transform_suite():
    """Two regression models on the SAME 2-feature frame:

    * ``y_scaled``  -- inner LGBM trained on STANDARD-SCALED ["x0","x1"]; its
      per-model pre_pipeline is a fitted StandardScaler whose ``transform`` is
      broken at predict time. The raw (unscaled) subset would feed it wrong
      values -> must be DROPPED, not served.
    * ``y_selector`` -- inner LGBM trained on RAW ["x0","x1"]; its pre_pipeline
      is a pure SelectKBest(k=2) (value-preserving) whose transform is broken
      too -> the subset-recovery is value-correct, so this model SURVIVES.

    Keeping a survivor proves the fix is a SURGICAL partial-failure drop (not a
    blanket all-fail RuntimeError) and that the legitimate selector recovery is
    untouched. ``y_selector``'s prediction must equal ``inner.predict(raw df)``.
    """
    from lightgbm import LGBMRegressor
    from sklearn.feature_selection import SelectKBest, f_regression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(RANDOM_SEED)
    n = 200
    # Large-magnitude raw features so unscaled vs scaled is dramatically different.
    x0 = (rng.standard_normal(n) * 100.0 + 500.0).astype(np.float64)
    x1 = (rng.standard_normal(n) * 50.0 - 300.0).astype(np.float64)
    y = (2.0 * x0 - 1.0 * x1 + 0.1 * rng.standard_normal(n)).astype(np.float64)
    df = pd.DataFrame({"x0": x0, "x1": x1})

    def _broken(_X, *_a, **_k):
        """Broken."""
        raise RuntimeError("simulated stale pre_pipeline state at predict")

    # --- value-transform model (must be dropped) ---
    scaler = Pipeline([("scale", StandardScaler())])
    X_scaled = pd.DataFrame(scaler.fit_transform(df), columns=["x0", "x1"], index=df.index)
    m_scaled = LGBMRegressor(n_estimators=30, num_leaves=15, min_child_samples=5, random_state=RANDOM_SEED, verbosity=-1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m_scaled.fit(X_scaled, y)
    scaler.transform = _broken
    obj_scaled = SimpleNamespace(model=m_scaled, pre_pipeline=scaler)

    # --- pure-selector model (must survive via raw-subset recovery) ---
    selector = Pipeline([("pre", SelectKBest(f_regression, k=2))])
    selector.fit(df, y)
    m_sel = LGBMRegressor(n_estimators=30, num_leaves=15, min_child_samples=5, random_state=RANDOM_SEED, verbosity=-1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m_sel.fit(df, y)
    selector.transform = _broken
    obj_sel = SimpleNamespace(model=m_sel, pre_pipeline=selector)
    expected_sel = m_sel.predict(df[["x0", "x1"]])

    models = {"regression": {"y_scaled": [obj_scaled], "y_selector": [obj_sel]}}
    metadata = {
        "columns": ["x0", "x1"],
        "raw_input_columns": ["x0", "x1"],
        "text_features": [],
        "embedding_features": [],
        "cat_features": [],
    }
    return df, models, metadata, expected_sel


def test_value_transform_pre_pipeline_failure_drops_model_not_silent_raw(caplog) -> None:
    """A value-transforming (StandardScaler) pre_pipeline that fails transform
    at predict MUST cause that model to be DROPPED (clean degrade), NOT served on
    raw unscaled input. Pre-fix served wrong predictions for it. A sibling
    pure-selector model whose transform also fails MUST still survive (its
    raw-subset recovery is value-correct)."""
    from mlframe.training.core.predict import predict_from_models

    df, models, metadata, expected_sel = _build_value_transform_suite()

    with caplog.at_level(logging.WARNING, logger="mlframe.training.core.predict"):
        result = predict_from_models(
            df=df,
            models=models,
            metadata=metadata,
            features_and_targets_extractor=None,
            return_probabilities=False,
            verbose=0,
        )

    preds_map = result["predictions"]
    # The scaler model is dropped; only the pure-selector model remains.
    _names = list(preds_map)
    assert not any("y_scaled" in _n for _n in _names), (
        "value-transforming pre_pipeline that fails transform must DROP the "
        "model, not serve predictions on un-transformed raw input; got "
        f"{_names} -- the silent-wrong-result bug is still present."
    )
    assert any("y_selector" in _n for _n in _names), f"the pure-selector model's raw-subset recovery is value-correct and must survive; got {_names}"
    (sel_pred,) = [v for k, v in preds_map.items() if "y_selector" in k]
    assert np.allclose(np.asarray(sel_pred), expected_sel), "pure-selector recovery must reproduce inner.predict(df[['x0','x1']])"
    assert any("value-transforms features" in rec.getMessage() for rec in caplog.records), (
        "expected the 'value-transforms features' re-raise warning; got: " + repr([r.getMessage() for r in caplog.records])
    )


def test_pure_selector_recovery_still_works():
    """Guard the legitimate path stays intact: the pure-selector detector
    classifies a bare name-keyed selector as value-preserving."""
    from mlframe.training.core._predict_pre_pipeline import _pre_pipeline_is_pure_selector
    from sklearn.feature_selection import SelectKBest, f_regression
    from sklearn.pipeline import Pipeline

    rng = np.random.default_rng(RANDOM_SEED)
    X = pd.DataFrame(rng.standard_normal((100, 4)), columns=[f"c{i}" for i in range(4)])
    y = X["c0"] * 2 + rng.standard_normal(100) * 0.1

    bare = SelectKBest(f_regression, k=2).fit(X, y)
    assert _pre_pipeline_is_pure_selector(bare) is True

    # Pipeline whose only step is the selector -> still pure.
    pipe_sel = Pipeline([("pre", SelectKBest(f_regression, k=2))]).fit(X, y)
    assert _pre_pipeline_is_pure_selector(pipe_sel) is True

    # Pipeline with a value-altering step alongside -> NOT pure.
    from sklearn.preprocessing import StandardScaler

    pipe_scaler = Pipeline([("scale", StandardScaler())]).fit(X, y)
    assert _pre_pipeline_is_pure_selector(pipe_scaler) is False
    assert _pre_pipeline_is_pure_selector(None) is False
