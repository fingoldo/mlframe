"""Lean-save serialization completeness: ``lean=True`` must NOT change predictions.

``save_mlframe_model(lean=True)`` strips a shallow-copy of the model namespace down
to the inference-ready fields (the heavy train/val/test preds+probs+targets, OOF
arrays, OD indices, trainset feature stats). This suite proves that the strip touches
ONLY inference-irrelevant training artefacts: a model saved lean and loaded predicts
BIT-IDENTICALLY to the same model saved non-lean and loaded, on the same holdout.

Covered estimator shapes (each round-trips through the real ``save_mlframe_model`` /
``load_mlframe_model`` -- no shims):
  * LightGBM regressor with a fitted sklearn ``pre_pipeline`` (imputer+scaler) and the
    full set of stripped fields populated (train/val/test preds+target, OOF, OD idx,
    trainset_features_stats) -- the inference-critical pre_pipeline + booster must survive.
  * ``CompositeTargetEstimator`` (linear_residual): the target-inverse alpha/beta params
    live on the estimator, which lean does NOT strip -- the residual-inverse must survive
    so predictions stay in target scale, not residual scale.

Both lean and non-lean go through ``predict_from_models`` so the assertion is on the
inference output the operator actually consumes.
"""

from __future__ import annotations

import os
import tempfile
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from mlframe.training.io import save_mlframe_model, load_mlframe_model, _LEAN_STRIP_FIELDS
from mlframe.training.core.predict import predict_from_models


# Single-threaded zstd avoids the Windows multi-thread ``flush of closed file`` quirk
# in atomic_write_bytes; the lean strip + dill graph are identical regardless of thread count.
_ZSTD = dict(level=4, write_checksum=True, write_content_size=True, threads=0)


def _make_holdout(n: int = 400, seed: int = 7) -> pd.DataFrame:
    """Helper that make holdout."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "f0": rng.normal(size=n).astype("float32"),
            "f1": rng.normal(size=n).astype("float32"),
            "f2": rng.normal(size=n).astype("float32"),
        }
    )


def _populate_strip_fields(entry: SimpleNamespace, n_train: int = 5_000) -> None:
    """Stamp every ``_LEAN_STRIP_FIELDS`` member with a non-None array so a regression that
    leaves an inference-critical field in the strip set (or drops a needed one) is detectable:
    if any of these leaked into predict, lean vs non-lean would diverge."""
    rng = np.random.default_rng(123)
    entry.train_preds = rng.standard_normal(n_train).astype(np.float32)
    entry.train_probs = None
    entry.train_target = rng.standard_normal(n_train).astype(np.float32)
    entry.val_preds = rng.standard_normal(800).astype(np.float32)
    entry.val_probs = None
    entry.val_target = rng.standard_normal(800).astype(np.float32)
    entry.test_preds = rng.standard_normal(800).astype(np.float32)
    entry.test_probs = None
    entry.test_target = rng.standard_normal(800).astype(np.float32)
    entry.oof_preds = rng.standard_normal(n_train).astype(np.float32)
    entry.oof_probs = None
    entry.train_od_idx = np.arange(50)
    entry.val_od_idx = np.arange(20)
    entry.trainset_features_stats = {f"f{i}": {"mean": 0.0, "std": 1.0} for i in range(3)}


def _roundtrip_predict(entry: SimpleNamespace, metadata: dict, df: pd.DataFrame, *, lean: bool):
    """Save the entry (lean or fat), load it back, predict the holdout, return the per-model preds."""
    with tempfile.NamedTemporaryFile(suffix=".dump", delete=False) as tf:
        fpath = tf.name
    try:
        ok = save_mlframe_model(entry, fpath, zstd_kwargs=_ZSTD, verbose=0, lean=lean, auto_lean_retry=False)
        assert ok is True, f"save_mlframe_model(lean={lean}) failed"
        loaded = load_mlframe_model(fpath, safe=True)
        assert loaded is not None, f"load_mlframe_model returned None for lean={lean}"
        models = {"regression": {"y": [loaded]}}
        res = predict_from_models(df=df, models=models, metadata=metadata, return_probabilities=False, verbose=0)
        assert res["models_used"], f"predict produced no models for lean={lean}"
        return res, loaded
    finally:
        if os.path.exists(fpath):
            os.remove(fpath)


@pytest.fixture(scope="module")
def _lgb_entry():
    """A fitted LGB regressor + sklearn pre_pipeline wrapped exactly as the suite stamps it."""
    pytest.importorskip("lightgbm")
    from lightgbm import LGBMRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(0)
    n = 600
    Xtr = pd.DataFrame(
        {
            "f0": rng.normal(size=n).astype("float32"),
            "f1": rng.normal(size=n).astype("float32"),
            "f2": rng.normal(size=n).astype("float32"),
        }
    )
    ytr = (1.7 * Xtr["f0"] - 0.9 * Xtr["f1"] + 0.4 * Xtr["f2"] + rng.normal(0, 0.2, n)).to_numpy()

    pre = Pipeline([("imp", SimpleImputer()), ("sc", StandardScaler())])
    Xtr_t = pd.DataFrame(pre.fit_transform(Xtr), columns=Xtr.columns)
    model = LGBMRegressor(n_estimators=40, num_leaves=15, verbose=-1, random_state=0)
    model.fit(Xtr_t, ytr)

    entry = SimpleNamespace(model=model, pre_pipeline=pre, columns=["f0", "f1", "f2"], metrics={})
    _populate_strip_fields(entry)
    return entry


def test_lean_save_does_not_change_lgb_predictions(_lgb_entry):
    """LGB + sklearn pre_pipeline: lean-saved predictions are BIT-IDENTICAL to non-lean-saved.
    A regression that strips the pre_pipeline / booster (or that lets a stripped training array
    leak into the predict path) would make these diverge."""
    df = _make_holdout()
    metadata = {"cat_features": [], "columns": ["f0", "f1", "f2"]}

    res_fat, loaded_fat = _roundtrip_predict(_lgb_entry, metadata, df, lean=False)
    res_lean, loaded_lean = _roundtrip_predict(_lgb_entry, metadata, df, lean=True)

    # Inference-critical state survived the lean strip.
    assert loaded_lean.model is not None, "lean dropped the fitted booster"
    assert loaded_lean.pre_pipeline is not None, "lean dropped the fitted pre_pipeline"
    # The fat bundle still carries a stripped field; the lean one must not.
    assert getattr(loaded_fat, "train_preds", None) is not None
    for field in _LEAN_STRIP_FIELDS:
        assert not hasattr(loaded_lean, field) or getattr(loaded_lean, field) is None, f"lean save leaked stripped field {field!r}"

    np.testing.assert_array_equal(
        np.asarray(res_fat["ensemble_predictions"]),
        np.asarray(res_lean["ensemble_predictions"]),
        err_msg="lean=True changed LGB predictions -- a stripped field was inference-critical or a needed field was dropped",
    )


def test_lean_save_does_not_change_composite_target_predictions():
    """CompositeTargetEstimator (linear_residual): the target-inverse alpha/beta params live ON
    the estimator (not in _LEAN_STRIP_FIELDS). Lean must preserve them so the residual-inverse
    keeps predictions in target scale -- a dropped inverse param silently collapses y_hat to the
    residual head (mean~0). Lean vs non-lean predictions must be BIT-IDENTICAL."""
    pytest.importorskip("lightgbm")
    from lightgbm import LGBMRegressor
    from mlframe.training.composite import CompositeTargetEstimator

    rng = np.random.default_rng(11)
    n = 800
    Xtr = pd.DataFrame(
        {
            "f0": rng.normal(size=n).astype("float32"),
            "f1": rng.normal(size=n).astype("float32"),
            "f2": rng.normal(size=n).astype("float32"),
        }
    )
    base = Xtr["f0"].to_numpy()
    ytr = (3.0 * base + 0.5 * Xtr["f1"].to_numpy() + rng.normal(0, 0.1, n)).astype("float64")

    inner = LGBMRegressor(n_estimators=40, num_leaves=15, verbose=-1, random_state=0)
    cte = CompositeTargetEstimator(
        base_estimator=inner,
        transform_name="linear_residual",
        base_column="f0",
    )
    cte.fit(Xtr, ytr)

    entry = SimpleNamespace(model=cte, pre_pipeline=None, columns=["f0", "f1", "f2"], metrics={})
    _populate_strip_fields(entry)

    df = _make_holdout()
    # CTE predict reads the raw base column from df_pre_pipeline; metadata carries no pipeline so
    # predict_from_models hands the raw frame straight through.
    metadata = {"cat_features": [], "columns": ["f0", "f1", "f2"]}

    res_fat, loaded_fat = _roundtrip_predict(entry, metadata, df, lean=False)
    res_lean, loaded_lean = _roundtrip_predict(entry, metadata, df, lean=True)

    # The residual-inverse params (alpha/beta) ride on the estimator and must survive lean.
    _fat_model, lean_model = loaded_fat.model, loaded_lean.model
    assert lean_model is not None, "lean dropped the CompositeTargetEstimator"
    np.testing.assert_array_equal(
        np.asarray(res_fat["ensemble_predictions"]),
        np.asarray(res_lean["ensemble_predictions"]),
        err_msg="lean=True changed CompositeTargetEstimator predictions -- the target-inverse params were lost across the lean strip",
    )
    # Predictions are in target scale (mean near 0 would mean the residual-inverse was lost).
    assert abs(float(np.mean(res_lean["ensemble_predictions"]))) > 0.1, "composite predictions collapsed toward residual scale; target-inverse not applied"
