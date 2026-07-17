"""Behavioral coverage of the predict-time feature-selection RECOVERY branch
in ``_apply_pre_pipeline_with_passthrough`` (mlframe.training.core.predict).

Production scenario (iter-59 family): a per-model ``pre_pipeline`` holds a
fitted feature selector (MRMR / RFECV) that, at predict time, can fail its
``.transform`` -- e.g. a clone-not-refit selector raising ``NotFittedError``,
or a stale internal state from a partially-restored bundle. The recovery block
catches that failure and, instead of dropping the model, SUBSETS the input to
the inner model's ``feature_names_in_`` / ``feature_names_`` and proceeds to
``model.predict`` (the main pipeline already encoded cat_features), emitting a
``Skipping pre_pipeline`` warning so the fallback is never silent.

Prior coverage of this branch was a test-local re-implementation plus an AST
source-scan -- ZERO behavioral exercise of the production recovery code. This
file drives ``predict_from_models`` end-to-end through a MINIMAL hand-built
models dict (no heavy ``train_mlframe_models_suite`` call) and asserts:

  (a) the model's predictions are produced and ``np.allclose`` to
      ``inner.predict(df[["x0","x1"]])`` -- proving the subset-recovery handed
      the inner model the RIGHT (pre-selector) input columns; and
  (b) the production "Skipping pre_pipeline" warning was logged.

The pre_pipeline is a REAL fitted ``MRMR`` (not a stub); its ``transform`` is
monkeypatched to raise so we exercise the genuine production recovery path,
not a copy. A control test confirms the recovery is load-bearing: when the
inner model exposes NO ``feature_names_in_`` (so the subset cannot run) the
post-transform frame still carries the dropped selector input and the same
warning fires -- the branch is genuinely entered, not bypassed.
"""

from __future__ import annotations

import logging
import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd

RANDOM_SEED = 17


def _make_frame(n: int = 120):
    """Synthetic 2-feature regression frame plus one EXTRA column the predict
    frame carries (``x2``) that the per-model feature list does not expect."""
    rng = np.random.default_rng(RANDOM_SEED)
    x0 = rng.standard_normal(n).astype(np.float64)
    x1 = rng.standard_normal(n).astype(np.float64)
    x2 = rng.standard_normal(n).astype(np.float64)  # extra col, dropped pre-pipeline
    y = (2.0 * x0 - 1.0 * x1 + 0.1 * rng.standard_normal(n)).astype(np.float64)
    df = pd.DataFrame({"x0": x0, "x1": x1, "x2": x2})
    return df, y


def _fit_real_mrmr(X_num: pd.DataFrame, y: np.ndarray):
    """Fit a REAL production ``MRMR`` on the numeric-only view so its
    ``feature_names_in_`` == ["x0","x1"] (the per-model expected feature list
    the predict validation block consults) and ``check_is_fitted`` passes."""
    from mlframe.feature_selection.filters.mrmr import MRMR
    from mlframe.feature_selection.filters.cat_fe_state import CatFEConfig

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sel = MRMR(
            verbose=0,
            interactions_max_order=1,
            fe_max_steps=0,
            cat_fe_config=CatFEConfig(enable=False),
        ).fit(X_num, pd.Series(y, name="y"))
    return sel


def _build_minimal_suite(*, inner_exposes_feature_names: bool = True):
    """Build the minimum surface ``predict_from_models`` needs to reach and
    exercise the FS recovery branch:

    - ``pre_pipeline`` is a real fitted MRMR with ``feature_names_in_`` =
      ["x0","x1"]; its ``transform`` is broken below so the recovery fires.
    - the inner model is an ``LGBMRegressor`` fitted on ["x0","x1"], so its
      ``feature_names_in_`` is exactly the columns the recovery must subset to;
      ``inner.predict(df[["x0","x1"]])`` is the ground-truth output.
    """
    from lightgbm import LGBMRegressor

    df, y = _make_frame()
    X_num = df[["x0", "x1"]]

    inner = LGBMRegressor(
        n_estimators=15,
        num_leaves=7,
        min_child_samples=5,
        random_state=RANDOM_SEED,
        verbosity=-1,
    )
    inner.fit(X_num, y)
    assert list(inner.feature_names_in_) == ["x0", "x1"]

    pre_pipeline = _fit_real_mrmr(X_num, y)
    assert list(pre_pipeline.feature_names_in_) == ["x0", "x1"]

    # Break the fitted selector's transform so the production recovery branch
    # catches the failure (clone-not-refit / stale-bundle symptom). Patch the
    # bound method on the instance so check_is_fitted still passes upstream.
    from sklearn.exceptions import NotFittedError

    def _broken_transform(_X, *_a, **_k):
        raise NotFittedError("simulated stale selector state at predict")

    pre_pipeline.transform = _broken_transform  # instance-level override

    if not inner_exposes_feature_names:
        # Strip feature_names_in_ so the subset-recovery cannot run; the branch
        # is still entered (transform raises) but falls through with the frame
        # unchanged -- used by the control test below.
        inner_obj = SimpleNamespace(model=_StripNamesModel(inner), pre_pipeline=pre_pipeline)
    else:
        inner_obj = SimpleNamespace(model=inner, pre_pipeline=pre_pipeline)

    models = {"regression": {"y_MRMR": [inner_obj]}}
    metadata = {
        "columns": ["x0", "x1", "x2"],
        "raw_input_columns": ["x0", "x1", "x2"],
        "text_features": [],
        "embedding_features": [],
        "cat_features": [],
    }
    expected = inner.predict(X_num)
    return df, models, metadata, inner, expected


class _StripNamesModel:
    """Wraps a fitted regressor but exposes NO ``feature_names_in_`` /
    ``feature_names_`` -- forces the recovery's subset step to be skipped so we
    can assert the branch is still entered (warning fires) without the subset."""

    def __init__(self, inner):
        self._inner = inner

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.loc[:, ["x0", "x1"]]
        return self._inner.predict(X)


def test_fs_recovery_subsets_to_inner_feature_names_and_predicts(caplog) -> None:
    """When the per-model pre_pipeline (a real MRMR) fails ``.transform`` at
    predict, the production recovery subsets the input to the inner model's
    ``feature_names_in_`` and predicts -- producing output ``np.allclose`` to
    ``inner.predict(df[["x0","x1"]])`` and logging ``Skipping pre_pipeline``."""
    from mlframe.training.core.predict import predict_from_models

    df, models, metadata, _inner, expected = _build_minimal_suite()

    with caplog.at_level(logging.WARNING, logger="mlframe.training.core.predict"):
        result = predict_from_models(
            df=df,
            models=models,
            metadata=metadata,
            features_and_targets_extractor=None,
            return_probabilities=False,
            verbose=0,
        )

    assert isinstance(result, dict)
    preds_map = result["predictions"]
    assert len(preds_map) == 1, f"expected one model output, got {list(preds_map)}"
    (got,) = preds_map.values()
    got = np.asarray(got)

    # (a) Subset-recovery handed the inner model the RIGHT pre-selector input.
    assert got.shape[0] == len(df)
    assert np.allclose(got, expected), "recovered predictions must match inner.predict(df[['x0','x1']]); a wrong subset would shift every value"

    # (b) The production fallback warning fired (never silent).
    assert any("Skipping pre_pipeline" in rec.getMessage() for rec in caplog.records), "expected the 'Skipping pre_pipeline' recovery warning; got: " + repr(
        [r.getMessage() for r in caplog.records]
    )


def test_fs_recovery_branch_is_load_bearing_when_no_inner_feature_names(caplog) -> None:
    """Control: the recovery branch is genuinely ENTERED (not bypassed). With
    an inner model exposing no ``feature_names_in_`` the subset step is skipped,
    yet the same ``Skipping pre_pipeline`` warning still fires -- proving the
    pre_pipeline.transform failure is what drives the path, and predict still
    succeeds because the inner model tolerates the unsubset frame.

    This pins that the test exercises the PROD recovery code: were the recovery
    block conceptually absent, the raised NotFittedError would propagate, the
    per-model try/except would drop the model, and ``predictions`` would be
    empty -- which this test forbids."""
    from mlframe.training.core.predict import predict_from_models

    df, models, metadata, _inner, expected = _build_minimal_suite(inner_exposes_feature_names=False)

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
    assert len(preds_map) == 1, (
        "recovery must keep the model alive even without inner feature_names_in_; an empty predictions dict would mean the model was dropped (recovery absent)"
    )
    (got,) = preds_map.values()
    assert np.allclose(np.asarray(got), expected)
    assert any("Skipping pre_pipeline" in rec.getMessage() for rec in caplog.records)
