"""Regression: ``predict_from_models`` must replay the fit-time
passthrough-cols re-attach when a per-model ``pre_pipeline`` contains
a feature selector (MRMR / RFECV) that drops text/embedding cols.

Pre-fix path (iter-59 100k seed=41 cb-regression with mrmr_fs=True):
1. At fit time the trainer wraps the MRMR pre_pipeline with
   ``_passthrough_cols_fit_transform(... passthrough_cols=
   text_features + embedding_features)``. MRMR drops text/embedding;
   the wrapper re-attaches them so CB.fit sees the full frame and
   stores ``text_features[0]`` as the integer index of ``text_col``
   in the re-attached frame.
2. At predict time the original code called
   ``model_obj.pre_pipeline.transform(input_for_model)`` directly --
   MRMR dropped text/embedding cols, CB then saw a narrower frame
   than at fit, and raised
   ``Invalid text_features[0] = N value: index must be < N``.
3. iter-46 surfaced this as the aggregated RuntimeError naming
   ``regression_y_MRMR``.

Post-fix: when ``metadata["text_features"]`` or
``metadata["embedding_features"]`` is non-empty, predict uses the
same ``_passthrough_cols_fit_transform`` wrapper around
``pre_pipeline.transform`` so the predict-time frame width matches
the fit-time frame width.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline


class _SelectFirstNCols:
    """Stand-in for a fitted MRMR/RFECV selector: at transform-time it
    drops every column except a fixed numeric set, mimicking how a
    fitted selector silently drops text/embedding cols at predict."""

    def __init__(self, kept_cols):
        self.kept_cols = list(kept_cols)
        # Mark fitted so sklearn check_is_fitted passes.
        self.feature_names_in_ = np.array(self.kept_cols, dtype=object)
        self.n_features_in_ = len(self.kept_cols)
        # Public selector indicator used by _is_fitted helpers in mlframe.
        self.support_ = np.ones(len(self.kept_cols), dtype=bool)

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X.loc[:, self.kept_cols]
        raise TypeError(f"unsupported X type: {type(X)}")

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
        return self.transform(X)


def _build_minimal_suite():
    """Build the minimum surface for predict_from_models that
    reproduces the iter-59 production fit-time wiring:

    - pre_pipeline (MRMR selector) was fitted on the NUMERIC-ONLY view
      of the frame because the trainer's
      ``_passthrough_cols_fit_transform`` HID text/embedding cols
      before pre_pipeline.fit_transform ran. So
      ``pre_pipeline.feature_names_in_`` = ["x0", "x1"] (no text_col).
    - The downstream model (a Ridge here mimicking CB's text_features
      shape contract) was fitted on the POST-MRMR + RE-ATTACHED frame
      = ["x0", "text_col"]; so ``model.feature_names_in_`` = those 2
      names.
    """
    rng = np.random.default_rng(0)
    n = 100
    df = pd.DataFrame({
        "x0": rng.standard_normal(n).astype(np.float64),
        "x1": rng.standard_normal(n).astype(np.float64),
        "text_col": np.array(["t" + str(i % 5) for i in range(n)], dtype=object),
    })
    y = rng.standard_normal(n).astype(np.float64)

    # pre_pipeline fitted on numeric-only (post-passthrough-hide).
    pre_pipeline = Pipeline([
        ("pre", _SelectFirstNCols(kept_cols=["x0"])),
    ])
    pre_pipeline.fit(df[["x0", "x1"]])

    # Inner model fitted on post-MRMR+reattach frame (numeric + text).
    inner = Ridge(alpha=1e-3).fit(df[["x0"]], y)
    # Override feature_names_in_ to mirror the trainer-side
    # post-MRMR + passthrough frame layout.
    inner.feature_names_in_ = np.array(["x0", "text_col"], dtype=object)

    obj = SimpleNamespace(model=inner, pre_pipeline=pre_pipeline)
    models = {"regression": {"y_MRMR": [obj]}}
    metadata = {
        "columns": ["x0", "x1", "text_col"],
        "raw_input_columns": ["x0", "x1", "text_col"],
        "text_features": ["text_col"],
        "embedding_features": [],
        "cat_features": [],
    }
    return df, models, metadata, inner


def test_predict_replays_passthrough_cols_when_pre_pipeline_drops_them() -> None:
    """The post-fix predict path must call ``pre_pipeline.transform``
    via ``_passthrough_cols_fit_transform`` so text_col makes it
    through to the inner model alongside the MRMR-selected numeric."""
    from mlframe.training.core.predict import predict_from_models
    df, models, metadata, inner = _build_minimal_suite()
    # Run predict; the inner Ridge will internally just use ``x0``,
    # but the pre_pipeline output the inner saw must include ``text_col``
    # (the fit-time contract). We don't have a hook on Ridge to assert
    # frame shape, so instead we verify ``predict_from_models`` returns
    # WITHOUT raising the iter-59 ValueError-equivalent.
    result = predict_from_models(
        df=df, models=models, metadata=metadata,
        features_and_targets_extractor=None,
        return_probabilities=False, verbose=0,
    )
    # Predict must succeed and produce one model's output.
    assert isinstance(result, dict)
    assert len(result["predictions"]) >= 1
    # The shape must match the input row count.
    for arr in result["predictions"].values():
        if hasattr(arr, "shape"):
            assert arr.shape[0] == len(df)


def test_passthrough_cols_actually_make_it_through_pre_pipeline() -> None:
    """Direct test of the wrapper contract: when pre_pipeline.transform
    is wrapped via ``_passthrough_cols_fit_transform``, text/embedding
    cols are re-attached. This isolates the fix without the full
    predict_from_models machinery."""
    from mlframe.training._pipeline_helpers import _passthrough_cols_fit_transform
    df, models, metadata, _ = _build_minimal_suite()
    pre_pipeline = models["regression"]["y_MRMR"][0].pre_pipeline

    # WITHOUT passthrough: text_col dropped, output has only ["x0"].
    plain = pre_pipeline.transform(df[["x0", "x1"]])
    assert "text_col" not in plain.columns
    assert list(plain.columns) == ["x0"]

    # WITH passthrough: text_col re-attached after pre_pipeline runs.
    wrapped = _passthrough_cols_fit_transform(
        pre_pipeline.transform,
        df[["x0", "x1", "text_col"]],
        passthrough_cols=["text_col"],
    )
    assert "text_col" in wrapped.columns
    assert "x0" in wrapped.columns
