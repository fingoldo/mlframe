"""Wave 90 (2026-05-21): pre_pipeline.transform + passthrough stashing
block (~192 lines) extracted from predict.py:1372 mega-try body to a
module-level helper ``_apply_pre_pipeline_with_passthrough``.

Behaviour preserved bit-for-bit. Net effect:
  - The mega-try body in ``predict_from_models`` shrinks again
    (525 -> 435 after wave 88, -> 348 after wave 89, -> ~156 here).
  - The helper is unit-testable in isolation. The five paths covered
    by these sensors are:
      1. No-op when ``model_obj`` has no ``pre_pipeline`` (or None).
      2. No-op when ``pre_pipeline is pipeline`` (the suite-level one).
      3. No-op + DEBUG log when ``check_is_fitted`` says unfitted.
      4. Active: stash text/embedding passthrough cols, run transform,
         re-attach to the post-transform pandas frame.
      5. Fallback to inner-model ``feature_names_in_`` subset when
         ``pre_pipeline.transform`` raises ``NotFittedError``.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError


def test_helper_is_module_level_callable() -> None:
    """Helper is module level callable."""
    from mlframe.training.core.predict import _apply_pre_pipeline_with_passthrough

    assert callable(_apply_pre_pipeline_with_passthrough)


def test_passthrough_when_model_obj_has_no_pre_pipeline() -> None:
    """Passthrough when model obj has no pre pipeline."""
    from mlframe.training.core.predict import _apply_pre_pipeline_with_passthrough

    class _ModelObj:
        """Groups tests covering model obj."""
        pass  # no pre_pipeline attribute

    df = pd.DataFrame({"x": [1.0, 2.0]})
    out = _apply_pre_pipeline_with_passthrough(
        df,
        model=object(),
        model_obj=_ModelObj(),
        pipeline=None,
        df=df,
        df_pre_pipeline=None,
        metadata={},
        model_name="no_pp",
        verbose=0,
    )
    assert out is df


def test_passthrough_when_pre_pipeline_is_suite_pipeline() -> None:
    """Passthrough when pre pipeline is suite pipeline."""
    from mlframe.training.core.predict import _apply_pre_pipeline_with_passthrough

    _sentinel_pipeline = object()

    class _ModelObj:
        """Groups tests covering model obj."""
        pre_pipeline = _sentinel_pipeline

    df = pd.DataFrame({"x": [1.0, 2.0]})
    out = _apply_pre_pipeline_with_passthrough(
        df,
        model=object(),
        model_obj=_ModelObj(),
        pipeline=_sentinel_pipeline,
        df=df,
        df_pre_pipeline=None,
        metadata={},
        model_name="same_pp",
        verbose=0,
    )
    assert out is df


def test_unfitted_pre_pipeline_skips_transform_and_logs_debug(caplog) -> None:
    """Unfitted pre pipeline skips transform and logs debug."""
    from mlframe.training.core.predict import _apply_pre_pipeline_with_passthrough

    class _UnfittedPP:
        # No fitted attributes; sklearn's check_is_fitted will raise.
        """Groups tests covering unfitted p p."""
        pass

    class _ModelObj:
        """Groups tests covering model obj."""
        pre_pipeline = _UnfittedPP()

    df = pd.DataFrame({"x": [1.0, 2.0]})
    caplog.set_level(logging.DEBUG, logger="mlframe.training.core.predict")
    out = _apply_pre_pipeline_with_passthrough(
        df,
        model=object(),
        model_obj=_ModelObj(),
        pipeline=None,
        df=df,
        df_pre_pipeline=None,
        metadata={},
        model_name="unfitted_pp",
        verbose=1,
    )
    assert out is df
    assert any("unfitted pre_pipeline" in rec.message for rec in caplog.records)


def test_fitted_pre_pipeline_runs_transform_and_reattaches_passthrough() -> None:
    """Fitted pre pipeline runs transform and reattaches passthrough."""
    from mlframe.training.core.predict import _apply_pre_pipeline_with_passthrough

    class _FittedPP(BaseEstimator, TransformerMixin):
        """Groups tests covering fitted p p."""
        def __init__(self) -> None:
            # Instance attribute ending in _ so sklearn check_is_fitted accepts as fitted.
            self.is_fitted_ = True

        def fit(self, X, y=None):
            """Fit."""
            return self

        def transform(self, X):
            # Drop the text passthrough col so the helper has to re-attach it.
            """Transform."""
            return X.drop(columns=[c for c in X.columns if c == "txt"])

    class _ModelObj:
        """Groups tests covering model obj."""
        pre_pipeline = _FittedPP()

    df_pre_pipeline = pd.DataFrame({"x": [1.0, 2.0, 3.0], "txt": ["a", "b", "c"]})
    df = df_pre_pipeline.copy()
    input_for_model = df_pre_pipeline.copy()

    out = _apply_pre_pipeline_with_passthrough(
        input_for_model,
        model=object(),
        model_obj=_ModelObj(),
        pipeline=None,
        df=df,
        df_pre_pipeline=df_pre_pipeline,
        metadata={"text_features": ["txt"]},
        model_name="fitted_pp",
        verbose=0,
    )
    assert isinstance(out, pd.DataFrame)
    # The transform dropped txt, then the helper re-attached it from the stash.
    assert "txt" in out.columns
    assert list(out["txt"]) == ["a", "b", "c"]


def test_not_fitted_error_in_transform_falls_back_to_feature_names_in() -> None:
    """Not fitted error in transform falls back to feature names in."""
    from mlframe.training.core.predict import _apply_pre_pipeline_with_passthrough

    class _RaisingPP(BaseEstimator, TransformerMixin):
        """Groups tests covering raising p p."""
        def __init__(self) -> None:
            # Instance attribute ending in _ so sklearn check_is_fitted accepts as fitted.
            self.is_fitted_ = True

        def fit(self, X, y=None):
            """Fit."""
            return self

        def transform(self, X):
            """Transform."""
            raise NotFittedError("cloned but not refit")

    class _ModelObj:
        """Groups tests covering model obj."""
        pre_pipeline = _RaisingPP()

    # Inner model exposes feature_names_in_; the helper should subset to it.
    class _InnerModel:
        """Groups tests covering inner model."""
        feature_names_in_ = np.array(["x"])

    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "extra": [9.0, 9.0, 9.0]})

    out = _apply_pre_pipeline_with_passthrough(
        df,
        model=_InnerModel(),
        model_obj=_ModelObj(),
        pipeline=None,
        df=df,
        df_pre_pipeline=df,
        metadata={},
        model_name="raising_pp",
        verbose=0,
    )
    assert isinstance(out, pd.DataFrame)
    # NotFittedError -> fallback subsetted to inner model's feature_names_in_.
    assert list(out.columns) == ["x"]


def test_outer_try_body_shrunk_after_wave90(monkeypatch) -> None:
    """predict_from_models hands each model's pre-pipeline step to the module-level helper, once per model, with the full keyword surface."""
    from sklearn.linear_model import LinearRegression

    from mlframe.training.core import predict as predict_mod
    from mlframe.training.core._predict_main_from_models import predict_from_models

    class _ModelWrapper:
        """Fitted model plus the pre_pipeline slot predict_from_models reads."""

        def __init__(self, model) -> None:
            self.model = model
            self.pre_pipeline = None

    X = pd.DataFrame({"a": np.arange(20.0), "b": np.arange(20.0) * 2})
    y = X["a"] + 5.0
    wrappers = [_ModelWrapper(LinearRegression().fit(X, y)), _ModelWrapper(LinearRegression().fit(X, y * 2))]
    calls: list = []
    real_helper = predict_mod._apply_pre_pipeline_with_passthrough

    def spy(input_for_model, **kwargs):
        """Record the call, then run the real helper."""
        calls.append((input_for_model, kwargs))
        return real_helper(input_for_model, **kwargs)

    monkeypatch.setattr(predict_mod, "_apply_pre_pipeline_with_passthrough", spy)
    meta = {"raw_input_columns": ["a", "b"], "columns": ["a", "b"], "cat_features": [], "text_features": [], "embedding_features": []}
    out = predict_from_models(X, {"regression": {"t": wrappers}}, meta, return_probabilities=False, verbose=0)
    assert "regression_t" in out["predictions"]
    assert len(calls) == len(wrappers)
    expected_keywords = {"model", "model_obj", "pipeline", "df", "df_pre_pipeline", "metadata", "model_name", "verbose"}
    for (frame, kwargs), wrapper in zip(calls, wrappers):
        assert isinstance(frame, pd.DataFrame)
        assert set(kwargs) == expected_keywords
        assert kwargs["model_obj"] is wrapper
