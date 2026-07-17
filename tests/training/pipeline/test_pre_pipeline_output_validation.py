"""Regression: a pre_pipeline whose output drops a column the FITTED model
expects must raise a clear, actionable error from
``_apply_pre_pipeline_transforms`` -- naming the missing column + the
pre_pipeline step -- instead of surfacing as an opaque downstream
sklearn/booster error deep inside ``.predict``.

Pre-fix there was no validation of the pre_pipeline output against the model's
expected feature names: a mis-shaped output (custom step dropping a needed
column) only blew up far from the cause. This pins the clear early error.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


class _DropColumn(BaseEstimator, TransformerMixin):
    """A pre_pipeline step that silently drops one column on transform."""

    def __init__(self, drop: str):
        self.drop = drop

    def fit(self, X, y=None):
        """Fit."""
        return self

    def transform(self, X):
        """Transform."""
        return X.drop(columns=[self.drop])


def _make_fitted_model(cols):
    """Make fitted model."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(80, len(cols))), columns=cols)
    y = (X[cols[0]] > 0).astype(int)
    return LogisticRegression(max_iter=200).fit(X, y)


def test_pre_pipeline_dropping_expected_column_raises_named_error():
    """Pre pipeline dropping expected column raises named error."""
    from mlframe.training.pipeline._pipeline_helpers import _apply_pre_pipeline_transforms

    cols = [f"f{i}" for i in range(4)]
    # Model already fitted on all 4 features -> feature_names_in_ is known.
    model = _make_fitted_model(cols)
    assert list(model.feature_names_in_) == cols

    # Pre-pipeline that DROPS f2 on transform -> output is missing a column the
    # model expects. Fit it so it's treated as a fitted pre_pipeline.
    pre = Pipeline([("dropper", _DropColumn(drop="f2"))])
    rng = np.random.default_rng(1)
    train_df = pd.DataFrame(rng.normal(size=(80, 4)), columns=cols)
    pre.fit(train_df)

    with pytest.raises(ValueError) as exc:
        _apply_pre_pipeline_transforms(
            model=model,
            pre_pipeline=pre,
            train_df=train_df,
            val_df=None,
            train_target=(train_df["f0"] > 0).astype(int),
            skip_pre_pipeline_transform=False,
            skip_preprocessing=False,
            use_cache=False,
            model_file_name="",
            verbose=False,
        )

    msg = str(exc.value)
    assert "f2" in msg, f"error must name the missing column f2; got: {msg}"
    assert "pre_pipeline" in msg.lower()


def test_pre_pipeline_unfitted_model_skips_validation():
    """First fit: model has no feature_names_in_ -> no over-validation."""
    from mlframe.training.pipeline._pipeline_helpers import _apply_pre_pipeline_transforms

    cols = [f"f{i}" for i in range(4)]
    pre = Pipeline([("dropper", _DropColumn(drop="f2"))])
    rng = np.random.default_rng(2)
    train_df = pd.DataFrame(rng.normal(size=(80, 4)), columns=cols)
    pre.fit(train_df)

    # model has no feature_names_in_ -> validation is skipped, no raise.
    out, _ = _apply_pre_pipeline_transforms(
        model=object(),
        pre_pipeline=pre,
        train_df=train_df,
        val_df=None,
        train_target=(train_df["f0"] > 0).astype(int),
        skip_pre_pipeline_transform=False,
        skip_preprocessing=False,
        use_cache=False,
        model_file_name="",
        verbose=False,
    )
    assert "f2" not in list(out.columns)
