"""Regression: a row-CHANGING pre_pipeline (imblearn resampler) must be rejected.

The feature-selection / pre_pipeline slot is row-PRESERVING. The driver
(``_apply_pre_pipeline_transforms``) returns only ``(train_df, val_df)`` -- it
cannot propagate a resampled target / sample_weight. So a SMOTE /
RandomOverSampler / RandomUnderSampler / FunctionSampler wired into a
``custom_pre_pipeline`` silently grows/shrinks ``train_df`` while
``train_target`` stays at the original length, misaligning X and y at model fit
(silent-wrong output, or an opaque backend crash, or -- with passthrough cols --
an opaque pandas index-shape error).

Pre-fix: no row-count check; the no-passthrough path returned a 16-row train_df
against a 10-row target (DESYNC), and the passthrough path raised an opaque
``Shape of passed values is (16, 2), indices imply (10, 2)``. Post-fix: both
paths raise a clear ValueError naming the row-count change and the supported
model-level imbalance knobs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.training.pipeline._pipeline_helpers import (
    _apply_pre_pipeline_transforms,
    _passthrough_cols_fit_transform,
)


def _imb_pipeline():
    """Imb pipeline."""
    imb_over = pytest.importorskip("imblearn.over_sampling")
    imb_pipe = pytest.importorskip("imblearn.pipeline")
    from sklearn.preprocessing import StandardScaler

    return imb_pipe.Pipeline([("over", imb_over.RandomOverSampler(random_state=0)), ("sc", StandardScaler())])


def _imbalanced_xy(n=20, pos=4):
    """Imbalanced xy."""
    rng = np.random.RandomState(0)
    X = pd.DataFrame({"a": rng.randn(n), "b": rng.randn(n)})
    y = np.array([0] * (n - pos) + [1] * pos)
    return X, y


def test_resampler_in_pre_pipeline_no_passthrough_raises_rowcount_error():
    """Growing resampler, no passthrough cols: the driver must raise (was silent X/y desync)."""
    from sklearn.linear_model import LogisticRegression

    X, y = _imbalanced_xy()
    with pytest.raises(ValueError, match=r"changed the train row count"):
        _apply_pre_pipeline_transforms(
            model=LogisticRegression(),
            pre_pipeline=_imb_pipeline(),
            train_df=X,
            val_df=None,
            train_target=y,
            skip_pre_pipeline_transform=False,
            skip_preprocessing=False,
            use_cache=False,
            model_file_name="",
            verbose=0,
            selector_passthrough_cols=None,
        )


def test_resampler_in_pre_pipeline_with_passthrough_raises_rowcount_error():
    """Growing resampler + passthrough col: clear error, not an opaque pandas index-shape crash."""
    X, y = _imbalanced_xy()
    X = X.assign(txt=["x"] * len(X))
    with pytest.raises(ValueError, match=r"changed the train row count"):
        _passthrough_cols_fit_transform(
            _imb_pipeline().fit_transform,
            X,
            passthrough_cols=["txt"],
            fit=True,
            target=y,
        )


def test_row_preserving_pre_pipeline_still_passes():
    """A genuine row-preserving FS/preprocessing pipeline must NOT trip the guard."""
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline as SkPipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    X, y = _imbalanced_xy()
    sk = SkPipeline([("imp", SimpleImputer()), ("sc", StandardScaler())])
    train_df, _ = _apply_pre_pipeline_transforms(
        model=LogisticRegression(),
        pre_pipeline=sk,
        train_df=X,
        val_df=None,
        train_target=y,
        skip_pre_pipeline_transform=False,
        skip_preprocessing=False,
        use_cache=False,
        model_file_name="",
        verbose=0,
        selector_passthrough_cols=None,
    )
    assert train_df.shape[0] == len(X)
