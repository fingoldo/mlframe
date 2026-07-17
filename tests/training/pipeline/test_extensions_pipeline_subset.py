"""Regression: ``_apply_extensions_pipeline`` must subset the predict
frame to the saved pipeline's ``feature_names_in_`` BEFORE calling
``ext_pipeline.transform``, because the training-time
``apply_preprocessing_extensions`` drops non-numeric (iter-43) and
all-null (iter-44) columns before the sklearn-bridge pipeline gets fit.

Pre-fix path (iter-49 300k seed=13 cb-regression):
1. Train: ``dim_reducer=PCA`` requested. The extensions stage dropped
   ``cat_low``, ``cat_mid`` (non-numeric) and ``x4``, ``x5`` (all-null)
   so the fitted pipeline saw 6 numeric cols. Output: ``pca0..pca5``.
2. Predict: raw input still carries the pre-filter cat/null cols.
   ``ext_pipeline.transform(df)`` triggers sklearn's strict feature-name
   check, raising ``ValueError: The feature names should match those
   that were passed during fit``. The error is logged + swallowed
   (line 115); df returned unchanged.
3. Downstream model.predict sees the raw frame minus the expected
   ``pca0..pca5`` and crashes with ``Model regression_y expects
   features missing from input: ['pca0', ..., 'pca5']``.
4. iter-46 surfaces this as the aggregated RuntimeError instead of an
   empty dict; iter-49 fixes the actual root cause.

Post-fix: at predict, subset ``df`` to the saved
``ext_pipeline.feature_names_in_`` (with column-order normalisation)
before calling ``.transform``. Missing fit-time cols still WARN; the
extra (pre-filter) cols are now silently dropped so transform passes.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _build_fitted_pipeline(seed: int = 0):
    """Build a Scaler+PCA pipeline fit on a frame matching the iter-49
    seed=13 post-filter shape: 6 numeric cols, n_components=5."""
    rng = np.random.default_rng(seed)
    n = 500
    train = pd.DataFrame({f"x{i}": rng.standard_normal(n).astype(np.float64) for i in range(6)})
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=5)),
        ]
    )
    pipe.fit(train)
    return pipe, train


def _predict_frame_with_extra_cols(train_like: pd.DataFrame, n: int = 100):
    """Build a predict frame that has all the train-fit cols PLUS extra
    pre-filter cols (cat_low, cat_mid as objects + x_null as all-NaN)
    so sklearn's strict feature-name check would reject it."""
    rng = np.random.default_rng(1)
    cols = {c: rng.standard_normal(n).astype(np.float64) for c in train_like.columns}
    cols["cat_low"] = pd.array(["a", "b", "c"] * (n // 3) + ["a"] * (n - 3 * (n // 3)))
    cols["cat_mid"] = pd.array(["x"] * n, dtype="object")
    cols["x_null"] = np.full(n, np.nan, dtype=np.float64)
    return pd.DataFrame(cols)


def test_extra_cols_dropped_before_transform() -> None:
    """Predict frame carries the train-fit numeric cols PLUS extras;
    after the subset, transform must succeed and return pca0..pca4
    columns matching the fit-time output."""
    from mlframe.training.core.predict import _apply_extensions_pipeline

    pipe, train_like = _build_fitted_pipeline()
    df = _predict_frame_with_extra_cols(train_like, n=80)
    out = _apply_extensions_pipeline(df, pipe, verbose=0)
    assert isinstance(out, pd.DataFrame)
    # PCA output columns must be present (the bug surfaced because they
    # were absent after the swallowed transform error).
    pca_cols = [c for c in out.columns if str(c).startswith("pca")]
    assert len(pca_cols) == 5, f"expected 5 PCA output cols (pca0..pca4); got {pca_cols}"
    assert out.shape[0] == 80


def test_column_order_normalised_before_transform() -> None:
    """Even when predict frame has only the fit-time cols but in a
    different ORDER, the subset must re-order to match
    ``feature_names_in_`` so sklearn's name check passes."""
    from mlframe.training.core.predict import _apply_extensions_pipeline

    pipe, train_like = _build_fitted_pipeline()
    rng = np.random.default_rng(2)
    # Same cols but reversed order.
    cols = list(reversed(list(train_like.columns)))
    df = pd.DataFrame({c: rng.standard_normal(60).astype(np.float64) for c in cols})
    out = _apply_extensions_pipeline(df, pipe, verbose=0)
    # Successful transform produces shape (60, 5).
    assert out.shape == (60, 5)


def test_missing_fit_col_warns_then_hard_fails(caplog) -> None:
    """Predict frame missing a fit-time col -> WARN naming it; transform then fails at
    sklearn's check (we can't fabricate the col). A2-10: the failure now HARD-FAILS
    (raises RuntimeError) instead of silently returning the raw frame -- serving raw
    columns the model was not trained on produces wrong predictions, so a loud error
    is the correct surface."""
    from mlframe.training.core.predict import _apply_extensions_pipeline

    pipe, train_like = _build_fitted_pipeline()
    rng = np.random.default_rng(3)
    # Build a frame missing x0.
    df = pd.DataFrame({c: rng.standard_normal(40).astype(np.float64) for c in train_like.columns if c != "x0"})
    with caplog.at_level(logging.WARNING, logger="mlframe.training.core.predict"):
        with pytest.raises(RuntimeError, match="transform failed at predict time"):
            _apply_extensions_pipeline(df, pipe, verbose=0)
    # The "missing fit-time column" WARN must have fired before the hard-fail.
    assert any("missing" in rec.message and "fit-time" in rec.message and "x0" in rec.message for rec in caplog.records), (
        f"expected missing-col WARN; got: {[r.message for r in caplog.records]}"
    )


def test_exact_fit_cols_passes_through_unchanged_subset() -> None:
    """Predict frame with EXACTLY the fit-time cols in the same order:
    subset is a no-op; transform succeeds. Locks the no-op path so we
    don't regress on the common case where caller already pre-filtered."""
    from mlframe.training.core.predict import _apply_extensions_pipeline

    pipe, train_like = _build_fitted_pipeline()
    rng = np.random.default_rng(4)
    df = pd.DataFrame({c: rng.standard_normal(50).astype(np.float64) for c in train_like.columns})
    out = _apply_extensions_pipeline(df, pipe, verbose=0)
    assert out.shape == (50, 5)


def test_pipeline_without_feature_names_in_falls_through() -> None:
    """Some custom pipelines lack ``feature_names_in_``; subset must
    skip (no AttributeError) and the original transform contract
    applies."""
    from mlframe.training.core.predict import _apply_extensions_pipeline

    class _IdentityNoNames:
        # Mimics a fitted estimator that doesn't expose feature_names_in_.
        def transform(self, X):
            return X.values if isinstance(X, pd.DataFrame) else np.asarray(X)

        def get_feature_names_out(self):
            return [f"col_{i}" for i in range(6)]

    pipe = _IdentityNoNames()
    rng = np.random.default_rng(5)
    df = pd.DataFrame({f"x{i}": rng.standard_normal(30) for i in range(6)})
    out = _apply_extensions_pipeline(df, pipe, verbose=0)
    # No crash, transform produced the identity-shape output.
    assert out.shape == (30, 6)
