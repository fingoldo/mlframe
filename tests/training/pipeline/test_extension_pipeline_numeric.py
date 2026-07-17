"""Regression: ``apply_preprocessing_extensions`` must drop non-numeric
columns before the sklearn-bridge pipeline.

Pre-fix path (1M-row harness ``_profile_fuzz_1m`` with seed=11):
1. Harness frame includes ``cat_mid`` (object dtype with values like
   ``M03``) and ``emb`` (object dtype with list-of-float embeddings).
2. The fuzz axes draw ``polynomial_degree=2``, so the sklearn-bridge
   pipeline activates: median imputer + polynomial features.
3. SimpleImputer(strategy="median") raised
   ``ValueError: Cannot use median strategy with non-numeric data,
   could not convert string to float: 'M03'``. (Fixed first by
   wrapping in a ColumnTransformer.)
4. With imputer wrapped, the non-numeric column then leaked to the
   PolynomialFeatures step which raised
   ``ValueError: The truth value of an array with more than one
   element is ambiguous`` during its internal validation.

Post-fix: ``apply_preprocessing_extensions`` filters its input frames
to numeric-only columns BEFORE constructing the sklearn-bridge
pipeline; non-numeric columns are logged at WARNING and dropped from
all three splits. The sklearn-bridge contract is now explicit:
"numeric only — encode strings upstream".
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from mlframe.training.configs import PreprocessingExtensionsConfig
from mlframe.training.pipeline import apply_preprocessing_extensions


def _make_mixed_frame(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Make mixed frame."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "x0": rng.standard_normal(n).astype(np.float32),
            "x1": rng.standard_normal(n).astype(np.float32),
            "x2": rng.standard_normal(n).astype(np.float32),
            # Unencoded string categorical (the harness's cat_mid shape).
            "cat_mid": np.array([f"M{i % 7:02d}" for i in range(n)], dtype=object),
        }
    )


def test_polynomial_with_string_column_drops_string_no_crash(caplog) -> None:
    """The polynomial-degree=2 path on a frame with cat_mid='M03' must
    not raise. Pre-fix this surfaced as ``ValueError: The truth value
    of an array with more than one element is ambiguous``."""
    df_train = _make_mixed_frame(200, seed=0)
    df_val = _make_mixed_frame(50, seed=1)
    df_test = _make_mixed_frame(50, seed=2)
    cfg = PreprocessingExtensionsConfig(polynomial_degree=2)
    with caplog.at_level(logging.WARNING, logger="mlframe.training.pipeline"):
        out = apply_preprocessing_extensions(
            df_train,
            df_val,
            df_test,
            cfg,
            verbose=0,
        )
    # Function returns (train, val, test, tfidf_pipes_or_None)
    assert out is not None
    out_train, out_val, out_test, _ = out
    assert isinstance(out_train, pd.DataFrame)
    # ``cat_mid`` must NOT appear in the output -- it was filtered
    # before the sklearn-bridge ever saw it.
    for _df, _label in ((out_train, "train"), (out_val, "val"), (out_test, "test")):
        if _df is None:
            continue
        assert "cat_mid" not in _df.columns, (
            f"cat_mid leaked into {_label} output -- the numeric-only filter at apply_preprocessing_extensions entry did not fire."
        )
    # The drop must be visible in the log (single WARN line with
    # the dropped column names).
    assert any("dropped" in rec.message and "cat_mid" in rec.message for rec in caplog.records), (
        f"expected dropped-non-numeric WARN; got: {[r.message for r in caplog.records]}"
    )


def test_scaler_with_object_column_no_truth_value_error() -> None:
    """RobustScaler is the other axis the 1M harness toggles. Same
    contract: object dtype must be filtered, not leaked."""
    df_train = _make_mixed_frame(150, seed=3)
    cfg = PreprocessingExtensionsConfig(scaler="RobustScaler")
    out = apply_preprocessing_extensions(df_train, None, None, cfg, verbose=0)
    assert out is not None
    out_train = out[0]
    assert "cat_mid" not in out_train.columns


def test_all_numeric_frame_unchanged_by_filter() -> None:
    """Baseline: a frame with no non-numeric columns must round-trip
    unchanged through the filter. Avoids the filter regressing valid
    pure-numeric callers."""
    df_train = pd.DataFrame(
        {
            "x0": np.random.default_rng(4).standard_normal(100),
            "x1": np.random.default_rng(5).standard_normal(100),
        }
    )
    cfg = PreprocessingExtensionsConfig(polynomial_degree=2)
    out = apply_preprocessing_extensions(df_train.copy(), None, None, cfg, verbose=0)
    assert out is not None
    out_train = out[0]
    # Polynomial degree=2 expands 2 cols to {x0, x1, x0^2, x0*x1, x1^2}
    # (5 cols when interaction_only=False). The exact count depends on
    # sklearn's PolynomialFeatures defaults; just assert it's >= the
    # input count, i.e. the filter didn't strip valid numeric input.
    assert out_train.shape[1] >= 2
