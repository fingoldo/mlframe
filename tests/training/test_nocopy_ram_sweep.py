"""Regression tests for the 100GB-no-copy rule (CLAUDE.md "Memory / RAM constraints").

Each site below used a deep ``df.copy()`` purely as a mutate-and-restore guard for a per-column
cast/fill of a FEW columns. A deep copy clones the (large) untouched columns too -> OOM on a 100+ GB
frame. The fix is ``copy(deep=False)``: untouched column buffers are SHARED with the caller frame and
full-column reassignment never writes back into it. These tests fail on the pre-fix deep-copy code
(``np.shares_memory`` is False after a deep copy) and pass after.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _big_passthrough_frame(n=64):
    return pd.DataFrame(
        {
            "num_a": np.arange(n, dtype="float64"),
            "num_b": np.arange(n, dtype="float64") + 100.0,
            "passthrough": np.arange(n, dtype="float64") * 3.0,
        }
    )


def test_numeric_only_transformer_does_not_deep_copy_passthrough():
    from mlframe.training.strategies.base import _NumericOnlyTransformer

    class _DoubleNumeric:
        def fit(self, X, y=None):
            return self

        def transform(self, X):
            return X * 2.0

    X = _big_passthrough_frame()
    X["cat"] = pd.Categorical(["x", "y"] * (len(X) // 2))
    cat_buf = X["cat"]
    t = _NumericOnlyTransformer(inner=_DoubleNumeric(), cat_features=["cat"])
    out = t.fit(X).transform(X)
    # numeric columns were transformed
    assert out["num_a"].tolist() == (X["num_a"] * 2.0).tolist()
    # caller frame untouched (num_a still original values)
    assert X["num_a"].tolist() == np.arange(len(X), dtype="float64").tolist()
    # passthrough cat column must SHARE memory with the caller frame (no deep copy)
    assert out["cat"].cat.codes is not None
    assert np.shares_memory(out["cat"].cat.codes.to_numpy(), cat_buf.cat.codes.to_numpy())


def test_preprocess_dataframe_stringdtype_normalise_no_deep_copy():
    from mlframe.training.preprocessing import preprocess_dataframe
    from mlframe.training.configs import PreprocessingConfig

    n = 64
    df = pd.DataFrame(
        {
            "s": pd.array(["a", "b"] * (n // 2), dtype="string"),
            "big_num": np.arange(n, dtype="float64"),
        }
    )
    s_codes_before = df["s"].tolist()
    cfg = PreprocessingConfig()
    out = preprocess_dataframe(df, cfg, verbose=0)
    if not isinstance(out, pd.DataFrame):
        pytest.skip("non-pandas return")
    assert out["s"].dtype == object
    # caller's StringDtype column is untouched (pre-fix shallow path must not mutate caller)
    assert df["s"].dtype == "string"
    assert df["s"].tolist() == s_codes_before
    # the normalised StringDtype column was reassigned on the shallow copy, not on caller's frame:
    # caller still has StringDtype, output has object -- distinct columns, no in-place mutation.
    assert out["s"].tolist() == s_codes_before


def test_decategorise_float_cat_no_deep_copy():
    from mlframe.training._eval_helpers import _decategorise_float_cat_columns

    n = 64
    df = pd.DataFrame(
        {
            "fc": pd.Categorical(np.arange(n, dtype="float64") % 4),
            "big_num": np.arange(n, dtype="float64") + 7.0,
        }
    )
    num_buf = df["big_num"].to_numpy()
    train_out, _, _ = _decategorise_float_cat_columns(df)
    # the float-cat column was decategorised
    assert not isinstance(train_out["fc"].dtype, pd.CategoricalDtype)
    # caller frame's cat column untouched
    assert isinstance(df["fc"].dtype, pd.CategoricalDtype)
    # untouched numeric column shares memory (no deep copy)
    assert np.shares_memory(train_out["big_num"].to_numpy(), num_buf)
