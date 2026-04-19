"""Round 11 sensors for the TRUE root cause of CB/XGB Polars fastpath
failures with Polars 1.x (diagnosed 2026-04-19 after rounds 7-10).

Root cause: Polars 1.x + pyarrow 15+ exports
  - ``pl.String`` → ``pa.large_string()`` (64-bit offsets)
  - ``pl.Categorical`` → ``Dictionary<uint32, large_string>``

CatBoost 1.2.10's ``_set_features_order_data_polars_categorical_column``
Cython fused cpdef has no dispatch signature for large_string variant
→ ``TypeError: No matching signature found`` + 20-50 s wasted per
failed attempt + 3× redundant polars→pandas conversions (fit, val
predict, test predict).

XGBoost 3.x's ``xgboost.data._arrow_dtype`` maps only numeric types;
plain large_string columns hit ``KeyError: DataType(large_string)``
inside ``_wrap_evaluation_matrices`` killing the entire suite.

Fix: ``_polars_df_emits_large_string`` in trainer.py detects the
issue via ``head(0).to_arrow()`` schema check. Called from
core.py's polars-fastpath block — if True, ``polars_fastpath_active``
flips to False BEFORE the fit, the pandas tier DF is built now,
and the rest of training uses pandas end-to-end. Same code path
handles both CB (would hit the TypeError) and XGB (would hit the
KeyError) — they share the detection + bypass.

Note on the XGB ``_arrow_dtype`` monkey-patch: an earlier iteration
tried to wrap ``xgboost.data._arrow_dtype`` to map ``large_string``
→ ``'c'``. Direct repro showed this only moves the crash deeper —
from a KeyError in ``_arrow_feature_info`` to a ``ValueError: too
many values to unpack`` in the downstream dictionary-handling code.
XGB really does require a Dictionary-encoded column (not plain
``large_string``) for the 'c' code to work. The proactive bypass
above avoids the issue entirely by never handing a large-string
Arrow frame to XGB — reverted the shim in favour of that.
"""
from __future__ import annotations

import numpy as np
import polars as pl
import pyarrow as pa
import pytest


# =============================================================================
# Fix — _polars_df_emits_large_string detector
# =============================================================================


class TestPolarsDfEmitsLargeStringDetector:

    def test_detects_plain_string_column(self):
        from mlframe.training.trainer import _polars_df_emits_large_string
        df = pl.DataFrame({"s": ["a", "b", "c"]})
        assert _polars_df_emits_large_string(df)

    def test_detects_categorical_with_large_string_values(self):
        """The production-common case: pl.Categorical columns become
        Dictionary<uint32, large_string> in the Arrow export. CB's
        fused cpdef doesn't dispatch on this shape; XGB's arrow
        handler chokes on the inner large_string too."""
        from mlframe.training.trainer import _polars_df_emits_large_string
        df = pl.DataFrame({"c": pl.Series("c", ["x", "y", "z"]).cast(pl.Categorical)})
        assert _polars_df_emits_large_string(df)

    def test_numeric_only_df_returns_false(self):
        """False-positive sensor: purely numeric DFs should NOT trip the
        detector (those work fine with the polars fastpath)."""
        from mlframe.training.trainer import _polars_df_emits_large_string
        df = pl.DataFrame({
            "num": np.arange(5, dtype=np.float32),
            "i": np.arange(5, dtype=np.int16),
            "b": [True, False, True, False, True],
        })
        assert not _polars_df_emits_large_string(df)

    def test_empty_df_does_not_crash(self):
        """head(0) call should work even on already-empty DFs."""
        from mlframe.training.trainer import _polars_df_emits_large_string
        df = pl.DataFrame({"s": pl.Series("s", [], dtype=pl.String)})
        # Even empty, the *schema* still has large_string, so detector
        # returns True.
        assert _polars_df_emits_large_string(df)

    def test_non_polars_input_returns_false(self):
        """Defensive: pandas DF / ndarray / None → returns False rather
        than crashing. Callers don't need to pre-check."""
        from mlframe.training.trainer import _polars_df_emits_large_string
        import pandas as pd
        assert not _polars_df_emits_large_string(pd.DataFrame({"a": [1, 2]}))
        assert not _polars_df_emits_large_string(None)
        assert not _polars_df_emits_large_string(np.arange(5))
