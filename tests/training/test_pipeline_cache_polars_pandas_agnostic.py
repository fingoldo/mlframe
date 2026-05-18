"""Pipeline cache key must be polars/pandas-agnostic for identical-dtype columns.

Pre-fix: polars and pandas inputs with the SAME logical dtypes (e.g., Int32 / int32) produced DIFFERENT cache keys because the ``_dt`` suffix was conditional on input type. Production TVT log showed the same logical preprocessing block being computed TWICE -- once with polars input (cached under one key) and once with pandas input (cached under another key) -- wasting wall-clock + RAM.

The fix canonicalises dtype names (``Int32`` / ``int32`` -> ``i32``; ``Float32`` / ``float32`` -> ``f32``; ``Boolean`` / ``bool`` -> ``b``; ``Utf8`` / ``String`` / ``object`` -> ``s``; ``Categorical`` / ``category`` -> ``c``) so the ``_dt`` suffix matches across backends.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from mlframe.training.core._phase_train_one_target import (
    _canonicalise_dtype,
    _canonical_dtype_pairs,
    _compute_pipeline_cache_key,
)


class TestCanonicaliseDtype:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("Int8", "i8"), ("Int16", "i16"), ("Int32", "i32"), ("Int64", "i64"),
            ("int8", "i8"), ("int16", "i16"), ("int32", "i32"), ("int64", "i64"),
            ("UInt32", "u32"), ("uint32", "u32"),
            ("Float32", "f32"), ("Float64", "f64"),
            ("float32", "f32"), ("float64", "f64"),
            ("Boolean", "b"), ("bool", "b"),
            ("Utf8", "s"), ("String", "s"), ("object", "s"),
            ("Categorical", "c"), ("category", "c"),
        ],
    )
    def test_canonical_form(self, raw: str, expected: str) -> None:
        assert _canonicalise_dtype(raw) == expected


class TestCanonicalDtypePairs:
    def test_polars_and_pandas_with_same_dtypes_match(self) -> None:
        arr_a = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        arr_b = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
        df_pl = pl.DataFrame({"a": arr_a, "b": arr_b})
        df_pd = pd.DataFrame({"a": arr_a, "b": arr_b})
        assert _canonical_dtype_pairs(df_pl) == _canonical_dtype_pairs(df_pd)

    def test_different_dtypes_produce_different_pairs(self) -> None:
        df_a = pd.DataFrame({"x": np.array([1, 2, 3], dtype=np.int32)})
        df_b = pd.DataFrame({"x": np.array([1.0, 2.0, 3.0], dtype=np.float64)})
        assert _canonical_dtype_pairs(df_a) != _canonical_dtype_pairs(df_b)


class TestPipelineCacheKeyAcrossBackends:
    def test_polars_and_pandas_dt_suffix_matches(self) -> None:
        arr_a = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        arr_b = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
        df_pl = pl.DataFrame({"a": arr_a, "b": arr_b})
        df_pd = pd.DataFrame({"a": arr_a, "b": arr_b})
        key_pl = _compute_pipeline_cache_key(
            "tree", None, (True, True), True, [], [], [], train_df=df_pl,
        )
        key_pd = _compute_pipeline_cache_key(
            "tree", None, (True, True), False, [], [], [], train_df=df_pd,
        )
        # _kind suffix differs (pl vs pd); _dt suffix MUST match.
        pl_dt = key_pl.split("_dt")[-1] if "_dt" in key_pl else None
        pd_dt = key_pd.split("_dt")[-1] if "_dt" in key_pd else None
        assert pl_dt is not None
        assert pd_dt is not None
        assert pl_dt == pd_dt, (
            f"polars vs pandas _dt suffix differs: pl={pl_dt!r} vs pd={pd_dt!r}"
        )

    def test_polars_int32_matches_pandas_int32_full_key_modulo_kind(self) -> None:
        """The full cache key should differ ONLY by the ``_kind`` segment (pl vs pd). Stripping that segment, polars and pandas keys MUST be byte-identical."""
        arr_a = np.array([1, 2, 3], dtype=np.int32)
        df_pl = pl.DataFrame({"a": arr_a})
        df_pd = pd.DataFrame({"a": arr_a})
        key_pl = _compute_pipeline_cache_key(
            "tree", "MRMR", "T", True, ["a"], [], [], train_df=df_pl,
        )
        key_pd = _compute_pipeline_cache_key(
            "tree", "MRMR", "T", False, ["a"], [], [], train_df=df_pd,
        )
        # Remove the _kind segment from both before comparing.
        assert key_pl.replace("_kindpl", "_kind") == key_pd.replace("_kindpd", "_kind")
