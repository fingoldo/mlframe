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
    """Groups tests covering canonicalise dtype."""
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("Int8", "i8"),
            ("Int16", "i16"),
            ("Int32", "i32"),
            ("Int64", "i64"),
            ("int8", "i8"),
            ("int16", "i16"),
            ("int32", "i32"),
            ("int64", "i64"),
            ("UInt32", "u32"),
            ("uint32", "u32"),
            ("Float32", "f32"),
            ("Float64", "f64"),
            ("float32", "f32"),
            ("float64", "f64"),
            ("Boolean", "b"),
            ("bool", "b"),
            ("Utf8", "s"),
            ("String", "s"),
            ("object", "s"),
            ("Categorical", "c"),
            ("category", "c"),
        ],
    )
    def test_canonical_form(self, raw: str, expected: str) -> None:
        """Canonical form."""
        assert _canonicalise_dtype(raw) == expected


class TestCanonicalDtypePairs:
    """Groups tests covering canonical dtype pairs."""
    def test_polars_and_pandas_with_same_dtypes_match(self) -> None:
        """Polars and pandas with same dtypes match."""
        arr_a = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        arr_b = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
        df_pl = pl.DataFrame({"a": arr_a, "b": arr_b})
        df_pd = pd.DataFrame({"a": arr_a, "b": arr_b})
        assert _canonical_dtype_pairs(df_pl) == _canonical_dtype_pairs(df_pd)

    def test_different_dtypes_produce_different_pairs(self) -> None:
        """Different dtypes produce different pairs."""
        df_a = pd.DataFrame({"x": np.array([1, 2, 3], dtype=np.int32)})
        df_b = pd.DataFrame({"x": np.array([1.0, 2.0, 3.0], dtype=np.float64)})
        assert _canonical_dtype_pairs(df_a) != _canonical_dtype_pairs(df_b)

    def test_polars_enum_canonicalises_to_c(self) -> None:
        """Polars Enum is in the same dtype FAMILY as Categorical (both backed by
        int codes + a dictionary), so it must canonicalise to "c" for cache-key
        purposes -- not to the full ``Enum(categories=['...', ...])`` repr its
        ``str()`` produces. Aligns Enum with Categorical, restoring cache-hit
        semantics across the iter470 polars->pandas bridge which promotes every
        Categorical column to an Enum with that column's actual category list."""
        df = pl.DataFrame({"c": pl.Series(["a", "b", "c"], dtype=pl.Enum(["a", "b", "c"]))})
        pairs = _canonical_dtype_pairs(df)
        assert pairs == (("c", "c"),)

    def test_polars_enum_and_categorical_canonicalise_to_same(self) -> None:
        """An Enum column and a Categorical column with identical column NAMES
        must produce identical canonical pairs (both -> "c"). The category lists
        are part of data content, not dtype shape, and are hashed separately by
        the content hashers. Without this, the iter470 bridge -- which produces
        Enum frames with column-specific category universes -- would cache-miss
        on what is logically the same preprocessing block."""
        df_enum = pl.DataFrame({"c": pl.Series(["a", "b"], dtype=pl.Enum(["a", "b"]))})
        df_cat = pl.DataFrame({"c": pl.Series(["a", "b"], dtype=pl.Categorical)})
        assert _canonical_dtype_pairs(df_enum) == _canonical_dtype_pairs(df_cat)

    def test_polars_enum_different_categories_canonicalise_same(self) -> None:
        """Two Enum columns with DIFFERENT category universes but the same
        column name must produce the same canonical pair. Cache hits across
        bridge slices (train/val/test) that each Enum'd with their own
        category subset rely on this."""
        df_a = pl.DataFrame({"c": pl.Series(["a", "b"], dtype=pl.Enum(["a", "b", "c", "d"]))})
        df_b = pl.DataFrame({"c": pl.Series(["x", "y"], dtype=pl.Enum(["x", "y", "z"]))})
        assert _canonical_dtype_pairs(df_a) == _canonical_dtype_pairs(df_b) == (("c", "c"),)


class TestPipelineCacheKeyAcrossBackends:
    """Groups tests covering pipeline cache key across backends."""
    def test_polars_and_pandas_dt_suffix_matches(self) -> None:
        """Polars and pandas dt suffix matches."""
        arr_a = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        arr_b = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
        df_pl = pl.DataFrame({"a": arr_a, "b": arr_b})
        df_pd = pd.DataFrame({"a": arr_a, "b": arr_b})
        key_pl = _compute_pipeline_cache_key(
            "tree",
            None,
            (True, True),
            True,
            [],
            [],
            [],
            train_df=df_pl,
        )
        key_pd = _compute_pipeline_cache_key(
            "tree",
            None,
            (True, True),
            False,
            [],
            [],
            [],
            train_df=df_pd,
        )
        # _kind suffix differs (pl vs pd); _dt suffix MUST match.
        pl_dt = key_pl.split("_dt")[-1] if "_dt" in key_pl else None
        pd_dt = key_pd.split("_dt")[-1] if "_dt" in key_pd else None
        assert pl_dt is not None
        assert pd_dt is not None
        assert pl_dt == pd_dt, f"polars vs pandas _dt suffix differs: pl={pl_dt!r} vs pd={pd_dt!r}"

    def test_polars_int32_matches_pandas_int32_full_key_modulo_kind(self) -> None:
        """The full cache key should differ ONLY by the ``_kind`` segment (pl vs pd). Stripping that segment, polars and pandas keys MUST be byte-identical."""
        arr_a = np.array([1, 2, 3], dtype=np.int32)
        df_pl = pl.DataFrame({"a": arr_a})
        df_pd = pd.DataFrame({"a": arr_a})
        key_pl = _compute_pipeline_cache_key(
            "tree",
            "MRMR",
            "T",
            True,
            ["a"],
            [],
            [],
            train_df=df_pl,
        )
        key_pd = _compute_pipeline_cache_key(
            "tree",
            "MRMR",
            "T",
            False,
            ["a"],
            [],
            [],
            train_df=df_pd,
        )
        # Remove the _kind segment from both before comparing.
        assert key_pl.replace("_kindpl", "_kind") == key_pd.replace("_kindpd", "_kind")
