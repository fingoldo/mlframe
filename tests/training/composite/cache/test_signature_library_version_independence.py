"""`data_signature` must key on the DATA, never on how the installed pandas / polars / numpy represent it.

Moving to pandas 3 changed seven pinned discovery-cache keys with no mlframe change: default strings became `str`
(was `object`), `date_range` / `to_timedelta` inferred `us` / `s` resolutions (was `ns`), and the key folded in
`str(dtype)`, `hash_pandas_object` and raw buffer bytes, all of which followed. Every entry written under pandas 2
became unreachable. These tests pin the representation-independence directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.training.composite.cache import _row_order_fingerprint, data_signature

pl = pytest.importorskip("polars")

_STRINGS = ["alpha", None, "beta", "gamma", "alpha", "", "delta"] * 30


def _frame(strings_dtype=object) -> pd.DataFrame:
    n = len(_STRINGS)
    return pd.DataFrame(
        {
            "y": np.linspace(0.0, 1.0, n),
            "num": np.arange(n, dtype=np.float64),
            "s": pd.Series(_STRINGS, dtype=strings_dtype),
        }
    )


def _sig(df) -> str:
    return data_signature(df, "y", ["num", "s"], sample_n=50)


class TestStringRepresentationDoesNotMoveTheKey:
    """object / string[python] / string[pyarrow] / pandas default `str` hold the same logical strings."""

    @pytest.mark.parametrize("dtype", ["string[python]", "str"])
    def test_string_dtypes_key_like_object(self, dtype):
        assert _sig(_frame(dtype)) == _sig(_frame(object))

    def test_pyarrow_strings_key_like_object(self):
        pytest.importorskip("pyarrow")
        assert _sig(_frame("string[pyarrow]")) == _sig(_frame(object))

    def test_categorical_strings_key_like_polars_categorical(self):
        df = _frame(object).astype({"s": "category"})
        assert _sig(df) == _sig(pl.from_pandas(df))


class TestFrameLibraryDoesNotMoveTheKey:
    """A polars frame of the same data keys like the pandas frame, so either caller hits the other's entries."""

    def test_mixed_frame_parity(self):
        n = 400
        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            {
                "y": rng.normal(size=n),
                "i": rng.integers(-5, 5, n),
                "f32": rng.normal(size=n).astype(np.float32),
                "b": rng.random(n) > 0.5,
                "s": pd.Series(np.array(["a", "bb", "ccc"])[rng.integers(0, 3, n)], dtype=object),
                "t": pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC"),
                "d": pd.to_timedelta(rng.integers(0, 1000, n), unit="s"),
            }
        )
        df.loc[3, "f32"] = np.nan
        feats = ["i", "f32", "b", "s", "t", "d"]
        assert data_signature(df, "y", feats) == data_signature(pl.from_pandas(df), "y", feats)
        assert _row_order_fingerprint(df) == _row_order_fingerprint(pl.from_pandas(df))


class TestDatetimeResolutionDoesNotMoveTheKey:
    """pandas 3 infers `us` / `s` where pandas 2 used `ns`; the instants and durations are the same data."""

    @pytest.mark.parametrize("unit", ["s", "ms", "us"])
    def test_datetime_unit(self, unit):
        base = pd.DataFrame({"y": np.arange(20.0), "t": pd.date_range("2026-01-01", periods=20, freq="D").astype("datetime64[ns]")})
        other = base.astype({"t": f"datetime64[{unit}]"})
        assert data_signature(other, "y", ["t"]) == data_signature(base, "y", ["t"])

    @pytest.mark.parametrize("unit", ["s", "ms", "us"])
    def test_timedelta_unit(self, unit):
        base = pd.DataFrame({"y": np.arange(20.0), "d": pd.to_timedelta(np.arange(20), unit="D").astype("timedelta64[ns]")})
        other = base.astype({"d": f"timedelta64[{unit}]"})
        assert data_signature(other, "y", ["d"]) == data_signature(base, "y", ["d"])


class TestNumericRepresentationRule:
    """Documented rule: ints key by value whatever their width / signedness / nullability (the platform default int
    width differs between numpy builds); nullable dtypes key like their numpy twins; float widths stay distinct because
    float32 holds different values; a null is a null whether stored as NaN or NA."""

    def test_int_width_does_not_move_the_key(self):
        a = pd.DataFrame({"y": np.arange(30.0), "i": np.arange(30, dtype=np.int32)})
        assert data_signature(a, "y", ["i"]) == data_signature(a.astype({"i": np.int64}), "y", ["i"])

    def test_nullable_without_nulls_keys_like_numpy(self):
        a = pd.DataFrame({"y": np.arange(30.0), "i": np.arange(30), "f": np.linspace(0, 1, 30), "b": np.arange(30) % 2 == 0})
        b = a.astype({"i": "Int64", "f": "Float64", "b": "boolean"})
        assert data_signature(a, "y", ["i", "f", "b"]) == data_signature(b, "y", ["i", "f", "b"])

    def test_float_nan_and_na_are_the_same_null(self):
        a = pd.DataFrame({"y": np.arange(30.0), "f": np.linspace(0, 1, 30)})
        a.loc[4, "f"] = np.nan
        b = a.astype({"f": "Float64"})
        assert pd.isna(b.loc[4, "f"])
        assert data_signature(a, "y", ["f"]) == data_signature(b, "y", ["f"])

    def test_float_width_moves_the_key(self):
        a = pd.DataFrame({"y": np.arange(30.0), "f": np.linspace(0, 1, 30)})
        assert data_signature(a, "y", ["f"]) != data_signature(a.astype({"f": np.float32}), "y", ["f"])

    def test_int_vs_float_moves_the_key(self):
        a = pd.DataFrame({"y": np.arange(30.0), "v": np.arange(30)})
        assert data_signature(a, "y", ["v"]) != data_signature(a.astype({"v": np.float64}), "y", ["v"])


class TestSensitivityIsKept:
    """Canonicalising the representation must not blunt the key."""

    def test_head_swap_moves_the_key(self):
        df = _frame(object)
        swapped = df.iloc[[1, 0, *list(range(2, len(df)))]].reset_index(drop=True)
        assert _sig(swapped) != _sig(df)

    def test_tail_swap_moves_the_key(self):
        n = 2000
        df = pd.DataFrame({"y": np.arange(n, dtype=np.float64), "num": np.arange(n, dtype=np.float64), "s": ["x"] * n})
        order = [*list(range(n - 2)), n - 1, n - 2]
        assert _sig(df.iloc[order].reset_index(drop=True)) != _sig(df)

    def test_appended_row_moves_the_key(self):
        df = _frame(object)
        grown = pd.concat([df, df.iloc[[5]]], ignore_index=True)
        assert _sig(grown) != _sig(df)

    def test_changed_string_moves_the_key(self):
        df = _frame(object)
        changed = df.copy()
        changed.loc[len(df) // 2, "s"] = "zeta"
        assert _sig(changed) != _sig(df)

    def test_null_vs_empty_string_moves_the_key(self):
        df = _frame(object)
        changed = df.copy()
        changed.loc[1, "s"] = ""
        assert _sig(changed) != _sig(df)


class TestNoLibraryHasherIsConsulted:
    """The regression guard for the actual bug: the key must be computable with pandas' and polars' own row hashers
    unavailable, proving neither (both unstable across library versions) reaches it."""

    def test_signature_without_library_hashers(self, monkeypatch):
        import pandas.util

        def _boom(*_a, **_k):
            raise AssertionError("library row hasher reached the cache key")

        expected_pd = _sig(_frame("str"))
        expected_pl = _sig(pl.from_pandas(_frame(object)))
        monkeypatch.setattr(pandas.util, "hash_pandas_object", _boom)
        monkeypatch.setattr(pd.util, "hash_pandas_object", _boom, raising=False)
        monkeypatch.setattr(pl.DataFrame, "hash_rows", _boom)
        monkeypatch.setattr(pl.Series, "hash", _boom)

        for frame in (_frame(object), _frame("str")):
            assert _row_order_fingerprint(frame) != ""
            assert _sig(frame) == expected_pd
        polars_frame = pl.from_pandas(_frame(object))
        assert _row_order_fingerprint(polars_frame) != ""
        assert _sig(polars_frame) == expected_pl == expected_pd
