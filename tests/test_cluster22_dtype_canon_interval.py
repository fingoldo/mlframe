"""Cluster 22 regression: dtype canonicalisation does not mistake interval dtypes for int (audit 2026-11-20, finding 22-F37)."""

import pytest

from mlframe._dtype_canon import canonicalise_dtype


def test_interval_dtype_is_not_an_int_family():
    """``interval[int64, right]`` keeps its own identity; it is not rewritten to an i-prefixed int code (22-F37)."""
    out = canonicalise_dtype("interval[int64, right]")
    assert not out.startswith("i64")
    assert out != canonicalise_dtype("int64")


@pytest.mark.parametrize("raw, expected", [("int64", "i64"), ("uint8", "u8"), ("float32", "f32")])
def test_numeric_families_still_canonicalise(raw, expected):
    """Control: plain numeric dtypes keep their short codes (22-F37)."""
    assert canonicalise_dtype(raw) == expected
