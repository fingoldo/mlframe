"""Reading several MPS parquet files where a later one lacks a requested column must work on polars 1.x and polars 2."""

from __future__ import annotations

import pytest

pl = pytest.importorskip("polars")

from mlframe.feature_engineering.mps import _read_parquet_insert_missing


def test_a_column_missing_from_a_later_file_is_read_as_nulls_instead_of_failing(tmp_path) -> None:
    """polars 2 dropped ``allow_missing_columns=True``; the old call raised TypeError and the caller swallowed it into ``None``."""
    first, second = tmp_path / "a.parquet", tmp_path / "b.parquet"
    pl.DataFrame({"ts": [1, 2], "grp": ["a", "a"], "price": [1.5, 2.5]}).write_parquet(first)
    pl.DataFrame({"ts": [3], "grp": ["b"]}).write_parquet(second)

    out = _read_parquet_insert_missing([first, second], ["ts", "grp", "price"])

    assert out.columns == ["ts", "grp", "price"]
    assert out["ts"].to_list() == [1, 2, 3]
    assert out["price"].to_list() == [1.5, 2.5, None]


def test_files_that_all_have_every_column_read_unchanged(tmp_path) -> None:
    """When nothing is missing the helper returns exactly what a plain read returns."""
    path = tmp_path / "full.parquet"
    frame = pl.DataFrame({"ts": [1, 2], "grp": ["a", "b"], "price": [1.5, 2.5]})
    frame.write_parquet(path)

    assert _read_parquet_insert_missing(path, ["ts", "grp", "price"]).equals(frame)
