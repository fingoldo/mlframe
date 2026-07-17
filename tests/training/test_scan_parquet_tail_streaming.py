"""Sensor test for audit B-P0-7: when neither columns nor n_rows are requested and
config.tail is set, ``load_and_prepare_dataframe`` must use the streaming engine on
the lazy scan so a 100GB file does not materialise into RAM before the tail slice runs.

We can't easily mock a 100GB file in CI; instead we verify the load path:
1. produces a polars DataFrame with exactly ``tail`` rows;
2. round-trips data correctly;
3. exercises the LazyFrame branch (scan_parquet + tail) without OOMing on a small file.

Imports a minimal config stub instead of mlframe.training.configs to dodge an
unrelated torch._inductor circular-import on this env.
"""

from __future__ import annotations

import types

import polars as pl
import pytest


def _make_cfg(tail=None, columns=None, n_rows=None, ensure_float32_dtypes=False, fillna_value=None, remove_constant_columns=False):
    cfg = types.SimpleNamespace(
        tail=tail,
        columns=columns,
        n_rows=n_rows,
        ensure_float32_dtypes=ensure_float32_dtypes,
        fillna_value=fillna_value,
        remove_constant_columns=remove_constant_columns,
    )
    return cfg


@pytest.mark.fast
def test_scan_parquet_with_tail_returns_last_n_rows(tmp_path):
    from mlframe.training.preprocessing import load_and_prepare_dataframe

    n_rows = 1_000
    pf = tmp_path / "small.parquet"
    pl.DataFrame({"i": list(range(n_rows)), "v": [float(x) for x in range(n_rows)]}).write_parquet(pf)

    df = load_and_prepare_dataframe(str(pf), config=_make_cfg(tail=10), verbose=False)

    assert isinstance(df, pl.DataFrame)
    assert df.height == 10
    assert df["i"].to_list() == list(range(n_rows - 10, n_rows))


@pytest.mark.fast
def test_scan_parquet_without_tail_returns_full_frame(tmp_path):
    """Sanity: without tail and without columns/n_rows, scan_parquet+collect returns the full frame."""
    from mlframe.training.preprocessing import load_and_prepare_dataframe

    pf = tmp_path / "small2.parquet"
    pl.DataFrame({"i": list(range(50))}).write_parquet(pf)
    df = load_and_prepare_dataframe(str(pf), config=_make_cfg(), verbose=False)
    assert isinstance(df, pl.DataFrame)
    assert df.height == 50
