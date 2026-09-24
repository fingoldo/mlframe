"""The batched polars encoder gives every column exactly the bytes the per-column encoder gives it.

The data signature's sample and row-order windows encoded each column with three or four tiny polars expressions, about 2 s
at 200k x 500. ``encode_polars_frame`` encodes a dtype group per ``select``; its bytes feed cache keys, so they must not move.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

pl = pytest.importorskip("polars")

from mlframe.training.composite._canonical_hash import encode_polars_frame, encode_polars_slice, polars_logical_type


def test_every_dtype_family_encodes_byte_identically():
    """Floats with NaN, nulls and -0.0, ints of several widths and UInt64, bools with nulls, strings, categoricals, dates,
    datetimes in ms: each column's batched bytes equal its own per-column encode."""
    rng = np.random.default_rng(0)
    n = 300
    f = rng.normal(size=n)
    f[::7] = np.nan
    f[1] = -0.0
    df = pl.DataFrame({
        "f64": f, "f32": rng.normal(size=n).astype(np.float32), "i8": rng.integers(-100, 100, n).astype(np.int8),
        "i64": rng.integers(0, 10**12, n), "u64": rng.integers(0, 10**12, n).astype(np.uint64), "b": rng.uniform(size=n) < 0.3,
        "s": [f"v{i % 11}" for i in range(n)],
    }).with_columns(
        pl.col("s").cast(pl.Categorical).alias("cat"),
        pl.Series("d", [dt.date(2020, 1, 1) + dt.timedelta(days=i) for i in range(n)]),
        pl.Series("ts", [dt.datetime(2020, 1, 1) + dt.timedelta(seconds=i) for i in range(n)]).dt.cast_time_unit("ms"),
        pl.when(pl.col("i8") > 50).then(None).otherwise(pl.col("i64")).alias("i64n"),
        pl.when(pl.col("b")).then(None).otherwise(pl.col("f32")).alias("f32n"),
        pl.when(pl.col("i8") > 60).then(None).otherwise(pl.col("b")).alias("bn"),
    )
    tokens = {c: polars_logical_type(t) for c, t in df.schema.items()}
    batched = encode_polars_frame(df, tokens)
    assert set(batched) == set(df.columns)
    for c in df.columns:
        assert batched[c] == encode_polars_slice(df.get_column(c), tokens[c]), c
