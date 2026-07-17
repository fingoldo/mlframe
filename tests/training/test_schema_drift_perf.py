"""Regression sensors for ``pipeline._warn_on_schema_drift``.

History (2026-05-08): the function compared dtypes via native ``!=``,
which on certain polars dtype values dispatched to pandas
``Index.__eq__`` -> ``Series.equals`` machinery, costing ~270ms per
single comparison. On c0034 (3-model multiclass + inject_inf_nan,
60k rows): 6 schema-drift checks burned 1.6s of suite time on
no-actual-drift comparisons.

Fix: compare via ``str()`` representation. These tests guard the fix:
1. Performance ceiling -- 100 dtype comparisons must stay under 100ms.
2. Equivalence -- warn fires correctly on real dtype drift / missing
   columns and stays silent on identical schemas.
"""

from __future__ import annotations

import io
import logging
import time

import polars as pl
import pytest

from mlframe.training.pipeline import _warn_on_schema_drift


def test_schema_drift_dtype_compare_under_100ms():
    """100 _warn_on_schema_drift calls on a 4-col schema must stay
    under 100ms (i.e. ~1ms each). Pre-fix: ~270ms per call ->
    27000ms for 100 calls. The ceiling at 100ms catches reintroduction
    of the slow pandas ``__eq__`` dispatch.
    """
    train = pl.DataFrame(
        {
            "a": [1.0, 2.0],
            "b": [1, 2],
            "c": ["x", "y"],
            "d": [True, False],
        }
    )
    val = pl.DataFrame(
        {
            "a": [3.0],
            "b": [3],
            "c": ["z"],
            "d": [True],
        }
    )
    train_schema = dict(train.schema)

    t0 = time.perf_counter()
    for _ in range(100):
        _warn_on_schema_drift(train_schema, val, "val")
    elapsed_ms = (time.perf_counter() - t0) * 1000

    assert elapsed_ms < 100.0, (
        f"_warn_on_schema_drift x100 took {elapsed_ms:.1f}ms; expected "
        f"<100ms. Slow path regression -- check whether dtype comparison "
        f"is dispatching to pandas Series.equals via Index.__eq__. The "
        f"fix uses str(dtype) compare which is microseconds; native "
        f"dtype != was hitting ~270ms per call."
    )


def test_schema_drift_silent_on_identical():
    """Identical schemas: no WARN logged."""
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setLevel(logging.WARNING)
    logger = logging.getLogger("mlframe.training.pipeline")
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)

    try:
        train = pl.DataFrame({"a": [1.0, 2.0], "b": ["x", "y"]})
        val = pl.DataFrame({"a": [3.0], "b": ["z"]})
        _warn_on_schema_drift(dict(train.schema), val, "val")
    finally:
        logger.removeHandler(handler)

    assert stream.getvalue() == "", f"identical schemas should not warn; got: {stream.getvalue()!r}"


def test_schema_drift_warns_on_dtype_change():
    """Float64 -> Int64 must warn."""
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setLevel(logging.WARNING)
    logger = logging.getLogger("mlframe.training.pipeline")
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)

    try:
        train = pl.DataFrame({"a": [1.0]})
        val = pl.DataFrame({"a": [1]})
        _warn_on_schema_drift(dict(train.schema), val, "val")
    finally:
        logger.removeHandler(handler)

    out = stream.getvalue()
    assert "dtype different" in out
    assert "Float64" in out and "Int64" in out


def test_schema_drift_warns_on_missing_column():
    """Missing column must warn."""
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setLevel(logging.WARNING)
    logger = logging.getLogger("mlframe.training.pipeline")
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)

    try:
        train = pl.DataFrame({"a": [1.0], "b": [2.0]})
        val = pl.DataFrame({"a": [3.0]})
        _warn_on_schema_drift(dict(train.schema), val, "val")
    finally:
        logger.removeHandler(handler)

    out = stream.getvalue()
    assert "missing" in out and "'b'" in out
