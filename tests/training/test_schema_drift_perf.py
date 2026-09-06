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

import polars as pl

from mlframe.training.pipeline import _warn_on_schema_drift


def test_schema_drift_compares_dtypes_by_string_not_natively(caplog):
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

    # The fix compares dtypes by ``str()`` rather than natively, because a native ``!=`` on some dtype
    # values dispatched into pandas' ``Index.__eq__`` / ``Series.equals`` machinery (~270ms per call). That
    # is a question about which comparison runs, so ask it directly: a tripwire dtype whose ``__eq__`` /
    # ``__ne__`` record being called, but whose ``str()`` matches the real dtype so no drift is reported.
    # A 100ms budget for 100 calls is 1ms each, which the nightly coverage job's line tracing inflates on
    # correct code, and which the slow path could still slip under on a frame with fewer columns.
    comparisons = []

    class _DtypeTripwire:
        """Stands in for a dtype whose native comparison is expensive."""

        def __init__(self, real):
            """Remember what this dtype renders as."""
            self._real = real

        def __eq__(self, other):
            """Record a native comparison."""
            comparisons.append("eq")
            return str(self._real) == str(other)

        def __ne__(self, other):
            """Record a native comparison."""
            comparisons.append("ne")
            return str(self._real) != str(other)

        def __hash__(self):
            """Keep the object usable as a dict value in a set-based caller."""
            return hash(str(self._real))

        def __str__(self):
            """Render exactly as the real dtype, so no drift is reported."""
            return str(self._real)

    tripwired = {col: _DtypeTripwire(dtype) for col, dtype in train_schema.items()}

    with caplog.at_level(logging.WARNING):
        for _ in range(100):
            _warn_on_schema_drift(tripwired, val, "val")

    assert not comparisons, f"the dtype check made {len(comparisons)} native comparisons; it is not comparing via str() and can dispatch into pandas' Index.__eq__ machinery"
    assert not [r for r in caplog.records if "dtype" in r.getMessage()], "the tripwire dtypes were reported as drift, so this test is not exercising the equal-dtype path it means to"


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
