"""E12: _drop_columns warns when a grouped predict would copy a large pandas
frame under Copy-on-Write OFF (no silent RAM doubling on 100+GB frames)."""

from __future__ import annotations

import logging

import pandas as pd
import pytest

from mlframe.training.composite import CompositeTargetEstimator


def _drop(X, cols):
    """Drop."""
    return CompositeTargetEstimator._drop_columns(X, cols)


def test_noop_when_column_absent_returns_same_object() -> None:
    """Noop when column absent returns same object."""
    X = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    assert _drop(X, ["missing"]) is X


def test_drops_present_column() -> None:
    """Drops present column."""
    X = pd.DataFrame({"a": [1, 2], "grp": ["x", "y"]})
    out = _drop(X, ["grp"])
    assert "grp" not in out.columns and "a" in out.columns


def test_large_frame_cow_off_warns(monkeypatch, caplog) -> None:
    """Large frame cow off warns."""
    X = pd.DataFrame({"a": [1.0, 2.0], "grp": ["x", "y"]})

    class _Big:
        """Groups tests covering big."""
        def sum(self):
            """Sum."""
            return 3 * 1024**3  # 3 GB

    monkeypatch.setattr(type(X), "memory_usage", lambda self, **k: _Big())
    monkeypatch.setattr(pd, "get_option", lambda opt: False if opt == "mode.copy_on_write" else None)
    with caplog.at_level(logging.WARNING):
        _drop(X, ["grp"])
    assert any("copies a" in r.message and "GB pandas" in r.message for r in caplog.records)


def test_large_frame_cow_on_does_not_warn(monkeypatch, caplog) -> None:
    """Large frame cow on does not warn."""
    X = pd.DataFrame({"a": [1.0, 2.0], "grp": ["x", "y"]})

    class _Big:
        """Groups tests covering big."""
        def sum(self):
            """Sum."""
            return 3 * 1024**3

    monkeypatch.setattr(type(X), "memory_usage", lambda self, **k: _Big())
    monkeypatch.setattr(pd, "get_option", lambda opt: True if opt == "mode.copy_on_write" else None)
    with caplog.at_level(logging.WARNING):
        _drop(X, ["grp"])
    assert not any("copies a" in r.message for r in caplog.records)


def test_polars_zero_copy_drop() -> None:
    """Polars zero copy drop."""
    pl = pytest.importorskip("polars")
    X = pl.DataFrame({"a": [1.0, 2.0], "grp": ["x", "y"]})
    out = _drop(X, ["grp"])
    assert "grp" not in out.columns and "a" in out.columns
