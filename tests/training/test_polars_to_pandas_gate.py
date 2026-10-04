"""Gated polars -> pandas bridge: identical result to ``to_pandas`` and a WARN once the frame passes the byte threshold."""
from __future__ import annotations

import logging

import pandas as pd
import pytest

pl = pytest.importorskip("polars")

from mlframe.training import _polars_to_pandas_gate as gate


def _frame():
    """Small mixed-dtype polars frame."""
    return pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": [1, 2, 3], "c": ["x", "y", "z"]})


def test_gated_conversion_equals_plain_to_pandas():
    """Same values and dtypes as the default conversion."""
    X = _frame()
    pd.testing.assert_frame_equal(gate.polars_to_pandas_gated(X, "t_equal"), X.to_pandas())


def test_warns_only_above_threshold(monkeypatch, caplog):
    """No WARN below the threshold; a WARN naming the site once the frame reaches it."""
    X = _frame()
    with caplog.at_level(logging.WARNING, logger=gate.logger.name):
        gate.polars_to_pandas_gated(X, "t_small")
    assert not caplog.records
    monkeypatch.setattr(gate, "PANDAS_BRIDGE_WARN_BYTES", 1)
    with caplog.at_level(logging.WARNING, logger=gate.logger.name):
        gate.polars_to_pandas_gated(X, "t_big")
    assert any("t_big" in r.getMessage() for r in caplog.records)


def test_ranker_fs_and_neural_prep_route_through_the_gate(monkeypatch, caplog):
    """The ranker FS and neural feature-prep conversions warn for an over-threshold frame."""
    from mlframe.training.neural.feature_prep import _as_pandas
    from mlframe.training.ranking._ranker_fs import _to_pandas_features

    monkeypatch.setattr(gate, "PANDAS_BRIDGE_WARN_BYTES", 1)
    with caplog.at_level(logging.WARNING, logger=gate.logger.name):
        assert isinstance(_to_pandas_features(_frame()), pd.DataFrame)
        assert isinstance(_as_pandas(_frame()), pd.DataFrame)
    msgs = " ".join(r.getMessage() for r in caplog.records)
    assert "ranker_fs" in msgs and "neural_feature_prep" in msgs
