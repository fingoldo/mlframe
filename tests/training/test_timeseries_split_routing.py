"""Unit tests for E2 main-split time-series routing (`_resolve_timeseries_timestamps`).

The helper turns a declared time-series cv_strategy + time_column into the ``timestamps`` array that
make_train_test_split uses for a chronological forward-walk; it is inert for the default random strategy.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from mlframe.training.core._phase_helpers_fit_split import _resolve_timeseries_timestamps


def _sc(**kw):
    base = dict(cv_strategy="random", time_column=None)
    base.update(kw)
    return SimpleNamespace(**base)


def test_random_strategy_is_inert():
    df = pd.DataFrame({"ts": np.arange(10), "x": np.arange(10)})
    assert _resolve_timeseries_timestamps(None, _sc(), df) is None


def test_timeseries_strategy_reads_time_column():
    df = pd.DataFrame({"ts": np.arange(10, 0, -1), "x": np.arange(10)})
    out = _resolve_timeseries_timestamps(None, _sc(cv_strategy="timeseries", time_column="ts"), df)
    assert out is not None
    np.testing.assert_array_equal(out, np.arange(10, 0, -1))


def test_purged_strategy_also_routes():
    df = pd.DataFrame({"ts": np.arange(5), "x": np.arange(5)})
    out = _resolve_timeseries_timestamps(None, _sc(cv_strategy="purged", time_column="ts"), df)
    np.testing.assert_array_equal(out, np.arange(5))


def test_existing_timestamps_preserved():
    df = pd.DataFrame({"ts": np.arange(5)})
    ext = np.array([9, 8, 7, 6, 5])
    out = _resolve_timeseries_timestamps(ext, _sc(cv_strategy="timeseries", time_column="ts"), df)
    np.testing.assert_array_equal(out, ext)  # upstream timestamps win, not overwritten


def test_missing_column_falls_back_to_none():
    df = pd.DataFrame({"x": np.arange(5)})
    out = _resolve_timeseries_timestamps(None, _sc(cv_strategy="timeseries", time_column="nope"), df)
    assert out is None  # unreadable column -> graceful fallback to random split


def test_polars_frame_supported():
    pl = __import__("polars")
    df = pl.DataFrame({"ts": np.arange(6), "x": np.arange(6)})
    out = _resolve_timeseries_timestamps(None, _sc(cv_strategy="timeseries", time_column="ts"), df)
    np.testing.assert_array_equal(np.asarray(out), np.arange(6))
