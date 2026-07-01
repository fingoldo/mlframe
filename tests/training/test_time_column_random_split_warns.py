"""Regression: a configured ``time_column`` combined with the default
``cv_strategy="random"`` emits a loud lookahead-leakage WARNING (but does NOT
silently override the split). Guards the time-data random-shuffle foot-gun.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from mlframe.training._preprocessing_configs import TrainingSplitConfig
from mlframe.training.core._phase_helpers_fit_split import _resolve_timeseries_timestamps

_LOGGER = "mlframe.training.core._phase_helpers_fit_split"


def _df():
    return pd.DataFrame({"t": np.arange(10), "x": np.arange(10)})


def test_time_column_with_random_strategy_warns(caplog):
    cfg = TrainingSplitConfig(time_column="t", cv_strategy="random")
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        out = _resolve_timeseries_timestamps(None, cfg, _df())
    # Split is NOT silently overridden: timestamps stay None (-> random split downstream).
    assert out is None
    msgs = " ".join(r.getMessage() for r in caplog.records)
    assert "time_column" in msgs
    assert "lookahead-leakage" in msgs.lower() or "random" in msgs.lower()


def test_no_time_column_does_not_warn(caplog):
    cfg = TrainingSplitConfig(cv_strategy="random")  # no time_column
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        _resolve_timeseries_timestamps(None, cfg, _df())
    assert not [r for r in caplog.records if "foot-gun" in r.getMessage()]


@pytest.mark.parametrize("strategy", ["timeseries", "purged"])
def test_time_column_with_temporal_strategy_does_not_warn(caplog, strategy):
    cfg = TrainingSplitConfig(time_column="t", cv_strategy=strategy)
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        out = _resolve_timeseries_timestamps(None, cfg, _df())
    # Temporal strategy honors the time column (returns the timestamps) and never warns.
    assert out is not None
    np.testing.assert_array_equal(np.asarray(out), np.arange(10))
    assert not [r for r in caplog.records if "foot-gun" in r.getMessage()]
