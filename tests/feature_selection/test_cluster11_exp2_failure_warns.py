"""Cluster 11 regression: a failing operator in the exp2 study is reported at WARNING, not DEBUG (audit 2026-11-20, finding 11-F5).

The study drops a failing operator and carries on; the drop is now visible at the default log level, once per operator.
"""

import logging

import numpy as np

from mlframe.feature_selection._benchmarks.fe_operator_factory.stat_study import exp2_general
from mlframe.utils.log_throttle import reset_throttle_counts


def _boom(u, v):
    """An operator that always fails, as a malformed one would."""
    raise ValueError("operator boom")


def test_failing_operator_is_logged_at_warning_once_and_dropped(monkeypatch, caplog):
    """A failing binary operator is dropped from the candidates and logged once at WARNING with its name and cause (later repeats are throttled) (11-F5)."""
    reset_throttle_counts()
    monkeypatch.setattr(exp2_general, "BI", {"boom_op_11f5": _boom})
    rng = np.random.default_rng(0)
    x, z = rng.random(200) + 0.5, rng.random(200) + 0.5
    with caplog.at_level(logging.WARNING, logger=exp2_general.logger.name):
        out = exp2_general.preset(x, z, np.arange(200))
    records = [r for r in caplog.records if "failed on the study inputs" in r.getMessage()]
    assert out == {}
    assert len(records) == 1
    assert records[0].levelno == logging.WARNING
    assert "operator boom" in records[0].getMessage()
