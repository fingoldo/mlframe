"""build_dcd_state: a failed DCD init the caller asked for is reported at WARNING whatever the verbosity."""

from __future__ import annotations

import logging

import numpy as np

from mlframe.feature_selection.filters import _dynamic_cluster_discovery as dcd
from mlframe.feature_selection.filters._screen_predictors_gate import build_dcd_state


def test_dcd_init_failure_warns_at_verbose_zero(monkeypatch, caplog):
    """verbose=0 and a raising make_dcd_state: the state is None and a WARNING names the failure."""

    def _boom(**kwargs):
        """Fail like a broken DCD init."""
        raise ValueError("init broke")

    monkeypatch.setattr(dcd, "make_dcd_state", _boom)
    with caplog.at_level(logging.WARNING):
        out = build_dcd_state({"enable": True}, np.zeros((5, 2), dtype=np.int32), [2, 2], ["a", "b"], [1], None, 0)
    assert out is None
    assert any(r.levelno == logging.WARNING and "DCD init failed" in r.getMessage() for r in caplog.records)
