"""The device-born binned-aggregate gate says so when it gives up and hands the work to the host gate."""

from __future__ import annotations

import logging
import sys
import types

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters import _binned_numeric_agg_resident as R


def test_gate_failure_falls_back_to_host_with_a_warning(monkeypatch, caplog):
    """A device error inside the scorer returns None (host gate) and logs a WARNING, not a debug line."""
    monkeypatch.setitem(sys.modules, "cupy", types.ModuleType("cupy"))

    def boom(*a, **k):
        """Scorer stand-in that fails like a driver fault."""
        raise RuntimeError("device fault")

    monkeypatch.setattr(R, "_binagg_device_scores", boom)
    feat = pd.DataFrame({"n1": np.arange(10.0)})
    raw = pd.DataFrame({"g": np.arange(10.0), "a": np.arange(10.0)})
    recipes = {"n1": {"group_col": "g", "agg_col": "a", "stat": "mean", "edges": [1.0, 2.0], "global": 0.0}}
    with caplog.at_level(logging.WARNING, logger=R.logger.name):
        out = R.local_mi_gate_binagg_resident(feat, np.arange(10) % 2, raw, recipes, floor=0.0)
    assert out is None
    assert any("device fault" in r.getMessage() and r.levelno >= logging.WARNING for r in caplog.records)
