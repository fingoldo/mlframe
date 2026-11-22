"""A recipe that fails to replay on the host must hand off to the per-job path with a WARNING, not a debug line."""

import logging

import numpy as np

from mlframe.feature_selection.filters import _fe_stability_vote as fsv
from mlframe.feature_selection.filters import engineered_recipes


def test_host_replay_failure_warns_and_returns_none(monkeypatch, caplog):
    """A raising recipe replay returns None (per-job path takes over) and logs at WARNING so the slower route is visible."""

    def _boom(*_a, **_k):
        """Stand in for a recipe replay that fails on this fold."""
        raise ValueError("synthetic replay fault")

    monkeypatch.setattr(engineered_recipes, "apply_recipe", _boom)
    with caplog.at_level(logging.WARNING, logger=fsv.logger.name):
        out = fsv._host_replay(object(), np.zeros((5, 2)), 5)
    assert out is None
    assert any(r.levelno == logging.WARNING and "per-job path" in r.getMessage() for r in caplog.records)
