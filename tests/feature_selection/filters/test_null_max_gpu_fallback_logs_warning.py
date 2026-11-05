"""When the batched GPU null maximum fails, the slower one-launch-per-permutation path is logged at WARNING.

It used to log at DEBUG, so a run that silently lost its batched path looked the same as one that never had it.
"""

from __future__ import annotations

import logging

import numpy as np

from mlframe.feature_selection.filters import _binned_numeric_agg_resident as resident
from mlframe.feature_selection.filters import _hermite_fe_mi


def test_failed_batched_binagg_null_logs_the_slow_path_at_warning(monkeypatch, caplog):
    """A failing batched null returns None, so the caller loops, and says so at WARNING level."""

    def _boom(*args, **kwargs):
        """Stand in for a device fault while binning the candidates."""
        raise RuntimeError("simulated device fault")

    monkeypatch.setattr(_hermite_fe_mi, "_resident_mi_candidates", _boom)
    rng = np.random.default_rng(0)
    with caplog.at_level(logging.WARNING, logger=resident.logger.name):
        out = resident._batched_null_max(None, None, None, np.arange(20), 0, 2, 4, rng)
    assert out is None
    assert any(r.levelno == logging.WARNING and "one launch per permutation" in r.getMessage() for r in caplog.records)
