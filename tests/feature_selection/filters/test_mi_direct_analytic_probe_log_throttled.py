"""The analytic-null availability probe sits on the per-candidate path, so its failure log is throttled rather than emitted once per call."""

from __future__ import annotations

import logging

import numpy as np

from mlframe.feature_selection.filters import _analytic_mi_null
from mlframe.feature_selection.filters._permutation_mi_direct_paths import _mi_direct_analytic_null
from mlframe.utils.log_throttle import reset_throttle_counts


def test_probe_failure_is_throttled(monkeypatch, caplog):
    """Twenty calls with a raising probe produce a handful of warnings, not twenty."""

    def boom():
        """Simulated probe fault."""
        raise RuntimeError("probe broke")

    monkeypatch.setattr(_analytic_mi_null, "analytic_null_enabled", boom)
    reset_throttle_counts()
    data = np.zeros((10, 2), dtype=np.int32)
    nb = np.array([2, 2])
    with caplog.at_level(logging.WARNING):
        for _ in range(20):
            res, _cy, _fy = _mi_direct_analytic_null(data, (0,), (1,), nb, np.int64, None, None, False)
            assert res is None
    warned = [r for r in caplog.records if "analytic-null availability" in r.getMessage()]
    assert len(warned) == 5, f"the throttle admits its default max_count of 5 warnings after a reset, got {len(warned)}"
