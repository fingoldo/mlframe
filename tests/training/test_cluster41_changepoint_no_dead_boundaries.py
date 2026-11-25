"""Cluster 41 regression: the PELT change-point helper drops its unused boundary loop (audit 2026-11-20, finding 41-F29).

The ``boundaries`` list was built and never read; the function returns the inner change-point pairs. The source no longer builds it, and the
returned pairs are unchanged for a simple breakpoint list.
"""

import inspect

import numpy as np

from mlframe.training.targets import _target_temporal_changepoint as cp


def test_dead_boundaries_block_removed():
    """The never-read ``boundaries`` accumulator is gone from the PELT helper (41-F29)."""
    body = inspect.getsource(cp.find_change_points_pelt)
    assert "boundaries: list[int] = []" not in body
    assert "boundaries.append" not in body


def test_step_signal_returns_paired_change_points():
    """Control: a clear level shift is found and returned as duplicated pairs (41-F29)."""
    rng = np.random.default_rng(0)
    rates = np.concatenate([rng.normal(0.0, 0.1, 60), rng.normal(5.0, 0.1, 60)])
    out = cp.find_change_points_pelt(rates, penalty=10.0)
    assert out, "a step of 5 sigma must produce a change point"
    assert len(out) % 2 == 0
    assert out[0::2] == out[1::2]
