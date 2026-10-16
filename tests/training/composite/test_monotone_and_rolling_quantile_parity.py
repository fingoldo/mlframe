"""Scalar references _monotone_fraction and _rolling_quantile_radius against their live vectorised/incremental forms."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.training.composite.conformal_online import _rolling_quantile_radius, _sorted_quantile_radius
from mlframe.training.composite.discovery._structural_hints import _MONOTONE_FRACTION, _monotone_fraction, structural_affinity_scores


def test_vectorised_monotone_score_equals_scalar_reference():
    """For time-kind columns the score IS the scalar monotone fraction; non-monotone columns are never tagged time."""
    rng = np.random.default_rng(3)
    n = 400
    t = np.arange(n, dtype=np.float64)
    steps = np.repeat(np.arange(n // 4, dtype=np.float64), 4)  # step-constant timestamps, ties excluded
    near = t.copy()
    near[::50] -= 3.0  # a few backward steps keep it just above the 0.98 gate
    noise = rng.normal(size=n)
    x = np.column_stack([t, steps, near, noise])
    names = ["t", "steps", "near", "noise"]
    y = rng.normal(size=n)
    scores, kinds = structural_affinity_scores(x, y, names)
    for j, name in enumerate(names):
        frac = _monotone_fraction(x[:, j])
        if frac >= _MONOTONE_FRACTION:
            assert kinds.get(name) == "time"
            assert scores[j] == pytest.approx(frac, abs=1e-15)
        else:
            assert kinds.get(name) != "time"
    assert kinds.get("t") == "time" and "noise" not in kinds


@pytest.mark.parametrize("alpha", [-0.1, 0.0, 0.05, 0.1, 0.5, 0.9, 1.0, 1.5])
@pytest.mark.parametrize("m", [0, 1, 5, 40])
def test_rolling_radius_equals_sorted_buffer_radius(m, alpha):
    """The scan-and-sort reference gives the same radius as the live sorted-buffer form."""
    rng = np.random.default_rng(m + 11)
    res = rng.normal(size=m)
    r_sorted = sorted(np.abs(res).tolist())
    assert _rolling_quantile_radius(res, alpha) == _sorted_quantile_radius(r_sorted, len(r_sorted), alpha)
