"""Calibration entry points refuse parameters that make the selection meaningless or write out of bounds (tracker 2026-11-02, VAL6-1, VAL6-2, VAL6-3)."""

import numpy as np
import pytest

from mlframe.calibration.policy import _ece_score, pick_best_calibrator


def _oof(n=300, seed=0):
    """Binary OOF probabilities and labels with some signal."""
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.05, 0.95, n)
    y = (rng.uniform(size=n) < p).astype(int)
    return p, y


def test_ece_with_zero_bins_raises_instead_of_writing_out_of_bounds():
    """n_bins=0 gave an empty accumulator indexed at -1 inside the njit kernel; it now raises first."""
    p, y = _oof()
    with pytest.raises(ValueError, match="n_bins"):
        _ece_score(y, p, 0)
    assert 0.0 <= _ece_score(y, p, 10) <= 1.0


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"alpha": 0.0}, "alpha"),
        ({"alpha": 1.5}, "alpha"),
        ({"alpha": float("nan")}, "alpha"),
        ({"n_bins": 0}, "n_bins"),
        ({"n_bootstrap": 0}, "n_bootstrap"),
        ({"inner_cv_splits": 1}, "inner_cv_splits"),
    ],
)
def test_pick_best_calibrator_refuses_degenerate_parameters(kwargs, match):
    """Each degenerate value raises a ValueError that names the parameter, before any fitting."""
    p, y = _oof()
    with pytest.raises(ValueError, match=match):
        pick_best_calibrator(None, None, p, y, **kwargs)
