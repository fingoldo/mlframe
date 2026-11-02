"""A drift refit keeps fit()'s rule that a T with MAD 0 gets no T-clip, instead of clamping to the observed range."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from mlframe.training.composite.estimator._update import _apply_drift_refit


class _Buffer:
    """Ring-buffer stand-in exposing the contiguous read the refit uses."""

    def __init__(self, values):
        self._values = np.asarray(values, dtype=float)

    def contiguous(self):
        """Return the stored values."""
        return self._values


def test_zero_mad_residual_leaves_the_t_clip_open():
    """A degenerate T (MAD 0) must reopen the clip, as fit() leaves it, not keep the stale bounds or clamp to the observed range."""
    rng = np.random.default_rng(0)
    base = rng.normal(size=200) + 5.0
    y = base.copy()  # y == base exactly: the refit's T is identically zero, MAD 0
    est = SimpleNamespace(
        _buffer_y_=_Buffer(y),
        _buffer_base_=_Buffer(base),
        transform_name="linear_residual",
        fitted_params_={"alpha": 0.0, "beta": 1.0, "t_clip_low": -1.0, "t_clip_high": 1.0},
    )
    _apply_drift_refit(est, {"change_point": -1}, 0.0, 1.0)
    assert est.fitted_params_["t_clip_low"] == float("-inf")
    assert est.fitted_params_["t_clip_high"] == float("inf")
