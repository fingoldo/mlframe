"""Large-offset regressions for the centred closed-form fits: composite OLS and the Platt Newton solver."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.training.composite import cache as cache_mod
from mlframe.training.composite.ensemble._calibration import _platt_mle_1d
from mlframe.training.composite.transforms.linear import _linear_residual_fit_batched, _linear_residual_fit_closed

_OFFSET = 1.7e9


def _offset_problem(seed: int = 0, n: int = 4000) -> tuple[np.ndarray, np.ndarray, float, float]:
    """An epoch-timestamp style base (large offset, unit spread) with a known slope and intercept."""
    rng = np.random.default_rng(seed)
    x = _OFFSET + rng.normal(size=n)
    alpha, beta = 2.5, -3.0
    y = alpha * (x - _OFFSET) + beta + rng.normal(scale=0.05, size=n)
    return x, y, alpha, beta


def test_closed_ols_recovers_slope_on_epoch_scale_base():
    """The closed-form OLS keeps the slope when the base sits 1.7e9 away from zero."""
    x, y, alpha, _ = _offset_problem()
    a, _b = _linear_residual_fit_closed(x, y)
    assert abs(a - alpha) < 0.01


def test_batched_ols_matches_closed_on_epoch_scale_base():
    """The batched solver agrees with the scalar closed form fold by fold on a large-offset base."""
    folds = [_offset_problem(seed=s, n=1500)[:2] for s in range(4)]
    alphas, betas = _linear_residual_fit_batched([f[0] for f in folds], [f[1] for f in folds])
    for i, (x, y) in enumerate(folds):
        a, b = _linear_residual_fit_closed(x, y)
        assert abs(alphas[i] - a) < 1e-9
        assert abs(betas[i] - b) < 1e-6


def test_platt_fit_is_translation_equivariant_in_the_logit():
    """Shifting the logit by a large constant moves B by -A*shift and leaves A unchanged."""
    rng = np.random.default_rng(3)
    z = rng.normal(size=3000)
    t01 = (rng.random(3000) < 1.0 / (1.0 + np.exp(-(1.4 * z - 0.3)))).astype(np.float64)
    a0, b0 = _platt_mle_1d(z, t01, None)
    shift = 40.0
    a1, b1 = _platt_mle_1d(z + shift, t01, None)
    assert abs(a1 - a0) < 1e-4
    assert abs((b1 + a1 * shift) - b0) < 1e-3


def test_unfingerprintable_frames_of_different_structure_get_different_keys(monkeypatch):
    """When the head/tail fingerprint fails, frames of different shape or columns must not share one cache key."""

    def _boom(*_args, **_kwargs):
        """Stand-in for a fingerprint routine that fails on an exotic frame type."""
        raise RuntimeError("fingerprint unavailable")

    monkeypatch.setattr(cache_mod._canonical, "row_order_fingerprint", _boom)
    a = pd.DataFrame({"x": [1, 2, 3]})
    b = pd.DataFrame({"y": [1, 2, 3]})
    c = pd.DataFrame({"x": [1, 2, 3, 4]})
    keys = {cache_mod._row_order_fingerprint(f) for f in (a, b, c)}
    assert len(keys) == 3
    assert all(k.startswith("unfingerprintable:") for k in keys)
    assert cache_mod._row_order_fingerprint(a) == cache_mod._row_order_fingerprint(pd.DataFrame({"x": [9, 9, 9]}))
