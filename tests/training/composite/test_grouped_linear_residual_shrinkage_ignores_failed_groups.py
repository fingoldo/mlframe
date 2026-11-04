"""The James-Stein noise estimate of the grouped linear residual covers the same groups as the shrinkage."""

from __future__ import annotations

import numpy as np

from mlframe.training.composite.transforms import get_transform
from mlframe.training.composite.transforms._call_gateway import call_transform


def _data(with_failed_group: bool):
    """Seven groups with their own slopes plus, optionally, a 60-row group (varying base) far from every fit."""
    rng = np.random.default_rng(0)
    ys, bs, gs = [], [], []
    for g in range(7):
        base = rng.normal(0, 1, 120)
        ys.append((1.0 + 0.15 * g) * base + rng.normal(0, 0.3, 120))
        bs.append(base)
        gs.append(np.full(120, g))
    if with_failed_group:
        ys.append(np.full(60, 1000.0))
        bs.append(np.linspace(0.0, 1.0, 60))
        gs.append(np.full(60, 99))
    return np.concatenate(ys), np.concatenate(bs), np.concatenate(gs)


def test_noise_estimate_covers_only_the_groups_that_were_fit(monkeypatch):
    """The shrinkage's residual sum and row count exclude a group whose OLS failed: before, its ~6e7 squared residual and its rows went in as if it were a fit."""
    from mlframe.training.composite.transforms import linear as linear_mod

    seen = {}
    real = linear_mod._js_shrinkage_for_groups

    def _spy(alphas, sizes, base_vars, alpha_global, total_resid_sq, total_n):
        """Record the noise statistics, then defer to the real shrinkage."""
        seen["total_resid_sq"], seen["total_n"] = total_resid_sq, total_n
        return real(alphas, sizes, base_vars, alpha_global, total_resid_sq, total_n)

    monkeypatch.setattr(linear_mod, "_js_shrinkage_for_groups", _spy)
    real_fit = linear_mod._linear_residual_fit

    def _fail_for_small_group(y, base, *args, **kwargs):
        """Make the per-group OLS raise for the 60-row group, the defensive path that marks a group as not fit."""
        if len(y) == 60:
            raise np.linalg.LinAlgError("forced failure")
        return real_fit(y, base, *args, **kwargs)

    monkeypatch.setattr(linear_mod, "_linear_residual_fit", _fail_for_small_group)
    y, b, g = _data(True)
    call_transform(get_transform("linear_residual_grouped"), "fit", y, b, groups=g)
    assert seen["total_n"] == 7 * 120, "the failed group's rows were counted"
    assert seen["total_resid_sq"] < 1e4, "the failed group's squared residual (~6e7) entered the noise estimate"
