"""Regression: linear_stack must use RidgeCV (alpha chosen by CV) by default,
not a hard-coded Ridge(alpha=1.0).

Pre-fix: ``ridge_alpha`` defaulted to 1.0 and Ridge was instantiated directly,
so the alpha was always 1.0 regardless of data characteristics.

Post-fix: ``ridge_alpha=None`` triggers RidgeCV over a small grid; the
chosen alpha is stored in ``notes["ridge_alpha"]`` and matches one of the
grid points. Explicit ``ridge_alpha=<float>`` still works for callers
that have tuned alpha externally.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from mlframe.training.composite_ensemble import CompositeCrossTargetEnsemble


def _build(seed: int, noise: float, alpha_grid: tuple[float, ...]):
    rng = np.random.default_rng(seed)
    n = 800
    y_train = rng.normal(loc=10.0, scale=2.0, size=n)
    p1 = y_train + 1.0 + rng.normal(scale=noise, size=n)
    p2 = y_train - 0.5 + rng.normal(scale=noise, size=n)
    p3 = y_train + 2.0 + rng.normal(scale=noise, size=n)
    preds = np.column_stack([p1, p2, p3])
    models = [MagicMock(name=f"c{i}") for i in range(3)]
    names = [f"c{i}" for i in range(3)]
    ens = CompositeCrossTargetEnsemble.from_linear_stack(
        component_models=models,
        component_names=names,
        component_predictions=preds,
        y_train=y_train,
        ridge_alpha_grid=alpha_grid,
    )
    return ens


def test_ridge_alpha_chosen_via_cv_not_always_one() -> None:
    """The chosen alpha must depend on the noise level: low noise -> small alpha,
    high noise -> large alpha (more shrinkage). Demonstrates CV is actually
    running rather than always returning 1.0.
    """
    grid = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
    chosen = []
    for seed, noise in [(0, 0.01), (0, 5.0), (1, 0.01), (1, 5.0)]:
        ens = _build(seed=seed, noise=noise, alpha_grid=grid)
        a = float(ens.notes["ridge_alpha"])
        assert a in grid, f"alpha {a} not in grid {grid}"
        assert ens.notes["ridge_alpha_was_cv_selected"] is True
        chosen.append(a)
    # At least two distinct alphas must appear across noise levels --
    # confirms CV actually selects, not just hard-coded 1.0.
    assert len(set(chosen)) >= 2, (
        f"CV must pick different alphas for different noise; saw {chosen}"
    )


def test_explicit_ridge_alpha_bypasses_cv() -> None:
    """When caller passes ``ridge_alpha=<float>``, the value is honoured and
    no CV runs (notes flag remains False)."""
    rng = np.random.default_rng(7)
    n = 300
    y = rng.normal(size=n)
    preds = np.column_stack([y + rng.normal(scale=0.5, size=n) for _ in range(3)])
    models = [MagicMock() for _ in range(3)]
    names = [f"c{i}" for i in range(3)]
    ens = CompositeCrossTargetEnsemble.from_linear_stack(
        component_models=models, component_names=names,
        component_predictions=preds, y_train=y, ridge_alpha=2.5,
    )
    assert ens.notes["ridge_alpha"] == 2.5
    assert ens.notes["ridge_alpha_was_cv_selected"] is False
    assert ens._linear_stack_ridge_alpha == 2.5
