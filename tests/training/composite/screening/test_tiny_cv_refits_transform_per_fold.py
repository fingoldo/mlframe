"""The tiny screening CV must refit the transform on each fold's train rows.

``_tiny_cv_rmse_y_scale`` is handed the params fit once on every screening row, so those params have already seen the
rows each fold holds out. A flexible residual absorbs part of the held-out noise into its own coefficients, and the CV
score it reports is optimistic -- which is the score the rerank order, the raw-baseline gate, the per-bin gate and the
WAIC band all read. ``refit_transform_on_fold`` existed and was tested but had no production caller.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from mlframe.training.composite.discovery._screening_tiny_perbin import _tiny_cv_rmse_y_scale
from mlframe.training.composite.transforms import Transform

_POLY_DEGREE = 13


def _poly_fit(y: np.ndarray, base: np.ndarray) -> dict[str, Any]:
    """Least-squares polynomial of y on base, flexible enough to chase noise on a small sample."""
    y = np.asarray(y, dtype=np.float64)
    base = np.asarray(base, dtype=np.float64)
    deg = min(_POLY_DEGREE, max(1, y.shape[0] - 1))
    return {"coef": np.polyfit(base, y, deg).tolist()}


def _poly_forward(y: np.ndarray, base: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    """T = y - poly(base)."""
    return np.asarray(y, dtype=np.float64) - np.polyval(np.asarray(params["coef"], dtype=np.float64), np.asarray(base, dtype=np.float64))


def _poly_inverse(t_hat: np.ndarray, base: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    """y = T + poly(base)."""
    return np.asarray(t_hat, dtype=np.float64) + np.polyval(np.asarray(params["coef"], dtype=np.float64), np.asarray(base, dtype=np.float64))


def _poly_domain(y: np.ndarray | None, base: np.ndarray) -> np.ndarray:
    """Finite rows only."""
    ok = np.isfinite(np.asarray(base, dtype=np.float64))
    return ok if y is None else ok & np.isfinite(np.asarray(y, dtype=np.float64))


def _poly_transform(fit=_poly_fit) -> Transform:
    """A deliberately over-flexible residual: the regime where global params flatter the held-out folds.

    ``Transform`` is frozen, so a variant with a different ``fit`` (one that returns the global params, or one that
    always fails) is built here rather than patched onto an instance.
    """
    return Transform(
        name="poly_residual_testonly",
        forward=_poly_forward,
        inverse=_poly_inverse,
        fit=fit,
        domain_check=_poly_domain,
        description="test-only flexible polynomial residual",
    )


def _data(n: int = 240, seed: int = 7):
    """A gentle cubic mean plus unit noise, so the residual a flexible fit absorbs is pure noise."""
    rng = np.random.default_rng(seed)
    base = np.sort(rng.uniform(-2.5, 2.5, size=n))
    y = 0.4 * base**3 - 1.1 * base + rng.normal(0.0, 1.0, size=n)
    x = np.column_stack([base, rng.normal(size=n)])
    return y.astype(np.float64), base.astype(np.float64), x.astype(np.float64)


def _cv(y, base, x, transform, params) -> float:
    """Run the screening CV under test with a fixed, deterministic tiny model."""
    return float(
        _tiny_cv_rmse_y_scale(
            y, base, transform, params, x,
            family="ridge", n_estimators=10, num_leaves=3, learning_rate=0.1,
            cv_folds=4, random_state=0, deterministic=True,
        )
    )


def test_screening_cv_does_not_score_folds_with_params_that_saw_them():
    """The reported CV RMSE must be the per-fold-refit number, not the flattering global-params one.

    Both runs are handed the same global params; the honest path refits them inside each fold. If the CV still scored
    with the global fit, the two numbers would be identical.
    """
    y, base, x = _data()
    transform = _poly_transform()
    global_params = _poly_fit(y, base)
    got = _cv(y, base, x, transform, global_params)

    # What the old path reported: a "refit" that hands back the global fit, i.e. scoring folds with params that saw them.
    leaky = _poly_transform(fit=lambda _y, _b, **_kw: {"coef": list(global_params["coef"])})
    leaky_score = _cv(y, base, x, leaky, global_params)

    assert np.isfinite(got) and np.isfinite(leaky_score)
    assert got > leaky_score, (
        f"the honest per-fold score must be worse than the one measured with params that saw the held-out rows: "
        f"per-fold={got:.4f}, global={leaky_score:.4f}"
    )


def test_a_degenerate_fold_keeps_the_global_params_instead_of_dropping_the_spec():
    """A transform whose per-fold fit always fails must still score: the helper returns None and the fold falls back."""
    y, base, x = _data()
    params = _poly_fit(y, base)

    def _always_fails(_y, _b, **_kw):
        raise RuntimeError("degenerate fold")

    score = _cv(y, base, x, _poly_transform(fit=_always_fails), params)
    assert np.isfinite(score), "a spec that scored before must not drop out when its per-fold refit fails"
    fixed = _poly_transform(fit=lambda _y, _b, **_kw: {"coef": list(params["coef"])})
    assert score == pytest.approx(_cv(y, base, x, fixed, params)), "the fallback must reproduce the global-params score exactly"
