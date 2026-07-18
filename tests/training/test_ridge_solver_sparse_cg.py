"""Regression test for the 2026-05-27 Ridge slow-solver bug.

Bug: prod Ridge fit took 44.5 min on 4M x 323 (TVT_regression.log
22:34->23:19) because sklearn's ``solver='auto'`` picked the SVD path
(O(min(n,p)^2 * max(n,p)) + an n x min(n,p) U matrix that OOMs at
n=4M). The same data fits in 1.13s under ``solver='sparse_cg'``.

Fix: pin ``solver='sparse_cg'`` in the linear / ridge model builders.

Sensor: both builders must request the explicit fast solver. A future
refactor that drops the kwarg regresses the user back to the 44-min
auto-SVD path.
"""

from __future__ import annotations


def test_linear_regressor_uses_sparse_cg_solver() -> None:
    """The plain linear regressor pins solver='sparse_cg' to avoid the 44+ minute SVD-auto path on 4M x 323 prod data."""
    from mlframe.training.models import _build_linear_regressor
    from mlframe.training._model_configs import LinearModelConfig

    cfg = LinearModelConfig(random_state=0)
    model = _build_linear_regressor(cfg)
    assert model.get_params()["solver"] == "sparse_cg", (
        f"linear_regressor must pin solver='sparse_cg' to avoid the 44+ min SVD-auto path on 4M x 323 prod data; got {model.get_params()['solver']!r}"
    )


def test_ridge_regressor_uses_sparse_cg_solver() -> None:
    """The Ridge regressor likewise pins solver='sparse_cg', matching the plain linear regressor's fix."""
    from mlframe.training.models import _build_ridge_regressor
    from mlframe.training._model_configs import LinearModelConfig

    cfg = LinearModelConfig(random_state=0)
    model = _build_ridge_regressor(cfg)
    assert model.get_params()["solver"] == "sparse_cg"
