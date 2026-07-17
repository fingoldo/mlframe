"""Module-split sensor for the composite-target estimator package.

The ``CompositeTargetEstimator`` class lives in ``composite/estimator/_estimator.py``
and is re-exported from the ``composite.estimator`` package facade so every existing
caller (``from mlframe.training.composite import CompositeTargetEstimator``) keeps
working unchanged.

The sibling imports the top-level helpers (_y_train_clip_bounds, _extract_base,
_extract_groups, _extract_base_matrix, _is_polars_df) from the facade's partial-module
state -- the facade's bottom re-export triggers the sibling load AFTER those helpers
are bound.
"""

from __future__ import annotations

from pathlib import Path


def test_class_still_importable_from_facade() -> None:
    from mlframe.training.composite.estimator import CompositeTargetEstimator

    assert CompositeTargetEstimator is not None
    from sklearn.base import BaseEstimator, RegressorMixin

    assert issubclass(CompositeTargetEstimator, BaseEstimator)
    assert issubclass(CompositeTargetEstimator, RegressorMixin)


def test_extract_helpers_still_importable() -> None:
    """The top-level helpers stay in the facade; the sibling pulls them
    back via partial-module import."""
    from mlframe.training.composite.estimator import (
        _y_train_clip_bounds,
        _extract_base,
        _extract_groups,
        _extract_base_matrix,
        _is_polars_df,
        predict_quantile_ensemble,
    )

    for fn in (
        _y_train_clip_bounds,
        _extract_base,
        _extract_groups,
        _extract_base_matrix,
        _is_polars_df,
        predict_quantile_ensemble,
    ):
        assert callable(fn), fn


def test_public_path_reexports_estimator() -> None:
    """``mlframe.training.composite`` re-exports the estimator surface."""
    from mlframe.training.composite import CompositeTargetEstimator, predict_quantile_ensemble

    assert CompositeTargetEstimator is not None
    assert callable(predict_quantile_ensemble)


def test_facade_below_1k_line_threshold() -> None:
    root = Path(__file__).resolve().parents[4] / "src" / "mlframe" / "training" / "composite" / "estimator"
    facade = root / "__init__.py"
    n = len(facade.read_text(encoding="utf-8").splitlines())
    assert n < 1000, f"estimator/__init__.py is {n} lines, still over the 1k threshold"


def test_sibling_owns_the_moved_class() -> None:
    """Identity: facade and sibling expose the SAME class object."""
    from mlframe.training.composite import estimator
    from mlframe.training.composite.estimator import _estimator

    assert estimator.CompositeTargetEstimator is _estimator.CompositeTargetEstimator


def test_sibling_helpers_resolve_to_parent_definitions() -> None:
    """The sibling's top-level import of facade helpers must resolve to
    the facade's definitions, not a re-defined local copy."""
    from mlframe.training.composite import estimator
    from mlframe.training.composite.estimator import _estimator

    for name in ("_y_train_clip_bounds", "_extract_base", "_extract_groups", "_extract_base_matrix", "_is_polars_df"):
        assert getattr(_estimator, name) is getattr(estimator, name), name


def test_y_train_clip_bounds_round_trip() -> None:
    """Functional smoke: the helper still works after the split."""
    import numpy as np
    from mlframe.training.composite.estimator import _y_train_clip_bounds

    lo, hi = _y_train_clip_bounds(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert np.isfinite(lo) and np.isfinite(hi)
    assert lo < hi
