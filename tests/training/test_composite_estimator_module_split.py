"""Wave 102 (2026-05-21): split training/composite_estimator.py
(1285 lines) into composite_estimator.py (now 349 lines) + new
_composite_target_estimator.py (989 lines).

Moved to the sibling file: the ``CompositeTargetEstimator`` class
(~945 lines). The class is the main composite-target estimator type;
re-exported from ``composite_estimator`` so every existing caller
(``from mlframe.training.composite_estimator import CompositeTargetEstimator``)
keeps working unchanged.

The sibling imports the top-level helpers (_y_train_clip_bounds,
_extract_base, _extract_groups, _extract_base_matrix, _is_polars_df)
from the parent's partial-module state -- the parent's bottom
re-export triggers the sibling load AFTER those helpers are bound.
"""
from __future__ import annotations

from pathlib import Path


def test_class_still_importable_from_facade() -> None:
    from mlframe.training.composite_estimator import CompositeTargetEstimator
    assert CompositeTargetEstimator is not None
    # Verify a few public class attributes survived the split.
    from sklearn.base import BaseEstimator, RegressorMixin
    assert issubclass(CompositeTargetEstimator, BaseEstimator)
    assert issubclass(CompositeTargetEstimator, RegressorMixin)


def test_extract_helpers_still_importable() -> None:
    """The top-level helpers stay in the parent module; the sibling
    pulls them back via partial-module import."""
    from mlframe.training.composite_estimator import (
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


def test_facade_below_1k_line_threshold() -> None:
    root = Path(__file__).resolve().parent.parent.parent / "src" / "mlframe" / "training"
    facade = root / "composite_estimator.py"
    n = len(facade.read_text(encoding="utf-8").splitlines())
    assert n < 1000, f"composite_estimator.py is {n} lines, still over the 1k threshold"


def test_sibling_owns_the_moved_class() -> None:
    """Identity: facade and sibling expose the SAME class object."""
    from mlframe.training import composite_estimator, _composite_target_estimator
    assert composite_estimator.CompositeTargetEstimator is _composite_target_estimator.CompositeTargetEstimator


def test_sibling_helpers_resolve_to_parent_definitions() -> None:
    """The sibling's top-level import of parent helpers must resolve to
    the parent's definitions, not a re-defined local copy."""
    from mlframe.training import composite_estimator, _composite_target_estimator
    for name in ("_y_train_clip_bounds", "_extract_base", "_extract_groups", "_extract_base_matrix", "_is_polars_df"):
        assert getattr(_composite_target_estimator, name) is getattr(composite_estimator, name), name


def test_y_train_clip_bounds_round_trip() -> None:
    """Functional smoke: the helper still works after the split."""
    import numpy as np
    from mlframe.training.composite_estimator import _y_train_clip_bounds

    lo, hi = _y_train_clip_bounds(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    # The bounds are derived percentiles; just verify they are finite + lo < hi.
    assert np.isfinite(lo) and np.isfinite(hi)
    assert lo < hi
