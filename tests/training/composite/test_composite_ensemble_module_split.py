"""Wave 101 (2026-05-21): split training/composite_ensemble.py
(1292 lines) into composite_ensemble.py (now 588 lines) + new
_composite_cross_target_ensemble.py (744 lines).

Moved to the sibling file: the ``CompositeCrossTargetEnsemble`` class
(~710 lines). The class is the central composite-target ensemble type,
re-exported from ``composite_ensemble`` so every existing caller
(``from mlframe.training.composite.ensemble import CompositeCrossTargetEnsemble``)
keeps working unchanged.
"""
from __future__ import annotations

from pathlib import Path


def test_class_still_importable_from_facade() -> None:
    from mlframe.training.composite.ensemble import CompositeCrossTargetEnsemble
    assert CompositeCrossTargetEnsemble is not None
    # Classmethods on the public API.
    assert hasattr(CompositeCrossTargetEnsemble, "from_uniform_weights")
    assert hasattr(CompositeCrossTargetEnsemble, "from_linear_stack")


def test_other_composite_ensemble_symbols_still_importable() -> None:
    from mlframe.training.composite.ensemble import (
        derive_seeds,
        detect_gpu_in_use,
        env_signature,
        compute_oof_holdout_predictions,
    )
    for fn in (
        derive_seeds,
        detect_gpu_in_use,
        env_signature,
        compute_oof_holdout_predictions,
    ):
        assert callable(fn), fn


def test_facade_below_1k_line_threshold() -> None:
    root = Path(__file__).resolve().parents[3] / "src" / "mlframe" / "training" / "composite" / "ensemble"
    facade = root / "__init__.py"
    n = len(facade.read_text(encoding="utf-8").splitlines())
    assert n < 1000, f"ensemble/__init__.py is {n} lines, still over the 1k threshold"


def test_sibling_owns_the_moved_class() -> None:
    """Identity: facade and sibling expose the SAME class object."""
    from mlframe.training.composite import ensemble as composite_ensemble
    from mlframe.training.composite.ensemble import _cross_target as _composite_cross_target_ensemble
    assert composite_ensemble.CompositeCrossTargetEnsemble is _composite_cross_target_ensemble.CompositeCrossTargetEnsemble


def test_from_uniform_weights_round_trip() -> None:
    """Functional smoke: the moved class's from_uniform_weights factory
    still constructs a valid instance after the split."""
    import numpy as np
    from mlframe.training.composite.ensemble import CompositeCrossTargetEnsemble

    class _ConstModel:
        """Tiny stub that satisfies the ensemble's predict contract."""
        def __init__(self, value: float):
            self._v = value

        def predict(self, X):
            return np.full(len(X), self._v)

    ensemble = CompositeCrossTargetEnsemble.from_uniform_weights(
        component_models=[_ConstModel(1.0), _ConstModel(3.0)],
        component_names=["a", "b"],
    )
    assert len(ensemble.component_models) == 2
    # Convex weights normalise to 0.5 / 0.5.
    np.testing.assert_allclose(ensemble.weights, [0.5, 0.5])
    assert ensemble.is_convex is True
    assert ensemble.strategy == "mean"


def test_from_uniform_weights_empty_components_raises_clear_error() -> None:
    """The fail-fast error message survived the split unchanged."""
    import pytest
    from mlframe.training.composite.ensemble import CompositeCrossTargetEnsemble
    with pytest.raises(ValueError, match="empty component list"):
        CompositeCrossTargetEnsemble.from_uniform_weights(
            component_models=[],
            component_names=[],
        )
