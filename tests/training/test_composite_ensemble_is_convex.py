"""Regressions for findings #6 (is_convex flag) and #7 (NNLS no-renormalise).

F6: ``from_linear_stack`` historically bypassed the sum=1 invariant by
mutating ``instance.weights`` after construction. Downstream code asserting
sum=1 misbehaved silently. Mark non-convex via the new ``is_convex`` attribute
so consumers can gate the invariant explicitly.

F7: ``from_nnls_stack`` normalised NNLS weights to sum=1 after the fit. The
deployed predictor then differed from the one NNLS solved for (and the gate
evaluated). Fix: keep the raw NNLS weights, mark is_convex=False.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from mlframe.training.composite_ensemble import CompositeCrossTargetEnsemble


def _components(n_train: int = 200, n_components: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    y = rng.normal(loc=5.0, scale=2.0, size=n_train)
    preds = np.column_stack(
        [y + rng.normal(scale=0.5, size=n_train) for _ in range(n_components)]
    )
    models = [MagicMock(name=f"m{i}") for i in range(n_components)]
    names = [f"m{i}" for i in range(n_components)]
    return models, names, preds, y


def test_from_uniform_weights_is_convex_true():
    models, names, _, _ = _components()
    ens = CompositeCrossTargetEnsemble.from_uniform_weights(models, names)
    assert ens.is_convex is True
    assert np.isclose(ens.weights.sum(), 1.0)


def test_from_linear_stack_is_convex_false():
    models, names, preds, y = _components()
    ens = CompositeCrossTargetEnsemble.from_linear_stack(
        component_models=models, component_names=names,
        component_predictions=preds, y_train=y, ridge_alpha=1.0,
    )
    assert ens.is_convex is False, "linear_stack must not be marked convex"
    # Weights are the Ridge coefficients; do NOT assume sum=1.
    # We don't pin a specific value here -- just that the invariant is OFF.


def test_from_nnls_stack_is_convex_false():
    models, names, preds, y = _components()
    ens = CompositeCrossTargetEnsemble.from_nnls_stack(
        component_models=models, component_names=names,
        component_predictions=preds, y_train=y,
    )
    assert ens.is_convex is False, "nnls_stack must not be marked convex (F6)"


def test_from_nnls_stack_weights_not_post_normalised():
    """F7: deployed weights must equal NNLS's raw output, not the renormalised version.

    Pre-fix the constructor divided weights by their sum, so the deployed model used
    w/sum(w) instead of w. We re-run NNLS on the same matrix and assert equality with
    the stored weights (no renormalisation in between).
    """
    from scipy.optimize import nnls

    models, names, preds, y = _components(seed=7)
    ens = CompositeCrossTargetEnsemble.from_nnls_stack(
        component_models=models, component_names=names,
        component_predictions=preds, y_train=y,
    )
    # Independent NNLS call -- same inputs (finite rows only).
    finite = np.isfinite(y) & np.all(np.isfinite(preds), axis=1)
    w_ref, _ = nnls(preds[finite], y[finite])
    np.testing.assert_allclose(ens.weights, w_ref, rtol=1e-9, atol=1e-9)
    # Sanity: if the bug was present, ens.weights would be w_ref / w_ref.sum()
    # which (for non-degenerate y in our fixture) does NOT equal w_ref.
    if w_ref.sum() > 0 and not np.isclose(w_ref.sum(), 1.0):
        assert not np.allclose(ens.weights, w_ref / w_ref.sum()), (
            "weights look renormalised; F7 regression"
        )


def test_constructor_is_convex_false_skips_renormalisation():
    # Bypass factories -- direct constructor invocation with raw weights.
    models, names, _, _ = _components(n_components=3)
    weights = np.array([2.0, 5.0, -1.0])  # negative + non-sum-1
    ens = CompositeCrossTargetEnsemble(
        component_models=models, component_names=names,
        weights=weights, strategy="linear_stack", is_convex=False,
    )
    np.testing.assert_allclose(ens.weights, weights)
    assert ens.is_convex is False


def test_constructor_is_convex_true_normalises():
    models, names, _, _ = _components(n_components=3)
    weights = np.array([1.0, 1.0, 2.0])
    ens = CompositeCrossTargetEnsemble(
        component_models=models, component_names=names,
        weights=weights, strategy="oof_weighted",
    )
    assert ens.is_convex is True
    np.testing.assert_allclose(ens.weights.sum(), 1.0)
