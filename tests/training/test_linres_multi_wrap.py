"""Regression: wrapping fitted inner models for multi-base composite specs
(``linear_residual_multi`` with non-empty ``extra_base_columns``) must
pass the full base column tuple to ``CompositeTargetEstimator.
from_fitted_inner`` so predict reconstructs the (n, K) base matrix
matching the K alphas saved in ``fitted_params``.

Pre-fix path (iter-52 300k seed=31 cb-regression):
1. Discovery's forward-stepwise auto-promotion produced two
   ``linear_residual_multi`` specs with base set {x0, x1} (one named
   "linresM-x0+x1", the other "linresM-x1+x0").
2. CatBoost was trained on the T-scale composite target -> 2 alphas
   saved in ``fitted_params``.
3. ``_phase_composite_post.py`` wrapped each fitted inner via
   ``CompositeTargetEstimator.from_fitted_inner(... base_column=
   spec["base_column"] ...)`` -- but did NOT pass
   ``extra_base_columns``, so the wrapper's ``base_columns`` stayed
   None and ``_resolve_base_columns()`` fell back to
   ``(base_column,)`` -- a 1-tuple.
4. At predict, ``_extract_base_for_transform`` returned a 1-D array.
   ``_linear_residual_multi_inverse`` reshapes to (n, 1) and raises:
   ``linear_residual_multi: base has 1 columns but fitted alphas has
   2 entries``.
5. iter-46 surfaced this as the per-model error in the aggregated
   predict_from_models output.

Post-fix: every ``from_fitted_inner`` call site that reads ``_spec``
now also reads ``spec.get("extra_base_columns")`` and builds the full
``base_columns`` tuple before calling the wrapper. Three sites fixed:
- core/_phase_composite_post.py (main wrap path, the iter-52 failure)
- composite_ensemble.py kfold OOF refit
- composite_ensemble.py single OOF refit
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from mlframe.training.composite import CompositeTargetEstimator
from mlframe.training.composite.transforms import _linear_residual_multi_fit


def _build_inner_and_spec_multi():
    """Build a Ridge inner already trained on a 2-base linres_multi
    T-scale target, plus a dict matching the on-disk ``_spec`` shape
    used by ``_phase_composite_post`` and the OOF-refit paths."""
    rng = np.random.default_rng(0)
    n = 300
    x0 = rng.standard_normal(n).astype(np.float64)
    x1 = rng.standard_normal(n).astype(np.float64)
    feat = rng.standard_normal(n).astype(np.float64)
    # y depends on x0 + x1 + a signal feature so multi-base joint OLS
    # is well-conditioned and produces 2 non-zero alphas.
    y = 0.7 * x0 + 0.4 * x1 + 0.5 * feat + rng.standard_normal(n) * 0.1
    base_matrix = np.column_stack([x0, x1])
    params = _linear_residual_multi_fit(y, base_matrix)
    assert not params["collinear_fallback"], "test fixture should not trigger fallback"
    assert len(params["alphas"]) == 2, "test fixture must produce 2 alphas"

    # T-scale target the inner sees at fit time.
    t = y - (base_matrix @ np.asarray(params["alphas"])) - params["beta"]
    # The inner is trained on (X, t). X carries the base cols AND
    # downstream features (mimics the production-train path).
    X = pd.DataFrame({"x0": x0, "x1": x1, "feat": feat})
    inner = Ridge(alpha=1e-3).fit(X, t)
    spec = {
        "transform_name": "linear_residual_multi",
        "base_column": "x0",
        "extra_base_columns": ("x1",),
        "fitted_params": params,
    }
    return inner, spec, X, y


def test_from_fitted_inner_with_extra_base_columns_predicts_without_error() -> None:
    """Construct the wrapper the way ``_phase_composite_post.py:119``
    does post-fix and verify the wrapper's predict() does NOT raise
    the iter-52 ValueError."""
    inner, spec, X, y = _build_inner_and_spec_multi()
    _extra = tuple(spec.get("extra_base_columns") or ())
    _base_columns = (spec["base_column"], *_extra) if _extra else None
    wrapper = CompositeTargetEstimator.from_fitted_inner(
        fitted_inner=inner,
        transform_name=spec["transform_name"],
        base_column=spec["base_column"],
        base_columns=_base_columns,
        transform_fitted_params=spec["fitted_params"],
        y_train=y,
    )
    # Pre-fix this raised:
    #   ValueError: linear_residual_multi: base has 1 columns but
    #   fitted alphas has 2 entries
    preds = wrapper.predict(X)
    assert preds.shape == (len(X),)
    assert np.all(np.isfinite(preds))
    # Sanity: predictions should be in roughly the same scale as y.
    assert abs(preds.mean() - y.mean()) < 1.0


def test_resolve_base_columns_returns_full_tuple_when_extras_passed() -> None:
    """Locks the wrapper's ``_resolve_base_columns()`` contract: when
    ``base_columns`` is supplied, it wins over ``base_column``."""
    inner, spec, X, y = _build_inner_and_spec_multi()
    wrapper = CompositeTargetEstimator.from_fitted_inner(
        fitted_inner=inner,
        transform_name=spec["transform_name"],
        base_column=spec["base_column"],
        base_columns=(spec["base_column"], *spec["extra_base_columns"]),
        transform_fitted_params=spec["fitted_params"],
        y_train=y,
    )
    assert wrapper._resolve_base_columns() == ("x0", "x1")


def test_pre_fix_path_raises_to_lock_regression() -> None:
    """Sensor that fails if a future change reintroduces the iter-52
    bug. Calls ``from_fitted_inner`` WITHOUT passing ``base_columns``
    (the broken pattern); predict must raise the iter-52 ValueError.
    If the wrapper grows graceful auto-recovery later, this test
    should be updated to reflect the new contract."""
    inner, spec, X, y = _build_inner_and_spec_multi()
    wrapper = CompositeTargetEstimator.from_fitted_inner(
        fitted_inner=inner,
        transform_name=spec["transform_name"],
        base_column=spec["base_column"],
        # base_columns OMITTED intentionally (the pre-fix path)
        transform_fitted_params=spec["fitted_params"],
        y_train=y,
    )
    with pytest.raises(ValueError) as excinfo:
        wrapper.predict(X)
    assert "base has 1 columns but fitted alphas has 2 entries" in str(excinfo.value)


def test_single_base_spec_still_works_without_extras() -> None:
    """Baseline: single-base linres_multi (1 alpha + extras tuple is
    empty) continues to work via the legacy single-column path. Locks
    the back-compat contract for the K=1 degenerate-multi case."""
    rng = np.random.default_rng(1)
    n = 200
    x0 = rng.standard_normal(n).astype(np.float64)
    feat = rng.standard_normal(n).astype(np.float64)
    y = 0.6 * x0 + 0.4 * feat + rng.standard_normal(n) * 0.1
    base_matrix = x0.reshape(-1, 1)
    params = _linear_residual_multi_fit(y, base_matrix)
    assert len(params["alphas"]) == 1
    t = y - (base_matrix @ np.asarray(params["alphas"])) - params["beta"]
    X = pd.DataFrame({"x0": x0, "feat": feat})
    inner = Ridge(alpha=1e-3).fit(X, t)
    spec = {
        "transform_name": "linear_residual_multi",
        "base_column": "x0",
        "extra_base_columns": (),
        "fitted_params": params,
    }
    _extra = tuple(spec.get("extra_base_columns") or ())
    _base_columns = (spec["base_column"], *_extra) if _extra else None
    assert _base_columns is None  # legacy single-base path
    wrapper = CompositeTargetEstimator.from_fitted_inner(
        fitted_inner=inner,
        transform_name=spec["transform_name"],
        base_column=spec["base_column"],
        base_columns=_base_columns,
        transform_fitted_params=spec["fitted_params"],
        y_train=y,
    )
    preds = wrapper.predict(X)
    assert preds.shape == (len(X),)
    assert np.all(np.isfinite(preds))
