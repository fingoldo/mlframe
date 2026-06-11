"""Monotonic-constraint passthrough for ``CompositeTargetEstimator``.

Carved out of ``_estimator.py`` to keep that module under the 1k-LOC limit.
``_apply_monotone_constraints`` (rebound onto the class at the parent's module
bottom) validates the constraint vector length / values against the post-drop
feature count -- from ``_frame_utils._count_feature_columns`` -- and forwards it
to the inner GBDT via ``set_params``.

The constraint is enforced on the inner's T (residual) target. For the
additive-residual cores (linear_residual / diff family) the inverse adds a
base-only term back, so monotonicity in T carries through to y at fixed base;
see ``CompositeTargetEstimator``'s ``monotone_constraints`` docstring.
"""
from __future__ import annotations

from typing import Any


def _apply_monotone_constraints(self: Any, estimator: Any, n_features: int) -> None:
    """Validate ``self.monotone_constraints`` length + forward to ``estimator``.

    ``n_features`` is the post-drop feature count (columns the inner trains on).
    The constraint vector must match it exactly so a +1/-1/0 lines up with the
    right feature; a length mismatch is a hard config error (silent truncation /
    zero-padding would constrain the WRONG feature).

    LightGBM, XGBoost and CatBoost all accept a ``monotone_constraints``
    estimator param, so we forward via ``set_params``. We probe support with
    ``set_params`` itself rather than ``get_params``: LightGBM passes
    ``monotone_constraints`` through to the booster as a ``**kwargs`` extra and
    does NOT list it in its default ``get_params`` (so a get_params membership
    check would wrongly reject LightGBM), whereas a non-GBDT estimator such as
    ``LinearRegression`` raises ``ValueError`` from ``set_params`` for the
    unknown param. We translate that into a clear ``TypeError`` so the user is
    not misled into thinking a constraint was applied.
    """
    constraints = list(self.monotone_constraints)
    if len(constraints) != n_features:
        raise ValueError(
            f"CompositeTargetEstimator: monotone_constraints has length "
            f"{len(constraints)} but the inner estimator trains on {n_features} "
            f"features (after dropping plumbing columns such as group_column). "
            f"The constraint vector must match the post-drop feature count exactly."
        )
    bad = [c for c in constraints if c not in (-1, 0, 1)]
    if bad:
        raise ValueError(
            f"CompositeTargetEstimator: monotone_constraints entries must each be "
            f"-1, 0, or +1; got disallowed values {sorted(set(bad))}."
        )
    try:
        estimator.set_params(monotone_constraints=constraints)
    except ValueError as err:
        # sklearn's set_params raises ValueError "Invalid parameter ..." when the
        # estimator does not accept monotone_constraints (e.g. LinearRegression).
        raise TypeError(
            f"CompositeTargetEstimator: inner estimator "
            f"'{type(estimator).__name__}' does not accept a 'monotone_constraints' "
            f"parameter ({err}); cannot enforce the requested monotonicity. Use a "
            f"GBDT that supports it (LightGBM / XGBoost / CatBoost)."
        ) from err
