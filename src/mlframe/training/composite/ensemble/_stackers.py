"""Richer OOF cross-target meta-learners for ``CompositeCrossTargetEnsemble``.

The default cross-target blend is NNLS (``from_nnls_stack``): a non-negative linear combination of the per-component
predictions. That is optimal only when the best blend is itself a convex / linear function of the component columns. When
the optimal blend is *interaction-shaped* or *non-convex* (e.g. "use component A in region R, component B elsewhere", or a
product of two component predictions), a linear solver cannot recover it and leaves accuracy on the table.

This module adds two alternative meta-learners that fit on the SAME leakage-free OOF prediction matrix the NNLS weights are
derived from, and are selected via an explicit ``stacker=`` choice:

- ``"ridge"`` -- Ridge regression over the OOF component matrix, with internal RidgeCV alpha selection and an optional
  ``non_negative`` constraint (``positive=True``). Still linear; useful when components are collinear (NNLS can be unstable
  there) and when a tiny negative coefficient genuinely helps. Reduces to a linear stack like ``from_linear_stack`` but is
  exposed here under the unified meta-stacker API + the non-negative knob.
- ``"gbm"`` -- a SHALLOW ``HistGradientBoostingRegressor`` (sklearn, always available, no extra dep) fit on the OOF matrix.
  This is a non-linear meta-learner: it can represent interactions / region-switching blends that NNLS and Ridge cannot.

Both attach a fitted ``_meta_model`` onto the ensemble; :meth:`CompositeCrossTargetEnsemble.predict` routes through that
meta-model when present (and ONLY then -- when ``_meta_model`` is absent the predict path is byte-for-byte the legacy linear
blend, so the default NNLS path stays bit-identical).

Leakage contract: the caller MUST pass an OUT-OF-FOLD component matrix (the same one the NNLS weights consume; produced by
``compute_oof_holdout_predictions``). This module does not and cannot verify out-of-foldness -- it mirrors the NNLS / Ridge
weight solvers, which carry the identical contract.

Default policy: NNLS stays the DEFAULT. The GBM / ridge stackers are opt-in via ``stacker=``. See the biz_value benchmark in
``tests/training/test_biz_val_composite_stackers.py``: GBM wins decisively on an interaction-shaped (region-switch / product)
blend, while on a plain convex blend all three are within noise (NNLS not regressed). A richer stacker did NOT win on the
MAJORITY of convex scenarios, so the default is unchanged (REJECTED-as-default != deleted: the option ships).
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Supported meta-stacker kinds. ``"nnls"`` is handled by the legacy ``from_nnls_stack`` path and listed here only so the
# dispatcher can validate the kwarg uniformly.
META_STACKER_KINDS = ("nnls", "ridge", "gbm")


def _clean_oof_matrix(
    oof_matrix: np.ndarray, oof_y: np.ndarray, n_components: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate + finite-filter the OOF (n, K) matrix and y vector.

    Returns ``(X_finite, y_finite, finite_mask)``. Raises ``ValueError`` on shape mismatch.
    """
    X = np.asarray(oof_matrix, dtype=np.float64)
    y = np.asarray(oof_y, dtype=np.float64).reshape(-1)
    if X.ndim != 2 or X.shape[1] != n_components:
        raise ValueError(
            f"meta-stacker: OOF matrix has shape {X.shape}; expected (n, {n_components})."
        )
    if y.shape[0] != X.shape[0]:
        raise ValueError(
            f"meta-stacker: y length {y.shape[0]} != OOF matrix rows {X.shape[0]}."
        )
    finite = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    return X[finite], y[finite], finite


def _validate_sample_weight(
    sample_weight: np.ndarray | None, n_rows: int, finite: np.ndarray,
) -> np.ndarray | None:
    if sample_weight is None:
        return None
    sw = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
    if sw.shape[0] != n_rows:
        raise ValueError(
            f"meta-stacker: sample_weight length {sw.shape[0]} != OOF matrix rows {n_rows}."
        )
    if not np.all(np.isfinite(sw)) or (sw < 0).any():
        raise ValueError(
            f"meta-stacker: sample_weight must be finite and non-negative; got "
            f"{int((~np.isfinite(sw)).sum())} non-finite and {int((sw < 0).sum())} negative entries (min={float(np.nanmin(sw))})."
        )
    return sw[finite]


def fit_ridge_meta_stacker(
    oof_matrix: np.ndarray,
    oof_y: np.ndarray,
    n_components: int,
    *,
    non_negative: bool = False,
    ridge_alpha: float | None = None,
    ridge_alpha_grid: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0),
    sample_weight: np.ndarray | None = None,
) -> Any:
    """Fit a Ridge meta-model on the leakage-free OOF component matrix.

    ``non_negative=True`` constrains coefficients to ``>= 0`` (``Ridge(positive=True)``), giving an NNLS-like non-negative
    linear stack with L2 regularisation -- more stable than raw NNLS when the component columns are collinear. With
    ``non_negative=False`` it is an unconstrained linear stack (negative coefficients allowed for anti-correlated members).

    When ``ridge_alpha is None`` the alpha is chosen by efficient leave-one-out ``RidgeCV`` over ``ridge_alpha_grid``;
    ``RidgeCV`` does not support ``positive=``, so under ``non_negative=True`` the alpha is selected by RidgeCV on the
    unconstrained problem, then a final ``Ridge(positive=True, alpha=chosen)`` is fit -- a cheap, standard two-step.

    Returns the fitted sklearn estimator (has ``.predict`` + ``.coef_`` + ``.intercept_``).
    """
    from sklearn.linear_model import Ridge, RidgeCV  # lazy: keep cold-import off the predict path

    X, y, finite = _clean_oof_matrix(oof_matrix, oof_y, n_components)
    sw = _validate_sample_weight(sample_weight, np.asarray(oof_matrix).shape[0], finite)
    if X.shape[0] < n_components + 2:
        raise ValueError(
            f"fit_ridge_meta_stacker: only {X.shape[0]} finite OOF rows for {n_components} components (need >= {n_components + 2})."
        )
    if ridge_alpha is None:
        rcv = RidgeCV(alphas=tuple(ridge_alpha_grid), fit_intercept=True)
        rcv.fit(X, y, sample_weight=sw)
        chosen = float(getattr(rcv, "alpha_", ridge_alpha_grid[0]))
    else:
        chosen = float(ridge_alpha)
    model = Ridge(alpha=chosen, fit_intercept=True, positive=bool(non_negative))
    model.fit(X, y, sample_weight=sw)
    return model


def fit_gbm_meta_stacker(
    oof_matrix: np.ndarray,
    oof_y: np.ndarray,
    n_components: int,
    *,
    max_depth: int = 3,
    max_iter: int = 200,
    learning_rate: float = 0.05,
    min_samples_leaf: int = 20,
    l2_regularization: float = 1.0,
    random_state: int = 0,
    sample_weight: np.ndarray | None = None,
) -> Any:
    """Fit a SHALLOW ``HistGradientBoostingRegressor`` meta-model on the leakage-free OOF component matrix.

    The defaults are intentionally conservative for a STACKER: a small K-column design (one column per component) is prone to
    over-fitting, so the depth is shallow (``max_depth=3``), the leaves are kept reasonably populated
    (``min_samples_leaf=20``), and L2 + early stopping (sklearn's built-in validation-fraction early stopping) keep the model
    from memorising the OOF rows. This is a non-linear meta-learner: it recovers interaction / region-switch blends that the
    linear (NNLS / Ridge) stackers cannot.

    sklearn is always available, so no extra dependency is incurred. Returns the fitted estimator (has ``.predict``).
    """
    from sklearn.ensemble import HistGradientBoostingRegressor  # lazy

    X, y, finite = _clean_oof_matrix(oof_matrix, oof_y, n_components)
    sw = _validate_sample_weight(sample_weight, np.asarray(oof_matrix).shape[0], finite)
    if X.shape[0] < max(n_components + 2, 30):
        raise ValueError(
            f"fit_gbm_meta_stacker: only {X.shape[0]} finite OOF rows; need >= {max(n_components + 2, 30)} to fit a GBM stacker without over-fitting."
        )
    # early_stopping on a small held-out fraction guards against memorising the K-column OOF design.
    model = HistGradientBoostingRegressor(
        max_depth=int(max_depth),
        max_iter=int(max_iter),
        learning_rate=float(learning_rate),
        min_samples_leaf=int(min_samples_leaf),
        l2_regularization=float(l2_regularization),
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=15,
        random_state=int(random_state),
    )
    model.fit(X, y, sample_weight=sw)
    return model


def build_meta_stack_ensemble(
    ensemble_cls: Any,
    component_models: list[Any],
    component_names: list[str],
    oof_matrix: np.ndarray,
    oof_y: np.ndarray,
    *,
    stacker: str = "ridge",
    non_negative: bool = False,
    sample_weight: np.ndarray | None = None,
    ridge_alpha: float | None = None,
    ridge_alpha_grid: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0),
    gbm_max_depth: int = 3,
    gbm_max_iter: int = 200,
    gbm_learning_rate: float = 0.05,
    random_state: int = 0,
) -> Any:
    """Construct a :class:`CompositeCrossTargetEnsemble` whose blend is a fitted meta-model over the OOF component matrix.

    ``stacker`` selects the meta-learner: ``"ridge"`` (optionally ``non_negative``) or ``"gbm"``. (Pass ``stacker="nnls"`` to
    use the legacy NNLS path via ``ensemble_cls.from_nnls_stack`` instead -- routed here for a single uniform entry point.)

    The returned ensemble carries a fitted ``_meta_model`` + ``_meta_stack_kind``; :meth:`predict` routes the per-component
    prediction matrix through that meta-model. On any meta-fit failure (degenerate / too-few-rows / solver error) this falls
    back to ``ensemble_cls.from_uniform_weights`` so the suite never loses the ensemble entirely.

    Leakage: ``oof_matrix`` MUST be out-of-fold (mirrors the NNLS / Ridge weight solvers' contract).
    """
    n = len(component_models)
    if n == 0:
        raise ValueError("build_meta_stack_ensemble: empty component list.")
    if len(component_names) != n:
        raise ValueError(
            f"build_meta_stack_ensemble: {n} models but {len(component_names)} names."
        )
    kind = str(stacker).lower()
    if kind not in META_STACKER_KINDS:
        raise ValueError(
            f"build_meta_stack_ensemble: unknown stacker {stacker!r}; choose one of {META_STACKER_KINDS}."
        )
    if kind == "nnls":
        return ensemble_cls.from_nnls_stack(
            component_models, component_names, oof_matrix, oof_y, sample_weight=sample_weight,
        )
    try:
        if kind == "ridge":
            meta = fit_ridge_meta_stacker(
                oof_matrix, oof_y, n,
                non_negative=non_negative,
                ridge_alpha=ridge_alpha,
                ridge_alpha_grid=ridge_alpha_grid,
                sample_weight=sample_weight,
            )
        else:  # gbm
            meta = fit_gbm_meta_stacker(
                oof_matrix, oof_y, n,
                max_depth=gbm_max_depth,
                max_iter=gbm_max_iter,
                learning_rate=gbm_learning_rate,
                random_state=random_state,
                sample_weight=sample_weight,
            )
    except Exception as exc:
        logger.warning(
            "[CompositeCrossTargetEnsemble] %s meta-stacker fit failed (%s); falling back to uniform-weight mean.",
            kind, exc,
        )
        return ensemble_cls.from_uniform_weights(component_models, component_names)

    # Placeholder weights kept for export / cap-by-|weight| compatibility: for a meta-stack the per-component
    # contribution is non-linear, so we record the meta-model's own notion of importance where it exposes one (GBM has
    # none cheaply; ridge exposes coef_). Uniform is a safe, sum-to-1 placeholder; predict ignores it (routes to _meta_model).
    placeholder_w = np.full(n, 1.0 / n, dtype=np.float64)
    notes: dict[str, Any] = {
        "meta_stacker": kind,
        "non_negative": bool(non_negative) if kind == "ridge" else None,
        "n_oof_rows": int(np.isfinite(np.asarray(oof_y, dtype=np.float64).reshape(-1)).sum()),
    }
    if kind == "ridge":
        notes["ridge_coef"] = np.asarray(getattr(meta, "coef_", []), dtype=np.float64).tolist()
        notes["ridge_intercept"] = float(getattr(meta, "intercept_", 0.0))
        notes["ridge_alpha"] = float(getattr(meta, "alpha", 0.0))
    instance = ensemble_cls(
        component_models=component_models,
        component_names=component_names,
        weights=placeholder_w,
        strategy=f"meta_{kind}",
        notes=notes,
        is_convex=True,  # placeholder weights ARE convex; predict short-circuits to _meta_model so this is never used for blending.
    )
    instance._meta_model = meta
    instance._meta_stack_kind = kind
    # Per-component OOF column means: the neutral stand-in predict uses when a component drops out at inference (the
    # meta-model needs its full K-column design). Cheap (K floats), kept on the instance (survives pickle).
    _Xclean, _yclean, _finite = _clean_oof_matrix(oof_matrix, oof_y, n)
    instance._meta_col_means = (
        _Xclean.mean(axis=0).tolist() if _Xclean.shape[0] > 0 else [0.0] * n
    )
    return instance
