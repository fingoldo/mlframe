"""Cross-fitted scoring of a stacked cross-target ensemble: weights fitted on some OOF rows, scored on the others.

NNLS minimises squared error over w >= 0 and every unit vector is feasible, so a stack scored on the matrix it was fitted
on can never lose to its best single component: the "fall back to the best single" gate could not fire for ``nnls_stack``,
and in practice not for a lightly regularised ``linear_stack`` either. Scoring each fold with weights fitted on the other
folds gives the gate an out-of-sample number to compare.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ...composite.discovery._splitter import make_discovery_splitter

_N_SPLITS = 5


def cross_fitted_stack_rmse(ens_cls: Any, strategy: str, components: Sequence[Any], names: Sequence[str], P: np.ndarray, y: np.ndarray,
                            random_state: int = 0) -> float:
    """RMSE of the ``strategy`` stack on each OOF fold when its weights are fitted on the remaining folds; NaN when too small."""
    P = np.asarray(P, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if P.shape[0] < 4 * _N_SPLITS or P.shape[1] == 0:
        return float("nan")
    build = ens_cls.from_linear_stack if strategy == "linear_stack" else ens_cls.from_nnls_stack
    pred = np.full(y.shape, np.nan)
    for tr, te in make_discovery_splitter(_N_SPLITS, random_state=random_state)[0].split(P):
        ens = build(component_models=list(components), component_names=list(names), component_predictions=P[tr], y_train=y[tr])
        w = np.asarray(ens.weights, dtype=np.float64)
        if getattr(ens, "is_convex", True):
            s = float(w.sum())
            pred[te] = P[te] @ (w / s if s > 0 else np.full_like(w, 1.0 / w.size))
        else:
            pred[te] = P[te] @ w + float(getattr(ens, "_linear_stack_intercept", 0.0))
    ok = np.isfinite(pred) & np.isfinite(y)
    return float(np.sqrt(np.mean((pred[ok] - y[ok]) ** 2))) if ok.any() else float("nan")


def refit_capped_stack(ens_cls: Any, strategy: str, capped: Any, oof_names: Sequence[str], P: Any, y: Any) -> Any:
    """Refit a capped non-convex stack on the OOF columns it kept, so its weights describe the predictor that ships.

    Capping kept the top components with their raw stack weights; the served blend lost the dropped weight mass on every
    row (the EST-03 bias, made deterministic). Convex strategies renormalise at predict and are returned unchanged.
    """
    if strategy not in ("nnls_stack", "linear_stack") or getattr(capped, "is_convex", True) or P is None or y is None:
        return capped
    names = list(capped.component_names)
    pos = [list(oof_names).index(n) for n in names if n in oof_names]
    if len(pos) != len(names):
        return capped
    build = ens_cls.from_linear_stack if strategy == "linear_stack" else ens_cls.from_nnls_stack
    return build(component_models=list(capped.component_models), component_names=names,
                 component_predictions=np.asarray(P, dtype=np.float64)[:, pos], y_train=np.asarray(y, dtype=np.float64))


def gate_stack_rmse(ens_cls: Any, strategy: str, components: Sequence[Any], names: Sequence[str], P: Any, y: Any, in_sample_pred: Any) -> float:
    """The ensemble RMSE the fallback gate compares: cross-fitted for a stack (in-sample it cannot lose), in-sample otherwise."""
    in_sample = float(np.sqrt(np.mean((np.asarray(in_sample_pred) - np.asarray(y)) ** 2)))
    if strategy not in ("nnls_stack", "linear_stack"):
        return in_sample
    cf = cross_fitted_stack_rmse(ens_cls, strategy, components, names, P, y)
    return cf if np.isfinite(cf) else in_sample

