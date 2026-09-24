"""Cross-fitted scoring of a stacked cross-target ensemble: weights fitted on some OOF rows, scored on the others.

NNLS minimises squared error over w >= 0 and every unit vector is feasible, so a stack scored on the matrix it was fitted
on can never lose to its best single component: the "fall back to the best single" gate could not fire for ``nnls_stack``,
and in practice not for a lightly regularised ``linear_stack`` either. Scoring each fold with weights fitted on the other
folds gives the gate an out-of-sample number to compare.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ...composite._row_roles import note_rows
from ...composite.discovery._splitter import make_discovery_splitter

_N_SPLITS = 5


def weighted_rmse(pred: Any, y: Any, sample_weight: Any = None) -> float:
    """RMSE over the rows where both are finite, weighted when ``sample_weight`` is given; NaN when no row qualifies."""
    pred, y = np.asarray(pred, dtype=np.float64).reshape(-1), np.asarray(y, dtype=np.float64).reshape(-1)
    ok = np.isfinite(pred) & np.isfinite(y)
    if not ok.any():
        return float("nan")
    w = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64).reshape(-1)[ok]
    return float(np.sqrt(np.average((pred[ok] - y[ok]) ** 2, weights=w)))


def column_rmses(P: Any, y: Any, sample_weight: Any = None) -> np.ndarray:
    """Per-component OOF RMSE over each column's finite rows, weighted when ``sample_weight`` is given."""
    P = np.asarray(P, dtype=np.float64)
    return np.array([weighted_rmse(P[:, j], y, sample_weight) for j in range(P.shape[1])], dtype=np.float64)


def oof_row_weights(sample_weight: Any, rows: Any) -> np.ndarray | None:
    """The sample weights of the OOF holdout rows (``rows`` are their positions among the weighted train rows), or None."""
    if sample_weight is None or rows is None:
        return None
    return np.asarray(np.asarray(sample_weight, dtype=np.float64).reshape(-1)[np.asarray(rows)])


def cross_fitted_stack_rmse(ens_cls: Any, strategy: str, components: Sequence[Any], names: Sequence[str], P: np.ndarray, y: np.ndarray,
                            random_state: int = 0, sample_weight: Any = None) -> float:
    """RMSE of the ``strategy`` stack on each OOF fold when its weights are fitted on the remaining folds; NaN when too small."""
    P = np.asarray(P, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if P.shape[0] < 4 * _N_SPLITS or P.shape[1] == 0:
        return float("nan")
    build = ens_cls.from_linear_stack if strategy == "linear_stack" else ens_cls.from_nnls_stack
    pred = np.full(y.shape, np.nan)
    for k, (tr, te) in enumerate(make_discovery_splitter(_N_SPLITS, random_state=random_state)[0].split(P)):
        note_rows("oof_rows", "fit", f"xt_stack_gate[fold {k}]", tr)
        note_rows("oof_rows", "report", f"xt_stack_gate[fold {k}]", te)
        ens = build(component_models=list(components), component_names=list(names), component_predictions=P[tr], y_train=y[tr],
                    sample_weight=None if sample_weight is None else np.asarray(sample_weight)[tr])
        w = np.asarray(ens.weights, dtype=np.float64)
        if getattr(ens, "is_convex", True):
            s = float(w.sum())
            pred[te] = P[te] @ (w / s if s > 0 else np.full_like(w, 1.0 / w.size))
        else:
            pred[te] = P[te] @ w + float(getattr(ens, "_linear_stack_intercept", 0.0))
    return weighted_rmse(pred, y, sample_weight)


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


def gate_stack_rmse(ens_cls: Any, strategy: str, components: Sequence[Any], names: Sequence[str], P: Any, y: Any, in_sample_pred: Any,
                    sample_weight: Any = None) -> float:
    """The ensemble RMSE the fallback gate compares: cross-fitted for a stack (in-sample it cannot lose), in-sample otherwise."""
    in_sample = weighted_rmse(in_sample_pred, y, sample_weight)
    if strategy not in ("nnls_stack", "linear_stack"):
        return in_sample
    cf = cross_fitted_stack_rmse(ens_cls, strategy, components, names, P, y, sample_weight=sample_weight)
    if np.isfinite(cf):
        return cf
    rows = np.arange(np.asarray(y).reshape(-1).size)  # too few rows to cross-fit: the weights are scored on the rows they were fit on
    note_rows("oof_rows", "fit", "xt_stack_gate[in-sample]", rows)
    note_rows("oof_rows", "report", "xt_stack_gate[in-sample]", rows)
    return in_sample
