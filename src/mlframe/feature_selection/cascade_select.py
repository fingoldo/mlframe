"""``cascade_select``: 3-stage noise-injection screen -> forward selection -> permutation backward elimination.

Source: AV WNS Hackathon 2018 top-3, Team Cheburek - "Feature selection employed Gini importance and noised
LightGBM importance for initial screening, followed by forward algorithm selection, then permutation
importance with backward elimination" on 4600+ generated features down to ~21/61.

Orchestrates three EXISTING mlframe primitives rather than reimplementing any of them:

1. :func:`mlframe.feature_selection.filters._boruta.boruta_select` - Boruta-style all-relevant screen
   (shadow-shuffled copies of every feature; a real feature is confirmed only when it repeatedly beats the
   best shadow importance more often than chance). Cuts a very large raw feature set down to the "clearly
   not noise" subset cheaply.
2. :func:`mlframe.feature_selection.forward_select.forward_select` (added alongside this orchestrator -
   the one genuinely missing primitive; MRMR's own greedy loop is MI/redundancy-driven, not CV-score-driven,
   and RFECV's SFFS pass is a refinement of an already-narrowed subset, not a from-scratch greedy build).
   Grows a candidate subset from the Boruta-confirmed features by best CV-score marginal improvement.
3. :class:`mlframe.feature_selection.wrappers.rfecv.RFECV` with ``importance_getter="permutation"`` -
   already supports permutation-importance-driven backward elimination natively; used here as a black box
   via its documented public API (constructor + ``fit`` + ``support_``), not modified.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import numpy as np

from .filters._boruta import boruta_select
from .forward_select import forward_select


def cascade_select(
    X: Any,
    y: np.ndarray,
    estimator_factory: Callable[[], Any],
    n_boruta_iterations: int = 20,
    boruta_alpha: float = 0.05,
    forward_max_features: Optional[int] = None,
    forward_min_improvement: float = 0.0,
    cv: int = 5,
    scoring: Optional[str] = None,
    random_state: int = 42,
    rfecv_kwargs: Optional[Dict[str, Any]] = None,
    X_holdout: Any = None,
    y_holdout: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Run the 3-stage cascade: Boruta screen -> forward selection -> permutation backward elimination.

    Parameters
    ----------
    X
        Feature frame (pandas DataFrame with named columns).
    y
        Target array.
    estimator_factory
        Zero-arg callable returning a fresh unfitted tree-based estimator exposing ``feature_importances_``
        (used for the Boruta screen's importance function) and usable by ``forward_select``/``RFECV``.
    n_boruta_iterations, boruta_alpha
        Passed to :func:`boruta_select`. Features decided ``"confirmed"`` or ``"tentative"`` survive to
        stage 2 (``"rejected"`` features are dropped - "tentative" is kept rather than dropped since a
        false rejection here is unrecoverable, while stages 2-3 will naturally deprioritize a truly-useless
        tentative feature anyway).
    forward_max_features, forward_min_improvement
        Passed to :func:`forward_select`.
    cv, scoring
        Shared CV configuration for stages 2 and 3.
    random_state
        Seed for the Boruta shadow shuffles and RFECV.
    rfecv_kwargs
        Extra keyword arguments forwarded to :class:`RFECV`'s constructor (``importance_getter`` defaults to
        ``"permutation"`` unless overridden here).

    Returns
    -------
    dict
        ``boruta_result`` (the raw :func:`boruta_select` output), ``boruta_confirmed`` (list of column
        names), ``forward_selected`` (list, in the order added), ``final_selected`` (list, the RFECV
        backward-elimination output), ``rfecv`` (the fitted :class:`RFECV` instance), and ``holdout_score``.

        The scores inside ``rfecv`` (``cv_results_["cv_mean_perf"]`` and the like) are a RANKING signal only: they were
        computed on the same rows Boruta and forward selection already used to narrow the candidates, so they are
        selection-optimistic and must not be reported as an estimate of generalisation. ``holdout_score`` is: the
        final subset refitted on ``X`` and scored (``scoring``, or the estimator's own ``score``) on the untouched
        ``X_holdout`` / ``y_holdout`` when the caller passes them, else None.
    """
    from .wrappers.rfecv import RFECV

    if not hasattr(X, "columns"):
        raise TypeError("cascade_select: X must be a pandas DataFrame with named columns.")

    def _importance_fn(X_shadowed: Any, y_arr: np.ndarray) -> np.ndarray:
        """Fit a fresh estimator on the (possibly shadow-augmented) features and return its feature importances."""
        model = estimator_factory()
        model.fit(X_shadowed, y_arr)
        return np.asarray(model.feature_importances_, dtype=np.float64)

    boruta_result = boruta_select(X, y, _importance_fn, n_iterations=n_boruta_iterations, alpha=boruta_alpha, random_state=random_state)
    boruta_confirmed = [name for name, decision in zip(boruta_result["feature_names"], boruta_result["decision"]) if decision in ("confirmed", "tentative")]

    if not boruta_confirmed:
        return {"boruta_result": boruta_result, "boruta_confirmed": [], "forward_selected": [], "final_selected": [], "rfecv": None}

    X_stage1 = X[boruta_confirmed]
    forward_selected = forward_select(
        X_stage1, y, estimator_factory, scoring=scoring, cv=cv, max_features=forward_max_features, min_improvement=forward_min_improvement,
    )

    if not forward_selected:
        return {"boruta_result": boruta_result, "boruta_confirmed": boruta_confirmed, "forward_selected": [], "final_selected": [], "rfecv": None}

    X_stage2 = X[forward_selected]
    rfecv_config = dict(rfecv_kwargs) if rfecv_kwargs else {}
    rfecv_config.setdefault("importance_getter", "permutation")
    rfecv_config.setdefault("cv", cv)
    rfecv_config.setdefault("random_state", random_state)
    rfecv_config.setdefault("verbose", 0)
    rfecv = RFECV(estimator=estimator_factory(), **rfecv_config)
    rfecv.fit(X_stage2, y)
    final_selected = list(np.asarray(forward_selected)[rfecv.support_])

    return {
        "boruta_result": boruta_result,
        "boruta_confirmed": boruta_confirmed,
        "forward_selected": forward_selected,
        "final_selected": final_selected,
        "rfecv": rfecv,
        "holdout_score": _holdout_score(estimator_factory, X, y, final_selected, X_holdout, y_holdout, scoring),
    }


def _holdout_score(estimator_factory, X, y, selected, X_holdout, y_holdout, scoring) -> Optional[float]:
    """Refit on ``selected`` columns of ``X`` and score on the caller's untouched holdout; None when there is none."""
    if X_holdout is None or y_holdout is None or not selected:
        return None
    model = estimator_factory()
    model.fit(X[selected], y)
    if scoring is None:
        return float(model.score(X_holdout[selected], y_holdout))
    from sklearn.metrics import get_scorer

    return float(get_scorer(scoring)(model, X_holdout[selected], y_holdout))


__all__ = ["cascade_select"]
