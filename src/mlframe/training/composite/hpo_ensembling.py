"""Turn an HPO trial OOF pool into a ready-to-use combined prediction.

``optimize_composite(..., collect_oof_pool=True)`` harvests every trial's leakage-free OOF
prediction array on :attr:`CompositeHPOResult.trial_oof_pool` (see ``hpo.py``). Naively
averaging the WHOLE pool is a known loser (poorly-tuned/early-diverging trials inject noise
that outweighs the diversity win -- honest measurement: RMSE 0.4335 vs single-best 0.3104 on
the composite-HPO synthetic). The biz_value test in ``test_biz_val_hpo_oof_pool.py`` shows a
manual top-K-by-CV-score-then-average fix beats single-best, but every caller who wants that
has to re-derive it by hand: filter ``trials`` by score, refit each kept config, average.

This module closes that gap by wiring the pool directly into an EXISTING, already-validated
selection primitive -- :func:`mlframe.models.ensembling.selection.stepwise_ensemble_selection`
(bidirectional forward-add/backward-remove ensemble member selection, proven this session to
beat naive forward-only selection) -- instead of hand-rolling a new top-K heuristic. Selection
runs directly on the OOF pool (no refit needed: the pool predictions are already leakage-free
CV-OOS), so it is essentially free relative to the HPO search that produced them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, List, Optional

import numpy as np

from ..utils import coerce_to_1d_numpy as _to_1d_numpy
from ...models.ensembling.selection import stepwise_ensemble_selection
from .hpo import _rmse

if TYPE_CHECKING:
    from .hpo import CompositeHPOResult

__all__ = ["OOFPoolSelectionResult", "select_oof_pool_ensemble"]


@dataclass
class OOFPoolSelectionResult:
    """Outcome of :func:`select_oof_pool_ensemble`.

    ``combined_oof`` is the uniform-mean blend of the ``kept_trial_indices`` members' OOF
    predictions -- ready to use directly as a stacking input, no further refit needed.
    ``kept_trial_indices`` / ``dropped_trial_indices`` index into the ORIGINAL
    ``trial_oof_pool`` / ``trials`` order (not the NaN-filtered internal working set), so a
    caller can cross-reference ``result.trials[i]`` for the ``(transform, inner_params)`` of
    any kept or dropped trial. ``score`` is the stepwise walk's held-out metric of the kept
    blend (RMSE by default, lower is better unless ``greater_is_better=True`` was passed).
    """

    kept_trial_indices: List[int]
    dropped_trial_indices: List[int]
    combined_oof: np.ndarray
    score: float


def select_oof_pool_ensemble(
    result: "CompositeHPOResult",
    y: object,
    *,
    metric: Optional[Callable] = None,
    greater_is_better: bool = False,
    max_picks: int = 100,
    max_rounds: int = 200,
    with_replacement: bool = False,
    tol: float = 0.0,
    min_models: int = 1,
) -> OOFPoolSelectionResult:
    """Run :func:`stepwise_ensemble_selection` directly over an HPO ``trial_oof_pool``.

    Convenience wiring for the ``optimize_composite(..., collect_oof_pool=True)`` ->
    ensemble-selection pipeline a caller would otherwise have to hand-assemble (filter
    ``result.trials`` by score, refit each kept config, average). Since every pool member is
    already a leakage-free OOF prediction, selection runs directly on the pool -- no refit.

    Parameters
    ----------
    result
        A :class:`CompositeHPOResult` from ``optimize_composite(..., collect_oof_pool=True)``.
    y
        The same target ``optimize_composite`` was fit against (1-D, length ``n_rows``).
    metric, greater_is_better, max_picks, max_rounds, with_replacement, tol, min_models
        Forwarded to :func:`stepwise_ensemble_selection`. ``metric=None`` defaults to RMSE
        (``greater_is_better=False``) to match ``optimize_composite``'s default scorer -- NOT
        ``stepwise_ensemble_selection``'s own default of ROC-AUC, which would silently score a
        regression pool with a classification metric.

    Returns
    -------
    OOFPoolSelectionResult
        Kept/dropped trial indices (into the ORIGINAL pool order) + the combined OOF array +
        the selection's held-out score.

    Raises
    ------
    ValueError
        ``result.trial_oof_pool`` is ``None`` (HPO was run with ``collect_oof_pool=False``), or
        every trial's OOF array contains at least one NaN (no fold ever fully succeeded for any
        trial, so no candidate is usable by the NaN-free selection routine).
    """
    if result.trial_oof_pool is None:
        raise ValueError("select_oof_pool_ensemble: result.trial_oof_pool is None -- rerun optimize_composite(..., collect_oof_pool=True).")

    y_arr = _to_1d_numpy(y)
    pool = result.trial_oof_pool

    # stepwise_ensemble_selection has no NaN handling; a trial with ANY failed fold (NaN row)
    # is dropped from the selectable set up front rather than propagating NaN into every score.
    valid_indices = [i for i, oof in enumerate(pool) if np.all(np.isfinite(oof))]
    if not valid_indices:
        raise ValueError("select_oof_pool_ensemble: every pooled trial has at least one NaN OOF row (no trial CV'd cleanly on all folds) -- nothing selectable.")

    stacked = np.stack([pool[i] for i in valid_indices], axis=0)
    resolved_metric = metric if metric is not None else _rmse

    selection = stepwise_ensemble_selection(
        stacked,
        y_arr,
        metric=resolved_metric,
        greater_is_better=greater_is_better,
        max_picks=max_picks,
        max_rounds=max_rounds,
        with_replacement=with_replacement,
        tol=tol,
        min_models=min_models,
    )

    kept_trial_indices = sorted(valid_indices[i] for i in selection.kept)
    dropped_trial_indices = sorted(i for i in range(len(pool)) if i not in kept_trial_indices)
    combined_oof = np.mean(np.stack([pool[i] for i in kept_trial_indices], axis=0), axis=0)

    return OOFPoolSelectionResult(
        kept_trial_indices=kept_trial_indices,
        dropped_trial_indices=dropped_trial_indices,
        combined_oof=combined_oof,
        score=float(selection.score),
    )
