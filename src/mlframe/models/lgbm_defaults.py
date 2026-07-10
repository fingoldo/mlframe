"""Default LightGBM hyperparameter preset, including ``extra_trees=True``.

LightGBM's ``extra_trees`` option picks split thresholds RANDOMLY within each feature's candidate range
(rather than exhaustively searching for the locally-optimal split, as sklearn's ``ExtraTreesClassifier``
does relative to a standard Random Forest) — this decorrelates the individual trees, trading a slightly
weaker single tree for a materially more diverse ensemble. Ubiquant's 7th place team reported "steady
improvement when the number of trees goes large" from just flipping this one flag. It is off by default in
LightGBM itself and easy to never discover; this preset makes it the mlframe default so users get the win
without having to know the flag exists.
"""
from __future__ import annotations

from typing import Any, Dict


def default_lgbm_params(
    objective: str = "regression",
    extra_trees: bool = True,
    **overrides: Any,
) -> Dict[str, Any]:
    """Sensible default LightGBM hyperparameters, with ``extra_trees`` ON by default.

    Parameters
    ----------
    objective
        LightGBM ``objective`` string (e.g. ``"regression"``, ``"binary"``, ``"multiclass"``).
    extra_trees
        Forwarded as LightGBM's ``extra_trees`` param. Default ``True`` — decorrelates trees via randomized
        split-threshold selection, which measurably helps once ``n_estimators`` is large (see module
        docstring). Set ``False`` to restore LightGBM's own default (standard greedy-best-split trees).
    **overrides
        Any additional LightGBM param, applied last (overrides anything set above, including
        ``extra_trees`` itself if explicitly passed here too).

    Returns
    -------
    dict
        A ``**kwargs``-ready dict for ``lightgbm.LGBMRegressor`` / ``LGBMClassifier`` / the sklearn API.
    """
    params: Dict[str, Any] = {
        "objective": objective,
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "extra_trees": extra_trees,
        "n_jobs": -1,
        "verbose": -1,
        "random_state": 0,
    }
    params.update(overrides)
    return params


__all__ = ["default_lgbm_params"]
