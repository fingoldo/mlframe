"""Round budget for OOF fold models: stop where the deployed model stopped instead of running every configured round."""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

_ROUND_PARAMS = ("iterations", "n_estimators")


def deployed_round_budget(model: Any) -> Optional[int]:
    """Number of boosting rounds the already-fit ``model`` kept after early stopping, or ``None`` when unknown.

    Reads LightGBM ``best_iteration_`` (1-based), XGBoost ``best_iteration`` (0-based) and CatBoost ``get_best_iteration()`` (0-based).
    A best round of ``None``/0 on a LightGBM model means early stopping never fired, which is not a budget.
    """
    inner = getattr(model, "steps", None)
    if inner:
        model = inner[-1][1]
    inner_reg = getattr(model, "regressor_", None)
    if inner_reg is not None:
        model = inner_reg
    best = getattr(model, "best_iteration_", None)
    if isinstance(best, int) and not isinstance(best, bool) and best > 0:
        return int(best)
    best = getattr(model, "best_iteration", None)
    if isinstance(best, int) and not isinstance(best, bool) and best >= 0 and hasattr(model, "get_booster"):
        return int(best) + 1
    getter = getattr(model, "get_best_iteration", None)
    if callable(getter):
        try:
            cb_best = getter()
        except Exception as exc:
            logger.debug("get_best_iteration unavailable on %s: %r", type(model).__name__, exc)
            return None
        if isinstance(cb_best, int) and cb_best >= 0:
            return int(cb_best) + 1
    return None


def apply_round_budget(estimator: Any, rounds: Optional[int]) -> Optional[str]:
    """Set the round-count hyperparameter of the unfitted OOF clone to ``rounds``; return the parameter name set, else ``None``."""
    if rounds is None or rounds <= 0:
        return None
    try:
        params = estimator.get_params(deep=False)
    except (AttributeError, TypeError):
        return None
    for name in _ROUND_PARAMS:
        if name in params:
            try:
                estimator.set_params(**{name: int(rounds)})
            except (ValueError, TypeError):
                return None
            return name
    return None
