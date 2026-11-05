"""Does an engineered feature carry held-out LINEAR value that the binned joint information of the selection cannot see?

The parsimony compaction drops an engineered feature whose removal leaves the bias-corrected binned joint information of the selection unchanged. For a
transform of a selected source that is exact: ``y = x**2 + 0.2 x`` is a function of the quantile bins of ``x``, so ``x__pade2`` adds nothing to ``x`` by the
data-processing inequality and is dropped, although a downstream LINEAR model cannot fit the quadratic from ``x`` alone. A feature is redundant only when it is
redundant in both views: here the held-out CV loss (log-loss of a scaled logistic model for classification, MAE of OLS for regression, the retention's own
scorers and fold split) of the selection with the feature is compared with the loss without it, and the feature is kept when removing it makes the loss worse
by more than the greedy's relative margin (1%).
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

_MAX_ROWS = 3000
_MARGIN = 0.01


def cv_loss(design: np.ndarray, y: np.ndarray, classification: bool, n_folds: int, seed: int) -> float:
    """Mean CV loss of a scaled linear model on the columns of ``design`` (log-loss for classification, MAE for regression); ``inf`` when it cannot be fit."""
    from sklearn.model_selection import KFold

    from .._usability_aware_selection import _scrub, _usability_scorers

    scorers = _usability_scorers(classification, y)
    if scorers is None:
        return float("inf")
    mk, logloss, y_enc, _n_classes = scorers
    x = _scrub(np.asarray(design, dtype=np.float64))  # Intentional float64: the retention gates score float64 columns
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if not np.isfinite(x).all():
        return float("inf")
    losses = []
    for tr, va in KFold(n_splits=n_folds, shuffle=True, random_state=seed).split(x):
        model = mk()
        if classification:
            model.fit(x[tr], y_enc[tr])
            losses.append(logloss(y_enc[va], model.predict_proba(x[va])))
        else:
            model.fit(x[tr], y[tr])
            losses.append(float(np.mean(np.abs(y[va] - model.predict(x[va])))))
    return float(np.mean(losses))


def make_linear_value_check(self: Any, X: Any, recipes_by_name: dict, seed: int = 0, n_folds: int = 3) -> Optional[Callable[[str, "list[str]"], bool]]:
    """``check(feature, rest)``: whether dropping ``feature`` from ``rest + [feature]`` worsens the held-out linear loss by more than the margin; ``None`` when no continuous target exists
    (the compaction then keeps its information-only rule)."""
    y_cont = getattr(self, "_fe_prewarp_y_continuous_", None)
    if y_cont is None:
        return None
    try:
        from .._fe_pure_form_retention import _detect_pure_form_task
        from .._fe_retention_subsumption import replay_candidate_continuous
        from ..engineered_recipes.shared import apply_recipe

        y_all = np.asarray(y_cont, dtype=np.float64).ravel()  # Intentional float64: as above
        n = y_all.shape[0]
        if n != len(X) or not np.isfinite(y_all).any():
            return None
        rng = np.random.default_rng(seed)
        rows = np.sort(rng.choice(n, _MAX_ROWS, replace=False)) if n > _MAX_ROWS else np.arange(n)
        y = y_all[rows]
        is_clf, _codes, classes = _detect_pure_form_task(y)
        if is_clf and classes.size < 2:
            return None
    except Exception as exc:
        logger.debug("compaction linear-value check unavailable: %s", exc)
        return None
    cache: "dict[str, np.ndarray]" = {}

    def column(name: str) -> "np.ndarray | None":
        """The continuous values of a selected raw column or engineered recipe on the sampled rows."""
        if name in cache:
            return cache[name]
        try:
            if name in recipes_by_name:
                vals = np.asarray(replay_candidate_continuous(recipes_by_name[name], X, apply_recipe), dtype=np.float64).ravel()  # Intentional float64: as above
            else:
                vals = np.asarray(X[name].to_numpy(), dtype=np.float64).ravel()  # Intentional float64: as above
            cache[name] = vals[rows] if vals.shape[0] == n else None
        except Exception as exc:
            logger.debug("compaction linear-value check: cannot read %s (%s)", name, exc)
            cache[name] = None
        return cache[name]

    def check(feature: str, rest: "list[str]") -> bool:
        """Whether ``feature`` lowers the held-out linear loss of ``rest`` by more than the margin."""
        cols = [column(nm) for nm in [*rest, feature]]
        if any(c is None for c in cols):
            return False
        with_e = cv_loss(np.column_stack(cols), y, is_clf, n_folds, seed)
        if len(cols) == 1:
            return bool(np.isfinite(with_e))
        without = cv_loss(np.column_stack(cols[:-1]), y, is_clf, n_folds, seed)
        return bool(np.isfinite(with_e) and (not np.isfinite(without) or without > (1.0 + _MARGIN) * with_e))

    def replacement_loses(candidate: str, replaced: "list[str]", rest: "list[str]") -> bool:
        """Whether swapping ``replaced`` for ``candidate`` beside ``rest`` raises the held-out linear loss by more than the margin."""
        old_cols = [column(nm) for nm in [*rest, *replaced]]
        new_cols = [column(nm) for nm in [*rest, candidate]]
        if any(c is None for c in old_cols) or any(c is None for c in new_cols):
            return False
        old_loss = cv_loss(np.column_stack(old_cols), y, is_clf, n_folds, seed)
        new_loss = cv_loss(np.column_stack(new_cols), y, is_clf, n_folds, seed)
        return bool(np.isfinite(old_loss) and (not np.isfinite(new_loss) or new_loss > (1.0 + _MARGIN) * old_loss))

    check.replacement_loses = replacement_loses  # type: ignore[attr-defined]
    return check
