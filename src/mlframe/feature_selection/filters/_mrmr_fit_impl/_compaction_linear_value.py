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
from typing import Any, Optional

import numpy as np

from .._linear_cv_loss import cv_loss

logger = logging.getLogger(__name__)

_MAX_ROWS = 3000
_MARGIN = 0.01


class LinearValueCheck:
    """Held-out linear loss of column sets of the selection, on at most ``_MAX_ROWS`` sampled rows (the retention's scorers, a fixed fold split)."""

    def __init__(self, X: Any, recipes_by_name: dict, y: np.ndarray, rows: np.ndarray, is_clf: bool, seed: int, n_folds: int, replay: Any, apply_recipe: Any) -> None:
        self._X = X
        self._recipes = recipes_by_name
        self._y = y
        self._rows = rows
        self._n_total = len(X)
        self._is_clf = is_clf
        self._seed = seed
        self._n_folds = n_folds
        self._replay = replay
        self._apply_recipe = apply_recipe
        self._cache: "dict[str, Optional[np.ndarray]]" = {}

    def _column(self, name: str) -> "Optional[np.ndarray]":
        """The continuous values of a selected raw column or engineered recipe on the sampled rows; ``None`` when they cannot be read."""
        if name in self._cache:
            return self._cache[name]
        vals: "Optional[np.ndarray]" = None
        try:
            if name in self._recipes:
                full = np.asarray(self._replay(self._recipes[name], self._X, self._apply_recipe), dtype=np.float64).ravel()  # Intentional float64: the retention gates score float64 columns
            else:
                full = np.asarray(self._X[name].to_numpy(), dtype=np.float64).ravel()  # Intentional float64: as above
            if full.shape[0] == self._n_total:
                vals = full[self._rows]
        except Exception as exc:
            logger.debug("compaction linear-value check: cannot read %s (%s)", name, exc)
        self._cache[name] = vals
        return vals

    def _design(self, names: "list[str]") -> "Optional[np.ndarray]":
        """The ``(rows, len(names))`` design of ``names``, or ``None`` when one of the columns cannot be read."""
        cols = [self._column(nm) for nm in names]
        arrays = [c for c in cols if c is not None]
        if not arrays or len(arrays) != len(cols):
            return None
        return np.column_stack(arrays)

    def _loss(self, names: "list[str]") -> float:
        """Held-out CV loss of the design of ``names`` (``nan`` when it cannot be built, ``inf`` when it cannot be fit)."""
        design = self._design(names)
        if design is None:
            return float("nan")
        return cv_loss(design, self._y, self._is_clf, self._n_folds, self._seed)

    def __call__(self, feature: str, rest: "list[str]") -> bool:
        """Whether ``feature`` lowers the held-out linear loss of ``rest`` by more than the margin (``False`` when a column cannot be read)."""
        with_e = self._loss([*rest, feature])
        if not np.isfinite(with_e):
            return False
        if not rest:
            return True
        without = self._loss(rest)
        if np.isnan(without):
            return False
        return bool(not np.isfinite(without) or without > (1.0 + _MARGIN) * with_e)

    def replacement_loses(self, candidate: str, replaced: "list[str]", rest: "list[str]") -> bool:
        """Whether swapping ``replaced`` for ``candidate`` beside ``rest`` raises the held-out linear loss by more than the margin."""
        old_loss = self._loss([*rest, *replaced])
        new_loss = self._loss([*rest, candidate])
        if np.isnan(old_loss) or np.isnan(new_loss) or not np.isfinite(old_loss):
            return False
        return bool(not np.isfinite(new_loss) or new_loss > (1.0 + _MARGIN) * old_loss)


def make_linear_value_check(self: Any, X: Any, recipes_by_name: dict, seed: int = 0, n_folds: int = 3) -> "Optional[LinearValueCheck]":
    """The check for a fitted selector ``self`` and its frame ``X``; ``None`` when no continuous target exists (the compaction then keeps its information-only rule)."""
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
    return LinearValueCheck(X, recipes_by_name, y, rows, is_clf, seed, n_folds, replay_candidate_continuous, apply_recipe)
