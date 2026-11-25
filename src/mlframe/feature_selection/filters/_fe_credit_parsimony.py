"""Parsimony of the output once a unary credit has fired (goal 4: 2 signal features give 2 to 3 columns, not 8 to 12).

When the linear-usability credit (``_fe_unary_credit``) finds the exact form of a column (``sin(x1)``, ``log(x1)``), every other engineered column that only re-expresses what the output already holds is noise for the
reader: basis legs of the same source (``fourier3``, ``T3``, hinges, ``round``), single-index directions, products with noise columns that survived the MI screens by chance, ``rff`` / ``mahal`` summaries of all
columns, and the raw source the form replaces. This pass removes them by backward elimination on a design of the final output columns, one column at a time, cheapest first. A column goes only when BOTH out-of-fold
legs agree that it is not needed (its removal costs less than ``R2_DROP`` of the variance of ``y``, an effect size, not a significance: with enough rows every column has a significant sliver):

* LINEAR leg: a ridge head on the columns themselves (what a linear model sees);
* SMOOTH leg: a ridge head on the column itself plus a hat-function smooth of its rank, per column (what an additive or tree model sees: any monotone or non-monotone shape per column, but no interaction between columns).

An earlier version asked the debiased conditional-MI test of the retention guard for the second leg; at 3600 rows it called ``rff`` and ``mahal`` "not subsumed" on a fixture where at 6000 rows it called them subsumed
(binned CMI given three conditioning columns is a sparse table). The out-of-fold smooth is stable across both sizes.

Protected: the credited forms themselves and every raw column that is not the source of a single-source engineered column. The pass runs ONLY when a credit fired in this fit (the evidence that a simple exact form explains structure
the families approximated), and only for a regression target; with no credit the selection is byte-identical. Rollback: ``MLFRAME_FE_CREDIT_PARSIMONY=0``."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from mlframe.utils.env_flags import FALSE_VALUES, env_reader
from mlframe.utils.log_throttle import log_throttle

logger = logging.getLogger(__name__)

#: A column is removable when its removal costs less than this share of the variance of ``y`` in out-of-fold R^2, on both legs.
R2_DROP = 0.003
#: Rows of the design (a seeded draw above it), ridge folds and strength (per row, relative to the column count), hat functions per column of the smooth leg.
MAX_ROWS = 6000
FOLDS = 5
RIDGE = 1e-4
HATS = 8

_enabled_reader = env_reader("MLFRAME_FE_CREDIT_PARSIMONY")


def credit_parsimony_enabled() -> bool:
    """On unless ``MLFRAME_FE_CREDIT_PARSIMONY`` is set to a false spelling."""
    value = _enabled_reader()
    return value is None or value.strip().lower() not in FALSE_VALUES - {""}


def _standardise(matrix: np.ndarray) -> np.ndarray:
    """Columns scaled to zero mean and unit variance, clipped at 8 standard deviations (a heavy-tailed engineered column must not own the ridge)."""
    matrix = matrix - matrix.mean(axis=0)
    sd = matrix.std(axis=0)
    return np.asarray(np.clip(matrix / np.where(sd > 0, sd, 1.0), -8.0, 8.0))


def oof_squared_error(design: np.ndarray, y: np.ndarray, folds: np.ndarray) -> np.ndarray:
    """Per-row out-of-fold squared error of a ridge fit of ``y`` on the intercept and the (standardised) ``design`` columns."""
    n, k = design.shape
    full = np.column_stack([np.ones(n), design])
    gram = full.T @ full
    moment = full.T @ y
    penalty = RIDGE * n * np.eye(k + 1)
    penalty[0, 0] = 0.0
    squared = np.empty(n)
    for fold in range(FOLDS):
        test = folds == fold
        held = full[test]
        weights = np.linalg.solve(gram - held.T @ held + penalty, moment - held.T @ y[test])
        squared[test] = (y[test] - held @ weights) ** 2
    return squared


def rank_hats(column: np.ndarray, k: int = HATS) -> np.ndarray:
    """``(n, k)`` piecewise-linear hat basis of the rank of ``column`` on equally spaced knots of [0, 1]: any shape of the column, invariant to a monotone rescaling."""
    order = np.empty(column.size)
    order[np.argsort(column, kind="stable")] = np.arange(column.size)
    u = (order + 0.5) / column.size
    knots = np.linspace(0.0, 1.0, k)
    return np.asarray(np.maximum(0.0, 1.0 - np.abs(u[:, None] - knots[None, :]) / (knots[1] - knots[0])))


def eliminate(columns: np.ndarray, y: np.ndarray, removable: "list[int]", *, seed: int = 0) -> "list[int]":
    """Column indices of ``columns`` to drop, by backward elimination over the ``removable`` ones (see the module docstring for the two legs).

    One removal per round, the one that costs least; the design shrinks after every removal so a later column is judged against what is left."""
    n = columns.shape[0]
    folds = np.random.default_rng(seed).permutation(n) % FOLDS
    design = _standardise(columns)
    # the linear term of the column rides in its block, so the coarse hats are not asked to rebuild a straight line (without it a second column that merely re-orders the same variable looked useful by refining the hats)
    smooth = [np.hstack([design[:, [j]], rank_hats(columns[:, j])]) for j in range(columns.shape[1])]
    variance = max(float(np.var(y)), 1e-300)
    alive = list(range(columns.shape[1]))
    dropped: "list[int]" = []
    while len(alive) > 1:
        base_lin = oof_squared_error(design[:, alive], y, folds).mean()
        base_smooth = oof_squared_error(np.hstack([smooth[j] for j in alive]), y, folds).mean()
        best = None
        for j in removable:
            if j not in alive:
                continue
            rest = [a for a in alive if a != j]
            loss_lin = (oof_squared_error(design[:, rest], y, folds).mean() - base_lin) / variance
            loss_smooth = (oof_squared_error(np.hstack([smooth[a] for a in rest]), y, folds).mean() - base_smooth) / variance
            if loss_lin < R2_DROP and loss_smooth < R2_DROP and (best is None or loss_lin + loss_smooth < best[0]):
                best = (loss_lin + loss_smooth, j)
        if best is None:
            break
        alive.remove(best[1])
        dropped.append(best[1])
    return dropped


def run_credit_parsimony(self: Any, X: Any, selected_vars: list, cols: Any, classes_y: Any, y: Any, verbose: int = 0) -> "tuple[bool, list]":
    """Remove the columns the credited forms make redundant; ``(changed, selected_vars)``. A no-op unless a credit fired in this fit and the target is continuous."""
    import pandas as pd

    from ._fe_retention_subsumption import replay_candidate_continuous
    from .engineered_recipes.shared import apply_recipe

    if not credit_parsimony_enabled() or not getattr(self, "_unary_credit_fired_", False) or not isinstance(X, pd.DataFrame):
        return False, selected_vars
    try:
        y_cont = getattr(self, "_fe_prewarp_y_continuous_", None)
        if y_cont is None:
            y_cont = np.asarray(y.values if hasattr(y, "values") else y).reshape(-1)
        y_cont = np.asarray(y_cont)
        if y_cont.shape[0] != len(X) or not np.issubdtype(y_cont.dtype, np.number) or np.unique(y_cont[:5000]).size < 20:
            return False, selected_vars
        raw_names = [str(cols[int(v)]) for v in selected_vars if 0 <= int(v) < len(cols)]
        recipes = list(self._engineered_recipes_ or [])
        soft = set(getattr(self, "_unary_credit_soft_", None) or ())  # single forms credited only through a product
        # a raw column is a candidate for removal when some engineered column is a function of it alone (a credited form, or the same form an MI-greedy family emitted under its own name): the two legs decide
        credited_sources = {r.src_names[0] for r in recipes if len(r.src_names) == 1}
        columns: list[Any] = []
        labels: list[Any] = []
        removable: list[Any] = []
        for nm in raw_names:
            if nm in X.columns and pd.api.types.is_numeric_dtype(X[nm].dtype):
                if nm in credited_sources:
                    removable.append(len(columns))
                columns.append(np.asarray(X[nm].to_numpy(), dtype=np.float64))
                labels.append(("raw", nm))
        for recipe in recipes:
            try:
                values = np.asarray(replay_candidate_continuous(recipe, X, apply_recipe), dtype=np.float64).ravel()
            except Exception as exc:  # an unreplayable column stays, untouched
                log_throttle(
                    logger, "credit_parsimony.replay", logging.WARNING, "credit parsimony: %r not replayed (%r); kept", getattr(recipe, "name", "?"), exc
                )
                continue
            if values.shape[0] != len(X) or not np.isfinite(values).all():
                continue
            if recipe.kind != "unary_credit" or str(recipe.name) in soft:
                removable.append(len(columns))
            columns.append(values)
            labels.append(("eng", str(recipe.name)))
        if len(columns) < 3 or not removable:
            return False, selected_vars
        n = len(X)
        rng = np.random.default_rng(int(getattr(self, "random_seed", 0) or 0))
        rows = np.sort(rng.choice(n, MAX_ROWS, replace=False)) if n > MAX_ROWS else np.arange(n)
        target = y_cont[rows].astype(np.float64)
        logger.debug("credit parsimony design: %s; removable: %s", [lab[1] for lab in labels], [labels[j][1] for j in removable])
        dropped = eliminate(np.column_stack([c[rows] for c in columns]), target, removable, seed=int(getattr(self, "random_seed", 0) or 0))
    except Exception as exc:  # never let the optional pass break a fit
        log_throttle(logger, "credit_parsimony.failed", logging.WARNING, "credit parsimony skipped (%s: %s)", type(exc).__name__, exc)
        return False, selected_vars
    if not dropped:
        return False, selected_vars
    drop_raw = {labels[j][1] for j in dropped if labels[j][0] == "raw"}
    drop_eng = {labels[j][1] for j in dropped if labels[j][0] == "eng"}
    if verbose:
        logger.info("MRMR credit parsimony: dropped %d column(s) the credited form(s) make redundant: %s", len(dropped), sorted(drop_raw | drop_eng))
    keep_pos = [i for i, r in enumerate(self._engineered_recipes_) if str(r.name) not in drop_eng]
    self._engineered_recipes_ = [self._engineered_recipes_[i] for i in keep_pos]
    self._engineered_features_ = [f for f in self._engineered_features_ if str(f) not in drop_eng]
    new_selected = [v for v in selected_vars if str(cols[int(v)]) not in drop_raw]
    self.support_ = np.array(new_selected, dtype=np.int64)
    self.n_features_ = len(new_selected) + len(self._engineered_recipes_)
    return True, new_selected
