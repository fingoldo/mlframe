"""Joint hyperparameter optimization for a composite target.

``optimize_composite`` searches JOINTLY over (transform choice, inner-estimator
hyperparameters) to minimise leakage-free CV (or purged-CV) out-of-sample error,
returning the best ``(transform_name, inner_params, selection_score)`` plus a
:class:`CompositeTargetEstimator` re-fitted on ALL rows with the winning config.

Why joint search: the best transform and the best inner depth are coupled -- a
log-residual base may favour a shallow tree while a raw-diff base needs depth to
recover curvature. Optimising the two independently (pick transform by MI, then
tune the inner) misses the interaction and ships a sub-optimal pair. Each trial's
CV error is leakage-free per FOLD (transform applied on the fit fold, inverted on
the score fold), so the winner is the pair that predicts ``y`` best out of sample.

IMPORTANT -- ``selection_score`` is a SELECTION score, NOT an unbiased OOS estimate.
It is the CV error of the WINNING trial, i.e. the minimum over ``n_trials`` noisy
CV estimates. Taking the min (argmax of goodness) over many noisy estimates is
optimistically biased downward ("winner's curse" / selection bias): the reported
number is systematically better than the true generalisation error of the chosen
config. There is NO nested CV here to de-bias it. To report an HONEST OOS number,
re-estimate the returned ``estimator`` on a DISJOINT holdout the search never saw
(or wrap ``optimize_composite`` in an outer CV loop for full nested CV).

Backend: uses Optuna when importable (TPE over a mixed categorical/numeric space);
falls back to a self-contained random search when Optuna is absent -- never a hard
dependency. Both honour ``n_trials`` exactly and use the SAME objective, so results
are comparable and the fallback is a drop-in.

Leakage safety: every trial scores via K-fold (or a caller-supplied splitter); the
inner estimator and the transform parameters are fit on the train fold ONLY and the
transform is inverted to original ``y`` scale before scoring the held-out fold.
When ``time_ordering`` is given the default splitter becomes
:class:`PurgedTimeSeriesSplit` so overlapping-label / autocorrelated series are
scored honestly. Frames are never copied (see CLAUDE.md memory rules): folds are
taken by frame-native ``.iloc`` / ``.filter`` on integer indices.
"""

from __future__ import annotations

import itertools
import logging
import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from sklearn.base import clone

from .cv import PurgedTimeSeriesSplit
from .estimator import CompositeTargetEstimator
from ..utils import coerce_to_1d_numpy as _to_1d_numpy

logger = logging.getLogger(__name__)

__all__ = ["optimize_composite", "CompositeHPOResult", "HPOSpace"]


# ----------------------------------------------------------------------
# Search-space description
# ----------------------------------------------------------------------

@dataclass
class HPOSpace:
    """A single inner hyperparameter's search range.

    ``kind`` is one of ``"int"`` / ``"float"`` / ``"categorical"``. ``low`` /
    ``high`` bound numeric params (``log=True`` samples geometrically -- right
    for learning-rate / regularisation scales); ``choices`` lists categorical
    options. Mirrors the minimal subset of the Optuna ``suggest_*`` API so the
    same :class:`HPOSpace` drives both backends.
    """

    kind: str
    low: float = 0.0
    high: float = 1.0
    log: bool = False
    choices: Tuple[Any, ...] = ()

    def sample(self, rng: random.Random) -> Any:
        """Draw one value with the fallback (random-search) backend."""
        if self.kind == "categorical":
            return rng.choice(self.choices)
        if self.kind == "int":
            if self.log:
                lo, hi = math.log(max(self.low, 1e-12)), math.log(self.high)
                return int(round(math.exp(rng.uniform(lo, hi))))
            return rng.randint(int(self.low), int(self.high))
        if self.kind == "float":
            if self.log:
                lo, hi = math.log(max(self.low, 1e-12)), math.log(self.high)
                return math.exp(rng.uniform(lo, hi))
            return rng.uniform(self.low, self.high)
        raise ValueError(f"HPOSpace: unknown kind {self.kind!r}")

    def suggest(self, trial: Any, name: str) -> Any:
        """Draw one value with the Optuna backend (TPE-aware)."""
        if self.kind == "categorical":
            return trial.suggest_categorical(name, list(self.choices))
        if self.kind == "int":
            return trial.suggest_int(name, int(self.low), int(self.high), log=self.log)
        if self.kind == "float":
            return trial.suggest_float(name, self.low, self.high, log=self.log)
        raise ValueError(f"HPOSpace: unknown kind {self.kind!r}")

    def enumerate_values(self) -> Optional[List[Any]]:
        """Finite value list for grid enumeration, or None when continuous.

        A ``categorical`` yields its choices; a non-log ``int`` yields the full
        inclusive integer range. ``float`` and log-scaled ``int`` are continuous
        (no exhaustive grid) and return None, forcing the random fallback.
        """
        if self.kind == "categorical":
            return list(self.choices)
        if self.kind == "int" and not self.log:
            return list(range(int(self.low), int(self.high) + 1))
        return None


@dataclass
class CompositeHPOResult:
    """Outcome of :func:`optimize_composite`.

    ``estimator`` is a :class:`CompositeTargetEstimator` already fitted on all
    rows with the winning ``(transform, inner_params)`` pair, ready to predict.
    ``selection_score`` is the CV error of the WINNING trial (lower is better).
    It is a SELECTION score, NOT an unbiased OOS estimate: it is the minimum over
    ``n_trials`` noisy CV estimates and is therefore optimistically biased
    (winner's curse). For an honest generalisation number, re-score ``estimator``
    on a disjoint holdout the search never saw. ``trials`` records every evaluated
    ``(transform, params, score)`` for audit.

    ``cv_score`` is retained as a read-only backward-compatible alias of
    ``selection_score`` (older callers); prefer ``selection_score`` in new code so
    the optimistic-bias caveat is visible at the call site.
    """

    transform: str
    inner_params: Dict[str, Any]
    selection_score: float
    estimator: CompositeTargetEstimator
    backend: str
    n_trials: int
    trials: List[Tuple[str, Dict[str, Any], float]] = field(default_factory=list)

    @property
    def cv_score(self) -> float:
        """Deprecated alias of :attr:`selection_score` (a selection score, not an
        unbiased OOS estimate). Kept so existing ``result.cv_score`` callers keep
        working; new code should read ``selection_score`` and honor its caveat."""
        return self.selection_score


# ----------------------------------------------------------------------
# Internals
# ----------------------------------------------------------------------

def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Default scorer: root-mean-squared error on original ``y`` scale.

    Non-finite predictions (an inverse-transform blow-up on a pathological
    fold) are penalised with ``inf`` so the optimizer steers away from the
    config rather than crashing the whole search.
    """
    diff = np.asarray(y_true, dtype=np.float64) - np.asarray(y_pred, dtype=np.float64)
    if not np.all(np.isfinite(diff)):
        return float("inf")
    return float(np.sqrt(np.mean(diff * diff)))


def _iloc_rows(X: Any, idx: np.ndarray) -> Any:
    """Frame-native row subset by integer positions -- NO whole-frame copy.

    Polars: ``X[idx]`` (gather on row index). Pandas: ``X.iloc[idx]``. Both
    return a view/new-frame of only the selected rows, never a clone of the
    full carrier (CLAUDE.md 100 GB rule)."""
    if hasattr(X, "iloc"):
        return X.iloc[idx]
    # polars DataFrame supports positional gather via __getitem__ on an int array.
    return X[idx]


def _default_inner_spaces(inner: Any) -> Dict[str, HPOSpace]:
    """Best-effort default search space for a tree/boosting inner estimator.

    Probes the estimator's params for the common depth / n_estimators /
    learning_rate / regularisation knobs and builds a small space for the ones
    it exposes. Callers wanting full control pass ``inner_spaces`` explicitly.
    """
    try:
        params = inner.get_params()
    except Exception:  # pragma: no cover - non-sklearn inner
        return {}
    spaces: Dict[str, HPOSpace] = {}
    if "max_depth" in params:
        spaces["max_depth"] = HPOSpace("int", low=1, high=8)
    if "n_estimators" in params:
        spaces["n_estimators"] = HPOSpace("int", low=50, high=300)
    if "learning_rate" in params:
        spaces["learning_rate"] = HPOSpace("float", low=1e-3, high=0.5, log=True)
    if "min_samples_leaf" in params:
        spaces["min_samples_leaf"] = HPOSpace("int", low=1, high=32)
    return spaces


def _build_estimator(
    inner_factory: Callable[[], Any],
    base_column: str,
    transform_name: str,
    inner_params: Dict[str, Any],
) -> CompositeTargetEstimator:
    """Construct a fresh composite for one (transform, params) candidate."""
    inner = inner_factory()
    if inner_params:
        inner = clone(inner)
        inner.set_params(**inner_params)
    return CompositeTargetEstimator(
        base_estimator=inner,
        transform_name=transform_name,
        base_column=base_column,
    )


def _cv_score_candidate(
    X: Any,
    y: np.ndarray,
    *,
    base_column: str,
    transform_name: str,
    inner_params: Dict[str, Any],
    inner_factory: Callable[[], Any],
    splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    scorer: Callable[[np.ndarray, np.ndarray], float],
) -> float:
    """Mean leakage-free CV OOS error for one candidate.

    For each fold the composite (transform params + inner estimator) is fit on
    the train rows ONLY and scored on the held-out rows in original ``y`` scale.
    A fold that raises (degenerate domain, singular inner) contributes ``inf``
    so the candidate is penalised, never aborting the whole search.
    """
    fold_scores: List[float] = []
    for train_idx, test_idx in splits:
        if train_idx.size == 0 or test_idx.size == 0:
            fold_scores.append(float("inf"))
            continue
        est = _build_estimator(inner_factory, base_column, transform_name, inner_params)
        X_tr, X_te = _iloc_rows(X, train_idx), _iloc_rows(X, test_idx)
        y_tr, y_te = y[train_idx], y[test_idx]
        try:
            est.fit(X_tr, y_tr)
            pred = np.asarray(est.predict(X_te), dtype=np.float64)
            fold_scores.append(float(scorer(y_te, pred)))
        except Exception as err:  # noqa: BLE001 -- penalise, don't crash search
            logger.debug("optimize_composite: candidate %r/%r fold failed: %r",
                         transform_name, inner_params, err)
            fold_scores.append(float("inf"))
    if not fold_scores:
        return float("inf")
    return float(np.mean(fold_scores))


def _resolve_splits(
    X: Any,
    n_rows: int,
    cv: Any,
    time_ordering: Optional[np.ndarray],
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Materialise the list of (train_idx, test_idx) folds ONCE.

    Precedence: an explicit ``cv`` splitter object (anything with ``.split``)
    wins; else when ``time_ordering`` is given a :class:`PurgedTimeSeriesSplit`
    on the time-sorted row order is used (honest temporal OOS); else a plain
    integer ``cv`` (or default 5) drives a contiguous KFold. Folds are computed
    once and reused across every trial so all candidates see identical splits.

    Caveat: the default integer-``cv`` KFold is CONTIGUOUS with NO shuffle -- each
    fold is a solid block of consecutive rows. For ordered-but-non-temporal data
    (rows sorted by an id, a category, or any non-random key that is NOT a time
    axis) this concentrates whole regions in single folds and gives a biased CV
    estimate; pass an explicit shuffled ``cv`` (e.g. ``KFold(shuffle=True)``) for
    such data, or a ``time_ordering`` when the order really is temporal.
    """
    if cv is not None and hasattr(cv, "split"):
        return [(np.asarray(tr), np.asarray(te)) for tr, te in cv.split(X)]

    n_splits = int(cv) if isinstance(cv, (int, np.integer)) else 5
    if n_splits < 2:
        n_splits = 2

    if time_ordering is not None:
        order = np.argsort(np.asarray(time_ordering), kind="stable")
        splitter = PurgedTimeSeriesSplit(n_splits=n_splits)
        out: List[Tuple[np.ndarray, np.ndarray]] = []
        for tr, te in splitter.split(np.empty(n_rows)):
            out.append((order[tr], order[te]))
        return out

    # Plain KFold on contiguous blocks (no shuffle -> deterministic, cheap).
    fold_sizes = np.full(n_splits, n_rows // n_splits, dtype=int)
    fold_sizes[: n_rows % n_splits] += 1
    bounds = np.concatenate([[0], np.cumsum(fold_sizes)])
    all_idx = np.arange(n_rows)
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for i in range(n_splits):
        te = all_idx[bounds[i]: bounds[i + 1]]
        tr = np.concatenate([all_idx[: bounds[i]], all_idx[bounds[i + 1]:]])
        splits.append((tr, te))
    return splits


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------

def optimize_composite(
    X: Any,
    y: Any,
    *,
    base_column: str,
    transform_candidates: Sequence[str],
    inner_factory: Callable[[], Any],
    n_trials: int = 30,
    cv: Any = 5,
    scorer: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    inner_spaces: Optional[Dict[str, HPOSpace]] = None,
    time_ordering: Optional[Any] = None,
    random_state: int = 0,
    prefer_optuna: bool = True,
) -> CompositeHPOResult:
    """Jointly optimise (transform, inner hyperparameters) for a composite.

    Parameters
    ----------
    X, y
        Feature frame (pandas / polars, NOT copied) and 1-D target. ``X`` MUST
        contain ``base_column``.
    base_column
        Column the transform residualises against (passed to the composite).
    transform_candidates
        Transform names to search over (e.g. ``("diff", "log_residual",
        "ratio")``); the optimizer picks the one that minimises CV OOS error.
    inner_factory
        Zero-arg callable returning a FRESH inner estimator. Called once per
        candidate so trials never share fitted state.
    n_trials
        Exact number of (transform, params) candidates evaluated. Honoured by
        BOTH backends.
    cv
        Either an int (number of KFold folds) or any sklearn-style splitter with
        ``.split(X)``. When ``time_ordering`` is given and ``cv`` is an int, a
        purged time-series splitter is used instead.
    scorer
        ``scorer(y_true, y_pred) -> float`` where LOWER is better. Default RMSE.
    inner_spaces
        Mapping of inner-param name -> :class:`HPOSpace`. ``None`` auto-derives a
        small space from the inner estimator's params (depth / n_estimators /
        learning_rate / min_samples_leaf where present).
    time_ordering
        Optional per-row timestamp / sort key; triggers purged-CV scoring.
    random_state
        Seed for the random-search fallback and the Optuna sampler.
    prefer_optuna
        When True (default) use Optuna if importable; set False to force the
        random-search fallback (used by the fallback unit test).

    Returns
    -------
    CompositeHPOResult
        Winning transform + inner params + ``selection_score`` (the winning
        trial's CV error -- a SELECTION score, optimistically biased by the
        min-over-trials selection; re-score on a disjoint holdout for an honest
        OOS number) + an all-rows-fitted composite estimator + the full trial log.
    """
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}")
    if not transform_candidates:
        raise ValueError("transform_candidates is empty")

    y_arr = _to_1d_numpy(y)
    n_rows = y_arr.shape[0]
    scorer = scorer or _rmse

    if inner_spaces is None:
        inner_spaces = _default_inner_spaces(inner_factory())

    splits = _resolve_splits(X, n_rows, cv,
                             None if time_ordering is None else np.asarray(time_ordering))

    trials_log: List[Tuple[str, Dict[str, Any], float]] = []

    def _evaluate(transform_name: str, inner_params: Dict[str, Any]) -> float:
        score = _cv_score_candidate(
            X, y_arr,
            base_column=base_column,
            transform_name=transform_name,
            inner_params=inner_params,
            inner_factory=inner_factory,
            splits=splits,
            scorer=scorer,
        )
        trials_log.append((transform_name, dict(inner_params), score))
        return score

    backend, best_transform, best_params, best_score = _run_search(
        transform_candidates=transform_candidates,
        inner_spaces=inner_spaces,
        n_trials=n_trials,
        random_state=random_state,
        prefer_optuna=prefer_optuna,
        evaluate=_evaluate,
    )

    # Re-fit the winner on ALL rows for the returned estimator.
    final_est = _build_estimator(inner_factory, base_column, best_transform, best_params)
    final_est.fit(X, y_arr)

    return CompositeHPOResult(
        transform=best_transform,
        inner_params=best_params,
        selection_score=best_score,
        estimator=final_est,
        backend=backend,
        n_trials=n_trials,
        trials=trials_log,
    )


def _run_search(
    *,
    transform_candidates: Sequence[str],
    inner_spaces: Dict[str, HPOSpace],
    n_trials: int,
    random_state: int,
    prefer_optuna: bool,
    evaluate: Callable[[str, Dict[str, Any]], float],
) -> Tuple[str, str, Dict[str, Any], float]:
    """Drive the chosen backend; return (backend, transform, params, score).

    Both backends call the SAME ``evaluate`` objective so the fallback is a
    drop-in for Optuna and the two are directly comparable.
    """
    optuna = None
    if prefer_optuna:
        try:
            import optuna as _optuna  # type: ignore
            optuna = _optuna
        except ImportError:
            optuna = None

    if optuna is not None:
        return _search_optuna(optuna, transform_candidates, inner_spaces,
                              n_trials, random_state, evaluate)
    return _search_random(transform_candidates, inner_spaces,
                          n_trials, random_state, evaluate)


def _search_optuna(
    optuna: Any,
    transform_candidates: Sequence[str],
    inner_spaces: Dict[str, HPOSpace],
    n_trials: int,
    random_state: int,
    evaluate: Callable[[str, Dict[str, Any]], float],
) -> Tuple[str, str, Dict[str, Any], float]:
    """TPE search over the joint (transform, inner) space via Optuna."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def _objective(trial: Any) -> float:
        transform_name = trial.suggest_categorical("transform", list(transform_candidates))
        params = {name: sp.suggest(trial, name) for name, sp in inner_spaces.items()}
        return evaluate(transform_name, params)

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(_objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_trial
    best_transform = best.params["transform"]
    best_params = {name: best.params[name] for name in inner_spaces if name in best.params}
    return "optuna", best_transform, best_params, float(best.value)


def _search_random(
    transform_candidates: Sequence[str],
    inner_spaces: Dict[str, HPOSpace],
    n_trials: int,
    random_state: int,
    evaluate: Callable[[str, Dict[str, Any]], float],
) -> Tuple[str, str, Dict[str, Any], float]:
    """Self-contained search -- the no-Optuna fallback.

    When the full joint (transform x inner-params) grid is finite and fits the
    ``n_trials`` budget, every grid point is evaluated exactly once (any leftover
    budget is spent on random draws), so the search cannot return worse than any
    config in its own space. When the grid is continuous or larger than the
    budget, it falls back to ``n_trials`` seeded random draws. Either way it runs
    exactly ``n_trials`` evaluations, a faithful drop-in for the Optuna path.
    """
    rng = random.Random(random_state)
    best_score = float("inf")
    best_transform = transform_candidates[0]
    best_params: Dict[str, Any] = {}

    grid = _enumerate_grid(transform_candidates, inner_spaces)
    evaluated = 0
    if grid is not None and len(grid) <= n_trials:
        for transform_name, params in grid:
            score = evaluate(transform_name, params)
            evaluated += 1
            if score < best_score:
                best_score, best_transform, best_params = score, transform_name, dict(params)

    for _ in range(n_trials - evaluated):
        transform_name = rng.choice(list(transform_candidates))
        params = {name: sp.sample(rng) for name, sp in inner_spaces.items()}
        score = evaluate(transform_name, params)
        if score < best_score:
            best_score, best_transform, best_params = score, transform_name, dict(params)
    return "random", best_transform, best_params, best_score


def _enumerate_grid(
    transform_candidates: Sequence[str],
    inner_spaces: Dict[str, HPOSpace],
) -> Optional[List[Tuple[str, Dict[str, Any]]]]:
    """Full (transform, inner-params) grid, or None if any space is continuous."""
    per_space: List[List[Any]] = []
    names = list(inner_spaces)
    for name in names:
        values = inner_spaces[name].enumerate_values()
        if values is None:
            return None
        per_space.append(values)
    grid: List[Tuple[str, Dict[str, Any]]] = []
    for transform_name in transform_candidates:
        for combo in itertools.product(*per_space) if per_space else [()]:
            grid.append((transform_name, dict(zip(names, combo))))
    return grid
