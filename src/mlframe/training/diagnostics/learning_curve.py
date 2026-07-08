"""Honest learning curve: holdout score vs increasing train size (OPT-IN).

A learning curve answers "would more data help?". It refits a FRESH estimator on
log-spaced fractions of a single train split and scores each fit on ONE fixed
holdout. The shape is the diagnostic:

  * holdout score still RISING at the largest size  -> data-starved; more data helps.
  * holdout score PLATEAUED at the largest sizes     -> saturated; collect features/model
    capacity instead of more rows.
  * large train-vs-holdout GAP that does not close    -> variance / overfit; the model
    memorises and more data narrows the gap slowly.

Cost is INHERENT and is the reason this is opt-in. There is no incremental
train-size mechanism elsewhere in mlframe; a learning curve is K full refits by
construction (one per size). ``LearningCurveConfig.enabled`` therefore defaults
to ``False`` -- the legitimate, documented cost-gated exception to "cheap
diagnostics default on". The integrator wires the suite call only when an
operator opts in.

Efficiency choices (the EFFICIENCY MANDATE applied):

  * ONE fixed holdout, not nested per-size CV: K fits total, not K*folds.
  * The full train pool is index-sorted ONCE; each size is a prefix slice of that
    one permutation, so the smaller sizes are nested subsets of the larger ones
    (a true learning curve) and no re-shuffle/re-sort happens per size.
  * Sizes run in parallel across cores via joblib (``n_jobs``); each worker holds
    only a column-view + an index slice, never a frame copy.
  * ``time_budget_s`` stops ADDING sizes once the elapsed wall exceeds the budget;
    skipped sizes are LOGGED (never silently dropped) and recorded on the result.
  * ``warm_start`` reuses one estimator and CONTINUES training across the nested
    prefixes (lgb/xgb ``n_estimators`` bump, sklearn ``warm_start=True``) so the K
    fits cost ~one full fit of incremental work instead of K independent fits --
    only taken when the estimator advertises support; otherwise fresh refits.
  * RAM-safe on 100GB-class frames: column views (``.iloc`` / ``.filter`` / slice),
    never ``.copy()`` of the frame; the holdout is split off by index once.

Public surface:
  * :func:`compute_learning_curve` -- the sweep.
  * :class:`LearningCurveResult` -- per-size train/holdout scores (+ std if cheap).
  * :func:`learning_curve_panel` -- a pure-data ``FigureSpec`` (LinePanelSpec).
  * :class:`LearningCurveConfig` -- opt-in (``enabled=False``) integrator config.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from timeit import default_timer as timer
from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Default log-spaced fractions of the train pool (after the holdout is removed). Six points span the
# bias-variance story without the per-size cost ballooning; integer-rounded + de-duplicated downstream.
DEFAULT_SIZES: Tuple[float, ...] = (0.1, 0.2, 0.4, 0.6, 0.8, 1.0)

# A learner is treated as warm-startable only when it exposes one of these incremental controls AND a
# ``warm_start`` flag (or is a known booster with ``n_estimators``). We never assume; we probe attributes.
_WARM_START_N_ATTRS: Tuple[str, ...] = ("n_estimators", "max_iter")


@dataclass(frozen=True)
class LearningCurveResult:
    """Per-size train + holdout scores from a single-holdout learning-curve sweep.

    Arrays are parallel and ascending in ``train_sizes`` (absolute row counts actually fit). ``*_score_std``
    is the std across inner repeats when cheap repeats were requested, else all-zeros. ``skipped_fractions``
    lists the requested fractions a ``time_budget_s`` cut before they ran (empty when nothing was skipped).
    """

    train_sizes: np.ndarray  # absolute row counts (ascending)
    train_scores: np.ndarray  # score on the fit subset itself
    holdout_scores: np.ndarray  # score on the ONE fixed holdout
    train_score_std: np.ndarray
    holdout_score_std: np.ndarray
    holdout_n: int  # rows in the fixed holdout
    scorer_name: str = "score"
    higher_is_better: bool = True
    warm_start_used: bool = False
    skipped_fractions: Tuple[float, ...] = ()
    elapsed_seconds: float = 0.0

    def holdout_slope_last(self, k: int = 3) -> float:
        """Least-squares slope of the holdout score over the last ``k`` sizes (x = log10 size).

        Sign is the verdict: > 0 means the holdout is still improving with more data (data-starved); ~0 / < 0
        means it has plateaued (saturated). log10(size) on the x-axis matches the log-spaced sweep so a constant
        relative gain in rows reads as a constant slope. Returns 0.0 when fewer than 2 usable points remain.
        """
        n = len(self.holdout_scores)
        if n < 2:
            return 0.0
        k = max(2, min(int(k), n))
        x = np.log10(self.train_sizes[-k:].astype(np.float64))
        y = self.holdout_scores[-k:].astype(np.float64)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 2:
            return 0.0
        x, y = x[ok], y[ok]
        # Orient so "rising" always means "more data still helps", regardless of metric direction.
        slope = float(np.polyfit(x, y, 1)[0])
        return slope if self.higher_is_better else -slope

    def verdict(self, plateau_tol: float = 1e-3) -> str:
        """``"data_starved"`` when the last-sizes slope clears ``plateau_tol``, else ``"saturated"``."""
        return "data_starved" if self.holdout_slope_last() > plateau_tol else "saturated"


@dataclass(frozen=True)
class LearningCurveConfig:
    """Opt-in config for the learning-curve diagnostic (default OFF: K refits is expensive).

    ``enabled`` defaults to ``False`` because a learning curve is, by construction, K full model fits -- there is
    no cheaper way to measure score-vs-train-size. This is the documented cost-gated exception to the project's
    "cheap diagnostics default on" rule; the integrator turns it on only when an operator asks. The remaining
    fields tune the sweep cost: fewer ``sizes`` and a ``time_budget_s`` bound the wall, ``n_jobs`` parallelises
    across sizes, ``warm_start`` continues training across nested prefixes for incremental learners.
    """

    enabled: bool = False
    sizes: Tuple[float, ...] = DEFAULT_SIZES
    holdout: float = 0.2
    n_jobs: int = -1
    warm_start: bool = False
    time_budget_s: Optional[float] = None
    random_state: int = 0
    score_repeats: int = 1  # >1 re-fits each size on reshuffled prefixes to get a cheap std band


def _n_rows(X: Any) -> int:
    """Row count for pandas / polars / ndarray without materialising anything."""
    if hasattr(X, "shape") and X.shape is not None:
        return int(X.shape[0])
    return len(X)


def _take_rows(X: Any, idx: np.ndarray) -> Any:
    """Row-subset view by integer positions; format-native, never a whole-frame copy.

    pandas -> ``.iloc`` (view-ish gather), polars -> positional ``gather`` on the lazy/eager frame, ndarray ->
    fancy index. None of these copy columns the caller did not ask for, so a 100GB frame stays a column-view slice.
    """
    if hasattr(X, "iloc"):
        return X.iloc[idx]
    if hasattr(X, "gather") and not isinstance(X, np.ndarray):  # polars DataFrame
        return X[idx]
    return np.asarray(X)[idx]


def _resolve_sizes(sizes: Optional[Sequence[float]], n_pool: int) -> Tuple[np.ndarray, Tuple[float, ...]]:
    """Map requested fractions to ascending, de-duplicated absolute row counts (>=2 rows each).

    Returns ``(abs_counts, frac_for_each_count)`` so a time-budget skip can be reported back in the original
    fraction vocabulary. A fraction that rounds to the same count as a neighbour is collapsed (a tiny pool can't
    distinguish 0.1 from 0.2); the largest size is clamped to the whole pool.
    """
    if sizes is None:
        sizes = DEFAULT_SIZES
    fr = np.asarray([f for f in sizes if 0.0 < f <= 1.0], dtype=np.float64)
    if fr.size == 0:
        raise ValueError("sizes must contain fractions in (0, 1]")
    fr = np.unique(fr)  # ascending
    counts = np.clip(np.round(fr * n_pool).astype(np.int64), 2, n_pool)
    keep_counts: List[int] = []
    keep_fracs: List[float] = []
    seen: set[int] = set()
    for c, f in zip(counts.tolist(), fr.tolist()):
        if c not in seen:
            seen.add(c)
            keep_counts.append(c)
            keep_fracs.append(f)
    return np.asarray(keep_counts, dtype=np.int64), tuple(keep_fracs)


def _supports_warm_start(est: Any) -> Optional[str]:
    """Return the incremental-count attribute name iff ``est`` can continue training, else None.

    We only claim warm-start when the estimator has a ``warm_start`` flag we can set True AND an incremental
    count attribute (``n_estimators`` for boosters, ``max_iter`` for linear/MLP). Probing attributes (not a
    hardcoded class list) keeps this working for lgb/xgb/sklearn wrappers and any future learner with the same API.
    """
    if not (hasattr(est, "set_params") and hasattr(est, "get_params")):
        return None
    try:
        params = est.get_params()
    except Exception:
        return None
    if "warm_start" not in params:
        return None
    for attr in _WARM_START_N_ATTRS:
        if attr in params:
            return attr
    return None


def _score(scorer: Any, est: Any, X: Any, y: np.ndarray) -> float:
    """Score via an sklearn-style ``scorer(estimator, X, y)``; fall back to ``estimator.score``."""
    try:
        return float(scorer(est, X, y))
    except TypeError:
        return float(est.score(X, y))


def _fit_one_size(
    estimator_factory: Callable[[], Any],
    X_pool: Any,
    y_pool: np.ndarray,
    perm: np.ndarray,
    count: int,
    X_hold: Any,
    y_hold: np.ndarray,
    scorer: Any,
    repeats: int,
    base_seed: int,
) -> Tuple[float, float, float, float]:
    """Fit a FRESH estimator on the first ``count`` rows of the permuted pool; score train-subset + holdout.

    ``repeats > 1`` re-fits on reshuffled prefixes of the same length to produce a cheap std band; repeats==1
    (the default) does a single fit. Returns ``(train_mean, train_std, holdout_mean, holdout_std)``.
    """
    tr_scores: List[float] = []
    ho_scores: List[float] = []
    for r in range(max(1, repeats)):
        if r == 0:
            sel = perm[:count]
        else:
            sub = np.random.default_rng(base_seed + r).permutation(len(perm))[:count]
            sel = perm[sub]
        Xs = _take_rows(X_pool, sel)
        ys = y_pool[sel]
        est = estimator_factory()
        est.fit(Xs, ys)
        tr_scores.append(_score(scorer, est, Xs, ys))
        ho_scores.append(_score(scorer, est, X_hold, y_hold))
    return (
        float(np.mean(tr_scores)), float(np.std(tr_scores)),
        float(np.mean(ho_scores)), float(np.std(ho_scores)),
    )


def _fit_warm_curve(
    estimator_factory: Callable[[], Any],
    X_pool: Any,
    y_pool: np.ndarray,
    perm: np.ndarray,
    counts: np.ndarray,
    n_attr: str,
    X_hold: Any,
    y_hold: np.ndarray,
    scorer: Any,
    deadline: Optional[float],
) -> Tuple[List[float], List[float], List[float], int]:
    """Warm-start path: ONE estimator, nested prefixes, incremental ``n_attr`` bump per size.

    Because the sizes are nested prefixes of one permutation, a warm-startable learner can keep its already-fit
    state and only learn the newly-added rows (and, for boosters, add trees) at each step -- ~one full fit of
    incremental work for the whole curve instead of K independent fits. Runs serially (state is shared) and
    honours ``deadline``; returns the scores for the sizes it managed plus the index of the last size reached.
    """
    est = estimator_factory()
    est.set_params(warm_start=True)
    try:
        base_n = int(est.get_params().get(n_attr) or 0)
    except Exception:
        base_n = 0
    per_step = max(1, base_n // len(counts)) if base_n else 50
    tr: List[float] = []
    ho: List[float] = []
    sizes_done: List[float] = []
    reached = 0
    for i, count in enumerate(counts.tolist()):
        if deadline is not None and timer() > deadline:
            break
        est.set_params(**{n_attr: per_step * (i + 1)})
        sel = perm[:count]
        Xs = _take_rows(X_pool, sel)
        ys = y_pool[sel]
        est.fit(Xs, ys)
        tr.append(_score(scorer, est, Xs, ys))
        ho.append(_score(scorer, est, X_hold, y_hold))
        sizes_done.append(float(count))
        reached = i + 1
    return tr, ho, sizes_done, reached


def compute_learning_curve(
    estimator_factory: Callable[[], Any],
    X: Any,
    y: Any,
    *,
    sizes: Optional[Sequence[float]] = None,
    scorer: Any,
    holdout: float = 0.2,
    n_jobs: int = -1,
    warm_start: bool = False,
    random_state: int = 0,
    time_budget_s: Optional[float] = None,
    score_repeats: int = 1,
    scorer_name: str = "score",
    higher_is_better: bool = True,
) -> LearningCurveResult:
    """Refit a fresh estimator on log-spaced fractions of one train split; score each on ONE fixed holdout.

    Parameters
    ----------
    estimator_factory : callable ``() -> unfitted estimator``
        Called once per fit so every size trains a FRESH model (no leakage of a prior fit's state, except on the
        opt-in ``warm_start`` path which deliberately continues one model across nested prefixes).
    X, y : array / DataFrame, length n
        ``X`` may be pandas / polars / ndarray (row-subset by view, never copied). ``y`` is coerced to ndarray once.
    sizes : fractions in (0, 1], default ``DEFAULT_SIZES`` (~6 log-spaced points)
        Mapped to ascending absolute row counts of the train POOL (after the holdout is removed); duplicates from
        rounding on a tiny pool are collapsed.
    scorer : sklearn-style ``scorer(estimator, X, y) -> float``
        Higher = better is the sklearn convention; set ``higher_is_better=False`` for a raw-loss scorer so the
        slope/verdict orient correctly.
    holdout : float in (0, 1)
        Fraction split off ONCE as the fixed holdout. The remaining rows are the pool the sizes index into; the
        holdout is disjoint from every train subset by construction.
    n_jobs : int
        Parallel workers across sizes (joblib). Ignored on the ``warm_start`` path (shared state -> serial).
    warm_start : bool
        When True AND the estimator supports it, continue ONE model across the nested prefixes instead of K fresh
        refits (much cheaper). Silently falls back to fresh refits when unsupported.
    random_state : int
        Seeds the single pool permutation (and the per-repeat reshuffles).
    time_budget_s : float, optional
        Soft wall budget: once exceeded, no further (larger) sizes are STARTED. Skipped fractions are LOGGED and
        returned on ``result.skipped_fractions`` -- never silently dropped.
    score_repeats : int
        >1 re-fits each size on reshuffled equal-length prefixes for a cheap std band (multiplies cost by repeats).

    Returns
    -------
    LearningCurveResult
        Parallel ascending arrays of train sizes, train scores, holdout scores (+ stds), the holdout row count,
        and the skipped-fraction list. ``.holdout_slope_last()`` / ``.verdict()`` summarise the bias-variance read.
    """
    t0 = timer()
    n = _n_rows(X)
    if n < 4:
        raise ValueError(f"learning curve needs >=4 rows, got {n}")
    if not (0.0 < holdout < 1.0):
        raise ValueError(f"holdout must be in (0, 1), got {holdout}")
    y_arr = np.asarray(y)
    if len(y_arr) != n:
        raise ValueError(f"len(y)={len(y_arr)} != len(X)={n}")

    rng = np.random.default_rng(random_state)
    shuffled = rng.permutation(n)
    n_hold = max(1, round(holdout * n))
    n_hold = min(n_hold, n - 2)  # leave >=2 rows for the pool
    hold_idx = np.sort(shuffled[:n_hold])
    pool_idx = np.sort(shuffled[n_hold:])
    n_pool = len(pool_idx)

    X_hold = _take_rows(X, hold_idx)
    y_hold = y_arr[hold_idx]
    X_pool = _take_rows(X, pool_idx)
    y_pool = y_arr[pool_idx]

    counts, fracs = _resolve_sizes(sizes, n_pool)
    # One permutation of the pool; every size is a prefix of it, so smaller sizes are nested subsets of larger
    # ones (a true learning curve) and we sort/permute exactly once.
    perm = rng.permutation(n_pool)

    deadline = None if time_budget_s is None else (t0 + float(time_budget_s))

    warm_attr = _supports_warm_start(estimator_factory()) if warm_start else None
    if warm_attr is not None:
        tr, ho, sizes_done, reached = _fit_warm_curve(
            estimator_factory, X_pool, y_pool, perm, counts, warm_attr,
            X_hold, y_hold, scorer, deadline,
        )
        train_sizes = np.asarray(sizes_done, dtype=np.int64)
        train_scores = np.asarray(tr, dtype=np.float64)
        holdout_scores = np.asarray(ho, dtype=np.float64)
        train_std = np.zeros_like(train_scores)
        holdout_std = np.zeros_like(holdout_scores)
        skipped = tuple(fracs[reached:])
        if skipped:
            logger.info(
                "compute_learning_curve: time budget %.1fs hit (warm-start); skipped %d size(s) %s",
                float(time_budget_s or 0.0), len(skipped), [round(f, 3) for f in skipped],
            )
        return LearningCurveResult(
            train_sizes=train_sizes, train_scores=train_scores, holdout_scores=holdout_scores,
            train_score_std=train_std, holdout_score_std=holdout_std, holdout_n=n_hold,
            scorer_name=scorer_name, higher_is_better=higher_is_better, warm_start_used=True,
            skipped_fractions=skipped, elapsed_seconds=timer() - t0,
        )

    def _job(c: int) -> Tuple[float, float, float, float]:
        return _fit_one_size(
            estimator_factory, X_pool, y_pool, perm, c, X_hold, y_hold,
            scorer, score_repeats, random_state,
        )

    run_counts: List[int] = []
    skipped_fracs: List[float] = []
    if deadline is not None:
        # Budget path: sizes are ascending, so fit smallest-first and STOP adding once the accumulated wall passes
        # the deadline (the budget bounds total fit-work). The decision to stop is inherently sequential; we still
        # always keep the first size so a too-tight budget yields at least one point.
        results: List[Tuple[float, float, float, float]] = []
        for i, (c, f) in enumerate(zip(counts.tolist(), fracs)):
            if i > 0 and timer() > deadline:
                skipped_fracs.append(f)
                continue
            results.append(_job(c))
            run_counts.append(c)
        if skipped_fracs:
            logger.info(
                "compute_learning_curve: time budget %.1fs hit; skipped %d size(s) %s",
                float(time_budget_s or 0.0), len(skipped_fracs), [round(f, 3) for f in skipped_fracs],
            )
    else:
        run_counts = counts.tolist()
        if len(run_counts) > 1 and n_jobs != 1:
            from joblib import Parallel, delayed

            results = Parallel(n_jobs=n_jobs, prefer="processes")(delayed(_job)(c) for c in run_counts)
        else:
            results = [_job(c) for c in run_counts]

    train_scores = np.asarray([r[0] for r in results], dtype=np.float64)
    train_std = np.asarray([r[1] for r in results], dtype=np.float64)
    holdout_scores = np.asarray([r[2] for r in results], dtype=np.float64)
    holdout_std = np.asarray([r[3] for r in results], dtype=np.float64)

    return LearningCurveResult(
        train_sizes=np.asarray(run_counts, dtype=np.int64),
        train_scores=train_scores, holdout_scores=holdout_scores,
        train_score_std=train_std, holdout_score_std=holdout_std, holdout_n=n_hold,
        scorer_name=scorer_name, higher_is_better=higher_is_better, warm_start_used=False,
        skipped_fractions=tuple(skipped_fracs), elapsed_seconds=timer() - t0,
    )


def learning_curve_panel(result: LearningCurveResult, *, title: str = "Learning curve") -> "Any":
    """Build a pure-data ``FigureSpec`` (one ``LinePanelSpec``) of train vs holdout score vs train size.

    Two series share the train-size x-axis: the train-subset score and the holdout score, each with a +-1 std
    band when repeats produced one. The gap between the curves and whether the holdout curve is still rising at
    the right edge is the bias-vs-variance / "would-more-data-help" read the panel exists to surface. Returns a
    one-panel ``FigureSpec`` so either backend renders it identically; the integrator drops it into the report grid.
    """
    from mlframe.reporting.spec import FigureSpec, LinePanelSpec

    x = result.train_sizes.astype(np.float64)
    train_y = result.train_scores.astype(np.float64)
    hold_y = result.holdout_scores.astype(np.float64)

    # Band on the holdout series (the headline curve): +-1 std when repeats gave a non-degenerate std, else none.
    band = None
    if np.any(result.holdout_score_std > 0):
        band = (hold_y - result.holdout_score_std, hold_y + result.holdout_score_std)

    verdict = result.verdict()
    subtitle = f"{title} ({result.scorer_name}) -- {verdict}; holdout n={result.holdout_n}" + (
        f"; {len(result.skipped_fractions)} size(s) skipped (budget)" if result.skipped_fractions else ""
    )

    line = LinePanelSpec(
        x=x,
        y=(train_y, hold_y),
        series_labels=("train score", "holdout score"),
        line_styles=("--", "lines+markers"),
        title=subtitle,
        xlabel="train size (rows)",
        ylabel=result.scorer_name,
        band=band,
        band_label="holdout +-1 std" if band is not None else None,
    )
    return FigureSpec(suptitle="", panels=((line,),), figsize=(7.0, 5.0))


__all__ = [
    "DEFAULT_SIZES",
    "LearningCurveConfig",
    "LearningCurveResult",
    "compute_learning_curve",
    "learning_curve_panel",
]
