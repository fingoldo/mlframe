"""Helper functions for the RFECV wrapper module."""
from __future__ import annotations

import logging
import random
import warnings
from typing import Any, Union

import numpy as np
import pandas as pd
import polars as pl

from ._enums import OptimumSearch, VotesAggregation

logger = logging.getLogger(__name__)


# Class-name fragments for estimators that already use native multi-threading.
# Parallelising CV folds on top of these over-subscribes cores: one outer
# joblib worker per fold * N inner threads per fit = N x core_count > cores.
_MULTITHREADED_ESTIMATOR_PATTERNS = (
    "CatBoost", "LGBM", "LightGBM", "XGB", "XGBoost",
    "RandomForest", "ExtraTrees", "GradientBoosting", "HistGradientBoosting",
)

# Thread-count constructor params recognised on common multi-threaded estimators.
# Used only when force_parallel=True to pin each fold's clone to single thread.
_THREAD_PARAMS = ("thread_count", "n_jobs", "n_threads", "nthread", "num_threads")


def _detect_multithreaded(estimator: object) -> bool:
    """True if the estimator's class name matches a known multi-threaded pattern."""
    name = type(estimator).__name__
    return any(p in name for p in _MULTITHREADED_ESTIMATOR_PATTERNS)


def _pin_threads_to_one(estimator: object) -> None:
    """Best-effort: set every known thread-count param to 1 on ``estimator``.

    E14 (Wave 4, 2026-05-28; reverted 2026-05-28): an earlier version also
    SET ``os.environ['OMP_NUM_THREADS']='1'`` here, but that leaked process-
    global single-threading into subsequent tests / fits that DID want
    multi-thread (LightGBM's histogram path changed split selection on
    pinned vs unpinned threads, breaking ``test_basic_regression`` for
    LGBMRegressor). Callers that need full OMP-pinning MUST wrap with
    their own try/finally; this function only touches estimator params.
    """
    if not hasattr(estimator, "set_params"):
        return
    try:
        valid = set(estimator.get_params().keys())  # type: ignore[attr-defined]
    except Exception:
        return
    pinned = {p: 1 for p in _THREAD_PARAMS if p in valid}
    if pinned:
        try:
            estimator.set_params(**pinned)
        except Exception as e:  # nosec B110 - swallow converted to debug-log, non-fatal by design
            logger.debug("suppressed in _helpers.py:58: %s", e)
            pass


def suppress_irritating_3rdparty_warnings() -> None:
    """Silence known-benign UserWarning noise emitted by third-party libraries during fits/CV loops (e.g. catboost's harmless JIT-optimisation notice), so it doesn't drown out real warnings in long training logs."""
    # "optimize" typo is verbatim from catboost's _catboost.pyx _jit_common_checks(); do not "fix" it or the filter stops matching.
    for message in [r"Can't optimize method \"evaluate\" because self argument is used"]:
        warnings.filterwarnings("ignore", category=UserWarning, message=message)


def split_into_train_test(
    X: pd.DataFrame | np.ndarray, y: pd.DataFrame | np.ndarray, train_index: np.ndarray, test_index: np.ndarray, features_indices: Union[np.ndarray, None] = None,
    X_estimator: np.ndarray | None = None, col_pos: dict | None = None,
) -> tuple:
    """Split X & y according to indices & dtypes. Handles pd.DataFrame, np.ndarray, and polars.

    perf (2026-06-05): when ``X_estimator`` (a contiguous numpy matrix mirroring the all-numeric DataFrame
    ``X`` column-for-column) and ``col_pos`` (name -> integer column position) are supplied, the X-side
    train/test slabs are taken from the NUMPY matrix by integer position instead of from the pandas frame.
    This feeds the inner estimator numpy directly so it skips the per-fit/per-predict ``_data_from_pandas``
    reconversion + the per-column dtype-validation storm (the RFECV fit hotspot on LightGBM). The y-side
    slicing is unchanged. The numpy matrix is float64 (bit-identical to pandas for the all-numeric case;
    float32 would alter LightGBM splits), so the selection is unaffected. Only the X-DataFrame branch
    consults ``X_estimator``; the polars / ndarray branches are untouched.
    """

    X_train: Any
    X_test: Any
    if X_estimator is not None and isinstance(X, pd.DataFrame):
        # All-numeric fast path: slice rows + (name->pos) cols out of the numpy mirror.
        tr_arr = np.asarray(train_index)
        te_arr = np.asarray(test_index)
        if features_indices is None:
            X_train = X_estimator[tr_arr, :]
            X_test = X_estimator[te_arr, :]
        elif col_pos is not None and not isinstance(features_indices[0], (int, np.integer)):
            pos = np.fromiter((col_pos[f] for f in features_indices), dtype=np.intp, count=len(features_indices))
            X_train = X_estimator[np.ix_(tr_arr, pos)]
            X_test = X_estimator[np.ix_(te_arr, pos)]
        else:
            fi_arr = np.asarray(features_indices, dtype=np.intp)
            X_train = X_estimator[np.ix_(tr_arr, fi_arr)]
            X_test = X_estimator[np.ix_(te_arr, fi_arr)]
        y_train = y.iloc[train_index, :] if isinstance(y, pd.DataFrame) else (y.iloc[train_index] if isinstance(y, pd.Series) else y[train_index])
        y_test = y.iloc[test_index, :] if isinstance(y, pd.DataFrame) else (y.iloc[test_index] if isinstance(y, pd.Series) else y[test_index])
        return X_train, y_train, X_test, y_test

    if isinstance(X, pd.DataFrame):
        # ``X.iloc[np.ix_(rows, cols)]`` selects both axes in one shot. The chained
        # ``X.iloc[rows].iloc[:, cols]`` form materialises the full row-slab BEFORE
        # column selection - on wide tables (200k rows x 10k cols) that's ~16 GB
        # of intermediate write when only the K-feature subset (~8 GB) is needed.
        if features_indices is None:
            X_train = X.iloc[train_index, :]
            X_test = X.iloc[test_index, :]
        else:
            tr_arr = np.asarray(train_index)
            te_arr = np.asarray(test_index)
            if isinstance(features_indices[0], (int, np.integer)):
                fi_arr = np.asarray(features_indices)
                X_train = X.iloc[np.ix_(tr_arr, fi_arr)]
                X_test = X.iloc[np.ix_(te_arr, fi_arr)]
            else:
                X_train = X.loc[X.index[tr_arr], list(features_indices)]
                X_test = X.loc[X.index[te_arr], list(features_indices)]
        y_train = y.iloc[train_index, :] if isinstance(y, pd.DataFrame) else (y.iloc[train_index] if isinstance(y, pd.Series) else y[train_index])
        y_test = y.iloc[test_index, :] if isinstance(y, pd.DataFrame) else (y.iloc[test_index] if isinstance(y, pd.Series) else y[test_index])
    elif isinstance(X, pl.DataFrame):
        # Polars rejects ``X[np.ix_(rows, cols)]``; select rows and cols in two steps.
        tr_idx = list(np.asarray(train_index))
        te_idx = list(np.asarray(test_index))
        if features_indices is None:
            X_train = X[tr_idx]
            X_test = X[te_idx]
        else:
            fi = np.asarray(features_indices)
            if fi.dtype.kind in ("i", "u"):
                cols_sel = [X.columns[int(i)] for i in fi]
            else:
                cols_sel = [str(c) for c in fi]
            X_train = X.select(cols_sel)[tr_idx]
            X_test = X.select(cols_sel)[te_idx]
        if hasattr(y, "to_numpy") and not isinstance(y, np.ndarray):
            y_np = y.to_numpy()
        else:
            y_np = y
        y_train = y_np[train_index, :] if hasattr(y_np, "shape") and len(y_np.shape) > 1 else y_np[train_index]
        y_test = y_np[test_index, :] if hasattr(y_np, "shape") and len(y_np.shape) > 1 else y_np[test_index]
    else:
        if features_indices is None:
            X_train = X[train_index, :]
            X_test = X[test_index, :]
        else:
            fi = np.asarray(features_indices)
            X_train = X[np.ix_(np.asarray(train_index), fi)]
            X_test = X[np.ix_(np.asarray(test_index), fi)]
        # train_index/test_index are POSITIONAL (KFold.split output). A pandas y reaching this
        # ndarray-X branch (mixed ndarray X + pandas y) must be sliced positionally via .iloc;
        # raw ``y[idx]`` on a Series with a non-RangeIndex resolves as a label lookup -> KeyError.
        if isinstance(y, pd.Series):
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        elif isinstance(y, pd.DataFrame):
            y_train, y_test = y.iloc[train_index, :], y.iloc[test_index, :]
        else:
            y_train = y[train_index, :] if len(y.shape) > 1 else y[train_index]
            y_test = y[test_index, :] if len(y.shape) > 1 else y[test_index]

    return X_train, y_train, X_test, y_test


def store_averaged_cv_scores(pos: int, scores: list, evaluated_scores_mean: dict, evaluated_scores_std: dict, self: Any) -> tuple:
    """Compute (mean, std, final_score) and store at ``pos`` ONLY if the new score
    beats any existing score at the same key. Returns (mean, std, final_score, was_stored).

    Gating on best-so-far avoids silently downgrading the curve and the cached
    selected_features when MBH re-explores the same nfeatures-count with a worse subset.
    """
    scores_arr = np.array(scores)
    n_nan = int(np.isnan(scores_arr).sum()) if scores_arr.size else 0
    if n_nan:
        logger.warning(
            "store_averaged_cv_scores @ pos=%d: %d / %d CV fold score(s) are NaN. "
            "Likely cause: single-class CV fold (stratified split would fix it) "
            "or scorer returning NaN on degenerate folds.",
            pos, n_nan, scores_arr.size,
        )

    # nanmean/nanstd: a single degenerate fold mustn't poison the entire iter's final_score.
    if scores_arr.size and n_nan and n_nan < scores_arr.size:
        scores_mean, scores_std = np.nanmean(scores_arr), np.nanstd(scores_arr)
    else:
        scores_mean, scores_std = np.mean(scores_arr), np.std(scores_arr)
    final_score = scores_mean * self.mean_perf_weight - scores_std * self.std_perf_weight

    existing_mean = evaluated_scores_mean.get(pos)
    existing_std = evaluated_scores_std.get(pos)
    if existing_mean is None:
        existing_final = -np.inf
    else:
        existing_final = existing_mean * self.mean_perf_weight - existing_std * self.std_perf_weight
    was_stored = (not np.isnan(final_score)) and final_score > existing_final
    if was_stored:
        evaluated_scores_mean[pos] = scores_mean
        evaluated_scores_std[pos] = scores_std

    return scores_mean, scores_std, final_score, was_stored


def get_next_features_subset(
    nsteps: int,
    original_features: list,
    feature_importances: dict,
    evaluated_scores_mean: dict,
    evaluated_scores_std: dict,
    use_all_fi_runs: bool,
    use_last_fi_run_only: bool,
    use_one_freshest_fi_run: bool,
    use_fi_ranking: bool,
    top_predictors_search_method: OptimumSearch = OptimumSearch.ScipyLocal,
    votes_aggregation_method: VotesAggregation = VotesAggregation.Borda,
    Optimizer: Any = None,
    fi_missing_policy: str = "worst",
    dichotomic_epsilon: float = 0.0,
    rng: Any = None,
    fi_decay_rate: float = 0.0,
    fi_run_order: Union[list, None] = None,
    importance_agg: str = "legacy",
    fi_family: Union[str, None] = None,
    signed_importances: Union[dict, None] = None,
    importance_agg_k_cv: float = 1.0,
    dichotomic_step: str = "midpoint",
    elimination_rule: str = "importance",
) -> list:
    """Generate the next 'next_nfeatures_to_check' candidate to evaluate.
    Combines FIs from prior runs into ranks via voting, returns the top-N."""
    if nsteps == 0:
        return original_features

    # +1 on the upper bound includes the all-features candidate.
    # sort: set-difference order is PYTHONHASHSEED-dependent; a positional pick must be over a stable order.
    remaining = sorted(set(np.arange(1, len(original_features) + 1)) - set(evaluated_scores_mean.keys()))
    if len(remaining) == 0:
        return []

    next_nfeatures_to_check: Union[int, None]
    if top_predictors_search_method == OptimumSearch.ExhaustiveRandom:
        # use the threaded seeded rng (np.random.default_rng); module-global random is unseeded -> not reproducible.
        if rng is not None and hasattr(rng, "integers"):
            next_nfeatures_to_check = int(remaining[int(rng.integers(0, len(remaining)))])
        else:  # nosec B311 - non-cryptographic sampling/jitter, not a security-sensitive use
            next_nfeatures_to_check = int(random.choice(remaining))  # nosec B311 - non-crypto sampling/jitter, not used for tokens/secrets
    elif top_predictors_search_method == OptimumSearch.ModelBasedHeuristic:
        next_nfeatures_to_check = Optimizer.suggest_candidate()
    elif top_predictors_search_method == OptimumSearch.ExhaustiveDichotomic:
        next_nfeatures_to_check = _suggest_dichotomic(
            remaining=sorted(remaining),
            evaluated_scores_mean=evaluated_scores_mean,
            n_total=len(original_features),
            epsilon=float(dichotomic_epsilon),
            rng=rng,
            step=dichotomic_step,
        )
    elif top_predictors_search_method == OptimumSearch.ScipyLocal:
        next_nfeatures_to_check = _suggest_scipy_local(
            remaining=remaining,
            evaluated_scores_mean=evaluated_scores_mean,
            n_total=len(original_features),
            epsilon=float(dichotomic_epsilon),
            rng=rng,
        )
    elif top_predictors_search_method == OptimumSearch.ScipyGlobal:
        next_nfeatures_to_check = _suggest_scipy_global(
            remaining=remaining,
            evaluated_scores_mean=evaluated_scores_mean,
            n_total=len(original_features),
            epsilon=float(dichotomic_epsilon),
            rng=rng,
        )
    else:
        raise NotImplementedError(
            f"top_predictors_search_method={top_predictors_search_method!r} "
            f"is declared in OptimumSearch but not implemented in "
            f"get_next_features_subset. Currently supported: "
            f"OptimumSearch.ExhaustiveRandom, OptimumSearch.ModelBasedHeuristic, "
            f"OptimumSearch.ExhaustiveDichotomic, OptimumSearch.ScipyLocal, "
            f"OptimumSearch.ScipyGlobal."
        )

    if next_nfeatures_to_check is None:
        return []

    fi_to_consider = select_appropriate_feature_importances(
        feature_importances=feature_importances,
        nfeatures=next_nfeatures_to_check,
        n_original_features=len(original_features),
        use_all_fi_runs=use_all_fi_runs,
        use_last_fi_run_only=use_last_fi_run_only,
        use_one_freshest_fi_run=use_one_freshest_fi_run,
        use_fi_ranking=use_fi_ranking,
        votes_aggregation_method=votes_aggregation_method,
    )
    # F8 (Wave 3, 2026-05-28): exponential decay of FI history. With ``fi_decay_rate > 0``, run K is weighted ``(1 - rate)^(age_K)``,
    # where ``age_K`` is the number of fresher runs that exist. So a 30-iter-old vote weighs (1-0.05)^30 = 0.21 vs the latest vote = 1.0
    # when rate=0.05. Default 0.0 keeps the legacy equal-weight voting. Recommended 0.02-0.1 for long runs (>=30 iters).
    _run_weights = None
    if fi_decay_rate and fi_decay_rate > 0.0 and fi_run_order:
        # fi_run_order is the insertion order (newest LAST). Map age -> weight.
        _decay = float(fi_decay_rate)
        _ordered_keys = [k for k in fi_run_order if k in fi_to_consider]
        _n_runs = len(_ordered_keys)
        if _n_runs > 0:
            _run_weights = {k: (1.0 - _decay) ** (_n_runs - 1 - i) for i, k in enumerate(_ordered_keys)}
    # elimination_rule='stability' (opt-in): rank for elimination by mean_importance discounted
    # by cross-fold selection-frequency at the elimination cut, protecting steady-mid-rank features
    # from one-fold-noise eviction. Operates on the RAW per-fold table independently of importance_agg
    # (no double-count: dispatched discounts value-CV in the ranking, stability discounts rank-volatility
    # around the cut). Falls through to the legacy/dispatched path when fewer than 2 runs are available.
    if elimination_rule == "stability" and len(fi_to_consider) >= 2:
        from ._helpers_importance_agg import aggregate_stability
        _stab = aggregate_stability(fi_to_consider, cut_k=next_nfeatures_to_check)
        if _stab:
            ranks = sorted(
                _stab.keys(),
                key=lambda k: (-(_stab[k] if np.isfinite(_stab[k]) else -np.inf), str(k)),
            )
            return ranks[:next_nfeatures_to_check]
    if importance_agg == "dispatched" and fi_family in ("tree", "linear"):
        from ._helpers_importance_agg import aggregate_importances_dispatched
        _signed_subset = None
        if signed_importances and fi_family == "linear":
            _signed_subset = {k: v for k, v in signed_importances.items() if k in fi_to_consider}
        ranks = aggregate_importances_dispatched(
            feature_importances=fi_to_consider,
            family=fi_family,
            votes_aggregation_method=votes_aggregation_method,
            signed_importances=_signed_subset,
            k_cv=float(importance_agg_k_cv),
            fi_missing_policy=fi_missing_policy,
            run_weights=_run_weights,
        )
    else:
        ranks = get_actual_features_ranking(
            feature_importances=fi_to_consider,
            votes_aggregation_method=votes_aggregation_method,
            fi_missing_policy=fi_missing_policy,
            run_weights=_run_weights,
        )
    return ranks[:next_nfeatures_to_check]


def _curve_is_flat_near_best(evaluated_scores_mean: dict, best_n: int) -> bool:
    """True when the score curve around ``best_n`` is flat (relative slope tiny).

    Compares the best score against the nearest evaluated neighbours on each side;
    a flat verdict means a big stride is safe because nearby N's score ~the same,
    so the optimum is not in the immediate neighbourhood we already mapped.
    """
    if len(evaluated_scores_mean) < 2:
        return True
    best_s = evaluated_scores_mean[best_n]
    lower = [n for n in evaluated_scores_mean if n < best_n]
    higher = [n for n in evaluated_scores_mean if n > best_n]
    neigh = []
    if lower:
        neigh.append(max(lower))
    if higher:
        neigh.append(min(higher))
    if not neigh:
        return True
    span = max(abs(v) for v in evaluated_scores_mean.values()) or 1.0
    # Max relative score change to the immediate evaluated neighbours.
    rel = max(abs(best_s - evaluated_scores_mean[n]) for n in neigh) / span
    return bool(rel < 0.01)


def _suggest_dichotomic(remaining: list, evaluated_scores_mean: dict, n_total: int, epsilon: float = 0.0, rng: Any = None, step: str = "auto") -> Union[int, None]:
    """Coarse-to-fine suggester for ExhaustiveDichotomic.

    ``step='midpoint'`` (default) is the legacy fixed bisection. ``step='auto'`` is the adaptive elimination-pace schedule:
    while the unevaluated pool is large AND the CV curve near the best is flat, it strides by ``max(1, floor(frac * n_remaining))``
    away from the best to map the curve cheaply; as the pool shrinks / the curve starts moving near the knee it collapses back to
    the midpoint bisection (effectively step->1), so the FINAL probed neighbourhood is identical to the fine search. 'auto' is an
    opt-in: it is selection-equivalent to midpoint but showed no replicated wall win (bench_dichotomic_adaptive_step.py).

    With one or zero evaluations: probe the midpoint of the full range. With >=2: identify the highest-scoring evaluated N
    and (adaptive) stride or (legacy) bisect toward the nearest unevaluated neighbour.

    When ``epsilon > 0`` and rng samples a Bernoulli(epsilon) success, pick a random unevaluated N OUTSIDE the best's
    neighbourhood (gap > p/4) -- prevents the two-plateau hill-climb trap. ``rng=None`` uses module ``random``.
    """
    remaining = sorted(remaining)
    if not remaining:
        return None
    if epsilon > 0 and rng is not None and len(evaluated_scores_mean) >= 3 and len(remaining) >= 2:
        if float(rng.random()) < float(epsilon):
            # Pick a remaining N far from the current best.
            best_evaluated = max(evaluated_scores_mean.items(), key=lambda kv: kv[1])[0]
            _far = [n for n in remaining if abs(n - best_evaluated) > max(2, n_total // 4)]
            pool = _far if _far else remaining
            return int(pool[int(rng.integers(0, len(pool))) if hasattr(rng, "integers") else rng.randrange(len(pool))])
    if len(evaluated_scores_mean) <= 1:
        # First-time call (or just-the-baseline): probe the midpoint of the FULL range.
        target = max(1, n_total // 2)
        return int(min(remaining, key=lambda n: abs(n - target)))
    best_evaluated = max(evaluated_scores_mean.items(), key=lambda kv: kv[1])[0]

    # Adaptive coarse step: only fire while the pool is still large AND the curve near best is flat. ``frac`` shrinks as
    # the pool drains (0.5 -> 0 over n_total), so strides start big and naturally taper to 1 near the knee, after which we
    # fall through to the legacy midpoint refinement. Bounded by n_total//4 so a stride can never overshoot the whole range.
    if step == "auto" and len(remaining) > 4 and _curve_is_flat_near_best(evaluated_scores_mean, best_evaluated):
        frac = 0.5 * (len(remaining) / max(n_total, 1))
        stride = int(np.floor(frac * len(remaining)))
        stride = max(1, min(stride, max(1, n_total // 4)))
        if stride > 1:
            # Stride toward whichever side has more unevaluated room (more information left to gain).
            lower = [n for n in remaining if n < best_evaluated]
            higher = [n for n in remaining if n > best_evaluated]
            side_hi = len(higher) >= len(lower)
            target = best_evaluated + stride if side_hi and higher else best_evaluated - stride
            if (side_hi and not higher) or (not side_hi and not lower):
                target = best_evaluated - stride if side_hi else best_evaluated + stride
            return int(min(remaining, key=lambda n: abs(n - target)))

    # Legacy fine refinement: bisect toward the nearest unevaluated neighbour on the wider-gap side.
    lower = [n for n in remaining if n < best_evaluated]
    higher = [n for n in remaining if n > best_evaluated]
    candidates = []
    if lower:
        gap_lo = best_evaluated - max(lower)
        candidates.append((gap_lo, (best_evaluated + max(lower)) // 2))
    if higher:
        gap_hi = min(higher) - best_evaluated
        candidates.append((gap_hi, (best_evaluated + min(higher)) // 2))
    if not candidates:
        # No unevaluated N adjacent to the best: fall back to any remaining N closest to it.
        return int(min(remaining, key=lambda n: abs(n - best_evaluated)))
    target = max(candidates, key=lambda gc: gc[0])[1]
    return int(min(remaining, key=lambda n: abs(n - target)))


def _suggest_scipy_local(remaining: list, evaluated_scores_mean: dict, n_total: int, epsilon: float = 0.0, rng: Any = None) -> Union[int, None]:
    """S5 (Wave 2, 2026-05-28): retained as a thin alias for ExhaustiveDichotomic.

    The previous implementation built a piecewise-linear interpolant over evaluated points and ran scipy's ``minimize_scalar`` on it.
    The argmax of a piecewise-linear function is ALWAYS one of its breakpoints (= ``xs`` = already-evaluated N's), so the scipy
    optimiser collapses to "nearest unevaluated N to an already-evaluated one" -- exactly what dichotomic returns, at the cost of a
    scipy import + roundtrip. We now delegate to dichotomic with optional epsilon kick; users keep the OptimumSearch.ScipyLocal enum
    value to avoid silent API breakage in pickled configs.
    """
    return _suggest_dichotomic(remaining, evaluated_scores_mean, n_total, epsilon=epsilon, rng=rng)


def _suggest_scipy_global(remaining: list, evaluated_scores_mean: dict, n_total: int, epsilon: float = 0.0, rng: Any = None) -> Union[int, None]:
    """S5 (Wave 2, 2026-05-28): retained as a thin alias for ExhaustiveDichotomic.

    Same reasoning as _suggest_scipy_local: differential_evolution over a piecewise-linear interpolant has no global structure to
    discover beyond the breakpoints. Delegate to dichotomic so the search has a single, well-understood code path.
    """
    return _suggest_dichotomic(remaining, evaluated_scores_mean, n_total, epsilon=epsilon, rng=rng)

# ----------------------------------------------------------------------
# Sibling-module re-exports. Knockoff helpers live in _knockoffs.py
# (carved out in Wave 5 to keep this file under 1k LOC). Legacy callers
# importing make_gaussian_knockoffs / select_features_fdr /
# knockoff_importance from this module keep working.
# ----------------------------------------------------------------------
from ._knockoffs import (
    make_gaussian_knockoffs,
    select_features_fdr,
    knockoff_importance,
)

# Importance computation + vote-ranking helpers live in _helpers_importance.py.
# Re-exported so legacy ``from ._helpers import get_feature_importances`` callers keep working.
from ._helpers_importance import (
    _conditional_permutation_importance,
    _impute_ragged_fi_table,
    get_actual_features_ranking,
    get_feature_importances,
    select_appropriate_feature_importances,
)
