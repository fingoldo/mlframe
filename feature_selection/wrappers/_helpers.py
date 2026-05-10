"""Helper functions for the RFECV wrapper module.

Split out of the prior monolithic wrappers.py to keep each layer focused:
- split_into_train_test: dataframe / ndarray / polars-aware fold splitting
- store_averaged_cv_scores: NaN-safe per-iter score aggregation with best-stored gating
- get_feature_importances: importance_getter dispatch (auto / coef_ / feature_importances_ / permutation / shap / callable)
- select_appropriate_feature_importances + get_actual_features_ranking: voting layer
- get_next_features_subset: optimiser-driven candidate selection
- multi-threaded estimator detection helpers (Phase 4 N3)
- suppress_irritating_3rdparty_warnings
"""
from __future__ import annotations

import logging
import random
import warnings
from typing import Callable, Union

import numpy as np
import pandas as pd
import polars as pl

from mlframe.votenrank import Leaderboard

from ._enums import OptimumSearch, VotesAggregation


logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Multi-threaded estimator detection (Phase 4 N3)
# ----------------------------------------------------------------------------

# Class-name fragments for estimators that already use native multi-threading.
# Parallelising CV folds on top of these over-subscribes cores - one outer
# joblib worker per fold * N inner threads per fit = N x core_count > cores.
# Phase 4 N3 detects this and either auto-falls-back to sequential (default)
# or pins inner threads to 1 (force_parallel=True path).
_MULTITHREADED_ESTIMATOR_PATTERNS = (
    "CatBoost", "LGBM", "LightGBM", "XGB", "XGBoost",
    "RandomForest", "ExtraTrees", "GradientBoosting", "HistGradientBoosting",
)

# Thread-count constructor params recognised on common multi-threaded estimators.
# Used only when force_parallel=True to pin each fold's clone to single thread.
_THREAD_PARAMS = ("thread_count", "n_jobs", "n_threads", "nthread", "num_threads")


def _detect_multithreaded(estimator: object) -> bool:
    """True if the estimator's class name matches a known multi-threaded
    pattern (CatBoost / LightGBM / XGBoost / RandomForest / ExtraTrees /
    GradientBoosting / HistGradientBoosting). Used by Phase 4 N3 to decide
    whether parallelising CV folds would over-subscribe cores."""
    name = type(estimator).__name__
    return any(p in name for p in _MULTITHREADED_ESTIMATOR_PATTERNS)


def _pin_threads_to_one(estimator: object) -> None:
    """Attempt to set every known thread-count param to 1 on the estimator's
    constructor params. Best-effort: some params may not exist on every
    estimator type; ignore those silently."""
    if not hasattr(estimator, "set_params"):
        return
    try:
        valid = set(estimator.get_params().keys())
    except Exception:
        return
    pinned = {p: 1 for p in _THREAD_PARAMS if p in valid}
    if pinned:
        try:
            estimator.set_params(**pinned)
        except Exception:
            pass


def suppress_irritating_3rdparty_warnings() -> None:
    for message in [r"Can't optimze method \"evaluate\" because self argument is used"]:
        # Filter out the specific warning message using a substring or regex pattern.
        warnings.filterwarnings("ignore", category=UserWarning, message=message)


# ----------------------------------------------------------------------------
# Splitting
# ----------------------------------------------------------------------------

def split_into_train_test(
    X: Union[pd.DataFrame, np.ndarray], y: Union[pd.DataFrame, np.ndarray], train_index: np.ndarray, test_index: np.ndarray, features_indices: np.ndarray = None
) -> tuple:
    """Split X & y according to indices & dtypes. Basically this accounts for diffeent dtypes (pd.DataFrame, np.ndarray) to perform the same."""

    if isinstance(X, pd.DataFrame):
        # Perf #2 fix: the prior integer-index path did
        # ``X.iloc[rows].iloc[:, cols]`` (chained), which materialises the
        # full row-slab BEFORE column selection. On wide tables (200k rows x
        # 10k cols) that's ~16 GB of intermediate write when only the
        # K-feature subset (~8 GB) was needed. ``X.iloc[np.ix_(rows, cols)]``
        # mirrors the numpy branch and selects both axes in one shot, cutting
        # 2-7% off total wall-clock on wide-table runs.
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
        # Polars branch (2026-04-21 fix 9.8): the generic numpy path below
        # used ``X[np.ix_(rows, cols)]``, which raises on polars. Polars
        # selects rows+cols in two steps.
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
        y_train = y[train_index, :] if len(y.shape) > 1 else y[train_index]
        y_test = y[test_index, :] if len(y.shape) > 1 else y[test_index]

    return X_train, y_train, X_test, y_test


# ----------------------------------------------------------------------------
# Scoring aggregation (F23 + F35 fixes)
# ----------------------------------------------------------------------------

def store_averaged_cv_scores(pos: int, scores: list, evaluated_scores_mean: dict, evaluated_scores_std: dict, self: object) -> tuple:
    """Compute (mean, std, final_score) and store at ``pos`` ONLY if the new score
    beats any existing score at the same key. Returns (mean, std, final_score, was_stored).

    F35 fix: the prior version unconditionally overwrote evaluated_scores_mean[pos],
    so when MBH re-explored the same nfeatures-count with a worse subset, both
    the curve and the cached selected_features were silently downgraded to the
    last evaluation rather than the best.
    """
    scores = np.array(scores)
    n_nan = int(np.isnan(scores).sum()) if scores.size else 0
    if n_nan:
        logger.warning(
            "store_averaged_cv_scores @ pos=%d: %d / %d CV fold score(s) are NaN. "
            "Likely cause: single-class CV fold (stratified split would fix it) "
            "or scorer returning NaN on degenerate folds.",
            pos, n_nan, scores.size,
        )

    # F23 fix: nanmean/nanstd so a single degenerate fold doesn't poison the
    # entire iter's final_score.
    if scores.size and n_nan and n_nan < scores.size:
        scores_mean, scores_std = np.nanmean(scores), np.nanstd(scores)
    else:
        scores_mean, scores_std = np.mean(scores), np.std(scores)
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


# ----------------------------------------------------------------------------
# Feature-importance dispatch (F38 + Phase 4 N6)
# ----------------------------------------------------------------------------

def get_feature_importances(
    model: object,
    current_features: list,
    importance_getter: Union[str, Callable],
    data: Union[pd.DataFrame, np.ndarray, None] = None,
    reference_data: Union[pd.DataFrame, np.ndarray, None] = None,
    target: Union[pd.Series, np.ndarray, None] = None,
) -> dict:
    """Compute per-feature importance for a fitted model.

    importance_getter:
        - 'auto': inspect model's attributes (feature_importances_ -> coef_)
        - 'feature_importances_' / 'coef_' / any other attr name
        - 'permutation' (Phase 4 N6): sklearn.inspection.permutation_importance
        - 'shap' (Phase 4 N6): shap.Explainer mean-abs values
        - Callable: importance_getter(model, data, reference_data, target)
    """
    if isinstance(importance_getter, str):
        if importance_getter == "permutation":
            if target is None:
                raise ValueError(
                    "importance_getter='permutation' requires target (y_test) "
                    "to score against. Pass target= explicitly."
                )
            from sklearn.inspection import permutation_importance
            pi = permutation_importance(
                model, data, target,
                n_repeats=5,
                random_state=0,
                n_jobs=1,
            )
            res = pi.importances_mean
        elif importance_getter == "shap":
            try:
                import shap as _shap
            except ImportError as _exc:
                raise ImportError(
                    "importance_getter='shap' requires the optional ``shap`` "
                    "package. Install via ``pip install shap``."
                ) from _exc
            try:
                explainer = _shap.Explainer(model, data)
                shap_values = explainer(data, check_additivity=False)
                vals = shap_values.values
                if vals.ndim > 2:
                    vals = np.abs(vals).mean(axis=tuple(range(2, vals.ndim)))
                res = np.abs(vals).mean(axis=0)
            except Exception as _exc:
                raise RuntimeError(
                    f"shap.Explainer failed for {type(model).__name__}: {_exc}. "
                    f"Try importance_getter='permutation' instead."
                ) from _exc
        else:
            if importance_getter == "auto":
                if hasattr(model, "feature_importances_"):
                    getter_attr = "feature_importances_"
                elif hasattr(model, "coef_"):
                    getter_attr = "coef_"
                else:
                    raise AttributeError(
                        f"importance_getter='auto' could not find feature_importances_ or coef_ on a fitted {type(model).__name__}."
                    )
            else:
                getter_attr = importance_getter
            res = getattr(model, getter_attr)
            if getter_attr == "coef_":
                res = np.abs(res)
            if res.ndim > 1:
                res = res.sum(axis=0)
    else:
        try:
            res = importance_getter(model=model, data=data, reference_data=reference_data, target=target)
        except TypeError:
            res = importance_getter(model=model, data=data, reference_data=reference_data)

    if len(res) != len(current_features):
        raise ValueError(f"Feature importances length {len(res)} doesn't match current_features length {len(current_features)}")

    try:
        res_arr = np.asarray(res, dtype=float)
        n_nan = int(np.isnan(res_arr).sum()) if res_arr.size else 0
    except (TypeError, ValueError):
        n_nan = 0
    if n_nan:
        logger.warning(
            "get_feature_importances: %d / %d importance value(s) are NaN from %s.",
            n_nan, res_arr.size, type(model).__name__,
        )
    return {feature_index: feature_importance for feature_index, feature_importance in zip(current_features, res)}


# ----------------------------------------------------------------------------
# Voting layer
# ----------------------------------------------------------------------------

def select_appropriate_feature_importances(
    feature_importances: dict,
    nfeatures: int,
    n_original_features: int,
    use_all_fi_runs: bool = True,
    use_last_fi_run_only: bool = False,
    use_one_freshest_fi_run: bool = False,
    use_fi_ranking: bool = False,
) -> dict:
    if use_last_fi_run_only:
        fi_to_consider = {key: value for key, value in feature_importances.items() if len(value) == n_original_features}
    else:
        if use_all_fi_runs:
            fi_to_consider = {key: value for key, value in feature_importances.items() if len(value) > 1} if n_original_features > 1 else feature_importances
        else:
            if use_one_freshest_fi_run:
                # F25 fix: range upper-bound was n_original_features (exclusive),
                # so the FI run on the full feature set was never picked.
                fi_to_consider = {}
                for possible_nfeatures in range(nfeatures + 1, n_original_features + 1):
                    for key, value in feature_importances.items():
                        if len(value) == possible_nfeatures:
                            fi_to_consider[key] = value
                    if fi_to_consider:
                        print(f"using freshest FI of {possible_nfeatures} features for nfeatures={nfeatures}")
                        break
            else:
                fi_to_consider = {key: value for key, value in feature_importances.items() if (len(value) > nfeatures and len(value) != 1)}
    if use_fi_ranking:
        fi_to_consider = {key: pd.Series(value).rank(ascending=True, pct=True).to_dict() for key, value in fi_to_consider.items()}
    return fi_to_consider


def get_actual_features_ranking(feature_importances: dict, votes_aggregation_method: VotesAggregation) -> list:
    """Vote-based rank of features given per-run importances.
    Borda/AM/GM/Dowdall use only ranks (cheap). Copeland needs majority_graph
    (now lazy-built in Leaderboard - see votenrank Phase 4 fix)."""
    lb = Leaderboard(table=pd.DataFrame(feature_importances))
    if votes_aggregation_method == VotesAggregation.Borda:
        ranks = lb.borda_ranking()
    elif votes_aggregation_method == VotesAggregation.AM:
        ranks = lb.mean_ranking(mean_type="arithmetic")
    elif votes_aggregation_method == VotesAggregation.GM:
        ranks = lb.mean_ranking(mean_type="geometric")
    elif votes_aggregation_method == VotesAggregation.Copeland:
        ranks = lb.copeland_ranking()
    elif votes_aggregation_method == VotesAggregation.Dowdall:
        ranks = lb.dowdall_ranking()
    elif votes_aggregation_method == VotesAggregation.Minimax:
        ranks = lb.minimax_ranking()
    elif votes_aggregation_method == VotesAggregation.OG:
        ranks = lb.optimality_gap_ranking(gamma=1)
    elif votes_aggregation_method == VotesAggregation.Plurality:
        ranks = lb.plurality_ranking()
    else:
        raise NotImplementedError(
            f"votes_aggregation_method={votes_aggregation_method!r} not handled"
        )
    return ranks.index.values.tolist()


# ----------------------------------------------------------------------------
# Candidate-search dispatch (F1 + F41 fixes)
# ----------------------------------------------------------------------------

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
    Optimizer: object = None,
) -> list:
    """Generate the next 'next_nfeatures_to_check' candidate to evaluate.
    Combines FIs from prior runs into ranks via voting, returns the top-N."""
    if nsteps == 0:
        return original_features

    # F41 fix: +1 includes the all-features candidate.
    remaining = list(set(np.arange(1, len(original_features) + 1)) - set(evaluated_scores_mean.keys()))
    if len(remaining) == 0:
        return []

    if top_predictors_search_method == OptimumSearch.ExhaustiveRandom:
        next_nfeatures_to_check = random.choice(remaining)
    elif top_predictors_search_method == OptimumSearch.ModelBasedHeuristic:
        next_nfeatures_to_check = Optimizer.suggest_candidate()
    else:
        raise NotImplementedError(
            f"top_predictors_search_method={top_predictors_search_method!r} "
            f"is declared in OptimumSearch but not implemented in "
            f"get_next_features_subset. Currently supported: "
            f"OptimumSearch.ExhaustiveRandom, OptimumSearch.ModelBasedHeuristic."
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
    )
    ranks = get_actual_features_ranking(feature_importances=fi_to_consider, votes_aggregation_method=votes_aggregation_method)
    return ranks[:next_nfeatures_to_check]
