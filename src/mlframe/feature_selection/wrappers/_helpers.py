"""Helper functions for the RFECV wrapper module."""
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

# Cell budget (n_rows * n_cols of the per-fold held-out set) below which the unspecified ('auto')
# importance default routes to PERMUTATION (the accuracy winner on the FS bench); above it 'auto' falls
# back to impurity for speed. ~40k x 100; tune via the dispatcher rather than hardcoding per call site.
_PERM_AUTO_CELL_CAP = 4_000_000


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
    # "optimze" typo is verbatim from catboost's _catboost.pyx _jit_common_checks(); do not "fix" it or the filter stops matching.
    for message in [r"Can't optimze method \"evaluate\" because self argument is used"]:
        warnings.filterwarnings("ignore", category=UserWarning, message=message)




def split_into_train_test(
    X: pd.DataFrame | np.ndarray, y: pd.DataFrame | np.ndarray, train_index: np.ndarray, test_index: np.ndarray, features_indices: np.ndarray = None
) -> tuple:
    """Split X & y according to indices & dtypes. Handles pd.DataFrame, np.ndarray, and polars."""

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


def store_averaged_cv_scores(pos: int, scores: list, evaluated_scores_mean: dict, evaluated_scores_std: dict, self: object) -> tuple:
    """Compute (mean, std, final_score) and store at ``pos`` ONLY if the new score
    beats any existing score at the same key. Returns (mean, std, final_score, was_stored).

    Gating on best-so-far avoids silently downgrading the curve and the cached
    selected_features when MBH re-explores the same nfeatures-count with a worse subset.
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

    # nanmean/nanstd: a single degenerate fold mustn't poison the entire iter's final_score.
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


def _conditional_permutation_importance(
    model,
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    n_repeats: int = 5,
    max_depth: Union[int, None] = None,
    min_samples_leaf: int = 10,
    random_state: int = 0,
) -> np.ndarray:
    """Strobl, Boulesteix, Zeileis, Hothorn 2008 conditional permutation importance.

    Vanilla permutation (Breiman 2001) shuffles X_j independently of X_{-j},
    creating out-of-distribution combinations on correlated feature sets and
    inflating measured importance. This conditional variant fits a shallow
    decision tree X_{-j} -> X_j, then permutes X_j WITHIN each leaf, which
    preserves P(X_j | X_{-j}) and removes the correlation-induced bias.

    F10 (Wave 3, 2026-05-28): max_depth=None grows the tree until
    min_samples_leaf binds. The pre-fix max_depth=5 cap under-conditioned on
    >5 correlated features and silently degenerated to vanilla permutation
    (Strobl 2008 recommends >=5 samples per leaf, no depth cap).

    Cost: ~2-3x vanilla permutation (per-feature tree fit + n_repeats
    leaf-grouped shuffles).

    Returns
    -------
    importances : ndarray of shape (p,)
        Per-feature mean score loss (baseline - permuted). Higher = more important.
    """
    from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

    is_dataframe = isinstance(X, pd.DataFrame)
    if is_dataframe:
        X_arr = X.to_numpy()
        cols = X.columns
        idx = X.index
    else:
        X_arr = np.asarray(X)
        cols = None
        idx = None

    if X_arr.ndim != 2:
        raise ValueError(f"conditional_permutation expects 2D X, got shape {X_arr.shape}")

    n, p = X_arr.shape
    rng = np.random.default_rng(random_state)
    baseline = float(model.score(X, y))
    importances = np.zeros(p, dtype=float)

    def _is_discrete(col: np.ndarray) -> bool:
        # Integer dtype OR <=10 unique non-null values: pragmatic proxy.
        if np.issubdtype(col.dtype, np.integer):
            return True
        try:
            mask = ~np.isnan(col.astype(float, copy=False))
            uniq = np.unique(col[mask])
        except (TypeError, ValueError):
            uniq = np.unique(col)
        return uniq.size <= 10

    def _is_discrete_v2(col: np.ndarray) -> bool:
        # F11 (Wave 3, 2026-05-28): tighter discrete detection.
        # Integer dtype is canonical discrete. For floats, require BOTH (a) low
        # unique count AND (b) cardinality << n_rows. Decile-binned continuous
        # variables (10 unique values across 100k rows) now correctly route to
        # regression instead of classification.
        if np.issubdtype(col.dtype, np.integer):
            return True
        try:
            mask = ~np.isnan(col.astype(float, copy=False))
            uniq = np.unique(col[mask])
        except (TypeError, ValueError):
            uniq = np.unique(col)
        _n = max(int(mask.sum()) if hasattr(mask, "sum") else len(col), 1)
        return uniq.size <= max(5, int(np.sqrt(_n))) and uniq.size <= 0.5 * _n

    for j in range(p):
        Xj = X_arr[:, j]
        Xnotj = np.delete(X_arr, j, axis=1)

        if Xnotj.shape[1] == 0:
            # Single-feature case: no conditioning set; fall back to vanilla shuffle.
            score_losses = []
            for _ in range(n_repeats):
                X_perm = X_arr.copy()
                X_perm[:, j] = rng.permutation(X_arr[:, j])
                X_for_score = (
                    pd.DataFrame(X_perm, columns=cols, index=idx) if is_dataframe else X_perm
                )
                score_losses.append(baseline - float(model.score(X_for_score, y)))
            importances[j] = float(np.mean(score_losses))
            continue

        # F10/F11: pass max_depth + min_samples_leaf. max_depth=None grows the tree
        # until min_samples_leaf binds (recommended by Strobl 2008 on >=5).
        # F11 (Wave 3, 2026-05-28): _is_discrete heuristic improved to require
        # integer-dtype OR n_unique<=max(5, sqrt(n)) so decile-binned continuous
        # variables don't trigger Classifier mis-detection.
        if _is_discrete_v2(Xj):
            tree = DecisionTreeClassifier(
                max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                random_state=random_state,
            )
        else:
            tree = DecisionTreeRegressor(
                max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                random_state=random_state,
            )

        try:
            tree.fit(Xnotj, Xj)
            leaves = tree.apply(Xnotj)
        except (ValueError, TypeError, MemoryError, RuntimeError):
            # E11 (Wave 4, 2026-05-28): widen the except to catch MemoryError
            # on 1M-row Xnotj and RuntimeError from custom-estimator paths
            # raising AttributeError-like wrapped exceptions. Conditioning
            # fit failed (constant Xj, all-NaN row, etc.); skip.
            importances[j] = 0.0
            continue

        score_losses = []
        for _ in range(n_repeats):
            X_perm = X_arr.copy()
            for leaf_id in np.unique(leaves):
                in_leaf = np.where(leaves == leaf_id)[0]
                if in_leaf.size <= 1:
                    continue
                shuffled_positions = rng.permutation(in_leaf)
                X_perm[in_leaf, j] = X_arr[shuffled_positions, j]
            X_for_score = (
                pd.DataFrame(X_perm, columns=cols, index=idx) if is_dataframe else X_perm
            )
            # E11 ext: wrap model.score in try/except so a custom scorer crash
            # on the permuted X doesn't kill the whole CPI loop. NaN signals
            # the failure to the consumer.
            try:
                score_losses.append(baseline - float(model.score(X_for_score, y)))
            except Exception:
                score_losses.append(np.nan)
        importances[j] = float(np.nanmean(score_losses)) if any(not np.isnan(s) for s in score_losses) else 0.0

    return importances


def get_feature_importances(
    model: object,
    current_features: list,
    importance_getter: str | Callable,
    data: pd.DataFrame | np.ndarray | None = None,
    reference_data: pd.DataFrame | np.ndarray | None = None,
    target: pd.Series | np.ndarray | None = None,
    train_data: pd.DataFrame | np.ndarray | None = None,
    multiclass_coef_aggregation: str = "max",
    coef_scale_source: str = "train",
    cpi_max_depth: Union[int, None] = None,
    cpi_min_samples_leaf: int = 10,
    n_repeats: int = 5,
    random_state: int = 0,
) -> dict:
    """Compute per-feature importance for a fitted model.

    importance_getter:
        - 'auto': inspect model's attributes (feature_importances_ -> coef_)
        - 'feature_importances_' / 'coef_' / any other attr name
        - 'permutation': sklearn.inspection.permutation_importance
        - 'conditional_permutation' (Strobl 2008): permute X_j WITHIN
          leaves of a shallow tree X_{-j} -> X_j; preserves P(X_j | X_{-j}) so
          correlated-feature pairs no longer inflate each other's importance.
        - 'shap': shap.Explainer mean-abs values
        - Callable: importance_getter(model, data, reference_data, target)

    Accuracy-first default (CLAUDE.md "Variant defaults: most accurate first"): when the caller leaves
    importance unspecified (None / 'auto') AND a held-out (data, target) is available, resolve to
    PERMUTATION importance. On the wide FS bench (6 scenarios x 2 seeds) permutation beat impurity on
    10/12 cells (best downstream LightGBM AUC 0.795 vs 0.790, and far cleaner: 2.5 vs 6.2 noise features
    kept), because impurity importance is in-bag and inflates high-variance / structure-bearing noise
    columns, whereas permutation measures held-out predictive degradation (noise -> ~0). SHAP main-effect
    importance behaved like impurity (kept noise) and was the slowest, so it is NOT the default. Cost gate:
    above ``_PERM_AUTO_CELL_CAP`` cells the per-fold permutation cost is prohibitive, so 'auto' falls back
    to impurity (speed) where it is the only affordable option. Pass importance_getter='feature_importances_'
    to force impurity, or 'permutation' to force it regardless of size.
    """
    if importance_getter is None:
        importance_getter = "auto"
    if importance_getter == "auto" and target is not None and data is not None:
        try:
            _shape = getattr(data, "shape", None)
            _cells = int(_shape[0]) * (int(_shape[1]) if len(_shape) > 1 else 1) if _shape else 0
        except Exception:
            _cells = 0
        if 0 < _cells <= _PERM_AUTO_CELL_CAP:
            importance_getter = "permutation"  # accuracy winner; below the cost cap
    if isinstance(importance_getter, str):
        if importance_getter == "permutation":
            if target is None:
                raise ValueError(
                    "importance_getter='permutation' requires target (y_test) "
                    "to score against. Pass target= explicitly."
                )
            from sklearn.inspection import permutation_importance
            # Forward the function's own n_repeats + caller-supplied random_state
            # (default 0 preserves legacy behaviour). Pre-fix these were hardcoded
            # n_repeats=5 / random_state=0, so the n_repeats kwarg was dead on this
            # path and the caller's seed never reached the permutation shuffles.
            pi = permutation_importance(
                model, data, target,
                n_repeats=n_repeats,
                random_state=random_state,
                n_jobs=1,
            )
            res = pi.importances_mean
        elif importance_getter == "conditional_permutation":
            if target is None:
                raise ValueError(
                    "importance_getter='conditional_permutation' requires target (y_test) "
                    "to score against. Pass target= explicitly."
                )
            # F10 (Wave 3, 2026-05-28): cpi_max_depth=None lets the auxiliary tree grow
            # until min_samples_leaf constraint kicks in (Strobl 2008 recommendation).
            # random_state forwarded (default 0 preserves legacy behaviour); lets
            # the caller thread a fold-derived seed so CPI shuffles vary per fold
            # and the run stays reproducible, matching the 'permutation' path.
            res = _conditional_permutation_importance(
                model, data, target,
                n_repeats=n_repeats,
                max_depth=cpi_max_depth,
                min_samples_leaf=cpi_min_samples_leaf,
                random_state=random_state,
            )
        elif importance_getter == "drop_column":
            # Drop-column importance: refit ``model`` on data with each column
            # individually dropped, measure score drop vs full-X baseline.
            # O(p * full_fit_time) -- infeasible on p>=1000. Useful as a
            # ground-truth oracle when benchmarking other importance methods.
            if data is None or target is None:
                raise ValueError(
                    "importance_getter='drop_column' requires data (X) and target (y) at the call site."
                )
            from sklearn.base import clone as _clone
            _Xnp = data.to_numpy(copy=False) if hasattr(data, "to_numpy") else np.asarray(data)
            _baseline = float(model.score(data, target))
            _scores = np.zeros(_Xnp.shape[1], dtype=float)
            for _j in range(_Xnp.shape[1]):
                _X_drop = np.delete(_Xnp, _j, axis=1)
                if hasattr(data, "columns"):
                    _X_drop = pd.DataFrame(_X_drop, columns=[c for i, c in enumerate(data.columns) if i != _j])
                _m = _clone(model)
                try:
                    _m.fit(_X_drop, target)
                    _scores[_j] = _baseline - float(_m.score(_X_drop, target))
                except Exception:
                    _scores[_j] = 0.0
            res = _scores
        elif importance_getter == "boruta":
            # Classical Boruta (Kursa & Rudnicki 2010, JSS-36): pair each real
            # feature with a SHADOW (shuffled copy) and judge importance vs
            # the max-shadow importance. No external dep; uses the supplied
            # ``model``'s feature_importances_ / coef_ after refitting on
            # [X, X_shadow]. Pure-Gini variant -- biased on high-cardinality
            # categoricals. Use 'boruta_shap' for the SHAP-based unbiased
            # version when shap is available.
            if data is None or target is None:
                raise ValueError(
                    "importance_getter='boruta' requires data (X) and target (y) at the call site."
                )
            from sklearn.base import clone as _clone
            rng = np.random.default_rng(0)
            _Xnp = data.to_numpy(copy=False) if hasattr(data, "to_numpy") else np.asarray(data)
            _p = _Xnp.shape[1]
            # Shadow = column-wise shuffled copy.
            _Xshadow = _Xnp.copy()
            for _j in range(_p):
                rng.shuffle(_Xshadow[:, _j])
            _Xjoint = np.hstack([_Xnp, _Xshadow])
            _model_clone = _clone(model)
            try:
                _model_clone.fit(_Xjoint, target)
            except Exception as _exc:
                raise RuntimeError(
                    f"Boruta refit on [X, shadow] failed for {type(model).__name__}: {_exc}"
                ) from _exc
            # Read importances of joint model.
            if hasattr(_model_clone, "feature_importances_"):
                _imps = np.asarray(_model_clone.feature_importances_)
            elif hasattr(_model_clone, "coef_"):
                _imps = np.abs(np.asarray(_model_clone.coef_))
                if _imps.ndim > 1:
                    _imps = _imps.max(axis=0)
            else:
                raise AttributeError(
                    f"'boruta' importance_getter requires feature_importances_ or coef_ on the refit model; "
                    f"{type(_model_clone).__name__} has neither."
                )
            _real = _imps[:_p]
            _shadow_max = float(_imps[_p:].max()) if len(_imps) > _p else 0.0
            # Per-feature score = real importance MINUS shadow-max threshold.
            # Positive = beats shadow; negative = noise. Caller's downstream
            # consumer treats it like any FI vector.
            res = _real - _shadow_max
        elif importance_getter == "boruta_shap":
            # L1 (Wave 5, 2026-05-28): Boruta-SHAP via the optional
            # ``BorutaShap`` package. Returns per-feature shadow-relative
            # importance: positive => beats max-shadow at the configured
            # p-value level; zero => indistinguishable from shadow.
            if target is None:
                raise ValueError("importance_getter='boruta_shap' requires target (y_test).")
            # data (X) is required: BorutaShap.fit needs the feature matrix.
            # Pre-fix this path fell back to ``X=target`` when data was None,
            # feeding y in as the feature matrix and silently producing a
            # nonsensical fit instead of a clear error.
            if data is None:
                raise ValueError("importance_getter='boruta_shap' requires data (X) at the call site.")
            try:
                from BorutaShap import BorutaShap as _BorutaShap
            except ImportError as _exc2:
                # No arfs/GrootCV fallback: GrootCV's constructor + fit signature
                # (GrootCV(objective=, cutoff=).fit(X, y) -> .selected_features_)
                # is incompatible with the BorutaShap call shape below, so
                # aliasing it would crash rather than degrade gracefully.
                raise ImportError(
                    "importance_getter='boruta_shap' requires the optional ``BorutaShap`` package. "
                    "Install via ``pip install BorutaShap``."
                ) from _exc2
            try:
                _bs = _BorutaShap(model=model, importance_measure="shap", classification=hasattr(model, "classes_"))
                _bs.fit(X=data, y=target, n_trials=15, random_state=0, verbose=False)
                # Output: BorutaShap stores accepted/rejected lists; build dense importance with shadow-relative scores.
                _accepted = set(getattr(_bs, "accepted", []) or [])
                _tentative = set(getattr(_bs, "tentative", []) or [])
                res = np.array([1.0 if c in _accepted else (0.5 if c in _tentative else 0.0)
                                for c in current_features], dtype=float)
            except Exception as _exc:
                raise RuntimeError(f"BorutaShap failed: {_exc}") from _exc
        elif importance_getter == "powershap":
            # L2 (Wave 5, 2026-05-28): PowerSHAP via optional ``powershap`` pkg.
            if target is None:
                raise ValueError("importance_getter='powershap' requires target (y_test).")
            try:
                from powershap import PowerShap as _PowerShap
            except ImportError as _exc:
                raise ImportError(
                    "importance_getter='powershap' requires the optional ``powershap`` package. "
                    "Install via ``pip install powershap``."
                ) from _exc
            try:
                _ps = _PowerShap(model=model)
                _ps.fit(data, target)
                # _ps stores _processed_shaps_df with p-values per feature; treat selected -> 1, else 0.
                _sel = set(_ps.selected_features_) if hasattr(_ps, "selected_features_") else set()
                res = np.array([1.0 if c in _sel else 0.0 for c in current_features], dtype=float)
            except Exception as _exc:
                raise RuntimeError(f"PowerSHAP failed: {_exc}") from _exc
        elif importance_getter in ("shap", "shap_oof"):
            # L4 (Wave 5, 2026-05-28): 'shap_oof' is an explicit alias for
            # 'shap'. The standard RFECV fold path fits the model on
            # X_train then calls this with data=X_test (held-out fold),
            # so the resulting mean(|SHAP|) is already an OOF importance.
            # The alias name makes that semantic explicit for callers who
            # want SHAP-OOF elimination without having to read the source.
            try:
                import shap as _shap
            except ImportError as _exc:
                raise ImportError(
                    f"importance_getter={importance_getter!r} requires the optional "
                    f"``shap`` package. Install via ``pip install shap``."
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
            # 2026-05-28 sklearn-parity: ``importance_getter`` may be a dotted
            # path such as ``regressor_.coef_`` (for TransformedTargetRegressor)
            # or ``named_steps.lr.coef_`` (for Pipeline). Resolve via
            # operator.attrgetter so the legacy single-attr behaviour AND the
            # dotted path both work. ``auto`` also unwraps Pipelines and other
            # wrappers to the underlying ``_final_estimator`` / ``regressor_``
            # before searching for feature_importances_ / coef_.
            def _unwrap_estimator(m):
                # Walk to the innermost fitted estimator: Pipeline -> _final_estimator;
                # TransformedTargetRegressor -> regressor_; ColumnTransformer-style
                # wrappers fall back to themselves.
                for _ in range(8):
                    if hasattr(m, "_final_estimator") and m._final_estimator is not m:
                        m = m._final_estimator
                        continue
                    if hasattr(m, "regressor_") and getattr(m, "regressor_") is not m:
                        m = m.regressor_
                        continue
                    if hasattr(m, "best_estimator_") and getattr(m, "best_estimator_") is not m:
                        m = m.best_estimator_
                        continue
                    break
                return m

            def _resolve_getter(obj, dotted: str):
                # operator.attrgetter handles the dotted-path traversal.
                from operator import attrgetter
                return attrgetter(dotted)(obj)

            if importance_getter == "auto":
                inner = _unwrap_estimator(model)
                if hasattr(inner, "feature_importances_"):
                    res = inner.feature_importances_
                    getter_attr = "feature_importances_"
                elif hasattr(inner, "coef_"):
                    res = inner.coef_
                    getter_attr = "coef_"
                else:
                    raise AttributeError(
                        f"importance_getter='auto' could not find feature_importances_ or coef_ "
                        f"on a fitted {type(model).__name__} (unwrapped to {type(inner).__name__})."
                    )
            else:
                getter_attr = importance_getter
                # Dotted path (sklearn convention).
                if "." in importance_getter:
                    try:
                        res = _resolve_getter(model, importance_getter)
                    except (AttributeError, KeyError) as _attr_exc:
                        raise AttributeError(
                            f"importance_getter={importance_getter!r}: could not resolve dotted "
                            f"path on {type(model).__name__}. Verify each step exists on the "
                            f"fitted estimator. Underlying error: {_attr_exc}"
                        ) from _attr_exc
                    # Normalise getter_attr to the LAST segment for downstream coef_-scaling logic.
                    getter_attr = importance_getter.rsplit(".", 1)[-1]
                else:
                    res = getattr(model, getter_attr)
            if getter_attr == "coef_":
                # F5 (Wave 3, 2026-05-28): multi-class collapse. 'max' (default)
                # uses max(|coef_class|, axis=0) -> a feature important for ANY
                # class is important. Pre-fix sum(|coef|) over OvR rows mixed
                # class-specific signals: a single-class discriminator looked
                # like a mid-relevance feature for every class. 'sum' is opt-in.
                res = np.abs(res)
                if res.ndim > 1:
                    if multiclass_coef_aggregation == "max":
                        res = res.max(axis=0)
                    else:
                        res = res.sum(axis=0)
                # F4 (Wave 3, 2026-05-28): scale correction with TRAIN stds.
                # Pre-fix used X_test stds -> leaks test variance into FI on small
                # folds. Use train_data when provided; fall back to data only if
                # train_data is absent (callable importance_getter path) and
                # coef_scale_source != 'none'.
                _scale_src = coef_scale_source
                if _scale_src == "none":
                    pass
                else:
                    _src_data = train_data if (train_data is not None and _scale_src == "train") else data
                    if _src_data is not None:
                        try:
                            if hasattr(_src_data, "values"):
                                _Xarr = _src_data.values
                            else:
                                _Xarr = np.asarray(_src_data)
                            _stds = np.nanstd(_Xarr, axis=0)
                            # Avoid blow-up on near-constant cols (stds ~ 0).
                            _stds = np.where(_stds > 1e-12, _stds, 1.0)
                            if len(_stds) == len(res):
                                res = res * _stds
                        except (TypeError, ValueError):
                            # Non-numeric data (object cols, mixed pl frames): skip scaling.
                            pass
            elif res.ndim > 1:
                # Tree-based feature_importances_ stays 1-D normally; this branch
                # handles unusual estimators (e.g. multi-output) and uses sum
                # because we can't distinguish "OvR class" from "output" here.
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


def select_appropriate_feature_importances(
    feature_importances: dict,
    nfeatures: int,
    n_original_features: int,
    use_all_fi_runs: bool = True,
    use_last_fi_run_only: bool = False,
    use_one_freshest_fi_run: bool = False,
    use_fi_ranking: bool = False,
    votes_aggregation_method: Union[VotesAggregation, None] = None,
) -> dict:
    if use_last_fi_run_only:
        fi_to_consider = {key: value for key, value in feature_importances.items() if len(value) == n_original_features}
    else:
        if use_all_fi_runs:
            fi_to_consider = {key: value for key, value in feature_importances.items() if len(value) > 1} if n_original_features > 1 else feature_importances
        else:
            if use_one_freshest_fi_run:
                # Upper bound is inclusive (n_original_features + 1) so the
                # FI run on the full feature set is also considered.
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
        # F12 (Wave 3, 2026-05-28): rank-based aggregation rules (Borda /
        # Copeland / Dowdall / Minimax / Plurality) internally rank the
        # input table anyway; pre-ranking it here is a no-op for them and
        # only adds tiebreaker-method drift (.rank default 'average' vs
        # Leaderboard's 'min'/'max'). Skip pre-ranking when the downstream
        # aggregator is itself rank-based.
        _rank_based = {
            VotesAggregation.Borda,
            VotesAggregation.Copeland,
            VotesAggregation.Dowdall,
            VotesAggregation.Minimax,
            VotesAggregation.Plurality,
        } if votes_aggregation_method is not None else set()
        if votes_aggregation_method in _rank_based:
            pass  # downstream Leaderboard handles ranking
        else:
            fi_to_consider = {key: pd.Series(value).rank(ascending=True, pct=True).to_dict() for key, value in fi_to_consider.items()}
    return fi_to_consider


def _impute_ragged_fi_table(table: pd.DataFrame, policy: str) -> pd.DataFrame:
    """F1+F2+F3 (Wave 1, 2026-05-28): impute missing per-run FI entries in a ragged voting table BEFORE handing it to Leaderboard.

    The historical RFECV vote let pandas' NaN propagate into Borda / Dowdall / Copeland / Minimax / Plurality with skipna=True semantics
    that systematically biased toward late-surviving features (a feature voting in 30/30 runs sums over 30 columns vs a feature voting in
    3/30 runs that sums over 3). AM/GM/OG already fill with the column median upstream; the other rules did not. Different rules + different
    pre-fixes = unpredictable user-facing behaviour. We now normalise the ragged table at the WRAPPER layer so every rule sees the same
    completed input.

    policy:
        'worst'  : missing -> min(col) - eps for each column. A feature absent from run K is treated as "ranked LAST in run K" by every
                   downstream rule. This matches the operator intuition that elimination at iter N means the feature lost the iter-N comparison.
        'median' : missing -> column median. Pre-2026 default for AM/GM/OG, generalised. Lets re-appearing features keep "average" treatment.
        'skip'   : raw pre-fix table (back-compat A/B path).
    """
    if policy == "skip":
        return table
    if not isinstance(table, pd.DataFrame) or table.empty or not table.isna().to_numpy().any():
        return table
    if policy == "worst":
        # For each column, the imputed value sits strictly below the smallest observed FI so the "missing -> last rank" guarantee
        # holds even on ties: the eps is scaled to the column range so a constant column doesn't collapse the gap.
        out = table.copy()
        col_min = out.min(axis=0, skipna=True)
        col_max = out.max(axis=0, skipna=True)
        col_range = (col_max - col_min).fillna(0.0)
        # Treat zero-range columns (every present value identical) as needing a finite eps so the imputed value still sorts strictly below.
        col_eps = col_range.where(col_range > 0.0, other=1.0) * 1e-3
        fill = col_min - col_eps
        # Any column whose every value is NaN cannot be imputed from itself; fall back to a global floor below the table-wide min.
        all_nan_cols = fill.isna()
        if all_nan_cols.any():
            global_floor = float(table.min(skipna=True).min(skipna=True))
            if not np.isfinite(global_floor):
                global_floor = 0.0
            fill = fill.where(~all_nan_cols, other=global_floor - 1.0)
        out = out.fillna(fill)
        return out
    if policy == "median":
        return table.fillna(table.median())
    raise ValueError(f"_impute_ragged_fi_table: unknown policy={policy!r}")


def get_actual_features_ranking(feature_importances: dict, votes_aggregation_method: VotesAggregation, fi_missing_policy: str = "worst", run_weights: Union[dict, None] = None) -> list:
    """Vote-based rank of features given per-run importances.

    Borda/AM/GM/Dowdall use only ranks (cheap). Copeland needs majority_graph,
    which Leaderboard builds lazily.

    Args:
        feature_importances: dict[run_key -> dict[feature -> importance]].
        votes_aggregation_method: rule from VotesAggregation enum.
        fi_missing_policy: how to complete ragged-NaN table (see _impute_ragged_fi_table).
        run_weights: optional dict[run_key -> weight] for F8 exponential-decay
            over FI history. When None, all runs vote with equal weight
            (legacy). Leaderboard normalises so the absolute scale doesn't
            matter; the RATIO between newer and older runs is what shifts
            the final ranking.

    F7 (Wave 3, 2026-05-28) tie-breaker: when two features end the rule with
    identical Leaderboard scores (very common on tree FI with many zeros),
    fall back to lexicographic ordering by feature name so the output is
    fully deterministic across Python set/dict iteration orders.
    """
    table = pd.DataFrame(feature_importances)
    table = _impute_ragged_fi_table(table, policy=fi_missing_policy)
    # F8: forward run_weights into Leaderboard. Leaderboard normalises by sum
    # so we just pass the float weights; the rule code multiplies per-column.
    if run_weights:
        lb = Leaderboard(table=table, weights=dict(run_weights))
    else:
        lb = Leaderboard(table=table)
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
    # F7 tie-breaker: lexicographic on feature name. Without this the order
    # of equal-rank features depends on Leaderboard's internal sort and on
    # the order of the original dict keys -> different runs of the SAME
    # input pick different "top N" features on tie-clusters.
    _scores = ranks.to_dict()
    out = sorted(_scores.keys(), key=lambda k: (-float(_scores[k]) if np.isfinite(_scores[k]) else 0.0, str(k)))
    return out


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
    fi_missing_policy: str = "worst",
    dichotomic_epsilon: float = 0.0,
    rng: object = None,
    fi_decay_rate: float = 0.0,
    fi_run_order: Union[list, None] = None,
) -> list:
    """Generate the next 'next_nfeatures_to_check' candidate to evaluate.
    Combines FIs from prior runs into ranks via voting, returns the top-N."""
    if nsteps == 0:
        return original_features

    # +1 on the upper bound includes the all-features candidate.
    remaining = list(set(np.arange(1, len(original_features) + 1)) - set(evaluated_scores_mean.keys()))
    if len(remaining) == 0:
        return []

    if top_predictors_search_method == OptimumSearch.ExhaustiveRandom:
        next_nfeatures_to_check = random.choice(remaining)
    elif top_predictors_search_method == OptimumSearch.ModelBasedHeuristic:
        next_nfeatures_to_check = Optimizer.suggest_candidate()
    elif top_predictors_search_method == OptimumSearch.ExhaustiveDichotomic:
        next_nfeatures_to_check = _suggest_dichotomic(
            remaining=sorted(remaining),
            evaluated_scores_mean=evaluated_scores_mean,
            n_total=len(original_features),
            epsilon=float(dichotomic_epsilon),
            rng=rng,
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
            _run_weights = {
                k: (1.0 - _decay) ** (_n_runs - 1 - i)
                for i, k in enumerate(_ordered_keys)
            }
    ranks = get_actual_features_ranking(
        feature_importances=fi_to_consider,
        votes_aggregation_method=votes_aggregation_method,
        fi_missing_policy=fi_missing_policy,
        run_weights=_run_weights,
    )
    return ranks[:next_nfeatures_to_check]


def _suggest_dichotomic(remaining: list, evaluated_scores_mean: dict,
                         n_total: int, epsilon: float = 0.0,
                         rng: object = None) -> int:
    """Binary-search-style suggester for ExhaustiveDichotomic.

    With one or zero evaluations: probe the midpoint of the full
    feature range. With >=2 evaluations: identify the highest-scoring
    evaluated N and probe the midpoint between it and the nearest
    unevaluated neighbour. Falls back to picking the unevaluated N
    closest to the global midpoint if nothing useful from history.

    S6 (Wave 2, 2026-05-28): when ``epsilon > 0`` and rng samples a
    Bernoulli(epsilon) success, pick a random unevaluated N OUTSIDE the
    neighbourhood of the current best (gap > p/4). Prevents the
    classic two-plateau hill-climb trap. ``rng=None`` uses module's
    ``random`` for determinism-via-seed only when explicitly passed.
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
        return min(remaining, key=lambda n: abs(n - target))
    best_evaluated = max(evaluated_scores_mean.items(), key=lambda kv: kv[1])[0]
    # Pick whichever side has the wider gap (more information).
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
        return min(remaining, key=lambda n: abs(n - best_evaluated))
    target = max(candidates, key=lambda gc: gc[0])[1]
    return min(remaining, key=lambda n: abs(n - target))


def _suggest_scipy_local(remaining: list, evaluated_scores_mean: dict,
                         n_total: int, epsilon: float = 0.0, rng=None) -> int:
    """S5 (Wave 2, 2026-05-28): retained as a thin alias for ExhaustiveDichotomic.

    The previous implementation built a piecewise-linear interpolant over evaluated points and ran scipy's ``minimize_scalar`` on it.
    The argmax of a piecewise-linear function is ALWAYS one of its breakpoints (= ``xs`` = already-evaluated N's), so the scipy
    optimiser collapses to "nearest unevaluated N to an already-evaluated one" -- exactly what dichotomic returns, at the cost of a
    scipy import + roundtrip. We now delegate to dichotomic with optional epsilon kick; users keep the OptimumSearch.ScipyLocal enum
    value to avoid silent API breakage in pickled configs.
    """
    return _suggest_dichotomic(remaining, evaluated_scores_mean, n_total, epsilon=epsilon, rng=rng)


def _suggest_scipy_global(remaining: list, evaluated_scores_mean: dict,
                          n_total: int, epsilon: float = 0.0, rng=None) -> int:
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
from ._knockoffs import (  # noqa: E402, F401
    make_gaussian_knockoffs,
    select_features_fdr,
    knockoff_importance,
)
