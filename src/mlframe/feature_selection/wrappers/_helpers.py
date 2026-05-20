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


def _detect_multithreaded(estimator: object) -> bool:
    """True if the estimator's class name matches a known multi-threaded pattern."""
    name = type(estimator).__name__
    return any(p in name for p in _MULTITHREADED_ESTIMATOR_PATTERNS)


def _pin_threads_to_one(estimator: object) -> None:
    """Best-effort: set every known thread-count param to 1 on ``estimator``."""
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


def make_gaussian_knockoffs(X, random_state=None, sdp_solve: bool = False) -> np.ndarray:
    """Generate fixed-design Gaussian knockoffs (Barber-Candes 2015).

    For each X_j a knockoff X_tilde_j is produced that has the same
    correlation with X_{-j} as X_j does, but is independent of y. This
    lets us identify 'real' importance: a feature is selected if its
    importance >> its knockoff's importance under the same fitted model.

    Equicorrelated construction (default): s_j = s for all j, with
    s = min(2 * lambda_min(Sigma), 1) where Sigma = corr(X). This is the
    cheap closed-form path; the SDP-based optimal s requires cvxpy and
    gives slightly tighter knockoffs but is rarely worth the dependency.

    Parameters
    ----------
    X : ndarray (n, p)
        Numeric design matrix. Will be standardized internally; the
        returned X_tilde matches the standardized scale.
    random_state : int or None
        Seed for the noise injection.
    sdp_solve : bool
        Reserved for future SDP-based s; currently raises
        NotImplementedError if True.

    Returns
    -------
    X_tilde : ndarray (n, p)
        Standardized knockoff matrix, same shape as X.
    """
    if sdp_solve:
        raise NotImplementedError("SDP knockoffs not yet implemented; use equicorrelated default.")

    rng = np.random.default_rng(random_state)
    X_arr = np.asarray(X, dtype=float)
    n, p = X_arr.shape
    if n < 2 or p < 1:
        raise ValueError(f"X must have at least 2 rows and 1 column; got {X_arr.shape}")

    # Standardise X (zero mean, unit variance per column) so Sigma is correlation
    means = np.nanmean(X_arr, axis=0)
    stds = np.nanstd(X_arr, axis=0)
    stds = np.where(stds > 1e-12, stds, 1.0)
    X_std = (X_arr - means) / stds
    # Replace any NaNs (from constant columns) with 0
    X_std = np.where(np.isnan(X_std), 0.0, X_std)

    # Sigma = correlation matrix. Tiny ridge keeps it positive definite on near-collinear inputs.
    Sigma = (X_std.T @ X_std) / max(1, n - 1)
    Sigma = Sigma + 1e-8 * np.eye(p)

    # Equicorrelated s: s_j = s for all j, where s = min(2*lambda_min, 1).
    eigvals = np.linalg.eigvalsh(Sigma)
    lam_min = float(max(eigvals[0], 1e-8))
    # When Sigma is near-singular (anti-correlated pairs X_j = -X_k or 100% collinear copies),
    # lam_min ~ 1e-8 -> s_val ~ 2e-8 -> X_tilde becomes ~ X (self-corr ~ 1, useless as knockoff).
    # Warn so the user knows knockoffs won't help here.
    if lam_min < 1e-4:
        logger.warning(
            "make_gaussian_knockoffs: input correlation matrix has "
            "lambda_min=%.2e (near-singular); knockoffs will be near-copies "
            "of original features (self-corr ~ 1) and W statistics ~ 0. "
            "Reduce collinearity (drop duplicates / use feature_groups) "
            "or use stability_selection instead.",
            lam_min,
        )
    s_val = min(2.0 * lam_min, 1.0) * 0.99  # 0.99 buffer for numerical PSD
    s = np.full(p, s_val)

    # Knockoff construction:
    #   X_tilde = X_std (I - Sigma^{-1} diag(s)) + Z C^T
    # where C C^T = 2 diag(s) - diag(s) Sigma^{-1} diag(s) (must be PSD).
    # Pseudo-inverse is a safety fallback on near-singular Sigma.
    try:
        Sigma_inv = np.linalg.inv(Sigma)
    except np.linalg.LinAlgError:
        Sigma_inv = np.linalg.pinv(Sigma)
    diag_s = np.diag(s)
    A = np.eye(p) - Sigma_inv @ diag_s

    M = 2.0 * diag_s - diag_s @ Sigma_inv @ diag_s
    # Ensure M is symmetric PSD numerically; add ridge if needed.
    M = 0.5 * (M + M.T)
    eigvals_M = np.linalg.eigvalsh(M)
    if eigvals_M[0] < 0:
        M = M + (-eigvals_M[0] + 1e-8) * np.eye(p)
    C = np.linalg.cholesky(M)

    Z = rng.standard_normal((n, p))
    X_tilde_std = X_std @ A + Z @ C.T

    # Return on the original scale; estimators don't typically standardise themselves.
    X_tilde = X_tilde_std * stds + means
    return X_tilde


def select_features_fdr(W: dict, q: float = 0.1) -> list:
    """Barber-Candes FDR-controlled feature selection from a knockoff W
    statistic dict.

    Picks features with W_j >= tau, where
        tau = min{t > 0 : (1 + #{j: W_j <= -t}) / max(1, #{j: W_j >= t}) <= q}
    The probability that a noise feature is in the selected set is bounded
    by q (Barber & Candes 2015, Theorem 1). Returns [] if no threshold
    achieves the target FDR (typical on small n / weak signal).

    Parameters
    ----------
    W : dict
        Mapping feature_name -> W_j statistic (output of
        ``knockoff_importance``).
    q : float
        Target FDR in (0, 1). Lower = more conservative selection.

    Returns
    -------
    list of feature names with W_j >= tau, sorted by W_j desc.
    """
    if not W:
        return []
    if not (0.0 < q < 1.0):
        raise ValueError(f"q must be in (0, 1); got {q}")
    abs_W = np.array([abs(v) for v in W.values()])
    candidates = sorted(set(abs_W[abs_W > 0]))
    tau = float("inf")
    for t in candidates:
        n_neg = sum(1 for v in W.values() if v <= -t)
        n_pos = sum(1 for v in W.values() if v >= t)
        ratio = (1 + n_neg) / max(1, n_pos)
        if ratio <= q:
            tau = t
            break
    if not np.isfinite(tau):
        return []
    selected = [(n, v) for n, v in W.items() if v >= tau]
    # Wave 58 (2026-05-20): secondary key on feature name; tied |W| (shrinkage
    # saturation) no longer makes downstream [:topN] slicing drift.
    selected.sort(key=lambda kv: (-kv[1], kv[0]))
    return [n for n, _ in selected]


def knockoff_importance(model_factory, X, y, current_features=None, random_state=None,
                        importance_getter: str = "auto") -> dict:
    """Compute knockoff-based importance: W_j = imp(X_j) - imp(X_tilde_j).

    Builds Gaussian knockoffs X_tilde, fits a fresh model on [X, X_tilde]
    (2p columns), reads the importance of each REAL feature j and its
    KNOCKOFF j, returns the difference. Real features driving y will have
    W_j >> 0; noise features have W_j ~ N(0, sigma) symmetric around 0.

    Parameters
    ----------
    model_factory : callable
        ``model_factory()`` must return a fresh unfitted estimator. We don't
        accept a pre-fit model because knockoffs need a fit on [X, X_tilde].
    X : DataFrame or ndarray (n, p)
    y : array-like (n,)
    current_features : list of feature names (optional)
        If None, defaults to X.columns or range(p).
    random_state : int or None
    importance_getter : same semantics as get_feature_importances

    Returns
    -------
    Dict mapping feature_name -> W_j (knockoff statistic).
    """
    is_df = hasattr(X, "columns")
    X_arr = X.values if is_df else np.asarray(X)
    n, p = X_arr.shape
    if current_features is None:
        current_features = list(X.columns) if is_df else list(range(p))

    X_tilde = make_gaussian_knockoffs(X_arr, random_state=random_state)
    X_joint = np.hstack([X_arr, X_tilde])
    real_names = [f"_real_{i}" for i in range(p)]
    fake_names = [f"_fake_{i}" for i in range(p)]
    if is_df:
        X_joint_df = pd.DataFrame(X_joint, columns=real_names + fake_names, index=X.index)
    else:
        X_joint_df = X_joint

    model = model_factory()
    model.fit(X_joint_df, y)

    fi = get_feature_importances(
        model=model,
        current_features=real_names + fake_names,
        data=X_joint_df,
        target=y,
        importance_getter=importance_getter,
    )
    W = {}
    for i, fname in enumerate(current_features):
        imp_real = float(fi.get(f"_real_{i}", 0.0))
        imp_fake = float(fi.get(f"_fake_{i}", 0.0))
        W[fname] = imp_real - imp_fake
    return W


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
    max_depth: int = 5,
    random_state: int = 0,
) -> np.ndarray:
    """Strobl, Boulesteix, Zeileis, Hothorn 2008 conditional permutation importance.

    Vanilla permutation (Breiman 2001) shuffles X_j independently of X_{-j},
    creating out-of-distribution combinations on correlated feature sets and
    inflating measured importance. This conditional variant fits a shallow
    decision tree X_{-j} -> X_j, then permutes X_j WITHIN each leaf, which
    preserves P(X_j | X_{-j}) and removes the correlation-induced bias.

    Cost: ~2-3x vanilla permutation (per-feature shallow tree fit + n_repeats
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

        if _is_discrete(Xj):
            tree = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
        else:
            tree = DecisionTreeRegressor(max_depth=max_depth, random_state=random_state)

        try:
            tree.fit(Xnotj, Xj)
            leaves = tree.apply(Xnotj)
        except (ValueError, TypeError):
            # Conditioning fit failed (constant Xj, all-NaN row, etc.); skip.
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
            score_losses.append(baseline - float(model.score(X_for_score, y)))
        importances[j] = float(np.mean(score_losses))

    return importances


def get_feature_importances(
    model: object,
    current_features: list,
    importance_getter: str | Callable,
    data: pd.DataFrame | np.ndarray | None = None,
    reference_data: pd.DataFrame | np.ndarray | None = None,
    target: pd.Series | np.ndarray | None = None,
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
        elif importance_getter == "conditional_permutation":
            if target is None:
                raise ValueError(
                    "importance_getter='conditional_permutation' requires target (y_test) "
                    "to score against. Pass target= explicitly."
                )
            res = _conditional_permutation_importance(
                model, data, target,
                n_repeats=5,
                max_depth=5,
                random_state=0,
            )
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
                # Multi-class: collapse class axis FIRST (sum |coef_| across classes
                # per feature) before scale-correction.
                res = np.abs(res)
                if res.ndim > 1:
                    res = res.sum(axis=0)
                # coef_ is scale-dependent: if X_j has 100x larger std than X_k,
                # |coef_j| is ~100x smaller for the same effect on y, biasing selection
                # toward small-variance features. Multiply by feature std so the
                # resulting "importance" is the magnitude of effect on y per
                # *standardised* unit of X. data is X_test in the call site; if absent
                # (callable path), fall back to unscaled |coef_|.
                if data is not None:
                    try:
                        if hasattr(data, "values"):
                            _Xarr = data.values
                        else:
                            _Xarr = np.asarray(data)
                        _stds = np.nanstd(_Xarr, axis=0)
                        # Avoid blow-up on near-constant cols (stds ~ 0).
                        _stds = np.where(_stds > 1e-12, _stds, 1.0)
                        if len(_stds) == len(res):
                            res = res * _stds
                    except (TypeError, ValueError):
                        # Non-numeric data (object cols, mixed pl frames): skip scaling.
                        pass
            elif res.ndim > 1:
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
        fi_to_consider = {key: pd.Series(value).rank(ascending=True, pct=True).to_dict() for key, value in fi_to_consider.items()}
    return fi_to_consider


def get_actual_features_ranking(feature_importances: dict, votes_aggregation_method: VotesAggregation) -> list:
    """Vote-based rank of features given per-run importances.

    Borda/AM/GM/Dowdall use only ranks (cheap). Copeland needs majority_graph,
    which Leaderboard builds lazily."""
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
        )
    elif top_predictors_search_method == OptimumSearch.ScipyLocal:
        next_nfeatures_to_check = _suggest_scipy_local(
            remaining=remaining,
            evaluated_scores_mean=evaluated_scores_mean,
            n_total=len(original_features),
        )
    elif top_predictors_search_method == OptimumSearch.ScipyGlobal:
        next_nfeatures_to_check = _suggest_scipy_global(
            remaining=remaining,
            evaluated_scores_mean=evaluated_scores_mean,
            n_total=len(original_features),
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
    )
    ranks = get_actual_features_ranking(feature_importances=fi_to_consider, votes_aggregation_method=votes_aggregation_method)
    return ranks[:next_nfeatures_to_check]


def _suggest_dichotomic(remaining: list, evaluated_scores_mean: dict,
                         n_total: int) -> int:
    """Binary-search-style suggester for ExhaustiveDichotomic.

    With one or zero evaluations: probe the midpoint of the full
    feature range. With >=2 evaluations: identify the highest-scoring
    evaluated N and probe the midpoint between it and the nearest
    unevaluated neighbour. Falls back to picking the unevaluated N
    closest to the global midpoint if nothing useful from history.
    """
    remaining = sorted(remaining)
    if not remaining:
        return None
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
                         n_total: int) -> int:
    """Local (univariate) suggester via scipy ``minimize_scalar``.

    Builds a piecewise-linear interpolant over the evaluated points,
    finds its local maximum within the bracket of evaluated N values
    using ``method='bounded'``, then snaps to the nearest unevaluated
    N. When fewer than 3 evaluated points exist, falls back to the
    dichotomic suggester (scipy needs >= 2 anchors for a bracket).
    """
    remaining = sorted(remaining)
    if not remaining:
        return None
    if len(evaluated_scores_mean) < 3:
        return _suggest_dichotomic(remaining, evaluated_scores_mean, n_total)
    try:
        from scipy.optimize import minimize_scalar
    except ImportError:
        return _suggest_dichotomic(remaining, evaluated_scores_mean, n_total)
    xs = sorted(evaluated_scores_mean.keys())
    ys = [evaluated_scores_mean[k] for k in xs]
    lo, hi = float(xs[0]), float(xs[-1])
    if hi <= lo:
        return _suggest_dichotomic(remaining, evaluated_scores_mean, n_total)

    def _neg_score(x: float) -> float:
        # Piecewise-linear interpolation between evaluated anchors.
        return -float(np.interp(x, xs, ys))

    try:
        res = minimize_scalar(_neg_score, bounds=(lo, hi),
                              method="bounded",
                              options={"xatol": 0.5})
        target = int(round(res.x))
    except Exception:
        return _suggest_dichotomic(remaining, evaluated_scores_mean, n_total)
    return min(remaining, key=lambda n: abs(n - target))


def _suggest_scipy_global(remaining: list, evaluated_scores_mean: dict,
                          n_total: int) -> int:
    """Global suggester via scipy ``differential_evolution`` on a
    spline surrogate of the evaluated points.

    Best when the score-vs-N curve has multiple plateaus or local
    maxima that a local optimiser would miss. Falls back to
    dichotomic when scipy is unavailable or evaluation history is
    insufficient (< 4 points).
    """
    remaining = sorted(remaining)
    if not remaining:
        return None
    if len(evaluated_scores_mean) < 4:
        return _suggest_dichotomic(remaining, evaluated_scores_mean, n_total)
    try:
        from scipy.optimize import differential_evolution
    except ImportError:
        return _suggest_dichotomic(remaining, evaluated_scores_mean, n_total)
    xs = sorted(evaluated_scores_mean.keys())
    ys = [evaluated_scores_mean[k] for k in xs]
    lo, hi = float(xs[0]), float(xs[-1])
    if hi <= lo:
        return _suggest_dichotomic(remaining, evaluated_scores_mean, n_total)

    def _neg_score(xv) -> float:
        x = float(xv[0]) if hasattr(xv, "__len__") else float(xv)
        return -float(np.interp(x, xs, ys))

    try:
        res = differential_evolution(
            _neg_score,
            bounds=[(lo, hi)],
            seed=42, maxiter=12, tol=1e-3, polish=False,
            updating="immediate", popsize=8,
        )
        target = int(round(float(res.x[0])))
    except Exception:
        return _suggest_dichotomic(remaining, evaluated_scores_mean, n_total)
    return min(remaining, key=lambda n: abs(n - target))
