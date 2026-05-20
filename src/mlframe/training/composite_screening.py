"""Screening helpers used by CompositeTargetDiscovery: column extraction (_extract_column_array / _is_numeric_column), correlation guards (_safe_corr / _safe_abs_corr_all / _residualise), mutual-information scoring (_mi_pair_bin / _mi_to_target), tiny-model CV-RMSE rerank (_build_tiny_model / _tiny_cv_rmse_raw_y / _tiny_cv_rmse_y_scale / *_multiseed variants), stratified sampling (_sample_indices). Split out of composite.py so discovery internals are isolated from the screening machinery; composite.py re-exports every symbol below at its bottom for full back-compat."""


from __future__ import annotations

import contextlib
import logging
import math
import os
import sys
import warnings
from timeit import default_timer as timer
from typing import (
    TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple, Union,
)

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone

try:
    import polars as pl  # type: ignore
    _HAS_POLARS = True
except Exception:  # pragma: no cover
    pl = None  # type: ignore
    _HAS_POLARS = False


def _is_polars_df(x: Any) -> bool:
    """ENS-P2-6: prefer explicit isinstance check over duck-typing."""
    return _HAS_POLARS and isinstance(x, pl.DataFrame)


from .composite_estimator import CompositeTargetEstimator, _y_train_clip_bounds
from .composite_transforms import get_transform

if TYPE_CHECKING:
    from .composite_transforms import Transform  # used as forward annotation in _tiny_cv_rmse_y_scale signature

logger = logging.getLogger(__name__)


def _extract_column_array(df: Any, col: str) -> np.ndarray:
    """Pull a single column out as a 1-D float64 ndarray. Polars / pandas
    only -- never materialise a whole-frame conversion."""
    if _is_polars_df(df):
        # Polars Series.to_numpy() already returns an ndarray; the prior
        # np.asarray wrapper allocated a redundant view. copy=False keeps
        # the astype zero-copy when the source dtype already matches.
        return df.get_column(col).to_numpy().astype(np.float64, copy=False)
    if isinstance(df, pd.DataFrame):
        return df[col].to_numpy(dtype=np.float64)
    raise TypeError(
        f"CompositeTargetDiscovery: unsupported df type {type(df).__name__}"
    )


def _is_numeric_column(df: Any, col: str) -> bool:
    """True if ``col`` is numeric in ``df``. Falls back to False on
    error -- discovery skips non-numeric base candidates rather than
    risking a cast bomb on object-dtype columns."""
    try:
        if _is_polars_df(df):
            import polars as pl  # lazy
            dtype = df.schema[col]
            # ``dtype.is_numeric()`` covers Float*, Int*, UInt* (added in
            # polars 0.19); fall back to a hard-coded set on older
            # versions that ship the dtypes but not the helper.
            try:
                return bool(dtype.is_numeric())
            except AttributeError:
                return dtype in {
                    pl.Float32, pl.Float64,
                    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                }
        if isinstance(df, pd.DataFrame):
            return pd.api.types.is_numeric_dtype(df[col])
    except Exception:
        return False
    return False


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation that returns 0.0 (not NaN) on degenerate
    inputs (constant array, all-NaN). Used in the forbidden-base
    near-derived filter where NaN would falsely pass the threshold.

    Implemented via explicit centred-vector dot product instead of
    ``np.corrcoef`` to avoid the 2x2 matrix construction overhead
    (~1.17x faster on 80K rows; benchmarked 2026-05-10).
    """
    finite = np.isfinite(a) & np.isfinite(b)
    n = int(finite.sum())
    if n < 3:
        return 0.0
    a_f = a[finite]
    b_f = b[finite]
    a_dev = a_f - a_f.mean()
    b_dev = b_f - b_f.mean()
    var_a = float(np.dot(a_dev, a_dev))
    var_b = float(np.dot(b_dev, b_dev))
    if var_a < 1e-24 or var_b < 1e-24:
        return 0.0
    return float(np.dot(a_dev, b_dev) / np.sqrt(var_a * var_b))


def _safe_abs_corr_all(
    y: np.ndarray, X: np.ndarray,
) -> np.ndarray:
    """Vectorised ``|corr(y, X[:, j])|`` for all j in one pass.

    Single matrix op gives 2.2x over the per-column loop on dense
    inputs (200 features x 80K rows: 558ms vs 1220ms). Returns 0.0
    for columns whose centred dot is below numerical tolerance.

    NaN handling differs from ``_safe_corr``: rows where ``y`` is
    non-finite are masked GLOBALLY (not per-column). For composite-
    target use this is acceptable because callers gate columns on
    sufficient finite count BEFORE passing them in (see
    ``_filter_features``); per-column NaN masking is reserved for
    the scalar ``_safe_corr`` path.
    """
    y_finite = np.isfinite(y)
    if y_finite.sum() < 3:
        return np.zeros(X.shape[1])
    y_f = y[y_finite]
    X_f = X[y_finite]
    y_dev = y_f - y_f.mean()
    var_y = float(np.dot(y_dev, y_dev))
    if var_y < 1e-24:
        return np.zeros(X.shape[1])
    X_means = X_f.mean(axis=0)
    X_dev = X_f - X_means
    var_X = (X_dev * X_dev).sum(axis=0)
    out = np.zeros(X.shape[1])
    safe = var_X >= 1e-24
    if safe.any():
        cov = X_dev[:, safe].T @ y_dev
        denom = np.sqrt(var_y * var_X[safe])
        out[safe] = np.abs(cov / denom)
    return out


def _residualise(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """OLS-residualise ``y`` against ``x``. Returns ``y - alpha*x - beta``.
    If x is constant, returns ``y - mean(y)``."""
    finite = np.isfinite(y) & np.isfinite(x)
    if finite.sum() < 3 or np.std(x[finite]) < 1e-12:
        out = y.astype(np.float64).copy()
        out -= float(np.mean(out[finite])) if finite.any() else 0.0
        return out
    X = np.column_stack([x[finite].astype(np.float64), np.ones(int(finite.sum()))])
    coef, *_ = np.linalg.lstsq(X, y[finite].astype(np.float64), rcond=None)
    alpha = float(coef[0])
    beta = float(coef[1])
    out = y.astype(np.float64) - alpha * x - beta
    return out


def _mi_pair_bin(
    x: np.ndarray, y: np.ndarray, *, nbins: int,
) -> float:
    """Discrete MI between two 1-D continuous arrays via quantile binning.

    Discretises both axes into ``nbins`` quantile bins (so each bin
    holds ~equal mass), then computes
    ``MI = sum_ij p(i, j) * log(p(i, j) / (p_x(i) * p_y(j)))``
    using the joint frequency table. Equivalent to the bin-based MI
    estimator widely used in feature-selection libraries.

    Tradeoffs vs the kNN Kraskov estimator (sklearn default):

    - **5-10x faster** on n>1000: O(n + nbins^2) vs O(n*log(n))
      kd-tree queries.
    - **Biased low on heavy-tail distributions** because the equal-mass
      bins concentrate rare-tail values into one bin, hiding
      structure.
    - **Less sensitive to small sample size**: the kNN estimator
      becomes unstable below n=50; bin-based stays usable down to
      ~5*nbins rows.

    Implementation notes (engineering-honest, after benchmarking):

    Several optimisation attempts were tried and rejected:

    - **numba JIT of the full pipeline** (commit history: tried with
      both partial-JIT and full-JIT kernels). On n=1000 the JIT
      gives a 2.6x speedup, but on n>=10000 it is *slower* than
      numpy because numpy's sort / searchsorted / bincount are
      SIMD-vectorised C, and numba's JIT'd Python loops cannot
      beat them. Plus a one-shot ~5 s compile cost on first call.
      Production callers always pass mi_sample_n>=20K rows, so
      numpy wins where it matters. Removed.
    - **np.partition instead of np.quantile** for cut edges. The
      single-position partition is 1.5x faster than np.quantile,
      but the multi-position np.partition (one call selecting all
      nbins-1 positions) becomes O(n * nbins) and ends up *slower*
      on n>=100K than np.quantile's optimised sort-based path.
      Reverted to np.quantile.

    Verdict: the numpy implementation here is at the
    speed-of-vectorised-C floor for this algorithm. Further wins
    require dropping to a different algorithm entirely (e.g. a
    streaming hash-bin estimator that avoids the O(n log n) sort
    altogether).
    """
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 5 * nbins:
        return 0.0
    x_f = x[finite]
    y_f = y[finite]
    qs = np.linspace(0, 1, nbins + 1)[1:-1]
    x_edges = np.quantile(x_f, qs)
    y_edges = np.quantile(y_f, qs)
    x_idx = np.searchsorted(x_edges, x_f, side="right").astype(np.int64)
    y_idx = np.searchsorted(y_edges, y_f, side="right").astype(np.int64)
    np.clip(x_idx, 0, nbins - 1, out=x_idx)
    np.clip(y_idx, 0, nbins - 1, out=y_idx)
    combo = x_idx * nbins + y_idx
    joint_counts = np.bincount(combo, minlength=nbins * nbins).reshape(nbins, nbins)
    n_total = float(joint_counts.sum())
    if n_total <= 0:
        return 0.0
    pxy = joint_counts.astype(np.float64) / n_total
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    nz = pxy > 0
    log_terms = np.zeros_like(pxy)
    log_terms[nz] = np.log(pxy[nz] / (px * py)[nz])
    mi = float((pxy * log_terms).sum())
    return max(0.0, mi)


def _prebin_feature_columns(
    feature_matrix: np.ndarray, *, nbins: int,
) -> np.ndarray:
    """Pre-compute quantile edges and bin indices for every feature column.

    Returns an (n_samples, n_features) int64 array of bin indices
    (0..nbins-1, with -1 marking non-finite rows).  Call once per
    screening pass; reuse across all candidate targets via
    ``_mi_to_target_prebinned``.  This saves the ``np.quantile`` +
    ``np.searchsorted`` cost per column per candidate -- for the
    typical 8-candidate × 25-feature sweep, roughly half the MI wall
    time.

    Uses ``np.nanquantile`` so NaN values don't corrupt the cut
    points; non-finite rows get sentinel -1 and are skipped by the
    downstream ``_mi_to_target_prebinned``.
    """
    n_rows, n_cols = feature_matrix.shape
    if n_rows < 5 * nbins:
        return np.full((n_rows, n_cols), -1, dtype=np.int64)
    q_edges = np.linspace(0.0, 1.0, nbins + 1)[1:-1]
    binned = np.empty((n_rows, n_cols), dtype=np.int64)
    for j in range(n_cols):
        col = feature_matrix[:, j]
        col_finite = np.isfinite(col)
        if col_finite.sum() < 5 * nbins:
            binned[:, j] = -1
            continue
        cut = np.nanquantile(col, q_edges)
        col_idx = np.full(n_rows, -1, dtype=np.int64)
        col_idx[col_finite] = np.searchsorted(
            cut, col[col_finite], side="right",
        ).astype(np.int64)
        np.clip(col_idx, 0, nbins - 1, out=col_idx, where=col_idx >= 0)
        binned[:, j] = col_idx
    return binned


def _mi_to_target_prebinned(
    feature_binned: np.ndarray,
    target: np.ndarray,
    *,
    nbins: int,
    aggregation: str = "mean",
) -> float:
    """MI between pre-binned feature columns and ``target``.

    ``feature_binned`` is the output of ``_prebin_feature_columns`` --
    integer bin indices (0..nbins-1, -1 for non-finite rows) with the
    SAME number of rows as ``target``.  Only the target is binned here
    (once), saving the per-column quantile + searchsorted cost.
    """
    if feature_binned.shape[0] == 0 or feature_binned.shape[1] == 0:
        return 0.0
    if feature_binned.shape[0] != target.shape[0]:
        return 0.0
    # Rows valid for MI: target finite AND first feature column not sentinel
    finite = np.isfinite(target) & (feature_binned[:, 0] >= 0)
    if finite.sum() < 5 * nbins:
        return 0.0
    t_f = target[finite]
    fb_f = feature_binned[finite]
    qs = np.linspace(0.0, 1.0, nbins + 1)[1:-1]
    t_edges = np.nanquantile(t_f, qs)
    t_idx = np.searchsorted(t_edges, t_f, side="right").astype(np.int64)
    np.clip(t_idx, 0, nbins - 1, out=t_idx)
    per_feat = np.empty(fb_f.shape[1], dtype=np.float64)
    for j in range(fb_f.shape[1]):
        col_b = fb_f[:, j]
        # Filter out -1 sentinel (non-finite feature rows) for this column
        col_valid = col_b >= 0
        if col_valid.sum() < 5 * nbins:
            per_feat[j] = 0.0
            continue
        per_feat[j] = _mi_from_binned_pair(
            col_b[col_valid], t_idx[col_valid], nbins=nbins,
        )
    if aggregation == "sum":
        return float(np.sum(per_feat))
    return float(np.mean(per_feat))


def _mi_from_binned_pair(
    x_idx: np.ndarray, y_idx: np.ndarray, *, nbins: int,
) -> float:
    """MI from two already-binned integer arrays (0..nbins-1)."""
    combo = x_idx * nbins + y_idx
    joint_counts = np.bincount(combo, minlength=nbins * nbins).reshape(nbins, nbins)
    n_total = float(joint_counts.sum())
    if n_total <= 0:
        return 0.0
    pxy = joint_counts.astype(np.float64) / n_total
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    nz = pxy > 0
    log_terms = np.zeros_like(pxy)
    log_terms[nz] = np.log(pxy[nz] / (px * py)[nz])
    return max(0.0, float((pxy * log_terms).sum()))


def _mi_per_feature_y_fixed(
    feature_matrix: np.ndarray,
    y: np.ndarray,
    *,
    nbins: int,
) -> np.ndarray:
    """Per-feature MI between each column of ``feature_matrix`` and a fixed ``y``.

    Specialises ``_mi_pair_bin`` for the (feature-loop, fixed-target) case
    used by ``CompositeTargetDiscovery._auto_base``: the naive
    ``[_mi_pair_bin(F[:, j], y, nbins) for j in cols]`` re-quantiles ``y``
    on every iteration; this hoists the y-binning out of the loop. The
    per-feature work reduces to ``np.quantile(col) + searchsorted +
    bincount`` -- mathematically identical to the per-call path, so the
    returned per-feature MI is bit-exact vs the naive baseline.

    Inputs are assumed already cross-column finite-masked by the caller
    (matches ``_mi_pair_bin``'s contract when used inside the feature
    loop at ``composite_discovery._auto_base``).

    Benchmark (n=500K, k=30, nbins=50, screening sample from
    ``_profile_fuzz_1m.py`` seed=99): naive 2888.5 ms -> hoisted 1729.3 ms
    => 1.67x faster, bit-exact (max abs diff 0.0).
    """
    n_rows, n_cols = feature_matrix.shape
    out = np.zeros(n_cols, dtype=np.float64)
    if n_rows < 5 * nbins or n_cols == 0:
        return out
    qs = np.linspace(0.0, 1.0, nbins + 1)[1:-1]
    y_edges = np.quantile(y, qs)
    y_idx = np.searchsorted(y_edges, y, side="right").astype(np.int64)
    np.clip(y_idx, 0, nbins - 1, out=y_idx)
    for j in range(n_cols):
        col = feature_matrix[:, j]
        x_edges = np.quantile(col, qs)
        x_idx = np.searchsorted(x_edges, col, side="right").astype(np.int64)
        np.clip(x_idx, 0, nbins - 1, out=x_idx)
        out[j] = _mi_from_binned_pair(x_idx, y_idx, nbins=nbins)
    return out


def _mi_to_target(
    feature_matrix: np.ndarray,
    target: np.ndarray,
    *,
    n_neighbors: int,
    random_state: int,
    estimator: str = "knn",
    nbins: int = 16,
    aggregation: str = "mean",
) -> float:
    """Aggregated MI of each feature column with ``target``.

    Two estimators:

    - ``"knn"`` (default): sklearn's Kraskov kNN estimator. Higher
      accuracy on heavy-tail distributions; slow on n>10k.
    - ``"bin"``: quantile-bin estimator (``_mi_pair_bin`` per column).
      5-10x faster; biased low on heavy-tail.

    Aggregation across feature columns:

    - ``"mean"`` (default since 2026-05-10, R10b stat #1):
      ``sum_j MI(x_j, target) / n_features``. Invariant to feature
      count -- the metric stays comparable when the feature set
      shrinks (because the base column was excluded). Statistician's
      review flagged the legacy ``"sum"`` aggregation as biased:
      sum-of-marginal-MI overcounts shared information when X is
      correlated, AND the over-count differs between numerator
      ``MI(T, X_no_base)`` and denominator ``MI(y, X_no_base)``
      because changing the target reweights how features overlap.
      Mean is the simplest fix that removes the dimension confound.
    - ``"sum"``: legacy behaviour. Set explicitly for backward-
      compatibility on existing benchmarks.
    """
    finite = np.isfinite(target) & np.all(np.isfinite(feature_matrix), axis=1)
    if finite.sum() < 50:
        return 0.0
    target_f = target[finite]
    fm_f = feature_matrix[finite]
    if fm_f.shape[1] == 0:
        return 0.0
    if estimator == "bin":
        per_feat = np.array([
            _mi_pair_bin(fm_f[:, j], target_f, nbins=nbins)
            for j in range(fm_f.shape[1])
        ])
    else:
        from sklearn.feature_selection import mutual_info_regression
        per_feat = mutual_info_regression(
            fm_f, target_f, n_neighbors=n_neighbors, random_state=random_state,
        )
    if aggregation == "sum":
        return float(np.sum(per_feat))
    # Default: mean (statistician #1).
    return float(np.mean(per_feat))


@contextlib.contextmanager
def _silence_tiny_model_output():
    """Context manager: silence the per-fold tiny-model fit/predict
    noise without changing the numeric path (no DataFrame->ndarray
    conversion; we keep the user's frame as-is for performance).

    Suppressed:
    - sklearn ``UserWarning`` for "X has feature names, but X was
      fitted without feature names" / vice versa (we mix ndarray-fit
      with DataFrame-predict on the cross-target ensemble path).
    - sklearn ``ConvergenceWarning`` from Ridge / linear models on
      degenerate folds.
    - ``RuntimeWarning`` from numpy on near-singular regressors.
    - LightGBM "No further splits with positive gain" info messages
      that escape ``verbose=-1`` via the C library (silenced through
      ``logging.getLogger('lightgbm')`` level bump).

    Not touched: errors, structured logging from mlframe itself,
    catboost / xgboost (already silenced via their own kwargs).
    """
    import logging as _logging
    _lgb_logger = _logging.getLogger("lightgbm")
    _prev_level = _lgb_logger.level
    _lgb_logger.setLevel(_logging.ERROR)
    try:
        from sklearn.exceptions import ConvergenceWarning
    except Exception:  # pragma: no cover - sklearn always installed in our deps
        ConvergenceWarning = UserWarning  # type: ignore[assignment]
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*feature names.*",
            category=UserWarning,
        )
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            yield
        finally:
            _lgb_logger.setLevel(_prev_level)


def _build_tiny_model(family: str, *, n_estimators: int, num_leaves: int,
                      learning_rate: float, random_state: int,
                      deterministic: bool = False) -> Any:
    """Lazy-build a tiny regressor for the requested family. Lazy
    imports keep the discovery module light when those libraries
    aren't installed.

    When ``deterministic=True``, inject the well-known per-family
    determinism flags so run-to-run results are bit-exact at a
    5-10% per-fit cost. See ``deterministic_screening_models`` config
    field for the rationale.

    LightGBM determinism set:
    - ``deterministic=True``: forces deterministic histograms +
      bin-construction + tree-learner.
    - ``force_row_wise=True``: row-wise histogram aggregation is
      deterministic; the column-wise default is faster but uses
      atomic adds whose order varies.
    - ``force_col_wise=False``: explicitly OFF; otherwise it overrides
      ``force_row_wise``.

    XGBoost determinism set:
    - ``tree_method="hist"``: explicit hist; the auto-pick may flip
      to ``"approx"`` with non-deterministic atomic ops.
    - ``predictor="auto"``: keep -- predict path is already deterministic.
    - XGB doesn't expose a single ``deterministic`` switch the way
      LGB does; ``hist`` is the deterministic path.

    CatBoost determinism set:
    - ``boosting_type="Plain"``: the ``Ordered`` default is faster
      but uses random ordering which differs run-to-run; ``Plain``
      is deterministic.

    Linear (Ridge) is already deterministic by construction.
    """
    family_lower = family.lower()
    if family_lower in ("lgb", "lightgbm"):
        import lightgbm as lgb
        kwargs = dict(
            n_estimators=n_estimators,
            num_leaves=num_leaves,
            learning_rate=learning_rate,
            random_state=random_state,
            n_jobs=-1, verbose=-1, force_col_wise=True,
        )
        if deterministic:
            # ``force_col_wise`` + ``force_row_wise`` are mutually
            # exclusive in LightGBM; flip the pair when going
            # deterministic.
            kwargs["force_col_wise"] = False
            kwargs["force_row_wise"] = True
            kwargs["deterministic"] = True
        return lgb.LGBMRegressor(**kwargs)
    if family_lower in ("xgb", "xgboost"):
        import xgboost as xgb
        kwargs = dict(
            n_estimators=n_estimators,
            max_depth=4,
            learning_rate=learning_rate,
            random_state=random_state,
            n_jobs=-1, verbosity=0,
        )
        if deterministic:
            kwargs["tree_method"] = "hist"
        return xgb.XGBRegressor(**kwargs)
    if family_lower in ("cb", "catboost"):
        import catboost as cb
        kwargs = dict(
            iterations=n_estimators,
            depth=4,
            learning_rate=learning_rate,
            random_state=random_state,
            verbose=False,
        )
        if deterministic:
            kwargs["boosting_type"] = "Plain"
        return cb.CatBoostRegressor(**kwargs)
    if family_lower in ("linear", "ridge"):
        from sklearn.linear_model import Ridge
        # Ridge is deterministic by construction; the flag is a no-op
        # here but accepting the kwarg keeps the call signature
        # uniform across families.
        return Ridge(alpha=1.0, random_state=random_state)
    raise ValueError(
        f"_build_tiny_model: unknown family '{family}'. "
        "Supported: lightgbm, xgboost, catboost, linear / ridge."
    )


def _tiny_cv_rmse_raw_y(
    y_train: np.ndarray,
    x_train_matrix: np.ndarray,
    *,
    family: str,
    n_estimators: int,
    num_leaves: int,
    learning_rate: float,
    cv_folds: int,
    random_state: int,
    n_jobs: int = 1,
    deterministic: bool = False,
    return_per_bin: bool = False,
    n_bins: int = 5,
    bin_var: np.ndarray | None = None,
    time_aware: bool = False,
    cv_splitter: Any = None,
):
    """CV-RMSE of a tiny model trained DIRECTLY on raw y (no transform).

    Used as the raw-y baseline against which composite-target tiny CV-RMSEs
    are compared in :meth:`CompositeTargetDiscovery._tiny_model_rerank`.
    Composite specs that fail to beat this baseline are rejected -- the
    primary safeguard that catches "wrong base" cases where MI-gain
    passes barely but the resulting target is harder for the model to
    predict than y itself (e.g. subtracting a spatial coordinate that has
    global trend with y but no structural residual signal).

    Same fit / fold / parallelism contract as :func:`_tiny_cv_rmse_y_scale`
    so the comparison is apples-to-apples.
    """
    from sklearn.model_selection import KFold, TimeSeriesSplit
    n = len(y_train)
    if n < cv_folds * 10:
        return float("nan")
    y_clean = y_train.astype(np.float64)
    if not np.all(np.isfinite(y_clean)):
        finite_mask = np.isfinite(y_clean)
        y_clean = y_clean[finite_mask]
        x_clean = x_train_matrix[finite_mask]
    else:
        x_clean = x_train_matrix
    if len(y_clean) < cv_folds * 10:
        return float("nan")

    if cv_splitter is not None:
        kf = cv_splitter
    elif time_aware:
        kf = TimeSeriesSplit(n_splits=cv_folds)
    else:
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    # bin_var aligns to the masked y_clean / x_clean. If caller
    # passed it, mask it the same way.
    if bin_var is not None and len(bin_var) == len(y_train):
        if not np.all(np.isfinite(y_train)):
            bin_var_clean = bin_var[np.isfinite(y_train)]
        else:
            bin_var_clean = bin_var
    else:
        bin_var_clean = None

    def _one_fold(
        train_fold: np.ndarray, val_fold: np.ndarray,
    ) -> tuple[float, np.ndarray | None]:
        try:
            model = _build_tiny_model(
                family,
                n_estimators=n_estimators,
                num_leaves=num_leaves,
                learning_rate=learning_rate,
                random_state=random_state,
                deterministic=deterministic,
            )
            if n_jobs > 1 and hasattr(model, "set_params"):
                try:
                    model.set_params(n_jobs=1)
                except Exception as _njobs_err:
                    # When the set_params raises (custom model, version skew rejecting the
                    # kwarg), every fold's inner model oversubscribes its own threads
                    # against the outer parallel-fold dispatch -- discovery wallclock
                    # blows up 4-8x with no log evidence pre-fix. Surface the model class
                    # so the operator can fix the wrapper that's rejecting n_jobs.
                    logger.warning(
                        "composite_screening: failed to cap n_jobs=1 on inner %s under "
                        "outer n_jobs=%d (parallel oversubscription risk; discovery "
                        "wallclock may regress 4-8x): %s: %s",
                        type(model).__name__, n_jobs, type(_njobs_err).__name__, _njobs_err,
                    )
            with _silence_tiny_model_output():
                model.fit(x_clean[train_fold], y_clean[train_fold])
                y_hat = np.asarray(model.predict(x_clean[val_fold])).reshape(-1)
            diff = y_hat.astype(np.float64) - y_clean[val_fold]
            finite = np.isfinite(diff)
            if finite.sum() == 0:
                return float("nan"), None
            rmse = float(np.sqrt(np.mean(diff[finite] * diff[finite])))
            per_bin = None
            if return_per_bin and bin_var_clean is not None:
                per_bin = _per_bin_rmse(
                    y_clean[val_fold], y_hat,
                    bin_var_clean[val_fold], n_bins=n_bins,
                )
            return rmse, per_bin
        except Exception:
            return float("nan"), None

    splits = list(kf.split(x_clean))
    if n_jobs > 1 and len(splits) > 1:
        try:
            from joblib import Parallel, delayed
            fold_results = Parallel(
                n_jobs=min(n_jobs, len(splits)),
                backend="threading",
            )(delayed(_one_fold)(tr, va) for tr, va in splits)
        except ImportError:
            fold_results = [_one_fold(tr, va) for tr, va in splits]
    else:
        fold_results = [_one_fold(tr, va) for tr, va in splits]
    fold_rmses = [r for r, _ in fold_results if math.isfinite(r)]
    if not fold_rmses:
        if return_per_bin:
            return float("nan"), np.full(n_bins, float("nan"))
        return float("nan")
    mean_rmse = float(np.mean(fold_rmses))
    if not return_per_bin:
        return mean_rmse
    per_bin_arrays = [pb for _, pb in fold_results if pb is not None]
    if not per_bin_arrays:
        return mean_rmse, np.full(n_bins, float("nan"))
    per_bin_stack = np.stack(per_bin_arrays, axis=0)
    with np.errstate(invalid="ignore"):
        per_bin_mean = np.nanmean(per_bin_stack, axis=0)
    return mean_rmse, per_bin_mean


def _tiny_cv_rmse_y_scale_multiseed(
    *args,
    n_seed_repeats: int = 1,
    base_random_state: int = 0,
    return_per_seed: bool = False,
    **kwargs,
):
    """Multi-seed wrapper around :func:`_tiny_cv_rmse_y_scale`.

    R10b improvement #10: with cv_folds=3, a single CV split has
    high variance. Repeat the K-fold split with different random
    seeds and return the MEDIAN of the per-seed mean RMSEs (instead
    of a single point estimate). When ``return_per_bin=True``, also
    returns the per-bin median across seeds.

    R10b statistician #4: when ``return_per_seed=True``, also returns
    the array of per-seed mean RMSEs so callers can run a paired
    Wilcoxon test against a reference (raw-y baseline) array.

    ``n_seed_repeats=1`` is the legacy single-seed path -- exact
    same numerical result as calling the underlying function once.
    """
    if n_seed_repeats <= 1:
        kwargs["random_state"] = base_random_state
        result = _tiny_cv_rmse_y_scale(*args, **kwargs)
        if return_per_seed:
            mean = result[0] if isinstance(result, tuple) else result
            per_seed_arr = np.array(
                [mean] if math.isfinite(mean) else [],
                dtype=np.float64,
            )
            if isinstance(result, tuple):
                return result + (per_seed_arr,)
            return result, per_seed_arr
        return result
    seed_results = []
    seed_per_bins = []
    return_pb = kwargs.get("return_per_bin", False)
    for s_idx in range(n_seed_repeats):
        kwargs["random_state"] = base_random_state + s_idx * 7919
        result = _tiny_cv_rmse_y_scale(*args, **kwargs)
        if return_pb and isinstance(result, tuple):
            mean_rmse, per_bin = result
            if math.isfinite(mean_rmse):
                seed_results.append(mean_rmse)
                seed_per_bins.append(per_bin)
        else:
            if math.isfinite(result):
                seed_results.append(result)
    seed_arr = np.array(seed_results, dtype=np.float64)
    if not seed_results:
        if return_pb:
            res = float("nan"), np.full(kwargs.get("n_bins", 5), float("nan"))
            return res + (seed_arr,) if return_per_seed else res
        return (float("nan"), seed_arr) if return_per_seed else float("nan")
    median_rmse = float(np.median(seed_results))
    if return_pb:
        if seed_per_bins:
            stack = np.stack(seed_per_bins, axis=0)
            with np.errstate(invalid="ignore"):
                median_per_bin = np.nanmedian(stack, axis=0)
        else:
            median_per_bin = np.full(kwargs.get("n_bins", 5), float("nan"))
        if return_per_seed:
            return median_rmse, median_per_bin, seed_arr
        return median_rmse, median_per_bin
    if return_per_seed:
        return median_rmse, seed_arr
    return median_rmse


def _tiny_cv_rmse_raw_y_multiseed(
    *args,
    n_seed_repeats: int = 1,
    base_random_state: int = 0,
    return_per_seed: bool = False,
    **kwargs,
):
    """Multi-seed wrapper around :func:`_tiny_cv_rmse_raw_y`. See
    :func:`_tiny_cv_rmse_y_scale_multiseed` for the rationale."""
    if n_seed_repeats <= 1:
        kwargs["random_state"] = base_random_state
        result = _tiny_cv_rmse_raw_y(*args, **kwargs)
        if return_per_seed:
            mean = result[0] if isinstance(result, tuple) else result
            per_seed_arr = np.array(
                [mean] if math.isfinite(mean) else [],
                dtype=np.float64,
            )
            if isinstance(result, tuple):
                return result + (per_seed_arr,)
            return result, per_seed_arr
        return result
    seed_results = []
    seed_per_bins = []
    return_pb = kwargs.get("return_per_bin", False)
    for s_idx in range(n_seed_repeats):
        kwargs["random_state"] = base_random_state + s_idx * 7919
        result = _tiny_cv_rmse_raw_y(*args, **kwargs)
        if return_pb and isinstance(result, tuple):
            mean_rmse, per_bin = result
            if math.isfinite(mean_rmse):
                seed_results.append(mean_rmse)
                seed_per_bins.append(per_bin)
        else:
            if math.isfinite(result):
                seed_results.append(result)
    seed_arr = np.array(seed_results, dtype=np.float64)
    if not seed_results:
        if return_pb:
            res = float("nan"), np.full(kwargs.get("n_bins", 5), float("nan"))
            return res + (seed_arr,) if return_per_seed else res
        return (float("nan"), seed_arr) if return_per_seed else float("nan")
    median_rmse = float(np.median(seed_results))
    if return_pb:
        if seed_per_bins:
            stack = np.stack(seed_per_bins, axis=0)
            with np.errstate(invalid="ignore"):
                median_per_bin = np.nanmedian(stack, axis=0)
        else:
            median_per_bin = np.full(kwargs.get("n_bins", 5), float("nan"))
        if return_per_seed:
            return median_rmse, median_per_bin, seed_arr
        return median_rmse, median_per_bin
    if return_per_seed:
        return median_rmse, seed_arr
    return median_rmse


def _per_bin_rmse(
    y_true: np.ndarray,
    y_hat: np.ndarray,
    bin_var: np.ndarray,
    n_bins: int = 5,
) -> np.ndarray:
    """RMSE within each quantile-bin of ``bin_var``. Returns
    array of shape ``(n_bins,)``; bins with too few rows return NaN.

    Used by the regime-aware gate to detect specs that beat raw on
    average but underperform within a particular slice of the data
    (e.g. logratio is correct on multiplicative-regime rows but
    actively wrong on additive-regime rows; mean RMSE hides this).
    """
    finite = np.isfinite(y_true) & np.isfinite(y_hat) & np.isfinite(bin_var)
    if finite.sum() < n_bins * 5:
        return np.full(n_bins, float("nan"))
    y_t = y_true[finite]
    y_p = y_hat[finite]
    bv = bin_var[finite]
    qs = np.linspace(0, 1, n_bins + 1)[1:-1]
    edges = np.quantile(bv, qs)
    bin_idx = np.searchsorted(edges, bv, side="right")
    np.clip(bin_idx, 0, n_bins - 1, out=bin_idx)
    out = np.full(n_bins, float("nan"))
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() < 5:
            continue
        diff = y_p[mask] - y_t[mask]
        out[b] = float(np.sqrt(np.mean(diff * diff)))
    return out


def _tiny_cv_rmse_y_scale(
    y_train: np.ndarray,
    base_train: np.ndarray,
    transform: Transform,
    fitted_params: dict[str, Any],
    x_train_matrix: np.ndarray,
    *,
    family: str,
    n_estimators: int,
    num_leaves: int,
    learning_rate: float,
    cv_folds: int,
    random_state: int,
    n_jobs: int = 1,
    deterministic: bool = False,
    return_per_bin: bool = False,
    n_bins: int = 5,
    time_aware: bool = False,
    early_stop_threshold: float = float("inf"),
):
    """Compute CV-RMSE of a tiny model on the y-scale (after inverse).

    1. Apply ``transform.forward`` to (y_train, base_train) -> T.
    2. K-fold split on the train rows. With ``time_aware=True`` the split
       is a sklearn ``TimeSeriesSplit`` -- past-only train / future
       holdout for each fold -- which matches the production ordering
       for autoregressive bases (``TVT_prev``, lag features). Random
       K-fold on a lag base leaks future->past and over-rates the spec.
    3. For each fold: fit tiny model on T_train_fold, predict T_hat
       on the held-out fold, apply transform.inverse to recover
       y_hat in the original scale, score against y_held.
    4. Return mean across folds.

    Folds run in parallel when ``n_jobs > 1`` via joblib. Each fold
    fit gets ``n_jobs_per_fit = max(1, total_cpus // n_jobs)`` cores
    so the inner LightGBM doesn't oversubscribe. NaN if anything
    degenerates so callers can deprioritise.
    """
    from sklearn.model_selection import KFold, TimeSeriesSplit
    n = len(y_train)
    if n < cv_folds * 10:
        return float("nan")
    valid = transform.domain_check(y_train, base_train)
    if valid.sum() < cv_folds * 10:
        return float("nan")
    y_clean = y_train[valid].astype(np.float64)
    base_clean = base_train[valid].astype(np.float64)
    x_clean = x_train_matrix[valid]
    t_clean = transform.forward(y_clean, base_clean, fitted_params)
    if not np.all(np.isfinite(t_clean)):
        return float("nan")

    if time_aware:
        kf = TimeSeriesSplit(n_splits=cv_folds)
    else:
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    def _one_fold(
        train_fold: np.ndarray, val_fold: np.ndarray,
    ) -> tuple[float, np.ndarray | None]:
        """Return (fold_rmse, per_bin_rmse_or_None)."""
        try:
            model = _build_tiny_model(
                family,
                n_estimators=n_estimators,
                num_leaves=num_leaves,
                learning_rate=learning_rate,
                random_state=random_state,
                deterministic=deterministic,
            )
            # When folds run in parallel, cap LightGBM's intra-fit
            # threads to avoid CPU oversubscription.
            if n_jobs > 1 and hasattr(model, "set_params"):
                try:
                    model.set_params(n_jobs=1)
                except Exception as _njobs_err:
                    # Same oversubscription warning as the sibling branch above
                    # (transformed-target variant): without this log, an operator
                    # tracking "discovery wallclock regressed" never connects it
                    # to a silently-failed n_jobs cap on the inner model.
                    logger.warning(
                        "composite_screening (transformed): failed to cap n_jobs=1 on "
                        "inner %s under outer n_jobs=%d (oversubscription risk): %s: %s",
                        type(model).__name__, n_jobs, type(_njobs_err).__name__, _njobs_err,
                    )
            with _silence_tiny_model_output():
                model.fit(x_clean[train_fold], t_clean[train_fold])
                t_hat = np.asarray(model.predict(x_clean[val_fold])).reshape(-1)
            y_hat = transform.inverse(
                t_hat, base_clean[val_fold], fitted_params,
            )
            # R10b improvement #4: wrapper-aware clipping. The
            # production CompositeTargetEstimator.predict applies
            # the same y-clip on inverse output to keep predictions
            # inside a physically plausible range. Mirror that here
            # so screening RMSE matches deployed RMSE (otherwise
            # heavy-tail transforms like logratio look better in
            # screening than they actually deliver).
            y_clip_low, y_clip_high = _y_train_clip_bounds(
                y_clean[train_fold]
            )
            y_hat = np.clip(
                y_hat.astype(np.float64), y_clip_low, y_clip_high,
            )
            # Domain-violation fallback: rows where the transform's
            # domain_check fails on val use y_train_median (matches
            # wrapper.predict). The wrapper computes domain_check on
            # (y, base) but at inference y is unknown -- so the
            # wrapper fallback uses y=None handling. Here on val we
            # know y_clean[val_fold]; emulate the wrapper logic by
            # fall-backing rows where y_hat is non-finite OR where
            # the inverse pushed beyond the clip.
            non_finite = ~np.isfinite(y_hat)
            if non_finite.any():
                y_train_median = float(np.median(
                    y_clean[train_fold][np.isfinite(y_clean[train_fold])]
                )) if np.isfinite(y_clean[train_fold]).any() else 0.0
                y_hat[non_finite] = y_train_median
            diff = y_hat - y_clean[val_fold]
            finite = np.isfinite(diff)
            if finite.sum() == 0:
                return float("nan"), None
            rmse = float(np.sqrt(np.mean(diff[finite] * diff[finite])))
            per_bin = None
            if return_per_bin:
                per_bin = _per_bin_rmse(
                    y_clean[val_fold], y_hat,
                    base_clean[val_fold], n_bins=n_bins,
                )
            return rmse, per_bin
        except Exception:
            return float("nan"), None

    splits = list(kf.split(x_clean))
    if n_jobs > 1 and len(splits) > 1:
        try:
            from joblib import Parallel, delayed
            fold_results = Parallel(
                n_jobs=min(n_jobs, len(splits)),
                backend="threading",
            )(delayed(_one_fold)(tr, va) for tr, va in splits)
        except ImportError:
            fold_results = [_one_fold(tr, va) for tr, va in splits]
    else:
        # Pack #7 serial early-stop: track partial-sum and break when
        # the final mean is GUARANTEED to exceed early_stop_threshold,
        # i.e. ``sum_so_far > early_stop_threshold * cv_folds``. Even if
        # all remaining folds return 0, the mean = sum / cv_folds > thr.
        # Saves 30-66% of fold-fit compute on candidates that the gate
        # will reject anyway.
        fold_results = []
        _sum_so_far = 0.0
        _n_finite_so_far = 0
        for _fi, (tr, va) in enumerate(splits):
            _rmse, _pb = _one_fold(tr, va)
            fold_results.append((_rmse, _pb))
            if math.isfinite(_rmse):
                _sum_so_far += _rmse
                _n_finite_so_far += 1
            if (
                math.isfinite(early_stop_threshold)
                and _n_finite_so_far > 0
                and _fi < len(splits) - 1
                and _sum_so_far > early_stop_threshold * cv_folds
            ):
                # Final mean cannot reach <= threshold; abort remaining folds.
                break

    fold_rmses = [r for r, _ in fold_results if math.isfinite(r)]
    if not fold_rmses:
        if return_per_bin:
            return float("nan"), np.full(n_bins, float("nan"))
        return float("nan")
    mean_rmse = float(np.mean(fold_rmses))
    if not return_per_bin:
        return mean_rmse
    # Aggregate per-bin: mean across folds (NaN-skipping).
    per_bin_arrays = [pb for _, pb in fold_results if pb is not None]
    if not per_bin_arrays:
        return mean_rmse, np.full(n_bins, float("nan"))
    per_bin_stack = np.stack(per_bin_arrays, axis=0)
    with np.errstate(invalid="ignore"):
        per_bin_mean = np.nanmean(per_bin_stack, axis=0)
    return mean_rmse, per_bin_mean


def _sample_indices(
    n: int, sample_n: int | None, random_state: int,
    *,
    strategy: str = "random",
    y: np.ndarray | None = None,
    n_strata: int = 10,
) -> np.ndarray:
    """Return a sorted array of row indices to use for MI screening.

    Two strategies:

    - ``"random"`` (default): uniform random sample of ``sample_n``
      rows from ``n``. Cheap, unbiased on average, but high-variance
      on heavy-tail targets (the rare-tail rows that carry most of
      the signal can be over- or under-represented in any one draw).

    - ``"stratified_quantile"``: bin ``y`` into ``n_strata`` quantile
      bins, then sample ``sample_n / n_strata`` rows from each bin.
      Tail rows get oversampled relative to natural frequency. Use
      when ``y`` is heavy-tail (financial returns, fraud scores,
      queue lengths) -- gives stable MI rankings across runs because
      each bin contributes a guaranteed number of rows.

    Sorted so the (mostly-temporal) row order is preserved -- avoids
    biasing the MI estimate on temporal data.
    """
    if sample_n is None or n <= sample_n:
        return np.arange(n)
    rng = np.random.default_rng(random_state)
    if strategy == "random" or y is None or n_strata < 2:
        idx = rng.choice(n, size=sample_n, replace=False)
        idx.sort()
        return idx
    if strategy != "stratified_quantile":
        raise ValueError(
            f"_sample_indices: unknown strategy '{strategy}'. "
            "Choose from 'random' or 'stratified_quantile'."
        )

    # Stratified quantile sampling. Bin y into n_strata quantile
    # bins, sample ceil(sample_n / n_strata) from each.
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
    if y_arr.size != n:
        # Caller passed mismatched y; fall back to random.
        idx = rng.choice(n, size=sample_n, replace=False)
        idx.sort()
        return idx
    finite_mask = np.isfinite(y_arr)
    if finite_mask.sum() < n_strata * 2:
        # Too few finite y; can't stratify, fall back to random.
        idx = rng.choice(n, size=sample_n, replace=False)
        idx.sort()
        return idx
    # Compute quantile cuts on finite y.
    qs = np.linspace(0, 1, n_strata + 1)[1:-1]
    cuts = np.quantile(y_arr[finite_mask], qs)
    # Assign each finite row to a stratum [0, n_strata-1]; non-finite
    # rows get a separate stratum at the end so they aren't dropped
    # silently.
    stratum = np.searchsorted(cuts, y_arr, side="right")
    np.clip(stratum, 0, n_strata - 1, out=stratum)
    stratum[~finite_mask] = n_strata  # extra "non-finite" bin

    per_stratum = max(1, sample_n // n_strata)
    picked: list[np.ndarray] = []
    for s in range(n_strata + 1):
        bin_rows = np.where(stratum == s)[0]
        if bin_rows.size == 0:
            continue
        take = min(bin_rows.size, per_stratum)
        if take == bin_rows.size:
            picked.append(bin_rows)
        else:
            chosen = rng.choice(bin_rows, size=take, replace=False)
            picked.append(chosen)
    out = np.concatenate(picked) if picked else np.arange(min(n, sample_n))
    out.sort()
    return out[:sample_n]
