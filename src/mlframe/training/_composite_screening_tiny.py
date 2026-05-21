"""Tiny-model RMSE / CV helpers for composite-target screening.

Wave 98 (2026-05-21): split out from ``composite_screening.py`` to keep
that file below the 1k-line threshold. Behaviour preserved bit-for-bit;
every moved symbol is re-exported from ``composite_screening`` so
existing ``from mlframe.training.composite_screening import _build_tiny_model``
(and the other six moved names) imports continue to work.

What lives here:
  - ``_silence_tiny_model_output`` (CB / LGB log-silencer context)
  - ``_build_tiny_model`` (CB / LGB / Ridge factory)
  - ``_tiny_cv_rmse_raw_y`` (single-seed CV on raw y)
  - ``_tiny_cv_rmse_y_scale_multiseed`` (multi-seed CV on y-scale)
  - ``_tiny_cv_rmse_raw_y_multiseed`` (multi-seed CV on raw y)
  - ``_per_bin_rmse`` (per-bin RMSE helper for piecewise scoring)
  - ``_tiny_cv_rmse_y_scale`` (single-seed CV on y-scale)
"""
from __future__ import annotations

import contextlib
import io
import logging
import math
import os
import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Hoisted from a lazy import inside ``_one_fold`` (wave 98 split-out). The lazy form raced
# under joblib threading + n_jobs > 1: two folds simultaneously executing
# ``from .composite_estimator import _y_train_clip_bounds`` could see a partially-loaded
# composite_estimator module on Python's import dance, leaving the local name unbound for
# the second thread. Symptom in production log (TVT 2026-05-21): 4x
# ``composite_screening: tiny-model CV fold failed (name '_y_train_clip_bounds' is not defined)``
# warnings -- the lazy import silently raised NameError, the outer ``except Exception`` swallowed
# it, and the fold returned NaN. Sibling ``composite_screening.py`` already imports at module
# level (line 34) so there is no circular-dep concern.
from .composite_estimator import _y_train_clip_bounds

logger = logging.getLogger(__name__)


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
        except Exception as _e:
            # Failed fold reported as NaN -> np.nanmean over surviving folds
            # silently shifts the screening RMSE toward well-behaved folds
            # (K_eff < K with no signal). At least WARN-log per failure so
            # operators see why the effective fold count dropped. The NaN
            # return semantics are preserved (caller knows to use nanmean).
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "composite_screening: tiny-model CV fold failed (%s); fold "
                "reported as NaN. Screening RMSE will use nanmean over surviving "
                "folds -- effective fold count is reduced.", _e,
            )
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
            # _y_train_clip_bounds is imported at module level above
            # (race-safe across joblib threading folds).
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
        except Exception as _e:
            # Failed fold reported as NaN -> np.nanmean over surviving folds
            # silently shifts the screening RMSE toward well-behaved folds
            # (K_eff < K with no signal). At least WARN-log per failure so
            # operators see why the effective fold count dropped. The NaN
            # return semantics are preserved (caller knows to use nanmean).
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "composite_screening: tiny-model CV fold failed (%s); fold "
                "reported as NaN. Screening RMSE will use nanmean over surviving "
                "folds -- effective fold count is reduced.", _e,
            )
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
