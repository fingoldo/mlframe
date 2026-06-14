"""Tiny-model RMSE / CV helpers for composite-target screening.

Every moved symbol is re-exported from ``composite_screening`` so existing
``from mlframe.training.composite_screening import _build_tiny_model`` (and the
other six moved names) imports continue to work.

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

import logging
import math
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
    from ..transforms import Transform

import numpy as np

# Hoisted from a lazy import inside ``_one_fold``. The lazy form raced under joblib threading + n_jobs > 1:
# two folds simultaneously executing ``from ..estimator import _y_train_clip_bounds`` could see a
# partially-loaded composite.estimator module on Python's import dance, leaving the local name unbound for
# the second thread -- the lazy import silently raised NameError, the outer ``except Exception`` swallowed
# it, and the fold returned NaN. Sibling ``composite_screening.py`` already imports at module level so there
# is no circular-dep concern.
from ..estimator import _y_train_clip_bounds

logger = logging.getLogger(__name__)

# Shuffled-KFold splits depend ONLY on (n_rows, cv_folds, random_state) -- not on
# the feature values (KFold.split shuffles np.arange(n) by the seeded RNG). Across
# all N_SPECS in one rerank sweep that triple is identical, so the fold-index lists
# repeat; cache them once per sweep (also skips KFold.split's per-call x re-validation).
# Bounded LRU-ish: cap entries so a long-lived process can't grow it unboundedly.
_KFOLD_SPLIT_CACHE: dict[tuple[int, int, int], list[tuple[np.ndarray, np.ndarray]]] = {}
_KFOLD_SPLIT_CACHE_MAX = 256



from ._screening_tiny import _build_tiny_model, _cached_kfold_splits, _silence_tiny_model_output  # noqa: E402 (cycle-safe: defined before parent's bottom re-export)

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


def _per_bin_from_fold_preds(
    fold_preds: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]],
    bin_var_clean: np.ndarray,
    n_bins: int = 5,
) -> np.ndarray:
    """Per-bin RMSE for ``bin_var_clean`` from cached raw-y fold predictions.

    ``fold_preds`` is the ``(val_y_true, val_y_hat, val_idx)`` list returned by
    :func:`_tiny_cv_rmse_raw_y` with ``return_fold_preds=True``; ``val_idx``
    indexes into the SAME finite-y-masked space as ``bin_var_clean`` (the caller
    masks ``bin_var`` with ``isfinite(y_train)`` exactly as the function does).

    Reproduces the in-function per-bin path bit-for-bit: per fold call
    :func:`_per_bin_rmse` on ``bin_var_clean[val_idx]`` then ``np.nanmean`` the
    surviving per-fold arrays. Because the raw-y model is independent of
    ``bin_var``, this yields the identical array a fresh ``_tiny_cv_rmse_raw_y``
    call with ``bin_var=`` would -- with no refit, so K specs over distinct
    bases share one fit.
    """
    per_bin_arrays = []
    for y_true, y_hat, val_idx in fold_preds:
        pb = _per_bin_rmse(
            y_true, y_hat, bin_var_clean[val_idx], n_bins=n_bins,
        )
        per_bin_arrays.append(pb)
    if not per_bin_arrays:
        return np.full(n_bins, float("nan"))
    per_bin_stack = np.stack(per_bin_arrays, axis=0)
    with np.errstate(invalid="ignore"):
        return np.nanmean(per_bin_stack, axis=0)


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
    inner_n_jobs: int = -1,
    deterministic: bool = False,
    return_per_bin: bool = False,
    n_bins: int = 5,
    time_aware: bool = False,
    early_stop_threshold: float = float("inf"),
    groups: np.ndarray | None = None,
    cv_splitter: Any = None,
    cv_selector_mode: str = "mean",
    cv_selector_alpha: float = 1.0,
    cv_selector_confidence: float = 0.9,
    cv_selector_quantile_level: float = 0.9,
):
    """Compute CV-RMSE of a tiny model on the y-scale (after inverse).

    1. Apply ``transform.forward`` to (y_train, base_train) -> T.
    2. K-fold split on the train rows. With ``time_aware=True`` the split
       is a sklearn ``TimeSeriesSplit`` -- past-only train / future
       holdout for each fold -- which matches the production ordering
       for autoregressive bases (lag features). Random
       K-fold on a lag base leaks future->past and over-rates the spec.
    3. For each fold: fit tiny model on T_train_fold, predict T_hat
       on the held-out fold, apply transform.inverse to recover
       y_hat in the original scale, score against y_held.
    4. Return mean across folds.

    Folds run in parallel when ``n_jobs > 1`` via joblib. Each fold
    fit gets ``n_jobs_per_fit = max(1, total_cpus // n_jobs)`` cores
    so the inner LightGBM doesn't oversubscribe. NaN if anything
    degenerates so callers can deprioritise.

    When the transform's fitted domain excludes some finite-y rows, the val
    population is the full finite-y set (raw-y baseline parity); the model trains
    only on the domain but each fold scores its full val rows with a y_train_median
    fallback on the off-domain ones (mirrors CompositeTargetEstimator.predict).
    The all-valid common case is bit-identical and keeps the no-copy fast path.
    cProfile note: the emulation adds one ``np.isfinite(y_train)`` + one ``.any()``
    on the common (all-valid) path -- <0.1 ms at n=20k vs the ~26 ms fold fits, far
    under the actionable threshold; no further optimization needed.
    """
    from sklearn.model_selection import GroupKFold, TimeSeriesSplit
    n = len(y_train)
    if n < cv_folds * 10:
        return (float("nan"), np.full(n_bins, float("nan"))) if return_per_bin else float("nan")
    valid = transform.domain_check(y_train, base_train)
    # Refine with the fitted-params-aware domain (log_y's offset,
    # centered_ratio's c+eps). Without this, rows that are valid pre-fit but
    # out of the TRUE fitted domain produce NaN T below -> the non-finite
    # guard nukes the WHOLE spec's rerank score, silently dropping a spec
    # that is perfectly valid on its real domain.
    _dcf = getattr(transform, "domain_check_fitted", None)
    if _dcf is not None and isinstance(fitted_params, dict):
        _valid_fitted = np.asarray(
            _dcf(y_train, base_train, fitted_params), dtype=bool,
        )
        if _valid_fitted.shape == valid.shape:
            valid = valid & _valid_fitted
    if valid.sum() < cv_folds * 10:
        return (float("nan"), np.full(n_bins, float("nan"))) if return_per_bin else float("nan")
    # Baseline-population parity with the raw-y sibling. The raw-y baseline scores EVERY finite-y row, but this transformed-target path previously scored only domain-valid rows -- so composite RMSE was measured on a STRICT SUBSET of the rows raw RMSE was measured on, a population mismatch that biases the raw-baseline gate exactly where a transform's domain excludes hard rows.
    # Production CompositeTargetEstimator.predict does NOT drop invalid rows: it predicts on the whole batch and fills domain-violation rows with y_train_median. We emulate that so the screening val population == the raw-y val population (all finite-y rows): the model still TRAINS only on the transform's domain (forward is undefined off-domain), but each fold is SCORED over its full finite-y val rows, with domain-invalid val rows filled by the train-fold median.
    # The _all_valid short-circuit keeps this bit-identical to the legacy path when the transform's domain covers every row (the common case) and preserves the no-copy fast path; ``finite_y`` mirrors raw-y's isfinite(y) mask and ``valid`` is a subset of it, so the split population (finite_y) is a superset of the trainable (valid) rows.
    _all_valid = bool(valid.all())
    finite_y = np.isfinite(np.asarray(y_train, dtype=np.float64))
    _emulate_fallback = (not _all_valid) and bool((finite_y & ~valid).any())
    if _emulate_fallback:
        # Split population = all finite-y rows (raw-y parity); ``valid_pop`` flags which are on the transform's fitted domain (trainable). The rest are scored via the median fallback on the val side only.
        pop_mask = finite_y
        y_pop = np.asarray(y_train, dtype=np.float64)[pop_mask]
        base_pop = np.asarray(base_train, dtype=np.float64)[pop_mask]
        x_pop = x_train_matrix[pop_mask]
        valid_pop = np.asarray(valid, dtype=bool)[pop_mask]
        if int(valid_pop.sum()) < cv_folds * 10:
            return (float("nan"), np.full(n_bins, float("nan"))) if return_per_bin else float("nan")
        # T computed only on the trainable rows; off-domain rows never enter forward(). A non-finite T on a domain-valid row still nukes the spec (legacy semantics), so keep the whole-spec finite guard.
        t_pop = np.full(y_pop.shape[0], np.nan, dtype=np.float64)
        t_pop[valid_pop] = transform.forward(y_pop[valid_pop], base_pop[valid_pop], fitted_params)
        if not np.all(np.isfinite(t_pop[valid_pop])):
            return (float("nan"), np.full(n_bins, float("nan"))) if return_per_bin else float("nan")
        y_clean = y_pop
        base_clean = base_pop
        x_clean = x_pop
        t_clean = t_pop
        _split_valid_mask = valid_pop
        _group_mask = pop_mask
    else:
        # Skip the full-matrix fancy-index copy of the wide feature matrix when
        # every row is valid (mirror the raw-y no-copy gate at the top of
        # _tiny_cv_rmse_raw_y). Downstream reads x_clean[train_fold]/x_clean[val_fold]
        # (which fancy-index a copy regardless), so passing the view is bit-identical.
        y_clean = y_train.astype(np.float64) if _all_valid else y_train[valid].astype(np.float64)
        base_clean = base_train.astype(np.float64) if _all_valid else base_train[valid].astype(np.float64)
        x_clean = x_train_matrix if _all_valid else x_train_matrix[valid]
        t_clean = transform.forward(y_clean, base_clean, fitted_params)
        if not np.all(np.isfinite(t_clean)):
            return (float("nan"), np.full(n_bins, float("nan"))) if return_per_bin else float("nan")
        _split_valid_mask = None  # every split row is trainable
        _group_mask = valid if not _all_valid else None

    groups_clean = None
    if groups is not None:
        _g_arr = np.asarray(groups)
        if _g_arr.shape[0] == len(y_train):
            # Align groups to whichever population the split runs on -- the
            # finite-y superset under fallback emulation, else the valid subset.
            groups_clean = (
                _g_arr[_group_mask] if _group_mask is not None else _g_arr
            )
            _n_groups = int(np.unique(groups_clean).size)
            if _n_groups < cv_folds:
                # Groups requested but too few distinct groups survive the
                # domain mask -> silent downgrade to KFold (different fold
                # population, no group separation). WARN so the operator sees
                # why the group split did not apply.
                _fallback_desc = (
                    "the caller-supplied cv_splitter" if cv_splitter is not None
                    else ("TimeSeriesSplit" if time_aware else "shuffled KFold")
                )
                logger.warning(
                    "_tiny_cv_rmse_y_scale: groups supplied but only %d distinct "
                    "group(s) survive the domain mask (< cv_folds=%d); falling "
                    "back to %s. Group separation is NOT enforced for this "
                    "spec -- reduce cv_folds or supply more groups to keep the "
                    "grouped fold contract.",
                    _n_groups, cv_folds, _fallback_desc,
                )
                groups_clean = None  # fall back to KFold if not enough groups
    if cv_splitter is not None:
        # Escape hatch (parity with _tiny_cv_rmse_raw_y): a caller-supplied
        # splitter wins over the groups/time_aware/KFold auto-pick.
        kf = cv_splitter
        splits = (
            list(kf.split(x_clean, groups=groups_clean))
            if groups_clean is not None
            else list(kf.split(x_clean))
        )
    elif groups_clean is not None:
        if time_aware:
            # Both group- and time-awareness were requested but the splitter
            # can honour only one; GroupKFold wins and the temporal ordering is
            # dropped (future->past leak risk on autoregressive bases). WARN once
            # per call so the operator can supply a grouped forward-chaining
            # cv_splitter if both must hold.
            logger.warning(
                "_tiny_cv_rmse_y_scale: both groups and time_aware were "
                "requested; GroupKFold takes precedence and the temporal order "
                "is NOT preserved (random group folds may leak future->past on "
                "autoregressive bases). Pass a grouped forward-chaining "
                "cv_splitter to honour both.",
            )
        kf = GroupKFold(n_splits=cv_folds)
        splits = list(kf.split(x_clean, groups=groups_clean))
    elif time_aware:
        kf = TimeSeriesSplit(n_splits=cv_folds)
        splits = list(kf.split(x_clean))
    else:
        # Hoisted split cache: identical (n,cv_folds,seed) across N_SPECS -> identical splits.
        splits = _cached_kfold_splits(x_clean.shape[0], cv_folds, random_state)

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
                inner_n_jobs=inner_n_jobs,
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
            # Under fallback emulation the train fold may contain off-domain rows (NaN T); fit ONLY on the domain-valid train rows (forward is undefined off-domain) -- production fits the inner the same way. ``_fit_rows`` == the full train fold on the legacy / all-valid path.
            if _split_valid_mask is not None:
                _tr_valid = train_fold[_split_valid_mask[train_fold]]
                if _tr_valid.shape[0] < 2:
                    return float("nan"), None
                _fit_rows = _tr_valid
            else:
                _fit_rows = train_fold
            with _silence_tiny_model_output(family):
                model.fit(x_clean[_fit_rows], t_clean[_fit_rows])
                t_hat = np.asarray(model.predict(x_clean[val_fold])).reshape(-1)
            # Domain-invalid val rows have a meaningless base for the inverse; supply a safe placeholder and overwrite with the median fallback below (mirrors estimator/_predict.py base_safe + y_train_median).
            if _split_valid_mask is not None:
                _val_valid = _split_valid_mask[val_fold]
                _base_for_inverse = np.where(_val_valid, base_clean[val_fold], 1.0)
            else:
                _val_valid = None
                _base_for_inverse = base_clean[val_fold]
            y_hat = transform.inverse(t_hat, _base_for_inverse, fitted_params)
            # Wrapper-aware clipping. The
            # production CompositeTargetEstimator.predict applies
            # the same y-clip on inverse output to keep predictions
            # inside a physically plausible range. Mirror that here
            # so screening RMSE matches deployed RMSE (otherwise
            # heavy-tail transforms like logratio look better in
            # screening than they actually deliver).
            # _y_train_clip_bounds is imported at module level above (race-safe across joblib threading folds). The y-clip envelope is fit on the domain-valid train rows (the rows the production estimator's clip bounds were learned from).
            y_clip_low, y_clip_high = _y_train_clip_bounds(
                y_clean[_fit_rows]
            )
            y_hat = np.clip(
                y_hat.astype(np.float64), y_clip_low, y_clip_high,
            )
            # Domain-violation fallback: rows where the transform's domain_check fails on val use y_train_median (matches wrapper.predict). The wrapper computes domain_check on (y, base) but at inference y is unknown, so it uses y=None handling; here on val we know y_clean[val_fold] and emulate the wrapper by falling back rows where y_hat is non-finite OR where the row is domain-invalid. The fallback median uses the domain-valid train y (estimator/_predict.py stores y_train_median over fit rows).
            _y_train_fallback: float | None = None
            if _val_valid is not None and (~_val_valid).any():
                _y_fit = y_clean[_fit_rows]
                _y_train_fallback = float(np.median(
                    _y_fit[np.isfinite(_y_fit)]
                )) if np.isfinite(_y_fit).any() else 0.0
                y_hat[~_val_valid] = _y_train_fallback
            non_finite = ~np.isfinite(y_hat)
            if non_finite.any():
                if _y_train_fallback is not None:
                    y_train_median = _y_train_fallback
                else:
                    _y_fit = y_clean[_fit_rows]
                    y_train_median = float(np.median(
                        _y_fit[np.isfinite(_y_fit)]
                    )) if np.isfinite(_y_fit).any() else 0.0
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

    # ``splits`` was built above (GroupKFold path inserted before this
    # function body so groups-aware tiny-CV uses the same fold contract).
    # Outer silence wrap: the lightgbm-logger level bump (and its whole-tree _clear_cache) happens ONCE here instead of once per fold via the reentrancy guard in _silence_tiny_model_output.
    with _silence_tiny_model_output(family):
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
            # Serial early-stop: track partial-sum and break when
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
                    and cv_selector_mode == "mean"  # partial-sum bound only guarantees the MEAN exceeds thr; median/quantile aggregates can stay below
                ):
                    # Final mean cannot reach <= threshold; abort remaining folds.
                    break

    # NaN-fold aggregate WARN (twin of the y-scale branch).
    _nan_fold_count = sum(1 for r, _ in fold_results if not math.isfinite(r))
    if _nan_fold_count > 0:
        logger.warning(
            "_tiny_cv_rmse_transformed_y: %d/%d folds returned NaN (silent failures). "
            "Screening RMSE will use nanmean over the remaining %d fold(s); "
            "effective fold count is reduced.",
            _nan_fold_count, len(fold_results), len(fold_results) - _nan_fold_count,
        )
    fold_rmses = [r for r, _ in fold_results if math.isfinite(r)]
    if not fold_rmses:
        if return_per_bin:
            return float("nan"), np.full(n_bins, float("nan"))
        return float("nan")
    from ..._cv_aggregation import aggregate_fold_scores
    mean_rmse = aggregate_fold_scores(
        fold_rmses,
        mode=cv_selector_mode,  # type: ignore[arg-type]
        direction="min",
        alpha=cv_selector_alpha,
        confidence=cv_selector_confidence,
        quantile_level=cv_selector_quantile_level,
    )
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
