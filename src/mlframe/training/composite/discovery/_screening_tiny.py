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

import contextlib
import logging
import math
import re
import threading
import warnings
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

import numpy as np

# Hoisted from a lazy import inside ``_one_fold``. The lazy form raced under joblib threading + n_jobs > 1:
# two folds simultaneously executing ``from ..estimator import _y_train_clip_bounds`` could see a
# partially-loaded composite.estimator module on Python's import dance, leaving the local name unbound for
# the second thread -- the lazy import silently raised NameError, the outer ``except Exception`` swallowed
# it, and the fold returned NaN. Sibling ``composite_screening.py`` already imports at module level so there
# is no circular-dep concern. Kept at module level here (not only in the carved ``_screening_tiny_perbin``
# sibling) so the race-safe hoist holds for importers of this parent module too.
from ..estimator import _y_train_clip_bounds  # noqa: F401

logger = logging.getLogger(__name__)

# Shuffled-KFold splits depend ONLY on (n_rows, cv_folds, random_state) -- not on
# the feature values (KFold.split shuffles np.arange(n) by the seeded RNG). Across
# all N_SPECS in one rerank sweep that triple is identical, so the fold-index lists
# repeat; cache them once per sweep (also skips KFold.split's per-call x re-validation).
# Bounded LRU-ish: cap entries so a long-lived process can't grow it unboundedly.
_KFOLD_SPLIT_CACHE: dict[tuple[int, int, int], list[tuple[np.ndarray, np.ndarray]]] = {}
_KFOLD_SPLIT_CACHE_MAX = 256


def _cached_kfold_splits(n_rows: int, cv_folds: int, random_state: int):
    """Return the shuffled-KFold ``(train_idx, val_idx)`` list for this
    ``(n_rows, cv_folds, random_state)``, building+caching it on first use.

    Bit-identical to ``list(KFold(n_splits=cv_folds, shuffle=True,
    random_state=random_state).split(x))`` for any ``x`` with ``n_rows`` rows --
    the split is a pure function of those three ints. Cached arrays are only
    READ downstream (fancy-indexed into copies), so reuse is safe without a copy."""
    key = (int(n_rows), int(cv_folds), int(random_state))
    cached = _KFOLD_SPLIT_CACHE.get(key)
    if cached is not None:
        return cached
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    splits = list(kf.split(np.empty(n_rows, dtype=np.uint8)))
    if len(_KFOLD_SPLIT_CACHE) >= _KFOLD_SPLIT_CACHE_MAX:
        _KFOLD_SPLIT_CACHE.clear()  # cheap bounded reset; sweeps reuse one key set
    _KFOLD_SPLIT_CACHE[key] = splits
    return splits


# Thread-local reentrancy state for the lightgbm-logger level bump in _silence_tiny_model_output.
# Only the outermost silence context on a given thread touches the logger level, so an outer per-CV-call wrap collapses N per-fold setLevel/_clear_cache pairs into a single one.
_silence_tiny_state = threading.local()

# Per-fold warning silencing installs the same four filters on every CV fold (23k+ times across a discovery sweep). ``warnings.filterwarnings`` re-compiles each ``message`` regex (with ``re.I``) on every call -- ~6.3us/call from the two regex compiles alone. Precompile the two message regexes once at module scope and prepend the four filter tuples directly to the fresh ``catch_warnings``-copied filters list, mirroring ``filterwarnings``' exact tuple shape ``(action, compiled_message_or_None, category, compiled_module_or_None, lineno)`` with ``re.I`` on the message. Behaviourally identical (same action/category/message-match), ~3.8x cheaper (6.3us -> 1.6us per fold).
_FEATURE_NAMES_RE = re.compile(".*feature names.*", re.I)
_SKIPPING_FEATURES_RE = re.compile(".*Skipping features without any observed values.*", re.I)


def _family_uses_lgb(family: str | None) -> bool:
    """True when ``family`` actually drives LightGBM. Only then is the
    lightgbm-logger level bump (logging.Manager._clear_cache, which walks the
    whole logger tree twice) worth paying. ``None`` keeps the legacy
    always-bump behaviour for callers that don't pass a family."""
    if family is None:
        return True
    return family.lower() in ("lgb", "lightgbm")


@contextlib.contextmanager
def _silence_tiny_model_output(family: str | None = None):
    """Context manager: silence the per-fold tiny-model fit/predict
    noise without changing the numeric path (no DataFrame->ndarray
    conversion; we keep the user's frame as-is for performance).

    ``family`` gates the lightgbm-logger level bump: for linear/ridge/cb/xgb
    families no LightGBM is involved, so the ``setLevel`` (which triggers
    ``logging.Manager._clear_cache`` over the whole logger tree, twice per
    fold) is skipped. ``None`` (default) keeps the legacy always-bump path for
    external callers that don't know the family.

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
    _bump_lgb = _family_uses_lgb(family)
    _lgb_logger = _logging.getLogger("lightgbm") if _bump_lgb else None
    # setLevel ALWAYS calls Manager._clear_cache() over the whole logger tree (even when the level is unchanged); on a large process logger tree that is ~100-150us per call, and it fires twice per CV fold (bump + restore).
    # Across a screening run the lightgbm logger sits at NOTSET and we always bump-to-ERROR / restore-to-NOTSET, so consecutive folds keep re-clearing the cache for no net level change. Use a reentrancy depth so only the OUTERMOST silence bump touches the level: nested / sequential-within-an-outer-silence entries skip both setLevel calls.
    # Numerically identical: throughout the yield the effective lightgbm level is ERROR (INFO/DEBUG suppressed) exactly as before, and the level is restored to its original value when the outermost context exits.
    _do_bump = False
    if _lgb_logger is not None:
        _depth = getattr(_silence_tiny_state, "lgb_depth", 0)
        if _depth == 0:
            _silence_tiny_state.lgb_prev_level = _lgb_logger.level
            _lgb_logger.setLevel(_logging.ERROR)
            _do_bump = True
        _silence_tiny_state.lgb_depth = _depth + 1
    try:
        from sklearn.exceptions import ConvergenceWarning
    except Exception:  # pragma: no cover - sklearn always installed in our deps
        ConvergenceWarning = UserWarning
    with warnings.catch_warnings():
        # Prepend the four ignore filters directly using precompiled message regexes (see ``_FEATURE_NAMES_RE`` note above). ``catch_warnings.__enter__`` has just copied the global filters into a fresh ``warnings.filters`` list, so a plain prepend (newest-first) reproduces the exact order four ``filterwarnings("ignore", ...)`` calls would leave -- last-inserted (RuntimeWarning) first. The second message filter squelches sklearn SimpleImputer's "Skipping features without any observed values" no-op warning (fires per CV fold * candidate spec in tiny-rerank); the actual remediation (drop fully-NaN columns) lives in the auto-base per-column NaN gate.
        warnings.filters[:0] = [  # type: ignore[index]  # warnings.filters is a private CPython list, not the Sequence the stub declares
            ("ignore", None, RuntimeWarning, None, 0),
            ("ignore", None, ConvergenceWarning, None, 0),
            ("ignore", _SKIPPING_FEATURES_RE, UserWarning, None, 0),
            ("ignore", _FEATURE_NAMES_RE, UserWarning, None, 0),
        ]
        warnings._filters_mutated()  # type: ignore[attr-defined]  # private CPython API (no typeshed stub); invalidates the warnings filter cache after the manual prepend above
        try:
            yield
        finally:
            if _lgb_logger is not None:
                _silence_tiny_state.lgb_depth -= 1
                if _do_bump:
                    _lgb_logger.setLevel(_silence_tiny_state.lgb_prev_level)


def _build_tiny_model(
    family: str, *, n_estimators: int, num_leaves: int, learning_rate: float, random_state: int, deterministic: bool = False, inner_n_jobs: int = -1
) -> Any:
    """Lazy-build a tiny regressor for the requested family. Lazy
    imports keep the discovery module light when those libraries
    aren't installed.

    No-actionable-speedup (bench: _benchmarks/bench_build_tiny_model_per_fold.py): a fresh per-fold construction is
    ~3.7us (ridge) / ~6-7us (lgb/xgb) / ~28us (cb) warm -- the lazy import is cached after first touch, so a 5-fold
    build sweep is sub-millisecond, orders below the fold FIT (ms-to-s). Pooling/reusing instances across folds saves
    nothing and risks cross-fold/spec state leakage; a fresh unfitted estimator per fold is the correct contract.

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
        kwargs: dict[str, Any] = dict(
            n_estimators=n_estimators,
            num_leaves=num_leaves,
            learning_rate=learning_rate,
            random_state=random_state,
            n_jobs=inner_n_jobs, verbose=-1, force_col_wise=True,
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
        xgb_kwargs: dict[str, Any] = dict(
            n_estimators=n_estimators,
            max_depth=4,
            learning_rate=learning_rate,
            random_state=random_state,
            n_jobs=inner_n_jobs, verbosity=0,
        )
        if deterministic:
            xgb_kwargs["tree_method"] = "hist"
        return xgb.XGBRegressor(**xgb_kwargs)
    if family_lower in ("cb", "catboost"):
        import catboost as cb
        cb_kwargs: dict[str, Any] = dict(
            iterations=n_estimators,
            depth=4,
            learning_rate=learning_rate,
            random_state=random_state,
            verbose=False,
            allow_writing_files=False,  # parallel fold/spec fits race on catboost_info/ (Windows file-lock -> silent NaN folds) and litter CWD
        )
        if deterministic:
            cb_kwargs["boosting_type"] = "Plain"
        return cb.CatBoostRegressor(**cb_kwargs)
    if family_lower in ("linear", "ridge"):
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
        # Wrap Ridge in a SimpleImputer pipeline because Ridge raises on
        # NaN inputs (observed in prod: tens of thousands of warnings
        # spammed per discovery run when ``tiny_screening_families``
        # included "linear" alongside "lightgbm" -- LGBM handles NaN
        # natively, Ridge doesn't). Mean-imputing the screening matrix
        # for the proxy is sound: this isn't the production model, just
        # a screening proxy for "would a downstream linear model find
        # this composite useful". Ridge is deterministic by construction;
        # the random_state kwarg accepted for call-signature uniformity.
        return Pipeline(
            [
                ("imp", SimpleImputer(strategy="mean")),
                ("ridge", Ridge(alpha=1.0, random_state=random_state)),
            ]
        )
    raise ValueError(f"_build_tiny_model: unknown family '{family}'. " "Supported: lightgbm, xgboost, catboost, linear / ridge.")


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
    inner_n_jobs: int = -1,
    deterministic: bool = False,
    return_per_bin: bool = False,
    n_bins: int = 5,
    bin_var: np.ndarray | None = None,
    return_fold_preds: bool = False,
    time_aware: bool = False,
    cv_splitter: Any = None,
    groups: np.ndarray | None = None,
    cv_selector_mode: str = "mean",
    cv_selector_alpha: float = 1.0,
    cv_selector_confidence: float = 0.9,
    cv_selector_quantile_level: float = 0.9,
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

    ``return_fold_preds`` appends a list of per-fold ``(val_y_true, val_y_hat,
    val_idx)`` records to the return value. The raw-y model is trained on
    ``x_train_matrix`` and never sees ``bin_var``, so the same fold predictions
    yield the per-bin baseline for ANY ``bin_var`` via
    :func:`_per_bin_from_fold_preds` -- letting the rerank compute the raw
    per-bin breakdown ONCE and reuse it across every base, instead of refitting
    the (bin_var-independent) raw-y model per base. Bit-identical to calling
    this function per base with ``bin_var=`` set.
    """
    from sklearn.model_selection import GroupKFold, KFold, TimeSeriesSplit
    n = len(y_train)
    if n < cv_folds * 10:
        return (float("nan"), np.full(n_bins, float("nan"))) if return_per_bin else float("nan")
    y_clean = y_train.astype(np.float64)
    finite_mask = None
    if not np.all(np.isfinite(y_clean)):
        finite_mask = np.isfinite(y_clean)
        y_clean = y_clean[finite_mask]
        x_clean = x_train_matrix[finite_mask]
    else:
        x_clean = x_train_matrix
    if len(y_clean) < cv_folds * 10:
        return (float("nan"), np.full(n_bins, float("nan"))) if return_per_bin else float("nan")

    groups_clean = None
    if groups is not None:
        _g = np.asarray(groups)
        if _g.shape[0] == len(y_train):
            groups_clean = _g[finite_mask] if finite_mask is not None else _g
            _n_groups = int(np.unique(groups_clean).size)
            if _n_groups < cv_folds:
                # Silent GroupKFold->KFold downgrade -> WARN (see y-scale twin).
                logger.warning(
                    "_tiny_cv_rmse_raw_y: groups supplied but only %d distinct "
                    "group(s) survive the finite mask (< cv_folds=%d); falling "
                    "back to %s split. Group separation is NOT enforced for the "
                    "raw-y baseline -- reduce cv_folds or supply more groups.",
                    _n_groups,
                    cv_folds,
                    "TimeSeriesSplit" if (cv_splitter is None and time_aware) else "shuffled KFold",
                )
                groups_clean = None
    if cv_splitter is not None:
        kf = cv_splitter
        _precomputed_splits = None
    elif groups_clean is not None:
        if time_aware:
            # Groups win over time_aware; temporal order dropped. WARN once.
            logger.warning(
                "_tiny_cv_rmse_raw_y: both groups and time_aware requested; "
                "GroupKFold takes precedence and temporal order is NOT preserved. "
                "Pass a grouped forward-chaining cv_splitter to honour both.",
            )
        kf = GroupKFold(n_splits=cv_folds)
        _precomputed_splits = list(kf.split(x_clean, groups=groups_clean))
    elif time_aware:
        kf = TimeSeriesSplit(n_splits=cv_folds)
        _precomputed_splits = None
    else:
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        _precomputed_splits = None

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
    ) -> tuple[float, np.ndarray | None, tuple[np.ndarray, np.ndarray, np.ndarray] | None]:
        """Fit one tiny CV fold's model and return (val RMSE, val predictions or None on failure, fold arrays for downstream reuse or None)."""
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
            with _silence_tiny_model_output(family):
                model.fit(x_clean[train_fold], y_clean[train_fold])
                y_hat = np.asarray(model.predict(x_clean[val_fold])).reshape(-1)
            y_hat_f64 = y_hat.astype(np.float64)
            diff = y_hat_f64 - y_clean[val_fold]
            finite = np.isfinite(diff)
            if finite.sum() == 0:
                return float("nan"), None, None
            rmse = float(np.sqrt(np.mean(diff[finite] * diff[finite])))
            per_bin = None
            if return_per_bin and bin_var_clean is not None:
                per_bin = _per_bin_rmse(
                    y_clean[val_fold], y_hat,
                    bin_var_clean[val_fold], n_bins=n_bins,
                )
            # Capture the per-fold val truth/prediction/index so the caller can
            # re-derive a per-bin breakdown for a DIFFERENT bin_var without
            # refitting the (bin_var-independent) raw-y model and folds. Store
            # the RAW y_hat (the exact array _per_bin_rmse receives in-loop) so
            # the derived per-bin is bit-identical, not the float64-cast copy.
            fold_pred = (y_clean[val_fold].copy(), y_hat, np.asarray(val_fold)) if return_fold_preds else None
            return rmse, per_bin, fold_pred
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
            return float("nan"), None, None

    splits = _precomputed_splits if _precomputed_splits is not None else list(kf.split(x_clean))
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
            fold_results = [_one_fold(tr, va) for tr, va in splits]
    # Emit a single WARN when ANY fold returned NaN. The
    # outer ``except Exception`` in ``_one_fold`` swallows every failure into a
    # NaN result, and downstream ``np.nanmean`` silently shifts the screening
    # RMSE toward the surviving folds. Without this WARN the operator never
    # sees that effective fold count dropped (a prod log had
    # 4 silent NaN-folds before the lazy-import race was fixed).
    _nan_fold_count = sum(1 for r, _, _ in fold_results if not math.isfinite(r))
    if _nan_fold_count > 0:
        logger.warning(
            "_tiny_cv_rmse_raw_y: %d/%d folds returned NaN (silent failures). "
            "Screening RMSE will use nanmean over the remaining %d fold(s); "
            "effective fold count is reduced. Inspect the per-fold WARN logs "
            "above for the underlying exception.",
            _nan_fold_count, len(fold_results), len(fold_results) - _nan_fold_count,
        )
    # Per-fold val truth/prediction records (only populated when the caller
    # asked for them); the per-bin baseline for any bin_var is derivable from
    # these without a refit, since the raw-y model never sees bin_var.
    fold_preds = [fp for _, _, fp in fold_results if fp is not None] if return_fold_preds else None
    fold_rmses = [r for r, _, _ in fold_results if math.isfinite(r)]
    if not fold_rmses:
        _nan_pb = np.full(n_bins, float("nan"))
        if return_per_bin and return_fold_preds:
            return float("nan"), _nan_pb, []
        if return_fold_preds:
            return float("nan"), []
        if return_per_bin:
            return float("nan"), _nan_pb
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
        if return_fold_preds:
            return mean_rmse, fold_preds
        return mean_rmse
    per_bin_arrays = [pb for _, pb, _ in fold_results if pb is not None]
    if not per_bin_arrays:
        per_bin_mean = np.full(n_bins, float("nan"))
    else:
        per_bin_stack = np.stack(per_bin_arrays, axis=0)
        with np.errstate(invalid="ignore"):
            per_bin_mean = np.nanmean(per_bin_stack, axis=0)
    if return_fold_preds:
        return mean_rmse, per_bin_mean, fold_preds
    return mean_rmse, per_bin_mean


def _seed_median_lower_bound(observed_finite: list[float], n_remaining: int) -> float:
    """Rigorous LOWER BOUND on ``median(observed_finite + <n_remaining more finite seed scores>)``,
    for ANY values those remaining scores turn out to take.

    RMSE-like seed scores are non-negative. Inserting a hypothetical 0 for every not-yet-run seed
    can only shift the sorted sample toward its low end, so ``median(observed_finite + [0]*n_remaining)``
    is <= the median of ``observed_finite`` plus ANY n_remaining non-negative reals (median is
    monotone non-decreasing in each element, and a failed/NaN future seed -- excluded from the
    eventual median -- is equivalent to inserting a value at the current position, never lower than
    inserting an explicit 0). Used to decide whether a candidate is GUARANTEED to still land at or
    above the raw-baseline rejection threshold regardless of unrun seeds, i.e. safe to early-stop.
    """
    if not observed_finite and n_remaining <= 0:
        return float("nan")
    padded = list(observed_finite) + [0.0] * max(0, n_remaining)
    return float(np.median(padded))


def _tiny_cv_rmse_y_scale_multiseed(
    *args,
    n_seed_repeats: int = 1,
    base_random_state: int = 0,
    return_per_seed: bool = False,
    seed_early_stop_threshold: float = float("inf"),
    **kwargs,
):
    """Multi-seed wrapper around :func:`_tiny_cv_rmse_y_scale`.

    With cv_folds=3, a single CV split has high variance. Repeat the
    K-fold split with different random seeds and return the MEDIAN of
    the per-seed mean RMSEs (instead of a single point estimate). When
    ``return_per_bin=True``, also returns the per-bin median across seeds.

    When ``return_per_seed=True``, also returns the array of per-seed
    mean RMSEs so callers can run a paired Wilcoxon test against a
    reference (raw-y baseline) array.

    The returned per-seed array is FIXED-LENGTH (one slot per seed, in
    ``base_random_state + s_idx*7919`` seed order) with NaN at positions
    where that seed's CV degenerated. The composite and raw-y sweeps run
    the SAME seed schedule, so a failed seed leaves a NaN in the SAME
    slot on both sides -- the consumer pairs by seed index and diffs only
    jointly-finite positions. A compacted (finite-only) contract would
    mis-pair a failed composite seed against a failed raw seed at a
    different position in the paired Wilcoxon gate. The median is still
    taken over finite seeds only, so the returned point estimate is
    unaffected.

    ``early_stop_threshold`` (forwarded transparently through ``**kwargs``
    into the underlying serial fold loop) lets the rerank caller abort a
    spec's per-fold fits once the running fold-mean is GUARANTEED to
    exceed the raw-baseline gate threshold (the spec will be rejected
    regardless). For this to be sound across seeds the caller passes the
    SAME threshold every seed -- each seed independently early-stops; the
    median over the surviving (finite) seeds is unchanged when the
    threshold is ``inf`` (the default), so the multiseed return value is
    bit-identical to the no-threshold call. The early-stop fires only on
    the serial path (``n_jobs<=1``) and only for ``cv_selector_mode='mean'``
    (the partial-sum bound is a mean bound); see ``_tiny_cv_rmse_y_scale``.

    ``n_seed_repeats=1`` is the legacy single-seed path -- exact
    same numerical result as calling the underlying function once.
    """
    # Seed-invariant splitters ignore random_state: TimeSeriesSplit (time_aware)
    # has a fixed forward-walk, GroupKFold partitions by group, and either makes
    # every "seed repeat" an EXACT duplicate. Collapse to one honest measurement
    # (per_seed length 1) -- this avoids N identical fits on the dominant rerank
    # phase AND avoids pseudo-replicating a single measurement into a falsely-
    # powered Wilcoxon (the gate then correctly skips on n=1).
    _seed_invariant = bool(kwargs.get("time_aware", False)) or (kwargs.get("groups", None) is not None)
    _effective_repeats = 1 if _seed_invariant else n_seed_repeats
    if _effective_repeats <= 1:
        kwargs["random_state"] = base_random_state
        result = _tiny_cv_rmse_y_scale(*args, **kwargs)
        if return_per_seed:
            mean = result[0] if isinstance(result, tuple) else result
            # Fixed length 1: NaN when the single seed degenerated, so the
            # consumer always sees one slot per scheduled seed.
            per_seed_arr = np.array([mean], dtype=np.float64)
            if isinstance(result, tuple):
                return (*result, per_seed_arr)
            return result, per_seed_arr
        return result
    seed_results = []
    seed_per_bins = []
    return_pb = kwargs.get("return_per_bin", False)
    # Sequential early-stop is only proven safe for cv_selector_mode == "mean" -- see
    # ``enable_multiseed_early_stop`` docstring in ``_composite_target_discovery_config_base``. Any
    # other fold-aggregation mode leaves ``_early_stop_active`` False, a strict no-op (all seeds run).
    _early_stop_active = math.isfinite(seed_early_stop_threshold) and kwargs.get("cv_selector_mode", "mean") == "mean"
    # One slot per scheduled seed; NaN where that seed failed.
    per_seed_full = np.full(_effective_repeats, float("nan"), dtype=np.float64)
    for s_idx in range(_effective_repeats):
        kwargs["random_state"] = base_random_state + s_idx * 7919
        result = _tiny_cv_rmse_y_scale(*args, **kwargs)
        if return_pb and isinstance(result, tuple):
            mean_rmse, per_bin = result
            if math.isfinite(mean_rmse):
                per_seed_full[s_idx] = mean_rmse
                seed_results.append(mean_rmse)
                seed_per_bins.append(per_bin)
        else:
            if math.isfinite(result):
                per_seed_full[s_idx] = result
                seed_results.append(result)
        if _early_stop_active and s_idx < _effective_repeats - 1:
            _lb = _seed_median_lower_bound(seed_results, _effective_repeats - 1 - s_idx)
            if math.isfinite(_lb) and _lb >= seed_early_stop_threshold:
                # True median can only be >= this lower bound (see ``_seed_median_lower_bound``);
                # the candidate is already guaranteed to be rejected -- stop running seeds.
                break
    seed_arr = per_seed_full
    if not seed_results:
        if return_pb:
            res = float("nan"), np.full(kwargs.get("n_bins", 5), float("nan"))
            return (*res, seed_arr) if return_per_seed else res
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
    seed_early_stop_threshold: float = float("inf"),
    **kwargs,
):
    """Multi-seed wrapper around :func:`_tiny_cv_rmse_raw_y`. See
    :func:`_tiny_cv_rmse_y_scale_multiseed` for the rationale (including
    the fixed-length NaN-padded per-seed contract).

    ``seed_early_stop_threshold`` is accepted for call-signature parity with the composite sibling
    but the raw-y baseline is never called with a finite threshold in practice: it IS the quantity the
    rejection threshold is derived from, so early-stopping it against its own derived threshold would
    be circular. Left wired through only so a future caller with an independent reference (e.g. a prior
    iteration's raw baseline) can opt in without a signature change."""
    # Seed-invariant splitters (TimeSeriesSplit / GroupKFold) collapse to one
    # honest measurement -- see _tiny_cv_rmse_y_scale_multiseed.
    _seed_invariant = bool(kwargs.get("time_aware", False)) or (kwargs.get("groups", None) is not None)
    _effective_repeats = 1 if _seed_invariant else n_seed_repeats
    if _effective_repeats <= 1:
        kwargs["random_state"] = base_random_state
        result = _tiny_cv_rmse_raw_y(*args, **kwargs)
        if return_per_seed:
            mean = result[0] if isinstance(result, tuple) else result
            # Fixed length 1: NaN when the single seed degenerated.
            per_seed_arr = np.array([mean], dtype=np.float64)
            if isinstance(result, tuple):
                return (*result, per_seed_arr)
            return result, per_seed_arr
        return result
    seed_results = []
    seed_per_bins = []
    return_pb = kwargs.get("return_per_bin", False)
    _early_stop_active = math.isfinite(seed_early_stop_threshold) and kwargs.get("cv_selector_mode", "mean") == "mean"
    # One slot per scheduled seed; NaN where that seed failed.
    per_seed_full = np.full(_effective_repeats, float("nan"), dtype=np.float64)
    for s_idx in range(_effective_repeats):
        kwargs["random_state"] = base_random_state + s_idx * 7919
        result = _tiny_cv_rmse_raw_y(*args, **kwargs)
        if return_pb and isinstance(result, tuple):
            mean_rmse, per_bin = result
            if math.isfinite(mean_rmse):
                per_seed_full[s_idx] = mean_rmse
                seed_results.append(mean_rmse)
                seed_per_bins.append(per_bin)
        else:
            if math.isfinite(result):
                per_seed_full[s_idx] = result
                seed_results.append(result)
        if _early_stop_active and s_idx < _effective_repeats - 1:
            _lb = _seed_median_lower_bound(seed_results, _effective_repeats - 1 - s_idx)
            if math.isfinite(_lb) and _lb >= seed_early_stop_threshold:
                break
    seed_arr = per_seed_full
    if not seed_results:
        if return_pb:
            res = float("nan"), np.full(kwargs.get("n_bins", 5), float("nan"))
            return (*res, seed_arr) if return_per_seed else res
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


# per-bin RMSE + y-scale tiny-CV helpers carved to _screening_tiny_perbin.py (1k-LOC ceiling).
from ._screening_tiny_perbin import (  # noqa: F401
    _per_bin_from_fold_preds,
    _per_bin_rmse,
    _tiny_cv_rmse_y_scale,
)
