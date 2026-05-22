"""``RFECV.fit`` carved out of ``mlframe.feature_selection.wrappers._rfecv``.

Holds only the main fit method. Bound onto the ``RFECV`` class at the
parent's module bottom so ``rfecv.fit(...)`` call sites work unchanged.

All module-level dependencies of ``fit`` are imported at this sibling's
top because none of them sits on a cycle path. The parent's ``RFECV``
class symbol itself is referenced inside the function body only (the
function takes ``self``), so no top-level ``from ._rfecv import RFECV``
is needed here.
"""
from __future__ import annotations

import copy
import hashlib
import logging
import textwrap
from contextlib import nullcontext
from os.path import exists
from timeit import default_timer as timer
from typing import Union

import numpy as np
import pandas as pd
import polars as pl

from pyutilz.system import tqdmu
from pyutilz.pythonlib import (
    suppress_stdout_stderr,
)

from sklearn.base import (
    clone,
    is_classifier,
    is_regressor,
)
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.pipeline import Pipeline

from mlframe.config import CATBOOST_MODEL_TYPES
from mlframe.utils.misc import set_random_seed
from mlframe.estimators.baselines import get_best_dummy_score
from mlframe.core.helpers import has_early_stopping_support
from mlframe.training.helpers import compute_cb_text_processing
from mlframe.preprocessing.transforms import pack_val_set_into_fit_params
from mlframe.metrics.core import compute_probabilistic_multiclass_error

from ._enums import OptimumSearch
from ._helpers import (
    _detect_multithreaded,
    _pin_threads_to_one,
    get_feature_importances,
    get_next_features_subset,
    split_into_train_test,
    store_averaged_cv_scores,
    suppress_irritating_3rdparty_warnings,
)
from ._rfecv_validate import _sanitize_X_inputs
from ._rfecv_cv_setup import _resolve_cv_and_val_cv
from ._rfecv_mbh_optimizer import _build_mbh_optimizer
from ._rfecv_finalize import _finalize_fit_results
from ._rfecv_checkpoint import _maybe_resume_from_checkpoint
from ._rfecv_must_include import _resolve_must_include

logger = logging.getLogger("mlframe.feature_selection.wrappers._rfecv")


def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.DataFrame, pd.Series, np.ndarray], groups: Union[pd.Series, np.ndarray] = None, sample_weight: Union[np.ndarray, pd.Series, None] = None, **fit_params):
    # sample_weight, when provided, is sliced per CV fold and threaded into both the cloned estimator's
    # ``fit(..., sample_weight=fold_train_w)`` call (if the estimator advertises support) and into the
    # sklearn scorer's ``__call__(..., sample_weight=fold_test_w)`` (if the scorer accepts the kwarg).
    # Default None preserves the legacy code path byte-for-byte (regression sentry); the gating flag
    # ``FeatureSelectionConfig.use_sample_weights_in_fs`` decides whether the caller forwards weights here.
    self._fit_sample_weight_ = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
    if self._fit_sample_weight_ is not None:
        _n_rows_for_sw = X.shape[0] if hasattr(X, "shape") else len(X)
        if self._fit_sample_weight_.ndim != 1:
            raise ValueError(f"RFECV.fit sample_weight must be 1-D, got shape {self._fit_sample_weight_.shape}")
        if self._fit_sample_weight_.shape[0] != _n_rows_for_sw:
            raise ValueError(f"RFECV.fit sample_weight length {self._fit_sample_weight_.shape[0]} != n_rows {_n_rows_for_sw}")
        if not np.all(np.isfinite(self._fit_sample_weight_)) or (self._fit_sample_weight_ < 0).any():
            raise ValueError("RFECV.fit sample_weight must be finite and non-negative")

    # Polars -> pandas at entry. RFECV uses pandas / numpy idioms throughout (KFold.split, current_features.index(...), passthrough_cols).
    # Inner estimators (notably CatBoost) crash on polars Enum columns, so convert once here and let every downstream caller see pandas.
    # Before lossy conversion, stash a "polars-detected monotonic datetime axis" hint so the CV auto-detect block below can route
    # to TimeSeriesSplit. After to_pandas() the original polars schema is gone and the .index becomes a plain RangeIndex.
    _polars_time_series_hint = False
    if isinstance(X, pl.DataFrame):
        try:
            _dt_cols = [
                n for n, d in X.schema.items()
                if d in (pl.Datetime, pl.Date) or str(d).startswith(("Datetime", "Date"))
            ]
            if len(_dt_cols) == 1:
                _col = X.get_column(_dt_cols[0])
                if _col.is_sorted(descending=False) and _col.null_count() == 0:
                    _polars_time_series_hint = True
        except Exception:
            pass
        try:
            X = X.to_pandas(use_pyarrow_extension_array=True, split_blocks=True, self_destruct=True)
        except TypeError:
            X = X.to_pandas()
    if isinstance(y, pl.Series):
        y = y.to_pandas()

    # Reject pathological y / X early instead of letting sklearn raise opaque errors deep in the splitter or estimator.
    try:
        y_arr = np.asarray(y)
    except Exception as exc:
        raise ValueError(f"y must be array-like; got {type(y).__name__}: {exc}") from exc
    if y_arr.size == 0:
        raise ValueError("y is empty; nothing to fit.")
    # NaN / Inf in y are silent miscompute traps in sklearn folds.
    if y_arr.dtype.kind in "fc":
        n_nan_y = int(np.isnan(y_arr).sum())
        n_inf_y = int(np.isinf(y_arr).sum())
        if n_nan_y or n_inf_y:
            raise ValueError(
                f"y contains {n_nan_y} NaN and {n_inf_y} +/-inf values. "
                f"sklearn CV splitters silently mishandle these. Drop or "
                f"impute these rows before passing y to RFECV."
            )
    # Single-class y for classification is a fold-collapse trap.
    if is_classifier(self.estimator if self.estimator is not None
                     else (self.estimators[0] if self.estimators else None)):
        unique_y = np.unique(y_arr)
        if len(unique_y) < 2:
            raise ValueError(
                f"y has only {len(unique_y)} unique class(es) "
                f"({unique_y.tolist()}). Classification CV requires at "
                f"least 2 classes. Check that y is not constant or that "
                f"upstream filtering didn't drop the minority class."
            )
        # Minority-class size must support the requested CV.
        class_counts = np.bincount(y_arr.astype(int)) if y_arr.dtype.kind in "iu" else None
        if class_counts is not None and len(class_counts) > 0:
            min_class = int(class_counts[class_counts > 0].min())
            cv_n = self.cv if isinstance(self.cv, int) else getattr(self.cv, "n_splits", 3)
            if min_class < cv_n:
                raise ValueError(
                    f"Minority class has {min_class} samples but cv={cv_n}. "
                    f"StratifiedKFold requires at least n_splits samples per "
                    f"class. Reduce cv or oversample the minority class."
                )

    # must_include + must_exclude intersection is a confusing config error.
    if self.must_include and self.must_exclude:
        mi_set = set(self.must_include)
        me_set = set(self.must_exclude)
        overlap = mi_set & me_set
        if overlap:
            raise ValueError(
                f"must_include and must_exclude both contain {sorted(overlap)}. "
                f"Resolve the conflict in your config."
            )

    # X-side input checks. Run after y validation so common operator mistakes surface clearly at fit entry.
    if isinstance(X, pd.DataFrame):
        # Estimator behaviour on Inf is undefined - LR crashes, CB silently treats as huge.
        try:
            _num = X.select_dtypes(include="number")
            if _num.size > 0:
                _inf_mask = np.isinf(_num.to_numpy())
                if _inf_mask.any():
                    _inf_cols = _num.columns[_inf_mask.any(axis=0)].tolist()
                    raise ValueError(
                        f"X contains +/-Inf values in column(s) {_inf_cols[:10]}. "
                        f"Estimator behaviour on Inf is undefined. Drop or "
                        f"clip these values before fit()."
                    )
        except (TypeError, ValueError) as exc:
            if "+/-Inf" in str(exc):
                raise

        # Tree-based estimators handle NaN; linear models don't.
        if getattr(self, "verbose", 0):
            _nan_count = int(X.isna().to_numpy().sum())
            if _nan_count > 0:
                logger.warning(
                    "RFECV: X has %d NaN cells. Tree-based estimators "
                    "(RF/CB/XGB/HGBM) handle NaN; linear models (LR, "
                    "Ridge, Lasso) do NOT and will crash on .fit(). "
                    "Pre-impute via SimpleImputer / KNNImputer if using "
                    "linear estimators.", _nan_count,
                )

        _obj_cols = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
        if _obj_cols:
            _user_cats = set(self.cat_features or [])
            _unhandled = [c for c in _obj_cols if c not in _user_cats]
            if _unhandled and getattr(self, "verbose", 0):
                logger.warning(
                    "RFECV: %d object/string/category column(s) %s have "
                    "NOT been listed in cat_features=. CB/XGB will crash "
                    "on string columns; LR will fail on .fit(). Either "
                    "encode them upstream or pass via cat_features.",
                    len(_unhandled), _unhandled[:10],
                )

    # n_samples < 2 * cv breaks every k-fold split.
    cv_n = self.cv if isinstance(self.cv, int) else getattr(self.cv, "n_splits", 3)
    if X.shape[0] < 2 * cv_n:
        raise ValueError(
            f"n_samples={X.shape[0]} < 2 * cv ({cv_n}); each fold would "
            f"have <2 train samples. Reduce cv or get more data."
        )

    X = _sanitize_X_inputs(self, X, y)

    # Inputs/outputs signature. Shape alone isn't enough - two datasets with identical (n, p) but different column identities must
    # trigger a retrain, otherwise self.support_ silently applies stale column selections. y-content is folded in via a blake2b
    # 16-byte digest because two semantically-different targets of the same length and shape (e.g. column-A binary vs column-B
    # binary picked off the same frame) used to replay the prior fit's support_; without the y-hash a per-target FS loop reusing
    # one RFECV instance silently selected features for whichever target arrived first.
    if isinstance(X, pd.DataFrame):
        columns_key = tuple(map(str, X.columns.tolist()))
    else:
        columns_key = ("__ndarray__", int(X.shape[1]))
    try:
        _y_arr = np.ascontiguousarray(
            y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)
        )
        _y_hash = hashlib.blake2b(_y_arr.tobytes(), digest_size=16).hexdigest()
    except (TypeError, ValueError):
        # Object-dtype y or otherwise non-bytes-castable; fall back to a stringified per-element hash that still discriminates content.
        _y_hash = hashlib.blake2b(
            ",".join(map(str, np.asarray(y).ravel().tolist())).encode("utf-8"),
            digest_size=16,
        ).hexdigest()
    # X-content sample (10 evenly-spaced rows, all columns) so two RUNS on
    # different X with identical shape + column names cannot collide on the
    # checkpoint signature. Cost is O(n_cols * 10) value reads -- negligible
    # vs a single RFECV iteration. Without this, a user reusing a checkpoint
    # across experiments with same schema gets silent stale resume.
    try:
        _n = int(X.shape[0])
        if _n > 0:
            _step = max(1, _n // 10)
            _positions = [i * _step for i in range(10) if i * _step < _n]
            if isinstance(X, pd.DataFrame):
                _x_sample_arr = X.iloc[_positions].to_numpy()
            else:
                _x_sample_arr = np.asarray(X)[_positions]
            _x_hash = hashlib.blake2b(
                np.ascontiguousarray(_x_sample_arr.astype(np.float64, copy=False, casting="unsafe") if _x_sample_arr.dtype.kind in "biufc" else np.asarray(_x_sample_arr, dtype=str)).tobytes(),
                digest_size=12,
            ).hexdigest()
        else:
            _x_hash = "empty"
    except (TypeError, ValueError):
        _x_hash = hashlib.blake2b(
            repr(np.asarray(X).ravel()[:100].tolist()).encode("utf-8"),
            digest_size=12,
        ).hexdigest()
    signature = (X.shape, y.shape, columns_key, _y_hash, _x_hash)
    # Invalidate stale support_/cache at fit entry so a partial-fit failure cannot leave a previous-fit's selection silently in place.
    # The cache is rebuilt below only on a successful path.
    self._selected_cols_cache = None
    if self.skip_retraining_on_same_shape:
        if signature == self.signature:
            if self.verbose:
                logger.info("Skipping retraining on the same inputs signature %s", signature)
            return self

    # Stability selection branch: uses bootstrap voting instead of the MBH+CV-fold-voting search. Returns early after setting
    # support_ / n_features_ / cv_results_-shim / feature_names_in_ etc.
    if self.stability_selection:
        return self._fit_stability_selection(X=X, y=y, signature=signature)

    # ``estimators`` (list) supersedes the singular ``estimator``. Work with a list internally; singular path is a len-1 list.
    # Score per fold = mean across estimators; FI runs stored under separate keys so the voting layer treats each estimator's
    # importance as an independent "run".
    estimators_list = list(self.estimators) if self.estimators else (
        [self.estimator] if self.estimator is not None else []
    )
    if not estimators_list:
        raise ValueError("RFECV requires either estimator= or estimators=.")
    # ``estimator`` retained for legacy code paths inside fit() that need a single object for type-dispatch (CV stratification,
    # scoring, importance_getter resolution). First estimator is the representative.
    estimator = estimators_list[0]
    # Extract suite-level ``timestamps`` hint BEFORE we shadow ``fit_params`` with ``self.fit_params``;
    # the time-series auto-detect block below picks it up to upgrade plain int cv -> TimeSeriesSplit
    # for callers whose X has no DatetimeIndex / polars datetime column but who know the row order
    # is temporal (e.g. epoch-seconds in a separate array).
    _ts_hint_from_caller = fit_params.pop("timestamps", None) if isinstance(fit_params, dict) else None
    fit_params = copy.copy(self.fit_params) if self.fit_params else {}
    if _ts_hint_from_caller is not None:
        fit_params["timestamps"] = _ts_hint_from_caller
    max_runtime_mins = self.max_runtime_mins
    max_refits = self.max_refits
    cv = self.cv
    cv_shuffle = self.cv_shuffle
    early_stopping_val_nsplits = self.early_stopping_val_nsplits
    early_stopping_rounds = self.early_stopping_rounds
    scoring = self.scoring
    top_predictors_search_method = self.top_predictors_search_method
    votes_aggregation_method = self.votes_aggregation_method
    use_all_fi_runs = self.use_all_fi_runs
    use_last_fi_run_only = self.use_last_fi_run_only
    use_one_freshest_fi_run = self.use_one_freshest_fi_run
    use_fi_ranking = self.use_fi_ranking
    importance_getter = self.importance_getter
    random_state = self.random_state
    # When random_state is None, derive a stable seed from the signature so re-fits on the SAME data are deterministic. Otherwise
    # self._rng is reseeded from system entropy on every fit, breaking reproducibility silently.
    # Use ``hashlib.blake2b`` (not Python's builtin ``hash``) so the seed is bit-stable across processes /
    # runs / PYTHONHASHSEED values. The signature tuple contains strings (column names + hex digests) and
    # the builtin hash of strings is salted per process, which silently broke the "same data -> same support_"
    # guarantee across worker spawns. The hashlib path mirrors the _y_hash / _x_hash construction ~30 lines
    # above so the in-tree style stays consistent.
    if random_state is None:
        import hashlib as _hashlib
        _sig_bytes = repr(signature).encode("utf-8", errors="replace")
        _seed = int.from_bytes(
            _hashlib.blake2b(_sig_bytes, digest_size=4).digest(), "big",
        )
        self._rng = np.random.default_rng(_seed)
    else:
        self._rng = np.random.default_rng(random_state)
    leave_progressbars = self.leave_progressbars
    verbose = self.verbose
    show_plot = self.show_plot
    cat_features = self.cat_features
    # Strip cat_features whose columns have already been numerically encoded by an upstream pipeline step (e.g. CatBoostEncoder turning
    # cat_0 -> float). With cat_0 still in cat_features, inner CatBoost.fit raises ``Invalid type for cat_feature``. Restrict to columns
    # whose dtype is still categorical/object - those are the ones CB/XGB can actually consume. LOCAL only - never mutate
    # self.cat_features (back-to-back fits across encoded/un-encoded frames must each pick the right subset for their X).
    if cat_features and isinstance(X, pd.DataFrame):
        try:
            _consumable_kinds = {"O", "U", "S"}
            _consumable = []
            for _c in cat_features:
                if _c not in X.columns:
                    continue
                _dt = X[_c].dtype
                if str(_dt).startswith("category") or getattr(_dt, "kind", "") in _consumable_kinds:
                    _consumable.append(_c)
            if len(_consumable) != len(cat_features):
                if verbose:
                    logger.info(
                        "wrappers.fit: %d/%d cat_features kept after dtype check; "
                        "the rest (%s) appear numerically encoded upstream and "
                        "are skipped for the inner estimator.",
                        len(_consumable), len(cat_features),
                        [c for c in cat_features if c not in _consumable],
                    )
                cat_features = _consumable
        except Exception:
            pass
    keep_estimators = self.keep_estimators
    feature_cost = self.feature_cost
    smooth_perf = self.smooth_perf
    frac = self.frac
    best_desired_score = self.best_desired_score
    max_noimproving_iters = self.max_noimproving_iters

    # Resolve effective n_jobs. Multi-threaded estimators (CB/LGB/XGB/RF/...) already use all cores natively; parallelising folds on top
    # oversubscribes and SLOWS DOWN. Auto-fallback to sequential unless force_parallel=True (then pin inner threads to 1 in _eval_fold).
    n_jobs_effective = int(self.n_jobs) if self.n_jobs else 1
    if n_jobs_effective < 0:
        # joblib semantics: -1 = all cores.
        try:
            import os as _os
            n_jobs_effective = max(1, (_os.cpu_count() or 1))
        except Exception:
            n_jobs_effective = 1
    _is_multithreaded = _detect_multithreaded(estimator)
    if n_jobs_effective > 1 and _is_multithreaded and not self.force_parallel:
        if verbose:
            logger.info(
                "RFECV: n_jobs=%d requested, but %s already uses native "
                "multi-threading. Auto-falling back to sequential CV folds "
                "to avoid core oversubscription. Pass ``force_parallel=True`` "
                "to override (will pin inner threads to 1).",
                n_jobs_effective, type(estimator).__name__,
            )
        n_jobs_effective = 1
    ndigits = self.report_ndigits

    start_time = timer()
    ran_out_of_time = False
    if max_runtime_mins:
        if verbose:
            logger.info("max_runtime_mins=%.2f", max_runtime_mins)

    if random_state is not None:
        set_random_seed(random_state)

    feature_importances = {}
    evaluated_scores_std = {}
    evaluated_scores_mean = {}

    original_features = X.columns.tolist() if isinstance(X, pd.DataFrame) else np.arange(X.shape[1])

    # must_include partition: the optimiser only sees the complement; pinned features are glued back into support_ at the end.
    original_features, must_include_resolved = _resolve_must_include(self, X=X, original_features=original_features, verbose=verbose)

    cv, val_cv, early_stopping_rounds = _resolve_cv_and_val_cv(
        cv=cv, X=X, y=y, groups=groups, estimator=estimator,
        cv_shuffle=cv_shuffle, random_state=random_state,
        fit_params=fit_params,
        early_stopping_val_nsplits=early_stopping_val_nsplits,
        early_stopping_rounds=early_stopping_rounds,
        _polars_time_series_hint=_polars_time_series_hint,
        verbose=verbose,
    )
    # Expose the resolved splitter for introspection (auto-detect tests / callers checking which CV strategy was picked).
    self.cv_ = cv

    progressbar_prefix = "RFECV:"
    iters_pbar = tqdmu(
        desc=progressbar_prefix,
        leave=leave_progressbars,
        total=min(len(original_features) + 1, max_refits) if max_refits else len(original_features) + 1,
    )

    suppress_irritating_3rdparty_warnings()

    if scoring is None:
        if is_classifier(estimator):
            logger.info(f"Scoring omitted, using probabilistic_multiclass_error by default.")
            # response_method='predict_proba' for sklearn 1.4+ (needs_proba is deprecated).
            scoring = make_scorer(score_func=compute_probabilistic_multiclass_error, response_method="predict_proba", greater_is_better=False)
        elif is_regressor(estimator):
            logger.info(f"Scoring omitted, using mean_squared_error by default.")
            scoring = make_scorer(score_func=mean_squared_error, greater_is_better=False)
        else:
            raise ValueError(f"Appropriate scoring not known for estimator type: {estimator}")
        self.scoring = scoring

    if verbose:
        logger.info("Scoring=%s", scoring)

    estimator_type = type(estimator.steps[-1][1]).__name__ if isinstance(estimator, Pipeline) else type(estimator).__name__

    # Defer importance_getter dispatch to get_feature_importances (it looks at the FITTED model's attributes);
    # a hardcoded ``LogisticRegression -> coef_, else -> feature_importances_`` crashes on Ridge / Lasso / SVC(linear) / SGDClassifier etc.
    if importance_getter is None:
        importance_getter = "auto"

    nsteps = 0
    dummy_scores = []
    fitted_estimators = {}
    selected_features_per_nfeatures = {}

    Optimizer = _build_mbh_optimizer(
        self,
        original_features=original_features,
        max_refits=max_refits,
        top_predictors_search_method=top_predictors_search_method,
    )

    prev_score, prev_nfeatures = -np.inf, 0
    n_noimproving_iters = 0
    best_nfeatures = 0
    best_iter = 0
    # -inf is the only safe initial value for an argmax-style accumulator: any finite floor (e.g. -1e6) is breached routinely by
    # negative-MSE scorers on noisy regression, leaving ``final_score > best_score`` False forever and tripping max_noimproving_iters
    # prematurely.
    best_score = -np.inf

    # Resume-from-checkpoint: restore mutable outer-loop state iff the checkpoint signature matches the current
    # (X.shape, y.shape, columns_key). Mismatch silently starts fresh, so users can keep the same checkpoint_path across experiments.
    _ckpt_state = _maybe_resume_from_checkpoint(self, signature=signature, verbose=verbose, state={
        "nsteps": nsteps,
        "evaluated_scores_mean": evaluated_scores_mean,
        "evaluated_scores_std": evaluated_scores_std,
        "feature_importances": feature_importances,
        "selected_features_per_nfeatures": selected_features_per_nfeatures,
        "prev_score": prev_score,
        "prev_nfeatures": prev_nfeatures,
        "n_noimproving_iters": n_noimproving_iters,
        "best_nfeatures": best_nfeatures,
        "best_iter": best_iter,
        "best_score": best_score,
        "dummy_scores": dummy_scores,
        "Optimizer": Optimizer,
    })
    nsteps = _ckpt_state["nsteps"]
    evaluated_scores_mean = _ckpt_state["evaluated_scores_mean"]
    evaluated_scores_std = _ckpt_state["evaluated_scores_std"]
    feature_importances = _ckpt_state["feature_importances"]
    selected_features_per_nfeatures = _ckpt_state["selected_features_per_nfeatures"]
    prev_score = _ckpt_state["prev_score"]
    prev_nfeatures = _ckpt_state["prev_nfeatures"]
    n_noimproving_iters = _ckpt_state["n_noimproving_iters"]
    best_nfeatures = _ckpt_state["best_nfeatures"]
    best_iter = _ckpt_state["best_iter"]
    best_score = _ckpt_state["best_score"]
    dummy_scores = _ckpt_state["dummy_scores"]
    Optimizer = _ckpt_state["Optimizer"]

    # Baseline RSS + best-effort frame footprint so the RAM-aware ``maybe_clean_ram_and_gpu`` short-circuit can decide whether a ``gc.collect()`` is justified each iter. The old "every 5th iter" trigger ran a ~290ms gc.collect even when nothing had accumulated, dominating wall on small problems; the helper only fires when RSS actually grew past a threshold or free RAM gets tight relative to frame size.
    from mlframe.training._ram_helpers import estimate_df_size_mb as _estimate_df_size_mb, get_process_rss_mb as _get_process_rss_mb, maybe_clean_ram_and_gpu as _maybe_clean_ram_and_gpu
    _ram_baseline_mb = _get_process_rss_mb()
    try:
        _ram_df_size_mb = _estimate_df_size_mb(X)
    except Exception:
        _ram_df_size_mb = 0.0

    while nsteps < len(original_features):

        # Select current set of features to work on, based on ranking received so far and the optimum search method.

        # RAM-aware GC: ``maybe_clean_ram_and_gpu`` checks RSS-vs-baseline and free-vs-frame BEFORE invoking gc.collect, so the ~290ms cost is paid only when something actually accumulated. The old unconditional ``clean_ram()`` every-5-iters trigger fired even on small problems where no garbage existed, dominating wall.
        _ram_baseline_mb = _maybe_clean_ram_and_gpu(
            _ram_baseline_mb, _ram_df_size_mb, verbose=False, reason=f"RFECV iter {nsteps}"
        )

        if self.special_feature_indices is not None and len(self.special_feature_indices) > 0:
            current_features = self.special_feature_indices
        else:

            current_features = get_next_features_subset(
                nsteps=nsteps,
                original_features=original_features,
                feature_importances=feature_importances,
                evaluated_scores_mean=evaluated_scores_mean,
                evaluated_scores_std=evaluated_scores_std,
                use_all_fi_runs=use_all_fi_runs,
                use_last_fi_run_only=use_last_fi_run_only,
                use_one_freshest_fi_run=use_one_freshest_fi_run,
                use_fi_ranking=use_fi_ranking,
                top_predictors_search_method=top_predictors_search_method,
                votes_aggregation_method=votes_aggregation_method,
                Optimizer=Optimizer,
            )

        if current_features is None or len(current_features) == 0:
            break  # nothing more to try
        if self.stop_file and exists(self.stop_file):
            logger.warning(f"Stop file {self.stop_file} detected, quitting.")
            break

        desc = f"{progressbar_prefix} trying {len(current_features):_}F"
        if nsteps > 0:
            desc += f", had {prev_nfeatures:_}F with score {prev_score:.{ndigits}f}, best was {best_nfeatures:_}F with score {best_score:.{ndigits}f} @iter={best_iter:_} "
        iters_pbar.set_description(
            desc,
            refresh=True,
        )

        # Defer the selected_features_per_nfeatures write until after we know whether this exploration's score beats the existing best
        # at the same N. An unconditional write silently downgrades a winning subset whenever MBH revisits the same N with a worse one.

        # Each split must be different: even with a fixed random_state, the per-iter CV random_state is derived deterministically.
        scores = []

        splitter = cv.split(X=X, y=y, groups=groups)

        # Pre-materialise fold args so we can dispatch sequentially or in parallel from the same code path. Each fold gets its own
        # pre-derived RNG seed rather than sharing self._rng, which would race in a parallel context.
        _fold_args: list = []
        for _nfold, (_tr_idx, _te_idx) in enumerate(splitter):
            _fold_seed = int(self._rng.integers(0, 2**31 - 1))
            _fold_args.append((_nfold, _tr_idx, _te_idx, _fold_seed))

        def _eval_fold(nfold, train_index, test_index, fold_seed, current_features=current_features, scores=scores):
            """Per-fold evaluation. Returns a dict or None on skip. Fold-local state (RNG, estimator clone, fit_params) is built fresh inside.

            ``current_features`` and ``scores`` are passed as default args so they bind at def-time to the current outer-iter values; this is
            safe because the closure is created fresh each outer iter and consumed within that iter (sequentially or via joblib.Parallel).
            """
            if self.min_train_size and len(train_index) < self.min_train_size:
                return None
            if frac:
                size = int(len(train_index) * frac)
                if size > 10:
                    # Per-fold local RNG seeded deterministically; avoids races on self._rng when joblib runs folds in parallel.
                    local_rng = np.random.default_rng(fold_seed)
                    train_index = local_rng.choice(train_index, size=size, replace=False)

            # Actual fit/score uses must_include + optimiser's pick. current_features already lives in the search-universe complement
            # (must_include filtered out at fit entry), so concatenation never duplicates.
            if must_include_resolved:
                fit_features = list(must_include_resolved) + list(current_features)
            else:
                fit_features = current_features
            X_train, y_train, X_test, y_test = split_into_train_test(
                X=X, y=y, train_index=train_index, test_index=test_index, features_indices=fit_features
            )  # this splits both dataframes & ndarrays in the same fashion
            if verbose > 2:
                print(f"Train set size={len(y_train):_}, train idx sum={train_index.sum():_}")

            if val_cv and has_early_stopping_support(estimator_type):

                # Additional early-stopping split from X_train.
                if groups is not None:
                    if isinstance(groups, pd.Series):
                        train_groups = groups.iloc[train_index]
                    else:
                        train_groups = groups[train_index]
                else:
                    train_groups = None

                splits = list(val_cv.split(X=X_train, y=y_train, groups=train_groups))
                true_train_index, val_index = splits[-1]

                X_train, y_train, X_val, y_val = split_into_train_test(X=X_train, y=y_train, train_index=true_train_index, test_index=val_index)
                if verbose > 2:
                    print(f"Val set size={len(y_val):_}, val idx sum={val_index.sum():_}")

                # If the estimator type is known, craft early-stopping fit params tailored to it.
                temp_cat_features = [current_features.index(var) for var in cat_features if var in current_features] if cat_features else None

                temp_fit_params = pack_val_set_into_fit_params(
                    model=estimator,
                    X_val=X_val,
                    y_val=y_val,
                    early_stopping_rounds=early_stopping_rounds,
                    cat_features=temp_cat_features,
                )
                # Filter feature-list keys (cat_features / text_features / embedding_features) coming in via fit_params to only columns
                # present in the current selector iteration. Otherwise names from the outer call reference columns dropped by the
                # current iteration and CB raises ``Error while processing column for feature 'cat_0'``.
                #
                # cat_features: pack_val_set_into_fit_params above already injected index-based temp_cat_features IFF that list was
                # non-empty. When empty (current_features doesn't intersect self.cat_features) we MUST still pass a name-list filtered
                # to current_features so CB doesn't fall back to auto-detect on numerically-encoded category columns (target-encoded
                # cats look like floats and trip CB's "Invalid type for cat_feature").
                # When current_features holds integer indices (ndarray X path), set-membership against string column names always
                # misses; in that case the user-supplied lists pass through unfiltered (the inner estimator can deal with missing
                # column references on its own).
                _features_are_integer = (
                    len(current_features) > 0
                    and isinstance(current_features[0], (int, np.integer))
                )
                _current_set = set(current_features) if not _features_are_integer else None
                _filtered_fit_params = {}
                for _k, _v in fit_params.items():
                    if _k == "cat_features":
                        if "cat_features" in temp_fit_params:
                            # temp's index-based cat_features wins; drop outer name list.
                            continue
                        if _v and _current_set is not None:
                            _filtered = [c for c in _v if c in _current_set]
                            _filtered_fit_params[_k] = _filtered or None
                        elif _v:
                            _filtered_fit_params[_k] = _v
                        continue
                    if _k in ("text_features", "embedding_features") and _v:
                        if _current_set is not None:
                            _filtered = [c for c in _v if c in _current_set]
                            _filtered_fit_params[_k] = _filtered or None
                        else:
                            _filtered_fit_params[_k] = _v
                        continue
                    _filtered_fit_params[_k] = _v
                temp_fit_params.update(_filtered_fit_params)

            else:

                temp_fit_params = {}
                X_val = None

            # Per-fold sample_weight slicing. Estimator-side: pass via temp_fit_params only when the estimator's
            # fit signature accepts sample_weight (sklearn convention). Scorer-side: detected at score-call time
            # below (sklearn's _BaseScorer.__call__ accepts sample_weight=).
            _fold_train_sw = None
            _fold_test_sw = None
            if self._fit_sample_weight_ is not None:
                _fold_train_sw = self._fit_sample_weight_[train_index]
                _fold_test_sw = self._fit_sample_weight_[test_index]

            # Fit on current train fold, score on test, get FI.

            # Always clone per fold via sklearn.base.clone. copy.copy is a SHALLOW copy that shares mutable state (cat_features list,
            # set_params side effects, warm_start buffers) across folds. clone() returns an unfitted estimator with the same
            # constructor params and NO fitted state.
            fitted_estimator = clone(estimator)

            # Dynamic CB ``text_processing`` calibration for THIS fold's clone (not the outer estimator). RFECV folds are typically
            # much smaller than the outer training set; with CB's default ``occurrence_lower_bound=50`` words that occur < 50 times in
            # the fold are pruned, leaving an empty dictionary and HANGING CB's C++ ``_train`` loop. ``compute_cb_text_processing``
            # returns a config that scales the floor proportionally to fold rows, or None when the fold is large enough.
            if val_cv and has_early_stopping_support(estimator_type):
                _temp_text_feats = _filtered_fit_params.get("text_features") or []
                if _temp_text_feats and "CatBoost" in type(fitted_estimator).__name__:
                    _fold_rows = X_train.shape[0] if hasattr(X_train, "shape") else None
                    _tp = compute_cb_text_processing(_fold_rows) if _fold_rows is not None else None
                    if _tp is not None and hasattr(fitted_estimator, "set_params"):
                        try:
                            fitted_estimator.set_params(text_processing=_tp)
                        except Exception as _tp_exc:
                            logger.warning(
                                "RFECV inner fold: failed to set CB text_processing "
                                "(fold_rows=%s, exc=%s).", _fold_rows, _tp_exc,
                            )

            # Empty-train guard: heavy upstream filtering (small n + outlier_detection + trainset_aging_limit) can collapse X_train /
            # y_train to 0 rows on a CV fold; CatBoost then raises "Labels variable is empty" deep in C++ Pool init. Skip the fold
            # cleanly with a NaN score; sklearn's RFECV does the same on degenerate inner folds.
            _x_n = X_train.shape[0] if hasattr(X_train, "shape") else None
            _y_n = len(y_train) if y_train is not None and hasattr(y_train, "__len__") else None
            if (_x_n is not None and _x_n == 0) or (_y_n is not None and _y_n == 0):
                # Always-on ERROR (not verbose-gated) so the operator sees the empty-fold collapse. Root cause is upstream filter
                # aggression (OD + trainset_aging_limit together can shrink the inner-CV training fraction below cv splits) - fix
                # that, not this guard.
                logger.error(
                    "wrappers.fit: skipping fold %s - empty train slice "
                    "(rows=%s, target_len=%s). Upstream filters reduced "
                    "the train batch to zero. Investigate "
                    "outlier_detection contamination + "
                    "trainset_aging_limit interactions.",
                    nfold, _x_n, _y_n,
                )
                scores.append(np.nan)
                feature_importances[f"{len(current_features)}_{nfold}"] = {}
                # The body has since been wrapped in a nested ``_eval_fold`` function so an outer-loop ``continue`` is no longer
                # legal; return None to skip the fold (matches the function's documented "or None on skip" contract).
                return None
            # Multi-estimator: fit ALL estimators (singular case is a len-1 list). Score per fold = mean across estimators; FI runs
            # stored under separate keys ("{N}_{fold}" for singular, "{N}_{fold}_e{idx}" for multi) so the voting layer treats each
            # estimator's importances as an independent run.
            _est_scores = []
            _est_fi_runs = []  # list of (key, fi_dict)
            for _est_idx, _est_proto in enumerate(estimators_list):
                if _est_idx == 0:
                    # First estimator was already cloned + text_processing tuned above as ``fitted_estimator``.
                    _fitted = fitted_estimator
                else:
                    _fitted = clone(_est_proto)
                    # Apply CB text_processing if applicable for THIS clone.
                    if val_cv and has_early_stopping_support(estimator_type):
                        _temp_text_feats = _filtered_fit_params.get("text_features") or []
                        if _temp_text_feats and "CatBoost" in type(_fitted).__name__:
                            _fold_rows = X_train.shape[0] if hasattr(X_train, "shape") else None
                            _tp = compute_cb_text_processing(_fold_rows) if _fold_rows is not None else None
                            if _tp is not None and hasattr(_fitted, "set_params"):
                                try:
                                    _fitted.set_params(text_processing=_tp)
                                except Exception:
                                    pass

                _model_type_name = type(_fitted).__name__
                _ctx = suppress_stdout_stderr() if _model_type_name in CATBOOST_MODEL_TYPES else nullcontext()
                # Build per-call fit kwargs: only attach sample_weight when the estimator's fit signature accepts it.
                # CatBoost / LGBM / sklearn estimators all accept it; some custom shims may not, so introspect.
                _per_est_fit_params = dict(temp_fit_params)
                if _fold_train_sw is not None and "sample_weight" not in _per_est_fit_params:
                    try:
                        import inspect as _inspect
                        _sig = _inspect.signature(_fitted.fit)
                        if "sample_weight" in _sig.parameters or any(p.kind == _inspect.Parameter.VAR_KEYWORD for p in _sig.parameters.values()):
                            _per_est_fit_params["sample_weight"] = _fold_train_sw
                    except (TypeError, ValueError):
                        pass
                with _ctx:
                    _fitted.fit(X=X_train, y=y_train, **_per_est_fit_params)
                # Scorer-side: forward fold-test sample_weight when sklearn scorer accepts it. make_scorer-wrapped
                # callables expose sample_weight via _BaseScorer.__call__; bare callables may not.
                if _fold_test_sw is not None:
                    try:
                        _score = scoring(_fitted, X_test, y_test, sample_weight=_fold_test_sw)
                    except TypeError:
                        _score = scoring(_fitted, X_test, y_test)
                else:
                    _score = scoring(_fitted, X_test, y_test)
                _est_scores.append(_score)

                # FI is computed on the actual fit_features.
                _fi_full = get_feature_importances(
                    model=_fitted, current_features=fit_features,
                    data=X_test, reference_data=X_val, target=y_test,
                    importance_getter=importance_getter,
                )
                if must_include_resolved:
                    must_set = set(must_include_resolved)
                    _fi = {k: v for k, v in _fi_full.items() if k not in must_set}
                else:
                    _fi = _fi_full

                _est_suffix = f"_e{_est_idx}" if len(estimators_list) > 1 else ""
                _key = f"{len(current_features)}_{nfold}{_est_suffix}"
                _est_fi_runs.append((_key, _fi))
                if keep_estimators:
                    fitted_estimators[_key] = _fitted

            # Aggregate fold score via worst-case (min) across estimators given sklearn's "higher is better". Mean would let one strong
            # estimator (e.g. RF on 2 informative features) mask the fact that another (e.g. LR) needs more features to converge;
            # worst-case forces N to be where ALL estimators agree it's sufficient. For the singular path (len-1 list), min == mean.
            if _est_scores:
                valid_scores = [s for s in _est_scores if not np.isnan(s)]
                score = float(np.min(valid_scores)) if valid_scores else float("nan")
            else:
                score = float("nan")
            scores.append(score)
            # Persist every estimator's FI run.
            for _k, _fi in _est_fi_runs:
                feature_importances[_k] = _fi

            if 0 not in evaluated_scores_mean:

                # Dummy baselines serve as fitness @ 0 features.
                if not self.nofeatures_dummy_scoring:
                    # Sign-direction-agnostic "worse than model" placeholder. ``score/10`` silently put the dummy ABOVE the model when
                    # _sign==+1 and score was negative (e.g. R^2 < 0). Subtracting a positive number always makes the score worse
                    # under both sklearn conventions; use a magnitude-relative fudge so it scales with the metric in play (log-loss
                    # ~0.5, MSE ~1e6, R^2 ~1).
                    fudge = max(abs(score), 1e-3) * 9.0
                    dummy_scores.append(score - fudge)
                else:
                    dummy_scores.append(
                        get_best_dummy_score(estimator=estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, scoring=scoring)
                    )

        # Dispatch _eval_fold sequentially or in parallel. n_jobs>1 uses prefer="threads" so we don't pickle X/y across workers
        # (datasets stay in shared memory) and the closure can keep mutating outer state under the GIL. When n_jobs>1 AND the
        # estimator is multi-threaded AND force_parallel=True, pin inner threads to 1.
        if n_jobs_effective > 1 and _is_multithreaded and self.force_parallel:
            _orig_eval_fold = _eval_fold
            def _eval_fold_pinned(*args, _orig=_orig_eval_fold):
                # The closure clones the estimator inside its body so we can't reach in. Pin once on the OUTER estimator; clone()
                # preserves params so each fold's clone inherits thread_count=1 / n_jobs=1.
                _pin_threads_to_one(estimator)
                return _orig(*args)
            _fold_runner = _eval_fold_pinned
        else:
            _fold_runner = _eval_fold

        if n_jobs_effective > 1 and len(_fold_args) > 1:
            from joblib import Parallel, delayed
            # prefer="threads": sklearn / CB / LGB / XGB all release GIL during fit, so threads give true parallelism without the
            # serialisation cost of multiprocessing.
            # Wave 27 P2 fix (2026-05-20): added ``require="sharedmem"``
            # to HARDEN the design. ``prefer`` is a soft hint that an
            # outer ``joblib.parallel_backend('loky')`` / sklearn
            # ``parallel_config`` can override; under loky the
            # ``_fold_runner`` closure-state mutations (``scores``,
            # ``feature_importances``, ``fitted_estimators``) happen
            # in worker process copies and the main-process state
            # silently stays empty -> ``final_score = nan`` with no
            # exception. ``require`` makes joblib RAISE if it can't
            # satisfy threading, surfacing the misconfiguration loud.
            Parallel(n_jobs=n_jobs_effective, prefer="threads", require="sharedmem")(
                delayed(_fold_runner)(*a) for a in _fold_args
            )
        else:
            if verbose:
                _iter = tqdmu(_fold_args, desc="CV folds", leave=False, total=len(_fold_args))
            else:
                _iter = _fold_args
            for a in _iter:
                _fold_runner(*a)

        if 0 not in evaluated_scores_mean:
            scores_mean, scores_std, final_score, _ = store_averaged_cv_scores(
                pos=0, scores=dummy_scores, evaluated_scores_mean=evaluated_scores_mean, evaluated_scores_std=evaluated_scores_std, self=self
            )
            nofeatures_score = final_score
            if verbose:
                logger.info(f"Baseline with 0 features, score={scores_mean:.{ndigits}f} +/- {scores_std:.{ndigits}f} ~ {final_score:.{ndigits}f}")
            if top_predictors_search_method == OptimumSearch.ModelBasedHeuristic:
                Optimizer.submit_evaluations(candidates=[0], evaluations=[final_score], durations=[None])

        scores_mean, scores_std, final_score, was_stored = store_averaged_cv_scores(
            pos=len(current_features),
            scores=scores,
            evaluated_scores_mean=evaluated_scores_mean,
            evaluated_scores_std=evaluated_scores_std,
            self=self,
        )
        # Only commit selected_features when this run actually won at its N.
        if was_stored:
            selected_features_per_nfeatures[len(current_features)] = current_features

        if top_predictors_search_method == OptimumSearch.ModelBasedHeuristic:
            Optimizer.submit_evaluations(candidates=[len(current_features)], evaluations=[final_score], durations=[None])

            if verbose:
                logger.info(
                    f"Tried {len(current_features):_} features ({textwrap.shorten(', '.join(map(str, current_features[:40])), 150)}), score={scores_mean:.{ndigits}f} +/- {scores_std:.{ndigits}f} ~ {final_score:.{ndigits}f}"
                )

        prev_nfeatures, prev_score = len(current_features), final_score
        iters_pbar.update(1)

        nsteps += 1

        # Persist outer-loop state so a crash mid-run is recoverable. fitted_estimators is intentionally NOT pickled (CB / RF
        # ensembles dominate file size); they are re-fit on resume when needed. Save errors are logged but do not abort the fit.
        if self.checkpoint_path is not None:
            try:
                self._save_checkpoint({
                    "version": self._CHECKPOINT_VERSION,
                    "signature": signature,
                    "nsteps": nsteps,
                    "evaluated_scores_mean": dict(evaluated_scores_mean),
                    "evaluated_scores_std": dict(evaluated_scores_std),
                    "feature_importances": dict(feature_importances),
                    "selected_features_per_nfeatures": dict(selected_features_per_nfeatures),
                    "prev_score": prev_score,
                    "prev_nfeatures": prev_nfeatures,
                    "n_noimproving_iters": n_noimproving_iters,
                    "best_nfeatures": best_nfeatures,
                    "best_iter": best_iter,
                    "best_score": best_score,
                    "dummy_scores": list(dummy_scores),
                    "optimizer": Optimizer,
                })
            except Exception as _ckpt_exc:
                if verbose:
                    logger.warning(
                        "RFECV: checkpoint save at nsteps=%d failed: %s",
                        nsteps, _ckpt_exc,
                    )

        if len(evaluated_scores_mean) == 2:
            # If the first explored subset (whatever MBH seeded; default seed = 2 features) is already worse than the dummy at 0
            # features, there's no point continuing.
            if final_score < nofeatures_score:
                logger.info(
                    f"Stopping RFECV early: dummy baseline at 0 features ({nofeatures_score:.{ndigits}f}) "
                    f"already beats the first explored subset of {len(current_features)} features "
                    f"({final_score:.{ndigits}f}). The model can't learn anything from this data."
                )
                break

        if max_runtime_mins and not ran_out_of_time:
            delta = timer() - start_time
            ran_out_of_time = delta > (max_runtime_mins * 60)
            if ran_out_of_time:
                if verbose:
                    logger.info(f"max_runtime_mins={max_runtime_mins:_.1f} reached.")
                break

        if max_refits and nsteps >= max_refits:
            if verbose:
                logger.info(f"max_refits={max_refits:_} reached.")
            break

        if final_score > best_score:
            best_score = final_score
            best_iter = nsteps
            best_nfeatures = len(current_features)
            n_noimproving_iters = 0
        else:
            n_noimproving_iters += 1

        if best_desired_score is not None and final_score > best_desired_score:
            if verbose:
                logger.info(f"best_desired_score {best_desired_score:_.{ndigits}f} reached.")
            break

        if max_noimproving_iters and n_noimproving_iters >= max_noimproving_iters:
            if verbose:
                logger.info("Max # of noimproved iters reached: %s", n_noimproving_iters)
            break

        # Abort early if every iter so far produced a NaN final_score. The most common cause is a custom scorer returning NaN on every
        # fold (e.g. ROC AUC on single-class CV folds). Without this, the noimproving counter would consume max_noimproving_iters
        # worth of useless CV fits. Detect 5 consecutive NaN iters and bail.
        if np.isnan(final_score):
            if not hasattr(self, "_consecutive_nan_iters"):
                self._consecutive_nan_iters = 0
            self._consecutive_nan_iters += 1
            if self._consecutive_nan_iters >= 5:
                raise RuntimeError(
                    "RFECV: scoring returned NaN on 5 consecutive iters. "
                    "Likely cause: custom scorer NaN-ing on degenerate folds, "
                    "or single-class fold with a binary-only metric. Switch "
                    "to a NaN-safe scorer or pass cv=StratifiedKFold(...)."
                )
        else:
            self._consecutive_nan_iters = 0

        if self.special_feature_indices is not None:
            if verbose:
                logger.info(f"Quitting as special_feature_indices were checked.")
            break

    # Truncated SFFS final-pass swap: run K paired swaps on the best subset found - replace each of the K worst-FI kept features with
    # each of the K best-FI dropped features, accept any swap that improves the CV score. Uses sklearn.cross_val_score directly so it
    # does NOT honour fit_params / val_cv / early stopping.
    _finalize_fit_results(
        self,
        X=X, y=y, estimator=estimator, cv=cv, scoring=scoring,
        best_nfeatures=best_nfeatures, best_score=best_score,
        selected_features_per_nfeatures=selected_features_per_nfeatures,
        feature_importances=feature_importances,
        original_features=original_features,
        evaluated_scores_mean=evaluated_scores_mean,
        evaluated_scores_std=evaluated_scores_std,
        verbose=verbose, ndigits=ndigits,
        fitted_estimators=fitted_estimators,
        must_include_resolved=must_include_resolved,
        feature_cost=feature_cost, smooth_perf=smooth_perf,
        use_all_fi_runs=use_all_fi_runs,
        use_last_fi_run_only=use_last_fi_run_only,
        use_one_freshest_fi_run=use_one_freshest_fi_run,
        use_fi_ranking=use_fi_ranking,
        votes_aggregation_method=votes_aggregation_method,
        show_plot=show_plot,
        signature=signature,
    )
    return self
