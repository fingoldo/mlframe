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
import logging
import textwrap
from os.path import exists
from timeit import default_timer as timer
from typing import Union

import numpy as np
import pandas as pd

from pyutilz.system import tqdmu

from sklearn.pipeline import Pipeline

from mlframe.utils.misc import set_random_seed

from ._enums import OptimumSearch
from ._helpers import (
    _pin_threads_to_one,
    get_next_features_subset,
    store_averaged_cv_scores,
    suppress_irritating_3rdparty_warnings,
)
from ._rfecv_cv_setup import _resolve_cv_and_val_cv
from ._rfecv_mbh_optimizer import _build_mbh_optimizer
from ._rfecv_finalize import _finalize_fit_results
from ._rfecv_checkpoint import _maybe_resume_from_checkpoint
from ._rfecv_must_include import _resolve_must_include
from ._rfecv_fit_init import _init_fit_state
from ._rfecv_fit_setup import (
    filter_cat_features_by_dtype,
    resolve_default_scoring,
    resolve_effective_n_jobs,
)
from ._rfecv_fit_fold import _eval_fold_body

logger = logging.getLogger("mlframe.feature_selection.wrappers._rfecv")


def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.DataFrame, pd.Series, np.ndarray], groups: Union[pd.Series, np.ndarray] = None, sample_weight: Union[np.ndarray, pd.Series, None] = None, **fit_params):
    X, y, signature, _polars_time_series_hint, _init_skip = _init_fit_state(
        self, X, y, groups, sample_weight,
    )
    if _init_skip:
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
    cat_features = filter_cat_features_by_dtype(X, self.cat_features, verbose)
    keep_estimators = self.keep_estimators
    feature_cost = self.feature_cost
    smooth_perf = self.smooth_perf
    frac = self.frac
    best_desired_score = self.best_desired_score
    max_noimproving_iters = self.max_noimproving_iters

    n_jobs_effective, _is_multithreaded = resolve_effective_n_jobs(
        self.n_jobs, estimator, self.force_parallel, verbose,
    )
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
        scoring = resolve_default_scoring(scoring, estimator)
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
    from mlframe.training import estimate_df_size_mb as _estimate_df_size_mb, get_process_rss_mb as _get_process_rss_mb, maybe_clean_ram_and_gpu as _maybe_clean_ram_and_gpu
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

        # ``current_features`` and ``scores`` are passed as default args so they bind at def-time to the current outer-iter values; this is safe because the closure is created fresh each outer iter and consumed within that iter (sequentially or via joblib.Parallel).
        def _eval_fold(nfold, train_index, test_index, fold_seed, _current_features=current_features, _scores=scores):
            return _eval_fold_body(
                nfold, train_index, test_index, fold_seed,
                self=self,
                current_features=_current_features,
                frac=frac,
                must_include_resolved=must_include_resolved,
                X=X, y=y,
                val_cv=val_cv,
                estimator_type=estimator_type,
                groups=groups,
                verbose=verbose,
                cat_features=cat_features,
                early_stopping_rounds=early_stopping_rounds,
                fit_params=fit_params,
                estimators_list=estimators_list,
                estimator=estimator,
                scoring=scoring,
                importance_getter=importance_getter,
                keep_estimators=keep_estimators,
                evaluated_scores_mean=evaluated_scores_mean,
                scores=_scores,
                feature_importances=feature_importances,
                fitted_estimators=fitted_estimators,
                dummy_scores=dummy_scores,
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
    try:
        from mlframe.training.provenance import record_provenance as _record_provenance
        _n_rows_done = int(X.shape[0]) if hasattr(X, "shape") else None
        _cv_n_done = self.cv if isinstance(self.cv, int) else getattr(self.cv, "n_splits", None)
        _record_provenance(
            getattr(self, "_provenance_sink_", None),
            "rfecv",
            source="train_only",
            n_rows=_n_rows_done,
            seed=int(getattr(self, "random_state", 0) or 0) if getattr(self, "random_state", None) is not None else None,
            extra={"cv_folds": int(_cv_n_done) if _cv_n_done is not None else None, "n_features_in": int(X.shape[1]) if hasattr(X, "shape") and len(X.shape) > 1 else None},
        )
        self.provenance_ = {
            "step": "rfecv", "source": "train_only", "n_rows": _n_rows_done,
            "cv_folds": int(_cv_n_done) if _cv_n_done is not None else None,
        }
    except Exception:
        pass
    return self
