"""Outer while-loop body of ``RFECV.fit`` carved out of ``_rfecv_fit.py``.

The outer while-loop in the original ``fit`` body owned ~13 mutable locals + 6 break conditions + the checkpoint persistence + the Optimizer.submit_evaluations calls + the RAM-aware GC hook. This sibling packages those mutables in ``OuterLoopState`` (a plain dataclass) and exposes ``run_outer_loop_iteration`` which executes ONE iteration and returns an ``IterationOutcome`` signaling whether the parent should continue, break, or raise.

Behavioural equivalence: every named local that crossed an iteration boundary in the original is now a dataclass field; every break path in the original is now a ``ShouldBreak`` outcome value the parent translates back to ``break``; the consecutive-NaN raise is preserved verbatim. The four shared mutable containers (``scores`` per iter is fresh; ``evaluated_scores_mean`` / ``evaluated_scores_std`` / ``feature_importances`` / ``fitted_estimators`` / ``selected_features_per_nfeatures`` / ``dummy_scores`` survive across iterations and are owned by the state).
"""
from __future__ import annotations

import logging
import textwrap
from dataclasses import dataclass, field
from enum import Enum
from os.path import exists
from timeit import default_timer as timer
from typing import Any, Optional

import numpy as np

from ._enums import OptimumSearch, VotesAggregation
from ._helpers import get_next_features_subset, store_averaged_cv_scores
from ._rfecv_fit_fold import _eval_fold_body


logger = logging.getLogger("mlframe.feature_selection.wrappers._rfecv")


class IterationOutcome(Enum):
    """Outcome of a single outer-loop iteration.

    Mirrors the original branching: an early ``break`` inside the iter body becomes a ``ShouldBreak`` return; a normal end-of-iter becomes ``ShouldContinue``. The 5-consecutive-NaN raise is still a raise (RuntimeError surfaces to the caller exactly as before).
    """

    CONTINUE = "continue"
    BREAK = "break"


@dataclass
class OuterLoopState:
    """Outer-while-loop mutable state for ``RFECV.fit``.

    Every name in this dataclass crossed an iteration boundary in the pre-carve body. Fresh-per-iter locals (``current_features``, ``splitter``, ``_fold_args``, ``scores``) are NOT stored here; they live as iteration-local variables inside ``run_outer_loop_iteration``.

    The mutable containers (``evaluated_scores_mean``, ``evaluated_scores_std``, ``feature_importances``, ``fitted_estimators``, ``selected_features_per_nfeatures``, ``dummy_scores``) are owned by the state object; ``_finalize_fit_results`` reads them directly from the state at end-of-fit, so the parent passes ``state.evaluated_scores_mean`` etc forward into finalize unchanged.
    """

    nsteps: int = 0
    prev_score: float = -np.inf
    prev_nfeatures: int = 0
    n_noimproving_iters: int = 0
    best_nfeatures: int = 0
    best_iter: int = 0
    # -inf is the only safe initial value for an argmax-style accumulator: any finite floor (e.g. -1e6) is breached routinely by negative-MSE scorers on noisy regression, leaving ``final_score > best_score`` False forever and tripping max_noimproving_iters prematurely.
    best_score: float = -np.inf
    nofeatures_score: float = float("nan")
    ran_out_of_time: bool = False

    evaluated_scores_mean: dict = field(default_factory=dict)
    evaluated_scores_std: dict = field(default_factory=dict)
    feature_importances: dict = field(default_factory=dict)
    fitted_estimators: dict = field(default_factory=dict)
    selected_features_per_nfeatures: dict = field(default_factory=dict)
    dummy_scores: list = field(default_factory=list)
    # 2026-05-28 sklearn-parity: per-fold scores keyed by N, for cv_results_["splitK_test_score"] schema.
    per_fold_scores: dict = field(default_factory=dict)  # dict[N -> list[float] of length n_splits]

    Optimizer: Any = None

    ram_baseline_mb: float = 0.0
    ram_df_size_mb: float = 0.0


def run_outer_loop_iteration(
    self,
    state: OuterLoopState,
    *,
    X,
    y,
    groups,
    cv,
    val_cv,
    original_features,
    must_include_resolved,
    estimators_list,
    estimator,
    estimator_type,
    scoring,
    importance_getter,
    cat_features,
    fit_params,
    keep_estimators,
    early_stopping_rounds,
    frac,
    n_jobs_effective,
    _is_multithreaded,
    use_all_fi_runs,
    use_last_fi_run_only,
    use_one_freshest_fi_run,
    use_fi_ranking,
    top_predictors_search_method,
    votes_aggregation_method,
    max_runtime_mins,
    max_refits,
    best_desired_score,
    max_noimproving_iters,
    verbose,
    ndigits,
    progressbar_prefix,
    iters_pbar,
    start_time,
    signature,
    maybe_clean_ram_and_gpu,
    pin_threads_to_one,
):
    """Run ONE iteration of the outer while-loop and return the outcome.

    Mutates ``state`` in place. Returns ``IterationOutcome.BREAK`` when any of the six original break conditions fires (no more features, stop_file, dummy-beats-first-explored, max_runtime_mins, max_refits, best_desired_score, max_noimproving_iters, special_feature_indices already covered); returns ``IterationOutcome.CONTINUE`` otherwise. Raises RuntimeError on 5-consecutive-NaN (verbatim pre-carve behaviour).
    """

    # RAM-aware GC: ``maybe_clean_ram_and_gpu`` checks RSS-vs-baseline and free-vs-frame BEFORE invoking gc.collect, so the ~290ms cost is paid only when something actually accumulated. The old unconditional ``clean_ram()`` every-5-iters trigger fired even on small problems where no garbage existed, dominating wall.
    state.ram_baseline_mb = maybe_clean_ram_and_gpu(
        state.ram_baseline_mb, state.ram_df_size_mb, verbose=False, reason=f"RFECV iter {state.nsteps}",
    )

    if self.special_feature_indices is not None and len(self.special_feature_indices) > 0:
        current_features = self.special_feature_indices
    else:
        # F6 (Wave 3): coerce votes_aggregation to Borda when multi-estimator + AM/GM and not allow_unsafe.
        _vam = votes_aggregation_method
        if estimators_list and len(estimators_list) > 1 and not getattr(self, "allow_unsafe_aggregation", False):
            if _vam in (VotesAggregation.AM, VotesAggregation.GM):
                _vam = VotesAggregation.Borda
        current_features = get_next_features_subset(
            nsteps=state.nsteps,
            original_features=original_features,
            feature_importances=state.feature_importances,
            evaluated_scores_mean=state.evaluated_scores_mean,
            evaluated_scores_std=state.evaluated_scores_std,
            use_all_fi_runs=use_all_fi_runs,
            use_last_fi_run_only=use_last_fi_run_only,
            use_one_freshest_fi_run=use_one_freshest_fi_run,
            use_fi_ranking=use_fi_ranking,
            top_predictors_search_method=top_predictors_search_method,
            votes_aggregation_method=_vam,
            Optimizer=state.Optimizer,
            fi_missing_policy=getattr(self, "fi_missing_policy", "worst"),
            dichotomic_epsilon=float(getattr(self, "dichotomic_epsilon", 0.0)),
            rng=getattr(self, "_rng", None),
            fi_decay_rate=float(getattr(self, "fi_decay_rate", 0.0)),
            fi_run_order=list(state.feature_importances.keys()),
        )

    if current_features is None or len(current_features) == 0:
        return IterationOutcome.BREAK
    if self.stop_file and exists(self.stop_file):
        logger.warning(f"Stop file {self.stop_file} detected, quitting.")
        return IterationOutcome.BREAK

    desc = f"{progressbar_prefix} trying {len(current_features):_}F"
    if state.nsteps > 0:
        desc += f", had {state.prev_nfeatures:_}F with score {state.prev_score:.{ndigits}f}, best was {state.best_nfeatures:_}F with score {state.best_score:.{ndigits}f} @iter={state.best_iter:_} "
    iters_pbar.set_description(desc, refresh=True)

    # Defer the selected_features_per_nfeatures write until after we know whether this exploration's score beats the existing best at the same N. An unconditional write silently downgrades a winning subset whenever MBH revisits the same N with a worse one.

    # Each split must be different: even with a fixed random_state, the per-iter CV random_state is derived deterministically.
    scores: list = []

    splitter = cv.split(X=X, y=y, groups=groups)

    # Pre-materialise fold args so we can dispatch sequentially or in parallel from the same code path. Each fold gets its own pre-derived RNG seed rather than sharing self._rng, which would race in a parallel context.
    _fold_args: list = []
    for _nfold, (_tr_idx, _te_idx) in enumerate(splitter):
        _fold_seed = int(self._rng.integers(0, 2**31 - 1))
        _fold_args.append((_nfold, _tr_idx, _te_idx, _fold_seed))

    # F9 (Wave 1, 2026-05-28): snapshot FI keys BEFORE the fold loop so we
    # can roll the just-added runs back if store_averaged_cv_scores rejects
    # the new subset at this N as worse than the previously-stored one.
    # Without rollback, the loser-subset's FI runs remain in
    # ``state.feature_importances`` forever and contaminate voting -- a
    # contemporary RFECV vote sums a mix of winning-subset and
    # losing-subset importances at the same N, biasing the next subset pick.
    # Opt-out via self.keep_loser_subset_fi (default False = new behaviour).
    _fi_keys_before = set(state.feature_importances.keys())

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
            evaluated_scores_mean=state.evaluated_scores_mean,
            scores=_scores,
            feature_importances=state.feature_importances,
            fitted_estimators=state.fitted_estimators,
            dummy_scores=state.dummy_scores,
        )

    # Dispatch _eval_fold sequentially or in parallel. n_jobs>1 uses prefer="threads" so we don't pickle X/y across workers (datasets stay in shared memory) and the closure can keep mutating outer state under the GIL. When n_jobs>1 AND the estimator is multi-threaded AND force_parallel=True, pin inner threads to 1.
    if n_jobs_effective > 1 and _is_multithreaded and self.force_parallel:
        _orig_eval_fold = _eval_fold

        def _eval_fold_pinned(*args, _orig=_orig_eval_fold):
            # The closure clones the estimator inside its body so we can't reach in. Pin once on the OUTER estimator; clone() preserves params so each fold's clone inherits thread_count=1 / n_jobs=1.
            pin_threads_to_one(estimator)
            return _orig(*args)
        _fold_runner = _eval_fold_pinned
    else:
        _fold_runner = _eval_fold

    if n_jobs_effective > 1 and len(_fold_args) > 1:
        from joblib import Parallel, delayed
        # prefer="threads": sklearn / CB / LGB / XGB all release GIL during fit, so threads give true parallelism without the serialisation cost of multiprocessing. require="sharedmem" HARDENS the design -- under loky the closure-state mutations would happen in worker process copies and the main-process state silently stays empty -> ``final_score = nan`` with no exception; require= makes joblib RAISE if it can't satisfy threading, surfacing the misconfiguration loud.
        Parallel(n_jobs=n_jobs_effective, prefer="threads", require="sharedmem")(
            delayed(_fold_runner)(*a) for a in _fold_args
        )
    else:
        if verbose:
            from pyutilz.system import tqdmu
            _iter = tqdmu(_fold_args, desc="CV folds", leave=False, total=len(_fold_args))
        else:
            _iter = _fold_args
        for a in _iter:
            _fold_runner(*a)

    if 0 not in state.evaluated_scores_mean:
        scores_mean, scores_std, final_score, _ = store_averaged_cv_scores(
            pos=0, scores=state.dummy_scores, evaluated_scores_mean=state.evaluated_scores_mean, evaluated_scores_std=state.evaluated_scores_std, self=self,
        )
        state.nofeatures_score = final_score
        if verbose:
            logger.info(
                "Baseline with 0 features, score=%s +/- %s ~ %s",
                f"{scores_mean:.{ndigits}f}", f"{scores_std:.{ndigits}f}", f"{final_score:.{ndigits}f}",
            )
        # C2 (Wave 1, 2026-05-28): NEW default does NOT submit the N=0 dummy
        # to the MBH surrogate. On imbalanced accuracy/F1 the prior-strategy
        # DummyClassifier scores close to the model and the surrogate's
        # score-vs-N curve gets anchored at the steep (0, dummy) point,
        # steering the optimizer toward small N. The dummy stays in
        # cv_results_ for reporting but does NOT influence acquisition.
        # Opt-in old behaviour via self.submit_dummy_to_optimizer=True.
        if top_predictors_search_method == OptimumSearch.ModelBasedHeuristic \
                and getattr(self, "submit_dummy_to_optimizer", False):
            state.Optimizer.submit_evaluations(candidates=[0], evaluations=[final_score], durations=[None])

    scores_mean, scores_std, final_score, was_stored = store_averaged_cv_scores(
        pos=len(current_features),
        scores=scores,
        evaluated_scores_mean=state.evaluated_scores_mean,
        evaluated_scores_std=state.evaluated_scores_std,
        self=self,
    )
    # Only commit selected_features when this run actually won at its N.
    if was_stored:
        state.selected_features_per_nfeatures[len(current_features)] = current_features
        # Per-fold scores for cv_results_["splitK_test_score"] schema.
        state.per_fold_scores[len(current_features)] = list(scores)
    else:
        # F9 rollback: this iter's FI got added to state.feature_importances inside
        # _eval_fold_body; if the subset lost the gate it must not poison voting.
        if not getattr(self, "keep_loser_subset_fi", False):
            _new_fi_keys = set(state.feature_importances.keys()) - _fi_keys_before
            for _k in _new_fi_keys:
                state.feature_importances.pop(_k, None)

    if top_predictors_search_method == OptimumSearch.ModelBasedHeuristic:
        # S8 (Wave 2, 2026-05-28): align optimizer target with the 1-SE
        # rule semantics in select_optimal_nfeatures_, which threshold on
        # RAW cv_mean_perf. The optimizer used to maximise
        # final_score = mean*w_mean - std*w_std - feature_cost*N
        # (a UCB-of-noise scalar). Post-processing then picked 1-SE on
        # the raw mean -> two criteria that can disagree, especially when
        # the user keeps default std_perf_weight=0.1 and feature_cost=0.0
        # but switches the rule.
        # Default: optimizer maximises ``scores_mean`` (consistent with
        # both 'argmax' (after multiplying by mean_perf_weight=1 default)
        # and 'one_se_*' (raw mean threshold)). Opt-out via
        # ``optimizer_target='final_score'`` (legacy).
        _target_name = getattr(self, "optimizer_target", "mean")
        if _target_name == "mean":
            _target_value = scores_mean
        elif _target_name == "final_score":
            _target_value = final_score
        else:
            raise ValueError(
                f"optimizer_target must be 'mean' or 'final_score'; got {_target_name!r}"
            )
        state.Optimizer.submit_evaluations(candidates=[len(current_features)], evaluations=[_target_value], durations=[None])

        if verbose:
            logger.info(
                "Tried %s features (%s), score=%s +/- %s ~ %s",
                f"{len(current_features):_}",
                textwrap.shorten(', '.join(map(str, current_features[:40])), 150),
                f"{scores_mean:.{ndigits}f}", f"{scores_std:.{ndigits}f}", f"{final_score:.{ndigits}f}",
            )

    state.prev_nfeatures, state.prev_score = len(current_features), final_score
    iters_pbar.update(1)

    state.nsteps += 1

    # Persist outer-loop state so a crash mid-run is recoverable. fitted_estimators is intentionally NOT pickled (CB / RF ensembles dominate file size); they are re-fit on resume when needed. Save errors are logged but do not abort the fit.
    if self.checkpoint_path is not None:
        try:
            self._save_checkpoint({
                "version": self._CHECKPOINT_VERSION,
                "signature": signature,
                "nsteps": state.nsteps,
                "evaluated_scores_mean": dict(state.evaluated_scores_mean),
                "evaluated_scores_std": dict(state.evaluated_scores_std),
                "feature_importances": dict(state.feature_importances),
                "selected_features_per_nfeatures": dict(state.selected_features_per_nfeatures),
                "prev_score": state.prev_score,
                "prev_nfeatures": state.prev_nfeatures,
                "n_noimproving_iters": state.n_noimproving_iters,
                "best_nfeatures": state.best_nfeatures,
                "best_iter": state.best_iter,
                "best_score": state.best_score,
                "dummy_scores": list(state.dummy_scores),
                "optimizer": state.Optimizer,
            })
        except Exception as _ckpt_exc:
            if verbose:
                logger.warning(
                    "RFECV: checkpoint save at nsteps=%d failed: %s",
                    state.nsteps, _ckpt_exc,
                )

    if len(state.evaluated_scores_mean) == 2:
        # If the first explored subset (whatever MBH seeded; default seed = 2 features) is already worse than the dummy at 0 features, there's no point continuing.
        if final_score < state.nofeatures_score:
            logger.info(
                "Stopping RFECV early: dummy baseline at 0 features (%s) already beats the first "
                "explored subset of %d features (%s). The model can't learn anything from this data.",
                f"{state.nofeatures_score:.{ndigits}f}",
                len(current_features),
                f"{final_score:.{ndigits}f}",
            )
            return IterationOutcome.BREAK

    if max_runtime_mins and not state.ran_out_of_time:
        delta = timer() - start_time
        state.ran_out_of_time = delta > (max_runtime_mins * 60)
        if state.ran_out_of_time:
            if verbose:
                logger.info("max_runtime_mins=%s reached.", f"{max_runtime_mins:_.1f}")
            return IterationOutcome.BREAK

    if max_refits and state.nsteps >= max_refits:
        if verbose:
            logger.info("max_refits=%s reached.", f"{max_refits:_}")
        return IterationOutcome.BREAK

    if final_score > state.best_score:
        state.best_score = final_score
        state.best_iter = state.nsteps
        state.best_nfeatures = len(current_features)
        state.n_noimproving_iters = 0
    else:
        # C8 (Wave 4, 2026-05-28): only increment the no-improve counter when
        # the OPTIMIZER actually proposed something it hadn't seen before
        # (was_stored=True or new N). MBH revisits of the same N with a worse
        # subset (was_stored=False) used to spike the counter and trip
        # max_noimproving_iters prematurely. Opt-out via
        # noimprove_counts_revisit=True.
        if was_stored or getattr(self, "noimprove_counts_revisit", False):
            state.n_noimproving_iters += 1

    if best_desired_score is not None and final_score > best_desired_score:
        if verbose:
            logger.info("best_desired_score %s reached.", f"{best_desired_score:_.{ndigits}f}")
        return IterationOutcome.BREAK

    if max_noimproving_iters and state.n_noimproving_iters >= max_noimproving_iters:
        if verbose:
            logger.info("Max # of noimproved iters reached: %s", state.n_noimproving_iters)
        return IterationOutcome.BREAK

    # S7 (Wave 2, 2026-05-28): tolerance-based convergence. ``n_noimproving_iters``
    # resets on ANY new-best even when the improvement is CV noise (e.g. 1e-6
    # bump). On noisy scoring the counter rarely hits max_noimproving_iters
    # and budget is spent grinding the plateau. tol+tol_window break-condition:
    # "stop when max(last K scores) - min(last K) < tol * |best_score|".
    # Default tol=None disables the check (legacy behaviour).
    _tol = getattr(self, "convergence_tol", None)
    if _tol is not None and _tol > 0 and not np.isnan(final_score):
        _tol_window = max(2, int(getattr(self, "convergence_tol_window", 10)))
        if not hasattr(state, "_recent_finals"):
            state._recent_finals = []
        state._recent_finals.append(final_score)
        if len(state._recent_finals) > _tol_window:
            state._recent_finals = state._recent_finals[-_tol_window:]
        if len(state._recent_finals) >= _tol_window:
            _spread = max(state._recent_finals) - min(state._recent_finals)
            _scale = max(abs(state.best_score), 1e-12) if state.best_score != -np.inf else 1.0
            if _spread < float(_tol) * _scale:
                if verbose:
                    logger.info(
                        "convergence_tol reached: spread %.6f < tol*|best| (%.6g * %.6f); stopping.",
                        _spread, _tol, _scale,
                    )
                return IterationOutcome.BREAK

    # Abort early if every iter so far produced a NaN final_score. The most common cause is a custom scorer returning NaN on every fold (e.g. ROC AUC on single-class CV folds). Without this, the noimproving counter would consume max_noimproving_iters worth of useless CV fits. Detect 5 consecutive NaN iters and bail.
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
        return IterationOutcome.BREAK

    return IterationOutcome.CONTINUE
