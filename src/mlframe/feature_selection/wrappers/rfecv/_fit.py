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
from timeit import default_timer as timer
from typing import Union

import numpy as np
import pandas as pd

from pyutilz.system import tqdmu

from sklearn.pipeline import Pipeline

from mlframe.utils.misc import set_random_seed

from .._helpers import (
    _pin_threads_to_one,
    suppress_irritating_3rdparty_warnings,
)
from ._cv_setup import _resolve_cv_and_val_cv
from ._mbh_optimizer import _build_mbh_optimizer
from ._finalize import _finalize_fit_results
from ._checkpoint import _maybe_resume_from_checkpoint
from ._must_include import _resolve_must_include
from ._fit_init import _init_fit_state
from ._fit_setup import (
    filter_cat_features_by_dtype,
    resolve_default_scoring,
    resolve_effective_n_jobs,
)
from ._fit_outer_loop import (
    IterationOutcome,
    OuterLoopState,
    run_outer_loop_iteration,
)

logger = logging.getLogger("mlframe.feature_selection.wrappers.rfecv")


def _apply_prescreen(self, *, X, y, candidate_features, verbose):
    """L7 (Wave 5, 2026-05-28): run the configured prescreen and return the filtered candidate-feature list.

    Supported prescreen values:
      - 'univariate_ht' : in-tree Mann-Whitney / Kruskal-Wallis / Kendall / chi-squared + BY-FDR (numba-compiled).
      - callable(X, y) -> list : user-supplied prescreen returning the kept feature names/indices.
    Returns the kept feature names. Falls back to ``candidate_features`` (no-op) on error.
    """
    _p = self.prescreen
    _topk = getattr(self, "prescreen_top_k", None)
    if callable(_p):
        try:
            _kept = list(_p(X[candidate_features] if isinstance(X, pd.DataFrame) else X, y))
        except Exception as exc:
            if verbose:
                logger.warning("Prescreen callable failed (%s); skipping prescreen.", exc)
            return candidate_features
        if _topk is not None and len(_kept) > _topk:
            _kept = _kept[:int(_topk)]
        return _kept
    # mRMR pre-screening (TODO from pre-Wave-1 high-priority, 2026-05-28):
    # use the existing ``mlframe.feature_selection.filters.mrmr.MRMR`` to
    # pick the top-K features by min-redundancy / max-relevance MI ranking
    # before the RFECV outer loop. Best when p >> n (e.g. p >= 5000).
    # Cost: O(p^2) MRMR vs O(p * iter) backward elimination -- net win
    # when p >= 5000.
    if isinstance(_p, str) and _p.lower() == "mrmr":
        try:
            from mlframe.feature_selection.filters.mrmr import MRMR
        except ImportError as exc:
            if verbose:
                logger.warning("prescreen='mrmr' could not import MRMR (%s); skipping.", exc)
            return candidate_features
        try:
            _Xsub = X[candidate_features] if isinstance(X, pd.DataFrame) else X
            _topk_arg = int(_topk) if _topk else min(500, len(candidate_features) // 2 or 1)
            _mrmr = MRMR(
                full_npermutations=3, cv=3,
                run_additional_rfecv_minutes=False,
                random_seed=getattr(self, "random_state", 0) or 0,
                min_features_fallback=max(1, _topk_arg // 4),
            )
            _mrmr.fit(_Xsub, y)
            _kept = list(_mrmr.get_feature_names_out())
            if _topk_arg and len(_kept) > _topk_arg:
                _kept = _kept[:_topk_arg]
        except Exception as exc:
            if verbose:
                logger.warning("mRMR prescreen failed (%s); skipping.", exc)
            return candidate_features
        _kept_set = set(_kept)
        return [c for c in candidate_features if c in _kept_set]

    # L7 native impl (Wave 5, 2026-05-28): in-tree univariate hypothesis-test
    # relevance scoring with BY-FDR correction. Backend selection per
    # (feature, target) dtype:
    #   binary target  + numeric  -> Mann-Whitney U
    #   multiclass     + numeric  -> Kruskal-Wallis H
    #   continuous     + numeric  -> Kendall tau-b
    #   any            + categorical -> Chi-squared independence
    # Numba-compiled rank / U / H / tau kernels (cache=True). No external
    # dependencies.
    if isinstance(_p, str) and _p.lower() == "univariate_ht":
        if not isinstance(X, pd.DataFrame):
            if verbose:
                logger.warning("Prescreen='univariate_ht' needs pandas DataFrame; skipping.")
            return candidate_features
        from .._univariate_ht import calculate_relevance_table as _crt
        try:
            _Xsub = X[candidate_features]
            _rel = _crt(_Xsub, np.asarray(y),
                        fdr_level=float(getattr(self, "prescreen_fdr_level", 0.05)),
                        ml_task="auto", n_jobs=1)
            _rel = _rel[_rel["relevant"]].sort_values("p_value", ascending=True)
            _kept = list(_rel["feature"])
        except Exception as exc:
            if verbose:
                logger.warning("univariate_ht prescreen failed (%s); skipping.", exc)
            return candidate_features
        if _topk is not None and len(_kept) > _topk:
            _kept = _kept[:int(_topk)]
        _kept_set = set(_kept)
        return [c for c in candidate_features if c in _kept_set]
    if verbose:
        logger.warning("Unknown prescreen=%r; skipping.", _p)
    return candidate_features


def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.DataFrame, pd.Series, np.ndarray], groups: Union[pd.Series, np.ndarray] = None, sample_weight: Union[np.ndarray, pd.Series, None] = None, **fit_params):
    # scipy.sparse X is not first-class across the dense-frame-centric FS pipeline; densify at the boundary so the
    # existing ndarray path handles it. Gated on dense size per the project RAM rule -- a sparse matrix whose dense
    # form would exceed ~2 GB is refused with a clear error rather than silently doubling host memory.
    try:
        from scipy.sparse import issparse as _issparse
    except Exception:
        _issparse = None
    if _issparse is not None and _issparse(X):
        _dense_bytes = int(X.shape[0]) * int(X.shape[1]) * 8
        if _dense_bytes > 2 * 1024 ** 3:
            raise NotImplementedError(
                f"RFECV does not accept scipy.sparse X whose dense form would be {_dense_bytes / 1024 ** 3:.1f} GB "
                f"(> 2 GB); densify a representative subset or pass a DataFrame/ndarray that fits in RAM."
            )
        X = np.asarray(X.toarray())

    # Multi-output (2D y) opt-in: fit one single-target RFECV per output column and aggregate support_ (union/intersect).
    # Default ``multioutput_strategy=None`` falls through to the historical clear NotImplementedError in _fit_init.
    _mo_strategy = getattr(self, "multioutput_strategy", None)
    if _mo_strategy is not None:
        try:
            _y_arr = np.asarray(y)
            _is_2d = _y_arr.ndim >= 2 and _y_arr.shape[-1] > 1
        except Exception:
            _is_2d = False
        if _is_2d:
            from ._multioutput import fit_multioutput
            _fp = copy.copy(self.fit_params) if self.fit_params else {}
            return fit_multioutput(self, X, y, groups, sample_weight, _fp, _mo_strategy)

    # TODO A (Wave 6 prelim, 2026-05-28): auto-tune. Compute a DataFingerprint
    # then push the rule-based suggestion into self.<flat-knob> for every
    # flat kwarg the caller didn't explicitly override. Stored decision lives
    # in self.auto_tune_decision_ for inspection.
    if getattr(self, "auto_tune", False) and not getattr(self, "_auto_tune_applied_", False):
        try:
            from .._auto_tune import DataFingerprint, suggest_configs, explain_suggestion
            _fp = DataFingerprint.from_xy(X, y)
            _sc, _fic, _rc = suggest_configs(_fp)
            _decision: dict = {"fingerprint": _fp, "explanation": explain_suggestion(_fp)}
            # For each suggested field, apply IFF the current attribute is at its constructor default.
            # We approximate "default" by comparing to the SearchConfig / FIConfig / RobustnessConfig
            # baseline defaults (no kwargs passed).
            from ._configs import SearchConfig as _SC, FIConfig as _FIC, RobustnessConfig as _RC
            _baselines = (_SC(), _FIC(), _RC())
            _applied: dict = {}
            for _cfg, _baseline in zip((_sc, _fic, _rc), _baselines):
                _set_fields = getattr(_cfg, "model_fields_set", None)
                if not _set_fields:
                    continue
                for _k in _set_fields:
                    _user_val = getattr(self, _k, None)
                    _baseline_val = getattr(_baseline, _k, None)
                    if _user_val == _baseline_val:
                        # User didn't override this knob; apply the auto-tune suggestion.
                        setattr(self, _k, getattr(_cfg, _k))
                        _applied[_k] = getattr(_cfg, _k)
            _decision["applied"] = _applied
            self.auto_tune_decision_ = _decision
            self._auto_tune_applied_ = True
            if getattr(self, "verbose", 0):
                logger.info("RFECV auto_tune: %s", _decision["explanation"])
        except Exception as _exc:
            logger.warning("auto_tune skipped (%s): %s", type(_exc).__name__, _exc)

    X, y, signature, _polars_time_series_hint, _init_skip = _init_fit_state(
        self, X, y, groups, sample_weight,
    )
    if _init_skip:
        return self

    # perf (2026-06-05): build ONE contiguous numpy mirror of the (now sanitised) DataFrame and feed the
    # inner estimator numpy column-SUBSETS by integer position throughout elimination / CV scoring /
    # permutation-FI re-prediction. This skips LightGBM's per-fit/per-predict ``_data_from_pandas``
    # reconversion + per-column dtype-validation storm (cProfile: ~47% of the scene 700x299 fit). Names
    # keep flowing through the voting / FI / finalize machinery unchanged, so support_ / ranking_ /
    # get_feature_names_out are bit-identical. Gated on an ALL-NUMERIC frame: object / category / string
    # columns (CatBoost cats, etc.) fall back to the historical pandas path so estimators that need the
    # original dtypes / names are untouched. float64 (NOT float32) preserves LightGBM's split points exactly.
    X_estimator = None
    col_pos = None
    # Escape hatch (default OFF): force the historical pandas path. Used by the selection-identity
    # regression test to A/B numpy-vs-pandas on the SAME seeded fixture, and available as a safety
    # opt-out should any estimator ever misbehave on numpy input.
    _force_pandas = bool(getattr(self, "_force_pandas_estimator_path", False))
    if (not _force_pandas) and isinstance(X, pd.DataFrame):
        try:
            from pandas.api.types import is_numeric_dtype as _is_num, is_bool_dtype as _is_bool
            # All columns must be numeric/bool AND have a finite-supporting dtype. NaN is fine (LightGBM /
            # tree handles it; float64 carries NaN bit-identically); object / category / string disqualify.
            _all_numeric = bool(len(X.columns)) and all(_is_num(X[c]) or _is_bool(X[c]) for c in X.columns)
        except Exception:
            _all_numeric = False
        if _all_numeric:
            try:
                # ``to_numpy(dtype=float64)`` materialises pyarrow-backed nullable numerics into a plain C
                # float64 array too (pd.NA -> nan); the try/except + shape guard below rejects any dtype
                # that fails to cast cleanly, so those callers transparently keep the pandas path.
                _np = np.ascontiguousarray(X.to_numpy(dtype=np.float64))
                if _np.shape == (int(X.shape[0]), int(X.shape[1])) and _np.dtype == np.float64:
                    X_estimator = _np
                    col_pos = {name: i for i, name in enumerate(X.columns)}
            except (TypeError, ValueError):
                X_estimator = None
                col_pos = None

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

    # L7 (Wave 5, 2026-05-28): optional prescreen pass. Restricts the MBH search universe to a smaller pre-filtered candidate set so the
    # outer loop converges faster on high-p data. The prescreen sees the (X[original_features], y) AFTER must_include filtering, so
    # pinned features always remain in the final support_ regardless of prescreen output.
    _prescreen = getattr(self, "prescreen", None)
    if _prescreen is not None and len(original_features) > 0:
        _kept = _apply_prescreen(
            self, X=X, y=y, candidate_features=original_features, verbose=verbose,
        )
        if len(_kept) > 0 and list(_kept) != list(original_features):
            if verbose:
                logger.info(
                    "RFECV prescreen=%s: %d -> %d features (keeping %d after prescreen).",
                    _prescreen, len(original_features), len(_kept), len(_kept),
                )
            original_features = list(_kept)

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

    # Wide-data perm-FI cost guard (2026-06-04). Permutation / conditional-permutation importance rescore the model
    # O(p * n_repeats) times PER FOLD. On wide frames one RFECV iteration can exceed the whole runtime budget (measured:
    # madelon p=500, n_repeats=5 -> ~208s/iter > a 180s budget), so only 2-3 iters complete, the CV curve has ~3 points,
    # and the N-rule (e.g. one_se_min) lands at the over-selection. When ``wide_data_fi_fallback`` (default True) and the
    # search universe exceeds ``wide_data_fi_threshold``, fall back to the estimator's native (gain/impurity) importance
    # for the elimination ranking so the outer loop can build a REAL multi-point curve in budget; below the threshold cap
    # n_repeats at ``wide_data_fi_n_repeats`` to soften the cliff. The fallback only changes the ELIMINATION RANKING that
    # picks the next candidate subset -- the CV SCORE that drives the optimum is unaffected -- so on wide noisy frames it
    # trades perm-FI's debiasing (a small-p win) for a usable curve. Opt out with wide_data_fi_fallback=False (and a
    # generous max_runtime_mins) to keep exact permutation FI regardless of p.
    _perm_getters = ("permutation", "conditional_permutation")
    _n_candidates = len(original_features)
    _eff_n_repeats = int(getattr(self, "n_repeats", 5))
    self._wide_data_fi_applied_ = None
    if (
        getattr(self, "wide_data_fi_fallback", True)
        and isinstance(importance_getter, str)
        and importance_getter in _perm_getters
    ):
        _threshold = int(getattr(self, "wide_data_fi_threshold", 200))
        if _n_candidates > _threshold:
            self._wide_data_fi_applied_ = {
                "reason": "fallback_to_native",
                "n_candidates": _n_candidates,
                "threshold": _threshold,
                "from_importance_getter": importance_getter,
                "to_importance_getter": "auto",
            }
            if verbose:
                logger.info(
                    "RFECV wide-data guard: %d candidate features > wide_data_fi_threshold=%d; "
                    "falling back from importance_getter=%r to native 'auto' for the elimination ranking "
                    "(permutation FI is ~O(p*n_repeats) rescores/fold and would blow the runtime budget). "
                    "Set wide_data_fi_fallback=False to keep exact permutation FI.",
                    _n_candidates, _threshold, importance_getter,
                )
            importance_getter = "auto"
        else:
            _cap = int(getattr(self, "wide_data_fi_n_repeats", 2))
            if _eff_n_repeats > _cap and _n_candidates > max(1, _threshold // 4):
                self._wide_data_fi_applied_ = {
                    "reason": "capped_n_repeats",
                    "n_candidates": _n_candidates,
                    "threshold": _threshold,
                    "from_n_repeats": _eff_n_repeats,
                    "to_n_repeats": _cap,
                }
                if verbose:
                    logger.info(
                        "RFECV wide-data guard: %d candidate features; capping permutation n_repeats %d -> %d "
                        "to keep per-iteration cost in budget.",
                        _n_candidates, _eff_n_repeats, _cap,
                    )
                _eff_n_repeats = _cap
    # Effective n_repeats consumed by the per-fold permutation FI + stability bootstraps; read via the private attr so
    # the user-facing self.n_repeats is never mutated by the guard.
    self._effective_n_repeats = _eff_n_repeats

    state = OuterLoopState(
        evaluated_scores_mean=evaluated_scores_mean,
        evaluated_scores_std=evaluated_scores_std,
        feature_importances=feature_importances,
    )
    state.Optimizer = _build_mbh_optimizer(
        self,
        original_features=original_features,
        max_refits=max_refits,
        top_predictors_search_method=top_predictors_search_method,
    )

    # Resume-from-checkpoint: restore mutable outer-loop state iff the checkpoint signature matches the current (X.shape, y.shape, columns_key). Mismatch silently starts fresh, so users can keep the same checkpoint_path across experiments.
    _ckpt_state = _maybe_resume_from_checkpoint(self, signature=signature, verbose=verbose, state={
        "nsteps": state.nsteps,
        "evaluated_scores_mean": state.evaluated_scores_mean,
        "evaluated_scores_std": state.evaluated_scores_std,
        "feature_importances": state.feature_importances,
        "selected_features_per_nfeatures": state.selected_features_per_nfeatures,
        "prev_score": state.prev_score,
        "prev_nfeatures": state.prev_nfeatures,
        "n_noimproving_iters": state.n_noimproving_iters,
        "best_nfeatures": state.best_nfeatures,
        "best_iter": state.best_iter,
        "best_score": state.best_score,
        "dummy_scores": state.dummy_scores,
        "Optimizer": state.Optimizer,
    })
    state.nsteps = _ckpt_state["nsteps"]
    state.evaluated_scores_mean = _ckpt_state["evaluated_scores_mean"]
    state.evaluated_scores_std = _ckpt_state["evaluated_scores_std"]
    state.feature_importances = _ckpt_state["feature_importances"]
    state.selected_features_per_nfeatures = _ckpt_state["selected_features_per_nfeatures"]
    state.prev_score = _ckpt_state["prev_score"]
    state.prev_nfeatures = _ckpt_state["prev_nfeatures"]
    state.n_noimproving_iters = _ckpt_state["n_noimproving_iters"]
    state.best_nfeatures = _ckpt_state["best_nfeatures"]
    state.best_iter = _ckpt_state["best_iter"]
    state.best_score = _ckpt_state["best_score"]
    state.dummy_scores = _ckpt_state["dummy_scores"]
    state.Optimizer = _ckpt_state["Optimizer"]

    # Baseline RSS + best-effort frame footprint so the RAM-aware ``maybe_clean_ram_and_gpu`` short-circuit can decide whether a ``gc.collect()`` is justified each iter. The old "every 5th iter" trigger ran a ~290ms gc.collect even when nothing had accumulated, dominating wall on small problems; the helper only fires when RSS actually grew past a threshold or free RAM gets tight relative to frame size.
    from mlframe.training import estimate_df_size_mb as _estimate_df_size_mb, get_process_rss_mb as _get_process_rss_mb, maybe_clean_ram_and_gpu as _maybe_clean_ram_and_gpu
    state.ram_baseline_mb = _get_process_rss_mb()
    try:
        state.ram_df_size_mb = _estimate_df_size_mb(X)
    except Exception:
        state.ram_df_size_mb = 0.0

    while state.nsteps < len(original_features):
        outcome = run_outer_loop_iteration(
            self,
            state,
            X=X, y=y, X_estimator=X_estimator, col_pos=col_pos, groups=groups,
            cv=cv, val_cv=val_cv,
            original_features=original_features,
            must_include_resolved=must_include_resolved,
            estimators_list=estimators_list,
            estimator=estimator,
            estimator_type=estimator_type,
            scoring=scoring,
            importance_getter=importance_getter,
            cat_features=cat_features,
            fit_params=fit_params,
            keep_estimators=keep_estimators,
            early_stopping_rounds=early_stopping_rounds,
            frac=frac,
            n_jobs_effective=n_jobs_effective,
            _is_multithreaded=_is_multithreaded,
            use_all_fi_runs=use_all_fi_runs,
            use_last_fi_run_only=use_last_fi_run_only,
            use_one_freshest_fi_run=use_one_freshest_fi_run,
            use_fi_ranking=use_fi_ranking,
            top_predictors_search_method=top_predictors_search_method,
            votes_aggregation_method=votes_aggregation_method,
            max_runtime_mins=max_runtime_mins,
            max_refits=max_refits,
            best_desired_score=best_desired_score,
            max_noimproving_iters=max_noimproving_iters,
            verbose=verbose,
            ndigits=ndigits,
            progressbar_prefix=progressbar_prefix,
            iters_pbar=iters_pbar,
            start_time=start_time,
            signature=signature,
            maybe_clean_ram_and_gpu=_maybe_clean_ram_and_gpu,
            pin_threads_to_one=_pin_threads_to_one,
        )
        if outcome is IterationOutcome.BREAK:
            break

    # Stash per-fold scores so finalize can build cv_results_["splitK_test_score"] (sklearn parity).
    self._per_fold_scores = dict(state.per_fold_scores)

    # Truncated SFFS final-pass swap: run K paired swaps on the best subset found - replace each of the K worst-FI kept features with each of the K best-FI dropped features, accept any swap that improves the CV score. Uses sklearn.cross_val_score directly so it does NOT honour fit_params / val_cv / early stopping.
    _finalize_fit_results(
        self,
        X=X, y=y, X_estimator=X_estimator, col_pos=col_pos, estimator=estimator, cv=cv, scoring=scoring,
        best_nfeatures=state.best_nfeatures, best_score=state.best_score,
        selected_features_per_nfeatures=state.selected_features_per_nfeatures,
        feature_importances=state.feature_importances,
        original_features=original_features,
        evaluated_scores_mean=state.evaluated_scores_mean,
        evaluated_scores_std=state.evaluated_scores_std,
        verbose=verbose, ndigits=ndigits,
        fitted_estimators=state.fitted_estimators,
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
