"""Stage helpers carved out of ``MRMR._fit_body`` so that method stays under the function-length gate.

Each stage is a verbatim block of the original ``_fit_body``; they pass the long-lived locals through the ``st`` namespace.
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
import warnings
from functools import partial
from typing import Any, cast

from .._dynamic_cluster_discovery import set_dcd_active as _set_dcd_active
from .._fe_deadline import clear_fe_deadline as _clear_fe_deadline
from .._fe_family_timing import log_fe_family_summary as _log_fe_wall
from .._fe_gpu_strict import clear_auto_fit_n as _clear_auto_fit_n
from .._fe_gpu_strict import set_auto_fit_n as _set_auto_fit_n
from .._fe_rejection_ledger import populate_fe_rejection_ledger as _pop_rej
from .._mrmr_fe_provenance import populate_fe_provenance as _pop_prov
from .._mrmr_fingerprints import _MRMR_IDENTITY_FP_CACHE
from .._mrmr_fingerprints import _MRMR_IDENTITY_FP_LOCK
from .._mrmr_fingerprints import _mrmr_identity_cache_key
from .._param_accuracy_warnings import warn_accuracy_suboptimal_params
from ..screen import _preserve_global_numpy_rng_state
from ._fe_budget_learning import apply_learned_fe_budgets
from ._fe_budget_learning import update_fe_budgets_after_fit
from ._mrmr_y_multioutput import _mrmr_y_is_multioutput
from mlframe.feature_selection.filters.info_theory.shared import set_group_mi as _set_group_mi
from ..info_theory import (
    get_bur_lambda,
    get_cmi_perm_stop,
    get_cpt_test,
    get_group_mi,
    get_pid_synergy_bonus,
    get_relaxmrmr_alpha,
    use_jmim_aggregator,
    use_mi_chao_shen,
    use_mi_miller_madow,
    use_su_normalization,
)

logger = logging.getLogger("mlframe.feature_selection.filters.mrmr")


class _MRMRFitStagesMixin:
    """Mixin carrying the ``_fit_body`` stage methods; see the module docstring."""

    def _fit_body_check_row_counts(self, X: Any, y: Any, st: Any) -> None:
        """Fail fast with a ValueError when X and y row counts differ (before any njit kernel can read out of bounds)."""
        # Row-count guard, first thing: no length-validation existed anywhere before the MI/screening pipeline, so a mismatched (X, y) reached numba-njit
        # kernels (bounds checking compiled OUT for speed) with an out-of-bounds row index instead of a Python exception. Off the JIT-disabled fallback path
        # this happened to raise a clean pandas ValueError from an unrelated internal column assignment deep in the pipeline (accidental, not an intentional
        # guard); WITH jit enabled the same out-of-bounds read reached compiled code first and corrupted the process -- "Windows fatal exception: access
        # violation", reproduced live via ``test_selectors_shared.py::TestSharedDegenerateInputs::test_y_length_mismatch_raises[MRMR]``. Validating here, before
        # anything touches a kernel, makes the failure mode a clean ValueError unconditionally (JIT on or off) instead of an accident of which code path happens
        # to run first. A polars LazyFrame has neither .shape nor len() (row count is unknown until materialised) -- duck-typed via .collect (present on
        # LazyFrame, absent on pandas/polars-eager/ndarray) so this guard skips it without a hard polars import (polars stays an optional dependency throughout
        # this module). The auto-collect step downstream turns it into an eager frame, and the row-count check still fires there (this guard's whole point --
        # reaching the mismatch as a clean ValueError before any njit kernel -- is preserved, just deferred to after collection).
        st._is_lazyframe = hasattr(X, "collect") and not hasattr(X, "shape")
        if st._is_lazyframe:
            _n_rows_X = None
        else:
            _n_rows_X = X.shape[0] if hasattr(X, "shape") else len(X)
        _n_rows_y = len(y)
        if _n_rows_X is not None and _n_rows_X != _n_rows_y:
            raise ValueError(f"MRMR.fit: X has {_n_rows_X} rows but y has {_n_rows_y} -- X and y must have the same length.")

    def _fit_body_snapshot_params_and_budgets(self, X: Any, st: Any) -> None:
        """Snapshot the pre-override ctor params and apply the learned FE-family budgets (originals saved in ``st`` for the finally block)."""
        # groups contract check and polars validate+bridge each moved verbatim to a named helper on _MRMRFitHelpersMixin (see their docstrings for the original
        # rationale) - zero behavior change, pure extraction. The GPU-breaker re-arm now happens in the outer ``fit()`` wrapper's ``_enter_active_fit_scope()``,
        # gated to the 0->1 in-flight-fit transition instead of running unconditionally on every call here.
        #
        # Pre-override ctor-params snapshot (bug found while testing the re-entrancy guard): the in-object "identical refit -> skip" signature
        # (``_fit_impl_core.py``'s ``_self_params_sig``) used to be computed from ``self.get_params()``
        # READ INSIDE ``_fit_impl`` - i.e. AFTER this method's OWN below-here overrides (cluster_
        # aggregate_enable, fast-search profile, default-screen-subsample, etc.) had already flipped
        # several ctor-param-named attributes to a TRANSIENT mid-fit value. The signature got stored
        # with that transient value baked in, but the ``finally`` block restores the true ctor value
        # before returning - so a SUBSEQUENT identical fit()'s freshly-read params (post-restore) could
        # NEVER match the stored signature for any config whose override actually fires (e.g. the
        # DEFAULT ``cluster_aggregate_enable=True``), permanently defeating the same-content skip
        # optimization for the common case and forcing a full re-fit (with a different tie-break outcome
        # on near-equal-gain features) on every "identical" refit. Snapshotting the STABLE, pre-override
        # params here and threading it into ``_fit_impl`` (which prefers it over a fresh
        # ``self.get_params()`` read) fixes the signature to always reflect the user-visible ctor state.
        self._pre_fit_ctor_params_snapshot_ = self.get_params(deep=True)
        # gt_07 FE-family budget: load a persisted per-family budget (keyed by dataset fingerprint)
        # and scale the triplet/quadruplet/adaptive-arity seed_k/top_count quotas by it - a lower
        # budget fraction means fewer candidates proposed for that family this fit. Snapshot the
        # ORIGINAL ctor values first (restored in the ``finally`` block below) so a crashing fit
        # never leaves the instance with a scaled-down quota baked in for a later ``get_params()``.
        st._fe_budget_quota_snapshot = {}
        st._fe_budget_setting = getattr(self, "fe_budget_learning", False)
        st._fe_budget_learning_effective = bool(st._fe_budget_setting)
        st._fe_loaded_budgets = None
        st._fe_budget_learning_effective, st._fe_loaded_budgets = self._fit_body_isinstance_fe_budget_setting(
            st._fe_budget_setting, X, st._fe_budget_learning_effective, st._fe_loaded_budgets
        )
        if st._fe_budget_learning_effective:
            apply_learned_fe_budgets(self, X, st._fe_loaded_budgets, st._fe_budget_quota_snapshot)

    def _fit_body_warn_accuracy_params(self) -> None:
        """Surface (once) any valid-but-accuracy-degrading parameter value; a diagnostic failure never breaks a fit."""
        # ACCURACY-CAVEAT WARNING. Surface (once) any parameter value that is valid but KNOWN to degrade
        # selection accuracy (an explicit opt-out of an on-by-default accuracy mechanism, or a numeric
        # knob pinned to a documented-bad setting). Silent on a default config. See
        # ``_param_accuracy_warnings.ACCURACY_SUBOPTIMAL`` - the single source of truth for the
        # ``# [ACCURACY-CAVEAT]`` markers on the flagged constructor parameters.
        try:
            warn_accuracy_suboptimal_params(self)
        except Exception as exc:
            logger.debug(
                "mrmr: accuracy-suboptimal param warning failed (diagnostic only): %r", exc, exc_info=True
            )  # a diagnostic warning must never break a fit

    def _fit_body_try_stability_outer(self, X: Any, y: Any, groups: Any, sample_weight: Any, fit_params: dict) -> Any:
        """Delegate to the stability-selection bootstrap aggregator when requested; returns its fitted result, or None to continue with the classic fit."""
        # Stability-selection outer-loop short-circuit.
        # When ``stability_selection_method`` is 'cluster' or
        # 'complementary_pairs', delegate to the bootstrap aggregator before
        # the legacy single-fit body executes.
        _stab_method = getattr(self, "stability_selection_method", "classic")
        if _stab_method not in ("classic", "cluster", "complementary_pairs"):
            raise ValueError(f"stability_selection_method must be 'classic', 'cluster', or 'complementary_pairs'; got {_stab_method!r}.")
        if _stab_method != "classic":
            try:
                _stab_result = self._stability_outer_fit(
                    X,
                    y,
                    groups=groups,
                    sample_weight=sample_weight,
                    **fit_params,
                )
            except Exception as _exc:
                warnings.warn(
                    f"MRMR stability_selection_method={_stab_method!r} outer-loop raised {type(_exc).__name__}: {_exc}. "
                    "Falling back to classic fit. This is usually a data-shape issue (too few rows for "
                    "stability_n_bootstrap subsamples, or a column that can't enter the correlation "
                    "clustering step) rather than a bug -- check n_rows / stability_n_bootstrap if it recurs.",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                if _stab_result is not None:
                    return _stab_result
        return None

    def _fit_body_try_multioutput(self, X: Any, y: Any, groups: Any, sample_weight: Any, fit_params: dict, st: Any) -> Any:
        """Reject NaN/Inf in y and route a 2D y through the per-column multi-output fit; returns its result, or None for the single-target path."""
        # Reject NaN/Inf in y at fit entry, matching the sibling selectors
        # (RFECV / ShapProxiedFS): NaN flowing into the MI scorer silently
        # degrades relevance numbers. Skip the check on object-dtype y
        # (categorical labels) where np.isnan would raise; numeric / float /
        # int paths get validated.
        st._y_check = np.asarray(y)
        self._fit_body_int_paths_get_validated(st._y_check)

        # Multi-output (2D y) opt-in. MRMR's merged-target greedy under-selects the 2nd genuine feature on a 2D y (the lazy confirmation step
        # drops it even though per-column MI is high), so fit one single-target selector per output column (the correct 1D path) and aggregate.
        _mo_strategy = getattr(self, "multioutput_strategy", "union")
        if _mo_strategy not in (None, "joint", "union", "intersect"):
            raise ValueError(f"multioutput_strategy must be None, 'joint', 'union', or 'intersect'; got {_mo_strategy!r}.")
        # 09_error_messages_ux.md: 'joint' is intentionally EQUIVALENT to None here (both fall through to
        # the legacy merged-target path below, per the ctor docstring: "None / 'joint': legacy
        # merged-target behaviour, byte-identical to the prior form") - not a validation-accepts-but-
        # runtime-ignores gap. Only 'union'/'intersect' route through the per-column multioutput path.
        if _mo_strategy in ("union", "intersect") and _mrmr_y_is_multioutput(y):
            return self._fit_multioutput(X, y, groups, sample_weight, _mo_strategy, fit_params)
        return None

    def _fit_body_try_identity_shortcut(self, X: Any, y: Any, st: Any) -> Any:
        """Probe the cross-target identity cache; returns ``self`` after the identity shortcut on a hit, else None."""
        # #2 cross-target identity cache.
        st._identity_skip = bool(getattr(self, "mrmr_skip_when_prior_was_identity", False))
        _include_y = bool(getattr(self, "mrmr_identity_cache_include_y", False))
        # Suite caller (train_mlframe_models_suite) can inject a ctx-scoped dict here via
        # ``_mlframe_identity_cache_override_`` so cache lifetime is bounded by the suite
        # rather than the process. When absent, fall back to the module-level dict for
        # cross-suite reuse (CI matrices opt in via mrmr_identity_cache_scope="process").
        # Entries are legacy bool OR (is_id, prior_y_sample) tuples (see the store below); the module-level
        # dict is declared dict[str, bool] for its legacy shape, so widen the local view to match actual usage.
        st._cache_dict = getattr(self, "_mlframe_identity_cache_override_", None)
        if st._cache_dict is None:
            st._cache_dict = cast(dict, _MRMR_IDENTITY_FP_CACHE)
        st._x_fp = None
        if st._identity_skip:
            # Covers X, y (when ``mrmr_identity_cache_include_y``) and this selector's params. Read, store and the self-refit check below all
            # use this one key.
            st._x_fp = _mrmr_identity_cache_key(self, X, y)
            with _MRMR_IDENTITY_FP_LOCK:
                _prior_entry = st._cache_dict.get(st._x_fp)
            # Entry is either a legacy bool or the (is_id, prior_y_sample) tuple stored below.
            if isinstance(_prior_entry, tuple):
                _prior_was_identity, st._prior_y_sample = _prior_entry
            else:
                _prior_was_identity, st._prior_y_sample = _prior_entry, None
            # A refit of this SAME instance on the exact (X, y) pair that produced this cache entry
            # is not a cross-target reuse - it's a self-refit, which _fit_impl's own signature /
            # _FIT_CACHE shortcuts already handle precisely (replaying the TRUE fitted support_ order
            # and mrmr_gains_). Taking the coarse identity-shortcut here instead would silently replace
            # the real MI-ranked selection order with raw arange(n_cols) and zero out mrmr_gains_/
            # provenance_ - caught live via a bit-identical-refit regression test.
            _is_self_refit = st._x_fp == getattr(self, "_own_last_identity_fp_", None)
            if _prior_was_identity is True and not _is_self_refit:
                _ycorr_thr = float(getattr(self, "mrmr_identity_cache_ycorr_threshold", 0.0) or 0.0)
                _ycorr_ok = True
                _measured_corr = None
                _measured_corr, _ycorr_ok = self._fit_body_ycorr_thr(_ycorr_thr, st._prior_y_sample, y, _measured_corr, _ycorr_ok)
                if _ycorr_ok:
                    logger.info(
                        "[MRMR] cross-target identity cache HIT for X fingerprint=%s (y-corr=%s, thr=%.3g) -- "
                        "prior fit returned identity, skipping ~minute(s) of FE pipeline.",
                        st._x_fp,
                        ("%.3f" % _measured_corr) if _measured_corr is not None else "n/a",
                        _ycorr_thr,
                    )
                    self._fit_identity_shortcut(X)
                    self._fit_sample_weight_ = None
                    self._identity_cache_ycorr_ = _measured_corr
                    return self
                logger.info(
                    "[MRMR] cross-target identity cache candidate REFUSED for X fingerprint=%s: "
                    "|y-corr|=%.3f < threshold %.3g; running a full fit for this distinct target.",
                    st._x_fp,
                    abs(_measured_corr) if _measured_corr is not None else float("nan"),
                    _ycorr_thr,
                )
        return None

    def _fit_body_sis_and_isolate(self, X: Any, y: Any, st: Any) -> Any:
        """Run the SIS front gate and take the shallow input-isolation copy of a pandas frame; returns the working ``X``."""
        # INPUT-MUTATION ISOLATION (P1): ``_fit_impl`` injects temporary
        # ``targ_*`` columns into the working pandas frame (X.loc[:, target_names] = ...)
        # AND the FE pipeline appends engineered columns in place (X[name]=..., pd.concat
        # rebinds, hinge/cat-FE generators). The targ_* injection is reversed in the finally
        # below, but the engineered FE columns were NEVER removed - so a caller-supplied
        # DataFrame came back with ['a','b'] -> ['a','b','a__relu_gt...',...] permanently
        # appended. This breaks the sklearn fit-must-not-mutate-input contract and, worse,
        # silently corrupts a frame REUSED across fits: the 2nd fit's FE builds on the 1st
        # fit's leaked columns -> different post-FE X content -> the _FIT_CACHE / replay
        # signature misses (wrong selection / no cache hit). Surfaced by
        # test_replay_fitted_state_isolation.py (A then B fit on the SAME X: A's leaked FE
        # cols made B's content hash differ -> no cache replay -> arrays not shared/frozen).
        #
        # FIX AT THE BOUNDARY (not in the FE-step machinery, which is being actively re-split):
        # operate on an INTERNAL copy so every downstream append lands on our copy and the
        # caller's frame is never touched. Polars input is immutable (with_columns returns a
        # new frame), numpy arrays carry no column index - neither is mutated by our code, so
        # only pandas needs isolating.
        #
        # ALWAYS a SHALLOW copy (deep=False), regardless of pandas' Copy-on-Write setting
        # a deep copy is a real O(n*p) alloc+memcpy of the
        # whole (possibly SIS-reduced) frame, on EVERY fit, and pandas < 3.0's default IS
        # CoW-off - so this was not a rare edge case, it was the common path for most installed
        # pandas versions. The deep-copy branch existed only as a defensive fallback for "a
        # shallow copy could write through an existing-cell mutation" - but every internal
        # mutation site (this boundary's own ``targ_*`` injection at ``X.loc[:, target_names] =
        # ...``, and every FE stage: ``X[name] = ...``, ``pd.concat`` rebinds, hinge/cat-FE
        # generators) only ever ADDS a NEW column key, never overwrites an EXISTING column's
        # cell values in place. A pandas ``DataFrame.copy(deep=False)`` shares the existing
        # columns' underlying arrays with the caller's frame, but assigning a NEW column key
        # allocates a fresh block on the COPY's own BlockManager - it cannot write through to
        # the original's shared arrays regardless of CoW, because no code path here ever
        # mutates an EXISTING column's values in place. Verified empirically (not just by
        # code-reading) with CoW forced off: fit an FE-heavy MRMR on a plain pandas frame,
        # confirm the caller's original frame is byte-identical after fit (columns + values +
        # dtypes), across test_mrmr_input_not_mutated.py, test_replay_fitted_state_isolation.py,
        # and the broader fe/ suite - all pass under a forced-shallow copy. Copy ONCE here, not
        # per FE step.
        # SIS FRONT GATE (Gate A) dispatch. When the frame is at/above ``sis_screen_threshold`` columns, run
        # the chunked O(p*n) screen (filters/_mrmr_sis_screen.sis_screen) to cut the pool to a few thousand
        # data-derived survivors BEFORE the super-linear MRMR machinery. Fastest-default dispatch, not opt-in.
        # Subsetting X to survivor columns (pandas/polars/numpy) keeps the rest of fit unchanged; the screen
        # is best-effort - any failure falls through to the full path.
        st._sis_input_space_info = None
        X, st._sis_input_space_info = self._fit_body_best_effort_any_failure(X, y, st._sis_input_space_info)

        if isinstance(X, pd.DataFrame):
            X = X.copy(deep=False)
        return X

    def _fit_body_activate_thread_locals(self, X: Any, y: Any, groups: Any, sample_weight: Any, st: Any) -> None:
        """Snapshot the MI thread-locals and activate this fit's SU/JMIM/BUR/MM/CS/group-MI/RelaxMRMR/PID/CMI/CPT/DCD toggles (restored in the finally block)."""
        from ._mrmr_class import _restore_toggles_snapshot_and_raise_impl, set_bur_lambda, set_cmi_perm_stop, set_cpt_test, set_jmim_aggregator, set_mi_chao_shen, set_mi_miller_madow, set_pid_synergy_bonus, set_relaxmrmr_alpha, set_su_normalization  # resolved on the class module so patches of its names reach this method
        # activate thread-local SU normalization when mi_normalization='su'.
        # The toggle is read by evaluation.py / Fleuret loops at the scoring site so
        # raw conditional_mi (and cached entropy numbers) stay legacy-bit-stable for
        # the default ``mi_normalization='none'`` path. Restored in finally so a
        # crashing _fit_impl can't leak SU mode into subsequent fits.
        # Snapshot the MI thread-locals at fit ENTRY so the ``finally`` can restore the caller's values
        # instead of hardcoded literals: a nested / outer MRMR fit (the worker path already does this in
        # _evaluation_driver.py) must not have its toggles clobbered to False/0.0 when an inner fit exits.
        st._toggles_snapshot = (
            use_su_normalization(),
            use_jmim_aggregator(),
            get_bur_lambda(),
            use_mi_miller_madow(),
            get_relaxmrmr_alpha(),
            get_pid_synergy_bonus(),
            get_cmi_perm_stop(),
            get_cpt_test(),
            use_mi_chao_shen(),
            get_group_mi(),
        )

        _restore_toggles_snapshot_and_raise = partial(_restore_toggles_snapshot_and_raise_impl, st._toggles_snapshot)

        _mi_norm = getattr(self, "mi_normalization", "none")
        if _mi_norm not in ("none", "su"):
            raise ValueError(f"MRMR.mi_normalization must be 'none' or 'su'; got {_mi_norm!r}.")
        st._prev_su = _mi_norm == "su"
        set_su_normalization(st._prev_su)
        # activate JMIM aggregator + BUR weight thread-locals.
        # Both default OFF (redundancy_aggregator=None, bur_lambda=0.0) so the
        # legacy Fleuret path stays bit-stable.
        _redundancy_agg = getattr(self, "redundancy_aggregator", None)
        if _redundancy_agg not in (None, "jmim", "auto"):
            # A typo (e.g. 'JMIM', 'jimm') would otherwise silently fall through to plain Fleuret with no signal
            # that the requested aggregator was ignored - fail loudly instead.
            _restore_toggles_snapshot_and_raise(ValueError(f"redundancy_aggregator must be one of None, 'jmim', 'auto'; got {_redundancy_agg!r}."))
        st._jmim_on = self._fit_body_requested_aggregator_was_ignored(_redundancy_agg, X, y)
        st._bur_lambda = float(getattr(self, "bur_lambda", 0.0) or 0.0)
        set_jmim_aggregator(st._jmim_on)
        set_bur_lambda(st._bur_lambda)
        # Miller-Madow / Chao-Shen relevance-MI bias correction. Both subtract/re-estimate away the plug-in
        # estimator's finite-sample bias from the OBSERVED relevance so high-cardinality noise no longer
        # out-ranks low-cardinality true signal at small n. Default 'none' keeps the legacy plug-in
        # estimator bit-exact. Reset in the finally. Chao-Shen
        # was previously an accepted-but-silently-ignored value (degraded to plug-in with a warning); it is
        # now fully wired into both the observed-relevance and permutation-null paths, mirroring
        # Miller-Madow's wiring exactly (see compute_relevance_score / mi_or_su_from_classes).
        st._mi_corr = getattr(self, "mi_correction", "none")
        st._mm_on = st._mi_corr == "miller_madow"
        st._cs_on = st._mi_corr == "chao_shen"
        set_mi_miller_madow(st._mm_on)
        set_mi_chao_shen(st._cs_on)
        # Group-aware relevance MI: per-group I(X;Y|G) so a between-group-level feature (high global MI, ~0 within-group)
        # is demoted. Row resampling under non-uniform sample_weight reshuffles X but not groups, so restrict to the
        # no-resample case (sample_weight is None); otherwise disable with a warning rather than mis-assign rows.
        st._gmi_payload = None
        self._fit_body_step1_no_resample_case(groups, X, _restore_toggles_snapshot_and_raise, sample_weight, st)
        _set_group_mi(st._gmi_payload)
        # Research-knob thread-locals (RelaxMRMR 3-D redundancy / PID synergy bonus / CMI permutation early-stop). All default OFF (alpha=0 / bonus=0 / stop=False) so the
        # legacy Fleuret per-candidate score is byte-identical; reset in the finally. Read in evaluation.py and forwarded to joblib workers like the SU/JMIM/BUR toggles.
        set_relaxmrmr_alpha(float(getattr(self, "relaxmrmr_alpha", 0.0) or 0.0))
        set_pid_synergy_bonus(float(getattr(self, "pid_synergy_bonus", 0.0) or 0.0))
        set_cmi_perm_stop(
            bool(getattr(self, "cmi_perm_stop", False)),
            float(getattr(self, "cmi_perm_alpha", 0.05)),
            int(getattr(self, "cmi_perm_n_permutations", 100)),
        )
        set_cpt_test(
            bool(getattr(self, "cpt_test", False)),
            int(getattr(self, "cpt_n_permutations", 200)),
        )
        # activate DCD thread-local. The DCDState dataclass
        # is constructed inside ``_screen_predictors`` (passed via dcd_config
        # kwarg) — joblib-safe; the thread-local is only the read-only branch
        # toggle. Reset in finally.
        st._dcd_on = bool(getattr(self, "dcd_enable", False))
        _set_dcd_active(st._dcd_on)
        # Critic1/H-3 fix: when DCD active and dcd_postoc_compose=False, suppress
        # the post-hoc cluster_aggregate FE-step (else double-aggregation). Save
        # and restore the original flag to keep the constructor-arg semantics
        # bit-stable across fits.
        st._orig_cluster_aggregate_enable = bool(getattr(self, "cluster_aggregate_enable", True))
        st._dcd_suppress_postoc = st._dcd_on and not bool(getattr(self, "dcd_postoc_compose", False))
        if st._dcd_suppress_postoc:
            self.cluster_aggregate_enable = False

    def _fit_body_apply_fit_overrides(self, X: Any, st: Any) -> None:
        """Apply the per-fit overrides (default screen subsample, fast-search profile, lazy seed / skip-content aliases) and snapshot the nbins knobs."""
        from ._mrmr_class import _UNSET  # resolved on the class module so patches of its names reach this method
        # FAST-SEARCH PROFILE. Apply the fast FE-search overrides for the duration of
        # this fit, recording each pre-fit value so the ``finally`` restores constructor-arg
        # semantics (clone / pickle / repeated-fit stability). Only knobs the user left at their
        # package default are overridden - an explicit user value always wins. See the
        # ``fe_fast_search`` docstring in __init__ for the rationale + measured wins.
        # DEFAULT SCREEN SUBSAMPLE. Apply the feature-recovery screen-subsample for large n on
        # EVERY fit (not just fe_fast_search): the FE MI-sweep / polynom-pair / conditional-gate DETECTION
        # are rank-stable under subsampling and the survivors replay at full n, so the default MRMR() fit
        # can screen on ~30k rows at n=100k (168.8s -> ~75s, both compounds still recovered). Only knobs at
        # their package default are shrunk; restored in ``finally``. n below the screen size is a no-op.
        st._default_screen_saved = {}
        st._default_screen_saved = self._fit_body_their_package_default_shrunk(X, st._default_screen_saved)
        st._fast_search_saved = {}
        # Resolve the auto knobs first; an explicit True/False is never touched, and the fast-search pass below leaves
        # these alone because a resolved value no longer equals the None default it compares against.
        st._fast_values = dict(self._FAST_SEARCH_OVERRIDES)
        st._fast_on = bool(getattr(self, "fe_fast_search", False))
        self._fit_body_these_alone_because_resolved(st._fast_search_saved, st._fast_on, st._fast_values)
        self._fit_body_fast(st._fast_on, st._fast_search_saved)
        # LAZY ctor-alias reconciliation (sklearn ``get_params`` stays byte-identical to what the user
        # passed). The constructor no longer promotes ``random_state`` -> ``random_seed``; that is
        # resolved HERE and the EFFECTIVE value is written onto the public attr for the fit duration so
        # every reader (this module + _fit_impl_core's skip check + the cross-file ``self.random_seed``
        # uses) sees it, then the original stored value is restored in ``finally`` (saved == _UNSET means
        # "not overridden, leave alone").
        st._eff_seed = self._effective_random_seed()
        st._orig_random_seed = _UNSET
        if st._eff_seed != getattr(self, "random_seed", None):
            st._orig_random_seed = getattr(self, "random_seed", None)
            self.random_seed = st._eff_seed
        # PICKLE-ONLY migration (NOT a ctor alias - the ctor no longer accepts
        # ``skip_retraining_on_same_shape`` at all): an already-pickled MRMR predating
        # the content/shape rename can still carry the old attribute verbatim in its ``__dict__``
        # (``__setstate__`` never removes it), so a genuinely-legacy saved model's explicit
        # True/False choice is still honoured here rather than silently reset to the current default.
        # A freshly-constructed instance never has this attribute, so this is a no-op for it.
        st._orig_skip_content = _UNSET
        st._skip_shape = getattr(self, "skip_retraining_on_same_shape", None)
        st._orig_skip_content = self._fit_body_freshly_constructed_instance_never(st._skip_shape, st._orig_skip_content)
        # _fit_impl's large-n regression adaptive-quantization
        # gate (adaptive_nbins_large_n_reg) permanently overwrote self.nbins_strategy/self.quantization_nbins
        # in place with no restore anywhere - breaking the sklearn clone()/get_params() round-trip contract
        # and permanently freezing a config the gate's own campaign data says LOSES at smaller n on any
        # subsequent .fit() call on the same instance. Snapshot unconditionally here (mirrors _orig_random_seed
        # / _orig_skip_content above) and restore in the finally block below regardless of whether the gate fired.
        st._orig_nbins_strategy = getattr(self, "nbins_strategy", None)
        st._orig_quantization_nbins = getattr(self, "quantization_nbins", None)

    def _fit_body_run_fit_impl(self, X: Any, y: Any, groups: Any, fit_params: dict) -> Any:
        """Run ``_fit_impl`` inside the global-RNG containment and fit-shape scopes; returns its result."""
        # GLOBAL-RNG CONTAINMENT + SEED DETERMINISM: a fit consumes process-global
        # ``np.random`` in places no per-call Generator covers (cat-confirm permutation shuffles,
        # the FE families' global shuffles, etc.). Two failures followed: (a) an UNSEEDED fit
        # advanced the caller's MT19937 -> a second fit in the same process drifted (run-order
        # flakiness under the xdist suite); (b) even a SEEDED fit (``random_seed`` set) was NON-
        # deterministic in those global-RNG parts because nothing reseeded the process RNG, so the
        # selection drifted run-to-run (e.g. the 5-class layer16 LogReg-macro-F1 gate flaked).
        # Scope the WHOLE fit: when ``random_seed`` is set the block reseeds numpy/numba/cupy to it
        # (the fit becomes reproducible) AND restores the caller's state on exit; ``None`` (no seed
        # requested) restores only. A seeded fit is now deterministic + leak-free; an unseeded fit
        # is leak-free with unchanged (entropy) behaviour.
        # Record the fit's row/column counts so the AUTO (unset MLFRAME_FE_GPU_STRICT) size-gated STRICT
        # default can engage GPU-resident FE on large-n fits (selection-equivalent to CPU by ~50k, ~2.5x
        # faster) OR on a wide-but-under-the-row-threshold fit whose total (n, p) work already clears the
        # same floor a per-call dispatch would need (a fix for when the row-only gate ignored column
        # count entirely), and stay on the exact CPU path otherwise. Cleared in finally so it never leaks
        # to a later fit.
        _fit_shape = getattr(X, "shape", None)
        _set_auto_fit_n(
            int(_fit_shape[0]) if _fit_shape is not None else None,
            int(_fit_shape[1]) if _fit_shape is not None and len(_fit_shape) > 1 else None,
        )
        try:
            with _preserve_global_numpy_rng_state(self._effective_random_seed()):
                result = self._fit_impl(X, y, groups, **fit_params)
        finally:
            _clear_auto_fit_n()
            # _fit_impl_core.py sets the FE-enrichment wall-clock deadline (_fe_deadline.py) near its
            # own top but - despite that module's docstring claiming "cleared in the finally below" -
            # nothing ever actually cleared it: a fit with max_runtime_mins set left the deadline as a
            # stale absolute timestamp for the rest of the process, silently short-circuiting every
            # later fe_deadline_passed() consumer (including any subsequent MRMR.fit call that itself
            # passed NO budget) once that timestamp elapsed. Mirrors _clear_auto_fit_n's own
            # established pattern here - clear exactly once, at the SAME outer call-site boundary,
            # regardless of how _fit_impl exited.
            _clear_fe_deadline()
        return result

    def _fit_body_post_fit(self, result: Any, X: Any, y: Any, st: Any) -> Any:
        """Post-fit bookkeeping (SIS remap, identity-cache stash, provenance, rejection ledger, FE budgets, summary); returns ``result``."""
        if st._sis_input_space_info is not None:
            # The fit ran on the SIS survivors; express support_ / feature_names_in_ / n_features_in_ in the caller's input space before
            # anything below reads them. Done per fit, so a cache replay of the survivor-space state is remapped the same way.
            from .._mrmr_sis_apply import _remap_sis_fit_to_input_space

            _remap_sis_fit_to_input_space(self, *st._sis_input_space_info)
        self._fit_body_step1_anything_below_reads(X)
        # Stash X-fingerprint -> identity-bool in cross-target cache so a SUBSEQUENT fit (different y, same X) can early-skip the FE pipeline.
        self._fit_body_stash_fingerprint_identity_bool(st._identity_skip, st._x_fp, X, y, st._cache_dict)
        # populate ``fe_provenance_`` from
        # the sibling module so users can audit which engineered
        # columns landed in support_, why (origin + mechanism
        # details) and what each contributed in the greedy gain
        # ledger. Pure metadata; never mutates the selection result.
        _pop_prov(self)
        # populate ``fe_rejection_ledger_`` - the rejection side of the provenance
        # surface: one row per FE candidate a gate dropped (which gate + the margin).
        # Pure metadata built from the records ``_run_fe_step`` accumulated; never
        # mutates the selection result.
        _pop_rej(self)
        # gt_07 FE-family budget: compute this fit's per-family credit/ROI from fe_provenance_ +
        # the wall-time ledger, reallocate next fit's budget, and persist it under the dataset
        # fingerprint computed above. Report is populated whenever the flag is on, independent of
        # whether the pre-fit quota scaling above succeeded (visibility is half the feature's
        # value even before enforcement, per the plan). Never allowed to affect the selection
        # result itself - wrapped defensively, same posture as provenance population above.
        # Reuses the SAME resolved flag the pre-fit block computed above ("auto" only persists /
        # reallocates when it actually fired this fit - a probe that found nothing cached must not
        # start writing a NEW cache entry, or "auto" would silently flip itself on forever after one
        # fit, defeating the whole "opt-in-once" contract).
        if st._fe_budget_learning_effective:
            update_fe_budgets_after_fit(self)
        _log_fe_wall()
        self._print_fit_summary()
        return result
