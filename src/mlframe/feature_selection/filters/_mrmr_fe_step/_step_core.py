"""MRMR._run_fe_step - the FE-step orchestrator for mlframe.feature_selection.filters.mrmr.

The single irreducible ~1.36k-line _run_fe_step function lives here, verbatim. The package
__init__ re-exports it (and the small _helpers symbols), and mrmr.py binds it back onto
the MRMR class at its module bottom, so self._run_fe_step(...) call sites are unchanged.

The parent module helpers (_lazy_chunks etc.) and every other intra-filters dependency are
imported lazily in-body (from ..mrmr import ... etc.) since .mrmr re-imports this package at
its bottom for method binding - a top-level import would create a hard cycle. The self-contained FE
sub-blocks (synergy bootstrap, order-2 maxT floor, FE summary log, cluster-aggregate emission) live in
the sibling _mrmr_fe_step_helpers module; the two small operand-pool helpers live in ._helpers.

bench-note (2026-06-24, MRMR-FE wall /loop): _run_fe_step's OWN body is AT FLOOR - it is pure
orchestration. cProfile F2 100k full fit: this function's tottime = 0.001s (cumtime ~13s over 2 calls is
ALL in carved-sibling callees). line_profiler over the body: glue (every getattr resolution, the
synergy-budget dict-comp at ~300, the original_cols dict-comp at ~464, sort_dict_by_value, the prewarp/
gate-med/rejection-ledger merge loops) sums to 0.166ms = 0.149% of the function's 11.16s; the other
99.85% is the 9 call sites into _step_pool / _step_pairmi / _step_pairs_rank / check_prospective_fe_pairs
/ _step_score. The remaining wall lives in (a) GPU-dispatched routes (cupy .get 2.0s,
gpu_materialise_discretize_codes_host 2.77s cum) and (b) njit compute at CPU-optimum
(_plugin_mi_classif_batch_njit 1.8s, _combine_factorize_njit 0.85s, _pair_combo_mi_njit_table_parallel
0.82s, conditional_mi 0.64s) - none in this body. No safe wall win exists at the orchestrator level; do
not re-profile the body. Next dominant wall hotspot = the _score_one_pair / check_prospective_fe_pairs
permutation-null + cupy-host-transfer path (already GPU-routed; iter9/11b/15).
"""
from __future__ import annotations


from ._step_core_helpers import (
    _resolve_prevalence_and_stratify_knobs,
    logger,
)
from types import SimpleNamespace as _SimpleNamespace
from ._step_core_parts import _run_fe_step_impl_part2, _run_fe_step_impl_part5, _run_fe_step_impl_part6, _run_fe_step_impl_part7


def _should_serialize_fe_pair_check(n_prospective_pairs: int, gpu_fe_active: bool, serial_min_pairs_per_worker: int) -> bool:
    """Route ``check_prospective_fe_pairs`` to the serial-main-thread branch instead of the joblib
    threading fan-out. True when the GPU-discretize path is active (N threads would otherwise all
    contend for the single device - see the GPU-FE SERIALIZE comment at this module's main call site) OR
    when there are too few prospective pairs to fill the worker pool. Extracted as a pure, directly
    Testable predicate; behavior is unchanged.
    """
    return True if gpu_fe_active else int(n_prospective_pairs) < int(serial_min_pairs_per_worker)


def _free_gpu_fe_mempool() -> bool:
    """Free the cupy default pool's UNUSED blocks at GPU FE-step teardown; return whether a free was issued.

    When the FE step ran GPU kernels (``MLFRAME_FE_GPU_STRICT`` / ``MLFRAME_CMI_GPU``), the cupy default pool
    RETAINS every block it cached this step for reuse. Across repeated fits (CV folds, repeated ``MRMR().fit``
    on a 4 GB card) the retained pool sits at its high-water mark near the device cap and the allocator thrashes
    cudaMalloc/sync - measured: consecutive 100k f32 STRICT fits degrade 11.2s -> 31.8s -> 32.3s, while freeing
    the unused blocks at each step teardown holds the footprint flat (~2.9 GB) and the wall at ~11.2s.
    ``free_all_blocks`` only returns blocks with no live reference, so a resident operand table held across the
    fit is untouched; this is post-compute teardown, never mid-pipeline. Best-effort + env-gated so CPU users /
    other backends are not touched.

    bench-attempt-rejected: releasing only a pool above a fraction of the card (so a modest 0.8 GB pool survives into the next step) was measured at 100k and 1M rows on the
    4 GB dev card: fit wall unchanged (12.0 / 12.3 s against 12.0 / 12.1 s). One 320 MB block re-allocation took 0.29 s in a traced run but not in the timed ones.
    """
    import os as _os
    if not (_os.environ.get("MLFRAME_FE_GPU_STRICT") or _os.environ.get("MLFRAME_CMI_GPU")):
        return False
    # Drop the resident y/z device cache in _cmi_cuda FIRST so its device arrays carry no
    # live reference -> free_all_blocks below can actually reclaim them (a fit-scoped cache, never persisted).
    try:
        from mlframe.feature_selection.filters.info_theory.shared import clear_cmi_resident_cache
        clear_cmi_resident_cache()
    except ImportError as exc:  # the CUDA CMI module is genuinely absent
        logger.debug("mrmr: no CMI resident device cache to clear at FE-step teardown (%s)", exc)
    except Exception as exc:
        # NOT a missing dependency: the module imported and the clear itself failed, which is a device or driver fault worth seeing.
        logger.warning("mrmr: clearing the CMI resident device cache at FE-step teardown failed (%s: %s); VRAM may not be reclaimed", type(exc).__name__, exc)
    # FE operand resident cache: drop the fit-constant operand device copies (y / z / base columns) so their
    # device arrays carry no live reference -> free_all_blocks below can reclaim them (a fit-scoped cache).
    try:
        from .._fe_resident_operands import clear_fe_resident_operands
        clear_fe_resident_operands()
    except ImportError as exc:  # the resident-operand module is genuinely absent
        logger.debug("mrmr: no FE resident operands to clear at teardown (%s)", exc)
    except Exception as exc:
        logger.warning("mrmr: clearing FE resident operands at teardown failed (%s: %s); VRAM may not be reclaimed", type(exc).__name__, exc)
    try:
        import cupy as _cp
        _cp.get_default_memory_pool().free_all_blocks()
        return True
    except ImportError as exc:  # cupy not installed: this is the CPU-only path, not a failure
        logger.debug("mrmr: cupy is not available, so there is no device pool to free at FE-step teardown (%s)", exc)
        return False
    except Exception as exc:
        # cupy imported and the free itself failed. Reported at warning because the documented consequence - the retained pool sitting at its
        # high-water mark and the allocator thrashing, measured as 11.2s -> 31.8s -> 32.3s across consecutive fits - is otherwise invisible.
        logger.warning(
            "mrmr: cupy free_all_blocks() at FE-step teardown failed (%s: %s); VRAM will not be reclaimed this step and repeated fits will "
            "degrade as the pool stays at its high-water mark",
            type(exc).__name__, exc,
        )
        return False


def _capture_prefe_screened_raw(self, cols, selected_vars) -> None:
    """Record the screening-confirmed raw columns on ``self._prefe_screened_raw_`` so support finalisation can re-add them."""
    try:
        _raw_set = set(self.feature_names_in_)
        self._prefe_screened_raw_ = [cols[v] for v in selected_vars if cols[v] in _raw_set]
    except Exception as exc:
        # Runs at most once per fit, so a warning costs nothing, and without it genuine raws can be dropped with no trace.
        logger.warning("mrmr: capturing the pre-FE screened-raw safety net failed; support finalisation cannot re-add these raws: %s: %s", type(exc).__name__, exc)
        self._prefe_screened_raw_ = []


def _run_fe_step(self, **kwargs):
    """Fit-scoped wrapper around ``_run_fe_step_impl``: opens ``dedup_collinear_memo_scope()`` for the
    WHOLE FE step so every opt-in FE family's ``_dedup_collinear_source_cols`` call within this one round
    (orth univariate, extra-basis, gpu-resident, wavelet, hinge - up to 7 call sites, see that function's
    docstring) shares ONE fit-scoped memo instead of each repeating the identical O(P^2) corrcoef pass on
    the same candidate columns. cProfile, 2026-07-16 wellbore-100k fit: _dedup_collinear_source_cols cost
    3.8s (3 calls, orth univariate path) + 2.5s (hinge path) SEPARATELY in one FE step despite operating on
    overlapping candidate-column sets. The memo mechanism (content-hash keyed, thread-local, nesting-safe)
    Already existed but was never actually wired to any call site - built, never engaged.
    Wrapping here (not each family's own call site) is the single choke point every family's dedup call
    passes through, so no per-family wiring can be missed. Purely additive: with no repeated (cols,
    corr_threshold) pair the memo is a single miss-then-store per key, i.e. free within measurement noise."""
    from .._orthogonal_univariate_fe._orth_dedup import dedup_collinear_memo_scope
    with dedup_collinear_memo_scope():
        # _free_gpu_fe_mempool teardown used to be a bare
        # trailing statement inside _run_fe_step_impl - any exception raised anywhere in that ~900-line
        # body after the GPU-STRICT path engaged skipped VRAM-pool cleanup entirely, silently
        # reintroducing the cross-fit VRAM-bloat degradation (11.2s -> 31.8s -> 32.3s, see the teardown's
        # own docstring) on the very "repeated MRMR().fit / CV folds" scenario it exists to prevent.
        # try/finally here (the single choke point every call passes through) guarantees it always runs,
        # success or exception, without touching the impl function's own body.
        try:
            return _run_fe_step_impl(self, **kwargs)
        finally:
            _free_gpu_fe_mempool()


def _run_fe_step_impl(
    self,
    *,
    # Mutable state from MRMR.fit (returned updated as tuple).
    data, cols, nbins, X,
    target_names, target_indices,
    selected_vars, categorical_vars,
    classes_y, classes_y_safe, freqs_y,
    cached_MIs, cached_confident_MIs,
    unary_transformations, binary_transformations,
    engineered_features, checked_pairs,
    # Parallel dict (name -> EngineeredRecipe) populated as new columns are added to data / cols so
    # transform() can replay them on test data. Mutated in place; MRMR.fit reads it after the FE loop
    # and copies surviving recipes into self._engineered_recipes_. ``None`` skips recipe construction.
    engineered_recipes=None,
    times_spent,
    num_fs_steps,
    # Service.
    n_jobs, prefetch_factor, parallel_kwargs,
    _is_polars_input, verbose,
    # FE config (frozen per fit).
    fe_max_steps, fe_npermutations, fe_max_pair_features,
    fe_print_best_mis_only, fe_min_nonzero_confidence,
    fe_min_engineered_mi_prevalence,
    fe_good_to_best_feature_mi_threshold,
    fe_max_external_validation_factors,
    fe_min_pair_mi, fe_min_pair_mi_prevalence,
    fe_smart_polynom_iters, fe_smart_polynom_optimization_steps,
    fe_min_polynom_degree, fe_max_polynom_degree,
    fe_min_polynom_coeff, fe_max_polynom_coeff,
    # Preset-name snapshot so recipes can rebuild the correct registry at replay time. Default "minimal"
    # matches MRMR.__init__ defaults; callers that override via self.fe_unary_preset / self.fe_binary_preset
    # get the actual values threaded through by fit().
    fe_unary_preset: str = "medium",
    fe_binary_preset: str = "minimal",
):
    """One Feature Engineering iteration. Extracted from ``MRMR.fit`` for testability and FE experimentation.

    Returns ``None`` if the FE step should not run (empty-screen + ``fe_fallback_to_all=False``); otherwise
    ``(data, cols, nbins, X, selected_vars, n_recommended_features)``. ``n_recommended_features == 0`` signals
    the outer loop to stop. Private; external callers should use ``MRMR.fit()`` or ``MRMR.fit_transform()``.
    """
    # VRAM POOL CAP: setup counterpart to the ``_free_gpu_fe_mempool`` teardown below. Cap MRMR's
    # OWN cupy default pool to ``MLFRAME_FE_GPU_POOL_FRACTION`` (default 0.6) of total VRAM ONCE per process at the
    # FE-step entry, so the pool cannot grow unbounded and eat a shared 4 GB card (starving the next launch / other
    # processes). Idempotent + best-effort (no-op without cupy); on exhaustion cupy raises OutOfMemoryError which
    # the GPU-FE try/excepts catch -> graceful CPU. Cheap: the once-flag short-circuits every call after the first.
    st = _SimpleNamespace()  # long-lived locals of this function (see the stage helpers below)
    try:
        from .._fe_gpu_vram import ensure_fe_gpu_pool_limit as _ensure_fe_gpu_pool_limit
        _ensure_fe_gpu_pool_limit()
    except Exception as e:
        logger.debug("mrmr: ensure_fe_gpu_pool_limit() (VRAM pool cap at FE-step entry) failed (best-effort): %s", e, exc_info=True)
    # SEPARATE KTC-free GPU-RESIDENT FE step (MLFRAME_FE_GPU_STRICT + MLFRAME_FE_GPU_STRICT_RESIDENT). Per
    # ``fe_gpu_strict_resident_enabled()``'s own docstring this is now DEFAULT-ON whenever MLFRAME_FE_GPU_STRICT
    # itself is on (this comment previously said "default OFF", which
    # was stale and had led a prior audit to independently mis-conclude this call site had zero production
    # callers). When the resident path is enabled it takes over the WHOLE FE step (operands uploaded once per
    # device, all compute on GPU kernels, no bulk D2H). Phase 0: the entry is a stub that raises
    # NotImplementedError, so this is INERT (falls through to the existing per-family path below) until later
    # phases implement it.
    from .._gpu_strict_fe import fe_gpu_strict_resident_enabled, run_fe_step_gpu_strict
    if fe_gpu_strict_resident_enabled():
        # Explicit allowlist of the forwarded kwargs, not a locals()-introspection snapshot.
        # The prior `{k: v for k, v in locals().items() if k not in (...)}` pattern depended on being written
        # BEFORE any other local was bound in this function - any future edit inserting a new local-binding
        # line ahead of this block would silently change the kwarg set forwarded to run_fe_step_gpu_strict,
        # a bug that would only surface once a later phase actually implements that function.
        _fe_args = {"data": data, "cols": cols, "nbins": nbins, "X": X, "target_names": target_names,
            "target_indices": target_indices, "selected_vars": selected_vars, "categorical_vars": categorical_vars,
            "classes_y": classes_y, "classes_y_safe": classes_y_safe, "freqs_y": freqs_y, "cached_MIs": cached_MIs,
            "cached_confident_MIs": cached_confident_MIs, "unary_transformations": unary_transformations,
            "binary_transformations": binary_transformations, "engineered_features": engineered_features,
            "checked_pairs": checked_pairs, "engineered_recipes": engineered_recipes, "times_spent": times_spent,
            "num_fs_steps": num_fs_steps, "n_jobs": n_jobs, "prefetch_factor": prefetch_factor,
            "parallel_kwargs": parallel_kwargs, "_is_polars_input": _is_polars_input, "verbose": verbose,
            "fe_max_steps": fe_max_steps, "fe_npermutations": fe_npermutations,
            "fe_max_pair_features": fe_max_pair_features, "fe_print_best_mis_only": fe_print_best_mis_only,
            "fe_min_nonzero_confidence": fe_min_nonzero_confidence,
            "fe_min_engineered_mi_prevalence": fe_min_engineered_mi_prevalence,
            "fe_good_to_best_feature_mi_threshold": fe_good_to_best_feature_mi_threshold,
            "fe_max_external_validation_factors": fe_max_external_validation_factors,
            "fe_min_pair_mi": fe_min_pair_mi, "fe_min_pair_mi_prevalence": fe_min_pair_mi_prevalence,
            "fe_smart_polynom_iters": fe_smart_polynom_iters,
            "fe_smart_polynom_optimization_steps": fe_smart_polynom_optimization_steps,
            "fe_min_polynom_degree": fe_min_polynom_degree, "fe_max_polynom_degree": fe_max_polynom_degree,
            "fe_min_polynom_coeff": fe_min_polynom_coeff, "fe_max_polynom_coeff": fe_max_polynom_coeff,
            "fe_unary_preset": fe_unary_preset, "fe_binary_preset": fe_binary_preset}
        try:
            return run_fe_step_gpu_strict(self, **_fe_args)
        except NotImplementedError:
            pass  # Phase 0 scaffold / unported config -> existing per-family FE path below
    # GUARDED-ADAPTIVE PREVALENCE, replacing a hardcoded threshold. When
    # ``fe_min_pair_mi_prevalence == "auto"``, keep the proven 1.05 ratio bar but apply it to the
    # MILLER-MADOW-DEBIASED pair MI (the analytic finite-sample joint-MI bias subtracted, the value
    # the maxT floor already uses) rather than the raw ``pair_mi``. Debiasing can ONLY LOWER the
    # observed joint MI, so the gate can only TIGHTEN - it drops the best-of-pool finite-sample-noise
    # pairs a fixed 1.05 admits (the bench showed a higher effective bar helps the bilinear archetype:
    # 0.207 -> 0.092) while a genuine high-signal pair (joint MI >> bias, e.g. F2's c*d) is untouched.
    # Resolve to the float HERE so every downstream float use is byte-identical to the default path;
    # only the per-pair prevalence comparison below switches to the debiased observed value. An
    # explicit float (incl. the default 1.05) is honoured verbatim -> byte-identical to pre-conversion.
    # R1 STRATIFIED SUBSAMPLE flag, resolved ONCE per FE step (UNCONDITIONALLY - both the polynom-pair
    # FE and the unary/binary ``check_prospective_fe_pairs`` calls below consume it, and the polynom
    # block is gated by ``if fe_smart_polynom_iters`` so this must live at function-body level). The
    # MRMR ``fe_subsample_stratify`` tri-state knob (None=auto / True / False) is resolved against the
    # best available target: the continuous y (``_fe_prewarp_y_continuous_``) when present (its
    # dtype/cardinality lets ``infer_classification`` pick clf-vs-reg and the regression heavy-tail
    # heuristic see the real distribution), else the discrete ``classes_y`` codes (classification).
    # Default-None auto-ON fires only on a small rare-class fraction / heavy-tailed target; otherwise
    # OFF -> byte-identical legacy uniform draw.
    st._fe_subsample_stratify, st._prevalence_debias_auto, fe_min_pair_mi_prevalence, st._synergy_prev_resolved = _resolve_prevalence_and_stratify_knobs(self, classes_y, fe_min_pair_mi_prevalence)
    _ret_run_fe_step_impl_part7 = _run_fe_step_impl_part7(
        verbose, selected_vars, self, st, cols, target_names, num_fs_steps, data, categorical_vars, X, nbins, target_indices, classes_y, freqs_y, cached_MIs,
        classes_y_safe, cached_confident_MIs, n_jobs, prefetch_factor, parallel_kwargs, fe_min_nonzero_confidence, fe_npermutations, fe_min_pair_mi,
        fe_min_pair_mi_prevalence, checked_pairs,
    )
    if _ret_run_fe_step_impl_part7[0]:
        return _ret_run_fe_step_impl_part7[1]
    check_prospective_fe_pairs, discretize_array, get_new_feature_name, parallel_run, selected_vars = _ret_run_fe_step_impl_part7[1:]

    data, X, cols, nbins = _run_fe_step_impl_part2(
        self, st, verbose, data, fe_smart_polynom_iters, classes_y, X, cols, _is_polars_input, nbins, engineered_features, engineered_recipes,
        fe_smart_polynom_optimization_steps, fe_min_polynom_degree, fe_max_polynom_degree, fe_min_polynom_coeff, fe_max_polynom_coeff,
        fe_min_engineered_mi_prevalence, n_jobs, num_fs_steps,
    )

    materialise_and_finalise_fe_candidates = _run_fe_step_impl_part5(
        self, st, cols, verbose, classes_y, n_jobs, X, check_prospective_fe_pairs, unary_transformations, binary_transformations, classes_y_safe, freqs_y,
        num_fs_steps, fe_max_steps, fe_npermutations, fe_max_pair_features, fe_print_best_mis_only, fe_min_nonzero_confidence,
        fe_min_engineered_mi_prevalence, fe_good_to_best_feature_mi_threshold, fe_max_external_validation_factors, times_spent, prefetch_factor,
        fe_min_pair_mi_prevalence, parallel_run, parallel_kwargs,
    )
    _ret_run_fe_step_impl_part6 = _run_fe_step_impl_part6(
        materialise_and_finalise_fe_candidates, self, st, data, cols, nbins, X, classes_y, selected_vars, engineered_features, engineered_recipes,
        checked_pairs, num_fs_steps, fe_max_steps, fe_unary_preset, fe_binary_preset, _is_polars_input, verbose, discretize_array, get_new_feature_name,
        unary_transformations, fe_min_pair_mi_prevalence, fe_min_engineered_mi_prevalence, fe_min_nonzero_confidence, fe_good_to_best_feature_mi_threshold,
        target_indices, categorical_vars, cached_MIs,
    )
    if _ret_run_fe_step_impl_part6[0]:
        return _ret_run_fe_step_impl_part6[1]
