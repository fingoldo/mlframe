"""``check_prospective_fe_pairs`` - the FE pair-search core (candidate generation
via unary+binary ops, batched MI + permutation noise-gate, prewarp / median-gate
pseudo-unaries, chunked materialise, kernel-tuning dispatch).

This is the irreducible single-function body of the ``_feature_engineering_pairs``
subpackage; the supporting kernels / gates / dispatch live in sibling submodules
and are re-exported from the package ``__init__``.
"""

from __future__ import annotations

from typing import Any, cast


import numba
import numpy as np

from pyutilz.system import tqdmu

# A column is near-constant when its std is within this factor of machine epsilon of its largest |value|: its spread is then rounding noise.
from ._pairs_core_helpers import (  # noqa: F401  -- carved helpers
    _short_fe_name,
    _check_prospective__use_subsample,
    _check_prospective__behaviour_change_wrapped_experimental,
    _check_prospective__scores_same_rows_fall,
    _check_prospective__shared_idx_none,
    _check_prospective__total_get_proportions_matching,
    _check_prospective__win_do_re_implement,
    _check_prospective__ignored_thread_multiplication_byte,
    _check_prospective__classes_codes_still_usable,
    _check_prospective__only_worth_chunking_least,
    _check_prospective__read_weakref_cache_no,
    _check_prospective__pair_res_entry_none,
    _check_prospective__no_config_nan_edge,
    _check_prospective__pair_scoped_copies_spec,
    _check_prospective__fitted_specs,
    _check_prospective__value_single_float_per,
)


from ._pairs_core_steps import (  # noqa: F401  -- carved helpers
    _DEGENERATE_REL_TOL,
    _abs_corr_finite_njit,
    _check_prospective_fe_step1_def_extval_raw,
    _check_prospective_fe_step2_classes_codes_still,
    _check_prospective_fe_step3_gpu_clock_variance,
    _check_prospective_fe_step4_int_code_array,
)
from ._pairs_core_parts import (_check_prospective_fe_pairs_part10, _check_prospective_fe_pairs_part3, _check_prospective_fe_pairs_part5, _check_prospective_fe_pairs_part6, _check_prospective_fe_pairs_part7, _check_prospective_fe_pairs_part8, _check_prospective_fe_pairs_part9)


@numba.njit(cache=True, fastmath=False)
def _abs_corr_zerofill_njit(a, b):
    """|Pearson corr| of ``a`` vs ``b`` treating any non-finite entry (NaN/+-inf) as 0.0 INLINE, in one pass -
    bit-equivalent to ``abs(np.corrcoef(np.nan_to_num(a), np.nan_to_num(b))[0, 1])`` without materialising either
    zero-filled copy or the 2x2 corrcoef matrix. This is the ZERO-FILL twin of ``_abs_corr_finite_njit`` (which
    instead MASKS non-finite rows out) - deliberately kept separate rather than unified: a veto/dedup check
    comparing two ENGINEERED columns (e.g. a winning composite against one of its own transformed operands, which
    can legitimately contain NaN/Inf from a domain-invalid unary like log/sqrt) that was written against
    nan_to_num-then-corrcoef semantics must keep that exact statistic - masking those rows out instead would
    silently change which rows influence the veto decision. Returns 0.0 when either side is (near-)constant
    post-zero-fill."""
    # Two-pass centred, for the same reason as the masked twin above: recovering the variance from raw power
    # sums by subtraction loses the answer entirely once the data carries an offset large relative to its
    # spread, and the destroyed variance then trips the near-constant branch and returns a confident 0.0.
    n = a.shape[0]
    if n == 0:
        return 0.0
    sa = 0.0
    sb = 0.0
    for i in range(n):
        av = a[i]
        bv = b[i]
        if not np.isfinite(av):
            av = 0.0
        if not np.isfinite(bv):
            bv = 0.0
        sa += av
        sb += bv
    ma = sa / n
    mb = sb / n
    va = 0.0
    vb = 0.0
    cab = 0.0
    amax = 0.0
    bmax = 0.0
    for i in range(n):
        av = a[i]
        bv = b[i]
        if not np.isfinite(av):
            av = 0.0
        if not np.isfinite(bv):
            bv = 0.0
        da = av - ma
        db = bv - mb
        va += da * da
        vb += db * db
        cab += da * db
        if abs(av) > amax:
            amax = abs(av)
        if abs(bv) > bmax:
            bmax = abs(bv)
    if va <= n * (_DEGENERATE_REL_TOL * amax) ** 2 or vb <= n * (_DEGENERATE_REL_TOL * bmax) ** 2:
        return 0.0
    denom = (va * vb) ** 0.5
    if denom <= 0.0:
        return 0.0
    r = cab / denom
    return -r if r < 0.0 else r


from ._pairs_gates import (
    _PREWARP_UNARY,
)
from ._pairs_score import _score_one_pair

# Subsample default for the FE pair-search entry point: the single ``_feature_engineering_mem_budget.UNIFIED_FE_SUBSAMPLE_N``
# source of truth (30k), imported from the leaf that defines it so this package does not load the facade at import time.
from .._feature_engineering_mem_budget import UNIFIED_FE_SUBSAMPLE_N

# GPU gate decisions live in ``_pairs_gpu_gates`` (carved out for the LOC ceiling); re-exported so call sites and tests keep their imports.
from ._pairs_gpu_gates import (  # noqa: F401  -- carved helpers
    _GPU_GATE_CACHE,
    _GPU_GATE_CACHE_LOCK,
    _GPU_GATE_CACHE_MAX,
    _fe_gpu_binning_enabled,
    _fe_gpu_binning_enabled_uncached,
    _fe_gpu_discretize_enabled,
    _fe_gpu_discretize_enabled_uncached,
    _gpu_gate_cached,
    _gpu_gate_env_signature,
)


# Notes on the parameters of check_prospective_fe_pairs (kept verbatim from the signature; [name] marks the parameter each block belongs to):
# [subsample_n]
# CRITICAL #2 follow-up: subsample rows for the MI sweep so the
# transformed_vars + shared_buffer allocations scale with subsample_n rather
# than the (possibly multi-million-row) full X. Survivor columns are still
# produced at full n via _rebuild_full_survivor_col so the caller contract
# is preserved. Default is the unified UNIFIED_FE_SUBSAMPLE_N (30k); the
# standalone accuracy bench in bench_fe_pair_subsample_accuracy.py shows
# jaccard=1.0 vs full at n_eff >= 50k on synthetic 3-pair-competition data.
# 0 = use full data (legacy).
# [shared_subsample_idx]
# ONE shared FE subsample. When the caller (``_mrmr_fe_step``) passes the fit's single
# shared row-index draw, the MI-sweep REUSES it verbatim instead of drawing its own subsample here -
# so the pair-search, the polynom path and the sufficient-summary maxT floor all score the SAME rows
# (one draw per fit, not N independent re-draws). ``None`` keeps the legacy per-call draw below.
# [fe_subsample_stratify]
# STRATIFIED SUBSAMPLE (R1). When True the MI-sweep row subsample below draws a
# TARGET-STRATIFIED set of rows (per-class proportional for classification - guaranteeing the
# rare class survives; y-quantile-bin proportional for regression - preserving the tails)
# instead of the plain uniform ``rng.choice``. False (default) keeps the byte-identical legacy
# uniform draw. The caller (``_mrmr_fe_step``) resolves the MRMR ``fe_subsample_stratify``
# tri-state knob (None=auto) to a concrete bool via ``_resolve_fe_subsample_stratify``.
# [prewarp_enable]
# PER-OPERAND PRE-WARP. When ``prewarp_enable`` is True the
# unary/binary search gains, per raw operand, one extra "pseudo-unary"
# ``prewarp(x)`` - a learned 1-D orthogonal-polynomial warp fit JOINTLY
# across the pair via the rank-1 ALS sweep (``hermite_fe.fit_pair_prewarp_als``,
# which reuses the orthogonal-poly path's ``warm_start_als_seed``; an
# INDEPENDENT 1-D fit cannot recover the b-side of a product target whose
# b-marginal is ~0). This lets the elementary unary/binary path represent a
# within-operand non-monotone distortion such as ``a**3 - 2a`` that no single
# library unary can express, so a target ``F3(F1(a), F2(b))`` with a
# non-monotone inner ``F1`` becomes recoverable.
# ``prewarp_y`` is the SAME discretised target codes (``classes_y``) the MI
# sweep already scores against; the fit is supervised but the produced column
# is a closed-form function of ``x`` alone, so replay is leak-safe (the
# fitted coeffs are stored in the recipe by ``_mrmr_fe_step``). Default OFF
# so behaviour on data that does not need it is byte-identical.
# [prewarp_y_continuous]
# PREWARP ALS RECONSTRUCTION TARGET. The rank-1 ALS warp fit /
# held-out validation / winning-spec reconstruction is a least-squares solve
# of ``y ~ f(a)*g(b)``; its fidelity depends on the RESOLUTION of the target it
# reconstructs. The target-rebin guard coarsens ``classes_y`` to the
# 10-bin equal-frequency screening codes (correctly - the MI screen/gates need
# the faithful coarse codes), but feeding those binned codes to the ALS dropped
# the F-POLY non-monotone product reconstruction |corr| 0.97 -> 0.88. When the
# CONTINUOUS target is threaded here it drives the ALS fit/validate/score ONLY;
# the MI screen + every gate keep using ``classes_y`` codes. None -> legacy
# behaviour (ALS reconstructs against ``classes_y``).
# [usability_y_continuous]
# LINEAR-USABILITY GUARD TARGET. The leader-equivalence tie-break and the
# noise-wrap |corr| guard must score against the CONTINUOUS regression target, NOT the
# binned ``classes_y`` codes: on a heavy-tailed target the Pearson |corr| of a magnitude-
# carrying form (e.g. ``a**2/b``) with the quantile-RANK codes COLLAPSES (~0.05) while a
# bounded monotone warp (e.g. ``a/sqrt(b)``) scores higher (~0.4) - the EXACT INVERSE of
# linear usability. The prewarp path already threaded continuous y, so the exhaustive
# (prewarp-on) path was correct; the fast (prewarp-off) path fell back to ``classes_y`` and
# picked the linearly-useless leg (biz_value_mrmr_fast_search MAE 0.05 -> 37). Thread the
# continuous y here UNCONDITIONALLY (independent of prewarp) so both paths agree. None ->
# legacy ``classes_y`` fallback (correct for classification / non-numeric y).
# [prewarp_uplift_threshold]
# Minimum ratio (best-prewarp-MI / best-nonprewarp-MI) for the alternative
# acceptance path. 1.20 = the prewarp must beat the elementary library by
# >= 20% engineered MI to be admitted past the joint-prevalence gate. Tuned
# to fire on the F-POLY non-monotone inner (measured uplift ~1.42x) while
# staying silent on linear/monotone/noise (uplift ~1.0x there).
# [prewarp_min_val_corr]
# Held-out floor for the out-of-sample prewarp validation: a warp fit on a
# train slice is kept only if its rank-1 reconstruction f(a)*g(b) tracks y on
# a held-out slice with |corr| >= this. Rejects supervised overfitting on noise
# operands at small n; 0.0 disables (legacy in-sample-only fit).
# [fe_gate_med_enable]
# PER-OPERAND MEDIAN GATE. When ``fe_gate_med_enable`` is True the
# unary/binary search gains, per raw operand, one extra "pseudo-unary"
# ``gate_med(x) = (x > train_median_x).astype(float)``. Combined with the
# existing ``mul`` binary this lets the elementary path represent the
# median-gated operators ``(a > median_a) * b`` (``mul(gate_med(a), b)``) and
# the conjunction ``(a > median_a) & (b > median_b)``
# (``mul(gate_med(a), gate_med(b))``) that a bilinear product cannot - the
# signal is non-product / conditional. Unlike a fixed threshold-0 gate, the
# median ADAPTS the split to each operand's distribution, so it recovers the
# gate on shifted / skewed operands where threshold-0 is useless (measured
# skew-bench: gated_med +0.0355 / thr_and_med +0.0435 downstream-AUC d_mean
# vs raw, beating products +0.022/+0.020 and threshold-0 +0.009/+0.0001).
# The fitted state is ONE float per operand (the TRAIN median), stored in the
# recipe by ``_mrmr_fe_step`` for leak-safe closed-form replay (no y, no
# test-time recompute). The median does not overfit, so - unlike prewarp -
# NO held-out validation is needed; the gate is still subject to the same
# MI-prevalence / external-validation acceptance gates every engineered
# feature passes (it competes on equal footing in the per-pair MI sweep and
# wins via ``best_mi`` only where the conditional form genuinely beats the
# library). Default OFF so behaviour is byte-identical when not requested.
# [serial_main_thread]
# OPT-A: True when the caller (``_mrmr_fe_step``) dispatches this on the
# SERIAL MAIN THREAD with NO joblib threading nest (the ``len(X) < 50000`` /
# ``len(prospective_pairs) < 2`` branch). On that path the FE materialise / searchsorted
# kernels may use their ``parallel=True`` column-prange twins (a numba prange is safe -
# nothing nests it). On the joblib ``backend="threading"`` path this stays False so the
# serial ``nogil`` kernels are used (a nested prange deadlocks the threading layer).
# Byte-identical either way - only thread-count of the embarrassingly-parallel per-column
# work changes. Default False = the always-safe serial path.
# [allow_engineered_operands]
# ENGINEERED-OPERAND FEED-FORWARD. When True (default), an operand var
# that is NOT a raw ``feature_names_in_`` column (i.e. it is an engineered column
# appended by a prior FE step, hence absent from ``original_cols``) is resolved by
# NAME from the augmented frame ``X`` rather than skipped. This lets the step-k>1
# pair search build COMPOSITES of two engineered features - e.g. the additive
# ``add(div(sqr(a),abs(b)), mul(log(c),sin(d)))`` that captures ~the entire
# deterministic signal of ``y = a**2/b + log(c)*sin(d)``. The number of engineered
# operands fed back is capped UPSTREAM (``fe_max_engineered_operands`` in
# ``_mrmr_fe_step``) so the O(k^2) pair count stays bounded. Set False to restore
# the legacy raw-only operand pool (no engineered x engineered composites).
# [engineered_operand_values]
# ENGINEERED-OPERAND CONTINUOUS VALUES. ``{engineered_col_name ->
# full-n float64 ndarray}`` of the CONTINUOUS engineered values produced by prior
# FE steps. ``_extval_raw_col`` reads this for engineered operands so the pair
# search combines the CONTINUOUS values rather than the augmented frame's
# DISCRETISED bin codes (combining codes is severely lossy - it sinks the
# additive composite below the engineered-MI gate). ``None`` (default) -> fall
# back to the by-name frame extract (bin codes), preserving the legacy behaviour.
# [fe_multi_emit_max_per_pair]
# MULTI-CANDIDATE DIVERSE EMISSION. Per raw pair, the search picks the
# single MAX-target-MI engineered form and discards every other. But MI is a RANK
# statistic blind to LINEAR usability: on F2 ``log(2c)*sin(d/3)`` the MI-winner
# ``sub(exp(c),cbrt(d))`` (MI 0.288) helps a LINEAR downstream by ~0 (MAE 0.092->0.093)
# while the lower-MI ``mul(log(c),sin(d))`` (MI 0.264, INSIDE the 0.85 leaders band) is
# the linearly-aligned additive term and cuts MAE 0.092->0.063. Emitting only one loses
# whichever the actual model needs. With ``fe_multi_emit_max_per_pair > 1`` the search
# additionally emits the next DISTINCT forms (greedy by target MI, skipping any whose
# continuous values correlate above ``fe_multi_emit_diversity_corr`` with an
# already-emitted column, down to ``fe_multi_emit_mi_floor`` x best_mi) - a tree-friendly
# AND a linear-friendly form both survive, and the downstream MRMR redundancy gate prunes
# any residual overlap. The cap + the diversity filter keep it from flooding near-duplicates.
# [fe_pair_usability_admission_enable]
# TAIL-CONCENTRATION USABILITY ADMISSION. When a raw pair's true signal is tail-concentrated
# (rank-MI collapses in the clean bulk under heavy outliers) the rank-MI winner-selection + engineered-MI
# prevalence gate both mis-rank / drop the genuine linearly-usable form. When enabled, a pair the detector
# flags (best raw bivariate-form |corr(continuous y)| clears ``min_corr`` AND beats the best single-operand
# form by ``pairness_margin``) promotes its |corr|-best engineered form as the winner and is admitted past
# the engineered-MI gate. Default ON; the detector is FALSE for canonical fixtures + non-tail data, so
# those are byte-identical. Threaded from the MRMR ``fe_pair_usability_admission_*`` knobs.
# [concurrent_workers]
# LARGE-N PEAK-MEMORY FIX. Number of ``check_prospective_fe_pairs`` calls
# that may run CONCURRENTLY in this process. On the serial-main-thread path this is 1; on
# the joblib ``backend="threading"`` path it is ``n_jobs`` (each thread allocates its OWN
# candidate / chunk / disc / MI buffers in the SHARED address space, so the per-call RAM
# budget must be divided by the worker count or N threads collectively OOM). Used only to
# SIZE the candidate buffers (chunk width + shared-buffer fit check) - never changes which
# candidates are produced or their MI, so selection stays byte-identical.
# [fe_mm_debias_prevalence]
# MILLER-MADOW DEBIAS of the joint-prevalence RATIO gate (+ #4).
# The gate ``best_mi / pair_mi > fe_min_engineered_mi_prevalence`` compares a 1-D
# engineered MI (over ~``quantization_nbins`` bins) against a 2-D joint MI (over
# ~``nbins^2`` bins). Both are RAW plug-in MIs whose positive finite-sample bias is
# ``(k_x-1)(k_y-1)/2n``; the JOINT denominator's term is ~``nbins``x larger, so the raw
# ratio is structurally depressed below 1.0 at small/moderate n. When True we subtract the
# Miller-Madow MI bias term (OCCUPIED bin counts, #4) from BOTH sides before the ratio,
# with a denominator-positivity guard that falls back to the raw ratio when the joint bias
# swamps the finite-sample joint MI; the order-2 maxT floor is MM-debiased CONSISTENTLY
# upstream (IRON RULE). bench-rejected as DEFAULT (2026-06-09): the isolated ratio fix is
# real but it adds 0 end-to-end recovery on clean synergy (the marginal-uplift fallback
# already recovers it) and ADMITS cross-mix noise on the weak F2 (cross-mix 3/10 -> 9/10
# seeds, genuine_ab 10/10 -> 8/10). Default False (opt-in); see the full numbers + rationale
# on ``MRMR.fe_mm_debias_prevalence``. False == legacy raw-plug-in ratio (byte-reproduction).
# [rejection_ledger_out]
# REJECTION-LEDGER out-param (additive). When a list is passed, every pair
# the per-pair acceptance gate REJECTS (joint-prevalence floor AND the marginal-uplift /
# joint-recovery fallback both declined) appends one record dict carrying the operands,
# the winning binary operator, the observed best_mi/pair_mi ratio, the prevalence
# threshold, and the marginal-uplift diagnostics - all values the gate ALREADY computed,
# no recompute. The caller (``_mrmr_fe_step``) drains it into ``self``'s fe rejection
# ledger. ``None`` (default) = legacy behaviour, no records captured.
def check_prospective_fe_pairs(
    prospective_pairs: dict,
    X: Any,
    unary_transformations: dict,
    binary_transformations: dict,
    classes_y: np.ndarray,
    classes_y_safe: np.ndarray,
    freqs_y: np.ndarray,
    num_fs_steps: int,
    cols: list[str],
    original_cols: list[str],
    fe_max_steps: int,
    fe_npermutations: int,
    fe_max_pair_features: int,
    fe_print_best_mis_only: bool,
    fe_min_nonzero_confidence: float,
    fe_min_engineered_mi_prevalence: float,
    fe_good_to_best_feature_mi_threshold: float,
    fe_max_external_validation_factors: int,
    numeric_vars_to_consider: list,
    quantization_nbins: int,
    quantization_method: str | None,
    quantization_dtype: type | str,
    times_spent: dict,
    verbose: int,
    subsample_n: int = UNIFIED_FE_SUBSAMPLE_N,
    subsample_seed: int = 42,
    shared_subsample_idx: np.ndarray | None = None,
    fe_subsample_stratify: bool = False,
    prewarp_enable: bool = False,
    prewarp_y: np.ndarray | None = None,
    prewarp_y_continuous: np.ndarray | None = None,
    usability_y_continuous: np.ndarray | None = None,
    prewarp_basis: str = "chebyshev",
    prewarp_max_degree: int = 4,
    prewarp_uplift_threshold: float = 1.20,
    prewarp_min_val_corr: float = 0.08,
    prewarp_specs_out: dict | None = None,
    fe_gate_med_enable: bool = False,
    gate_med_specs_out: dict | None = None,
    serial_main_thread: bool = False,
    allow_engineered_operands: bool = True,
    engineered_operand_values: dict | None = None,
    fe_multi_emit_max_per_pair: int = 1,
    fe_multi_emit_mi_floor: float = 0.5,
    fe_multi_emit_diversity_corr: float = 0.90,
    fe_pair_usability_admission_enable: bool = True,
    fe_pair_usability_admission_min_corr: float = 0.6,
    fe_pair_usability_admission_pairness_margin: float = 1.05,
    concurrent_workers: int = 1,
    fe_mm_debias_prevalence: bool = False,
    rejection_ledger_out: list | None = None,
) -> dict:
    """Search ``prospective_pairs`` for engineered unary/binary transforms whose MI against ``classes_y`` clears the
    joint-prevalence and external-validation gates, and materialise the surviving engineered columns.

    This is the core FE pair-search kernel: for each candidate pair it builds the unary-transform pool of both
    operands (optionally extended with a per-operand pre-warp and/or median gate), evaluates every configured
    binary operator on top, scores each resulting form's MI against the (optionally subsampled, optionally
    stratified) target, and admits the best form(s) per pair once they beat the raw pair's joint MI by
    ``fe_min_engineered_mi_prevalence`` (or pass one of the marginal-uplift / tail-concentration fallback
    admission paths). Engineered-operand feed-forward lets step-k>1 searches compose earlier engineered columns
    into higher-order composites. Returns the accepted engineered features plus their recipes for later
    leak-safe replay; declined pairs are recorded to ``rejection_ledger_out`` when provided.
    """
    st, pair_pbar, pair_mi, _can_hoist_shared_buffer, _rebuild_full_survivor_col, discretize_array, discretize_2d_quantile_batch, get_new_feature_name, gpu_compatible_unary_names, logger, mi_direct, batch_mi_with_noise_gate, use_su_normalization, X, _full_n_rows, _use_subsample, _sample_idx, classes_y, classes_y_safe, freqs_y, _extval_raw_col_cache = _check_prospective_fe_pairs_part3(
        X, subsample_seed, shared_subsample_idx, subsample_n, fe_subsample_stratify, classes_y, classes_y_safe, freqs_y, verbose,
    )

    _extval_raw_col, transformed_vars, vars_transformations, _operand_marginal_mi_cache, _corr_y_cont, _corr_y_cont_finite, _safe_abs_corr = _check_prospective_fe_pairs_part10(
        _extval_raw_col_cache, original_cols, X, allow_engineered_operands, cols, engineered_operand_values, _use_subsample, _full_n_rows, _sample_idx,
        prospective_pairs, prewarp_enable, prewarp_y, prewarp_y_continuous, prewarp_basis, prewarp_max_degree, prewarp_min_val_corr, fe_gate_med_enable, st,
        unary_transformations, verbose, logger, binary_transformations, concurrent_workers, gpu_compatible_unary_names, usability_y_continuous, classes_y,
    )

    _raw_operand_abs_corr, _transformed_operand_abs_corr, _operand_marginal_mi, _operand_discretized = _check_prospective_fe_pairs_part7(
        _extval_raw_col, _safe_abs_corr, vars_transformations, transformed_vars, st, _operand_marginal_mi_cache, quantization_nbins, quantization_method,
        quantization_dtype, classes_y, classes_y_safe, freqs_y, fe_min_nonzero_confidence, fe_npermutations, prospective_pairs, X, verbose,
    )

    # The executor and the double chunk buffer are torn down in a `finally`. They used to be created here
    # and shut down ~200 lines later at the end of the sweep, with nothing covering the span: any exception
    # in the pair loop -- a cupy OutOfMemoryError, a kernel launch failure, a KeyboardInterrupt -- skipped the
    # shutdown. ThreadPoolExecutor registers its worker with threading._register_atexit, so the thread
    # survived to interpreter exit, and a still-pending future kept the shared buffers alive with it: at a
    # 2M-row chunk over 40 operands that is 640 MB per buffer, 1.28 GB for the pair, retained for the rest
    # of the process.
    try:

        _sweep_best_mi, _gpu_mat_on, _op_code_arr_all, _op_code_arr, _mi_band, _fe_env_gate, _pair_family_alpha = _check_prospective_fe_pairs_part5(
            binary_transformations, quantization_nbins, classes_y, freqs_y, prospective_pairs,
        )
        for (
            raw_vars_pair,
            pair_mi,
        ), _uplift in (
            pair_pbar := tqdmu(prospective_pairs.items(), desc="pair", leave=False, disable=not verbose)
        ):  # better to start considering form the most prospective pairs with highest mis ratio!

            _pair_res_entry, best_config, best_mi = _score_one_pair(
                raw_vars_pair=raw_vars_pair, pair_mi=pair_mi, pair_family_alpha=_pair_family_alpha, chunk_state=st._chunk_state,
                rejection_records=st._rejection_records, rejection_ledger_out=rejection_ledger_out, X=X, transformed_vars=transformed_vars,
                vars_transformations=vars_transformations, binary_transformations=binary_transformations, unary_transformations=unary_transformations,
                pair_combs=st.pair_combs, final_transformed_vals_shared=st.final_transformed_vals_shared, _need_recompute_map=st._need_recompute_map,
                _chunk_global_batch=st._chunk_global_batch, _chunk_buffer=st._chunk_buffer, _pair_to_chunk=st._pair_to_chunk, _fe_chunks=st._fe_chunks,
                _pair_valid_combs=st._pair_valid_combs, _fe_defer_float=st._fe_defer_float, _gpu_mat_on=_gpu_mat_on, _op_code_arr=_op_code_arr,
                _op_code_arr_all=_op_code_arr_all, _mi_band=_mi_band, _fe_env_gate=_fe_env_gate, classes_y=classes_y, classes_y_safe=classes_y_safe,
                freqs_y=freqs_y, fe_npermutations=fe_npermutations, fe_min_nonzero_confidence=fe_min_nonzero_confidence,
                quantization_nbins=quantization_nbins, quantization_method=quantization_method, quantization_dtype=quantization_dtype,
                num_fs_steps=num_fs_steps, fe_min_engineered_mi_prevalence=fe_min_engineered_mi_prevalence,
                fe_good_to_best_feature_mi_threshold=fe_good_to_best_feature_mi_threshold,
                fe_max_external_validation_factors=fe_max_external_validation_factors, numeric_vars_to_consider=numeric_vars_to_consider,
                fe_max_steps=fe_max_steps, fe_print_best_mis_only=fe_print_best_mis_only, fe_mm_debias_prevalence=fe_mm_debias_prevalence,
                _prewarp_active=st._prewarp_active, prewarp_uplift_threshold=prewarp_uplift_threshold, _PREWARP_UNARY=_PREWARP_UNARY,
                _corr_y_cont=_corr_y_cont, _corr_y_cont_finite=_corr_y_cont_finite, _NOISE_WRAP_CORR_COLLAPSE_FRAC=st._NOISE_WRAP_CORR_COLLAPSE_FRAC,
                _NOISE_WRAP_MIN_OPERAND_CORR=st._NOISE_WRAP_MIN_OPERAND_CORR, fe_multi_emit_max_per_pair=fe_multi_emit_max_per_pair,
                fe_multi_emit_mi_floor=fe_multi_emit_mi_floor, fe_multi_emit_diversity_corr=fe_multi_emit_diversity_corr,
                fe_pair_usability_admission_enable=fe_pair_usability_admission_enable,
                fe_pair_usability_admission_min_corr=fe_pair_usability_admission_min_corr,
                fe_pair_usability_admission_pairness_margin=fe_pair_usability_admission_pairness_margin, cols=cols, original_cols=original_cols,
                _use_subsample=_use_subsample, _X_full=st._X_full, _full_n_rows=_full_n_rows, _prewarp_spec_by_var=st._prewarp_spec_by_var,
                _gate_med_median_by_var=st._gate_med_median_by_var, engineered_operand_values=engineered_operand_values, _rng_extval=st._rng_extval,
                _n_workers=st._n_workers, times_spent=times_spent, verbose=verbose, serial_main_thread=serial_main_thread, _extval_raw_col=_extval_raw_col,
                _safe_abs_corr=_safe_abs_corr, _raw_operand_abs_corr=_raw_operand_abs_corr, _transformed_operand_abs_corr=_transformed_operand_abs_corr,
                _operand_marginal_mi=_operand_marginal_mi, _operand_discretized=_operand_discretized, batch_mi_with_noise_gate=batch_mi_with_noise_gate,
                use_su_normalization=use_su_normalization, discretize_array=discretize_array, discretize_2d_quantile_batch=discretize_2d_quantile_batch,
                mi_direct=mi_direct, get_new_feature_name=get_new_feature_name, _rebuild_full_survivor_col=_rebuild_full_survivor_col,
                _can_hoist_shared_buffer=_can_hoist_shared_buffer, _fe_gpu_discretize_enabled=_fe_gpu_discretize_enabled,
            )
            _check_prospective_fe_pairs_part9(
                _pair_res_entry, st, raw_vars_pair, best_config, cols, pair_mi, rejection_ledger_out, verbose, best_mi, _sweep_best_mi, pair_pbar,
            )

        _check_prospective_fe_pairs_part6(st, prospective_pairs, prewarp_specs_out, gate_med_specs_out)

        # Tear down the chunk-pipeline executor (all futures resolved by the pair loop; an unconsumed prefetch
        # - e.g. an early exit - is awaited by shutdown so the worker never outlives the shared buffers).
    finally:
        _check_prospective_fe_pairs_part8(st)

    return cast(dict, st.res)
