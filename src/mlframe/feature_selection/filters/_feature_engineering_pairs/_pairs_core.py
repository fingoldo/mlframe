"""``check_prospective_fe_pairs`` -- the FE pair-search core (candidate generation
via unary+binary ops, batched MI + permutation noise-gate, prewarp / median-gate
pseudo-unaries, chunked materialise, kernel-tuning dispatch).

This is the irreducible single-function body of the ``_feature_engineering_pairs``
subpackage; the supporting kernels / gates / dispatch live in sibling submodules
and are re-exported from the package ``__init__``.
"""
from __future__ import annotations

from itertools import combinations
from timeit import default_timer as timer

import numpy as np
import pandas as pd
from pandas.api.extensions import ExtensionDtype
from numpy.polynomial.hermite import hermval

from pyutilz.pythonlib import sort_dict_by_value
from pyutilz.system import tqdmu

from ._pairs_chunks import (
    _FE_CHUNK_MAX_COLS_HARD_CAP,
    _compute_one_fe_chunk,
    _plan_fe_chunks,
)
from ._pairs_common import _TIMES_SPENT_LOCK
from ._pairs_dispatch import _dispatch_batch_mi_with_noise_gate
from ._pairs_gates import (
    _FE_MARGINAL_UPLIFT_MIN_JOINT_RATIO,
    _FE_MARGINAL_UPLIFT_MIN_RATIO,
    _FE_MARGINAL_UPLIFT_STRICT_JOINT_RATIO,
    _FE_MARGINAL_UPLIFT_SYNERGY_UPLIFT,
    _FE_REJECTION_RESULT_KEY,
    _GATE_MED_SPECS_RESULT_KEY,
    _GATE_MED_UNARY,
    _PREWARP_SPECS_RESULT_KEY,
    _PREWARP_UNARY,
    _gate_med_apply,
    _select_single_best,
)
from ._pairs_materialise import (
    _fe_use_parallel_kernels,
    _materialise_extval_njit,
    _narrow_code_dtype,
    _njit_binary_op_codes,
)

def _short_fe_name(name, maxlen: int = 30) -> str:
    """Truncate a (possibly long engineered) feature expression for live progress-bar
    display, keeping head + tail so both operator and operand stay legible
    (``mul(log(c),sin(d))`` -> ``mul(log(c..sin(d))``). Robust to non-str input."""
    try:
        s = str(name)
    except Exception:
        return "?"
    if len(s) <= maxlen:
        return s
    head = (maxlen - 2) // 2
    tail = maxlen - 2 - head
    return s[:head] + ".." + s[-tail:]


# Shared subsample default across the two FE entry points. ``polynom_pair_fe``
# already uses 200_000 (validated 2026-05-18: 100k could lose a marginal hermite
# feature, 200k kept it). The accuracy bench for ``check_prospective_fe_pairs``
# at this n landed at jaccard=1.0 vs full -- see
# bench_fe_pair_subsample_accuracy.py. Keep both call sites pinned to ONE knob
# so a future re-tune lands consistently across the FE block.
FE_DEFAULT_SUBSAMPLE_N: int = 200_000


def check_prospective_fe_pairs(
    prospective_pairs,
    X,
    unary_transformations,
    binary_transformations,
    classes_y,
    classes_y_safe,
    freqs_y,
    num_fs_steps,
    cols,
    original_cols,
    fe_max_steps,
    fe_npermutations,
    fe_max_pair_features,
    fe_print_best_mis_only,
    fe_min_nonzero_confidence,
    fe_min_engineered_mi_prevalence,
    fe_good_to_best_feature_mi_threshold,
    fe_max_external_validation_factors,
    numeric_vars_to_consider,
    quantization_nbins,
    quantization_method,
    quantization_dtype,
    times_spent,
    verbose,
    # CRITICAL #2 follow-up (2026-05-21): subsample rows for the MI sweep so the
    # transformed_vars + shared_buffer allocations scale with subsample_n rather
    # than the (possibly multi-million-row) full X. Survivor columns are still
    # produced at full n via _rebuild_full_survivor_col so the caller contract
    # is preserved. Default 200_000 matches polynom_pair_fe's existing knob
    # (FE_DEFAULT_SUBSAMPLE_N); the standalone accuracy bench in
    # bench_fe_pair_subsample_accuracy.py shows jaccard=1.0 vs full at n_eff
    # >= 50k on synthetic 3-pair-competition data. 0 = use full data (legacy).
    subsample_n: int = FE_DEFAULT_SUBSAMPLE_N,
    subsample_seed: int = 42,
    # PER-OPERAND PRE-WARP (2026-06-02). When ``prewarp_enable`` is True the
    # unary/binary search gains, per raw operand, one extra "pseudo-unary"
    # ``prewarp(x)`` -- a learned 1-D orthogonal-polynomial warp fit JOINTLY
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
    prewarp_enable: bool = False,
    prewarp_y: np.ndarray | None = None,
    # PREWARP ALS RECONSTRUCTION TARGET (2026-06-11). The rank-1 ALS warp fit /
    # held-out validation / winning-spec reconstruction is a least-squares solve
    # of ``y ~ f(a)*g(b)``; its fidelity depends on the RESOLUTION of the target it
    # reconstructs. The 2026-06-10 target-rebin guard coarsens ``classes_y`` to the
    # 10-bin equal-frequency screening codes (correctly -- the MI screen/gates need
    # the faithful coarse codes), but feeding those binned codes to the ALS dropped
    # the F-POLY non-monotone product reconstruction |corr| 0.97 -> 0.88. When the
    # CONTINUOUS target is threaded here it drives the ALS fit/validate/score ONLY;
    # the MI screen + every gate keep using ``classes_y`` codes. None -> legacy
    # behaviour (ALS reconstructs against ``classes_y``).
    prewarp_y_continuous: np.ndarray | None = None,
    # LINEAR-USABILITY GUARD TARGET (2026-06-17). The leader-equivalence tie-break and the
    # noise-wrap |corr| guard must score against the CONTINUOUS regression target, NOT the
    # binned ``classes_y`` codes: on a heavy-tailed target the Pearson |corr| of a magnitude-
    # carrying form (e.g. ``a**2/b``) with the quantile-RANK codes COLLAPSES (~0.05) while a
    # bounded monotone warp (e.g. ``a/sqrt(b)``) scores higher (~0.4) -- the EXACT INVERSE of
    # linear usability. The prewarp path already threaded continuous y, so the exhaustive
    # (prewarp-on) path was correct; the fast (prewarp-off) path fell back to ``classes_y`` and
    # picked the linearly-useless leg (biz_value_mrmr_fast_search MAE 0.05 -> 37). Thread the
    # continuous y here UNCONDITIONALLY (independent of prewarp) so both paths agree. None ->
    # legacy ``classes_y`` fallback (correct for classification / non-numeric y).
    usability_y_continuous: np.ndarray | None = None,
    prewarp_basis: str = "chebyshev",
    prewarp_max_degree: int = 4,
    # Minimum ratio (best-prewarp-MI / best-nonprewarp-MI) for the alternative
    # acceptance path. 1.20 = the prewarp must beat the elementary library by
    # >= 20% engineered MI to be admitted past the joint-prevalence gate. Tuned
    # to fire on the F-POLY non-monotone inner (measured uplift ~1.42x) while
    # staying silent on linear/monotone/noise (uplift ~1.0x there).
    prewarp_uplift_threshold: float = 1.20,
    # Held-out floor for the out-of-sample prewarp validation: a warp fit on a
    # train slice is kept only if its rank-1 reconstruction f(a)*g(b) tracks y on
    # a held-out slice with |corr| >= this. Rejects supervised overfitting on noise
    # operands at small n; 0.0 disables (legacy in-sample-only fit).
    prewarp_min_val_corr: float = 0.08,
    prewarp_specs_out: dict | None = None,
    # PER-OPERAND MEDIAN GATE (2026-06-04). When ``fe_gate_med_enable`` is True the
    # unary/binary search gains, per raw operand, one extra "pseudo-unary"
    # ``gate_med(x) = (x > train_median_x).astype(float)``. Combined with the
    # existing ``mul`` binary this lets the elementary path represent the
    # median-gated operators ``(a > median_a) * b`` (``mul(gate_med(a), b)``) and
    # the conjunction ``(a > median_a) & (b > median_b)``
    # (``mul(gate_med(a), gate_med(b))``) that a bilinear product cannot -- the
    # signal is non-product / conditional. Unlike a fixed threshold-0 gate, the
    # median ADAPTS the split to each operand's distribution, so it recovers the
    # gate on shifted / skewed operands where threshold-0 is useless (measured
    # skew-bench: gated_med +0.0355 / thr_and_med +0.0435 downstream-AUC d_mean
    # vs raw, beating products +0.022/+0.020 and threshold-0 +0.009/+0.0001).
    # The fitted state is ONE float per operand (the TRAIN median), stored in the
    # recipe by ``_mrmr_fe_step`` for leak-safe closed-form replay (no y, no
    # test-time recompute). The median does not overfit, so -- unlike prewarp --
    # NO held-out validation is needed; the gate is still subject to the same
    # MI-prevalence / external-validation acceptance gates every engineered
    # feature passes (it competes on equal footing in the per-pair MI sweep and
    # wins via ``best_mi`` only where the conditional form genuinely beats the
    # library). Default OFF so behaviour is byte-identical when not requested.
    fe_gate_med_enable: bool = False,
    gate_med_specs_out: dict | None = None,
    # OPT-A (2026-06-07): True when the caller (``_mrmr_fe_step``) dispatches this on the
    # SERIAL MAIN THREAD with NO joblib threading nest (the ``len(X) < 50000`` /
    # ``len(prospective_pairs) < 2`` branch). On that path the FE materialise / searchsorted
    # kernels may use their ``parallel=True`` column-prange twins (a numba prange is safe --
    # nothing nests it). On the joblib ``backend="threading"`` path this stays False so the
    # serial ``nogil`` kernels are used (a nested prange deadlocks the threading layer).
    # Byte-identical either way -- only thread-count of the embarrassingly-parallel per-column
    # work changes. Default False = the always-safe serial path.
    serial_main_thread: bool = False,
    # ENGINEERED-OPERAND FEED-FORWARD (2026-06-08). When True (default), an operand var
    # that is NOT a raw ``feature_names_in_`` column (i.e. it is an engineered column
    # appended by a prior FE step, hence absent from ``original_cols``) is resolved by
    # NAME from the augmented frame ``X`` rather than skipped. This lets the step-k>1
    # pair search build COMPOSITES of two engineered features -- e.g. the additive
    # ``add(div(sqr(a),abs(b)), mul(log(c),sin(d)))`` that captures ~the entire
    # deterministic signal of ``y = a**2/b + log(c)*sin(d)``. The number of engineered
    # operands fed back is capped UPSTREAM (``fe_max_engineered_operands`` in
    # ``_mrmr_fe_step``) so the O(k^2) pair count stays bounded. Set False to restore
    # the legacy raw-only operand pool (no engineered x engineered composites).
    allow_engineered_operands: bool = True,
    # ENGINEERED-OPERAND CONTINUOUS VALUES (2026-06-08). ``{engineered_col_name ->
    # full-n float64 ndarray}`` of the CONTINUOUS engineered values produced by prior
    # FE steps. ``_extval_raw_col`` reads this for engineered operands so the pair
    # search combines the CONTINUOUS values rather than the augmented frame's
    # DISCRETISED bin codes (combining codes is severely lossy -- it sinks the
    # additive composite below the engineered-MI gate). ``None`` (default) -> fall
    # back to the by-name frame extract (bin codes), preserving the legacy behaviour.
    engineered_operand_values: dict | None = None,
    # MULTI-CANDIDATE DIVERSE EMISSION (2026-06-12). Per raw pair, the search picks the
    # single MAX-target-MI engineered form and discards every other. But MI is a RANK
    # statistic blind to LINEAR usability: on F2 ``log(2c)*sin(d/3)`` the MI-winner
    # ``sub(exp(c),cbrt(d))`` (MI 0.288) helps a LINEAR downstream by ~0 (MAE 0.092->0.093)
    # while the lower-MI ``mul(log(c),sin(d))`` (MI 0.264, INSIDE the 0.85 leaders band) is
    # the linearly-aligned additive term and cuts MAE 0.092->0.063. Emitting only one loses
    # whichever the actual model needs. With ``fe_multi_emit_max_per_pair > 1`` the search
    # additionally emits the next DISTINCT forms (greedy by target MI, skipping any whose
    # continuous values correlate above ``fe_multi_emit_diversity_corr`` with an
    # already-emitted column, down to ``fe_multi_emit_mi_floor`` x best_mi) -- a tree-friendly
    # AND a linear-friendly form both survive, and the downstream MRMR redundancy gate prunes
    # any residual overlap. The cap + the diversity filter keep it from flooding near-duplicates.
    fe_multi_emit_max_per_pair: int = 1,
    fe_multi_emit_mi_floor: float = 0.5,
    fe_multi_emit_diversity_corr: float = 0.90,
    # LARGE-N PEAK-MEMORY FIX (2026-06-08). Number of ``check_prospective_fe_pairs`` calls
    # that may run CONCURRENTLY in this process. On the serial-main-thread path this is 1; on
    # the joblib ``backend="threading"`` path it is ``n_jobs`` (each thread allocates its OWN
    # candidate / chunk / disc / MI buffers in the SHARED address space, so the per-call RAM
    # budget must be divided by the worker count or N threads collectively OOM). Used only to
    # SIZE the candidate buffers (chunk width + shared-buffer fit check) -- never changes which
    # candidates are produced or their MI, so selection stays byte-identical.
    concurrent_workers: int = 1,
    # MILLER-MADOW DEBIAS of the joint-prevalence RATIO gate (2026-06-09, backlog #1 + #4).
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
    fe_mm_debias_prevalence: bool = False,
    # REJECTION-LEDGER out-param (additive, 2026-06-11). When a list is passed, every pair
    # the per-pair acceptance gate REJECTS (joint-prevalence floor AND the marginal-uplift /
    # joint-recovery fallback both declined) appends one record dict carrying the operands,
    # the winning binary operator, the observed best_mi/pair_mi ratio, the prevalence
    # threshold, and the marginal-uplift diagnostics -- all values the gate ALREADY computed,
    # no recompute. The caller (``_mrmr_fe_step``) drains it into ``self``'s fe rejection
    # ledger. ``None`` (default) = legacy behaviour, no records captured.
    rejection_ledger_out: list | None = None,
):
    # Starting from the most heavily connected pairs, create a big pool of original features + their unary transforms. Individual vars referenced more than once go
    # to the global pool, the rest to the local (not stored)?

    # Lazy import of parent-resident helpers: ``.predict`` re-imports
    # this sibling at its bottom, so a top-level ``from .predict
    # import ...`` would create a hard cycle the meta-test flags.
    from ..feature_engineering import FE_DEFAULT_SUBSAMPLE_N, _FE_BUFFER_RAM_BUDGET_RATIO, _can_hoist_shared_buffer, _estimate_fe_shared_buffer_bytes, _fe_effective_buffer_budget_bytes, _rebuild_full_survivor_col, discretize_array, discretize_2d_quantile_batch, get_new_feature_name, gpu_compatible_unary_names, logger, mi_direct
    # 2026-06-05: batched FE-candidate MI + permutation noise-gate (bit-identical to the
    # per-candidate mi_direct on the default outer/n_workers=1 path -- see kernel docstring).
    from ..info_theory import batch_mi_with_noise_gate, use_su_normalization
    res = {}
    # REJECTION-LEDGER local accumulator (additive, 2026-06-11): per-pair acceptance-gate
    # drops collect here, then are exported via BOTH the ``rejection_ledger_out`` side channel
    # (serial / threading path) AND the reserved ``res`` key (survives the loky-parallel path,
    # where the caller's list cannot be mutated cross-process). Records carry only values the
    # gate already computed -- no recompute.
    _rejection_records: list = []

    # Seeded RNG for the external-validation factor subsample below. Pre-fix code used the
    # process-global ``np.random.choice`` there, which (a) made the choice depend on whatever
    # had consumed the global numpy RNG earlier in the process (so two fits of the SAME (X, y)
    # in one session could pick DIFFERENT validation factors -> a different tie-break -> a
    # different engineered-recipe SET / column NAMES), and (b) raced under the joblib
    # ``backend="threading"`` chunked path (N workers sharing one global RNG). Derive a local
    # Generator from ``subsample_seed`` (instance-controlled) so the factor pick is reproducible
    # from the MRMR seed and thread-safe, mirroring the seeded ``_rng_sub`` and the fleuret LCG fix.
    _rng_extval = np.random.default_rng(int(subsample_seed))

    # SUBSAMPLE-SETUP: when caller asks for subsample_n > 0 AND len(X) exceeds it,
    # build subsampled views of X / classes_y / classes_y_safe / freqs_y. The MI
    # sweep operates on these views; survivor packing always rebuilds at full n.
    # When subsample_n is 0 / negative / >= len(X) the legacy full-data path runs
    # unchanged (everything below uses ``X`` / ``classes_y`` / ... directly).
    _X_full = X
    _full_n_rows = len(_X_full)
    _use_subsample = isinstance(subsample_n, int) and 0 < subsample_n < _full_n_rows
    if _use_subsample:
        _rng_sub = np.random.default_rng(int(subsample_seed))
        _sample_idx = np.sort(_rng_sub.choice(_full_n_rows, size=int(subsample_n), replace=False))
        if isinstance(_X_full, pd.DataFrame):
            X = _X_full.iloc[_sample_idx].reset_index(drop=True)
        else:
            # Polars path -- row indexing returns a fresh frame; preserves zero-copy where possible.
            X = _X_full[_sample_idx]
        # Realign per-row target encodings; recompute freqs from the subsampled
        # class labels so MI estimates use the actual subsample distribution
        # rather than the full-n freq table (which would bias the MI estimator
        # toward classes that shrank under the random subset).
        _cy = np.asarray(classes_y)
        _cy_safe = np.asarray(classes_y_safe)
        classes_y = _cy[_sample_idx]
        classes_y_safe = _cy_safe[_sample_idx]
        # Recompute freqs from subsampled class labels. merge_vars returns
        # freqs_y as a FLOAT proportions array (sum=1.0), not raw counts; the
        # subsample needs the same shape. bincount gives counts -> divide by
        # total to get proportions matching the caller's expectation.
        if classes_y.size > 0 and classes_y.dtype.kind in ("i", "u"):
            _counts = np.bincount(classes_y.astype(np.int64))
            _total = _counts.sum()
            if _total > 0:
                freqs_y = (_counts.astype(np.float64) / float(_total))
        # else: leave the caller-supplied freqs_y; mi_direct handles its own
        # validation and would crash anyway on a non-integer class table.
        if verbose:
            logger.info(
                "check_prospective_fe_pairs: subsample_n=%d active (full_n=%d, %.1f%% sample); "
                "MI sweep runs on the subsample, survivor columns rebuilt at full n.",
                int(subsample_n), _full_n_rows, 100.0 * subsample_n / _full_n_rows,
            )

    # EXTERNAL-VALIDATION raw-column EXTRACTION MEMO (2026-06-07, LEVER 1).
    # The lazy external-validation tie-break (below) extracts the RAW values of
    # every external factor via ``X.iloc[:, original_cols[ext]].values`` (or the
    # polars ``.to_numpy()``) once per tied-leader config. The external-factor set
    # is ``numeric_vars_to_consider`` minus the 2 pair operands, so the SAME raw
    # column is re-extracted across every config AND every raw pair: on the wide
    # scene bed (2407x299) that is ~276k pandas ``.iloc`` calls / ~43s of pure
    # pandas-indexing (call-site cProfile). The extraction is DETERMINISTIC -- a
    # var-index maps to a FIXED column-values ndarray for the lifetime of this
    # call (``X`` is fixed after the subsample reassignment above) -- so memoising
    # it by the var key (the ``external_factor`` index into ``original_cols``)
    # yields BYTE-IDENTICAL values while collapsing the per-config re-extraction to
    # one extraction per distinct external factor. Keyed by the var id only, never
    # by array contents, so it cannot collide across distinct columns. The cache is
    # a plain local dict scoped to this call -> never pickled (no __getstate__
    # concern; the MRMR estimator never holds a reference to it).
    _extval_raw_col_cache: dict = {}

    def _densify_nullable(_arr):
        """Cast a pandas nullable extension array (Int64/Float64/boolean + pd.NA) to a plain
        float64 ndarray (pd.NA -> np.nan) so the numba/numpy unary-transform kernels can type it.
        No-op for ordinary numpy float64/int64 input -- only ExtensionArrays are converted."""
        if isinstance(getattr(_arr, "dtype", None), ExtensionDtype):
            return _arr.to_numpy(dtype=np.float64, na_value=np.nan)
        return _arr

    def _extval_raw_col(_var):
        """Memoised operand-values ndarray for var ``_var`` (cols-space index).

        For a RAW operand (``_var in original_cols``) returns ``X``'s column at the
        ``original_cols[_var]`` position (the RAW position into ``feature_names_in_``),
        bit-identical to the legacy ``X.iloc[...].values`` / ``.to_numpy()`` extract.

        ENGINEERED-OPERAND FEED-FORWARD (2026-06-08): at FE step k>1 the operand pool
        also carries the engineered columns appended by the prior step(s)
        (``selected_vars`` includes their cols-space indices, so the pair-MI sweep
        surfaces ``(eng_i, eng_j)`` pairs -- e.g. the additive composite of the two
        real step-1 features that captures ~the entire deterministic signal). Those
        columns are NOT in ``original_cols`` (which holds raw ``feature_names_in_``
        positions only), but they ARE present in the AUGMENTED frame ``X`` under their
        ``cols[_var]`` name (``_mrmr_fe_step`` appends each engineered column to BOTH
        ``cols`` and ``X`` in lockstep). When ``allow_engineered_operands`` is on we
        fetch them by NAME so ``(eng_i, eng_j)`` can produce a real composite candidate.
        Returns ``None`` only when the var is neither a raw position nor a resolvable
        augmented-frame column (the caller then skips it, exactly as before)."""
        if _var in _extval_raw_col_cache:
            return _extval_raw_col_cache[_var]
        if _var in original_cols:
            if isinstance(X, pd.DataFrame):
                _vals = _densify_nullable(X.iloc[:, original_cols[_var]].values)
            else:
                _vals = X[:, original_cols[_var]].to_numpy()
            _extval_raw_col_cache[_var] = _vals
            return _vals
        # Engineered operand: resolve by name. PREFER the CONTINUOUS engineered values
        # (``engineered_operand_values[name]``) over the augmented frame's column, which
        # holds the DISCRETISED bin codes -- combining bin codes (e.g. ``add(codes_a,
        # codes_b)``) is severely lossy and sinks the composite below the engineered-MI
        # gate (measured: 0.88 from codes vs 1.81 -- the full signal -- from continuous
        # values). Fall back to the by-name frame extract when no continuous value is
        # stored (e.g. an engineered column produced by a stage that did not register one).
        if allow_engineered_operands and 0 <= _var < len(cols):
            _name = cols[_var]
            _vals = None
            if engineered_operand_values is not None:
                _cv = engineered_operand_values.get(_name)
                if _cv is not None:
                    _cv = np.asarray(_cv)
                    # The continuous store is full-n; align to the (possibly subsampled) X.
                    if _cv.shape[0] == len(X):
                        _vals = _cv
                    elif _use_subsample and _cv.shape[0] == _full_n_rows:
                        _vals = _cv[_sample_idx]
            if _vals is None:
                try:
                    if isinstance(X, pd.DataFrame):
                        _vals = _densify_nullable(X[_name]) if isinstance(X[_name].dtype, ExtensionDtype) else (
                            X[_name].to_numpy() if hasattr(X[_name], "to_numpy") else X[_name].values
                        )
                    elif hasattr(X, "columns") and _name in getattr(X, "columns", []):
                        _vals = X[_name].to_numpy()  # polars
                    else:
                        _vals = None
                except Exception:
                    _vals = None
            if _vals is not None:
                _vals = np.asarray(_vals)
                _extval_raw_col_cache[_var] = _vals
                return _vals
        _extval_raw_col_cache[_var] = None
        return None

    # PER-OPERAND PRE-WARP setup (2026-06-02). When enabled, fit ONE learned
    # 1-D pre-warp per raw operand against the (subsample-aligned) target, and
    # expose it as an extra pseudo-unary named ``_PREWARP_UNARY`` so the existing
    # unary x unary x binary search naturally considers ``binary(prewarp(a),
    # prewarp(b))``, ``binary(prewarp(a), b)`` etc. The fitted spec per var is
    # kept in ``_prewarp_spec_by_var`` for survivor recipe construction; warped
    # values are written into ``transformed_vars`` like any other unary.
    _prewarp_active = bool(prewarp_enable) and prewarp_y is not None
    _prewarp_spec_by_var: dict[int, dict] = {}
    _prewarp_y_eff = None
    if _prewarp_active:
        from ..hermite_fe import apply_operand_prewarp, fit_pair_prewarp_als
        # The ALS reconstruction target: prefer the CONTINUOUS y when supplied (it
        # is the faithful least-squares target; the binned ``classes_y`` codes the
        # target-rebin guard produces are for the MI screen, not for reconstructing
        # a continuous f(a)*g(b)). Fall back to ``classes_y`` codes when no
        # continuous target was threaded (legacy / non-numeric / multi-output y).
        _pw_y_src = prewarp_y_continuous if prewarp_y_continuous is not None else prewarp_y
        _pw_y = np.asarray(_pw_y_src)
        if _use_subsample and _pw_y.shape[0] == _full_n_rows:
            _pw_y = _pw_y[_sample_idx]
        _prewarp_y_eff = np.ascontiguousarray(_pw_y, dtype=np.float64)

        # JOINT per-pair ALS pre-fit. For each prospective pair fit BOTH operand
        # warps together (rank-1 ALS); an independent 1-D fit cannot recover the
        # b-side of a product target whose b-marginal is ~0. First pairing wins
        # for a var shared across pairs (pairs are processed most-prospective-
        # first, so a shared var binds to its strongest interaction). None specs
        # leave the pseudo-unary unregistered for that var.
        # Q8 (2026-06-07): route the prewarp operand extraction through the SHARED
        # ``_extval_raw_col`` memo (the single {var: raw-ndarray} cache) instead of a
        # second un-memoised ``X.iloc[...].values`` per call. The prewarp loop reads each
        # pair's two operands, and a var shared across many prospective pairs was previously
        # re-extracted once PER pair; the shared memo extracts each distinct var ONCE.
        # Bit-identical: ``_extval_raw_col`` performs the IDENTICAL ``.values`` / ``.to_numpy()``
        # extract (same None-on-missing guard) -- only the redundant re-reads are removed.
        _operand_vals = _extval_raw_col

        # OUT-OF-SAMPLE PREWARP VALIDATION (2026-06-03). The ALS prewarp is a
        # SUPERVISED per-operand fit; at small n it overfits noise operands (the
        # in-sample uplift is inflated by the fit AND by the multiple operand/pair
        # comparisons), so a noise-paired warp clears the in-sample uplift gate,
        # gets engineered, and ABSORBS a genuine feature -- the raw column then
        # reads as redundant and is dropped, leaving a noise-diluted feature
        # (measured: a genuine X5 dropped at n=500). Guard: fit the warp on a TRAIN
        # slice and keep it only if its rank-1 reconstruction f(a)*g(b) still tracks
        # y on a HELD-OUT slice. Genuine synergy (incl. zero-marginal XOR) and
        # genuine non-monotone inners generalise; overfit-on-noise collapses on the
        # held-out slice. At large n train ~= full so genuine recovery is untouched.
        # ``fe_pair_prewarp_min_val_corr`` (default 0.08) is the held-out floor; 0.0
        # restores the legacy in-sample-only fit.
        _pw_min_val_corr = float(prewarp_min_val_corr or 0.0)
        _pw_n = int(_prewarp_y_eff.shape[0]) if hasattr(_prewarp_y_eff, "shape") else len(_prewarp_y_eff)
        # Deterministic stride split (no RNG): every 3rd row -> validation (~33%).
        _pw_cv_ok = _pw_min_val_corr > 0.0 and _pw_n >= 60
        if _pw_cv_ok:
            _val_mask = (np.arange(_pw_n) % 3 == 0)
            _tr_mask = ~_val_mask
            _y_tr = _prewarp_y_eff[_tr_mask]
            _y_val = _prewarp_y_eff[_val_mask] - float(np.mean(_prewarp_y_eff[_val_mask]))

        def _prewarp_generalises(_va_full, _vb_full):
            """True if an ALS warp fit on the train slice still tracks y on the
            held-out slice (rank-1 reconstruction correlation >= floor)."""
            if not _pw_cv_ok:
                return True  # CV disabled / n too small -> accept (legacy behaviour)
            try:
                _a = np.asarray(_va_full, dtype=np.float64).reshape(-1)
                _b = np.asarray(_vb_full, dtype=np.float64).reshape(-1)
                if _a.shape[0] != _pw_n or _b.shape[0] != _pw_n:
                    return True  # length mismatch (subsample edge) -> don't block
                _sa_tr, _sb_tr = fit_pair_prewarp_als(
                    _a[_tr_mask], _b[_tr_mask], _y_tr,
                    basis=prewarp_basis, max_degree=prewarp_max_degree,
                )
                if _sa_tr is None or _sb_tr is None:
                    return False
                _wa = apply_operand_prewarp(_a[_val_mask], _sa_tr)
                _wb = apply_operand_prewarp(_b[_val_mask], _sb_tr)
                _recon = _wa * _wb
                if float(np.std(_recon)) < 1e-12 or float(np.std(_y_val)) < 1e-12:
                    return False
                # measure-experiment-rejected (2026-06-03): a dcor / MI held-out
                # floor (to recover non-monotone XOR prewarps that |corr| might
                # under-credit at small n) gives NO gain. The rank-1 reconstruction
                # f(a)*g(b) is FIT to approximate y, so it is linear-in-y by
                # construction -> |Pearson| is the right measure and is already
                # high for genuine synergy (XOR-sign reconstruction |corr| 0.64-0.75
                # at n=200, far above this 0.08 floor); benched 0/20 cases where
                # |corr|<floor BUT dcor>=0.15 across mul/xor-sign/sq*abs/a*sin(b).
                return abs(float(np.corrcoef(_recon, _y_val)[0, 1])) >= _pw_min_val_corr
            except Exception:
                return True  # validation failure -> fall back to accepting the warp

        for (raw_vars_pair, _), _ in prospective_pairs.items():
            _va, _vb = raw_vars_pair[0], raw_vars_pair[1]
            if _va in _prewarp_spec_by_var and _vb in _prewarp_spec_by_var:
                continue
            _vals_a = _operand_vals(_va)
            _vals_b = _operand_vals(_vb)
            if _vals_a is None or _vals_b is None:
                continue
            # Reject the warp for this pair if it does not generalise out-of-sample
            # (overfit-on-noise). Leaves the operands unregistered -> the pair search
            # falls back to the library unaries, which do not overfit.
            if not _prewarp_generalises(_vals_a, _vals_b):
                continue
            _sa, _sb = fit_pair_prewarp_als(
                _vals_a, _vals_b, _prewarp_y_eff,
                basis=prewarp_basis, max_degree=prewarp_max_degree,
            )
            if _va not in _prewarp_spec_by_var:
                _prewarp_spec_by_var[_va] = _sa
            if _vb not in _prewarp_spec_by_var:
                _prewarp_spec_by_var[_vb] = _sb

    # PER-OPERAND MEDIAN GATE setup (2026-06-04). When enabled, fit ONE TRAIN
    # median per raw operand (on the subsample-aligned slice, exactly like the
    # operand values the unary search consumes) and expose it as an extra
    # pseudo-unary named ``_GATE_MED_UNARY``. The fit is a single ``np.median``
    # per operand -- no supervision, no held-out validation (a median does not
    # overfit). The fitted float per var is kept in ``_gate_med_median_by_var``
    # for survivor recipe construction; the gated 0/1 column is written into
    # ``transformed_vars`` like any other unary. Operands missing from
    # ``original_cols`` or with no usable variance leave the pseudo-unary
    # unregistered for that var (search falls back to the real unaries).
    _gate_med_active = bool(fe_gate_med_enable)
    _gate_med_median_by_var: dict[int, float] = {}
    if _gate_med_active:
        for (raw_vars_pair, _), _ in prospective_pairs.items():
            for _gv in raw_vars_pair:
                if _gv in _gate_med_median_by_var:
                    continue
                if _gv not in original_cols:
                    continue
                # Q8: shared {var: raw-ndarray} memo (bit-identical extract).
                _gvals = _extval_raw_col(_gv)
                if _gvals is None:
                    continue
                _gf = np.asarray(_gvals, dtype=np.float64)
                # Reject no-variance operands (a constant gate is dead).
                _gmed = float(np.nanmedian(_gf)) if _gf.size else 0.0
                if not np.isfinite(_gmed):
                    continue
                _gate_med_median_by_var[_gv] = _gmed

    # Effective unary name list: the real registry plus the pre-warp pseudo-unary
    # when active. Used everywhere a per-pair combination over unary names is
    # built so the pseudo-unary participates exactly like a real one.
    _unary_names_eff = list(unary_transformations.keys())
    if _prewarp_active:
        _unary_names_eff = _unary_names_eff + [_PREWARP_UNARY]
    if _gate_med_active:
        _unary_names_eff = _unary_names_eff + [_GATE_MED_UNARY]

    # Exact preallocation. ``n_pairs * n_unary * 2`` over-counts because (var, tr_name) keys are de-duplicated in ``vars_transformations``; the unique-key set is the
    # true upper bound.
    unique_keys: set = set()
    for (raw_vars_pair, _), _ in prospective_pairs.items():
        for var in raw_vars_pair:
            for tr_name in _unary_names_eff:
                unique_keys.add((var, tr_name))

    if verbose >= 2:
        logger.info(
            "Creating a pool of %d unary transformations for feature engineering "
            "(legacy upper bound was %d).",
            len(unique_keys),
            len(prospective_pairs) * len(unary_transformations) * 2,
        )

    transformed_vars = np.empty(shape=(len(X), len(unique_keys)), dtype=np.float32)

    # Hoist ``final_transformed_vals`` outside the per-pair loop: precompute each pair's ``combs``, find the max length, allocate one shared buffer. Each pair writes
    # then reads the same ``[:, i]`` slice so stale tail data is never observed.
    #
    # OPT-C structural-candidate-dedup analysed + REJECTED (2026-06-07, 0% yield): a proposal to
    # dedup PROVABLY-EQUAL candidate columns (symmetric binaries on equal operand-key sets) before
    # discretize+MI was measured to remove NOTHING -- this generation ALREADY eliminates every
    # provable structural duplicate:
    #   (1) ``combinations`` below emits each UNORDERED operand pair ONCE -> no symmetric-op order
    #       dups within a pair (``mul((a,u1),(b,u2))`` and ``mul((b,u2),(a,u1))`` never both appear);
    #   (2) ``tp[0][0] != tp[1][0]`` keeps operands from DISTINCT raw vars;
    #   (3) ``prospective_pairs`` keys are unique raw {a,b} pairs -> no two candidate columns across
    #       pairs share the same (op, operand-key set);
    #   (4) the shared UNARY operand columns (``sqr(a)`` reused across pairs (a,b),(a,c)) are already
    #       materialised ONCE via the ``vars_transformations`` {(var,unary):col} dict below.
    # Empirical: a canonical-(op, frozenset-of-operand-keys) scan over a 40-raw-pair x 13-unary x
    # 9-op pool = 60840 candidate columns (33800 symmetric-op) found ZERO provable duplicates
    # (0.00%). Any remaining "duplicate" would be VALUE-equal (not structurally provable) operands,
    # which the proposal explicitly excludes. So a structural-dedup pass is pure complexity for no
    # win; do NOT re-implement without a NEW source of structural collisions.
    pair_combs: dict = {}
    max_n_combs = 0
    for (raw_vars_pair, _), _ in prospective_pairs.items():
        combs = list(
            combinations(
                [(raw_vars_pair[0], k) for k in _unary_names_eff]
                + [(raw_vars_pair[1], k) for k in _unary_names_eff],
                2,
            )
        )
        combs = [tp for tp in combs if tp[0][0] != tp[1][0]]
        pair_combs[raw_vars_pair] = combs
        if len(combs) > max_n_combs:
            max_n_combs = len(combs)

    # CRITICAL #2 (2026-05-21): memory-aware dispatch. The full hoisted buffer is the fast path
    # but on n=4M with medium preset this lands at ~17.6 GiB and crashes the suite. Estimate the
    # required buffer, check psutil.virtual_memory().available, and either keep the buffer (fast)
    # or set it to None and switch to recompute-from-metadata in the inner loop and survivor
    # rebuild stages (memory-safe; ~1% extra bin_func calls per pair).
    _n_binary = len(binary_transformations)
    final_transformed_vals_shared = None
    # CROSS-PAIR chunk budget (cols). Default 0 == not chunkable (per-pair buffer).
    # Filled below when the single-pair buffer fits RAM: the chunk buffer reuses the
    # SAME available-RAM budget but may hold MANY pairs (each pair packed whole).
    _fe_chunk_max_cols = 0
    # LARGE-N PEAK-MEMORY FIX (2026-06-08): the candidate buffers (chunk float32 + disc
    # int8 codes + batch-MI working set + the held-alive single-pair buffer) coexist while a
    # chunk is scored, and on the joblib ``backend="threading"`` path ``concurrent_workers``
    # copies of them are alive at once. So BOTH the single-pair fit check and the chunk-width
    # cap use the overhead+worker-aware envelope (raw 0.4*available / _FE_PEAK_OVERHEAD_FACTOR
    # / concurrent_workers) instead of the raw 0.4*available that under-counted the siblings
    # and ignored the thread multiplication. Byte-identical selection (width only).
    _n_workers = max(1, int(concurrent_workers))
    if max_n_combs > 0:
        _buf_bytes = _estimate_fe_shared_buffer_bytes(len(X), max_n_combs, _n_binary)
        _can_hoist, _bb, _avail = _can_hoist_shared_buffer(_buf_bytes, n_workers=_n_workers)
        if _can_hoist:
            try:
                final_transformed_vals_shared = np.empty(
                    shape=(len(X), max_n_combs * _n_binary),
                    dtype=np.float32,
                )
                # Cross-pair chunk width cap: the largest column count whose
                # ``n_rows * cols * 4`` float32 buffer stays inside the overhead+worker-aware
                # envelope (the SUM of the coexisting per-worker buffers fits the SAME 0.4
                # global budget). One pair (``max_n_combs * _n_binary`` cols) already passed,
                # so this is always >= one pair; we pack whole pairs up to it.
                _per_row_bytes = max(1, int(len(X)) * 4)
                _eff_budget_bytes = _fe_effective_buffer_budget_bytes(_avail, n_workers=_n_workers)
                if _eff_budget_bytes >= 0:
                    _fe_chunk_max_cols = int(_eff_budget_bytes // _per_row_bytes)
                else:
                    # No psutil reading -> bound the chunk to the hard cap only.
                    _fe_chunk_max_cols = _FE_CHUNK_MAX_COLS_HARD_CAP
                _fe_chunk_max_cols = max(
                    max_n_combs * _n_binary,
                    min(_fe_chunk_max_cols, _FE_CHUNK_MAX_COLS_HARD_CAP),
                )
            except MemoryError:
                # psutil over-reported available; falling back stays safe.
                final_transformed_vals_shared = None
                if verbose:
                    logger.warning(
                        "check_prospective_fe_pairs: shared buffer (%.1f GiB) allocation raised "
                        "MemoryError despite passing the available-RAM check (%.1f GiB available, "
                        "%.0f%% budget); switching to recompute-from-metadata fallback (~1%% extra "
                        "bin_func calls per pair).",
                        _bb / 2**30, _avail / 2**30 if _avail >= 0 else float("nan"),
                        _FE_BUFFER_RAM_BUDGET_RATIO * 100.0,
                    )
        else:
            if verbose:
                logger.warning(
                    "check_prospective_fe_pairs: shared buffer would need %.1f GiB but only %.1f GiB "
                    "RAM is available (%.0f%% budget = %.1f GiB cap); using recompute-from-metadata "
                    "fallback path (~1%% extra bin_func calls per pair, identical survivors). To force "
                    "the fast path either free RAM or raise _FE_BUFFER_RAM_BUDGET_RATIO; to bound "
                    "compute, pass subsample_n>0 from the MRMR config.",
                    _bb / 2**30, _avail / 2**30 if _avail >= 0 else float("nan"),
                    _FE_BUFFER_RAM_BUDGET_RATIO * 100.0,
                    (_avail * _FE_BUFFER_RAM_BUDGET_RATIO) / 2**30 if _avail >= 0 else float("nan"),
                )
    # bench-attempt-rejected (2026-06-17): BLOCK-STREAMING the candidate buffer when the full
    # per-pair buffer does not hoist (alloc a narrow N-col block buffer, flush materialise ->
    # discretize_2d_quantile_batch -> batched-MI per block; route the 7 downstream
    # ``final_transformed_vals[:, cfg]`` reads -- usability-corr, winner occupied-K, multi-emit,
    # ext-val, survivor -- through a recompute helper). Implemented + VALIDATED bit-identical
    # (block-on selection == per-column selection on the n=100k user fixture), but MEASURED SLOWER
    # in every config: n=100k 4-core box, per-column fallback 103s vs block 119s (joblib default) /
    # 192s (n_jobs=1). The per-block discretise+MI dispatch + the recompute of the downstream
    # winner/leader columns outweigh the parallel-kernel gain, and the per-column fallback ALREADY
    # RAM-bounds (one column at a time via ``_col_buf_1d``), so block-streaming adds no OOM benefit
    # either. The genuine speed lever is the FULL single-buffer batch path (n=100k 54s when it hoists)
    # firing more often, NOT blocking it. Do not re-attempt block-streaming for speed; if 1M-on-16GB
    # ever needs it for RAM, gate it on n and re-bench against the per-column serial baseline first.
    # In the recompute-fallback path we need to look up the
    # ``(transformations_pair, bin_func_name)`` for any index ``i`` that
    # was assigned in the inner loop, so the validation + survivor-packing
    # phases can rebuild the column on demand. The shared-buffer path
    # ignores this dict; either way it is bounded by ``max_n_combs *
    # _n_binary`` lightweight tuples per pair.
    _need_recompute_map = final_transformed_vals_shared is None

    vars_transformations = {}
    i = 0
    for (raw_vars_pair, _pair_mi), _uplift in prospective_pairs.items():
        for var in raw_vars_pair:
            # Q8 (2026-06-07): SHARED {var: raw-ndarray} memo. This main unary-materialise
            # loop iterates over (pair, var); a var shared across prospective pairs was
            # previously re-extracted via ``X.iloc[...].values`` once per occurrence. Reading
            # the shared ``_extval_raw_col`` memo (same Polars/pandas extract: ``X[:, idx].to_numpy()``
            # / ``X.iloc[:, idx].values``) extracts each distinct var ONCE. Bit-identical raw values.
            #
            # ENGINEERED-OPERAND FEED-FORWARD (2026-06-08): ``_extval_raw_col`` now also
            # resolves engineered operands (var not in ``original_cols``) by NAME from the
            # augmented frame, so a step-k>1 ``(eng_i, eng_j)`` pair materialises a real
            # composite candidate. It returns ``None`` only when the var resolves to neither
            # a raw position nor an augmented-frame column (a temp / dropped index); skip
            # silently in that case rather than KeyError out of the whole FE block.
            vals = _extval_raw_col(var)
            if vals is None:
                continue
            for tr_name in _unary_names_eff:
                tr_func = unary_transformations.get(tr_name)
                key = (var, tr_name)
                # Per-operand learned pre-warp: the joint ALS spec was pre-fit
                # above (per pair). When the var has no usable spec (solve failed
                # / non-polynomial basis) the pseudo-unary is simply not
                # registered and the search proceeds with the real unaries only.
                # The fitted spec is stashed for survivor recipe construction
                # (leak-safe replay from coeffs alone).
                if tr_name == _PREWARP_UNARY:
                    if _prewarp_spec_by_var.get(var) is None:
                        continue
                # Median-gate pseudo-unary: skip vars with no fitted median (no
                # variance / not in original_cols). The fitted float is stashed
                # in ``_gate_med_median_by_var`` for survivor recipe construction
                # (leak-safe replay from the stored median alone).
                if tr_name == _GATE_MED_UNARY:
                    if var not in _gate_med_median_by_var:
                        continue
                if key not in vars_transformations:
                    try:
                        if tr_name == _PREWARP_UNARY:
                            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                                transformed_vars[:, i] = apply_operand_prewarp(vals, _prewarp_spec_by_var[var])
                        elif tr_name == _GATE_MED_UNARY:
                            transformed_vars[:, i] = _gate_med_apply(vals, _gate_med_median_by_var[var])
                        elif "poly_" in tr_name:
                            transformed_vars[:, i] = hermval(vals, c=tr_func)
                        else:
                            # WAVE 5 (1/4): if CUDA is available, the
                            # transformation is GPU-compatible, AND the
                            # column is large enough to amortise the H2D
                            # + D2H round trip, run the elementwise op on
                            # GPU via cupy. The numpy-vs-cupy crossover for
                            # this n_samples is resolved per-host via the
                            # shared get_or_tune orchestrator (kernel
                            # "unary_elementwise"), with the old fixed
                            # ~500k-cell breakeven as the measurement-backed
                            # fallback. A "cupy" choice is still gated below
                            # on live CUDA availability + per-op compat.
                            _gpu_used = False
                            from pyutilz.performance.kernel_tuning import array_location

                            from .._unary_elementwise_tuning import unary_elementwise_backend_choice
                            # residency-aware: VRAM-resident input skips H2D, which flips the
                            # numpy/cupy crossover (measured), so pass where ``vals`` lives.
                            _want_gpu = unary_elementwise_backend_choice(int(vals.size), array_location(vals)) == "cupy"
                            if (
                                _want_gpu
                                and tr_name in gpu_compatible_unary_names()
                            ):
                                try:
                                    from pyutilz.core.pythonlib import is_cuda_available
                                    if is_cuda_available():
                                        import cupy as cp
                                        _cp_fn = getattr(cp, tr_name, None)
                                        if _cp_fn is not None:
                                            d_vals = cp.asarray(vals)
                                            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                                                d_res = _cp_fn(d_vals)
                                            transformed_vars[:, i] = cp.asnumpy(d_res)
                                            _gpu_used = True
                                except Exception:
                                    _gpu_used = False  # fall through to CPU
                            if not _gpu_used:
                                # Suppress unary-transform NaN/inf RuntimeWarnings
                                # (eg ``overflow in exp``, ``divide by zero in
                                # log``). The downstream nan_to_num + MI-gate
                                # already filter pathological rows; the bare
                                # numpy emit only spams stderr.
                                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                                    transformed_vars[:, i] = tr_func(vals)
                    except Exception as e:
                        # ``np.isnan`` / ``np.isinf`` / ``np.nanmin`` only work on float dtypes. When ``vals`` is object/string (e.g. a polars Utf8 cat column not encoded
                        # before reaching FE), calling them inside the error-log formatter itself raises -- masking the real transformation error and aborting MRMR
                        # entirely. Compute numeric-only diagnostics conditionally.
                        if np.issubdtype(vals.dtype, np.floating):
                            _diag = (
                                f", isnan={np.isnan(vals).sum()}, "
                                f"isinf={np.isinf(vals).sum()}, nanmin={np.nanmin(vals)}"
                            )
                        else:
                            _diag = f", dtype={vals.dtype} (numeric diagnostics skipped)"
                        logger.error(
                            f"Error when performing {tr_name} on array {vals[:5]}, "
                            f"var={cols[var]}: {str(e)}{_diag}"
                        )
                    else:
                        vars_transformations[key] = i
                        i += 1

    if verbose >= 2:
        logger.info("Created. For every pair from the pool, trying all known functions...")

    # Per-operand marginal MI cache for the MARGINAL-UPLIFT alternative acceptance gate.
    # The marginal MI is the discretised RAW operand (its ``identity`` unary, already
    # materialised in ``transformed_vars``) scored against the target with the SAME
    # estimator the engineered columns use, so the uplift ratio is apples-to-apples.
    # Operands recur across pairs, so memoise by cols-space var index.
    _operand_marginal_mi_cache: dict = {}

    # NOISE-WRAP CORR-COLLAPSE GUARD (2026-06-15). A subsample-aligned CONTINUOUS target for a cheap |corr|
    # discriminator that catches the failure mode where the per-pair search WRAPS a strong, clean operand
    # (e.g. a univariate-basis column ``a__T2`` ~ a**2) with a PURE-NOISE operand (``e``) via an extreme
    # heavy-tailed transform (``sub(log(e),invqubed(a__T2))``). That composite's binned-MI ``best_mi/pair_mi``
    # ratio CLEARS the joint-prevalence gate (the extreme transform inflates BOTH MIs), yet its |corr| with the
    # target COLLAPSES to ~0 while the clean operand's |corr| is ~1.0 -- so the artefact displaces the clean
    # basis from the support and recovery dies. Genuine synergy pairs (a*b, log(c)*sin(d)) do NOT collapse this
    # way: the engineered column still tracks y monotonically. Prefer the continuous y; fall back to the binned
    # ``classes_y`` codes (still a usable monotone proxy) when no continuous target was threaded.
    _corr_y_cont = None
    try:
        _cyc_src = (
            usability_y_continuous if usability_y_continuous is not None
            else (prewarp_y_continuous if prewarp_y_continuous is not None else classes_y)
        )
        _cyc = np.asarray(_cyc_src, dtype=np.float64).ravel()
        if _use_subsample and _cyc.shape[0] == _full_n_rows:
            _cyc = _cyc[_sample_idx]
        if _cyc.shape[0] == len(classes_y) and np.isfinite(_cyc).any() and float(np.nanstd(_cyc)) > 1e-12:
            _corr_y_cont = _cyc
    except Exception:
        _corr_y_cont = None

    def _safe_abs_corr(_v) -> float:
        """|Pearson corr| of a column with the (subsample-aligned) target over their jointly-finite rows;
        0.0 when the guard target is unavailable or either side is degenerate. Cheap (one corrcoef)."""
        if _corr_y_cont is None:
            return 0.0
        try:
            _a = np.asarray(_v, dtype=np.float64).ravel()
            if _a.shape[0] != _corr_y_cont.shape[0]:
                return 0.0
            _m = np.isfinite(_a) & np.isfinite(_corr_y_cont)
            if int(_m.sum()) < 8:
                return 0.0
            _av, _yv = _a[_m], _corr_y_cont[_m]
            if float(np.std(_av)) <= 1e-12 or float(np.std(_yv)) <= 1e-12:
                return 0.0
            return abs(float(np.corrcoef(_av, _yv)[0, 1]))
        except Exception:
            return 0.0

    # The winning composite is condemned as a noise-wrap when its |corr| with the target is a small FRACTION
    # of the best single operand's |corr| AND that operand is genuinely strong on its own. Calibrated wide:
    # the artefact collapses to ~0.02 vs the clean operand's ~0.99 (a >40x collapse), while a genuine synergy
    # pair's engineered column tracks y at least comparably to its strongest operand. ``0.5`` keeps a 2x margin
    # so a real synergy that modestly trades linear |corr| for a higher-order MI gain is never condemned.
    _NOISE_WRAP_CORR_COLLAPSE_FRAC: float = 0.5
    _NOISE_WRAP_MIN_OPERAND_CORR: float = 0.30

    # bench-attempt-rejected (2026-06-07): BATCH all distinct operands' marginal MI through
    # one discretize_2d_quantile_batch + one _dispatch_batch_mi_with_noise_gate (Q9), instead
    # of this per-var single-column discretize_array + mi_direct. It WOULD be bit-identical
    # (the batch discretise + batch gate are already proven equal to the per-column path), but
    # it is not worth the invasive control-flow change + the risk to the load-bearing uplift
    # gate comparison below: _operand_marginal_mi is already memoised (each distinct operand
    # computed at most once) AND it only fires on the marginal-uplift FALLBACK gate (pairs that
    # miss the joint + prewarp gates), so on the scene scene-profile the ENTIRE mi_direct family
    # is only 0.1-0.3% of fit wall -- batching it saves a sub-noise fraction that is dwarfed by
    # the GPU-clock variance between runs. Re-evaluate only if a workload makes this gate hot.
    def _operand_marginal_mi(_var) -> float:
        if _var in _operand_marginal_mi_cache:
            return _operand_marginal_mi_cache[_var]
        _mi_val = 0.0
        _idx = vars_transformations.get((_var, "identity"))
        if _idx is not None:
            try:
                _disc = discretize_array(
                    arr=transformed_vars[:, _idx], n_bins=quantization_nbins,
                    method=quantization_method, dtype=quantization_dtype,
                )
                _m, _ = mi_direct(
                    _disc.reshape(-1, 1),
                    x=np.array([0], dtype=np.int64), y=None,
                    factors_nbins=np.array([quantization_nbins], dtype=np.int64),
                    classes_y=classes_y, classes_y_safe=classes_y_safe, freqs_y=freqs_y,
                    min_nonzero_confidence=fe_min_nonzero_confidence, npermutations=fe_npermutations,
                )
                _mi_val = float(_m)
            except Exception as _mm_exc:
                # FAIL-CLOSED (audit A3, 2026-06-13): the previous ``0.0`` was FAIL-OPEN -- it fed
                # the marginal-uplift gate's ``max(operand marginals)``, so a FAILED marginal on the
                # operand that actually has the LARGER marginal would shrink that max and LOOSEN the
                # admission bar (``best_nonprewarp_mi >= max_marginal * _FE_MARGINAL_UPLIFT_MIN_RATIO``),
                # wrongly admitting a feature whose uplift was never validated. Return +inf instead so
                # an UNKNOWN marginal can only TIGHTEN the gate (the pair fails the uplift fallback and
                # is dropped-on-uncertainty); it can still be admitted by the joint/prewarp gates, which
                # do not use this marginal. The whole-pair both-operand-fail case was already
                # fail-closed via the ``_max_operand_marginal > 0.0`` guard.
                if verbose:
                    logger.warning(
                        "MRMR FE: operand %s marginal-MI computation failed (%s); failing the "
                        "marginal-uplift gate CLOSED (+inf) so it cannot loosen admission.",
                        _var, type(_mm_exc).__name__,
                    )
                _mi_val = float("inf")
        _operand_marginal_mi_cache[_var] = _mi_val
        return _mi_val

    # MM-DEBIAS (2026-06-09, backlog #1 + #4): per-operand DISCRETISED codes for the
    # occupied-joint-K of a raw pair. Memoised + bit-identical to the discretise the
    # gate's ``pair_mi`` was computed over (same ``quantization_nbins`` / method /
    # ``transformed_vars`` identity column ``_operand_marginal_mi`` uses). Returns the
    # int code array, or None when the operand has no identity transform.
    _operand_disc_cache: dict = {}

    def _operand_discretized(_var):
        if _var in _operand_disc_cache:
            return _operand_disc_cache[_var]
        _codes = None
        _idx = vars_transformations.get((_var, "identity"))
        if _idx is not None:
            try:
                _codes = discretize_array(
                    arr=transformed_vars[:, _idx], n_bins=quantization_nbins,
                    method=quantization_method, dtype=quantization_dtype,
                )
            except Exception:
                _codes = None
        _operand_disc_cache[_var] = _codes
        return _codes

    # CROSS-PAIR (CHUNK) BATCHING precompute (2026-06-06). Only on the hoist+quantile
    # path (the per-pair 3-phase batch path). We partition the prospective pairs into
    # CHUNKS whose total candidate-column count fits the RAM-budgeted chunk buffer,
    # then -- per chunk -- materialise ALL the chunk's candidate columns into ONE wide
    # buffer, run ONE discretize_2d + ONE batch_mi over the whole chunk, and stash the
    # per-pair results (ordered candidate list + per-candidate MI + buffer columns) in
    # ``_chunk_mi_cache``. The per-pair loop below consumes the cache (reads MI + the
    # chunk buffer columns) instead of doing its own per-pair Phase 1/2/3 -- enlarging
    # the njit-prange + GPU-dispatch batch from one pair's K to the whole chunk's
    # K_chunk, so the kernels saturate the cores / cross the GPU threshold. Bit-identical
    # (each column is scored independently; shuffle seeded by (0, perm_index) only).
    #
    # ``_chunk_mi_cache[raw_vars_pair]`` -> (ordered list of
    # (transformations_pair, bin_func_name, buf_col, uses_pw), fe_mi_array_aligned,
    # local_times_for_pair). ``_chunk_buffer`` is the wide buffer the buf_col indices
    # point into; it is held alive for the whole pair loop so survivor packing can read
    # ``_chunk_buffer[:, buf_col]`` exactly as the per-pair path read final_transformed_vals.
    _chunk_global_batch = (
        (final_transformed_vals_shared is not None)
        and (quantization_method == "quantile")
        and (_fe_chunk_max_cols > max_n_combs * _n_binary)  # chunk holds > 1 pair's worth
        and (len(prospective_pairs) > 1)
    )
    _chunk_mi_cache: dict = {}
    _chunk_buffer = None
    _pair_to_chunk: dict = {}   # raw_vars_pair -> chunk index
    _fe_chunks: list = []
    _pair_valid_combs: dict = {}
    if _chunk_global_batch:
        _fe_chunks, _pair_valid_combs, _chunk_buf_width = _plan_fe_chunks(
            prospective_pairs=prospective_pairs,
            pair_combs=pair_combs,
            vars_transformations=vars_transformations,
            n_binary=_n_binary,
            chunk_max_cols=_fe_chunk_max_cols,
        )
        # Only worth chunking when at least one chunk groups MORE than one pair.
        if _fe_chunks and max(len(c) for c in _fe_chunks) > 1 and _chunk_buf_width > 0:
            try:
                _chunk_buffer = np.empty((len(X), _chunk_buf_width), dtype=np.float32)
                for _ci_chunk, _chunk in enumerate(_fe_chunks):
                    for _p in _chunk:
                        _pair_to_chunk[_p] = _ci_chunk
                if verbose:
                    logger.info(
                        "check_prospective_fe_pairs: cross-pair chunk batching active "
                        "(%d pairs -> %d chunks, buffer %d cols, widest chunk %d pairs).",
                        len(prospective_pairs), len(_fe_chunks), _chunk_buf_width,
                        max(len(c) for c in _fe_chunks),
                    )
            except MemoryError:
                _chunk_buffer = None
                _pair_to_chunk = {}
                if verbose:
                    logger.warning(
                        "check_prospective_fe_pairs: cross-pair chunk buffer (%d x %d) raised "
                        "MemoryError; using per-pair batching.", len(X), _chunk_buf_width,
                    )
        else:
            _chunk_global_batch = False
    # Index of the chunk currently materialised in ``_chunk_buffer`` (-1 = none).
    _loaded_chunk_idx = -1

    # Sweep-wide leader for the live "pair" bar postfix: the best engineered MI found
    # so far across ALL pairs in this FE step + the feature that produced it. Both are
    # already computed per pair (``best_mi`` / ``best_config``) -- display-only, no extra
    # MI work. Starts blank so the no-candidate / NaN edge case renders gracefully.
    _sweep_best_mi = -1.0
    _sweep_best_name = None

    # For every pair from the pool, try all known functions of 2 variables (not storing results in persistent RAM). Record best pairs.
    for (
        raw_vars_pair,
        pair_mi,
    ), _uplift in (pair_pbar := tqdmu(
        prospective_pairs.items(), desc="pair", leave=False, disable=not verbose
    )):  # better to start considering form the most prospective pairs with highest mis ratio!

        messages = []

        combs = pair_combs[raw_vars_pair]

        best_config, best_mi = None, -1
        this_pair_features = set()
        var_pairs_perf = {}
        # Pre-warp uplift tracking (2026-06-02): the best engineered MI achievable
        # with ONLY the elementary library unaries (no ``prewarp`` operand) vs the
        # best USING a prewarp operand. A 1-D engineered summary of a 2-D pair
        # cannot retain ``fe_min_engineered_mi_prevalence`` of the 2-D JOINT MI,
        # so on a non-monotone inner distortion (where the elementary library is
        # representationally blind) the prewarp winner is rejected by the joint
        # prevalence gate despite being a large, real uplift over the best the
        # library can do. The alternative acceptance path below admits a prewarp
        # winner when it beats the best non-prewarp engineered MI by a margin --
        # directed (only fires where the prewarp adds representational power) and
        # noise-safe (on linear/monotone/noise data the prewarp does not beat the
        # elementary library, so the margin is never cleared).
        best_nonprewarp_mi = -1.0
        best_nonprewarp_config = None
        best_prewarp_config, best_prewarp_mi = None, -1.0

        # CRITICAL #2 dispatch: hoist path uses the shared buffer (writes into
        # ``[:, i]``); recompute-fallback path uses a tiny 1D scratch + a
        # config-by-i map for on-demand survivor recomputation later.
        # CROSS-PAIR: when this pair was batched across the chunk, its survivor
        # columns live in the wide ``_chunk_buffer`` (the config's ``i`` is the
        # chunk-buffer column), so point ``final_transformed_vals`` at it. The chunk
        # is materialised LAZILY: pairs are processed in chunk-plan order, so when we
        # reach the FIRST pair of a not-yet-loaded chunk we fill the buffer + MI cache
        # for that whole chunk in ONE batched pass. By the time the next chunk's first
        # pair arrives, all of this chunk's pairs (incl. their survivor packing, which
        # reads the buffer) have already been processed -> safe to overwrite.
        _chunk_entry = None
        if _chunk_global_batch and (_chunk_buffer is not None):
            _my_chunk = _pair_to_chunk.get(raw_vars_pair)
            if _my_chunk is not None:
                if _my_chunk != _loaded_chunk_idx:
                    _chunk_mi_cache = _compute_one_fe_chunk(
                        chunk_pairs=_fe_chunks[_my_chunk],
                        pair_valid_combs=_pair_valid_combs,
                        chunk_buffer=_chunk_buffer,
                        vars_transformations=vars_transformations,
                        transformed_vars=transformed_vars,
                        binary_transformations=binary_transformations,
                        quantization_nbins=quantization_nbins,
                        quantization_dtype=quantization_dtype,
                        classes_y=classes_y,
                        classes_y_safe=classes_y_safe,
                        freqs_y=freqs_y,
                        fe_npermutations=fe_npermutations,
                        fe_min_nonzero_confidence=fe_min_nonzero_confidence,
                        batch_mi_kernel=batch_mi_with_noise_gate,
                        use_su=use_su_normalization(),
                        prewarp_unary=_PREWARP_UNARY,
                        logger=logger,
                        discretize_2d_quantile_batch=discretize_2d_quantile_batch,
                        serial_main_thread=serial_main_thread,  # OPT-A
                    )
                    _loaded_chunk_idx = _my_chunk
                _chunk_entry = _chunk_mi_cache.get(raw_vars_pair)
        if _chunk_entry is not None:
            final_transformed_vals = _chunk_buffer
        else:
            final_transformed_vals = final_transformed_vals_shared
        _col_buf_1d: np.ndarray | None = (
            np.empty(len(X), dtype=np.float32) if _need_recompute_map else None
        )
        _config_by_i: dict[int, tuple] = {} if _need_recompute_map else None

        i = 0
        # Per-pair thread-local timing accumulator; merged into the shared
        # ``times_spent`` under the lock once per pair (see end of pair loop).
        _local_times: dict = {}

        # BATCHED-DISCRETIZE dispatch (2026-06-04): per-candidate ``discretize_array``
        # (np.linspace + np.nanpercentile->partition + searchsorted) is the FE-pair-search
        # hotspot -- millions of tiny per-column numpy calls -> serial-dispatch-bound,
        # idle CPU. On the HOIST path (shared buffer present) with the quantile method we
        # split this pair's sweep into 3 phases: (1) materialise ALL candidate columns into
        # the buffer + nan_to_num + record (config, idx, uses_pw); (2) batch-discretise the
        # filled buffer slice in ONE ``np.nanpercentile(axis=0)`` (amortises dispatch over K
        # columns -- bit-identical to per-column, see ``discretize_2d_quantile_batch``);
        # (3) replay the EXACT per-candidate mi_direct + best/prewarp/config tracking.
        # MI stays per-candidate (mi_direct's permutation confidence is NOT batched). The
        # recompute-fallback (no buffer) and the uniform method keep the original
        # per-candidate path verbatim -- only the hoist+quantile case is batched.
        _use_batch_disc = (final_transformed_vals is not None) and (quantization_method == "quantile")

        if _use_batch_disc:
            # CROSS-PAIR fast path: this pair was batched together with the rest of its
            # chunk in ``_compute_one_fe_chunk``. Its candidate columns already live
            # in ``_chunk_buffer`` (the buf_col index is the config ``i``), its MI is
            # already computed, and its per-bin_func materialise timings are recorded.
            # We only replay the EXACT per-candidate tracking below. Bit-identical: the
            # chunk's ONE discretize_2d + ONE batch_mi score each column independently,
            # and the candidate order per pair is the SAME (combs x bin_funcs) order the
            # per-pair Phase 1 produced.
            if _chunk_entry is not None:
                _batch_candidates, _fe_mi_by_col, _pair_local_times = _chunk_entry
                for _bf_name, _dt in _pair_local_times.items():
                    _local_times[_bf_name] = _local_times.get(_bf_name, 0.0) + _dt
                # ``_fe_mi_arr`` is indexed by the chunk-buffer column (buf_col), so the
                # replay's ``_fe_mi_arr[_ci]`` lookup is correct without re-indexing.
                _fe_mi_arr = _fe_mi_by_col
            else:
                # Phase 1: materialise + nan_to_num + record. ``i`` advances exactly as in the
                # per-candidate path so ``config``'s buffer index and ``_config_by_i`` are identical.
                _batch_candidates = []  # (transformations_pair, bin_func_name, i, uses_pw)
                for transformations_pair in combs:
                    if (transformations_pair[0] not in vars_transformations) or (transformations_pair[1] not in vars_transformations):
                        continue
                    param_a = transformed_vars[:, vars_transformations[transformations_pair[0]]]
                    param_b = transformed_vars[:, vars_transformations[transformations_pair[1]]]
                    _uses_pw = (
                        transformations_pair[0][1] == _PREWARP_UNARY
                        or transformations_pair[1][1] == _PREWARP_UNARY
                    )
                    # Same wide errstate scope as the original per-pair-comb path.
                    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                        for bin_func_name, bin_func in binary_transformations.items():
                            start = timer()
                            try:
                                final_transformed_vals[:, i] = bin_func(param_a, param_b)
                            except Exception:
                                logger.error(f"Error when performing {bin_func}")
                            else:
                                # DEFER the NaN/inf scrub to ONE vectorised pass over the packed
                                # buffer slice [:, :i] below (was a per-column ``nan_to_num`` here:
                                # K tiny 50k-element isposinf/isneginf calls per pair -> profiled at
                                # 16.5s / 5834 calls on the 5-feat x 50000-row repro, pure serial
                                # numpy dispatch with the cores idle). ``nan_to_num`` is elementwise
                                # so scrubbing the whole [:, :i] block at once is byte-identical to
                                # scrubbing each column as it is written, and runs one C loop over a
                                # contiguous (n x K) buffer instead of K strided ones.
                                _local_times[bin_func_name] = _local_times.get(bin_func_name, 0.0) + (timer() - start)
                                _batch_candidates.append((transformations_pair, bin_func_name, i, _uses_pw))
                                i += 1

                # Phase 1b: ONE vectorised NaN/inf scrub over every materialised column
                # [:, :i] (replaces the per-column ``nan_to_num`` removed above). Elementwise
                # -> byte-identical to the per-column scrub; one contiguous-block C pass.
                if i > 0:
                    np.nan_to_num(final_transformed_vals[:, :i], copy=False, nan=0.0, posinf=0.0, neginf=0.0)

                # Phase 2: ONE batch discretisation over the materialised columns [:, :n].
                # Bit-identical to per-column ``discretize_array(method='quantile')`` -- the
                # buffer dtype (float32) is NOT cast; per-column edges/codes match exactly.
                _fe_mi_arr = None
                if _batch_candidates:
                    # ``i`` advanced once per materialised candidate from 0 (reset per raw-pair),
                    # so the filled buffer slice is exactly [:, :i], densely packed 0..i-1.
                    _disc_2d = discretize_2d_quantile_batch(
                        final_transformed_vals[:, :i], n_bins=quantization_nbins,
                        dtype=_narrow_code_dtype(quantization_nbins, quantization_dtype),  # OPT-B narrow codes
                        # OPT-A extension (2026-06-07): same main-thread parallel searchsorted
                        # gate as the chunk + marginal-uplift discretise -- byte-identical
                        # column-prange twin when serial_main_thread (no joblib nest).
                        parallel=_fe_use_parallel_kernels(i, serial_main_thread),
                        # The ``np.nan_to_num(..., copy=False)`` directly above scrubbed this exact
                        # buffer slice, so the per-call ``np.isnan().any()`` scan inside the discretiser
                        # is guaranteed-False wasted work; skip it (bit-identical on a NaN-free buffer).
                        assume_finite=True,
                    )

                    # Phase 3: BATCHED MI + permutation noise-gate across ALL K candidate
                    # columns in ONE kernel call. Bit-identical to the per-candidate
                    # ``mi_direct`` loop on the default FE path (parallelism='outer',
                    # n_workers=1 -> parallel_mi_prange, base_seed=0): every candidate is
                    # tested against the SAME npermutations shuffles of y (the shuffle is
                    # seeded by (base_seed, perm_index) ONLY, never by classes_x), so a single
                    # batched kernel can shuffle y once per permutation and score all columns
                    # against it -- amortising both the MI compute and the shuffle across K.
                    # ``_dispatch_batch_mi_with_noise_gate`` routes CPU-njit vs a GPU batched
                    # path by n*K via the kernel_tuning_cache (no hardcoded threshold).
                    _fe_mi_arr = _dispatch_batch_mi_with_noise_gate(
                        disc_2d=_disc_2d,
                        quantization_nbins=quantization_nbins,
                        classes_y=classes_y,
                        classes_y_safe=classes_y_safe,
                        freqs_y=freqs_y,
                        npermutations=fe_npermutations,
                        min_nonzero_confidence=fe_min_nonzero_confidence,
                        use_su=use_su_normalization(),
                        batch_mi_kernel=batch_mi_with_noise_gate,
                    )

            # Replay best/prewarp/config tracking in the SAME order candidates were
            # produced -> identical tie-break behaviour. ``_fe_mi_arr`` is indexed by the
            # buffer column (per-pair: 0..K-1; cross-pair: the chunk-buffer column).
            if _batch_candidates and _fe_mi_arr is not None:
                for transformations_pair, bin_func_name, _ci, _uses_pw in _batch_candidates:
                    # Cast to Python float so ``var_pairs_perf`` / downstream tracking see
                    # the same scalar type ``mi_direct`` returned (numba njit returns a
                    # python float at the call boundary). Value is bit-identical.
                    fe_mi = float(_fe_mi_arr[_ci])

                    config = (transformations_pair, bin_func_name, _ci)
                    var_pairs_perf[config] = fe_mi
                    if _need_recompute_map:
                        _config_by_i[_ci] = (transformations_pair[0], transformations_pair[1], bin_func_name)

                    if fe_mi > best_mi:
                        best_mi = fe_mi
                        best_config = config
                    if _uses_pw:
                        if fe_mi > best_prewarp_mi:
                            best_prewarp_mi = fe_mi
                            best_prewarp_config = config
                    else:
                        if fe_mi > best_nonprewarp_mi:
                            best_nonprewarp_mi = fe_mi
                            best_nonprewarp_config = config
                    if fe_mi > best_mi * 0.85:
                        if not fe_print_best_mis_only or (fe_mi == best_mi):
                            if verbose > 2:
                                print(f"MI of transformed pair {bin_func_name}({transformations_pair})={fe_mi:.4f}, MI of the plain pair {pair_mi:.4f}")
        else:
            for transformations_pair in combs:
                if (transformations_pair[0] not in vars_transformations) or (transformations_pair[1] not in vars_transformations):
                    continue
                param_a = transformed_vars[:, vars_transformations[transformations_pair[0]]]
                param_b = transformed_vars[:, vars_transformations[transformations_pair[1]]]

                # A config "uses prewarp" iff either operand's unary name is the
                # pseudo-unary. Invariant across the bin_func loop -> compute once.
                _uses_pw = (
                    transformations_pair[0][1] == _PREWARP_UNARY
                    or transformations_pair[1][1] == _PREWARP_UNARY
                )

                # ``bin_func`` produces NaN/+-inf on extreme Optuna-picked params
                # (overflow in mul/exp, divide-by-zero in log); the downstream
                # nan_to_num + MI gate already sanitise, so the bare numpy
                # RuntimeWarnings carry zero diagnostic value. Suppress them for the
                # whole binary-transform sweep: entering np.errstate per inner
                # iteration cost ~6.8us/iter (measured ~490ms over 72k iters),
                # dwarfing the bin_func work itself; one context per pair-comb
                # removes that. numba kernels (discretize/mi_direct) ignore errstate
                # and nan_to_num emits nothing, so the wider scope is value-identical.
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    for bin_func_name, bin_func in binary_transformations.items():

                        start = timer()
                        try:
                            if final_transformed_vals is not None:
                                final_transformed_vals[:, i] = bin_func(param_a, param_b)
                                _col_view = final_transformed_vals[:, i]
                            else:
                                # Recompute fallback: write into the shared 1D scratch.
                                # bin_func returns a fresh ndarray; copy into the scratch
                                # so downstream nan_to_num + discretize see contiguous
                                # data. Avoids accumulating one alloc per inner iter.
                                _col_buf_1d[:] = bin_func(param_a, param_b)
                                _col_view = _col_buf_1d
                        except Exception:
                            logger.error(f"Error when performing {bin_func}")
                        else:
                            np.nan_to_num(_col_view, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

                            # Wave 27 P1: ``times_spent`` is shared across mrmr.py's
                            # parallel threading dispatch. Accumulate this pair's
                            # per-bin_func timings in a thread-LOCAL dict and merge them
                            # under ``_TIMES_SPENT_LOCK`` once per pair (below); the old
                            # per-inner-iteration lock was a serialization point on the
                            # hot path. Totals are identical.
                            _local_times[bin_func_name] = _local_times.get(bin_func_name, 0.0) + (timer() - start)

                            discretized_transformed_values = discretize_array(
                                arr=_col_view, n_bins=quantization_nbins, method=quantization_method, dtype=quantization_dtype
                            )
                            fe_mi, fe_conf = mi_direct(
                                discretized_transformed_values.reshape(-1, 1),
                                x=np.array([0], dtype=np.int64),
                                y=None,
                                factors_nbins=np.array([quantization_nbins], dtype=np.int64),
                                classes_y=classes_y,
                                classes_y_safe=classes_y_safe,
                                freqs_y=freqs_y,
                                min_nonzero_confidence=fe_min_nonzero_confidence,
                                npermutations=fe_npermutations,
                            )

                            config = (transformations_pair, bin_func_name, i)
                            var_pairs_perf[config] = fe_mi
                            if _need_recompute_map:
                                # Map i -> (a_key, b_key, bin_func_name) for downstream
                                # rebuild; bin_func is looked up via the original dict.
                                _config_by_i[i] = (transformations_pair[0], transformations_pair[1], bin_func_name)

                            if fe_mi > best_mi:
                                best_mi = fe_mi
                                best_config = config
                            # Track best-with-prewarp vs best-without so the alternative
                            # uplift gate below can decide whether the prewarp earned its
                            # place (``_uses_pw`` hoisted above the bin_func loop).
                            if _uses_pw:
                                if fe_mi > best_prewarp_mi:
                                    best_prewarp_mi = fe_mi
                                    best_prewarp_config = config
                            else:
                                if fe_mi > best_nonprewarp_mi:
                                    best_nonprewarp_mi = fe_mi
                                    best_nonprewarp_config = config
                            if fe_mi > best_mi * 0.85:
                                if not fe_print_best_mis_only or (fe_mi == best_mi):
                                    if verbose > 2:
                                        print(f"MI of transformed pair {bin_func_name}({transformations_pair})={fe_mi:.4f}, MI of the plain pair {pair_mi:.4f}")
                            i += 1

        # Merge this pair's per-bin_func timings into the shared accumulator in
        # ONE locked pass (the increment was previously locked per inner
        # iteration -- a serialization point under the parallel pair dispatch).
        if _local_times:
            with _TIMES_SPENT_LOCK:
                for _bf, _dt in _local_times.items():
                    times_spent[_bf] += _dt

        if verbose > 2:
            print(f"For pair {raw_vars_pair}, best config is {best_config} with best mi= {best_mi}")

        # experiment-rejected (2026-06-03): a held-out-CV firewall here (score
        # per-combo MI on a TRAIN stride slice for honest selection, then keep the
        # winner only if its held-out VAL-slice MI retains >= ratio of train MI) was
        # implemented and benched END-TO-END on Layer-49 -- NO gain. In an isolated
        # probe it separated cleanly (genuine synergy val/train 0.90-1.04 vs noise-FE
        # 0.12-0.36), BUT in the real pipeline the tighter prevalence-gate defaults
        # (fe_synergy_min_prevalence 1.5 / fe_min_engineered_mi_prevalence 0.97)
        # already remove the pure noise*noise products, and the RESIDUAL "noise" FE
        # are signal*noise combos (e.g. max(log(L4_s2),noise_3) -- L4_s2 is a real
        # sensor) that genuinely generalise (val/train > 0.5) and SHOULD be kept; the
        # firewall's train-based selection (half the rows) then merely added selection
        # noise (+1 support). Prevalence gating subsumes the win -> not shipped.
        # Standard acceptance: the best engineered MI clears the configured
        # fraction of the 2-D pair-joint MI.
        #
        # MILLER-MADOW DEBIAS (2026-06-09, backlog #1 + #4). The RAW ratio
        # ``best_mi / pair_mi`` compares a 1-D engineered MI (over ~``quantization_nbins``
        # bins) against a 2-D joint MI (over ~``nbins^2`` bins). Both are plug-in MIs whose
        # positive bias is ``(k_x-1)(k_y-1)/2n``; the JOINT denominator's term is ~``nbins``x
        # larger, so the raw ratio is structurally depressed below 1.0 even when the 1-D
        # feature captures all the joint information (worst at small/moderate n) -- this is
        # exactly the documented reason the marginal-uplift fallback gate had to be added.
        # When ``fe_mm_debias_prevalence`` we subtract the MM MI-bias term from BOTH sides,
        # using the OCCUPIED bin counts (#4: nominal ``nbins`` over-corrects heavy-tailed
        # columns that collapse), with a denominator-positivity guard that defers to the raw
        # ratio when the joint bias term swamps the finite-sample joint MI. ``->`` raw ratio
        # as ``n -> inf`` (bias terms vanish) => large-n selection byte-untouched. The order-2
        # maxT floor (the outer guard) is MM-debiased CONSISTENTLY upstream (the IRON RULE),
        # so admitting more pairs here does NOT weaken the best-of-pool noise floor.
        #
        # bench-attempt-rejected (2026-06-09, FS backlog #5 "permutation-null-calibrated
        # prevalence bar"). The idea: REPLACE the hardcoded ``fe_min_engineered_mi_prevalence``
        # (0.90) with a SELF-CALIBRATING per-pool null ratio -- in the SAME K y-shuffles the
        # order-2 maxT floor runs, ALSO mirror the max-over-transforms search (discretise the
        # elementary binary bank mul/add/sub/div/max/min over the CONTINUOUS operands ONCE --
        # permutation-invariant -- then per shuffle take max 1-D engineered MI / joint pair MI),
        # and gate ``best_mi/pair_mi`` against the q95 of that null-ratio distribution (the chance
        # ceiling), admitting only ABOVE it. Unlike #1 (a DETERMINISTIC bias subtraction that
        # uniformly relaxes the bar) the null ratio is calibrated to what NOISE actually produces.
        # MEASURED (standalone probe, N_BINS=8, K=25, q=0.95):
        #   * PURE NOISE (n=2000, p=12): null q95 ratio ~0.16; real noise-pair ratios <=0.17 ->
        #     ~5% admitted = pure (1-q) chance rate. The HARD noise-FP gate PASSES on clean noise.
        #   * He2(a)*b genuine synergy (n=500/2000/8000): real ratio 0.28/0.275/0.268 >> null
        #     0.15/0.16/0.17 -> ADMIT, while the hardcoded 0.90 bar REJECTS at every n. In a mixed
        #     He2-signal+8-noise frame the null bar admits the genuine (a,b) pair and 0-1/28 noise
        #     pairs (chance rate). So in ISOLATION #5 is a genuine improvement over #1.
        # BUT bench-REJECTED on the case that matters -- the user's WEAK F2
        # (``0.2*a**2/b + log(c*2)*sin(d/3)``, the SAME target that rejected #1/#8/#19): the null
        # ceiling is ~0.167 (calibrated to clean-noise pairs, ratio ~0.13-0.17), but EVERY weak-F2
        # pair sits FAR above it (5 seeds, n=20000): genuine_ab ~0.81, genuine_cd ~0.73, AND all four
        # cross-mix pairs 0.56-0.72 (cross(b,d) ~0.717 >= genuine_cd). So the null bar ADMITS every
        # cross-mix on every seed -- the IRON-RULE failure mode, identical to #1 (cross-mix 3/10 ->
        # 9/10). ROOT CAUSE is the documented fundamental detectability limit (see
        # ``test_mrmr_weak_f2_seed_stability.py`` "THREE DIRECT LEVERS EXHAUSTED"): the cross-mix
        # smuggles the dominant MONOTONE predictor ``c`` across the pair boundary, so its 1-D
        # engineered summary recovers a large fraction of its (real, cross) joint -- a HIGH ratio
        # indistinguishable from genuine synergy by ANY MI threshold. The null bar measures the
        # noise floor, but the weak-F2 problem is NOT noise admission; it is a real-monotone-predictor
        # cross-mix whose ratio is nowhere near the noise floor. AND the existing marginal-uplift /
        # prewarp FALLBACK already recovers the genuine pairs end-to-end at n=500/2000/8000, so #5
        # adds ZERO incremental recovery while WEAKENING cross-mix rejection. #5 is structurally a
        # 4th MI-threshold lever and fails by construction like #1/#8/#19; do NOT re-attempt an
        # MI-threshold/ratio fix here. Numbers + verdict in D:/Temp/null_prev_results.md.
        _gate_ratio = (best_mi / pair_mi) if pair_mi > 0.0 else 0.0
        if fe_mm_debias_prevalence and pair_mi > 0.0 and best_config is not None:
            from ._pairs_gates import _occupied_k, mm_debiased_prevalence_ratio
            _n_rows = int(len(classes_y))
            _k_y = int(np.asarray(freqs_y).shape[0])
            # Engineered winner occupied-K: discretise its CONTINUOUS column (the buffer
            # column ``best_config[2]``) with the SAME quantiser the MI was scored under.
            _k_eng = quantization_nbins
            try:
                _win_codes = discretize_array(
                    arr=np.nan_to_num(final_transformed_vals[:, best_config[2]]),
                    n_bins=quantization_nbins, method=quantization_method, dtype=quantization_dtype,
                )
                _k_eng = _occupied_k(_win_codes)
            except Exception:
                _k_eng = quantization_nbins
            # 2-D joint occupied-K of the raw operands (bit-identical discretise to the
            # pair_mi compute); fall back to nominal ``nbins^2`` if either operand is
            # missing an identity transform.
            _ca = _operand_discretized(raw_vars_pair[0])
            _cb = _operand_discretized(raw_vars_pair[1])
            if _ca is not None and _cb is not None:
                _nb_b = int(np.asarray(_cb).max()) + 1 if np.asarray(_cb).size else quantization_nbins
                _joint_codes = np.asarray(_ca, dtype=np.int64) * _nb_b + np.asarray(_cb, dtype=np.int64)
                _k_joint = _occupied_k(_joint_codes)
            else:
                _k_joint = quantization_nbins * quantization_nbins
            _gate_ratio = mm_debiased_prevalence_ratio(
                best_mi, pair_mi, k_eng=_k_eng, k_joint=_k_joint, k_y=_k_y, n=_n_rows,
            )
        _passes_joint_gate = _gate_ratio > fe_min_engineered_mi_prevalence * (1.0 if num_fs_steps < 1 else 1.025)

        # Alternative pre-warp acceptance (2026-06-02): the joint-prevalence gate
        # structurally rejects a 1-D summary of a 2-D pair on a non-monotone inner
        # distortion. Admit the prewarp winner when it beats the best NON-prewarp
        # engineered MI by ``prewarp_uplift_threshold`` AND clears the pair-MI
        # noise floor (its MI must exceed the larger individual operand MI -- the
        # same notion the smart_polynom baseline uplift uses), so it cannot fire
        # on noise (where prewarp does not beat the library) or pure-linear data
        # (where the elementary library already saturates and the prewarp adds no
        # uplift). When it fires, the prewarp config becomes the winner.
        _prewarp_accept = False
        if (
            _prewarp_active
            and not _passes_joint_gate
            and best_prewarp_config is not None
            and best_nonprewarp_mi > 0.0
            and best_prewarp_mi >= best_nonprewarp_mi * float(prewarp_uplift_threshold)
        ):
            _prewarp_accept = True
            # Promote the prewarp winner to the pair's winner so the standard
            # leading-features / single-best materialisation path emits it.
            best_config, best_mi = best_prewarp_config, best_prewarp_mi
            if verbose:
                messages.append(
                    f"pre-warp uplift gate: best prewarp MI={best_prewarp_mi:.4f} "
                    f"beats best non-prewarp MI={best_nonprewarp_mi:.4f} by "
                    f">= {float(prewarp_uplift_threshold):.2f}x (joint-prevalence "
                    f"gate {best_mi / pair_mi:.3f} < {fe_min_engineered_mi_prevalence:.2f} "
                    f"would have rejected it); admitting the prewarp feature."
                )

        # MARGINAL-UPLIFT alternative acceptance: admit a pair the joint-prevalence
        # gate rejects when its best ELEMENTARY-LIBRARY (non-prewarp) engineered column
        # beats the LARGER individual operand marginal MI by ``_FE_MARGINAL_UPLIFT_MIN_RATIO``
        # AND still recovers at least ``_FE_MARGINAL_UPLIFT_MIN_JOINT_RATIO`` of the inflated
        # 2-D joint. Rationale + thresholds: see the module-level constants. Genuine synergy
        # pairs (a**2/b, log(c)*sin(d)) clear both; cross-pair artefacts that merely recapture
        # one operand's marginal fail the uplift bar, and structureless noise pairs never reach
        # here (the upstream pair screen + order-2 maxT floor remove them). Only fires when the
        # primary joint gate AND the prewarp path both declined, so it is purely additive recall
        # for genuine pairs the strict joint bar drops. We score + promote the best NON-PREWARP
        # winner: the prewarp pseudo-unary has its own dedicated acceptance path above
        # (``_prewarp_accept``), and promoting a prewarp form here would require the per-operand
        # warp spec to round-trip into the recipe -- which is only guaranteed on the prewarp path.
        _marginal_uplift_accept = False
        if (
            not _passes_joint_gate
            and not _prewarp_accept
            and best_nonprewarp_config is not None
            and best_nonprewarp_mi > 0.0
            and pair_mi > 0.0
        ):
            _max_operand_marginal = max(
                _operand_marginal_mi(raw_vars_pair[0]),
                _operand_marginal_mi(raw_vars_pair[1]),
            )
            _joint_ratio = best_nonprewarp_mi / pair_mi
            _uplift_ratio = (best_nonprewarp_mi / _max_operand_marginal) if _max_operand_marginal > 0.0 else 0.0
            # HW-robust two-tier joint-recovery floor (see the constants above): a genuine
            # same-signal pair clears EITHER the strict joint floor on its own OR is a clear-synergy
            # pair (high uplift) that clears the relaxed base floor. A cross-signal artefact clears
            # neither, so a small cross-HW MI perturbation cannot flip it into the support.
            _joint_recovery_ok = (
                _joint_ratio >= _FE_MARGINAL_UPLIFT_STRICT_JOINT_RATIO
                or (
                    _uplift_ratio >= _FE_MARGINAL_UPLIFT_SYNERGY_UPLIFT
                    and _joint_ratio >= _FE_MARGINAL_UPLIFT_MIN_JOINT_RATIO
                )
            )
            if (
                _max_operand_marginal > 0.0
                and best_nonprewarp_mi >= _max_operand_marginal * _FE_MARGINAL_UPLIFT_MIN_RATIO
                and _joint_recovery_ok
            ):
                _marginal_uplift_accept = True
                # Promote the best non-prewarp form so the standard single-best
                # materialisation path emits a recipe-replayable winner.
                best_config, best_mi = best_nonprewarp_config, best_nonprewarp_mi
                if verbose:
                    messages.append(
                        f"marginal-uplift gate: best non-prewarp engineered MI={best_nonprewarp_mi:.4f} "
                        f"beats the larger operand marginal MI={_max_operand_marginal:.4f} by "
                        f">= {_FE_MARGINAL_UPLIFT_MIN_RATIO:.2f}x and recovers {_joint_ratio:.3f} "
                        f"of the 2-D joint (joint-prevalence gate "
                        f"{fe_min_engineered_mi_prevalence:.2f} would have rejected it); "
                        f"admitting the genuine synergy pair."
                    )

        # NOISE-WRAP CORR-COLLAPSE VETO (2026-06-15). Whatever path admitted the winner, VETO it when the
        # winning composite WRAPS a strong, clean operand with a (near-)noise operand: its |corr| with the
        # target collapses to a small fraction of the best single operand's |corr| while that operand is
        # genuinely strong on its own. This is the ``sub(log(e),invqubed(a__T2))`` failure -- an extreme
        # heavy-tailed transform inflates the binned ``best_mi/pair_mi`` so it clears the joint-prevalence
        # gate, yet the column carries ~0 linear/monotone signal (|corr|~0.02) versus the clean operand's
        # |corr|~0.99, so it would DISPLACE the clean univariate basis from the support and kill recovery.
        # Genuine synergy (a*b, log(c)*sin(d)) keeps the engineered column tracking y (no collapse), so the
        # wide 2x fraction margin never condemns it. Pure-noise pairs never reach here (upstream screens).
        if (
            (_passes_joint_gate or _prewarp_accept or _marginal_uplift_accept)
            and _corr_y_cont is not None
            and best_config is not None
        ):
            try:
                _win_vals = final_transformed_vals[:, best_config[2]] if final_transformed_vals is not None else None
                _win_corr = _safe_abs_corr(_win_vals) if _win_vals is not None else None
                # Compare against the strongest CLEAN per-operand column the winner actually used: each operand
                # under its CHOSEN unary (``sqr(a)`` for the ``a`` side, not raw ``a`` -- raw ``a`` is ~0 corr
                # for an even target like ``exp(-a**2)``), falling back to the raw operand value. This is the
                # genuine single-source signal the wrap is diluting.
                _op_corr = 0.0
                _tp = best_config[0]
                for _side in (0, 1):
                    _opk = _tp[_side] if isinstance(_tp, (tuple, list)) and len(_tp) > _side else None
                    if _opk is not None and _opk in vars_transformations:
                        _op_corr = max(_op_corr, _safe_abs_corr(transformed_vars[:, vars_transformations[_opk]]))
                    _op_corr = max(_op_corr, _safe_abs_corr(_extval_raw_col(raw_vars_pair[_side])))
                if (
                    _win_corr is not None
                    and _op_corr >= _NOISE_WRAP_MIN_OPERAND_CORR
                    and _win_corr < _op_corr * _NOISE_WRAP_CORR_COLLAPSE_FRAC
                ):
                    _passes_joint_gate = _prewarp_accept = _marginal_uplift_accept = False
                    if verbose:
                        messages.append(
                            f"noise-wrap corr-collapse veto: winning composite |corr| with target "
                            f"{_win_corr:.3f} collapsed below {_NOISE_WRAP_CORR_COLLAPSE_FRAC:.2f}x the "
                            f"best operand |corr| {_op_corr:.3f}; the pair wraps a clean strong operand with "
                            f"a near-noise operand (binned-MI inflated by an extreme transform) -- rejecting "
                            f"so it cannot displace the clean operand."
                        )
            except Exception:
                pass

        # REJECTION LEDGER (additive): record a pair the per-pair acceptance gate is about
        # to DROP -- the joint-prevalence floor declined AND both the prewarp and the
        # marginal-uplift (abs-MAD / joint-recovery) fallbacks declined. Attribute to whichever
        # floor it primarily missed: the engineered-MI prevalence floor (the 0.97 floor the
        # session hand-diagnoses) is the primary gate; if the ratio DID clear that bar (so the
        # prewarp/uplift path declined for another reason) tag the marginal-uplift floor.
        # All values were already computed above (no recompute).
        if not (_passes_joint_gate or _prewarp_accept or _marginal_uplift_accept):
            try:
                _rej_thr = float(fe_min_engineered_mi_prevalence) * (1.0 if num_fs_steps < 1 else 1.025)
                _rej_op = None
                if best_config is not None:
                    try:
                        _rej_op = best_config[1]  # binary func name of the best engineered form
                    except Exception:
                        _rej_op = None
                if not _passes_joint_gate:
                    _rej_rec = {
                        "gate": "engineered_mi_prevalence",
                        "candidate": str(raw_vars_pair),
                        "operands": tuple(raw_vars_pair),
                        "operator": _rej_op,
                        "observed": float(_gate_ratio),
                        "threshold": _rej_thr,
                        "reason": "best_mi_over_pair_mi_below_floor",
                    }
                else:
                    _rej_rec = {
                        "gate": "marginal_uplift_floor",
                        "candidate": str(raw_vars_pair),
                        "operands": tuple(raw_vars_pair),
                        "operator": _rej_op,
                        "observed": float(_gate_ratio),
                        "threshold": _rej_thr,
                        "reason": "marginal_uplift_and_prewarp_declined",
                    }
                _rejection_records.append(_rej_rec)
                if rejection_ledger_out is not None:
                    rejection_ledger_out.append(_rej_rec)
            except Exception:
                pass

        if _passes_joint_gate or _prewarp_accept or _marginal_uplift_accept:  # Best transformation is good enough

            # If there is a group of leaders with almost the same performance, approve them through one of the other variables.
            # если будут возникать такие группы примерно одинаковых по силе лидеров, их придётся разрешать с помощью одного из других влияющих факторов
            # When the pair was admitted ONLY via the marginal-uplift path (the joint /
            # prewarp gates declined), the winner MUST be a non-prewarp form so the recipe
            # is replayable -- restrict the leaders to elementary-library configs.
            _restrict_to_nonprewarp = _marginal_uplift_accept and not (_passes_joint_gate or _prewarp_accept)
            leading_features = []
            for next_config, next_mi in sort_dict_by_value(var_pairs_perf).items():
                if next_mi > best_mi * fe_good_to_best_feature_mi_threshold:
                    if _restrict_to_nonprewarp and (
                        next_config[0][0][1] == _PREWARP_UNARY or next_config[0][1][1] == _PREWARP_UNARY
                    ):
                        continue
                    leading_features.append(next_config)

            # LINEAR-USABILITY TIE-BREAK over the MI-leaders (2026-06-16). MI is a RANK
            # statistic blind to linear usability, so a raw pair's leading-features
            # equivalence class can hold forms with IDENTICAL target MI but wildly
            # different linear usability (canonical: on a ``y=1.5*a*b`` bilinear target the
            # forms ``mul(a,b)``, ``log(a)+log(b)`` and ``1/(a**2*b**2)`` are ALL strictly-
            # monotone in ``a*b`` -> bit-identical binned MI 0.4561, but |corr(y)| 0.76 /
            # 0.61 / 0.004). Pre-fix ``_select_single_best`` broke the MI-tie by extval-MI
            # then NAME, so a linearly-useless inverse-square form could win and cap the
            # downstream LINEAR model (test-R2 0.884 < 0.90 floor). Score each leader's
            # |corr(continuous y)| from its materialised column so the tie-break prefers the
            # linearly-usable leg (the project's "prefer the linearly-usable member" rule).
            # Tie-break is gated on EQUAL MI inside _select_single_best, so it never overrides
            # a higher-MI form; trees are rank-indifferent so this cannot hurt the tree list.
            _leader_usability: dict = {}
            if len(leading_features) > 1 and _corr_y_cont is not None:
                for _lc in leading_features:
                    try:
                        _li = _lc[2]
                        _lvals = final_transformed_vals[:, _li] if final_transformed_vals is not None else None
                        if _lvals is not None:
                            _leader_usability[_lc] = _safe_abs_corr(_lvals)
                    except Exception:
                        continue

            if len(leading_features) > 1:
                if len(numeric_vars_to_consider) > 2:

                    if verbose > 2:
                        print(f"Taking {len(leading_features)} new features for a separate validation step!")

                    # Test all candidates as-is against the rest of the approved factors (also as-is). Candidates significantly outstanding (in terms of MI with target)
                    # against any other approved factor are kept.
                    valid_pairs_perf = {}
                    # LAZY EXTERNAL VALIDATION (2026-06-06): valid_pairs_perf feeds _select_single_best ONLY as the
                    # SECONDARY tie-break, decisive solely among leaders whose PRIMARY (target) MI is EXACTLY equal.
                    # The external loop below (all external_factors x binary_funcs x per-candidate discretize +
                    # mi_direct) was the single-threaded FE hotspot (py-spy). Run it ONLY for the leaders tied at the
                    # max primary MI; a unique top leader wins outright with no external work. Bit-identical: a
                    # lower-primary leader can never win the (primary, secondary, name) max key regardless of its
                    # (uncomputed) secondary.
                    _lead_primary = {c: var_pairs_perf[c] for c in leading_features if c in var_pairs_perf}
                    _max_primary = max(_lead_primary.values()) if _lead_primary else None
                    _ev_configs = [c for c, _m in _lead_primary.items() if _m == _max_primary] if _max_primary is not None else []
                    for transformations_pair, bin_func_name, i in (_ev_configs if len(_ev_configs) > 1 else []):
                        if final_transformed_vals is not None:
                            param_a = final_transformed_vals[:, i]
                        else:
                            # CRITICAL #2 recompute-fallback: rebuild the survivor column from its
                            # (a_key, b_key, bin_func_name) metadata. transformed_vars is small
                            # (deduped unary table); the bin_func call is cheap (one ufunc).
                            _a_key, _b_key, _bin_name = _config_by_i[i]
                            _pa = transformed_vars[:, vars_transformations[_a_key]]
                            _pb = transformed_vars[:, vars_transformations[_b_key]]
                            param_a = binary_transformations[_bin_name](_pa, _pb)
                            np.nan_to_num(param_a, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

                        best_valid_mi = -1
                        config = (transformations_pair, bin_func_name, i)

                        # ``sorted`` first: a bare ``set`` difference iterates in hash order, which is
                        # PYTHONHASHSEED-randomised for str keys, so the candidate order (and hence the
                        # sampled subset) would differ across processes / fits. Sort to a stable order,
                        # then sample with the instance-seeded ``_rng_extval`` so the chosen validation
                        # factors are fully reproducible from the MRMR seed.
                        external_factors = sorted(set(numeric_vars_to_consider) - set(raw_vars_pair))
                        if fe_max_external_validation_factors and len(external_factors) > fe_max_external_validation_factors:
                            external_factors = _rng_extval.choice(external_factors, fe_max_external_validation_factors, replace=False)

                        # BATCHED EXTERNAL VALIDATION (2026-06-07): the per-(external_factor x
                        # valid_bin_func) ``discretize_array`` + ``mi_direct`` double loop was the
                        # single dominant serial FE hotspot at wide p (call-site profile on scene
                        # 2407x299: 228k discretize_array + 228k mi_direct here, ~80% of fit wall;
                        # CPU near-idle => GIL-bound per-candidate dispatch). ``best_valid_mi`` is a
                        # pure ``max`` over an order-INDEPENDENT per-candidate MI, and every
                        # candidate is scored against the SAME y with the SAME estimator the per-pair
                        # sweep already batches, so we materialise ALL candidate columns into one
                        # buffer, run ONE ``discretize_2d_quantile_batch`` + ONE
                        # ``_dispatch_batch_mi_with_noise_gate`` (CPU njit / GPU by size), then take
                        # the max. BIT-IDENTICAL to the loop on the default FE path
                        # (``parallelism='outer'``, ``n_workers=1``, ``base_seed=0``,
                        # ``npermutations=fe_npermutations<32`` so no GPU permutation route) -- the
                        # batch kernel shuffles y once per permutation and scores all columns against
                        # it, exactly matching the per-candidate ``mi_direct`` noise-gate. Only the
                        # ``quantile`` method is batched (matches ``discretize_2d_quantile_batch``'s
                        # bit-identity domain); any other method falls back to the per-candidate
                        # loop below.
                        _ev_param_bs = []
                        for external_factor in external_factors:
                            # Memoised raw-values extract (LEVER 1): one extraction
                            # per distinct external factor for the whole call, reused
                            # across every config + raw pair. ``None`` => factor not in
                            # ``original_cols`` -> skip (identical to the prior guard).
                            _pb_vals = _extval_raw_col(external_factor)
                            if _pb_vals is None:
                                continue
                            _ev_param_bs.append(_pb_vals)

                        # Memory guard: the batch buffer is (n_rows x ext_factors*n_binary)
                        # float64. On the common wide-but-shallow bed (e.g. scene 2407x299:
                        # ~1680 cols -> 32 MB) this is trivial, but an unbounded ext-factor set
                        # on a multi-million-row frame could OOM. Reuse the SAME available-RAM
                        # budget the shared-buffer hoist uses; if the batch buffer would not fit,
                        # fall back to the (bit-identical) per-candidate loop below.
                        _ev_n_bin = len(binary_transformations)
                        _ev_buf_bytes = len(X) * max(1, len(_ev_param_bs)) * _ev_n_bin * 8
                        # LARGE-N FIX (2026-06-08): this float64 ext-val buffer coexists with the
                        # chunk/disc/MI buffers and is allocated per concurrent worker, so use the
                        # SAME overhead+worker-aware envelope as the candidate buffer above.
                        _ev_can_batch, _, _ = _can_hoist_shared_buffer(_ev_buf_bytes, n_workers=_n_workers)
                        if quantization_method == "quantile" and _ev_param_bs and _ev_can_batch:
                            _ev_bin_funcs = list(binary_transformations.values())
                            _ev_K = len(_ev_param_bs) * len(_ev_bin_funcs)
                            # float64 buffer: the per-candidate path discretises the RAW
                            # ``valid_bin_func(...)`` output (numpy bin_funcs return float64) with
                            # NO nan_to_num -- ``discretize_array``/``discretize_2d_quantile_batch``
                            # both bin via ``np.nanpercentile`` (NaN-ignoring edges) + per-column
                            # ``searchsorted`` (NaN -> rightmost bin), identically. Writing into a
                            # float64 buffer (not float32) preserves the bin_func's native precision
                            # so the percentile edges match the 1-D path to the bit.
                            _ev_buf = np.empty((len(X), _ev_K), dtype=np.float64)
                            _ev_op_codes = _njit_binary_op_codes(binary_transformations)
                            if _ev_op_codes is not None:
                                # NJIT materialise: ALL (ext x op) candidate columns in one nogil
                                # kernel (bit-identical to the numpy bin_funcs; see
                                # ``_materialise_extval_njit``). Column order ext-outer/op-inner ==
                                # the numpy ``for ext: for bin_func`` order, so the discretise +
                                # MI + max reduction below is unchanged. ``param_a`` may be a
                                # float32 buffer slice; the kernel upcasts per-element to float64.
                                # bench-attempt-rejected (2026-06-07): "drop the _ev_pb_mat repack"
                                # (Q7). The external-factor columns are DISTINCT memoised arrays
                                # (_extval_raw_col per var) so they genuinely must be assembled into
                                # a 2-D matrix for the njit kernel; there is no view to substitute.
                                # This per-column-assign loop is already the fastest assembly
                                # (n_ext=50/150/300: 0.68/1.39/3.13ms vs np.column_stack
                                # 0.96/2.10/3.98ms) and is a tiny fraction of the per-call kernel +
                                # discretise + MI cost (never appears in the scene sampler top-30).
                                # No actionable speedup; kept as-is.
                                _ev_pb_mat = np.empty((len(X), len(_ev_param_bs)), dtype=np.float64)
                                for _ei, _pb_vals in enumerate(_ev_param_bs):
                                    _ev_pb_mat[:, _ei] = _pb_vals
                                _materialise_extval_njit(
                                    np.ascontiguousarray(param_a), _ev_pb_mat, _ev_op_codes,
                                    _ev_buf[:, :_ev_K],
                                )
                                _ev_col = _ev_K
                            else:
                                # NUMPY FALLBACK: a bin_func is not njit-coded (maximal-preset
                                # special) -> materialise per-candidate with the exact numpy ufuncs.
                                _ev_col = 0
                                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                                    for _pb_vals in _ev_param_bs:
                                        for valid_bin_func in _ev_bin_funcs:
                                            _ev_buf[:, _ev_col] = valid_bin_func(param_a, _pb_vals)
                                            _ev_col += 1
                            _ev_disc = discretize_2d_quantile_batch(
                                _ev_buf[:, :_ev_col], n_bins=quantization_nbins,
                                dtype=_narrow_code_dtype(quantization_nbins, quantization_dtype),  # OPT-B narrow codes
                                # OPT-A extension (2026-06-07): the marginal-uplift gate's
                                # discretise ran the SERIAL searchsorted kernel on the main
                                # thread (post-OPT-D the top sampler hotspot, ~21% of fit) while
                                # the other cores sat idle. ``check_prospective_fe_pairs`` carries
                                # ``serial_main_thread`` down from _mrmr_fe_step's ``len(X)<50000``
                                # dispatch, so the same OPT-A predicate that already gates the
                                # main chunk's discretise (line ~907) safely selects the
                                # byte-identical column-prange twin here too (no joblib nest).
                                parallel=_fe_use_parallel_kernels(_ev_col, serial_main_thread),
                            )
                            _ev_mi = _dispatch_batch_mi_with_noise_gate(
                                disc_2d=_ev_disc,
                                quantization_nbins=quantization_nbins,
                                classes_y=classes_y,
                                classes_y_safe=classes_y_safe,
                                freqs_y=freqs_y,
                                npermutations=fe_npermutations,
                                min_nonzero_confidence=fe_min_nonzero_confidence,
                                use_su=use_su_normalization(),
                                batch_mi_kernel=batch_mi_with_noise_gate,
                            )
                            if _ev_mi is not None and len(_ev_mi):
                                best_valid_mi = float(np.max(_ev_mi))
                        else:
                            for _pb_vals in _ev_param_bs:
                                param_b = _pb_vals
                                for valid_bin_func_name, valid_bin_func in binary_transformations.items():

                                    valid_vals = valid_bin_func(param_a, param_b)

                                    discretized_transformed_values = discretize_array(
                                        arr=valid_vals, n_bins=quantization_nbins, method=quantization_method, dtype=quantization_dtype
                                    )
                                    fe_mi, fe_conf = mi_direct(
                                        discretized_transformed_values.reshape(-1, 1),
                                        x=np.array([0], dtype=np.int64),
                                        y=None,
                                        factors_nbins=np.array([quantization_nbins], dtype=np.int64),
                                        classes_y=classes_y,
                                        classes_y_safe=classes_y_safe,
                                        freqs_y=freqs_y,
                                        min_nonzero_confidence=fe_min_nonzero_confidence,
                                        npermutations=fe_npermutations,
                                    )

                                    if fe_mi > best_valid_mi:
                                        best_valid_mi = fe_mi

                        valid_pairs_perf[config] = best_valid_mi

                    # ONE-BEST-PER-PAIR (2026-06-01): the leading-features
                    # equivalence class holds many near-identical representations
                    # of the same algebraic target (a**2/b == div(sqr(a),b) ==
                    # mul(sqr(a),reciproc(b)) == div(a,sqrt(b)) ...). The
                    # pre-refactor code materialised EXACTLY ONE per raw pair;
                    # the refactor regressed to emitting the whole class (~15
                    # cols on the canonical fixture). Pick the single best by
                    # TARGET MI (``var_pairs_perf`` -- the primary objective),
                    # using the external-validation MI (``valid_pairs_perf``)
                    # only as a tie-break among target-MI-equal leaders. (Prior
                    # bug: selected by external-validation MI alone, discarding
                    # the true max-target-MI form -- e.g. picking add(log(c),1/d)
                    # MI=0.25 over the true mul(log(c),sin(d)) MI=0.32.)
                    _primary_perf = {c: var_pairs_perf[c] for c in leading_features if c in var_pairs_perf}
                    _winner = _select_single_best(_primary_perf, cols, secondary=valid_pairs_perf,
                                                  usability=_leader_usability)
                    if _winner is not None:
                        new_feature_name = get_new_feature_name(fe_tuple=_winner, cols_names=cols)
                        if verbose:
                            messages.append(
                                f"{new_feature_name} is recommended to use as a new feature! (won in validation with other factors) best_mi={best_mi:.4f}, pair_mi={pair_mi:.4f}, rat={best_mi/pair_mi:.4f}"
                            )
                        this_pair_features.add((_winner, 0))
                else:
                    # Can't narrow by external validation (only 2 vars total) --
                    # still emit ONE best representative (highest engineered MI,
                    # deterministic name tie-break) rather than the whole class.
                    _lead_perf = {c: var_pairs_perf[c] for c in leading_features if c in var_pairs_perf}
                    _winner = _select_single_best(_lead_perf, cols, usability=_leader_usability)
                    if _winner is not None:
                        if verbose:
                            messages.append(
                                f"{get_new_feature_name(fe_tuple=_winner, cols_names=cols)} is recommended to use as a new feature! (best of {len(leading_features)} near-equivalent leaders) best_mi={best_mi:.4f}, pair_mi={pair_mi:.4f}, rat={best_mi/pair_mi:.4f}"
                            )
                        this_pair_features.add((_winner, 0))
            else:
                new_feature_name = get_new_feature_name(fe_tuple=best_config, cols_names=cols)
                if verbose:
                    messages.append(
                        f"{new_feature_name} is recommended to use as a new feature! (clear winner) best_mi={best_mi:.4f}, pair_mi={pair_mi:.4f}, rat={best_mi/pair_mi:.4f}"
                    )
                j = 0
                this_pair_features.add((best_config, j))

            # MULTI-CANDIDATE DIVERSE EMISSION (2026-06-12): the blocks above emit the
            # single MAX-MI engineered form. MI is rank-based and blind to LINEAR usability,
            # so the MI-winner can be a tree-friendly monotone warp that a linear model
            # cannot use, while a lower-MI form is the linearly-aligned one (F2:
            # sub(exp(c),cbrt(d)) MI 0.288 vs the linearly-usable mul(log(c),sin(d)) MI 0.264).
            # When ``fe_multi_emit_max_per_pair > 1`` additionally emit the next DISTINCT
            # forms by target MI (skip any whose continuous values correlate above
            # ``fe_multi_emit_diversity_corr`` with an already-emitted column, down to
            # ``fe_multi_emit_mi_floor`` x best_mi) so both survive; the downstream MRMR
            # redundancy gate prunes residual overlap. Purely additive: never emits FEWER
            # than the single-best path, byte-identical when max_per_pair == 1.
            if (
                int(fe_multi_emit_max_per_pair) > 1
                and final_transformed_vals is not None
                and this_pair_features
                and best_mi > 0
            ):
                _emit_floor = float(best_mi) * float(fe_multi_emit_mi_floor)
                _div_corr = float(fe_multi_emit_diversity_corr)
                _already = {c for c, _ in this_pair_features}
                _emitted_cols = []
                for _c in _already:
                    try:
                        _emitted_cols.append(np.asarray(final_transformed_vals[:, _c[2]], dtype=np.float64))
                    except Exception:
                        pass
                for _cfg, _cfg_mi in sort_dict_by_value(var_pairs_perf).items():
                    if len(this_pair_features) >= int(fe_multi_emit_max_per_pair):
                        break
                    if _cfg_mi < _emit_floor:
                        break  # sorted desc: nothing below the floor remains
                    if _cfg in _already:
                        continue
                    try:
                        _col = np.asarray(final_transformed_vals[:, _cfg[2]], dtype=np.float64)
                    except Exception:
                        continue
                    _col = np.nan_to_num(_col, nan=0.0, posinf=0.0, neginf=0.0)
                    if float(np.std(_col)) <= 1e-9:
                        continue
                    # DIVERSITY: skip a near-duplicate of any already-emitted column.
                    _dup = False
                    for _ec in _emitted_cols:
                        if float(np.std(_ec)) <= 1e-9:
                            continue
                        _r = np.corrcoef(_col, _ec)[0, 1]
                        if np.isfinite(_r) and abs(_r) > _div_corr:
                            _dup = True
                            break
                    if _dup:
                        continue
                    this_pair_features.add((_cfg, 0))
                    _already.add(_cfg)
                    _emitted_cols.append(_col)
                    if verbose:
                        messages.append(
                            f"{get_new_feature_name(fe_tuple=_cfg, cols_names=cols)} also emitted "
                            f"(diverse multi-candidate, MI={_cfg_mi:.4f} vs best {best_mi:.4f})"
                        )

            transformed_vals, new_cols, new_nbins = None, None, None

            if this_pair_features:

                # Bulk add the found & checked best features.
                # ``this_pair_features`` is a SET of (config, j) tuples
                # with sparse, non-contiguous ``j`` indices into
                # ``final_transformed_vals``. The consumer (mrmr.py
                # ``_run_fe_step``) iterates
                # ``for k in range(len(this_pair_features)):
                # transformed_vals[:, k]``, so the buffer MUST have
                # exactly ``len(this_pair_features)`` columns packed
                # densely 0..N-1, not the sparse ``j``-indexed layout
                # with holes. Pre-fix code wrote to ``transformed_vals[:, j]``
                # then sliced to ``[:, :last_j + 1]`` -- this gives
                # either a too-short buffer (if last_j was small) and
                # IndexError downstream, or holes (if last_j was large).
                # Pack each (config, j) into a compact column index
                # ``idx = 0..len(this_pair_features)-1`` instead.
                #
                # 2026-06-01 (ROOT CAUSE 5 fix): materialise the survivor
                # columns whenever FE runs (``fe_max_steps >= 1``), not only on
                # multi-step (``> 1``). Previously, with the default
                # ``fe_max_steps=1`` the recommended features were LOGGED but
                # ``transformed_vals`` stayed ``None`` -- so the consumer
                # (_mrmr_fe_step) had nothing to append, the columns never
                # entered ``data``/``selected_vars``, and ``_engineered_features_``
                # stayed empty. Producing the buffer unconditionally lets the
                # single-step default actually emit engineered columns.
                # Materialise each survivor into a temp column FIRST, then apply
                # the NON-CONSTANT guard (2026-06-01): a column that replays as
                # constant (std<=1e-9) or non-finite is a DEAD feature and must
                # never be appended -- it reaches the downstream model carrying
                # zero variance. Several div(sqr(a),b)-family combos replayed
                # constant on the canonical fixture (degenerate quantile binning
                # of the heavy-tailed a**2/b). One-best-per-pair already keeps the
                # non-constant MI winner; this guard is defence-in-depth and also
                # compacts ``this_pair_features`` / buffers so the recipe builder
                # downstream never constructs a recipe for a dropped column.
                _kept_configs = []   # list[(config, j)] that survived the guard
                _kept_cols_vals = []  # list[np.ndarray] aligned with _kept_configs
                _kept_names = []

                for idx, (config, j) in enumerate(this_pair_features):
                    new_feature_name = get_new_feature_name(fe_tuple=config, cols_names=cols)
                    transformations_pair, bin_func_name, i = config

                    if fe_max_steps >= 1:
                        if _use_subsample:
                            # SUBSAMPLE path: rebuild from raw _X_full so the survivor column
                            # carries the FULL n rows the caller expects (mrmr.py appends it
                            # back to its full-n ``data`` array). The MI sweep used a 200k
                            # subset; the survivor IDENTITIES are correct (bench shows
                            # jaccard=1.0 vs full-n at n_eff>=50k), so we just need to
                            # rematerialise the values at full resolution.
                            _col_full = _rebuild_full_survivor_col(
                                config, _X_full, original_cols,
                                unary_transformations, binary_transformations,
                                prewarp_spec_by_var=_prewarp_spec_by_var,
                                gate_med_median_by_var=_gate_med_median_by_var,
                                cols=cols,
                                engineered_operand_values=engineered_operand_values,
                            )
                        elif final_transformed_vals is not None:
                            _col_full = final_transformed_vals[:, i]
                        else:
                            # CRITICAL #2 recompute-fallback (no subsample, tight RAM): rebuild
                            # the survivor column from its (a_key, b_key, bin_func_name)
                            # metadata via the cached unary table. transformed_vars is at
                            # full n in this path so the column lands at full n directly.
                            _a_key, _b_key, _bin_name = _config_by_i[i]
                            _pa = transformed_vars[:, vars_transformations[_a_key]]
                            _pb = transformed_vars[:, vars_transformations[_b_key]]
                            _col = binary_transformations[_bin_name](_pa, _pb)
                            np.nan_to_num(_col, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                            _col_full = _col

                        # Keep the RAW (float) engineered values, scrubbed of
                        # nan/inf. CRITICAL (2026-06-02): do NOT cast to the
                        # integer ``quantization_dtype`` here. ``transformed_vals``
                        # feeds two consumers downstream -- (a) ``_mrmr_fe_step``
                        # discretises it via ``discretize_array(method=quantile)``
                        # into the ``data`` bin-code matrix, and (b) the recipe
                        # builder computes its quantile EDGES from these values for
                        # leak-safe replay. A premature int cast TRUNCATES the
                        # heavy-tailed engineered values (e.g. mul(log(c),sin(d)) in
                        # (-inf,0] collapses to ~2 integers), so the subsequent
                        # quantile binning sees only 2-3 distinct values and the
                        # column reaches the model with a fraction of its MI
                        # (measured: 0.14 vs the true 0.32). Keeping float lets the
                        # downstream quantile discretiser produce the full nbins
                        # codes and the recipe pin correct edges.
                        _col_arr = np.nan_to_num(
                            np.asarray(_col_full, dtype=np.float64),
                            nan=0.0, posinf=0.0, neginf=0.0,
                        )
                        if float(np.std(_col_arr)) <= 1e-9:
                            if verbose:
                                messages.append(
                                    f"{new_feature_name} dropped at materialisation: dead column "
                                    f"(std={float(np.std(_col_arr)):.2e}, non-constant guard)."
                                )
                            continue
                        _kept_cols_vals.append(_col_arr)
                    _kept_configs.append((config, j))
                    _kept_names.append(new_feature_name)

                # Rebuild the survivor set / buffers from ONLY the kept columns so
                # the recipe builder and the downstream dense consumer stay aligned.
                this_pair_features = set(_kept_configs)
                new_cols = list(_kept_names)
                if fe_max_steps >= 1 and _kept_cols_vals:
                    # float buffer: holds RAW engineered values (discretised to
                    # codes downstream; see the non-constant-guard comment above).
                    transformed_vals = np.empty(shape=(_full_n_rows, len(_kept_cols_vals)), dtype=np.float64)
                    for _ci, _cv in enumerate(_kept_cols_vals):
                        transformed_vals[:, _ci] = _cv
                    new_nbins = [quantization_nbins] * len(_kept_cols_vals)
                else:
                    transformed_vals, new_nbins = None, []

            res[raw_vars_pair] = (this_pair_features, transformed_vals, new_cols, new_nbins, messages)

        # Live progress: surface the best engineered feature found so far in this sweep
        # (its MI with y) plus the pair just evaluated, on the "pair" bar. ``best_mi`` /
        # ``best_config`` are already computed for this pair -- no extra MI compute. Robust
        # to the no-config / NaN edge cases (we only adopt a finite, improving best_mi).
        if verbose:
            try:
                _bm = float(best_mi)
                if best_config is not None and np.isfinite(_bm) and _bm > _sweep_best_mi:
                    _sweep_best_mi = _bm
                    _sweep_best_name = get_new_feature_name(fe_tuple=best_config, cols_names=cols)
                _cur_pair = f"{cols[raw_vars_pair[0]]},{cols[raw_vars_pair[1]]}"
                _pf = {"pair": _short_fe_name(_cur_pair, 22)}
                if _sweep_best_name is not None:
                    _pf["best"] = f"{_short_fe_name(_sweep_best_name)}={_sweep_best_mi:.4f}"
                pair_pbar.set_postfix(_pf, refresh=False)
            except (TypeError, ValueError, IndexError):
                pass

    # Surface the fitted per-operand pre-warp specs (keyed by cols-space var
    # index) so the caller (``_mrmr_fe_step``) can persist them in each survivor
    # recipe for leak-safe replay. Only the non-None specs that were actually
    # fitted are exported. We populate BOTH the optional ``prewarp_specs_out``
    # side-channel (works for the in-process serial path) AND a reserved key in
    # the returned ``res`` (survives the loky-parallel path where the side
    # channel dict cannot be mutated cross-process; the caller merges per-chunk
    # results). The reserved key is a private 3-tuple that can never collide with
    # a real ``raw_vars_pair`` (which is always length 2).
    _fitted_specs = {_v: _s for _v, _s in _prewarp_spec_by_var.items() if _s is not None}
    if _fitted_specs:
        if prewarp_specs_out is not None:
            prewarp_specs_out.update(_fitted_specs)
        res[_PREWARP_SPECS_RESULT_KEY] = _fitted_specs

    # Same dual-channel export for the fitted per-operand TRAIN medians so the
    # caller can persist them in each survivor recipe for leak-safe replay. The
    # value is a single float per cols-space var index.
    _fitted_medians = {_v: float(_m) for _v, _m in _gate_med_median_by_var.items()}
    if _fitted_medians:
        if gate_med_specs_out is not None:
            gate_med_specs_out.update(_fitted_medians)
        res[_GATE_MED_SPECS_RESULT_KEY] = _fitted_medians

    # REJECTION LEDGER export via the reserved result key (survives the loky-parallel path).
    if _rejection_records:
        res[_FE_REJECTION_RESULT_KEY] = _rejection_records

    return res
