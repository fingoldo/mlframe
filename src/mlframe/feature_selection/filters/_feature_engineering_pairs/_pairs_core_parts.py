"""Helper blocks extracted from the over-long functions of ``_pairs_core`` (function-length gate); behaviour is unchanged."""
from __future__ import annotations
from ._pairs_core_helpers import _check_prospective__use_subsample, _check_prospective__behaviour_change_wrapped_experimental, _check_prospective__scores_same_rows_fall, _check_prospective__win_do_re_implement, _check_prospective__ignored_thread_multiplication_byte, _check_prospective__classes_codes_still_usable, _check_prospective__only_worth_chunking_least, _check_prospective__read_weakref_cache_no, _check_prospective__pair_res_entry_none, _check_prospective__no_config_nan_edge, _check_prospective__pair_scoped_copies_spec, _check_prospective__fitted_specs, _check_prospective__value_single_float_per
from ._pairs_core_steps import _check_prospective_fe_step1_def_extval_raw, _check_prospective_fe_step2_classes_codes_still, _check_prospective_fe_step3_gpu_clock_variance, _check_prospective_fe_step4_int_code_array
from ._pairs_chunks import _plan_fe_chunks
from ._pairs_dispatch import resolve_fe_dispatch_env_gate
from ._pairs_gates import (
    _FE_REJECTION_RESULT_KEY,
    _GATE_MED_UNARY,
    _PREWARP_UNARY,
    mi_tie_band,
)
from ._pairs_materialise import _njit_binary_op_codes
from ._pairs_setup import _build_operand_table, _fit_prewarp_and_gate_med


from typing import Any

import os

import numpy as np
from pandas.api.extensions import ExtensionDtype

# A column is near-constant when its std is within this factor of machine epsilon of its largest |value|: its spread is then rounding noise.
from types import SimpleNamespace as _SimpleNamespace


def _check_prospective_fe_pairs_part10(
    _extval_raw_col_cache: Any, original_cols: list[str], X: Any, allow_engineered_operands: bool, cols: list[str], engineered_operand_values: dict | None,
    _use_subsample: Any, _full_n_rows: Any, _sample_idx: Any, prospective_pairs: dict, prewarp_enable: bool, prewarp_y: np.ndarray | None,
    prewarp_y_continuous: np.ndarray | None, prewarp_basis: str, prewarp_max_degree: int, prewarp_min_val_corr: float, fe_gate_med_enable: bool, st: Any,
    unary_transformations: dict, verbose: int, logger: Any, binary_transformations: dict, concurrent_workers: int, gpu_compatible_unary_names: Any,
    usability_y_continuous: np.ndarray | None, classes_y: np.ndarray,
) -> tuple[Any, ...]:
    """Extracted block of ``check_prospective_fe_pairs`` (part 10); returns the names the caller still uses."""
    _extval_raw_col = _check_prospective_fe_pairs_part4(
        _extval_raw_col_cache, original_cols, X, allow_engineered_operands, cols, engineered_operand_values, _use_subsample, _full_n_rows, _sample_idx,
        prospective_pairs, prewarp_enable, prewarp_y, prewarp_y_continuous, prewarp_basis, prewarp_max_degree, prewarp_min_val_corr, fe_gate_med_enable, st,
    )

    transformed_vars, vars_transformations, _operand_marginal_mi_cache, _corr_y_cont, _corr_y_cont_finite, _safe_abs_corr = _check_prospective_fe_pairs_part2(
        unary_transformations, st, prospective_pairs, verbose, logger, X, binary_transformations, concurrent_workers, _extval_raw_col, cols,
        gpu_compatible_unary_names, usability_y_continuous, prewarp_y_continuous, classes_y, _use_subsample, _full_n_rows, _sample_idx,
    )
    return _extval_raw_col, transformed_vars, vars_transformations, _operand_marginal_mi_cache, _corr_y_cont, _corr_y_cont_finite, _safe_abs_corr


def _check_prospective_fe_pairs_part9(
    _pair_res_entry: Any, st: Any, raw_vars_pair: Any, best_config: Any, cols: list[str], pair_mi: Any, rejection_ledger_out: list | None, verbose: int,
    best_mi: Any, _sweep_best_mi: Any, pair_pbar: Any,
) -> None:
    """Extracted block of ``check_prospective_fe_pairs`` (part 9); returns the names the caller still uses."""
    _check_prospective__pair_res_entry_none(_pair_res_entry, st.res, raw_vars_pair, best_config, cols, pair_mi, st._rejection_records, rejection_ledger_out)

    # Live progress: surface the best engineered feature found so far in this sweep
    # (its MI with y) plus the pair just evaluated, on the "pair" bar. ``best_mi`` /
    # ``best_config`` are already computed for this pair - no extra MI compute. Robust
    # to the no-config / NaN edge cases (we only adopt a finite, improving best_mi).
    _check_prospective__no_config_nan_edge(verbose, best_mi, best_config, _sweep_best_mi, cols, raw_vars_pair, pair_pbar)


def _check_prospective_fe_pairs_part8(st: Any) -> None:
    """Extracted block of ``check_prospective_fe_pairs`` (part 8); returns the names the caller still uses."""
    # All futures are resolved by the pair loop in the normal case; an unconsumed prefetch -- an early
    # exit, or an exception -- is awaited here so the worker never outlives the buffers it holds.
    _pl_ex = st._chunk_state.pop("pipeline_ex", None)
    if _pl_ex is not None:
        _pl_ex.shutdown(wait=True)
    st._chunk_state.pop("pipeline_buffers", None)
    st._chunk_state.pop("pipeline_futures", None)


def _check_prospective_fe_pairs_part7(
    _extval_raw_col: Any, _safe_abs_corr: Any, vars_transformations: Any, transformed_vars: Any, st: Any, _operand_marginal_mi_cache: Any,
    quantization_nbins: int, quantization_method: str | None, quantization_dtype: type | str, classes_y: np.ndarray, classes_y_safe: np.ndarray,
    freqs_y: np.ndarray, fe_min_nonzero_confidence: float, fe_npermutations: int, prospective_pairs: dict, X: Any, verbose: int,
) -> tuple[Any, ...]:
    """Extracted block of ``check_prospective_fe_pairs`` (part 7); returns the names the caller still uses."""
    _raw_operand_abs_corr, _transformed_operand_abs_corr, _operand_marginal_mi, _operand_discretized = _check_prospective_fe_pairs_part1(
        _extval_raw_col, _safe_abs_corr, vars_transformations, transformed_vars, st, _operand_marginal_mi_cache, quantization_nbins, quantization_method,
        quantization_dtype, classes_y, classes_y_safe, freqs_y, fe_min_nonzero_confidence, fe_npermutations, prospective_pairs, X, verbose,
    )

    # CHUNK PIPELINE (max-GPU phase): overlap chunk k+1's GPU produce with chunk k's host
    # consume. A single-worker executor + a SECOND chunk buffer ride in ``chunk_state``; the lazy-load seam
    # in ``_score_one_pair`` awaits the prefetched future, then submits the next chunk into the alternate
    # buffer before replaying this chunk's pairs (depth-1 double buffer - the reused slot's pairs were fully
    # consumed one iteration earlier). Gated on the STRICT-resident path + >=2 chunks + the second buffer
    # actually fitting host RAM (a MemoryError just disables the pipeline); opt-out
    # MLFRAME_FE_PIPELINE_CHUNKS=0. The operand table's device mirror is pre-warmed so both threads only
    # READ the weakref cache (no first-call race). Selection identical: same compute, same chunk/pair order.
    _check_prospective__read_weakref_cache_no(
        st._chunk_global_batch, st._chunk_buffer, st._fe_chunks, X, st._chunk_buf_width, transformed_vars, st._chunk_state, verbose
    )
    return _raw_operand_abs_corr, _transformed_operand_abs_corr, _operand_marginal_mi, _operand_discretized


def _check_prospective_fe_pairs_part6(st: Any, prospective_pairs: dict, prewarp_specs_out: dict | None, gate_med_specs_out: dict | None) -> None:
    """Extracted block of ``check_prospective_fe_pairs`` (part 6); returns the names the caller still uses."""
    # Surface the fitted per-operand pre-warp specs (keyed by cols-space var
    # index) so the caller (``_mrmr_fe_step``) can persist them in each survivor
    # recipe for leak-safe replay. Only the non-None specs that were actually
    # fitted are exported. We populate BOTH the optional ``prewarp_specs_out``
    # side-channel (works for the in-process serial path) AND a reserved key in
    # the returned ``res`` (survives the loky-parallel path where the side
    # channel dict cannot be mutated cross-process; the caller merges per-chunk
    # results). The reserved key is a private 3-tuple that can never collide with
    # a real ``raw_vars_pair`` (which is always length 2).
    _fitted_specs = {_v: _s for _v, _s in st._prewarp_spec_by_var.items() if _s is not None}
    # Pair-scoped copies: the spec each prospective pair of THIS call actually materialised with (see _prewarp_pair_spec_key).
    _check_prospective__pair_scoped_copies_spec(prospective_pairs, st._prewarp_spec_by_var, _fitted_specs)
    _check_prospective__fitted_specs(_fitted_specs, prewarp_specs_out, st.res)

    # Same dual-channel export for the fitted per-operand TRAIN medians so the
    # caller can persist them in each survivor recipe for leak-safe replay. The
    # value is a single float per cols-space var index.
    _fitted_medians = {_v: float(_m) for _v, _m in st._gate_med_median_by_var.items()}
    _check_prospective__value_single_float_per(_fitted_medians, gate_med_specs_out, st.res)

    # REJECTION LEDGER export via the reserved result key (survives the loky-parallel path).
    if st._rejection_records:
        st.res[_FE_REJECTION_RESULT_KEY] = st._rejection_records


def _check_prospective_fe_pairs_part5(
    binary_transformations: dict, quantization_nbins: int, classes_y: np.ndarray, freqs_y: np.ndarray, prospective_pairs: dict,
) -> tuple[Any, ...]:
    """Extracted block of ``check_prospective_fe_pairs`` (part 5); returns the names the caller still uses."""
    # Sweep-wide leader for the live "pair" bar postfix: the best engineered MI found
    # so far across ALL pairs in this FE step + the feature that produced it. Both are
    # already computed per pair (``best_mi`` / ``best_config``) - display-only, no extra
    # MI work. Starts blank so the no-candidate / NaN edge case renders gracefully.
    _sweep_best_mi = -1.0
    _sweep_best_name = None

    # PER-CALL HOISTS (env-var gate + op-code table + MI tie-band): all three are invariant across
    # the WHOLE pair loop below (the env var never changes mid-call, the op-code table depends only
    # on ``binary_transformations``, and the tie-band depends only on quantization_nbins/classes_y/
    # freqs_y) - computing them once here instead of once per pair (or, for the op-code table, once
    # per tied-leader inside the emit tail) avoids millions of redundant env reads / dict rebuilds /
    # float divisions over a wide fit with no behaviour change (every downstream site consumed the
    # exact same values before this hoist). ``_op_code_arr_all`` is the UNGATED table (feeds the
    # external-validation njit materialise in the emit tail, which was never gated by the GPU-
    # materialise escape hatch); ``_op_code_arr`` additionally applies that gate (feeds the per-pair /
    # per-chunk GPU-fused materialise attempt, which WAS gated) - both derive from one table build.
    _gpu_mat_on = os.environ.get("MLFRAME_FE_GPU_MATERIALISE", "1").strip().lower() not in ("0", "false", "no", "off")
    _op_code_arr_all = _njit_binary_op_codes(binary_transformations)
    _op_code_arr = _op_code_arr_all if _gpu_mat_on else None
    _mi_band = mi_tie_band(int(quantization_nbins), len(classes_y), int(np.asarray(freqs_y).shape[0]))
    # Same rationale, for the noise-gate dispatcher's own CUDA-opt-out / resident-gate env reads (it runs
    # once per pair AND once per chunk AND once per ext-val tied-leader-set - see resolve_fe_dispatch_env_gate).
    _fe_env_gate = resolve_fe_dispatch_env_gate()

    # For every pair from the pool, try all known functions of 2 variables (not storing results in persistent RAM). Record best pairs.
    # Bonferroni across the pairs under test: each pair's family-wise test runs at alpha / (number of pairs), so the chance that any
    # noise pair passes stays near the 1% family level for the whole pool rather than per pair.
    _pair_family_alpha = 0.001 / max(1, len(prospective_pairs))
    return _sweep_best_mi, _gpu_mat_on, _op_code_arr_all, _op_code_arr, _mi_band, _fe_env_gate, _pair_family_alpha


def _check_prospective_fe_pairs_part4(
    _extval_raw_col_cache: Any, original_cols: list[str], X: Any, allow_engineered_operands: bool, cols: list[str], engineered_operand_values: dict | None,
    _use_subsample: Any, _full_n_rows: Any, _sample_idx: Any, prospective_pairs: dict, prewarp_enable: bool, prewarp_y: np.ndarray | None,
    prewarp_y_continuous: np.ndarray | None, prewarp_basis: str, prewarp_max_degree: int, prewarp_min_val_corr: float, fe_gate_med_enable: bool, st: Any,
) -> Any:
    """Extracted block of ``check_prospective_fe_pairs`` (part 4); returns the names the caller still uses."""
    def _densify_nullable(_arr):
        """Cast a pandas nullable extension array (Int64/Float64/boolean + pd.NA) to a plain
        float64 ndarray (pd.NA -> np.nan) so the numba/numpy unary-transform kernels can type it.
        No-op for ordinary numpy float64/int64 input - only ExtensionArrays are converted."""
        if isinstance(getattr(_arr, "dtype", None), ExtensionDtype):
            return _arr.to_numpy(dtype=np.float64, na_value=np.nan)
        return _arr

    _extval_raw_col = _check_prospective_fe_step1_def_extval_raw(
        _extval_raw_col_cache,
        original_cols,
        X,
        _densify_nullable,
        allow_engineered_operands,
        cols,
        engineered_operand_values,
        _use_subsample,
        _full_n_rows,
        _sample_idx,
    )

    # Per-operand learned pre-warp + median-gate fit (carved to _pairs_setup.py).
    st._prewarp_active, st._prewarp_spec_by_var, st._gate_med_active, st._gate_med_median_by_var = _fit_prewarp_and_gate_med(
        prospective_pairs=prospective_pairs,
        prewarp_enable=prewarp_enable,
        prewarp_y=prewarp_y,
        prewarp_y_continuous=prewarp_y_continuous,
        prewarp_basis=prewarp_basis,
        prewarp_max_degree=prewarp_max_degree,
        prewarp_min_val_corr=prewarp_min_val_corr,
        fe_gate_med_enable=fe_gate_med_enable,
        original_cols=original_cols,
        _use_subsample=_use_subsample,
        _full_n_rows=_full_n_rows,
        _sample_idx=_sample_idx,
        _extval_raw_col=_extval_raw_col,
    )
    return _extval_raw_col


def _check_prospective_fe_pairs_part3(
    X: Any, subsample_seed: int, shared_subsample_idx: np.ndarray | None, subsample_n: int, fe_subsample_stratify: bool, classes_y: np.ndarray,
    classes_y_safe: np.ndarray, freqs_y: np.ndarray, verbose: int,
) -> tuple[Any, ...]:
    """Extracted block of ``check_prospective_fe_pairs`` (part 3); returns the names the caller still uses."""
    # Starting from the most heavily connected pairs, create a big pool of original features + their unary transforms. Individual vars referenced more than once go
    # to the global pool, the rest to the local (not stored)?

    # Lazy import of parent-resident helpers: ``.predict`` re-imports
    # this sibling at its bottom, so a top-level ``from .predict
    # import ...`` would create a hard cycle the meta-test flags.
    st = _SimpleNamespace()  # long-lived locals of this function (see the stage helpers below)
    pair_pbar: Any = None
    pair_mi: Any = None
    from ..feature_engineering import (
        _can_hoist_shared_buffer,
        _rebuild_full_survivor_col,
        discretize_array,
        discretize_2d_quantile_batch,
        get_new_feature_name,
        gpu_compatible_unary_names,
        logger,
        mi_direct,
    )

    # Batched FE-candidate MI + permutation noise-gate (bit-identical to the
    # per-candidate mi_direct on the default outer/n_workers=1 path - see kernel docstring).
    from ..info_theory import batch_mi_with_noise_gate, use_su_normalization

    # P-SEAM (matrix-native FE replatform): the SINGLE integration point for the framework-agnostic
    # matrix path. GATED behind MLFRAME_FE_MATRIX_P0 - default OFF, so this is a pure no-op and X is
    # byte-untouched (the legacy pandas path runs unchanged). When enabled, X is routed through the
    # single-copy float32 matrix adapter (a round-trip here today; on-device kernels in later phases),
    # so the SAME numba/cupy path can serve pandas and polars. The float32 cast is the intended P0
    # behaviour change. Wrapped so the experimental path can never break the production FE pipeline.

    X = _check_prospective__behaviour_change_wrapped_experimental(X)
    st.res = {}
    # REJECTION-LEDGER local accumulator (additive): per-pair acceptance-gate
    # drops collect here, then are exported via BOTH the ``rejection_ledger_out`` side channel
    # (serial / threading path) AND the reserved ``res`` key (survives the loky-parallel path,
    # where the caller's list cannot be mutated cross-process). Records carry only values the
    # gate already computed - no recompute.
    st._rejection_records = []

    # Seeded RNG for the external-validation factor subsample below. Pre-fix code used the
    # process-global ``np.random.choice`` there, which (a) made the choice depend on whatever
    # had consumed the global numpy RNG earlier in the process (so two fits of the SAME (X, y)
    # in one session could pick DIFFERENT validation factors -> a different tie-break -> a
    # different engineered-recipe SET / column NAMES), and (b) raced under the joblib
    # ``backend="threading"`` chunked path (N workers sharing one global RNG). Derive a local
    # Generator from ``subsample_seed`` (instance-controlled) so the factor pick is reproducible
    # from the MRMR seed and thread-safe, mirroring the seeded ``_rng_sub`` and the fleuret LCG fix.
    st._rng_extval = np.random.default_rng(int(subsample_seed))

    # SUBSAMPLE-SETUP: when caller asks for subsample_n > 0 AND len(X) exceeds it,
    # build subsampled views of X / classes_y / classes_y_safe / freqs_y. The MI
    # sweep operates on these views; survivor packing always rebuilds at full n.
    # When subsample_n is 0 / negative / >= len(X) the legacy full-data path runs
    # unchanged (everything below uses ``X`` / ``classes_y`` / ... directly).
    st._X_full = X
    _full_n_rows = len(st._X_full)
    # Prefer the fit's ONE shared row-index draw when the caller supplies it (so every FE consumer
    # scores the SAME rows); fall back to the legacy per-call draw otherwise.
    st._shared_idx = None
    st._shared_idx = _check_prospective__scores_same_rows_fall(shared_subsample_idx, _full_n_rows, st._shared_idx)
    _use_subsample = (st._shared_idx is not None) or (isinstance(subsample_n, int) and 0 < subsample_n < _full_n_rows)
    _sample_idx = None  # set below when subsampling; threaded into _fit_prewarp_and_gate_med
    X, _sample_idx, classes_y, classes_y_safe, freqs_y = _check_prospective__use_subsample(
        _use_subsample,
        st._shared_idx,
        subsample_seed,
        fe_subsample_stratify,
        classes_y,
        subsample_n,
        _full_n_rows,
        _sample_idx,
        st._X_full,
        classes_y_safe,
        freqs_y,
        verbose,
        X,
    )

    # EXTERNAL-VALIDATION raw-column EXTRACTION MEMO (LEVER 1).
    # The lazy external-validation tie-break (below) extracts the RAW values of
    # every external factor via ``X.iloc[:, original_cols[ext]].values`` (or the
    # polars ``.to_numpy()``) once per tied-leader config. The external-factor set
    # is ``numeric_vars_to_consider`` minus the 2 pair operands, so the SAME raw
    # column is re-extracted across every config AND every raw pair: on the wide
    # scene bed (2407x299) that is ~276k pandas ``.iloc`` calls / ~43s of pure
    # pandas-indexing (call-site cProfile). The extraction is DETERMINISTIC - a
    # var-index maps to a FIXED column-values ndarray for the lifetime of this
    # call (``X`` is fixed after the subsample reassignment above) - so memoising
    # it by the var key (the ``external_factor`` index into ``original_cols``)
    # yields BYTE-IDENTICAL values while collapsing the per-config re-extraction to
    # one extraction per distinct external factor. Keyed by the var id only, never
    # by array contents, so it cannot collide across distinct columns. The cache is
    # a plain local dict scoped to this call -> never pickled (no __getstate__
    # concern; the MRMR estimator never holds a reference to it).
    _extval_raw_col_cache: dict = {}
    return st, pair_pbar, pair_mi, _can_hoist_shared_buffer, _rebuild_full_survivor_col, discretize_array, discretize_2d_quantile_batch, get_new_feature_name, gpu_compatible_unary_names, logger, mi_direct, batch_mi_with_noise_gate, use_su_normalization, X, _full_n_rows, _use_subsample, _sample_idx, classes_y, classes_y_safe, freqs_y, _extval_raw_col_cache


def _check_prospective_fe_pairs_part2(
    unary_transformations: dict, st: Any, prospective_pairs: dict, verbose: int, logger: Any, X: Any, binary_transformations: dict, concurrent_workers: int,
    _extval_raw_col: Any, cols: list[str], gpu_compatible_unary_names: Any, usability_y_continuous: np.ndarray | None,
    prewarp_y_continuous: np.ndarray | None, classes_y: np.ndarray, _use_subsample: Any, _full_n_rows: Any, _sample_idx: Any,
) -> tuple[Any, ...]:
    """Extracted block of ``check_prospective_fe_pairs`` (part 2); returns the names the caller still uses."""
    # Effective unary name list: the real registry plus the pre-warp pseudo-unary
    # when active. Used everywhere a per-pair combination over unary names is
    # built so the pseudo-unary participates exactly like a real one.
    st._unary_names_eff = list(unary_transformations.keys())
    if st._prewarp_active:
        st._unary_names_eff = [*st._unary_names_eff, _PREWARP_UNARY]
    if st._gate_med_active:
        st._unary_names_eff = [*st._unary_names_eff, _GATE_MED_UNARY]

    # Exact preallocation. ``n_pairs * n_unary * 2`` over-counts because (var, tr_name) keys are de-duplicated in ``vars_transformations``; the unique-key set is the
    # true upper bound. Every raw var appearing in ANY pair is combined with every ``_unary_names_eff`` entry
    # unconditionally (no per-var filtering above), so the true unique-key count is provably
    # ``len({var for pair in prospective_pairs for var in pair}) * len(_unary_names_eff)`` - avoids materialising
    # the O(pairs x unaries) set just to take its length.
    st.unique_vars = {var for raw_vars_pair, _ in prospective_pairs.keys() for var in raw_vars_pair}
    st.n_unique_keys = len(st.unique_vars) * len(st._unary_names_eff)

    if verbose >= 2:
        logger.info(
            "Creating a pool of %d unary transformations for feature engineering " "(legacy upper bound was %d).",
            st.n_unique_keys,
            len(prospective_pairs) * len(unary_transformations) * 2,
        )

    transformed_vars = np.empty(shape=(len(X), st.n_unique_keys), dtype=np.float32)

    # Hoist ``final_transformed_vals`` outside the per-pair loop: precompute each pair's ``combs``, find the max length, allocate one shared buffer. Each pair writes
    # then reads the same ``[:, i]`` slice so stale tail data is never observed.
    #
    # OPT-C structural-candidate-dedup analysed + REJECTED (2026-06-07, 0% yield): a proposal to
    # dedup PROVABLY-EQUAL candidate columns (symmetric binaries on equal operand-key sets) before
    # discretize+MI was measured to remove NOTHING - this generation ALREADY eliminates every
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
    st.pair_combs = {}
    st.max_n_combs = 0
    st.max_n_combs = _check_prospective__win_do_re_implement(prospective_pairs, st._unary_names_eff, st.pair_combs, st.max_n_combs)

    # CRITICAL #2: memory-aware dispatch. The full hoisted buffer is the fast path
    # but on n=4M with medium preset this lands at ~17.6 GiB and crashes the suite. Estimate the
    # required buffer, check psutil.virtual_memory().available, and either keep the buffer (fast)
    # or set it to None and switch to recompute-from-metadata in the inner loop and survivor
    # rebuild stages (memory-safe; ~1% extra bin_func calls per pair).
    st._n_binary = len(binary_transformations)
    st.final_transformed_vals_shared = None
    # CROSS-PAIR chunk budget (cols). Default 0 == not chunkable (per-pair buffer).
    # Filled below when the single-pair buffer fits RAM: the chunk buffer reuses the
    # SAME available-RAM budget but may hold MANY pairs (each pair packed whole).
    st._fe_chunk_max_cols = 0
    # LARGE-N PEAK-MEMORY FIX: the candidate buffers (chunk float32 + disc
    # int8 codes + batch-MI working set + the held-alive single-pair buffer) coexist while a
    # chunk is scored, and on the joblib ``backend="threading"`` path ``concurrent_workers``
    # copies of them are alive at once. So BOTH the single-pair fit check and the chunk-width
    # cap use the overhead+worker-aware envelope (raw 0.4*available / _FE_PEAK_OVERHEAD_FACTOR
    # / concurrent_workers) instead of the raw 0.4*available that under-counted the siblings
    # and ignored the thread multiplication. Byte-identical selection (width only).
    st._n_workers = max(1, int(concurrent_workers))
    st._fe_chunk_max_cols, st.final_transformed_vals_shared = _check_prospective__ignored_thread_multiplication_byte(
        st.max_n_combs, X, st._n_binary, st._n_workers, verbose, st._fe_chunk_max_cols, st.final_transformed_vals_shared
    )
    # bench-attempt-rejected (2026-06-17): BLOCK-STREAMING the candidate buffer when the full
    # per-pair buffer does not hoist (alloc a narrow N-col block buffer, flush materialise ->
    # discretize_2d_quantile_batch -> batched-MI per block; route the 7 downstream
    # ``final_transformed_vals[:, cfg]`` reads - usability-corr, winner occupied-K, multi-emit,
    # ext-val, survivor - through a recompute helper). Implemented + VALIDATED bit-identical
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
    st._need_recompute_map = st.final_transformed_vals_shared is None

    vars_transformations = _build_operand_table(
        prospective_pairs=prospective_pairs,
        transformed_vars=transformed_vars,
        _unary_names_eff=st._unary_names_eff,
        unary_transformations=unary_transformations,
        _extval_raw_col=_extval_raw_col,
        _prewarp_active=st._prewarp_active,
        _prewarp_spec_by_var=st._prewarp_spec_by_var,
        _gate_med_median_by_var=st._gate_med_median_by_var,
        cols=cols,
        verbose=verbose,
        logger=logger,
        gpu_compatible_unary_names=gpu_compatible_unary_names,
    )

    if verbose >= 2:
        logger.info("Created. For every pair from the pool, trying all known functions...")

    # Per-operand marginal MI cache for the MARGINAL-UPLIFT alternative acceptance gate.
    # The marginal MI is the discretised RAW operand (its ``identity`` unary, already
    # materialised in ``transformed_vars``) scored against the target with the SAME
    # estimator the engineered columns use, so the uplift ratio is apples-to-apples.
    # Operands recur across pairs, so memoise by cols-space var index.
    _operand_marginal_mi_cache: dict = {}

    # NOISE-WRAP CORR-COLLAPSE GUARD. A subsample-aligned CONTINUOUS target for a cheap |corr|
    # discriminator that catches the failure mode where the per-pair search WRAPS a strong, clean operand
    # (e.g. a univariate-basis column ``a__T2`` ~ a**2) with a PURE-NOISE operand (``e``) via an extreme
    # heavy-tailed transform (``sub(log(e),invqubed(a__T2))``). That composite's binned-MI ``best_mi/pair_mi``
    # ratio CLEARS the joint-prevalence gate (the extreme transform inflates BOTH MIs), yet its |corr| with the
    # target COLLAPSES to ~0 while the clean operand's |corr| is ~1.0 - so the artefact displaces the clean
    # basis from the support and recovery dies. Genuine synergy pairs (a*b, log(c)*sin(d)) do NOT collapse this
    # way: the engineered column still tracks y monotonically. Prefer the continuous y; fall back to the binned
    # ``classes_y`` codes (still a usable monotone proxy) when no continuous target was threaded.
    _corr_y_cont = None
    _corr_y_cont_finite = None  # cached np.isfinite(_corr_y_cont); _corr_y_cont is never mutated after assignment
    _corr_y_cont, _corr_y_cont_finite = _check_prospective__classes_codes_still_usable(
        usability_y_continuous, prewarp_y_continuous, classes_y, _use_subsample, _full_n_rows, _sample_idx, _corr_y_cont, _corr_y_cont_finite
    )

    _safe_abs_corr = _check_prospective_fe_step2_classes_codes_still(_corr_y_cont, _corr_y_cont_finite)
    return transformed_vars, vars_transformations, _operand_marginal_mi_cache, _corr_y_cont, _corr_y_cont_finite, _safe_abs_corr


def _check_prospective_fe_pairs_part1(
    _extval_raw_col: Any, _safe_abs_corr: Any, vars_transformations: Any, transformed_vars: Any, st: Any, _operand_marginal_mi_cache: Any,
    quantization_nbins: int, quantization_method: str | None, quantization_dtype: type | str, classes_y: np.ndarray, classes_y_safe: np.ndarray,
    freqs_y: np.ndarray, fe_min_nonzero_confidence: float, fe_npermutations: int, prospective_pairs: dict, X: Any, verbose: int,
) -> tuple[Any, ...]:
    """Extracted block of ``check_prospective_fe_pairs`` (part 1); returns the names the caller still uses."""
    # Unlike ``_config_corr``'s per-pair-specific candidate columns (never revisited across pairs, so
    # correctly left unmemoised), a raw/transformed OPERAND's |corr| recurs across every admitted pair
    # that shares it (var ``a`` in pairs (a,b),(a,c),(a,d),...). Mirror the ``_operand_marginal_mi`` /
    # ``_operand_discretized`` var-keyed memoisation pattern above so the noise-wrap veto's per-operand
    # |corr| (an O(n) njit pass) is computed once per distinct operand for the whole call, not once per
    # admitted pair it appears in.
    _raw_operand_abs_corr_cache: dict = {}

    def _raw_operand_abs_corr(_var) -> float:
        """Memoised |corr(raw operand, y)| by raw var key."""
        if _var in _raw_operand_abs_corr_cache:
            return float(_raw_operand_abs_corr_cache[_var])
        _vals = _extval_raw_col(_var)
        _c = _safe_abs_corr(_vals) if _vals is not None else 0.0
        _raw_operand_abs_corr_cache[_var] = _c
        return _c

    _transformed_operand_abs_corr_cache: dict = {}

    def _transformed_operand_abs_corr(_opk) -> float:
        """Memoised |corr(transformed operand column, y)| by ``(var, unary)`` key."""
        if _opk in _transformed_operand_abs_corr_cache:
            return float(_transformed_operand_abs_corr_cache[_opk])
        _idx = vars_transformations.get(_opk)
        _c = _safe_abs_corr(transformed_vars[:, _idx]) if _idx is not None else 0.0
        _transformed_operand_abs_corr_cache[_opk] = _c
        return _c

    # The winning composite is condemned as a noise-wrap when its |corr| with the target is a small FRACTION
    # of the best single operand's |corr| AND that operand is genuinely strong on its own. Calibrated wide:
    # the artefact collapses to ~0.02 vs the clean operand's ~0.99 (a >40x collapse), while a genuine synergy
    # pair's engineered column tracks y at least comparably to its strongest operand. ``0.5`` keeps a 2x margin
    # so a real synergy that modestly trades linear |corr| for a higher-order MI gain is never condemned.
    st._NOISE_WRAP_CORR_COLLAPSE_FRAC = 0.5
    st._NOISE_WRAP_MIN_OPERAND_CORR = 0.30

    # bench-attempt-rejected (2026-06-07): BATCH all distinct operands' marginal MI through
    # one discretize_2d_quantile_batch + one _dispatch_batch_mi_with_noise_gate (Q9), instead
    # of this per-var single-column discretize_array + mi_direct. It WOULD be bit-identical
    # (the batch discretise + batch gate are already proven equal to the per-column path), but
    # it is not worth the invasive control-flow change + the risk to the load-bearing uplift
    # gate comparison below: _operand_marginal_mi is already memoised (each distinct operand
    # computed at most once) AND it only fires on the marginal-uplift FALLBACK gate (pairs that
    # miss the joint + prewarp gates), so on the scene scene-profile the ENTIRE mi_direct family
    # is only 0.1-0.3% of fit wall - batching it saves a sub-noise fraction that is dwarfed by
    # the GPU-clock variance between runs. Re-evaluate only if a workload makes this gate hot.
    _operand_marginal_mi = _check_prospective_fe_step3_gpu_clock_variance(
        _operand_marginal_mi_cache,
        vars_transformations,
        transformed_vars,
        quantization_nbins,
        quantization_method,
        quantization_dtype,
        classes_y,
        classes_y_safe,
        freqs_y,
        fe_min_nonzero_confidence,
        fe_npermutations,
    )

    # MM-DEBIAS (+ #4): per-operand DISCRETISED codes for the
    # occupied-joint-K of a raw pair. Memoised + bit-identical to the discretise the
    # gate's ``pair_mi`` was computed over (same ``quantization_nbins`` / method /
    # ``transformed_vars`` identity column ``_operand_marginal_mi`` uses). Returns the
    # int code array, or None when the operand has no identity transform.
    _operand_disc_cache: dict = {}

    _operand_discretized = _check_prospective_fe_step4_int_code_array(
        _operand_disc_cache, vars_transformations, transformed_vars, quantization_nbins, quantization_method, quantization_dtype
    )

    # CROSS-PAIR (CHUNK) BATCHING precompute. Only on the hoist+quantile
    # path (the per-pair 3-phase batch path). We partition the prospective pairs into
    # CHUNKS whose total candidate-column count fits the RAM-budgeted chunk buffer,
    # then - per chunk - materialise ALL the chunk's candidate columns into ONE wide
    # buffer, run ONE discretize_2d + ONE batch_mi over the whole chunk, and stash the
    # per-pair results (ordered candidate list + per-candidate MI + buffer columns) in
    # ``_chunk_mi_cache``. The per-pair loop below consumes the cache (reads MI + the
    # chunk buffer columns) instead of doing its own per-pair Phase 1/2/3 - enlarging
    # the njit-prange + GPU-dispatch batch from one pair's K to the whole chunk's
    # K_chunk, so the kernels saturate the cores / cross the GPU threshold. Bit-identical
    # (each column is scored independently; shuffle seeded by (0, perm_index) only).
    #
    # ``_chunk_mi_cache[raw_vars_pair]`` -> (ordered list of
    # (transformations_pair, bin_func_name, buf_col, uses_pw), fe_mi_array_aligned,
    # local_times_for_pair). ``_chunk_buffer`` is the wide buffer the buf_col indices
    # point into; it is held alive for the whole pair loop so survivor packing can read
    # ``_chunk_buffer[:, buf_col]`` exactly as the per-pair path read final_transformed_vals.
    st._chunk_global_batch = (
        (st.final_transformed_vals_shared is not None)
        and (quantization_method == "quantile")
        and (st._fe_chunk_max_cols > st.max_n_combs * st._n_binary)  # chunk holds > 1 pair's worth
        and (len(prospective_pairs) > 1)
    )
    # RESIDENCY DEFERRAL gate (default OFF). When ON, the chunk's GPU FUSED codes path skips the (n,K)
    # float D2H (out_cand=None) and the few intermediate buffer reads below RE-MATERIALISE their column on
    # the GPU via ``_fe_materialise_block_gpu`` - the SAME kernel that filled the buffer, so the bytes are
    # BIT-IDENTICAL (no cupy-vs-numpy ULP shift; the prior numpy-recompute scaffold flipped the clean-form
    # demotion). The operand table ``transformed_vars`` is uploaded ONCE per deferred chunk and cached;
    # per-buf_col columns are cached too (a column may be read several times). Only the GPU-fused path is
    # eligible (else the CPU binning still needs the host buffer); CPU/no-CUDA path is unchanged.
    st._fe_defer_float = os.environ.get("MLFRAME_FE_GPU_DEFER_FLOAT", "1").strip().lower() in ("1", "true", "on", "yes")
    # Cross-pair chunk-materialise state, threaded through ``_score_one_pair`` as ONE mutable dict so the
    # lazy per-chunk load / reset semantics persist across pairs exactly as the in-loop locals did:
    #   loaded_idx     : index of the chunk currently materialised in ``_chunk_buffer`` (-1 = none).
    #   mi_cache       : ``_compute_one_fe_chunk`` result for the loaded chunk (per-pair MI + buf cols).
    #   float_deferred : the chunk's ``__float_deferred__`` signal (GPU buffer-D2H deferral).
    #   defer_meta     : (a_cols, b_cols, ops) int arrays for on-demand GPU re-materialise, per chunk.
    #   tv_gpu         : cupy upload of ``transformed_vars`` (the operand table), per chunk.
    #   resolved_cols  : buf_col -> host float32 column (re-materialised), per chunk.
    st._chunk_state = {
        "loaded_idx": -1,
        "mi_cache": {},
        "float_deferred": False,
        "defer_meta": None,
        "tv_gpu": None,
        "resolved_cols": {},
    }
    st._chunk_buffer = None
    st._pair_to_chunk = {}  # raw_vars_pair -> chunk index
    st._fe_chunks = []
    st._pair_valid_combs = {}
    st._chunk_buf_width = 0
    if st._chunk_global_batch:
        st._fe_chunks, st._pair_valid_combs, st._chunk_buf_width = _plan_fe_chunks(
            prospective_pairs=prospective_pairs,
            pair_combs=st.pair_combs,
            vars_transformations=vars_transformations,
            n_binary=st._n_binary,
            chunk_max_cols=st._fe_chunk_max_cols,
        )
        # Only worth chunking when at least one chunk groups MORE than one pair.
        st._chunk_buffer, st._chunk_global_batch, st._pair_to_chunk = _check_prospective__only_worth_chunking_least(
            st._fe_chunks, st._chunk_buf_width, X, st._pair_to_chunk, verbose, prospective_pairs, st._chunk_buffer, st._chunk_global_batch
        )
    return _raw_operand_abs_corr, _transformed_operand_abs_corr, _operand_marginal_mi, _operand_discretized
