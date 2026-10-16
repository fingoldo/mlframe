"""External validation of FE pair candidates (step 2 of ``_pairs_emit._emit_pair_features``).

For every surviving pair config the candidates are rebuilt against the external validation factors, discretised and scored in batches
(chunked by the RAM budget), and the best validation MI per config is recorded; the pair's candidates then go through the family-wise gate.
Carved out of ``_pairs_emit`` (module size limit); the parent re-exports both names.
"""

from __future__ import annotations

import logging

import numpy as np

from mlframe.utils.env_flags import env_int
from mlframe.utils.log_throttle import log_throttle

logger = logging.getLogger(__name__)

# The external-validation MI only breaks ties between leaders of exactly equal primary MI, but the exact family test over every validation
# candidate of a pair needs all of their codes at once and a permutation pass over all of them: 118 tied configurations x 252 external factors
# x 6 operations at 30000 rows is 5.4e9 code cells, more than a host holds even as int8. A family above this many million cells is decided by
# the null-MI table floor alone (see ``_pairs_emit._apply_pair_family_floor``); the exact test stays for every family that fits.
# TODO(KTC): a measured crossover per machine; 400 million cells keeps the first 256-permutation block near 1e11 histogram updates.
_FAMILY_EXACT_MAX_CELLS = env_int("MLFRAME_PAIR_FAMILY_EXACT_MAX_MCELLS", 400, minimum=0) * 1_000_000


def _extval_host_buffer(param_a, _ev_param_bs, _ev_op_codes, _ev_bin_funcs, _ev_K, n_rows, _materialise_extval_njit):
    """Materialise the (n, K) float64 external-validation candidate buffer on the host (njit kernel if every op is coded, else numpy);
    returns ``(buffer, n_columns)``."""
    _ev_buf = np.empty((n_rows, _ev_K), dtype=np.float64)
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
        _ev_pb_mat = np.empty((n_rows, len(_ev_param_bs)), dtype=np.float64)
        for _ei, _pb_vals in enumerate(_ev_param_bs):
            _ev_pb_mat[:, _ei] = _pb_vals
        _materialise_extval_njit(
            np.ascontiguousarray(param_a),
            _ev_pb_mat,
            _ev_op_codes,
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
    return _ev_buf, _ev_col


def _emit_pair_features_step2_read_here_once(  # noqa: C901 -- one step kept whole; the complexity ratchet tracks it
    _ev_configs,
    final_transformed_vals,
    _this_chunk_deferred,
    _resolve_col,
    _config_by_i,
    st,
    transformed_vars,
    vars_transformations,
    binary_transformations,
    _ext_factors_sorted,
    fe_max_external_validation_factors,
    _rng_extval,
    _extval_raw_col,
    X,
    _can_hoist_shared_buffer,
    _n_workers,
    quantization_method,
    _narrow_code_dtype,
    quantization_nbins,
    quantization_dtype,
    _ev_op_codes,
    _materialise_extval_njit,
    discretize_2d_quantile_batch,
    _fe_use_parallel_kernels,
    serial_main_thread,
    _dispatch_batch_mi_with_noise_gate,
    classes_y,
    classes_y_safe,
    freqs_y,
    fe_npermutations,
    fe_min_nonzero_confidence,
    use_su_normalization,
    batch_mi_with_noise_gate,
    _fe_env_gate,
    discretize_array,
    mi_direct,
    valid_pairs_perf,
):
    """Step 2 of _emit_pair_features: lines starting at ``for transformations_pair, bin_func_name, i in (_ev_configs if len(_ev_``."""
    from mlframe.feature_selection.filters._gpu_strict_fe import fe_gpu_strict_resident_enabled as _ev_resident_enabled

    _pair_batches: list = []  # (config, host codes n x K) of every batch of this pair: one family for the family-wise gate below
    # Size of the family before building it: configurations x external factors x operations x rows. A family past the cell budget is never
    # held; each configuration only reports its best validation MI and the pair is gated by the table floor afterwards.
    _n_ext_est = len(_ext_factors_sorted)
    if fe_max_external_validation_factors:
        _n_ext_est = min(_n_ext_est, int(fe_max_external_validation_factors))
    _family_cells = len(_ev_configs) * _n_ext_est * len(binary_transformations) * len(X)
    _family_exact = _family_cells <= _FAMILY_EXACT_MAX_CELLS
    _k_seen = 0  # candidates scored over every configuration of this pair (the family size K of the table floor)
    for transformations_pair, bin_func_name, i in (_ev_configs if len(_ev_configs) > 1 else []):
        _pa_dev = None
        if final_transformed_vals is not None:
            param_a = final_transformed_vals[:, i]
        elif _this_chunk_deferred:
            # DEFERRED-float GPU path: re-materialise the survivor column on the GPU
            # (bit-identical to the buffer -> the external-validation MI is unchanged).
            # Under the strict-resident path the column stays on the device (the ext-val candidates are built from it there); the host copy is only
            # taken below if a host branch actually needs it.
            _pa_dev = getattr(_resolve_col, "device", lambda *_: None)(i) if _ev_resident_enabled() else None
            param_a = None if _pa_dev is not None else _resolve_col(i)
        else:
            # CRITICAL #2 recompute-fallback: rebuild the survivor column from its
            # (a_key, b_key, bin_func_name) metadata. transformed_vars is small
            # (deduped unary table); the bin_func call is cheap (one ufunc).
            st._a_key, st._b_key, st._bin_name = _config_by_i[i]
            st._pa = transformed_vars[:, vars_transformations[st._a_key]]
            st._pb = transformed_vars[:, vars_transformations[st._b_key]]
            param_a = binary_transformations[st._bin_name](st._pa, st._pb)
            np.nan_to_num(param_a, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        best_valid_mi = -1.0
        config = (transformations_pair, bin_func_name, i)
        _else_cols: list = []  # per-candidate codes of this config on the non-batch path; gated with the pair's family

        # ``sorted`` first: a bare ``set`` difference iterates in hash order, which is
        # PYTHONHASHSEED-randomised for str keys, so the candidate order (and hence the
        # sampled subset) would differ across processes / fits. Sort to a stable order,
        # then sample with the instance-seeded ``_rng_extval`` so the chosen validation
        # factors are fully reproducible from the MRMR seed.
        external_factors = _ext_factors_sorted  # hoisted above; deterministic, raw_vars_pair-invariant
        if fe_max_external_validation_factors and len(external_factors) > fe_max_external_validation_factors:
            external_factors = _rng_extval.choice(external_factors, fe_max_external_validation_factors, replace=False)

        # BATCHED EXTERNAL VALIDATION: the per-(external_factor x
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
        # ``npermutations=fe_npermutations<32`` so no GPU permutation route) - the
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
        # LARGE-N FIX: this float64 ext-val buffer coexists with the
        # chunk/disc/MI buffers and is allocated per concurrent worker, so use the
        # SAME overhead+worker-aware envelope as the candidate buffer above.
        _ev_can_batch, _, _ = _can_hoist_shared_buffer(_ev_buf_bytes, n_workers=_n_workers)
        # Chunk the external factors when the whole (n, factors x ops) float64 buffer does not fit the RAM budget: the per-candidate MI and
        # codes are column-independent, so scoring the candidates chunk by chunk is identical to one batch, and far cheaper than the
        # per-candidate loop below (54k single-column discretise + MI calls on a 50000-row, 3795-candidate fit).
        _ev_chunk = len(_ev_param_bs)
        while _ev_chunk > 1 and not _ev_can_batch:
            _ev_chunk = max(1, _ev_chunk // 2)
            _ev_can_batch, _, _ = _can_hoist_shared_buffer(len(X) * _ev_chunk * _ev_n_bin * 8, n_workers=_n_workers)
        if quantization_method == "quantile" and _ev_param_bs and _ev_can_batch:
            _ev_all_param_bs = _ev_param_bs
            _ev_chunk_codes: list = []
            _ev_all_scored = True
            for _ev_c0 in range(0, len(_ev_all_param_bs), _ev_chunk):
                _ev_param_bs = _ev_all_param_bs[_ev_c0 : _ev_c0 + _ev_chunk]
                _ev_bin_funcs = list(binary_transformations.values())
                _ev_K = len(_ev_param_bs) * len(_ev_bin_funcs)
                # float64 buffer: the per-candidate path discretises the RAW
                # ``valid_bin_func(...)`` output (numpy bin_funcs return float64) with
                # NO nan_to_num - ``discretize_array``/``discretize_2d_quantile_batch``
                # both bin via ``np.nanpercentile`` (NaN-ignoring edges) + per-column
                # ``searchsorted`` (NaN -> rightmost bin), identically. Writing into a
                # float64 buffer (not float32) preserves the bin_func's native precision
                # so the percentile edges match the 1-D path to the bit.
                _ev_code_dtype = _narrow_code_dtype(quantization_nbins, quantization_dtype)  # OPT-B narrow codes
                _ev_col = _ev_K
                _ev_disc = None
                # DEVICE-BORN EXT-VAL CANDIDATES (Phase-1 residency, 2026-07-01). Build out[:, e*n_ops+o] =
                # op(param_a, ext_e) RESIDENT in float64 + quantile-bin RESIDENT, so the (n, K) candidate
                # matrix NEVER crosses H2D - only param_a + the external-factor columns upload (once, small),
                # killing the ~46 MB gpu_discretize_codes_host bulk H2D at full n. Engaged only on the strict-
                # resident path (no KTC crossover - residency contract, wall-loss accepted) AND when every op
                # is registry-coded; any cupy fault returns None -> the exact host njit + upload path below.
                # The device ops are the float64 numpy bin_func semantics op-for-op and NaN/inf are NOT
                # scrubbed (routed to the rightmost bin by the same resident binner, as nanpercentile +
                # searchsorted do on the host) -> codes selection-equivalent (see
                # _gpu_resident_extval.gpu_materialise_extval_codes_host).
                if _ev_op_codes is not None:
                    try:
                        from mlframe.feature_selection.filters._gpu_strict_fe import fe_gpu_strict_resident_enabled as _ev_resident_on

                        _ev_use_dev = bool(_ev_resident_on())
                    except Exception as e:
                        log_throttle(
                            logger,
                            "gpu_strict.resident_probe",
                            logging.WARNING,
                            "the resident-GPU eligibility probe failed (%s), using the non-resident path",
                            e,
                        )
                        _ev_use_dev = False
                    if _ev_use_dev:
                        from mlframe.feature_selection.filters._gpu_resident_extval import gpu_materialise_extval_codes_host

                        _ev_disc = gpu_materialise_extval_codes_host(
                            param_a,
                            _ev_param_bs,
                            _ev_op_codes,
                            int(quantization_nbins),
                            dtype=_ev_code_dtype,
                            defer_host_fill=True,
                            param_a_dev=_pa_dev,
                        )
                if _ev_disc is None:
                    # HOST PATH (unchanged): materialise the (n, K) float64 buffer then discretise.
                    if param_a is None:
                        param_a = _resolve_col(i)
                    _ev_buf, _ev_col = _extval_host_buffer(param_a, _ev_param_bs, _ev_op_codes, _ev_bin_funcs, _ev_K, len(X), _materialise_extval_njit)
                    # GPU BINNING: the ext-val survivor binning (full n) gets the same
                    # dedicated binning crossover - bit-identical to the CPU njit binning (maxdiff 0)
                    # and much faster at large n. Any GPU failure falls back to the CPU discretise below.
                    try:
                        from mlframe.feature_selection.filters._feature_engineering_pairs._pairs_core import _fe_gpu_binning_enabled

                        if _fe_gpu_binning_enabled(_ev_buf.shape[0], _ev_col):
                            from mlframe.feature_selection.filters._gpu_resident_fe import gpu_discretize_codes_host

                            # defer_host_fill: the codes flow straight into _dispatch_batch_mi_with_noise_gate,
                            # whose resident-CUDA gate consumes the DEVICE codes in place; the host buffer is
                            # filled lazily only on a host-reading branch. Skips the (n, K) codes D2H whenever
                            # the resident gate is the consumer. Bit-identical (host buffer == device.get()).
                            _ev_disc = gpu_discretize_codes_host(
                                _ev_buf[:, :_ev_col],
                                int(quantization_nbins),
                                dtype=_ev_code_dtype,
                                defer_host_fill=True,
                            )
                    except Exception as e:
                        log_throttle(
                            logger, "pairs_emit_resident_discretize", logging.WARNING, "resident discretization failed, falling back to the host path: %s", e
                        )
                        _ev_disc = None
                    if _ev_disc is None:
                        _ev_disc = discretize_2d_quantile_batch(
                            _ev_buf[:, :_ev_col],
                            n_bins=quantization_nbins,
                            dtype=_ev_code_dtype,
                            # OPT-A extension: the marginal-uplift gate's
                            # discretise ran the SERIAL searchsorted kernel on the main
                            # thread (post-OPT-D the top sampler hotspot, ~21% of fit) while
                            # the other cores sat idle. ``check_prospective_fe_pairs`` carries
                            # ``serial_main_thread`` down from _mrmr_fe_step's ``len(X)<50000``
                            # dispatch, so the same OPT-A predicate that already gates the
                            # main chunk's discretise (line ~907) safely selects the
                            # byte-identical column-prange twin here too (no joblib nest).
                            parallel=_fe_use_parallel_kernels(_ev_col, serial_main_thread),
                        )
                if _family_exact and isinstance(_ev_disc, np.ndarray):
                    # The family gate reads these codes on the host after the dispatch; an unfilled deferred GPU buffer would read as zeros (the
                    # dispatch drops the deferred-fill record when it ends), so fill it before. A family past the budget is never read: skip the D2H.
                    from mlframe.feature_selection.filters._gpu_resident_fe import ensure_host_codes_filled

                    ensure_host_codes_filled(_ev_disc)
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
                    env_gate=_fe_env_gate,
                )
                if _ev_mi is None or not len(_ev_mi) or _ev_disc is None:
                    _ev_all_scored = False
                if _ev_mi is not None and len(_ev_mi) and _ev_disc is not None:
                    _k_seen += int(_ev_col)
                if _family_exact and _ev_mi is not None and len(_ev_mi) and _ev_disc is not None:
                    _host_codes = _ev_disc[:, : int(_ev_col)]
                    _ev_chunk_codes.append(_host_codes.get() if hasattr(_host_codes, "get") else np.asarray(_host_codes))
                if _ev_mi is not None and len(_ev_mi):
                    best_valid_mi = max(best_valid_mi, float(np.max(_ev_mi)))
            _ev_param_bs = _ev_all_param_bs
            # The family gate needs every candidate of the pair; a chunk without scores leaves the pair out of it, as the one-batch path did.
            if _family_exact and _ev_chunk_codes and _ev_all_scored:
                _pair_batches.append((config, _ev_chunk_codes[0] if len(_ev_chunk_codes) == 1 else np.hstack(_ev_chunk_codes)))
        else:
            if param_a is None:
                param_a = _resolve_col(i)
            for _pb_vals in _ev_param_bs:
                param_b = _pb_vals
                for valid_bin_func in binary_transformations.values():

                    valid_vals = valid_bin_func(param_a, param_b)

                    discretized_transformed_values = discretize_array(
                        arr=valid_vals, n_bins=quantization_nbins, method=quantization_method, dtype=quantization_dtype
                    )
                    _k_seen += 1
                    if _family_exact:
                        _else_cols.append(np.asarray(discretized_transformed_values).reshape(-1, 1))
                    fe_mi, _fe_conf = mi_direct(
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

        if _else_cols:
            _pair_batches.append((config, np.hstack(_else_cols)))
        valid_pairs_perf[config] = best_valid_mi

    from ._pairs_emit import _apply_pair_family_floor, _apply_pair_family_gate

    if _family_exact:
        _apply_pair_family_gate(_pair_batches, valid_pairs_perf, classes_y, int(quantization_nbins))
    else:
        _apply_pair_family_floor(valid_pairs_perf, classes_y, int(quantization_nbins), len(X), _k_seen)
