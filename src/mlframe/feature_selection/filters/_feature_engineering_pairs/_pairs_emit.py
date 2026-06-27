"""Admitted-pair emission tail of ``check_prospective_fe_pairs`` (carved
2026-06-22, Tier E -- second sibling).

When a prospective pair clears the per-pair acceptance gates (joint-prevalence /
prewarp-uplift / marginal-uplift, minus the noise-wrap veto), this module turns
the MI-leaders equivalence class into the emitted survivor feature(s): the
linear-usability tie-break over the leaders, the lazy external-validation
secondary tie-break, the single-best ``_select_single_best`` pick, the diverse
multi-candidate emission, and the survivor-column materialisation + non-constant
guard. Carved VERBATIM out of ``_pairs_score._score_one_pair`` so that module
(and the parent ``_pairs_core``) stay under the 1k-LOC ceiling; every local it
reads is an explicit parameter, the ``_resolve_col`` re-materialise closure is
passed in, and the single ``_pair_res_entry = (...)`` result is returned.
Selection is byte-for-byte identical to the pre-carve in-function block.
"""
from __future__ import annotations

import numpy as np

from pyutilz.pythonlib import sort_dict_by_value

from ._pairs_gates import _PREWARP_UNARY, _select_single_best, mi_tie_band


def _emit_pair_features(
    *,
    raw_vars_pair,
    pair_mi,
    best_mi,
    best_config,
    var_pairs_perf,
    this_pair_features,
    _passes_joint_gate,
    _prewarp_accept,
    _marginal_uplift_accept,
    final_transformed_vals,
    _this_chunk_deferred,
    _config_by_i,
    _resolve_col,
    _corr_y_cont,
    _safe_abs_corr,
    transformed_vars,
    vars_transformations,
    binary_transformations,
    unary_transformations,
    numeric_vars_to_consider,
    fe_good_to_best_feature_mi_threshold,
    fe_max_external_validation_factors,
    fe_max_steps,
    fe_multi_emit_max_per_pair,
    fe_multi_emit_mi_floor,
    fe_multi_emit_diversity_corr,
    quantization_nbins,
    quantization_method,
    quantization_dtype,
    classes_y,
    classes_y_safe,
    freqs_y,
    fe_npermutations,
    fe_min_nonzero_confidence,
    cols,
    original_cols,
    _use_subsample,
    _X_full,
    _full_n_rows,
    _prewarp_spec_by_var,
    _gate_med_median_by_var,
    engineered_operand_values,
    _extval_raw_col,
    _rng_extval,
    _n_workers,
    serial_main_thread,
    verbose,
    messages,
    # framework callables (lazily imported in the parent)
    discretize_array,
    discretize_2d_quantile_batch,
    mi_direct,
    get_new_feature_name,
    _rebuild_full_survivor_col,
    _can_hoist_shared_buffer,
    _narrow_code_dtype,
    _materialise_extval_njit,
    _njit_binary_op_codes,
    _fe_use_parallel_kernels,
    _dispatch_batch_mi_with_noise_gate,
    batch_mi_with_noise_gate,
    use_su_normalization,
    X,
):
    """Emit the survivor feature(s) for an ADMITTED pair. Returns the
    ``res[raw_vars_pair]`` tuple ``(this_pair_features, transformed_vals,
    new_cols, new_nbins, messages)``. Appends to ``messages`` (caller-owned)."""
    _pair_res_entry = None

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
                if final_transformed_vals is not None:
                    _lvals = final_transformed_vals[:, _li]
                elif _this_chunk_deferred:
                    # DEFERRED-float GPU path: re-materialise the leader column on the GPU
                    # (bit-identical to the buffer -> same linear-usability tie-break).
                    _lvals = _resolve_col(_li)
                elif _config_by_i is not None and _li in _config_by_i:
                    # Recompute fallback (no hoisted buffer): rebuild the leader's
                    # CONTINUOUS column from its (a_key, b_key, bin_func_name) metadata
                    # so the linear-usability tie-break is identical to the buffered path
                    # (the two paths MUST select the same survivor among MI-equal leaders).
                    _a_key, _b_key, _bin_name = _config_by_i[_li]
                    _pa = transformed_vars[:, vars_transformations[_a_key]]
                    _pb = transformed_vars[:, vars_transformations[_b_key]]
                    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                        _lvals = binary_transformations[_bin_name](_pa, _pb)
                    _lvals = np.nan_to_num(np.asarray(_lvals, dtype=np.float32), copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                else:
                    _lvals = None
                if _lvals is not None:
                    _leader_usability[_lc] = _safe_abs_corr(_lvals)
            except Exception:
                continue

    # ABSOLUTE binned-MI tie band (2026-06-24, F2 ``mixed`` distribution-robustness fix). Two FORMS of
    # the same raw pair that are monotone re-expressions of one algebraic target have MI equal up to the
    # plug-in bias scale; without a tie band an MI EPSILON (pure binning noise) crowns a form whose linear
    # usability is far worse (mixed: additive ``add(log(a),invsqrt(b))`` MI 0.1180 > exact ratio
    # ``div(sqr(a),b)`` 0.1167, yet |corr(y)| 0.25 vs 0.46 -- the noise winner does not fuse cleanly and
    # leaves the ratio form as a fragment). Snapping the primary MI key to this band inside
    # ``_select_single_best`` lets the EXISTING linear-usability tie-break pick the linearly-usable form.
    _mi_band = mi_tie_band(int(quantization_nbins), int(len(classes_y)), int(np.asarray(freqs_y).shape[0]))

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
            # Hoisted out of the per-config loop: depends only on ``raw_vars_pair`` (loop-invariant for the
            # whole pair), so the set-difference + sort is recomputed once instead of once per tied leader.
            # The RNG draw (``_rng_extval.choice`` below) stays per-config so its state consumption — and
            # therefore every later pair's tie-break — is bit-identical.
            _ext_factors_sorted = sorted(set(numeric_vars_to_consider) - set(raw_vars_pair))
            for transformations_pair, bin_func_name, i in (_ev_configs if len(_ev_configs) > 1 else []):
                if final_transformed_vals is not None:
                    param_a = final_transformed_vals[:, i]
                elif _this_chunk_deferred:
                    # DEFERRED-float GPU path: re-materialise the survivor column on the GPU
                    # (bit-identical to the buffer -> the external-validation MI is unchanged).
                    param_a = _resolve_col(i)
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
                external_factors = _ext_factors_sorted  # hoisted above; deterministic, raw_vars_pair-invariant
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
                    _ev_disc = None
                    # GPU BINNING (2026-06-23): the ext-val survivor binning (full n) gets the same
                    # dedicated binning crossover -- bit-identical to the CPU njit binning (maxdiff 0)
                    # and much faster at large n. Any GPU failure falls back to the CPU discretise below.
                    _ev_code_dtype = _narrow_code_dtype(quantization_nbins, quantization_dtype)  # OPT-B narrow codes
                    try:
                        from ._pairs_core import _fe_gpu_binning_enabled
                        if _fe_gpu_binning_enabled(_ev_buf.shape[0], _ev_col):
                            from .._gpu_resident_fe import gpu_discretize_codes_host
                            # defer_host_fill: the codes flow straight into _dispatch_batch_mi_with_noise_gate,
                            # whose resident-CUDA gate consumes the DEVICE codes in place; the host buffer is
                            # filled lazily only on a host-reading branch. Skips the (n, K) codes D2H whenever
                            # the resident gate is the consumer. Bit-identical (host buffer == device.get()).
                            _ev_disc = gpu_discretize_codes_host(
                                _ev_buf[:, :_ev_col], int(quantization_nbins), dtype=_ev_code_dtype,
                                defer_host_fill=True,
                            )
                    except Exception:
                        _ev_disc = None
                    if _ev_disc is None:
                        _ev_disc = discretize_2d_quantile_batch(
                            _ev_buf[:, :_ev_col], n_bins=quantization_nbins,
                            dtype=_ev_code_dtype,
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
                                          usability=_leader_usability, mi_band=_mi_band)
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
            _winner = _select_single_best(_lead_perf, cols, usability=_leader_usability, mi_band=_mi_band)
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
        and (final_transformed_vals is not None or _this_chunk_deferred)
        and this_pair_features
        and best_mi > 0
    ):
        _emit_floor = float(best_mi) * float(fe_multi_emit_mi_floor)
        _div_corr = float(fe_multi_emit_diversity_corr)
        _already = {c for c, _ in this_pair_features}
        _emitted_cols = []
        # Stable, hash-independent order (set of name-string-bearing configs); see the determinism
        # note at the survivor-packing loop below.
        for _c in sorted(_already, key=lambda _cfg: get_new_feature_name(fe_tuple=_cfg, cols_names=cols)):
            try:
                _emitted_cols.append(np.asarray(_resolve_col(_c[2]), dtype=np.float64))
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
                _col = np.asarray(_resolve_col(_cfg[2]), dtype=np.float64)
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

        # ``this_pair_features`` is a SET of ``(config, j)`` tuples whose ``config`` holds
        # transformation-NAME strings, so a bare ``enumerate(set)`` iterates in PYTHONHASHSEED-
        # randomised order (string hashing is salted per process) whenever a pair emits more than
        # one form (multi-emit). That order becomes the ``new_cols`` / ``transformed_vals`` column
        # order below, which propagates into the engineered-column ordering downstream tie-breaks
        # depend on -- a latent cross-process determinism hole. Iterate in a STABLE, hash-independent
        # order keyed on the engineered feature NAME (the canonical algebraic identity of the form)
        # so the emitted column order is reproducible across processes.
        _ordered_pair_features = sorted(
            this_pair_features,
            key=lambda _cj: get_new_feature_name(fe_tuple=_cj[0], cols_names=cols),
        )
        for idx, (config, j) in enumerate(_ordered_pair_features):
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
                elif _this_chunk_deferred:
                    # DEFERRED-float GPU path, NON-subsample (transformed_vars is at full n here):
                    # re-materialise the survivor column on the GPU (bit-identical to the buffer,
                    # full-n directly). The subsample case took the _rebuild_full_survivor_col
                    # branch above (full-n rebuild from raw), so this only handles full-n fits.
                    _col_full = _resolve_col(i)
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

    _pair_res_entry = (this_pair_features, transformed_vals, new_cols, new_nbins, messages)
    return _pair_res_entry
