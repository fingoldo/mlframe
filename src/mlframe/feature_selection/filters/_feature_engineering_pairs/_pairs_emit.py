"""Admitted-pair emission tail of ``check_prospective_fe_pairs`` (carved
2026-06-22, Tier E - second sibling).

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

import logging
from typing import Optional, Any

import numpy as np

from pyutilz.pythonlib import sort_dict_by_value

from ._pairs_gates import _PREWARP_UNARY, _select_single_best
from types import SimpleNamespace as _SimpleNamespace

logger = logging.getLogger(__name__)

# External validation (step 2) lives in the sibling module; re-exported here so the step chain and patch targets keep this name.
from ._pairs_emit_extval import _emit_pair_features_step2_read_here_once, _extval_host_buffer  # noqa: F401


def _survivor_device_column(resolve_col, buf_col):
    """The survivor column ``buf_col`` as a scrubbed float32 DEVICE array when the deferred-float path can produce it there, else ``None`` (the caller then
    takes the host column from ``resolve_col``)."""
    return getattr(resolve_col, "device", lambda *_: None)(buf_col)


def _survivor_std(col) -> float:
    """Standard deviation of a survivor column held on the host or the device (one scalar read for a device column)."""
    if type(col).__module__.split(".", 1)[0] == "cupy":
        import cupy as cp

        return float(cp.std(col.astype(cp.float64)))
    return float(np.std(col))


def _stack_survivor_columns(cols, n_rows: int):
    """The kept survivor columns as one float64 ``(n_rows, k)`` matrix: a device-backed lazy matrix when every column is on the device (nothing is copied
    back until a host consumer reads it), else the plain host buffer."""
    if cols and all(type(c).__module__.split(".", 1)[0] == "cupy" for c in cols):
        import cupy as cp

        from mlframe.feature_selection.filters._lazy_host_codes import LazyHostCodes

        return LazyHostCodes(cp.stack([c.astype(cp.float64, copy=False) for c in cols], axis=1), np.float64)
    buf = np.empty(shape=(n_rows, len(cols)), dtype=np.float64)
    for ci, cv in enumerate(cols):
        buf[:, ci] = cv if type(cv).__module__.split(".", 1)[0] != "cupy" else _device_to_host(cv)
    return buf


def _device_to_host(col) -> np.ndarray:
    """Host copy of one device survivor column, for the mixed host/device fallback."""
    import cupy as cp

    return np.asarray(cp.asnumpy(col))


def _finalize_survivor_column(col_full: np.ndarray) -> np.ndarray:
    """Return the survivor column produced by any of the ``_col_full`` branches (rebuild / buffer-view /
    GPU-resolve / recompute-fallback) unchanged, WITHOUT a second NaN/Inf scrub or a forced float64 upcast.

    Every producing branch already scrubs NaN/Inf before returning (``_rebuild_full_survivor_col``'s own
    ``nan_to_num``, the buffer-fill scrub in ``_pairs_score._score_one_pair``, ``_resolve_col``'s D2H
    ``nan_to_num``, and the recompute-fallback's inline ``nan_to_num`` a few lines above this call site), so
    a second scrub here is provably dead work on every call. dtype is intentionally NOT forced to float32:
    most branches already yield float32 (matching ``transformed_vars`` / the resident buffers), but the
    recompute-fallback's ``binary_transformations["div"]`` (``_safe_div``) internally upcasts to float64 for
    its eps-floor division and returns float64 - forcing float32 here would silently truncate that result,
    which is NOT a pure redundancy removal. Passing the dtype through as-is keeps every branch bit-identical
    to the pre-fix ``nan_to_num(asarray(..., float64))`` (a no-op scrub + a lossless upcast when the source
    was float32, an exact no-op when it was already float64); the caller then assigns each column into a
    float64 ``transformed_vals`` buffer via plain elementwise ``__setitem__``, which upcasts correctly
    regardless of the source dtype, so no consumer needs this intermediate forced to any particular dtype."""
    return col_full if type(col_full).__module__.split(".", 1)[0] == "cupy" else np.asarray(col_full)


from .._null_mi_threshold import pair_search_floor

PAIR_FAMILY_PERMUTATIONS = 9999  # resolution for alpha down to 1e-4 at the per-pair level


def _apply_pair_family_gate(pair_batches, valid_pairs_perf, classes_y, n_bins, alpha=0.01):
    """Family-wise gate over EVERY configuration scored for one admitted pair (the winner is the maximum over all of them).

    Cascade: the null-MI table floor for the whole family (K = all configurations) rejects the pair cheaply when even its best configuration sits
    below it. Otherwise the exact family-wise permutation test runs, and each configuration whose batch does not clear the family maximum is
    zeroed. Batches are gated together, never one at a time, so a small batch cannot pass against a threshold built from itself alone.
    """
    if not pair_batches:
        return
    from .._family_maxt_gate import family_pass_batches
    from .._null_mi_threshold import pair_search_floor

    k_total = int(sum(codes.shape[1] for _, codes in pair_batches))
    n_rows = int(pair_batches[0][1].shape[0])
    best = max(float(valid_pairs_perf.get(cfg, 0.0)) for cfg, _ in pair_batches)
    table_floor = pair_search_floor(n_rows, classes_y, int(n_bins), k_total)
    if table_floor is not None and best <= table_floor:
        for cfg, _ in pair_batches:
            valid_pairs_perf[cfg] = 0.0
        return
    passed = family_pass_batches([codes for _, codes in pair_batches], np.asarray(classes_y).ravel(), alpha=alpha, n_perm=PAIR_FAMILY_PERMUTATIONS)
    for (cfg, _), ok in zip(pair_batches, passed):
        if not ok:
            valid_pairs_perf[cfg] = 0.0


def _apply_pair_family_floor(valid_pairs_perf, classes_y, n_bins, n_rows, k_total):
    """Family gate of a pair whose validation candidates are too many to hold: the null-MI table floor for the whole family (``k_total``
    candidates) rejects every configuration when even the best validation MI sits below it. Above the floor nothing is zeroed, because the exact
    family-wise permutation test needs all the codes at once (see ``_pairs_emit_extval._FAMILY_EXACT_MAX_CELLS``)."""
    if not valid_pairs_perf or k_total <= 0:
        return
    from .._null_mi_threshold import pair_search_floor

    best = max(float(v) for v in valid_pairs_perf.values())
    table_floor = pair_search_floor(int(n_rows), classes_y, int(n_bins), int(k_total))
    if table_floor is not None and best <= table_floor:
        for cfg in list(valid_pairs_perf):  # a snapshot of the keys, so rewriting values can never be mistaken for resizing the dict
            valid_pairs_perf[cfg] = 0.0


def apply_pair_family_gate_from_state(st, classes_y, n_bins, alpha=0.01) -> bool:
    """Family-wise gate over every configuration of the current pair, read from the pair state. Returns True when the gate rejected the winner
    (the caller must then emit nothing for this pair)."""
    codes = st.__dict__.get("var_pairs_codes") or {}
    if not codes or not st.var_pairs_perf:
        return False
    old_best = st.best_config
    batches = [(cfg, codes[cfg]) for cfg in st.var_pairs_perf if cfg in codes]
    _apply_pair_family_gate(batches, st.var_pairs_perf, classes_y, int(n_bins), alpha)
    return bool(old_best is not None and old_best in st.var_pairs_perf and st.var_pairs_perf[old_best] == 0.0 and float(st.best_mi) > 0.0)


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
    _usability_primary=False,
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
    # Miller-Madow tie-band + op-code table + noise-gate-dispatch env gate: all resolved ONCE per
    # ``check_prospective_fe_pairs`` call (invariant across every pair) and threaded down through
    # ``_score_one_pair`` - not recomputed here per admitted pair / per tied-leader.
    _mi_band,
    _op_code_arr,
    _fe_env_gate,
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
    _fe_use_parallel_kernels,
    _dispatch_batch_mi_with_noise_gate,
    batch_mi_with_noise_gate,
    use_su_normalization,
    X,
):
    """Emit the survivor feature(s) for an ADMITTED pair. Returns the
    ``res[raw_vars_pair]`` tuple ``(this_pair_features, transformed_vals,
    new_cols, new_nbins, messages)``. Appends to ``messages`` (caller-owned)."""
    # Family-wise gate over the pair's K configurations: the winner is the maximum over K candidates, so its MI must clear the
    # 1 - alpha quantile of the maximum null over K noise candidates for this (n, bins, target entropy) fingerprint. Without it a pure-noise pair
    # wins its family by chance at roughly the per-candidate rate, times K.
    _pair_floor = pair_search_floor(int(_full_n_rows), classes_y, int(quantization_nbins), len(var_pairs_perf))
    if _pair_floor is not None and float(best_mi) <= _pair_floor:
        return this_pair_features, None, None, None, messages
    _cached_name, _name_cache, leading_features, st = _emit_pair_features_step1_st_simplenamespace_long(
        get_new_feature_name,
        cols,
        _marginal_uplift_accept,
        _passes_joint_gate,
        _prewarp_accept,
        var_pairs_perf,
        best_mi,
        fe_good_to_best_feature_mi_threshold,
        _corr_y_cont,
        final_transformed_vals,
        _this_chunk_deferred,
        _resolve_col,
        _config_by_i,
        transformed_vars,
        vars_transformations,
        binary_transformations,
        _safe_abs_corr,
    )

    # ABSOLUTE binned-MI tie band. Two FORMS of
    # the same raw pair that are monotone re-expressions of one algebraic target have MI equal up to the
    # plug-in bias scale; without a tie band an MI EPSILON (pure binning noise) crowns a form whose linear
    # usability is far worse (mixed: additive ``add(log(a),invsqrt(b))`` MI 0.1180 > exact ratio
    # ``div(sqr(a),b)`` 0.1167, yet |corr(y)| 0.25 vs 0.46 - the noise winner does not fuse cleanly and
    # leaves the ratio form as a fragment). Snapping the primary MI key to this band inside
    # ``_select_single_best`` lets the EXISTING linear-usability tie-break pick the linearly-usable form.
    # ``_mi_band`` is the caller-hoisted, call-invariant tie band (a parameter - see the per-call hoist
    # comment in ``check_prospective_fe_pairs``), not recomputed per admitted pair.

    if len(leading_features) > 1:
        if len(numeric_vars_to_consider) > 2:

            _ev_configs, _ev_op_codes, _ext_factors_sorted, valid_pairs_perf = _emit_pair_features_step1_comment_check_prospective(
                verbose, leading_features, var_pairs_perf, numeric_vars_to_consider, raw_vars_pair, _op_code_arr
            )
            _emit_pair_features_step2_read_here_once(
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
            )

            # ONE-BEST-PER-PAIR: the leading-features
            # equivalence class holds many near-identical representations
            # of the same algebraic target (a**2/b == div(sqr(a),b) ==
            # mul(sqr(a),reciproc(b)) == div(a,sqrt(b)) ...). The
            # pre-refactor code materialised EXACTLY ONE per raw pair;
            # the refactor regressed to emitting the whole class (~15
            # cols on the canonical fixture). Pick the single best by
            # TARGET MI (``var_pairs_perf`` - the primary objective),
            # using the external-validation MI (``valid_pairs_perf``)
            # only as a tie-break among target-MI-equal leaders. (Prior
            # bug: selected by external-validation MI alone, discarding
            # the true max-target-MI form - e.g. picking add(log(c),1/d)
            # MI=0.25 over the true mul(log(c),sin(d)) MI=0.32.)
            _emit_pair_features_step3_mi_over_true(
                leading_features,
                var_pairs_perf,
                cols,
                valid_pairs_perf,
                st,
                _mi_band,
                _usability_primary,
                _name_cache,
                _cached_name,
                verbose,
                messages,
                best_mi,
                pair_mi,
                this_pair_features,
            )
        else:
            # Can't narrow by external validation (only 2 vars total) -
            # still emit ONE best representative (highest engineered MI,
            # deterministic name tie-break) rather than the whole class.
            _emit_pair_features_step4_deterministic_name_tie(
                leading_features,
                var_pairs_perf,
                cols,
                st,
                _mi_band,
                _usability_primary,
                _name_cache,
                verbose,
                messages,
                _cached_name,
                best_mi,
                pair_mi,
                this_pair_features,
            )
    else:
        new_feature_name = _cached_name(best_config)
        if verbose:
            messages.append(
                f"{new_feature_name} is recommended to use as a new feature! (clear winner) best_mi={best_mi:.4f}, pair_mi={pair_mi:.4f}, rat={best_mi/pair_mi:.4f}"
            )
        j = 0
        this_pair_features.add((best_config, j))

    # MULTI-CANDIDATE DIVERSE EMISSION: the blocks above emit the
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
    _emit_pair_features_step2_than_single_best(
        fe_multi_emit_max_per_pair,
        final_transformed_vals,
        _this_chunk_deferred,
        this_pair_features,
        best_mi,
        fe_multi_emit_mi_floor,
        fe_multi_emit_diversity_corr,
        _cached_name,
        _resolve_col,
        var_pairs_perf,
        st,
        verbose,
        messages,
    )

    st.transformed_vals, st.new_cols, st.new_nbins = None, None, None

    this_pair_features = _emit_pair_features_step3_pair_features(
        this_pair_features,
        _cached_name,
        fe_max_steps,
        _use_subsample,
        _rebuild_full_survivor_col,
        _X_full,
        original_cols,
        unary_transformations,
        binary_transformations,
        _prewarp_spec_by_var,
        _gate_med_median_by_var,
        cols,
        engineered_operand_values,
        final_transformed_vals,
        _this_chunk_deferred,
        _resolve_col,
        _config_by_i,
        st,
        transformed_vars,
        vars_transformations,
        verbose,
        messages,
        _full_n_rows,
        quantization_nbins,
    )

    st._pair_res_entry = (this_pair_features, st.transformed_vals, st.new_cols, st.new_nbins, messages)
    return st._pair_res_entry


def _emit_pair_features_step1_comment_check_prospective(verbose, leading_features, var_pairs_perf, numeric_vars_to_consider, raw_vars_pair, _op_code_arr):
    """Step 1 of _emit_pair_features: lines starting at ``if verbose > 2:``."""
    if verbose > 2:
        logger.debug("Taking %d new features for a separate validation step!", len(leading_features))

    # Test all candidates as-is against the rest of the approved factors (also as-is). Candidates significantly outstanding (in terms of MI with target)
    # against any other approved factor are kept.
    valid_pairs_perf: dict[Any, Any] = {}
    # LAZY EXTERNAL VALIDATION: valid_pairs_perf feeds _select_single_best ONLY as the
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
    # ``_op_code_arr`` (the op-code table) is call-invariant - resolved ONCE per
    # ``check_prospective_fe_pairs`` call (see the per-call hoist comment there) - so it is
    # read here once per pair rather than rebuilt on every tied-leader iteration below.
    _ev_op_codes = _op_code_arr
    return _ev_configs, _ev_op_codes, _ext_factors_sorted, valid_pairs_perf


def _emit_pair_features_step3_mi_over_true(
    leading_features,
    var_pairs_perf,
    cols,
    valid_pairs_perf,
    st,
    _mi_band,
    _usability_primary,
    _name_cache,
    _cached_name,
    verbose,
    messages,
    best_mi,
    pair_mi,
    this_pair_features,
):
    """Step 3 of _emit_pair_features: lines starting at ``_primary_perf = {c: var_pairs_perf[c] for c in leading_features if c i``."""
    _primary_perf = {c: var_pairs_perf[c] for c in leading_features if c in var_pairs_perf}
    _winner = _select_single_best(
        _primary_perf,
        cols,
        secondary=valid_pairs_perf,
        usability=st._leader_usability,
        mi_band=_mi_band,
        usability_primary=_usability_primary,
        name_cache=_name_cache,
    )
    if _winner is not None:
        new_feature_name = _cached_name(_winner)
        if verbose:
            messages.append(
                f"{new_feature_name} is recommended to use as a new feature! (won in validation with other factors) best_mi={best_mi:.4f}, pair_mi={pair_mi:.4f}, rat={best_mi/pair_mi:.4f}"
            )
        this_pair_features.add((_winner, 0))


def _emit_pair_features_step4_deterministic_name_tie(
    leading_features, var_pairs_perf, cols, st, _mi_band, _usability_primary, _name_cache, verbose, messages, _cached_name, best_mi, pair_mi, this_pair_features
):
    """Step 4 of _emit_pair_features: lines starting at ``_lead_perf = {c: var_pairs_perf[c] for c in leading_features if c in v``."""
    _lead_perf = {c: var_pairs_perf[c] for c in leading_features if c in var_pairs_perf}
    _winner = _select_single_best(
        _lead_perf,
        cols,
        usability=st._leader_usability,
        mi_band=_mi_band,
        usability_primary=_usability_primary,
        name_cache=_name_cache,
    )
    if _winner is not None:
        this_pair_features.add((_winner, 0))
        if verbose:
            messages.append(
                f"{_cached_name(_winner)} is recommended to use as a new feature! (best of {len(leading_features)} near-equivalent leaders) best_mi={best_mi:.4f}, pair_mi={pair_mi:.4f}, rat={best_mi/pair_mi:.4f}"
            )


def _emit_pair_features_step1_st_simplenamespace_long(
    get_new_feature_name,
    cols,
    _marginal_uplift_accept,
    _passes_joint_gate,
    _prewarp_accept,
    var_pairs_perf,
    best_mi,
    fe_good_to_best_feature_mi_threshold,
    _corr_y_cont,
    final_transformed_vals,
    _this_chunk_deferred,
    _resolve_col,
    _config_by_i,
    transformed_vars,
    vars_transformations,
    binary_transformations,
    _safe_abs_corr,
):
    """Step 1 of _emit_pair_features: lines starting at ``st = _SimpleNamespace() # long-lived locals of this function (see the ``."""
    st = _SimpleNamespace()  # long-lived locals of this function (see the stage helpers below)
    st._pair_res_entry = None

    # ``get_new_feature_name(config)`` is pure over ``(config, cols)`` (both fixed for this
    # one admitted pair), but the same config's name is independently needed at several
    # points below (the emit message, the multi-emit diversity sort, the survivor-packing
    # sort, the materialise loop) plus inside ``_select_single_best``'s own name tie-break.
    # One dict shared across every lookup (incl. the ``_select_single_best`` calls below)
    # so each distinct config's name is computed once per pair, not once per lookup site.
    _name_cache: dict = {}

    def _cached_name(_cfg) -> str:
        """Return ``_cfg``'s generated feature name, computing and memoizing it on first use."""
        _n: Optional[str] = _name_cache.get(_cfg)
        if _n is None:
            _n = get_new_feature_name(fe_tuple=_cfg, cols_names=cols)
            _name_cache[_cfg] = _n
        return _n

    # If there is a group of leaders with almost the same performance, approve them through one of the other variables.
    # если будут возникать такие группы примерно одинаковых по силе лидеров, их придётся разрешать с помощью одного из других влияющих факторов
    # When the pair was admitted ONLY via the marginal-uplift path (the joint /
    # prewarp gates declined), the winner MUST be a non-prewarp form so the recipe
    # is replayable - restrict the leaders to elementary-library configs.
    st._restrict_to_nonprewarp = _marginal_uplift_accept and not (_passes_joint_gate or _prewarp_accept)
    leading_features = []
    for next_config, next_mi in sort_dict_by_value(var_pairs_perf).items():
        if next_mi > best_mi * fe_good_to_best_feature_mi_threshold:
            if st._restrict_to_nonprewarp and (next_config[0][0][1] == _PREWARP_UNARY or next_config[0][1][1] == _PREWARP_UNARY):
                continue
            leading_features.append(next_config)

    # LINEAR-USABILITY TIE-BREAK over the MI-leaders. MI is a RANK
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
    st._leader_usability = {}
    if len(leading_features) > 1 and _corr_y_cont is not None:
        _corr_y_cont_finite = np.isfinite(_corr_y_cont)
        for _lc in leading_features:
            try:
                _li = _lc[2]
                if final_transformed_vals is not None:
                    _lvals = final_transformed_vals[:, _li]
                elif _this_chunk_deferred:
                    # DEFERRED-float GPU path: re-materialise the leader column on the GPU
                    # (bit-identical to the buffer -> same linear-usability tie-break).
                    _dev_corr = getattr(_resolve_col, "abs_corr", lambda *_: None)(_li, _corr_y_cont, _corr_y_cont_finite)
                    if _dev_corr is not None:
                        st._leader_usability[_lc] = _dev_corr
                        continue
                    _lvals = _resolve_col(_li)
                elif _config_by_i is not None and _li in _config_by_i:
                    # Recompute fallback (no hoisted buffer): rebuild the leader's
                    # CONTINUOUS column from its (a_key, b_key, bin_func_name) metadata
                    # so the linear-usability tie-break is identical to the buffered path
                    # (the two paths MUST select the same survivor among MI-equal leaders).
                    st._a_key, st._b_key, st._bin_name = _config_by_i[_li]
                    st._pa = transformed_vars[:, vars_transformations[st._a_key]]
                    st._pb = transformed_vars[:, vars_transformations[st._b_key]]
                    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                        _lvals = binary_transformations[st._bin_name](st._pa, st._pb)
                    _lvals = np.nan_to_num(np.asarray(_lvals, dtype=np.float32), copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                else:
                    _lvals = None
                if _lvals is not None:
                    st._leader_usability[_lc] = _safe_abs_corr(_lvals)
            except Exception as e:  # nosec B112 - best-effort path
                logger.debug("leader usability computation for %r failed, skipping: %s", _lc, e)
                continue
    return _cached_name, _name_cache, leading_features, st


def _emit_pair_features_step2_than_single_best(
    fe_multi_emit_max_per_pair,
    final_transformed_vals,
    _this_chunk_deferred,
    this_pair_features,
    best_mi,
    fe_multi_emit_mi_floor,
    fe_multi_emit_diversity_corr,
    _cached_name,
    _resolve_col,
    var_pairs_perf,
    st,
    verbose,
    messages,
):
    """Step 2 of _emit_pair_features: lines starting at ``if int(fe_multi_emit_max_per_pair) > 1 and (final_transformed_vals is ``."""
    if int(fe_multi_emit_max_per_pair) > 1 and (final_transformed_vals is not None or _this_chunk_deferred) and this_pair_features and best_mi > 0:
        _emit_floor = float(best_mi) * float(fe_multi_emit_mi_floor)
        _div_corr = float(fe_multi_emit_diversity_corr)
        _already = {c for c, _ in this_pair_features}
        _emitted_cols = []
        # Stable, hash-independent order (set of name-string-bearing configs); see the determinism
        # note at the survivor-packing loop below.
        for _c in sorted(_already, key=_cached_name):
            try:
                _emitted_cols.append(np.asarray(_resolve_col(_c[2]), dtype=np.float64))
            except Exception as e:  # noqa: PERF203 -- per-iteration fault isolation is intentional
                logger.debug("resolving already-emitted column %r failed, skipping: %s", _c, e)
        for _cfg, _cfg_mi in sort_dict_by_value(var_pairs_perf).items():
            if len(this_pair_features) >= int(fe_multi_emit_max_per_pair):
                break
            if _cfg_mi < _emit_floor:
                break  # sorted desc: nothing below the floor remains
            if _cfg in _already:
                continue
            try:
                st._col = np.asarray(_resolve_col(_cfg[2]), dtype=np.float64)
            except Exception as e:  # nosec B112 - best-effort path
                logger.debug("resolving column %r failed, skipping: %s", _cfg, e)
                continue
            st._col = np.nan_to_num(st._col, nan=0.0, posinf=0.0, neginf=0.0)
            if float(np.std(st._col)) <= 1e-9:
                continue
            # DIVERSITY: skip a near-duplicate of any already-emitted column.
            _dup = False
            from mlframe.feature_selection.filters._feature_engineering_pairs._pairs_core import _abs_corr_zerofill_njit

            for _ec in _emitted_cols:
                if float(np.std(_ec)) <= 1e-9:
                    continue
                # One-pass njit correlation (bit-equivalent to the previous np.corrcoef on
                # nan_to_num'd inputs - both _col and _ec are already finite by contract via
                # _resolve_col; see _abs_corr_zerofill_njit's docstring) instead of a 2x2 corrcoef matrix.
                if _abs_corr_zerofill_njit(st._col, _ec) > _div_corr:
                    _dup = True
                    break
            if _dup:
                continue
            this_pair_features.add((_cfg, 0))
            _already.add(_cfg)
            _emitted_cols.append(st._col)
            if verbose:
                messages.append(f"{_cached_name(_cfg)} also emitted " f"(diverse multi-candidate, MI={_cfg_mi:.4f} vs best {best_mi:.4f})")


def _emit_pair_features_step3_pair_features(
    this_pair_features,
    _cached_name,
    fe_max_steps,
    _use_subsample,
    _rebuild_full_survivor_col,
    _X_full,
    original_cols,
    unary_transformations,
    binary_transformations,
    _prewarp_spec_by_var,
    _gate_med_median_by_var,
    cols,
    engineered_operand_values,
    final_transformed_vals,
    _this_chunk_deferred,
    _resolve_col,
    _config_by_i,
    st,
    transformed_vars,
    vars_transformations,
    verbose,
    messages,
    _full_n_rows,
    quantization_nbins,
):
    """Step 3 of _emit_pair_features: lines starting at ``if this_pair_features:``."""
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
        # then sliced to ``[:, :last_j + 1]`` - this gives
        # either a too-short buffer (if last_j was small) and
        # IndexError downstream, or holes (if last_j was large).
        # Pack each (config, j) into a compact column index
        # ``idx = 0..len(this_pair_features)-1`` instead.
        #
        # materialise the survivor
        # columns whenever FE runs (``fe_max_steps >= 1``), not only on
        # multi-step (``> 1``). Previously, with the default
        # ``fe_max_steps=1`` the recommended features were LOGGED but
        # ``transformed_vals`` stayed ``None`` - so the consumer
        # (_mrmr_fe_step) had nothing to append, the columns never
        # entered ``data``/``selected_vars``, and ``_engineered_features_``
        # stayed empty. Producing the buffer unconditionally lets the
        # single-step default actually emit engineered columns.
        # Materialise each survivor into a temp column FIRST, then apply
        # the NON-CONSTANT guard: a column that replays as
        # constant (std<=1e-9) or non-finite is a DEAD feature and must
        # never be appended - it reaches the downstream model carrying
        # zero variance. Several div(sqr(a),b)-family combos replayed
        # constant on the canonical fixture (degenerate quantile binning
        # of the heavy-tailed a**2/b). One-best-per-pair already keeps the
        # non-constant MI winner; this guard is defence-in-depth and also
        # compacts ``this_pair_features`` / buffers so the recipe builder
        # downstream never constructs a recipe for a dropped column.
        _kept_configs = []  # list[(config, j)] that survived the guard
        _kept_cols_vals = []  # list[np.ndarray] aligned with _kept_configs
        _kept_names = []

        # ``this_pair_features`` is a SET of ``(config, j)`` tuples whose ``config`` holds
        # transformation-NAME strings, so a bare ``enumerate(set)`` iterates in PYTHONHASHSEED-
        # randomised order (string hashing is salted per process) whenever a pair emits more than
        # one form (multi-emit). That order becomes the ``new_cols`` / ``transformed_vals`` column
        # order below, which propagates into the engineered-column ordering downstream tie-breaks
        # depend on - a latent cross-process determinism hole. Iterate in a STABLE, hash-independent
        # order keyed on the engineered feature NAME (the canonical algebraic identity of the form)
        # so the emitted column order is reproducible across processes.
        _ordered_pair_features = sorted(
            this_pair_features,
            key=lambda _cj: _cached_name(_cj[0]),
        )
        for _idx, (config, j) in enumerate(_ordered_pair_features):
            new_feature_name = _cached_name(config)
            _transformations_pair, _bin_func_name, i = config

            if fe_max_steps >= 1:
                if _use_subsample:
                    # SUBSAMPLE path: rebuild from raw _X_full so the survivor column
                    # carries the FULL n rows the caller expects (mrmr.py appends it
                    # back to its full-n ``data`` array). The MI sweep used a 200k
                    # subset; the survivor IDENTITIES are correct (bench shows
                    # jaccard=1.0 vs full-n at n_eff>=50k), so we just need to
                    # rematerialise the values at full resolution.
                    _col_full = _rebuild_full_survivor_col(
                        config,
                        _X_full,
                        original_cols,
                        unary_transformations,
                        binary_transformations,
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
                    # Under the strict-resident path the survivor stays a DEVICE column: the downstream stage bins it, derives the recipe edges and
                    # stores its continuous values from the device, and only copies back what a host consumer actually reads.
                    _col_full = _survivor_device_column(_resolve_col, i)
                    if _col_full is None:
                        _col_full = _resolve_col(i)
                else:
                    # CRITICAL #2 recompute-fallback (no subsample, tight RAM): rebuild
                    # the survivor column from its (a_key, b_key, bin_func_name)
                    # metadata via the cached unary table. transformed_vars is at
                    # full n in this path so the column lands at full n directly.
                    st._a_key, st._b_key, st._bin_name = _config_by_i[i]
                    st._pa = transformed_vars[:, vars_transformations[st._a_key]]
                    st._pb = transformed_vars[:, vars_transformations[st._b_key]]
                    st._col = binary_transformations[st._bin_name](st._pa, st._pb)
                    np.nan_to_num(st._col, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                    _col_full = st._col

                # Keep the RAW (float) engineered values, scrubbed of
                # nan/inf. CRITICAL: do NOT cast to the
                # integer ``quantization_dtype`` here. ``transformed_vals``
                # feeds two consumers downstream - (a) ``_mrmr_fe_step``
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
                _col_arr = _finalize_survivor_column(_col_full)
                _col_std = _survivor_std(_col_arr)
                if _col_std <= 1e-9:
                    if verbose:
                        messages.append(f"{new_feature_name} dropped at materialisation: dead column " f"(std={_col_std:.2e}, non-constant guard).")
                    continue
                _kept_cols_vals.append(_col_arr)
            _kept_configs.append((config, j))
            _kept_names.append(new_feature_name)

        # Rebuild the survivor set / buffers from ONLY the kept columns so
        # the recipe builder and the downstream dense consumer stay aligned.
        this_pair_features = set(_kept_configs)
        st.new_cols = list(_kept_names)
        if fe_max_steps >= 1 and _kept_cols_vals:
            # float buffer: holds RAW engineered values (discretised to
            # codes downstream; see the non-constant-guard comment above).
            st.transformed_vals = _stack_survivor_columns(_kept_cols_vals, _full_n_rows)
            st.new_nbins = [quantization_nbins] * len(_kept_cols_vals)
        else:
            st.transformed_vals, st.new_nbins = None, []
    return this_pair_features
