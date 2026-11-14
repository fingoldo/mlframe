"""Prevalence gate of one pair in ``score_prospective_pairs``: the tail-concentration relaxation of the first sweep, the cached int64 column codes and the per-pair prevalence verdict.

Carved out of ``_step_pairs_rank`` (which re-exports every name here) to keep that module under the line budget.
"""

from __future__ import annotations

import logging
from typing import Any, cast

import numpy as np

from mlframe.utils.env_flags import env_int
from mlframe.utils.log_throttle import log_throttle

logger = logging.getLogger("mlframe.feature_selection.filters.mrmr")


# Share of the free host RAM one prescan block may hold (its operand rows and the usability kernel's working copies).
_PRESCAN_FREE_FRACTION = 0.25


def _prescan_budget_bytes(n_rows: int, dtype: Any) -> tuple[int, int]:
    """``(budget, per_row)``: the bytes one usability-scan block may upload and the bytes of one operand row of ``n_rows`` values.

    The budget is the host share of the free RAM, and when cupy is present the free VRAM less the cushion the dispatcher keeps,
    the dispatcher's own admission test (its footprint is the operand matrix at its dtype plus the double outputs)."""
    from .._orthogonal_univariate_fe._streamed_gpu_mi import _host_free_bytes

    n = max(1, int(n_rows))
    budget = _PRESCAN_FREE_FRACTION * _host_free_bytes()
    try:
        import cupy as cp

        from .._fe_gpu_vram import _cushion_bytes

        free_vram, total_vram = cp.cuda.runtime.memGetInfo()
        budget = min(budget, max(0, int(free_vram) - _cushion_bytes(int(total_vram))) * 0.9)
    except Exception as exc:  # no cupy or no device: the host budget governs
        logger.debug("prescan block sizing uses the host budget only: %s", type(exc).__name__)
    return int(budget), n * np.dtype(dtype).itemsize


def _host_free_bytes_safe() -> int:
    """Free host RAM in bytes (0 when it cannot be read)."""
    try:
        from .._orthogonal_univariate_fe._streamed_gpu_mi import _host_free_bytes

        return int(_host_free_bytes())
    except Exception as exc:  # unreadable host memory: no budget, so the smallest path runs
        log_throttle(
            logger, "host_free_bytes_unreadable", logging.WARNING,
            "host free memory unreadable (%s), budgeting zero bytes so the smallest path runs", type(exc).__name__,
        )
        return 0


def _resident_cand(codes):
    """Upload an ALREADY-BINNED host int64 candidate column ONCE and return the RESIDENT cupy codes, routed
    through the content-keyed resident cache so the SAME content the gate / earlier scorers already uploaded
    HITS that copy (no re-upload at the ``cmi_cand_x`` / ``card_cand_x`` / ``permnull_cand_x`` sites). Returns
    the host ``codes`` unchanged on any cupy fault / when residency is off (the device sites then upload it)."""
    try:
        from .._fe_resident_operands import resident_code_operand
        return resident_code_operand(np.asarray(codes).ravel(), "cmi_cand_x")
    except Exception as e:
        logger.debug("_resident_cand: resident-operand cache upload failed, returning the host codes unchanged: %s", e)
        return codes


from .._fe_usability_signal import (  # shared leaf helpers (numpy-only, no cycle)
    _crit_np_dtype,
    pair_is_tail_concentrated_rankaware as _pair_is_tail_concentrated_rankaware,
)


def _collect_prescan_pair_keys(cached_MIs, sorted_pairs, sort_dict_by_value, numeric_vars_to_consider, cached_operand, n_rows, max_prescan) -> list:
    """Return the pair keys to usability-scan, highest joint MI first, skipping pairs whose operands are missing or not ``n_rows`` long.

    ``max_prescan`` caps how many candidate pairs are visited (0 means no cap), so the scan stays bounded on a wide pool.
    """
    if sorted_pairs is not None:
        _sorted_pairs_local = sorted_pairs
    elif sort_dict_by_value is not None:
        _sorted_pairs_local = sort_dict_by_value(cached_MIs)
    else:
        from pyutilz.pythonlib import sort_dict_by_value as _sort_dict_by_value_fallback
        _sorted_pairs_local = _sort_dict_by_value_fallback(cached_MIs)
    scan_pks = []
    scanned = 0
    usable: dict = {}  # operand index -> whether its column exists and is ``n_rows`` long; each operand is looked up once, not once per pair (376k extractions at 600 columns)

    def _usable(idx: int) -> bool:
        """Whether operand ``idx`` can be scanned (looked up on first use)."""
        ok = usable.get(idx)
        if ok is None:
            op = cached_operand(idx)
            ok = usable[idx] = op is not None and op.shape[0] == n_rows
        return bool(ok)

    for _pk in reversed(_sorted_pairs_local.keys()):
        if len(_pk) != 2:
            continue
        if _pk[0] not in numeric_vars_to_consider or _pk[1] not in numeric_vars_to_consider:
            continue
        if max_prescan > 0 and scanned >= max_prescan:
            break
        scanned += 1
        if not (_usable(_pk[0]) and _usable(_pk[1])):
            continue
        scan_pks.append(_pk)
    return scan_pks


def _dominant_pair_by_linear_corr(scan_pks, ya_pre, cached_operand):
    """The pair with the highest linear |corr| against the continuous target, scored in memory-sized blocks; ``None`` when no pair scores."""
    _dom_cp, _dom_pk = -1.0, None
    from .._fe_usability_signal import _corr_stride
    from ..batch_pair_usability_corr_gpu import ALL_PAIR_FORM_IDS, dispatch_batch_pair_usability_corr

    # Match usability_form_corrs's OWN internal _subsample_for_corr stride exactly - it
    # subsamples (y, x0, x1) TOGETHER to _ABS_PEARSON_MAX_ROWS before ever computing a form;
    # skipping this here would silently run the batched call on the FULL row count instead
    # (wrong answer AND far slower - the whole point of the cap). Also cast to
    # _crit_np_dtype() (f32 by default) BEFORE batching - usability_form_corrs casts its
    # own y/x0/x1 internally, so a caller passing raw f64 here would compute every form at
    # STRICTLY HIGHER precision than the reference (a real, if small, ~1e-9 divergence from
    # comparing f32-then-f64-accumulated vs pure-f64 arithmetic - found via direct A/B
    # against usability_form_corrs, not assumed).
    _stride_pre = _corr_stride(ya_pre.shape[0])
    _dtype_pre = _crit_np_dtype()
    _y_batch = np.asarray(ya_pre[::_stride_pre] if _stride_pre > 1 else ya_pre, dtype=_dtype_pre)

    def _op_rows_of(idx: int) -> np.ndarray:
        """One operand's row, subsampled with the y rows."""
        _op_full = cached_operand(idx)
        return cast(np.ndarray, _op_full[::_stride_pre] if _stride_pre > 1 else _op_full)

    # Every scanned pair is scored. The pairs are grouped by the chunks of operands they touch, so a group's operand matrix (at most ``rows_per_block`` rows) is uploaded once and scores all of its
    # pairs in ONE call; with the pairs taken in scan order in blocks of ``budget / (2 rows)`` pairs, a budget below two operand rows (the card nearly full after the earlier stages) meant one pair
    # per call: 188191 calls at 600 columns. The winner is the largest value and, among equal values, the earliest pair in scan order: the pair one call over all pairs would return.
    _n_scan_operands = len({_i for _pk in scan_pks for _i in _pk})
    _budget_pre, _row_pre = _prescan_budget_bytes(_y_batch.shape[0], _dtype_pre)
    _rows_per_block = max(2, int(_budget_pre // max(1, _row_pre)))
    if _rows_per_block < 64 and _n_scan_operands > _rows_per_block:
        from .._orthogonal_univariate_fe._streamed_gpu_mi import _host_free_bytes

        # the device share is nearly used up: group by the host share and let the dispatcher admit or decline the device for each group
        _rows_per_block = max(_rows_per_block, int(_PRESCAN_FREE_FRACTION * _host_free_bytes() // max(1, _row_pre)))
    _chunk = max(1, min(_n_scan_operands, _rows_per_block // 2))
    _order: dict = {}
    for _pk in scan_pks:
        for _idx in _pk:
            _order.setdefault(_idx, len(_order))
    _groups: dict = {}
    for _pos, _pk in enumerate(scan_pks):
        _ca, _cb = _order[_pk[0]] // _chunk, _order[_pk[1]] // _chunk
        _groups.setdefault((min(_ca, _cb), max(_ca, _cb)), []).append(_pos)
    _row_cache: dict = {}
    _cache_rows = _n_scan_operands * _row_pre <= max(_budget_pre, 0) or _n_scan_operands * _row_pre <= _PRESCAN_FREE_FRACTION * _host_free_bytes_safe()

    def _row_of(idx: int) -> np.ndarray:
        """The subsampled operand row of ``idx`` in the scan dtype (kept for later groups when the rows of all operands fit the budget)."""
        row = _row_cache.get(idx)
        if row is None:
            row = np.asarray(_op_rows_of(idx), dtype=_dtype_pre)
            if _cache_rows:
                _row_cache[idx] = row
        return row

    _dom_cp, _dom_pos = -1.0, -1
    for _key in sorted(_groups):
        _positions = _groups[_key]
        _blk_row_of: dict = {}
        _blk_rows: list = []
        for _pos in _positions:
            for _idx in scan_pks[_pos]:
                if _idx not in _blk_row_of:
                    _blk_row_of[_idx] = len(_blk_rows)
                    _blk_rows.append(_row_of(_idx))
        _pair_a = np.array([_blk_row_of[scan_pks[_pos][0]] for _pos in _positions], dtype=np.int64)
        _pair_b = np.array([_blk_row_of[scan_pks[_pos][1]] for _pos in _positions], dtype=np.int64)
        _pair_form_corrs, _ = dispatch_batch_pair_usability_corr(
            _y_batch, np.vstack(_blk_rows), _pair_a, _pair_b, form_ids=ALL_PAIR_FORM_IDS,
        )
        _cp_block = _pair_form_corrs.max(axis=1)
        _bi = int(np.argmax(_cp_block))
        _val, _pos_best = float(_cp_block[_bi]), _positions[_bi]
        if _dom_pos < 0 or _val > _dom_cp or (_val == _dom_cp and _pos_best < _dom_pos):
            _dom_cp, _dom_pos = _val, _pos_best
    _dom_pk = scan_pks[_dom_pos] if _dom_pos >= 0 else None
    return _dom_pk


def _maybe_relax_prevalence_for_tail_concentrated_pool(
    *, self, cached_MIs, numeric_vars_to_consider, data, num_fs_steps, verbose,
    fe_min_pair_mi_prevalence, cached_operand, sorted_pairs=None, sort_dict_by_value=None,
):
    """TAIL-CONCENTRATION FIRST-SWEEP PREVALENCE RELAXATION. A co-signal half whose pair joint
    MI is only marginally above its (high) marginal sum - the F2 (c,d) ``mul(log(c),sin(d))`` half, ratio
    ~1.24/1.20 < the strict 1.05 bar - FAILS the strict pair-MI prevalence gate and never enters the FIRST
    FE sweep's prospective pool; today it builds ONLY in the adaptive-threshold RETRY, which fires solely
    when the first sweep yields ZERO features and then REPLACES (not merges) the pass. When a TAIL-
    CONCENTRATED pair is present (e.g. the outlier a/b half), the usability winner-promotion below makes the
    first sweep emit that half, so the retry is skipped and the co-signal half never builds in the SAME
    sweep - leaving C2 additive-fusion (which fuses two disjoint-token engineered halves) nothing to fuse.
    FIX: when usability admission is on AND a RANK-AWARE tail-concentrated pair exists in this pool (linear
    |corr(continuous y)| high AND genuinely pairwise AND its RANK association COLLAPSED - the outlier
    signature, FALSE for balanced canonical / the 4 passing profiles where rank and linear AGREE), rebind
    ``fe_min_pair_mi_prevalence`` to the SAME relaxed value the adaptive retry uses (``max(1.001, bar *
    fe_adaptive_relax_factor)``). The rest of the caller is then byte-identical to a retry call with that
    value, so BOTH the promoted tail half AND the co-signal half enter one sweep's pool and both build for
    fusion. No tail-concentrated pair -> the bar is untouched (byte-identical). The maxT floor + every
    downstream FE gate still apply. Best-effort: any failure keeps the strict bar (returns the input
    unchanged). Returns the (possibly relaxed) ``fe_min_pair_mi_prevalence``."""
    try:
        if not (bool(getattr(self, "fe_pair_usability_admission_enable", True)) and int(num_fs_steps) == 0):
            return fe_min_pair_mi_prevalence
        _yc_pre = getattr(self, "_fe_prewarp_y_continuous_", None)
        if _yc_pre is None or len(_yc_pre) != data.shape[0]:
            return fe_min_pair_mi_prevalence
        _ya_pre = np.asarray(_yc_pre, dtype=_crit_np_dtype()).reshape(-1)
        _min_corr_pre = float(getattr(self, "fe_pair_usability_admission_min_corr", 0.6))
        _margin_pre = float(getattr(self, "fe_pair_usability_admission_pairness_margin", 1.05))
        _rank_frac_pre = float(getattr(self, "fe_pair_usability_admission_rank_frac", 0.7))
        _max_prescan = int(getattr(self, "fe_pair_usability_prescan_max_pairs", None) or 0)  # 0 = scan every pair, in memory-sized blocks
        # DOMINANT-PAIR GATE (specificity): fire ONLY if the pool's dominant pair - the one with
        # the highest linear |corr(continuous y)| - is rank-collapsed. A small divisor operand makes
        # SPURIOUS forms (e.g. d/b ~ 1/b, which tracks an a**2/b target while d is noise) look tail-
        # concentrated on lower-|corr| pairs; those must NOT trigger the relaxation and disrupt cases
        # the existing path already fuses. The GENUINE tail-concentrated half is the dominant-|corr|
        # pair (a**2/b tracks y at |corr| ~0.99), so gating on the max-|corr| pair keeps the relaxation
        # specific: it fires for with_outliers (dominant (a,b) rank-collapsed) but NOT for a balanced /
        # naturally-heavy-tailed user case where the dominant ratio tracks y in rank too (rank ~ linear).
        _pool_tail_concentrated = False
        _dom_p0, _dom_p1, _dom_cp, _dom_pk = None, None, -1.0, None
        # BATCHED: the pre-fix version called _usability_form_corrs once per
        # scanned pair in a serial Python loop - measured 4.2-4.9x SLOWER at prescan-representative
        # scale (2k-30k pairs) than collecting the pool first and dispatching ONE batched call
        # (batch_pair_usability_corr_gpu's njit(parallel=True) backend, prange over the flattened
        # (pair, form) index - see bench_batch_pair_usability_corr_gpu.py). Bit-identical: same
        # per-pair two-pass reduction, same 5 pair-forms, same np.argmax first-on-tie semantics as
        # the original loop's strict `>` comparison (numpy argmax also returns the FIRST occurrence
        # of the max). dispatch_batch_pair_usability_corr's un-forced default is the warp-coalesced
        # CUDA kernel when CUDA is available (measured 3.96-4.52x faster than CPU; see that module's
        # docstring), and it honors the MLFRAME_DISABLE_GPU / CUDA_VISIBLE_DEVICES opt-out.
        # Scan candidates in DESCENDING joint-MI order (a principled
        # importance proxy for "how likely is this the pool's dominant pair"), not raw cached_MIs insertion
        # order (batch-precompute chunks, then legacy-sweep entries, then any GBM/gradient-seeded additions -
        # an accident of WHEN a pair was computed, not how relevant it is). For a pool wider than
        # fe_pair_usability_prescan_max_pairs the old insertion-order scan could silently miss the actual
        # highest-|corr| dominant pair if it happened to be computed late, weakening the "fires only for the
        # pool's genuinely dominant pair" specificity this function's docstring claims.
        # sort_dict_by_value's default is ASCENDING, so the highest-MI pairs are at the END.
        _scan_pks = _collect_prescan_pair_keys(
            cached_MIs, sorted_pairs, sort_dict_by_value, numeric_vars_to_consider, cached_operand, _ya_pre.shape[0], _max_prescan
        )
        if _scan_pks:
            _dom_pk = _dominant_pair_by_linear_corr(_scan_pks, _ya_pre, cached_operand)
            if _dom_pk is not None:
                _dom_p0 = cached_operand(_dom_pk[0])
                _dom_p1 = cached_operand(_dom_pk[1])
        if _dom_p0 is not None and _dom_p1 is not None and _pair_is_tail_concentrated_rankaware(
            _ya_pre, _dom_p0, _dom_p1,
            min_corr=_min_corr_pre, pairness_margin=_margin_pre, max_rank_frac=_rank_frac_pre,
        ):
            _pool_tail_concentrated = True
        if _pool_tail_concentrated:
            _relax = float(getattr(self, "fe_adaptive_relax_factor", 0.9))
            _relaxed_bar = max(1.001, float(fe_min_pair_mi_prevalence) * _relax)
            if _relaxed_bar < float(fe_min_pair_mi_prevalence):
                if verbose >= 2:
                    logger.info(
                        "Tail-concentrated pair detected in the pool; relaxing first-sweep pair-MI "
                        "prevalence %.3f -> %.3f so the co-signal half builds in the SAME sweep for C2 fusion.",
                        float(fe_min_pair_mi_prevalence), _relaxed_bar,
                    )
                return _relaxed_bar
        return fe_min_pair_mi_prevalence
    except Exception as e:  # nosec B110 - optional/best-effort path, rationale documented
        logger.debug("relaxed prevalence-bar computation failed, keeping the strict bar: %s", e)
        return fe_min_pair_mi_prevalence  # any failure keeps the strict prevalence bar (byte-identical)


def _get_col_codes_i64(data, col_idx: int, cache: "dict | None") -> np.ndarray:
    """Contiguous int64 codes for one column of ``data``, memoized per column index when ``cache`` is
    given. ``data[:, col_idx]`` on a row-major 2D array is never contiguous (column stride skips every
    other column), so ``np.ascontiguousarray`` on it is a REAL copy every call, not the near-free no-op
    it is on an already-contiguous 1-D array. A small set of bootstrap/anchor columns gets paired against
    many different partner columns across the prevalence-gate pre-pass (found live, wellbore-50k cProfile:
    41902 ``ascontiguousarray`` calls / 22.7s tottime here, confirmed by microbench that a column-slice
    copy costs ~1ms vs ~0.4us for an already-contiguous array) - caching collapses the repeats onto one
    copy per distinct column actually touched within the pre-pass."""
    if cache is None:
        return np.ascontiguousarray(data[:, int(col_idx)], dtype=np.int64)
    cached = cache.get(col_idx)
    if cached is None:
        cached = np.ascontiguousarray(data[:, int(col_idx)], dtype=np.int64)
        cache[col_idx] = cached
    return cached


def _resolve_pair_prevalence_gate(
    raw_vars_pair, pair_mi, ind_elems_mi_sum, *,
    self, data, classes_y, _pair_resident, _pair_mm_bias, _pair_maxt_floor,
    _synergy_added_idx, fe_min_pair_mi_prevalence, _synergy_prev_resolved, _prevalence_debias_auto,
    _col_codes_cache=None, single_mis=(0.0, 0.0), conditional_gate=None, relax_verdicts=None,
):
    """Pure (no side effects on ``vars_usage_counter`` / ``prospective_pairs`` / the rejection ledger)
    resolution of the prevalence + order-2 maxT gate state for ONE pair - factored out of the main loop
    (explicit kwargs, no closure capture, matching this module's own stated design) so the usability-
    admission pre-pass (which needs to know, for every pair, whether it will reach the usability gate) and
    the main loop itself always agree bit-for-bit, computed once each, never twice: neither this function's
    inputs nor its output depend on anything that changes DURING the main loop (``vars_usage_counter`` is
    write-only from this function's perspective), so evaluating it once, ahead of time, and caching the
    result is exactly equivalent to evaluating it inline per pair.

    SYNERGY pairs (>=1 bootstrap-added operand) must clear a STRICTER uplift bar than selected-selected
    pairs: their operands are unselected (usually noise), and adding one as a 2nd joint dimension inflates
    the finite-sample joint MI by ~5-15% bias, which would clear the lenient 1.05 gate and inject a spurious
    feature (observed regressing F-MONO). Genuine synergy has joint MI far above the marginal sum, so the
    stricter bar keeps it.

    ASYMMETRIC-SYNERGY RELAXATION (2026-06-24, DOMINANT-CAPTURE fix for the F2 ``mul(log(c),sin(d))``
    half). The strict 1.5 synergy bar exists to suppress the finite-sample JOINT-MI bias inflation that lets
    a TWO-NOISE-operand synergy pair clear the lenient 1.05 gate. But it also blocks a GENUINE asymmetric
    synergy half: the F2 (c,d) pair has exactly ONE bootstrap operand (c), whose own MARGINAL MI is ~0
    (E[sin(d)]~=0 zeroes MI(c;y)), yet log(c) carries real CONDITIONAL signal jointly with d. Its joint
    ratio (~1.13) clears 1.05 but fails 1.5, so the clean c/d half is never built and C2 has nothing to
    fuse. Relax to the REGULAR 1.05 bar ONLY for an asymmetric pair (exactly one bootstrap operand) whose
    bootstrap operand's CONDITIONAL signal given the partner is GENUINE - a leak-safe held-out test, NOT a
    marginal-only heuristic. A marginal-only gate cannot separate the real (c,d) half from a noise cross-mix
    like canonical's (c,e): BOTH have a ~0-marginal bootstrap operand. The discriminator is CMI(boot; y |
    partner) vs its conditional-permutation null: genuine synergy clears the null by a wide margin (F2
    (c,d): CMI 0.020 vs floor 0.004, excess 0.018) while a noise operand sits ON the null (canonical (c,e):
    CMI 0.004 vs floor 0.004, excess 0.003). Require the CMI to clear the null AND its excess over the null
    mean to be at least the null FLOOR magnitude (a chance-fluctuation scale) - (c,d) excess 0.018 >= floor
    0.004 PASSES, (c,e) excess 0.003 < floor 0.004 FAILS. Best-effort: any error leaves the strict 1.5 bar
    in place (no relaxation), so canonical is never loosened on failure."""
    _mm_pair_bias = _pair_mm_bias.get(tuple(sorted(raw_vars_pair)), 0.0)
    _pair_mi_floor_cmp = pair_mi - _mm_pair_bias
    _is_synergy_pair = bool(_synergy_added_idx) and (raw_vars_pair[0] in _synergy_added_idx or raw_vars_pair[1] in _synergy_added_idx)
    _prev_thresh = fe_min_pair_mi_prevalence
    if _is_synergy_pair:
        _prev_thresh = max(fe_min_pair_mi_prevalence, _synergy_prev_resolved)
        _idx0, _idx1 = raw_vars_pair[0], raw_vars_pair[1]
        _b0 = _idx0 in _synergy_added_idx
        _b1 = _idx1 in _synergy_added_idx
        if _b0 != _b1 and relax_verdicts is not None and raw_vars_pair in relax_verdicts:
            if relax_verdicts[raw_vars_pair]:  # decided by the batched pre-pass (same CMI, same null, same seed)
                _prev_thresh = fe_min_pair_mi_prevalence
        elif _b0 != _b1:  # exactly one bootstrap operand -> asymmetric
            _boot_idx = _idx0 if _b0 else _idx1
            _partner_idx = _idx1 if _b0 else _idx0
            try:
                from .._fe_cmi_redundancy_gate import _conditional_perm_null
                from .._mi_greedy_cmi_fe import _cmi_from_binned
                _bcodes = _get_col_codes_i64(data, _boot_idx, _col_codes_cache)
                _pcodes = _get_col_codes_i64(data, _partner_idx, _col_codes_cache)
                _yc = np.ascontiguousarray(classes_y, dtype=np.int64)
                # SCORING SUBSAMPLE. This bootstrap-prevalence relaxation runs an observed
                # CMI + a within-stratum permutation null on the FULL 1M pair codes - one of the top
                # _conditional_perm_null callers (~2.4s at 1M). The prevalence-relax verdict is a wide-
                # margin CMI/floor DECISION, selection-equivalent under a large strided subsample: cap the
                # candidate, partner, and y TOGETHER (aligned rows) above MLFRAME_PAIR_NULL_MAX_ROWS
                # (default 250k, 0=full-n) so the observed CMI + null decide on one consistent slice.
                _pn_max = env_int("MLFRAME_PAIR_NULL_MAX_ROWS", 250000, minimum=0)
                _pn_n = int(_bcodes.shape[0])
                if _pn_max > 0 and _pn_n > _pn_max:
                    _pn_st = int(_pn_n // _pn_max)
                    if _pn_st > 1:
                        _bcodes = np.ascontiguousarray(_bcodes[::_pn_st])
                        _pcodes = np.ascontiguousarray(_pcodes[::_pn_st])
                        _yc = np.ascontiguousarray(_yc[::_pn_st])
                # RESIDENT bootstrap-operand candidate (scored x) -> no re-upload; partner stays host z.
                _bcand = _resident_cand(_bcodes) if _pair_resident else _bcodes
                # kx/kz from the HOST code columns (numpy .max, no device sync): _bcand's resident
                # codes share _bcodes' cardinality, so passing it skips the device int(dx.max()) read
                # in _cmi_from_binned_cupy (the observed CMI's dominant per-pair scalar sync).
                _kx = (int(_bcodes.max()) + 1) if _bcodes.size else 1
                _kz = (int(_pcodes.max()) + 1) if _pcodes.size else 1
                _cmi_obs = float(_cmi_from_binned(_bcand, _yc, _pcodes, kx=_kx, kz=_kz))
                _cfloor, _cnull_mean = _conditional_perm_null(
                    _bcand, _yc, _pcodes,
                    seed=int(getattr(self, "random_seed", 0) or 0) + 6271 * int(_boot_idx) + int(_partner_idx),
                )
                if _cmi_obs > _cfloor and (_cmi_obs - _cnull_mean) >= _cfloor:
                    _prev_thresh = fe_min_pair_mi_prevalence
            except Exception as e:  # nosec B110 - optional/best-effort path, rationale documented
                logger.debug("permutation-null prevalence check failed, keeping the strict 1.5 bar: %s", e)
    # ORDER-2 maxT floor (computed once above) applied IN ADDITION to the per-pair prevalence gate: the
    # pair's JOINT MI must clear the pool's permutation-null max as well, rejecting best-of-p chance-max
    # noise pairs the per-pair prevalence bar misses. No-op when floor==0.0. ``_pair_mi_floor_cmp`` is
    # the MM-debiased joint MI (IRON RULE, see above).
    # GUARDED "auto": compare the MM-DEBIASED pair MI against the same ratio bar -
    # tightens the gate by the per-pair finite-sample bias, never loosens. Default (explicit float) uses
    # the raw pair_mi -> byte-identical.
    _obs_for_prevalence = _pair_mi_floor_cmp if _prevalence_debias_auto else pair_mi
    _passes_prevalence = _obs_for_prevalence > ind_elems_mi_sum * _prev_thresh
    _passes_maxt = _pair_mi_floor_cmp >= _pair_maxt_floor
    # The floor is the chance-max of a pair's joint MI over the pool. When one operand clears it on its own (a selected composite), every partner clears it too, so the joint MI is no evidence for
    # the PAIR: the weaker operand must stand out, as the best of all the candidate partners, in the information it carries beyond the stronger one.
    if _passes_prevalence and _passes_maxt and conditional_gate is not None and _pair_maxt_floor > 0.0 and max(single_mis) >= _pair_maxt_floor:
        _strong, _weak = (0, 1) if single_mis[0] >= single_mis[1] else (1, 0)
        try:
            _passes_maxt = conditional_gate.clears(raw_vars_pair[_weak], raw_vars_pair[_strong])
        except Exception as e:  # nosec B110 - optional/best-effort path, rationale documented
            logger.debug("family-wise partner test failed, keeping the floor verdict: %s", e)
    return _passes_prevalence, _passes_maxt, _is_synergy_pair, _prev_thresh, _pair_mi_floor_cmp
