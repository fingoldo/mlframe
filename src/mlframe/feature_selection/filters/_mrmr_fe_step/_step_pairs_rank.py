"""Per-pair joint-MI uplift + order-2 maxT scoring stage of ``MRMR._run_fe_step``.

Carved verbatim from ``_step_core.py`` to keep that module under the 1k-LOC ceiling.
``score_prospective_pairs`` ranks the gate-surviving candidate pairs by joint-MI uplift, applies the
per-pair prevalence + order-2 Westfall-Young maxT floor (with the MM-debias / data-driven conditional-
permutation admission paths), records rejections, and returns the ``prospective_pairs`` ranking dict
plus the prevalence-failed synergy rescue ledger. The loop locals are threaded in as explicit keyword
args (no closure capture); selection is byte-for-byte identical to the inline block.
"""
from __future__ import annotations

import logging
import os
from mlframe.utils.env_flags import env_int
from typing import Any, Callable, Optional

import numpy as np
from collections import defaultdict, OrderedDict

logger = logging.getLogger("mlframe.feature_selection.filters.mrmr")

from mlframe.utils.log_throttle import log_throttle

from .._fe_rejection_ledger import record_fe_rejection as _record_fe_rejection

from .._fe_usability_signal import (  # shared leaf helpers (numpy-only, no cycle)
    _single_operand_usability_corr,
    usability_operand_continuous as _usability_operand_continuous,
)
from mlframe.feature_selection.filters._fe_family_timing import fe_timed

# Pair forms the usability scan evaluates per pair (the five pair forms of ALL_PAIR_FORM_IDS).


def _pair_gate_resident_enabled(*, n: int | None = None, p: int | None = None) -> bool:
    """True when the device-born candidate-code residency is active for the prospective-pair CMI gate.

    Mirrors the CMI-redundancy gate (409d63fe): default ON under ``fe_gpu_strict_resident_enabled`` +
    ``_cmi_gpu_enabled``; opt-out ``MLFRAME_FE_GATE_RESIDENT_CANDS=0``. Any import fault -> off (host path).

    ``n``/``p`` (optional): the calling dispatch's own shape, forwarded to ``_cmi_gpu_enabled`` so the
    STRICT/AUTO decision is size-aware for THIS call. Omit to preserve the shape-blind default."""
    if os.environ.get("MLFRAME_FE_GATE_RESIDENT_CANDS", "1").strip().lower() not in ("1", "true", "on", "yes"):
        return False
    try:
        from .._gpu_strict_fe import fe_gpu_strict_resident_enabled
        from .._mi_greedy_cmi_fe import _cmi_gpu_enabled
        return bool(fe_gpu_strict_resident_enabled()) and bool(_cmi_gpu_enabled(n=n, p=p))
    except Exception as e:
        logger.warning("_pair_gate_resident_enabled: residency check failed, the prospective-pair CMI gate stays on the host path: %s", e)
        return False


def _batch_usability_admission_verdicts(self, *, need_usability, y_continuous, cached_operand, cached_single_corr) -> dict:
    """Batched replacement for calling ``pair_is_tail_concentrated_rankaware`` once per pair in
    ``need_usability`` (a list of ``raw_vars_pair`` tuples already known to need the usability-admission
    verdict - see the pre-pass in :func:`score_prospective_pairs`). Returns ``{raw_vars_pair: bool}``;
    best-effort - any failure returns an empty dict so every caller lookup defaults to False (the strict
    rank-MI decision stands), matching the original inline try/except's failure mode exactly.

    Two-stage, matching ``batch_pair_tail_concentration_rankaware``'s own internal design: the expensive
    reduction (best pair-form |corr|) is batched across ALL pairs in one dispatch; the rank-correlation leg
    - which needs the winning form's actual VALUES, not just its |corr| - only materializes those values
    for the small subset that clears the cheap min_corr/pairness gate first. See that function's own
    docstring in ``batch_pair_usability_corr_gpu.py`` for the full numerical contract."""
    _usability_verdict: dict = {}
    if not need_usability:
        return _usability_verdict
    try:
        from .._fe_usability_signal import _corr_stride, _crit_np_dtype
        from ..batch_pair_usability_corr_gpu import batch_pair_tail_concentration_rankaware

        _min_corr_m = float(getattr(self, "fe_pair_usability_admission_min_corr", 0.6))
        _margin_m = float(getattr(self, "fe_pair_usability_admission_pairness_margin", 1.05))
        _rank_frac_m = float(getattr(self, "fe_pair_usability_admission_rank_frac", 0.7))

        _ya_m = np.asarray(y_continuous, dtype=_crit_np_dtype()).reshape(-1)
        _stride_m = _corr_stride(_ya_m.shape[0])
        _dtype_m = _crit_np_dtype()
        _y_batch_m = np.asarray(_ya_m[::_stride_m] if _stride_m > 1 else _ya_m, dtype=_dtype_m)

        _op_row_of_m: dict = {}
        _op_rows_m: list = []
        _sc_rows_m: list = []
        for _pk in need_usability:
            for _idx in (_pk[0], _pk[1]):
                if _idx not in _op_row_of_m:
                    _op_row_of_m[_idx] = len(_op_rows_m)
                    _full = cached_operand(_idx)
                    _sub = _full[::_stride_m] if _stride_m > 1 else _full
                    _op_rows_m.append(np.asarray(_sub, dtype=_dtype_m))
                    _sc = cached_single_corr(_idx)
                    _sc_rows_m.append(float(_sc) if _sc is not None else 0.0)
        _operand_matrix_m = np.vstack(_op_rows_m)
        _single_corr_m = np.asarray(_sc_rows_m, dtype=_dtype_m)
        _pair_a_m = np.array([_op_row_of_m[_pk[0]] for _pk in need_usability], dtype=np.int64)
        _pair_b_m = np.array([_op_row_of_m[_pk[1]] for _pk in need_usability], dtype=np.int64)

        _verdicts_m = batch_pair_tail_concentration_rankaware(
            _y_batch_m, _operand_matrix_m, _pair_a_m, _pair_b_m,
            min_corr=_min_corr_m, pairness_margin=_margin_m, max_rank_frac=_rank_frac_m,
            single_corr=_single_corr_m,
        )
        for _pk, _v in zip(need_usability, _verdicts_m):
            _usability_verdict[_pk] = bool(_v)
        return _usability_verdict
    except Exception as e:  # nosec B110 - optional/best-effort path, rationale documented
        # Every lookup then defaults to False (the strict rank-MI decision stands), which removes the usability admission route for the
        # whole step, so the failure is reported with its scope.
        logger.warning(
            "usability-verdict batch lookup failed (%s: %s); the usability admission route is off for all %d candidate pair(s) in this step",
            type(e).__name__, e, len(need_usability),
        )
        return {}


from ._step_pairs_prevalence import (  # noqa: F401  -- carved: the per-pair prevalence gate
    _PRESCAN_FREE_FRACTION,
    _prescan_budget_bytes,
    _host_free_bytes_safe,
    _collect_prescan_pair_keys,
    _dominant_pair_by_linear_corr,
    _maybe_relax_prevalence_for_tail_concentrated_pool,
    _get_col_codes_i64,
    _resolve_pair_prevalence_gate,
    _resident_cand,
)


def _prepass_gate_and_usability_candidates(
    *, cached_MIs, checked_pairs, numeric_vars_to_consider, data, sort_dict_by_value,
    cached_operand, usability_enabled, yc_shape_ok, gate_kwargs, sorted_pairs=None,
):
    """Walk every 2-tuple candidate pair ONCE (side-effect-free: no ``vars_usage_counter`` / rejection-ledger
    writes) to (a) resolve + cache the prevalence/maxT/synergy gate state every pair needs - see
    :func:`_resolve_pair_prevalence_gate` - and (b) collect the subset that will reach the usability-
    admission gate (fails the normal gates, has positive ``pair_mi``, usability admission is enabled, and
    both operands resolve) for the batched verdict pass. Filter conditions mirror ``score_prospective_pairs``'
    main loop's own OWN filter chain exactly (``len==2``, not in ``checked_pairs``, both operands considered,
    ``ind_elems_mi_sum>0``) so the returned ``_gate_cache`` has an entry for every pair the main loop will
    look up.

    ``sorted_pairs``: - pass the caller's already-computed
    ``sort_dict_by_value(cached_MIs)`` view (an unmutated dict between the two calls in
    ``score_prospective_pairs``) to avoid a second full O(n log n) sort + O(n) dict-copy of the whole
    candidate pool. Falls back to computing it here (the original behaviour) when omitted, so any other
    caller keeps working unchanged. A single ``_col_codes_cache`` dict is shared across every pair in this
    walk (``data`` is fixed for the whole call) so the asymmetric-synergy branch's bootstrap/anchor column
    codes are copied at most once per distinct column touched, not once per pair - see
    ``_get_col_codes_i64``."""
    _gate_cache: dict = {}
    _need_usability: list = []
    _col_codes_cache: dict = {}
    from ._conditional_maxt import ConditionalPartnerGate

    _conditional_gate = ConditionalPartnerGate(data, gate_kwargs["classes_y"], sorted(numeric_vars_to_consider), int(getattr(gate_kwargs["self"], "random_seed", 0) or 0))
    if sorted_pairs is None:
        sorted_pairs = sort_dict_by_value(cached_MIs)
    from ._pair_admission_batch import batched_asymmetric_relax_verdicts

    _eligible = [
        _pk for _pk in sorted_pairs
        if len(_pk) == 2 and _pk not in checked_pairs and _pk[0] in numeric_vars_to_consider and _pk[1] in numeric_vars_to_consider and cached_MIs[(_pk[0],)] + cached_MIs[(_pk[1],)] > 0
    ]
    _relax = batched_asymmetric_relax_verdicts(gate_kwargs["self"], _eligible, gate_kwargs.get("_synergy_added_idx"), data, gate_kwargs["classes_y"])
    for _pk, _pmi in sorted_pairs.items():
        if len(_pk) != 2 or _pk in checked_pairs:
            continue
        if _pk[0] not in numeric_vars_to_consider or _pk[1] not in numeric_vars_to_consider:
            continue
        _ies = cached_MIs[(_pk[0],)] + cached_MIs[(_pk[1],)]
        if _ies <= 0:
            continue
        _gate_cache[_pk] = _resolve_pair_prevalence_gate(
            _pk, _pmi, _ies, **gate_kwargs, _col_codes_cache=_col_codes_cache, single_mis=(cached_MIs[(_pk[0],)], cached_MIs[(_pk[1],)]), conditional_gate=_conditional_gate,
            relax_verdicts=_relax,
        )
        if not usability_enabled or not yc_shape_ok:
            continue
        _pp, _pt, _is_syn_ign, _prevt_ign, _floorcmp_ign = _gate_cache[_pk]
        if (_pp and _pt) or _pmi <= 0:
            continue  # usability never consulted: normal gates already satisfied, or pair_mi<=0
        _o0 = cached_operand(_pk[0])
        _o1 = cached_operand(_pk[1])
        if _o0 is None or _o1 is None or _o0.shape[0] != data.shape[0] or _o1.shape[0] != data.shape[0]:
            continue
        _need_usability.append(_pk)
    return _gate_cache, _need_usability


def _make_cached_operand(self: Any, X: Any, cols: Any, *, max_entries: int) -> Callable[[int], Optional[np.ndarray]]:
    """Build the per-call memo for an operand's continuous usability value, with least-recently-used eviction.

    Carved out of ``score_prospective_pairs`` to keep it under its length ceiling. The prescan is capped, but the gate-failure path asks
    for an operand for every pair with a positive pair MI, so an unbounded memo could hold one full-length column per operand in the pool:
    at 5000 operands and a million rows, tens of GB standing behind a memoization whose justification was CPU time. Least-recently-used
    eviction keeps the win, since a small pool of operands recurs across a much larger pool of pairs, without the footprint. Values are
    stored at the dtype both consumers cast to anyway, so nothing they compute changes.

    Args:
        self: the estimator the operand value is computed against.
        X: the frame the operand column is read from.
        cols: the column names, indexed by operand index.
        max_entries: the most operands held before the least recently used one is evicted.

    Returns:
        A one-argument callable mapping an operand index to its value, or ``None`` when it has none.
    """
    from .._fe_usability_signal import _crit_np_dtype as _op_cache_dtype

    _operand_cache: OrderedDict = OrderedDict()
    # The entry cap is a floor: the cache also grows to the operands that fit a tenth of the free host RAM, because a pool wider than the cap made every pair re-extract its two operand
    # columns from the frame (247104 extractions, 37 s, at 600 columns; the cache held 512 of ~700 operands and the scan order cycles through all of them).
    try:
        _row_bytes = max(1, len(X)) * np.dtype(_op_cache_dtype()).itemsize
        max_entries = max(int(max_entries), int(0.1 * _host_free_bytes_safe() // _row_bytes))
    except Exception as exc:
        logger.debug("operand cache sizing kept the entry cap: %s", exc)

    def _cached_operand(_idx):
        """Return operand ``_idx``'s continuous usability value, computing and memoizing it on first use."""
        _hit = _operand_cache.get(_idx)
        if _hit is not None:
            _operand_cache.move_to_end(_idx)
            return _hit
        _val = _usability_operand_continuous(self, X, cols, _idx)
        if _val is None:
            return None
        _val = np.asarray(_val, dtype=_op_cache_dtype())
        _operand_cache[_idx] = _val
        if len(_operand_cache) > max_entries:
            _operand_cache.popitem(last=False)
        return _val

    return _cached_operand


@fe_timed("stage:score_prospective_pairs")
def score_prospective_pairs(
    self,
    *,
    cached_MIs: dict,
    numeric_vars_to_consider: set,
    checked_pairs: set,
    _pair_mm_bias: Any,
    _pair_maxt_floor: Any,
    _synergy_added_idx: Any,
    fe_min_pair_mi_prevalence: float,
    _synergy_prev_resolved: Any,
    _prevalence_debias_auto: bool,
    data: np.ndarray,
    classes_y: np.ndarray,
    X: Any,
    cols: list[str],
    num_fs_steps: int,
    verbose: int,
    sort_dict_by_value: Callable[[dict], Any],
) -> tuple:
    """Rank the gate-surviving candidate pairs; return (prospective_pairs, _prevalence_failed_synergy)."""
    vars_usage_counter: defaultdict = defaultdict(int)
    prospective_pairs: dict = {}
    # Per-call memoization for ``_usability_operand_continuous``: both loops below call
    # it twice per candidate PAIR, but each raw operand appears in O(n_candidates) pairs, so the SAME column
    # gets re-extracted from ``X`` (pandas getitem + dtype cast + ravel) once per pair it participates in.
    # Measured 170,160 calls / 3.27s cumtime on a 100k-row production profile, almost entirely this
    # redundancy (X/cols/self are fixed for the lifetime of this one call, so the result for a given
    # ``var_idx`` never changes within it). A plain local dict is safe here - this function is called
    # serially from ``_step_core.py`` (no threading/loky dispatch at this level), so no shared-mutable-state
    # race risk (contrast the numba-typed-dict caches elsewhere in this package that DO cross worker threads
    # and need explicit per-worker copies).
    _cached_operand = _make_cached_operand(self, X, cols, max_entries=max(512, 2 * int(getattr(self, "fe_pair_usability_prescan_max_pairs", 256) or 256)))

    # Per-call memoization for the SINGLE-operand half of usability_form_corrs's ``_cs`` (perf
    # fix): ``pair_is_tail_concentrated_rankaware`` runs once per candidate PAIR (~85k calls in a wide 100k-row
    # FE fit) against a target fixed for this whole call, but only 2 of its 9 internal abs_pearson forms
    # (this operand's own value and its square) actually change per operand - the OTHER operand's forms and
    # all 5 pair forms genuinely depend on the specific pairing. Since a small pool of raw operands recurs
    # across a much larger pool of pairs (same rationale as ``_cached_operand`` above), caching THIS half by
    # operand index turns ~4 abs_pearson calls/pair into ~2 calls per UNIQUE operand - cProfile on a 100k-row
    # production run showed usability_form_corrs's single-operand share alone (a quarter of its 9 forms) was a
    # meaningful slice of the ~85k-call, ~95s cumtime this predicate costs. Bit-identical: see
    # ``_single_operand_usability_corr``'s docstring for why ``max()`` of the two cached halves equals the
    # non-cached flat ``max()`` over all 4 values exactly.
    _yc_cont_ = getattr(self, "_fe_prewarp_y_continuous_", None)
    _single_corr_cache: dict = {}

    def _cached_single_corr(_idx):
        """Return operand ``_idx``'s single-operand usability correlation, memoized across calls."""
        if _idx in _single_corr_cache:
            return _single_corr_cache[_idx]
        if _yc_cont_ is None:
            return None
        _op = _cached_operand(_idx)
        if _op is None or len(_yc_cont_) != _op.shape[0]:
            _single_corr_cache[_idx] = None
            return None
        _val = _single_operand_usability_corr(_yc_cont_, _op)
        _single_corr_cache[_idx] = _val
        return _val
 # PREVALENCE-FAILED SYNERGY RESCUE LEDGER (F2 a**2/b miss): a synergy
    # pair (>=1 bootstrap-added operand) whose JOINT MI cleared the order-2 maxT floor
    # but missed the STRICTER ``fe_synergy_min_prevalence`` raw-MI ratio bar is recorded
    # here as ``{(pair_idx_tuple): pair_mi}``. The raw-MI prevalence ratio structurally
    # UNDER-estimates a smooth non-bilinear ratio interaction (the genuine ``a**2/b``
    # scores ratio ~1.11 - below the 1.5 synergy bar - yet a leak-safe rank-1 ALS pair
    # warp beats its best single-operand warp held-out by ratio 1.24, cleanly separated
    # from cross-mix/noise at ~1.0). Rather than LOWER the raw-MI bar (bench-rejected:
    # injects optimisation-inflated noise products), these pairs are handed to the
    # auto-escalation as a SECOND CHANCE: its ``_propose_poly`` re-tests each with the
    # held-out pair-vs-single |corr| margin (``fe_escalation_pairness_margin``=1.15) and
    # min-val-corr floor, and the proposed candidate then faces the FULL admission gates
    # (order-2 maxT on MM-debiased MI + marginal-permutation floor + S5 CMI redundancy).
    # A false rescue only PROPOSES; the gates decide. Measured on F2: rescues the genuine
    # (a,b) term (esc_poly candidate MI 0.032 ~ the simple a**2/b, downstream R^2
    # 0.947 -> 0.997) while cross-mix/noise pairs propose nothing or are gated out.
    _prevalence_failed_synergy: dict = {}
    _score_prospective_pairs_part2(
        data, numeric_vars_to_consider, sort_dict_by_value, cached_MIs, self, num_fs_steps, verbose, fe_min_pair_mi_prevalence, _cached_operand, classes_y,
        _pair_mm_bias, _pair_maxt_floor, _synergy_added_idx, _synergy_prev_resolved, _prevalence_debias_auto, checked_pairs, _cached_single_corr,
        vars_usage_counter, prospective_pairs, cols, _prevalence_failed_synergy,
    )

    return prospective_pairs, _prevalence_failed_synergy


def _score_prospective_pairs_part2(
    data: np.ndarray, numeric_vars_to_consider: set, sort_dict_by_value: Callable[[dict], Any], cached_MIs: dict, self: Any, num_fs_steps: int, verbose: int,
    fe_min_pair_mi_prevalence: float, _cached_operand: Any, classes_y: np.ndarray, _pair_mm_bias: Any, _pair_maxt_floor: Any, _synergy_added_idx: Any,
    _synergy_prev_resolved: Any, _prevalence_debias_auto: bool, checked_pairs: set, _cached_single_corr: Any, vars_usage_counter: Any, prospective_pairs: Any,
    cols: list[str], _prevalence_failed_synergy: Any,
) -> None:
    """Extracted block of ``score_prospective_pairs`` (part 2); returns the names the caller still uses."""
    # DEVICE-BORN candidate-code residency for the conditional-MI gates below (asymmetric-synergy relaxation +
    # the data-driven perm-null admission). Both read a candidate operand's ALREADY-BINNED int codes from
    # ``data`` and score CMI / conditional-perm-null on them; under residency the SCORED candidate is uploaded
    # ONCE and threaded resident (``_cmi_from_binned`` cupy resident-input branch + ``_conditional_perm_null``
    # resident-input branch) so it never re-crosses H2D at the ``cmi_cand_x`` / ``card_cand_x`` /
    # ``permnull_cand_x`` sites. The partner / anchor stays host (it is the conditioning ``z``, uploaded by the
    # primitives). Computed once (fit-constant predicate). Byte-identical: same int codes, same partition.
    _pair_resident = _pair_gate_resident_enabled(n=int(data.shape[0]), p=len(numeric_vars_to_consider))
    # Compute the sorted view ONCE, here (before its first
    # consumer), and reuse it in every downstream consumer in this function - cached_MIs is not written to
    # anywhere between here and the main loop below, so re-sorting it a second or third time was a pure
    # duplicate O(n log n) sort + O(n) dict-copy of the whole candidate pool (singles + all pairs) for no
    # reason. This same view also gives the dominant-pair prescan below a principled top-K-by-MI candidate
    # source instead of raw insertion order, at zero extra sort cost.
    _sorted_pairs = sort_dict_by_value(cached_MIs)
    # See _maybe_relax_prevalence_for_tail_concentrated_pool's docstring for the full rationale (moved out
    # of this function to keep score_prospective_pairs' own cyclomatic complexity manageable).
    fe_min_pair_mi_prevalence = _maybe_relax_prevalence_for_tail_concentrated_pool(
        self=self, cached_MIs=cached_MIs, numeric_vars_to_consider=numeric_vars_to_consider, data=data,
        num_fs_steps=num_fs_steps, verbose=verbose, fe_min_pair_mi_prevalence=fe_min_pair_mi_prevalence,
        cached_operand=_cached_operand, sorted_pairs=_sorted_pairs,
    )

    # BATCHED USABILITY-ADMISSION PRE-PASS. Restores the
    # previously-dead-but-now-fixed ``_admit_via_usability`` gate (see its own comment below) at PRODUCTION
    # scale: the per-pair inline call used to run ``_pair_is_tail_concentrated_rankaware`` (9 abs_pearson
    # passes + a rank transform) once per pair reaching this gate - ~85k calls in a wide 100k-row
    # production fit, one of the largest hotspots profiled this session. As a side effect of needing to know
    # - for every pair, ahead of time - whether it will reach the usability gate, the prevalence/maxT/
    # synergy gate state (``_resolve_pair_prevalence_gate``) is ALSO now resolved once per pair here and
    # cached, rather than inline in the main loop: same total work, just relocated (the main loop was the
    # only place this used to run, so no computation happens twice).
    #
    # Two-stage, matching the batched kernel's own internal design: stage 1 batches ONLY the cheap
    # best-pair-form |corr| reduction for every candidate (the streamed on-core kernel never materializes a
    # 9x-wider form matrix - see ``batch_pair_usability_corr_gpu.py``); stage 2, the rank-correlation leg
    # (which genuinely needs the winning form's VALUES, not just its |corr|), runs only on the SMALL subset
    # that clears the min_corr/pairness gate - most candidate pairs reaching this gate are noise and fail
    # it immediately, so materializing winning-form arrays for only that subset (instead of all ~85k
    # candidates) is the whole point of the two-stage split. Bit-identical to calling
    # ``_pair_is_tail_concentrated_rankaware`` once per pair - verified by
    # ``test_score_prospective_pairs_usability_admission_batching.py`` (mirrors both code paths on synthetic
    # data, including the argmax-first-on-tie / subsample-stride / dtype-cast details the prescan batching
    # fix above already had to get right once). Best-effort: any failure leaves ``_usability_verdict`` empty
    # so every pair's lookup below defaults to False - the strict rank-MI decision stands, exactly as the
    # original inline try/except's failure mode.
    _usability_enabled = bool(getattr(self, "fe_pair_usability_admission_enable", True))
    _yc_pass = getattr(self, "_fe_prewarp_y_continuous_", None)
    _yc_shape_ok = _yc_pass is not None and len(_yc_pass) == data.shape[0]
    _gate_kwargs = dict(
        self=self, data=data, classes_y=classes_y, _pair_resident=_pair_resident, _pair_mm_bias=_pair_mm_bias,
        _pair_maxt_floor=_pair_maxt_floor, _synergy_added_idx=_synergy_added_idx,
        fe_min_pair_mi_prevalence=fe_min_pair_mi_prevalence, _synergy_prev_resolved=_synergy_prev_resolved,
        _prevalence_debias_auto=_prevalence_debias_auto,
    )
    # _sorted_pairs was already computed once, above, before the prevalence-relaxation prescan - reused here.
    _gate_cache, _need_usability = _prepass_gate_and_usability_candidates(
        cached_MIs=cached_MIs, checked_pairs=checked_pairs, numeric_vars_to_consider=numeric_vars_to_consider,
        data=data, sort_dict_by_value=sort_dict_by_value, cached_operand=_cached_operand,
        usability_enabled=_usability_enabled, yc_shape_ok=_yc_shape_ok, gate_kwargs=_gate_kwargs,
        sorted_pairs=_sorted_pairs,
    )
    _usability_verdict: dict = {}
    if _need_usability:
        _usability_verdict = _batch_usability_admission_verdicts(
            self, need_usability=_need_usability, y_continuous=_yc_pass,
            cached_operand=_cached_operand, cached_single_corr=_cached_single_corr,
        )

    from ._pair_admission_batch import batched_perm_null_admissions

    # the permutation-null admission verdicts of every eligible pair, one batched CMI per anchor block (``_pair_perm_null_admission`` reads them)
    self._pair_perm_admit_cache = batched_perm_null_admissions(self, _gate_cache, cached_MIs, data, classes_y)
    for raw_vars_pair, pair_mi in _sorted_pairs.items():
        if len(raw_vars_pair) == 2:
            if raw_vars_pair in checked_pairs:
                continue
            if raw_vars_pair[0] in numeric_vars_to_consider and raw_vars_pair[1] in numeric_vars_to_consider:
                ind_elems_mi_sum = cached_MIs[(raw_vars_pair[0],)] + cached_MIs[(raw_vars_pair[1],)]
                # Guard against ZeroDivisionError: when both individual features have zero MI with target
                # (canonical 3-way XOR case: MI(x_i, y) = 0 for all i but the joint signal exists), any positive pair_mi
                # qualifies as infinite uplift - keep the pair.
                # MM-DEBIAS (IRON RULE): the maxT floor was computed on the
                # Miller-Madow-debiased joint-MI scale (per-pair bias subtracted inside the
                # null kernel), so subtract the SAME per-pair joint-MI bias from the observed
                # ``pair_mi`` before the ``>= floor`` comparison - consistent debias on both
                # sides keeps the outer best-of-pool guard at full strength even though the
                # prevalence ratio bar downstream was lowered. No-op (0.0) when MM is off or
                # the pool is below the floor's min-pairs self-gate.
                _mm_pair_bias = _pair_mm_bias.get(tuple(sorted(raw_vars_pair)), 0.0)
                _pair_mi_floor_cmp = pair_mi - _mm_pair_bias
                if ind_elems_mi_sum <= 0:
                    # ORDER-2 maxT floor (computed once above): a zero-individual-MI
                    # pair enters via the canonical XOR branch on ANY positive joint
                    # MI, so on a wide noise matrix a noise pair whose joint MI is
                    # merely the best chance hit slips through. Require the joint MI
                    # to clear the pool's permutation-null max before keeping it;
                    # genuine pure-synergy (XOR / sign product) joint MI is FAR above
                    # the chance ceiling, so it survives. No-op when floor==0.0.
                    if pair_mi > 0 and _pair_mi_floor_cmp >= _pair_maxt_floor:
                        if verbose >= 2:
                            logger.info(
                                "Factors pair %s has zero individual MI but pair_mi=%.4f -- canonical hidden-pair case (e.g. XOR), keeping for FE",
                                raw_vars_pair, pair_mi,
                            )
                        prospective_pairs[(raw_vars_pair, pair_mi)] = vars_usage_counter[raw_vars_pair[0]] + vars_usage_counter[raw_vars_pair[1]]
                        for var in raw_vars_pair:
                            vars_usage_counter[var] += 1
                    else:
                        # REJECTION LEDGER (additive): a zero-individual-MI (XOR-branch) pair was
                        # dropped because its MM-debiased joint MI did not clear the order-2 max-T
                        # permutation null floor. Record observed=joint MI vs threshold=floor.
                        _record_fe_rejection(
                            self, gate="order2_maxt_floor",
                            candidate=str(raw_vars_pair), operands=raw_vars_pair, operator="pair",
                            operand_names="(" + ", ".join(str(cols[_i]) for _i in raw_vars_pair) + ")",
                            observed=_pair_mi_floor_cmp, threshold=_pair_maxt_floor,
                            reason=("xor_zero_marginal_below_maxt" if pair_mi > 0 else "xor_zero_pair_mi"),
                            step=int(num_fs_steps),
                        )
                    continue
                # Prevalence + order-2 maxT gate state (synergy debiasing, asymmetric-synergy CMI
                # relaxation, MM-debias floor comparison - see ``_resolve_pair_prevalence_gate``'s
                # docstring above for the full rationale) is resolved ONCE for every pair in the
                # usability-admission pre-pass above and cached here - not recomputed inline, so this
                # lookup is the SAME computation the pre-pass already paid for, just relocated earlier.
                # The pre-pass mirrors this loop's filter chain; if the two ever drift, skip the pair visibly rather than abort the fit.
                _gate_state = _gate_cache.get(raw_vars_pair)
                if _gate_state is None:
                    log_throttle(
                        logger, "fe_pair_gate_cache_miss", logging.WARNING,
                        "MRMR FE: no pre-pass gate state for pair %s (the pre-pass and scoring filter chains disagree); skipping it", raw_vars_pair,
                    )
                    continue
                _pair_mi_floor_cmp, var = _score_prospective_pairs_part1(
                    _gate_state, self, raw_vars_pair, cached_MIs, data, classes_y, _pair_resident, verbose, pair_mi, _usability_verdict, ind_elems_mi_sum,
                    vars_usage_counter, prospective_pairs, cols, num_fs_steps, _prevalence_failed_synergy, _pair_maxt_floor,
                )
    self._pair_perm_admit_cache = None  # the verdicts belong to this call only


def _score_prospective_pairs_part1(
    _gate_state: Any, self: Any, raw_vars_pair: Any, cached_MIs: dict, data: np.ndarray, classes_y: np.ndarray, _pair_resident: Any, verbose: int,
    pair_mi: Any, _usability_verdict: Any, ind_elems_mi_sum: Any, vars_usage_counter: Any, prospective_pairs: Any, cols: list[str], num_fs_steps: int,
    _prevalence_failed_synergy: Any, _pair_maxt_floor: Any,
) -> tuple[Any, ...]:
    """Extracted block of ``score_prospective_pairs`` (part 1); returns the names the caller still uses."""
    var: Any = None
    _passes_prevalence, _passes_maxt, _is_synergy_pair, _prev_thresh, _pair_mi_floor_cmp = _gate_state
    # DATA-DRIVEN PREVALENCE (EXPERIMENTAL, default OFF pending the
    # 3-model RMSE A/B/C): the HARDCODED ratio bar over the MM-debiased joint MI
    # under-admits an ASYMMETRIC interaction whose one operand has a strong
    # marginal (the joint's analytic bias subtraction exceeds the marginals',
    # dropping the ratio below the bar even when the OTHER operand adds genuine
    # conditional signal - F2's (c,d): MM-ratio ~1.03 < 1.05 yet CMI(d;y|c)
    # clears its within-stratum permutation null by +0.085, while noise (c,e)
    # sits ON the null). When the ratio bar fails but the pair cleared the maxT
    # floor, re-decide with a CONDITIONAL-PERMUTATION NULL (cancels the
    # finite-sample bias by construction). CAVEAT under measurement: CMI cannot
    # separate a multiplicative interaction (c,d) from an additive cross-mix
    # (a,c) - both show CMI>0 - so this over-admits cross-mix; whether the
    # resulting fused features HELP or HURT downstream is being decided by RMSE,
    # not assumed. ``fe_pair_perm_null_admission_enable`` (default False).
    _admit_via_perm = False
    _admit_via_perm = _pair_perm_null_admission(self, _passes_prevalence, _passes_maxt, raw_vars_pair, cached_MIs, data, classes_y, _pair_resident, verbose, _admit_via_perm)
    # bench-attempt-rejected : the cheap proxy below (2-operand joint OLS R^2 of the
    # CONTINUOUS y on the BINNED operand codes) does NOT recover with_outliers at any threshold
    # (0.05/0.15/0.3 all leave the selection byte-identical to OFF). The signal is too diluted:
    # y = a**2/b + f/5 + log(c)*sin(d), so a 2-var LINEAR fit on binned a,b captures only the linear
    # projection of ONE nonlinear third of y -> R^2 stays below any useful bar. The true
    # distinguisher (the materialised form div(sqr(a),b) vs y, |corr| 0.986) lives DOWNSTREAM of
    # this admission gate; a residual-target fit (y minus the already-captured c/d half) or actual
    # form materialisation is required, not an operand proxy. Kept default-OFF + wired (reject =
    # not-default, never deleted) so the next attempt builds on it instead of re-trying the proxy.
    # USABILITY ADMISSION (tail-concentrated ratio credit, default ON). Under heavy
    # operand outliers a genuine ratio (a**2/b) is TAIL-CONCENTRATED: its rank-MI collapses (bulk
    # Spearman ~0, signal only in the 5% tail) so it fails BOTH prevalence and maxT, and the
    # rank-CMI perm path is gated behind maxT too - yet it carries strong LINEAR usability the
    # rank gates never see. Unlike the prior binned-code OLS proxy (bench note above, which failed
    # because binning CLIPS the outlier tail that carries the a**2 magnitude), this scores the pair
    # on the RAW CONTINUOUS operands (outliers intact) via the max |Pearson corr(continuous y)|
    # over a small scale/sign-robust bivariate form dictionary - the header-validated
    # distinguisher (div(sqr(a),b) reads 0.986 vs y while the spurious rank-MI winner reads 0.371).
    # Fires ONLY when the rank-MI gates did not BOTH pass (can only ADD pairs the rank path dropped,
    # never remove) AND the best PAIR form beats the best SINGLE-operand form by a pairness margin,
    # so a cross-mix / noise pair where ONE operand dominates (no genuine pairness) is rejected -
    # this protects the 'e' noise operand and the canonical fixtures WITHOUT touching the rank-MI
    # decision on any pair the rank path admitted (the 4 passing profiles never enter this branch:
    # their (a,b) pair clears the rank gates). A false admit only PROPOSES; the full downstream FE
    # winner-selection + escalation |corr| re-test + redundancy gates still decide the final form.
    # Best-effort: any error leaves the strict rank-MI rejection in place (canonical never loosened
    # on failure). Toggle ``fe_pair_usability_admission_enable`` (default True; set False for the
    # legacy rank-MI-only, byte-identical admission).
    #
    # BUG FOUND AND FIXED: `_admit_via_usability` was computed below (via
    # `_pair_is_tail_concentrated_rankaware`, ~85k calls, one of the largest hotspots in a
    # 100k-row production profile) but its result was NEVER consulted in the accept condition -
    # it read `(_passes_prevalence and _passes_maxt) or _admit_via_perm`, missing
    # `or _admit_via_usability` entirely (grepped the whole file: no reassignment of
    # `_passes_prevalence`/`_passes_maxt` inside this block, no other reader of the variable, no
    # test anywhere references `fe_pair_usability_admission_enable` by name - this bug had zero
    # test coverage). Wiring it in (below) was verified NOT to change the canonical
    # `ratio_sqr/with_outliers` fixture this block's own comments target (identical selected-
    # feature lists with and without, at both n=10000 and n=30000) - that fixture's (a,b)
    # recovery comes from the SEPARATE first-sweep prevalence-relaxation prescan (this file's
    # dominant-pair rank-aware scan, above), which already relaxes the bar for the whole sweep
    # whenever the POOL's dominant pair is tail-concentrated. This per-pair block's real,
    # currently-untested value is the case the prescan structurally cannot reach: a NON-dominant
    # pair in the scan pool that is individually tail-concentrated while the pool's dominant pair
    # is not, so no global relaxation ever fires for it - exactly the gap the two-test check
    # above cannot exercise (both fixtures happen to have their tail-concentrated pair already be
    # the dominant one). Fixed rather than removed.
    # RANK-AWARE (not the loose |corr| check): admit via usability ONLY when the pair is linearly
    # usable AND its RANK association has COLLAPSED (the tail-concentration signature). A
    # genuinely usable ratio that also tracks y in rank (naturally-heavy-tailed / balanced data)
    # is NOT admitted here - the rank-MI gates already judge it, so canonical + the 4 passing
    # profiles stay byte-identical. Computed via the batched pre-pass above (``_usability_verdict``,
    # keyed by ``raw_vars_pair``) instead of an inline per-pair call - a pair absent from the
    # dict (trigger condition false, an operand failed to resolve, or the whole batch failed)
    # defaults to False here exactly as the original inline try/except's failure mode did.
    _admit_via_usability = False
    if (not (_passes_prevalence and _passes_maxt)) and pair_mi > 0 and bool(getattr(self, "fe_pair_usability_admission_enable", True)):
        _admit_via_usability = _usability_verdict.get(raw_vars_pair, False)
        if _admit_via_usability and verbose >= 2:
            logger.info(
                "Factors pair %s ADMITTED via usability (rank-collapsed tail-concentrated " "linear signal) despite rank-MI rejection.",
                raw_vars_pair,
            )
    if (_passes_prevalence and _passes_maxt) or _admit_via_perm or _admit_via_usability:
        uplift = pair_mi / ind_elems_mi_sum if ind_elems_mi_sum > 0 else float("inf")
        if verbose >= 2 and not _admit_via_perm:
            logger.info(
                "Factors pair %s will be considered for Feature Engineering, %.4f->%.4f, rat=%.2f",
                raw_vars_pair, ind_elems_mi_sum, pair_mi, uplift,
            )
        prospective_pairs[(raw_vars_pair, pair_mi)] = vars_usage_counter[raw_vars_pair[0]] + vars_usage_counter[raw_vars_pair[1]]
        for var in raw_vars_pair:
            vars_usage_counter[var] += 1
    else:
        # REJECTION LEDGER (additive): the pair failed the marginal pair-MI prevalence
        # pre-screen and/or the order-2 max-T null floor. Attribute to whichever leg
        # failed (prevalence first - it is the primary screen the session diagnoses).
        if not _passes_prevalence:
            # observed = joint MI; threshold = marginal-sum * prevalence bar.
            _record_fe_rejection(
                self, gate="marginal_pair_mi_prescreen",
                candidate=str(raw_vars_pair), operands=raw_vars_pair, operator="pair",
                operand_names="(" + ", ".join(str(cols[_i]) for _i in raw_vars_pair) + ")",
                observed=pair_mi, threshold=ind_elems_mi_sum * _prev_thresh,
                reason=("synergy_prevalence" if _is_synergy_pair else "prevalence_ratio"),
                step=int(num_fs_steps),
            )
            # SECOND-CHANCE RESCUE (see ``_prevalence_failed_synergy`` above):
            # a SYNERGY pair that cleared the order-2 maxT floor but only missed
            # the stricter raw-MI synergy ratio is handed to the auto-escalation,
            # where a leak-safe held-out ALS pair-warp test (not the raw-MI ratio)
            # re-decides. Gated on ``fe_synergy_prevalence_rescue_enable`` (default
            # on); only synergy pairs (selected-selected prevalence misses are
            # genuinely additive and stay out), only when the joint MI cleared the
            # maxT null (a pure-chance noise pair never reaches escalation).
            # PAIRNESS-ROUTED RESCUE: synergy pairs always route here
            # (existing behaviour). With ``fe_prevalence_rescue_all_pairs`` ON, a
            # SELECTED-SELECTED pair that cleared the maxT floor ALSO routes here, so
            # the escalation's held-out ALS pairness test (which separates a genuine
            # multiplicative interaction from an additive cross-mix) re-decides it.
            _route_to_rescue = (
                (True if _is_synergy_pair else bool(getattr(self, "fe_prevalence_rescue_all_pairs", False)))
                and _passes_maxt
                and bool(getattr(self, "fe_synergy_prevalence_rescue_enable", True))
            )
            if _route_to_rescue:
                _prevalence_failed_synergy[tuple(raw_vars_pair)] = float(pair_mi)
        else:
            # prevalence passed but the MM-debiased joint MI missed the max-T floor.
            _record_fe_rejection(
                self, gate="order2_maxt_floor",
                candidate=str(raw_vars_pair), operands=raw_vars_pair, operator="pair",
                operand_names="(" + ", ".join(str(cols[_i]) for _i in raw_vars_pair) + ")",
                observed=_pair_mi_floor_cmp, threshold=_pair_maxt_floor,
                reason="below_maxt_floor", step=int(num_fs_steps),
            )
    return _pair_mi_floor_cmp, var


def _pair_perm_null_admission(self, _passes_prevalence, _passes_maxt, raw_vars_pair, cached_MIs, data, classes_y, _pair_resident, verbose, _admit_via_perm):
    """Admit pairs that fail prevalence but pass max-T through the permutation null."""
    if (not _passes_prevalence) and _passes_maxt and bool(getattr(self, "fe_pair_perm_null_admission_enable", False)):
        batched = getattr(self, "_pair_perm_admit_cache", None)
        if batched and raw_vars_pair in batched:
            return bool(batched[raw_vars_pair])
        try:
            from mlframe.feature_selection.filters._fe_cmi_redundancy_gate import _conditional_perm_null
            from mlframe.feature_selection.filters._mi_greedy_cmi_fe import _cmi_from_binned
            _ia, _ib = int(raw_vars_pair[0]), int(raw_vars_pair[1])
            if cached_MIs[(_ia,)] >= cached_MIs[(_ib,)]:
                _anchor_i, _cand_i = _ia, _ib
            else:
                _anchor_i, _cand_i = _ib, _ia
            _anchor_mi = float(cached_MIs[(_anchor_i,)])
            _cand_codes = np.ascontiguousarray(data[:, _cand_i], dtype=np.int64)
            _anchor_codes = np.ascontiguousarray(data[:, _anchor_i], dtype=np.int64)
            _y_codes = np.ascontiguousarray(classes_y, dtype=np.int64)
            # Same scoring-subsample cap as the bootstrap block above: this data-driven prevalence
            # admission's observed CMI + permutation null are a wide-margin decision, subsample-safe.
            _pn_max2 = env_int("MLFRAME_PAIR_NULL_MAX_ROWS", 250000, minimum=0)
            _pn_n2 = int(_cand_codes.shape[0])
            if _pn_max2 > 0 and _pn_n2 > _pn_max2:
                _pn_st2 = int(_pn_n2 // _pn_max2)
                if _pn_st2 > 1:
                    _cand_codes = np.ascontiguousarray(_cand_codes[::_pn_st2])
                    _anchor_codes = np.ascontiguousarray(_anchor_codes[::_pn_st2])
                    _y_codes = np.ascontiguousarray(_y_codes[::_pn_st2])
            # RESIDENT candidate (scored x) -> no re-upload; anchor stays host z.
            _cand_dev = _resident_cand(_cand_codes) if _pair_resident else _cand_codes
            _cmi_obs = float(_cmi_from_binned(_cand_dev, _y_codes, _anchor_codes))
            _floor, _null_mean = _conditional_perm_null(
                _cand_dev, _y_codes, _anchor_codes,
                seed=int(getattr(self, "random_seed", 0) or 0) + 7919 * _ia + _ib,
            )
            _excess_frac = float(getattr(self, "fe_pair_perm_null_excess_frac", 0.05))
            if _cmi_obs > _floor and (_cmi_obs - _null_mean) >= _excess_frac * max(_anchor_mi, 1e-9):
                _admit_via_perm = True
                if verbose >= 2:
                    logger.info(
                        "Factors pair %s ADMITTED via conditional-permutation null "
                        "(CMI(%d|%d)=%.4f > floor %.4f, excess %.4f) -- data-driven prevalence.",
                        raw_vars_pair, _cand_i, _anchor_i, _cmi_obs, _floor, _cmi_obs - _null_mean,
                    )
        except Exception as e:
            logger.warning("permutation-based admission check failed, not admitting via permutation: %s", e)
            _admit_via_perm = False
    return _admit_via_perm
