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

import numpy as np
from collections import defaultdict

logger = logging.getLogger("mlframe.feature_selection.filters.mrmr")

from .._fe_rejection_ledger import record_fe_rejection as _record_fe_rejection


def score_prospective_pairs(
    self,
    *,
    cached_MIs,
    numeric_vars_to_consider,
    checked_pairs,
    _pair_mm_bias,
    _pair_maxt_floor,
    _synergy_added_idx,
    fe_min_pair_mi_prevalence,
    _synergy_prev_resolved,
    _prevalence_debias_auto,
    data,
    classes_y,
    num_fs_steps,
    verbose,
    sort_dict_by_value,
):
    """Rank the gate-surviving candidate pairs; return (prospective_pairs, _prevalence_failed_synergy)."""
    vars_usage_counter = defaultdict(int)
    prospective_pairs = {}
    # PREVALENCE-FAILED SYNERGY RESCUE LEDGER (2026-06-12, F2 a**2/b miss): a synergy
    # pair (>=1 bootstrap-added operand) whose JOINT MI cleared the order-2 maxT floor
    # but missed the STRICTER ``fe_synergy_min_prevalence`` raw-MI ratio bar is recorded
    # here as ``{(pair_idx_tuple): pair_mi}``. The raw-MI prevalence ratio structurally
    # UNDER-estimates a smooth non-bilinear ratio interaction (the genuine ``a**2/b``
    # scores ratio ~1.11 -- below the 1.5 synergy bar -- yet a leak-safe rank-1 ALS pair
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
    for raw_vars_pair, pair_mi in sort_dict_by_value(cached_MIs).items():
        if len(raw_vars_pair) == 2:
            if raw_vars_pair in checked_pairs:
                continue
            if raw_vars_pair[0] in numeric_vars_to_consider and raw_vars_pair[1] in numeric_vars_to_consider:
                ind_elems_mi_sum = cached_MIs[(raw_vars_pair[0],)] + cached_MIs[(raw_vars_pair[1],)]
                # Guard against ZeroDivisionError: when both individual features have zero MI with target
                # (canonical 3-way XOR case: MI(x_i, y) = 0 for all i but the joint signal exists), any positive pair_mi
                # qualifies as infinite uplift -- keep the pair.
                # MM-DEBIAS (2026-06-09, IRON RULE): the maxT floor was computed on the
                # Miller-Madow-debiased joint-MI scale (per-pair bias subtracted inside the
                # null kernel), so subtract the SAME per-pair joint-MI bias from the observed
                # ``pair_mi`` before the ``>= floor`` comparison -- consistent debias on both
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
                        uplift = float("inf")
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
                            observed=_pair_mi_floor_cmp, threshold=_pair_maxt_floor,
                            reason=("xor_zero_marginal_below_maxt" if pair_mi > 0 else "xor_zero_pair_mi"),
                            step=int(num_fs_steps),
                        )
                    continue
                # SYNERGY pairs (>=1 bootstrap-added operand) must clear a STRICTER
                # uplift bar than selected-selected pairs: their operands are
                # unselected (usually noise), and adding one as a 2nd joint
                # dimension inflates the finite-sample joint MI by ~5-15% bias,
                # which would clear the lenient 1.05 gate and inject a spurious
                # feature (observed regressing F-MONO). Genuine synergy has joint MI
                # far above the marginal sum, so the stricter bar keeps it.
                _is_synergy_pair = bool(_synergy_added_idx) and (
                    raw_vars_pair[0] in _synergy_added_idx or raw_vars_pair[1] in _synergy_added_idx
                )
                _prev_thresh = fe_min_pair_mi_prevalence
                if _is_synergy_pair:
                    _prev_thresh = max(fe_min_pair_mi_prevalence, _synergy_prev_resolved)
                # ORDER-2 maxT floor (computed once above) applied IN ADDITION to the
                # per-pair prevalence gate: the pair's JOINT MI must clear the pool's
                # permutation-null max as well, rejecting best-of-p chance-max noise
                # pairs the per-pair prevalence bar misses. No-op when floor==0.0.
                # ``_pair_mi_floor_cmp`` is the MM-debiased joint MI (IRON RULE, see above).
                # GUARDED "auto" (2026-06-13): compare the MM-DEBIASED pair MI against the same
                # ratio bar -- tightens the gate by the per-pair finite-sample bias, never loosens.
                # Default (explicit float) uses the raw pair_mi -> byte-identical.
                _obs_for_prevalence = _pair_mi_floor_cmp if _prevalence_debias_auto else pair_mi
                _passes_prevalence = _obs_for_prevalence > ind_elems_mi_sum * _prev_thresh
                _passes_maxt = _pair_mi_floor_cmp >= _pair_maxt_floor
                # DATA-DRIVEN PREVALENCE (2026-06-12, EXPERIMENTAL, default OFF pending the
                # 3-model RMSE A/B/C): the HARDCODED ratio bar over the MM-debiased joint MI
                # under-admits an ASYMMETRIC interaction whose one operand has a strong
                # marginal (the joint's analytic bias subtraction exceeds the marginals',
                # dropping the ratio below the bar even when the OTHER operand adds genuine
                # conditional signal -- F2's (c,d): MM-ratio ~1.03 < 1.05 yet CMI(d;y|c)
                # clears its within-stratum permutation null by +0.085, while noise (c,e)
                # sits ON the null). When the ratio bar fails but the pair cleared the maxT
                # floor, re-decide with a CONDITIONAL-PERMUTATION NULL (cancels the
                # finite-sample bias by construction). CAVEAT under measurement: CMI cannot
                # separate a multiplicative interaction (c,d) from an additive cross-mix
                # (a,c) -- both show CMI>0 -- so this over-admits cross-mix; whether the
                # resulting fused features HELP or HURT downstream is being decided by RMSE,
                # not assumed. ``fe_pair_perm_null_admission_enable`` (default False).
                _admit_via_perm = False
                if (
                    (not _passes_prevalence) and _passes_maxt
                    and bool(getattr(self, "fe_pair_perm_null_admission_enable", False))
                ):
                    try:
                        from .._fe_cmi_redundancy_gate import _conditional_perm_null
                        from .._mi_greedy_cmi_fe import _cmi_from_binned
                        _ia, _ib = int(raw_vars_pair[0]), int(raw_vars_pair[1])
                        if cached_MIs[(_ia,)] >= cached_MIs[(_ib,)]:
                            _anchor_i, _cand_i = _ia, _ib
                        else:
                            _anchor_i, _cand_i = _ib, _ia
                        _anchor_mi = float(cached_MIs[(_anchor_i,)])
                        _cand_codes = np.ascontiguousarray(data[:, _cand_i], dtype=np.int64)
                        _anchor_codes = np.ascontiguousarray(data[:, _anchor_i], dtype=np.int64)
                        _y_codes = np.ascontiguousarray(classes_y, dtype=np.int64)
                        _cmi_obs = float(_cmi_from_binned(_cand_codes, _y_codes, _anchor_codes))
                        _floor, _null_mean = _conditional_perm_null(
                            _cand_codes, _y_codes, _anchor_codes,
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
                    except Exception:
                        _admit_via_perm = False
                if (_passes_prevalence and _passes_maxt) or _admit_via_perm:
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
                    # failed (prevalence first -- it is the primary screen the session diagnoses).
                    if not _passes_prevalence:
                        # observed = joint MI; threshold = marginal-sum * prevalence bar.
                        _record_fe_rejection(
                            self, gate="marginal_pair_mi_prescreen",
                            candidate=str(raw_vars_pair), operands=raw_vars_pair, operator="pair",
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
                        # PAIRNESS-ROUTED RESCUE (2026-06-12): synergy pairs always route here
                        # (existing behaviour). With ``fe_prevalence_rescue_all_pairs`` ON, a
                        # SELECTED-SELECTED pair that cleared the maxT floor ALSO routes here, so
                        # the escalation's held-out ALS pairness test (which separates a genuine
                        # multiplicative interaction from an additive cross-mix) re-decides it.
                        _route_to_rescue = (
                            (_is_synergy_pair or bool(getattr(self, "fe_prevalence_rescue_all_pairs", False)))
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
                            observed=_pair_mi_floor_cmp, threshold=_pair_maxt_floor,
                            reason="below_maxt_floor", step=int(num_fs_steps),
                        )

    return prospective_pairs, _prevalence_failed_synergy
