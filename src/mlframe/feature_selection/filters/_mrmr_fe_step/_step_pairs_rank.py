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

import numpy as np
from collections import defaultdict

logger = logging.getLogger("mlframe.feature_selection.filters.mrmr")

from .._fe_rejection_ledger import record_fe_rejection as _record_fe_rejection


def _pair_gate_resident_enabled() -> bool:
    """True when the device-born candidate-code residency is active for the prospective-pair CMI gate.

    Mirrors the CMI-redundancy gate (409d63fe): default ON under ``fe_gpu_strict_resident_enabled`` +
    ``_cmi_gpu_enabled``; opt-out ``MLFRAME_FE_GATE_RESIDENT_CANDS=0``. Any import fault -> off (host path)."""
    if os.environ.get("MLFRAME_FE_GATE_RESIDENT_CANDS", "1").strip().lower() not in ("1", "true", "on", "yes"):
        return False
    try:
        from .._gpu_strict_fe import fe_gpu_strict_resident_enabled
        from .._mi_greedy_cmi_fe import _cmi_gpu_enabled
        return bool(fe_gpu_strict_resident_enabled()) and bool(_cmi_gpu_enabled())
    except Exception:
        return False


def _resident_cand(codes):
    """Upload an ALREADY-BINNED host int64 candidate column ONCE and return the RESIDENT cupy codes, routed
    through the content-keyed resident cache so the SAME content the gate / earlier scorers already uploaded
    HITS that copy (no re-upload at the ``cmi_cand_x`` / ``card_cand_x`` / ``permnull_cand_x`` sites). Returns
    the host ``codes`` unchanged on any cupy fault / when residency is off (the device sites then upload it)."""
    try:
        from .._fe_resident_operands import resident_code_operand
        return resident_code_operand(np.asarray(codes).ravel(), "cmi_cand_x")
    except Exception:
        return codes


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
    # DEVICE-BORN candidate-code residency for the conditional-MI gates below (asymmetric-synergy relaxation +
    # the data-driven perm-null admission). Both read a candidate operand's ALREADY-BINNED int codes from
    # ``data`` and score CMI / conditional-perm-null on them; under residency the SCORED candidate is uploaded
    # ONCE and threaded resident (``_cmi_from_binned`` cupy resident-input branch + ``_conditional_perm_null``
    # resident-input branch) so it never re-crosses H2D at the ``cmi_cand_x`` / ``card_cand_x`` /
    # ``permnull_cand_x`` sites. The partner / anchor stays host (it is the conditioning ``z``, uploaded by the
    # primitives). Computed once (fit-constant predicate). Byte-identical: same int codes, same partition.
    _pair_resident = _pair_gate_resident_enabled()
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
                    # ASYMMETRIC-SYNERGY RELAXATION (2026-06-24, DOMINANT-CAPTURE fix for the F2
                    # ``mul(log(c),sin(d))`` half). The strict 1.5 synergy bar exists to suppress the
                    # finite-sample JOINT-MI bias inflation that lets a TWO-NOISE-operand synergy pair
                    # clear the lenient 1.05 gate. But it also blocks a GENUINE asymmetric synergy half:
                    # the F2 (c,d) pair has exactly ONE bootstrap operand (c), whose own MARGINAL MI is
                    # ~0 (E[sin(d)]~=0 zeroes MI(c;y)), yet log(c) carries real CONDITIONAL signal
                    # jointly with d. Its joint ratio (~1.13) clears 1.05 but fails 1.5, so the clean
                    # c/d half is never built and C2 has nothing to fuse. Relax to the REGULAR 1.05 bar
                    # ONLY for an asymmetric pair (exactly one bootstrap operand) whose bootstrap
                    # operand's CONDITIONAL signal given the partner is GENUINE -- a leak-safe held-out
                    # test, NOT a marginal-only heuristic. A marginal-only gate cannot separate the real
                    # (c,d) half from a noise cross-mix like canonical's (c,e): BOTH have a ~0-marginal
                    # bootstrap operand. The discriminator is CMI(boot; y | partner) vs its
                    # conditional-permutation null: genuine synergy clears the null by a wide margin
                    # (F2 (c,d): CMI 0.020 vs floor 0.004, excess 0.018) while a noise operand sits ON
                    # the null (canonical (c,e): CMI 0.004 vs floor 0.004, excess 0.003). Require the
                    # CMI to clear the null AND its excess over the null mean to be at least the null
                    # FLOOR magnitude (a chance-fluctuation scale) -- (c,d) excess 0.018 >= floor 0.004
                    # PASSES, (c,e) excess 0.003 < floor 0.004 FAILS. Best-effort: any error leaves the
                    # strict 1.5 bar in place (no relaxation), so canonical is never loosened on failure.
                    _idx0, _idx1 = raw_vars_pair[0], raw_vars_pair[1]
                    _b0 = _idx0 in _synergy_added_idx
                    _b1 = _idx1 in _synergy_added_idx
                    if _b0 != _b1:  # exactly one bootstrap operand -> asymmetric
                        _boot_idx = _idx0 if _b0 else _idx1
                        _partner_idx = _idx1 if _b0 else _idx0
                        try:
                            from .._fe_cmi_redundancy_gate import _conditional_perm_null
                            from .._mi_greedy_cmi_fe import _cmi_from_binned
                            _bcodes = np.ascontiguousarray(data[:, int(_boot_idx)], dtype=np.int64)
                            _pcodes = np.ascontiguousarray(data[:, int(_partner_idx)], dtype=np.int64)
                            _yc = np.ascontiguousarray(classes_y, dtype=np.int64)
                            # RESIDENT bootstrap-operand candidate (scored x) -> no re-upload; partner stays host z.
                            _bcand = _resident_cand(_bcodes) if _pair_resident else _bcodes
                            _cmi_obs = float(_cmi_from_binned(_bcand, _yc, _pcodes))
                            _cfloor, _cnull_mean = _conditional_perm_null(
                                _bcand, _yc, _pcodes,
                                seed=int(getattr(self, "random_seed", 0) or 0) + 6271 * int(_boot_idx) + int(_partner_idx),
                            )
                            if _cmi_obs > _cfloor and (_cmi_obs - _cnull_mean) >= _cfloor:
                                _prev_thresh = fe_min_pair_mi_prevalence
                        except Exception:
                            pass  # keep the strict 1.5 bar on any failure
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
                    except Exception:
                        _admit_via_perm = False
                # bench-attempt-rejected (2026-06-25): the cheap proxy below (2-operand joint OLS R^2 of the
                # CONTINUOUS y on the BINNED operand codes) does NOT recover with_outliers at any threshold
                # (0.05/0.15/0.3 all leave the selection byte-identical to OFF). The signal is too diluted:
                # y = a**2/b + f/5 + log(c)*sin(d), so a 2-var LINEAR fit on binned a,b captures only the linear
                # projection of ONE nonlinear third of y -> R^2 stays below any useful bar. The true
                # distinguisher (the materialised form div(sqr(a),b) vs y, |corr| 0.986) lives DOWNSTREAM of
                # this admission gate; a residual-target fit (y minus the already-captured c/d half) or actual
                # form materialisation is required, not an operand proxy. Kept default-OFF + wired (reject =
                # not-default, never deleted) so the next attempt builds on it instead of re-trying the proxy.
                # USABILITY ADMISSION (2026-06-25, tail-concentrated signal credit, default OFF). Under heavy
                # outliers a genuine ratio (a**2/b) is TAIL-CONCENTRATED: its rank-MI is suppressed (bulk
                # Spearman ~0, signal only in the tail) so it fails BOTH prevalence and maxT, and the rank-CMI
                # perm path is gated behind maxT too -- yet it carries strong LINEAR usability the gate never
                # sees (discrete codes only). Re-decide a rank-MI-rejected pair on the 2-operand joint OLS R^2
                # against the CONTINUOUS y: a tail-concentrated true pair has linear-usable joint structure
                # (high R^2) while a noise pair does not. Fires ONLY when BOTH rank-MI gates failed (so it can
                # only ADD pairs the rank path drops, never remove), and the FULL downstream FE admission gates
                # + escalation |corr| re-test still decide -- a false admit only PROPOSES. Default OFF
                # (canonical byte-identical); ``fe_pair_usability_admission_enable`` turns it on.
                _admit_via_usability = False
                if (
                    (not _passes_prevalence) and (not _passes_maxt)
                    and bool(getattr(self, "fe_pair_usability_admission_enable", False))
                ):
                    try:
                        _yc = getattr(self, "_fe_prewarp_y_continuous_", None)
                        if _yc is not None and len(_yc) == data.shape[0]:
                            _ya = np.asarray(_yc, dtype=np.float64).reshape(-1)
                            _A = np.column_stack([
                                data[:, int(raw_vars_pair[0])].astype(np.float64),
                                data[:, int(raw_vars_pair[1])].astype(np.float64),
                                np.ones(data.shape[0], dtype=np.float64),
                            ])
                            _coef, _, _, _ = np.linalg.lstsq(_A, _ya, rcond=None)
                            _ss_res = float(((_A @ _coef - _ya) ** 2).sum())
                            _ss_tot = float(((_ya - _ya.mean()) ** 2).sum())
                            _r2 = (1.0 - _ss_res / _ss_tot) if _ss_tot > 0 else 0.0
                            if _r2 >= float(getattr(self, "fe_pair_usability_admission_min_r2", 0.15)):
                                _admit_via_usability = True
                                if verbose >= 2:
                                    logger.info(
                                        "Factors pair %s ADMITTED via usability (joint OLS R^2=%.4f vs y) despite "
                                        "rank-MI rejection -- tail-concentrated linear signal.",
                                        raw_vars_pair, _r2,
                                    )
                    except Exception:
                        _admit_via_usability = False
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
