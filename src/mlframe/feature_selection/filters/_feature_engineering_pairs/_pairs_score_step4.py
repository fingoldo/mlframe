"""Step 4 of ``_score_one_pair`` (prewarp clean-form demotion, usability admission) and its ``_config_corr`` helper, carved out of ``_pairs_score_steps``."""

from __future__ import annotations

from typing import Any
import logging

import numpy as np

from ._pairs_operand_floor import _beats_the_larger_operand
from ._pairs_abs_corr_gpu import candidate_abs_corr
from ._pairs_score_helpers import (
    logger,
    _score_one_pair_iteration_serialization_point_under,
    _score_one_pair_monotone_equivalent_warp_scores,
    _score_one_pair_mi_threshold_ratio_fix,
    _score_one_pair_uplift_fires_prewarp_config,
    _score_one_pair_warp_spec_round_trip,
)
from mlframe.utils.log_throttle import log_throttle


def _config_corr(
    _cfg,
    final_transformed_vals,
    _this_chunk_deferred,
    _resolve_col,
    _config_by_i,
    transformed_vars,
    vars_transformations,
    binary_transformations,
    _safe_abs_corr,
):
    """|corr(continuous y)| of a config's materialised continuous column; -1.0 when
    it cannot be rebuilt (so an unrecoverable form never wins the comparison)."""
    try:
        _ci = _cfg[2]
        if final_transformed_vals is not None:
            _v = final_transformed_vals[:, _ci]
        elif _this_chunk_deferred:
            # DEFERRED-float GPU path: re-materialise this column on the GPU (bit-identical
            # to the bulk buffer -> the clean-form demotion uses the EXACT same |corr| it
            # would have under the host buffer; a numpy recompute here flips it at ULP).
            _dev_corr = candidate_abs_corr(_resolve_col, _safe_abs_corr, _ci)
            if _dev_corr is not None:
                return _dev_corr
            _v = _resolve_col(_ci)
        elif _config_by_i is not None and _ci in _config_by_i:
            _ak, _bk, _bn = _config_by_i[_ci]
            _pa = transformed_vars[:, vars_transformations[_ak]]
            _pb = transformed_vars[:, vars_transformations[_bk]]
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                _v = binary_transformations[_bn](_pa, _pb)
            _v = np.nan_to_num(np.asarray(_v, dtype=np.float32), copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            return -1.0
        return _safe_abs_corr(_v)
    except Exception as e:
        # Unmeasured, not unrecoverable: -1.0 here used to switch the clean-form demotion OFF when the CLEAN side failed.
        log_throttle(
            logger,
            "pairs_score_config_corr_failed",
            logging.WARNING,
            "_config_corr: could not measure |corr| for config %r (%s: %s); the prewarp demotion treats it as unmeasured",
            _cfg,
            type(e).__name__,
            e,
        )
        return None


def _score_one_pair_step4_iteration_serialization_point(
    st,
    times_spent,
    verbose,
    raw_vars_pair,
    _corr_y_cont,
    _PREWARP_UNARY,
    fe_good_to_best_feature_mi_threshold,
    final_transformed_vals,
    _this_chunk_deferred,
    _resolve_col,
    _config_by_i,
    transformed_vars,
    vars_transformations,
    binary_transformations,
    _safe_abs_corr,
    pair_mi,
    fe_mm_debias_prevalence,
    classes_y,
    freqs_y,
    quantization_nbins,
    discretize_array,
    quantization_method,
    quantization_dtype,
    _operand_discretized,
    fe_min_engineered_mi_prevalence,
    num_fs_steps,
    _operand_marginal_mi,
    _prewarp_active,
    prewarp_uplift_threshold,
):
    """Step 4 of _score_one_pair: lines starting at ``_score_one_pair_iteration_serialization_point_under(st._local_times, t``."""
    _score_one_pair_step4_iteration_serialization_point_part1(
        st, times_spent, verbose, raw_vars_pair, _corr_y_cont, _PREWARP_UNARY, fe_good_to_best_feature_mi_threshold, final_transformed_vals,
        _this_chunk_deferred, _resolve_col, _config_by_i, transformed_vars, vars_transformations, binary_transformations, _safe_abs_corr, pair_mi,
    )
    st._gate_ratio = _score_one_pair_mi_threshold_ratio_fix(
        fe_mm_debias_prevalence,
        pair_mi,
        st.best_config,
        classes_y,
        freqs_y,
        quantization_nbins,
        discretize_array,
        _resolve_col,
        quantization_method,
        quantization_dtype,
        _operand_discretized,
        raw_vars_pair,
        st.best_mi,
        st._gate_ratio,
    )
    st._passes_joint_gate = st._gate_ratio > fe_min_engineered_mi_prevalence * (1.0 if num_fs_steps < 1 else 1.025)
    st._passes_joint_gate = st._passes_joint_gate and _beats_the_larger_operand(
        st.best_mi, _operand_marginal_mi, raw_vars_pair, st.messages if verbose else None, pair_mi
    )

    # Alternative pre-warp acceptance: the joint-prevalence gate
    # structurally rejects a 1-D summary of a 2-D pair on a non-monotone inner
    # distortion. Admit the prewarp winner when it beats the best NON-prewarp
    # engineered MI by ``prewarp_uplift_threshold`` AND clears the pair-MI
    # noise floor (its MI must exceed the larger individual operand MI - the
    # same notion the smart_polynom baseline uplift uses), so it cannot fire
    # on noise (where prewarp does not beat the library) or pure-linear data
    # (where the elementary library already saturates and the prewarp adds no
    # uplift). When it fires, the prewarp config becomes the winner.
    st._prewarp_accept = False
    st._prewarp_accept, st.best_config, st.best_mi = _score_one_pair_uplift_fires_prewarp_config(
        _prewarp_active,
        st._passes_joint_gate,
        st.best_prewarp_config,
        st.best_nonprewarp_mi,
        st.best_prewarp_mi,
        prewarp_uplift_threshold,
        verbose,
        st.messages,
        pair_mi,
        fe_min_engineered_mi_prevalence,
        st._prewarp_accept,
        st.best_config,
        st.best_mi,
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
    # warp spec to round-trip into the recipe - which is only guaranteed on the prewarp path.
    st._marginal_uplift_accept = False
    st._marginal_uplift_accept, st.best_config, st.best_mi = _score_one_pair_warp_spec_round_trip(
        st._passes_joint_gate,
        st._prewarp_accept,
        st.best_nonprewarp_config,
        st.best_nonprewarp_mi,
        pair_mi,
        _operand_marginal_mi,
        raw_vars_pair,
        verbose,
        st.messages,
        fe_min_engineered_mi_prevalence,
        st._marginal_uplift_accept,
        st.best_config,
        st.best_mi,
    )

    # TAIL-CONCENTRATION USABILITY ADMISSION + WINNER PROMOTION. Under heavy operand outliers a
    # genuine ratio (a**2/b) is TAIL-CONCENTRATED: in the clean bulk its rank association with y collapses
    # (Spearman ~0), so its rank-MI sits in a tight noise band where a SPURIOUS form out-ranks the true ratio
    # by a REAL margin the tie band cannot bridge (F2 with_outliers: min(reciproc(a),sign(b)) rank-MI 0.073
    # beats the true div(sqr(a),b) 0.063), AND the whole pair's best rank-MI fails the engineered-MI joint /
    # marginal-uplift gates (0.073/0.094 = 0.78 < the 0.90+ bar). Both failures are the SAME rank-MI blind
    # spot: the distinguishing signal is LINEAR (|corr(continuous y)| 0.986 true vs 0.371 spurious, corr
    # outlier-inflated - exactly right for a tail signal). Detect tail concentration on the RAW operands vs
    # the continuous y (``pair_is_tail_concentrated`` - best bivariate-form |corr| clears the bar AND beats
    # the best single-operand form by the pairness margin, so cross-mix / lone-dominant-operand / noise pairs
    # are rejected). When detected, PROMOTE the |corr|-best engineered FORM as the winner: this (a) admits the
    # pair when every rank-MI gate declined (``_usability_accept``) and (b) both widens the emit leader set to
    # include the true form - its rank-MI is BELOW ``fe_good_to_best_feature_mi_threshold`` x the spurious
    # leader, so it would otherwise be excluded - and flips the winner-selection primary key to |corr|
    # (``_usability_primary``) so the true ratio wins. The rank-vs-|corr| DISAGREEMENT gate
    # (``tail_concentration_form_override``) makes it a strict no-op on the 4 passing F2 profiles + canonical
    # fixtures (there the ratio is BOTH the rank-MI and |corr| leader, so nothing is promoted). Best-effort:
    # any error leaves the rank-MI decision untouched.
    st._usability_accept = False
    st._usability_primary = False


def _score_one_pair_step4_iteration_serialization_point_part1(
    st: Any, times_spent: Any, verbose: Any, raw_vars_pair: Any, _corr_y_cont: Any, _PREWARP_UNARY: Any, fe_good_to_best_feature_mi_threshold: Any,
    final_transformed_vals: Any, _this_chunk_deferred: Any, _resolve_col: Any, _config_by_i: Any, transformed_vars: Any, vars_transformations: Any,
    binary_transformations: Any, _safe_abs_corr: Any, pair_mi: Any,
) -> None:
    """Extracted block of ``_score_one_pair_step4_iteration_serialization_point`` (part 1); returns the names the caller still uses."""
    _score_one_pair_iteration_serialization_point_under(st._local_times, times_spent)

    if verbose > 2:
        logger.debug("For pair %s, best config is %s with best mi= %s", raw_vars_pair, st.best_config, st.best_mi)

    # CLEAN-FORM DEMOTION over the per-pair MI winner. The ``prewarp``
    # pseudo-unary fits a learned 1-D orthogonal-poly warp per operand; on a target whose
    # inner function is already LIBRARY-expressible up to a MONOTONE distortion (e.g.
    # ``log(c)*sin(d)`` - ``mul(log(c),sin(d))`` is the clean form, while the warp learns a
    # monotone re-expression ``mul(prewarp(c),sin(d))``) the prewarp form has IDENTICAL
    # ordering -> bit-equal binned MI, but MI is RANK-only so it cannot prefer the clean leg.
    # The warp then wins ``best_config`` by an MI tie/epsilon and propagates a DISTORTED form
    # (``log(div(sqr(a),neg(b)))`` for the a/b half, double-``prewarp(c)`` for the c/d half)
    # that the step-k>1 composite chains, displacing the clean additive compound AND dragging
    # in a redundant raw operand. Demote: when the global winner USES a prewarp operand but a
    # clean elementary-library (non-prewarp) form scores essentially the SAME target MI, keep
    # the prewarp winner ONLY if it has a real LINEAR-USABILITY uplift (|corr(continuous y)|)
    # over the best clean form. This preserves prewarp's INTENDED case - a genuinely
    # non-monotone inner (``a**3-2a``) where the warp's reconstruction is MORE linearly usable
    # than any library form, so the uplift is real and the prewarp form is kept - while making
    # the monotone-equivalent case (no |corr| uplift) fall back to the clean library compound.
    if (
        st.best_config is not None
        and st.best_nonprewarp_config is not None
        and st.best_config is not st.best_nonprewarp_config
        and st.best_nonprewarp_mi > 0.0
        and _corr_y_cont is not None
    ):
        _bc_uses_pw = (
            isinstance(st.best_config[0], (tuple, list))
            and len(st.best_config[0]) == 2
            and (st.best_config[0][0][1] == _PREWARP_UNARY or st.best_config[0][1][1] == _PREWARP_UNARY)
        )
        # Only act when the winner is a prewarp form AND the clean form is MI-equivalent
        # (within the same 0.85 leaders band already used as the equivalence notion). A
        # strictly-higher-MI prewarp winner is left untouched - the prewarp/marginal gates
        # below decide it on its own merits; demotion is for the MI-tie monotone case only.
        if _bc_uses_pw and st.best_nonprewarp_mi >= st.best_mi * fe_good_to_best_feature_mi_threshold:

            _corr_args = (
                final_transformed_vals,
                _this_chunk_deferred,
                _resolve_col,
                _config_by_i,
                transformed_vars,
                vars_transformations,
                binary_transformations,
                _safe_abs_corr,
            )
            _pw_corr = _config_corr(st.best_config, *_corr_args)
            _clean_corr = _config_corr(st.best_nonprewarp_config, *_corr_args)
            # Demote to the clean form unless the prewarp form is MEANINGFULLY more linearly
            # usable. ``1.05`` = the prewarp must beat the clean |corr| by >= 5% to justify the
            # distorted re-expression; a genuinely non-monotone inner clears this comfortably
            # (its warp reconstruction is the ONLY linearly-aligned form), while a
            # monotone-equivalent warp scores <= the clean form and is demoted.
            st.best_config, st.best_mi = _score_one_pair_monotone_equivalent_warp_scores(
                _pw_corr,
                _clean_corr,
                st.best_nonprewarp_config,
                st.best_nonprewarp_mi,
                st.var_pairs_perf,
                _PREWARP_UNARY,
                verbose,
                st.messages,
                st.best_config,
                st.best_mi,
            )

    # experiment-rejected (2026-06-03): a held-out-CV firewall here (score
    # per-combo MI on a TRAIN stride slice for honest selection, then keep the
    # winner only if its held-out VAL-slice MI retains >= ratio of train MI) was
    # implemented and benched END-TO-END on Layer-49 - NO gain. In an isolated
    # probe it separated cleanly (genuine synergy val/train 0.90-1.04 vs noise-FE
    # 0.12-0.36), BUT in the real pipeline the tighter prevalence-gate defaults
    # (fe_synergy_min_prevalence 1.5 / fe_min_engineered_mi_prevalence 0.97)
    # already remove the pure noise*noise products, and the RESIDUAL "noise" FE
    # are signal*noise combos (e.g. max(log(L4_s2),noise_3) - L4_s2 is a real
    # sensor) that genuinely generalise (val/train > 0.5) and SHOULD be kept; the
    # firewall's train-based selection (half the rows) then merely added selection
    # noise (+1 support). Prevalence gating subsumes the win -> not shipped.
    # Standard acceptance: the best engineered MI clears the configured
    # fraction of the 2-D pair-joint MI.
    #
    # MILLER-MADOW DEBIAS (+ #4). The RAW ratio
    # ``best_mi / pair_mi`` compares a 1-D engineered MI (over ~``quantization_nbins``
    # bins) against a 2-D joint MI (over ~``nbins^2`` bins). Both are plug-in MIs whose
    # positive bias is ``(k_x-1)(k_y-1)/2n``; the JOINT denominator's term is ~``nbins``x
    # larger, so the raw ratio is structurally depressed below 1.0 even when the 1-D
    # feature captures all the joint information (worst at small/moderate n) - this is
    # exactly the documented reason the marginal-uplift fallback gate had to be added.
    # When ``fe_mm_debias_prevalence`` we subtract the MM MI-bias term from BOTH sides,
    # using the OCCUPIED bin counts (#4: nominal ``nbins`` over-corrects heavy-tailed
    # columns that collapse), with a denominator-positivity guard that defers to the raw
    # ratio when the joint bias term swamps the finite-sample joint MI. ``->`` raw ratio
    # as ``n -> inf`` (bias terms vanish) => large-n selection byte-untouched. The order-2
    # maxT floor (the outer guard) is MM-debiased CONSISTENTLY upstream (the IRON RULE),
    # so admitting more pairs here does NOT weaken the best-of-pool noise floor.
    #
    # bench-attempt-rejected (2026-06-09, FS "permutation-null-calibrated
    # prevalence bar"). The idea: REPLACE the hardcoded ``fe_min_engineered_mi_prevalence``
    # (0.90) with a SELF-CALIBRATING per-pool null ratio - in the SAME K y-shuffles the
    # order-2 maxT floor runs, ALSO mirror the max-over-transforms search (discretise the
    # elementary binary bank mul/add/sub/div/max/min over the CONTINUOUS operands ONCE -
    # permutation-invariant - then per shuffle take max 1-D engineered MI / joint pair MI),
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
    # BUT bench-REJECTED on the case that matters - the user's WEAK F2
    # (``0.2*a**2/b + log(c*2)*sin(d/3)``, the SAME target that rejected #1/#8/#19): the null
    # ceiling is ~0.167 (calibrated to clean-noise pairs, ratio ~0.13-0.17), but EVERY weak-F2
    # pair sits FAR above it (5 seeds, n=20000): genuine_ab ~0.81, genuine_cd ~0.73, AND all four
    # cross-mix pairs 0.56-0.72 (cross(b,d) ~0.717 >= genuine_cd). So the null bar ADMITS every
    # cross-mix on every seed - the IRON-RULE failure mode, identical to #1 (cross-mix 3/10 ->
    # 9/10). ROOT CAUSE is the documented fundamental detectability limit (see
    # ``test_mrmr_weak_f2_seed_stability.py`` "THREE DIRECT LEVERS EXHAUSTED"): the cross-mix
    # smuggles the dominant MONOTONE predictor ``c`` across the pair boundary, so its 1-D
    # engineered summary recovers a large fraction of its (real, cross) joint - a HIGH ratio
    # indistinguishable from genuine synergy by ANY MI threshold. The null bar measures the
    # noise floor, but the weak-F2 problem is NOT noise admission; it is a real-monotone-predictor
    # cross-mix whose ratio is nowhere near the noise floor. AND the existing marginal-uplift /
    # prewarp FALLBACK already recovers the genuine pairs end-to-end at n=500/2000/8000, so #5
    # adds ZERO incremental recovery while WEAKENING cross-mix rejection. #5 is structurally a
    # 4th MI-threshold lever and fails by construction like #1/#8/#19; do NOT re-attempt an
    # MI-threshold/ratio fix here. Numbers + verdict in D:/Temp/null_prev_results.md.
    st._gate_ratio = (st.best_mi / pair_mi) if pair_mi > 0.0 else 0.0
