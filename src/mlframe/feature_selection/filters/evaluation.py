"""Per-candidate evaluation: gain computation, confirmation checks.

See ``screen.py`` for the screening orchestrator that calls these functions.
"""
from __future__ import annotations

import logging
import os
from timeit import default_timer as timer
from typing import Sequence

import numba
import numpy as np
from numba import njit
from numba.core import types

from pyutilz.numbalib import generate_combinations_recursive_njit
# Module-level import so cloudpickle can resolve tqdmu when the function ships to a joblib worker.

from ._internals import LARGE_CONST, MAX_CONFIRMATION_CAND_NBINS
from ._numba_utils import arr2str, count_cand_nbins
from .gpu import mi_direct_gpu
from .info_theory import (
    conditional_mi, entropy, merge_vars, mi,
    # 2026-05-28: SU normalization dispatcher. ``cmi_or_csu`` reads a thread-local
    # toggle set by ``MRMR.fit`` when ``mi_normalization='su'``; legacy path
    # (toggle off) is one extra Python call ahead of njit kernels.
    use_su_normalization, conditional_symmetric_uncertainty, use_mi_miller_madow,
    # 2026-05-30 Wave 8: JMIM aggregator + BUR weight thread-local toggles
    # used in evaluate_gain / evaluate_candidate.
    use_jmim_aggregator, get_bur_lambda,
    # 2026-05-30 Wave 9.1 iter 5: setters for re-publishing the toggles into
    # joblib worker threads.
    get_relaxmrmr_alpha, get_pid_synergy_bonus, get_cmi_perm_stop,
)
from .permutation import mi_direct
from .info_theory._state_and_dispatch import get_group_mi
from .info_theory._group_mi import group_relevance_mi

logger = logging.getLogger(__name__)

# S-F3: the JMIM joint-MI aggregation applies the same ``nexisting`` discount exponent the Fleuret CMI branch uses. For a conditional MI (usually in [0,1] nats) ``x**k``
# shrinks the value (a discount for deeper interactions), but a JMIM JOINT MI ``I({X,Z};Y)`` is routinely >1 nat, and ``x**k`` AMPLIFIES values >1 -- the opposite of a
# discount. When set, ``MLFRAME_JMIM_EXPONENT_DISCOUNT_ONLY=1`` clamps the exponent so it can only ever shrink (never amplify) the joint MI, removing that spurious
# amplification. Read once at import so the njit ``evaluate_gain`` captures it as a compile-time constant (a bench toggles it per subprocess). Default off = bit-identical.
# bench-attempt-rejected: bench_sf3_jmim_exponent_selection.py (8 seeds, synergy fixture) -- the discount-only correction NEVER improved selection (0 seed wins), was
# selection-identical on 7/8 and regressed 1 seed (recall 0.8->0.6), mean recall 0.800 (exponent) vs 0.775 (corrected). The exponent is load-bearing, so it stays the default;
# the correction is kept as an off-by-default option for re-testing on other data/hardware.
_JMIM_EXPONENT_DISCOUNT_ONLY = os.environ.get("MLFRAME_JMIM_EXPONENT_DISCOUNT_ONLY", "0") == "1"

# 2026-06-19: observability sink for the JMIM joint-MI cache (see _evaluate_candidates_inner).
# Each completed JMIM-mode candidate-evaluation appends {"size", "hits"}. Bounded so a long
# fit cannot grow it without limit. Read by tests / benches to confirm the cache engages;
# carries no algorithmic meaning and is safe to clear() at any time.
from collections import deque as _deque
_JMIM_CACHE_STATS: "_deque" = _deque(maxlen=4096)

# Significance level for the permutation-null debiasing of the relevance MI. A candidate whose relevance permutation p-value is BELOW this alpha is SIGNIFICANT -- it sits
# above its own null distribution -- and keeps its FULL observed MI (no subtraction); a candidate at/above alpha is NOT significant and has the empirical null mean subtracted
# (``max(0, observed - null_mean)``), demoting it toward zero. This replaces the earlier fixed keep-fraction clamp with the textbook discriminator the clamp was a proxy for:
# the null MEAN alone cannot tell a WEAK GENUINE signal (whose coarse-binning null is a large fraction of its observed MI) apart from SPURIOUS NOISE (whose null is high because
# it IS noise) -- but the permutation p-value can, because weak genuine signal sits ABOVE its null (significant) while noise sits WITHIN its null (not significant). alpha=0.05 is
# a STANDARD statistical level, not a fixture-tuned coefficient; the selection is stable across a wide alpha range (0.02-0.10) precisely because real signal clears p ~ 0 and
# noise sits near p ~ 1, far from any alpha in that band. Subtraction is scale-preserving (stays in MI units) so the downstream min_relevance_gain / relative-gain / FDR floors
# need no recalibration. Override via the ``MLFRAME_MRMR_NULL_SIGNIF_ALPHA`` env var.
#
# RESOLUTION LIMIT: the p_value the gate compares against is the add-one Monte-Carlo estimate ``(1 + nfailed) / (full_budget + 1)`` (see permutation._perm_pvalue) over the
# null budget ``MLFRAME_MRMR_NULL_PERMS`` (default 32). It is therefore QUANTIZED in steps of ~1/(32+1) ~ 0.0303: the only attainable p-values near alpha are 0/33=0.0,
# 1/33~0.0303, 2/33~0.0606, ... -- so any alpha in the open interval (0.0303, 0.0606) is INDISTINGUISHABLE from alpha=0.0606, and a feature whose null is tied/beaten by exactly
# one shuffle ("1 tie") reads p~0.0606 >= 0.05 and is treated as NON-significant (its full null mean is subtracted). At the default budget the gate cannot resolve the
# 0.0303..0.0606 band; raise ``MLFRAME_MRMR_NULL_PERMS`` (finer steps ~1/(B+1), at proportional permutation cost) if you need to separate alpha values inside that band.
import os as _os
_MRMR_NULL_SIGNIF_ALPHA = float(_os.environ.get("MLFRAME_MRMR_NULL_SIGNIF_ALPHA", "0.05"))


def _materialize_var(factors_data, var_idx, factors_nbins, dtype=np.int32):
    """Densely-encode one variable index (or multi-index tuple/array) into a 1-D int column plus its effective cardinality K, via ``merge_vars`` (handles joints + bin
    pruning identically to the MI kernels). Used only by the research knobs (RelaxMRMR / PID / CMI-perm) whose standalone kernels take materialized columns, not indices."""
    idx = np.asarray(var_idx if hasattr(var_idx, "__len__") else [var_idx], dtype=np.int64)
    classes, _freqs, nclasses = merge_vars(
        factors_data=factors_data, vars_indices=idx, var_is_nominal=None,
        factors_nbins=factors_nbins, verbose=False, dtype=dtype,
    )
    return np.asarray(classes, dtype=np.int64), int(nclasses)


def _su_normalize_relevance(direct_gain: float, X, y, factors_data, factors_nbins, dtype) -> float:
    """SU-scale a marginal-relevance MI when ``mi_normalization='su'`` is active (no-op otherwise / on a non-positive gain).

    Scales by the SU denominator ``2/(H(X)+H(Y))`` so the value the ``min_relevance_gain`` floor compares against is the
    cardinality-scrubbed score, matching the unit SU definition. Shared by BOTH relevance-MI entry points so they stay on the
    SAME scale: the fresh ``mi_direct`` else-path AND the ``cached_confident_MIs`` branch (a confirmed candidate's bootstrapped
    gain, which is otherwise raw MI and would be compared against an SU-scale floor). A degenerate joint keeps the raw value.
    """
    if not (use_su_normalization() and direct_gain > 0.0):
        return direct_gain
    try:
        _x_idx = np.asarray(X, dtype=np.int64)
        _y_idx = np.asarray(y, dtype=np.int64)
        _, _freqs_x_su, _ = merge_vars(
            factors_data=factors_data, vars_indices=_x_idx, var_is_nominal=None,
            factors_nbins=factors_nbins, verbose=False, dtype=dtype,
        )
        _, _freqs_y_su, _ = merge_vars(
            factors_data=factors_data, vars_indices=_y_idx, var_is_nominal=None,
            factors_nbins=factors_nbins, verbose=False, dtype=dtype,
        )
        _denom_su = entropy(freqs=_freqs_x_su) + entropy(freqs=_freqs_y_su)
        if _denom_su > 1e-12:
            return 2.0 * direct_gain / _denom_su
    except Exception:
        # SU denominator unavailable (degenerate joint) -> keep the raw debiased relevance. Log at DEBUG so a
        # GENUINE merge_vars/entropy bug (not just a degenerate joint) is diagnosable instead of silently
        # masked -- a swallowed bug here can quietly change which features/interactions get selected.
        logger.debug("evaluation: SU normalization skipped; using raw debiased gain", exc_info=True)
    return direct_gain


def get_candidate_name(candidate_indices: list, factors_names: list) -> str:
    cand_name = "-".join([factors_names[el] for el in candidate_indices])
    return cand_name


def should_skip_candidate(
    cand_idx: int,
    X: tuple,
    interactions_order: int,
    failed_candidates: set,
    added_candidates: set,
    expected_gains: np.ndarray,
    selected_vars: list,
    selected_interactions_vars: list,
    only_unknown_interactions: bool = True,
    engineered_lineage: dict = None,
    dcd_state=None,
) -> tuple:
    """Decide if current candidate for predictors should be skipped (already accepted, failed, or computed).

    ``engineered_lineage``: optional ``{engineered_idx -> frozenset(parent_indices)}`` from the cat-FE step. When set, a k-way candidate is skipped if it
    combines an engineered column with one of its own parents (conditional MI degenerates and confidence gates waste budget). Legacy/numeric path leaves it ``None``.

    ``dcd_state`` (Wave 9): optional ``DCDState`` reference. When provided, a
    candidate (single index OR k-way tuple) is skipped if its ``pool_pruned_mask``
    bit is set per ``should_be_pruned`` semantics (Critic1/B-3: tuple of indices
    skipped iff ALL components pruned). Bit-stable when ``None``.
    """

    nexisting = 0

    if (cand_idx in failed_candidates) or (cand_idx in added_candidates) or expected_gains[cand_idx]:
        return True, nexisting

    # 2026-05-30 Wave 9 — DCD prune-mask short-circuit (Critic1/B-1 fix:
    # uses the mask instead of mutating the candidates list).
    if dcd_state is not None:
        try:
            from ._dynamic_cluster_discovery import should_be_pruned as _should_be_pruned
            target = X if interactions_order > 1 else int(cand_idx)
            if _should_be_pruned(dcd_state, target):
                return True, nexisting
        except Exception:
            # DCD is best-effort; never break candidate evaluation.
            pass

    if interactions_order > 1:  # disabled for single predictors 'cause Fleuret formula won't detect pairs predictors

        # Lineage filter: skip k-way candidates that combine an engineered column with one of its own parent columns.
        if engineered_lineage:
            X_set = set(X)
            for subel in X:
                parents = engineered_lineage.get(subel)
                if parents is not None and not parents.isdisjoint(X_set):
                    return True, nexisting

        # Check if any sub-element is already selected at this stage.
        skip_cand = False
        for subel in X:
            if subel in selected_interactions_vars:
                skip_cand = True
                break
        if skip_cand:
            return True, nexisting

        # Or all selected at the lower stages.
        skip_cand = [(subel in selected_vars) for subel in X]
        nexisting = sum(skip_cand)
        if (only_unknown_interactions and any(skip_cand)) or all(skip_cand):
            return True, nexisting

    return False, nexisting


def handle_best_candidate(
    current_gain: float,
    best_gain: float,
    X: Sequence,
    best_candidate: Sequence,
    factors_names: list,
    verbose: int = 1,
    ndigits: int = 5,
    max_runtime_mins: float = None,
    start_time: float = None,
    min_relevance_gain: float = None,
):
    # Save best known candidate, to enable early stopping.
    run_out_of_time = False

    if current_gain > best_gain:
        best_candidate = X
        best_gain = current_gain
        if verbose > 2:
            logger.info(
                "\t%s is so far the best candidate with best_gain=%.*f",
                get_candidate_name(best_candidate, factors_names=factors_names), ndigits, best_gain,
            )
    else:
        if min_relevance_gain and verbose > 2 and current_gain > min_relevance_gain:
            logger.info("\t\t%s current_gain=%.*f", get_candidate_name(X, factors_names=factors_names), ndigits, current_gain)

    if max_runtime_mins and not run_out_of_time:
        run_out_of_time = (timer() - start_time) > max_runtime_mins * 60

    return best_gain, best_candidate, run_out_of_time


@njit(cache=True)
def evaluate_gain(
    current_gain: float,
    last_checked_k: int,
    direct_gain: float,
    X: np.ndarray,
    y: np.ndarray,
    nexisting: int,
    best_gain: float,
    factors_data: np.ndarray,
    factors_nbins: np.ndarray,
    selected_vars: np.ndarray,
    mrmr_relevance_algo: str = "fleuret",
    mrmr_redundancy_algo: str = "fleuret",
    max_veteranes_interactions_order: int = 2,
    extra_knowledge_multipler: float = -1.0,
    sink_threshold: float = -1.0,
    entropy_cache: dict = None,
    cached_cond_MIs: dict = None,
    # 2026-06-19: JMIM joint-MI cache. Mirrors ``cached_cond_MIs`` but keyed on the
    # multiset ``{X} u Z`` only (``arr2str(xz_combined)``); y is fixed per fit so it
    # is not part of the key. Stores the RAW ``mi({X,Z}; y)``; the nexisting exponent
    # is applied at READ time, exactly as the plain-CMI branch does for its cache.
    cached_jmim_MIs: dict = None,
    # 2026-06-19: 1-element int64 array hit counter for the JMIM cache. An array
    # (not a scalar) so the @njit kernel can mutate it in place and the caller can
    # read the post-loop count -- proves the cache actually HITS, not just that it
    # is harmless. Counter[0] is incremented on every JMIM cache read-hit.
    jmim_hit_counter: np.ndarray = None,
    can_use_x_cache=False,
    can_use_y_cache=False,
    dtype=np.int32,
    confidence_mode: bool = False,
    max_confirmation_cand_nbins: int = MAX_CONFIRMATION_CAND_NBINS,
    use_su: bool = False,  # 2026-05-28: SU normalization toggle threaded from process_candidates / mrmr_fit_impl.
    use_jmim: bool = False,  # 2026-05-30 Wave 8: JMIM aggregator toggle, threaded from the Python-level caller for the same reason as ``use_su`` -- ``@njit`` cannot do ``from X import Y`` or call a non-njit thread-local reader at runtime (IMPORT_NAME opcode is unsupported).
    use_mm: bool = False,  # N-F2: Miller-Madow toggle for the redundancy CMI, threaded like use_su so the redundancy carries the SAME bias correction as the MM relevance. Default False -> plug-in redundancy, unchanged.
) -> tuple:
    """``max_confirmation_cand_nbins`` is a parameter (not a baked-in constant). Default mirrors the legacy value; ``MRMR.fit`` overrides with ``quantization_nbins ** interactions_max_order * 2`` unless the user pins it explicitly."""

    positive_mode = False
    stopped_early = False
    sink_reasons = None

    k = 0
    for interactions_order in range(max_veteranes_interactions_order):
        combs = generate_combinations_recursive_njit(np.array(selected_vars, dtype=np.int32), interactions_order + 1)[::-1]

        for Z in combs:

            if k > last_checked_k:
                if confidence_mode and count_cand_nbins(Z, factors_nbins) > max_confirmation_cand_nbins:
                    additional_knowledge = 0.0  # this is needed to skip checking agains hi cardinality approved factors
                else:
                    if mrmr_relevance_algo == "fleuret":
                        # additional_knowledge = I(X; Y | Z) = H(X, Z) + H(Y, Z) - H(Z) - H(X, Y, Z); I(X, Z) = entropy_x + entropy_z - entropy_xz.
                        key_found = False
                        if not confidence_mode:
                            # 2026-05-30 Wave 9.1 fix (loop iter 14):
                            # ``arr2str`` already uses ``_`` as its element
                            # separator, so the prior boundary ``"_"`` between
                            # ``arr2str(X)`` and ``arr2str(Z)`` aliased every
                            # partition of the multiset ``X u Z``. Confirmed:
                            # X=[1,2] Z=[3,4], X=[1] Z=[2,3,4], X=[1,2,3] Z=[4]
                            # all produced ``"1_2_3_4"`` -> wrong-answer cache
                            # hits, silent. Affects every config with
                            # ``max_veteranes_interactions_order >= 2`` or
                            # engineered multi-element X/Z, biasing
                            # conditional-MI scoring. Switching to ``"|"``
                            # (a character ``arr2str`` cannot emit) makes the
                            # boundary unambiguous.
                            # 2026-05-30 Wave 9.1 fix (loop iter 37):
                            # cache the RAW (pre-exponent) conditional MI
                            # and apply ``** (nexisting + 1)`` at every
                            # read site. Pre-fix the cache stored the
                            # already-exponentiated value but the key
                            # omitted ``nexisting``, so a subsequent
                            # ``(X, Z)`` lookup with a different
                            # ``nexisting`` returned the WRONG exponent
                            # (and silently biased the Fleuret/CMIM
                            # redundancy score). Caching the raw value
                            # lets all nexisting tiers share the
                            # underlying CMI compute.
                            key = arr2str(X) + "|" + arr2str(Z)
                            if key in cached_cond_MIs:
                                additional_knowledge = cached_cond_MIs[key]
                                # Apply the nexisting exponent at read time.
                                if nexisting > 0:
                                    additional_knowledge = additional_knowledge ** (nexisting + 1)
                                key_found = True

                        if not key_found:

                            # 2026-05-30 Wave 8 — JMIM aggregator (Bennasar 2015).
                            # When the thread-local JMIM toggle is on, replace
                            # Fleuret's conditional MI ``I(X; Y | Z)`` with the
                            # joint MI ``I({X, Z}; Y)``. Both feed the same
                            # outer ``min_k`` aggregator. JMIM preserves the
                            # synergistic information that CMIM rejects on
                            # multi-collinear groups (Brown 2012 / Bennasar 2015
                            # framework). Caching skipped on the JMIM branch
                            # because the joint-MI computation has its own
                            # per-pair entropy cost; future sprint may add a
                            # joint-MI cache keyed on the multiset {X, Z, Y}.
                            # 2026-05-28: SU branch bypasses entropy caches because the SU denominator
                            # is path-dependent and re-using cached unconditional CMI numbers would
                            # silently desync the normalization. Cheap (no caching) is correct here;
                            # legacy raw-CMI path keeps cache wired in. ``use_su`` AND ``use_jmim``
                            # are threaded from the Python-level caller (evaluate_candidate ->
                            # evaluate_gain) -- the njit kernel cannot read the Python thread-locals
                            # at runtime (IMPORT_NAME / dynamic Python call are unsupported in @njit).
                            if use_jmim:
                                # Build the joint variable index multiset {X, Z} for JMIM's
                                # I({X, Z}; Y). np.concatenate is supported in @njit but the
                                # numba type-inferencer can fail to unify dtypes when X and Z
                                # come from different upstream call paths; coerce both sides
                                # to int64 explicitly so the unification is bit-identical to
                                # mi()'s internal x = np.asarray(x, dtype=np.int64) cast.
                                _x_int = np.asarray(X, dtype=np.int64)
                                _z_int = np.asarray(Z, dtype=np.int64)
                                xz_combined = np.unique(np.concatenate((_x_int, _z_int)))
                                # 2026-06-19: JMIM joint-MI cache. The multiset {X} u Z recurs
                                # across greedy rounds (and across (X, Z) swaps within a round),
                                # so memoise the raw mi({X,Z}; y) keyed on arr2str(xz_combined).
                                # Stored RAW; nexisting exponent applied below at read time, as
                                # the plain-CMI branch does. ``y`` is fixed per fit -> not in key.
                                # Independent of the cond-MI cache so JMIM and CMI values never
                                # collide under the same arr2str(X)+"|"+arr2str(Z) key.
                                # ``cached_jmim_MIs`` / ``jmim_hit_counter`` may be None when a
                                # direct caller invokes evaluate_gain without the cache wired
                                # (e.g. focused unit tests). Guard with ``is not None`` -- numba
                                # narrows the Optional so the typed-dict ``in``/index ops only
                                # type-check on the non-None branch; the uncached fallback just
                                # recomputes mi() every time (pre-cache behaviour).
                                _jmim_key = arr2str(xz_combined)
                                if (not confidence_mode) and (cached_jmim_MIs is not None) and (_jmim_key in cached_jmim_MIs):
                                    additional_knowledge = cached_jmim_MIs[_jmim_key]
                                    if jmim_hit_counter is not None:
                                        jmim_hit_counter[0] += 1
                                    key_found = True
                                else:
                                    additional_knowledge = mi(
                                        factors_data=factors_data,
                                        x=xz_combined, y=y,
                                        factors_nbins=factors_nbins, dtype=dtype,
                                    )
                                    if (not confidence_mode) and (cached_jmim_MIs is not None):
                                        cached_jmim_MIs[_jmim_key] = additional_knowledge
                                if nexisting > 0:
                                    _disc = additional_knowledge ** (nexisting + 1)
                                    if _JMIM_EXPONENT_DISCOUNT_ONLY and _disc > additional_knowledge:
                                        # joint MI > 1 -> the exponent would amplify; clamp to discount-only (never increase the joint MI).
                                        pass
                                    else:
                                        additional_knowledge = _disc
                            elif use_su:
                                additional_knowledge = conditional_symmetric_uncertainty(
                                    factors_data=factors_data, x=X, y=y, z=Z,
                                    factors_nbins=factors_nbins, dtype=dtype,
                                )
                            else:
                                additional_knowledge = conditional_mi(
                                    factors_data=factors_data,
                                    x=X,
                                    y=y,
                                    z=Z,
                                    var_is_nominal=None,
                                    factors_nbins=factors_nbins,
                                    entropy_cache=entropy_cache,
                                    can_use_x_cache=can_use_x_cache,
                                    can_use_y_cache=can_use_y_cache,
                                    dtype=dtype,
                                    use_mm=use_mm,  # N-F2: redundancy carries the SAME MM bias correction as the relevance
                                )

                            # 2026-05-30 Wave 9.1 fix (loop iter 37):
                            # write the RAW additional_knowledge to the
                            # cache BEFORE applying the nexisting
                            # exponent. Apply the exponent only to the
                            # local ``additional_knowledge`` after the
                            # write. This makes the cache nexisting-
                            # independent and lets all callers share
                            # the underlying CMI compute.
                            # 2026-06-19: JMIM manages its OWN cache + exponent above
                            # (cached_jmim_MIs, keyed on the {X} u Z multiset). Skip the
                            # shared cond-MI write/exponent here so JMIM values are neither
                            # stored under the CMI key nor double-exponentiated. The SU
                            # branch is uncached by design; the plain-CMI branch keeps the
                            # legacy write/exponent. ``key`` is undefined when use_jmim, so
                            # this guard also avoids a NameError under JMIM.
                            if not use_jmim:
                                if not confidence_mode:
                                    cached_cond_MIs[key] = additional_knowledge
                                if nexisting > 0:
                                    additional_knowledge = additional_knowledge ** (nexisting + 1)

                    # Account for possible extra knowledge from conditioning on Z; must update best_gain globally and log such cases. Order of discovery is
                    # not guaranteed, but cases are too precious to ignore. Also enables skipping higher-order interactions containing all approved candidates.
                    # CAVEAT (extra_knowledge_multipler > 0): this is NOT MRMR redundancy for that candidate. Once a single Z yields additional_knowledge above the
                    # multiplied direct_gain, ``positive_mode`` flips and the score becomes the MAX over Z of that extra-knowledge term, NOT the MIN-redundancy MRMR
                    # objective -- the candidate is ranked by its single BEST synergistic relationship instead of its worst-case redundancy. Default is -1.0 (off), so
                    # the default selection stays pure MRMR; enable this only when you deliberately want a synergy-seeking (max-extra-knowledge) ranking.
                    if extra_knowledge_multipler > 0 and additional_knowledge > direct_gain * extra_knowledge_multipler:
                        if not positive_mode:
                            current_gain = additional_knowledge
                            positive_mode = True
                        else:
                            # rare chance that a candidate has many excellent relationships
                            if additional_knowledge > current_gain:
                                current_gain = additional_knowledge

                if not positive_mode and (additional_knowledge < current_gain):

                    current_gain = additional_knowledge

                    if best_gain is not None and current_gain <= best_gain:
                        # No point checking other Zs -- current_gain already won't beat best_gain (assuming best_gain was estimated confidently; checked at end).
                        # Also fix what Z caused X (the most) to sink.
                        if sink_threshold > -1 and current_gain < sink_threshold:
                            sink_reasons = Z

                        stopped_early = True
                        return stopped_early, current_gain, k, sink_reasons
            k += 1

    return stopped_early, current_gain, k - 1, sink_reasons


def evaluate_candidate(
    cand_idx: int,
    X: Sequence[int],
    y: Sequence[int],
    nexisting: int,
    best_gain: float,
    factors_data: np.ndarray,
    factors_nbins: np.ndarray,
    factors_names: Sequence[str],
    expected_gains: np.ndarray,
    partial_gains: dict,
    selected_vars: list,
    baseline_npermutations: int,
    classes_y: np.ndarray = None,
    classes_y_safe: np.ndarray = None,
    freqs_y: np.ndarray = None,
    freqs_y_safe: np.ndarray = None,
    use_gpu: bool = True,
    cached_MIs: dict = None,
    cached_confident_MIs: dict = None,
    cached_cond_MIs: dict = None,
    cached_jmim_MIs: dict = None,
    jmim_hit_counter: np.ndarray = None,
    entropy_cache: dict = None,
    mrmr_relevance_algo: str = "fleuret",
    mrmr_redundancy_algo: str = "fleuret",
    max_veteranes_interactions_order: int = 2,
    extra_knowledge_multipler: float = -1.0,
    sink_threshold: float = -1.0,
    dtype=np.int32,
    verbose: int = 1,
    ndigits: int = 5,
    use_simple_mode: bool = True,
) -> None:
    sink_reasons = set()

    # Is this candidate any good for target 1-vs-1?
    if X in cached_confident_MIs:  # cached_confident_MIs first -- more reliable (but don't fill them here).
        # cached_confident_MIs stores (bootstrapped_gain, confidence) tuples (see confirm_candidate); take the gain.
        # INVARIANT: a candidate only lands in cached_confident_MIs AFTER permutation confirmation, at which point its
        # cand_idx is in added_/failed_candidates and should_skip_candidate filters it out before re-entry here -- so under
        # a normal fit this branch is unreachable. It is kept (and made explicit) only as a safety net: if the skip
        # invariant ever breaks, the bootstrapped gain MUST still pass the same SU floor-scaling the else-path applies,
        # otherwise a raw-MI-scale value would be compared against an SU-scale min_relevance_gain floor (it is already
        # permutation-confirmed, so it does NOT re-enter the null-debiasing/significance gate -- re-applying that would
        # double-penalise). Log once if reached so a broken invariant is diagnosable rather than silently mis-scoring.
        logger.warning(
            "evaluate_candidate: confirmed candidate %s re-entered the cached_confident_MIs branch (skip-invariant broken); "
            "scoring it through the SU floor-scaling guard.", X,
        )
        direct_gain, _ = cached_confident_MIs[X]
        direct_gain = _su_normalize_relevance(direct_gain, X, y, factors_data, factors_nbins, dtype)
    else:
        _gmi = get_group_mi()
        _grp_gain = float("nan")
        if _gmi is not None and X not in cached_MIs:
            # Group-aware relevance: per-group I(X;Y|G) (MM-debiased) instead of the global MI. Bypasses the GPU +
            # permutation-null path (the per-group MM debias handles the small-sample bias). Returns nan when the
            # segments do not row-align (subsample), in which case we fall through to the global path. Default OFF.
            _si, _off, _mr, _sw = _gmi
            _grp_gain = group_relevance_mi(
                factors_data, X, classes_y, factors_nbins, len(freqs_y),
                _si, _off, min_rows=_mr, size_weighted=_sw, dtype=dtype,
            )
        if _grp_gain == _grp_gain:  # not nan -> group-aware relevance succeeded
            direct_gain = _grp_gain
            cached_MIs[X] = direct_gain
        elif X in cached_MIs:
            direct_gain = cached_MIs[X]
        else:
            # 2026-05-30 Wave 9.1 fix (XOR-synergy regression):
            # use UNANIMOUS-rejection baseline (require ALL perms to
            # beat observed before rejecting). The prior
            # ``min_nonzero_confidence=1.0`` hardcode + the
            # ``max_failed = max(1, ...)`` floor at permutation.py:348
            # combined to require ZERO of ``baseline_npermutations``
            # (default 2) perms meet/exceed observed - one chance
            # perm killed genuine synergy candidates. For order-2+
            # tuples with high joint cardinality (5x5=25 cells), the
            # null distribution has heavy tails and 1/2 perms beating
            # observed is COMMON for legitimately-significant XOR-
            # family candidates.
            # The middle ground: ``max_failed=npermutations`` means
            # the screen rejects ONLY when ALL baseline perms beat
            # observed. This kills obvious-noise candidates (where
            # nearly every shuffle matches observed because there's
            # no signal) while letting genuinely-significant
            # candidates (where most shuffles fall short) through to
            # the strict confirmation test at ``full_npermutations``.
            # Empirically this gives ~30% baseline reject rate on
            # all-noise and >90% pass rate on signal/synergy.
            _bnp = max(2, int(baseline_npermutations))
            if use_gpu:
                direct_gain, _, null_mean, p_value = mi_direct_gpu(
                    factors_data,
                    x=X,
                    y=y,
                    factors_nbins=factors_nbins,
                    classes_y=classes_y,
                    classes_y_safe=classes_y_safe,
                    freqs_y=freqs_y,
                    freqs_y_safe=freqs_y_safe,
                    min_nonzero_confidence=0.0,
                    max_failed=_bnp,
                    npermutations=_bnp,
                    dtype=dtype,
                    return_null_mean=True,
                )
                # SAME significance-gated relevance debiasing as the CPU branch below (audit5-P1). Without it the
                # GPU path returned RAW plug-in MI, inflating high-cardinality / heavy-tailed / monotone-datetime
                # / engineered columns above genuine lower-cardinality signal, AND making selection
                # hardware-dependent (GPU-present hosts selected differently from CPU-only ones).
                if p_value >= _MRMR_NULL_SIGNIF_ALPHA:
                    direct_gain = max(0.0, direct_gain - null_mean)
            else:
                # Significance-gated empirical-null debiasing of the relevance MI. On a wide composite-FE candidate pool the in-sample plug-in MI is upward-biased for
                # high-cardinality (50-level categoricals), heavy-tailed (Student-t), monotone-datetime and engineered columns, so they out-rank genuine lower-cardinality signal
                # (e.g. a strong Gaussian leg) and get selected while the real signal is dropped. ``return_null_mean=True`` runs the relevance null and returns BOTH the per-feature
                # null mean (the average MI of X against y-PERMUTATIONS the kernel already computes) AND the permutation p-value (the fraction of shuffles that tied/beat observed).
                direct_gain, _, null_mean, p_value = mi_direct(
                    factors_data,
                    x=X,
                    y=y,
                    factors_nbins=factors_nbins,
                    classes_y=classes_y,
                    classes_y_safe=classes_y_safe,
                    freqs_y=freqs_y,
                    min_nonzero_confidence=0.0,
                    max_failed=_bnp,
                    npermutations=_bnp,
                    dtype=dtype,
                    return_null_mean=True,
                )
                # Gate the subtraction on permutation SIGNIFICANCE. The null mean alone cannot distinguish a WEAK GENUINE signal (whose coarse-binning null is a large fraction of
                # its observed MI) from SPURIOUS NOISE (whose null is high because it IS noise) -- subtracting the full null would over-correct the weak signal below the
                # relative-gain floor and drop it. The permutation p-value IS that discriminator: a significant feature (``p_value < alpha``) sits ABOVE its null and keeps its
                # full observed MI; a non-significant feature (``p_value >= alpha``) sits WITHIN its null and gets the full null mean subtracted, collapsing toward 0. alpha is a
                # textbook level (0.05), not a fixture-tuned fraction, and the selection is stable across a wide alpha band because real signal clears p ~ 0 and noise sits at p ~ 1.
                if p_value >= _MRMR_NULL_SIGNIF_ALPHA:
                    direct_gain = max(0.0, direct_gain - null_mean)
            # SU-NORMALIZE THE MARGINAL RELEVANCE under mi_normalization='su'. The conditional/redundancy term already uses
            # conditional_symmetric_uncertainty, but the marginal relevance (the value the min_relevance_gain floor compares against,
            # and the first-pick score) was left as RAW MI -- so a high-cardinality noise column whose raw MI clears the entropy-relative
            # floor (e.g. 80-level hi_* with MI ~0.11 > 0.16*H(y)=0.111) was admitted even though its SU ~0.044 sits far below. Scale the
            # debiased relevance by the SU denominator 2/(H(X)+H(Y)) so the floor sees the cardinality-scrubbed score, matching the unit
            # SU definition. Done only when direct_gain > 0 (a zero stays zero) and the SU toggle is on; legacy path is byte-identical.
            direct_gain = _su_normalize_relevance(direct_gain, X, y, factors_data, factors_nbins, dtype)
            cached_MIs[X] = direct_gain

    # Synergy candidates can have direct_gain == 0 (pure XOR, parity, etc.: the
    # marginal MI is zero by construction but the conditional MI given an
    # already-selected variable surfaces the synergy). When the fleuret-style
    # complex mode is active (``selected_vars`` non-empty AND
    # ``use_simple_mode=False``) we MUST still evaluate the conditional gain --
    # gating on ``direct_gain > 0`` skips exactly the synergy class the
    # algorithm is supposed to find. The simple-mode path (no conditional MI)
    # still short-circuits on zero direct gain since there's nothing else to
    # compute.
    _force_cond = (
        selected_vars and not use_simple_mode
        and str(mrmr_relevance_algo) == "fleuret"
    )
    if direct_gain > 0 or _force_cond:
        if selected_vars and not use_simple_mode:
            # Some factors already selected. Best gain from including X is min I(X; Y | Z) over Z in selected_vars. But a variable correlated to every real
            # predictor plus noise has real value zero -- I(X; Y | Z) alone still gives significant impact. Summing I(X, Z) over Zs reveals it shares all
            # knowledge with the rest and has no value alone, but that requires all real factors already in S; otherwise 'connected-to-all' trash dominates.
            # Solution: compute sum(X, Z) not only when adding Z, but repeat for all Zs once a new X is added -- some Zs may become useless after.
            # P2 FIX (2026-06-13): the (current_gain, last_checked_k) cross-step resume is only VALID
            # when newly-selected vars APPEND to the end of the combination sequence -- true at the
            # default max_veteranes_interactions_order == 1 (a new singleton appends at the tail), but
            # NOT at order >= 2, where growing selected_vars inserts the new singleton in the MIDDLE of
            # the global k-sequence. Resuming from a stale last_checked_k then SKIPS measuring the new
            # var's redundancy against the candidate, so a feature fully redundant with the
            # most-recently-selected var could survive. For order >= 2 evaluate from scratch (re-min over
            # ALL combinations incl. the new ones); order == 1 keeps the byte-identical resume optimisation.
            # S-F4: the early-exit + resume below assume the stored partial gain is a running MIN (a monotone lower
            # bound). That holds for the standard MRMR min-redundancy objective, but NOT when
            # ``extra_knowledge_multipler > 0``: evaluate_gain then flips to positive_mode = MAX over Z of the
            # extra-knowledge term, so a later Z (or a newly-selected var) can yield a HIGHER score. The
            # ``_stored_gain <= best_gain -> return`` would wrongly prune a candidate whose best synergy is with a
            # newly-added var, and positive_mode is reset each entry (not persisted across resume calls). Re-evaluate
            # from scratch on the extra-knowledge path (perf cost only; the knob is off by default, -1.0). No-op on
            # the default path, which keeps the byte-identical resume optimisation.
            if cand_idx in partial_gains and not (extra_knowledge_multipler > 0):
                _stored_gain, _stored_k = partial_gains[cand_idx]
                # EARLY-EXIT is safe at ANY interactions_order: the stored partial gain is a running MIN
                # over the combinations checked so far, so the FINAL gain (min over MORE combinations) can
                # only be <= it. If the partial is already <= best_gain this candidate cannot win -> return.
                if best_gain is not None and (_stored_gain <= best_gain):
                    return _stored_gain, sink_reasons
                # RESUME (continue from last_checked_k) is only valid at order < 2, where a newly-selected
                # var APPENDS at the tail of the combination sequence. At order >= 2 the new singleton
                # inserts in the MIDDLE, so a stale last_checked_k would skip measuring its redundancy
                # against the candidate -> re-evaluate from scratch (re-min over ALL combinations).
                if max_veteranes_interactions_order < 2:
                    current_gain, last_checked_k = _stored_gain, _stored_k
                else:
                    current_gain = LARGE_CONST
                    last_checked_k = -1
            else:
                current_gain = LARGE_CONST
                last_checked_k = -1

            # 2026-06-19: the @njit evaluate_gain types the JMIM-cache ``in`` check
            # unconditionally (even when use_jmim is off), so it needs a REAL numba
            # typed dict + a real counter array, never None. Callers that do not
            # thread a shared cache (e.g. the serial/parallel _confirm_predictor
            # path) leave these None; self-provision a fresh per-call typed dict +
            # zeroed counter here so the kernel always type-checks. A fresh local
            # dict gives within-call memoisation only (the cross-round reuse needs
            # the driver to thread a shared dict, mirroring cached_cond_MIs) and is
            # purely local => never leaks onto an instance / breaks pickle.
            if cached_jmim_MIs is None:
                cached_jmim_MIs = numba.typed.Dict.empty(
                    key_type=types.unicode_type, value_type=types.float64,
                )
            if jmim_hit_counter is None:
                jmim_hit_counter = np.zeros(1, dtype=np.int64)

            stopped_early, current_gain, k, sink_reasons = evaluate_gain(
                current_gain=current_gain,
                last_checked_k=last_checked_k,
                direct_gain=direct_gain,
                X=X,
                y=y,
                nexisting=nexisting,
                best_gain=best_gain,
                factors_data=factors_data,
                factors_nbins=factors_nbins,
                selected_vars=selected_vars,
                mrmr_relevance_algo=mrmr_relevance_algo,
                mrmr_redundancy_algo=mrmr_redundancy_algo,
                max_veteranes_interactions_order=max_veteranes_interactions_order,
                extra_knowledge_multipler=extra_knowledge_multipler,
                sink_threshold=sink_threshold,
                entropy_cache=entropy_cache,
                cached_cond_MIs=cached_cond_MIs,
                cached_jmim_MIs=cached_jmim_MIs,
                jmim_hit_counter=jmim_hit_counter,
                can_use_x_cache=True,
                can_use_y_cache=True,
                use_su=use_su_normalization(),
                use_jmim=use_jmim_aggregator(),
                use_mm=(use_mi_miller_madow() and not use_su_normalization()),  # N-F2: MM redundancy when MM relevance is active
            )

            partial_gains[cand_idx] = current_gain, k
            if not stopped_early:  # no break -- current_gain computed fully.
                expected_gains[cand_idx] = current_gain
        else:
            # No factors selected yet -- current_gain is just direct_gain.
            current_gain = direct_gain
            expected_gains[cand_idx] = current_gain
    else:
        # Float (not int) so the no-gain return stays type-consistent with the float
        # ``direct_gain`` path. The empirical-null debiasing can drive ``direct_gain`` to
        # exactly 0.0, which now routes a non-synergy candidate through this branch.
        current_gain = 0.0

    # 2026-05-30 Wave 8 — BUR additive bonus (Gao 2022). Off by default
    # (``bur_lambda`` thread-local = 0.0). When enabled, adds
    # ``lambda * (I(X; Y) - max_j I(X; X_j))`` to the post-Fleuret score.
    # ``direct_gain`` already holds ``I(X; Y)``; max-correlation to selected
    # is computed once per candidate. Cheap because no new MI estimator —
    # reuses existing ``mi`` njit on multi-index slices. Floored at zero so
    # a fully-redundant candidate gets zero bonus, not a penalty.
    _bur_lambda = get_bur_lambda()
    if _bur_lambda > 0.0 and direct_gain > 0 and selected_vars:
        try:
            max_xz = 0.0
            for _z in selected_vars:
                _z_arr = np.asarray(_z if hasattr(_z, "__len__") else [_z],
                                     dtype=np.int64)
                xz = mi(
                    factors_data=factors_data, x=X, y=_z_arr,
                    factors_nbins=factors_nbins, dtype=dtype,
                )
                if xz > max_xz:
                    max_xz = float(xz)
            bonus = max(0.0, float(direct_gain) - max_xz) * _bur_lambda
            current_gain = float(current_gain) + bonus
            if cand_idx in partial_gains:
                _g, _k = partial_gains[cand_idx]
                partial_gains[cand_idx] = (float(_g) + bonus, _k)
            # 2026-05-30 Wave 9.1 fix (loop iter 26): publish the BUR-
            # bonus-inflated gain into ``expected_gains[cand_idx]``
            # unconditionally. Pre-fix the publish was gated on
            # ``if expected_gains[cand_idx]:`` which:
            #   - raised KeyError on the dict-path stopped_early branch
            #     (line 575-577 only sets expected_gains when fully
            #     evaluated -- on stopped_early it stays absent)
            #   - returned 0 (falsy) on the ndarray-path stopped_early
            #     branch, skipping the publish
            # In both cases ``partial_gains[cand_idx]`` and ``current_gain``
            # carried the BUR bonus but the dense ranking vector
            # ``expected_gains`` did not -- the confirmation loop's
            # lexsort then ranked the BUR-bonus winner BELOW peers
            # whose ``expected_gains`` entries reflected pre-bonus
            # scores. The bare ``except Exception: pass`` silently
            # masked the KeyError on dict-path. Drop both the gate
            # and the bare-except so BUR actually affects ranking and
            # programming errors surface.
            expected_gains[cand_idx] = current_gain
        except (KeyError, IndexError) as _bur_publish_exc:
            # ``expected_gains`` is preallocated by ``screen_predictors``
            # before any candidate evaluation, so a missing slot here
            # would indicate a true invariant break - re-raise.
            raise RuntimeError(
                f"BUR publish failed for cand_idx={cand_idx}: {_bur_publish_exc!r}"
            ) from _bur_publish_exc
        except Exception as _bur_exc:
            # Real numerical / mi-kernel failures (degenerate joints,
            # all-NaN slices) stay best-effort: warn and continue rather
            # than aborting the whole screen.
            logger.warning("BUR bonus computation failed silently: %r", _bur_exc)

    # RelaxMRMR 3-D-redundancy score (Vinh 2016). Default alpha=0.0 -> dispatch skipped, legacy Fleuret score untouched (byte-identical). When alpha>0 the candidate's
    # complex-mode score is REPLACED by ``relax_mrmr_score`` (relevance - mean pairwise redundancy + alpha-weighted 3-way interaction term), which down-weights redundancy
    # detected only at the 3-feature level. Only meaningful in fleuret complex mode with a non-empty selected set (the simple-mode / first-pick path keeps direct_gain).
    _relax_alpha = get_relaxmrmr_alpha()
    if _relax_alpha > 0.0 and selected_vars and not use_simple_mode and str(mrmr_relevance_algo) == "fleuret":
        try:
            from ._relaxmrmr_3d import relax_mrmr_score
            x_col, k_x = _materialize_var(factors_data, X, factors_nbins, dtype=dtype)
            y_col, k_y = _materialize_var(factors_data, y, factors_nbins, dtype=dtype)
            sel_cols, sel_nbins = [], []
            for _z in selected_vars:
                _zc, _zk = _materialize_var(factors_data, _z, factors_nbins, dtype=dtype)
                sel_cols.append(_zc)
                sel_nbins.append(_zk)
            current_gain = float(relax_mrmr_score(x_col, sel_cols, y_col, k_x, sel_nbins, k_y, alpha=_relax_alpha))
            if cand_idx in partial_gains:
                _g, _k = partial_gains[cand_idx]
                partial_gains[cand_idx] = (current_gain, _k)
            expected_gains[cand_idx] = current_gain
        except Exception as _relax_exc:
            logger.warning("RelaxMRMR score computation failed silently: %r", _relax_exc)

    # PID synergy bonus (Williams-Beer / Ince I_ccs). Default bonus=0.0 -> skipped (byte-identical). When bonus>0, add ``bonus * max_j synergy(X, Z_j; Y)`` so a candidate
    # that is synergistic with an already-selected feature (XOR-like joint information neither carries alone) is rewarded -- the standard redundancy gate would otherwise drop
    # it. Mirrors the BUR additive-bonus shape: computed once per candidate, floored at zero, republished into the dense ranking vector.
    _pid_bonus = get_pid_synergy_bonus()
    if _pid_bonus > 0.0 and selected_vars:
        try:
            from ._pid_decomposition import pid_decomposition
            x_col, k_x = _materialize_var(factors_data, X, factors_nbins, dtype=dtype)
            y_col, k_y = _materialize_var(factors_data, y, factors_nbins, dtype=dtype)
            max_syn = 0.0
            for _z in selected_vars:
                _zc, _zk = _materialize_var(factors_data, _z, factors_nbins, dtype=dtype)
                syn = float(pid_decomposition(x_col, _zc, y_col, k_x, _zk, k_y)["synergistic"])
                if syn > max_syn:
                    max_syn = syn
            bonus = max(0.0, max_syn) * _pid_bonus
            current_gain = float(current_gain) + bonus
            if cand_idx in partial_gains:
                _g, _k = partial_gains[cand_idx]
                partial_gains[cand_idx] = (float(_g) + bonus, _k)
            expected_gains[cand_idx] = current_gain
        except Exception as _pid_exc:
            logger.warning("PID synergy bonus computation failed silently: %r", _pid_exc)

    # CMI permutation early-stop (Yu & Principe 2019). Default off -> skipped (byte-identical). When active, permute the candidate (preserving its marginal) and re-estimate
    # ``I(X; Y | selected)``; if the observed conditional MI is NOT significant at alpha (p >= alpha), the candidate carries no conditional signal given the selected set and is
    # dropped (gain forced to 0.0). Drops a conditionally-redundant candidate the relative-gain floor alone would admit.
    _cmi_active, _cmi_alpha, _cmi_nperm = get_cmi_perm_stop()
    if _cmi_active and selected_vars and current_gain > 0.0:
        try:
            from ._cmi_perm_stop import cmi_permutation_stop
            x_col, k_x = _materialize_var(factors_data, X, factors_nbins, dtype=dtype)
            y_col, k_y = _materialize_var(factors_data, y, factors_nbins, dtype=dtype)
            sel_cols, sel_nbins = [], []
            for _z in selected_vars:
                _zc, _zk = _materialize_var(factors_data, _z, factors_nbins, dtype=dtype)
                sel_cols.append(_zc)
                sel_nbins.append(_zk)
            is_signif, _obs, _pval = cmi_permutation_stop(
                x_col, y_col, sel_cols, k_x, k_y, sel_nbins,
                n_permutations=_cmi_nperm, alpha=_cmi_alpha, seed=int(cand_idx),
            )
            if not is_signif:
                current_gain = 0.0
                if cand_idx in partial_gains:
                    _g, _k = partial_gains[cand_idx]
                    partial_gains[cand_idx] = (0.0, _k)
                expected_gains[cand_idx] = 0.0
        except Exception as _cmi_exc:
            logger.warning("CMI permutation early-stop failed silently: %r", _cmi_exc)

    return current_gain, sink_reasons


def find_best_partial_gain(
    partial_gains: dict, failed_candidates: set, added_candidates: set, candidates: list, selected_vars: list, skip_indices: tuple = (),
    dcd_state=None,
) -> float:
    # 2026-06-02 Wave 9 fix: a DCD-pruned candidate must NOT be returned as a
    # redirect target. ``partial_gains`` persists across the confirmation
    # ``while`` retries within one interactions-order; when DCD prunes a
    # candidate AFTER it was scored (``discover_cluster_members`` sets
    # ``pool_pruned_mask`` once a same-cluster member is selected), the
    # candidate is skipped from RE-scoring (``should_skip_candidate``) but its
    # now-STALE high partial gain stays in the dict. Pre-fix
    # ``find_best_partial_gain`` had no view of the prune mask, so it kept
    # returning that pruned candidate's stale gain as "the best other option",
    # the confirmation loop redirected to it forever (it can never be confirmed
    # -- it is skipped), and the genuinely-good candidate that DID confirm was
    # never committed -> the screen stopped early and dropped real signal
    # (sensor-mesh: 6 features -> 2, -4% downstream AUC). Skipping pruned
    # candidates here closes the redirect loop. ``None`` dcd_state is the
    # legacy/bit-stable path (no DCD).
    _should_be_pruned = None
    if dcd_state is not None:
        try:
            from ._dynamic_cluster_discovery import should_be_pruned as _should_be_pruned
        except Exception:
            _should_be_pruned = None
    best_partial_gain = -LARGE_CONST
    best_key = None
    # Hoist selected_vars to a set: the inner ``subel in selected_vars`` membership is O(len) on the list, and it runs
    # per sub-element per candidate per confirmation-retry -> an O(1) set lookup is ~1.6x on a wide candidate pool
    # (bit-identical -- same membership test). selected_vars is small so building the set once is negligible.
    _selected_set = set(selected_vars)
    for key, value in partial_gains.items():
        if (key not in failed_candidates) and (key not in added_candidates) and (key not in skip_indices):
            skip_cand = False
            for subel in candidates[key]:
                if subel in _selected_set:
                    skip_cand = True  # the sub-element or var itself is already selected.
                    break
            if skip_cand:
                continue
            if _should_be_pruned is not None and _should_be_pruned(dcd_state, candidates[key]):
                continue  # DCD-pruned: out of contention, never a valid redirect target.
            partial_gain, _ = value
            if partial_gain > best_partial_gain:
                best_partial_gain = partial_gain
                best_key = key
    return best_partial_gain, best_key


# Tier E carve (2026-06-22): the candidate-evaluation driver cluster (the batched-GPU cond-MI
# cache pre-fill + the thread-worker entry points) was moved verbatim to ``_evaluation_driver.py``
# to bring this module under the 1k-LOC ceiling. Re-exported at the BOTTOM so the sibling's
# top-level ``from .evaluation import evaluate_candidate, handle_best_candidate, _JMIM_CACHE_STATS``
# resolves against the already-executed parent body (no circular-import hazard), and the public
# API of ``mlframe.feature_selection.filters.evaluation`` stays byte-for-byte unchanged.
from ._evaluation_driver import (  # noqa: E402,F401
    _gpu_cmi_prefill_enabled,
    _prefill_cond_MIs_gpu,
    evaluate_candidates,
    _evaluate_candidates_inner,
)

