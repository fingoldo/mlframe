"""Per-candidate evaluation: gain computation, confirmation checks.

See ``screen.py`` for the screening orchestrator that calls these functions.
"""
from __future__ import annotations

import gc
import logging
import math
import time
from timeit import default_timer as timer
from typing import Sequence

import numba
import numpy as np
from numba import njit
from numba.core import types

from pyutilz.numbalib import generate_combinations_recursive_njit, python_dict_2_numba_dict
# Module-level import so cloudpickle can resolve tqdmu when the function ships to a joblib worker.
from pyutilz.system import tqdmu

from ._internals import LARGE_CONST, MAX_CONFIRMATION_CAND_NBINS, MAX_ITERATIONS_TO_TRACK
from ._numba_utils import arr2str, count_cand_nbins, unpack_and_sort
from .gpu import mi_direct_gpu
from .info_theory import (
    compute_mi_from_classes, conditional_mi, entropy, merge_vars, mi,
    # 2026-05-28: SU normalization dispatcher. ``cmi_or_csu`` reads a thread-local
    # toggle set by ``MRMR.fit`` when ``mi_normalization='su'``; legacy path
    # (toggle off) is one extra Python call ahead of njit kernels.
    cmi_or_csu, use_su_normalization, conditional_symmetric_uncertainty,
    # 2026-05-30 Wave 8: JMIM aggregator + BUR weight thread-local toggles
    # used in evaluate_gain / evaluate_candidate.
    use_jmim_aggregator, get_bur_lambda,
    # 2026-05-30 Wave 9.1 iter 5: setters for re-publishing the toggles into
    # joblib worker threads.
    set_su_normalization, set_jmim_aggregator, set_bur_lambda,
)
from .permutation import mi_direct

logger = logging.getLogger(__name__)


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


def evaluate_candidates(
    workload: list,
    y: Sequence[int],
    best_gain: float,
    factors_data: np.ndarray,
    factors_nbins: np.ndarray,
    factors_names: Sequence[str],
    partial_gains: dict,
    selected_vars: list,
    baseline_npermutations: int,
    classes_y: np.ndarray = None,
    freqs_y: np.ndarray = None,
    freqs_y_safe: np.ndarray = None,
    use_gpu: bool = True,
    cached_MIs: dict = None,
    cached_confident_MIs: dict = None,
    cached_cond_MIs: dict = None,
    entropy_cache: dict = None,
    mrmr_relevance_algo: str = "fleuret",
    mrmr_redundancy_algo: str = "fleuret",
    max_veteranes_interactions_order: int = 1,
    dtype=np.int32,
    max_runtime_mins: float = None,
    start_time: float = None,
    min_relevance_gain: float = None,
    verbose: int = 1,
    ndigits: int = 5,
    use_simple_mode: bool = True,
    # 2026-05-30 Wave 9.1 iter 5: Wave 8 thread-locals MUST be re-published
    # into the worker thread. ``threading.local`` does NOT propagate across
    # joblib workers (even with backend='threading', new threads in the
    # pool have their own local namespace), so reads from inside the worker
    # would silently see the default ``False`` / ``0.0`` even though the
    # main thread set them. Snapshot the main-thread values in
    # ``_confirm_predictor`` and forward as explicit kwargs; the worker
    # republishes them at entry and resets in finally.
    use_su: bool = False,
    use_jmim: bool = False,
    bur_lambda: float = 0.0,
) -> None:

    # Worker-thread re-publish of Wave 8 toggles (iter 5 fix). The
    # try/finally guarantees we don't pollute the worker thread's locals
    # if the same worker is re-used for a subsequent dispatch with
    # different settings.
    _prev_su = use_su_normalization()
    _prev_jmim = use_jmim_aggregator()
    _prev_bur = get_bur_lambda()
    set_su_normalization(bool(use_su))
    set_jmim_aggregator(bool(use_jmim))
    set_bur_lambda(float(bur_lambda))
    try:
        return _evaluate_candidates_inner(
            workload=workload, y=y, best_gain=best_gain,
            factors_data=factors_data, factors_nbins=factors_nbins,
            factors_names=factors_names, partial_gains=partial_gains,
            selected_vars=selected_vars,
            baseline_npermutations=baseline_npermutations,
            classes_y=classes_y, freqs_y=freqs_y,
            freqs_y_safe=freqs_y_safe, use_gpu=use_gpu,
            cached_MIs=cached_MIs, cached_confident_MIs=cached_confident_MIs,
            cached_cond_MIs=cached_cond_MIs, entropy_cache=entropy_cache,
            mrmr_relevance_algo=mrmr_relevance_algo,
            mrmr_redundancy_algo=mrmr_redundancy_algo,
            max_veteranes_interactions_order=max_veteranes_interactions_order,
            dtype=dtype, max_runtime_mins=max_runtime_mins,
            start_time=start_time, min_relevance_gain=min_relevance_gain,
            verbose=verbose, ndigits=ndigits,
            use_simple_mode=use_simple_mode,
        )
    finally:
        set_su_normalization(_prev_su)
        set_jmim_aggregator(_prev_jmim)
        set_bur_lambda(_prev_bur)


def _evaluate_candidates_inner(
    workload, y, best_gain, factors_data, factors_nbins, factors_names,
    partial_gains, selected_vars, baseline_npermutations,
    classes_y=None, freqs_y=None, freqs_y_safe=None, use_gpu=True,
    cached_MIs=None, cached_confident_MIs=None, cached_cond_MIs=None,
    entropy_cache=None, mrmr_relevance_algo="fleuret",
    mrmr_redundancy_algo="fleuret", max_veteranes_interactions_order=1,
    dtype=np.int32, max_runtime_mins=None, start_time=None,
    min_relevance_gain=None, verbose=1, ndigits=5, use_simple_mode=True,
):

    best_gain = -LARGE_CONST
    best_candidate = None
    expected_gains = {}

    entropy_cache_dict = numba.typed.Dict.empty(
        key_type=types.unicode_type,
        value_type=types.float64,
    )
    python_dict_2_numba_dict(python_dict=entropy_cache, numba_dict=entropy_cache_dict)

    cached_cond_MIs_dict = numba.typed.Dict.empty(
        key_type=types.unicode_type,
        value_type=types.float64,
    )
    python_dict_2_numba_dict(python_dict=cached_cond_MIs, numba_dict=cached_cond_MIs_dict)

    classes_y_safe = classes_y.copy()

    for cand_idx, X, nexisting in tqdmu(workload, leave=False, desc="Thread Candidates", disable=not verbose):

        current_gain, sink_reasons = evaluate_candidate(
            cand_idx=cand_idx,
            X=X,
            y=y,
            nexisting=nexisting,
            best_gain=best_gain,
            factors_data=factors_data,
            factors_nbins=factors_nbins,
            factors_names=factors_names,
            classes_y=classes_y,
            classes_y_safe=classes_y_safe,
            freqs_y=freqs_y,
            use_gpu=use_gpu,
            freqs_y_safe=freqs_y_safe,
            partial_gains=partial_gains,
            baseline_npermutations=baseline_npermutations,
            mrmr_relevance_algo=mrmr_relevance_algo,
            mrmr_redundancy_algo=mrmr_redundancy_algo,
            max_veteranes_interactions_order=max_veteranes_interactions_order,
            expected_gains=expected_gains,
            selected_vars=selected_vars,
            cached_MIs=cached_MIs,
            cached_confident_MIs=cached_confident_MIs,
            cached_cond_MIs=cached_cond_MIs_dict,
            entropy_cache=entropy_cache_dict,
            verbose=verbose,
            ndigits=ndigits,
            dtype=dtype,
            use_simple_mode=use_simple_mode,
        )

        best_gain, best_candidate, run_out_of_time = handle_best_candidate(
            current_gain=current_gain,
            best_gain=best_gain,
            X=X,
            best_candidate=best_candidate,
            factors_names=factors_names,
            verbose=verbose,
            ndigits=ndigits,
            max_runtime_mins=max_runtime_mins,
            start_time=start_time,
            min_relevance_gain=min_relevance_gain,
        )

        if run_out_of_time:
            break

    entropy_cache = dict(entropy_cache_dict)
    cached_cond_MIs = dict(cached_cond_MIs_dict)

    return best_gain, best_candidate, partial_gains, expected_gains, cached_MIs, cached_cond_MIs, entropy_cache


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
    can_use_x_cache=False,
    can_use_y_cache=False,
    dtype=np.int32,
    confidence_mode: bool = False,
    max_confirmation_cand_nbins: int = MAX_CONFIRMATION_CAND_NBINS,
    use_su: bool = False,  # 2026-05-28: SU normalization toggle threaded from process_candidates / mrmr_fit_impl.
    use_jmim: bool = False,  # 2026-05-30 Wave 8: JMIM aggregator toggle, threaded from the Python-level caller for the same reason as ``use_su`` -- ``@njit`` cannot do ``from X import Y`` or call a non-njit thread-local reader at runtime (IMPORT_NAME opcode is unsupported).
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
                                additional_knowledge = mi(
                                    factors_data=factors_data,
                                    x=xz_combined, y=y,
                                    factors_nbins=factors_nbins, dtype=dtype,
                                )
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
                                )

                            # 2026-05-30 Wave 9.1 fix (loop iter 37):
                            # write the RAW additional_knowledge to the
                            # cache BEFORE applying the nexisting
                            # exponent. Apply the exponent only to the
                            # local ``additional_knowledge`` after the
                            # write. This makes the cache nexisting-
                            # independent and lets all callers share
                            # the underlying CMI compute.
                            if not confidence_mode:
                                cached_cond_MIs[key] = additional_knowledge
                            if nexisting > 0:
                                additional_knowledge = additional_knowledge ** (nexisting + 1)

                    # Account for possible extra knowledge from conditioning on Z; must update best_gain globally and log such cases. Order of discovery is
                    # not guaranteed, but cases are too precious to ignore. Also enables skipping higher-order interactions containing all approved candidates.
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
        # The main loop normally skips this branch (a confirmed candidate lands in added_/failed_candidates), so it
        # was latent, but assigning the whole tuple to direct_gain crashes the next ``direct_gain > 0`` if ever reached.
        direct_gain, _ = cached_confident_MIs[X]
    else:
        if X in cached_MIs:
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
                direct_gain, _ = mi_direct_gpu(
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
                )
            else:
                direct_gain, _ = mi_direct(
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
                )
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
            if cand_idx in partial_gains:
                current_gain, last_checked_k = partial_gains[cand_idx]
                if best_gain is not None and (current_gain <= best_gain):
                    return current_gain, sink_reasons
            else:
                current_gain = LARGE_CONST
                last_checked_k = -1

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
                can_use_x_cache=True,
                can_use_y_cache=True,
                use_su=use_su_normalization(),
                use_jmim=use_jmim_aggregator(),
            )

            partial_gains[cand_idx] = current_gain, k
            if not stopped_early:  # no break -- current_gain computed fully.
                expected_gains[cand_idx] = current_gain
        else:
            # No factors selected yet -- current_gain is just direct_gain.
            current_gain = direct_gain
            expected_gains[cand_idx] = current_gain
    else:
        current_gain = 0

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
    for key, value in partial_gains.items():
        if (key not in failed_candidates) and (key not in added_candidates) and (key not in skip_indices):
            skip_cand = False
            for subel in candidates[key]:
                if subel in selected_vars:
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

