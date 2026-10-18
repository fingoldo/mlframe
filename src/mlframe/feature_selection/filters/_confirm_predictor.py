"""Single-predictor confirmation primitives, carved out of
``mlframe.feature_selection.filters._screen_predictors`` so that file stays well below the
1k-line monolith threshold and the (most-frequently-patched) confirmation math lives in one place:

* :func:`score_candidates`  - evaluate every feasible candidate's conditional-MI
                                  gain given the current ``selected_vars`` (serial +
                                  joblib-parallel paths), filling ``expected_gains`` /
                                  ``partial_gains`` and returning the running best.
* :func:`confirm_candidate`  - permutation-confirm a single candidate ``X`` (direct
                                  MI bootstrap + Fleuret conditional recheck), returning
                                  ``(bootstrapped_gain, confidence)``.
* :func:`confirm_one_predictor`-- the full single-predictor confirmation cycle (the
                                  ``while True`` loop with lexsort tiebreak, partial-gain
                                  recompute/retry, and patience accounting), composed from
                                  the two primitives above.

``screen_predictors`` calls :func:`confirm_one_predictor` once per added predictor.

All shared/static state (data arrays, algorithm parameters, the four MI caches, and
the per-interactions-order mutable collections) is bundled in :class:`ScreenContext`;
per-predictor scalars (``best_gain`` / ``best_candidate`` / ``expected_gains`` /
``confidence`` / accumulators) are threaded as call arguments and return values so the
moved bodies stay byte-for-byte identical to the original inline blocks.
"""
from __future__ import annotations

import logging
from timeit import default_timer as timer
from typing import Optional, cast, Any

import os

import numba
import numpy as np
from ._fe_usability_signal import _crit_np_dtype
from numba.core import types


from ._internals import MAX_ITERATIONS_TO_TRACK, NMAX_NONPARALLEL_ITERS
from .evaluation import (
    evaluate_candidate,
    evaluate_candidates,
    find_best_partial_gain,
    get_candidate_name,
    handle_best_candidate,
    should_skip_candidate,
    _prefill_cond_MIs_gpu,
)
from .gpu import mi_direct_gpu
from .permutation import mi_direct

logger = logging.getLogger(__name__)

# OPT-IN (default OFF): ``score_candidates``'s ``evaluate_candidates`` threading-pool dispatch used to gate on
# ``n_workers > 1`` alone. Isolated/warmed/best-of-3+ A/B at this call site's realistic scales found the pool
# NEVER wins over the serial fallback below: m=10 candidates -> 0.03x, m=320 (wellbore-scale) -> 0.72-0.73x,
# m=820/n_workers=8 -> 0.81x, confirming GIL-bound dispatch-boundary contention. The serial path therefore
# stays the default; ``MLFRAME_MRMR_EVAL_POOL=1`` re-enables the parallel branch so it remains reachable (not
# dead code) for a workload shape a future host-specific measurement might favor. The parallel merge-back is
# selection-identical to the serial path. ``n_workers`` is still accepted/threaded through regardless (other
# call sites, e.g. the Fleuret conditional-confirmation gate further down, branch on it).
_EVALUATE_CANDIDATES_POOL_ENABLED = os.environ.get("MLFRAME_MRMR_EVAL_POOL", "0").strip().lower() in ("1", "true", "on", "yes")


# ``_mrmr_fit_impl`` imports ``_extract_single_raw_parent`` from this module by name (raw-retention pass).
from ._confirm_predictor_engineered import (
    _confirmable_engineered_child,
    _extract_single_raw_parent,  # noqa: F401 - re-exported for _mrmr_fit_impl/_fit_impl_core.py
    _prefer_engineered_order,
)

# ScreenContext carved out into
# _confirm_predictor_context.py to clear the repo's enforced hard 1000-LOC CI gate (this file had crept
# back to 1007 lines). Re-exported here so every existing import keeps working unchanged.
from ._confirm_predictor_context import ScreenContext
from ._confirm_predictor_score import score_candidates_parallel, score_candidates_serial
from ._confirm_predictor_helpers import confirm_in_gain_order, fleuret_conditional_recheck, marginal_confirm_use_gpu


def score_candidates(ctx: ScreenContext, best_gain: float, best_candidate: Any, expected_gains: np.ndarray) -> tuple:
    """Evaluate every feasible candidate's gain given ``ctx.selected_vars``.

    Mutates ``expected_gains`` (dense, keyed by global candidate index), ``ctx.partial_gains``,
    and the MI caches in place. Returns ``(best_gain, best_candidate, run_out_of_time)``. This is
    the verbatim scoring block from the original ``screen_predictors`` confirmation loop.
    """

    # Alias shared/static state so the moved body stays identical to the original inline block.
    candidates = ctx.candidates
    interactions_order = ctx.interactions_order
    only_unknown_interactions = ctx.only_unknown_interactions
    failed_candidates = ctx.failed_candidates
    added_candidates = ctx.added_candidates
    selected_vars = ctx.selected_vars
    selected_interactions_vars = ctx.selected_interactions_vars
    engineered_lineage = ctx.engineered_lineage
    dcd_state = getattr(ctx, "dcd_state", None)  # dynamic cluster discovery state
    reduce_gain_on_subelement_chosen = ctx.reduce_gain_on_subelement_chosen
    n_workers = ctx.n_workers
    use_simple_mode = ctx.use_simple_mode
    cached_MIs = ctx.cached_MIs
    num_possible_candidates = ctx.num_possible_candidates
    # JMIM joint-MI cache + hit counter (see ScreenContext). Lazily
    # provisioned for direct ScreenContext callers that don't set them, so the
    # serial evaluate_candidate path always gets a real numba dict (the @njit
    # kernel types the cache ``in`` check unconditionally).
    cached_jmim_MIs = getattr(ctx, "cached_jmim_MIs", None)
    jmim_hit_counter = getattr(ctx, "jmim_hit_counter", None)
    if cached_jmim_MIs is None:
        cached_jmim_MIs = numba.typed.Dict.empty(
            key_type=types.unicode_type,
            # Intentional float64: numba typed-dict value contract shared with the njit JMIM cache (must match warmup_typed_dict)
            value_type=types.float64,
        )
        ctx.cached_jmim_MIs = cached_jmim_MIs
    if jmim_hit_counter is None:
        jmim_hit_counter = np.zeros(1, dtype=np.int64)
        ctx.jmim_hit_counter = jmim_hit_counter
    verbose = ctx.verbose

    run_out_of_time = False

    if verbose and len(selected_vars) < MAX_ITERATIONS_TO_TRACK:
        eval_start = timer()

    feasible_candidates = []
    for cand_idx, X in enumerate(candidates):
        skip, nexisting = should_skip_candidate(
            cand_idx=cand_idx,
            X=X,
            interactions_order=interactions_order,
            only_unknown_interactions=only_unknown_interactions,
            failed_candidates=failed_candidates,
            added_candidates=added_candidates,
            expected_gains=expected_gains,
            selected_vars=selected_vars,
            selected_interactions_vars=selected_interactions_vars,
            engineered_lineage=engineered_lineage,  # type: ignore[arg-type]
            dcd_state=dcd_state,
        )
        if skip:
            continue

        feasible_candidates.append((cand_idx, X, nexisting if reduce_gain_on_subelement_chosen else 0))

    if (
        _EVALUATE_CANDIDATES_POOL_ENABLED
        and n_workers > 1
        and (not use_simple_mode or len(cached_MIs) < num_possible_candidates)
        and len(feasible_candidates) > NMAX_NONPARALLEL_ITERS
    ):
        best_gain, best_candidate, run_out_of_time = score_candidates_parallel(
            ctx, feasible_candidates, best_gain, best_candidate, expected_gains, evaluate_candidates,
        )
    else:
        best_gain, best_candidate, run_out_of_time = score_candidates_serial(
            ctx, feasible_candidates, best_gain, best_candidate, expected_gains, cached_jmim_MIs, jmim_hit_counter,
            _prefill_cond_MIs_gpu, evaluate_candidate, handle_best_candidate,
        )

    if verbose > 2 and len(selected_vars) < MAX_ITERATIONS_TO_TRACK:
        logger.info("evaluate_candidates took %.1f sec.", timer() - eval_start)

    # Publish JMIM cache observability (size + cumulative hits) so tests /
    # benches can confirm the cache engages. Only meaningful in JMIM mode; harmless else.
    try:
        from .info_theory import use_jmim_aggregator as _use_jmim_obs
        if _use_jmim_obs():
            from . import evaluation as _ev_obs

            _ev_obs._JMIM_CACHE_STATS.append({"size": len(cached_jmim_MIs), "hits": int(jmim_hit_counter[0])})
    except Exception as e:  # nosec B110 - optional dependency import guard
        logger.debug("skipping JMIM cache-stats publish: %s", e)

    return best_gain, best_candidate, run_out_of_time


def confirm_candidate(ctx: ScreenContext, X: tuple, next_best_gain: float) -> tuple:
    """Permutation-confirm a single candidate ``X`` given ``ctx.selected_vars``.

    Direct-MI bootstrap (cached in ``cached_confident_MIs[X]``) plus the optional Fleuret
    conditional recheck. Returns ``(bootstrapped_gain, confidence)``. Verbatim port of the
    original confirmation block.
    """

    # Alias shared/static state.
    full_npermutations = ctx.full_npermutations
    selected_vars = ctx.selected_vars
    use_simple_mode = ctx.use_simple_mode
    cached_confident_MIs = ctx.cached_confident_MIs
    use_gpu = ctx.use_gpu
    factors_data = ctx.factors_data
    factors_nbins = ctx.factors_nbins
    factors_names = ctx.factors_names
    classes_y = ctx.classes_y
    classes_y_safe = ctx.classes_y_safe
    freqs_y = ctx.freqs_y
    freqs_y_safe = ctx.freqs_y_safe
    min_nonzero_confidence = ctx.min_nonzero_confidence
    n_workers = ctx.n_workers
    workers_pool = ctx.workers_pool
    parallel_kwargs = ctx.parallel_kwargs
    y = ctx.y
    random_seed = ctx.random_seed
    verbose = ctx.verbose
    ndigits = ctx.ndigits

    if full_npermutations:

        if verbose > 2:
            logger.info(
                "confirming candidate %s, next_best_gain=%.*f",
                get_candidate_name(X, factors_names=factors_names), ndigits, next_best_gain,
            )

        if X in cached_confident_MIs:
            bootstrapped_gain, confidence = cached_confident_MIs[X]
        else:
            # fold the candidate's own identity into the seed (mirrors the
            # _fleuret_base_seed fix below) so ``random_seed`` actually reaches this marginal-confirmation
            # permutation gate (pre-fix: neither mi_direct nor mi_direct_gpu received any seed here, so the
            # CPU branch always drew the identical base_seed=0 permutation stream for every candidate, and
            # the GPU branch drew fresh unseeded entropy every call - non-reproducible and unaffected by
            # ctx.random_seed either way).
            _marginal_base_seed = int(((int(random_seed or 0) * 2654435761) + hash(X)) & 0xFFFFFFFF)
            # Route the confirm to CPU ``mi_direct`` (which carries the analytic large-n null - the
            # nperm->inf limit of the permutation test: faster, decision-equivalent) whenever the
            # analytic path would engage (n >= threshold) OR the GPU is globally disabled. The GPU
            # permutation confirm (``mi_direct_gpu``, cupy argsort) was 37% of a 300k fit on a weak
            # GPU (cupy argsort + GPU-sync sleep, CPU idle), and ignored MLFRAME_DISABLE_GPU. At large
            # n the analytic CPU path supersedes it everywhere; small-n + enabled GPU still uses GPU.
            _confirm_use_gpu = marginal_confirm_use_gpu(use_gpu, factors_data)
            if _confirm_use_gpu:
                # Mirror evaluate_candidate's GPU
                # try/except (evaluation.py, fix) - this call previously had NO exception
                # handling, unlike every sibling GPU dispatch point in this codebase. A transient CUDA
                # fault here (driver hiccup, OOM, a poisoned context from an earlier unrelated GPU call)
                # would otherwise propagate all the way up and crash the whole (possibly hours-long)
                # MRMR.fit() instead of degrading to CPU for this one candidate.
                try:
                    bootstrapped_gain, confidence = mi_direct_gpu(
                        factors_data,
                        x=X,
                        y=y,  # type: ignore[arg-type]
                        factors_nbins=factors_nbins,  # type: ignore[arg-type]
                        classes_y=classes_y,
                        freqs_y=freqs_y,
                        freqs_y_safe=freqs_y_safe,  # type: ignore[arg-type]
                        classes_y_safe=classes_y_safe,  # type: ignore[arg-type]
                        min_nonzero_confidence=min_nonzero_confidence,
                        npermutations=full_npermutations,
                        base_seed=_marginal_base_seed,
                    )
                except Exception as _gpu_confirm_exc:
                    logger.warning(
                        "confirm_candidate: GPU marginal confirmation failed (%s: %s); falling back to CPU for this candidate.",
                        type(_gpu_confirm_exc).__name__, _gpu_confirm_exc,
                    )
                    bootstrapped_gain, confidence = mi_direct(
                        factors_data,
                        x=X,
                        y=y,  # type: ignore[arg-type]
                        factors_nbins=factors_nbins,  # type: ignore[arg-type]
                        classes_y=classes_y,
                        freqs_y=freqs_y,
                        classes_y_safe=classes_y_safe,  # type: ignore[arg-type]
                        min_nonzero_confidence=min_nonzero_confidence,
                        npermutations=full_npermutations,
                        n_workers=n_workers,
                        workers_pool=workers_pool,
                        parallel_kwargs=parallel_kwargs,
                        base_seed=_marginal_base_seed,
                    )
            else:
                if verbose and len(selected_vars) < MAX_ITERATIONS_TO_TRACK:
                    eval_start = timer()
                bootstrapped_gain, confidence = mi_direct(
                    factors_data,
                    x=X,
                    y=y,  # type: ignore[arg-type]
                    factors_nbins=factors_nbins,  # type: ignore[arg-type]
                    classes_y=classes_y,
                    freqs_y=freqs_y,
                    classes_y_safe=classes_y_safe,  # type: ignore[arg-type]
                    min_nonzero_confidence=min_nonzero_confidence,
                    npermutations=full_npermutations,
                    n_workers=n_workers,
                    workers_pool=workers_pool,
                    parallel_kwargs=parallel_kwargs,
                    base_seed=_marginal_base_seed,
                    # belt-and-braces - this branch runs
                    # precisely when the caller already decided NOT to use GPU (_confirm_use_gpu=False);
                    # pin prefer_gpu=False explicitly rather than trusting mi_direct's own internal gate
                    # (which is now also fixed to check gpu_globally_disabled(), but a caller that already
                    # made the decision should not need to re-trust a callee it opted out of).
                    prefer_gpu=False,
                )
                if verbose and len(selected_vars) < MAX_ITERATIONS_TO_TRACK:
                    logger.debug("mi_direct bootstrapped eval took %.1f sec.", timer() - eval_start)
            cached_confident_MIs[X] = bootstrapped_gain, confidence
    else:
        if X in cached_confident_MIs:
            bootstrapped_gain, confidence = cached_confident_MIs[X]
        else:
            bootstrapped_gain, confidence = next_best_gain, 1.0

    if full_npermutations and bootstrapped_gain > 0 and selected_vars and not use_simple_mode:  # additional check of Fleuret criteria

        bootstrapped_gain, confidence = fleuret_conditional_recheck(ctx, X, next_best_gain, bootstrapped_gain, confidence)

    return bootstrapped_gain, confidence


def confirm_one_predictor(
    ctx: ScreenContext,
    nconsec_unconfirmed: int,
    total_checked: int,
    total_disproved: int,
    patience_triggered: bool,
) -> tuple:
    """Run one full single-predictor confirmation cycle (the original ``while True`` loop).

    Re-scores all feasible candidates, then permutation-confirms them in expected-gain order
    (with the partial-gain recompute/retry + patience logic), composing :func:`score_candidates`
    and :func:`confirm_candidate`. Returns
    ``(best_candidate, best_gain, confidence, run_out_of_time, nconsec_unconfirmed,
    total_checked, total_disproved, patience_triggered)``.
    """

    next_best_gain: Any = None
    next_best_candidate_idx: Any = None
    X: Any = None
    candidates = ctx.candidates
    min_relevance_gain = ctx.min_relevance_gain
    selected_vars = ctx.selected_vars
    added_candidates = ctx.added_candidates
    max_consec_unconfirmed = ctx.max_consec_unconfirmed
    factors_names = ctx.factors_names
    verbose = ctx.verbose
    ndigits = ctx.ndigits

    best_candidate = None
    best_gain = min_relevance_gain - 1
    expected_gains = np.zeros(len(candidates), dtype=_crit_np_dtype())
    confidence = 1.0
    run_out_of_time = False

    # Order-invariant gain tie-break. The descending-gain selection order below
    # used ``np.arange`` (candidate POSITION) as its secondary key, so on a gain
    # tie the lower-positioned candidate won - making the whole greedy path
    # depend on input column order: reversing the columns reverses the positions
    # and a DIFFERENT feature wins the tie, then conditions every later pick,
    # yielding a different selected set (column-order-invariance contract break,
    # reproducible at n=400 AND n=25000). Tie-break instead on the candidate's
    # NAME, which is invariant under column reordering; rank gives a stable,
    # contiguous integer key for ``np.lexsort``.
    _rank_cache = cast(Optional[tuple[list, list[str], np.ndarray]], ctx._name_rank_cache)
    _name_rank = _confirm_one_predic_contiguous_integer_key_np(_rank_cache, candidates, factors_names, ctx)

    while True:  # confirmation loop (by random permutations)

        # Time EVERY score_candidates call (the whole remaining-pool rescore, cost
        # scaling with pool_size x |selected_vars|) separately from the rest of this retry loop, and
        # accumulate onto ctx so it survives across retries/predictors/interactions_orders. This is
        # what let a prior investigation reconstruct, after the fact, that wall time here (NOT the
        # FE pair-precompute step logged elsewhere) drove a slow screen_predictors round; logging it
        # live makes that reconstruction unnecessary next time.
        _sc_t0 = timer()
        best_gain, best_candidate, run_out_of_time = score_candidates(ctx, best_gain, best_candidate, expected_gains)
        ctx.sc_wall += timer() - _sc_t0
        ctx.sc_calls += 1
        if verbose >= 2 and ctx.sc_calls % 25 == 0:
            logger.info(
                "score_candidates progress: pool=%d |Z|=%d cumulative_wall=%.1fs calls=%d (avg %.3fs/call)",
                len(candidates), len(selected_vars), ctx.sc_wall, ctx.sc_calls, ctx.sc_wall / ctx.sc_calls,
            )

        if run_out_of_time:
            break

        if best_gain < min_relevance_gain:
            if verbose >= 2:
                logger.info("Minimum expected gain reached or no candidates to check anymore.")
            break  # exit confirmation while loop

        # ---------------------------------------------------------------------------------------------------------------
        # Now need to confirm best expected gain with a permutation test
        # ---------------------------------------------------------------------------------------------------------------

        cand_confirmed = False
        any_cand_considered = False
        # Descending-gain order with a stable index tie-break, then the directed-FE
        # promotion: on a near-tie (gain within ``prefer_engineered_rel_eps`` of the
        # leader) prefer an engineered candidate over its raw parent. This is the
        # decisive selection ordering, so the promotion here fixes BOTH the wrong
        # pick (raw parent winning an MI-tie) AND the backend nondeterminism (the
        # promotion reads only the backend-identical ``expected_gains`` array).
        _gain_order = np.lexsort((_name_rank, -np.asarray(expected_gains)))
        _gain_order = _prefer_engineered_order(_gain_order, expected_gains, ctx)
        (
            best_gain, cand_confirmed, any_cand_considered, nconsec_unconfirmed, total_checked, total_disproved, patience_triggered,
            next_best_candidate_idx, X, next_best_gain, confidence,
        ) = confirm_in_gain_order(
            ctx, _gain_order, expected_gains, best_gain, nconsec_unconfirmed, total_checked, total_disproved, patience_triggered, confidence,
            confirm_candidate, find_best_partial_gain, get_candidate_name,
        )

        # ---------------------------------------------------------------------------------------------------------------
        # Let's act upon results of the permutation test
        # ---------------------------------------------------------------------------------------------------------------

        if cand_confirmed:
            # Per-signal prefer-engineered substitution. By the data-processing
            # inequality a transform E=f(P) cannot out-score its raw parent P on
            # mutual information, so when a raw feature wins a near-tie against
            # its own transform the MI criterion always keeps the raw one - yet
            # the transform is the representation a shallow downstream actually
            # needs. If the confirmed winner X is raw and a transform of X sits
            # within the prefer-engineered band below it AND independently
            # confirms, select the transform instead (the raw parent then
            # becomes redundant and is marked accepted so it is not reselected).
            _sub_idx, _sub_X, _sub_gain = next_best_candidate_idx, X, next_best_gain
            _child = _confirmable_engineered_child(ctx, X, next_best_candidate_idx, next_best_gain, expected_gains)
            _sub_X, _sub_gain, _sub_idx = _confirm_one_predic_becomes_redundant_marked_accepted(_child, ctx, added_candidates, next_best_candidate_idx, verbose, X, factors_names, ndigits, next_best_gain, _sub_X, _sub_gain, _sub_idx)
            added_candidates.add(_sub_idx)  # so it won't be selected again
            best_candidate = _sub_X
            best_gain = _sub_gain
            break  # exit confirmation while loop
        else:
            if not any_cand_considered:
                best_gain = min_relevance_gain - 1
                if verbose:
                    logger.info("No more candidates to confirm.")
                break  # exit confirmation while loop
            else:
                best_gain = min_relevance_gain - 1
                if max_consec_unconfirmed and (nconsec_unconfirmed > max_consec_unconfirmed):
                    patience_triggered = True
                    break  # exit confirmation while loop
                else:
                    pass  # retry all cands evaluation

    return (
        best_candidate,
        best_gain,
        confidence,
        run_out_of_time,
        nconsec_unconfirmed,
        total_checked,
        total_disproved,
        patience_triggered,
    )


def _confirm_one_predic_contiguous_integer_key_np(_rank_cache, candidates, factors_names, ctx):
    """Block of confirm_one_predictor starting at ``if _rank_cache is not None and _rank_cache[0] is candidates:``."""
    if _rank_cache is not None and _rank_cache[0] is candidates:
        _cand_names, _name_rank = _rank_cache[1], _rank_cache[2]
    else:
        _cand_names = [get_candidate_name(c, factors_names=factors_names) for c in candidates]
        _name_rank = np.empty(len(candidates), dtype=np.int64)
        for _rank, _pos in enumerate(sorted(range(len(_cand_names)), key=lambda i: _cand_names[i])):
            _name_rank[_pos] = _rank
        ctx._name_rank_cache = (candidates, _cand_names, _name_rank)
    return _name_rank


def _confirm_one_predic_becomes_redundant_marked_accepted(_child, ctx, added_candidates, next_best_candidate_idx, verbose, X, factors_names, ndigits, next_best_gain, _sub_X, _sub_gain, _sub_idx):
    """Block of confirm_one_predictor starting at ``if _child is not None:``."""
    if _child is not None:
        _child_idx, _child_X, _child_gain = _child
        _child_boot, _child_conf = confirm_candidate(ctx, _child_X, _child_gain)
        if _child_boot > 0:
            added_candidates.add(next_best_candidate_idx)  # raw parent now redundant
            _sub_idx, _sub_X, _sub_gain = _child_idx, _child_X, _child_gain * _child_conf
            if verbose >= 2:
                logger.info(
                    "prefer-engineered substitution: %s -> %s (raw gain %.*f, transform gain %.*f)",
                    get_candidate_name(X, factors_names=factors_names),
                    get_candidate_name(_child_X, factors_names=factors_names),
                    ndigits, next_best_gain, ndigits, _sub_gain,
                )
    return _sub_X, _sub_gain, _sub_idx
