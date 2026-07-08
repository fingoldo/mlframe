"""Single-predictor confirmation primitives, carved out of
``mlframe.feature_selection.filters._screen_predictors`` so that file stays well below the
1k-line monolith threshold and the (most-frequently-patched) confirmation math lives in one place:

* :func:`score_candidates`     -- evaluate every feasible candidate's conditional-MI
                                  gain given the current ``selected_vars`` (serial +
                                  joblib-parallel paths), filling ``expected_gains`` /
                                  ``partial_gains`` and returning the running best.
* :func:`confirm_candidate`    -- permutation-confirm a single candidate ``X`` (direct
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
from dataclasses import dataclass, field
from timeit import default_timer as timer
from typing import Sequence

import numba
import numpy as np
from numba.core import types
from joblib import delayed

from pyutilz.parallel import split_list_into_chunks

from ._internals import MAX_ITERATIONS_TO_TRACK, NMAX_NONPARALLEL_ITERS, sanitize
from ._numba_utils import count_cand_nbins
from .evaluation import (
    evaluate_candidate,
    evaluate_candidates,
    find_best_partial_gain,
    get_candidate_name,
    handle_best_candidate,
    should_skip_candidate,
    _prefill_cond_MIs_gpu,
)
from .fleuret import get_fleuret_criteria_confidence, get_fleuret_criteria_confidence_parallel
from .gpu import mi_direct_gpu
from .permutation import mi_direct, _addone_pvalue_enabled

logger = logging.getLogger(__name__)


# Re-export the full engineered-helper surface: ``confirm_one_predictor`` / ``confirm_candidate`` use three of them, and
# ``_mrmr_fit_impl`` imports ``_extract_single_raw_parent`` from this module by name (raw-retention pass).
from ._confirm_predictor_engineered import (  # noqa: F401
    _candidate_is_engineered,
    _confirmable_engineered_child,
    _conditioning_rows_per_cell,
    _extract_single_raw_parent,
    _PARENT_TOKEN_SPLIT,
    _prefer_engineered_order,
)


@dataclass
class ScreenContext:
    """Bundle of shared/static screening state threaded into the confirmation primitives.

    Static fields (data, algorithm parameters, thresholds) are set once by the orchestrator;
    the four caches are mutated in place across candidates; the per-order fields
    (``candidates`` .. ``failed_candidates``) are reassigned by the caller between predictors
    and interaction orders.
    """

    # --- data / target ---
    factors_data: np.ndarray
    factors_nbins: Sequence[int]
    factors_names: Sequence[str]
    y: Sequence[int]
    data_copy: np.ndarray
    classes_y: np.ndarray
    classes_y_safe: object
    freqs_y: np.ndarray
    freqs_y_safe: object
    # --- algorithm params ---
    mrmr_relevance_algo: str
    mrmr_redundancy_algo: str
    reduce_gain_on_subelement_chosen: bool
    max_veteranes_interactions_order: int
    only_unknown_interactions: bool
    use_gpu: bool
    use_simple_mode: bool
    extra_x_shuffling: bool
    engineered_lineage: object
    # --- performance ---
    n_workers: int
    workers_pool: object
    parallel_kwargs: dict
    # --- confidence / stopping ---
    baseline_npermutations: int
    full_npermutations: int
    min_nonzero_confidence: float
    max_failed: int
    min_relevance_gain: float
    max_consec_unconfirmed: int
    max_runtime_mins: float
    max_confirmation_cand_nbins: int
    random_seed: int
    # --- misc ---
    verbose: int
    ndigits: int
    start_time: float
    num_possible_candidates: int
    # --- shared MI caches (mutated in place) ---
    cached_MIs: dict
    cached_confident_MIs: dict
    cached_cond_MIs: object
    # 2026-06-19: JMIM joint-MI cache (numba typed dict) + 1-elem int64 hit-counter
    # array, mirroring ``cached_cond_MIs``. Lazily built in ``score_candidates`` on
    # first use so they persist across greedy rounds (cross-round reuse of the
    # ``{X} u Z`` multiset key). ``None`` default keeps direct ScreenContext callers
    # backward-compatible. Both are converted to plain values at any pickling
    # boundary (the typed dict never escapes onto an instance).
    cached_jmim_MIs: object = None
    jmim_hit_counter: object = None
    entropy_cache: object = None
    # --- per-interactions-order / per-node mutable state ---
    # These five are ALWAYS populated by the orchestrator before use (never read as None at
    # runtime); ``None`` is only a dataclass placeholder default (a required-after-defaulted-
    # field constraint), so they stay typed as their concrete container rather than Optional
    # -- which would force null-checks onto every one of their many non-nullable call sites.
    candidates: list = None  # type: ignore[assignment]
    # 2026-05-30 Wave 9 — DCD state forwarded into ``should_skip_candidate``
    # for pool_pruned_mask check. ``None`` preserves legacy bit-stable.
    dcd_state: object = None
    interactions_order: int = 1
    selected_vars: list = None  # type: ignore[assignment]
    selected_interactions_vars: list = None  # type: ignore[assignment]
    partial_gains: dict = None  # type: ignore[assignment]
    added_candidates: set = field(default=None)  # type: ignore[assignment,arg-type]
    failed_candidates: set = field(default=None)  # type: ignore[assignment,arg-type]
    # 2026-06-02 — directed-FE tie-break. ``raw_feature_names`` is the set of
    # ORIGINAL (pre-FE) column names; any ``factors_names[idx]`` not in it is an
    # engineered transform of its raw parent(s). On a near-tie in selection gain
    # (within ``prefer_engineered_rel_eps`` relative tolerance) the greedy pick
    # deterministically prefers the engineered candidate over a raw one: an
    # engineered column is a function of its parent, so on an MI-tie it dominates
    # representationally (a shallow downstream can use x1**2-1 but not raw x1),
    # and the deterministic rule removes the njit-vs-njit_par pick nondeterminism
    # that the prior index-order tie-break introduced. ``None`` raw-name set
    # falls back to the syntactic heuristic (a name containing ``(`` or ``__`` is
    # engineered) so direct callers still get deterministic behaviour. Setting
    # the rel-eps to ``0.0`` restores the legacy pure-index tie-break.
    raw_feature_names: object = None
    prefer_engineered_rel_eps: float = 0.0
    # 2026-06-02 RC2 — sample-size-aware Fleuret confirmation. When the
    # conditioning joint ``(X u selected_vars)`` has fewer than this many rows
    # per occupied cell, the conditional-MI permutation test is finite-sample
    # unreliable (shuffled-y null CMI ~= real CMI -> over-rejection / premature
    # stop) so ``confirm_candidate`` falls back to a MARGINAL-MI permutation
    # test (the X-marginal joint is far better sampled). ``0.0`` restores the
    # strict legacy conditional test for every candidate. Default ``5.0`` set
    # by the MRMR ctor (new behaviour ON since ``use_simple_mode=False`` is the
    # default). Dedup is unaffected (handled by the gain's redundancy term).
    fe_confirm_undersample_rows_per_cell: float = 0.0


def score_candidates(ctx: ScreenContext, best_gain: float, best_candidate, expected_gains: np.ndarray):
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
    dcd_state = getattr(ctx, "dcd_state", None)  # Wave 9
    reduce_gain_on_subelement_chosen = ctx.reduce_gain_on_subelement_chosen
    n_workers = ctx.n_workers
    use_simple_mode = ctx.use_simple_mode
    cached_MIs = ctx.cached_MIs
    num_possible_candidates = ctx.num_possible_candidates
    cached_cond_MIs = ctx.cached_cond_MIs
    # 2026-06-19: JMIM joint-MI cache + hit counter (see ScreenContext). Lazily
    # provisioned for direct ScreenContext callers that don't set them, so the
    # serial evaluate_candidate path always gets a real numba dict (the @njit
    # kernel types the cache ``in`` check unconditionally).
    cached_jmim_MIs = getattr(ctx, "cached_jmim_MIs", None)
    jmim_hit_counter = getattr(ctx, "jmim_hit_counter", None)
    if cached_jmim_MIs is None:
        cached_jmim_MIs = numba.typed.Dict.empty(
            key_type=types.unicode_type, value_type=types.float64,
        )
        ctx.cached_jmim_MIs = cached_jmim_MIs
    if jmim_hit_counter is None:
        jmim_hit_counter = np.zeros(1, dtype=np.int64)
        ctx.jmim_hit_counter = jmim_hit_counter
    entropy_cache = ctx.entropy_cache
    workers_pool = ctx.workers_pool
    y = ctx.y
    factors_data = ctx.factors_data
    factors_nbins = ctx.factors_nbins
    factors_names = ctx.factors_names
    classes_y = ctx.classes_y
    classes_y_safe = ctx.classes_y_safe
    freqs_y = ctx.freqs_y
    use_gpu = ctx.use_gpu
    freqs_y_safe = ctx.freqs_y_safe
    partial_gains = ctx.partial_gains
    baseline_npermutations = ctx.baseline_npermutations
    mrmr_relevance_algo = ctx.mrmr_relevance_algo
    mrmr_redundancy_algo = ctx.mrmr_redundancy_algo
    max_veteranes_interactions_order = ctx.max_veteranes_interactions_order
    cached_confident_MIs = ctx.cached_confident_MIs
    max_runtime_mins = ctx.max_runtime_mins
    start_time = ctx.start_time
    min_relevance_gain = ctx.min_relevance_gain
    verbose = ctx.verbose
    ndigits = ctx.ndigits

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

    if n_workers > 1 and (not use_simple_mode or len(cached_MIs) < num_possible_candidates) and len(feasible_candidates) > NMAX_NONPARALLEL_ITERS:
        temp_cached_cond_MIs = sanitize(cached_cond_MIs)
        temp_entropy_cache = sanitize(entropy_cache)
        # 2026-05-30 Wave 9.1 iter 5 fix: snapshot main-thread Wave 8
        # toggles. ``threading.local`` storage from info_theory.py does NOT
        # propagate across joblib workers (each worker thread gets its own
        # local namespace), so without explicit forwarding the workers
        # silently read ``False`` / ``0.0`` defaults and Wave 8 features
        # (SU normalization, JMIM, BUR) become no-ops in the parallel hot
        # path - the common case since backend='threading' is the default.
        from .info_theory import (
            use_su_normalization as _use_su, use_jmim_aggregator as _use_jmim,
            get_bur_lambda as _get_bur, get_relaxmrmr_alpha as _get_relax,
            get_pid_synergy_bonus as _get_pid, get_cmi_perm_stop as _get_cmi,
            use_mi_miller_madow as _use_mm,
        )
        from .info_theory._state_and_dispatch import get_group_mi as _get_gmi
        _gmi_snapshot = _get_gmi()
        _su_snapshot = bool(_use_su())
        _jmim_snapshot = bool(_use_jmim())
        _bur_snapshot = float(_get_bur())
        _relax_snapshot = float(_get_relax())
        _pid_snapshot = float(_get_pid())
        _cmi_snapshot = _get_cmi()
        _mm_snapshot = bool(_use_mm())
        res = workers_pool(  # type: ignore[operator]
            delayed(evaluate_candidates)(
                workload=workload,
                y=y,
                best_gain=best_gain,
                factors_data=factors_data,
                factors_nbins=factors_nbins,
                factors_names=factors_names,
                classes_y=classes_y,
                freqs_y=freqs_y,
                use_gpu=use_gpu,
                freqs_y_safe=freqs_y_safe,
                # Per-worker COPIES of the two dicts the worker writes in-place (unlike cond_MIs/entropy,
                # which each worker re-copies into its own numba dict). Sharing one object let several
                # threading workers __setitem__ it concurrently -> a "dict changed size during iteration"
                # crash at scale. sanitize() is evaluated once per workload so each worker owns its copy;
                # the merge-back below consolidates them (workers own disjoint candidate keys, so the
                # merged result is byte-identical to the old shared-write path).
                partial_gains=sanitize(partial_gains),
                baseline_npermutations=baseline_npermutations,
                mrmr_relevance_algo=mrmr_relevance_algo,
                mrmr_redundancy_algo=mrmr_redundancy_algo,
                max_veteranes_interactions_order=max_veteranes_interactions_order,
                selected_vars=selected_vars,
                cached_MIs=sanitize(cached_MIs),
                cached_confident_MIs=cached_confident_MIs,
                cached_cond_MIs=temp_cached_cond_MIs,
                entropy_cache=temp_entropy_cache,
                max_runtime_mins=max_runtime_mins,
                start_time=start_time,
                min_relevance_gain=min_relevance_gain,
                verbose=verbose,
                ndigits=ndigits,
                use_simple_mode=use_simple_mode,
                use_su=_su_snapshot,
                use_jmim=_jmim_snapshot,
                bur_lambda=_bur_snapshot,
                relaxmrmr_alpha=_relax_snapshot,
                pid_synergy_bonus=_pid_snapshot,
                cmi_perm=_cmi_snapshot,
                mi_miller_madow=_mm_snapshot,
                group_mi=_gmi_snapshot,
            )
            for workload in split_list_into_chunks(feasible_candidates, max(1, len(feasible_candidates) // n_workers))
        )

        for (
            worker_best_gain,
            worker_best_candidate,
            worker_partial_gains,
            worker_expected_gains,
            worker_cached_MIs,
            worker_cached_cond_MIs,
            worker_entropy_cache,
        ) in res:

            if worker_best_gain > best_gain:
                best_candidate = worker_best_candidate
                best_gain = worker_best_gain

            # sync caches
            for local_storage, global_storage in [
                (worker_expected_gains, expected_gains),
                (worker_cached_MIs, cached_MIs),
                (worker_cached_cond_MIs, cached_cond_MIs),
                (worker_entropy_cache, entropy_cache),
            ]:
                for key, value in local_storage.items():
                    global_storage[key] = value  # type: ignore[index]

            for cand_idx, (worker_current_gain, worker_z_idx) in worker_partial_gains.items():
                if cand_idx in partial_gains:
                    current_gain, z_idx = partial_gains[cand_idx]
                else:
                    z_idx = -2
                if worker_z_idx > z_idx:
                    partial_gains[cand_idx] = (worker_current_gain, worker_z_idx)

        if use_simple_mode:
            # need to sort all cands by perfs
            pass
        if max_runtime_mins and not run_out_of_time:
            run_out_of_time = (timer() - start_time) > max_runtime_mins * 60
            if run_out_of_time:
                logger.info("Time limit exhausted. Finalizing the search...")

    else:
        # Batched-GPU conditional-MI cache pre-fill for the SERIAL path (default ON;
        # env kill-switch MLFRAME_MRMR_GPU_CMI=0). Same win as the parallel wiring in
        # _evaluate_candidates_inner: pre-populate the shared python ``cached_cond_MIs``
        # with batched I(X; Y | Z) so the @njit evaluate_gain loop hits the cache and
        # skips the serial scalar conditional_mi. Unlike the parallel path (which writes
        # a per-worker numba dict to avoid a concurrent "dict changed size" race), the
        # serial loop has NO concurrent iteration, so writing the plain python dict in
        # place BEFORE the loop is race-free. The helper reuses the exact arr2str key
        # format and the same size/HW dispatch + SU/JMIM/order-1 gating; bit-parity
        # kernel => selection unchanged; any failure / non-eligible regime is a silent
        # no-op (scalar path).
        _prefill_cond_MIs_gpu(
            workload=feasible_candidates,
            y=y,
            factors_data=factors_data,
            factors_nbins=factors_nbins,
            selected_vars=selected_vars,
            cached_cond_MIs=cached_cond_MIs,
            use_simple_mode=use_simple_mode,
            mrmr_relevance_algo=mrmr_relevance_algo,
            max_veteranes_interactions_order=max_veteranes_interactions_order,
        )

        for cand_idx, X, nexisting in feasible_candidates:

            current_gain, sink_reasons = evaluate_candidate(
                cand_idx=cand_idx,
                X=X,
                y=y,
                nexisting=nexisting,
                best_gain=best_gain,
                factors_data=factors_data,
                factors_nbins=factors_nbins,  # type: ignore[arg-type]
                factors_names=factors_names,
                classes_y=classes_y,
                classes_y_safe=classes_y_safe,  # type: ignore[arg-type]
                freqs_y=freqs_y,
                use_gpu=use_gpu,
                freqs_y_safe=freqs_y_safe,  # type: ignore[arg-type]
                partial_gains=partial_gains,
                baseline_npermutations=baseline_npermutations,
                mrmr_relevance_algo=mrmr_relevance_algo,
                mrmr_redundancy_algo=mrmr_redundancy_algo,
                max_veteranes_interactions_order=max_veteranes_interactions_order,
                expected_gains=expected_gains,
                selected_vars=selected_vars,
                cached_MIs=cached_MIs,
                cached_confident_MIs=cached_confident_MIs,
                cached_cond_MIs=cached_cond_MIs,  # type: ignore[arg-type]
                cached_jmim_MIs=cached_jmim_MIs,
                jmim_hit_counter=jmim_hit_counter,
                entropy_cache=entropy_cache,  # type: ignore[arg-type]
                verbose=verbose,
                ndigits=ndigits,
                use_simple_mode=use_simple_mode,
            )

            best_gain, best_candidate, run_out_of_time = handle_best_candidate(
                current_gain=current_gain,
                best_gain=best_gain,
                X=X,
                best_candidate=best_candidate,
                factors_names=factors_names,  # type: ignore[arg-type]
                max_runtime_mins=max_runtime_mins,
                start_time=start_time,
                min_relevance_gain=min_relevance_gain,
                verbose=verbose,
                ndigits=ndigits,
            )

            if run_out_of_time:
                if verbose:
                    logger.info("Time limit exhausted. Finalizing the search...")
                break

    if verbose > 2 and len(selected_vars) < MAX_ITERATIONS_TO_TRACK:
        logger.info("evaluate_candidates took %.1f sec.", timer() - eval_start)

    # 2026-06-19: publish JMIM cache observability (size + cumulative hits) so tests /
    # benches can confirm the cache engages. Only meaningful in JMIM mode; harmless else.
    try:
        from .info_theory import use_jmim_aggregator as _use_jmim_obs
        if _use_jmim_obs():
            from . import evaluation as _ev_obs

            _ev_obs._JMIM_CACHE_STATS.append({"size": len(cached_jmim_MIs), "hits": int(jmim_hit_counter[0])})
    except Exception:  # nosec B110 - optional dependency import guard
        pass

    return best_gain, best_candidate, run_out_of_time


def confirm_candidate(ctx: ScreenContext, X: tuple, next_best_gain: float):
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
    max_confirmation_cand_nbins = ctx.max_confirmation_cand_nbins
    data_copy = ctx.data_copy
    y = ctx.y
    max_failed = ctx.max_failed
    mrmr_relevance_algo = ctx.mrmr_relevance_algo
    mrmr_redundancy_algo = ctx.mrmr_redundancy_algo
    max_veteranes_interactions_order = ctx.max_veteranes_interactions_order
    cached_cond_MIs = ctx.cached_cond_MIs
    entropy_cache = ctx.entropy_cache
    extra_x_shuffling = ctx.extra_x_shuffling
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
            # Route the confirm to CPU ``mi_direct`` (which carries the analytic large-n null -- the
            # nperm->inf limit of the permutation test: faster, decision-equivalent) whenever the
            # analytic path would engage (n >= threshold) OR the GPU is globally disabled. The GPU
            # permutation confirm (``mi_direct_gpu``, cupy argsort) was 37% of a 300k fit on a weak
            # GPU (cupy argsort + GPU-sync sleep, CPU idle), and ignored MLFRAME_DISABLE_GPU. At large
            # n the analytic CPU path supersedes it everywhere; small-n + enabled GPU still uses GPU.
            _confirm_use_gpu = use_gpu
            if _confirm_use_gpu:
                try:
                    import os as _os
                    from ._analytic_mi_null import analytic_null_min_n, analytic_null_enabled
                    _cvd = _os.environ.get("CUDA_VISIBLE_DEVICES", None)
                    _gpu_off = _os.environ.get("MLFRAME_DISABLE_GPU", "") == "1" or (_cvd is not None and _cvd.strip() == "")
                    _analytic_n = analytic_null_enabled() and int(factors_data.shape[0]) >= analytic_null_min_n()
                    if _gpu_off or _analytic_n:
                        _confirm_use_gpu = False
                except Exception as e:  # nosec B110 - swallow converted to debug-log, non-fatal by design
                    logger.debug("suppressed in _confirm_predictor.py:528: %s", e)
                    pass
            if _confirm_use_gpu:
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
                )
                if verbose and len(selected_vars) < MAX_ITERATIONS_TO_TRACK:
                    logger.info("mi_direct bootstrapped eval took %.1f sec.", timer() - eval_start)
            cached_confident_MIs[X] = bootstrapped_gain, confidence
    else:
        if X in cached_confident_MIs:
            bootstrapped_gain, confidence = cached_confident_MIs[X]
        else:
            bootstrapped_gain, confidence = next_best_gain, 1.0

    if full_npermutations and bootstrapped_gain > 0 and selected_vars and not use_simple_mode:  # additional check of Fleuret criteria

        if count_cand_nbins(X, factors_nbins) <= max_confirmation_cand_nbins:

            skip_cand = [(subel in selected_vars) for subel in X]
            nexisting = sum(skip_cand)

            if verbose and len(selected_vars) < MAX_ITERATIONS_TO_TRACK:
                eval_start = timer()

            # RC2 sample-size-aware confirmation. The Fleuret CONDITIONAL
            # permutation test estimates ``I(X; Y | Z)`` over the (X, Y, Z)
            # joint; on a small-n / high-cardinality conditioning joint that
            # joint is severely undersampled (e.g. diabetes s5 -> ~0.4
            # rows/cell) so the shuffled-y NULL conditional MI is ~= the REAL
            # conditional MI and the gate OVER-REJECTS every genuine feature
            # after the first (premature stop, catastrophic under-selection).
            # When the conditioning joint has fewer than
            # ``fe_confirm_undersample_rows_per_cell`` rows per occupied cell,
            # confirm the candidate by a MARGINAL-MI permutation test instead
            # (the X-marginal joint is far better sampled). The marginal
            # bootstrap (``bootstrapped_gain`` / ``confidence`` computed above
            # via ``mi_direct`` / ``mi_direct_gpu``, which permute y and
            # compare real vs permuted I(X; y)) IS exactly that test, so the
            # fallback is simply to keep its verdict and SKIP the unreliable
            # conditional recheck. Pure noise is still rejected because its
            # marginal permutation test rejects it; redundant duplicates are
            # still dropped because the relevance-minus-redundancy GAIN
            # (``next_best_gain``, already net of the redundancy term) handles
            # dedup independently of this gate. Set the threshold to 0 (knob
            # ``fe_confirm_undersample_rows_per_cell=0``) to always use the
            # strict conditional test (legacy behaviour).
            _undersample_threshold = float(getattr(ctx, "fe_confirm_undersample_rows_per_cell", 0.0) or 0.0)
            if _undersample_threshold > 0.0:
                _rows_per_cell = _conditioning_rows_per_cell(ctx, X)
                if _rows_per_cell < _undersample_threshold:
                    if verbose and len(selected_vars) < MAX_ITERATIONS_TO_TRACK:
                        logger.info(
                            "confirm %s: conditioning joint undersampled (%.2f rows/cell < %.2f); " "marginal-MI fallback (conf=%.*f)",
                            get_candidate_name(X, factors_names=factors_names),
                            _rows_per_cell, _undersample_threshold, ndigits, confidence,
                        )
                    # Keep the marginal bootstrap verdict; do not run the
                    # unreliable conditional permutation gate.
                    return bootstrapped_gain, confidence

            _fleuret_base_seed = int(((int(random_seed or 0) * 2654435761) + len(selected_vars) + 1) & 0xFFFFFFFFFFFFFFFF)
            if n_workers and n_workers > 1 and full_npermutations > NMAX_NONPARALLEL_ITERS:
                bootstrapped_gain, confidence, parallel_entropy_cache = get_fleuret_criteria_confidence_parallel(
                    data_copy=data_copy,
                    factors_nbins=factors_nbins,  # type: ignore[arg-type]
                    x=X,
                    y=y,  # type: ignore[arg-type]
                    selected_vars=selected_vars,
                    bootstrapped_gain=next_best_gain,
                    npermutations=full_npermutations,
                    max_failed=max_failed,
                    nexisting=nexisting,
                    mrmr_relevance_algo=mrmr_relevance_algo,
                    mrmr_redundancy_algo=mrmr_redundancy_algo,
                    max_veteranes_interactions_order=max_veteranes_interactions_order,
                    cached_cond_MIs=cached_cond_MIs,  # type: ignore[arg-type]
                    entropy_cache=entropy_cache,  # type: ignore[arg-type]
                    extra_x_shuffling=extra_x_shuffling,
                    n_workers=n_workers,
                    workers_pool=workers_pool,
                    parallel_kwargs=parallel_kwargs,
                    base_seed=_fleuret_base_seed,
                )
                for key, value in parallel_entropy_cache.items():
                    entropy_cache[key] = value  # type: ignore[index]
            else:
                nfailed, nchecked = get_fleuret_criteria_confidence(
                    data_copy=data_copy,
                    factors_nbins=factors_nbins,
                    x=X,
                    y=y,
                    selected_vars=selected_vars,
                    bootstrapped_gain=next_best_gain,
                    npermutations=full_npermutations,
                    max_failed=max_failed,
                    nexisting=nexisting,
                    mrmr_relevance_algo=mrmr_relevance_algo,
                    mrmr_redundancy_algo=mrmr_redundancy_algo,
                    max_veteranes_interactions_order=max_veteranes_interactions_order,
                    cached_cond_MIs=cached_cond_MIs,
                    entropy_cache=entropy_cache,
                    extra_x_shuffling=extra_x_shuffling,
                    base_seed=np.uint64(_fleuret_base_seed),
                )
                # The SERIAL confirm runs only at the screen's tiny default budget (full_npermutations=3,
                # below NMAX_NONPARALLEL_ITERS) where the parallel path is never taken -- so there is no serial-vs-
                # parallel comparison to reconcile here. The add-one estimator's ceiling at budget 3 is 1-1/4=0.75,
                # which lowers gain*confidence ~25% below min_relevance_gain and STARVES the screen (a prior C4
                # attempt to add-one this path regressed cluster-aggregate / FE recovery). Keep the raw rate: a clean
                # null (nfailed=0) correctly yields confidence 1.0 so genuine signals clear the small-budget screen.
                confidence = 1 - nfailed / nchecked
                if nfailed >= max_failed:
                    bootstrapped_gain = 0.0

            if verbose and len(selected_vars) < MAX_ITERATIONS_TO_TRACK:
                logger.info("get_fleuret_criteria_confidence bootstrapped eval took %.1f sec.", timer() - eval_start)

    return bootstrapped_gain, confidence


def confirm_one_predictor(
    ctx: ScreenContext,
    nconsec_unconfirmed: int,
    total_checked: int,
    total_disproved: int,
    patience_triggered: bool,
):
    """Run one full single-predictor confirmation cycle (the original ``while True`` loop).

    Re-scores all feasible candidates, then permutation-confirms them in expected-gain order
    (with the partial-gain recompute/retry + patience logic), composing :func:`score_candidates`
    and :func:`confirm_candidate`. Returns
    ``(best_candidate, best_gain, confidence, run_out_of_time, nconsec_unconfirmed,
    total_checked, total_disproved, patience_triggered)``.
    """

    candidates = ctx.candidates
    min_relevance_gain = ctx.min_relevance_gain
    selected_vars = ctx.selected_vars
    failed_candidates = ctx.failed_candidates
    added_candidates = ctx.added_candidates
    partial_gains = ctx.partial_gains
    full_npermutations = ctx.full_npermutations
    max_consec_unconfirmed = ctx.max_consec_unconfirmed
    factors_names = ctx.factors_names
    verbose = ctx.verbose
    ndigits = ctx.ndigits

    best_candidate = None
    best_gain = min_relevance_gain - 1
    expected_gains = np.zeros(len(candidates), dtype=np.float64)
    confidence = 1.0
    run_out_of_time = False

    # Order-invariant gain tie-break. The descending-gain selection order below
    # used ``np.arange`` (candidate POSITION) as its secondary key, so on a gain
    # tie the lower-positioned candidate won -- making the whole greedy path
    # depend on input column order: reversing the columns reverses the positions
    # and a DIFFERENT feature wins the tie, then conditions every later pick,
    # yielding a different selected set (column-order-invariance contract break,
    # reproducible at n=400 AND n=25000). Tie-break instead on the candidate's
    # NAME, which is invariant under column reordering; rank gives a stable,
    # contiguous integer key for ``np.lexsort``.
    _cand_names = [get_candidate_name(c, factors_names=factors_names) for c in candidates]
    _name_rank = np.empty(len(candidates), dtype=np.int64)
    for _rank, _pos in enumerate(sorted(range(len(_cand_names)), key=lambda i: _cand_names[i])):
        _name_rank[_pos] = _rank

    while True:  # confirmation loop (by random permutations)

        best_gain, best_candidate, run_out_of_time = score_candidates(ctx, best_gain, best_candidate, expected_gains)

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
        for n, next_best_candidate_idx in enumerate(_gain_order):
            next_best_gain = expected_gains[next_best_candidate_idx]
            if next_best_gain >= min_relevance_gain:  # only can consider here candidates fully checked against every Z

                X = candidates[next_best_candidate_idx]

                if n > 0:
                    best_partial_gain, best_key = find_best_partial_gain(
                        partial_gains=partial_gains,
                        failed_candidates=failed_candidates,
                        added_candidates=added_candidates,
                        candidates=candidates,
                        selected_vars=selected_vars,
                        dcd_state=getattr(ctx, "dcd_state", None),
                    )

                    if best_partial_gain > next_best_gain:
                        best_gain = next_best_gain
                        if verbose > 2:
                            logger.debug(
                                "Have no best_candidate anymore. Need to recompute partial gains. best_partial_gain of candidate %s was %s",
                                get_candidate_name(candidates[best_key], factors_names=factors_names),
                                best_partial_gain,
                            )
                        break  # out of best candidates confirmation, to retry all cands evaluation

                any_cand_considered = True

                if full_npermutations:
                    total_checked += 1

                bootstrapped_gain, confidence = confirm_candidate(ctx, X, next_best_gain)

                if bootstrapped_gain > 0:

                    nconsec_unconfirmed = 0

                    # Budget-aware confidence for the gain multiplier. The marginal bootstrap confidence
                    # from ``mi_direct`` uses the add-one Monte-Carlo p-value (calibrated for the REPORTED
                    # significance), whose ceiling at a clean null is ``budget/(budget+1)`` -- only 0.75 at
                    # the screen's tiny default budget (full_npermutations=3). Multiplying the gain by that
                    # ceiling deflates a perfectly clean signal ~25%, dropping it below ``min_relevance_gain``
                    # and starving the FE pair-pool (engineered features collapse to ~0). Dividing by the
                    # add-one ceiling exactly recovers the raw exceedance-rate confidence (1.0 at a clean
                    # null) the screen used pre-add-one, so genuine signals clear the small-budget screen
                    # while the calibrated add-one confidence is still what gets STORED/reported. No-op when
                    # add-one is disabled (ceiling == 1.0) and on large budgets (ceiling -> 1.0).
                    _conf_for_gain = confidence
                    if _addone_pvalue_enabled() and full_npermutations:
                        _addone_ceiling = full_npermutations / (full_npermutations + 1.0)
                        if _addone_ceiling > 0.0:
                            _conf_for_gain = min(1.0, confidence / _addone_ceiling)
                    next_best_gain = next_best_gain * _conf_for_gain
                    expected_gains[next_best_candidate_idx] = next_best_gain

                    best_partial_gain, best_key = find_best_partial_gain(
                        partial_gains=partial_gains,
                        failed_candidates=failed_candidates,
                        added_candidates=added_candidates,
                        candidates=candidates,
                        selected_vars=selected_vars,
                        skip_indices=(next_best_candidate_idx,),
                        dcd_state=getattr(ctx, "dcd_state", None),
                    )
                    if best_partial_gain > next_best_gain:
                        if verbose > 2:
                            logger.info(
                                "\t\tCandidate's lowered confidence %s requires re-checking other candidates, "
                                "as now its expected gain is only %.*f, vs %.*f, of %s",
                                confidence, ndigits, next_best_gain, ndigits, best_partial_gain,
                                get_candidate_name(candidates[best_key], factors_names=factors_names),
                            )
                        break  # out of best candidates confirmation, to retry all cands evaluation
                    else:
                        cand_confirmed = True
                        if full_npermutations:
                            if verbose > 2:
                                logger.info("\t\tconfirmed with confidence %.*f", ndigits, confidence)
                        break  # out of best candidates confirmation, to add candidate to the list, and go to more candidates
                else:
                    expected_gains[next_best_candidate_idx] = 0.0
                    failed_candidates.add(next_best_candidate_idx)
                    if verbose > 2:
                        logger.info("\t\tconfirmation failed with confidence %.*f", ndigits, confidence)

                    nconsec_unconfirmed += 1
                    total_disproved += 1
                    if max_consec_unconfirmed and (nconsec_unconfirmed > max_consec_unconfirmed):
                        patience_triggered = True
                        if verbose:
                            logger.info("Maximum consecutive confirmation failures reached.")
                        break  # out of best candidates confirmation, to finish the level

            else:  # next_best_gain = 0
                break  # nothing wrong, just retry all cands evaluation

        # ---------------------------------------------------------------------------------------------------------------
        # Let's act upon results of the permutation test
        # ---------------------------------------------------------------------------------------------------------------

        if cand_confirmed:
            # Per-signal prefer-engineered substitution. By the data-processing
            # inequality a transform E=f(P) cannot out-score its raw parent P on
            # mutual information, so when a raw feature wins a near-tie against
            # its own transform the MI criterion always keeps the raw one -- yet
            # the transform is the representation a shallow downstream actually
            # needs. If the confirmed winner X is raw and a transform of X sits
            # within the prefer-engineered band below it AND independently
            # confirms, select the transform instead (the raw parent then
            # becomes redundant and is marked accepted so it is not reselected).
            _sub_idx, _sub_X, _sub_gain = next_best_candidate_idx, X, next_best_gain
            _child = _confirmable_engineered_child(ctx, X, next_best_candidate_idx, next_best_gain, expected_gains)
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
