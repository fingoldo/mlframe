"""Pure helpers of ``_confirm_predictor.confirm_candidate``: the GPU/CPU routing of the marginal confirmation and the Fleuret conditional recheck."""
from __future__ import annotations

import logging
from timeit import default_timer as timer
from typing import Any, Callable

import numpy as np
from joblib import Parallel

from ._confirm_predictor_engineered import _conditioning_rows_per_cell
from ._internals import MAX_ITERATIONS_TO_TRACK, NMAX_NONPARALLEL_ITERS
from ._numba_utils import count_cand_nbins, get_fleuret_criteria_confidence
from .evaluation import get_candidate_name
from .fleuret import get_fleuret_criteria_confidence_parallel
from .permutation import _addone_pvalue_enabled

logger = logging.getLogger("mlframe.feature_selection.filters._confirm_predictor")


def marginal_confirm_use_gpu(use_gpu: bool, factors_data: np.ndarray) -> bool:
    """Whether the marginal permutation confirmation should run on GPU: ``use_gpu`` unless the GPU is globally off or the analytic null engages."""
    _confirm_use_gpu = use_gpu
    if _confirm_use_gpu:
        try:
            import os as _os
            from ._analytic_mi_null import analytic_null_min_n, analytic_null_enabled
            _cvd = _os.environ.get("CUDA_VISIBLE_DEVICES", None)
            _gpu_disabled_env = _os.environ.get("MLFRAME_DISABLE_GPU", "") == "1"
            _gpu_hidden_env = _cvd is not None and _cvd.strip() == ""
            _gpu_off = False
            if _gpu_disabled_env:
                _gpu_off = True
            elif _gpu_hidden_env:
                _gpu_off = True
            _analytic_n = analytic_null_enabled() and int(factors_data.shape[0]) >= analytic_null_min_n()
            if _gpu_off or _analytic_n:
                _confirm_use_gpu = False
        except Exception as e:  # nosec B110 - swallow converted to debug-log, non-fatal by design
            logger.debug("suppressed: %s", e)
            pass
    return _confirm_use_gpu


def fleuret_conditional_recheck(ctx: Any, X: tuple, next_best_gain: float, bootstrapped_gain: float, confidence: float) -> tuple:
    """Fleuret conditional permutation recheck of a candidate that passed the marginal bootstrap; returns the (possibly zeroed) ``(bootstrapped_gain, confidence)``."""
    full_npermutations = ctx.full_npermutations
    selected_vars = ctx.selected_vars
    factors_nbins = ctx.factors_nbins
    factors_names = ctx.factors_names
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
                    logger.debug(
                        "confirm %s: conditioning joint undersampled (%.2f rows/cell < %.2f); " "marginal-MI fallback (conf=%.*f)",
                        get_candidate_name(X, factors_names=factors_names),
                        _rows_per_cell, _undersample_threshold, ndigits, confidence,
                    )
                # Keep the marginal bootstrap verdict; do not run the
                # unreliable conditional permutation gate.
                return bootstrapped_gain, confidence

        # fold the candidate's own identity (``hash(X)``) into the seed -
        # pre-fix this only depended on (random_seed, len(selected_vars)), so every distinct candidate
        # confirmed at the same selected_vars depth within one greedy round drew the IDENTICAL
        # permutation stream for the Fleuret conditional recheck.
        _fleuret_base_seed = int(((int(random_seed or 0) * 2654435761) + len(selected_vars) + 1 + hash(X)) & 0xFFFFFFFFFFFFFFFF)
        if n_workers and n_workers > 1 and full_npermutations > NMAX_NONPARALLEL_ITERS:
            # Build the Fleuret parallel pool ONCE per
            # screen round and cache it on ``ctx`` - pre-fix, ``workers_pool`` was always None here
            # (deliberately, per the RETIRED note above, for the UNRELATED evaluate_candidates path),
            # so ``get_fleuret_criteria_confidence_parallel`` rebuilt a fresh joblib.Parallel pool on
            # EVERY candidate needing the conditional recheck, even though the same ctx/round can
            # confirm dozens of candidates. Reusing one pool across the round amortises the
            # construction cost while every other candidate/round still gets a fresh pool as before.
            if workers_pool is None:
                workers_pool = Parallel(n_jobs=n_workers, **(parallel_kwargs or {}))
                ctx.workers_pool = workers_pool
            bootstrapped_gain, confidence, parallel_entropy_cache = get_fleuret_criteria_confidence_parallel(
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
                n_workers=n_workers,
                workers_pool=workers_pool,
                parallel_kwargs=parallel_kwargs,
                base_seed=_fleuret_base_seed,
            )
            for key, value in parallel_entropy_cache.items():
                entropy_cache[key] = value
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
            # below NMAX_NONPARALLEL_ITERS) where the parallel path is never taken - so there is no serial-vs-
            # parallel comparison to reconcile here. The add-one estimator's ceiling at budget 3 is 1-1/4=0.75,
            # which lowers gain*confidence ~25% below min_relevance_gain and STARVES the screen (a prior C4
            # attempt to add-one this path regressed cluster-aggregate / FE recovery). Keep the raw rate: a clean
            # null (nfailed=0) correctly yields confidence 1.0 so genuine signals clear the small-budget screen.
            confidence = 1 - nfailed / nchecked
            if nfailed >= max_failed:
                bootstrapped_gain = 0.0

        if verbose and len(selected_vars) < MAX_ITERATIONS_TO_TRACK:
            logger.debug("get_fleuret_criteria_confidence bootstrapped eval took %.1f sec.", timer() - eval_start)
    return bootstrapped_gain, confidence


def confirm_in_gain_order(
    ctx: Any,
    _gain_order: np.ndarray,
    expected_gains: np.ndarray,
    best_gain: float,
    nconsec_unconfirmed: int,
    total_checked: int,
    total_disproved: int,
    patience_triggered: bool,
    confidence: float,
    confirm_candidate: Callable[..., Any],
    find_best_partial_gain: Callable[..., Any],
    get_candidate_name: Callable[..., Any],
) -> tuple:
    """Walk the candidates in descending expected gain and permutation-confirm the first that holds (with the partial-gain retry and patience accounting).

    ``confirm_candidate`` / ``find_best_partial_gain`` / ``get_candidate_name`` are handed in by ``_confirm_predictor`` so patching them there keeps working.
    Returns ``(best_gain, cand_confirmed, any_cand_considered, nconsec_unconfirmed, total_checked, total_disproved, patience_triggered,
    next_best_candidate_idx, X, next_best_gain, confidence)``.
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
    next_best_gain: Any = None
    next_best_candidate_idx: Any = None
    X: Any = None
    cand_confirmed = False
    any_cand_considered = False

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
                # significance), whose ceiling at a clean null is ``budget/(budget+1)`` - only 0.75 at
                # the screen's tiny default budget (full_npermutations=3). Multiplying the gain by that
                # ceiling deflates a perfectly clean signal ~25%, dropping it below ``min_relevance_gain``
                # and starving the FE pair-pool (engineered features collapse to ~0). Dividing by the
                # add-one ceiling exactly recovers the raw exceedance-rate confidence (1.0 at a clean
                # null) the screen used pre-add-one, so genuine signals clear the small-budget screen
                # while the calibrated add-one confidence is still what gets STORED/reported. No-op when
                # add-one is disabled (ceiling == 1.0) and on large budgets (ceiling -> 1.0).
                _conf_for_gain = confidence
                _conf_for_gain = _confirm_one_predic_add_one_disabled_ceiling(full_npermutations, confidence, _conf_for_gain)
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
    return (
        best_gain, cand_confirmed, any_cand_considered, nconsec_unconfirmed, total_checked, total_disproved, patience_triggered,
        next_best_candidate_idx, X, next_best_gain, confidence,
    )


def _confirm_one_predic_add_one_disabled_ceiling(full_npermutations, confidence, _conf_for_gain):
    """Block of confirm_one_predictor starting at ``if _addone_pvalue_enabled() and full_npermutations:``."""
    if _addone_pvalue_enabled() and full_npermutations:
        _addone_ceiling = full_npermutations / (full_npermutations + 1.0)
        if _addone_ceiling > 0.0:
            _conf_for_gain = min(1.0, confidence / _addone_ceiling)
    return _conf_for_gain
