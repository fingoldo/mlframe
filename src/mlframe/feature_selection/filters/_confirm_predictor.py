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
import re
from dataclasses import dataclass, field
from timeit import default_timer as timer
from typing import Sequence

import numpy as np
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
)
from .fleuret import get_fleuret_criteria_confidence, get_fleuret_criteria_confidence_parallel
from .gpu import mi_direct_gpu
from .info_theory import merge_vars
from .permutation import mi_direct

logger = logging.getLogger(__name__)


def _conditioning_rows_per_cell(ctx, X: tuple) -> float:
    """Rows-per-nonempty-cell of the conditioning joint ``(X u selected_vars)``.

    RC2 sample-size diagnostic for the Fleuret conditional-MI permutation test.
    The conditional test estimates ``I(X; Y | Z)`` over the joint ``(X, Y, Z)``
    histogram; when that joint is severely undersampled (few rows per occupied
    cell) the plug-in conditional-MI estimate is dominated by finite-sample bias
    and the SHUFFLED-y null conditional MI is ~= the REAL conditional MI, so the
    permutation gate over-rejects every genuine feature after the first
    (proven on sklearn diabetes, n=442, s5 10-bin -> ~0.4 rows/cell).

    We measure the occupied-cell count of the FULL ``(X, Y, selected_vars)``
    joint - the exact support over which the conditional-MI permutation test
    operates (``I(X; Y | Z)`` plug-in needs the H(X,Y,Z) histogram). Counting
    only ``(X, Z)`` is too coarse: on diabetes after MDLP binning the
    ``(bmi, s5)`` joint occupies ~25 cells (~17.7 rows/cell, looks healthy) but
    adding the 10-bin target Y fragments it to ~1000 cells (~0.4 rows/cell) -
    the regime where the shuffled-y null CMI matches the real CMI. ``merge_vars``
    returns ``current_nclasses`` = number of NON-empty bins after pruning, which
    is exactly the occupied-cell count. Returns ``+inf`` when the joint is
    empty/degenerate so the strict conditional path is kept.
    """
    factors_data = ctx.factors_data
    n_samples = len(factors_data)
    if n_samples == 0:
        return float("inf")
    n_cols = factors_data.shape[1] if factors_data.ndim == 2 else 0
    cond_indices = []
    cond_indices.extend(int(c) for c in X)
    # Include the target Y: the conditional-MI estimator histograms (X, Y, Z).
    # In the standard MRMR setup ``targets_data is factors_data`` so y indexes
    # into ``factors_data``; guard the (rare) separate-target-array case by
    # skipping out-of-range y indices (the (X, Z) cell count is then a lower
    # bound on the support, still a valid undersampling signal).
    for yi in ctx.y:
        if 0 <= int(yi) < n_cols:
            cond_indices.append(int(yi))
    for z in ctx.selected_vars:
        if hasattr(z, "__len__"):
            cond_indices.extend(int(c) for c in z)
        else:
            cond_indices.append(int(z))
    try:
        _, _, n_nonempty_cells = merge_vars(
            factors_data=factors_data,
            vars_indices=cond_indices,
            var_is_nominal=None,
            factors_nbins=ctx.factors_nbins,
            dtype=np.int32,
        )
    except Exception:
        return float("inf")
    if n_nonempty_cells <= 0:
        return float("inf")
    return n_samples / float(n_nonempty_cells)


def _candidate_is_engineered(X, factors_names, raw_feature_names) -> bool:
    """True iff candidate ``X`` (single index or k-way tuple of cols-indices) is
    engineered, i.e. at least one of its component columns is NOT a raw input.

    When ``raw_feature_names`` (a set of the original pre-FE column names) is
    provided, a component is engineered iff its ``factors_names`` entry is not in
    that set -- the authoritative test. When it is ``None`` (direct callers that
    don't thread the raw-name set), fall back to the syntactic convention that
    engineered names contain ``(`` (functional forms like ``add(x0,x2)``,
    ``div(sqr(a),b)``) or ``__`` (basis transforms like ``x1__He2``); raw column
    names never contain those tokens in any mlframe FE producer.
    """
    for sub in X:
        try:
            name = factors_names[sub]
        except (IndexError, TypeError, KeyError):
            continue
        if raw_feature_names is not None:
            if name not in raw_feature_names:
                return True
        else:
            if ("(" in name) or ("__" in name):
                return True
    return False


def _prefer_engineered_order(order, expected_gains, ctx) -> np.ndarray:
    """Reorder the descending-gain ``order`` so that, among the leading run of
    candidates whose gain is within ``ctx.prefer_engineered_rel_eps`` (relative)
    of the top gain, engineered candidates precede raw ones.

    Gated and minimal: only the candidates tied (within rel-eps) with the current
    front-runner are eligible to move, and the relative order INSIDE the
    engineered group and inside the raw group is preserved (stable), so a clear
    winner -- one with no engineered peer within eps -- is never displaced. The
    promotion is deterministic and independent of the scoring backend because it
    reads only the (backend-identical) ``expected_gains`` array and the static
    raw-name set. Returns the (possibly) reordered index array.
    """
    rel_eps = float(getattr(ctx, "prefer_engineered_rel_eps", 0.0) or 0.0)
    if rel_eps <= 0.0 or len(order) < 2:
        return order

    gains = np.asarray(expected_gains, dtype=np.float64)
    top_idx = order[0]
    top_gain = gains[top_idx]
    # Only meaningful when the leader carries a positive gain; near-ties at or
    # below zero are noise and must keep the legacy (index) ordering.
    if not (top_gain > 0.0):
        return order

    factors_names = ctx.factors_names
    raw_names = getattr(ctx, "raw_feature_names", None)
    candidates = ctx.candidates
    tol = rel_eps * abs(top_gain)
    # An already-engineered leader stays first: it lands at the head of the
    # stable ``engineered_in_band`` group below, so the order is unchanged.

    leading, engineered_in_band, rest = [], [], []
    seen_band = True
    for pos, cand_idx in enumerate(order):
        if seen_band and (top_gain - gains[cand_idx]) <= tol:
            # Within the relative-eps band of the leader.
            if _candidate_is_engineered(candidates[cand_idx], factors_names, raw_names):
                engineered_in_band.append(cand_idx)
            else:
                leading.append(cand_idx)
        else:
            seen_band = False
            rest.append(cand_idx)

    if not engineered_in_band:
        return order  # no engineered peer in the band -> clear winner, untouched

    # Engineered band-members first (stable), then the raw band-members (stable),
    # then the untouched tail.
    return np.asarray(engineered_in_band + leading + rest, dtype=order.dtype)


# Split an engineered name into identifier-ish sub-tokens. ``__`` is a word char
# so a basis transform ``b__He2`` survives as a single token (its raw parent is
# the part before the FIRST ``__``); functional / cross forms ``div(sqr(a),b)``,
# ``c1*c2`` split on the operator/paren/comma punctuation into their operands.
_PARENT_TOKEN_SPLIT = re.compile(r"[^A-Za-z0-9_]+")


def _extract_single_raw_parent(cand, factors_names, raw_names):
    """Return the raw parent NAME iff candidate ``cand`` (an iterable of column
    indices) is ENGINEERED and is a function of EXACTLY ONE raw input column;
    otherwise ``None``.

    A component column counts as raw when its ``factors_names`` entry is in
    ``raw_names``. An engineered component contributes the raw column(s) it is
    built from: the operands of a functional/cross form (``div(sqr(a),b)`` ->
    {a, b}; ``c1*c2`` -> {c1, c2}) and, for a basis transform ``b__He2``, the
    prefix before the first ``__`` (``b``). Token equality against ``raw_names``
    is EXACT (``x1`` never matches inside ``x10``); the ``__``-prefix strip only
    fires when the whole token is not itself a raw name, so a raw column that
    legitimately contains ``__`` is preserved. Returns the parent only when the
    candidate references a single distinct raw column -- the unambiguous
    "transform of one parent" case the substitution is allowed to act on.
    """
    parents = set()
    is_engineered = False
    for sub in cand:
        try:
            name = factors_names[sub]
        except (IndexError, TypeError, KeyError):
            return None
        if name in raw_names:
            parents.add(name)
            continue
        is_engineered = True
        for tok in _PARENT_TOKEN_SPLIT.split(name):
            if not tok:
                continue
            if tok in raw_names:
                parents.add(tok)
            elif "__" in tok:
                base = tok.split("__", 1)[0]
                if base in raw_names:
                    parents.add(base)
        if len(parents) > 1:
            return None  # multi-parent -> not a single-parent transform
    if not is_engineered or len(parents) != 1:
        return None
    return next(iter(parents))


def _confirmable_engineered_child(ctx, X, winner_idx, winner_gain, expected_gains):
    """When a RAW single-feature candidate ``X`` just won confirmation, find an
    engineered single-parent transform of ``X`` that is within the
    prefer-engineered relative band of the raw winner's gain, and return it as a
    drop-in substitute. Returns ``(child_idx, child_cand, child_gain)`` or
    ``None``.

    Why a substitution and not a reorder: by the data-processing inequality an
    engineered transform ``E = f(P)`` can NEVER carry more mutual information
    about ``y`` than its raw parent ``P`` (``I(E; y) <= I(P; y)`` for any
    binning), so an MI/CMIM criterion ALWAYS scores the raw parent at or above
    its transform and the parent wins every near-tie. Yet ``E`` is strictly
    richer for a shallow downstream model (a linear model can exploit ``b**2-1``
    but not raw ``b`` for a quadratic signal) -- the whole point of the
    engineered feature. ``_prefer_engineered_order`` already encodes this
    preference, but only within the GLOBAL leader band; a secondary-signal pair
    (``b`` vs ``b__He2`` while some unrelated feature leads) never enters that
    band. This applies the SAME preference per-signal at the decision point:
    when the raw parent is the confirmed winner and its transform sits within
    ``prefer_engineered_rel_eps`` below it (the only side DPI allows) and the
    transform independently confirms, prefer the transform.

    Scope is narrow and raw-safe: only fires for a single-feature raw winner
    with a configured ``prefer_engineered_rel_eps``; the transform must reference
    exactly this one raw parent, be unselected/unfailed, and clear the band
    floor. The transform's effective gain is recovered from ``partial_gains``
    when CMIM pruning zeroed its ``expected_gains`` slot. No-op (returns ``None``)
    when prefer-engineered is disabled, no raw-name set is threaded, or no such
    transform exists -- legacy bit-stable.
    """
    raw_names = getattr(ctx, "raw_feature_names", None)
    rel_eps = float(getattr(ctx, "prefer_engineered_rel_eps", 0.0) or 0.0)
    if not raw_names or rel_eps <= 0.0 or len(X) != 1 or not (winner_gain > 0.0):
        return None
    factors_names = ctx.factors_names
    try:
        parent_name = factors_names[X[0]]
    except (IndexError, TypeError, KeyError):
        return None
    if parent_name not in raw_names:
        return None  # the winner is itself engineered -> nothing to substitute

    candidates = ctx.candidates
    partial_gains = getattr(ctx, "partial_gains", None) or {}
    added = ctx.added_candidates or set()
    failed = ctx.failed_candidates or set()
    selected = ctx.selected_vars or []
    band_floor = winner_gain * (1.0 - rel_eps)

    best_child = None  # (gain, idx, cand)
    for idx, cand in enumerate(candidates):
        if idx == winner_idx or idx in added or idx in failed:
            continue
        if any(sub in selected for sub in cand):
            continue
        if _extract_single_raw_parent(cand, factors_names, raw_names) != parent_name:
            continue
        g = float(expected_gains[idx]) if idx < len(expected_gains) else 0.0
        pg = partial_gains.get(idx)
        if pg is not None:
            try:
                g = max(g, float(pg[0]))
            except (TypeError, IndexError, ValueError):
                pass
        if g < band_floor or g >= 1e29:
            continue
        if best_child is None or g > best_child[0]:
            best_child = (g, idx, cand)

    if best_child is None:
        return None
    return best_child[1], best_child[2], best_child[0]


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
    entropy_cache: object
    # --- per-interactions-order / per-node mutable state ---
    candidates: list = None
    # 2026-05-30 Wave 9 — DCD state forwarded into ``should_skip_candidate``
    # for pool_pruned_mask check. ``None`` preserves legacy bit-stable.
    dcd_state: object = None
    interactions_order: int = 1
    selected_vars: list = None
    selected_interactions_vars: list = None
    partial_gains: dict = None
    added_candidates: set = field(default=None)
    failed_candidates: set = field(default=None)
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
            engineered_lineage=engineered_lineage,
            dcd_state=dcd_state,
        )
        if skip:
            continue

        feasible_candidates.append((cand_idx, X, nexisting if reduce_gain_on_subelement_chosen else 0))

    if (
        n_workers > 1
        and (not use_simple_mode or len(cached_MIs) < num_possible_candidates)
        and len(feasible_candidates) > NMAX_NONPARALLEL_ITERS
    ):
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
            get_bur_lambda as _get_bur,
        )
        _su_snapshot = bool(_use_su())
        _jmim_snapshot = bool(_use_jmim())
        _bur_snapshot = float(_get_bur())
        res = workers_pool(
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
                partial_gains=partial_gains,
                baseline_npermutations=baseline_npermutations,
                mrmr_relevance_algo=mrmr_relevance_algo,
                mrmr_redundancy_algo=mrmr_redundancy_algo,
                max_veteranes_interactions_order=max_veteranes_interactions_order,
                selected_vars=selected_vars,
                cached_MIs=cached_MIs,
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
                    global_storage[key] = value

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
        if use_simple_mode and False:
            # No need to check every can out of order: let's just return next best known candidate
            best_gain, best_candidate, run_out_of_time = 1, 1, False
        else:
            for cand_idx, X, nexisting in feasible_candidates:

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
                    cached_cond_MIs=cached_cond_MIs,
                    entropy_cache=entropy_cache,
                    verbose=verbose,
                    ndigits=ndigits,
                    use_simple_mode=use_simple_mode,
                )

                best_gain, best_candidate, run_out_of_time = handle_best_candidate(
                    current_gain=current_gain,
                    best_gain=best_gain,
                    X=X,
                    best_candidate=best_candidate,
                    factors_names=factors_names,
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
            if use_gpu:
                bootstrapped_gain, confidence = mi_direct_gpu(
                    factors_data,
                    x=X,
                    y=y,
                    factors_nbins=factors_nbins,
                    classes_y=classes_y,
                    freqs_y=freqs_y,
                    freqs_y_safe=freqs_y_safe,
                    classes_y_safe=classes_y_safe,
                    min_nonzero_confidence=min_nonzero_confidence,
                    npermutations=full_npermutations,
                )
            else:
                if verbose and len(selected_vars) < MAX_ITERATIONS_TO_TRACK:
                    eval_start = timer()
                bootstrapped_gain, confidence = mi_direct(
                    factors_data,
                    x=X,
                    y=y,
                    factors_nbins=factors_nbins,
                    classes_y=classes_y,
                    freqs_y=freqs_y,
                    classes_y_safe=classes_y_safe,
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
                            "confirm %s: conditioning joint undersampled (%.2f rows/cell < %.2f); "
                            "marginal-MI fallback (conf=%.*f)",
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
        _gain_order = np.lexsort((np.arange(len(expected_gains)), -np.asarray(expected_gains)))
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
                    )

                    if best_partial_gain > next_best_gain:
                        best_gain = next_best_gain
                        if verbose > 2:
                            print(
                                "Have no best_candidate anymore. Need to recompute partial gains. best_partial_gain of candidate",
                                get_candidate_name(candidates[best_key], factors_names=factors_names),
                                "was",
                                best_partial_gain,
                            )
                        break  # out of best candidates confirmation, to retry all cands evaluation

                any_cand_considered = True

                if full_npermutations:
                    total_checked += 1

                bootstrapped_gain, confidence = confirm_candidate(ctx, X, next_best_gain)

                if bootstrapped_gain > 0:

                    nconsec_unconfirmed = 0

                    next_best_gain = next_best_gain * confidence
                    expected_gains[next_best_candidate_idx] = next_best_gain

                    best_partial_gain, best_key = find_best_partial_gain(
                        partial_gains=partial_gains,
                        failed_candidates=failed_candidates,
                        added_candidates=added_candidates,
                        candidates=candidates,
                        selected_vars=selected_vars,
                        skip_indices=(next_best_candidate_idx,),
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
