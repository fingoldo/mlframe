"""``screen_predictors`` carved out of
``mlframe.feature_selection.filters.screen``.

Re-imported at the parent's module bottom so historical
``from mlframe.feature_selection.filters.screen import screen_predictors``
resolves transparently.
"""

from __future__ import annotations

import logging
from timeit import default_timer as timer
from typing import TYPE_CHECKING, Callable, Optional
from collections.abc import Iterable, Sequence

import numba
import numpy as np
from mlframe.feature_selection.filters._fe_usability_signal import _crit_np_dtype
from joblib import Parallel
from numba.core import types

from pyutilz.numbalib import set_numba_random_seed

from ._internals import MAX_CONFIRMATION_CAND_NBINS, MAX_JOBLIB_NBYTES
from ._confirm_predictor import ScreenContext
from ._screen_predictors_gate import build_dcd_state
from ._screen_predictors_prescreen import cardinality_prescreen, compute_fdr_gain_floor
from .info_theory import merge_vars
from ._numba_warmup import warmup_typed_dict
from ._doc_attach import set_doc
from ._screen_predictors_doc import SCREEN_PREDICTORS_DOC
from ._screen_predictors_step1 import _screen_predictors_step1_array_returned_encodings

if TYPE_CHECKING:
    from ._dynamic_cluster_discovery import DCDState

from types import SimpleNamespace as _SimpleNamespace

logger = logging.getLogger(__name__)

# The typed.Dict(unicode->float64) warm-up used to run HERE, at import. That made every process which merely imports
# this module transitively - each loky FE worker, and anything importing mlframe.preprocessing or mlframe.reporting -
# pay it without ever calling ``screen_predictors``: measured 36.4 s per worker to import
# ``filters.polynom_pair_fe`` in four workers, against 19.x s with MLFRAME_SKIP_NUMBA_WARMUP=1, and loky respawns
# workers after its idle timeout. It now runs on the first ``screen_predictors`` call instead, where it is actually
# needed; the per-process saving for a screening process is unchanged (the warm-up is idempotent and still happens
# before any fit work). Opt out with MLFRAME_SKIP_NUMBA_WARMUP=1. See _numba_warmup for the full rationale.


# The interaction-order enumeration step lives in the sibling module (module size limit), re-exported here.
from ._screen_predictors_steps import _screen_predictors_step2_interactions_order_subsets, _short_name  # noqa: F401


def _pool_warmup_noop(i):
    """Module-level no-op handed to the joblib pool so worker spawn cost is paid before
    the screening loop starts. Must be module-level (not a closure) so the ``loky`` backend
    can pickle it. Mirrors ``screen._pool_warmup_noop``; defined here too because the warmup
    call lives in this module's ``screen_predictors`` (pre-2026-05 this reference was an
    unbound ``NameError`` on the ``n_workers>1`` path)."""
    return None


def _init_screen_caches(seed_caches):
    """Seed the screen caches from the prior round when supplied, else build empty typed dicts; returns the four caches."""
    if seed_caches is not None:
        entropy_cache, cached_MIs, cached_confident_MIs, cached_cond_MIs = seed_caches
    else:
        cached_MIs = dict()
        cached_confident_MIs = dict()
        # PERF NOTE (profiled 2026-07): constructing the first numba typed.Dict
        # (unicode->float64) in a process JIT-compiles the whole typed-dict method
        # suite - ~5s of LLVM codegen (27% of a cold F2 fit wall), recurring per
        # process and NOT numba-disk-cacheable. That machinery is now warmed
        # synchronously at import (warmup_typed_dict, above), moving the ~5s off
        # every fit's critical path (fit 42.9s -> 37.9s; import +5.2s). A background
        # daemon-thread warm-up was tried and rejected first: numba's global compile
        # lock serialises it against the fit's own njit compilation and this
        # host-serial pipeline has no concurrent GPU work to overlap it with.
        # The warm-up only SHIFTS the cost (per-process, amortises to zero across
        # fits in a long-running process); ELIMINATING it would need replacing the
        # unicode-keyed caches below with a 128-bit-hash (UniTuple int64) key - a
        # ~7-file core rewrite gated on the selection-equivalence contract, deferred
        # as poor ROI (saves ~5s, already zero for multi-fit processes).
        entropy_cache = numba.typed.Dict.empty(
            key_type=types.unicode_type,
            value_type=types.float64,  # Intentional float64: numba typed-dict value type must match the float64 values the kernel stores
        )
        cached_cond_MIs = numba.typed.Dict.empty(
            key_type=types.unicode_type,
            value_type=types.float64,  # Intentional float64: numba typed-dict value type must match the float64 values the kernel stores
        )
    return entropy_cache, cached_MIs, cached_confident_MIs, cached_cond_MIs


# ``classes_y_safe`` is OVERLOADED across two consumers with INCOMPATIBLE
# array-backend requirements, so we keep a CPU numpy buffer ALWAYS and a
# separate CuPy device buffer only when GPU is on:
#   * the FE step (returned ``classes_y_safe`` -> check_prospective_fe_pairs
#     -> the batched MI noise-gate) runs njit kernels (CPU fallback AND the
#     bit-identical GPU twin, which does ``np.asarray(classes_y_safe)`` +
#     a numba Fisher-Yates shuffle) -> it REQUIRES a numpy host array.
#   * the confirm / baseline-eval GPU branches (``mi_direct_gpu``) want the
#     pre-warmed CuPy DEVICE buffer to skip the H2D copy.
# Pre-the GPU branch reassigned the single ``classes_y_safe``
# local to a CuPy array and returned THAT to the FE step, so any
# ``MRMR(use_gpu=True)`` fit that reached the FE pair-search crashed with
# ``TypingError: Cannot determine Numba type of <class 'cupy.ndarray'>``
# (or ``np.asarray`` on a cupy array raising in the subsample path). Now the
# numpy buffer survives for the return; the device buffer only goes to ctx.
#
# RESOLVED (7-site joblib.Parallel audit - follow-up to the
# UNRESOLVED note this comment used to carry): the "releases the GIL" claim below was never
# actually confirmed for THIS call path, and the isolated/warmed/best-of-3+ A/B settles it in the
# other direction. Measured at ``evaluate_candidates``'s realistic scales: m=10 candidates ->
# 0.03x (SLOWER than serial), m=320 (wellbore-scale) -> 0.72-0.73x, m=820/n_workers=8 -> 0.81x -
# this ``Parallel(backend="threading")`` pool NEVER wins over serial at any tested scale, confirming
# the pool is GIL-bound at the dispatch boundary exactly as suspected (evaluate_candidates
# eventually calls njit kernels that DO release the GIL, but the per-call joblib dispatch/aggregation
# overhead dominates at these candidate counts). The pool is therefore never constructed here anymore
# - ``evaluate_candidates`` always runs on the serial fast path in ``confirm_one_predictor``/
# ``_confirm_predictor.py``. ``n_workers`` is kept as an accepted parameter (other call sites, e.g.
# the Fleuret conditional-confirmation gate, still branch on it) but no longer causes a pool to be
# built here; ``seed_workers_pool`` is likewise accepted-but-unused for this purpose.
def _resident_screen_targets(use_gpu, classes_y, freqs_y, classes_y_safe):
    """Return ``(classes_y_safe, freqs_y_safe)`` for the screen: a shuffle-safe resident device copy of the target codes on GPU, else the host arrays."""
    if use_gpu:
        # RESIDENT UPLOAD: ``screen_predictors`` runs 3-10 rounds per fit with an IDENTICAL
        # ``classes_y``/``freqs_y`` across rounds (the target never changes mid-fit; ``seed_maxt_floor_cache``
        # / ``seed_workers_pool`` above already thread other round-carried state through this same call, so
        # the pattern for reusing state across rounds is established) - resident_operand needs no caller-side
        # plumbing beyond this call site since it self-caches by content, unlike those explicit seed_* kwargs.
        #
        # ``.copy()`` on ``classes_y_safe`` is NOT optional: ``ctx.classes_y_safe`` reaches
        # ``gpu.py:mi_direct_gpu``'s pre-warmed-buffer path, which SHUFFLES it IN PLACE
        # (``classes_y_safe[:] = classes_y_safe[cp.argsort(...)]``) once per permutation, for every
        # candidate confirmed this round. Pre-fix that was harmless - ``classes_y_safe`` was a fresh
        # disposable ``cp.asarray`` built new every ``screen_predictors()`` call, never read by anyone
        # else. Handing the RESIDENT (shared, content-keyed, cross-call, cross-file) buffer straight into
        # that mutator would let one candidate's permutation shuffle permanently corrupt the row-order
        # every OTHER resident-cache consumer of the same target content relies on (e.g.
        # ``_gpu_batched.py``'s ``classes_y_gpu``, which assumes its rows stay aligned with
        # ``classes_x_gpu``) - a real cross-file aliasing regression this exact copy is NOT theoretical
        # (it was live: 4 GPU/CPU selection-equivalence tests hit it before this ``.copy()`` was added).
        # ``.copy()`` is a cheap device-to-device memcpy (not a host re-upload), so the H2D-avoidance win
        # from ``resident_operand`` is preserved; only the shuffle-safety copy is paid, once per round.
        # ``freqs_y_safe`` is never mutated in place anywhere in this package (verified by grep), so it is
        # shared directly with no copy.
        from ._fe_resident_operands import resident_operand

        classes_y_safe = resident_operand(classes_y, "screen_classes_y", dtype=np.int32).copy()  # device buffer for ctx -> mi_direct_gpu
        freqs_y_safe = resident_operand(freqs_y, "screen_freqs_y", dtype=_crit_np_dtype())
    else:
        freqs_y_safe = None
    return classes_y_safe, freqs_y_safe


def _seed_screen_rngs(st: _SimpleNamespace, random_seed: int | None, use_gpu: bool) -> None:
    """Capture fresh restore seeds on ``st`` and seed numba (and cupy when ``use_gpu``) for the duration of the screen; no-op when ``random_seed`` is None."""
    if random_seed is not None:
        # Capture a fresh entropy-derived seed to restore numba/cupy with on
        # finally; mathematically equivalent (from the consumer's view) to
        # "the seed they would have had if no inner seed call had fired".
        # _fresh_seed masks the draw to 32 bits: a raw 64-bit draw is >= 2**63 about half the time, which numba's int64 seed argument
        # rejects (and numpy's legacy seeding under NUMBA_DISABLE_JIT=1 rejects anything >= 2**32); the restore below swallows that
        # into a debug log, so the stream would silently stay where this screen left it.
        from mlframe.utils.shared import fresh_seed as _fresh_seed

        st._numba_restore_seed = _fresh_seed()
        # Only capture cupy restore-seed when GPU path is actually requested.
        # Otherwise the finally block below would import cupy purely to
        # "restore" a seed that was never set, triggering cupy's Windows
        # DLL-load _diagnose_import_error recursion on hosts where cupy
        # is installed but its CUDA stack is broken (cuTENSOR/cuBLAS
        # missing) - that recursion blows the C stack in batch pytest
        # contexts and tears down the test runner.
        if use_gpu:
            st._cp_restore_seed = _fresh_seed()
        set_numba_random_seed(random_seed)
        # The prior
        # ``try: cp.random.seed(random_seed); except NameError: pass``
        # always swallowed NameError because ``cp`` is only imported
        # inside ``if use_gpu:`` ~80 lines down. So the documented
        # "seed it (numpy + numba + cupy) for the screening duration"
        # was a dead path - CuPy RNG was NEVER seeded, breaking GPU
        # reproducibility. Import cupy here when seeding is requested
        # (regardless of use_gpu) so the seed call actually fires.
        if use_gpu:
            try:
                import cupy as cp

                cp.random.seed(random_seed)
            except Exception as e:  # nosec B110 - non-trivial body
                logger.debug("cupy RNG seeding failed: %s", e)
                # CuPy absent -> CPU fallback below. Also tolerate the legacy global cuRAND host generator
                # failing to init (CURAND_STATUS_INITIALIZATION_FAILED on some driver/lib combos): seeding
                # it is best-effort reproducibility and the GPU kernels use the modern Generator API anyway,
                # so a seed failure must not crash the whole screen.
                pass


def _init_jmim_cache() -> tuple:
    """Return the JMIM joint-MI cache (typed dict) and its int64 hit counter.

    Built once per fit alongside ``cached_cond_MIs`` so it persists across greedy rounds (the {X} u Z multiset key recurs as selected_vars
    grows). The plain int64 array counts cache HITS for observability. Both are forwarded through ScreenContext to evaluate_candidate; never
    pickled onto the instance.
    """
    cached_jmim_MIs = numba.typed.Dict.empty(
        key_type=types.unicode_type,
        value_type=types.float64,  # Intentional float64: numba typed-dict value type must match the float64 values the kernel stores
    )
    return cached_jmim_MIs, np.zeros(1, dtype=np.int64)


def _screen_fdr_floor(
    factors_data: np.ndarray,
    factors_nbins: Sequence[int],
    factors_names: Sequence[str],
    st: _SimpleNamespace,
    y: Sequence[int],
    subsample_idx: np.ndarray | None,
    screen_full_factors: np.ndarray | None,
    cardinality_bias_correction: bool,
    raw_cardinality_cols: set | None,
    fdr_kwargs: dict,
    random_seed: int | None,
    verbose: int,
    seed_maxt_floor_cache: dict | None,
) -> float:
    """Run the cardinality pre-screen (updating ``st.x``) and return the maxT permutation-null gain floor for the finalised order-1 pool."""
    # Cardinality-bias pre-screen: drop columns whose Miller-Madow plug-in-MI bias is too large to score honestly (nbins_x > 2*sqrt(n)). See ``cardinality_prescreen``.
    if cardinality_bias_correction and factors_data.shape[1] > 0:
        st.x, _cardinality_refused_cols = cardinality_prescreen(
            factors_data,
            factors_nbins,
            factors_names,
            st.x,
            y,
            verbose,
            raw_cardinality_cols=raw_cardinality_cols,
        )
    else:
        _cardinality_refused_cols = set()

    # — Westfall-Young maxT permutation-null gain floor (computed ONCE on the finalised order-1 pool, applied at the selection gate). Fires on a WIDE pool
    # (>= ``screen_fdr_min_features``, where best-of-p selection bias dominates) OR on a NARROW pool that meets the target-over-split gate - an MDLP-over-split heavy-tailed
    # regression target whose plug-in MI bias lifts pure-noise columns past the gain floors after the signals are picked. The narrow-pool gate is itself self-gating: it only
    # engages where the (X,y) joint table is dense enough that the floor is reliable, so a dense weak-signal pool (diabetes) keeps the floor OFF. Below both gates the floor
    # is 0.0 (no-op) so the clean low-cardinality tabular suite is untouched. See ``_permutation_null.py``.
    return compute_fdr_gain_floor(
        factors_data,
        np.asarray(factors_nbins),
        st.x,
        y,
        honest_full_factors=screen_full_factors,
        honest_eval_rows=(np.asarray(subsample_idx, dtype=np.int64) if screen_full_factors is not None else None),
        cardinality_bias_correction=cardinality_bias_correction,
        random_seed=random_seed,
        verbose=verbose,
        maxt_floor_cache=seed_maxt_floor_cache,
        **fdr_kwargs,
    )


def _log_screen_termination(ctx: ScreenContext, selected_vars: list, patience_triggered: bool, max_consec_unconfirmed: int) -> None:
    """Emit the always-on termination-reason summary line for ``screen_predictors``."""
    # Termination-reason summary (always emitted). ``patience_triggered`` distinguishes "gave up confirming" (max_consec_unconfirmed hit) from natural exhaustion below
    # ``min_relevance_gain``. If patience keeps tripping, raise ``max_consec_unconfirmed`` or smooth the relevance signals.
    # Total wall time inside score_candidates + call count + final |Z| appended to the
    # SAME existing always-emitted summary line (cheaper than a whole new logging mechanism) -
    # this is exactly the trio (pool_size x |Z| scaling driver, not the FE-pair-precompute step
    # logged elsewhere) a prior investigation had to reconstruct from code instead of the log.
    _sc_summary = f" [score_candidates: {ctx.sc_wall:.1f}s over {ctx.sc_calls} call(s), final |Z|={len(selected_vars)}]"
    if patience_triggered:
        logger.warning(
            "screen_predictors terminated early via max_consec_unconfirmed=%d "
            "patience (at least one level exhausted). Returned %d selected "
            "feature(s). If you expected more, increase max_consec_unconfirmed "
            "or reduce the relevance-gain threshold.%s",
            max_consec_unconfirmed,
            len(selected_vars),
            _sc_summary,
        )
    else:
        logger.info(
            "screen_predictors finished naturally (no patience trip). " "Returned %d selected feature(s).%s",
            len(selected_vars),
            _sc_summary,
        )


def _influencing_vars(cached_confident_MIs: dict) -> set:
    """Return the union of variable indices from every confidently-scored combination whose bootstrapped gain is positive."""
    any_influencing: set = set()
    for vars_combination, (bootstrapped_gain, _confidence) in cached_confident_MIs.items():
        if bootstrapped_gain > 0:
            any_influencing.update(set(vars_combination))

    """Выбрать группы/кластера скоррелированных факторов. Вместо использования 1 самого крутого, рассмотреть средние от всех
        отброшенных факторов, имеющих высокое прямое direct_MI с таргетом, но близкое к 0 additional_knowledge с каждым
        "победившим" фактором, проверить, не могут ли они усилить свой победивший фактор через ансамблирование. Усиление в смысле
        среднего и вариативности MI с таргетом на бутстрепе подвыборок?

        key = arr2str(X) + "_" + arr2str(Z)
        if key in cached_cond_MIs:
            additional_knowledge = cached_cond_MIs[key]
    """
    return any_influencing


# Notes on the locals of ``screen_predictors`` (patience flag, cache seeding, DCD state, subsample, data_copy aliasing, ScreenContext), in code order.
# True if inner confirmation loop hit ``max_consec_unconfirmed`` patience at least once. Surfaced at function exit so callers can distinguish "gave up confirming"
# from "natural gain threshold reached".
# Seed from the PRIOR screen round's caches when supplied (see seed_caches in the parameter notes) instead of always starting empty - a cache key's value
# is a deterministic function of the (stable, append-only) data, so reuse across rounds is exact, not approximate.
# Dynamic Cluster Discovery state construction.
# Built only when ``dcd_config`` is provided AND ``enable=True``. The
# state lives on the screen-local namespace (NOT thread-local) so the
# joblib parallel-backend path remains safe per Critic1/F fix.
# DCD state construction (config parsing + make_dcd_state call + fallback-on-error)
# lives in ``build_dcd_state`` (``_screen_predictors_gate.py``).
# SCREEN SUBSAMPLE ("score on the shared rows"): slice the working factors/targets to the fit's
# one shared draw so the relevance sweep + maxT FDR floor below run on ~30k rows (consistent with
# the candidates they gate; no full-n permutation work). ``_screen_full_factors`` keeps the full
# array so the RETURNED encodings are recomputed at full n (row-aligned for the FE pipeline).
# Mutate-and-restore instead of a whole-matrix copy: ``data_copy`` aliases ``factors_data`` and the Fleuret permutation njit
# (``get_fleuret_criteria_confidence``) saves+restores ONLY the few columns it shuffles (x u y, ~O(n) each), so peak extra RAM is
# O((|x|+|y|)*n) not O(p*n). Frames here can be 100+ GB; the per-screen-call full copy doubled peak RAM. The parallel confirm path
# already copies per-worker, so it never mutated this buffer - the serial path now matches that pristine-start semantics.
# Shared confirmation context. Static fields + the four MI caches are set once; per-interactions-order fields
# (``candidates`` / ``partial_gains`` / ``added_candidates`` / ``failed_candidates`` / ``interactions_order`` /
# ``num_possible_candidates``) are refreshed at the top of each order below. ``selected_vars`` /
# ``selected_interactions_vars`` are the same list objects throughout (appended to, never reassigned), so the
# confirmation primitives always see the up-to-date selection.
@set_doc(SCREEN_PREDICTORS_DOC)
def screen_predictors(
    factors_data: np.ndarray, factors_nbins: Sequence[int], factors_names: Sequence[str] | None = None,
    factors_names_to_use: Sequence[str] | None = None, factors_to_use: Sequence[int] | None = None, targets_data: np.ndarray | None = None,
    targets_nbins: Sequence[int] | None = None, y: Sequence[int] | None = None, mrmr_relevance_algo: str = "fleuret",
    mrmr_redundancy_algo: str = "fleuret", reduce_gain_on_subelement_chosen: bool = True, extra_x_shuffling: bool = True, dtype: type = np.int32,
    random_seed: int | None = None, subsample_idx: np.ndarray | None = None, use_gpu: bool = False, n_workers: int = 1,
    min_occupancy: int | None = None, min_nonzero_confidence: float = 0.99, full_npermutations: int = 3, baseline_npermutations: int = 2,
    fe_confirm_undersample_rows_per_cell: float = 5.0, min_relevance_gain: float = 0.0001, min_relevance_gain_relative_to_first: float = 0.05,
    cardinality_bias_correction: bool = True, raw_cardinality_cols: set | None = None, max_consec_unconfirmed: int = 10,
    max_runtime_mins: float | None = None, interactions_min_order: int = 1, interactions_max_order: int = 1,
    interactions_order_reversed: bool = False, max_veteranes_interactions_order: int = 1, only_unknown_interactions: bool = False,
    max_confirmation_cand_nbins: int | None = None, fe_fallback_to_all: bool = False, verbose: int = 0, ndigits: int = 5,
    parallel_kwargs: dict | None = None, stop_file: str | None = None, use_simple_mode: bool = False, engineered_lineage: dict | None = None,
    dcd_config: dict | None = None, engineered_recipes: dict | None = None, raw_feature_names: Optional[Iterable[str]] = None,
    prefer_engineered_rel_eps: float = 0.15, screen_fdr_null_permutations: int = 999, screen_fdr_null_quantile: float = 0.975,
    honest_null_codes_fn: "Callable | None" = None, screen_fdr_min_features: int = 30, screen_fdr_target_oversplit_ratio: float = 1.0,
    screen_fdr_min_rows_per_joint_cell: float = 7.0, existing_dcd_state: "DCDState | None" = None, seed_caches: "tuple | None" = None,
    seed_maxt_floor_cache: "dict | None" = None, seed_workers_pool: "Parallel | None" = None,
) -> tuple:
    """Finds best predictors for the target; the full documentation is attached by ``set_doc``."""
    factors_names, max_confirmation_cand_nbins, parallel_kwargs, st, targets_data = _screen_predictors_step1_block(
        parallel_kwargs, max_confirmation_cand_nbins, mrmr_relevance_algo, mrmr_redundancy_algo, y, factors_data, targets_data, targets_nbins, factors_nbins, factors_names
    )
    _screen_predictors_step2_initialize_factor_indices(factors_to_use, st, factors_names_to_use, factors_names, factors_data, targets_data, verbose, y)
    assert y is not None  # _screen_predictors_step1_block rejects y=None before this point
    _seed_screen_rngs(st, random_seed, use_gpu)

    try:
        max_failed = int(full_npermutations * (1 - min_nonzero_confidence))
        if max_failed <= 1:
            max_failed = 1

        selected_interactions_vars: list = []
        selected_vars: list = []  # stores just indices. can't use set 'cause the order is important for efficient computing
        predictors: list = []  # stores more details.

        patience_triggered: bool = False

        entropy_cache, cached_MIs, cached_confident_MIs, cached_cond_MIs = _init_screen_caches(seed_caches)
        cached_jmim_MIs, jmim_hit_counter = _init_jmim_cache()

        dcd_state = build_dcd_state(dcd_config, factors_data, factors_nbins, factors_names, y, existing_dcd_state, verbose)

        _screen_full_factors = None
        _screen_full_factors, factors_data = _screen_predictors_step1_array_returned_encodings(subsample_idx, factors_data, targets_data, _screen_full_factors)

        data_copy = factors_data

        classes_y, freqs_y, _ = merge_vars(factors_data=factors_data, vars_indices=y, var_is_nominal=None, factors_nbins=factors_nbins, dtype=dtype)
        classes_y_safe = classes_y.copy()

        _fdr_gain_floor = _screen_fdr_floor(
            factors_data, factors_nbins, factors_names, st, y, subsample_idx, _screen_full_factors, cardinality_bias_correction, raw_cardinality_cols,
            dict(
                screen_fdr_null_permutations=screen_fdr_null_permutations,
                screen_fdr_null_quantile=screen_fdr_null_quantile,
                honest_null_codes_fn=honest_null_codes_fn,
                screen_fdr_min_features=screen_fdr_min_features,
                screen_fdr_target_oversplit_ratio=screen_fdr_target_oversplit_ratio,
                screen_fdr_min_rows_per_joint_cell=screen_fdr_min_rows_per_joint_cell,
            ),
            random_seed, verbose, seed_maxt_floor_cache,
        )

        # ``classes_y_safe`` is kept as a CPU numpy buffer (``classes_y_safe_host``, returned to the FE step) while the confirm GPU branches get a device copy;
        # see the note above ``_resident_screen_targets``.
        classes_y_safe_host = classes_y_safe  # numpy, returned to the FE step
        classes_y_safe, freqs_y_safe = _resident_screen_targets(use_gpu, classes_y, freqs_y, classes_y_safe)

        # The joblib pool is never constructed here: see the note above ``_resident_screen_targets`` on why ``evaluate_candidates`` always runs serially.
        workers_pool = None

        ctx = ScreenContext(
            factors_data=factors_data, factors_nbins=factors_nbins, factors_names=factors_names, y=y, data_copy=data_copy,
            classes_y=classes_y, classes_y_safe=classes_y_safe, freqs_y=freqs_y, freqs_y_safe=freqs_y_safe,
            mrmr_relevance_algo=mrmr_relevance_algo, mrmr_redundancy_algo=mrmr_redundancy_algo,
            reduce_gain_on_subelement_chosen=reduce_gain_on_subelement_chosen,
            max_veteranes_interactions_order=max_veteranes_interactions_order, only_unknown_interactions=only_unknown_interactions,
            use_gpu=use_gpu, use_simple_mode=use_simple_mode, extra_x_shuffling=extra_x_shuffling, engineered_lineage=engineered_lineage,
            # — directed-FE tie-break inputs. Normalize the raw-name
            # collection to a set for O(1) membership; ``None`` stays ``None`` so
            # the confirm primitive falls back to the syntactic heuristic.
            raw_feature_names=(set(raw_feature_names) if raw_feature_names is not None else None),
            prefer_engineered_rel_eps=float(prefer_engineered_rel_eps or 0.0),
            dcd_state=dcd_state,  # forward DCD state into confirm context
            n_workers=n_workers, workers_pool=workers_pool, parallel_kwargs=parallel_kwargs,
            baseline_npermutations=baseline_npermutations, full_npermutations=full_npermutations, min_nonzero_confidence=min_nonzero_confidence,
            fe_confirm_undersample_rows_per_cell=float(fe_confirm_undersample_rows_per_cell or 0.0),
            max_failed=max_failed, min_relevance_gain=min_relevance_gain, max_consec_unconfirmed=max_consec_unconfirmed,
            max_runtime_mins=max_runtime_mins if max_runtime_mins is not None else 0.0,
            max_confirmation_cand_nbins=max_confirmation_cand_nbins, random_seed=random_seed if random_seed is not None else 0,
            verbose=verbose, ndigits=ndigits, start_time=st.start_time, num_possible_candidates=0,
            cached_MIs=cached_MIs, cached_confident_MIs=cached_confident_MIs, cached_cond_MIs=cached_cond_MIs,
            cached_jmim_MIs=cached_jmim_MIs, jmim_hit_counter=jmim_hit_counter, entropy_cache=entropy_cache,
            selected_vars=selected_vars, selected_interactions_vars=selected_interactions_vars,
            honest_null_codes_fn=honest_null_codes_fn,
            honest_full_factors=(_screen_full_factors if _screen_full_factors is not None else factors_data),
            honest_eval_rows=(np.asarray(subsample_idx, dtype=np.int64) if _screen_full_factors is not None else None),
        )

        subsets = range(interactions_min_order, interactions_max_order + 1)
        if interactions_order_reversed:
            subsets = subsets[::-1]

        if verbose >= 2:
            logger.info(
                "Starting work with full_npermutations=%d, min_nonzero_confidence=%.*f, max_failed=%d", full_npermutations, ndigits, min_nonzero_confidence, max_failed
            )

        num_possible_candidates = 0  # needed to refrain from multiprocessing when all direct MIs are in cache already

        factors_nbins, patience_triggered = _screen_predictors_step2_interactions_order_subsets(
            subsets, verbose, st, num_possible_candidates, ctx, stop_file, patience_triggered, min_relevance_gain, cardinality_bias_correction,
            factors_data, y, factors_nbins, min_relevance_gain_relative_to_first, selected_vars, predictors, _fdr_gain_floor, cached_MIs,
            selected_interactions_vars, dcd_state, factors_names, data_copy, entropy_cache, full_npermutations, engineered_recipes, ndigits,
        )

        if verbose >= 2:
            logger.info("Finished.")

        _log_screen_termination(ctx, selected_vars, patience_triggered, max_consec_unconfirmed)
        any_influencing = _influencing_vars(cached_confident_MIs)

        # When the screen SCORED on a subsample, recompute the RETURNED target encodings at FULL n: the
        # downstream FE pipeline consumes classes_y / classes_y_safe_host / freqs_y row-aligned to the
        # full data (the subsample only bounded the screen's relevance sweep + maxT floor cost).
        if _screen_full_factors is not None:
            classes_y, freqs_y, _ = merge_vars(factors_data=_screen_full_factors, vars_indices=y, var_is_nominal=None, factors_nbins=factors_nbins, dtype=dtype)
            classes_y_safe_host = classes_y.copy()
        # Return the CPU numpy ``classes_y_safe_host`` (NOT the CuPy device buffer that
        # ``ctx`` holds for ``mi_direct_gpu``): the caller threads this into the FE step's
        # njit MI noise-gate, which cannot accept a cupy array. ``classes_y_safe_host`` is
        # defined unconditionally above; on the CPU path it IS ``classes_y_safe``.
        return (
            selected_vars, predictors, any_influencing, entropy_cache, cached_MIs, cached_confident_MIs, cached_cond_MIs,
            classes_y, classes_y_safe_host, freqs_y, dcd_state, workers_pool,
        )
    finally:
        # numpy global state is no longer mutated by this function (a local
        # Generator is used instead), so there is nothing to restore for numpy.
        # Still restore numba + cupy (the prior comment block
        # acknowledged the leak; this closes it with a fresh-entropy reseed).
        _screen_predictors_step3_acknowledged_leak_closes(st)


def _screen_predictors_step3_acknowledged_leak_closes(st):
    """Step 3 of screen_predictors: lines starting at ``if st._numba_restore_seed is not None:``."""
    if st._numba_restore_seed is not None:
        try:
            set_numba_random_seed(int(st._numba_restore_seed))
        except Exception as e:  # nosec B110 - swallow converted to debug-log, non-fatal by design
            logger.debug("suppressed: %s", e)
            pass
    if st._cp_restore_seed is not None:
        # Mirror the
        # entry-block fix. ``cp`` only exists in this scope when
        # use_gpu was True AND CuPy was actually importable; import
        # defensively here so the restore call really fires.
        try:
            import cupy as cp

            cp.random.seed(int(st._cp_restore_seed))
        except Exception as e:  # nosec B110 - optional dependency import guard
            logger.debug("Could not restore cupy random seed (%s: %s)", type(e).__name__, e)


def _screen_predictors_step1_block(
    parallel_kwargs,
    max_confirmation_cand_nbins,
    mrmr_relevance_algo,
    mrmr_redundancy_algo,
    y,
    factors_data,
    targets_data,
    targets_nbins,
    factors_nbins,
    factors_names,
):
    """Step 1 of screen_predictors: lines starting at ``st = _SimpleNamespace() # long-lived locals of this function (see the ``."""
    st = _SimpleNamespace()  # long-lived locals of this function (see the stage helpers below)
    warmup_typed_dict()  # idempotent, once per process; see the note next to the import
    if parallel_kwargs is None:
        # backend="threading" mirrors the mrmr.py default flip (iter-371 fix):
        # joblib ThreadPoolExecutor in-process shares the data arrays zero-copy
        # so the screen pass no longer triples RAM via per-worker memmap copies
        # under Windows paging pressure. Numba kernels release the GIL so the
        # threadpool genuinely parallelises on CPU cores.
        parallel_kwargs = dict(max_nbytes=MAX_JOBLIB_NBYTES, backend="threading")

    if max_confirmation_cand_nbins is None:
        max_confirmation_cand_nbins = MAX_CONFIRMATION_CAND_NBINS

    # Converted the "Input checks" block of 7 asserts
    # to explicit ValueError. Under -O all of these stripped and bad user
    # input slipped into the MRMR loop with cryptic failure modes.
    if mrmr_relevance_algo not in ("fleuret", "pld"):
        raise ValueError(f"mrmr_relevance_algo must be 'fleuret' or 'pld'; got {mrmr_relevance_algo!r}.")
    if mrmr_redundancy_algo not in ("fleuret", "pld_max", "pld_mean"):
        raise ValueError(f"mrmr_redundancy_algo must be one of 'fleuret', 'pld_max', " f"'pld_mean'; got {mrmr_redundancy_algo!r}.")

    if y is None:
        raise ValueError("y (target column indices) must be provided.")

    if len(factors_data) < 10:
        raise ValueError(f"factors_data must have at least 10 rows; got {len(factors_data)}.")
    if targets_data is None:
        targets_data = factors_data
    else:
        if len(factors_data) != len(targets_data):
            raise ValueError(f"factors_data ({len(factors_data)} rows) and targets_data " f"({len(targets_data)} rows) must have equal length.")

    if targets_nbins is None:
        targets_nbins = factors_nbins

    if targets_data.shape[1] != len(targets_nbins):
        raise ValueError(f"targets_data.shape[1]={targets_data.shape[1]} must equal " f"len(targets_nbins)={len(targets_nbins)}.")
    if factors_data.shape[1] != len(factors_nbins):
        raise ValueError(f"factors_data.shape[1]={factors_data.shape[1]} must equal " f"len(factors_nbins)={len(factors_nbins)}.")

    # Two bugs collapsed into one
    # input-validation block.
    # (a) ``factors_names=None`` (the documented default) crashed at
    #     ``len(None)`` before reaching the auto-name branch.
    # (b) The auto-name fallback generated ``len(factors_data)`` (=n_rows)
    #     names instead of ``factors_data.shape[1]`` (=n_cols) - immediate
    #     downstream length-mismatch raise for any caller actually hitting
    #     the empty-list branch.
    if factors_names is None or len(factors_names) == 0:
        factors_names = ["F" + str(i) for i in range(factors_data.shape[1])]
    else:
        if factors_data.shape[1] != len(factors_names):
            raise ValueError(f"factors_data.shape[1]={factors_data.shape[1]} must equal " f"len(factors_names)={len(factors_names)}.")

    # Initialize x (factor indices to consider) with appropriate defaults
    return factors_names, max_confirmation_cand_nbins, parallel_kwargs, st, targets_data


def _screen_predictors_step2_initialize_factor_indices(factors_to_use, st, factors_names_to_use, factors_names, factors_data, targets_data, verbose, y):
    """Step 2 of screen_predictors: lines starting at ``if factors_to_use is not None:``."""
    if factors_to_use is not None:
        st.x = set(factors_to_use)
    elif factors_names_to_use is not None:
        st.x = [i for i, col_name in enumerate(factors_names) if col_name in factors_names_to_use]
    else:
        st.x = set(range(factors_data.shape[1]))

    # warn if inputs are identical to targets
    if factors_data.shape == targets_data.shape:
        if np.shares_memory(factors_data, targets_data):
            if factors_to_use is None and factors_names_to_use is None:
                if verbose > 2:
                    logger.info(
                        "factors_data and targets_data share the same memory. factors_to_use will be determined automatically to not contain any target columns."
                    )
                st.x = set(range(factors_data.shape[1])) - set(y)
            else:
                if factors_to_use is not None:
                    st.x = set(factors_to_use) - set(y)
                    if verbose > 2:
                        logger.info("Using only %d predefined factors: %s", len(factors_to_use), factors_to_use)
                else:
                    assert factors_names_to_use is not None  # guaranteed by the outer `factors_to_use is None and factors_names_to_use is None` check
                    st.x = [i for i, col_name in enumerate(factors_names) if col_name in factors_names_to_use and i not in y]
                    if verbose > 2:
                        logger.info("Using only %d predefined factors: %s", len(factors_names_to_use), factors_names_to_use)
        else:

            # Assert -> RuntimeError. If true, MRMR
            # would loop on self-target - silent correctness bug under -O.
            if set(y).issubset(set(st.x)):
                raise RuntimeError(
                    "MRMR invariant violated: target index set is a subset of "
                    "the factor index set; MRMR would loop on self-target. "
                    "Check that targets_data / factors_data slicing didn't "
                    "alias columns."
                )

    # ---------------------------------------------------------------------------------------------------------------
    # Inits
    # ---------------------------------------------------------------------------------------------------------------

    st.start_time = timer()
    st.run_out_of_time = False

    # RNG hygiene. numpy: the modern screening / permutation / fleuret kernels
    # each thread ``random_seed`` explicitly (inline-LCG Fisher-Yates, or their
    # own local ``default_rng``), so we no longer seed or snapshot the
    # process-global MT19937 state here - the prior ``np.random.seed`` mutated
    # a process-wide generator (racy under threads/joblib workers and a hidden
    # side effect on the caller's state). Instead we build a LOCAL Generator
    # from ``random_seed`` for any numpy-side draw; the caller's global
    # ``np.random`` state is left untouched. All current downstream numpy draws
    # are threaded via ``random_seed`` into their own local Generators / LCG, so
    # no direct global-numpy draw remains on this path.
    #
    # numba/cupy: these expose no portable Generator threading for the njit
    # kernels, so their global seeds are still set for the screening duration
    # and restored in ``finally`` with a fresh-entropy reseed, which
    # is byte-indistinguishable to any downstream consumer.
    st._numba_restore_seed = None
    st._cp_restore_seed = None
