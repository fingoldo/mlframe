"""``screen_predictors`` carved out of
``mlframe.feature_selection.filters.screen``.

Re-imported at the parent's module bottom so historical
``from mlframe.feature_selection.filters.screen import screen_predictors``
resolves transparently.
"""
from __future__ import annotations

import logging
from itertools import combinations
from os.path import exists
from timeit import default_timer as timer
from typing import Sequence

import numba
import numpy as np
from joblib import Parallel, delayed
from numba.core import types

from pyutilz.numbalib import set_numba_random_seed
from pyutilz.system import tqdmu

from ._internals import MAX_CONFIRMATION_CAND_NBINS, MAX_JOBLIB_NBYTES
from ._confirm_predictor import ScreenContext, confirm_one_predictor
from .evaluation import get_candidate_name
from .info_theory import merge_vars

logger = logging.getLogger(__name__)


def _pool_warmup_noop(i):
    """Module-level no-op handed to the joblib pool so worker spawn cost is paid before
    the screening loop starts. Must be module-level (not a closure) so the ``loky`` backend
    can pickle it. Mirrors ``screen._pool_warmup_noop``; defined here too because the warmup
    call lives in this module's ``screen_predictors`` (pre-2026-05 this reference was an
    unbound ``NameError`` on the ``n_workers>1`` path)."""
    return None


def screen_predictors(
    # factors
    factors_data: np.ndarray,
    factors_nbins: Sequence[int],
    factors_names: Sequence[str] = None,
    factors_names_to_use: Sequence[str] = None,
    factors_to_use: Sequence[int] = None,
    # targets
    targets_data: np.ndarray = None,
    targets_nbins: Sequence[int] = None,
    y: Sequence[int] = None,
    # algorithm
    mrmr_relevance_algo: str = "fleuret",
    mrmr_redundancy_algo: str = "fleuret",
    reduce_gain_on_subelement_chosen: bool = True,
    # performance
    extra_x_shuffling: bool = True,
    dtype=np.int32,
    random_seed: int = None,
    use_gpu: bool = False,
    n_workers: int = 1,
    # confidence
    min_occupancy: int = None,
    min_nonzero_confidence: float = 0.99,
    full_npermutations: int = 1_000,
    baseline_npermutations: int = 100,
    # stopping conditions
    min_relevance_gain: float = 0.00001,
    max_consec_unconfirmed: int = 30,
    max_runtime_mins: float = None,
    interactions_min_order: int = 1,
    interactions_max_order: int = 1,
    interactions_order_reversed: bool = False,
    max_veteranes_interactions_order: int = 1,
    only_unknown_interactions: bool = False,
    # Confirmation-step cardinality cutoff. ``None`` falls back to ``MAX_CONFIRMATION_CAND_NBINS``; ``MRMR.fit`` overrides with ``quantization_nbins ** interactions_max_order * 2``.
    max_confirmation_cand_nbins: int = None,
    # When screening returns zero selected_vars, legacy FE fell back to running on ALL features. False skips FE instead (safer default: FE on empty screen amplifies noise).
    fe_fallback_to_all: bool = True,
    # verbosity and formatting
    verbose: int = 1,
    ndigits: int = 5,
    parallel_kwargs: dict = None,
    stop_file: str = None,
    use_simple_mode: bool = True,
    # ``engineered_lineage`` -- mapping ``{engineered_col_idx: frozenset(parent_indices)}``. When set, k-way candidate enumeration skips combinations of an engineered
    # column with one of its own parents (e.g. ``(orig_i, kway(orig_i, orig_j))`` is redundant since the engineered col already contains orig_i's info). Threaded
    # through ``should_skip_candidate``. ``None`` preserves legacy behaviour.
    engineered_lineage: dict = None,
) -> float:
    """Finds best predictors for the target. ``factors_data`` must be an n-by-m array of integers (ordinal encoded).

    ``max_confirmation_cand_nbins=None`` falls back to the module constant for backward compat; ``MRMR.fit`` overrides explicitly. ``fe_fallback_to_all`` is consumed
    by ``MRMR.fit`` and only threaded here for caller pass-through.

    Parameters:
        full_npermutations: when computing every MI, repeat calculations with randomly shuffled indices that many times
        min_nonzero_confidence: if in random permutation tests this or higher % of cases had worse current_gain than original, current_gain value is considered valid, otherwise, it's set to zero.
        only_unknown_interactions: True for speed, False for completeness of higher order interactions discovery.
        verbose: int  1=log only important info,>1=also log additional details
        mrmr_relevance_algo:str
                        "fleuret": max(min(I(X,Y|Z)),max(I(X,Y|Z)-I(X,Y))) Possible to use n-way interactions here.
                        "pld": I(X,Y)
        mrmr_redundancy_algo:str
                        "fleuret": 0 ('cause redundancy already accounted for)
                        "pld_max": max(I(veterane,cand)) Possible to use n-way interactions here.
                        "pld_mean": mean(I(veterane,cand)) Possible to use n-way interactions here.

    Returns:
        1) best set of non-redundant single features influencing the target
        2) subsets of size 2..interactions_max_order influencing the target. Such subsets will be candidates for predictors and OtherVarsEncoding.
        3) all 1-vs-1 influencers (not necessarily in mRMR)
    """
    # ---------------------------------------------------------------------------------------------------------------
    # Input checks
    # ---------------------------------------------------------------------------------------------------------------

    if parallel_kwargs is None:
        # backend="threading" mirrors the mrmr.py default flip (iter-371 fix):
        # joblib ThreadPoolExecutor in-process shares the data arrays zero-copy
        # so the screen pass no longer triples RAM via per-worker memmap copies
        # under Windows paging pressure. Numba kernels release the GIL so the
        # threadpool genuinely parallelises on CPU cores.
        parallel_kwargs = dict(max_nbytes=MAX_JOBLIB_NBYTES, backend="threading")

    if max_confirmation_cand_nbins is None:
        max_confirmation_cand_nbins = MAX_CONFIRMATION_CAND_NBINS

    # Wave 31 (2026-05-20): converted the "Input checks" block of 7 asserts
    # to explicit ValueError. Under -O all of these stripped and bad user
    # input slipped into the MRMR loop with cryptic failure modes.
    if mrmr_relevance_algo not in ("fleuret", "pld"):
        raise ValueError(
            f"mrmr_relevance_algo must be 'fleuret' or 'pld'; got {mrmr_relevance_algo!r}."
        )
    if mrmr_redundancy_algo not in ("fleuret", "pld_max", "pld_mean"):
        raise ValueError(
            f"mrmr_redundancy_algo must be one of 'fleuret', 'pld_max', "
            f"'pld_mean'; got {mrmr_redundancy_algo!r}."
        )

    if len(factors_data) < 10:
        raise ValueError(
            f"factors_data must have at least 10 rows; got {len(factors_data)}."
        )
    if targets_data is None:
        targets_data = factors_data
    else:
        if len(factors_data) != len(targets_data):
            raise ValueError(
                f"factors_data ({len(factors_data)} rows) and targets_data "
                f"({len(targets_data)} rows) must have equal length."
            )

    if targets_nbins is None:
        targets_nbins = factors_nbins

    if targets_data.shape[1] != len(targets_nbins):
        raise ValueError(
            f"targets_data.shape[1]={targets_data.shape[1]} must equal "
            f"len(targets_nbins)={len(targets_nbins)}."
        )
    if factors_data.shape[1] != len(factors_nbins):
        raise ValueError(
            f"factors_data.shape[1]={factors_data.shape[1]} must equal "
            f"len(factors_nbins)={len(factors_nbins)}."
        )

    if len(factors_names) == 0:
        factors_names = ["F" + str(i) for i in range(len(factors_data))]
    else:
        if factors_data.shape[1] != len(factors_names):
            raise ValueError(
                f"factors_data.shape[1]={factors_data.shape[1]} must equal "
                f"len(factors_names)={len(factors_names)}."
            )

    # Initialize x (factor indices to consider) with appropriate defaults
    if factors_to_use is not None:
        x = set(factors_to_use)
    elif factors_names_to_use is not None:
        x = [i for i, col_name in enumerate(factors_names) if col_name in factors_names_to_use]
    else:
        x = set(range(factors_data.shape[1]))

    # warn if inputs are identical to targets
    if factors_data.shape == targets_data.shape:
        if np.shares_memory(factors_data, targets_data):
            if factors_to_use is None and factors_names_to_use is None:
                if verbose > 2:
                    logger.info(
                        "factors_data and targets_data share the same memory. factors_to_use will be determined automatically to not contain any target columns."
                    )
                x = set(range(factors_data.shape[1])) - set(y)
            else:
                if factors_to_use is not None:
                    x = set(factors_to_use) - set(y)
                    if verbose > 2:
                        logger.info("Using only %d predefined factors: %s", len(factors_to_use), factors_to_use)
                else:
                    x = [i for i, col_name in enumerate(factors_names) if col_name in factors_names_to_use and i not in y]
                    if verbose > 2:
                        logger.info("Using only %d predefined factors: %s", len(factors_names_to_use), factors_names_to_use)
        else:

            # Wave 31 (2026-05-20): assert -> RuntimeError. If true, MRMR
            # would loop on self-target -- silent correctness bug under -O.
            if set(y).issubset(set(x)):
                raise RuntimeError(
                    "MRMR invariant violated: target index set is a subset of "
                    "the factor index set; MRMR would loop on self-target. "
                    "Check that targets_data / factors_data slicing didn't "
                    "alias columns."
                )

    # ---------------------------------------------------------------------------------------------------------------
    # Inits
    # ---------------------------------------------------------------------------------------------------------------

    start_time = timer()
    run_out_of_time = False

    # Global-RNG hygiene: snapshot the process-wide MT19937 state on entry,
    # seed it (numpy + numba + cupy) for the screening duration so downstream
    # ``np.random.shuffle`` calls in permutation / fleuret kernels stay
    # deterministic, then restore in a ``finally`` so the caller's state is
    # byte-identical on EVERY exit path (happy return AND any mid-screen raise).
    # Wave 49 (2026-05-20): also restore numba and cupy seeds (the prior
    # implementation acknowledged the leak in comments but didn't fix it).
    # Numba/CuPy expose no portable get_state, so we re-seed them at finally
    # time with a high-entropy random64 captured pre-entry from os.urandom --
    # not byte-identical but indistinguishable to any downstream consumer.
    _np_state_snapshot = None
    _numba_restore_seed = None
    _cp_restore_seed = None
    if random_seed is not None:
        _np_state_snapshot = np.random.get_state()
        # Capture a fresh entropy-derived seed to restore numba/cupy with on
        # finally; mathematically equivalent (from the consumer's view) to
        # "the seed they would have had if no inner seed call had fired".
        import os as _os, struct as _struct
        _numba_restore_seed = _struct.unpack("<Q", _os.urandom(8))[0]
        _cp_restore_seed = _struct.unpack("<Q", _os.urandom(8))[0]
        np.random.seed(random_seed)
        set_numba_random_seed(random_seed)
        try:
            cp.random.seed(random_seed)
        except NameError:
            pass  # CuPy not imported

    try:
        max_failed = int(full_npermutations * (1 - min_nonzero_confidence))
        if max_failed <= 1:
            max_failed = 1

        selected_interactions_vars = []
        selected_vars = []  # stores just indices. can't use set 'cause the order is important for efficient computing
        predictors = []  # stores more details.

        # True if inner confirmation loop hit ``max_consec_unconfirmed`` patience at least once. Surfaced at function exit so callers can distinguish "gave up confirming"
        # from "natural gain threshold reached".
        patience_triggered: bool = False

        cached_MIs = dict()
        # cached_cond_MIs = dict()
        cached_confident_MIs = dict()
        entropy_cache = numba.typed.Dict.empty(
            key_type=types.unicode_type,
            value_type=types.float64,
        )
        cached_cond_MIs = numba.typed.Dict.empty(
            key_type=types.unicode_type,
            value_type=types.float64,
        )

        data_copy = factors_data.copy()

        classes_y, freqs_y, _ = merge_vars(factors_data=factors_data, vars_indices=y, var_is_nominal=None, factors_nbins=factors_nbins, dtype=dtype)
        classes_y_safe = classes_y.copy()

        if use_gpu:
            import cupy as cp

            classes_y_safe = cp.asarray(classes_y.astype(np.int32))
            freqs_y_safe = cp.asarray(freqs_y)
        else:
            freqs_y_safe = None

        if n_workers and n_workers > 1:
            if verbose >= 2:
                logger.info("Starting parallel pool with n_workers=%d", n_workers)

            # Threading backend: worker fn ``evaluate_candidates`` eventually calls njit (compute_mi_from_classes, shuffle_arr, parallel_mi) which release the GIL, so
            # threading gives near-process-pool speedup with zero IPC / pickle / memmap overhead. Threading also avoids the Windows-only joblib resource-tracker
            # KeyError on shutdown caused by loky's auto-memmap of large numpy arrays desyncing across screen-iteration boundaries when the same Parallel object is
            # re-called. Override via ``parallel_kwargs={"backend": "loky"}`` if process-isolation is needed (rare on this workload).
            pk = dict(parallel_kwargs)
            pk.setdefault("backend", "threading")
            # Disable auto-memmap regardless of backend: large factors_data arrays should stay in shared process memory, not be redundantly serialized to disk.
            pk.setdefault("max_nbytes", None)

            if pk.get("backend") == "loky":
                try:
                    from loky import set_loky_pickler
                    set_loky_pickler("cloudpickle")
                except ImportError:
                    pass

            workers_pool = Parallel(n_jobs=n_workers, **pk)
            # Warmup: spawn workers eagerly via a no-op call so spawn cost is paid before the screening loop starts.
            workers_pool(delayed(_pool_warmup_noop)(i) for i in range(n_workers))
        else:
            workers_pool = None

        # Shared confirmation context. Static fields + the four MI caches are set once; per-interactions-order fields
        # (``candidates`` / ``partial_gains`` / ``added_candidates`` / ``failed_candidates`` / ``interactions_order`` /
        # ``num_possible_candidates``) are refreshed at the top of each order below. ``selected_vars`` /
        # ``selected_interactions_vars`` are the same list objects throughout (appended to, never reassigned), so the
        # confirmation primitives always see the up-to-date selection.
        ctx = ScreenContext(
            factors_data=factors_data,
            factors_nbins=factors_nbins,
            factors_names=factors_names,
            y=y,
            data_copy=data_copy,
            classes_y=classes_y,
            classes_y_safe=classes_y_safe,
            freqs_y=freqs_y,
            freqs_y_safe=freqs_y_safe,
            mrmr_relevance_algo=mrmr_relevance_algo,
            mrmr_redundancy_algo=mrmr_redundancy_algo,
            reduce_gain_on_subelement_chosen=reduce_gain_on_subelement_chosen,
            max_veteranes_interactions_order=max_veteranes_interactions_order,
            only_unknown_interactions=only_unknown_interactions,
            use_gpu=use_gpu,
            use_simple_mode=use_simple_mode,
            extra_x_shuffling=extra_x_shuffling,
            engineered_lineage=engineered_lineage,
            n_workers=n_workers,
            workers_pool=workers_pool,
            parallel_kwargs=parallel_kwargs,
            baseline_npermutations=baseline_npermutations,
            full_npermutations=full_npermutations,
            min_nonzero_confidence=min_nonzero_confidence,
            max_failed=max_failed,
            min_relevance_gain=min_relevance_gain,
            max_consec_unconfirmed=max_consec_unconfirmed,
            max_runtime_mins=max_runtime_mins,
            max_confirmation_cand_nbins=max_confirmation_cand_nbins,
            random_seed=random_seed,
            verbose=verbose,
            ndigits=ndigits,
            start_time=start_time,
            num_possible_candidates=0,
            cached_MIs=cached_MIs,
            cached_confident_MIs=cached_confident_MIs,
            cached_cond_MIs=cached_cond_MIs,
            entropy_cache=entropy_cache,
            selected_vars=selected_vars,
            selected_interactions_vars=selected_interactions_vars,
        )

        subsets = range(interactions_min_order, interactions_max_order + 1)
        if interactions_order_reversed:
            subsets = subsets[::-1]

        if verbose >= 2:
            logger.info(
                "Starting work with full_npermutations=%d, min_nonzero_confidence=%.*f, max_failed=%d",
                full_npermutations, ndigits, min_nonzero_confidence, max_failed,
            )

        num_possible_candidates = 0  # needed to refrain from multiprocessing when all direct MIs are in cache already

        for interactions_order in (subsets_pbar := tqdmu(subsets, desc="Interactions order", leave=False)):

            if run_out_of_time:
                break
            subsets_pbar.set_description(f"{interactions_order}-way interactions")

            # ---------------------------------------------------------------------------------------------------------------
            # Generate candidates
            # ---------------------------------------------------------------------------------------------------------------

            candidates = [tuple(el) for el in combinations(x, interactions_order)]

            num_possible_candidates += len(candidates)

            # ---------------------------------------------------------------------------------------------------------------
            # Subset level inits
            # ---------------------------------------------------------------------------------------------------------------

            total_disproved = 0
            total_checked = 0
            partial_gains = {}
            added_candidates = set()
            failed_candidates = set()
            nconsec_unconfirmed = 0

            # Refresh the confirmation context for this interactions order.
            ctx.candidates = candidates
            ctx.interactions_order = interactions_order
            ctx.partial_gains = partial_gains
            ctx.added_candidates = added_candidates
            ctx.failed_candidates = failed_candidates
            ctx.num_possible_candidates = num_possible_candidates

            for _n_confirmed_predictors in (predictors_pbar := tqdmu(range(len(candidates)), leave=False, desc="Confirmed predictors")):
                if run_out_of_time:
                    break
                if stop_file and exists(stop_file):
                    logger.warning(f"Stop file {stop_file} detected, quitting.")
                    break

                # The full single-predictor confirmation cycle (score all candidates, then permutation-confirm in
                # expected-gain order with partial-gain recompute/retry + patience accounting) lives in the
                # ``confirm_one_predictor`` primitive (``_confirm_predictor.py``), keeping this file below the
                # 1k-line monolith threshold and the frequently patched confirmation math in one place.
                (
                    best_candidate,
                    best_gain,
                    confidence,
                    run_out_of_time,
                    nconsec_unconfirmed,
                    total_checked,
                    total_disproved,
                    patience_triggered,
                ) = confirm_one_predictor(
                    ctx,
                    nconsec_unconfirmed=nconsec_unconfirmed,
                    total_checked=total_checked,
                    total_disproved=total_disproved,
                    patience_triggered=patience_triggered,
                )

                # ---------------------------------------------------------------------------------------------------------------
                # Add best candidate to the list, if criteria are met, or proceed to the next interactions_order
                # ---------------------------------------------------------------------------------------------------------------

                if best_gain >= (min_relevance_gain if interactions_order == 1 else min_relevance_gain ** (1 / (interactions_order + 1))):
                    for var in best_candidate:
                        if var not in selected_vars:
                            selected_vars.append(var)
                            if interactions_order > 1:
                                selected_interactions_vars.append(var)
                    cand_name = get_candidate_name(best_candidate, factors_names=factors_names)

                    res = {"name": cand_name, "indices": best_candidate, "gain": best_gain}
                    if full_npermutations:
                        res["confidence"] = confidence
                    predictors.append(res)

                    if verbose >= 2:
                        mes = f"Added new predictor {cand_name} to the list with expected gain={best_gain:.{ndigits}f}"
                        if full_npermutations:
                            mes += f" and confidence={confidence:.3f}"
                        logger.info(mes)

                else:
                    if verbose >= 2:
                        if total_checked > 0:
                            details = f" Total candidates disproved: {total_disproved:_}/{total_checked:_} ({total_disproved*100/total_checked:.2f}%)"
                        else:
                            details = ""
                        logger.info("Can't add anything valuable anymore for interactions_order=%s.%s", interactions_order, details)
                    predictors_pbar.total = len(candidates)
                    predictors_pbar.close()
                    break

        # postprocess_candidates(selected_vars)
        # print(caching_hits_xyz, caching_hits_z, caching_hits_xz, caching_hits_yz)
        if verbose >= 2:
            logger.info("Finished.")

        # Termination-reason summary (always emitted). ``patience_triggered`` distinguishes "gave up confirming" (max_consec_unconfirmed hit) from natural exhaustion below
        # ``min_relevance_gain``. If patience keeps tripping, raise ``max_consec_unconfirmed`` or smooth the relevance signals.
        if patience_triggered:
            logger.warning(
                "screen_predictors terminated early via max_consec_unconfirmed=%d "
                "patience (at least one level exhausted). Returned %d selected "
                "feature(s). If you expected more, increase max_consec_unconfirmed "
                "or reduce the relevance-gain threshold.",
                max_consec_unconfirmed, len(selected_vars),
            )
        else:
            logger.info(
                "screen_predictors finished naturally (no patience trip). "
                "Returned %d selected feature(s).",
                len(selected_vars),
            )

        any_influencing = set()
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
        return selected_vars, predictors, any_influencing, entropy_cache, cached_MIs, cached_confident_MIs, cached_cond_MIs, classes_y, classes_y_safe, freqs_y
    finally:
        # Restore the global numpy RNG state captured at entry. Executes on the
        # happy return AND on any raise inside the try -- pre-fix code only restored on
        # the happy path, leaving the global state seeded after mid-screen exceptions.
        if _np_state_snapshot is not None:
            np.random.set_state(_np_state_snapshot)
        # Wave 49 (2026-05-20): also restore numba + cupy (the prior comment block
        # acknowledged the leak; this closes it with a fresh-entropy reseed).
        if _numba_restore_seed is not None:
            try:
                set_numba_random_seed(int(_numba_restore_seed))
            except Exception:
                pass
        if _cp_restore_seed is not None:
            try:
                cp.random.seed(int(_cp_restore_seed))
            except (NameError, Exception):
                pass
