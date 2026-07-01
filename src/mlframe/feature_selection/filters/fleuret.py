"""Fleuret-criterion permutation-confidence step. Three layers: ``get_fleuret_criteria_confidence_parallel`` (joblib pool spawner) ->
``parallel_fleuret`` (joblib worker) -> ``get_fleuret_criteria_confidence`` (``@njit`` core looping over the permutation budget).

Known weakness: the Fleuret formulation ``gain = I(X; Y) - max_k I(X; Y | S_k)`` rejects synergistic features (a candidate
uninformative on its own but critical in combination with an already-selected variable scores ``gain < 0``). The synergy-aware mitigation
is shipped: ``MRMR(redundancy_aggregator='jmim')`` (Bennasar 2015, ``min_j I(X_k, X_j; Y)``) and the data-dependent gate
``redundancy_aggregator='auto'`` (``_synergy_detector.detect_synergy`` routes to JMIM only when synergy is detected, else plain Fleuret).

BENCHMARKED DECISION (2026-06-19, ``_benchmarks/fs_quality/mrmr_synergy_regime_bench.py``, 10 seeds x n in {4k,8k}, fixed 60/40 split,
held-out LGBM+logit AUC + ground-truth recovery; planted XOR/sign-product pairs vs planted additive main-effects):

  * The synergy detector (per-pair INTERACTION INFORMATION ``I({X,Z};Y) - I(X;Y) - I(Z;Y)`` vs a label-permuted null, threshold =
    null_mult x null_scale from kernel_tuning_cache) fired on EVERY synergistic cell and on NO additive cell (clean separation:
    real excess ~0.12 vs threshold ~0.05 synergistic; ~0.028 vs ~0.05 additive). Using ``joint - max(marginal)`` was tried and
    REJECTED -- it false-positives on additive redundancy (two noisy views of one driver jointly beat either view alone).
  * ``auto`` reproduced ``jmim`` selection EXACTLY on synergistic data and ``default``/plain-Fleuret EXACTLY on additive data
    (paired auto-vs-default additive: recall 0/20/0, holdout-AUC 0/20/0 -- bit-for-bit no regression). HARD GATE satisfied.
  * Plain Fleuret's synergistic miss + JMIM's recovery is REAL but MODEST on these regimes (jmim-vs-default synergistic recall
    4W/15T/1L, AUC 3W/14T/3L -- balanced XOR at n<=8000 sits near the noise floor, AUC ~0.5). JMIM's additive cost is the documented
    OVER-SELECTION (precision 0.67->0.33, nsel 3.6-5 -> 9.0; downstream AUC roughly tied). ``auto`` pays NONE of that additive cost.
  * VERDICT: ``auto`` is strictly safe (captures JMIM's synergy gain, avoids JMIM's additive over-selection) and ships as a
    VALIDATED OPT-IN. It is NOT made the default: the synergistic win is too marginal on the tested regimes to override the
    bit-stable Fleuret default + the detector's pre-fit probe cost. Earlier large-n campaigns (mrmr_largeN_campaign*.py) already
    found JMIM loses F1 9/0/51 on additive/decoy data -- consistent. Next agent: re-run the bench at larger n / stronger synergy
    before reconsidering a default flip.
"""
from __future__ import annotations

import gc

import numpy as np
import numba
from joblib import Parallel, delayed
from numba import njit
from numba.core import types

from pyutilz.numbalib import python_dict_2_numba_dict

from .permutation import distribute_permutations, _perm_pvalue, _DEFAULT_BASE_SEED
from ._internals import LARGE_CONST
from .evaluation import evaluate_gain
from .info_theory import use_su_normalization


def get_fleuret_criteria_confidence_parallel(
    data_copy: np.ndarray,
    factors_nbins: np.ndarray,
    x: tuple,
    y: tuple,
    selected_vars: list,
    bootstrapped_gain: float,
    npermutations: int,
    max_failed: int,
    nexisting: int,
    mrmr_relevance_algo: str = "fleuret",
    mrmr_redundancy_algo: str = "fleuret",
    max_veteranes_interactions_order: int = 1,
    extra_knowledge_multipler: float = -1.0,
    sink_threshold: float = -1.0,
    cached_cond_MIs: dict = None,
    n_workers: int = 1,
    workers_pool: object = None,
    parallel_kwargs: dict = None,
    entropy_cache: dict = None,
    extra_x_shuffling: bool = True,
    dtype=np.int32,
    base_seed: int = 0,
) -> tuple:
    if parallel_kwargs is None:
        parallel_kwargs = {}
    nfailed = 0

    if workers_pool is None:
        workers_pool = Parallel(n_jobs=n_workers, **parallel_kwargs)

    # 2026-05-28: read SU toggle once here (Python-level) and thread into every joblib worker.
    _use_su = use_su_normalization()

    gc.collect()
    # Per-worker seed derivation: outer base_seed * Knuth multiplicative hash + worker index keeps streams independent yet aggregate is reproducible from outer base_seed.
    _worker_loads = list(distribute_permutations(npermutations=npermutations, n_workers=n_workers))
    res = workers_pool(
        delayed(parallel_fleuret)(
            data=data_copy,
            factors_nbins=factors_nbins,
            x=x,
            y=y,
            selected_vars=selected_vars,
            npermutations=worker_npermutations,
            bootstrapped_gain=bootstrapped_gain,
            max_failed=max_failed,
            nexisting=nexisting,
            mrmr_relevance_algo=mrmr_relevance_algo,
            mrmr_redundancy_algo=mrmr_redundancy_algo,
            max_veteranes_interactions_order=max_veteranes_interactions_order,
            extra_knowledge_multipler=extra_knowledge_multipler,
            sink_threshold=sink_threshold,
            cached_cond_MIs=dict(cached_cond_MIs),
            entropy_cache=dict(entropy_cache),
            extra_x_shuffling=extra_x_shuffling,
            dtype=dtype,
            base_seed=int((int(base_seed) * 2654435761 + (_widx + 1)) & 0xFFFFFFFFFFFFFFFF),
            use_su=_use_su,
        )
        for _widx, worker_npermutations in enumerate(_worker_loads)
    )

    nchecked = 0
    for worker_nfailed, worker_i, entropy_cache_dict in res:
        nfailed += worker_nfailed
        nchecked += worker_i
        for key, value in entropy_cache_dict.items():
            entropy_cache[key] = value

    if nfailed >= max_failed:
        bootstrapped_gain = 0.0

    # Permutation p-value for this candidate via the canonical ``_perm_pvalue`` estimator, which applies BOTH:
    #   1. Add-one (Davison & Hinkley 1997; Phipson & Smyth 2010): ``(1 + nfailed) / (1 + budget)`` -- the observed gain
    #      is itself one draw under the null, so the Monte-Carlo p can never be exactly 0; a naive ``nfailed/nchecked``
    #      returns 0 on a null feature that never fails, spuriously reporting p=0 / confidence=1.
    #   2. ``full_budget`` de-biasing of the ``max_failed`` early-stop: the worker loop breaks as soon as ``nfailed``
    #      reaches ``max_failed``, so ``nchecked`` is data-dependent and the stopped ratio overstates the failure rate
    #      (it stops precisely where failures cluster). Passing the full ``npermutations`` budget as the denominator
    #      makes the reported p independent of WHERE the early break fired -- an honest (conservative) estimate.
    # NOTE (multiple testing): this p-value is computed independently PER CANDIDATE over the whole greedy path with NO
    # family-wise / FDR correction. The reported confidence is therefore an UNCORRECTED per-candidate significance;
    # callers needing a calibrated false-discovery rate should apply a BH/BY correction across the candidate family.
    p_value = _perm_pvalue(nfailed, nchecked, full_budget=int(npermutations))
    confidence = 1.0 - p_value

    return bootstrapped_gain, confidence, entropy_cache


def parallel_fleuret(
    data: np.ndarray,
    factors_nbins: np.ndarray,
    x: tuple,
    y: tuple,
    selected_vars: list,
    npermutations: int,
    bootstrapped_gain: float,
    max_failed: int,
    nexisting: int,
    mrmr_relevance_algo: str = "fleuret",
    mrmr_redundancy_algo: str = "fleuret",
    max_veteranes_interactions_order: int = 1,
    extra_knowledge_multipler: float = -1.0,
    sink_threshold: float = -1.0,
    cached_cond_MIs: dict = None,
    entropy_cache: dict = None,
    extra_x_shuffling: bool = True,
    dtype=np.int32,
    base_seed: int = 0,
    use_su: bool = False,  # 2026-05-28: threaded from get_fleuret_criteria_confidence_parallel.
):
    """Joblib worker: rebuild numba.typed.Dict from pickled Python dicts, run the njit core, return a Python dict for the parent's union."""
    data_copy = data.copy()

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

    nfailed, i = get_fleuret_criteria_confidence(
        data_copy=data_copy,
        factors_nbins=factors_nbins,
        x=x,
        y=y,
        selected_vars=selected_vars,
        npermutations=npermutations,
        bootstrapped_gain=bootstrapped_gain,
        max_failed=max_failed,
        nexisting=nexisting,
        mrmr_relevance_algo=mrmr_relevance_algo,
        mrmr_redundancy_algo=mrmr_redundancy_algo,
        max_veteranes_interactions_order=max_veteranes_interactions_order,
        cached_cond_MIs=cached_cond_MIs_dict,
        entropy_cache=entropy_cache_dict,
        extra_x_shuffling=extra_x_shuffling,
        dtype=dtype,
        base_seed=np.uint64(base_seed),
        use_su=use_su,
    )

    return nfailed, i, dict(entropy_cache_dict)


@njit(cache=True)
def _fleuret_shuffle_col_lcg(col: np.ndarray, state: np.uint64) -> np.uint64:
    """Inline Fisher-Yates over a 1-D column slice using the same LCG schedule as ``permutation.shuffle_arr_lcg``.

    Returns the post-shuffle state so the caller threads it across per-column and per-permutation calls. Replaces ``np.random.shuffle(data_copy[:, idx])`` whose global numpy RNG state is process-wide
    and races under joblib parallel workers / multi-suite invocations.
    """
    n = len(col)
    for j in range(n - 1, 0, -1):
        state = state * np.uint64(6364136223846793005) + np.uint64(1442695040888963407)
        k = int(state >> np.uint64(33)) % (j + 1)
        tmp = col[j]
        col[j] = col[k]
        col[k] = tmp
    return state


@njit(cache=True)
def get_fleuret_criteria_confidence(
    data_copy: np.ndarray,
    factors_nbins: np.ndarray,
    x: tuple,
    y: tuple,
    selected_vars: list,
    npermutations: int,
    bootstrapped_gain: float,
    max_failed: int,
    nexisting: int,
    mrmr_relevance_algo: str = "fleuret",
    mrmr_redundancy_algo: str = "fleuret",
    max_veteranes_interactions_order: int = 1,
    extra_knowledge_multipler: float = -1.0,
    sink_threshold: float = -1.0,
    cached_cond_MIs: dict = None,
    entropy_cache: dict = None,
    extra_x_shuffling: bool = True,
    dtype=np.int32,
    base_seed: np.uint64 = _DEFAULT_BASE_SEED,
    use_su: bool = False,  # 2026-05-28: threaded from Python-level parallel_fleuret.
) -> tuple:
    """Sub to njit work with random shuffling as well.

    Zero-permutation guard returns ``(0, 0)`` cleanly. The legacy code raised ``UnboundLocalError`` on ``i`` because the for-loop never assigned it.

    ``base_seed`` threads through inline LCG Fisher-Yates so per-permutation column shuffles are reproducible from a single seed; pre-fix code used ``np.random.shuffle`` on the process-global numpy RNG which raced under joblib parallel workers and made two parallel suite calls produce non-deterministic confidence values.
    """
    if npermutations == 0:
        return 0, 0

    nfailed = 0
    _i = 0
    _lcg_state = np.uint64(base_seed) * np.uint64(2654435761) + np.uint64(1)
    for _i in range(npermutations):

        for idx in y:
            _lcg_state = np.uint64(_fleuret_shuffle_col_lcg(data_copy[:, idx], _lcg_state))

        if extra_x_shuffling:
            for idx in x:
                _lcg_state = np.uint64(_fleuret_shuffle_col_lcg(data_copy[:, idx], _lcg_state))

        stopped_early, current_gain, k, sink_reasons = evaluate_gain(
            current_gain=LARGE_CONST,
            last_checked_k=-1,
            X=x,
            y=y,
            best_gain=None,
            factors_data=data_copy,
            factors_nbins=factors_nbins,
            selected_vars=selected_vars,
            nexisting=nexisting,
            direct_gain=bootstrapped_gain,
            mrmr_relevance_algo=mrmr_relevance_algo,
            mrmr_redundancy_algo=mrmr_redundancy_algo,
            max_veteranes_interactions_order=max_veteranes_interactions_order,
            extra_knowledge_multipler=extra_knowledge_multipler,
            sink_threshold=sink_threshold,
            cached_cond_MIs=cached_cond_MIs,
            entropy_cache=entropy_cache,
            can_use_x_cache=not extra_x_shuffling,
            can_use_y_cache=False,
            confidence_mode=True,
            use_su=use_su,
        )

        if current_gain >= bootstrapped_gain:
            nfailed += 1
            if nfailed >= max_failed:
                break

    return nfailed, _i + 1
