"""Fleuret-criterion permutation-confidence step. Three layers: ``get_fleuret_criteria_confidence_parallel`` (splits the budget into worker segments) ->
``parallel_fleuret`` (one segment) -> ``get_fleuret_criteria_confidence`` (``@njit`` core looping over the permutation budget). The segments run one after the other in the calling thread: there is no
joblib pool any more (the njit core releases the GIL, the screen parallelises over candidates with its prefetch threads, and the device path of ``_fleuret_confirm_gpu`` replaces the pool for large
row counts); the segmentation is kept because the early stop at ``max_failed`` is per segment, which the counts of this route have always depended on.

Known weakness: the Fleuret formulation ``gain = I(X; Y) - max_k I(X; Y | S_k)`` rejects synergistic features (a candidate
uninformative on its own but critical in combination with an already-selected variable scores ``gain < 0``). The synergy-aware mitigation
is shipped: ``MRMR(redundancy_aggregator='jmim')`` (Bennasar 2015, ``min_j I(X_k, X_j; Y)``) and the data-dependent gate
``redundancy_aggregator='auto'`` (``_synergy_detector.detect_synergy`` routes to JMIM only when synergy is detected, else plain Fleuret).

BENCHMARKED DECISION (2026-06-19, ``_benchmarks/fs_quality/mrmr_synergy_regime_bench.py``, 10 seeds x n in {4k,8k}, fixed 60/40 split,
held-out LGBM+logit AUC + ground-truth recovery; planted XOR/sign-product pairs vs planted additive main-effects):

  * The synergy detector (per-pair INTERACTION INFORMATION ``I({X,Z};Y) - I(X;Y) - I(Z;Y)`` vs a label-permuted null, threshold =
    null_mult x null_scale from kernel_tuning_cache) fired on EVERY synergistic cell and on NO additive cell (clean separation:
    real excess ~0.12 vs threshold ~0.05 synergistic; ~0.028 vs ~0.05 additive). Using ``joint - max(marginal)`` was tried and
    REJECTED - it false-positives on additive redundancy (two noisy views of one driver jointly beat either view alone).
  * ``auto`` reproduced ``jmim`` selection EXACTLY on synergistic data and ``default``/plain-Fleuret EXACTLY on additive data
    (paired auto-vs-default additive: recall 0/20/0, holdout-AUC 0/20/0 - bit-for-bit no regression). HARD GATE satisfied.
  * Plain Fleuret's synergistic miss + JMIM's recovery is REAL but MODEST on these regimes (jmim-vs-default synergistic recall
    4W/15T/1L, AUC 3W/14T/3L - balanced XOR at n<=8000 sits near the noise floor, AUC ~0.5). JMIM's additive cost is the documented
    OVER-SELECTION (precision 0.67->0.33, nsel 3.6-5 -> 9.0; downstream AUC roughly tied). ``auto`` pays NONE of that additive cost.
  * VERDICT: ``auto`` is strictly safe (captures JMIM's synergy gain, avoids JMIM's additive over-selection) and ships as a
    VALIDATED OPT-IN. It is NOT made the default: the synergistic win is too marginal on the tested regimes to override the
    bit-stable Fleuret default + the detector's pre-fit probe cost. Earlier large-n campaigns (mrmr_largeN_campaign*.py) already
    found JMIM loses F1 9/0/51 on additive/decoy data - consistent. Next agent: re-run the bench at larger n / stronger synergy
    before reconsidering a default flip.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import numba
from numba.core import types

from pyutilz.numbalib import python_dict_2_numba_dict

from ._numba_utils import _fleuret_shuffle_col_lcg, distribute_permutations, evaluate_gain, get_fleuret_criteria_confidence  # noqa: F401 -- re-exported; get_fleuret_criteria_confidence is used below
from .permutation import _perm_pvalue
from .evaluation import jmim_exponent_discount_only
from .info_theory import use_su_normalization, use_mi_miller_madow, use_jmim_aggregator


def fleuret_segment_counts(
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
    cached_cond_MIs: Optional[dict] = None,
    n_workers: int = 1,
    workers_pool: Optional[object] = None,
    parallel_kwargs: Optional[dict] = None,
    entropy_cache: Optional[dict] = None,
    extra_x_shuffling: bool = True,
    dtype: type = np.int32,
    base_seed: int = 0,
    toggles: Optional[tuple] = None,
) -> tuple:
    """``(nfailed, nchecked, entropy_cache)`` of the Fleuret permutation-confirmation: ``npermutations`` split across ``n_workers`` segments (independent-yet-reproducible per-segment offsets), the SU/Miller-Madow/JMIM scoring toggles forwarded, the segments run one after the other in the calling thread (no joblib), the stop at ``max_failed`` applying per segment. ``toggles`` is the ``(use_su, use_mm, use_jmim, jmim_discount_only)`` to forward; ``None`` reads the calling thread's settings (a prefetch thread has the defaults, so a caller that runs in one passes the settings it read in the fitting thread)."""
    if entropy_cache is None:
        entropy_cache = {}
    nfailed = 0

    # Read the SU toggle once here (Python-level) and thread it into every segment.
    _use_su = use_su_normalization()
    # N-F2: same for the Miller-Madow toggle - when MM relevance is active the redundancy CMI carries the SAME bias
    # correction (SU path excluded: it uses conditional_symmetric_uncertainty, not conditional_mi).
    _use_mm = use_mi_miller_madow() and not _use_su
    # S-F2: JMIM candidates are SCORED by joint-MI; the confirmation must use the SAME statistic, not the default CMIM
    # null. Thread the JMIM toggle so evaluate_gain confirms JMIM picks against the JMIM criterion.
    _use_jmim = use_jmim_aggregator()
    _jmim_discount_only = jmim_exponent_discount_only()
    if toggles is not None:
        _use_su, _use_mm, _use_jmim, _jmim_discount_only = toggles

    # Seeding is per PERMUTATION (outer base_seed plus the permutation's global number): the workers share the base seed and each starts at its offset, so the
    # draws are the same for any worker count.
    _worker_loads = list(distribute_permutations(npermutations=npermutations, n_workers=n_workers))
    _offsets = np.concatenate(([0], np.cumsum(_worker_loads)[:-1])) if _worker_loads else np.zeros(0, dtype=np.int64)
    _calls = [
        dict(
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
            cached_cond_MIs=dict(cached_cond_MIs or {}),
            entropy_cache=dict(entropy_cache or {}),
            extra_x_shuffling=extra_x_shuffling,
            dtype=dtype,
            base_seed=int(base_seed),
            perm_offset=int(_offsets[_widx]),
            use_su=_use_su,
            use_mm=_use_mm,
            use_jmim=_use_jmim,
            jmim_discount_only=_jmim_discount_only,
        )
        for _widx, worker_npermutations in enumerate(_worker_loads)
    ]
    res = [parallel_fleuret(**_kw) for _kw in _calls]

    nchecked = 0
    for worker_nfailed, worker_i, entropy_cache_dict in res:
        nfailed += worker_nfailed
        nchecked += worker_i
        for key, value in entropy_cache_dict.items():
            entropy_cache[key] = value

    return nfailed, nchecked, entropy_cache


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
    cached_cond_MIs: Optional[dict] = None,
    n_workers: int = 1,
    workers_pool: Optional[object] = None,
    parallel_kwargs: Optional[dict] = None,
    entropy_cache: Optional[dict] = None,
    extra_x_shuffling: bool = True,
    dtype: type = np.int32,
    base_seed: int = 0,
    toggles: Optional[tuple] = None,
) -> tuple:
    """Fleuret permutation-confirmation of one candidate: :func:`fleuret_segment_counts`, then the gain zeroed at ``max_failed`` failures and the confidence from the add-one p-value of the full budget. ``workers_pool`` and ``parallel_kwargs`` are accepted for the old signature and unused."""
    nfailed, nchecked, entropy_cache = fleuret_segment_counts(
        data_copy=data_copy,
        factors_nbins=factors_nbins,
        x=x,
        y=y,
        selected_vars=selected_vars,
        bootstrapped_gain=bootstrapped_gain,
        npermutations=npermutations,
        max_failed=max_failed,
        nexisting=nexisting,
        mrmr_relevance_algo=mrmr_relevance_algo,
        mrmr_redundancy_algo=mrmr_redundancy_algo,
        max_veteranes_interactions_order=max_veteranes_interactions_order,
        extra_knowledge_multipler=extra_knowledge_multipler,
        sink_threshold=sink_threshold,
        cached_cond_MIs=cached_cond_MIs,
        n_workers=n_workers,
        workers_pool=workers_pool,
        parallel_kwargs=parallel_kwargs,
        entropy_cache=entropy_cache,
        extra_x_shuffling=extra_x_shuffling,
        dtype=dtype,
        base_seed=base_seed,
        toggles=toggles,
    )
    if nfailed >= max_failed:
        bootstrapped_gain = 0.0

    # Permutation p-value for this candidate via the canonical ``_perm_pvalue`` estimator, which applies BOTH:
    #   1. Add-one (Davison & Hinkley 1997; Phipson & Smyth 2010): ``(1 + nfailed) / (1 + budget)`` - the observed gain
    #      is itself one draw under the null, so the Monte-Carlo p can never be exactly 0; a naive ``nfailed/nchecked``
    #      returns 0 on a null feature that never fails, spuriously reporting p=0 / confidence=1.
    #   2. ``full_budget`` de-biasing of the ``max_failed`` early-stop: the worker loop breaks as soon as ``nfailed``
    #      reaches ``max_failed``, so ``nchecked`` is data-dependent and the stopped ratio overstates the failure rate
    #      (it stops precisely where failures cluster). Passing the full ``npermutations`` budget as the denominator
    #      makes the reported p independent of WHERE the early break fired - an honest (conservative) estimate.
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
    cached_cond_MIs: Optional[dict] = None,
    entropy_cache: Optional[dict] = None,
    extra_x_shuffling: bool = True,
    dtype: type = np.int32,
    base_seed: int = 0,
    perm_offset: int = 0,
    use_su: bool = False,  # Threaded from get_fleuret_criteria_confidence_parallel.
    use_mm: bool = False,  # N-F2: threaded from get_fleuret_criteria_confidence_parallel (MM redundancy consistency).
    use_jmim: bool = False,  # S-F2: threaded from get_fleuret_criteria_confidence_parallel (JMIM confirmation statistic).
    jmim_discount_only: bool = False,  # threaded like use_jmim
) -> tuple:
    """Joblib worker: rebuild numba.typed.Dict from pickled Python dicts, run the njit core, return a Python dict for the parent's union."""
    data_copy = data.copy()

    entropy_cache_dict = numba.typed.Dict.empty(
        key_type=types.unicode_type,
        # Intentional float64: numba typed-dict value type must match the f64 values the @njit MI kernels store.
        value_type=types.float64,
    )
    python_dict_2_numba_dict(python_dict=entropy_cache, numba_dict=entropy_cache_dict)

    cached_cond_MIs_dict = numba.typed.Dict.empty(
        key_type=types.unicode_type,
        # Intentional float64: numba typed-dict value type must match the f64 values the @njit MI kernels store.
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
        perm_offset=perm_offset,
        use_su=use_su,
        use_mm=use_mm,
        use_jmim=use_jmim,
        jmim_discount_only=jmim_discount_only,
    )

    return nfailed, i, dict(entropy_cache_dict)
