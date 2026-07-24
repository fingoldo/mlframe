"""Per-thread candidate-evaluation driver carved out of ``evaluation.py`` (Tier E).

Holds the workload orchestration cluster: the batched-GPU conditional-MI cache pre-fill helpers
(``_gpu_cmi_prefill_enabled`` / ``_prefill_cond_MIs_gpu``) and the thread-worker entry points
(``evaluate_candidates`` -> ``_evaluate_candidates_inner``) that loop over a workload and call
``evaluate_candidate`` / ``handle_best_candidate`` (which remain in ``evaluation.py`` and are
imported below). Re-exported from ``evaluation.py`` so every import path still resolves.

See ``screen.py`` for the screening orchestrator that calls these functions.
"""
from __future__ import annotations

import logging
import os as _os
from typing import Any, Optional, Sequence

import numba
import numpy as np
from numba.core import types

from pyutilz.numbalib import python_dict_2_numba_dict
# Module-level import so cloudpickle can resolve tqdmu when the function ships to a joblib worker.
from pyutilz.system import tqdmu

from ._internals import LARGE_CONST
from ._numba_utils import arr2str
from .info_theory import (
    use_su_normalization, use_jmim_aggregator, get_bur_lambda,
    set_su_normalization, set_jmim_aggregator, set_bur_lambda,
    get_relaxmrmr_alpha, get_pid_synergy_bonus, get_cmi_perm_stop, get_cpt_test,
    set_relaxmrmr_alpha, set_pid_synergy_bonus, set_cmi_perm_stop, set_cpt_test,
    use_mi_miller_madow, set_mi_miller_madow,
    get_group_mi, set_group_mi,
)

# Helpers + the module-level JMIM cache stats deque live in the parent ``evaluation.py``. These are
# imported LAZILY inside ``_evaluate_candidates_inner`` (see ``# lazy: avoids import cycle`` there)
# rather than at module top, so the static module-level import graph has no ``_evaluation_driver ->
# evaluation -> _evaluation_driver`` cycle while runtime behaviour is identical.

logger = logging.getLogger(__name__)


def _gpu_cmi_prefill_enabled() -> bool:
    """Env kill-switch for the batched-GPU conditional-MI cache pre-fill (default ON).

    ``MLFRAME_MRMR_GPU_CMI=0`` forces the legacy scalar CPU path (no pre-fill); any other
    value (or unset) leaves the dispatcher in charge of CPU-vs-GPU routing. Read at call time
    (not import) so tests can toggle it per-run via the environment / monkeypatch.
    """
    return _os.environ.get("MLFRAME_MRMR_GPU_CMI", "1") != "0"


def _prefill_cond_MIs_gpu(
    workload,
    y,
    factors_data,
    factors_nbins,
    selected_vars,
    cached_cond_MIs,
    use_simple_mode,
    mrmr_relevance_algo,
    max_veteranes_interactions_order,
    dtype=np.int32,
    force=None,
):
    """Pre-populate ``cached_cond_MIs`` with batched-GPU ``I(X; Y | Z)`` so the ``@njit``
    ``evaluate_gain`` redundancy loop hits the cache instead of running the serial scalar
    ``conditional_mi`` (info_theory._entropy_kernels) per candidate.

    SCOPE — applies ONLY to the plain-CMI Fleuret branch that ``evaluate_gain`` actually takes:

      * ``mrmr_relevance_algo == 'fleuret'`` (the only branch keyed by ``arr2str(X)+'|'+arr2str(Z)``)
      * NOT ``use_su`` and NOT ``use_jmim`` (those branches skip the cache entirely)
      * ``max_veteranes_interactions_order == 1`` -- the batched kernel conditions on a SINGLE var
        ``Z=[z]``; order >= 2 mixes multi-element ``Z`` the kernel does not cover, so it is left
        on the exact scalar path (selection unchanged)
      * ``selected_vars`` non-empty AND ``not use_simple_mode`` -- otherwise the conditional branch
        is never reached
      * single-var (order-1) candidates only (``len(X) == 1``)

    For every selected var ``Z=[z]`` it calls the dispatch ONCE over all order-1 candidates and
    writes the RAW CMI (the ``nexisting`` exponent is applied at READ time in ``evaluate_gain``;
    do NOT pre-apply). Key format replicates ``evaluate_gain`` EXACTLY via the same ``@njit``
    ``arr2str`` -- a mismatch would silently disable the cache.

    SELECTION-EQUIVALENCE (not bit-identical): the GPU CMI reduces the four entropies from the joint
    counts on-device (``_cmi_cuda._entropy_from_counts_axis``), parity ~1e-9 to the CPU scalar
    ``conditional_mi`` (a different float64 reduction order over the nonzero bins). This is the SAME
    parity bound the rest of the GPU MI path carries and is treated identically: a ranking flip needs two
    candidates whose Fleuret gains sit within ~1e-9 -- a pathological near-tie. Crucially the prefill is
    ALL-OR-NOTHING per round: if ANY candidate is multi-element it returns 0 and the WHOLE round stays on
    the exact scalar CPU path (see the ``len(X) != 1`` guard), and within a round it writes every order-1
    ``(cand, z)`` CMI, so the candidates compared against each other use a CONSISTENT backend rather than a
    GPU/CPU mix. A bit-exact alternative (D2H the integer counts + CPU-reduce) would defeat the entire
    point of the batched kernel for a ~1e-9 P2 near-tie, so it is intentionally not taken.

    Routing is delegated to ``conditional_mi_batched_dispatch`` (size/HW gate via the
    kernel_tuning_cache; GPU only when beneficial+available). ANY failure falls back silently:
    the cache is simply left un-prefilled and ``evaluate_gain`` runs the scalar path. Determinism
    preserved (the kernel is bit-parity with the CPU loop; an empty pre-fill is a pure no-op)."""

    # Gate: only the plain Fleuret CMI branch, single-var Z, conditional path actually taken.
    if not _gpu_cmi_prefill_enabled():
        return 0
    if str(mrmr_relevance_algo) != "fleuret":
        return 0
    if use_simple_mode or not selected_vars:
        return 0
    if int(max_veteranes_interactions_order) != 1:
        return 0
    # SU / JMIM branches bypass the cache -> a pre-fill would be dead weight (and SU is a different
    # quantity). Read the thread-locals the kernel respects.
    if use_su_normalization() or use_jmim_aggregator():
        return 0

    try:
        from .info_theory._cmi_cuda import conditional_mi_batched_dispatch

        # Order-1 single-var candidates only; collect their column indices.
        cand_indices = []
        for _cand_idx, X, _nexisting in workload:
            if len(X) != 1:
                return 0  # any multi-element candidate -> abandon pre-fill, keep scalar path
            cand_indices.append(int(X[0]))
        if not cand_indices:
            return 0

        cand_indices_arr = np.asarray(cand_indices, dtype=np.int64)
        y_index = int(np.asarray(y).ravel()[0])
        factors_nbins_arr = np.asarray(factors_nbins)

        # Candidate-key strings are z-independent -- build them once, reuse across every z below.
        cand_keys = [arr2str(np.asarray([ci], dtype=np.int64)) for ci in cand_indices]

        # Shadow the numba typed-dict's keys in a plain Python set for this call's membership checks
        # (cProfile, 2026-07-16 wellbore fit: numba.typed.typeddict.__contains__ cost 40.5s / 1.6M calls,
        # ~1613570 of them from the missing_pos scan below -- EVERY (candidate, z) membership check crosses
        # the Python<->numba boundary individually. A python 'x in str_key' against a numba typed Dict pays
        # real per-call marshaling overhead: measured ~25us/check vs ~0.2us against a plain python set/dict,
        # ~120x. One bulk `set(cached_cond_MIs.keys())` per call (cost scales with dict SIZE, done once)
        # replaces N_missing_checks cross-boundary lookups (cost scales with candidates x selected_vars,
        # done every round) -- a clear net win once check-count exceeds dict-size, which happens quickly as
        # selected_vars grows across rounds. Read-only snapshot; the real numba dict below is still the one
        # ``evaluate_gain`` reads from and the one every write (here + evaluation.py's scalar-path cache
        # fill) targets, so no staleness risk -- a key written by that OTHER path after this snapshot is
        # simply treated as "missing" here too (re-dispatched, then overwritten with the same value at
        # write time -- redundant work, never a correctness bug: the CMI value is a pure function of
        # (cand, z), not of which path filled it first).
        cached_keys_shadow = set(cached_cond_MIs.keys())
        n_written = 0
        for z in selected_vars:
            z_idx = int(z)
            z_key_arr = np.asarray([z_idx], dtype=np.int32)
            z_str = arr2str(z_key_arr)
            # Dispatch ONLY the (cand, z) pairs missing from the cache. The greedy loop calls this
            # prefill EVERY round with the full selected_vars list, but each round adds just ONE new z --
            # every (surviving-candidate, older-z) CMI was already computed and cached in a prior round,
            # and the old code recomputed the ENTIRE (candidates x z) matrix per round only to discard
            # the already-cached values at write time (measured on the wellbore-100k profile: 4375
            # dispatches / ~500s of _cpu_cmi_loop, the single largest FE-phase hotspot, mostly redundant).
            missing_pos = [j for j, ck in enumerate(cand_keys) if (ck + "|" + z_str) not in cached_keys_shadow]
            if not missing_pos:
                continue
            missing_arr = cand_indices_arr[np.asarray(missing_pos, dtype=np.int64)]
            cmi_vec = conditional_mi_batched_dispatch(
                factors_data=factors_data,
                cand_indices=missing_arr,
                y_index=y_index,
                z_index=z_idx,
                factors_nbins=factors_nbins_arr,
                dtype=dtype,
                force=force,
            )
            for j, val in zip(missing_pos, cmi_vec):
                new_key = cand_keys[j] + "|" + z_str
                cached_cond_MIs[new_key] = float(val)
                cached_keys_shadow.add(new_key)
                n_written += 1
        return n_written
    except Exception as _exc:
        logger.debug("gpu cmi pre-fill skipped (%s); scalar CPU path used", _exc)
        return 0


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
    classes_y: Optional[np.ndarray] = None,
    freqs_y: Optional[np.ndarray] = None,
    freqs_y_safe: Optional[np.ndarray] = None,
    use_gpu: bool = True,
    cached_MIs: Optional[dict] = None,
    cached_confident_MIs: Optional[dict] = None,
    cached_cond_MIs: Optional[dict] = None,
    cached_jmim_MIs: Optional[dict] = None,
    entropy_cache: Optional[dict] = None,
    mrmr_relevance_algo: str = "fleuret",
    mrmr_redundancy_algo: str = "fleuret",
    max_veteranes_interactions_order: int = 1,
    dtype=np.int32,
    max_runtime_mins: Optional[float] = None,
    start_time: Optional[float] = None,
    min_relevance_gain: Optional[float] = None,
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
    relaxmrmr_alpha: float = 0.0,
    pid_synergy_bonus: float = 0.0,
    cmi_perm: tuple = (False, 0.05, 100),
    cpt: tuple = (False, 200),
    mi_miller_madow: bool = False,
    group_mi=None,
    random_seed: Optional[int] = None,
) -> tuple:
    """Thread-worker entry point: re-publish the Wave 8 ``threading.local`` toggles (SU/JMIM/Bur-lambda/relax-mrmr/PID/CMI-perm/CPT/Miller-Madow/group-MI) into this worker thread, run ``_evaluate_candidates_inner`` over the workload, then restore the prior thread-local values in ``finally`` so a reused worker doesn't leak this call's settings into the next dispatch."""
    # Worker-thread re-publish of Wave 8 toggles (iter 5 fix). The
    # try/finally guarantees we don't pollute the worker thread's locals
    # if the same worker is re-used for a subsequent dispatch with
    # different settings. mi_miller_madow is forwarded alongside the other six:
    # threading.local does not cross into joblib workers, so without it the
    # mi_correction='miller_madow' bias-correction was a silent no-op in the
    # parallel greedy loop (mi_or_su / the class-MI kernels consult this toggle).
    _prev_su = use_su_normalization()
    _prev_jmim = use_jmim_aggregator()
    _prev_bur = get_bur_lambda()
    _prev_relax = get_relaxmrmr_alpha()
    _prev_pid = get_pid_synergy_bonus()
    _prev_cmi = get_cmi_perm_stop()
    _prev_cpt = get_cpt_test()
    _prev_mm = use_mi_miller_madow()
    _prev_gmi = get_group_mi()
    set_group_mi(group_mi)
    set_su_normalization(bool(use_su))
    set_jmim_aggregator(bool(use_jmim))
    set_bur_lambda(float(bur_lambda))
    set_relaxmrmr_alpha(float(relaxmrmr_alpha))
    set_pid_synergy_bonus(float(pid_synergy_bonus))
    set_cmi_perm_stop(bool(cmi_perm[0]), float(cmi_perm[1]), int(cmi_perm[2]))
    set_cpt_test(bool(cpt[0]), int(cpt[1]))
    set_mi_miller_madow(bool(mi_miller_madow))
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
            cached_cond_MIs=cached_cond_MIs, cached_jmim_MIs=cached_jmim_MIs,
            entropy_cache=entropy_cache,
            mrmr_relevance_algo=mrmr_relevance_algo,
            mrmr_redundancy_algo=mrmr_redundancy_algo,
            max_veteranes_interactions_order=max_veteranes_interactions_order,
            dtype=dtype, max_runtime_mins=max_runtime_mins,
            start_time=start_time, min_relevance_gain=min_relevance_gain,
            verbose=verbose, ndigits=ndigits,
            use_simple_mode=use_simple_mode,
            random_seed=random_seed,
        )
    finally:
        set_su_normalization(_prev_su)
        set_jmim_aggregator(_prev_jmim)
        set_bur_lambda(_prev_bur)
        set_relaxmrmr_alpha(_prev_relax)
        set_pid_synergy_bonus(_prev_pid)
        set_cmi_perm_stop(_prev_cmi[0], _prev_cmi[1], _prev_cmi[2])
        set_cpt_test(_prev_cpt[0], _prev_cpt[1])
        set_mi_miller_madow(_prev_mm)
        set_group_mi(_prev_gmi)


def _evaluate_candidates_inner(
    workload, y, best_gain, factors_data, factors_nbins, factors_names,
    partial_gains, selected_vars, baseline_npermutations,
    classes_y=None, freqs_y=None, freqs_y_safe=None, use_gpu=True,
    cached_MIs=None, cached_confident_MIs=None, cached_cond_MIs=None,
    cached_jmim_MIs=None,
    entropy_cache=None, mrmr_relevance_algo="fleuret",
    mrmr_redundancy_algo="fleuret", max_veteranes_interactions_order=1,
    dtype=np.int32, max_runtime_mins=None, start_time=None,
    min_relevance_gain=None, verbose=1, ndigits=5, use_simple_mode=True,
    random_seed=None,
) -> tuple:
    """Score every candidate in ``workload`` (serial loop over ``evaluate_candidate``), tracking the running best gain/candidate and updating ``partial_gains``/the MI caches in place."""
    # lazy: parent-defined helpers + cache deque, imported here to avoid the
    # _evaluation_driver <-> evaluation module-level import cycle.
    from .evaluation import evaluate_candidate, handle_best_candidate, _JMIM_CACHE_STATS

    best_gain = -LARGE_CONST
    best_candidate: Optional[Sequence] = None
    expected_gains: dict = {}

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

    # 2026-06-19: JMIM joint-MI cache, built the SAME way as cached_cond_MIs (numba typed
    # dict seeded from the optional python dict the caller threads through; merged back to a
    # plain python dict at the return boundary so nothing non-picklable escapes onto an
    # instance). Keyed on arr2str({X} u Z); only the JMIM branch of evaluate_gain touches it.
    cached_jmim_MIs_dict = numba.typed.Dict.empty(
        key_type=types.unicode_type,
        value_type=types.float64,
    )
    python_dict_2_numba_dict(python_dict=cached_jmim_MIs or {}, numba_dict=cached_jmim_MIs_dict)
    # 1-element int64 hit counter, mutated in place by the @njit evaluate_gain on every
    # JMIM cache read-hit (an array, since @njit cannot mutate a python scalar by ref).
    jmim_hit_counter = np.zeros(1, dtype=np.int64)

    # Batched-GPU conditional-MI cache pre-fill (default ON; env kill-switch MLFRAME_MRMR_GPU_CMI=0).
    # Realises the GPU win: pre-populate the LOCAL numba cond-MI dict with batched I(X; Y | Z) so the
    # @njit evaluate_gain loop hits the cache and skips the serial scalar conditional_mi. Writing into
    # the per-call numba dict (NOT the shared python ``cached_cond_MIs``, which several threading
    # workers alias) avoids a "dict changed size during iteration" race. Bit-parity kernel => selection
    # unchanged; any failure / non-eligible regime is a silent no-op (scalar path).
    _prefill_cond_MIs_gpu(
        workload=workload,
        y=y,
        factors_data=factors_data,
        factors_nbins=factors_nbins,
        selected_vars=selected_vars,
        cached_cond_MIs=cached_cond_MIs_dict,
        use_simple_mode=use_simple_mode,
        mrmr_relevance_algo=mrmr_relevance_algo,
        max_veteranes_interactions_order=max_veteranes_interactions_order,
        dtype=dtype,
    )

    classes_y_safe = classes_y.copy()

    # RelaxMRMR: y_col/k_y and every already-selected column's materialize_var
    # result depend only on (y, factors_data, factors_nbins, dtype, selected_vars), none of which
    # change across this workload's per-candidate loop -- selected_vars is a fixed list threaded
    # through unmutated (see the loop below). Hoist the SAME condition evaluate_candidate gates the
    # RelaxMRMR block on and compute once per greedy iteration instead of once per candidate. Gated
    # behind relaxmrmr_alpha>0 (default OFF); any failure here falls through to per-candidate
    # recomputation inside evaluate_candidate (its own precomputed-args check no-ops to None).
    _relax_y_col: Optional[np.ndarray] = None
    _relax_k_y: Optional[int] = None
    _relax_sel_cols: Optional[list] = None
    _relax_sel_nbins: Optional[list] = None
    if get_relaxmrmr_alpha() > 0.0 and selected_vars and not use_simple_mode and str(mrmr_relevance_algo) == "fleuret":
        try:
            from .evaluation import _materialize_var
            _relax_y_col, _relax_k_y = _materialize_var(factors_data, y, factors_nbins, dtype=dtype)
            _relax_sel_cols, _relax_sel_nbins = [], []
            for _z in selected_vars:
                _zc, _zk = _materialize_var(factors_data, _z, factors_nbins, dtype=dtype)
                _relax_sel_cols.append(_zc)
                _relax_sel_nbins.append(_zk)
        except Exception:
            _relax_y_col = _relax_k_y = _relax_sel_cols = _relax_sel_nbins = None

    for cand_idx, X, nexisting in tqdmu(workload, leave=False, desc="Thread Candidates", disable=not verbose):

        current_gain, _sink_reasons = evaluate_candidate(
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
            expected_gains=expected_gains,  # type: ignore[arg-type]  # evaluate_candidate accepts dict or ndarray by int-key indexing; annotation says ndarray
            selected_vars=selected_vars,
            cached_MIs=cached_MIs,
            cached_confident_MIs=cached_confident_MIs,
            cached_cond_MIs=cached_cond_MIs_dict,
            cached_jmim_MIs=cached_jmim_MIs_dict,
            jmim_hit_counter=jmim_hit_counter,
            entropy_cache=entropy_cache_dict,
            verbose=verbose,
            ndigits=ndigits,
            dtype=dtype,
            use_simple_mode=use_simple_mode,
            _relax_y_col=_relax_y_col,
            _relax_k_y=_relax_k_y,
            _relax_sel_cols=_relax_sel_cols,
            _relax_sel_nbins=_relax_sel_nbins,
            random_seed=random_seed,
        )

        best_gain, best_candidate, run_out_of_time = handle_best_candidate(
            current_gain=current_gain,
            best_gain=best_gain,
            X=X,
            best_candidate=best_candidate,  # type: ignore[arg-type]  # None on the first iteration; handle_best_candidate only reads it when replacing, so this is safe
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
    # 2026-06-19: convert the JMIM typed dict back to a plain python dict at the
    # boundary (mirrors cached_cond_MIs) so nothing non-picklable can leak onto an
    # instance. The cache is currently per-call (not threaded through the driver's
    # 7-tuple return, which other modules unpack positionally), so it is discarded
    # here -- its only effect is within-call memoisation of repeated {X} u Z keys.
    # The final size is published to the module-level stats deque purely for
    # observability (the parity test reads it to PROVE the cache populated/hit).
    # bench (2026-06-19, n=4000 p=40 order-2 JMIM): cache delivers ~117k HITS over ~1.26M
    # entries -> real cross-round reuse; selection byte-identical (test_jmim_cache_parity).
    # WALL-TIME A/B (3 seeds, distinct frames per arm to dodge the re-fit content cache, cache ON vs a
    # forced-miss kill path): 183.2s on vs 190.0s off => ~1.04x (-6.8s, ~4%) -- a real but MODEST wall win,
    # since the avoided mi() calls are a small slice of the full JMIM fit (discretize / relevance / FE /
    # stability-vote dominate). Net: keep it (positive + harmless at order 1), but it is not a headline lever.
    # At interactions_max_order==1 the (current_gain, last_checked_k) resume already evaluates
    # each (X, Z) once across the whole fit, so the cache populates but never HITS (n=6000
    # p=150: ~456k entries, 0 hits) -- harmless, kept for the order>=2 win.
    cached_jmim_MIs = dict(cached_jmim_MIs_dict)
    if use_jmim_aggregator():
        _JMIM_CACHE_STATS.append({"size": len(cached_jmim_MIs), "hits": int(jmim_hit_counter[0])})

    return best_gain, best_candidate, partial_gains, expected_gains, cached_MIs, cached_cond_MIs, entropy_cache


# X_EFFICIENCY_ARCHITECTURE-1 fix: find_best_partial_gain carved out of
# evaluation.py (which had crept to 1011 LOC, over the repo's enforced hard 1000-LOC CI gate) into this
# already-established sibling. Re-exported from evaluation.py so every import path still resolves.
def find_best_partial_gain(
    partial_gains: dict, failed_candidates: set, added_candidates: set, candidates: list, selected_vars: list, skip_indices: tuple = (),
    dcd_state=None,
) -> "tuple[float, Any]":
    """Find the highest-scoring already-evaluated-but-not-yet-confirmed candidate in ``partial_gains`` (used to redirect the confirmation loop to the next-best option when the current top candidate fails confirmation), excluding failed/added/skip_indices candidates and any candidate DCD has since pruned."""
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
    _should_be_pruned: Any = None
    if dcd_state is not None:
        try:
            from ._dynamic_cluster_discovery import should_be_pruned as _should_be_pruned
        except Exception as exc:
            # SCREEN_CONFIRM_A-7 fix: log so a genuine import-time bug is not
            # indistinguishable from "DCD not configured".
            logger.debug("mrmr: importing should_be_pruned failed; DCD pruning disabled for this call: %r", exc, exc_info=True)
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
