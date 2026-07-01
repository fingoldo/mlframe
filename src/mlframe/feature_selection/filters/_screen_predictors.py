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
from ._screen_predictors_prescreen import cardinality_prescreen, compute_fdr_gain_floor
from .evaluation import get_candidate_name
from .info_theory import merge_vars

logger = logging.getLogger(__name__)


def _short_name(name, maxlen: int = 28) -> str:
    """Truncate a (possibly long engineered) feature name for live progress-bar display.

    Keeps the head and tail so both the operation and the operand stay legible
    (``mul(log(c),sin(d))`` -> ``mul(log(..in(d))``). Robust to non-str input.
    """
    try:
        s = str(name)
    except Exception:
        return "?"
    if len(s) <= maxlen:
        return s
    head = (maxlen - 2) // 2
    tail = maxlen - 2 - head
    return s[:head] + ".." + s[-tail:]


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
    dtype: type = np.int32,
    random_seed: int = None,
    # ONE shared FE subsample (2026-06-25, "score the screen on the same rows"). When supplied, the
    # order-1 relevance SWEEP + its maxT FDR floor are computed on THESE rows (consistent estimator scale,
    # and the full-n permutation work disappears); the RETURNED target encodings (classes_y / freqs_y)
    # are recomputed at FULL n so the downstream FE pipeline stays row-aligned. None -> full-n screen.
    subsample_idx: np.ndarray = None,
    use_gpu: bool = False,
    n_workers: int = 1,
    # confidence
    # Statistical defaults aligned with the MRMR constructor (mrmr.py) so a direct ``screen_predictors``
    # caller -- and any consumer that does not override every knob (e.g. ad-hoc screening) -- gets the same
    # behaviour MRMR.fit produces, rather than the far-stricter legacy 1000/100 permutation counts that made
    # standalone screening ~300x slower and over-reject. MRMR.fit still explicitly overrides each of these.
    min_occupancy: int = None,
    min_nonzero_confidence: float = 0.99,
    full_npermutations: int = 3,
    baseline_npermutations: int = 2,
    # 2026-06-02 RC2 — sample-size-aware Fleuret confirmation threshold (rows
    # per occupied cell of the conditioning joint). Below it the conditional-MI
    # permutation gate is finite-sample unreliable and ``confirm_candidate``
    # falls back to a marginal-MI permutation test. 0.0 = always use the strict
    # conditional test (legacy). Threaded into ``ScreenContext``.
    fe_confirm_undersample_rows_per_cell: float = 5.0,
    # stopping conditions
    min_relevance_gain: float = 0.0001,
    # 2026-05-30: diminishing-returns stop. Stops greedy selection when the
    # current candidate's gain falls below this fraction of the FIRST-selected
    # feature's gain. Catches "trailing noise" leakage on imbalanced y where
    # tiny-but-statistically-positive gains squeeze past min_relevance_gain
    # (e.g. Layer 13: signal gain 0.0176, noise gain 0.0004 - both clear the
    # H(y)-relative floor at 1% imbalance, but noise is 2.5% of signal).
    # 0.0 disables; default 0.05 = stop once gain drops below 5% of first
    # gain. Only applies from the second selected feature onward.
    min_relevance_gain_relative_to_first: float = 0.05,
    # 2026-05-30: Miller-Madow MI bias correction at the selection gate.
    # Plug-in mutual information OVERESTIMATES MI for high-cardinality
    # features (Paninski 2003, Miller 1955). For a binned feature with
    # |X| bins, target with |Y| classes, and n samples the plug-in MI
    # picks up a bias of ~(|X|-1)*(|Y|-1)/(2n). On a 1200-level user_id
    # at n=2500 with binary y that's ~0.24 nats - enough to make pure
    # noise outrank real numeric signal (Layer 10 seed=101 hijack:
    # user_id raw gain 0.328, after MM correction 0.088; num_signal_1
    # raw 0.187 -> corrected 0.185; the corrected ordering puts the
    # real signal first and demotes user_id to #3, where the relative-
    # gain floor then excludes it). True = subtract MM bias from gains
    # at the floor comparison only (does NOT mutate mrmr_gains_ which
    # remains the raw plug-in value for downstream consumers).
    cardinality_bias_correction: bool = True,
    max_consec_unconfirmed: int = 10,
    max_runtime_mins: float = None,
    interactions_min_order: int = 1,
    interactions_max_order: int = 1,
    interactions_order_reversed: bool = False,
    max_veteranes_interactions_order: int = 1,
    only_unknown_interactions: bool = False,
    # Confirmation-step cardinality cutoff. ``None`` falls back to ``MAX_CONFIRMATION_CAND_NBINS``; ``MRMR.fit`` overrides with ``quantization_nbins ** interactions_max_order * 2``.
    max_confirmation_cand_nbins: int = None,
    # When screening returns zero selected_vars, legacy FE fell back to running on ALL features. False skips FE instead (safer default: FE on empty screen amplifies noise). Aligned with the MRMR ctor default (False).
    fe_fallback_to_all: bool = False,
    # verbosity and formatting
    verbose: int = 0,
    ndigits: int = 5,
    parallel_kwargs: dict = None,
    stop_file: str = None,
    # Aligned with the MRMR ctor (False): full Fleuret conditional-MI redundancy is the point of MRMR. True is the faster dedup-free path on very wide pools.
    use_simple_mode: bool = False,
    # ``engineered_lineage`` -- mapping ``{engineered_col_idx: frozenset(parent_indices)}``. When set, k-way candidate enumeration skips combinations of an engineered
    # column with one of its own parents (e.g. ``(orig_i, kway(orig_i, orig_j))`` is redundant since the engineered col already contains orig_i's info). Threaded
    # through ``should_skip_candidate``. ``None`` preserves legacy behaviour.
    engineered_lineage: dict = None,
    # 2026-05-30 Wave 9 — Dynamic Cluster Discovery (DCD) config dict. When
    # not None, DCD is active for this screen call. Keys: enable, tau_cluster,
    # distance, cluster_size_threshold, swap_gain_threshold, swap_method,
    # pairwise_cache_max, min_cluster_size, max_cluster_size, X_raw,
    # quantization_method, quantization_nbins, quantization_dtype, factors_cols.
    # None preserves pre-Wave-9 behaviour bit-stable.
    dcd_config: dict = None,
    # 2026-05-31 Layer 43 (PART A) — host-MRMR engineered_recipes dict (name ->
    # EngineeredRecipe). Threaded into ``commit_swap`` so the DCD PC1 aggregate
    # gets registered as a replayable recipe. Pre-fix it was hardcoded to None
    # at the commit_swap callsite -> aggregate appeared in ``selected_vars`` /
    # ``cols`` but the remap in _mrmr_fit_impl.py dropped it from
    # ``_engineered_recipes_`` so ``get_feature_names_out`` silently lost it.
    engineered_recipes: dict = None,
    # 2026-06-02 — directed-FE tie-break (see ScreenContext.raw_feature_names).
    # ``raw_feature_names`` is the set of ORIGINAL (pre-FE) input column names;
    # any candidate whose ``factors_names`` entry is not in it is engineered.
    # On a near-tie in selection gain (within ``prefer_engineered_rel_eps``
    # relative tolerance) the greedy pick prefers the engineered candidate over a
    # raw one -- deterministic, and the whole point of directed FE: surface the
    # nonlinear combination, not its raw parent (which a shallow downstream can't
    # use). ``None`` raw-name set falls back to the syntactic ``(``/``__``
    # heuristic; rel-eps ``0.0`` restores the legacy pure-index tie-break.
    raw_feature_names: object = None,
    # 2026-06-02 RC1: widened 0.01 -> 0.15. On non-monotone targets (e.g.
    # y=sign(x1^2-1)) the binned-MI of the raw parent x1 edges out its engineered
    # linearizer x1__He2 by more than 1% (estimation noise; in EXACT MI they are
    # equal because He2 is a deterministic function of x1), so a 1% band left the
    # raw parent winning and the engineered linearizer pruned -- the downstream a
    # linear model needs (x1^2-1) was lost (AUC ~0.5 vs ~0.99). A 0.15 band lets
    # the engineered linearizer win over its near-tied raw parent. Still gated to
    # the leading band + engineered-only promotion (clear raw winners untouched).
    prefer_engineered_rel_eps: float = 0.15,
    # 2026-06-03 — Westfall-Young maxT permutation-null gain floor. In a wide
    # candidate pool (embedding / TF-IDF matrices, p >> sqrt(n)) the MAX marginal
    # MI over p pure-noise columns is a positive order statistic that grows with
    # p; per-candidate Miller-Madow correction centres each column's EXPECTED
    # bias near zero but cannot remove this max-over-p selection inflation, so the
    # best-of-p noise clears the abs/rel floors by chance and the greedy admits a
    # noise cloud (Layer-20 p=500: 15 noise dims pass a <=15 bound). The null
    # shuffles y K times, records the per-shuffle MAX corrected marginal MI over
    # the pool, and floors order-1 selection at the q-th quantile of that
    # distribution - the chance ceiling for THIS pool. SELF-GATING: tiny at small
    # p (keeps weak genuine signals), large at high p (rejects the noise cloud).
    # Applied when the pool has >= ``screen_fdr_min_features`` candidates (wide pool: embedding / TF-IDF best-of-p bias) OR when a NARROW pool meets the high-cardinality-target
    # gate (``screen_fdr_target_oversplit_ratio`` / ``screen_fdr_min_rows_per_joint_cell``). The narrow-pool gate catches a distinct finite-sample-bias regime: a heavy-tailed
    # (log-normal) regression target whose quantile binning yields a high-cardinality target (~10 equal-frequency bins, matching feature cardinality), lifting pure-noise columns
    # past the abs/rel gain floors after the genuine signals are picked. The gate fires ONLY when the target is high-cardinality (nbins_y >= median feature nbins -- a continuous
    # regression target, not a low-card classification one) AND the (X,y) joint table is dense enough that the floor is itself reliable -- so a dense weak-signal regression pool
    # like sklearn diabetes (nbins_y=10 but ~3.3 rows per joint cell at n=330) keeps the floor OFF and preserves its 10 weak features, while lognormal (~25-50 rows per joint cell)
    # fires. The original ratio=3 keyed on the MDLP ~30-bin over-split that the 2026-06-10 target-rebin guard now removes. See ``target_oversplit_floor_applies`` in
    # ``_permutation_null.py``. ``screen_fdr_null_permutations=0`` disables. Default 200 (raised from 25): the floor is the 95th percentile of a per-shuffle MAX, an
    # extreme upper-tail order statistic whose K=25 estimate is high-variance (~1.25 draws above the quantile); 200 draws stabilise the floor several-fold run-to-run
    # (bench ``_benchmarks/bench_maxt_floor_stability.py``) -- a lower-variance noise floor is the correct behavior, the rescore cost stays sub-second at production widths.
    screen_fdr_null_permutations: int = 200,
    screen_fdr_null_quantile: float = 0.95,
    screen_fdr_min_features: int = 30,
    screen_fdr_target_oversplit_ratio: float = 1.0,
    screen_fdr_min_rows_per_joint_cell: float = 8.0,
    # When MRMR.fit re-screens after a feature-engineering step (confirm-rescreen
    # loop), the DCDState from the prior pass is threaded back in here so cluster
    # discovery accumulates across passes instead of being rebuilt empty each
    # time (which dropped the screen-1 dup cluster from the published summary).
    existing_dcd_state: object = None,
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

    # 2026-05-30 Wave 9.1 fix (loop iter 25): two bugs collapsed into one
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

    # RNG hygiene. numpy: the modern screening / permutation / fleuret kernels
    # each thread ``random_seed`` explicitly (inline-LCG Fisher-Yates, or their
    # own local ``default_rng``), so we no longer seed or snapshot the
    # process-global MT19937 state here -- the prior ``np.random.seed`` mutated
    # a process-wide generator (racy under threads/joblib workers and a hidden
    # side effect on the caller's state). Instead we build a LOCAL Generator
    # from ``random_seed`` for any numpy-side draw; the caller's global
    # ``np.random`` state is left untouched. All current downstream numpy draws
    # are threaded via ``random_seed`` into their own local Generators / LCG, so
    # no direct global-numpy draw remains on this path.
    #
    # numba/cupy: these expose no portable Generator threading for the njit
    # kernels, so their global seeds are still set for the screening duration
    # and restored in ``finally`` with a fresh-entropy reseed (Wave 49), which
    # is byte-indistinguishable to any downstream consumer.
    _numba_restore_seed = None
    _cp_restore_seed = None
    if random_seed is not None:
        # Capture a fresh entropy-derived seed to restore numba/cupy with on
        # finally; mathematically equivalent (from the consumer's view) to
        # "the seed they would have had if no inner seed call had fired".
        import os as _os, struct as _struct
        _numba_restore_seed = _struct.unpack("<Q", _os.urandom(8))[0]
        # Only capture cupy restore-seed when GPU path is actually requested.
        # Otherwise the finally block below would import cupy purely to
        # "restore" a seed that was never set, triggering cupy's Windows
        # DLL-load _diagnose_import_error recursion on hosts where cupy
        # is installed but its CUDA stack is broken (cuTENSOR/cuBLAS
        # missing) -- that recursion blows the C stack in batch pytest
        # contexts and tears down the test runner.
        if use_gpu:
            _cp_restore_seed = _struct.unpack("<Q", _os.urandom(8))[0]
        set_numba_random_seed(random_seed)
        # 2026-05-30 Wave 9.1 fix (loop iter 25): the prior
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
            except Exception:
                # CuPy absent -> CPU fallback below. Also tolerate the legacy global cuRAND host generator
                # failing to init (CURAND_STATUS_INITIALIZATION_FAILED on some driver/lib combos): seeding
                # it is best-effort reproducibility and the GPU kernels use the modern Generator API anyway,
                # so a seed failure must not crash the whole screen.
                pass

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
        # 2026-06-19: JMIM joint-MI cache, built once per fit alongside cached_cond_MIs so
        # it persists across greedy rounds (the {X} u Z multiset key recurs as selected_vars
        # grows). Plain int64 array counts cache HITS for observability. Both are forwarded
        # through ScreenContext to evaluate_candidate; never pickled onto the instance.
        cached_jmim_MIs = numba.typed.Dict.empty(
            key_type=types.unicode_type,
            value_type=types.float64,
        )
        jmim_hit_counter = np.zeros(1, dtype=np.int64)

        # 2026-05-30 Wave 9 — Dynamic Cluster Discovery state construction.
        # Built only when ``dcd_config`` is provided AND ``enable=True``. The
        # state lives on the screen-local namespace (NOT thread-local) so the
        # joblib parallel-backend path remains safe per Critic1/F fix.
        dcd_state = None
        if dcd_config is not None and dcd_config.get("enable", False):
            try:
                from ._dynamic_cluster_discovery import make_dcd_state
                # Layer 47 (2026-05-31): tau_cluster passes through as-is so
                # the literal ``'auto'`` sentinel reaches make_dcd_state's
                # calibration branch. Numeric values are float()-coerced.
                _raw_tau = dcd_config.get("tau_cluster", 0.7)
                _tau_arg = (
                    _raw_tau if isinstance(_raw_tau, str) else float(_raw_tau)
                )
                dcd_state = make_dcd_state(
                    X_raw=dcd_config.get("X_raw"),
                    factors_data=factors_data,
                    factors_nbins=factors_nbins,
                    cols=list(factors_names) if factors_names is not None else None,
                    nbins=factors_nbins,
                    target_indices=np.asarray(y, dtype=np.int64),
                    quantization_method=dcd_config.get("quantization_method", "quantile"),
                    quantization_nbins=int(dcd_config.get("quantization_nbins", 10)),
                    quantization_dtype=dcd_config.get("quantization_dtype", np.int32),
                    tau_cluster=_tau_arg,
                    distance=str(dcd_config.get("distance", "su")),
                    cluster_size_threshold=int(dcd_config.get("cluster_size_threshold", 4)),
                    swap_gain_threshold=float(dcd_config.get("swap_gain_threshold", 0.05)),
                    swap_method=str(dcd_config.get("swap_method", "pca_pc1")),
                    pairwise_cache_max=int(dcd_config.get("pairwise_cache_max", 50_000)),
                    min_cluster_size=int(dcd_config.get("min_cluster_size", 2)),
                    max_cluster_size=int(dcd_config.get("max_cluster_size", 12)),
                    # Layer 47 (2026-05-31): forward auto-tau calibration knobs.
                    tau_calibration_n_pairs=int(dcd_config.get(
                        "tau_calibration_n_pairs", 100,
                    )),
                    tau_calibration_seed=int(dcd_config.get(
                        "tau_calibration_seed", 0,
                    )),
                    existing_state=existing_dcd_state,
                )
            except Exception as _dcd_init_exc:
                if verbose:
                    logger.warning(
                        "DCD init failed silently; falling back to legacy path: %r",
                        _dcd_init_exc,
                    )
                dcd_state = None

        # SCREEN SUBSAMPLE ("score on the shared rows"): slice the working factors/targets to the fit's
        # one shared draw so the relevance sweep + maxT FDR floor below run on ~30k rows (consistent with
        # the candidates they gate; no full-n permutation work). ``_screen_full_factors`` keeps the full
        # array so the RETURNED encodings are recomputed at full n (row-aligned for the FE pipeline).
        _screen_full_factors = None
        if subsample_idx is not None:
            try:
                _sidx = np.asarray(subsample_idx)
                if _sidx.ndim == 1 and 0 < _sidx.shape[0] < len(factors_data) and int(_sidx.max()) < len(factors_data):
                    _sidx = _sidx.astype(np.int64, copy=False)
                    _same_t = targets_data is factors_data
                    _screen_full_factors = factors_data
                    factors_data = factors_data[_sidx]
                    if _same_t:
                        targets_data = factors_data
                    elif targets_data is not None and len(targets_data) == len(_screen_full_factors):
                        targets_data = targets_data[_sidx]
            except Exception:
                _screen_full_factors = None

        data_copy = factors_data.copy()

        classes_y, freqs_y, _ = merge_vars(factors_data=factors_data, vars_indices=y, var_is_nominal=None, factors_nbins=factors_nbins, dtype=dtype)
        classes_y_safe = classes_y.copy()

        # Cardinality-bias pre-screen: drop columns whose Miller-Madow plug-in-MI bias is too large to score honestly (nbins_x > 2*sqrt(n)). See ``cardinality_prescreen``.
        if cardinality_bias_correction and factors_data.shape[1] > 0:
            x, _cardinality_refused_cols = cardinality_prescreen(
                factors_data, factors_nbins, factors_names, x, y, verbose,
            )
        else:
            _cardinality_refused_cols = set()

        # 2026-06-03 — Westfall-Young maxT permutation-null gain floor (computed ONCE on the finalised order-1 pool, applied at the selection gate). Fires on a WIDE pool
        # (>= ``screen_fdr_min_features``, where best-of-p selection bias dominates) OR on a NARROW pool that meets the target-over-split gate -- an MDLP-over-split heavy-tailed
        # regression target whose plug-in MI bias lifts pure-noise columns past the gain floors after the signals are picked. The narrow-pool gate is itself self-gating: it only
        # engages where the (X,y) joint table is dense enough that the floor is reliable, so a dense weak-signal pool (diabetes) keeps the floor OFF. Below both gates the floor
        # is 0.0 (no-op) so the clean low-cardinality tabular suite is untouched. See ``_permutation_null.py``.
        _fdr_gain_floor = compute_fdr_gain_floor(
            factors_data,
            factors_nbins,
            x,
            y,
            screen_fdr_null_permutations=screen_fdr_null_permutations,
            screen_fdr_null_quantile=screen_fdr_null_quantile,
            screen_fdr_min_features=screen_fdr_min_features,
            screen_fdr_target_oversplit_ratio=screen_fdr_target_oversplit_ratio,
            screen_fdr_min_rows_per_joint_cell=screen_fdr_min_rows_per_joint_cell,
            cardinality_bias_correction=cardinality_bias_correction,
            random_seed=random_seed,
            verbose=verbose,
        )

        # ``classes_y_safe`` is OVERLOADED across two consumers with INCOMPATIBLE
        # array-backend requirements, so we keep a CPU numpy buffer ALWAYS and a
        # separate CuPy device buffer only when GPU is on:
        #   * the FE step (returned ``classes_y_safe`` -> check_prospective_fe_pairs
        #     -> the batched MI noise-gate) runs njit kernels (CPU fallback AND the
        #     bit-identical GPU twin, which does ``np.asarray(classes_y_safe)`` +
        #     a numba Fisher-Yates shuffle) -> it REQUIRES a numpy host array.
        #   * the confirm / baseline-eval GPU branches (``mi_direct_gpu``) want the
        #     pre-warmed CuPy DEVICE buffer to skip the H2D copy.
        # Pre-2026-06-11 the GPU branch reassigned the single ``classes_y_safe``
        # local to a CuPy array and returned THAT to the FE step, so any
        # ``MRMR(use_gpu=True)`` fit that reached the FE pair-search crashed with
        # ``TypingError: Cannot determine Numba type of <class 'cupy.ndarray'>``
        # (or ``np.asarray`` on a cupy array raising in the subsample path). Now the
        # numpy buffer survives for the return; the device buffer only goes to ctx.
        classes_y_safe_host = classes_y_safe  # numpy, returned to the FE step
        if use_gpu:
            import cupy as cp

            classes_y_safe = cp.asarray(classes_y.astype(np.int32))  # device buffer for ctx -> mi_direct_gpu
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
            # 2026-06-02 — directed-FE tie-break inputs. Normalize the raw-name
            # collection to a set for O(1) membership; ``None`` stays ``None`` so
            # the confirm primitive falls back to the syntactic heuristic.
            raw_feature_names=(set(raw_feature_names) if raw_feature_names is not None else None),
            prefer_engineered_rel_eps=float(prefer_engineered_rel_eps or 0.0),
            dcd_state=dcd_state,  # Wave 9 — forward DCD state into confirm context
            n_workers=n_workers,
            workers_pool=workers_pool,
            parallel_kwargs=parallel_kwargs,
            baseline_npermutations=baseline_npermutations,
            full_npermutations=full_npermutations,
            min_nonzero_confidence=min_nonzero_confidence,
            fe_confirm_undersample_rows_per_cell=float(fe_confirm_undersample_rows_per_cell or 0.0),
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
            cached_jmim_MIs=cached_jmim_MIs,
            jmim_hit_counter=jmim_hit_counter,
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

        for interactions_order in (subsets_pbar := tqdmu(subsets, desc="Interactions order", leave=False, disable=not verbose)):

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

            # Running leader for the live "Confirmed predictors" postfix (reset per
            # interactions_order). ``_best_confirmed_gain`` starts at -inf so the first
            # confirmed feature -- even a negative-gain one -- becomes the displayed top.
            _best_confirmed_gain = float("-inf")
            _best_confirmed_name = None
            for _n_confirmed_predictors in (predictors_pbar := tqdmu(range(len(candidates)), leave=False, desc="Confirmed predictors", disable=not verbose)):
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

                _abs_floor = (min_relevance_gain if interactions_order == 1 else min_relevance_gain ** (1 / (interactions_order + 1)))
                # 2026-05-30 Miller-Madow: subtract finite-sample bias from gain at gate. For
                # joint candidates (k-way interactions) use product-of-bin-counts as effective
                # cardinality. Bias = (nbins_x_eff - 1) * (nbins_y - 1) / (2*n). The same
                # correction is applied to the first-selected feature's stored gain so the
                # relative-floor comparison is consistent across cardinalities.
                _best_gain_for_gate = float(best_gain)
                _first_gain_for_gate = 0.0
                # MM gate only applies to single-feature candidates (interactions_order=1).
                # For joint candidates (order >= 2) the bias (|joint_X|-1)*(|Y|-1)/(2n) grows
                # multiplicatively in component nbins (product), which over-corrects: a 39 x 39
                # joint at n=1500 carries bias 0.51 nats - enough to kill the XOR-product
                # synergy signal even when the joint MI is genuinely informative. The pre-
                # screen filter (cells > 0.5*n) already refuses high-cardinality SINGLE
                # columns before they're combined, so joints with all-safe components are
                # implicitly bounded; explicit MM correction on joints is double-counting.
                if (cardinality_bias_correction
                        and best_candidate is not None
                        and interactions_order == 1):
                    _n_samples_for_mm = int(factors_data.shape[0])
                    _y_idx = int(y[0]) if hasattr(y, "__len__") else int(y)
                    _nbins_y = int(factors_nbins[_y_idx])
                    _nbins_x_eff = 1
                    try:
                        for _v in best_candidate:
                            _nbins_x_eff *= int(factors_nbins[int(_v)])
                        _mm_bias_cand = (_nbins_x_eff - 1) * (_nbins_y - 1) / (2.0 * _n_samples_for_mm)
                        _best_gain_for_gate = float(best_gain) - _mm_bias_cand
                    except (TypeError, ValueError):
                        # best_candidate isn't iterable / contains non-int; skip MM gate
                        pass
                # 2026-05-30 diminishing-returns floor: from the SECOND selected feature onward,
                # require corrected best_gain >= MAX(corrected gain over already-selected) *
                # min_relevance_gain_relative_to_first. Using MAX (not just first) is critical when
                # the first-picked feature has a cardinality-inflated raw MI that the Miller-Madow
                # correction collapses (Layer 10 seed=101: user_id raw 0.328 -> corrected 0.088,
                # num_signal_1 raw 0.187 -> corrected 0.185; the corrected MAX over the running
                # set is num_signal_1's 0.185, so the floor at 5% is 0.009 - high enough to exclude
                # both the cardinality-biased user_id residual AND any trailing noise). The absolute
                # floor catches "no signal at all"; the relative floor catches "trailing noise that
                # statistically clears the absolute floor but is 100x smaller than the strongest
                # already-selected signal". 0.0 disables.
                _rel_floor = 0.0
                if min_relevance_gain_relative_to_first and selected_vars and predictors:
                    _max_corrected_gain = 0.0
                    _n_samples_for_mm = int(factors_data.shape[0]) if cardinality_bias_correction else 0
                    _y_idx_for_mm = int(y[0]) if hasattr(y, "__len__") else int(y)
                    _nbins_y_for_mm = int(factors_nbins[_y_idx_for_mm]) if cardinality_bias_correction else 0
                    for _pred in predictors:
                        _g_raw = float(_pred.get("gain", 0.0))
                        _p_indices = _pred.get("indices", ())
                        if cardinality_bias_correction and len(_p_indices) == 1:
                            _p_nbins_eff = 1
                            for _v in _p_indices:
                                _p_nbins_eff *= int(factors_nbins[int(_v)])
                            _g_corr = _g_raw - (_p_nbins_eff - 1) * (_nbins_y_for_mm - 1) / (2.0 * _n_samples_for_mm)
                        else:
                            _g_corr = _g_raw
                        if _g_corr > _max_corrected_gain:
                            _max_corrected_gain = _g_corr
                    _first_gain_for_gate = _max_corrected_gain
                    if _max_corrected_gain > 0.0:
                        _rel_floor = _max_corrected_gain * float(min_relevance_gain_relative_to_first)
                # 2026-06-03 maxT permutation-null floor (order-1 single candidates
                # only). CRITICAL: compare the candidate's corrected MARGINAL MI -
                # the exact statistic the null is built on - NOT the Fleuret
                # conditional gain ``_best_gain_for_gate``. Once the genuine signals
                # are selected, the conditional gain of a noise column is dominated by
                # conditioning-bias on the sparse high-dim joint (3 selected x 14-bin
                # features => ~2700 cells at n=1500): the noise's conditional gain
                # inflates to ~2x its marginal MI and clears the abs/rel floors, which
                # is exactly the embedding noise-cloud hijack. The marginal MI carries
                # no such conditioning bias, so flooring it at the maxT chance ceiling
                # cleanly separates the 3 genuine signals (marginal >> floor) from the
                # 497 noise dims (marginal < floor). ``cached_MIs[X]`` is the direct
                # marginal I(X;Y) computed during candidate scoring (no extra cost).
                # The floor is 0.0 for narrow pools (< ``screen_fdr_min_features``), so
                # this is a no-op on the tabular suite; higher-order joints are
                # untouched (they keep their own ``_abs_floor``).
                _fdr_floor_eff = _fdr_gain_floor if interactions_order == 1 else 0.0
                _fdr_pass = True
                if _fdr_floor_eff > 0.0 and best_candidate is not None and len(best_candidate) == 1:
                    _cand_marg_raw = cached_MIs.get(best_candidate, None)
                    if _cand_marg_raw is not None:
                        _cand_marg_corr = float(_cand_marg_raw)
                        if cardinality_bias_correction:
                            _nb_x_fdr = int(factors_nbins[int(best_candidate[0])])
                            _y_idx_fdr2 = int(y[0]) if hasattr(y, "__len__") else int(y)
                            _nb_y_fdr = int(factors_nbins[_y_idx_fdr2])
                            _cand_marg_corr -= (_nb_x_fdr - 1) * (_nb_y_fdr - 1) / (2.0 * int(factors_data.shape[0]))
                        _fdr_pass = _cand_marg_corr >= _fdr_floor_eff
                if (_best_gain_for_gate >= _abs_floor
                        and _best_gain_for_gate >= _rel_floor
                        and _fdr_pass):
                    for var in best_candidate:
                        if var not in selected_vars:
                            selected_vars.append(var)
                            if interactions_order > 1:
                                selected_interactions_vars.append(var)
                            # 2026-05-30 Wave 9 — Dynamic Cluster Discovery hook.
                            # After each accepted predictor, prune the Pool by
                            # SU(c, var) > tau_cluster. Mutates ``pool_pruned_mask``
                            # in-place (Critic1/B-1: NO mutation of candidates
                            # list — uses ``should_skip_candidate``'s mask check).
                            if dcd_state is not None:
                                # DCD discover/swap block carved into ``_screen_dcd_swap.py``
                                # (Tier E). The helper threads the loop-locals it reads/writes
                                # explicitly and RETURNS the four matrix refs it reassigns on a
                                # committed swap; ``dcd_state`` / ``selected_vars`` / ``predictors``
                                # / ``ctx`` / the caches are mutated in place. Behaviour is
                                # byte-for-byte identical to the prior inline block.
                                from ._screen_dcd_swap import screen_dcd_discover_and_swap
                                factors_data, factors_nbins, factors_names, data_copy = screen_dcd_discover_and_swap(
                                    dcd_state=dcd_state,
                                    var=var,
                                    factors_data=factors_data,
                                    factors_nbins=factors_nbins,
                                    factors_names=factors_names,
                                    data_copy=data_copy,
                                    selected_vars=selected_vars,
                                    entropy_cache=entropy_cache,
                                    cached_MIs=cached_MIs,
                                    full_npermutations=full_npermutations,
                                    y=y,
                                    engineered_recipes=engineered_recipes,
                                    predictors=predictors,
                                    ctx=ctx,
                                    verbose=verbose,
                                )
                    cand_name = get_candidate_name(best_candidate, factors_names=factors_names)

                    res = {"name": cand_name, "indices": best_candidate, "gain": best_gain}
                    if full_npermutations:
                        res["confidence"] = confidence
                    predictors.append(res)

                    # Live progress: surface the winning feature + its mrmr_gain on the
                    # "Confirmed predictors" bar. ``best_gain`` was just computed by
                    # confirm_one_predictor -- display-only, zero extra MI work. Track the
                    # strongest-confirmed so the postfix always names the current leader.
                    if verbose:
                        try:
                            _g = float(best_gain)
                            if _g > _best_confirmed_gain:
                                _best_confirmed_gain = _g
                                _best_confirmed_name = cand_name
                            _pf = {"last": _short_name(cand_name), "gain": f"{_g:.{ndigits}f}"}
                            if _best_confirmed_name is not None and _best_confirmed_name != cand_name:
                                _pf["top"] = f"{_short_name(_best_confirmed_name)}={_best_confirmed_gain:.{ndigits}f}"
                            predictors_pbar.set_postfix(_pf, refresh=False)
                        except (TypeError, ValueError):
                            pass

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
        return selected_vars, predictors, any_influencing, entropy_cache, cached_MIs, cached_confident_MIs, cached_cond_MIs, classes_y, classes_y_safe_host, freqs_y, dcd_state
    finally:
        # numpy global state is no longer mutated by this function (a local
        # Generator is used instead), so there is nothing to restore for numpy.
        # Wave 49 (2026-05-20): still restore numba + cupy (the prior comment block
        # acknowledged the leak; this closes it with a fresh-entropy reseed).
        if _numba_restore_seed is not None:
            try:
                set_numba_random_seed(int(_numba_restore_seed))
            except Exception:
                pass
        if _cp_restore_seed is not None:
            # 2026-05-30 Wave 9.1 fix (loop iter 25): mirror the
            # entry-block fix. ``cp`` only exists in this scope when
            # use_gpu was True AND CuPy was actually importable; import
            # defensively here so the restore call really fires.
            try:
                import cupy as cp
                cp.random.seed(int(_cp_restore_seed))
            except Exception:
                pass
