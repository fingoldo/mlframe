"""Permutation-based pair-confirmation for ``cat_interactions``.

Split out of ``cat_interactions.py`` to keep the parent below the 1k-line
monolith threshold. Behaviour preserved bit-for-bit; the parent re-exports
the public-looking confirm function so the orchestrator in
``run_cat_interaction_step`` continues to call it via the same name.

What lives here:
  - IPF + within-strata conditional shufflers
    (``_full_conditional_shuffle_ipf``, ``_conditional_shuffle_within_strata``)
  - per-pair joint-independence counting kernels (CPU prange + cupy)
  - perm-kernel CPU/GPU dispatch (``_perm_kernel_dispatch_use_gpu``)
  - same-shuffle three-MI test (``_shuffle_and_compute_three_mis``,
    ``_bulk_shuffle_and_compute_three_mis``)
  - Westfall-Young corrected p-value + FWER correction
  - the public ``_confirm_pairs_via_permutation`` entry point
"""
from __future__ import annotations

import logging
import math

import numpy as np
from numba import njit, prange

from .cat_fe_state import CatFEConfig
from .info_theory import compute_mi_from_classes, merge_vars

logger = logging.getLogger(__name__)


# ============================================================================
# Permutation confirmation (same-shuffle three-MI)
#
# Two-stage strategy:
# 1. Search phase: pair search computes point-estimate II for all candidates with no permutations (zero cost). Top-K selected by argpartition.
# 2. Confirmation phase: for each top-K survivor, run a permutation test of II_observed against the null distribution. SAME shuffled Y feeds all three MI computations
#    (I(merged;Y), I(X1;Y), I(X2;Y)), so II_perm = those three differences.
#
# Naming honesty: the test rejects "(X1, X2) jointly independent of Y" -- NOT "no synergy beyond marginals". Surfaced as ``joint_dependence_confidence`` not ``confidence``.
# ============================================================================


@njit(cache=True)
def _full_conditional_shuffle_ipf(
    classes_x2_safe: np.ndarray,
    classes_x1: np.ndarray,
    classes_y: np.ndarray,
    n_x1_classes: int,
    n_y_classes: int,
) -> None:
    """Full conditional permutation via IPF-style double-stratification. Shuffle X2 within strata of (X1, Y) so that P(X2 | X1, Y) is preserved on average across
    permutations -- the strictest synergy null that holds both marginals fixed while breaking X1-X2 conditional dependence.

    Reference: Patefield 1981 (R x C contingency tables with fixed marginals); Anderson & ter Braak 2003 (multi-factorial permutation).

    Implementation: for each unique (X1, Y) stratum, Fisher-Yates shuffle classes_x2_safe restricted to rows in that stratum.
    """
    n = len(classes_y)
    for cx1 in range(n_x1_classes):
        for cy in range(n_y_classes):
            # Gather indices in this (X1, Y) stratum
            positions = np.empty(n, dtype=np.int64)
            pos_count = 0
            for i in range(n):
                if classes_x1[i] == cx1 and classes_y[i] == cy:
                    positions[pos_count] = i
                    pos_count += 1
            # Fisher-Yates within stratum
            for idx in range(pos_count - 1, 0, -1):
                j = np.random.randint(0, idx + 1)
                a = positions[idx]
                b = positions[j]
                tmp = classes_x2_safe[a]
                classes_x2_safe[a] = classes_x2_safe[b]
                classes_x2_safe[b] = tmp


@njit(cache=True)
def _conditional_shuffle_within_strata(
    classes_x2_safe: np.ndarray,
    classes_y: np.ndarray,
    n_y_classes: int,
) -> None:
    """Conditional permutation: shuffle ``classes_x2_safe`` IN PLACE, restricting the shuffle to within each stratum of ``classes_y``.

    This is the correct null distribution for testing ``H0: X1 ⊥ X2 | Y`` (no synergy beyond marginals). Plain shuffle-Y tests ``H0: Y ⊥ (X1, X2)`` (joint independence) instead.

    Implementation: for each Y-stratum, collect the indices where ``classes_y[i] == c``, Fisher-Yates shuffle the corresponding slice of ``classes_x2_safe`` in place.
    Per Anderson & ter Braak 2003 ("Permutation tests for multi-factorial analysis of variance").

    Preserves ``P(X2 | Y)`` -- so each marginal ``I(X2; Y)`` is
    unchanged under the shuffle, but the conditional ``I(X1; X2 | Y)``
    is broken. The orchestrator combines this with three-MI calls
    above to compute II_perm under the conditional null.
    """
    n = len(classes_y)
    # For each Y class, gather positions and Fisher-Yates within the slice.
    for c in range(n_y_classes):
        # Collect indices manually (numba doesn't support boolean masks
        # for in-place writes the same way numpy does).
        positions = np.empty(n, dtype=np.int64)
        pos_count = 0
        for i in range(n):
            if classes_y[i] == c:
                positions[pos_count] = i
                pos_count += 1
        # Fisher-Yates shuffle within the stratum
        for idx in range(pos_count - 1, 0, -1):
            j = np.random.randint(0, idx + 1)
            a = positions[idx]
            b = positions[j]
            tmp = classes_x2_safe[a]
            classes_x2_safe[a] = classes_x2_safe[b]
            classes_x2_safe[b] = tmp


@njit(parallel=True, cache=True)
def _count_nfailed_joint_indep_prange(
    classes_pair: np.ndarray,
    freqs_pair: np.ndarray,
    classes_x1: np.ndarray,
    freqs_x1: np.ndarray,
    classes_x2: np.ndarray,
    freqs_x2: np.ndarray,
    classes_y: np.ndarray,
    freqs_y: np.ndarray,
    ii_obs: float,
    n_perms: int,
    base_seed: int,
    dtype,
) -> int:
    """Parallel permutation loop for the joint-independence null.

    Each thread gets its own copy of ``classes_y`` and own LCG seed (derived from ``base_seed + thread_idx``) so the count of failures is reproducible across re-runs at
    the same ``base_seed`` (modulo numpy version drift in ``random.shuffle``). Returns total ``nfailed`` summed across threads.

    Inner per-permutation work is fused into a SINGLE pass over N (vs three separate ``compute_mi_from_classes`` calls). Joint MI summation on the small (K_x, K_y)
    joint-count matrices is unchanged.
    """
    n = len(classes_y)
    K_pair = len(freqs_pair)
    K_x1 = len(freqs_x1)
    K_x2 = len(freqs_x2)
    K_y = len(freqs_y)
    nfailed_total = 0
    for tid in prange(n_perms):
        # Per-thread copy of Y so each prange iteration shuffles independently.
        cy_local = classes_y.copy()
        # Per-iteration LCG (PCG-style step) for the Fisher-Yates RNG -- ~2.7x faster than numpy's RandomState.seed + randint inside numba prange.
        # The LCG produces a different perm sequence than np.random, so absolute nfailed is not bit-equivalent to the legacy path; statistical behaviour
        # (mean nfailed / total perms) is unchanged because the LCG is also uniform on the shuffle space.
        state = np.uint64(base_seed) * np.uint64(2654435761) + np.uint64(tid + 1)
        # Fisher-Yates in place
        for i in range(n - 1, 0, -1):
            state = state * np.uint64(6364136223846793005) + np.uint64(1442695040888963407)
            j = int(state >> np.uint64(33)) % (i + 1)
            tmp = cy_local[i]
            cy_local[i] = cy_local[j]
            cy_local[j] = tmp

        # Single-pass joint-counts for the three (X*, Y) pairs.
        joint_pair = np.zeros((K_pair, K_y), dtype=dtype)
        joint_x1 = np.zeros((K_x1, K_y), dtype=dtype)
        joint_x2 = np.zeros((K_x2, K_y), dtype=dtype)
        for k in range(n):
            cy = cy_local[k]
            joint_pair[classes_pair[k], cy] += 1
            joint_x1[classes_x1[k], cy] += 1
            joint_x2[classes_x2[k], cy] += 1

        inv_n = 1.0 / n
        i_pair = 0.0
        for i in range(K_pair):
            px = freqs_pair[i]
            for j in range(K_y):
                jc = joint_pair[i, j]
                if jc:
                    jf = jc * inv_n
                    i_pair += jf * math.log(jf / (px * freqs_y[j]))

        i_x1 = 0.0
        for i in range(K_x1):
            px = freqs_x1[i]
            for j in range(K_y):
                jc = joint_x1[i, j]
                if jc:
                    jf = jc * inv_n
                    i_x1 += jf * math.log(jf / (px * freqs_y[j]))

        i_x2 = 0.0
        for i in range(K_x2):
            px = freqs_x2[i]
            for j in range(K_y):
                jc = joint_x2[i, j]
                if jc:
                    jf = jc * inv_n
                    i_x2 += jf * math.log(jf / (px * freqs_y[j]))

        if (i_pair - i_x1 - i_x2) >= ii_obs:
            nfailed_total += 1
    return nfailed_total


def _count_nfailed_joint_indep_cupy(
    classes_pair: np.ndarray,
    freqs_pair: np.ndarray,
    classes_x1: np.ndarray,
    freqs_x1: np.ndarray,
    classes_x2: np.ndarray,
    freqs_x2: np.ndarray,
    classes_y: np.ndarray,
    freqs_y: np.ndarray,
    ii_obs: float,
    n_perms: int,
    base_seed: int,
) -> int:
    """CuPy GPU variant of ``_count_nfailed_joint_indep_prange``.

    Same statistical contract: for each of ``n_perms`` shuffles of Y, compute joint MIs ``I(pair;Y_shuf)``, ``I(X1;Y_shuf)``, ``I(X2;Y_shuf)`` and count how often
    ``II_perm >= ii_obs``. Joint counts via flat-index ``cp.bincount(classes_x * K_y + y_perm, minlength=K_x*K_y)`` (single kernel launch per matrix, no scatter-add atomic
    contention). MI sum is a vectorised matrix op. Per-call cost is dominated by host->device transfer (~50-150 ms for 4 int arrays of size N + small freq arrays) plus
    n_perms inner iterations (~3-5 ms each at N=1M). Returns the same int ``n_failed`` total as the CPU numba version. Caller chooses CPU vs GPU dispatch.

    Bit-exact-equivalence note: cupy and numba may differ in last-bit fp rounding of ``log(jf / (px * py))``; for counting ``ii >= ii_obs`` this matters only when ii_obs
    sits EXACTLY at a perm-MI value (probability zero on continuous data). Real-world n_failed counts match.
    """
    import cupy as cp  # lazy import

    # Single host->device transfer per array.
    classes_pair_g = cp.asarray(classes_pair, dtype=cp.int64)
    classes_x1_g = cp.asarray(classes_x1, dtype=cp.int64)
    classes_x2_g = cp.asarray(classes_x2, dtype=cp.int64)
    classes_y_g = cp.asarray(classes_y, dtype=cp.int64)
    freqs_pair_g = cp.asarray(freqs_pair, dtype=cp.float64)
    freqs_x1_g = cp.asarray(freqs_x1, dtype=cp.float64)
    freqs_x2_g = cp.asarray(freqs_x2, dtype=cp.float64)
    freqs_y_g = cp.asarray(freqs_y, dtype=cp.float64)

    K_pair = int(freqs_pair_g.size)
    K_x1 = int(freqs_x1_g.size)
    K_x2 = int(freqs_x2_g.size)
    K_y = int(freqs_y_g.size)
    N = int(classes_y_g.size)
    inv_n = 1.0 / N if N > 0 else 0.0

    # Precompute marginals row-vector products for MI denominators.
    # ``freqs_x[:, None] * freqs_y[None, :]`` shape (K_x, K_y).
    denom_pair = freqs_pair_g[:, None] * freqs_y_g[None, :]
    denom_x1 = freqs_x1_g[:, None] * freqs_y_g[None, :]
    denom_x2 = freqs_x2_g[:, None] * freqs_y_g[None, :]

    nfailed_total = 0
    # Wave 49 (2026-05-20): use a local cupy RandomState per permutation rather
    # than mutating cp.random's global state. Reproducibility is preserved (same
    # base_seed -> same per-iter local RNG); caller's cupy global stream is no
    # longer clobbered.
    for p in range(n_perms):
        _local_cp_rng = cp.random.RandomState(base_seed + p)
        y_perm = _local_cp_rng.permutation(classes_y_g)

        # Joint counts via flat-index bincount.
        flat_pair = classes_pair_g * K_y + y_perm
        joint_pair = cp.bincount(flat_pair, minlength=K_pair * K_y).reshape(K_pair, K_y)
        flat_x1 = classes_x1_g * K_y + y_perm
        joint_x1 = cp.bincount(flat_x1, minlength=K_x1 * K_y).reshape(K_x1, K_y)
        flat_x2 = classes_x2_g * K_y + y_perm
        joint_x2 = cp.bincount(flat_x2, minlength=K_x2 * K_y).reshape(K_x2, K_y)

        jf_pair = joint_pair * inv_n
        jf_x1 = joint_x1 * inv_n
        jf_x2 = joint_x2 * inv_n

        # MI sum = sum(jf * log(jf / denom)) over (K_x, K_y); skip
        # zero-count cells (their contribution is 0 in the limit).
        i_pair = float(cp.where(jf_pair > 0, jf_pair * cp.log(jf_pair / cp.maximum(denom_pair, 1e-300)), 0.0).sum())
        i_x1 = float(cp.where(jf_x1 > 0, jf_x1 * cp.log(jf_x1 / cp.maximum(denom_x1, 1e-300)), 0.0).sum())
        i_x2 = float(cp.where(jf_x2 > 0, jf_x2 * cp.log(jf_x2 / cp.maximum(denom_x2, 1e-300)), 0.0).sum())

        if (i_pair - i_x1 - i_x2) >= ii_obs:
            nfailed_total += 1
    return nfailed_total


# Crossover threshold for CPU vs GPU on the permutation kernel.
# Calibrated empirically via ``profiling/bench_perm_kernel_gpu.py``
# (GTX-class consumer GPU, 6-core numba CPU):
#   100k x 3       CPU 4 ms  vs GPU 36 ms    -> CPU wins
#   100k x 50      CPU 49 ms vs GPU 241 ms   -> CPU wins
#   1M x 3         CPU 56 ms vs GPU 42 ms    -> GPU wins 1.3x
#   1M x 50        CPU 612 ms vs GPU 419 ms  -> GPU wins 1.5x
#   1M x 100       CPU 1135 ms vs GPU 834 ms -> GPU wins 1.4x
#   5M x 10        CPU 1018 ms vs GPU 373 ms -> GPU wins 2.7x
# Crossover sits at ~N=1_000_000 regardless of n_perms. Below 1M the
# per-call GPU launch + transfer cost (~30-40 ms) dominates the
# tiny CPU compute; above 1M, GPU bincount parallelism amortises the
# transfer and consistently wins.
_GPU_PERM_KERNEL_THRESHOLD_N: int = 1_000_000


def _perm_kernel_dispatch_use_gpu(
    n_samples: int, n_perms: int, backend: str,
) -> bool:
    """Decide whether to run the permutation kernel on GPU.

    Honours ``cfg.backend`` from CatFEConfig:
      - ``"gpu"`` : forced GPU (raises downstream if cupy missing).
      - ``"cpu"`` : forced CPU; never tries GPU.
      - ``"auto"``: GPU above ``_GPU_PERM_KERNEL_THRESHOLD_N`` AND
        cupy is importable. Otherwise CPU.
    """
    if backend == "gpu":
        try:
            import cupy as _cp  # noqa: F401
            return True
        except ImportError:
            return False
    if backend == "auto":
        # Wave 23 P1 fix (2026-05-20): _GPU_PERM_KERNEL_THRESHOLD_N is
        # tuned on a specific dev box; the GPU launch cost (30-40 ms)
        # depends on PCIe gen + driver model (WDDM vs TCC) + cc. Consult
        # kernel_tuning_cache for HW-tuned crossover; fall through to the
        # source-code default when no entry exists yet.
        try:
            from pyutilz.system.kernel_tuning_cache import KernelTuningCache
            _cache = KernelTuningCache.load_or_create()
            _entry = _cache.lookup(
                "cat_fe_perm_kernel",
                n_samples=int(n_samples), n_perms=int(n_perms),
            )
            _threshold = int(_entry["crossover_n"]) if _entry and "crossover_n" in _entry else _GPU_PERM_KERNEL_THRESHOLD_N
        except Exception:
            _threshold = _GPU_PERM_KERNEL_THRESHOLD_N
        if n_samples >= _threshold:
            try:
                import cupy as _cp  # noqa: F401
                return True
            except ImportError:
                return False
    return False


@njit(cache=True)
def _shuffle_and_compute_three_mis(
    classes_pair: np.ndarray,
    freqs_pair: np.ndarray,
    classes_x1: np.ndarray,
    freqs_x1: np.ndarray,
    classes_x2: np.ndarray,
    freqs_x2: np.ndarray,
    classes_y_safe: np.ndarray,
    freqs_y: np.ndarray,
    dtype,
) -> tuple:
    """Shuffle classes_y_safe in place (Fisher-Yates) and compute three MIs against the shuffled Y in a SINGLE pass over N rows.

    One pass increments all three joint-count matrices simultaneously, then closed-form MI summation on each (only depends on the small (K_x, K_y) matrices). Bit-exact
    equivalent of the three-separate-call form -- joint counts are integer increments, log/exp rounding identical.

    Profiled 2026-05-20 (iter3 of /loop fuzz-combo cycle): a ``parallel=True``
    variant with per-thread joint accumulators + final reduction was tried at
    N in {50k, 200k, 1M} and lost by 35-50% across the board (1.0->1.5ms,
    4.4->5.8ms, 31.0->34.2ms). Reason: the actual joint accumulation is only
    ~5-10ms of the 31ms total at N=1M; the rest is the SERIAL Fisher-Yates
    shuffle (RNG isn't safe across prange iterations - parallelising it
    would corrupt the permutation) plus the small inner MI loops. The
    parallel start-up + per-thread alloc + final reduction eats more than
    the accumulator win. Documented "no actionable speedup" so the next
    profile pass doesn't re-flag it - the seq form below is the winner."""
    n = len(classes_y_safe)
    # Fisher-Yates shuffle in place
    for i in range(n - 1, 0, -1):
        j = np.random.randint(0, i + 1)
        tmp = classes_y_safe[i]
        classes_y_safe[i] = classes_y_safe[j]
        classes_y_safe[j] = tmp

    K_pair = len(freqs_pair)
    K_x1 = len(freqs_x1)
    K_x2 = len(freqs_x2)
    K_y = len(freqs_y)

    joint_pair = np.zeros((K_pair, K_y), dtype=dtype)
    joint_x1 = np.zeros((K_x1, K_y), dtype=dtype)
    joint_x2 = np.zeros((K_x2, K_y), dtype=dtype)

    # SINGLE pass over N: increment all three joint matrices.
    for k in range(n):
        cy = classes_y_safe[k]
        joint_pair[classes_pair[k], cy] += 1
        joint_x1[classes_x1[k], cy] += 1
        joint_x2[classes_x2[k], cy] += 1

    inv_n = 1.0 / n

    # MI from joint counts. Inner loop is O(K_x * K_y) -- typically
    # K_x, K_y in {2..100} so this dominates by orders of magnitude
    # less than the N-pass above on N=1M.
    i_pair = 0.0
    for i in range(K_pair):
        px = freqs_pair[i]
        for j in range(K_y):
            jc = joint_pair[i, j]
            if jc:
                jf = jc * inv_n
                i_pair += jf * math.log(jf / (px * freqs_y[j]))

    i_x1 = 0.0
    for i in range(K_x1):
        px = freqs_x1[i]
        for j in range(K_y):
            jc = joint_x1[i, j]
            if jc:
                jf = jc * inv_n
                i_x1 += jf * math.log(jf / (px * freqs_y[j]))

    i_x2 = 0.0
    for i in range(K_x2):
        px = freqs_x2[i]
        for j in range(K_y):
            jc = joint_x2[i, j]
            if jc:
                jf = jc * inv_n
                i_x2 += jf * math.log(jf / (px * freqs_y[j]))

    return i_pair, i_x1, i_x2


@njit(parallel=True, nogil=True, cache=True)
def _bulk_shuffle_and_compute_three_mis(
    classes_pair: np.ndarray,
    freqs_pair: np.ndarray,
    classes_x1: np.ndarray,
    freqs_x1: np.ndarray,
    classes_x2: np.ndarray,
    freqs_x2: np.ndarray,
    classes_y: np.ndarray,
    freqs_y: np.ndarray,
    n_perms: int,
    base_seed: np.uint64,
    dtype,
):
    """Run ``n_perms`` independent shuffle+three-MI computations in parallel.

    Each prange iteration runs a FULL serial Fisher-Yates with its own LCG
    state (seeded from ``base_seed`` + iter index) into a thread-local
    working copy of ``classes_y``, then computes the three joint-MI values.
    Bench (n=200k, K=10/5/5/3, 8 perms): 46.4ms serial loop -> 7.7ms parallel
    (6.03x on 8 cores).

    Why this DIFFERS from the parallel attempt in ``_shuffle_and_compute_three_mis``
    docstring: that one tried to parallelise the INNER joint accumulator within
    ONE shuffle (lost 35-50% because the shuffle itself, not the accumulator,
    dominated). This one parallelises ACROSS multiple INDEPENDENT shuffles,
    each thread doing a serial-friendly chain (shuffle + accumulator) on its
    own buffer. The bandit Phase 1 always issues ``min_perms`` shuffles per
    pair up front; those are a clean parallel target. Phase 2 stays sequential
    because UCB1 needs each result to pick the next allocation.

    bench-attempt-rejected (2026-05-21, c0109 / iter135): folding the joint
    accumulation INTO the Fisher-Yates loop (position i becomes final after
    the swap at iter i) saves one full N-pass over local[] but runs
    18-31% SLOWER at n>=200k. Cause: the larger per-iteration working set
    (6 arrays touched per iter vs streaming patterns of the 2-pass form)
    blows L1; cache misses dominate the saved pass. Numerically bit-identical
    (max abs diff == 0). Bench:
    profiling/bench_bulk_shuffle_three_mis_fused.py.

    LCG sequence (state * 6364136223846793005 + 1442695040888963407, take top
    bits via >> 33) matches the one used in ``parallel_mi_besag_clifford`` so
    permutations are statistically equivalent to numpy's default rng for the
    purposes of MI permutation testing; the test outcome (kept_mask) may
    differ run-to-run from the np.random path purely because the RNG sequence
    is different, NOT because the test is biased.
    """
    n = len(classes_y)
    K_pair = len(freqs_pair)
    K_x1 = len(freqs_x1)
    K_x2 = len(freqs_x2)
    K_y = len(freqs_y)

    out_i_pair = np.zeros(n_perms, dtype=np.float64)
    out_i_x1 = np.zeros(n_perms, dtype=np.float64)
    out_i_x2 = np.zeros(n_perms, dtype=np.float64)

    for p in prange(n_perms):
        state = np.uint64(base_seed) + np.uint64(p) * np.uint64(2654435761)
        local = classes_y.copy()

        for i in range(n - 1, 0, -1):
            state = state * np.uint64(6364136223846793005) + np.uint64(1442695040888963407)
            k = int(state >> np.uint64(33)) % (i + 1)
            tmp = local[i]
            local[i] = local[k]
            local[k] = tmp

        joint_pair = np.zeros((K_pair, K_y), dtype=dtype)
        joint_x1 = np.zeros((K_x1, K_y), dtype=dtype)
        joint_x2 = np.zeros((K_x2, K_y), dtype=dtype)

        for k in range(n):
            cy = local[k]
            joint_pair[classes_pair[k], cy] += 1
            joint_x1[classes_x1[k], cy] += 1
            joint_x2[classes_x2[k], cy] += 1

        inv_n = 1.0 / n

        i_pair = 0.0
        for i in range(K_pair):
            px = freqs_pair[i]
            for j in range(K_y):
                jc = joint_pair[i, j]
                if jc:
                    jf = jc * inv_n
                    i_pair += jf * math.log(jf / (px * freqs_y[j]))

        i_x1 = 0.0
        for i in range(K_x1):
            px = freqs_x1[i]
            for j in range(K_y):
                jc = joint_x1[i, j]
                if jc:
                    jf = jc * inv_n
                    i_x1 += jf * math.log(jf / (px * freqs_y[j]))

        i_x2 = 0.0
        for i in range(K_x2):
            px = freqs_x2[i]
            for j in range(K_y):
                jc = joint_x2[i, j]
                if jc:
                    jf = jc * inv_n
                    i_x2 += jf * math.log(jf / (px * freqs_y[j]))

        out_i_pair[p] = i_pair
        out_i_x1[p] = i_x1
        out_i_x2[p] = i_x2

    return out_i_pair, out_i_x1, out_i_x2


def _compute_westfall_young_corrected_p(
    factors_data: np.ndarray,
    pairs_a: np.ndarray,
    pairs_b: np.ndarray,
    ii_obs_arr: np.ndarray,
    selected_idx: np.ndarray,
    nbins: np.ndarray,
    classes_y: np.ndarray,
    freqs_y: np.ndarray,
    marginal_mi: np.ndarray,
    n_perms: int,
    dtype,
    verbose: int,
) -> dict:
    """Full Westfall-Young: per shuffle, compute II_perm for ALL search-phase pairs, take the MAX, and accumulate the max-II distribution. Each survivor's p-value is
    ``(1 + #{b: max_II_perm[b] >= II_obs}) / (B + 1)``.

    The proper WY procedure (Westfall & Young 1993) naturally accounts for inter-pair correlation: pairs that share a column have correlated permutation distributions and
    the max-II statistic captures this. Strictly more powerful than Bonferroni on the same B.

    Cost: per shuffle, compute joint MI for all m = ``len(pairs_a)`` pairs. At m=4950 and B=100 that's 495k MI computations. Heavy -- enable only when
    ``cfg.fwer_correction='westfall_young'`` AND the user accepts the cost. Savings vs Bonferroni: typically need 2-5x fewer permutations for the same effective alpha.

    Returns ``{(i, j): corrected_p_value}`` ONLY for the survivors in
    ``selected_idx``.
    """
    n_samples = factors_data.shape[0]
    m = len(pairs_a)
    classes_y_safe = classes_y.copy()

    if verbose:
        logger.info(
            "cat-FE: full Westfall-Young permutation -- %d pairs x %d shuffles",
            m, n_perms,
        )

    # Pre-merge classes for all m search pairs ONCE. Memory: m * n * 4 B
    # = 19.8 MB at m=4950, n=1000; 198 MB at n=10000. Heavy but bounded.
    # If memory is a concern, fall back to Bonferroni-equivalent path.
    pair_classes_buf = np.empty((m, n_samples), dtype=dtype)
    pair_freqs_list: list = []
    for k in range(m):
        i = int(pairs_a[k])
        jj = int(pairs_b[k])
        cls_pair, fq_pair, _ = merge_vars(
            factors_data=factors_data,
            vars_indices=np.array([i, jj], dtype=np.int64),
            var_is_nominal=None, factors_nbins=nbins, dtype=dtype,
        )
        pair_classes_buf[k] = cls_pair
        pair_freqs_list.append(fq_pair)

    # Pre-merge marginal classes for each unique column in pairs_a/pairs_b
    unique_cols = np.unique(np.concatenate([pairs_a, pairs_b]))
    marginal_classes: dict = {}
    marginal_freqs: dict = {}
    for c in unique_cols:
        ci = int(c)
        cls_c, fq_c, _ = merge_vars(
            factors_data=factors_data,
            vars_indices=np.array([ci], dtype=np.int64),
            var_is_nominal=None, factors_nbins=nbins, dtype=dtype,
        )
        marginal_classes[ci] = cls_c
        marginal_freqs[ci] = fq_c

    # Permutation loop: for each shuffle, compute max II across all m pairs
    max_ii_per_perm = np.zeros(n_perms, dtype=np.float64)
    for b in range(n_perms):
        np.random.shuffle(classes_y_safe)
        # Compute MI(merged; Y_shuffled) for all pairs, and marginals for
        # all touched columns. Then II = joint - marginal_i - marginal_j.
        marginal_perm: dict = {}
        for ci, cls_c in marginal_classes.items():
            marginal_perm[ci] = compute_mi_from_classes(
                classes_x=cls_c, freqs_x=marginal_freqs[ci],
                classes_y=classes_y_safe, freqs_y=freqs_y, dtype=dtype,
            )
        max_ii = -np.inf
        for k in range(m):
            joint_perm = compute_mi_from_classes(
                classes_x=pair_classes_buf[k], freqs_x=pair_freqs_list[k],
                classes_y=classes_y_safe, freqs_y=freqs_y, dtype=dtype,
            )
            ii_perm = joint_perm - marginal_perm[int(pairs_a[k])] - marginal_perm[int(pairs_b[k])]
            if ii_perm > max_ii:
                max_ii = ii_perm
        max_ii_per_perm[b] = max_ii

    # Compute corrected p for each survivor
    corrected_p: dict = {}
    for k in selected_idx:
        i = int(pairs_a[k])
        jj = int(pairs_b[k])
        ii_obs = float(ii_obs_arr[k])
        n_exceed = int((max_ii_per_perm >= ii_obs).sum())
        corrected_p[(i, jj)] = (n_exceed + 1) / (n_perms + 1)
    return corrected_p


def _apply_fwer_correction(
    confidence_dict: dict, cfg: CatFEConfig, n_search_pairs: int,
) -> dict:
    """Apply multiple-testing correction to per-pair p-values, returning the corrected confidence dict. Supports:

    - ``"none"``: identity. FWER unchecked; user accepts inflation.
    - ``"bonferroni"``: ``p_corr = min(1, p * m)`` where m is the effective search-family size (``n_search_pairs``, NOT len of survivors). Conservative.
    - ``"bh_fdr"``: Benjamini-Hochberg step-up FDR. Less conservative than Bonferroni, controls expected proportion of false discoveries.
    - ``"westfall_young"``: proper WY requires recomputing the max-II across ALL search-phase pairs under each shuffle. This branch is reached only as a FALLBACK
      (typically when memory is too tight for ``_compute_westfall_young_corrected_p``); it approximates with Bonferroni-on-survivors, which is conservative-equivalent
      for the typical case where survivors' II values dominate the per-shuffle max.

    ``n_search_pairs`` is the count of pairs CONSIDERED at search time, NOT len(survivors). A user who screened 100 candidate cols saw N(N-1)/2 = 4950 pairs; that's the
    family size, not 64 survivors.
    """
    if cfg.fwer_correction == "none" or not confidence_dict:
        return dict(confidence_dict)

    # Convert confidences to p-values
    p_vals = {k: 1.0 - v for k, v in confidence_dict.items()}
    m = max(n_search_pairs, len(p_vals))

    if cfg.fwer_correction == "bonferroni":
        return {k: 1.0 - min(1.0, p * m) for k, p in p_vals.items()}

    if cfg.fwer_correction == "bh_fdr":
        # Benjamini-Hochberg step-up
        sorted_items = sorted(p_vals.items(), key=lambda kv: kv[1])
        n = len(sorted_items)
        # Adjusted p_(i) = min over j>=i of (p_(j) * m / j)
        corrected = {}
        prev = 1.0
        for rank, (k, p) in enumerate(reversed(sorted_items), start=1):
            i = n - rank + 1  # 1-indexed BH position
            adj = min(prev, p * m / i)
            prev = adj
            corrected[k] = adj
        return {k: 1.0 - corrected[k] for k in p_vals}

    if cfg.fwer_correction == "westfall_young":
        # Fallback path: orchestrator normally runs ``_compute_westfall_young_corrected_p`` directly. Bonferroni-on-survivors is the conservative-equivalent fallback.
        return {k: 1.0 - min(1.0, p * m) for k, p in p_vals.items()}

    raise ValueError(f"Unknown fwer_correction: {cfg.fwer_correction!r}")


def _confirm_pairs_via_permutation(
    factors_data: np.ndarray,
    pairs_a: np.ndarray,
    pairs_b: np.ndarray,
    selected_idx: np.ndarray,
    ii_arr: np.ndarray,
    nbins: np.ndarray,
    classes_y: np.ndarray,
    freqs_y: np.ndarray,
    cfg: CatFEConfig,
    n_search_pairs: int,
    dtype,
    verbose: int,
) -> tuple:
    """Run permutation confirmation on top-K survivors. Returns ``(selected_idx_kept, confidence_per_pair_dict)``.

    For each survivor, sample ``cfg.full_npermutations`` Fisher-Yates shuffles of Y, compute II_perm via same-shuffle three-MI, and count how often ``II_perm >= II_obs``.
    Confidence = 1 - failures / npermutations. Pairs with confidence < min_nonzero_confidence are dropped from selected_idx.

    Confidence is the "joint dependence confidence" -- it tests the null that (X1, X2) is jointly independent of Y, NOT the null that II = 0.
    """
    if cfg.full_npermutations <= 0:
        # Warn the user they're flying blind on FWER.
        if len(selected_idx) > 0 and verbose:
            logger.warning(
                "cat-FE: full_npermutations=0 surfaced %d pair(s) ranked by "
                "point estimate only. No statistical confirmation; results "
                "reflect selection bias from the search-phase 4950-pair family.",
                len(selected_idx),
            )
        return selected_idx, {}

    if len(selected_idx) == 0:
        return selected_idx, {}

    n_samples = factors_data.shape[0]
    n_perms = cfg.full_npermutations
    min_conf = 0.95  # cat-FE default; separate from MRMR.min_nonzero_confidence

    # Optional subsample for the permutation null distribution. When ``cfg.permutation_subsample`` is an int < n, draw a stable random subset ONCE (seed=base_seed) and
    # pass the subsampled (classes_pair, classes_x1, classes_x2, classes_y) to the inner permutation kernel. ``ii_obs`` was computed earlier on the FULL frame so the
    # test stays "subsampled-null vs full-data observed". Cost-vs-rigour trade-off documented on the config field.
    _perm_subsample = getattr(cfg, "permutation_subsample", None)
    if _perm_subsample is not None and _perm_subsample > 0 and n_samples > _perm_subsample:
        _ss_rng = np.random.default_rng(int(_perm_subsample) ^ n_samples)
        _ss_idx = _ss_rng.choice(n_samples, size=int(_perm_subsample), replace=False)
        _ss_idx.sort()  # contiguous-friendly access for downstream indexing
        _ss_classes_y = classes_y[_ss_idx]
        _ss_n_y_classes_local = int(_ss_classes_y.max()) + 1 if _ss_classes_y.size else 1
        _ss_freqs_y = np.bincount(_ss_classes_y, minlength=_ss_n_y_classes_local).astype(np.float64) / max(1, _ss_classes_y.size)
        if verbose:
            logger.info(
                "cat-FE: permutation_subsample=%d active; perm kernels see "
                "%d/%d rows (ii_obs still on full %d).",
                _perm_subsample, _ss_idx.size, n_samples, n_samples,
            )
    else:
        _ss_idx = None
        _ss_classes_y = classes_y
        _ss_freqs_y = freqs_y

    # Pre-merge classes for each survivor pair (and its marginals). Memory: O(top_k * 3 * n) -- at top_k=64, n=1M, dtype=int32: ~768 MB worst case; OK for top_k of order 100; user controls.
    confidence_dict: dict = {}
    kept_mask = np.ones(len(selected_idx), dtype=bool)

    if verbose:
        logger.info(
            "cat-FE: confirming %d pair(s) via %d permutation tests each",
            len(selected_idx), n_perms,
        )

    # Bandit UCB1 budget allocation is dispatched upstream (see top-level driver, _confirm_pairs_bandit_ucb1 call ~line 2837): when cfg.perm_budget_strategy ==
    # "bandit_ucb1" the caller takes the bandit branch and this fixed-budget function is not entered. By the time control reaches here, fixed budget is in effect.

    # Conditional permutation null requires per-pair freshly-merged X2 column to shuffle within strata of Y. The joint MI I(X1, X2; Y) after shuffling X2 within strata
    # of Y requires RE-merging X1 with the shuffled X2 -- the joint table can't be reused from cls_pair. So this null is materially more expensive than the joint-
    # independence null. Caller opts in via cfg.
    use_conditional = cfg.permutation_null == "conditional"
    n_y_classes = int(classes_y.max()) + 1

    for j, k in enumerate(selected_idx):
        i = int(pairs_a[k])
        jj = int(pairs_b[k])
        ii_obs = float(ii_arr[k])

        cls_pair, fq_pair, _ = merge_vars(
            factors_data=factors_data,
            vars_indices=np.array([i, jj], dtype=np.int64),
            var_is_nominal=None, factors_nbins=nbins, dtype=dtype,
        )
        cls_x1, fq_x1, _ = merge_vars(
            factors_data=factors_data,
            vars_indices=np.array([i], dtype=np.int64),
            var_is_nominal=None, factors_nbins=nbins, dtype=dtype,
        )
        cls_x2, fq_x2, _ = merge_vars(
            factors_data=factors_data,
            vars_indices=np.array([jj], dtype=np.int64),
            var_is_nominal=None, factors_nbins=nbins, dtype=dtype,
        )

        n_failed = 0
        if use_conditional:
            # Shuffle X2 within strata of Y. The null preserves P(X1, Y) AND P(X2, Y); only I(X1; X2 | Y) is broken.
            # If cfg.enable_full_conditional_perm is True, use the stricter (X1, Y) double-stratification which also preserves P(X1, X2) marginally -- the IPF-style synergy null.
            classes_x2_safe = cls_x2.astype(np.int64, copy=True)
            classes_x1_arr = cls_x1.astype(np.int64, copy=False)
            n_samples_local = factors_data.shape[0]
            use_full_cond = bool(getattr(cfg, "enable_full_conditional_perm", False))
            n_x1_classes = int(cls_x1.max()) + 1 if cls_x1.size else 1
            for _ in range(n_perms):
                if use_full_cond:
                    _full_conditional_shuffle_ipf(
                        classes_x2_safe, classes_x1_arr, classes_y,
                        n_x1_classes, n_y_classes,
                    )
                else:
                    _conditional_shuffle_within_strata(
                        classes_x2_safe, classes_y, n_y_classes,
                    )
                # Re-merge X1 with the shuffled X2 to get the conditional-null joint. We materialise a 2-col array on the fly.
                local_data = np.empty((n_samples_local, 2), dtype=dtype)
                local_data[:, 0] = classes_x1_arr.astype(dtype, copy=False)
                local_data[:, 1] = classes_x2_safe.astype(dtype, copy=False)
                local_nbins = np.array(
                    [int(cls_x1.max()) + 1, int(classes_x2_safe.max()) + 1],
                    dtype=np.int64,
                )
                cls_joint_perm, fq_joint_perm, _ = merge_vars(
                    factors_data=local_data,
                    vars_indices=np.array([0, 1], dtype=np.int64),
                    var_is_nominal=None, factors_nbins=local_nbins, dtype=dtype,
                )
                i_pair_p = compute_mi_from_classes(
                    classes_x=cls_joint_perm, freqs_x=fq_joint_perm,
                    classes_y=classes_y, freqs_y=freqs_y, dtype=dtype,
                )
                # Marginals are PRESERVED by conditional shuffle -- use the originals from the loop entry. (We still subtract the same marginals as the observed II.)
                i_x1_p = compute_mi_from_classes(
                    classes_x=cls_x1, freqs_x=fq_x1,
                    classes_y=classes_y, freqs_y=freqs_y, dtype=dtype,
                )
                # I(X2_shuffled; Y) -- per the conditional-null property, this equals I(X2; Y) up to floating-point noise; we recompute for safety.
                fq_x2_perm = np.bincount(
                    classes_x2_safe.astype(np.int64), minlength=int(classes_x2_safe.max()) + 1
                ).astype(np.float64) / n_samples_local
                i_x2_p = compute_mi_from_classes(
                    classes_x=classes_x2_safe.astype(dtype, copy=False),
                    freqs_x=fq_x2_perm.astype(np.float64),
                    classes_y=classes_y, freqs_y=freqs_y, dtype=dtype,
                )
                ii_perm = i_pair_p - i_x1_p - i_x2_p
                if ii_perm >= ii_obs:
                    n_failed += 1
        else:
            # Default joint-independence null: shuffle Y once, recompute all three MIs against the shuffled Y. Tests "(X1,X2) ⊥ Y".
            # Parallel via numba prange over permutations. Per-thread local copy of Y so shuffles don't race. Seed derived from j (survivor index) so re-runs are reproducible.
            # When permutation_subsample is set, the kernel runs against the subsampled (classes_x*, classes_y) arrays; ii_obs is unchanged.
            if _ss_idx is not None:
                _k_cls_pair = cls_pair[_ss_idx]
                _k_cls_x1 = cls_x1[_ss_idx]
                _k_cls_x2 = cls_x2[_ss_idx]
                _k_fq_pair = np.bincount(_k_cls_pair, minlength=len(fq_pair)).astype(np.float64) / max(1, _k_cls_pair.size)
                _k_fq_x1 = np.bincount(_k_cls_x1, minlength=len(fq_x1)).astype(np.float64) / max(1, _k_cls_x1.size)
                _k_fq_x2 = np.bincount(_k_cls_x2, minlength=len(fq_x2)).astype(np.float64) / max(1, _k_cls_x2.size)
            else:
                _k_cls_pair, _k_cls_x1, _k_cls_x2 = cls_pair, cls_x1, cls_x2
                _k_fq_pair, _k_fq_x1, _k_fq_x2 = fq_pair, fq_x1, fq_x2
            # Dispatch the permutation kernel to GPU when (N * n_perms) is above the crossover threshold. Below, CPU numba prange wins -- GPU launch + transfer cost
            # dominates short permutation budgets. See ``_perm_kernel_dispatch_use_gpu`` for the policy.
            _kernel_n = (
                int(_k_cls_pair.size) if _ss_idx is not None else n_samples
            )
            if _perm_kernel_dispatch_use_gpu(_kernel_n, n_perms, cfg.backend):
                try:
                    n_failed = _count_nfailed_joint_indep_cupy(
                        _k_cls_pair, _k_fq_pair, _k_cls_x1, _k_fq_x1,
                        _k_cls_x2, _k_fq_x2,
                        _ss_classes_y, _ss_freqs_y, ii_obs, n_perms,
                        base_seed=int(j) * 1000003 + 7,
                    )
                except Exception as _gpu_exc:
                    logger.warning(
                        "cat-FE: GPU permutation kernel failed (%s); "
                        "falling back to CPU numba.", _gpu_exc,
                    )
                    n_failed = _count_nfailed_joint_indep_prange(
                        _k_cls_pair, _k_fq_pair, _k_cls_x1, _k_fq_x1,
                        _k_cls_x2, _k_fq_x2,
                        _ss_classes_y, _ss_freqs_y, ii_obs, n_perms,
                        base_seed=int(j) * 1000003 + 7, dtype=dtype,
                    )
            else:
                n_failed = _count_nfailed_joint_indep_prange(
                    _k_cls_pair, _k_fq_pair, _k_cls_x1, _k_fq_x1,
                    _k_cls_x2, _k_fq_x2,
                    _ss_classes_y, _ss_freqs_y, ii_obs, n_perms,
                    base_seed=int(j) * 1000003 + 7, dtype=dtype,
                )
        # Continuity-corrected p-value: (n_failed + 1) / (n_perms + 1) gives a non-zero p-value floor even with zero failures and is the standard convention for
        # empirical permutation p.
        p = (n_failed + 1) / (n_perms + 1)
        conf = 1.0 - p
        confidence_dict[(i, jj)] = conf

    # FWER correction applied AFTER raw p collection so the correction strategy can see all survivors' p-values together (needed for BH-FDR step-up).
    corrected_conf = _apply_fwer_correction(
        confidence_dict, cfg, n_search_pairs=n_search_pairs,
    )

    # Drop pairs whose CORRECTED confidence falls below the floor.
    kept_mask = np.array([
        corrected_conf[(int(pairs_a[k]), int(pairs_b[k]))] >= min_conf
        for k in selected_idx
    ])
    if verbose:
        for j, k in enumerate(selected_idx):
            ipair = (int(pairs_a[k]), int(pairs_b[k]))
            if not kept_mask[j]:
                logger.info(
                    "cat-FE: pair %s failed FWER-corrected confirmation "
                    "(raw_conf=%.3f, corrected_conf=%.3f, threshold=%.2f, "
                    "correction=%s, m=%d)",
                    ipair, confidence_dict[ipair], corrected_conf[ipair],
                    min_conf, cfg.fwer_correction, n_search_pairs,
                )

    return selected_idx[kept_mask], corrected_conf

