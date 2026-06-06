"""``check_prospective_fe_pairs`` carved out of
``mlframe.feature_selection.filters.feature_engineering``.

Re-imported at the parent's module bottom so historical
``from mlframe.feature_selection.filters.feature_engineering import check_prospective_fe_pairs``
resolves transparently.
"""
from __future__ import annotations

import logging
import threading
from collections import defaultdict
from itertools import combinations
from timeit import default_timer as timer
from typing import Sequence

import numpy as np
import pandas as pd
from numba import njit, prange
from numpy.polynomial.hermite import hermval
from scipy import special as sp

from pyutilz.pythonlib import sort_dict_by_value
from pyutilz.system import tqdmu

# Module-level logger for module-scope helpers (e.g. _dispatch_batch_mi_with_noise_gate).
# ``check_prospective_fe_pairs`` still lazy-imports the parent's ``logger`` for its own body.
_module_logger = logging.getLogger(__name__)


# Pseudo-unary name for the per-operand learned pre-warp (2026-06-02). Lives in
# the same namespace as the real unary names (``identity``, ``sqr``, ...) so it
# flows through combination generation, naming, and survivor packing untouched;
# materialisation and recipe construction special-case it via this constant.
_PREWARP_UNARY = "prewarp"

# Reserved (never-colliding, length-3) key under which ``check_prospective_fe_pairs``
# returns the fitted pre-warp specs inside its result dict; real result keys are
# ``raw_vars_pair`` 2-tuples. Lets the caller recover specs across the loky path.
_PREWARP_SPECS_RESULT_KEY = ("__prewarp_specs__", -1, -1)


# Code-version for the per-host CPU-vs-GPU split of the batched FE-candidate
# MI+permutation noise-gate, keyed on the work size (n_rows x n_cols). The CPU
# njit-prange kernel wins for small/medium batches (its joint-hist pass is tiny and
# the H2D copy of the discretized frame + per-shuffle launch overhead would dominate
# on GPU); GPU pays off only for very large K at large n. The CPU/GPU backend is
# resolved per-host via the canonical ``KernelTuningCache.get_or_tune`` orchestrator
# (no hardcoded threshold). This is computed over the participating CPU + GPU fns via
# ``compute_code_version`` so a kernel-numerics edit invalidates stale per-host cache
# entries automatically; falls back to a static string when the GPU module / pyutilz
# code-versioning is unavailable.
try:  # GPU twin owns the canonical code_version (covers CPU + cupy + cuda bodies).
    from .batch_mi_noise_gate_gpu import _batch_mi_noise_gate_code_version as _bming_code_version
    _BATCH_MI_NOISE_GATE_CODE_VERSION = _bming_code_version() or "batch_mi_noise_gate-v2"
except Exception:
    _BATCH_MI_NOISE_GATE_CODE_VERSION = "batch_mi_noise_gate-v2"


def _dispatch_batch_mi_with_noise_gate(
    disc_2d: np.ndarray,
    quantization_nbins: int,
    classes_y: np.ndarray,
    classes_y_safe: np.ndarray,
    freqs_y: np.ndarray,
    npermutations: int,
    min_nonzero_confidence: float,
    use_su: bool,
    batch_mi_kernel,
) -> np.ndarray:
    """Route the batched FE-candidate MI + permutation noise-gate to the CPU njit kernel
    or (best-effort) a GPU batched path, by work size ``n * K`` via the per-host
    kernel_tuning_cache. The CPU kernel is the required win and the always-correct
    fallback; the GPU branch is gated behind ``is_cuda_available`` + try/except so any
    failure transparently falls back to CPU (mirrors ``mi_direct``'s GPU fastpath).

    Returns ``fe_mi[K]`` -- the per-column observed MI, zeroed where the permutation gate
    rejects. Bit-identical to a per-candidate ``mi_direct`` loop on the default FE path.
    """
    n = disc_2d.shape[0]
    K = disc_2d.shape[1]
    factors_nbins = np.full(K, int(quantization_nbins), dtype=np.int64)

    # Per-host CPU/GPU backend choice via the canonical ``get_or_tune`` orchestrator
    # (per-host cache, code-version checked -> on-miss tuner -> measurement-backed
    # fallback; no hardcoded threshold). The GPU twin (``batch_mi_noise_gate_gpu``)
    # is bit-identical (GPU does only the integer counting; entropy stays on the CPU
    # bit-exact path), so the tuner runs a real CPU-vs-GPU sweep and the cache routes
    # large (n_rows x n_cols) batches to GPU where it measurably wins on this host.
    backend = "cpu"
    try:
        from pyutilz.performance.kernel_tuning.cache import KernelTuningCache

        from .batch_mi_noise_gate_gpu import (
            _run_batch_mi_noise_gate_sweep,
            _batch_mi_noise_gate_fallback_choice,
        )

        # load_or_create() returns the MEMOIZED per-host cache singleton (~0ms/call); a bare
        # KernelTuningCache() ctor re-loads cache state from disk EVERY call (~0.75ms), which
        # on a wide near-saturated frame (thousands of FE raw-pairs, one dispatch each) added
        # seconds of pure overhead. This dispatcher is on the per-raw-pair hot path -> singleton.
        _res = KernelTuningCache.load_or_create().get_or_tune(
            "batch_mi_noise_gate",
            dims={"n_rows": int(n), "n_cols": int(K)},
            tuner=_run_batch_mi_noise_gate_sweep,  # real CPU-vs-GPU sweep (bit-identical GPU)
            axes=["n_rows", "n_cols"],
            fallback={"backend_choice": _batch_mi_noise_gate_fallback_choice(int(n), int(K))},
            code_version=_BATCH_MI_NOISE_GATE_CODE_VERSION,
            async_sweep=True,  # FIT-TIME: never block the FE pair-search on the sweep; measure in the background
        )
        if isinstance(_res, str):
            backend = _res
        elif _res:
            backend = str(_res.get("backend_choice", "cpu"))
    except Exception:  # pyutilz missing / cache error -> CPU (always correct)
        backend = "cpu"

    # GPU region: route to the bit-identical GPU twin. Any failure (cupy/cuda
    # unavailable, OOM, shape edge) returns None / raises and falls through to the
    # always-correct CPU njit kernel below (mirrors mi_direct's GPU fastpath).
    if backend in ("gpu", "cupy", "cuda"):
        try:
            _res = _batch_mi_with_noise_gate_gpu(
                disc_2d=disc_2d,
                factors_nbins=factors_nbins,
                classes_y=classes_y,
                classes_y_safe=classes_y_safe,
                freqs_y=freqs_y,
                npermutations=npermutations,
                min_nonzero_confidence=min_nonzero_confidence,
                use_su=use_su,
                force_backend=(backend if backend in ("cupy", "cuda") else None),
            )
            if _res is not None:
                return _res
        except Exception as _exc:  # pragma: no cover - GPU optional
            _module_logger.debug(
                "batch_mi_with_noise_gate GPU path failed (%s: %s); CPU fallback",
                type(_exc).__name__, _exc,
            )

    # CPU njit-prange kernel (the required win and always-correct fallback).
    return batch_mi_kernel(
        disc_2d=disc_2d,
        factors_nbins=factors_nbins,
        classes_y=classes_y,
        classes_y_safe=classes_y_safe,
        freqs_y=freqs_y,
        npermutations=int(npermutations),
        base_seed=np.uint64(0),
        min_nonzero_confidence=float(min_nonzero_confidence),
        use_su=bool(use_su),
        dtype=np.int32,
    )


def _batch_mi_with_noise_gate_gpu(
    disc_2d,
    factors_nbins,
    classes_y,
    classes_y_safe,
    freqs_y,
    npermutations,
    min_nonzero_confidence,
    use_su,
    force_backend=None,
):
    """Bit-identical GPU twin of ``batch_mi_with_noise_gate``; returns ``fe_mi[K]``
    when a GPU backend is available + chosen, else ``None`` (-> CPU fallback).

    Delegates to ``batch_mi_noise_gate_gpu.dispatch_batch_mi_with_noise_gate_gpu``,
    which runs the cupy / numba.cuda joint-histogram counting on the GPU and the
    entropy reduction on the bit-exact CPU path (the y-shuffles use the IDENTICAL
    CPU LCG/Fisher-Yates stream, ``base_seed*2654435761 + (i+1)`` then the PCG
    step). ``base_seed`` is 0 on the default FE path -- matching the CPU kernel call
    below -- so the GPU and CPU shuffle streams (and thus the noise-gate rejection)
    are identical to the bit.
    """
    try:
        from .batch_mi_noise_gate_gpu import dispatch_batch_mi_with_noise_gate_gpu
    except Exception:
        return None
    _out = dispatch_batch_mi_with_noise_gate_gpu(
        disc_2d=disc_2d,
        factors_nbins=factors_nbins,
        classes_y=classes_y,
        classes_y_safe=classes_y_safe,
        freqs_y=freqs_y,
        npermutations=int(npermutations),
        base_seed=np.uint64(0),
        min_nonzero_confidence=float(min_nonzero_confidence),
        use_su=bool(use_su),
        dtype=np.int32,
        force_backend=force_backend,
    )
    if _out is None:
        return None
    fe_mi, _backend_name = _out
    return fe_mi

# MARGINAL-UPLIFT alternative acceptance for the per-pair engineered winner. The
# primary joint-prevalence gate (``best_mi / pair_mi > fe_min_engineered_mi_prevalence``)
# rejects a genuine signal pair whenever the 2-D JOINT MI is finite-sample-inflated
# ABOVE what any 1-D engineered summary can retain -- and that inflation is severe when
# the target is discretised into many bins (the adaptive ``categorize_dataset`` routinely
# produces 25-30 target bins) while the engineered column is binned to ``quantization_nbins``
# (~10). On the canonical ``y = a**2/b + log(c)*sin(d)`` fixture the genuine ``a**2/b`` pair
# recovers only 0.83-0.97 of its inflated joint and is dropped, leaving a single engineered
# feature, and on the uniform-input variant a cross-pair artefact wins instead of the true
# ``mul(log(c),sin(d))``. A 1-D summary of a 2-D pair structurally cannot retain >~0.90 of
# the inflated joint, so requiring 0.97 asks the impossible for genuine algebraic recovery.
#
# The MARGINAL-UPLIFT criterion is scale-robust: a genuine joint pair's engineered column
# clears the LARGER individual-operand marginal MI by a wide margin (the synergy is real),
# whereas a cross-pair artefact -- whose engineered form mostly recapitulates one operand's
# marginal plus noise -- barely beats it. Measured on both fixtures the genuine pairs sit at
# uplift 1.36-2.32 while the artefacts sit at 1.03-1.44 BUT only ever with a LOW joint-recovery
# ratio; pairing the uplift bar with a relaxed joint floor admits the genuine pairs and keeps
# the artefacts out. This is the SAME notion the prewarp acceptance path and the smart_polynom
# baseline-uplift use ("engineered MI must beat the larger operand marginal"); noise pairs never
# clear it (the upstream pair screen + order-2 maxT floor already removed structureless pairs).
_FE_MARGINAL_UPLIFT_MIN_RATIO: float = 1.30
# Relaxed joint-recovery floor that pairs with the marginal-uplift bar. A 1-D engineered
# summary can recover ~0.82+ of even a heavily over-binned joint when the structure is genuine;
# cross-pair artefacts that DO clear the uplift bar are gated out here (their joint recovery is
# either much lower or -- when high -- comes with a marginal uplift that fails the bar above).
_FE_MARGINAL_UPLIFT_MIN_JOINT_RATIO: float = 0.82

# MEDIAN-GATE pseudo-unary (2026-06-04). Mirrors ``_PREWARP_UNARY`` exactly: it
# lives in the same namespace as the real unary names (``identity``, ``sqr``, ...)
# so it flows through combination generation, naming, and survivor packing
# untouched; materialisation and recipe construction special-case it via this
# constant. ``gate_med(x) = (x > train_median_x).astype(float)`` -- a STATEFUL
# pseudo-unary whose only fitted state is one float (the TRAIN median of the
# operand). Combined with the existing ``mul`` binary it expresses the
# median-gated operators ``(a > median_a) * b`` (gated_med) and
# ``(a > median_a) & (b > median_b)`` -> via ``mul(gate_med(a), gate_med(b))``
# (thr_and_med). Measured (skew bench) gated_med +0.0355 / thr_and_med +0.0435
# downstream-AUC d_mean vs raw, beating plain products (+0.022/+0.020) and
# threshold-0 gates (+0.009/+0.0001) -- the median adapts the split to each
# operand's distribution. The fit is a single ``np.median`` (no overfit risk,
# no held-out validation needed), the produced column is a closed-form function
# of ``x`` + the stored median, so replay is leak-safe; the gate still passes
# every existing FE prevalence / MI acceptance gate like any engineered feature.
_GATE_MED_UNARY = "gate_med"

# Reserved (never-colliding, length-3) key for the fitted median specs in the
# result dict (loky-parallel recovery path). Distinct from the prewarp key.
_GATE_MED_SPECS_RESULT_KEY = ("__gate_med_specs__", -1, -1)


def _gate_med_apply(vals: np.ndarray, median: float) -> np.ndarray:
    """Replay the median-gate pseudo-unary closed-form: ``(x > train_median).astype(float)``.

    NaN inputs (``x > median`` is False for NaN) collapse to 0.0, matching the
    downstream nan_to_num scrubbing. Returns float64 so the binary combine + MI
    discretisation see a continuous-typed 0/1 column (same dtype contract as the
    real unaries). The ``median`` is the TRAIN-fitted float; passing it in keeps
    this function pure (no y, no recompute) -> leak-safe at transform time."""
    return (np.asarray(vals, dtype=np.float64) > float(median)).astype(np.float64)


def _select_single_best(perf: dict, cols_names: Sequence, secondary: dict | None = None):
    """Pick ONE winning ``config`` from a ``{config: mi}`` mapping.

    Selection key, in priority order:
      1. PRIMARY: maximum ``perf[config]`` -- this MUST be the engineered
         feature's MI WITH THE TARGET. (Regression guard: a prior version
         passed the external-validation score -- MI of the candidate recombined
         with an unrelated third factor -- as the primary key here, so the
         search would discard the true max-target-MI form. e.g. on
         y = log(c)*sin(d) it picked add(log(c),1/d) at MI=0.25 over the true
         mul(log(c),sin(d)) at MI=0.32. Primary MUST be target MI.)
      2. SECONDARY (optional tie-break): maximum ``secondary[config]`` -- the
         external-validation MI. Only decisive among leaders whose target MI is
         exactly equal; prefers the representation that also generalises against
         other approved factors.
      3. deterministic tie-break by the engineered feature name (ascending).

    Used to collapse the leading-features equivalence class (many near-identical
    representations of the same algebraic target) down to a single representative
    per raw pair, restoring the pre-refactor 1-per-pair materialisation. Returns
    ``None`` when ``perf`` is empty.
    """
    if not perf:
        return None
    # measure-experiment-rejected (2026-06-03): a |corr| tie-break among the
    # MI-leading equivalence class (to prefer the most linearly-usable algebraic
    # form) was benchmarked and gives NO gain -- the forms within ~5% of the max
    # target MI are syntactic variants of the SAME function (a2/b == sqr(a)/b),
    # so their |corr| with y is identical (0/6 cases differed, OOS-Ridge delta
    # +0.000). Genuinely-different forms (a/b, a2/sqrt|b|) have DIFFERENT MI and
    # fall outside the band. The MI primary key + external-val secondary is right.
    # Lazy import (parent re-imports this module at its bottom -> avoid a
    # top-level cycle); mirrors the in-function import at the call sites.
    from .feature_engineering import get_new_feature_name
    _sec = secondary or {}
    return max(
        perf.items(),
        key=lambda kv: (
            kv[1],
            float(_sec.get(kv[0], 0.0)),
            _neg_name_key(get_new_feature_name(fe_tuple=kv[0], cols_names=cols_names)),
        ),
    )[0]


class _neg_name_key:
    """Reverse-ordering wrapper so a HIGHER MI still wins but, on an MI tie, the
    lexicographically-SMALLEST name wins (we negate the name comparison)."""

    __slots__ = ("s",)

    def __init__(self, s: str):
        self.s = s

    def __lt__(self, other: "_neg_name_key") -> bool:
        # Inverted: smaller string sorts as "greater" so max() prefers it.
        return self.s > other.s

    def __eq__(self, other) -> bool:  # pragma: no cover - completeness
        return isinstance(other, _neg_name_key) and self.s == other.s


# Wave 27 P1 (2026-05-20): ``check_prospective_fe_pairs`` is dispatched via
# ``parallel_run`` from mrmr.py with backend='threading'. The function
# accumulates per-binary-transform timings into a shared ``times_spent``
# defaultdict via ``+=``. Python's ``+=`` on a float is load-add-store and
# NOT atomic even under the GIL between threads; concurrent workers can
# drop updates silently, under-reporting the diagnostic at mrmr.py:1691.
# This module-level lock serialises the increment; threading workers
# synchronise correctly. Under loky/spawn each worker gets its own
# defaultdict copy (no shared state); the lock has no effect there but
# also doesn't break.
_TIMES_SPENT_LOCK = threading.Lock()

# CRITICAL: the hoisted shared buffer at
# ``check_prospective_fe_pairs`` allocates ``(n, max_n_combs * len(binary))``
# float32. With n=4M and the medium preset that's ~17.6 GiB -- production
# MRMR crashed with numpy.core._exceptions._ArrayMemoryError on a real run.
# The hoist landed in Wave Pack G (commit 068acdd) under small-n benchmarks
# and never measured peak RAM on million-row data.
#
# Two-strategy dispatch:
#   Fast path (current): if buffer < ``_FE_BUFFER_RAM_BUDGET_RATIO`` * available
#     RAM, allocate the shared buffer and use the hoist (cheapest if it fits).
#   Recompute fallback: drop the multi-column buffer, scratch into a fresh 1D
#     ``np.empty(n, float32)`` per inner iteration, and rebuild the ~10
#     survivor columns from their (transformations_pair, bin_func_name) metadata
#     after the inner loop. Extra recompute cost: ~K bin_func calls per pair
#     (K = num survivors, typically <= fe_max_pair_features + |leading|);
#     <= 1% of the ~max_combs*|binary| calls already done in the inner loop.
#
# Subsample path remains a separate opt-in (``subsample_n`` parameter); this
# memory dispatcher is the deterministic, accuracy-preserving fallback that
# auto-engages when the shared buffer would OOM.
_FE_BUFFER_RAM_BUDGET_RATIO: float = 0.4

# CROSS-PAIR (CHUNK) BATCHING (2026-06-06). The per-pair 3-phase batch (materialise
# -> ONE discretize_2d -> ONE batch_mi) below is bit-identical to the per-candidate
# path but its batches are too SMALL to saturate the cores: the njit-prange kernels
# (discretize_2d_quantile_batch, batch_mi_with_noise_gate) release the GIL, yet one
# pair's K candidates (~tens-to-low-hundreds of columns) is too little work for the
# prange to spread across cores AND stays below the GPU dispatch threshold. So the
# GIL-bound Python orchestration BETWEEN kernel calls (combo materialisation,
# nan_to_num, per-candidate best/prewarp/config bookkeeping) serialises on ONE core
# and dominates -> ~15-21% CPU (1 of 8), GPU idle (mrmr.py keeps backend="threading"
# because loky memmaps crash Windows paging on 1M-row data).
#
# Fix: process a CHUNK of many raw-pairs together -- accumulate ALL the chunk's
# candidate columns into ONE wide buffer, run ONE big discretize_2d + ONE big
# batch_mi over the whole chunk (the prange now has K_chunk = sum of per-pair K
# columns of work -> enough to spread across cores via nogil, AND the batch crosses
# the GPU dispatch threshold so the cupy/cuda backend can engage), THEN replay the
# EXACT per-pair best/prewarp/config tracking by grouping the chunk's candidates back
# by pair. Bit-identical because BOTH kernels score each column INDEPENDENTLY
# (discretize: per-column percentile axis=0 / searchsorted; batch_mi: prange over
# columns with per-column joint histogram, and the permutation shuffle is seeded by
# (base_seed=0, perm_index) ONLY -- never by the column -- so concatenating columns
# from many pairs into one disc_2d yields the same per-column MI as scoring each pair
# separately). Only the hoist+quantile path is chunked; the recompute-fallback (no
# buffer) and uniform method keep the per-pair path verbatim.
#
# The chunk's buffer width is bounded by the SAME RAM budget the per-pair buffer uses
# (``n_rows * chunk_cols * 4`` bytes within ``_FE_BUFFER_RAM_BUDGET_RATIO`` of
# available RAM); we pack as many whole pairs as fit (a pair is never split across
# chunks so the per-pair replay is intact) but always at least one pair per chunk.
_FE_CHUNK_MAX_COLS_HARD_CAP: int = 65536

# NJIT PARALLEL MATERIALISATION (2026-06-06). The chunk's candidate columns were filled by a PYTHON loop
# (``out[:, col] = bin_func(a, b)`` per candidate) -- one thread, GIL-held -> serial, the remaining single-core
# bottleneck after the discretize/MI kernels were batched. ``_materialise_chunk_njit`` fills the WHOLE chunk in a
# ``prange`` over candidates with ``nogil=True`` -> threads run truly parallel across cores. BIT-IDENTICAL to the
# numpy bin_funcs: every op is computed in float32 (the buffer + transformed_vars dtype), and the Python-float
# constants (``1e-9``, ``1.0``) are weak/value-based so the numpy expressions stay float32 too -> same hardware
# float32 ops, same per-element nan_to_num(nan=0, +-inf=0). Only the elementary closed ops (the default ``minimal``
# preset + the non-symmetric medium ops) are coded; a chunk containing any other op (hypot / maximal-preset
# specials) falls back to the per-candidate numpy path for bit-safety.
_NJIT_BINARY_OP_CODES: dict = {
    "mul": 0, "add": 1, "sub": 2, "div": 3, "max": 4, "min": 5,
    "abs_diff": 6, "signed": 7, "ratio_abs": 8,
}


def _njit_binary_op_codes(binary_transformations) -> "np.ndarray | None":
    """Return an int8 op-code per binary-op name (in registry order), or None if ANY op is not njit-coded
    (the caller then uses the per-candidate numpy path -- bit-safe for hypot / scipy.special / comparison ops)."""
    codes = []
    for name in binary_transformations:
        c = _NJIT_BINARY_OP_CODES.get(name)
        if c is None:
            return None
        codes.append(c)
    return np.asarray(codes, dtype=np.int8)


@njit(parallel=True, nogil=True, cache=True)
def _materialise_chunk_njit(tv, a_cols, b_cols, op_codes, out):
    """Fill ``out[:, k] = op_k(tv[:, a_cols[k]], tv[:, b_cols[k]])`` for all K candidates in a parallel prange,
    then nan_to_num each value to 0. float32 throughout -> bit-identical to the numpy bin_funcs (see header)."""
    n = tv.shape[0]
    K = op_codes.shape[0]
    eps = np.float32(1e-9)
    one = np.float32(1.0)
    zero = np.float32(0.0)
    for k in prange(K):
        ai = a_cols[k]
        bi = b_cols[k]
        op = op_codes[k]
        for r in range(n):
            a = tv[r, ai]
            b = tv[r, bi]
            if op == 0:        # mul
                v = a * b
            elif op == 1:      # add
                v = a + b
            elif op == 2:      # sub
                v = a - b
            elif op == 3:      # div = x / (y + sign(y)*eps + eps). numpy/numba promote the float64 ``eps`` literal,
                # so _safe_div computes in float64 then casts to float32 -> match that exactly (an all-float32 div
                # would differ in the last bit).
                sgn = zero if b == zero else (one if b > zero else -one)
                v = np.float32(np.float64(a) / (np.float64(b) + np.float64(sgn) * 1e-9 + 1e-9))
            elif op == 4:      # max = np.maximum (nan-propagating)
                if a != a or b != b:
                    v = a + b  # nan + anything -> nan (matches np.maximum nan propagation)
                else:
                    v = a if a > b else b
            elif op == 5:      # min = np.minimum (nan-propagating)
                if a != a or b != b:
                    v = a + b
                else:
                    v = a if a < b else b
            elif op == 6:      # abs_diff = |a - b|
                v = abs(a - b)
            elif op == 7:      # signed = sign(a)*|b|. np.sign(nan)=nan and np.abs(nan)=nan -> propagate nan.
                if a != a or b != b:
                    v = a + b
                else:
                    sgn = zero if a == zero else (one if a > zero else -one)
                    v = sgn * abs(b)
            else:              # op == 8: ratio_abs = a/(|b|+1). numpy/numba promote the float64 ``1.0`` literal ->
                # compute in float64 then cast, matching the numpy expression's last bit.
                v = np.float32(np.float64(a) / (np.float64(abs(b)) + 1.0))
            # np.nan_to_num(nan=0, posinf=0, neginf=0)
            if not (v == v and v != np.inf and v != -np.inf):
                v = zero
            out[r, k] = v

# Shared subsample default across the two FE entry points. ``polynom_pair_fe``
# already uses 200_000 (validated 2026-05-18: 100k could lose a marginal hermite
# feature, 200k kept it). The accuracy bench for ``check_prospective_fe_pairs``
# at this n landed at jaccard=1.0 vs full -- see
# bench_fe_pair_subsample_accuracy.py. Keep both call sites pinned to ONE knob
# so a future re-tune lands consistently across the FE block.
FE_DEFAULT_SUBSAMPLE_N: int = 200_000


def _plan_fe_chunks(*, prospective_pairs, pair_combs, vars_transformations, n_binary, chunk_max_cols):
    """Partition ``prospective_pairs`` (in iteration order) into CHUNKS of raw-pairs
    whose summed candidate-column count fits ``chunk_max_cols``.

    A pair is NEVER split across chunks (so the per-pair replay stays intact); a single
    pair larger than the cap becomes its own chunk. Pairs with zero valid candidates
    (both-operand registration fails for every comb) are dropped from the plan -- the
    caller's per-pair path no-ops them identically.

    Returns ``(chunks, pair_valid_combs, buf_width)``:
      * ``chunks`` -- list of lists of ``raw_vars_pair`` (chunk membership, in order).
      * ``pair_valid_combs`` -- ``raw_vars_pair -> [valid transformations_pair, ...]``.
      * ``buf_width`` -- the column width the shared chunk buffer must have (the widest
        chunk's candidate count); 0 when nothing is chunkable.
    """
    chunk_max_cols = max(int(chunk_max_cols), 1)
    pair_valid_combs: dict = {}
    pair_cols: dict = {}
    for (raw_vars_pair, _pm), _u in prospective_pairs.items():
        combs = pair_combs.get(raw_vars_pair, [])
        valid = [
            tp for tp in combs
            if (tp[0] in vars_transformations) and (tp[1] in vars_transformations)
        ]
        pair_valid_combs[raw_vars_pair] = valid
        pair_cols[raw_vars_pair] = len(valid) * n_binary

    chunks: list = []
    cur: list = []
    cur_cols = 0
    widest = 0
    for (raw_vars_pair, _pm), _u in prospective_pairs.items():
        need = pair_cols.get(raw_vars_pair, 0)
        if need == 0:
            continue
        if cur and (cur_cols + need > chunk_max_cols):
            chunks.append(cur)
            widest = max(widest, cur_cols)
            cur = []
            cur_cols = 0
        cur.append(raw_vars_pair)
        cur_cols += need
    if cur:
        chunks.append(cur)
        widest = max(widest, cur_cols)
    return chunks, pair_valid_combs, widest


def _compute_one_fe_chunk(
    *,
    chunk_pairs,
    pair_valid_combs,
    chunk_buffer,
    vars_transformations,
    transformed_vars,
    binary_transformations,
    quantization_nbins,
    quantization_dtype,
    classes_y,
    classes_y_safe,
    freqs_y,
    fe_npermutations,
    fe_min_nonzero_confidence,
    batch_mi_kernel,
    use_su,
    prewarp_unary,
    logger,
    discretize_2d_quantile_batch,
):
    """Fill ``chunk_buffer[:, :K]`` with EVERY candidate column of EVERY pair in
    ``chunk_pairs``, then run ONE ``discretize_2d_quantile_batch`` + ONE
    ``_dispatch_batch_mi_with_noise_gate`` over the whole chunk. Returns a dict
    ``raw_vars_pair -> (candidates, fe_mi_by_col, local_times)``.

    This is the cross-pair batch: K = sum of the chunk's per-pair candidate counts, so
    the njit-prange / GPU dispatch sees the WHOLE chunk's work at once (saturates cores
    / can cross the GPU threshold). BIT-IDENTITY: both kernels score each column
    INDEPENDENTLY (per-column percentile/searchsorted; prange over columns with a
    per-column joint histogram and a permutation shuffle seeded by (base_seed=0,
    perm_index) ONLY -- never by the column), so a candidate's codes + MI are exactly
    what the per-pair batch produced. The materialise order (combs x bin_func) and
    nan_to_num/timing per pair mirror the per-pair Phase 1 exactly. The caller must
    extract any survivor columns from ``chunk_buffer`` BEFORE the next chunk overwrites
    it (the caller processes a chunk's pairs immediately after this returns).
    """
    from timeit import default_timer as timer

    col = 0
    chunk_records: dict = {}  # raw_vars_pair -> (candidates, local_times)
    _op_codes_by_name = _njit_binary_op_codes(binary_transformations)  # None if ANY op is not njit-coded

    if _op_codes_by_name is not None:
        # NJIT PARALLEL PATH: record the candidate specs (NO per-candidate Python materialise), then fill the
        # ENTIRE chunk in ONE prange(nogil) kernel -> threads spread the work across cores. Candidate ORDER is the
        # SAME (pair x transformations_pair x bin_func) as the numpy path, so the config buffer index ``col`` is
        # identical, and the float32 op kernel is bit-identical to the numpy bin_funcs (see _materialise_chunk_njit).
        _name_list = list(binary_transformations.keys())
        _a_cols: list = []
        _b_cols: list = []
        _ops: list = []
        _cands_by_pair: dict = {}
        for raw_vars_pair in chunk_pairs:
            cands = []
            for transformations_pair in pair_valid_combs[raw_vars_pair]:
                _ai = vars_transformations[transformations_pair[0]]
                _bi = vars_transformations[transformations_pair[1]]
                uses_pw = (
                    transformations_pair[0][1] == prewarp_unary
                    or transformations_pair[1][1] == prewarp_unary
                )
                for _opn, bin_func_name in enumerate(_name_list):
                    _a_cols.append(_ai)
                    _b_cols.append(_bi)
                    _ops.append(int(_op_codes_by_name[_opn]))
                    cands.append((transformations_pair, bin_func_name, col, uses_pw))
                    col += 1
            _cands_by_pair[raw_vars_pair] = cands
        _t0 = timer()
        if col > 0:
            _materialise_chunk_njit(
                transformed_vars,
                np.asarray(_a_cols, dtype=np.int64),
                np.asarray(_b_cols, dtype=np.int64),
                np.asarray(_ops, dtype=np.int8),
                chunk_buffer[:, :col],
            )
        # Fold the single batched materialise time evenly across the bin_func names (diagnostic only -- the
        # per-bin_func breakdown does not affect recovery; the MI/feature selection below is index-driven).
        _per = (timer() - _t0) / max(len(_name_list), 1)
        for raw_vars_pair in chunk_pairs:
            chunk_records[raw_vars_pair] = (_cands_by_pair[raw_vars_pair], {nm: _per for nm in _name_list})
    else:
        # NUMPY FALLBACK: a chunk op is not njit-coded (hypot / maximal-preset specials) -> per-candidate path.
        for raw_vars_pair in chunk_pairs:
            candidates = []
            local_times: dict = {}
            for transformations_pair in pair_valid_combs[raw_vars_pair]:
                param_a = transformed_vars[:, vars_transformations[transformations_pair[0]]]
                param_b = transformed_vars[:, vars_transformations[transformations_pair[1]]]
                uses_pw = (
                    transformations_pair[0][1] == prewarp_unary
                    or transformations_pair[1][1] == prewarp_unary
                )
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    for bin_func_name, bin_func in binary_transformations.items():
                        start = timer()
                        try:
                            chunk_buffer[:, col] = bin_func(param_a, param_b)
                            _col_view = chunk_buffer[:, col]
                        except Exception:
                            logger.error(f"Error when performing {bin_func}")
                            continue
                        np.nan_to_num(_col_view, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                        local_times[bin_func_name] = local_times.get(bin_func_name, 0.0) + (timer() - start)
                        candidates.append((transformations_pair, bin_func_name, col, uses_pw))
                        col += 1
            chunk_records[raw_vars_pair] = (candidates, local_times)

    out: dict = {}
    if col == 0:
        for raw_vars_pair in chunk_pairs:
            candidates, local_times = chunk_records[raw_vars_pair]
            out[raw_vars_pair] = (candidates, None, local_times)
        return out

    disc_2d = discretize_2d_quantile_batch(
        chunk_buffer[:, :col], n_bins=quantization_nbins, dtype=quantization_dtype
    )
    fe_mi_arr = _dispatch_batch_mi_with_noise_gate(
        disc_2d=disc_2d,
        quantization_nbins=quantization_nbins,
        classes_y=classes_y,
        classes_y_safe=classes_y_safe,
        freqs_y=freqs_y,
        npermutations=fe_npermutations,
        min_nonzero_confidence=fe_min_nonzero_confidence,
        use_su=use_su,
        batch_mi_kernel=batch_mi_kernel,
    )
    fe_mi_full = np.asarray(fe_mi_arr, dtype=np.float64)
    for raw_vars_pair in chunk_pairs:
        candidates, local_times = chunk_records[raw_vars_pair]
        out[raw_vars_pair] = (candidates, fe_mi_full, local_times)
    return out


def check_prospective_fe_pairs(
    prospective_pairs,
    X,
    unary_transformations,
    binary_transformations,
    classes_y,
    classes_y_safe,
    freqs_y,
    num_fs_steps,
    cols,
    original_cols,
    fe_max_steps,
    fe_npermutations,
    fe_max_pair_features,
    fe_print_best_mis_only,
    fe_min_nonzero_confidence,
    fe_min_engineered_mi_prevalence,
    fe_good_to_best_feature_mi_threshold,
    fe_max_external_validation_factors,
    numeric_vars_to_consider,
    quantization_nbins,
    quantization_method,
    quantization_dtype,
    times_spent,
    verbose,
    # CRITICAL #2 follow-up (2026-05-21): subsample rows for the MI sweep so the
    # transformed_vars + shared_buffer allocations scale with subsample_n rather
    # than the (possibly multi-million-row) full X. Survivor columns are still
    # produced at full n via _rebuild_full_survivor_col so the caller contract
    # is preserved. Default 200_000 matches polynom_pair_fe's existing knob
    # (FE_DEFAULT_SUBSAMPLE_N); the standalone accuracy bench in
    # bench_fe_pair_subsample_accuracy.py shows jaccard=1.0 vs full at n_eff
    # >= 50k on synthetic 3-pair-competition data. 0 = use full data (legacy).
    subsample_n: int = FE_DEFAULT_SUBSAMPLE_N,
    subsample_seed: int = 42,
    # PER-OPERAND PRE-WARP (2026-06-02). When ``prewarp_enable`` is True the
    # unary/binary search gains, per raw operand, one extra "pseudo-unary"
    # ``prewarp(x)`` -- a learned 1-D orthogonal-polynomial warp fit JOINTLY
    # across the pair via the rank-1 ALS sweep (``hermite_fe.fit_pair_prewarp_als``,
    # which reuses the orthogonal-poly path's ``warm_start_als_seed``; an
    # INDEPENDENT 1-D fit cannot recover the b-side of a product target whose
    # b-marginal is ~0). This lets the elementary unary/binary path represent a
    # within-operand non-monotone distortion such as ``a**3 - 2a`` that no single
    # library unary can express, so a target ``F3(F1(a), F2(b))`` with a
    # non-monotone inner ``F1`` becomes recoverable.
    # ``prewarp_y`` is the SAME discretised target codes (``classes_y``) the MI
    # sweep already scores against; the fit is supervised but the produced column
    # is a closed-form function of ``x`` alone, so replay is leak-safe (the
    # fitted coeffs are stored in the recipe by ``_mrmr_fe_step``). Default OFF
    # so behaviour on data that does not need it is byte-identical.
    prewarp_enable: bool = False,
    prewarp_y: np.ndarray | None = None,
    prewarp_basis: str = "chebyshev",
    prewarp_max_degree: int = 4,
    # Minimum ratio (best-prewarp-MI / best-nonprewarp-MI) for the alternative
    # acceptance path. 1.20 = the prewarp must beat the elementary library by
    # >= 20% engineered MI to be admitted past the joint-prevalence gate. Tuned
    # to fire on the F-POLY non-monotone inner (measured uplift ~1.42x) while
    # staying silent on linear/monotone/noise (uplift ~1.0x there).
    prewarp_uplift_threshold: float = 1.20,
    # Held-out floor for the out-of-sample prewarp validation: a warp fit on a
    # train slice is kept only if its rank-1 reconstruction f(a)*g(b) tracks y on
    # a held-out slice with |corr| >= this. Rejects supervised overfitting on noise
    # operands at small n; 0.0 disables (legacy in-sample-only fit).
    prewarp_min_val_corr: float = 0.08,
    prewarp_specs_out: dict | None = None,
    # PER-OPERAND MEDIAN GATE (2026-06-04). When ``fe_gate_med_enable`` is True the
    # unary/binary search gains, per raw operand, one extra "pseudo-unary"
    # ``gate_med(x) = (x > train_median_x).astype(float)``. Combined with the
    # existing ``mul`` binary this lets the elementary path represent the
    # median-gated operators ``(a > median_a) * b`` (``mul(gate_med(a), b)``) and
    # the conjunction ``(a > median_a) & (b > median_b)``
    # (``mul(gate_med(a), gate_med(b))``) that a bilinear product cannot -- the
    # signal is non-product / conditional. Unlike a fixed threshold-0 gate, the
    # median ADAPTS the split to each operand's distribution, so it recovers the
    # gate on shifted / skewed operands where threshold-0 is useless (measured
    # skew-bench: gated_med +0.0355 / thr_and_med +0.0435 downstream-AUC d_mean
    # vs raw, beating products +0.022/+0.020 and threshold-0 +0.009/+0.0001).
    # The fitted state is ONE float per operand (the TRAIN median), stored in the
    # recipe by ``_mrmr_fe_step`` for leak-safe closed-form replay (no y, no
    # test-time recompute). The median does not overfit, so -- unlike prewarp --
    # NO held-out validation is needed; the gate is still subject to the same
    # MI-prevalence / external-validation acceptance gates every engineered
    # feature passes (it competes on equal footing in the per-pair MI sweep and
    # wins via ``best_mi`` only where the conditional form genuinely beats the
    # library). Default OFF so behaviour is byte-identical when not requested.
    fe_gate_med_enable: bool = False,
    gate_med_specs_out: dict | None = None,
):
    # Starting from the most heavily connected pairs, create a big pool of original features + their unary transforms. Individual vars referenced more than once go
    # to the global pool, the rest to the local (not stored)?

    # Lazy import of parent-resident helpers: ``.predict`` re-imports
    # this sibling at its bottom, so a top-level ``from .predict
    # import ...`` would create a hard cycle the meta-test flags.
    from .feature_engineering import FE_DEFAULT_SUBSAMPLE_N, _FE_BUFFER_RAM_BUDGET_RATIO, _can_hoist_shared_buffer, _estimate_fe_shared_buffer_bytes, _rebuild_full_survivor_col, discretize_array, discretize_2d_quantile_batch, get_new_feature_name, gpu_compatible_unary_names, logger, mi_direct
    # 2026-06-05: batched FE-candidate MI + permutation noise-gate (bit-identical to the
    # per-candidate mi_direct on the default outer/n_workers=1 path -- see kernel docstring).
    from .info_theory import batch_mi_with_noise_gate, use_su_normalization
    res = {}

    # SUBSAMPLE-SETUP: when caller asks for subsample_n > 0 AND len(X) exceeds it,
    # build subsampled views of X / classes_y / classes_y_safe / freqs_y. The MI
    # sweep operates on these views; survivor packing always rebuilds at full n.
    # When subsample_n is 0 / negative / >= len(X) the legacy full-data path runs
    # unchanged (everything below uses ``X`` / ``classes_y`` / ... directly).
    _X_full = X
    _full_n_rows = len(_X_full)
    _use_subsample = isinstance(subsample_n, int) and 0 < subsample_n < _full_n_rows
    if _use_subsample:
        _rng_sub = np.random.default_rng(int(subsample_seed))
        _sample_idx = np.sort(_rng_sub.choice(_full_n_rows, size=int(subsample_n), replace=False))
        if isinstance(_X_full, pd.DataFrame):
            X = _X_full.iloc[_sample_idx].reset_index(drop=True)
        else:
            # Polars path -- row indexing returns a fresh frame; preserves zero-copy where possible.
            X = _X_full[_sample_idx]
        # Realign per-row target encodings; recompute freqs from the subsampled
        # class labels so MI estimates use the actual subsample distribution
        # rather than the full-n freq table (which would bias the MI estimator
        # toward classes that shrank under the random subset).
        _cy = np.asarray(classes_y)
        _cy_safe = np.asarray(classes_y_safe)
        classes_y = _cy[_sample_idx]
        classes_y_safe = _cy_safe[_sample_idx]
        # Recompute freqs from subsampled class labels. merge_vars returns
        # freqs_y as a FLOAT proportions array (sum=1.0), not raw counts; the
        # subsample needs the same shape. bincount gives counts -> divide by
        # total to get proportions matching the caller's expectation.
        if classes_y.size > 0 and classes_y.dtype.kind in ("i", "u"):
            _counts = np.bincount(classes_y.astype(np.int64))
            _total = _counts.sum()
            if _total > 0:
                freqs_y = (_counts.astype(np.float64) / float(_total))
        # else: leave the caller-supplied freqs_y; mi_direct handles its own
        # validation and would crash anyway on a non-integer class table.
        if verbose:
            logger.info(
                "check_prospective_fe_pairs: subsample_n=%d active (full_n=%d, %.1f%% sample); "
                "MI sweep runs on the subsample, survivor columns rebuilt at full n.",
                int(subsample_n), _full_n_rows, 100.0 * subsample_n / _full_n_rows,
            )

    # PER-OPERAND PRE-WARP setup (2026-06-02). When enabled, fit ONE learned
    # 1-D pre-warp per raw operand against the (subsample-aligned) target, and
    # expose it as an extra pseudo-unary named ``_PREWARP_UNARY`` so the existing
    # unary x unary x binary search naturally considers ``binary(prewarp(a),
    # prewarp(b))``, ``binary(prewarp(a), b)`` etc. The fitted spec per var is
    # kept in ``_prewarp_spec_by_var`` for survivor recipe construction; warped
    # values are written into ``transformed_vars`` like any other unary.
    _prewarp_active = bool(prewarp_enable) and prewarp_y is not None
    _prewarp_spec_by_var: dict[int, dict] = {}
    _prewarp_y_eff = None
    if _prewarp_active:
        from .hermite_fe import apply_operand_prewarp, fit_pair_prewarp_als
        _pw_y = np.asarray(prewarp_y)
        if _use_subsample and _pw_y.shape[0] == _full_n_rows:
            _pw_y = _pw_y[_sample_idx]
        _prewarp_y_eff = np.ascontiguousarray(_pw_y, dtype=np.float64)

        # JOINT per-pair ALS pre-fit. For each prospective pair fit BOTH operand
        # warps together (rank-1 ALS); an independent 1-D fit cannot recover the
        # b-side of a product target whose b-marginal is ~0. First pairing wins
        # for a var shared across pairs (pairs are processed most-prospective-
        # first, so a shared var binds to its strongest interaction). None specs
        # leave the pseudo-unary unregistered for that var.
        def _operand_vals(_var):
            if _var not in original_cols:
                return None
            if isinstance(X, pd.DataFrame):
                return X.iloc[:, original_cols[_var]].values
            return X[:, original_cols[_var]].to_numpy()

        # OUT-OF-SAMPLE PREWARP VALIDATION (2026-06-03). The ALS prewarp is a
        # SUPERVISED per-operand fit; at small n it overfits noise operands (the
        # in-sample uplift is inflated by the fit AND by the multiple operand/pair
        # comparisons), so a noise-paired warp clears the in-sample uplift gate,
        # gets engineered, and ABSORBS a genuine feature -- the raw column then
        # reads as redundant and is dropped, leaving a noise-diluted feature
        # (measured: a genuine X5 dropped at n=500). Guard: fit the warp on a TRAIN
        # slice and keep it only if its rank-1 reconstruction f(a)*g(b) still tracks
        # y on a HELD-OUT slice. Genuine synergy (incl. zero-marginal XOR) and
        # genuine non-monotone inners generalise; overfit-on-noise collapses on the
        # held-out slice. At large n train ~= full so genuine recovery is untouched.
        # ``fe_pair_prewarp_min_val_corr`` (default 0.08) is the held-out floor; 0.0
        # restores the legacy in-sample-only fit.
        _pw_min_val_corr = float(prewarp_min_val_corr or 0.0)
        _pw_n = int(_prewarp_y_eff.shape[0]) if hasattr(_prewarp_y_eff, "shape") else len(_prewarp_y_eff)
        # Deterministic stride split (no RNG): every 3rd row -> validation (~33%).
        _pw_cv_ok = _pw_min_val_corr > 0.0 and _pw_n >= 60
        if _pw_cv_ok:
            _val_mask = (np.arange(_pw_n) % 3 == 0)
            _tr_mask = ~_val_mask
            _y_tr = _prewarp_y_eff[_tr_mask]
            _y_val = _prewarp_y_eff[_val_mask] - float(np.mean(_prewarp_y_eff[_val_mask]))

        def _prewarp_generalises(_va_full, _vb_full):
            """True if an ALS warp fit on the train slice still tracks y on the
            held-out slice (rank-1 reconstruction correlation >= floor)."""
            if not _pw_cv_ok:
                return True  # CV disabled / n too small -> accept (legacy behaviour)
            try:
                _a = np.asarray(_va_full, dtype=np.float64).reshape(-1)
                _b = np.asarray(_vb_full, dtype=np.float64).reshape(-1)
                if _a.shape[0] != _pw_n or _b.shape[0] != _pw_n:
                    return True  # length mismatch (subsample edge) -> don't block
                _sa_tr, _sb_tr = fit_pair_prewarp_als(
                    _a[_tr_mask], _b[_tr_mask], _y_tr,
                    basis=prewarp_basis, max_degree=prewarp_max_degree,
                )
                if _sa_tr is None or _sb_tr is None:
                    return False
                _wa = apply_operand_prewarp(_a[_val_mask], _sa_tr)
                _wb = apply_operand_prewarp(_b[_val_mask], _sb_tr)
                _recon = _wa * _wb
                if float(np.std(_recon)) < 1e-12 or float(np.std(_y_val)) < 1e-12:
                    return False
                # measure-experiment-rejected (2026-06-03): a dcor / MI held-out
                # floor (to recover non-monotone XOR prewarps that |corr| might
                # under-credit at small n) gives NO gain. The rank-1 reconstruction
                # f(a)*g(b) is FIT to approximate y, so it is linear-in-y by
                # construction -> |Pearson| is the right measure and is already
                # high for genuine synergy (XOR-sign reconstruction |corr| 0.64-0.75
                # at n=200, far above this 0.08 floor); benched 0/20 cases where
                # |corr|<floor BUT dcor>=0.15 across mul/xor-sign/sq*abs/a*sin(b).
                return abs(float(np.corrcoef(_recon, _y_val)[0, 1])) >= _pw_min_val_corr
            except Exception:
                return True  # validation failure -> fall back to accepting the warp

        for (raw_vars_pair, _), _ in prospective_pairs.items():
            _va, _vb = raw_vars_pair[0], raw_vars_pair[1]
            if _va in _prewarp_spec_by_var and _vb in _prewarp_spec_by_var:
                continue
            _vals_a = _operand_vals(_va)
            _vals_b = _operand_vals(_vb)
            if _vals_a is None or _vals_b is None:
                continue
            # Reject the warp for this pair if it does not generalise out-of-sample
            # (overfit-on-noise). Leaves the operands unregistered -> the pair search
            # falls back to the library unaries, which do not overfit.
            if not _prewarp_generalises(_vals_a, _vals_b):
                continue
            _sa, _sb = fit_pair_prewarp_als(
                _vals_a, _vals_b, _prewarp_y_eff,
                basis=prewarp_basis, max_degree=prewarp_max_degree,
            )
            if _va not in _prewarp_spec_by_var:
                _prewarp_spec_by_var[_va] = _sa
            if _vb not in _prewarp_spec_by_var:
                _prewarp_spec_by_var[_vb] = _sb

    # PER-OPERAND MEDIAN GATE setup (2026-06-04). When enabled, fit ONE TRAIN
    # median per raw operand (on the subsample-aligned slice, exactly like the
    # operand values the unary search consumes) and expose it as an extra
    # pseudo-unary named ``_GATE_MED_UNARY``. The fit is a single ``np.median``
    # per operand -- no supervision, no held-out validation (a median does not
    # overfit). The fitted float per var is kept in ``_gate_med_median_by_var``
    # for survivor recipe construction; the gated 0/1 column is written into
    # ``transformed_vars`` like any other unary. Operands missing from
    # ``original_cols`` or with no usable variance leave the pseudo-unary
    # unregistered for that var (search falls back to the real unaries).
    _gate_med_active = bool(fe_gate_med_enable)
    _gate_med_median_by_var: dict[int, float] = {}
    if _gate_med_active:
        for (raw_vars_pair, _), _ in prospective_pairs.items():
            for _gv in raw_vars_pair:
                if _gv in _gate_med_median_by_var:
                    continue
                if _gv not in original_cols:
                    continue
                if isinstance(X, pd.DataFrame):
                    _gvals = X.iloc[:, original_cols[_gv]].values
                else:
                    _gvals = X[:, original_cols[_gv]].to_numpy()
                _gf = np.asarray(_gvals, dtype=np.float64)
                # Reject no-variance operands (a constant gate is dead).
                _gmed = float(np.nanmedian(_gf)) if _gf.size else 0.0
                if not np.isfinite(_gmed):
                    continue
                _gate_med_median_by_var[_gv] = _gmed

    # Effective unary name list: the real registry plus the pre-warp pseudo-unary
    # when active. Used everywhere a per-pair combination over unary names is
    # built so the pseudo-unary participates exactly like a real one.
    _unary_names_eff = list(unary_transformations.keys())
    if _prewarp_active:
        _unary_names_eff = _unary_names_eff + [_PREWARP_UNARY]
    if _gate_med_active:
        _unary_names_eff = _unary_names_eff + [_GATE_MED_UNARY]

    # Exact preallocation. ``n_pairs * n_unary * 2`` over-counts because (var, tr_name) keys are de-duplicated in ``vars_transformations``; the unique-key set is the
    # true upper bound.
    unique_keys: set = set()
    for (raw_vars_pair, _), _ in prospective_pairs.items():
        for var in raw_vars_pair:
            for tr_name in _unary_names_eff:
                unique_keys.add((var, tr_name))

    if verbose >= 2:
        logger.info(
            "Creating a pool of %d unary transformations for feature engineering "
            "(legacy upper bound was %d).",
            len(unique_keys),
            len(prospective_pairs) * len(unary_transformations) * 2,
        )

    transformed_vars = np.empty(shape=(len(X), len(unique_keys)), dtype=np.float32)

    # Hoist ``final_transformed_vals`` outside the per-pair loop: precompute each pair's ``combs``, find the max length, allocate one shared buffer. Each pair writes
    # then reads the same ``[:, i]`` slice so stale tail data is never observed.
    pair_combs: dict = {}
    max_n_combs = 0
    for (raw_vars_pair, _), _ in prospective_pairs.items():
        combs = list(
            combinations(
                [(raw_vars_pair[0], k) for k in _unary_names_eff]
                + [(raw_vars_pair[1], k) for k in _unary_names_eff],
                2,
            )
        )
        combs = [tp for tp in combs if tp[0][0] != tp[1][0]]
        pair_combs[raw_vars_pair] = combs
        if len(combs) > max_n_combs:
            max_n_combs = len(combs)

    # CRITICAL #2 (2026-05-21): memory-aware dispatch. The full hoisted buffer is the fast path
    # but on n=4M with medium preset this lands at ~17.6 GiB and crashes the suite. Estimate the
    # required buffer, check psutil.virtual_memory().available, and either keep the buffer (fast)
    # or set it to None and switch to recompute-from-metadata in the inner loop and survivor
    # rebuild stages (memory-safe; ~1% extra bin_func calls per pair).
    _n_binary = len(binary_transformations)
    final_transformed_vals_shared = None
    # CROSS-PAIR chunk budget (cols). Default 0 == not chunkable (per-pair buffer).
    # Filled below when the single-pair buffer fits RAM: the chunk buffer reuses the
    # SAME available-RAM budget but may hold MANY pairs (each pair packed whole).
    _fe_chunk_max_cols = 0
    if max_n_combs > 0:
        _buf_bytes = _estimate_fe_shared_buffer_bytes(len(X), max_n_combs, _n_binary)
        _can_hoist, _bb, _avail = _can_hoist_shared_buffer(_buf_bytes)
        if _can_hoist:
            try:
                final_transformed_vals_shared = np.empty(
                    shape=(len(X), max_n_combs * _n_binary),
                    dtype=np.float32,
                )
                # Cross-pair chunk width cap: the largest column count whose
                # ``n_rows * cols * 4`` float32 buffer stays inside the SAME RAM
                # budget the single-pair check used (``_FE_BUFFER_RAM_BUDGET_RATIO``
                # of available). One pair (``max_n_combs * _n_binary`` cols) already
                # passed, so this is always >= one pair; we pack whole pairs up to it.
                _per_row_bytes = max(1, int(len(X)) * 4)
                if _avail >= 0:
                    _fe_chunk_max_cols = int((_avail * _FE_BUFFER_RAM_BUDGET_RATIO) // _per_row_bytes)
                else:
                    # No psutil reading -> bound the chunk to the hard cap only.
                    _fe_chunk_max_cols = _FE_CHUNK_MAX_COLS_HARD_CAP
                _fe_chunk_max_cols = max(
                    max_n_combs * _n_binary,
                    min(_fe_chunk_max_cols, _FE_CHUNK_MAX_COLS_HARD_CAP),
                )
            except MemoryError:
                # psutil over-reported available; falling back stays safe.
                final_transformed_vals_shared = None
                if verbose:
                    logger.warning(
                        "check_prospective_fe_pairs: shared buffer (%.1f GiB) allocation raised "
                        "MemoryError despite passing the available-RAM check (%.1f GiB available, "
                        "%.0f%% budget); switching to recompute-from-metadata fallback (~1%% extra "
                        "bin_func calls per pair).",
                        _bb / 2**30, _avail / 2**30 if _avail >= 0 else float("nan"),
                        _FE_BUFFER_RAM_BUDGET_RATIO * 100.0,
                    )
        else:
            if verbose:
                logger.warning(
                    "check_prospective_fe_pairs: shared buffer would need %.1f GiB but only %.1f GiB "
                    "RAM is available (%.0f%% budget = %.1f GiB cap); using recompute-from-metadata "
                    "fallback path (~1%% extra bin_func calls per pair, identical survivors). To force "
                    "the fast path either free RAM or raise _FE_BUFFER_RAM_BUDGET_RATIO; to bound "
                    "compute, pass subsample_n>0 from the MRMR config.",
                    _bb / 2**30, _avail / 2**30 if _avail >= 0 else float("nan"),
                    _FE_BUFFER_RAM_BUDGET_RATIO * 100.0,
                    (_avail * _FE_BUFFER_RAM_BUDGET_RATIO) / 2**30 if _avail >= 0 else float("nan"),
                )
    # In the recompute-fallback path we need to look up the
    # ``(transformations_pair, bin_func_name)`` for any index ``i`` that
    # was assigned in the inner loop, so the validation + survivor-packing
    # phases can rebuild the column on demand. The shared-buffer path
    # ignores this dict; either way it is bounded by ``max_n_combs *
    # _n_binary`` lightweight tuples per pair.
    _need_recompute_map = final_transformed_vals_shared is None

    vars_transformations = {}
    i = 0
    for (raw_vars_pair, _pair_mi), _uplift in prospective_pairs.items():
        for var in raw_vars_pair:
            # ``original_cols`` is built only for cols that survived the prior
            # selection pass; a temp / dropped column index may not be present.
            # Skip silently rather than KeyError out of the whole FE block.
            if var not in original_cols:
                continue
            # Polars vs pandas int-column indexing: ``X[:, idx].to_numpy()`` (polars, zero-copy for numerics) vs ``X.iloc[:, idx].values`` (pandas).
            if isinstance(X, pd.DataFrame):
                vals = X.iloc[:, original_cols[var]].values
            else:
                vals = X[:, original_cols[var]].to_numpy()
            for tr_name in _unary_names_eff:
                tr_func = unary_transformations.get(tr_name)
                key = (var, tr_name)
                # Per-operand learned pre-warp: the joint ALS spec was pre-fit
                # above (per pair). When the var has no usable spec (solve failed
                # / non-polynomial basis) the pseudo-unary is simply not
                # registered and the search proceeds with the real unaries only.
                # The fitted spec is stashed for survivor recipe construction
                # (leak-safe replay from coeffs alone).
                if tr_name == _PREWARP_UNARY:
                    if _prewarp_spec_by_var.get(var) is None:
                        continue
                # Median-gate pseudo-unary: skip vars with no fitted median (no
                # variance / not in original_cols). The fitted float is stashed
                # in ``_gate_med_median_by_var`` for survivor recipe construction
                # (leak-safe replay from the stored median alone).
                if tr_name == _GATE_MED_UNARY:
                    if var not in _gate_med_median_by_var:
                        continue
                if key not in vars_transformations:
                    try:
                        if tr_name == _PREWARP_UNARY:
                            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                                transformed_vars[:, i] = apply_operand_prewarp(vals, _prewarp_spec_by_var[var])
                        elif tr_name == _GATE_MED_UNARY:
                            transformed_vars[:, i] = _gate_med_apply(vals, _gate_med_median_by_var[var])
                        elif "poly_" in tr_name:
                            transformed_vars[:, i] = hermval(vals, c=tr_func)
                        else:
                            # WAVE 5 (1/4): if CUDA is available, the
                            # transformation is GPU-compatible, AND the
                            # column is large enough to amortise the H2D
                            # + D2H round trip, run the elementwise op on
                            # GPU via cupy. The numpy-vs-cupy crossover for
                            # this n_samples is resolved per-host via the
                            # shared get_or_tune orchestrator (kernel
                            # "unary_elementwise"), with the old fixed
                            # ~500k-cell breakeven as the measurement-backed
                            # fallback. A "cupy" choice is still gated below
                            # on live CUDA availability + per-op compat.
                            _gpu_used = False
                            from pyutilz.performance.kernel_tuning import array_location

                            from ._unary_elementwise_tuning import unary_elementwise_backend_choice
                            # residency-aware: VRAM-resident input skips H2D, which flips the
                            # numpy/cupy crossover (measured), so pass where ``vals`` lives.
                            _want_gpu = unary_elementwise_backend_choice(int(vals.size), array_location(vals)) == "cupy"
                            if (
                                _want_gpu
                                and tr_name in gpu_compatible_unary_names()
                            ):
                                try:
                                    from pyutilz.core.pythonlib import is_cuda_available
                                    if is_cuda_available():
                                        import cupy as cp
                                        _cp_fn = getattr(cp, tr_name, None)
                                        if _cp_fn is not None:
                                            d_vals = cp.asarray(vals)
                                            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                                                d_res = _cp_fn(d_vals)
                                            transformed_vars[:, i] = cp.asnumpy(d_res)
                                            _gpu_used = True
                                except Exception:
                                    _gpu_used = False  # fall through to CPU
                            if not _gpu_used:
                                # Suppress unary-transform NaN/inf RuntimeWarnings
                                # (eg ``overflow in exp``, ``divide by zero in
                                # log``). The downstream nan_to_num + MI-gate
                                # already filter pathological rows; the bare
                                # numpy emit only spams stderr.
                                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                                    transformed_vars[:, i] = tr_func(vals)
                    except Exception as e:
                        # ``np.isnan`` / ``np.isinf`` / ``np.nanmin`` only work on float dtypes. When ``vals`` is object/string (e.g. a polars Utf8 cat column not encoded
                        # before reaching FE), calling them inside the error-log formatter itself raises -- masking the real transformation error and aborting MRMR
                        # entirely. Compute numeric-only diagnostics conditionally.
                        if np.issubdtype(vals.dtype, np.floating):
                            _diag = (
                                f", isnan={np.isnan(vals).sum()}, "
                                f"isinf={np.isinf(vals).sum()}, nanmin={np.nanmin(vals)}"
                            )
                        else:
                            _diag = f", dtype={vals.dtype} (numeric diagnostics skipped)"
                        logger.error(
                            f"Error when performing {tr_name} on array {vals[:5]}, "
                            f"var={cols[var]}: {str(e)}{_diag}"
                        )
                    else:
                        vars_transformations[key] = i
                        i += 1

    if verbose >= 2:
        logger.info("Created. For every pair from the pool, trying all known functions...")

    # Per-operand marginal MI cache for the MARGINAL-UPLIFT alternative acceptance gate.
    # The marginal MI is the discretised RAW operand (its ``identity`` unary, already
    # materialised in ``transformed_vars``) scored against the target with the SAME
    # estimator the engineered columns use, so the uplift ratio is apples-to-apples.
    # Operands recur across pairs, so memoise by cols-space var index.
    _operand_marginal_mi_cache: dict = {}

    def _operand_marginal_mi(_var) -> float:
        if _var in _operand_marginal_mi_cache:
            return _operand_marginal_mi_cache[_var]
        _mi_val = 0.0
        _idx = vars_transformations.get((_var, "identity"))
        if _idx is not None:
            try:
                _disc = discretize_array(
                    arr=transformed_vars[:, _idx], n_bins=quantization_nbins,
                    method=quantization_method, dtype=quantization_dtype,
                )
                _m, _ = mi_direct(
                    _disc.reshape(-1, 1),
                    x=np.array([0], dtype=np.int64), y=None,
                    factors_nbins=np.array([quantization_nbins], dtype=np.int64),
                    classes_y=classes_y, classes_y_safe=classes_y_safe, freqs_y=freqs_y,
                    min_nonzero_confidence=fe_min_nonzero_confidence, npermutations=fe_npermutations,
                )
                _mi_val = float(_m)
            except Exception:
                _mi_val = 0.0
        _operand_marginal_mi_cache[_var] = _mi_val
        return _mi_val

    # CROSS-PAIR (CHUNK) BATCHING precompute (2026-06-06). Only on the hoist+quantile
    # path (the per-pair 3-phase batch path). We partition the prospective pairs into
    # CHUNKS whose total candidate-column count fits the RAM-budgeted chunk buffer,
    # then -- per chunk -- materialise ALL the chunk's candidate columns into ONE wide
    # buffer, run ONE discretize_2d + ONE batch_mi over the whole chunk, and stash the
    # per-pair results (ordered candidate list + per-candidate MI + buffer columns) in
    # ``_chunk_mi_cache``. The per-pair loop below consumes the cache (reads MI + the
    # chunk buffer columns) instead of doing its own per-pair Phase 1/2/3 -- enlarging
    # the njit-prange + GPU-dispatch batch from one pair's K to the whole chunk's
    # K_chunk, so the kernels saturate the cores / cross the GPU threshold. Bit-identical
    # (each column is scored independently; shuffle seeded by (0, perm_index) only).
    #
    # ``_chunk_mi_cache[raw_vars_pair]`` -> (ordered list of
    # (transformations_pair, bin_func_name, buf_col, uses_pw), fe_mi_array_aligned,
    # local_times_for_pair). ``_chunk_buffer`` is the wide buffer the buf_col indices
    # point into; it is held alive for the whole pair loop so survivor packing can read
    # ``_chunk_buffer[:, buf_col]`` exactly as the per-pair path read final_transformed_vals.
    _chunk_global_batch = (
        (final_transformed_vals_shared is not None)
        and (quantization_method == "quantile")
        and (_fe_chunk_max_cols > max_n_combs * _n_binary)  # chunk holds > 1 pair's worth
        and (len(prospective_pairs) > 1)
    )
    _chunk_mi_cache: dict = {}
    _chunk_buffer = None
    _pair_to_chunk: dict = {}   # raw_vars_pair -> chunk index
    _fe_chunks: list = []
    _pair_valid_combs: dict = {}
    if _chunk_global_batch:
        _fe_chunks, _pair_valid_combs, _chunk_buf_width = _plan_fe_chunks(
            prospective_pairs=prospective_pairs,
            pair_combs=pair_combs,
            vars_transformations=vars_transformations,
            n_binary=_n_binary,
            chunk_max_cols=_fe_chunk_max_cols,
        )
        # Only worth chunking when at least one chunk groups MORE than one pair.
        if _fe_chunks and max(len(c) for c in _fe_chunks) > 1 and _chunk_buf_width > 0:
            try:
                _chunk_buffer = np.empty((len(X), _chunk_buf_width), dtype=np.float32)
                for _ci_chunk, _chunk in enumerate(_fe_chunks):
                    for _p in _chunk:
                        _pair_to_chunk[_p] = _ci_chunk
                if verbose:
                    logger.info(
                        "check_prospective_fe_pairs: cross-pair chunk batching active "
                        "(%d pairs -> %d chunks, buffer %d cols, widest chunk %d pairs).",
                        len(prospective_pairs), len(_fe_chunks), _chunk_buf_width,
                        max(len(c) for c in _fe_chunks),
                    )
            except MemoryError:
                _chunk_buffer = None
                _pair_to_chunk = {}
                if verbose:
                    logger.warning(
                        "check_prospective_fe_pairs: cross-pair chunk buffer (%d x %d) raised "
                        "MemoryError; using per-pair batching.", len(X), _chunk_buf_width,
                    )
        else:
            _chunk_global_batch = False
    # Index of the chunk currently materialised in ``_chunk_buffer`` (-1 = none).
    _loaded_chunk_idx = -1

    # For every pair from the pool, try all known functions of 2 variables (not storing results in persistent RAM). Record best pairs.
    for (
        raw_vars_pair,
        pair_mi,
    ), _uplift in tqdmu(
        prospective_pairs.items(), desc="pair", leave=False, disable=not verbose
    ):  # better to start considering form the most prospective pairs with highest mis ratio!

        messages = []

        combs = pair_combs[raw_vars_pair]

        best_config, best_mi = None, -1
        this_pair_features = set()
        var_pairs_perf = {}
        # Pre-warp uplift tracking (2026-06-02): the best engineered MI achievable
        # with ONLY the elementary library unaries (no ``prewarp`` operand) vs the
        # best USING a prewarp operand. A 1-D engineered summary of a 2-D pair
        # cannot retain ``fe_min_engineered_mi_prevalence`` of the 2-D JOINT MI,
        # so on a non-monotone inner distortion (where the elementary library is
        # representationally blind) the prewarp winner is rejected by the joint
        # prevalence gate despite being a large, real uplift over the best the
        # library can do. The alternative acceptance path below admits a prewarp
        # winner when it beats the best non-prewarp engineered MI by a margin --
        # directed (only fires where the prewarp adds representational power) and
        # noise-safe (on linear/monotone/noise data the prewarp does not beat the
        # elementary library, so the margin is never cleared).
        best_nonprewarp_mi = -1.0
        best_nonprewarp_config = None
        best_prewarp_config, best_prewarp_mi = None, -1.0

        # CRITICAL #2 dispatch: hoist path uses the shared buffer (writes into
        # ``[:, i]``); recompute-fallback path uses a tiny 1D scratch + a
        # config-by-i map for on-demand survivor recomputation later.
        # CROSS-PAIR: when this pair was batched across the chunk, its survivor
        # columns live in the wide ``_chunk_buffer`` (the config's ``i`` is the
        # chunk-buffer column), so point ``final_transformed_vals`` at it. The chunk
        # is materialised LAZILY: pairs are processed in chunk-plan order, so when we
        # reach the FIRST pair of a not-yet-loaded chunk we fill the buffer + MI cache
        # for that whole chunk in ONE batched pass. By the time the next chunk's first
        # pair arrives, all of this chunk's pairs (incl. their survivor packing, which
        # reads the buffer) have already been processed -> safe to overwrite.
        _chunk_entry = None
        if _chunk_global_batch and (_chunk_buffer is not None):
            _my_chunk = _pair_to_chunk.get(raw_vars_pair)
            if _my_chunk is not None:
                if _my_chunk != _loaded_chunk_idx:
                    _chunk_mi_cache = _compute_one_fe_chunk(
                        chunk_pairs=_fe_chunks[_my_chunk],
                        pair_valid_combs=_pair_valid_combs,
                        chunk_buffer=_chunk_buffer,
                        vars_transformations=vars_transformations,
                        transformed_vars=transformed_vars,
                        binary_transformations=binary_transformations,
                        quantization_nbins=quantization_nbins,
                        quantization_dtype=quantization_dtype,
                        classes_y=classes_y,
                        classes_y_safe=classes_y_safe,
                        freqs_y=freqs_y,
                        fe_npermutations=fe_npermutations,
                        fe_min_nonzero_confidence=fe_min_nonzero_confidence,
                        batch_mi_kernel=batch_mi_with_noise_gate,
                        use_su=use_su_normalization(),
                        prewarp_unary=_PREWARP_UNARY,
                        logger=logger,
                        discretize_2d_quantile_batch=discretize_2d_quantile_batch,
                    )
                    _loaded_chunk_idx = _my_chunk
                _chunk_entry = _chunk_mi_cache.get(raw_vars_pair)
        if _chunk_entry is not None:
            final_transformed_vals = _chunk_buffer
        else:
            final_transformed_vals = final_transformed_vals_shared
        _col_buf_1d: np.ndarray | None = (
            np.empty(len(X), dtype=np.float32) if _need_recompute_map else None
        )
        _config_by_i: dict[int, tuple] = {} if _need_recompute_map else None

        i = 0
        # Per-pair thread-local timing accumulator; merged into the shared
        # ``times_spent`` under the lock once per pair (see end of pair loop).
        _local_times: dict = {}

        # BATCHED-DISCRETIZE dispatch (2026-06-04): per-candidate ``discretize_array``
        # (np.linspace + np.nanpercentile->partition + searchsorted) is the FE-pair-search
        # hotspot -- millions of tiny per-column numpy calls -> serial-dispatch-bound,
        # idle CPU. On the HOIST path (shared buffer present) with the quantile method we
        # split this pair's sweep into 3 phases: (1) materialise ALL candidate columns into
        # the buffer + nan_to_num + record (config, idx, uses_pw); (2) batch-discretise the
        # filled buffer slice in ONE ``np.nanpercentile(axis=0)`` (amortises dispatch over K
        # columns -- bit-identical to per-column, see ``discretize_2d_quantile_batch``);
        # (3) replay the EXACT per-candidate mi_direct + best/prewarp/config tracking.
        # MI stays per-candidate (mi_direct's permutation confidence is NOT batched). The
        # recompute-fallback (no buffer) and the uniform method keep the original
        # per-candidate path verbatim -- only the hoist+quantile case is batched.
        _use_batch_disc = (final_transformed_vals is not None) and (quantization_method == "quantile")

        if _use_batch_disc:
            # CROSS-PAIR fast path: this pair was batched together with the rest of its
            # chunk in ``_compute_one_fe_chunk``. Its candidate columns already live
            # in ``_chunk_buffer`` (the buf_col index is the config ``i``), its MI is
            # already computed, and its per-bin_func materialise timings are recorded.
            # We only replay the EXACT per-candidate tracking below. Bit-identical: the
            # chunk's ONE discretize_2d + ONE batch_mi score each column independently,
            # and the candidate order per pair is the SAME (combs x bin_funcs) order the
            # per-pair Phase 1 produced.
            if _chunk_entry is not None:
                _batch_candidates, _fe_mi_by_col, _pair_local_times = _chunk_entry
                for _bf_name, _dt in _pair_local_times.items():
                    _local_times[_bf_name] = _local_times.get(_bf_name, 0.0) + _dt
                # ``_fe_mi_arr`` is indexed by the chunk-buffer column (buf_col), so the
                # replay's ``_fe_mi_arr[_ci]`` lookup is correct without re-indexing.
                _fe_mi_arr = _fe_mi_by_col
            else:
                # Phase 1: materialise + nan_to_num + record. ``i`` advances exactly as in the
                # per-candidate path so ``config``'s buffer index and ``_config_by_i`` are identical.
                _batch_candidates = []  # (transformations_pair, bin_func_name, i, uses_pw)
                for transformations_pair in combs:
                    if (transformations_pair[0] not in vars_transformations) or (transformations_pair[1] not in vars_transformations):
                        continue
                    param_a = transformed_vars[:, vars_transformations[transformations_pair[0]]]
                    param_b = transformed_vars[:, vars_transformations[transformations_pair[1]]]
                    _uses_pw = (
                        transformations_pair[0][1] == _PREWARP_UNARY
                        or transformations_pair[1][1] == _PREWARP_UNARY
                    )
                    # Same wide errstate scope as the original per-pair-comb path.
                    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                        for bin_func_name, bin_func in binary_transformations.items():
                            start = timer()
                            try:
                                final_transformed_vals[:, i] = bin_func(param_a, param_b)
                                _col_view = final_transformed_vals[:, i]
                            except Exception:
                                logger.error(f"Error when performing {bin_func}")
                            else:
                                np.nan_to_num(_col_view, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                                _local_times[bin_func_name] = _local_times.get(bin_func_name, 0.0) + (timer() - start)
                                _batch_candidates.append((transformations_pair, bin_func_name, i, _uses_pw))
                                i += 1

                # Phase 2: ONE batch discretisation over the materialised columns [:, :n].
                # Bit-identical to per-column ``discretize_array(method='quantile')`` -- the
                # buffer dtype (float32) is NOT cast; per-column edges/codes match exactly.
                _fe_mi_arr = None
                if _batch_candidates:
                    # ``i`` advanced once per materialised candidate from 0 (reset per raw-pair),
                    # so the filled buffer slice is exactly [:, :i], densely packed 0..i-1.
                    _disc_2d = discretize_2d_quantile_batch(
                        final_transformed_vals[:, :i], n_bins=quantization_nbins, dtype=quantization_dtype
                    )

                    # Phase 3: BATCHED MI + permutation noise-gate across ALL K candidate
                    # columns in ONE kernel call. Bit-identical to the per-candidate
                    # ``mi_direct`` loop on the default FE path (parallelism='outer',
                    # n_workers=1 -> parallel_mi_prange, base_seed=0): every candidate is
                    # tested against the SAME npermutations shuffles of y (the shuffle is
                    # seeded by (base_seed, perm_index) ONLY, never by classes_x), so a single
                    # batched kernel can shuffle y once per permutation and score all columns
                    # against it -- amortising both the MI compute and the shuffle across K.
                    # ``_dispatch_batch_mi_with_noise_gate`` routes CPU-njit vs a GPU batched
                    # path by n*K via the kernel_tuning_cache (no hardcoded threshold).
                    _fe_mi_arr = _dispatch_batch_mi_with_noise_gate(
                        disc_2d=_disc_2d,
                        quantization_nbins=quantization_nbins,
                        classes_y=classes_y,
                        classes_y_safe=classes_y_safe,
                        freqs_y=freqs_y,
                        npermutations=fe_npermutations,
                        min_nonzero_confidence=fe_min_nonzero_confidence,
                        use_su=use_su_normalization(),
                        batch_mi_kernel=batch_mi_with_noise_gate,
                    )

            # Replay best/prewarp/config tracking in the SAME order candidates were
            # produced -> identical tie-break behaviour. ``_fe_mi_arr`` is indexed by the
            # buffer column (per-pair: 0..K-1; cross-pair: the chunk-buffer column).
            if _batch_candidates and _fe_mi_arr is not None:
                for transformations_pair, bin_func_name, _ci, _uses_pw in _batch_candidates:
                    # Cast to Python float so ``var_pairs_perf`` / downstream tracking see
                    # the same scalar type ``mi_direct`` returned (numba njit returns a
                    # python float at the call boundary). Value is bit-identical.
                    fe_mi = float(_fe_mi_arr[_ci])

                    config = (transformations_pair, bin_func_name, _ci)
                    var_pairs_perf[config] = fe_mi
                    if _need_recompute_map:
                        _config_by_i[_ci] = (transformations_pair[0], transformations_pair[1], bin_func_name)

                    if fe_mi > best_mi:
                        best_mi = fe_mi
                        best_config = config
                    if _uses_pw:
                        if fe_mi > best_prewarp_mi:
                            best_prewarp_mi = fe_mi
                            best_prewarp_config = config
                    else:
                        if fe_mi > best_nonprewarp_mi:
                            best_nonprewarp_mi = fe_mi
                            best_nonprewarp_config = config
                    if fe_mi > best_mi * 0.85:
                        if not fe_print_best_mis_only or (fe_mi == best_mi):
                            if verbose > 2:
                                print(f"MI of transformed pair {bin_func_name}({transformations_pair})={fe_mi:.4f}, MI of the plain pair {pair_mi:.4f}")
        else:
            for transformations_pair in combs:
                if (transformations_pair[0] not in vars_transformations) or (transformations_pair[1] not in vars_transformations):
                    continue
                param_a = transformed_vars[:, vars_transformations[transformations_pair[0]]]
                param_b = transformed_vars[:, vars_transformations[transformations_pair[1]]]

                # A config "uses prewarp" iff either operand's unary name is the
                # pseudo-unary. Invariant across the bin_func loop -> compute once.
                _uses_pw = (
                    transformations_pair[0][1] == _PREWARP_UNARY
                    or transformations_pair[1][1] == _PREWARP_UNARY
                )

                # ``bin_func`` produces NaN/+-inf on extreme Optuna-picked params
                # (overflow in mul/exp, divide-by-zero in log); the downstream
                # nan_to_num + MI gate already sanitise, so the bare numpy
                # RuntimeWarnings carry zero diagnostic value. Suppress them for the
                # whole binary-transform sweep: entering np.errstate per inner
                # iteration cost ~6.8us/iter (measured ~490ms over 72k iters),
                # dwarfing the bin_func work itself; one context per pair-comb
                # removes that. numba kernels (discretize/mi_direct) ignore errstate
                # and nan_to_num emits nothing, so the wider scope is value-identical.
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    for bin_func_name, bin_func in binary_transformations.items():

                        start = timer()
                        try:
                            if final_transformed_vals is not None:
                                final_transformed_vals[:, i] = bin_func(param_a, param_b)
                                _col_view = final_transformed_vals[:, i]
                            else:
                                # Recompute fallback: write into the shared 1D scratch.
                                # bin_func returns a fresh ndarray; copy into the scratch
                                # so downstream nan_to_num + discretize see contiguous
                                # data. Avoids accumulating one alloc per inner iter.
                                _col_buf_1d[:] = bin_func(param_a, param_b)
                                _col_view = _col_buf_1d
                        except Exception:
                            logger.error(f"Error when performing {bin_func}")
                        else:
                            np.nan_to_num(_col_view, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

                            # Wave 27 P1: ``times_spent`` is shared across mrmr.py's
                            # parallel threading dispatch. Accumulate this pair's
                            # per-bin_func timings in a thread-LOCAL dict and merge them
                            # under ``_TIMES_SPENT_LOCK`` once per pair (below); the old
                            # per-inner-iteration lock was a serialization point on the
                            # hot path. Totals are identical.
                            _local_times[bin_func_name] = _local_times.get(bin_func_name, 0.0) + (timer() - start)

                            discretized_transformed_values = discretize_array(
                                arr=_col_view, n_bins=quantization_nbins, method=quantization_method, dtype=quantization_dtype
                            )
                            fe_mi, fe_conf = mi_direct(
                                discretized_transformed_values.reshape(-1, 1),
                                x=np.array([0], dtype=np.int64),
                                y=None,
                                factors_nbins=np.array([quantization_nbins], dtype=np.int64),
                                classes_y=classes_y,
                                classes_y_safe=classes_y_safe,
                                freqs_y=freqs_y,
                                min_nonzero_confidence=fe_min_nonzero_confidence,
                                npermutations=fe_npermutations,
                            )

                            config = (transformations_pair, bin_func_name, i)
                            var_pairs_perf[config] = fe_mi
                            if _need_recompute_map:
                                # Map i -> (a_key, b_key, bin_func_name) for downstream
                                # rebuild; bin_func is looked up via the original dict.
                                _config_by_i[i] = (transformations_pair[0], transformations_pair[1], bin_func_name)

                            if fe_mi > best_mi:
                                best_mi = fe_mi
                                best_config = config
                            # Track best-with-prewarp vs best-without so the alternative
                            # uplift gate below can decide whether the prewarp earned its
                            # place (``_uses_pw`` hoisted above the bin_func loop).
                            if _uses_pw:
                                if fe_mi > best_prewarp_mi:
                                    best_prewarp_mi = fe_mi
                                    best_prewarp_config = config
                            else:
                                if fe_mi > best_nonprewarp_mi:
                                    best_nonprewarp_mi = fe_mi
                                    best_nonprewarp_config = config
                            if fe_mi > best_mi * 0.85:
                                if not fe_print_best_mis_only or (fe_mi == best_mi):
                                    if verbose > 2:
                                        print(f"MI of transformed pair {bin_func_name}({transformations_pair})={fe_mi:.4f}, MI of the plain pair {pair_mi:.4f}")
                            i += 1

        # Merge this pair's per-bin_func timings into the shared accumulator in
        # ONE locked pass (the increment was previously locked per inner
        # iteration -- a serialization point under the parallel pair dispatch).
        if _local_times:
            with _TIMES_SPENT_LOCK:
                for _bf, _dt in _local_times.items():
                    times_spent[_bf] += _dt

        if verbose > 2:
            print(f"For pair {raw_vars_pair}, best config is {best_config} with best mi= {best_mi}")

        # experiment-rejected (2026-06-03): a held-out-CV firewall here (score
        # per-combo MI on a TRAIN stride slice for honest selection, then keep the
        # winner only if its held-out VAL-slice MI retains >= ratio of train MI) was
        # implemented and benched END-TO-END on Layer-49 -- NO gain. In an isolated
        # probe it separated cleanly (genuine synergy val/train 0.90-1.04 vs noise-FE
        # 0.12-0.36), BUT in the real pipeline the tighter prevalence-gate defaults
        # (fe_synergy_min_prevalence 1.5 / fe_min_engineered_mi_prevalence 0.97)
        # already remove the pure noise*noise products, and the RESIDUAL "noise" FE
        # are signal*noise combos (e.g. max(log(L4_s2),noise_3) -- L4_s2 is a real
        # sensor) that genuinely generalise (val/train > 0.5) and SHOULD be kept; the
        # firewall's train-based selection (half the rows) then merely added selection
        # noise (+1 support). Prevalence gating subsumes the win -> not shipped.
        # Standard acceptance: the best engineered MI clears the configured
        # fraction of the 2-D pair-joint MI.
        _passes_joint_gate = best_mi / pair_mi > fe_min_engineered_mi_prevalence * (1.0 if num_fs_steps < 1 else 1.025)

        # Alternative pre-warp acceptance (2026-06-02): the joint-prevalence gate
        # structurally rejects a 1-D summary of a 2-D pair on a non-monotone inner
        # distortion. Admit the prewarp winner when it beats the best NON-prewarp
        # engineered MI by ``prewarp_uplift_threshold`` AND clears the pair-MI
        # noise floor (its MI must exceed the larger individual operand MI -- the
        # same notion the smart_polynom baseline uplift uses), so it cannot fire
        # on noise (where prewarp does not beat the library) or pure-linear data
        # (where the elementary library already saturates and the prewarp adds no
        # uplift). When it fires, the prewarp config becomes the winner.
        _prewarp_accept = False
        if (
            _prewarp_active
            and not _passes_joint_gate
            and best_prewarp_config is not None
            and best_nonprewarp_mi > 0.0
            and best_prewarp_mi >= best_nonprewarp_mi * float(prewarp_uplift_threshold)
        ):
            _prewarp_accept = True
            # Promote the prewarp winner to the pair's winner so the standard
            # leading-features / single-best materialisation path emits it.
            best_config, best_mi = best_prewarp_config, best_prewarp_mi
            if verbose:
                messages.append(
                    f"pre-warp uplift gate: best prewarp MI={best_prewarp_mi:.4f} "
                    f"beats best non-prewarp MI={best_nonprewarp_mi:.4f} by "
                    f">= {float(prewarp_uplift_threshold):.2f}x (joint-prevalence "
                    f"gate {best_mi / pair_mi:.3f} < {fe_min_engineered_mi_prevalence:.2f} "
                    f"would have rejected it); admitting the prewarp feature."
                )

        # MARGINAL-UPLIFT alternative acceptance: admit a pair the joint-prevalence
        # gate rejects when its best ELEMENTARY-LIBRARY (non-prewarp) engineered column
        # beats the LARGER individual operand marginal MI by ``_FE_MARGINAL_UPLIFT_MIN_RATIO``
        # AND still recovers at least ``_FE_MARGINAL_UPLIFT_MIN_JOINT_RATIO`` of the inflated
        # 2-D joint. Rationale + thresholds: see the module-level constants. Genuine synergy
        # pairs (a**2/b, log(c)*sin(d)) clear both; cross-pair artefacts that merely recapture
        # one operand's marginal fail the uplift bar, and structureless noise pairs never reach
        # here (the upstream pair screen + order-2 maxT floor remove them). Only fires when the
        # primary joint gate AND the prewarp path both declined, so it is purely additive recall
        # for genuine pairs the strict joint bar drops. We score + promote the best NON-PREWARP
        # winner: the prewarp pseudo-unary has its own dedicated acceptance path above
        # (``_prewarp_accept``), and promoting a prewarp form here would require the per-operand
        # warp spec to round-trip into the recipe -- which is only guaranteed on the prewarp path.
        _marginal_uplift_accept = False
        if (
            not _passes_joint_gate
            and not _prewarp_accept
            and best_nonprewarp_config is not None
            and best_nonprewarp_mi > 0.0
            and pair_mi > 0.0
        ):
            _max_operand_marginal = max(
                _operand_marginal_mi(raw_vars_pair[0]),
                _operand_marginal_mi(raw_vars_pair[1]),
            )
            _joint_ratio = best_nonprewarp_mi / pair_mi
            if (
                _max_operand_marginal > 0.0
                and best_nonprewarp_mi >= _max_operand_marginal * _FE_MARGINAL_UPLIFT_MIN_RATIO
                and _joint_ratio >= _FE_MARGINAL_UPLIFT_MIN_JOINT_RATIO
            ):
                _marginal_uplift_accept = True
                # Promote the best non-prewarp form so the standard single-best
                # materialisation path emits a recipe-replayable winner.
                best_config, best_mi = best_nonprewarp_config, best_nonprewarp_mi
                if verbose:
                    messages.append(
                        f"marginal-uplift gate: best non-prewarp engineered MI={best_nonprewarp_mi:.4f} "
                        f"beats the larger operand marginal MI={_max_operand_marginal:.4f} by "
                        f">= {_FE_MARGINAL_UPLIFT_MIN_RATIO:.2f}x and recovers {_joint_ratio:.3f} "
                        f"of the 2-D joint (joint-prevalence gate "
                        f"{fe_min_engineered_mi_prevalence:.2f} would have rejected it); "
                        f"admitting the genuine synergy pair."
                    )

        if _passes_joint_gate or _prewarp_accept or _marginal_uplift_accept:  # Best transformation is good enough

            # If there is a group of leaders with almost the same performance, approve them through one of the other variables.
            # если будут возникать такие группы примерно одинаковых по силе лидеров, их придётся разрешать с помощью одного из других влияющих факторов
            # When the pair was admitted ONLY via the marginal-uplift path (the joint /
            # prewarp gates declined), the winner MUST be a non-prewarp form so the recipe
            # is replayable -- restrict the leaders to elementary-library configs.
            _restrict_to_nonprewarp = _marginal_uplift_accept and not (_passes_joint_gate or _prewarp_accept)
            leading_features = []
            for next_config, next_mi in sort_dict_by_value(var_pairs_perf).items():
                if next_mi > best_mi * fe_good_to_best_feature_mi_threshold:
                    if _restrict_to_nonprewarp and (
                        next_config[0][0][1] == _PREWARP_UNARY or next_config[0][1][1] == _PREWARP_UNARY
                    ):
                        continue
                    leading_features.append(next_config)

            if len(leading_features) > 1:
                if len(numeric_vars_to_consider) > 2:

                    if verbose > 2:
                        print(f"Taking {len(leading_features)} new features for a separate validation step!")

                    # Test all candidates as-is against the rest of the approved factors (also as-is). Candidates significantly outstanding (in terms of MI with target)
                    # against any other approved factor are kept.
                    valid_pairs_perf = {}

                    for transformations_pair, bin_func_name, i in leading_features:
                        if final_transformed_vals is not None:
                            param_a = final_transformed_vals[:, i]
                        else:
                            # CRITICAL #2 recompute-fallback: rebuild the survivor column from its
                            # (a_key, b_key, bin_func_name) metadata. transformed_vars is small
                            # (deduped unary table); the bin_func call is cheap (one ufunc).
                            _a_key, _b_key, _bin_name = _config_by_i[i]
                            _pa = transformed_vars[:, vars_transformations[_a_key]]
                            _pb = transformed_vars[:, vars_transformations[_b_key]]
                            param_a = binary_transformations[_bin_name](_pa, _pb)
                            np.nan_to_num(param_a, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

                        best_valid_mi = -1
                        config = (transformations_pair, bin_func_name, i)

                        external_factors = list(set(numeric_vars_to_consider) - set(raw_vars_pair))
                        if fe_max_external_validation_factors and len(external_factors) > fe_max_external_validation_factors:
                            external_factors = np.random.choice(external_factors, fe_max_external_validation_factors)

                        for external_factor in tqdmu(external_factors, desc="external validation factor", leave=False, disable=not verbose):
                            if external_factor not in original_cols:
                                continue
                            if isinstance(X, pd.DataFrame):
                                param_b = X.iloc[:, original_cols[external_factor]].values
                            else:
                                param_b = X[:, original_cols[external_factor]].to_numpy()

                            for valid_bin_func_name, valid_bin_func in binary_transformations.items():

                                valid_vals = valid_bin_func(param_a, param_b)

                                discretized_transformed_values = discretize_array(
                                    arr=valid_vals, n_bins=quantization_nbins, method=quantization_method, dtype=quantization_dtype
                                )
                                fe_mi, fe_conf = mi_direct(
                                    discretized_transformed_values.reshape(-1, 1),
                                    x=np.array([0], dtype=np.int64),
                                    y=None,
                                    factors_nbins=np.array([quantization_nbins], dtype=np.int64),
                                    classes_y=classes_y,
                                    classes_y_safe=classes_y_safe,
                                    freqs_y=freqs_y,
                                    min_nonzero_confidence=fe_min_nonzero_confidence,
                                    npermutations=fe_npermutations,
                                )

                                if fe_mi > best_valid_mi:
                                    best_valid_mi = fe_mi
                                    if verbose > 2:
                                        print(
                                            f"MI of transformed pair {valid_bin_func_name}({(transformations_pair,bin_func_name)} with ext factor {external_factor})={fe_mi:.4f}"
                                        )

                        valid_pairs_perf[config] = best_valid_mi

                    # ONE-BEST-PER-PAIR (2026-06-01): the leading-features
                    # equivalence class holds many near-identical representations
                    # of the same algebraic target (a**2/b == div(sqr(a),b) ==
                    # mul(sqr(a),reciproc(b)) == div(a,sqrt(b)) ...). The
                    # pre-refactor code materialised EXACTLY ONE per raw pair;
                    # the refactor regressed to emitting the whole class (~15
                    # cols on the canonical fixture). Pick the single best by
                    # TARGET MI (``var_pairs_perf`` -- the primary objective),
                    # using the external-validation MI (``valid_pairs_perf``)
                    # only as a tie-break among target-MI-equal leaders. (Prior
                    # bug: selected by external-validation MI alone, discarding
                    # the true max-target-MI form -- e.g. picking add(log(c),1/d)
                    # MI=0.25 over the true mul(log(c),sin(d)) MI=0.32.)
                    _primary_perf = {c: var_pairs_perf[c] for c in valid_pairs_perf if c in var_pairs_perf}
                    _winner = _select_single_best(_primary_perf, cols, secondary=valid_pairs_perf)
                    if _winner is not None:
                        new_feature_name = get_new_feature_name(fe_tuple=_winner, cols_names=cols)
                        if verbose:
                            messages.append(
                                f"{new_feature_name} is recommended to use as a new feature! (won in validation with other factors) best_mi={best_mi:.4f}, pair_mi={pair_mi:.4f}, rat={best_mi/pair_mi:.4f}"
                            )
                        this_pair_features.add((_winner, 0))
                else:
                    # Can't narrow by external validation (only 2 vars total) --
                    # still emit ONE best representative (highest engineered MI,
                    # deterministic name tie-break) rather than the whole class.
                    _lead_perf = {c: var_pairs_perf[c] for c in leading_features if c in var_pairs_perf}
                    _winner = _select_single_best(_lead_perf, cols)
                    if _winner is not None:
                        if verbose:
                            messages.append(
                                f"{get_new_feature_name(fe_tuple=_winner, cols_names=cols)} is recommended to use as a new feature! (best of {len(leading_features)} near-equivalent leaders) best_mi={best_mi:.4f}, pair_mi={pair_mi:.4f}, rat={best_mi/pair_mi:.4f}"
                            )
                        this_pair_features.add((_winner, 0))
            else:
                new_feature_name = get_new_feature_name(fe_tuple=best_config, cols_names=cols)
                if verbose:
                    messages.append(
                        f"{new_feature_name} is recommended to use as a new feature! (clear winner) best_mi={best_mi:.4f}, pair_mi={pair_mi:.4f}, rat={best_mi/pair_mi:.4f}"
                    )
                j = 0
                this_pair_features.add((best_config, j))

            transformed_vals, new_cols, new_nbins = None, None, None

            if this_pair_features:

                # Bulk add the found & checked best features.
                # ``this_pair_features`` is a SET of (config, j) tuples
                # with sparse, non-contiguous ``j`` indices into
                # ``final_transformed_vals``. The consumer (mrmr.py
                # ``_run_fe_step``) iterates
                # ``for k in range(len(this_pair_features)):
                # transformed_vals[:, k]``, so the buffer MUST have
                # exactly ``len(this_pair_features)`` columns packed
                # densely 0..N-1, not the sparse ``j``-indexed layout
                # with holes. Pre-fix code wrote to ``transformed_vals[:, j]``
                # then sliced to ``[:, :last_j + 1]`` -- this gives
                # either a too-short buffer (if last_j was small) and
                # IndexError downstream, or holes (if last_j was large).
                # Pack each (config, j) into a compact column index
                # ``idx = 0..len(this_pair_features)-1`` instead.
                #
                # 2026-06-01 (ROOT CAUSE 5 fix): materialise the survivor
                # columns whenever FE runs (``fe_max_steps >= 1``), not only on
                # multi-step (``> 1``). Previously, with the default
                # ``fe_max_steps=1`` the recommended features were LOGGED but
                # ``transformed_vals`` stayed ``None`` -- so the consumer
                # (_mrmr_fe_step) had nothing to append, the columns never
                # entered ``data``/``selected_vars``, and ``_engineered_features_``
                # stayed empty. Producing the buffer unconditionally lets the
                # single-step default actually emit engineered columns.
                # Materialise each survivor into a temp column FIRST, then apply
                # the NON-CONSTANT guard (2026-06-01): a column that replays as
                # constant (std<=1e-9) or non-finite is a DEAD feature and must
                # never be appended -- it reaches the downstream model carrying
                # zero variance. Several div(sqr(a),b)-family combos replayed
                # constant on the canonical fixture (degenerate quantile binning
                # of the heavy-tailed a**2/b). One-best-per-pair already keeps the
                # non-constant MI winner; this guard is defence-in-depth and also
                # compacts ``this_pair_features`` / buffers so the recipe builder
                # downstream never constructs a recipe for a dropped column.
                _kept_configs = []   # list[(config, j)] that survived the guard
                _kept_cols_vals = []  # list[np.ndarray] aligned with _kept_configs
                _kept_names = []

                for idx, (config, j) in enumerate(this_pair_features):
                    new_feature_name = get_new_feature_name(fe_tuple=config, cols_names=cols)
                    transformations_pair, bin_func_name, i = config

                    if fe_max_steps >= 1:
                        if _use_subsample:
                            # SUBSAMPLE path: rebuild from raw _X_full so the survivor column
                            # carries the FULL n rows the caller expects (mrmr.py appends it
                            # back to its full-n ``data`` array). The MI sweep used a 200k
                            # subset; the survivor IDENTITIES are correct (bench shows
                            # jaccard=1.0 vs full-n at n_eff>=50k), so we just need to
                            # rematerialise the values at full resolution.
                            _col_full = _rebuild_full_survivor_col(
                                config, _X_full, original_cols,
                                unary_transformations, binary_transformations,
                                prewarp_spec_by_var=_prewarp_spec_by_var,
                                gate_med_median_by_var=_gate_med_median_by_var,
                            )
                        elif final_transformed_vals is not None:
                            _col_full = final_transformed_vals[:, i]
                        else:
                            # CRITICAL #2 recompute-fallback (no subsample, tight RAM): rebuild
                            # the survivor column from its (a_key, b_key, bin_func_name)
                            # metadata via the cached unary table. transformed_vars is at
                            # full n in this path so the column lands at full n directly.
                            _a_key, _b_key, _bin_name = _config_by_i[i]
                            _pa = transformed_vars[:, vars_transformations[_a_key]]
                            _pb = transformed_vars[:, vars_transformations[_b_key]]
                            _col = binary_transformations[_bin_name](_pa, _pb)
                            np.nan_to_num(_col, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                            _col_full = _col

                        # Keep the RAW (float) engineered values, scrubbed of
                        # nan/inf. CRITICAL (2026-06-02): do NOT cast to the
                        # integer ``quantization_dtype`` here. ``transformed_vals``
                        # feeds two consumers downstream -- (a) ``_mrmr_fe_step``
                        # discretises it via ``discretize_array(method=quantile)``
                        # into the ``data`` bin-code matrix, and (b) the recipe
                        # builder computes its quantile EDGES from these values for
                        # leak-safe replay. A premature int cast TRUNCATES the
                        # heavy-tailed engineered values (e.g. mul(log(c),sin(d)) in
                        # (-inf,0] collapses to ~2 integers), so the subsequent
                        # quantile binning sees only 2-3 distinct values and the
                        # column reaches the model with a fraction of its MI
                        # (measured: 0.14 vs the true 0.32). Keeping float lets the
                        # downstream quantile discretiser produce the full nbins
                        # codes and the recipe pin correct edges.
                        _col_arr = np.nan_to_num(
                            np.asarray(_col_full, dtype=np.float64),
                            nan=0.0, posinf=0.0, neginf=0.0,
                        )
                        if float(np.std(_col_arr)) <= 1e-9:
                            if verbose:
                                messages.append(
                                    f"{new_feature_name} dropped at materialisation: dead column "
                                    f"(std={float(np.std(_col_arr)):.2e}, non-constant guard)."
                                )
                            continue
                        _kept_cols_vals.append(_col_arr)
                    _kept_configs.append((config, j))
                    _kept_names.append(new_feature_name)

                # Rebuild the survivor set / buffers from ONLY the kept columns so
                # the recipe builder and the downstream dense consumer stay aligned.
                this_pair_features = set(_kept_configs)
                new_cols = list(_kept_names)
                if fe_max_steps >= 1 and _kept_cols_vals:
                    # float buffer: holds RAW engineered values (discretised to
                    # codes downstream; see the non-constant-guard comment above).
                    transformed_vals = np.empty(shape=(_full_n_rows, len(_kept_cols_vals)), dtype=np.float64)
                    for _ci, _cv in enumerate(_kept_cols_vals):
                        transformed_vals[:, _ci] = _cv
                    new_nbins = [quantization_nbins] * len(_kept_cols_vals)
                else:
                    transformed_vals, new_nbins = None, []

            res[raw_vars_pair] = (this_pair_features, transformed_vals, new_cols, new_nbins, messages)

    # Surface the fitted per-operand pre-warp specs (keyed by cols-space var
    # index) so the caller (``_mrmr_fe_step``) can persist them in each survivor
    # recipe for leak-safe replay. Only the non-None specs that were actually
    # fitted are exported. We populate BOTH the optional ``prewarp_specs_out``
    # side-channel (works for the in-process serial path) AND a reserved key in
    # the returned ``res`` (survives the loky-parallel path where the side
    # channel dict cannot be mutated cross-process; the caller merges per-chunk
    # results). The reserved key is a private 3-tuple that can never collide with
    # a real ``raw_vars_pair`` (which is always length 2).
    _fitted_specs = {_v: _s for _v, _s in _prewarp_spec_by_var.items() if _s is not None}
    if _fitted_specs:
        if prewarp_specs_out is not None:
            prewarp_specs_out.update(_fitted_specs)
        res[_PREWARP_SPECS_RESULT_KEY] = _fitted_specs

    # Same dual-channel export for the fitted per-operand TRAIN medians so the
    # caller can persist them in each survivor recipe for leak-safe replay. The
    # value is a single float per cols-space var index.
    _fitted_medians = {_v: float(_m) for _v, _m in _gate_med_median_by_var.items()}
    if _fitted_medians:
        if gate_med_specs_out is not None:
            gate_med_specs_out.update(_fitted_medians)
        res[_GATE_MED_SPECS_RESULT_KEY] = _fitted_medians

    return res
