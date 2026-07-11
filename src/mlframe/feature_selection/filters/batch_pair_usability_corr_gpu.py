"""GPU-batched variant of the FE usability |corr| signal (:mod:`_fe_usability_signal`).

``_fe_usability_signal.abs_pearson``/``usability_form_corrs``/``pair_is_tail_concentrated_rankaware`` are
CPU-only (numpy-only leaf module, no cupy import by design) -- 2026-07-11 profiling on a 79,237-row x
544-column production wellbore fit showed ~105s cumtime here, ~85,000 calls, with ZERO GPU offload anywhere
in the module. Unlike the batched pair-MI kernel (:mod:`batch_pair_mi_gpu`), a per-pair GPU dispatch here
would lose badly (each reduction runs over only ``_ABS_PEARSON_MAX_ROWS`` <= 30_000 subsampled rows -- far
below the ~400k-row CPU/CUDA crossover the numerical-kernel ladder documents) -- so the design bet was that
the win would exist as one launch batching MANY (pair, form) reductions at once (production shape: ~85k
pairs x up to 9 forms each ~= 765k independent reductions).

MEASURED RESULT (2026-07-11, see ``_benchmarks/bench_batch_pair_usability_corr_gpu.py``): that bet did NOT
pay off on this dev host (GTX 1050 Ti). The CUDA backend is bit-identical (see below) but SLOWER than the
CPU ``prange`` backend at every tested scale from 144 to 1.35M total (pair, form) reductions -- 0.05x at
the smallest, converging to ~0.53-0.57x (worse, not better) right at and beyond the real ~85k-pair
production scale.

This was FIRST measured end-to-end (host arrays in, host array out -- includes H2D upload + D2H download
inside the timed call) and initially attributed to poor memory coalescing without isolating transfer cost
from compute -- a real gap: a kernel that loses end-to-end can still win on pure compute if the caller's
data is already GPU-resident, and the fix there would be residency, not a backend revert (see
CLAUDE.md's GPU-profiling-traps rule this shortcut skipped). Re-measured with H2D/kernel/D2H DECOMPOSED:
uploading the full (500, 30_000) f64 operand matrix (120MB) took 0.020s ONE-TIME (amortizable, negligible
next to the kernel's own 0.1-48s range across the sweep); D2H of the (n_pairs, n_forms) result was
0.0004-0.005s (negligible). The kernel's OWN execution time with operands already resident on-device was
still ~0.53-0.60x vs CPU at every scale -- statistically indistinguishable from the original end-to-end
number. Transfer cost was never the story: the reduction genuinely is memory-bandwidth-bound, not
launch-overhead-bound, confirmed on the resident-input measurement, not asserted from the end-to-end one.
Each thread does two full sequential passes over its own (pair, form)'s 30_000 rows, and since different
threads read DIFFERENT operand-matrix rows, the reads are inherently uncoalesced -- more batch volume
cannot amortize this the way it amortizes a fixed launch cost. This matches this card's already-documented
0.26-0.66x underperformance on OTHER resident FE kernels (``_permutation_null_pair_resident.py``). The
dispatcher's un-forced default is therefore CPU -- see :func:`dispatch_batch_pair_usability_corr`'s
docstring. The CUDA backend is kept (REJECTED != DELETED): fully implemented, tested, bit-identical, and
reachable via ``force_backend="cuda"``
for a stronger production GPU or a future ``kernel_tuning_cache`` sweep that might find a real per-host
crossover this dev card does not have.

Two backends:

* ``batch_pair_usability_corr_njit_parallel`` -- CPU reference (``@njit(parallel=True)``, ``prange`` over
  (pair, form) flattened index). Numerical baseline; also the fallback when CUDA is unavailable.
* ``batch_pair_usability_corr_cuda`` -- ``numba.cuda`` JIT kernel. One THREAD per (pair, form) -- mirrors
  ``_batch_mi_noise_gate_kernels._cuda_mi_from_counts_kernel_factory``'s "one thread per independent
  reduction" shape (not a shared-memory block-per-reduction design: each reduction here is a flat
  two-pass moment computation, not a histogram, so there is no shared-memory accumulator to stage).

Both backends reproduce ``_abs_pearson_njit``'s EXACT algorithm (two-pass mean-then-center, branchless
isfinite masking, the ``_cv2=1e-16`` coefficient-of-variation degenerate floor) for EVERY one of the 9
candidate forms (``x0, x1, x0**2, x1**2, x0/x1, x1/x0, x0**2/x1, x1**2/x0, x0*x1``), computed ON-DEVICE
from two shared raw-operand columns per pair -- the host never materializes a 9-times-wider form matrix.
A single-pass (mean-and-variance-in-one-sweep) formula was deliberately NOT used: it reproduces the exact
catastrophic-cancellation bug ``_abs_pearson_njit``'s own docstring documents rejecting (a near-constant
column returning a spurious nonzero |corr| instead of ~0).

Numerical equivalence vs ``_abs_pearson_njit`` is verified by
``tests/feature_selection/gpu/test_batch_pair_usability_corr_gpu.py`` (CPU always; GPU variant when CUDA
is available, auto-skip otherwise).
"""
from __future__ import annotations

import logging
import math
from typing import Any

import numba
import numpy as np
from numba import prange

logger = logging.getLogger(__name__)

try:
    from numba import cuda as _nb_cuda
except Exception:
    _nb_cuda = None

try:
    from pyutilz.core.pythonlib import is_cuda_available as _pyutilz_is_cuda_available
    _CUDA_AVAIL = _pyutilz_is_cuda_available()
except Exception:
    try:
        _CUDA_AVAIL = bool(getattr(_nb_cuda, "is_available", lambda: False)()) if _nb_cuda is not None else False
    except Exception:
        _CUDA_AVAIL = False

# Require numba.cuda to actually compile+launch a kernel (not just device presence) so a cudatoolkit/NVVM
# mismatch routes to CPU instead of raising NvvmSupportError mid-dispatch (mirrors batch_pair_mi_gpu.py).
try:
    from ._internals import numba_cuda_can_compile as _numba_cuda_can_compile
    _CUDA_AVAIL = _CUDA_AVAIL and _numba_cuda_can_compile()
except Exception:
    _CUDA_AVAIL = False

# Integer dispatch enum for the 9 forms -- matches usability_form_corrs's _single_forms + _pair_forms order
# EXACTLY (single: x0, x1, x0sq, x1sq; pair: x0/x1f, x1/x0f, x0sq/x1f, x1sq/x0f, x0*x1) so a caller can map
# its existing form bookkeeping straight onto these integer ids.
FORM_X0 = 0
FORM_X1 = 1
FORM_X0_SQ = 2
FORM_X1_SQ = 3
FORM_X0_DIV_X1 = 4
FORM_X1_DIV_X0 = 5
FORM_X0SQ_DIV_X1 = 6
FORM_X1SQ_DIV_X0 = 7
FORM_X0_MUL_X1 = 8

ALL_SINGLE_FORM_IDS = np.array([FORM_X0, FORM_X1, FORM_X0_SQ, FORM_X1_SQ], dtype=np.int64)
ALL_PAIR_FORM_IDS = np.array(
    [FORM_X0_DIV_X1, FORM_X1_DIV_X0, FORM_X0SQ_DIV_X1, FORM_X1SQ_DIV_X0, FORM_X0_MUL_X1], dtype=np.int64,
)
ALL_FORM_IDS = np.concatenate([ALL_SINGLE_FORM_IDS, ALL_PAIR_FORM_IDS])

# Matches _fe_usability_signal.py's constants exactly -- see that module for the numerical justification.
_EPS_DENOM_FLOOR = 1e-12
_CV2_DEGENERATE_FLOOR = 1e-16


@numba.njit(cache=True, inline="always")
def _eval_form(form_id, x0, x1, eps):
    """Evaluate ONE of the 9 candidate forms at a single (x0, x1) row pair. Mirrors
    usability_form_corrs's per-row form construction (the eps-floored ratio denominator -> NaN on a
    near-zero divisor, dropped by the caller's isfinite mask -- no spurious inf)."""
    if form_id == FORM_X0:
        return x0
    if form_id == FORM_X1:
        return x1
    if form_id == FORM_X0_SQ:
        return x0 * x0
    if form_id == FORM_X1_SQ:
        return x1 * x1
    if form_id == FORM_X0_DIV_X1:
        return x0 / x1 if abs(x1) >= eps else np.nan
    if form_id == FORM_X1_DIV_X0:
        return x1 / x0 if abs(x0) >= eps else np.nan
    if form_id == FORM_X0SQ_DIV_X1:
        return (x0 * x0) / x1 if abs(x1) >= eps else np.nan
    if form_id == FORM_X1SQ_DIV_X0:
        return (x1 * x1) / x0 if abs(x0) >= eps else np.nan
    # FORM_X0_MUL_X1
    return x0 * x1


@numba.njit(cache=True, inline="always", fastmath={"reassoc", "contract", "arcp", "afn", "nsz"})
def _abs_pearson_form_reduction(y, operand_matrix, a_idx, b_idx, form_id, eps, cv2):
    """The exact ``_abs_pearson_njit`` two-pass reduction, evaluated on-the-fly for ONE (pair, form) --
    reproduces that kernel's numerical contract exactly: f64 accumulation, branchless isfinite masking,
    the coefficient-of-variation degenerate-column floor, AND the same fastmath set (``nnan``/``ninf``
    deliberately excluded, same as ``_abs_pearson_njit``, so the ``math.isfinite`` NaN-drop survives --
    only the accumulation-reordering flags are enabled). Without this, the two kernels agreed only to
    ~1e-9 (a real but selection-safe difference from summation-order alone); with matching fastmath flags
    they agree to the same ~1e-13 ULP level ``_abs_pearson_njit``'s own reassoc-delta test documents.
    Shared by both the njit_parallel and (via a thin cuda.jit wrapper) the CUDA backend, so the two
    backends can never numerically diverge from EACH OTHER (both compile this same function)."""
    n_rows = y.shape[0]
    cnt = 0
    sa = 0.0
    sv = 0.0
    for i in range(n_rows):
        yi = np.float64(y[i])
        x0 = np.float64(operand_matrix[a_idx, i])
        x1 = np.float64(operand_matrix[b_idx, i])
        v = _eval_form(form_id, x0, x1, eps)
        finite = math.isfinite(yi) and math.isfinite(v)
        av = yi if finite else 0.0
        bv = v if finite else 0.0
        cnt += 1 if finite else 0
        sa += av
        sv += bv
    if cnt < 2:
        return 0.0
    inv = 1.0 / cnt
    ma = sa * inv
    mv = sv * inv
    saa = 0.0
    svv = 0.0
    sav = 0.0
    for i in range(n_rows):
        yi = np.float64(y[i])
        x0 = np.float64(operand_matrix[a_idx, i])
        x1 = np.float64(operand_matrix[b_idx, i])
        v = _eval_form(form_id, x0, x1, eps)
        finite = math.isfinite(yi) and math.isfinite(v)
        da = (yi - ma) if finite else 0.0
        dv = (v - mv) if finite else 0.0
        saa += da * da
        svv += dv * dv
        sav += da * dv
    if saa <= cnt * cv2 * ma * ma or svv <= cnt * cv2 * mv * mv:
        return 0.0
    den = (saa * svv) ** 0.5
    if den <= 0.0:
        return 0.0
    c = sav / den
    if not math.isfinite(c):
        return 0.0
    return -c if c < 0.0 else c


# ---------------------------------------------------------------------------
# CPU reference / fallback: njit(parallel=True), prange over (pair, form)
# ---------------------------------------------------------------------------


@numba.njit(cache=True, parallel=True)
def batch_pair_usability_corr_njit_parallel(y, operand_matrix, pair_a, pair_b, form_ids):
    """CPU reference backend. ``prange`` over the flattened (pair, form) index -- each iteration is one
    independent ``_abs_pearson_form_reduction`` call, matching the numerical baseline every other backend
    is checked against.

    Parameters
    ----------
    y : (n,) float64 -- shared target (already subsampled/cast by the caller).
    operand_matrix : (n_operands, n) float64 -- unique raw operand columns, row-major (one row per operand).
    pair_a, pair_b : (n_pairs,) int64 -- operand-matrix row indices for each pair's two operands.
    form_ids : (n_forms,) int64 -- which of the 9 FORM_* ids to evaluate for every pair.

    Returns
    -------
    (n_pairs, n_forms) float64 -- ``|corr|`` for each (pair, form).
    """
    n_pairs = pair_a.shape[0]
    n_forms = form_ids.shape[0]
    out = np.zeros((n_pairs, n_forms), dtype=np.float64)
    total = n_pairs * n_forms
    for tid in prange(total):
        p = tid // n_forms
        f = tid - p * n_forms
        out[p, f] = _abs_pearson_form_reduction(
            y, operand_matrix, pair_a[p], pair_b[p], form_ids[f], _EPS_DENOM_FLOOR, _CV2_DEGENERATE_FLOOR,
        )
    return out


# ---------------------------------------------------------------------------
# numba.cuda variant
# ---------------------------------------------------------------------------

_CUDA_KERNEL: Any = None


def _cuda_kernel_factory():
    """Build (once) the CUDA kernel: one THREAD per (pair, form). Lazy so importing this module on a
    CPU-only host never triggers a CUDA driver lookup."""
    if not _CUDA_AVAIL or _nb_cuda is None:
        return None

    @_nb_cuda.jit
    def _kernel(y, operand_matrix, pair_a, pair_b, form_ids, eps, cv2, out):
        n_forms = form_ids.shape[0]
        n_pairs = pair_a.shape[0]
        tid = _nb_cuda.blockIdx.x * _nb_cuda.blockDim.x + _nb_cuda.threadIdx.x
        if tid >= n_pairs * n_forms:
            return
        p = tid // n_forms
        f = tid - p * n_forms
        out[p, f] = _abs_pearson_form_reduction(y, operand_matrix, pair_a[p], pair_b[p], form_ids[f], eps, cv2)

    return _kernel


def batch_pair_usability_corr_cuda(y: np.ndarray, operand_matrix: np.ndarray, pair_a: np.ndarray, pair_b: np.ndarray, form_ids: np.ndarray) -> np.ndarray:
    """CUDA backend: one thread per (pair, form), device-resident inputs uploaded once. Raises if CUDA is
    unavailable -- callers should go through :func:`dispatch_batch_pair_usability_corr` for the
    availability-checked + VRAM-guarded + auto-fallback path."""
    global _CUDA_KERNEL
    if _CUDA_KERNEL is None:
        _CUDA_KERNEL = _cuda_kernel_factory()
    if _CUDA_KERNEL is None:
        raise RuntimeError("CUDA is not available for batch_pair_usability_corr_cuda")

    y_d = _nb_cuda.to_device(np.ascontiguousarray(y, dtype=np.float64))
    operand_d = _nb_cuda.to_device(np.ascontiguousarray(operand_matrix, dtype=np.float64))
    pair_a_d = _nb_cuda.to_device(np.ascontiguousarray(pair_a, dtype=np.int64))
    pair_b_d = _nb_cuda.to_device(np.ascontiguousarray(pair_b, dtype=np.int64))
    form_ids_d = _nb_cuda.to_device(np.ascontiguousarray(form_ids, dtype=np.int64))

    n_pairs = pair_a.shape[0]
    n_forms = form_ids.shape[0]
    out_d = _nb_cuda.device_array((n_pairs, n_forms), dtype=np.float64)

    total = n_pairs * n_forms
    threads_per_block = 128
    blocks = (total + threads_per_block - 1) // threads_per_block
    _CUDA_KERNEL[blocks, threads_per_block](y_d, operand_d, pair_a_d, pair_b_d, form_ids_d, _EPS_DENOM_FLOOR, _CV2_DEGENERATE_FLOOR, out_d)
    return np.asarray(out_d.copy_to_host())


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def _required_gpu_bytes(operand_matrix: np.ndarray, n_pairs: int, n_forms: int) -> int:
    """Rough upload + output footprint for the VRAM cushion check -- operands + y (input) + the (n_pairs,
    n_forms) output matrix. Deliberately generous (float64 sizing even if inputs are f32) since this is a
    pre-flight guard, not a tight allocator."""
    operand_bytes = operand_matrix.size * 8
    output_bytes = n_pairs * n_forms * 8
    return int(operand_bytes + output_bytes + (1 << 20))  # +1MiB slack for y / index arrays


def dispatch_batch_pair_usability_corr(
    y: np.ndarray,
    operand_matrix: np.ndarray,
    pair_a: np.ndarray,
    pair_b: np.ndarray,
    form_ids: np.ndarray | None = None,
    force_backend: str | None = None,
) -> tuple[np.ndarray, str]:
    """Dispatch batched usability |corr| computation to the fastest available backend.

    Parameters mirror :func:`batch_pair_usability_corr_njit_parallel`; ``form_ids`` defaults to
    :data:`ALL_FORM_IDS` (all 9 forms) when omitted. ``force_backend`` in ``{"cpu", "cuda"}`` bypasses the
    size heuristic (for benchmarking / testing).

    Returns ``(result, backend_name)`` -- mirrors :func:`batch_pair_mi_gpu.dispatch_batch_pair_mi`'s
    contract. On ANY CUDA failure (compile, OOM, driver error) falls back to the CPU backend and logs at
    WARNING -- the per-pair result is otherwise identical (selection-equivalent, see the module docstring),
    so falling back never changes correctness, only speed.
    """
    if form_ids is None:
        form_ids = ALL_FORM_IDS
    form_ids = np.asarray(form_ids, dtype=np.int64)
    n_pairs = int(pair_a.shape[0])
    n_forms = int(form_ids.shape[0])

    use_cuda = _CUDA_AVAIL
    if force_backend == "cpu":
        use_cuda = False
    elif force_backend == "cuda":
        use_cuda = True
    elif force_backend is not None:
        raise ValueError(f"force_backend={force_backend!r}; expected 'cpu' or 'cuda' or None")
    else:
        # Un-forced default: CPU. 2026-07-11 measured A/B on this dev host (GTX 1050 Ti) at the real
        # production shape (30_000-row subsample per reduction, matching _ABS_PEARSON_MAX_ROWS) found the
        # CUDA backend NEVER wins -- 0.05x-0.65x across n_pairs in {16..150_000} (~144 to 1.35M total
        # reductions), the ratio converging to ~0.53-0.57x at and beyond the real ~85k-pair production
        # scale, not improving with more batch volume the way a launch-overhead-bound kernel normally
        # would. Consistent with this weak card's already-documented 0.26-0.66x underperformance on OTHER
        # resident FE kernels (see ``_permutation_null_pair_resident.py``) -- this reduction is memory-
        # bandwidth-bound (two full sequential passes per thread, one thread per (pair, form), inherently
        # poor coalescing since each thread reads a DIFFERENT operand row) rather than launch-overhead-
        # bound, so batching more pairs cannot amortize the way it does for e.g. batch_pair_mi_gpu's
        # histogram kernel. NOT auto-engaging CUDA here is therefore the measured-correct default on this
        # host -- unlike the per-call n*p heuristic elsewhere in this package, there is no known (n_pairs,
        # n_forms) region where CUDA wins to threshold on. Kept available via force_backend="cuda" (fully
        # tested, bit-identical) for a stronger production GPU or a future kernel_tuning_cache sweep that
        # finds a real per-host crossover -- see bench_batch_pair_usability_corr_gpu.py for the numbers.
        use_cuda = False

    if use_cuda:
        try:
            from ._fe_gpu_vram import fe_gpu_has_vram_cushion
            if not fe_gpu_has_vram_cushion(_required_gpu_bytes(operand_matrix, n_pairs, n_forms)):
                logger.warning(
                    "batch_pair_usability_corr: insufficient VRAM cushion for n_pairs=%d n_forms=%d -- falling back to CPU.",
                    n_pairs, n_forms,
                )
                use_cuda = False
        except Exception as exc:
            logger.debug("batch_pair_usability_corr: VRAM cushion check unavailable (%s); proceeding.", exc)

    if use_cuda:
        try:
            result = batch_pair_usability_corr_cuda(y, operand_matrix, pair_a, pair_b, form_ids)
            return result, "cuda"
        except Exception as exc:
            logger.warning(
                "batch_pair_usability_corr: CUDA backend failed (%s: %s) -- falling back to CPU.",
                type(exc).__name__, exc,
            )

    result = batch_pair_usability_corr_njit_parallel(
        np.ascontiguousarray(y, dtype=np.float64),
        np.ascontiguousarray(operand_matrix, dtype=np.float64),
        np.ascontiguousarray(pair_a, dtype=np.int64),
        np.ascontiguousarray(pair_b, dtype=np.int64),
        form_ids,
    )
    return result, "cpu"


# ---------------------------------------------------------------------------
# Batched rank-aware tail-concentration verdict (one bool per pair)
# ---------------------------------------------------------------------------


def _eval_pair_form_numpy(form_id: int, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
    """Vectorised (numpy, not numba) evaluation of ONE of the 5 ``ALL_PAIR_FORM_IDS`` forms across a whole
    array -- mirrors ``_eval_form``'s per-row logic and ``_fe_usability_signal.usability_form_corrs``'s own
    pair-form construction (the eps-floored ratio denominator -> NaN on a near-zero divisor) EXACTLY, so a
    caller reconstructing the winning form's VALUES (needed for the rank-correlation leg -- a |corr| alone
    is not enough) gets bit-identical values to what the serial reference would have used. Only called on
    the small subset of pairs that clear :func:`batch_pair_tail_concentration_rankaware`'s cheap min_corr/
    pairness gate -- never on the full candidate pool (the whole point of the two-stage split)."""
    _eps = _EPS_DENOM_FLOOR
    if form_id == FORM_X0_DIV_X1:
        x1f = np.where(np.abs(x1) < _eps, np.nan, x1)
        return np.asarray(x0 / x1f)
    if form_id == FORM_X1_DIV_X0:
        x0f = np.where(np.abs(x0) < _eps, np.nan, x0)
        return np.asarray(x1 / x0f)
    if form_id == FORM_X0SQ_DIV_X1:
        x1f = np.where(np.abs(x1) < _eps, np.nan, x1)
        return np.asarray((x0 * x0) / x1f)
    if form_id == FORM_X1SQ_DIV_X0:
        x0f = np.where(np.abs(x0) < _eps, np.nan, x0)
        return np.asarray((x1 * x1) / x0f)
    # FORM_X0_MUL_X1
    return np.asarray(x0 * x1)


def batch_pair_tail_concentration_rankaware(
    y: np.ndarray,
    operand_matrix: np.ndarray,
    pair_a: np.ndarray,
    pair_b: np.ndarray,
    *,
    min_corr: float,
    pairness_margin: float,
    max_rank_frac: float = 0.7,
    single_corr: np.ndarray | None = None,
) -> np.ndarray:
    """Batched equivalent of calling ``_fe_usability_signal.pair_is_tail_concentrated_rankaware`` once per
    pair in ``(pair_a, pair_b)`` -- returns a bool array, one verdict per pair. ``y``/``operand_matrix`` must
    already be subsampled + cast to the SAME dtype the serial reference uses internally (``_crit_np_dtype()``)
    -- this function does not re-subsample; the caller applies ``_corr_stride``/``_crit_np_dtype()`` ONCE for
    the whole pool (mirrors the existing dominant-pair prescan's pattern) instead of once per pair.

    ``single_corr``: optional array of each OPERAND-MATRIX ROW's own single-form |corr| (from
    ``_fe_usability_signal._single_operand_usability_corr``, aligned to ``operand_matrix`` rows via the
    caller's own row-index mapping) -- when given, pair i's ``_cs`` is ``max(single_corr[pair_a[i]],
    single_corr[pair_b[i]])`` (bit-identical to the serial per-pair ``precomputed_single_corr`` path); when
    omitted, computed via this same batched kernel over the 4 single forms.

    TWO-STAGE, avoiding materializing all 5 pair-forms for every pair (the whole point of batching -- most
    candidate pairs reaching this gate are noise and fail the cheap gate immediately): stage 1 batches ONLY
    the |corr| reduction (streamed on-core, no full form array ever stored) for every pair to get the best
    pair-form's value AND which form won; stage 2 rebuilds the actual winning-form VALUES (a plain numpy
    elementwise op, via :func:`_eval_pair_form_numpy`) only for the SUBSET that clears stage 1's min_corr/
    pairness gate, to run the rank-correlation leg -- mirrors the serial reference's own control flow
    (``_cp >= min_corr and _cp >= margin*_cs`` gates BEFORE the rank transform is ever computed)."""
    from ._fe_usability_signal import _rank_transform, abs_pearson

    n_pairs = int(pair_a.shape[0])
    verdict = np.zeros(n_pairs, dtype=bool)
    if n_pairs == 0:
        return verdict

    pair_corrs, _ = dispatch_batch_pair_usability_corr(y, operand_matrix, pair_a, pair_b, form_ids=ALL_PAIR_FORM_IDS)
    best_form_i = np.argmax(pair_corrs, axis=1)
    cp = pair_corrs[np.arange(n_pairs), best_form_i]

    if single_corr is not None:
        cs = np.maximum(single_corr[pair_a], single_corr[pair_b])
    else:
        single_corrs, _ = dispatch_batch_pair_usability_corr(y, operand_matrix, pair_a, pair_b, form_ids=ALL_SINGLE_FORM_IDS)
        cs = single_corrs.max(axis=1)

    _gate = (cp >= float(min_corr)) & (cp >= float(pairness_margin) * cs)
    _cand_idx = np.flatnonzero(_gate)
    if _cand_idx.size == 0:
        return verdict

    y_arr = np.asarray(y)
    for _i in _cand_idx:
        _form_id = int(ALL_PAIR_FORM_IDS[best_form_i[_i]])
        _x0 = operand_matrix[pair_a[_i]]
        _x1 = operand_matrix[pair_b[_i]]
        _form_vals = np.asarray(_eval_pair_form_numpy(_form_id, _x0, _x1))
        _m = np.isfinite(_form_vals) & np.isfinite(y_arr)
        if int(_m.sum()) < 3:
            continue
        _rank_corr = abs_pearson(_rank_transform(y_arr[_m]), _rank_transform(_form_vals[_m]))
        verdict[_i] = bool(_rank_corr <= float(max_rank_frac) * cp[_i])
    return verdict
