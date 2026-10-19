"""Bit-set permutation noise gate: the ``(P, K)`` plug-in MI of every column against the original and every permuted target, from device bit sets.

The histogram gate bins each column against each of the ``P`` target vectors with one shared-memory atomic per row: ``n`` increments per (column, permutation) block. The joint count of bin ``a``
and class ``c`` is also ``popcount(bin_a & class_c)`` over 64-row words, and ``family_null_bits_v2`` (``_family_maxt_cuda_bits``) evaluates exactly that for every (permutation, column) pair with the class bit
sets of a permutation in shared memory. This module builds the two bit-set families on the device from the resident integer codes and the resident ``(P, n)`` target matrix, so nothing but the ``(P, K)``
MI matrix leaves the device.

The MI is the plug-in value ``(sum xlogx(joint) + n log n - sum xlogx(px) - sum xlogx(py)) / n`` in nats, the same quantity the count-table reduction computes with a different order of additions: the two agree
to a few ulps (selection-equivalent), and rows ``p = 0`` and ``p > 0`` share one arithmetic, so equal joint tables give equal values. The backend (this one or the histogram pair) is a kernel-tuning-cache choice
over rows, columns, classes and bins; the fallback while the host has no entry is the measured crossover of ``nbins * (classes - 1)``.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

MAX_BINS = 32  # the bin bit sets of one (column, word) are held in registers: ``MAX_BINS`` words per thread
_C_TYPES = {
    np.dtype(np.int8): "signed char",
    np.dtype(np.uint8): "unsigned char",
    np.dtype(np.int16): "short",
    np.dtype(np.int32): "int",
    np.dtype(np.int64): "long long",
}
_BUDGET_CAP_BYTES = 768 * 1024 * 1024  # bit sets of one column group, at most

_PACK_SRC = r"""
extern "C" __global__ void pack_bins(const {DT}* __restrict__ disc, const long long n, const int k_total, const int k0, const int kg, const int words,
                                     unsigned long long* __restrict__ bits, long long* __restrict__ px) {
    const int kl = blockIdx.x * blockDim.x + threadIdx.x;
    const int w = blockIdx.y;
    if (kl >= kg) return;
    const int col = k0 + kl;
    unsigned long long acc[CAP];
    #pragma unroll
    for (int a = 0; a < CAP; ++a) acc[a] = 0ull;
    const long long r0 = (long long)w * 64;
    for (int r = 0; r < 64; ++r) {
        const long long row = r0 + r;
        if (row >= n) break;
        const int code = (int)disc[row * k_total + col];
        #pragma unroll
        for (int a = 0; a < CAP; ++a) acc[a] |= ((unsigned long long)(code == a)) << r;
    }
    #pragma unroll
    for (int a = 0; a < CAP; ++a) {
        bits[((long long)a * words + w) * kg + kl] = acc[a];
        atomicAdd((unsigned long long*)&px[(long long)kl * CAP + a], (unsigned long long)__popcll(acc[a]));
    }
}
extern "C" __global__ void pack_cls(const int* __restrict__ y_all, const long long n, const int words, const int n_y,
                                    unsigned long long* __restrict__ cls, long long* __restrict__ cnt) {
    const int w = blockIdx.x * blockDim.x + threadIdx.x;
    const int p = blockIdx.y;
    if (w >= words) return;
    unsigned long long acc[NYC];
    #pragma unroll
    for (int c = 0; c < NYC; ++c) acc[c] = 0ull;
    const long long r0 = (long long)w * 64;
    for (int r = 0; r < 64; ++r) {
        const long long row = r0 + r;
        if (row >= n) break;
        const int y = y_all[(long long)p * n + row];
        #pragma unroll
        for (int c = 0; c < NYC; ++c) acc[c] |= ((unsigned long long)(y == c)) << r;
    }
    #pragma unroll
    for (int c = 0; c < NYC; ++c) {
        cls[((long long)p * n_y + c) * words + w] = acc[c];
        atomicAdd((unsigned long long*)&cnt[(long long)p * n_y + c], (unsigned long long)__popcll(acc[c]));
    }
}
"""

_modules: dict = {}
_LOCK = threading.Lock()


def _pack_module(cp: Any, cap: int, n_y: int, c_type: str) -> Any:
    """The compiled packing kernels for ``cap`` bins, ``n_y`` classes and one code type, built once per process."""
    key = (cap, n_y, c_type)
    mod = _modules.get(key)
    if mod is None:
        with _LOCK:
            mod = _modules.get(key)
            if mod is None:
                mod = cp.RawModule(code=_PACK_SRC.replace("{DT}", c_type), options=(f"-DCAP={int(cap)}", f"-DNYC={int(n_y)}"))
                _modules[key] = mod
    return mod


def bits_gate_mi(d_disc: Any, nbins_arr: np.ndarray, d_y_all: Any, n_y: int) -> Optional[np.ndarray]:
    """``(P, K)`` plug-in MI (host float64) of every column of the resident ``(n, K)`` codes against every row of the resident ``(P, n)`` target matrix, or ``None`` when the bit-set kernel does not apply.

    It declines a code type it has no packing kernel for, more bins than ``MAX_BINS``, class bit sets that do not fit a block's shared memory, and any device failure (the caller then runs the histogram gate).
    """
    try:
        import cupy as cp

        from ._family_maxt_cuda_bits import _THREADS_V2, _get_kernel_v2, _v2_applies
        from ._family_maxt_null_v2_njit import xlogx_table

        codes = cp.asarray(d_disc)
        if codes.ndim != 2 or codes.dtype not in _C_TYPES:
            return None
        n, k_total = int(codes.shape[0]), int(codes.shape[1])
        nbins = np.asarray(nbins_arr, dtype=np.int64)
        cap = int(nbins.max()) if nbins.size else 0
        words = (n + 63) // 64
        if n == 0 or k_total == 0 or not 1 <= cap <= MAX_BINS or not _v2_applies(cp, n_y, words):
            return None
        y_all = cp.asarray(d_y_all)
        if y_all.dtype != cp.int32:
            y_all = y_all.astype(cp.int32)
        n_perm = int(y_all.shape[0])
        mod = _pack_module(cp, cap, n_y, _C_TYPES[codes.dtype])
        cls = cp.empty((n_perm, n_y, words), dtype=cp.uint64)
        cnt = cp.zeros((n_perm, n_y), dtype=cp.int64)
        mod.get_function("pack_cls")(((words + 127) // 128, n_perm), (128,), (y_all, np.int64(n), np.int32(words), np.int32(n_y), cls, cnt))
        # Intentional float64: the xlogx table of the MI formula (exact permutation ties); the kernel takes its element type from this buffer
        table = cp.asarray(xlogx_table(n))
        const_b = table[n] - table[cnt].sum(axis=1)
        n_x_all = cp.asarray(nbins)
        kern = _get_kernel_v2(cp, n_y - 1, "float" if table.dtype == cp.float32 else "double")
        free_bytes = int(cp.cuda.runtime.memGetInfo()[0])
        group = max(1, min(_BUDGET_CAP_BYTES, free_bytes // 4) // (cap * words * 8))
        out = cp.empty((n_perm, k_total), dtype=table.dtype)
        for lo in range(0, k_total, group):
            hi = min(lo + group, k_total)
            kg = hi - lo
            bits = cp.empty((cap, words, kg), dtype=cp.uint64)
            px = cp.zeros((kg, cap), dtype=cp.int64)
            mod.get_function("pack_bins")(((kg + 127) // 128, words), (128,), (codes, np.int64(n), np.int32(k_total), np.int32(lo), np.int32(kg), np.int32(words), bits, px))
            c_a = table[px].sum(axis=1)
            part = cp.empty((n_perm, kg), dtype=table.dtype)
            kern(
                (n_perm, (kg + _THREADS_V2 - 1) // _THREADS_V2),
                (_THREADS_V2,),
                (bits, n_x_all[lo:hi], cls, table, c_a, px, const_b, np.int64(n), np.int32(cap), np.int32(words), np.int32(n_y), np.int32(kg), np.int32(n_perm), part),
                shared_mem=(n_y - 1) * words * 8,
            )
            out[:, lo:hi] = part
            del bits, px, part
        host: np.ndarray = cp.asnumpy(out).astype(np.float64, copy=False)
        return host
    except Exception as exc:  # nosec B110 - any device or compile failure falls back to the histogram gate
        logger.warning("bit-set noise gate unavailable (%s: %s); using the histogram gate", type(exc).__name__, exc)
        return None


# --- backend choice (kernel tuning cache) -------------------------------------------------------------------------------------------------------------------

_SWEEP_N_SAMPLES = [4000, 20000]
_SWEEP_N_COLS = [400, 4000]
_SWEEP_N_CLASSES = [2, 10, 30]
_SWEEP_N_BINS = [6, 12, 24]
_SWEEP_PERMUTATIONS = 99
_FALLBACK_MAX_CELLS = 700  # nbins * (classes - 1) up to which the bit-set gate is the fallback choice (measured crossover on the GTX 1050 Ti: 10 bins x 10 classes is 8x faster)
_SALT = 1


def _fallback_choice(n_samples: int, n_cols: int, n_classes: int, nbins: int) -> str:
    """The backend while the host has no tuning-cache entry: the bit-set gate while ``nbins * (classes - 1)`` is within the measured crossover, else the histogram pair."""
    return "bits" if int(nbins) * (int(n_classes) - 1) <= _FALLBACK_MAX_CELLS else "hist"


def _sweep_inputs(dims: dict) -> tuple:
    """Synthetic ``(codes, nbins, classes_y, classes_y_safe, freqs_y)`` of the requested rows, columns, classes and bins, with a few columns that carry signal."""
    rng = np.random.default_rng(0)
    n, k, k_y, nb = int(dims["n_samples"]), int(dims["n_cols"]), int(dims["n_classes"]), int(dims["nbins"])
    y = rng.integers(0, k_y, n).astype(np.int64)
    codes = rng.integers(0, nb, (n, k)).astype(np.int8)
    for j in range(min(8, k)):
        keep = rng.random(n) < 0.12 * (j + 1)
        codes[:, j] = np.where(keep, codes[:, j], (y * nb // k_y)).astype(np.int8)
    freqs = np.bincount(y, minlength=k_y).astype(np.float64) / n  # Intentional float64: class frequencies of the MI formula
    return codes, np.full(k, nb, dtype=np.int64), y, y.astype(np.int32), freqs


def _run_sweep() -> list:
    """Sweep the histogram pair against the bit-set gate over rows x columns x classes x bins; fastest equivalent backend per cell."""
    from pyutilz.dev.benchmarking import sweep_backend_grid

    from . import batch_mi_noise_gate_gpu as gate

    seed = np.uint64(12345)

    def _call(use_bits: bool):
        """One backend as a sweep variant: the resident noise gate with the bit-set path forced on or off."""

        def run(codes, nbins, y, y_safe, freqs):
            """The gate decision of one backend; raises when the bit-set path declines so the sweep leaves the cell to the histogram pair."""
            import cupy as cp

            token = gate.force_noise_gate_backend("bits" if use_bits else "hist")
            try:
                out = gate.batch_mi_with_noise_gate_cuda_resident(codes, nbins, y, y_safe, freqs, _SWEEP_PERMUTATIONS, seed, 0.99, False, np.int32, None, cp.asarray(codes))
            finally:
                gate.force_noise_gate_backend(token)
            return np.asarray(out)

        return run

    return sweep_backend_grid(
        {"hist": _call(False), "bits": _call(True)},
        {"n_samples": _SWEEP_N_SAMPLES, "n_cols": _SWEEP_N_COLS, "n_classes": _SWEEP_N_CLASSES, "nbins": _SWEEP_N_BINS},
        _sweep_inputs,
        reference="hist",
        repeats=3,
        equiv_rtol=1e-9,
        equiv_atol=1e-12,
    )


def _register_spec() -> Any:
    """Register the backend choice with the kernel tuning registry (discovered by ``mlframe-tune-kernels``)."""
    from pyutilz.performance.kernel_tuning.registry import kernel_tuner

    return kernel_tuner(
        kernel_name="noise_gate_bits",
        variant_fns=(),
        tuner=_run_sweep,
        axes={"n_samples": list(_SWEEP_N_SAMPLES), "n_cols": list(_SWEEP_N_COLS), "n_classes": list(_SWEEP_N_CLASSES), "nbins": list(_SWEEP_N_BINS)},
        fallback=_fallback_choice,
        gpu_capable=True,
        salt=_SALT,
        cli_label="noise_gate_bits",
    )


_SPEC = _register_spec()


def noise_gate_backend_choice(n_samples: int, n_cols: int, n_classes: int, nbins: int) -> str:
    """``"bits"`` or ``"hist"`` for this shape: the tuning-cache choice, or the measured-crossover fallback when the cache has no entry or fails."""
    try:
        choice = str(_SPEC.choose(n_samples=int(n_samples), n_cols=int(n_cols), n_classes=int(n_classes), nbins=int(nbins)))
        if choice in ("bits", "hist"):
            return choice
    except Exception as exc:  # nosec B110 - a failed lookup takes the measured fallback
        logger.debug("noise_gate_bits tuning-cache lookup failed (%s: %s); using the fallback", type(exc).__name__, exc)
    return _fallback_choice(n_samples, n_cols, n_classes, nbins)
