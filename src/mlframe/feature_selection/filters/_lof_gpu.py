"""Fused device LOF scoring of new rows against a frozen reference set (version 2 of ``_lof_fe._lof_transform``).

Win axis: the host version builds a ``(rows, n_ref)`` distance matrix per row block (matmul, clip, square root, ``argpartition`` over every distance), about 4.7 s for 100000 rows against 2000 reference points.
Here one thread scores one row: it walks the reference points, forms the squared distance from the coordinates directly, keeps the ``k`` nearest in an insertion-sorted list and finishes the reach-distance and
density ratio in the same kernel, so no distance matrix exists. It wins for any row count in the tens of thousands or more and for up to 16 coordinates and 64 neighbours; outside that the host path runs.

Distances and sums are ``double`` (``{ACC}``), the stored scores are ``{REAL}``. The k-nearest *set* equals the host's except at exact distance ties on the k-th neighbour.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, cast

import numpy as np

from ._ktc_fallback import device_from_rows

from pyutilz.performance.gpu_hw_launch import block_size_for

logger = logging.getLogger(__name__)

MAX_DIMS = 16
MAX_NEIGHBOURS = 64
#: Fallback crossover (rows) when the kernel-tuning cache has no entry; ``MLFRAME_LOF_DEVICE=0|1`` forces a path.
LOF_DEVICE_FALLBACK_ROWS = 20_000

_SRC = r"""
#define K {K}
#define P {P}
#define INF_D __longlong_as_double(0x7FF0000000000000LL)
#if K > 32
#define TOPK_DECL extern __shared__ unsigned char topk_smem[]; double* bd_base = (double*)topk_smem; int* bj_base = (int*)(topk_smem + K * blockDim.x * sizeof(double));
#define BD(s) bd_base[(s) * blockDim.x + threadIdx.x]
#define BJ(s) bj_base[(s) * blockDim.x + threadIdx.x]
#else
#define TOPK_DECL double bd_regs[K]; int bj_regs[K];
#define BD(s) bd_regs[s]
#define BJ(s) bj_regs[s]
#endif
// coalescing-ok: each thread owns one point and reads its P coordinates X[i * P + t] once, P being a compile-time constant (3 by default); the cost is the neighbour scan, which reads all points in a block coalesced
extern "C" __global__ void lof_scores(const {REAL}* __restrict__ X, const {REAL}* __restrict__ ref, const {REAL}* __restrict__ kdist_ref, const {REAL}* __restrict__ lrd_ref,
                                      const long long n, const int m, const int cnt, {REAL}* __restrict__ out) {
    const long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= n) return;
    TOPK_DECL
    double xi[P];
#pragma unroll
    for (int t = 0; t < P; ++t) xi[t] = (double)X[i * P + t];
#pragma unroll
    for (int s = 0; s < K; ++s) { BD(s) = INF_D; BJ(s) = -1; }
    for (int j = 0; j < m; ++j) {
        double d2 = 0.0;
#pragma unroll
        for (int t = 0; t < P; ++t) { const double dd = xi[t] - (double)ref[(long long)j * P + t]; d2 += dd * dd; }
        if (d2 >= BD(K - 1)) continue;
        // one pass carries the candidate down the sorted list: every slot index is a compile-time constant
        double cd = d2;
        int cj = j;
#pragma unroll
        for (int s = 0; s < K; ++s) {
            if (cd < BD(s)) {
                const double td = BD(s);
                const int tj = BJ(s);
                BD(s) = cd;
                BJ(s) = cj;
                cd = td;
                cj = tj;
            }
        }
    }
    double sum_reach = 0.0, sum_lrd = 0.0;
#pragma unroll
    for (int t = 0; t < K; ++t) {
        if (BJ(t) < 0) continue;
        const double d = sqrt(BD(t));
        const double kd = (double)kdist_ref[BJ(t)];
        sum_reach += kd > d ? kd : d;
        sum_lrd += (double)lrd_ref[BJ(t)];
    }
    const double mean_reach = sum_reach / cnt;
    const double lrd_new = 1.0 / (mean_reach > 1e-12 ? mean_reach : 1e-12);
    out[i] = ({REAL})((sum_lrd / cnt) / (lrd_new > 1e-12 ? lrd_new : 1e-12));
}
"""

DEFAULT_K = 20
DEFAULT_P = 3
_REG_TOPK_MAX = 32
_SHARED_TOPK_THREADS_CAP = {k: max(32, (48 * 1024 // (12 * k)) // 32 * 32) for k in range(_REG_TOPK_MAX + 1, MAX_NEIGHBOURS + 1)}
_MODULES: dict = {}
_LOCK = threading.Lock()


def kernel(cp: Any, real: Any, k: int = DEFAULT_K, p: int = DEFAULT_P) -> Any:
    """``lof_scores`` for value type ``real`` with the neighbour count ``k`` and dimension ``p`` compiled in (one module per combination, cached)."""
    key = ("float" if np.dtype(real) == np.float32 else "double", int(k), int(p))
    with _LOCK:
        mod = _MODULES.get(key)
        if mod is None:
            src = _SRC.replace("{REAL}", key[0]).replace("{K}", str(key[1])).replace("{P}", str(key[2]))
            mod = _MODULES[key] = cp.RawModule(code=src, options=("-std=c++14",))
        return mod.get_function("lof_scores")


def lof_transform_gpu(X_new: np.ndarray, X_ref: np.ndarray, lrd_ref: np.ndarray, k_distance_ref: np.ndarray, k_eff: int, rows_per_launch: int = 4_000_000) -> "np.ndarray | None":
    """LOF scores of the rows of ``X_new`` against the frozen reference, or ``None`` when the shape is outside the kernel (the host path then runs)."""
    n, p = int(X_new.shape[0]), int(X_new.shape[1])
    if p > MAX_DIMS or not 1 <= int(k_eff) <= MAX_NEIGHBOURS or int(X_ref.shape[0]) < int(k_eff):
        return None
    try:
        import cupy as cp
    except Exception as exc:
        logger.debug("device LOF unavailable (%s)", exc)
        return None
    # Intentional float64: output keeps the input width (float32 or float64)
    dt = np.float32 if X_new.dtype.itemsize == 4 else np.float64
    ref = cp.asarray(np.ascontiguousarray(X_ref, dtype=dt))
    kd = cp.asarray(np.ascontiguousarray(k_distance_ref, dtype=dt))
    lrd = cp.asarray(np.ascontiguousarray(lrd_ref, dtype=dt))
    krn = kernel(cp, dt, int(k_eff), p)
    out = np.empty(n, dtype=dt)
    block = max(1, min(n, int(rows_per_launch)))
    for s in range(0, n, block):
        part = cp.asarray(np.ascontiguousarray(X_new[s : s + block], dtype=dt))
        res = cp.empty(part.shape[0], dtype=dt)
        threads = block_size_for(krn, work_items=part.shape[0])
        shared = 0
        if int(k_eff) > _REG_TOPK_MAX:
            threads = min(threads, _SHARED_TOPK_THREADS_CAP[int(k_eff)])
            shared = int(k_eff) * threads * 12
        krn(((part.shape[0] + threads - 1) // threads,), (threads,), (part, ref, kd, lrd, np.int64(part.shape[0]), np.int32(ref.shape[0]), np.int32(int(k_eff)), res), shared_mem=shared)
        out[s : s + block] = cp.asnumpy(res)
    return cast(np.ndarray, out)


def use_device_lof(n_rows: int) -> bool:
    """Whether ``n_rows`` rows are scored on the device: env override, else the per-host kernel-tuning cache, else the measured fallback."""
    forced = os.environ.get("MLFRAME_LOF_DEVICE", "").strip()
    if forced in ("0", "1"):
        return forced == "1"
    if _SPEC is None:
        return n_rows >= LOF_DEVICE_FALLBACK_ROWS
    try:
        return str(_SPEC.choose(n_samples=int(n_rows))) == "device"
    except Exception as exc:
        logger.debug("fe_lof_device: KTC choose() failed, using the fallback: %s", exc)
        return n_rows >= LOF_DEVICE_FALLBACK_ROWS


_SWEEP_ROWS = [2_000, 20_000, 100_000]
_SALT = 4  # 4: the kernel is specialised per (k, p) with a shared-memory top-k above 32, so its timings are measured again


def _make_inputs(dims: dict) -> tuple:
    """A 3-coordinate frame of ``dims["n_samples"]`` rows, a 2000-point reference and its density internals."""
    from ._lof_fe import _lof_fit_reference

    rng = np.random.default_rng(0)
    X = rng.normal(size=(int(dims["n_samples"]), 3)).astype(np.float32)
    ref = X[rng.choice(X.shape[0], size=min(2000, X.shape[0]), replace=False)]
    lrd, kd, k_eff = _lof_fit_reference(ref, 20)
    return (X, ref, lrd, kd, k_eff)


def _run_sweep() -> list:
    """Time the host and the device scorer per row count; the fastest variant whose scores agree with the host's wins each region."""
    from pyutilz.dev.benchmarking import sweep_backend_grid

    from ._lof_fe import _lof_transform_host

    def device(X, ref, lrd, kd, k_eff):
        """Device variant."""
        got = lof_transform_gpu(X, ref, lrd, kd, k_eff)
        if got is None:
            raise RuntimeError("device LOF unavailable")
        return got

    return cast(
        list,
        sweep_backend_grid(
            {"host": _lof_transform_host, "device": device},
            {"n_samples": _SWEEP_ROWS},
            _make_inputs,
            reference="host",
            repeats=3,
            equiv_rtol=1e-5,  # float32 scores: the device scorer differs from the host by about 3e-7 relative (a 1e-9 tolerance rejected the device everywhere and the tuner stored 'host' for every size)
            equiv_atol=1e-6,
        ),
    )


def _lof_fallback_choice(n_samples: int) -> str:
    """Pre-sweep fallback: the measured crossover."""
    return device_from_rows(n_samples, LOF_DEVICE_FALLBACK_ROWS)


try:
    from pyutilz.performance.kernel_tuning.registry import TunerSpec, kernel_tuner

    _SPEC: "TunerSpec | None" = kernel_tuner(
        kernel_name="fe_lof_device",
        variant_fns=(),  # GPU-only cupy path; covered by salt
        tuner=_run_sweep,
        axes={"n_samples": list(_SWEEP_ROWS)},
        fallback=_lof_fallback_choice,
        gpu_capable=True,
        salt=_SALT,
        cli_label="fe_lof_device",
    )
except Exception as e:
    logger.warning("kernel_tuner spec build for fe_lof_device failed: %s", e)
    _SPEC = None

__all__ = ["lof_transform_gpu", "use_device_lof", "kernel"]
