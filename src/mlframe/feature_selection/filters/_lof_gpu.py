"""Fused device LOF scoring of new rows against a frozen reference set (version 2 of ``_lof_fe._lof_transform``).

Win axis: the host version builds a ``(rows, n_ref)`` distance matrix per row block (matmul, clip, square root, ``argpartition`` over every distance), about 4.7 s for 100000 rows against 2000 reference points.
Here one thread scores one row: it walks the reference points, forms the squared distance from the coordinates directly, keeps the ``k`` nearest in an insertion-sorted list and finishes the reach-distance and
density ratio in the same kernel, so no distance matrix exists. It wins for any row count in the tens of thousands or more and for up to 16 coordinates and 64 neighbours; outside that the host path runs.

Distances and sums are ``double`` (``{ACC}``), the stored scores are ``{REAL}``. The k-nearest *set* equals the host's except at exact distance ties on the k-th neighbour.
"""

from __future__ import annotations

import logging
import os
from typing import cast

import numpy as np

from ._ktc_fallback import device_from_rows

from pyutilz.performance.gpu_hw_launch import block_size_for

from ._raw_module_cache import RawModuleCache

logger = logging.getLogger(__name__)

MAX_DIMS = 16
MAX_NEIGHBOURS = 64
#: Fallback crossover (rows) when the kernel-tuning cache has no entry; ``MLFRAME_LOF_DEVICE=0|1`` forces a path.
LOF_DEVICE_FALLBACK_ROWS = 20_000

_SRC = r"""
#define KMAX 64
#define PMAX 16
extern "C" __global__ void lof_scores(const {REAL}* __restrict__ X, const {REAL}* __restrict__ ref, const {REAL}* __restrict__ kdist_ref, const {REAL}* __restrict__ lrd_ref,
                                      const long long n, const int m, const int p, const int k, {REAL}* __restrict__ out) {
    const long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= n) return;
    double xi[PMAX];
    for (int t = 0; t < p; ++t) xi[t] = (double)X[i * p + t];
    double best_d[KMAX];
    int best_j[KMAX];
    int cnt = 0;
    for (int j = 0; j < m; ++j) {
        double d2 = 0.0;
        for (int t = 0; t < p; ++t) { const double dd = xi[t] - (double)ref[(long long)j * p + t]; d2 += dd * dd; }
        if (cnt == k && d2 >= best_d[k - 1]) continue;
        int pos = cnt < k ? cnt : k - 1;           // slot to fill / overwrite, then bubble towards the front
        while (pos > 0 && best_d[pos - 1] > d2) { best_d[pos] = best_d[pos - 1]; best_j[pos] = best_j[pos - 1]; --pos; }
        best_d[pos] = d2; best_j[pos] = j;
        if (cnt < k) ++cnt;
    }
    double sum_reach = 0.0, sum_lrd = 0.0;
    for (int t = 0; t < k; ++t) {
        const double d = sqrt(best_d[t]);
        const double kd = (double)kdist_ref[best_j[t]];
        sum_reach += kd > d ? kd : d;
        sum_lrd += (double)lrd_ref[best_j[t]];
    }
    const double mean_reach = sum_reach / k;
    const double lrd_new = 1.0 / (mean_reach > 1e-12 ? mean_reach : 1e-12);
    out[i] = ({REAL})((sum_lrd / k) / (lrd_new > 1e-12 ? lrd_new : 1e-12));
}
"""

_MODULES_BY_REAL = {"double": RawModuleCache(_SRC.replace("{REAL}", "double")), "float": RawModuleCache(_SRC.replace("{REAL}", "float"))}


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
    kernel = _MODULES_BY_REAL["float" if dt == np.float32 else "double"].get(cp).get_function("lof_scores")
    out = np.empty(n, dtype=dt)
    block = max(1, min(n, int(rows_per_launch)))
    for s in range(0, n, block):
        part = cp.asarray(np.ascontiguousarray(X_new[s : s + block], dtype=dt))
        res = cp.empty(part.shape[0], dtype=dt)
        threads = block_size_for(kernel, work_items=part.shape[0])
        kernel(((part.shape[0] + threads - 1) // threads,), (threads,), (part, ref, kd, lrd, np.int64(part.shape[0]), np.int32(ref.shape[0]), np.int32(p), np.int32(int(k_eff)), res))
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
_SALT = 2  # 2: the equivalence tolerance is 1e-9 / 1e-12 (the device scorer agrees with the host to about 1e-15), so the regions are measured again


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
            equiv_rtol=1e-9,
            equiv_atol=1e-12,
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

__all__ = ["lof_transform_gpu", "use_device_lof"]
