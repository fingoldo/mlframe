"""Device form of the ordered-pair ratio / log-ratio redundancy screen (host twin: ``_ratio_delta_screen._screen_njit``).

One block per ordered pair ``(a, b)`` of the p resident source columns, ``p (p - 1)`` blocks. The block builds the ratio (or log-ratio) value of every row in registers, twice (no ``(p, p, n)``
candidate matrix is ever stored), and reduces with double accumulators in shared memory: pass 1 the candidate's all-row sum and square (its spread), and per source the joint-finite count and sums;
pass 2 the centred variances and covariance of the masked rows with the same near-constant rule as ``_abs_corr_finite_njit``. The block writes the accept flag of the pair. The sources are read in
the column dtype ``{REAL}`` (float32 by default) in place from the ``(p, n)`` matrix; rows are strided over the block's threads (a row block per thread pass, nothing sized by n is held in shared memory; the p x n source matrix is the only operand, resident).

``ratio_pair_screen_on_device`` is the per-host choice between the two forms (kernel_tuner ``ratio_pair_screen_backend``, a sweep over the row count at identical accept matrices).
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

import numpy as np

from ._kernel_types import render
from ._ktc_fallback import host_device_sweep, rows_fallback

logger = logging.getLogger(__name__)

THREADS = 256  # the TH define of the kernel source: keep the two equal (test_cuda_kernel_register_pressure compiles both)

_SRC = r"""
#define NQ 9
#define TH 256  // equals THREADS below
extern "C" __global__ void ratio_pair_screen(const {REAL}* __restrict__ mat, const long long n, const int p, const int kind, const {ACC} eps, const {ACC} thr, const {ACC} rel_tol,
                                             unsigned char* __restrict__ accept) {
    __shared__ {ACC} sh[TH];
    const int a = blockIdx.x / p, b = blockIdx.x % p;
    if (a == b) { if (threadIdx.x == 0) accept[blockIdx.x] = 0; return; }
    const {REAL}* __restrict__ ca = mat + (long long)a * n;
    const {REAL}* __restrict__ cb = mat + (long long)b * n;
    const {REAL} reps = ({REAL})eps;  // the host builds the candidate in the column dtype
    {ACC} q[NQ];
    for (int k = 0; k < NQ; ++k) q[k] = 0.0;
    for (long long i = threadIdx.x; i < n; i += blockDim.x) {
        const {REAL} av = ca[i], bv = cb[i];
        {REAL} v;
        if (kind == 0) { const {REAL} den = (bv >= 0 ? ({REAL})1 : ({REAL})-1) * fmax(fabs(bv), reps); v = av / den; }
        else { v = log1p(fabs(av) + reps) - log1p(fabs(bv) + reps); }
        if (!isfinite(v)) v = 0;
        q[0] += 1.0; q[1] += ({ACC})v; q[2] += ({ACC})v * ({ACC})v;
        if (isfinite(av)) { q[3] += 1.0; q[4] += ({ACC})av; q[5] += ({ACC})v; }
        if (isfinite(bv)) { q[6] += 1.0; q[7] += ({ACC})bv; q[8] += ({ACC})v; }
    }
    {ACC} tot[NQ];
    for (int k = 0; k < NQ; ++k) {
        sh[threadIdx.x] = q[k];
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) { if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s]; __syncthreads(); }
        tot[k] = sh[0];
        __syncthreads();
    }
    const {ACC} nn = tot[0];
    const {ACC} mc = tot[1] / nn;
    const {ACC} varc_all = tot[2] / nn - mc * mc;
    {REAL} r[2][5];  // per-thread partial sums of the CENTRED values in the column dtype (about n / blockDim terms each), promoted to {ACC} in the block reduction
    for (int s = 0; s < 2; ++s) for (int k = 0; k < 5; ++k) r[s][k] = 0;
    const {ACC} ms[2] = {tot[3] > 0.0 ? tot[4] / tot[3] : 0.0, tot[6] > 0.0 ? tot[7] / tot[6] : 0.0};
    const {ACC} mcs[2] = {tot[3] > 0.0 ? tot[5] / tot[3] : 0.0, tot[6] > 0.0 ? tot[8] / tot[6] : 0.0};
    for (long long i = threadIdx.x; i < n; i += blockDim.x) {
        const {REAL} av = ca[i], bv = cb[i];
        {REAL} v;
        if (kind == 0) { const {REAL} den = (bv >= 0 ? ({REAL})1 : ({REAL})-1) * fmax(fabs(bv), reps); v = av / den; }
        else { v = log1p(fabs(av) + reps) - log1p(fabs(bv) + reps); }
        if (!isfinite(v)) v = 0;
        for (int s = 0; s < 2; ++s) {
            const {REAL} sv = s == 0 ? av : bv;
            if (isfinite(sv)) {
                const {REAL} da = ({REAL})(({ACC})sv - ms[s]), dc = ({REAL})(({ACC})v - mcs[s]);
                r[s][0] += da * da; r[s][1] += dc * dc; r[s][2] += da * dc;
                r[s][3] = fmax(r[s][3], fabs(sv)); r[s][4] = fmax(r[s][4], fabs(v));
            }
        }
    }
    {ACC} fin[2][5];
    for (int s = 0; s < 2; ++s) for (int k = 0; k < 5; ++k) {
        sh[threadIdx.x] = ({ACC})r[s][k];
        __syncthreads();
        for (int t = blockDim.x / 2; t > 0; t >>= 1) {
            if (threadIdx.x < t) sh[threadIdx.x] = k < 3 ? sh[threadIdx.x] + sh[threadIdx.x + t] : fmax(sh[threadIdx.x], sh[threadIdx.x + t]);
            __syncthreads();
        }
        fin[s][k] = sh[0];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        bool ok = varc_all > 1e-24;
        for (int s = 0; s < 2 && ok; ++s) {
            const {ACC} cnt = s == 0 ? tot[3] : tot[6];
            if (cnt < 2.0) continue;
            const {ACC} va = fin[s][0], vc = fin[s][1], cov = fin[s][2], amax = fin[s][3], cmax = fin[s][4];
            if (va <= cnt * (rel_tol * amax) * (rel_tol * amax) || vc <= cnt * (rel_tol * cmax) * (rel_tol * cmax)) continue;
            const {ACC} denom = sqrt(va * vc);
            if (denom <= 0.0) continue;
            if (fabs(cov / denom) > thr) ok = false;
        }
        accept[blockIdx.x] = ok ? 1 : 0;
    }
}
"""

_KERNELS: dict = {}
_LOCK = threading.Lock()


def kernel(cp: Any, real: Any) -> Any:
    """``ratio_pair_screen`` for a ``(p, n)`` source matrix of dtype ``real``; compile point of :func:`ratio_pair_screen_device` and of the register-pressure registry."""
    name = "float" if np.dtype(real) == np.float32 else "double"
    with _LOCK:
        ker = _KERNELS.get(name)
        if ker is None:
            ker = _KERNELS[name] = cp.RawKernel(render(_SRC, np.float32 if name == "float" else np.float64), "ratio_pair_screen")  # Intentional float64: the double kernel reads double columns
    return ker


def ratio_pair_screen_device(mat_dev: Any, kind: int, eps: float, threshold: float) -> np.ndarray:
    """Accept matrix (p, p) of the ordered pairs of the rows of the resident ``mat_dev`` (p, n). A source without spread has zero centred variance, which the kernel's near-constant rule skips as the host does."""
    import cupy as cp

    from ._feature_engineering_pairs._pairs_core_steps import _DEGENERATE_REL_TOL

    p, n = (int(mat_dev.shape[0]), int(mat_dev.shape[1]))
    out = cp.zeros(p * p, dtype=cp.uint8)
    kernel(cp, mat_dev.dtype)((p * p,), (THREADS,), (mat_dev, np.int64(n), np.int32(p), np.int32(kind), np.float64(eps), np.float64(threshold), np.float64(_DEGENERATE_REL_TOL), out))  # Intentional float64: scalar arguments  # launch-ok: the shared reduction array is statically sized by the TH define (256), so the block is fixed by the source
    return np.asarray(cp.asnumpy(out)).reshape(p, p).astype(bool)


DEVICE_FALLBACK_ROWS = 5_000  # before any sweep: the sweep put the device ahead from 5000 rows (24 columns)
_SWEEP_ROWS = [5_000, 20_000, 100_000, 400_000]
_SWEEP_COLS = 24
_SALT = 1


def _sweep_inputs(dims: dict) -> tuple:
    """``(columns,)``: ``_SWEEP_COLS`` float32 columns of ``dims["n_samples"]`` rows, one with a few non-finite entries and one near-collinear pair."""
    rng = np.random.default_rng(0)
    n = int(dims["n_samples"])
    m = rng.normal(size=(_SWEEP_COLS, n)).astype(np.float32)
    m[1, ::997] = np.nan
    m[3] = m[2] * 2.0 + 0.001 * m[3]
    m[5] = 1.0 + 0.001 * m[5]  # a near-constant denominator: a / m[5] is redundant with a
    m[7] = 0.0  # a constant column
    return (m,)


def _sweep_host(columns: np.ndarray) -> np.ndarray:
    """The host screen on the sweep inputs, as float."""
    from ._ratio_delta_screen import RATIO, ratio_pair_screen_host

    return ratio_pair_screen_host(list(columns), RATIO, 1e-9, 0.99).astype(np.float64)  # Intentional float64: comparison vector


_SWEEP_RESIDENT: dict = {}


def _sweep_device(columns: np.ndarray) -> np.ndarray:
    """The device screen on the sweep inputs, the matrix uploaded once per input (resident sources)."""
    import cupy as cp

    from ._ratio_delta_screen import RATIO

    hit = _SWEEP_RESIDENT.get(id(columns))
    if hit is None or hit[0] is not columns:
        _SWEEP_RESIDENT.clear()
        hit = _SWEEP_RESIDENT[id(columns)] = (columns, cp.asarray(columns))
    return ratio_pair_screen_device(hit[1], RATIO, 1e-9, 0.99).astype(np.float64)  # Intentional float64: comparison vector


try:
    from pyutilz.performance.kernel_tuning.registry import kernel_tuner

    _SPEC: Any = kernel_tuner(
        kernel_name="ratio_pair_screen_backend",
        variant_fns=(),  # the two backends are timed by the sweep callable
        tuner=host_device_sweep(__name__, _sweep_host, _sweep_device, rtol=0.0, atol=0.0),
        axes={"n_samples": list(_SWEEP_ROWS)},
        fallback=rows_fallback(__name__),
        gpu_capable=True,
        salt=_SALT,
        cli_label="ratio_pair_screen_backend",
    )
except Exception as e:  # best-effort: without a tuner entry the row-count fallback decides
    logger.warning("kernel_tuner spec build for ratio_pair_screen_backend failed: %s", e)
    _SPEC = None


def ratio_pair_screen_on_device(n_rows: int) -> bool:
    """Whether the pair screen over ``n_rows`` rows runs on the device (STRICT forces it): ``MLFRAME_RATIO_SCREEN_DEVICE=0|1`` forces a side, else the per-host tuning cache, else the row mark."""
    forced = os.environ.get("MLFRAME_RATIO_SCREEN_DEVICE", "").strip()
    if forced in ("0", "1"):
        return forced == "1"
    if _SPEC is None:
        return int(n_rows) >= DEVICE_FALLBACK_ROWS
    try:
        return str(_SPEC.choose(n_samples=int(n_rows))) == "device"
    except Exception as exc:
        logger.debug("ratio_pair_screen_backend: KTC choose() failed, using the row-count fallback: %s", exc)
        return int(n_rows) >= DEVICE_FALLBACK_ROWS
