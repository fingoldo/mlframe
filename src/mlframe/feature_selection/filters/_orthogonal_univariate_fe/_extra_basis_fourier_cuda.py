"""One device launch for a whole block of Fourier / chirp basis columns.

``_extra_basis_resident._fourier_col_gpu`` builds one column per call from about eight elementwise cupy operations, and the caller stacks the columns with one copy launch each; a pool of
several hundred engineered Fourier names is then a few thousand tiny launches. This kernel evaluates every name in one pass over a ``(names, rows)`` block: ``name j`` reads source column
``src[j]`` of the stacked resident operands, forms the axis ``z`` (the linear power axis ``(x**p - lo) / span`` or the chirp axis ``(sign(zc) * zc**2 - lo) / span`` with ``zc = (x - mean) / std``)
and writes ``sin|cos((2 pi freq) * z)``.

The arithmetic is that of the per-column function operation by operation, in the element type of the operand column, with the scalar parameters rounded to that type as cupy rounds a Python float
combined with an array of that type; the one product followed by a subtraction (the chirp axis) uses a rounded multiply so it is not contracted into a fused multiply-add, as two separate elementwise
kernels would not. (Compiling the whole kernel without fused multiply-add is not equivalent: it changes the device ``sin`` of double precision.) The columns are bit-identical to the per-column build.

The launch goes in column blocks of names (``_device_blocks``: only the source columns a block references are stacked, the block is halved on an out-of-memory error), the numba CPU twin is `_fourier_block_host`
(the reference the block is checked against, and the host side of the kernel-tuning entry `fourier_block_backend`, which decides from how many names the block replaces the per-column build).
"""

from __future__ import annotations

import logging
import math
import threading
from typing import Any, Optional

import numpy as np

from pyutilz.performance.gpu_hw_launch import block_size_for

logger = logging.getLogger(__name__)

_SRC = r"""
#define MUL_RN {MUL_RN}
extern "C" __global__ void fourier_block(const {REAL}* __restrict__ xs, const int* __restrict__ src, const int* __restrict__ cosine, const int* __restrict__ power,
                                         const int* __restrict__ quad, const {REAL}* __restrict__ lo, const {REAL}* __restrict__ span, const {REAL}* __restrict__ mean,
                                         const {REAL}* __restrict__ std, const {REAL}* __restrict__ w, const long long n, const int K, {REAL}* __restrict__ out) {
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)K * n) return;
    const int j = (int)(idx / n);
    const long long i = idx - (long long)j * n;
    const {REAL} x = xs[(long long)src[j] * n + i];
    {REAL} z;
    if (quad[j]) {
        const {REAL} zc = (x - mean[j]) / std[j];
        const {REAL} sgn = (zc > ({REAL})0) - (zc < ({REAL})0);
        const {REAL} u = MUL_RN(sgn, zc * zc);  // rounded product: not contracted with the subtraction below, as in two separate kernels
        z = (u - lo[j]) / span[j];
    } else {
        const int p = power[j];
        {REAL} xp = x;
        if (p != 1) xp = pow(x, ({REAL})p);
        z = (xp - lo[j]) / span[j];
    }
    const {REAL} ang = w[j] * z;
    out[idx] = cosine[j] ? cos(ang) : sin(ang);
}
"""

_kernels: dict = {}
_LOCK = threading.Lock()


def _kernel(cp: Any, real: str) -> Any:
    """The compiled kernel for one element type, built once per process."""
    with _LOCK:
        kern = _kernels.get(real)
        if kern is None:
            source = _SRC.replace("{MUL_RN}", "__fmul_rn" if real == "float" else "__dmul_rn").replace("{REAL}", real)
            kern = cp.RawModule(code=source).get_function("fourier_block")
            _kernels[real] = kern
        return kern


def _axis_params(m: dict) -> "tuple[float, float, float, float, int, int]":
    """``(lo, span, mean, std, power, quad)`` of one name's axis with the guards of ``_fourier_col_gpu`` (span and std floored at 1e-12)."""
    lo = float(m["lo"])
    span = float(m["span"])
    span = span if span > 1e-12 else 1e-12
    if m.get("arg") == "quadratic":
        std = float(m["std"])
        std = std if std > 1e-12 else 1e-12
        return lo, span, float(m["mean"]), std, 1, 1
    return lo, span, 0.0, 1.0, int(m.get("power", 1)), 0


def fourier_block_cuda(cp: Any, columns: "list[Any]", metas: "list[dict]") -> Optional[Any]:
    """``(len(metas), n)`` block of the Fourier / chirp columns, row ``j`` built from the device column ``columns[j]`` with ``metas[j]``, or ``None`` when the kernel is unavailable.

    ``columns`` are resident ``(n,)`` operands of one floating dtype (single or double precision); the block is in that dtype.
    """
    try:
        if not columns:
            return None
        dtype = columns[0].dtype
        if dtype not in (np.dtype(np.float32), np.dtype(np.float64)) or any(c.dtype != dtype for c in columns):  # Intentional float64: the two element types the kernel is templated for
            return None
        n = int(columns[0].shape[0])
        if any(int(c.shape[0]) != n for c in columns):
            return None
        params = np.array([_axis_params(m) for m in metas], dtype=np.float64)  # Intentional float64: axis parameters are formed in double precision and rounded once to the element type
        cosine = np.array([1 if m.get("kind") != "sin" else 0 for m in metas], dtype=np.int32)
        # Intentional float64: the angle weight 2 pi freq is formed in double precision (as in the per-column function) and rounded once to the element type
        weight = np.array([(2.0 * math.pi) * float(m["freq"]) for m in metas], dtype=np.float64)
        real = "float" if dtype == np.dtype(np.float32) else "double"
        k = len(metas)
        out = cp.empty((k, n), dtype=dtype)

        def up(a: np.ndarray) -> Any:
            """The array on the device in the element type of the columns."""
            return cp.asarray(np.ascontiguousarray(a.astype(dtype)))

        fourier = _kernel(cp, real)
        threads = block_size_for(fourier, work_items=k * n)

        def name_block(lo: int, hi: int) -> int:
            """Launch the names ``lo:hi`` into ``out[lo:hi]``: only the source columns this block references are stacked, so the temporary copy of the operands is bounded by the block, not the pool."""
            index: dict = {}
            stacked: list = []
            src = np.empty(hi - lo, dtype=np.int32)
            for j in range(lo, hi):
                col = columns[j]
                slot = index.get(id(col))
                if slot is None:
                    slot = len(stacked)
                    index[id(col)] = slot
                    stacked.append(col)
                src[j - lo] = slot
            xs = cp.stack(stacked)
            w_b = hi - lo
            fourier(
                (((w_b * n) + threads - 1) // threads,),
                (threads,),
                (xs, cp.asarray(src), cp.asarray(cosine[lo:hi]), cp.asarray(params[lo:hi, 4].astype(np.int32)), cp.asarray(params[lo:hi, 5].astype(np.int32)), up(params[lo:hi, 0]), up(params[lo:hi, 1]), up(params[lo:hi, 2]), up(params[lo:hi, 3]), up(weight[lo:hi]), np.int64(n), np.int32(w_b), out[lo:hi]),
            )
            return w_b

        from .._device_blocks import first_block, run_column_blocks

        run_column_blocks(cp, k, first_block(cp, n, k, int(dtype.itemsize), 2), name_block)  # column blocks of names, halved when a block does not fit
        return out
    except Exception as exc:  # nosec B110 - any device or compile failure falls back to the per-column build
        logger.debug("fourier_block_cuda unavailable (%s: %s); using the per-column build", type(exc).__name__, exc)
        return None


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Block kernel versus the per-column device build: kernel-tuning entry ``fourier_block_backend``, swept by the number of names of a pool (the numba twin ``_fourier_block_host`` is the host reference)
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
BLOCK_MIN_NAMES_FALLBACK = 4  # names from which the block kernel replaces the per-column build before a sweep has run (measured on the GTX 1050 Ti: 0.25 ms per name per column against 0.75 ms fixed)
_SWEEP_NAMES = [1, 2, 4, 8, 16, 64, 256]
_SWEEP_ROWS = 100_000
_SALT = 1


def _sweep_metas(n_names: int) -> "list[dict]":
    """``n_names`` linear and chirp names over four sources."""
    metas = []
    for j in range(n_names):
        m: dict = {"kind": "sin" if j % 2 == 0 else "cos", "freq": 0.25 * (1 + j % 12), "lo": -3.0, "span": 6.0, "src": j % 4}
        if j % 4 == 3:
            m.update({"arg": "quadratic", "mean": 0.01, "std": 1.02, "lo": -4.0, "span": 8.0})
        else:
            m["power"] = 1 + (j % 3 == 1)
        metas.append(m)
    return metas


def _sweep_columns(dims: dict) -> tuple:
    """Resident float32 operand columns and the names of the sweep point."""
    import cupy as cp

    rng = np.random.default_rng(0)
    xs = [cp.asarray(rng.standard_normal(_SWEEP_ROWS).astype(np.float32)) for _ in range(4)]
    metas = _sweep_metas(int(dims["n_names"]))
    return ([xs[m["src"]] for m in metas], metas)


def _per_column_build(columns: list, metas: list) -> np.ndarray:
    """The per-column device build (``_fourier_col_gpu`` per name), summarised for the comparison."""
    import cupy as cp

    from ._extra_basis_resident import _fourier_col_gpu

    out = cp.stack([_fourier_col_gpu(cp, c, m) for c, m in zip(columns, metas)])
    cp.cuda.Device().synchronize()
    return np.asarray(cp.asnumpy(out.sum(axis=1)), dtype=np.float64)  # Intentional float64: per-name column sums for the equivalence check


def _block_build(columns: list, metas: list) -> np.ndarray:
    """The one-launch block kernel; a failure raises so the sweep never records a fallback as a block time."""
    import cupy as cp

    out = fourier_block_cuda(cp, columns, metas)
    if out is None:
        raise RuntimeError("fourier_block_cuda unavailable")
    cp.cuda.Device().synchronize()
    return np.asarray(cp.asnumpy(out.sum(axis=1)), dtype=np.float64)  # Intentional float64: per-name column sums for the equivalence check


def _block_sweep() -> list:
    """Time the per-column build and the block kernel per number of names; the column sums must agree to float32 summation error."""
    from pyutilz.dev.benchmarking import sweep_backend_grid

    return list(
        sweep_backend_grid(
            {"per_column": _per_column_build, "block": _block_build},
            {"n_names": _SWEEP_NAMES},
            _sweep_columns,
            reference="per_column",
            repeats=3,
            equiv_rtol=1e-3,
            equiv_atol=1e-2,
        )
    )


def _block_fallback(n_names: int) -> str:
    """Pre-sweep choice: the block kernel from ``BLOCK_MIN_NAMES_FALLBACK`` names."""
    return "block" if int(n_names) >= BLOCK_MIN_NAMES_FALLBACK else "per_column"


try:
    from pyutilz.performance.kernel_tuning.registry import kernel_tuner

    _SPEC: Any = kernel_tuner(
        kernel_name="fourier_block_backend",
        variant_fns=(),  # the two builds are timed by the sweep callable
        tuner=_block_sweep,
        axes={"n_names": list(_SWEEP_NAMES)},
        fallback=_block_fallback,
        gpu_capable=True,
        salt=_SALT,
        cli_label="fourier_block_backend",
    )
except Exception as e:  # best-effort: without a tuner entry the fallback threshold decides
    logger.warning("kernel_tuner spec build for fourier_block_backend failed: %s", e)
    _SPEC = None


def block_preferred(n_names: int) -> bool:
    """Whether a pool of ``n_names`` Fourier / chirp names is built by the block kernel: the per-host cache by the number of names, else ``BLOCK_MIN_NAMES_FALLBACK``."""
    if _SPEC is None:
        return int(n_names) >= BLOCK_MIN_NAMES_FALLBACK
    try:
        return str(_SPEC.choose(n_names=int(n_names))) == "block"
    except Exception as exc:
        logger.debug("fourier_block_backend: KTC choose() failed, using the fallback: %s", exc)
        return int(n_names) >= BLOCK_MIN_NAMES_FALLBACK
