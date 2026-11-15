"""Device twin of ``_fe_synergy_screen._combo_mm_mi_cols_njit``: the Miller-Madow-corrected joint MI with the target of many pairs and triples of code columns in one launch.

The synergy screen scores up to a few thousand combos of two or three discrete columns, each by one pass of the host kernel over the rows. Here one block owns one combo: the rows fill a shared
``(joint cell, class)`` integer count table (integer atomics, so thread order does not change the counts), the threads sum ``pj * log(pj / (pxc * py))`` over the occupied cells and one warp-shuffle block
reduction gives the plug-in MI; the Miller-Madow debit ``(occ_x - 1 + occ_y - 1 - (occ_cells - 1)) / (2 n)`` is formed from the occupied counts exactly as on the host. A combo whose table does not fit the
shared memory (``_MAX_CELLS`` entries) returns NaN and the caller scores it with the host kernel. The code matrix goes up once, as bytes when every cardinality is at most 127; combos go in blocks of a bounded
count (``run_column_blocks``: a block of combos at a time, halved on an out-of-memory error). Host or device is chosen per call from the kernel-tuning entry ``synergy_combo_backend`` by rows x combos.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Mapping, Sequence

import numpy as np

from ._kernel_types import render
from ._ktc_fallback import host_device_sweep, rows_fallback

logger = logging.getLogger(__name__)

__all__ = ["combo_mm_mi_gpu", "combo_mm_mi_dispatch", "synergy_device_chosen"]

_MAX_CELLS = 8192  # (joint cell, class) entries of 4 bytes that fit the shared table
_THREADS = 256

_SRC = render(
    r"""
extern "C" __global__ __launch_bounds__(256)
void synergy_combo(const {CT}* __restrict__ codes, const int* __restrict__ c0, const int* __restrict__ c1, const int* __restrict__ c2, const int* __restrict__ card0,
                   const int* __restrict__ card1, const int* __restrict__ card2, const int* __restrict__ target, const long long n, const int kt, const int occ_y,
                   {ACC}* __restrict__ out) {
    extern __shared__ int tab[];                          // joint[cell * kt + class]
    __shared__ {ACC} red[32];
    __shared__ int red_c[32];
    __shared__ int pys[16];                               // class totals of the table (kt <= 16)
    const int p = blockIdx.x;
    const int k1 = card1[p];
    const int k2 = c2[p] >= 0 ? card2[p] : 1;
    const int n_cells = card0[p] * k1 * k2;
    const int size = n_cells * kt;
    for (int t = threadIdx.x; t < size; t += blockDim.x) tab[t] = 0;
    __syncthreads();
    const {CT}* a = codes + (long long)c0[p] * n;
    const {CT}* b = codes + (long long)c1[p] * n;
    const {CT}* c = c2[p] >= 0 ? codes + (long long)c2[p] * n : codes;
    const bool triple = c2[p] >= 0;
    for (long long i = threadIdx.x; i < n; i += blockDim.x) {
        int cell = (int)a[i] * k1 + (int)b[i];
        if (triple) cell = cell * k2 + (int)c[i];
        atomicAdd(&tab[cell * kt + target[i]], 1);
    }
    __syncthreads();
    if (threadIdx.x < kt) {
        int tot_t = 0;
        for (int cell = 0; cell < n_cells; ++cell) tot_t += tab[cell * kt + threadIdx.x];
        pys[threadIdx.x] = tot_t;
    }
    __syncthreads();
    const {ACC} inv_n = 1.0 / ({ACC})n;
    {ACC} mi = 0.0;
    int occ = 0;
    for (int cell = threadIdx.x; cell < n_cells; cell += blockDim.x) {
        int px_i = 0;
        for (int t = 0; t < kt; ++t) px_i += tab[cell * kt + t];
        if (px_i <= 0) continue;
        occ += 1;
        const {ACC} pxc = ({ACC})px_i * inv_n;
        for (int t = 0; t < kt; ++t) {
            const int cnt = tab[cell * kt + t];
            if (cnt > 0) {
                const int py_i = pys[t];
                const {ACC} pj = ({ACC})cnt * inv_n;
                mi += pj * log(pj / (pxc * (({ACC})py_i * inv_n)));
            }
        }
    }
    for (int o = 16; o > 0; o >>= 1) {
        mi += __shfl_down_sync(0xffffffffu, mi, o);
        occ += __shfl_down_sync(0xffffffffu, occ, o);
    }
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) { red[warp] = mi; red_c[warp] = occ; }
    __syncthreads();
    if (threadIdx.x == 0) {
        {ACC} tot = 0.0;
        int tc = 0;
        for (int w = 0; w < (blockDim.x >> 5); ++w) { tot += red[w]; tc += red_c[w]; }
        const {ACC} mm = (({ACC})(tc - 1 + occ_y - 1 - (tc - 1))) / (2.0 * ({ACC})n);
        const {ACC} val = tot + mm;
        out[p] = val > 0.0 ? val : 0.0;
    }
}
""",
)

_KERNELS: dict = {}


def _kernel(cp: Any, narrow: bool) -> Any:
    """The compiled kernel, one per code type: bytes when every cardinality is at most 127, else 32-bit integers."""
    if narrow not in _KERNELS:
        _KERNELS[narrow] = cp.RawKernel(_SRC.replace("{CT}", "signed char" if narrow else "int"), "synergy_combo")
    return _KERNELS[narrow]


def kernel(cp: Any, real: Any) -> Any:
    """``synergy_combo`` for codes of dtype ``real`` (uint8: byte codes, else 32-bit integers); the register-pressure registry compiles it through this."""
    return _kernel(cp, np.dtype(real) == np.uint8)


def combo_mm_mi_gpu(code_cols: Sequence[np.ndarray], combos: Sequence[Sequence[int]], cards: Sequence[int], target: np.ndarray, kt: int) -> np.ndarray:
    """MM-corrected joint MI of each combo (indices into ``code_cols``, two or three of them, ``cards`` the cardinality of every column) with ``target``; NaN for a combo over the table limit. Raises on a device fault."""
    import cupy as cp

    from ._device_blocks import run_column_blocks

    k = len(combos)
    out = np.full(k, np.nan)  # Intentional float64: MI statistic compared to the host's at full precision
    n = int(target.shape[0])
    if k == 0 or n == 0:
        return out
    cards_a = np.asarray(cards, dtype=np.int64)
    cards_l = [int(c) for c in cards]
    cells = np.array([math.prod(cards_l[c] for c in combo) * int(kt) for combo in combos], dtype=np.int64)  # python ints: a numpy reduction per combo cost more than the launch
    fits = np.nonzero(cells <= _MAX_CELLS)[0] if int(kt) <= 16 else np.zeros(0, dtype=np.int64)
    if fits.size == 0:
        return out
    narrow = bool(cards_a.max() <= 127)
    mat = np.empty((len(code_cols), n), dtype=np.int8 if narrow else np.int32)
    for j, col in enumerate(code_cols):
        mat[j] = col
    codes_d = cp.asarray(mat)
    target_d = cp.asarray(np.ascontiguousarray(target, dtype=np.int32))
    occ_y = int((np.bincount(np.asarray(target, dtype=np.int64), minlength=int(kt)) > 0).sum())
    kernel = _kernel(cp, narrow)

    def block_work(lo: int, hi: int) -> int:
        """Score the combos ``fits[lo:hi]`` (a block of combos; the code matrix is already resident)."""
        sel = fits[lo:hi]
        m = int(sel.shape[0])
        c0 = np.empty(m, dtype=np.int32)
        c1 = np.empty(m, dtype=np.int32)
        c2 = np.full(m, -1, dtype=np.int32)
        k0 = np.empty(m, dtype=np.int32)
        k1 = np.empty(m, dtype=np.int32)
        k2 = np.ones(m, dtype=np.int32)
        for q, idx in enumerate(sel):
            combo = combos[int(idx)]
            c0[q], c1[q] = combo[0], combo[1]
            k0[q], k1[q] = cards_a[combo[0]], cards_a[combo[1]]
            if len(combo) >= 3:
                c2[q], k2[q] = combo[2], cards_a[combo[2]]
        res = cp.empty(m, dtype=cp.float64)  # Intentional float64: MI statistic output
        shared = int(cells[sel].max()) * 4
        kernel(
            (m,),
            (_THREADS,),
            (codes_d, cp.asarray(c0), cp.asarray(c1), cp.asarray(c2), cp.asarray(k0), cp.asarray(k1), cp.asarray(k2), target_d, np.int64(n), np.int32(kt), np.int32(occ_y), res),
            shared_mem=shared,
        )  # launch-ok: one block per combo is the design
        out[sel] = cp.asnumpy(res)
        return m

    run_column_blocks(cp, int(fits.size), int(fits.size), block_work)  # one launch for all combos; halved only if the launch runs out of memory
    return out


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
DEVICE_FALLBACK_ROWS = 500_000  # rows x combos from which the device scores the combos before a sweep has run (crossover between 10 and 40 combos of 20000 rows)
_SWEEP_N = 20_000
_SWEEP_ROWS = [_SWEEP_N * k for k in (10, 40, 160, 640)]
_SALT = 1


def _sweep_inputs(dims: dict) -> tuple:
    """Eight columns of 4 levels over 20000 rows, ``dims["n_samples"] / 20000`` pairs and triples over them, a binary target that is the parity of the first two."""
    import itertools

    rng = np.random.default_rng(0)
    cols = [rng.integers(0, 4, _SWEEP_N).astype(np.int64) for _ in range(8)]
    target = ((cols[0] + cols[1]) % 2).astype(np.int64)
    combos = [c for r in (2, 3) for c in itertools.combinations(range(8), r)]
    k = max(1, int(dims["n_samples"]) // _SWEEP_N)
    combos = (combos * (k // len(combos) + 1))[:k]
    return (cols, combos, [4] * 8, target, 2)


def _sweep_host(cols: list, combos: list, cards: list, target: np.ndarray, kt: int) -> np.ndarray:
    """The numba column kernel per combo."""
    from ._fe_synergy_screen import _combo_mm_mi_cols_njit

    out = np.empty(len(combos))
    for q, combo in enumerate(combos):
        cs = np.array([cards[c] for c in combo], dtype=np.int64)
        c2 = cols[combo[2]] if len(combo) >= 3 else cols[combo[1]]
        out[q] = _combo_mm_mi_cols_njit(cols[combo[0]], cols[combo[1]], c2, len(combo), cs, target, kt, int(np.prod(cs)))
    return out


def _sweep_device(cols: list, combos: list, cards: list, target: np.ndarray, kt: int) -> np.ndarray:
    """The device kernel; every sweep combo fits, so a failed launch raises and is never recorded as a host time."""
    return combo_mm_mi_gpu(cols, combos, cards, target, kt)


try:
    from pyutilz.performance.kernel_tuning.registry import kernel_tuner

    _SPEC: Any = kernel_tuner(
        kernel_name="synergy_combo_backend",
        variant_fns=(),  # the two backends are timed by the sweep callable
        tuner=host_device_sweep(__name__, _sweep_host, _sweep_device, rtol=1e-9, atol=1e-12),
        axes={"n_samples": list(_SWEEP_ROWS)},
        fallback=rows_fallback(__name__),
        gpu_capable=True,
        salt=_SALT,
        cli_label="synergy_combo_backend",
    )
except Exception as e:  # best-effort: without a tuner entry the floor decides
    logger.warning("kernel_tuner spec build for synergy_combo_backend failed: %s", e)
    _SPEC = None


def synergy_device_chosen(n: int, n_combos: int) -> bool:
    """Whether ``n_combos`` combos over ``n`` rows are scored on the device: the per-host cache by rows x combos, else the ``DEVICE_FALLBACK_ROWS`` floor."""
    work = int(n) * int(n_combos)
    if _SPEC is None:
        return work >= DEVICE_FALLBACK_ROWS
    try:
        return str(_SPEC.choose(n_samples=work)) == "device"
    except Exception as exc:
        logger.debug("synergy_combo_backend: KTC choose() failed, using the floor: %s", exc)
        return work >= DEVICE_FALLBACK_ROWS


def combo_mm_mi_dispatch(host: Any, code_cols: "Sequence[np.ndarray] | Mapping[int, np.ndarray]", combos: Sequence[Sequence[int]], cards: "Sequence[int] | Mapping[int, int]", target: np.ndarray, kt: int) -> np.ndarray:
    """The MM-corrected joint MI of every combo: on the device when the tuned choice says so and a device is usable (combos over the table limit, and every combo after a device fault, from ``host``), else ``host``.
    ``host(code_cols, combos, cards, target, kt)`` returns the array of values on the host."""
    if len(combos) and synergy_device_chosen(int(target.shape[0]), len(combos)):
        try:
            from ._gpu_policy import cuda_available_for_run

            if cuda_available_for_run():
                if isinstance(code_cols, Mapping):  # columns keyed by feature index: dense positions for the device matrix
                    keys = list(code_cols)
                    pos = {key: i for i, key in enumerate(keys)}
                    cols_seq = [code_cols[key] for key in keys]
                    cards_seq = [cards[key] for key in keys]
                    combos_dense = [tuple(pos[c] for c in combo) for combo in combos]
                else:
                    cols_seq, cards_seq, combos_dense = list(code_cols), list(cards), [tuple(c) for c in combos]
                out = combo_mm_mi_gpu(cols_seq, combos_dense, cards_seq, target, kt)
                todo = np.nonzero(np.isnan(out))[0]
                if todo.size:
                    out[todo] = host(code_cols, [combos[int(i)] for i in todo], cards, target, kt)
                return out
        except Exception as exc:  # a device fault must not lose the screen
            from mlframe.utils.log_throttle import log_throttle

            log_throttle(logger, "synergy_combo.device_failed", logging.WARNING, "synergy combo device path failed (%s: %s); numba kernel", type(exc).__name__, exc)
    return np.asarray(host(code_cols, combos, cards, target, kt))
