"""Device twin of the bit-set family null (``_family_maxt_null_v2_njit.null_max_bitsets``).

The joint count of bin ``a`` and class ``c`` is ``popcount(bin_a & class_c)`` over 64-row words; the last class is the bin total minus the others. One thread handles one (permutation, column):
it walks the words of the column's bin bit sets (stored word-major, so the threads of a warp read consecutive columns: coalesced) against the permutation's class bit sets (the same words for every thread of the block, a broadcast read), so the
histogram's scattered increments become independent AND / population-count / add operations. The column bit sets (``K x bins x words`` uint64) are uploaded once per family and reused by every
block of permutations; only the ``B x classes x words`` class bit sets of a block cross the bus.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, cast

import numpy as np

from pyutilz.performance.gpu_hw_launch import block_size_for

from ._fe_batched_mi_cmi import _sm_count

logger = logging.getLogger(__name__)

_SOURCE = r"""
// coalescing-ok: the only strided read is px[j * cap + a], once per bin of a column; the word loop that carries the cost reads the bit sets coalesced (gld efficiency 99.5 percent, nvprof)
extern "C" __global__ void family_null_bits(const unsigned long long* __restrict__ bits, const long long* __restrict__ n_x, const unsigned long long* __restrict__ cls,
                                            const {REAL}* __restrict__ table, const {REAL}* __restrict__ c_a, const long long* __restrict__ px,
                                            const {REAL}* __restrict__ const_b, const long long n, const int cap, const int words, const int n_y, const int K, const int B,
                                            {REAL}* __restrict__ out) {
    const long long idx = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= (long long)B * K) return;
    const int b = (int)(idx / K);
    const int j = (int)(idx - (long long)b * K);
    double s = 0.0;
    const int nb = (int)n_x[j];
    for (int a = 0; a < nb; ++a) {
        const unsigned long long* row = bits + (long long)a * words * K + j;
        long long rest = px[(long long)j * cap + a];
        for (int c = 0; c < n_y - 1; ++c) {
            const unsigned long long* cl = cls + ((long long)b * n_y + c) * words;
            int cnt = 0;
            for (int w = 0; w < words; ++w) cnt += __popcll(row[(long long)w * K] & cl[w]);
            s += (double)table[cnt];
            rest -= cnt;
        }
        s += (double)table[rest];
    }
    out[idx] = ({REAL})((s + (double)const_b[b] - (double)c_a[j]) / (double)n);
}
"""

_kernels: dict = {}
_KERNEL_CACHE_LOCK = threading.RLock()

# One block per (permutation, tile of columns). The permutation's class bit sets sit in shared memory (every thread of the block reads the same word: a broadcast), and each thread reads a word of its
# column's bin bit set ONCE and ANDs it against every class, accumulating the joint counts in registers. The first kernel re-read the bin words once per class and the class words from global memory;
# this one reads the bin words ``n_y - 1`` times less. The sum is accumulated in the same order as the first kernel (per bin: classes in order, then the remainder), so the result is bit-identical.
_SOURCE_V2 = r"""
// coalescing-ok: the only strided read is px[j * cap + a], once per bin of a column; the word loop that carries the cost reads the bit sets coalesced (gld efficiency 99.5 percent, nvprof)
extern "C" __global__ void family_null_bits_v2(const unsigned long long* __restrict__ bits, const long long* __restrict__ n_x, const unsigned long long* __restrict__ cls,
                                               const {REAL}* __restrict__ table, const {REAL}* __restrict__ c_a, const long long* __restrict__ px,
                                               const {REAL}* __restrict__ const_b, const long long n, const int cap, const int words, const int n_y, const int K, const int B,
                                               {REAL}* __restrict__ out) {
    extern __shared__ unsigned long long sh_cls[];
    const int b = blockIdx.x;
    const int j = blockIdx.y * blockDim.x + threadIdx.x;
    for (int i = threadIdx.x; i < NY1 * words; i += blockDim.x) sh_cls[i] = cls[((long long)b * n_y) * words + i];
    __syncthreads();
    if (j >= K) return;
    double s = 0.0;
    const int nb = (int)n_x[j];
    for (int a = 0; a < nb; ++a) {
        const unsigned long long* row = bits + (long long)a * words * K + j;
        int cnt[NY1];
        #pragma unroll
        for (int c = 0; c < NY1; ++c) cnt[c] = 0;
        for (int w = 0; w < words; ++w) {
            const unsigned long long r = row[(long long)w * K];
            #pragma unroll
            for (int c = 0; c < NY1; ++c) cnt[c] += __popcll(r & sh_cls[c * words + w]);
        }
        long long rest = px[(long long)j * cap + a];
        #pragma unroll
        for (int c = 0; c < NY1; ++c) { s += (double)table[cnt[c]]; rest -= cnt[c]; }
        s += (double)table[rest];
    }
    out[(long long)b * K + j] = ({REAL})((s + (double)const_b[b] - (double)c_a[j]) / (double)n);
}
"""

_kernels_v2: dict = {}
_THREADS_V2 = 128  # the width without device information; the launches take it from ``v2_block_width``


def v2_block_width(kern: Any, shared_bytes: int, columns: int) -> int:
    """Threads per block of the v2 kernel from the device's limits: its class bitsets (``shared_bytes``, fixed per block) are shared by the block's columns, so the width that keeps the most threads resident on an SM wins
    (256 at 100000 rows on a Pascal SM: 19 ms against 28 ms at 128, 300 permutations x 250 columns); ``MLFRAME_GPU_BLOCK_FAMILY_BITS`` overrides; ``_THREADS_V2`` without a device.
    """
    import os

    forced = os.environ.get("MLFRAME_GPU_BLOCK_FAMILY_BITS", "").strip()
    if forced.isdigit() and 32 <= int(forced) <= 1024 and int(forced) % 32 == 0:
        return int(forced)
    from pyutilz.performance.gpu_hw_launch import fixed_shared_block_size

    try:
        width = fixed_shared_block_size(
            shared_bytes, regs_per_thread=int(kern.num_regs), static_smem=int(kern.shared_size_bytes), min_width=32, max_width=1024, work_items=columns
        )
    except Exception as _silent_exc:
        logger.warning("_family_maxt_cuda_bits.py: block width not derived (%r); the default width is used", _silent_exc)
        width = 0
    return width if width > 0 else _THREADS_V2


_MAX_CLASSES_V2 = 24  # registers hold ``n_y - 1`` counters per thread
_SHARED_MARGIN = 1024


def _get_kernel_v2(cp, ny1: int, real: str):
    """Compile the shared-memory kernel for ``ny1 = n_y - 1`` stored classes and one element type (one binary per pair)."""
    with _KERNEL_CACHE_LOCK:
        kern = _kernels_v2.get((ny1, real))
        if kern is None:
            kern = cp.RawModule(code=_SOURCE_V2.replace("{REAL}", real), options=(f"-DNY1={int(ny1)}",)).get_function("family_null_bits_v2")
            _kernels_v2[(ny1, real)] = kern
    return kern


# Win axis of v3 against v2 (both kept; ``MLFRAME_FAMILY_BITS_V3=0`` forces v2). v2 re-reads every column's bin bit sets once per permutation: 40 permutations x 100 MB = 4 GB through L2 for a 9 ms call
# (nvprof: 78% of issue stalls on memory dependency, L2 190 GB/s, DRAM 23 GB/s; unrolling the word loop gains 6%, the compiler already unrolls it). v3 gives each block ``PB`` permutations and each thread the counters of
# every bin of its column (``CAP`` bins, a compile-time constant), walks the words in tiles of ``TW`` class words staged in shared memory (4 KB: the whole-permutation tile of v2 is 12.5 KB per class and caps the
# resident blocks), and loads each bin word ONCE for all ``PB`` permutations. The sums are accumulated per pair in v2's order (per bin: classes in order, then the remainder): bit-identical results.
# Applies when ``CAP * PB * NY1 <= 32`` counters fit registers with ``PB >= 2``.
_SOURCE_V3 = r"""
#define TW 128
// coalescing-ok: the only strided read is px[j * cap + a], once per bin of a column; the word loop that carries the cost reads the bit sets coalesced (gld efficiency 98.1 percent, nvprof)
extern "C" __global__ void family_null_bits_v3(const unsigned long long* __restrict__ bits, const long long* __restrict__ n_x, const unsigned long long* __restrict__ cls,
                                               const {REAL}* __restrict__ table, const {REAL}* __restrict__ c_a, const long long* __restrict__ px,
                                               const {REAL}* __restrict__ const_b, const long long n, const int cap, const int words, const int n_y, const int K, const int B,
                                               {REAL}* __restrict__ out) {
    __shared__ unsigned long long sh_cls[PB * NY1 * TW];
    __shared__ int sh_tot[PB * NY1];  // rows of each stored class per permutation of this block (bins partition the rows, so the last bin of a column is a difference)
    const int b0 = blockIdx.x * PB;
    const int j = blockIdx.y * blockDim.x + threadIdx.x;
    const bool ok = j < K;
    const int nb = ok ? (int)n_x[j] : 0;
    long long px_sum = 0;
    if (CAP >= 8) {
        for (int a = 0; a < nb; ++a) px_sum += px[(long long)j * cap + a];
    }
    const bool exact = CAP >= 8 && ok && nb >= 1 && px_sum == n;  // every row has a code in [0, nb): the last bin is then not counted but derived; measured a win from 8 bins, a 9 percent loss at 4
    const int last = exact ? nb - 1 : nb;
    for (int i = threadIdx.x; i < PB * NY1; i += blockDim.x) sh_tot[i] = 0;
    int cnt[CAP][PB][NY1];
    #pragma unroll
    for (int a = 0; a < CAP; ++a)
        #pragma unroll
        for (int p = 0; p < PB; ++p)
            #pragma unroll
            for (int c = 0; c < NY1; ++c) cnt[a][p][c] = 0;
    for (int w0 = 0; w0 < words; w0 += TW) {
        const int tw = min(TW, words - w0);
        __syncthreads();
        for (int i = threadIdx.x; i < PB * NY1 * TW; i += blockDim.x) {
            const int p = i / (NY1 * TW);
            const int rem = i - p * NY1 * TW;
            const int c = rem / TW;
            const int w = rem - c * TW;
            const int bp = b0 + p;
            const unsigned long long v = (bp < B && w < tw) ? cls[((long long)bp * n_y + c) * words + w0 + w] : 0ULL;
            sh_cls[i] = v;
            if (CAP >= 8 && v) atomicAdd(&sh_tot[p * NY1 + c], __popcll(v));
        }
        __syncthreads();
        #pragma unroll
        for (int a = 0; a < CAP; ++a) {
            if (a < last) {
                const unsigned long long* row = bits + (long long)a * words * K + (long long)w0 * K + j;
                for (int w = 0; w < tw; ++w) {
                    const unsigned long long r = row[(long long)w * K];
                    #pragma unroll
                    for (int c = 0; c < NY1; ++c) {
                        #pragma unroll
                        for (int p = 0; p < PB; ++p) cnt[a][p][c] += __popcll(r & sh_cls[(p * NY1 + c) * TW + w]);
                    }
                }
            }
        }
    }
    __syncthreads();
    if (!ok) return;
    #pragma unroll
    for (int p = 0; p < PB; ++p) {
        double s = 0.0;
        int run[NY1];  // stored-class rows seen in the bins before this one: the last bin of an exact column is the class total minus them
        #pragma unroll
        for (int c = 0; c < NY1; ++c) run[c] = 0;
        #pragma unroll
        for (int a = 0; a < CAP; ++a) {
            if (a < nb) {
                long long rest = px[(long long)j * cap + a];
                #pragma unroll
                for (int c = 0; c < NY1; ++c) {
                    const int v = (exact && a == nb - 1) ? sh_tot[p * NY1 + c] - run[c] : cnt[a][p][c];
                    run[c] += v;
                    s += (double)table[v];
                    rest -= v;
                }
                s += (double)table[rest];
            }
        }
        if (b0 + p < B) out[(long long)(b0 + p) * K + j] = ({REAL})((s + (double)const_b[b0 + p] - (double)c_a[j]) / (double)n);
    }
}
"""

_kernels_v3: dict = {}
_V3_COUNTERS = 32  # CAP * PB * NY1 int registers of counters
_V3_MAX_PB = 4
_V3_TILE_WORDS = 128


def _v3_plan(n_y: int, cap: int) -> "int | None":
    """Permutations per block of the v3 kernel, or ``None`` when v2 runs (disabled, or the counters of ``cap`` bins x ``n_y - 1`` classes leave under two permutations in registers)."""
    import os

    if os.environ.get("MLFRAME_FAMILY_BITS_V3", "1").strip() == "0" or n_y < 2 or cap < 1:
        return None
    pb = min(_V3_MAX_PB, _V3_COUNTERS // (int(cap) * (n_y - 1)))
    return pb if pb >= 2 else None


def _get_kernel_v3(cp, ny1: int, pb: int, cap: int, real: str):
    """Compile the permutation-blocked kernel for ``ny1`` stored classes, ``pb`` permutations per block, ``cap`` bins and one element type."""
    with _KERNEL_CACHE_LOCK:
        kern = _kernels_v3.get((ny1, pb, cap, real))
        if kern is None:
            kern = cp.RawModule(code=_SOURCE_V3.replace("{REAL}", real), options=(f"-DNY1={int(ny1)}", f"-DPB={int(pb)}", f"-DCAP={int(cap)}")).get_function(
                "family_null_bits_v3"
            )
            _kernels_v3[(ny1, pb, cap, real)] = kern
    return kern


def kernel_v3(cp, real, ny1: int = 1, pb: int = _V3_MAX_PB, cap: int = 8):
    """The v3 kernel for the value type ``real`` (float32 or float64), for the register-pressure registry: one stored class, four permutations per block, eight bins."""
    return _get_kernel_v3(cp, ny1, pb, cap, "float" if np.dtype(real) == np.float32 else "double")


_FILL_BLOCKS_PER_SM = 2  # the launch aims for at least this many blocks per multiprocessor when the columns allow it


def fill_block_width(width: int, columns: int, other_blocks: int, sms: int) -> int:
    """Threads per block, narrowed from ``width`` (never under one warp) so the grid of ``other_blocks`` x ``ceil(columns / width)`` blocks reaches ``_FILL_BLOCKS_PER_SM`` blocks per multiprocessor.

    The result is only a launch shape: every column is still computed by one thread, so the numbers do not change. At 100 columns and 40 permutations a 256-thread block gave 10 x 1 = 10 blocks on a 24-SM card.
    """
    want_tiles = max(1, -(-_FILL_BLOCKS_PER_SM * max(1, int(sms)) // max(1, int(other_blocks))))
    narrow = -(-max(1, int(columns)) // want_tiles)
    narrow = ((narrow + 31) // 32) * 32
    return int(max(32, min(int(width), narrow)))


def family_null_launcher(cp, n_y: int, words: int, table, cls, n: int, cap: int, b: int):
    """The kernel of the shared-memory family null (v3 when it applies, else v2) for ``b`` permutations, resolved ONCE, as a function ``launch(bits, n_x, c_a, px, const_b, k, out)`` for one
    column group of ``k`` columns whose ``(b, k)`` output is ``out``.

    Resolving the kernel (a cached compile) is the only cupy work besides the launch itself, so a loop over column groups calls this once and then only launches. The kernel's element
    type follows the resident MI ``table``."""
    real = "float" if table.dtype == cp.float32 else "double"
    pb = _v3_plan(n_y, cap)
    if pb is not None:
        kern = _get_kernel_v3(cp, n_y - 1, pb, cap, real)

        def launch_v3(bits, n_x, c_a, px, const_b, k: int, out) -> None:
            """Launch the v3 kernel over one column group."""
            width = fill_block_width(v2_block_width(kern, pb * (n_y - 1) * _V3_TILE_WORDS * 8, k), k, (b + pb - 1) // pb, _sm_count())
            kern(
                ((b + pb - 1) // pb, (k + width - 1) // width),
                (width,),
                (bits, n_x, cls, table, c_a, px, const_b, np.int64(n), np.int32(cap), np.int32(words), np.int32(n_y), np.int32(k), np.int32(b), out),
            )

        return launch_v3
    kern2 = _get_kernel_v2(cp, n_y - 1, real)

    def launch_v2(bits, n_x, c_a, px, const_b, k: int, out) -> None:
        """Launch the v2 kernel over one column group."""
        width = fill_block_width(v2_block_width(kern2, (n_y - 1) * words * 8, k), k, b, _sm_count())
        kern2(
            (b, (k + width - 1) // width),
            (width,),
            (bits, n_x, cls, table, c_a, px, const_b, np.int64(n), np.int32(cap), np.int32(words), np.int32(n_y), np.int32(k), np.int32(b), out),
            shared_mem=(n_y - 1) * words * 8,
        )

    return launch_v2


def launch_family_null(cp, n_y: int, words: int, bits, n_x, cls, table, c_a, px, const_b, n: int, cap: int, k: int, b: int, out) -> None:
    """Launch the shared-memory family null kernel (v3 when it applies, else v2) over ``b`` permutations and ``k`` columns; ``out`` is ``(b, k)``.

    One-shot form of :func:`family_null_launcher` (a loop over column groups builds the launcher once instead)."""
    family_null_launcher(cp, n_y, words, table, cls, n, cap, b)(bits, n_x, c_a, px, const_b, k, out)


def _v2_applies(cp, n_y: int, words: int) -> bool:
    """Whether the shared-memory kernel can hold the class bit sets of one permutation: at least two classes, a bounded count, and a block's worth of shared memory."""
    if n_y < 2 or n_y - 1 > _MAX_CLASSES_V2:
        return False
    try:
        limit = int(cp.cuda.Device().attributes.get("MaxSharedMemoryPerBlock", 48 * 1024))
    except Exception as e:
        logger.warning("Could not read MaxSharedMemoryPerBlock (%s: %s); assuming 48 KiB", type(e).__name__, e)
        limit = 48 * 1024
    return (n_y - 1) * words * 8 <= limit - _SHARED_MARGIN


def _get_kernel(cp, real: str):
    """Compile the kernel once per element type (``"float"`` or ``"double"``)."""
    with _KERNEL_CACHE_LOCK:
        kern = _kernels.get(real)
        if kern is None:
            kern = cp.RawKernel(_SOURCE.replace("{REAL}", real), "family_null_bits")
            _kernels[real] = kern
    return kern


class DeviceBitsets:
    """A family's column bit sets (and the tables the MI needs) resident on the device, grouped within the memory budget."""

    def __init__(self, cp, prep, n_x: np.ndarray, n: int):
        table, c_a, px, groups = prep
        self.cp = cp
        self.n = int(n)
        self.table = cp.asarray(table)
        self.c_a = cp.asarray(c_a, dtype=self.table.dtype)
        self.px = cp.asarray(px)
        self.n_x = cp.asarray(np.ascontiguousarray(n_x, dtype=np.int64))
        self.groups: list = []
        for lo, hi, bits in groups:  # a generator is consumed one group at a time: the host bit sets of a group are dropped once it is on the device
            if not self.groups:
                self.cap = int(bits.shape[1])
                self.words = int(bits.shape[2])
            # word-major per group: (bins, words, columns), so the threads of a warp (consecutive columns) read consecutive addresses
            self.groups.append((lo, hi, cp.ascontiguousarray(cp.asarray(bits).transpose(1, 2, 0))))


def null_max_bits_device(dev: DeviceBitsets, cls_host: np.ndarray, py_const: float, acc: Any = None) -> np.ndarray | None:
    """Per-permutation maximum MI over the family. ``cls_host`` is ``(B, n_y, words)`` uint64; ``py_const = n log n - sum_b xlogx(py_b)``.

    ``acc``: an optional device double vector of length ``B``. When given, the maxima are folded into it on the device and nothing
    is copied to the host (returns ``None``); the caller reads the whole family's maxima back once, after every column group."""
    # Intentional float64: label-entropy constant of the MI formula (exact permutation ties)
    const_b = np.full(cls_host.shape[0], float(py_const), dtype=np.float64)
    return _run(dev, cls_host, const_b, reduce_max=True, acc=acc)


def mi_matrix_bits_device(dev: DeviceBitsets, cls_host: np.ndarray, const_b: np.ndarray) -> np.ndarray:
    """``(B, K)`` plug-in MI of every column against every label vector; ``const_b[b] = n log n - sum_c xlogx(count_c)`` of label vector ``b`` (they differ when each vector has its own class counts)."""
    return cast(np.ndarray, _run(dev, cls_host, const_b, reduce_max=False))  # no accumulator, so the matrix is always returned


def _run(dev: DeviceBitsets, cls_host: np.ndarray, const_b: np.ndarray, reduce_max: bool, acc: Any = None) -> np.ndarray | None:
    """Launch the bit-set kernel over every column group: the per-permutation maximum, or the full ``(B, K)`` matrix.

    The kernel's element type follows the resident MI table (``dev.table``); the constants and the output are built in that type, the sums are accumulated in double.
    """
    cp = dev.cp
    real = "float" if dev.table.dtype == cp.float32 else "double"
    B, n_y, _ = cls_host.shape
    cls_d = cp.asarray(cls_host)
    # Intentional float64: label-entropy constants of the MI formula, converted to the table's element type below
    const_d = cp.asarray(np.ascontiguousarray(const_b, dtype=np.float64)).astype(dev.table.dtype, copy=False)
    best = (
        cp.zeros(B, dtype=cp.float64) if acc is None else acc
    )  # Intentional float64: statistic vector (MI / null max / p-value inputs), exact permutation ties
    parts = []
    use_v2 = _v2_applies(cp, n_y, dev.words)
    kern = None if use_v2 else _get_kernel(cp, real)
    launch = family_null_launcher(cp, n_y, dev.words, dev.table, cls_d, dev.n, dev.cap, B) if use_v2 else None
    widths = [hi - lo for lo, hi, _ in dev.groups]
    # one backing buffer for every group's (B, k) block (a block is a contiguous slice), or one reused block when only the per-permutation maximum is kept
    flat = cp.empty(B * (max(widths) if reduce_max else sum(widths)), dtype=dev.table.dtype) if widths else None
    offsets = np.concatenate([[0], np.cumsum(widths)]).astype(np.int64) if widths else None
    outs = [flat[: B * k].reshape(B, k) if reduce_max else flat[B * int(offsets[g]) : B * int(offsets[g + 1])].reshape(B, k) for g, k in enumerate(widths)]
    for (lo, hi, bits), out in zip(dev.groups, outs):
        k = hi - lo
        if launch is not None:
            launch(bits, dev.n_x[lo:hi], dev.c_a[lo:hi], dev.px[lo:hi], const_d, k, out)
        else:
            assert kern is not None  # use_v2 is False on this branch, so the v1 kernel was fetched
            flat_threads = block_size_for(kern, work_items=B * k)
            kern(
                ((B * k + flat_threads - 1) // flat_threads,),
                (flat_threads,),
                (
                    bits,
                    dev.n_x[lo:hi],
                    cls_d,
                    dev.table,
                    dev.c_a[lo:hi],
                    dev.px[lo:hi],
                    const_d,
                    np.int64(dev.n),
                    np.int32(dev.cap),
                    np.int32(dev.words),
                    np.int32(n_y),
                    np.int32(k),
                    np.int32(B),
                    out,
                ),
            )
        if reduce_max:
            cp.maximum(best, out.max(axis=1), out=best)  # in place: the accumulator may be the caller's device vector
        else:
            parts.append(out)
    if reduce_max:
        if acc is not None:
            return None  # folded into the caller's device accumulator; no host copy here
        host: np.ndarray = cp.asnumpy(best)
        return host
    return np.asarray(cp.asnumpy(parts[0] if len(parts) == 1 else cp.concatenate(parts, axis=1)))
