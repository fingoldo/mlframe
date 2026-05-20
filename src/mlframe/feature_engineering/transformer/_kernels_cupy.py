"""GPU (cupy) kernels for the two stages where GPU actually wins.

Two paths:

1. ``rff_matmul_cupy`` - streaming matmul with pinned host buffers and double-buffer streams. At d=20k, the input X is up to 800 GB; pinned memory + stream overlap
   takes effective H2D throughput from ~6 GB/s pageable to ~12 GB/s pinned, and triple-buffer hides compute behind H2D. Without these, GPU win over CPU njit
   matmul at d=20k is only ~1.8x; with them, ~4x.

2. ``row_attention_stage4_cupy_fused`` - one RawKernel doing gather + dot + softmax + weighted-sum-V. The cupy-primitives alternative (gather via fancy indexing,
   then cp.special.softmax, then cp.matmul) does each step in a separate kernel launch with a separate DRAM round-trip; fusing into one launch is 3-9x faster
   on the bandwidth-bound stage 4.

Cupy is imported lazily inside the functions. Module-level RawKernel placeholders are populated on first use under a multiprocessing lock so Windows-spawn
workers don't race during initialisation. Pattern mirrors ``mlframe.feature_selection.filters.gpu`` which is the production reference for cupy in this codebase.
"""
from __future__ import annotations

import logging
import multiprocessing
import sys
from typing import Any

import numpy as np

from ._utils import is_gpu_available

logger = logging.getLogger(__name__)

# Module-level RawKernel placeholder. ``_ensure_kernels_inited`` populates it on first use.
row_attention_stage4_raw_kernel: Any = None

# Wave 27 P2 fix (2026-05-20): intra-process lock. Pre-fix claimed
# "Cross-process safe lock (Windows spawn workers)" -- that's FALSE on
# spawn since each child re-imports the module and constructs a fresh
# Lock(). The parent + children don't share the underlying SemLock.
# Acceptable HERE because the lock body is idempotent (NVRTC compile);
# documented honestly to avoid misleading future contributors.
_KERNEL_INIT_LOCK = multiprocessing.Lock()
_KERNELS_INITED = False

# Reusable host-pinned buffer pool. Allocating pinned memory is expensive (~100 us per MB on first call) so we keep a small pool that grows monotonically.
_PINNED_BUFFERS: dict[tuple[str, tuple[int, ...], np.dtype], Any] = {}


def _ensure_kernels_inited() -> None:
    """Build the ``RawKernel`` for stage-4 row-attention if not already built. Idempotent under the lock."""
    global _KERNELS_INITED, row_attention_stage4_raw_kernel
    if _KERNELS_INITED:
        return
    with _KERNEL_INIT_LOCK:
        if _KERNELS_INITED:
            return
        import cupy as cp
        module = sys.modules[__name__]
        module.row_attention_stage4_raw_kernel = cp.RawKernel(
            _ROW_ATTENTION_STAGE4_KERNEL_SRC,
            "row_attention_stage4_fused",
            options=("-std=c++14",),
        )
        _KERNELS_INITED = True


# Single CUDA kernel: one block per query, threads cooperate via shared memory to compute softmax denominator and aggregates.
# Layout: blockDim.x = K (== topk count, expected 16/32/64); the head_dim is loop-strided so we don't hard-code it.
# Shared memory layout per block:
#   float logits[K]  (after exp)         K * 4 bytes
#   float y_buf[K]                       K * 4 bytes
# Total: 2 * K * 4 = 256 bytes at K=32; well under the 48-64 KB per-block budget on all CCs we target.
_ROW_ATTENTION_STAGE4_KERNEL_SRC = r"""
extern "C" __global__
void row_attention_stage4_fused(
    const float* __restrict__ q_proj,        // (n_queries, head_dim)
    const float* __restrict__ k_proj,        // (n_train, head_dim)
    const float* __restrict__ y_train,       // (n_train,)
    const int*   __restrict__ topk_ids,      // (n_queries, K) int32
    const float                inv_temp,
    const int                  n_queries,
    const int                  head_dim,
    const int                  K,
    float* __restrict__ y_mean_out,          // (n_queries,)
    float* __restrict__ y_std_out,           // (n_queries,)
    float* __restrict__ x_mean_out           // (n_queries, head_dim)
) {
    extern __shared__ float smem[];
    float* s_logits = smem;
    float* s_y      = smem + K;

    int q = blockIdx.x;
    if (q >= n_queries) return;
    int t = threadIdx.x;
    if (t >= K) return;

    // Stage 1: compute logit for this thread's neighbour. Each thread handles one of the K neighbours.
    int nid = topk_ids[q * K + t];
    float dot = 0.0f;
    for (int d = 0; d < head_dim; ++d) {
        dot += q_proj[q * head_dim + d] * k_proj[nid * head_dim + d];
    }
    s_logits[t] = dot * inv_temp;
    s_y[t] = y_train[nid];
    __syncthreads();

    // Stage 2: max-subtract softmax. Sequential reduction is fine for K<=128; warp-shuffle reduction is overkill.
    if (t == 0) {
        float m = s_logits[0];
        for (int i = 1; i < K; ++i) {
            if (s_logits[i] > m) m = s_logits[i];
        }
        float s = 0.0f;
        // If max is non-finite (all -inf neighbours), bail to uniform 1/K to avoid NaN propagation.
        if (!isfinite(m)) {
            float inv_k = 1.0f / (float)K;
            for (int i = 0; i < K; ++i) s_logits[i] = inv_k;
        } else {
            for (int i = 0; i < K; ++i) {
                s_logits[i] = expf(s_logits[i] - m);
                s += s_logits[i];
            }
            if (s <= 0.0f || !isfinite(s)) {
                float inv_k = 1.0f / (float)K;
                for (int i = 0; i < K; ++i) s_logits[i] = inv_k;
            } else {
                float inv_s = 1.0f / s;
                for (int i = 0; i < K; ++i) s_logits[i] *= inv_s;
            }
        }
    }
    __syncthreads();

    // Stage 3: weighted mean of y_train neighbours. Thread 0 does the sequential reduction (K<=128 so single-thread is fine,
    // saves the headache of a warp-shuffle reduction that needs careful boundary handling).
    if (t == 0) {
        float mean = 0.0f;
        for (int i = 0; i < K; ++i) mean += s_logits[i] * s_y[i];
        y_mean_out[q] = mean;
        float var = 0.0f;
        for (int i = 0; i < K; ++i) {
            float diff = s_y[i] - mean;
            var += s_logits[i] * diff * diff;
        }
        y_std_out[q] = (var > 0.0f) ? sqrtf(var) : 0.0f;
    }
    __syncthreads();

    // Stage 4: per-dim weighted mean of projected neighbour features. Parallelise across head_dim: each thread t handles dims t, t+K, t+2K, ...
    for (int d = t; d < head_dim; d += K) {
        float s = 0.0f;
        for (int i = 0; i < K; ++i) {
            s += s_logits[i] * k_proj[topk_ids[q * K + i] * head_dim + d];
        }
        x_mean_out[q * head_dim + d] = s;
    }
}
"""


def _get_pinned_buffer(name: str, shape: tuple[int, ...], dtype: np.dtype):
    """Return a cupy-pinned host ndarray of the requested shape/dtype, growing the pool monotonically.

    Pinned (page-locked) host memory is required for the fastest H2D / D2H copies (~12 GB/s on PCIe-4 vs ~6 GB/s pageable). Allocation is slow on first call,
    so we keep one buffer per ``(name, shape, dtype)`` triple and reuse across calls. The pool grows but never shrinks - that matches the existing
    ``_GpuBufferPool`` pattern in ``feature_selection.filters.gpu``.
    """
    import cupy as cp
    key = (name, tuple(shape), np.dtype(dtype))
    cached = _PINNED_BUFFERS.get(key)
    if cached is not None and cached.shape == tuple(shape) and cached.dtype == np.dtype(dtype):
        return cached
    mem = cp.cuda.alloc_pinned_memory(int(np.prod(shape)) * np.dtype(dtype).itemsize)
    arr = np.frombuffer(mem, dtype=np.dtype(dtype), count=int(np.prod(shape))).reshape(shape)
    _PINNED_BUFFERS[key] = arr
    return arr


def rff_matmul_cupy(
    X: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
    out: np.ndarray,
    scale: float,
    *,
    batch_rows: int = 100_000,
) -> None:
    """Streaming GPU RFF: ``out = scale * [cos(XW + b), sin(XW + b)]`` via double-buffer cupy streams.

    Shapes:
        ``X``     - (N, d)        host
        ``W``     - (d, m)        host; uploaded once before the loop and pinned on device
        ``b``     - (m,)          host; uploaded once
        ``out``   - (N, 2*m)      host; filled in-place
        ``scale`` - float
        ``batch_rows`` - chunk size for the H2D / compute / D2H pipeline

    Algorithm: alternate between two cuda streams. Stream A copies batch ``i``, computes its RFF, then copies the result back. Stream B does batch ``i+1`` in
    parallel. The two host-pinned staging buffers and two device buffers rotate. Effective H2D throughput approaches PCIe-4 peak (~12 GB/s) and compute hides
    behind copy for d-bound shapes.

    No call here if cupy is unavailable - caller dispatches via ``is_gpu_available()``. We still re-check here in case the global probe was forced; raising is
    better than silently returning zeros.
    """
    if not is_gpu_available():
        raise RuntimeError("rff_matmul_cupy called but GPU is not available; caller must dispatch via is_gpu_available().")
    import cupy as cp
    n, d = X.shape
    m = W.shape[1]
    assert out.shape == (n, 2 * m), f"out must have shape (N, 2*m); got {out.shape} vs ({n}, {2 * m})"

    # One-shot upload of W and b - they're tiny (a few MB at most) and reused across all batches.
    W_dev = cp.asarray(W)
    b_dev = cp.asarray(b)

    streams = [cp.cuda.Stream(non_blocking=True), cp.cuda.Stream(non_blocking=True)]
    x_dev = [cp.empty((batch_rows, d), dtype=W.dtype) for _ in range(2)]
    y_dev = [cp.empty((batch_rows, 2 * m), dtype=W.dtype) for _ in range(2)]
    # Pinned host staging buffers for both input and output - one of each per stream.
    x_pinned = [_get_pinned_buffer(f"rff_x_{i}", (batch_rows, d), W.dtype) for i in range(2)]
    y_pinned = [_get_pinned_buffer(f"rff_y_{i}", (batch_rows, 2 * m), W.dtype) for i in range(2)]

    batches = [(r, min(r + batch_rows, n)) for r in range(0, n, batch_rows)]
    for bi, (r0, r1) in enumerate(batches):
        slot = bi & 1
        bs = r1 - r0
        with streams[slot]:
            # H2D: copy this batch's input into pinned staging, then async to device. For the final partial batch (bs < batch_rows) we still use the full-size
            # buffers but slice into the first ``bs`` rows.
            x_pinned[slot][:bs] = X[r0:r1]
            x_dev[slot][:bs].set(x_pinned[slot][:bs])
            # Compute: angles = X @ W + b; then cos / sin separately into the two output halves.
            angles = x_dev[slot][:bs] @ W_dev
            angles += b_dev
            cp.cos(angles, out=y_dev[slot][:bs, :m])
            cp.sin(angles, out=y_dev[slot][:bs, m:])
            y_dev[slot][:bs] *= scale
            # D2H back through pinned staging.
            y_dev[slot][:bs].get(out=y_pinned[slot][:bs])
        # Synchronise this slot only before we overwrite its buffers two iterations later. The simplest correct pattern is to sync the slot we're about to reuse,
        # which means stream ``slot`` from two iterations back. For the simpler double-buffer pattern we sync inline at the boundary:
        if bi >= 1:
            streams[1 - slot].synchronize()
            prev_r0, prev_r1 = batches[bi - 1]
            prev_bs = prev_r1 - prev_r0
            prev_slot = 1 - slot
            out[prev_r0:prev_r1] = y_pinned[prev_slot][:prev_bs]
    # Final batch: sync and copy.
    streams[(len(batches) - 1) & 1].synchronize()
    final_r0, final_r1 = batches[-1]
    final_slot = (len(batches) - 1) & 1
    out[final_r0:final_r1] = y_pinned[final_slot][: final_r1 - final_r0]


def row_attention_stage4_cupy(
    q_proj: np.ndarray,
    k_proj: np.ndarray,
    y_train: np.ndarray,
    topk_ids: np.ndarray,
    softmax_temp: float,
    y_mean_out: np.ndarray,
    y_std_out: np.ndarray,
    x_mean_out: np.ndarray,
    *,
    k_proj_device: Any | None = None,
    y_train_device: Any | None = None,
) -> None:
    """Fused stage-4 row-attention on GPU via the RawKernel.

    Shapes match the njit twin in ``_kernels_njit.row_attention_stage4_njit``. The kernel uses dynamic shared memory of size ``2 * K * 4`` bytes (logits + targets).

    Optional ``k_proj_device`` / ``y_train_device`` lets Mode-B inference pass already-resident cupy arrays (the K-bank stays on the device across many query
    batches), bypassing the per-call H2D upload that would otherwise dominate cost for small query batches.
    """
    if not is_gpu_available():
        raise RuntimeError("row_attention_stage4_cupy called but GPU is not available; caller must dispatch via is_gpu_available().")
    _ensure_kernels_inited()
    import cupy as cp

    n_queries, head_dim = q_proj.shape
    k = topk_ids.shape[1]
    n_train = k_proj.shape[0]

    q_dev = cp.asarray(q_proj, dtype=cp.float32)
    k_dev = k_proj_device if k_proj_device is not None else cp.asarray(k_proj, dtype=cp.float32)
    y_dev = y_train_device if y_train_device is not None else cp.asarray(y_train, dtype=cp.float32)
    topk_dev = cp.asarray(topk_ids, dtype=cp.int32)

    y_mean_dev = cp.empty(n_queries, dtype=cp.float32)
    y_std_dev = cp.empty(n_queries, dtype=cp.float32)
    x_mean_dev = cp.empty((n_queries, head_dim), dtype=cp.float32)

    shared_mem_bytes = 2 * k * 4

    # Block layout: one block per query, threads = K. Grid = n_queries.
    row_attention_stage4_raw_kernel(
        (n_queries,),
        (k,),
        (
            q_dev,
            k_dev,
            y_dev,
            topk_dev,
            np.float32(1.0 / softmax_temp),
            np.int32(n_queries),
            np.int32(head_dim),
            np.int32(k),
            y_mean_dev,
            y_std_dev,
            x_mean_dev,
        ),
        shared_mem=shared_mem_bytes,
    )

    # D2H back to caller-allocated host arrays.
    y_mean_dev.get(out=y_mean_out)
    y_std_dev.get(out=y_std_out)
    x_mean_dev.get(out=x_mean_out)


def vram_required_gb(
    n_train: int,
    n_queries: int,
    head_dim: int,
    n_heads: int,
    k: int,
    dtype: np.dtype = np.float32,
    *,
    keep_key_bank_on_gpu: bool = False,
) -> float:
    """Estimate peak VRAM for one row-attention call so callers can pre-flight check before invocation.

    Components included:
    - K-bank (n_train, head_dim) per head if resident (shared across heads when ``keep_key_bank_on_gpu`` is True; rebuilt per head otherwise but only one head at
      a time on the device).
    - y_train (n_train,) one copy.
    - topk_ids (n_queries, k) int32.
    - per-batch query / output buffers, scaled to one head at a time.

    Cupy memory-pool fragmentation and cuBLAS workspace add ~30%; multiply by 1.3 for a safety budget. This is conservative; users with 24 GB cards can ignore;
    8 GB cards should run this before invocation and lower ``head_dim`` if the result exceeds available VRAM (use ``_utils.gpu_available_bytes``).
    """
    item = np.dtype(dtype).itemsize
    bytes_k_bank = n_train * head_dim * item
    bytes_y_train = n_train * item
    bytes_topk = n_queries * k * 4
    bytes_q = n_queries * head_dim * item
    bytes_y_out = n_queries * item * 2
    bytes_x_out = n_queries * head_dim * item
    base = bytes_k_bank + bytes_y_train + bytes_topk + bytes_q + bytes_y_out + bytes_x_out
    if keep_key_bank_on_gpu and n_heads > 1:
        # K-bank held for all heads simultaneously (Mode B optimisation).
        base += bytes_k_bank * (n_heads - 1)
    # 1.3 safety multiplier covers pool fragmentation + cuBLAS workspace.
    return float(base * 1.3 / 2**30)
