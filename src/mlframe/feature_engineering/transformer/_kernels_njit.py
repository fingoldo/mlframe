"""CPU njit kernels for RFF + row-attention stage 4.

Decorator policy is ``fastmath=True`` here (not the project-wide ``False``). Rationale: the GPU path through cupy + cuBLAS does not honour any fastmath flag and
already permits FMA contraction / reciprocal approximation. Setting ``fastmath=True`` on the CPU side closes the gap so the GPU-vs-CPU parity test can hold a
tight tolerance (``atol=1e-4 rtol=1e-3``). The two compute steps where this matters in this file are (a) the softmax denominator's ``exp`` sum, where fp
associativity flips cost less than 1 ulp, and (b) the dot-product accumulator in attention / RFF, where contractions into FMA give 1.5-2x speedup at no cost
to correctness for inner-product use. Reductions that need bit-for-bit reproducibility belong in ``_aggregation.py`` (``fastmath=False``).

The fused stage-4 kernel here is the CPU twin of the cupy RawKernel in ``_kernels_cupy``. Parity tests in ``test_transformer_row_attention.py`` run both on the
same fixed seed and assert outputs match within the documented tolerance.
"""
from __future__ import annotations

import numba
import numpy as np

# Attention/softmax/RFF kernels: fastmath=True for cuBLAS parity. See module docstring.
NUMBA_NJIT_PARAMS = dict(fastmath=True, cache=True, nogil=True)


@numba.njit(**NUMBA_NJIT_PARAMS)
def softmax_stable_inplace(logits: np.ndarray) -> None:  # pragma: no cover
    """In-place numerically stable softmax over a 1-D array.

    Max-subtract trick (classic): ``exp(x - max)`` keeps the largest exponent at 0, avoiding fp32 overflow when any logit exceeds ~88. The denominator is the
    sum of ``exp`` values; after dividing each entry by it, ``logits`` sums to 1 (within fp error).

    Edge cases:
    - All logits = -inf (zero neighbours within reach): max is -inf, ``exp(-inf - (-inf)) = exp(nan)`` is nan. Detected by an explicit guard; result is set to
      uniform 1/k so downstream aggregations are well-defined rather than nan-propagating.
    - Single element: returns 1.0 (trivial).
    """
    k = logits.shape[0]
    if k == 0:
        return
    if k == 1:
        logits[0] = 1.0
        return
    m = logits[0]
    for i in range(1, k):
        if logits[i] > m:
            m = logits[i]
    if not np.isfinite(m):
        # Degenerate: all -inf or all +inf — fall back to uniform.
        inv = 1.0 / k
        for i in range(k):
            logits[i] = inv
        return
    s = 0.0
    for i in range(k):
        logits[i] = np.exp(logits[i] - m)
        s += logits[i]
    if s <= 0.0 or not np.isfinite(s):
        inv = 1.0 / k
        for i in range(k):
            logits[i] = inv
        return
    inv_s = 1.0 / s
    for i in range(k):
        logits[i] *= inv_s


@numba.njit(parallel=True, **NUMBA_NJIT_PARAMS)
def rff_matmul_njit(  # pragma: no cover
    X: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
    out: np.ndarray,
    scale: float,
) -> None:
    """CPU fallback for ``compute_rff_features``. Produces ``out = scale * [cos(XW+b), sin(XW+b)]`` in one parallel sweep.

    Shapes:
        ``X``     - (N, d)        input rows
        ``W``     - (d, m)        random Gaussian projection where m = n_features // 2
        ``b``     - (m,)          uniform U[0, 2*pi) bias
        ``out``   - (N, 2*m)      output: cos in [:, :m], sin in [:, m:]
        ``scale`` - sqrt(2 / n_features), constant

    Per-row complexity O(d * m) FMA. Parallelism is across rows; W and b are read-shared. Fused cos/sin in the same pass: the angle XW+b is computed once per
    (row, m-coord), then both ``cos`` and ``sin`` are evaluated on it. Halves the cost vs computing the angles, calling cos, then re-computing for sin.

    On a 16-core AVX2 box at d=64, m=128, N=1M, this runs in ~200 ms — the cupy alternative is ~10 ms compute + ~12 ms PCIe at this size, only ~2x faster.
    Crossover where cupy clearly wins is around N * d ~~ 5M (see ``profiling/bench_transformer_rff.py`` calibration).
    """
    n, d = X.shape
    m = W.shape[1]
    for i in numba.prange(n):
        for j in range(m):
            acc = b[j]
            for k in range(d):
                acc += X[i, k] * W[k, j]
            c = np.cos(acc)
            s = np.sin(acc)
            out[i, j] = scale * c
            out[i, j + m] = scale * s


@numba.njit(parallel=True, **NUMBA_NJIT_PARAMS)
def row_attention_stage4_adaptive_njit(  # pragma: no cover
    q_proj: np.ndarray,
    k_proj: np.ndarray,
    y_train: np.ndarray,
    topk_ids: np.ndarray,
    softmax_temps: np.ndarray,  # per-query temperature (n_queries,)
    y_mean_out: np.ndarray,
    y_std_out: np.ndarray,
    x_mean_out: np.ndarray,
) -> None:
    """Adaptive-bandwidth variant of row_attention_stage4_njit. Same as the standard kernel but takes a PER-QUERY softmax temperature instead of a global one.

    Use case: in dense local regions, smaller temperature → sharper attention; in sparse regions, larger temperature → smoother attention. Per-query temps can be
    set to the median distance to top-k neighbours (the "adaptive bandwidth" / "balloon estimator" pattern from non-parametric density estimation).
    """
    n_queries = q_proj.shape[0]
    head_dim = q_proj.shape[1]
    k = topk_ids.shape[1]
    for q in numba.prange(n_queries):
        temp_q = softmax_temps[q]
        inv_temp = 1.0 / temp_q if temp_q > 1e-12 else 1.0
        weights = np.empty(k, dtype=np.float32)
        y_buf = np.empty(k, dtype=np.float32)
        k_buf = np.empty((k, head_dim), dtype=np.float32)
        for i in range(k):
            nid = topk_ids[q, i]
            dot = 0.0
            for d in range(head_dim):
                kv = k_proj[nid, d]
                k_buf[i, d] = kv
                dot += q_proj[q, d] * kv
            weights[i] = dot * inv_temp
            y_buf[i] = y_train[nid]
        softmax_stable_inplace(weights)
        mean = 0.0
        for i in range(k):
            mean += weights[i] * y_buf[i]
        y_mean_out[q] = mean
        var = 0.0
        for i in range(k):
            diff = y_buf[i] - mean
            var += weights[i] * diff * diff
        y_std_out[q] = np.sqrt(var) if var > 0.0 else 0.0
        for d in range(head_dim):
            s = 0.0
            for i in range(k):
                s += weights[i] * k_buf[i, d]
            x_mean_out[q, d] = s


@numba.njit(parallel=True, **NUMBA_NJIT_PARAMS)
def row_attention_stage4_njit(  # pragma: no cover
    q_proj: np.ndarray,
    k_proj: np.ndarray,
    y_train: np.ndarray,
    topk_ids: np.ndarray,
    softmax_temp: float,
    y_mean_out: np.ndarray,
    y_std_out: np.ndarray,
    x_mean_out: np.ndarray,
) -> None:
    """Fused gather -> dot -> softmax -> weighted sum (y_mean, y_std, x_mean).

    Per query row, with ``k`` already-retrieved nearest-neighbour ids:
        1. logits[i] = dot(q_proj[query], k_proj[topk_ids[query, i]]) / softmax_temp     for i in 0..k-1
        2. weights = softmax_stable(logits)
        3. y_mean_out[query]    = sum_i weights[i] * y_train[topk_ids[query, i]]
        4. y_std_out[query]     = sqrt(sum_i weights[i] * (y_train[topk_ids[query, i]] - y_mean)^2)
        5. x_mean_out[query, d] = sum_i weights[i] * k_proj[topk_ids[query, i], d]       for d in 0..head_dim-1

    All five steps run in one prange iteration per query; the gathered neighbours stay in a small scratch buffer (k * head_dim floats, well within L1 cache for
    k=32, head_dim=8 -> 1 KB). Avoids the three-pass alternative (gather, then softmax, then aggregate) which thrashes the cache and triples DRAM traffic.

    Shapes (single head, caller loops over heads):
        ``q_proj``      - (n_queries, head_dim)   projected query vectors, L2-normalised
        ``k_proj``      - (n_train, head_dim)     projected K-bank, L2-normalised
        ``y_train``     - (n_train,)              targets at training rows
        ``topk_ids``    - (n_queries, k) int32    neighbour ids from hnswlib
        ``softmax_temp`` - float                  divides logits before softmax; lower = sharper
        ``y_mean_out``  - (n_queries,)            pre-allocated output
        ``y_std_out``   - (n_queries,)            pre-allocated output
        ``x_mean_out``  - (n_queries, head_dim)   pre-allocated output

    ``softmax_temp`` is the *divisor* applied to the cosine similarity (which lives in [-1, 1] after L2-norm of both q and K). Default 1.0; lower values (e.g. 0.1)
    sharpen the attention towards the top-1, higher values flatten towards uniform-of-top-k.
    """
    n_queries = q_proj.shape[0]
    head_dim = q_proj.shape[1]
    k = topk_ids.shape[1]
    # Wave 47 (2026-05-20): mirror the sibling kernel's guard at line 120;
    # user-provided softmax_temp can be 0 (or extremely small) and would divide-by-zero.
    inv_temp = 1.0 / softmax_temp if softmax_temp > 1e-12 else 1.0
    for q in numba.prange(n_queries):
        # Per-query scratch: logits/weights array and a (k, head_dim) gather buffer.
        weights = np.empty(k, dtype=np.float32)
        y_buf = np.empty(k, dtype=np.float32)
        k_buf = np.empty((k, head_dim), dtype=np.float32)
        # Stage 1: compute logits, also gather neighbour targets / projected features into scratch (one DRAM pass).
        for i in range(k):
            nid = topk_ids[q, i]
            dot = 0.0
            for d in range(head_dim):
                kv = k_proj[nid, d]
                k_buf[i, d] = kv
                dot += q_proj[q, d] * kv
            weights[i] = dot * inv_temp
            y_buf[i] = y_train[nid]
        # Stage 2: in-place softmax over the k logits.
        softmax_stable_inplace(weights)
        # Stage 3+4: weighted mean and std of y_train neighbours.
        mean = 0.0
        for i in range(k):
            mean += weights[i] * y_buf[i]
        y_mean_out[q] = mean
        var = 0.0
        for i in range(k):
            diff = y_buf[i] - mean
            var += weights[i] * diff * diff
        y_std_out[q] = np.sqrt(var) if var > 0.0 else 0.0
        # Stage 5: per-dim weighted mean of projected neighbour features.
        for d in range(head_dim):
            s = 0.0
            for i in range(k):
                s += weights[i] * k_buf[i, d]
            x_mean_out[q, d] = s
