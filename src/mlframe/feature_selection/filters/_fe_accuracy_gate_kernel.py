"""Fused logistic-ridge probe kernel: one CUDA block solves ONE (design, fold) of the accuracy gate, standardisation, Newton fit, held-out scores and the fold's ROC-AUC inside the block.

Win axis: many small designs over the SAME rows, which is exactly what the gate's pass 1 asks for (a candidate's design is its raw source plus the engineered column, and every candidate sees the
same subsample rows and the same folds, see ``_fe_accuracy_gate_batched``). One launch covers every candidate on every fold. It does NOT win for a handful of designs (the host probes those on a
thread pool at about 26 ms each) and it does not apply above ``MAX_K`` design columns or when a held-out fold has more than ``MAX_VA`` rows; the caller keeps the host probe for those.

Design, against the rules of CUDA.md section 3:

* one block per work item: block ``(d, f)`` standardises, fits design ``d`` on the rows outside fold ``f``, scores the held-out rows and reduces them to that fold's ROC-AUC, so the only thing
  that leaves the device is one number per (design, fold) and no intermediate table is written (a separate score kernel would have needed a ``(designs, folds, rows)`` buffer: 320 MB at 800
  designs, 10 folds and 5000 rows);
* the design matrix is FEATURE-MAJOR ``(n_designs, k, n)``, so the threads of a warp read consecutive rows of one column (stride 1); a row-major ``(n_designs, n, k)`` layout would stride by ``k``;
* the fold is a mask, not a gather: every block walks all ``n`` rows in order and takes a row as training when ``fold_of[row] != fold``, which keeps the reads coalesced where an index gather
  would not be, and the same walk scores the held-out rows;
* reductions are warp shuffles into a per-warp shared slot, then one thread folds the warps, with a barrier between the write and the read; the fold order is fixed, so a rerun gives the same bits;
* the held-out AUC is the Mann-Whitney count ``(#{s+ > s-} + 0.5 * #{s+ == s-}) / (n+ * n-)``, which is the mid-rank quantity ``fast_roc_auc`` returns, counted over the fold's scores held in
  shared memory;
* ``{REAL}`` is the type of the values read (the crit dtype) and ``{ACC}`` the accumulator of every sum and of the objective, so a float32 project dtype still sums in double;
* the standardised values round through ``{REAL}`` exactly as the host probe's cast to the frame dtype does, and the zero-spread rule is the host's (a deviation at or below
  ``10 * eps * max(|mean|, 1)`` becomes 1.0);
* block size 128 (four warps, a power of two for the tree fold of the warp slots); shared memory is fixed and small: the moments, the gradient, the Hessian, the step, the weights, the warp
  slots and the held-out scores.

The Newton iteration mirrors ``_fe_accuracy_gate_newton.fit_logistic_ridge_newton_numpy`` term for term: the gradient-norm stop test ``max|grad| < grad_tol * sqrt(max(1, n_train))``,
``reg + 1e-12`` on the Hessian diagonal, the intercept unpenalised, and a backtracking line search of at most twelve halvings on the penalised loss. A design that does not converge writes NaN,
which the caller reads as "probe this candidate on the host", the same way the host path falls through to the scikit-learn fit when Newton returns ``None``.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import numpy as np

from ._kernel_types import render

logger = logging.getLogger(__name__)

#: Design columns the kernel covers (without the intercept); the Hessian is ``(k+1)^2`` shared accumulators, so this bounds the shared block.
MAX_K = 8
#: Held-out rows a block reduces in shared memory (a 5000-row subsample over at least five folds leaves at most 1000).
MAX_VA = 1024
#: Threads per block: four warps, a power of two for the tree fold of the warp slots (CUDA.md section 3).
BLOCK = 128
_WARPS = BLOCK // 32

_SRC_TEMPLATE = r"""
#define MAXK __MAXK__
#define MAXP (MAXK + 1)
#define WARPS __WARPS__
#define MAXVA __MAXVA__
#define NTHREADS __BLOCK__

__device__ __forceinline__ {ACC} warp_sum({ACC} v) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
    return v;
}

// Block-wide sum of ``v``; every thread leaves with the total. One warp slot per warp, one thread folds the slots, the total is broadcast through shared memory.
__device__ __forceinline__ {ACC} block_sum({ACC} v, {ACC}* slots) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    v = warp_sum(v);
    if (lane == 0) slots[warp] = v;
    __syncthreads();
    if (threadIdx.x == 0) {
        {ACC} t = 0.0;
        for (int w = 0; w < WARPS; ++w) t += slots[w];
        slots[WARPS] = t;
    }
    __syncthreads();
    {ACC} total = slots[WARPS];
    __syncthreads();
    return total;
}

extern "C" __global__
void gate_logistic_auc(
    const {REAL}* __restrict__ designs,
    const {REAL}* __restrict__ yv,
    const signed char* __restrict__ fold_of,
    const int* __restrict__ eval_folds,
    const long long n, const int k, const int n_designs, const int n_folds,
    const {ACC} lam, const {ACC} grad_tol, const int max_iter, const int round_real,
    {ACC}* __restrict__ auc_out
) {
    const int d = blockIdx.x, fi = blockIdx.y;
    if (d >= n_designs || fi >= n_folds) return;
    const int fold = eval_folds[fi];
    const int p = k + 1;
    const size_t out_at = (size_t)d * n_folds + fi;

    __shared__ {ACC} mean_s[MAXK], sd_s[MAXK], grad_s[MAXP], hess_s[MAXP * MAXP], step_s[MAXP], w_s[MAXP];
    __shared__ {ACC} slots[WARPS + 1], scal_s[4];
    __shared__ {ACC} va_score[MAXVA];
    __shared__ signed char va_label[MAXVA];
    __shared__ int va_n;

    const {REAL}* col0 = designs + (size_t)d * k * n;

    {ACC} cnt = 0.0;
    for (long long i = threadIdx.x; i < n; i += NTHREADS) cnt += (fold_of[i] != fold) ? 1.0 : 0.0;
    const {ACC} n_tr = block_sum(cnt, slots);
    if (n_tr < 2.0 || (n - (long long)n_tr) > MAXVA) {
        if (threadIdx.x == 0) auc_out[out_at] = nan("");
        return;
    }
    for (int j = 0; j < k; ++j) {
        {ACC} s = 0.0;
        const {REAL}* cj = col0 + (size_t)j * n;
        for (long long i = threadIdx.x; i < n; i += NTHREADS) s += (fold_of[i] != fold) ? ({ACC})cj[i] : 0.0;
        const {ACC} tot = block_sum(s, slots);
        if (threadIdx.x == 0) mean_s[j] = tot / n_tr;
        __syncthreads();
    }
    for (int j = 0; j < k; ++j) {
        const {ACC} mu = mean_s[j];
        {ACC} s = 0.0;
        const {REAL}* cj = col0 + (size_t)j * n;
        for (long long i = threadIdx.x; i < n; i += NTHREADS) {
            if (fold_of[i] != fold) { const {ACC} dv = ({ACC})cj[i] - mu; s += dv * dv; }
        }
        const {ACC} tot = block_sum(s, slots);
        if (threadIdx.x == 0) {
            const {ACC} sd = sqrt(tot / n_tr);
            const {ACC} spread_floor = 10.0 * 2.220446049250313e-16 * fmax(fabs(mu), 1.0);
            sd_s[j] = (sd > spread_floor) ? sd : 1.0;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) { for (int j = 0; j < p; ++j) w_s[j] = 0.0; }
    __syncthreads();

    {ACC} f_cur;
    {
        {ACC} s = 0.0;
        for (long long i = threadIdx.x; i < n; i += NTHREADS) if (fold_of[i] != fold) s += log(2.0);
        f_cur = block_sum(s, slots);
    }

    int converged = 0;
    for (int it = 0; it < max_iter; ++it) {
        {ACC} g[MAXP], h[MAXP * MAXP];
        for (int a = 0; a < p; ++a) { g[a] = 0.0; for (int b = 0; b < p; ++b) h[a * p + b] = 0.0; }
        for (long long i = threadIdx.x; i < n; i += NTHREADS) {
            if (fold_of[i] == fold) continue;
            {ACC} x[MAXP];
            {ACC} z = 0.0;
            for (int j = 0; j < k; ++j) {
                {ACC} v = (({ACC})col0[(size_t)j * n + i] - mean_s[j]) / sd_s[j];
                if (round_real) v = ({ACC})({REAL})v;
                x[j] = v;
                z += v * w_s[j];
            }
            x[k] = 1.0;
            z += w_s[k];
            const {ACC} mu = 1.0 / (1.0 + exp(-z));
            const {ACC} r = mu - ({ACC})yv[i];
            const {ACC} sw = mu * (1.0 - mu);
            for (int a = 0; a < p; ++a) {
                g[a] += x[a] * r;
                for (int b = 0; b <= a; ++b) h[a * p + b] += x[a] * x[b] * sw;
            }
        }
        for (int a = 0; a < p; ++a) {
            const {ACC} tot = block_sum(g[a], slots);
            if (threadIdx.x == 0) grad_s[a] = tot + ((a < k) ? lam * w_s[a] : 0.0);
            __syncthreads();
        }
        for (int a = 0; a < p; ++a) {
            for (int b = 0; b <= a; ++b) {
                const {ACC} tot = block_sum(h[a * p + b], slots);
                if (threadIdx.x == 0) {
                    {ACC} v = tot;
                    if (a == b) v += ((a < k) ? lam : 0.0) + 1e-12;
                    hess_s[a * p + b] = v;
                    hess_s[b * p + a] = v;
                }
                __syncthreads();
            }
        }
        if (threadIdx.x == 0) {
            {ACC} m = 0.0;
            for (int a = 0; a < p; ++a) m = fmax(m, fabs(grad_s[a]));
            scal_s[0] = m;
        }
        __syncthreads();
        if (scal_s[0] < grad_tol * sqrt(fmax(1.0, n_tr))) { converged = 1; break; }

        if (threadIdx.x == 0) {
            {ACC} L[MAXP * MAXP], t1[MAXP];
            int bad = 0;
            for (int a = 0; a < p; ++a) {
                for (int b = 0; b <= a; ++b) {
                    {ACC} s = hess_s[a * p + b];
                    for (int c = 0; c < b; ++c) s -= L[a * p + c] * L[b * p + c];
                    if (a == b) {
                        if (!(s > 0.0)) { bad = 1; s = 1.0; }
                        L[a * p + a] = sqrt(s);
                    } else {
                        L[a * p + b] = s / L[b * p + b];
                    }
                }
            }
            for (int a = 0; a < p; ++a) {
                {ACC} s = grad_s[a];
                for (int c = 0; c < a; ++c) s -= L[a * p + c] * t1[c];
                t1[a] = s / L[a * p + a];
            }
            for (int a = p - 1; a >= 0; --a) {
                {ACC} s = t1[a];
                for (int c = a + 1; c < p; ++c) s -= L[c * p + a] * step_s[c];
                step_s[a] = s / L[a * p + a];
            }
            scal_s[1] = bad ? 1.0 : 0.0;
        }
        __syncthreads();
        if (scal_s[1] != 0.0) break;

        {ACC} t = 1.0;
        int accepted = 0;
        for (int ls = 0; ls < 12; ++ls) {
            {ACC} s = 0.0;
            for (long long i = threadIdx.x; i < n; i += NTHREADS) {
                if (fold_of[i] == fold) continue;
                {ACC} z = 0.0;
                for (int j = 0; j < k; ++j) {
                    {ACC} v = (({ACC})col0[(size_t)j * n + i] - mean_s[j]) / sd_s[j];
                    if (round_real) v = ({ACC})({REAL})v;
                    z += v * (w_s[j] - t * step_s[j]);
                }
                z += w_s[k] - t * step_s[k];
                s += ((z > 0.0) ? (z + log1p(exp(-z))) : log1p(exp(z))) - ({ACC})yv[i] * z;
            }
            const {ACC} loss = block_sum(s, slots);
            if (threadIdx.x == 0) {
                {ACC} pen = 0.0;
                for (int j = 0; j < k; ++j) { const {ACC} wj = w_s[j] - t * step_s[j]; pen += wj * wj; }
                scal_s[2] = loss + 0.5 * lam * pen;
            }
            __syncthreads();
            const {ACC} f_new = scal_s[2];
            if (isfinite(f_new) && f_new <= f_cur + 1e-11 * fabs(f_cur)) { accepted = 1; f_cur = f_new; break; }
            t *= 0.5;
        }
        if (!accepted) break;
        if (threadIdx.x == 0) { for (int a = 0; a < p; ++a) w_s[a] -= t * step_s[a]; }
        __syncthreads();
    }

    int finite = 1;
    for (int a = 0; a < p; ++a) if (!isfinite(w_s[a])) finite = 0;
    if (!converged || !finite) {
        if (threadIdx.x == 0) auc_out[out_at] = nan("");
        return;
    }

    if (threadIdx.x == 0) {
        int m = 0;
        for (long long i = 0; i < n; ++i) {
            if (fold_of[i] != fold) continue;
            {ACC} z = w_s[k];
            for (int j = 0; j < k; ++j) {
                {ACC} v = (({ACC})col0[(size_t)j * n + i] - mean_s[j]) / sd_s[j];
                if (round_real) v = ({ACC})({REAL})v;
                z += v * w_s[j];
            }
            va_score[m] = 1.0 / (1.0 + exp(-z));
            va_label[m] = (({ACC})yv[i] > 0.5) ? 1 : 0;
            ++m;
        }
        va_n = m;
    }
    __syncthreads();

    const int m = va_n;
    {ACC} u = 0.0, npos = 0.0;
    for (int a = threadIdx.x; a < m; a += NTHREADS) {
        if (!va_label[a]) continue;
        npos += 1.0;
        const {ACC} sa = va_score[a];
        {ACC} acc = 0.0;
        for (int b = 0; b < m; ++b) {
            if (va_label[b]) continue;
            const {ACC} sb = va_score[b];
            acc += (sa > sb) ? 1.0 : ((sa == sb) ? 0.5 : 0.0);
        }
        u += acc;
    }
    const {ACC} u_tot = block_sum(u, slots);
    const {ACC} pos_tot = block_sum(npos, slots);
    if (threadIdx.x == 0) {
        const {ACC} neg_tot = ({ACC})m - pos_tot;
        auc_out[out_at] = (pos_tot > 0.0 && neg_tot > 0.0) ? (u_tot / (pos_tot * neg_tot)) : 0.5;
    }
}
"""

_MODULES: dict = {}
_LOCK = threading.Lock()


def kernel(cp: Any, real: Any) -> Any:
    """The fused probe kernel compiled for the value type ``real`` (one module per type, built once and cached)."""
    key = np.dtype(real).str
    with _LOCK:
        mod = _MODULES.get(key)
        if mod is None:
            src = render(_SRC_TEMPLATE, real=real)
            src = src.replace("__MAXK__", str(MAX_K)).replace("__WARPS__", str(_WARPS)).replace("__MAXVA__", str(MAX_VA)).replace("__BLOCK__", str(BLOCK))
            mod = _MODULES[key] = cp.RawModule(code=src, options=("-std=c++11",))
        return mod.get_function("gate_logistic_auc")


__all__ = ["BLOCK", "MAX_K", "MAX_VA", "kernel"]
