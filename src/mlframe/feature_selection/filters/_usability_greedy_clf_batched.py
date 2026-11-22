"""Cross-validated logloss of every shortlisted candidate from one batched Newton fit per fold.

``_usability_greedy_clf_gpu_resident._cv_candidate`` refits the logistic model of ``selected + [candidate]`` on every fold, one candidate at a time: a round over a shortlist of 40 with 4 folds is 160 small
fits, each a few Newton iterations of tiny launches ending in a host sync. The fits are independent, so this module stacks the candidates of one fold into a ``(candidates, train rows, k + 1)`` design and runs
the Newton iteration on all of them at once with batched matmuls and one batched solve, with one host sync per iteration for the whole batch.

Each member follows the single-fit recipe operation by operation: train-fold standardization (population std, columns below ``1e-12`` held at scale 1), unpenalised intercept, L2 penalty ``1 / C``, and the stop
``max|step| < tol`` evaluated per member (a member that has met its stop is frozen, as the single fit returns at that point). A non-finite weight or a missing convergence raises the caller's fallback error.
"""

from __future__ import annotations

import threading

from typing import Any

import numpy as np

_BATCH_ELEMENT_BUDGET = 48_000_000  # design elements per chunk (three live copies of the chunk design at 4 bytes each stay near 600 MB)


_FUSED_SRC = r"""
#define MAXKK 40
// dtype-ok: Intentional float64, reviewed 2026-11-20: the Newton scratch (standardized columns, weights, residuals) and the log-loss output are double scratch buffers the host allocates as float64 explicitly (the fit solves a k x k system and is compared with the host fit at 1e-9).
extern "C" __global__ void logistic_cv_newton(const VT_* __restrict__ V, const long long ldv, const long long* __restrict__ tr, const int ntr, const long long* __restrict__ va, const int nva,
                                              const int* __restrict__ sel, const int s, const int* __restrict__ cand, const int cand_off, const long long* __restrict__ yenc,
                                              const double lam, const double tol, const int max_iter, const double eps,
                                              double* __restrict__ Xs, double* __restrict__ wd_scr, double* __restrict__ r_scr,
                                              double* __restrict__ out_ll, int* __restrict__ out_status, const int out_off) {
    // one block per candidate: standardize the train columns, run the L2 Newton iteration of the logistic fit to its stop, score the validation fold; every step stays inside the block
    extern __shared__ double sh[];
    const int kk = s + 2;                       // s selected columns, the candidate, the intercept
    double* H = sh;                              // kk * kk
    double* g = H + kk * kk;                     // kk
    double* w = g + kk;                          // kk
    double* mu = w + kk;                         // s + 1
    double* sd = mu + (s + 1);                   // s + 1
    double* red = sd + (s + 1);                  // blockDim.x
    __shared__ int flag;                         // 0 running, 1 converged, 2 non-finite
    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    const int nthr = blockDim.x;
    double* X = Xs + (long long)b * ntr * kk;
    double* wd = wd_scr + (long long)b * ntr;
    double* r = r_scr + (long long)b * ntr;
    const int ccol = cand[cand_off + b];
    // column means and population standard deviations over the train rows (below 1e-12 -> scale 1)
    for (int j = 0; j <= s; ++j) {
        const int col = (j < s) ? sel[j] : ccol;
        const long long vo = (long long)col * ldv;   // offset of this column in the column-major V
        double a = 0.0;
        for (int i = tid; i < ntr; i += nthr) a += V[vo + tr[i]];
        red[tid] = a; __syncthreads();
        for (int st = nthr >> 1; st > 0; st >>= 1) { if (tid < st) red[tid] += red[tid + st]; __syncthreads(); }
        const double m = red[0] / ntr; __syncthreads();
        double q = 0.0;
        for (int i = tid; i < ntr; i += nthr) { const double d = V[vo + tr[i]] - m; q += d * d; }
        red[tid] = q; __syncthreads();
        for (int st = nthr >> 1; st > 0; st >>= 1) { if (tid < st) red[tid] += red[tid + st]; __syncthreads(); }
        if (tid == 0) { double sdv = sqrt(red[0] / ntr); mu[j] = m; sd[j] = (sdv < 1e-12) ? 1.0 : sdv; }
        __syncthreads();
    }
    // X is column-major (column j occupies X[j * ntr .. j * ntr + ntr)) so that threads reading one column read consecutive rows
    for (int i = tid; i < ntr; i += nthr) {
        for (int j = 0; j <= s; ++j) { const int col = (j < s) ? sel[j] : ccol; const long long vo = (long long)col * ldv; X[(long long)j * ntr + i] = (V[vo + tr[i]] - mu[j]) / sd[j]; }
        X[(long long)(s + 1) * ntr + i] = 1.0;
    }
    for (int k = tid; k < kk; k += nthr) w[k] = 0.0;
    if (tid == 0) flag = 0;
    __syncthreads();
    int status = 2;                              // 2: no convergence within max_iter
    for (int it = 0; it < max_iter; ++it) {
        for (int i = tid; i < ntr; i += nthr) {
            double z = 0.0;
            for (int j = 0; j < kk; ++j) z += X[(long long)j * ntr + i] * w[j];
            const double p = 1.0 / (1.0 + exp(-z));
            wd[i] = p * (1.0 - p);
            r[i] = p - ((yenc[tr[i]] == 1) ? 1.0 : 0.0);
        }
        __syncthreads();
        // H and g entries one at a time: the whole block sweeps the train rows of one column pair (or column with r) with a strided loop
        // and tree-reduces the partial sums, so the row index is the thread-derived one and the column index is uniform across the block
        const int npair = kk * (kk + 1) / 2;
        for (int e = 0; e < npair + kk; ++e) {
            const bool isg = (e >= npair);
            int j = 0, l = 0;
            double acc = 0.0;
            if (!isg) {
                int rem = e;
                while (rem > j) { rem -= j + 1; ++j; }   // e = j (j + 1) / 2 + l, l <= j
                l = rem;
                for (int i = tid; i < ntr; i += nthr) acc += wd[i] * X[(long long)j * ntr + i] * X[(long long)l * ntr + i];
            } else {
                j = e - npair;
                for (int i = tid; i < ntr; i += nthr) acc += r[i] * X[(long long)j * ntr + i];
            }
            red[tid] = acc; __syncthreads();
            for (int st = nthr >> 1; st > 0; st >>= 1) { if (tid < st) red[tid] += red[tid + st]; __syncthreads(); }
            if (tid == 0) {
                if (!isg) { H[j * kk + l] = red[0]; H[l * kk + j] = red[0]; }
                else g[j] = red[0] + ((j < kk - 1) ? lam * w[j] : 0.0);
            }
            __syncthreads();
        }
        if (tid == 0) {
            for (int j = 0; j < kk - 1; ++j) H[j * kk + j] += lam;
            // Gauss elimination with partial pivoting on [H | g]; the step lands in g
            int bad = 0;
            for (int c = 0; c < kk && !bad; ++c) {
                int piv = c; double best = fabs(H[c * kk + c]);
                for (int rr = c + 1; rr < kk; ++rr) { const double v = fabs(H[rr * kk + c]); if (v > best) { best = v; piv = rr; } }
                if (!(best > 1e-300)) { bad = 1; break; }
                if (piv != c) { for (int cc = 0; cc < kk; ++cc) { const double t = H[c * kk + cc]; H[c * kk + cc] = H[piv * kk + cc]; H[piv * kk + cc] = t; } const double t = g[c]; g[c] = g[piv]; g[piv] = t; }
                for (int rr = c + 1; rr < kk; ++rr) {
                    const double f = H[rr * kk + c] / H[c * kk + c];
                    for (int cc = c; cc < kk; ++cc) H[rr * kk + cc] -= f * H[c * kk + cc];
                    g[rr] -= f * g[c];
                }
            }
            double maxstep = 0.0;
            if (!bad) {
                for (int c = kk - 1; c >= 0; --c) {
                    double acc = g[c];
                    for (int cc = c + 1; cc < kk; ++cc) acc -= H[c * kk + cc] * g[cc];
                    g[c] = acc / H[c * kk + c];
                }
                for (int j = 0; j < kk; ++j) { w[j] -= g[j]; maxstep = fmax(maxstep, fabs(g[j])); if (!isfinite(w[j])) bad = 1; }
            }
            flag = bad ? 2 : ((maxstep < tol) ? 1 : 0);
        }
        __syncthreads();
        if (flag != 0) { status = (flag == 1) ? 0 : 1; break; }
    }
    __syncthreads();
    double acc = 0.0;
    if (status == 0) {
        for (int i = tid; i < nva; i += nthr) {
            double z = w[s + 1];
            for (int j = 0; j <= s; ++j) { const int col = (j < s) ? sel[j] : ccol; const long long vo = (long long)col * ldv; z += (V[vo + va[i]] - mu[j]) / sd[j] * w[j]; }
            double p1 = 1.0 / (1.0 + exp(-z));
            p1 = fmin(fmax(p1, eps), 1.0 - eps);
            const double yv = (yenc[va[i]] == 1) ? 1.0 : 0.0;
            acc += yv * log(p1) + (1.0 - yv) * log(1.0 - p1);
        }
    }
    red[tid] = acc; __syncthreads();
    for (int st = nthr >> 1; st > 0; st >>= 1) { if (tid < st) red[tid] += red[tid + st]; __syncthreads(); }
    if (tid == 0) { out_ll[out_off + b] = -red[0] / nva; out_status[out_off + b] = status; }
}
"""
_FUSED_KERNEL: dict = {}
_MAX_FUSED_KK = 40


def kernel_logistic_cv(cp: Any, real: Any = np.float64) -> Any:  # Intentional float64: the double-precision value matrix is the default instance
    """``logistic_cv_newton`` (the whole per-candidate cross-validated logistic fit in one block) for a ``float32`` or ``float64`` value matrix; compile point of the fused path and of the
    register-pressure registry. The kernel reads the values at their own width and computes in double."""
    key = np.dtype(real).name
    with _LOGISTIC_KERNELS_LOCK:
        kern = _FUSED_KERNEL.get(key)
        if kern is None:
            kern = _FUSED_KERNEL[key] = cp.RawKernel(_FUSED_SRC.replace("VT_", "float" if key == "float32" else "double"), "logistic_cv_newton")
    return kern


def cv_candidates_binary(
    cp: Any, Vdev: Any, selected: "list[int]", cand_idx: "list[int]", folds: "list[tuple[Any, Any, bool]]", yenc_dev: Any, *, dtype: Any, newton_max_iter: int, tol: float, lam: float,
    logloss_eps: float, fallback_error: type,
) -> np.ndarray:
    """:func:`cv_candidates_binary_reference` with each (fold, chunk of candidates) as ONE launch of ``logistic_cv_newton``: the standardization, the Newton iteration of every candidate and the
    validation logloss all run inside the kernel, so a call is a few launches and one host read (the stacked-matmul form was about 270 launches a fold plus 180 per Newton fit). The fit is the same
    recipe, computed in double (the reference computed the Newton step in ``dtype``), so the logloss agrees to the reference's own precision. Shapes the kernel does not cover (a non-double or
    non-integer-indexed input, more than 38 selected columns) take the reference form."""
    s = len(selected)
    if (
        getattr(Vdev, "dtype", None) not in (cp.float32, cp.float64)  # Intentional float64: the value matrix is read at its own width
        or s + 2 > _MAX_FUSED_KK
        or not cand_idx
        or any(getattr(a, "dtype", None) is None or a.dtype.kind not in "iu" for f in folds for a in (f[0], f[1]))
    ):
        return cv_candidates_binary_reference(cp, Vdev, selected, cand_idx, folds, yenc_dev, dtype=dtype, newton_max_iter=newton_max_iter, tol=tol, lam=lam, logloss_eps=logloss_eps, fallback_error=fallback_error)
    m_all = len(cand_idx)
    nf = len(folds)
    kk = s + 2
    V = cp.ascontiguousarray(Vdev.T)  # column-major copy (columns x rows): the kernel gathers one column over the threads' rows, so consecutive threads read consecutive elements
    kern = kernel_logistic_cv(cp, V.dtype)
    sel_dev = cp.asarray(np.asarray(selected if s else [0], dtype=np.int32))
    cand_dev = cp.asarray(np.asarray(cand_idx, dtype=np.int32))
    y64 = yenc_dev if yenc_dev.dtype == cp.int64 else yenc_dev.astype(cp.int64)
    ll = cp.zeros((nf, m_all), dtype=cp.float64)  # Intentional float64: per-fold logloss accumulator
    status = cp.zeros((nf, m_all), dtype=cp.int32)
    threads = 128  # launch-ok: the block reduces over rows and solves a k x k system; the width is part of the kernel's shared-memory layout (one double per thread)
    trainable_h = np.array([bool(f[2]) for f in folds])
    for fo, (tr, va, trainable) in enumerate(folds):
        if not trainable:
            continue
        tr = cp.ascontiguousarray(tr.astype(cp.int64, copy=False))
        va = cp.ascontiguousarray(va.astype(cp.int64, copy=False))
        ntr, nva = int(tr.shape[0]), int(va.shape[0])
        chunk = max(1, (_BATCH_ELEMENT_BUDGET // 2) // max(1, ntr * kk))
        xs = cp.empty(min(chunk, m_all) * ntr * kk, dtype=cp.float64)  # Intentional float64: the standardized train designs of one chunk
        wd = cp.empty(min(chunk, m_all) * ntr, dtype=cp.float64)  # Intentional float64: IRLS weights
        rr = cp.empty(min(chunk, m_all) * ntr, dtype=cp.float64)  # Intentional float64: residuals
        shared = (kk * kk + 3 * kk + 2 * (s + 1) + threads) * 8
        for lo in range(0, m_all, chunk):
            mb = min(chunk, m_all - lo)
            kern((mb,), (threads,), (V, np.int64(V.shape[1]), tr, np.int32(ntr), va, np.int32(nva), sel_dev, np.int32(s), cand_dev, np.int32(lo), y64, np.float64(lam), np.float64(tol), np.int32(newton_max_iter), np.float64(logloss_eps), xs, wd, rr, ll, status, np.int32(fo * m_all + lo)), shared_mem=shared)  # launch-ok: the block width is part of the kernel's shared-memory layout (one double per thread), see ``threads`` above  # Intentional float64: the Newton solver and the log-loss run at double, read back in one copy
    host = cp.asnumpy(cp.concatenate([ll.ravel(), status.ravel().astype(cp.float64)]))  # ONE read: the logloss of every (fold, candidate) and the fit statuses
    ll_h = host[: nf * m_all].reshape(nf, m_all)
    st_h = host[nf * m_all :].reshape(nf, m_all)
    if (st_h[trainable_h] != 0).any():
        raise fallback_error()
    out = np.empty((m_all, nf), dtype=np.float64)  # Intentional float64: per-fold logloss accumulator
    for fo in range(nf):
        out[:, fo] = ll_h[fo] if trainable_h[fo] else np.inf
    return out


def cv_candidates_binary_reference(
    cp: Any, Vdev: Any, selected: "list[int]", cand_idx: "list[int]", folds: "list[tuple[Any, Any, bool]]", yenc_dev: Any, *, dtype: Any, newton_max_iter: int, tol: float, lam: float,
    logloss_eps: float, fallback_error: type,
) -> np.ndarray:
    """``(len(cand_idx), n_folds)`` validation logloss of ``selected + [candidate]`` for every candidate.

    ``folds`` lists ``(train_mask, val_mask, trainable)`` per fold on device; a fold whose train labels hold a single class (``trainable`` False) yields ``inf`` for every candidate.
    """
    m_all = len(cand_idx)
    out = np.empty((m_all, len(folds)), dtype=np.float64)  # Intentional float64: per-fold logloss accumulator
    s = len(selected)
    k = s + 1
    cand_arr = cp.asarray(np.asarray(cand_idx, dtype=np.int64))
    cand_cols = Vdev[:, cand_arr]
    for fo, (tr, va, trainable) in enumerate(folds):
        if not trainable:
            out[:, fo] = np.inf
            continue
        ytr = (yenc_dev[tr] == 1).astype(dtype)
        yva = (yenc_dev[va] == 1).astype(dtype)
        ntr = int(ytr.shape[0])
        nva = int(yva.shape[0])
        ctr = cand_cols[tr]
        cva = cand_cols[va]
        if s:
            str_, sva = Vdev[tr][:, selected], Vdev[va][:, selected]
            smu, ssd = str_.mean(axis=0), str_.std(axis=0)
            ssd = cp.where(ssd < 1e-12, 1.0, ssd)
            str_ = (str_ - smu) / ssd
            sva = (sva - smu) / ssd
        cmu, csd = ctr.mean(axis=0), ctr.std(axis=0)
        csd = cp.where(csd < 1e-12, 1.0, csd)
        ctr = (ctr - cmu) / csd
        cva = (cva - cmu) / csd
        chunk = max(1, _BATCH_ELEMENT_BUDGET // max(1, ntr * (k + 1)))
        for lo in range(0, m_all, chunk):
            hi = min(m_all, lo + chunk)
            mb = hi - lo
            A = cp.empty((mb, ntr, k + 1), dtype=dtype)
            if s:
                A[:, :, :s] = str_[None, :, :]
            A[:, :, s] = ctr[:, lo:hi].T
            A[:, :, k] = 1.0
            w = _newton(cp, A, ytr, mb, k + 1, dtype, newton_max_iter, tol, lam, fallback_error)
            B = cp.empty((mb, nva, k), dtype=dtype)
            if s:
                B[:, :, :s] = sva[None, :, :]
            B[:, :, s] = cva[:, lo:hi].T
            z = cp.einsum("bnk,bk->bn", B, w[:, :k]) + w[:, k : k + 1]
            p1 = cp.clip(1.0 / (1.0 + cp.exp(-z)), logloss_eps, 1.0 - logloss_eps)
            ll = -cp.mean(yva[None, :] * cp.log(p1) + (1.0 - yva[None, :]) * cp.log(1.0 - p1), axis=1)
            out[lo:hi, fo] = cp.asnumpy(ll).astype(np.float64)  # Intentional float64: per-fold logloss accumulator
    return out


_LOGISTIC_KERNELS: dict = {}
_LOGISTIC_KERNELS_LOCK = threading.RLock()  # serialises the get-or-build of the cache above (one build per key even with concurrent fits)


def _logistic_p_and_weight(cp: Any, z: Any) -> "tuple[Any, Any]":
    """``p = 1 / (1 + exp(-z))`` and the IRLS weight ``p (1 - p)`` of the logits ``z`` in one launch at ``z``'s dtype (the unfused form was five cupy operations per Newton step)."""
    with _LOGISTIC_KERNELS_LOCK:
        if not hasattr(cp, "ElementwiseKernel"):  # the host twin runs this loop on NumPy
            p = 1.0 / (1.0 + cp.exp(-z))
            return p, p * (1.0 - p)
        dt = str(z.dtype)
        kernel = _LOGISTIC_KERNELS.get(dt)
        if kernel is None:
            c = {"float32": "float", "float64": "double"}[dt]  # Intentional float64: dtype-name table of the kernel source rendering: float64 data maps to the C type double
            kernel = _LOGISTIC_KERNELS[dt] = cp.ElementwiseKernel("%s z" % dt, "%s p, %s w" % (dt, dt), "p = (%s)1 / ((%s)1 + exp(-z)); w = p * ((%s)1 - p);" % (c, c, c), "mlframe_logistic_p_w_" + dt)
        out: "tuple[Any, Any]" = kernel(z)
        return out


def _newton(cp: Any, A: Any, yb: Any, mb: int, kk: int, dtype: Any, max_iter: int, tol: float, lam: float, fallback_error: type) -> Any:
    """Batched L2 Newton for the logistic weights of ``mb`` designs ``A`` ``(mb, n, kk)`` (last column the unpenalised intercept) against the shared labels ``yb``."""
    w = cp.zeros((mb, kk), dtype=dtype)
    reg = cp.ones(kk, dtype=dtype)
    reg[kk - 1] = 0.0
    active = cp.ones(mb, dtype=bool)
    eye_idx = cp.arange(kk)
    At = A.transpose(0, 2, 1)
    for _ in range(max_iter):
        z = cp.einsum("bnk,bk->bn", A, w)
        p, wd = _logistic_p_and_weight(cp, z)
        grad = cp.einsum("bnk,bn->bk", A, p - yb[None, :]) + lam * reg[None, :] * w
        H = cp.matmul(At, A * wd[:, :, None])
        H[:, eye_idx, eye_idx] += lam * reg[None, :]
        try:
            step = cp.linalg.solve(H, grad[:, :, None])[:, :, 0]
        except Exception:
            raise fallback_error() from None
        w = w - step * active[:, None].astype(dtype)
        done = cp.max(cp.abs(step), axis=1) < tol
        active = active & ~done
        finite_and_active = cp.stack((cp.all(cp.isfinite(w)), active.any()))
        finite_and_active = finite_and_active.get() if hasattr(finite_and_active, "get") else finite_and_active  # one blocking read per iteration, not two
        if not bool(finite_and_active[0]):
            raise fallback_error()
        if not bool(finite_and_active[1]):
            return w
    raise fallback_error()


def make_cv_candidates_binary(cp: Any, Vdev: Any, yenc_dev: Any, tr_masks: list, va_masks: list, trainable: "list[bool]", **fit_kwargs: Any) -> Any:
    """``f(selected, candidates)`` giving the ``(len(candidates), n_folds)`` validation logloss matrix, with the fold masks and fit settings bound."""
    # boolean fold masks become row index arrays once here (the fused kernel gathers by index; one nonzero per mask for the whole greedy run, not per candidate batch)
    def _rows(m: Any) -> Any:
        """The row indices of a boolean mask; an integer index array passes unchanged."""
        return cp.flatnonzero(m) if getattr(m, "dtype", None) == cp.bool_ else m

    specs = [(_rows(tr_masks[fo]), _rows(va_masks[fo]), bool(trainable[fo])) for fo in range(len(tr_masks))]

    def _cv_candidates(selected: "list[int]", cand_idx: "list[int]") -> np.ndarray:
        """Per-fold logloss of every candidate from one batched Newton fit per fold."""
        return cv_candidates_binary(cp, Vdev, list(selected), list(cand_idx), specs, yenc_dev, **fit_kwargs)

    return _cv_candidates


__all__ = ["cv_candidates_binary", "make_cv_candidates_binary"]
