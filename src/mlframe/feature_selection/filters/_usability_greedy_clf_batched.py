"""Cross-validated logloss of every shortlisted candidate from one batched Newton fit per fold.

``_usability_greedy_clf_gpu_resident._cv_candidate`` refits the logistic model of ``selected + [candidate]`` on every fold, one candidate at a time: a round over a shortlist of 40 with 4 folds is 160 small
fits, each a few Newton iterations of tiny launches ending in a host sync. The fits are independent, so this module stacks the candidates of one fold into a ``(candidates, train rows, k + 1)`` design and runs
the Newton iteration on all of them at once with batched matmuls and one batched solve, with one host sync per iteration for the whole batch.

Each member follows the single-fit recipe operation by operation: train-fold standardization (population std, columns below ``1e-12`` held at scale 1), unpenalised intercept, L2 penalty ``1 / C``, and the stop
``max|step| < tol`` evaluated per member (a member that has met its stop is frozen, as the single fit returns at that point). A non-finite weight or a missing convergence raises the caller's fallback error.
"""

from __future__ import annotations

from typing import Any

import numpy as np

_BATCH_ELEMENT_BUDGET = 48_000_000  # design elements per chunk (three live copies of the chunk design at 4 bytes each stay near 600 MB)


def cv_candidates_binary(
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
    sel_cols = Vdev[:, selected] if s else None
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
            str_, sva = sel_cols[tr], sel_cols[va]
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
        p = 1.0 / (1.0 + cp.exp(-z))
        wd = p * (1.0 - p)
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
        if not bool(cp.all(cp.isfinite(w))):
            raise fallback_error()
        if not bool(active.any()):
            return w
    raise fallback_error()


def make_cv_candidates_binary(cp: Any, Vdev: Any, yenc_dev: Any, tr_masks: list, va_masks: list, trainable: "list[bool]", **fit_kwargs: Any) -> Any:
    """``f(selected, candidates)`` giving the ``(len(candidates), n_folds)`` validation logloss matrix, with the fold masks and fit settings bound."""
    specs = [(tr_masks[fo], va_masks[fo], bool(trainable[fo])) for fo in range(len(tr_masks))]

    def _cv_candidates(selected: "list[int]", cand_idx: "list[int]") -> np.ndarray:
        """Per-fold logloss of every candidate from one batched Newton fit per fold."""
        return cv_candidates_binary(cp, Vdev, list(selected), list(cand_idx), specs, yenc_dev, **fit_kwargs)

    return _cv_candidates


__all__ = ["cv_candidates_binary", "make_cv_candidates_binary"]
