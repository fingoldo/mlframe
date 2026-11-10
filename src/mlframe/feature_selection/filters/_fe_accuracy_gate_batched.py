"""Held-out uplift of MANY engineered columns against their raw sources in one batched pass on the device.

The accuracy gate probes each candidate on its own: 10 folds of a 5000-row subsample, each fold a standardised logistic (or ridge) fit and a held-out score, about 26 ms per candidate
(``audits/2026-10-22/profiles/_fe_accuracy_gate.md``). At 1000 columns that is the largest host block of the fit, about 65 thread-seconds, and the profile note already names the lever:
one batched Newton over all candidates and folds.

Batching applies because every candidate sees the SAME rows and the SAME folds: ``keep_engineered_over_source`` draws its subsample with ``default_rng(seed)`` and one seed for the whole
gate, and ``_probe_splits`` builds the folds with ``StratifiedKFold(..., random_state=seed)`` over ``y`` alone (the design enters only by its row count). Only the columns differ: the base
design is the raw source, the augmented one the source plus the engineered column. So a fold is a stack of ``(mb, n_train, k)`` designs of ONE width, which cupy solves as a batch
(``einsum`` for the gradient, ``matmul`` for the Hessian, ``cholesky``/``solve`` for the step).

The arithmetic mirrors ``_fe_accuracy_gate_newton.fit_logistic_ridge_newton_numpy`` term for term, including the gradient-norm stop test and the backtracking line search, and the held-out
score mirrors ``mlframe.metrics.core.fast_roc_auc`` (mid-ranks on ties). The batch reduces in a different ORDER than the per-candidate path, so the uplift of a candidate can differ in the
last bits; the bar for this family is an unchanged SELECTION, which ``tests/feature_selection/fe/test_accuracy_gate_batched.py`` pins by comparing the kept set against the host path.
A candidate whose batched fit does not converge is handed back as ``nan`` so the caller probes it on the host, exactly as a non-converged host fit falls through to the scikit-learn fit.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional, Sequence

import numpy as np

from ._fe_accuracy_gate_newton import _GRAD_TOL, _MAX_ITER
from ._fe_usability_signal import _crit_np_dtype

logger = logging.getLogger(__name__)

#: Candidates from which the batched device path is worth its uploads (below it the per-candidate host probe wins; the dispatcher's tuned entry overrides this).
_MIN_CANDIDATES = 24
#: Designs held on the device at once: one fold's block is ``mb * n_rows * k`` doubles, and 512 designs of 5000 rows by 3 columns are 61 MB.
_MAX_BLOCK = 512


def batched_enabled() -> bool:
    """Whether the batched device probe may run (``MLFRAME_GATE_BATCHED=0`` keeps the per-candidate host path)."""
    return os.environ.get("MLFRAME_GATE_BATCHED", "1").strip() not in ("0", "false", "off", "no")


def _newton_batch(cp: Any, A: Any, y: Any, lam: float) -> Any:
    """``(mb, k)`` logistic-ridge weights of the designs ``A`` ``(mb, n, k)`` (last column the unpenalised intercept) against the shared labels ``y``; a design that does not converge gets NaN.

    Mirrors the host Newton: the stop test is ``max|grad| < _GRAD_TOL * sqrt(max(1, n))``, the Hessian carries ``reg + 1e-12`` on its diagonal, and a step that does not lower the objective
    is halved up to twelve times. Designs converge at different iterations, so a converged one is frozen (its step is multiplied by zero) instead of being dropped from the batch.
    """
    mb, n, k = A.shape
    dt = A.dtype
    reg = cp.full(k, float(lam), dtype=dt)
    reg[k - 1] = 0.0
    w = cp.zeros((mb, k), dtype=dt)
    At = cp.ascontiguousarray(A.transpose(0, 2, 1))
    eye = cp.arange(k)
    tol = float(_GRAD_TOL) * max(1.0, float(n)) ** 0.5
    z = cp.zeros((mb, n), dtype=dt)
    f = _objective_batch(cp, z, y, w, float(lam))
    live = cp.ones(mb, dtype=cp.bool_)  # designs still iterating
    dead = cp.zeros(mb, dtype=cp.bool_)  # designs whose fit failed (no finite step, or the line search ran out)
    for _ in range(int(_MAX_ITER)):
        mu = 1.0 / (1.0 + cp.exp(-z))
        grad = cp.einsum("bnk,bn->bk", A, mu - y[None, :]) + reg[None, :] * w
        converged = cp.max(cp.abs(grad), axis=1) < tol
        live = live & ~converged & ~dead
        if not bool(live.any().get() if hasattr(live.any(), "get") else live.any()):
            break
        s = mu * (1.0 - mu)
        hess = cp.matmul(At, A * s[:, :, None])
        hess[:, eye, eye] += reg[None, :] + 1e-12
        step = _solve_batch(cp, hess, grad)
        bad = ~cp.all(cp.isfinite(step), axis=1)
        dead = dead | (bad & live)
        step = cp.where(cp.isfinite(step), step, 0.0)
        t = cp.ones((mb, 1), dtype=dt)
        accepted = cp.zeros(mb, dtype=cp.bool_)
        w_new, z_new, f_new = w, z, f
        for _ls in range(12):
            cand = w - t * step * live[:, None]
            z_cand = cp.einsum("bnk,bk->bn", A, cand)
            f_cand = _objective_batch(cp, z_cand, y, cand, float(lam))
            take = (~accepted) & cp.isfinite(f_cand) & (f_cand <= f)
            w_new = cp.where(take[:, None], cand, w_new)
            z_new = cp.where(take[:, None], z_cand, z_new)
            f_new = cp.where(take, f_cand, f_new)
            accepted = accepted | take
            if bool((accepted | ~live).all().get() if hasattr((accepted | ~live).all(), "get") else (accepted | ~live).all()):
                break
            t = t * 0.5
        dead = dead | (live & ~accepted)  # the line search ran out: the host path returns None here
        w, z, f = w_new, z_new, f_new
    out = cp.where((dead | live)[:, None], cp.nan, w)  # ``live`` still set means the iteration budget ran out, which the host path also reports as a failure
    return out


def _objective_batch(cp: Any, z: Any, y: Any, w: Any, lam: float) -> Any:
    """``(mb,)`` penalised logistic loss of each design, the batched form of ``_fe_accuracy_gate_newton._objective``: ``sum(log1p(exp(z)) - y z) + lam/2 * |w[:-1]|^2``."""
    loss = cp.sum(cp.logaddexp(0.0, z) - y[None, :] * z, axis=1)
    return loss + 0.5 * float(lam) * cp.sum(w[:, :-1] * w[:, :-1], axis=1)


def _solve_batch(cp: Any, hess: Any, grad: Any) -> Any:
    """``(mb, k)`` Newton steps ``hess^-1 grad``; a singular design's step comes back non-finite (the caller marks it failed), as the host path's ``LinAlgError`` does."""
    try:
        return cp.linalg.solve(hess, grad[:, :, None])[:, :, 0]
    except Exception as exc:  # a singular design in the batch: cupy raises for the whole call, so fall back to a per-design solve
        logger.debug("batched Newton: the batch solve failed (%s); solving the designs one by one", exc)
        out = cp.empty_like(grad)
        for b in range(int(hess.shape[0])):
            try:
                out[b] = cp.linalg.solve(hess[b], grad[b])
            except Exception:  # noqa: PERF203 - a singular design is reported as non-finite, like the host path's LinAlgError
                out[b] = cp.nan
        return out


def _standardised(cp: Any, X: Any, rows: Any) -> Any:
    """``(mb, len(rows), k+1)`` designs: the columns of ``X`` ``(mb, n, k)`` taken at ``rows``, centred and scaled by THOSE rows' mean and population deviation, plus a ones column.

    The zero-spread rule is the host probe's: a deviation at or below ``10 * eps * max(|mean|, 1)`` is replaced by 1.0, so a constant column becomes zeros instead of NaN. The standardised
    values go through the criterion dtype exactly as the host probe does (``_held_out_probe_score`` casts them to the frame's dtype, and the Newton fit widens them again), so a float32
    project dtype rounds them here too and the fit sees the same numbers.
    """
    sub = X[:, rows, :]
    mu = sub.mean(axis=1, keepdims=True)
    sd = sub.std(axis=1, keepdims=True)
    floor = 10.0 * float(np.finfo(np.float64).eps) * cp.maximum(cp.abs(mu), 1.0)  # Intentional float64: the host probe's zero-spread rule is written in double
    sd = cp.where(sd > floor, sd, 1.0)
    out = (sub - mu) / sd
    crit = _crit_np_dtype()
    if crit != np.float64:
        out = out.astype(crit).astype(sub.dtype)  # the host probe's round trip through the frame dtype
    ones = cp.ones((out.shape[0], out.shape[1], 1), dtype=out.dtype)
    return cp.concatenate([out, ones], axis=2)


def _auc_batch(cp: Any, scores: Any, y: Any) -> Any:
    """``(mb,)`` ROC-AUC of each row of ``scores`` ``(mb, n)`` against the shared 0/1 labels ``y``, with mid-ranks on ties (the quantity ``fast_roc_auc`` returns); 0.5 when one class is absent."""
    n_pos = float(cp.sum(y).get() if hasattr(cp.sum(y), "get") else cp.sum(y))
    n = int(y.shape[0])
    n_neg = float(n) - n_pos
    if n_pos <= 0 or n_neg <= 0:
        return cp.full(int(scores.shape[0]), 0.5, dtype=scores.dtype)
    order = cp.argsort(scores, axis=1, kind="stable")
    srt = cp.take_along_axis(scores, order, axis=1)
    ranks = cp.broadcast_to(cp.arange(1, n + 1, dtype=scores.dtype), srt.shape).copy()
    # mid-rank of every run of equal scores: the run's first and last positions average to the rank the tie shares
    is_start = cp.ones(srt.shape, dtype=cp.bool_)
    is_start[:, 1:] = srt[:, 1:] != srt[:, :-1]
    grp = cp.cumsum(is_start, axis=1) - 1
    nb = int(grp[:, -1].max().get() if hasattr(grp[:, -1].max(), "get") else grp[:, -1].max()) + 1
    rows = cp.broadcast_to(cp.arange(int(scores.shape[0]), dtype=cp.int64)[:, None], srt.shape)
    flat = (rows * nb + grp).ravel()
    sums = cp.bincount(flat, weights=ranks.ravel(), minlength=int(scores.shape[0]) * nb).reshape(int(scores.shape[0]), nb)
    counts = cp.bincount(flat, minlength=int(scores.shape[0]) * nb).reshape(int(scores.shape[0]), nb)
    mid = sums / cp.maximum(counts, 1)
    ranks = cp.take_along_axis(mid, grp, axis=1)
    y_srt = cp.take_along_axis(cp.broadcast_to(y[None, :], srt.shape).copy(), order, axis=1)
    rank_sum = cp.sum(ranks * y_srt, axis=1)
    return (rank_sum - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)


def _fold_scores(cp: Any, X: Any, y_dev: Any, folds: Sequence[tuple], lam: float) -> Any:
    """``(mb, n_folds)`` held-out AUC of every design of ``X`` ``(mb, n, k)`` on every fold; a design whose fit failed on a fold gets NaN there."""
    out = cp.empty((int(X.shape[0]), len(folds)), dtype=X.dtype)
    for j, (tr, va) in enumerate(folds):
        A = _standardised(cp, X, tr)
        w = _newton_batch(cp, A, y_dev[tr], lam)
        Av = _standardised(cp, X, va)
        z = cp.einsum("bnk,bk->bn", Av, cp.where(cp.isfinite(w), w, 0.0))
        auc = _auc_batch(cp, 1.0 / (1.0 + cp.exp(-z)), y_dev[va])
        out[:, j] = cp.where(cp.all(cp.isfinite(w), axis=1), auc, cp.nan)
    return out


def batched_uplift(
    base_cols: Sequence[np.ndarray], eng_cols: Sequence[np.ndarray], y: np.ndarray, *, folds: Sequence[tuple], base_of: Sequence[int], lam: float = 1.0
) -> "Optional[np.ndarray]":
    """``(n_candidates,)`` mean held-out AUC uplift of each engineered column over its base column, or ``None`` when the device path does not apply.

    ``base_cols`` holds the DISTINCT base columns (each ``(n,)``), ``eng_cols`` the candidates' engineered columns, and ``base_of[i]`` the index in ``base_cols`` of candidate ``i``'s base,
    so a base shared by several candidates is fitted once. ``folds`` are the ``(train rows, validation rows)`` index arrays the host probe would use. A candidate whose batched fit did not
    converge on some fold comes back as ``nan``: the caller probes that one on the host, as it does for a host fit that returns ``None``.
    """
    if not batched_enabled() or not eng_cols:
        return None
    try:
        import cupy as cp

        from ._gpu_policy import cuda_available_for_run

        if not cuda_available_for_run():
            return None
        y_arr = np.ascontiguousarray(np.asarray(y).ravel(), dtype=np.float64)  # Intentional float64: the Newton solve and the loss are in double, as on the host
        n = int(y_arr.shape[0])
        if n == 0 or any(int(np.asarray(c).shape[0]) != n for c in base_cols) or any(int(np.asarray(c).shape[0]) != n for c in eng_cols):
            return None
        y_dev = cp.asarray(y_arr)
        base_stack = cp.asarray(np.ascontiguousarray(np.stack([np.asarray(c, dtype=np.float64).ravel() for c in base_cols])[:, :, None]))  # Intentional float64: double designs, as on the host
        base_auc = _fold_scores(cp, base_stack, y_dev, folds, lam)
        del base_stack
        uplift = np.empty(len(eng_cols), dtype=np.float64)  # Intentional float64: the uplift is a mean of double AUC differences
        for start in range(0, len(eng_cols), _MAX_BLOCK):
            idx = range(start, min(len(eng_cols), start + _MAX_BLOCK))
            aug = np.empty((len(idx), n, 2), dtype=np.float64)  # Intentional float64: double designs, as on the host
            for row, i in enumerate(idx):
                aug[row, :, 0] = np.asarray(base_cols[base_of[i]], dtype=np.float64).ravel()
                aug[row, :, 1] = np.asarray(eng_cols[i], dtype=np.float64).ravel()
            aug_auc = _fold_scores(cp, cp.asarray(np.ascontiguousarray(aug)), y_dev, folds, lam)
            bases = cp.asarray(np.asarray([base_of[i] for i in idx], dtype=np.int64))
            delta = aug_auc - base_auc[bases]
            uplift[start : start + len(idx)] = cp.asnumpy(cp.mean(delta, axis=1))
        return uplift
    except Exception as exc:  # any device fault: the caller keeps the per-candidate host probe
        logger.warning("batched accuracy-gate probe not used (%s: %s); the per-candidate host probe decides", type(exc).__name__, exc)
        return None


def batched_uplifts_for_pairs(
    src_arrays: Sequence[np.ndarray], eng_arrays: Sequence[np.ndarray], base_keys: Sequence[Any], y: np.ndarray, *, seed: int = 0, max_probe_n: int = 5000, n_splits: int = 10
) -> "Optional[np.ndarray]":
    """``(n_candidates,)`` uplift of each ``(source, engineered)`` pair, ``nan`` for a pair the batch does not cover, or ``None`` when it does not apply at all.

    Covers the pairs the per-candidate probe would decide with the Newton fit on the SHARED rows and folds: a binary classification target, a pair whose values are all finite over the
    subsample (``measure_feature_uplift`` drops non-finite rows per pair, which would give that pair its own rows), and enough rows for the folds. Everything else comes back ``nan`` and the
    caller probes it as before. ``base_keys[i]`` names candidate ``i``'s source so a source shared by several candidates is fitted once.
    """
    from ._fe_accuracy_gate import _newton_probe_enabled, _probe_splits, infer_classification

    if not batched_enabled() or not eng_arrays or not _newton_probe_enabled():
        return None
    y_arr = np.asarray(y).ravel()
    n_full = int(y_arr.shape[0])
    if n_full == 0 or not infer_classification(y_arr):
        return None
    rows = np.arange(n_full)
    if n_full > int(max_probe_n):
        rows = np.random.default_rng(int(seed)).choice(n_full, int(max_probe_n), replace=False)  # the per-candidate probe draws this same subsample from the same seed
    y_sub = y_arr[rows]
    if y_sub.size < 40:
        return None
    classes = np.unique(y_sub)
    if classes.size != 2:  # a multiclass probe scores the hit rate of a scikit-learn fit, which this batch does not reproduce
        return None
    probe = _probe_splits(np.zeros((y_sub.size, 1)), y_sub, classification=True, n_splits=int(n_splits), seed=int(seed), groups=None)
    if probe is None:
        return None
    split_iter, y_enc, binary, _n_splits = probe
    if not binary:
        return None
    folds = [(tr, va) for tr, va in split_iter if np.unique(y_enc[tr]).size >= 2]
    if not folds:
        return None
    out = np.full(len(eng_arrays), np.nan)  # Intentional float64: the uplift is a mean of double AUC differences
    # the per-candidate probe reads its columns at the criterion dtype (``measure_feature_uplift`` casts them), so the batch starts from the same values
    crit = _crit_np_dtype()
    subs = [np.asarray(a, dtype=crit).ravel()[rows] for a in src_arrays]
    engs = [np.asarray(a, dtype=crit).ravel()[rows] for a in eng_arrays]
    covered = [i for i in range(len(engs)) if np.isfinite(subs[i]).all() and np.isfinite(engs[i]).all()]
    if not covered:
        return out
    order: "dict[Any, int]" = {}
    for i in covered:
        order.setdefault(base_keys[i], len(order))
    base_cols = [None] * len(order)
    for i in covered:
        base_cols[order[base_keys[i]]] = subs[i]
    got = batched_uplift(base_cols, [engs[i] for i in covered], y_enc.astype(np.float64), folds=folds, base_of=[order[base_keys[i]] for i in covered])  # Intentional float64: double labels for the Newton solve
    if got is None:
        return None
    out[np.asarray(covered, dtype=np.int64)] = got
    return out


__all__ = ["batched_enabled", "batched_uplift", "batched_uplifts_for_pairs"]
