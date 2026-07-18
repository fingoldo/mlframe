"""BATCHED cross-column twin of ``detect_hinge_breakpoints_gpu`` (2026-07-16, cProfile-driven).

THE FINDING: warmed steady-state per-call timing of ``detect_hinge_breakpoints_gpu`` floors at ~10-12ms
even on a column with zero genuine kink (n=99401), dominated by ~8 host<->device synchronisation points
per call (``float(sse_lin)``, two ``cp.asnumpy(cp.quantile(...))`` cut-vectors, the precheck
``best_drop``, and one ``sse_B``/``argmin`` pair per breakpoint round) -- NOT actual compute (a full-scan
column at n=99401 with k<=4 design columns is trivially cheap in raw FLOPs). ``generate_hinge_features``
calls this ONE COLUMN AT A TIME (216 calls measured on the wellbore-100k fit, 7.6s cumulative, ~35ms/call
average incl. one-time compile warmup) -- each call pays its own fixed sync-latency floor independently.

THE FIX: the plausibility PRECHECK (which the CPU docstring's own history records as normally filtering
out ~87.5% of columns, "~8x fewer solves") is batchable in CLOSED FORM across every candidate column at
once, because the caller (``generate_hinge_features``) already REQUIRES every surviving column to be
all-finite (any column with a single NaN is skipped before reaching the detector) -- so every column
shares the exact same row count and the same finite ``y``, with NO ragged-array complication. Batching:

  * one ``X_batch.mean(axis=0)`` / centered-sums pass computes every column's simple-OLS SSE
    (``SSE_lin[k] = SYY - SXY[k]^2/SXX[k]``) in ONE reduction instead of K Gram-solves;
  * one ``cp.quantile(X_batch, precheck_qs, axis=0)`` gets every column's 3 precheck cut positions;
  * the FWL relu-leg SSE-drop at each cut is the SAME projection identity as the single-column path,
    re-derived in closed form (b_v = cov(x,relu)/var(x), r_relu = relu - mean(relu) - b_v*(x-mean(x)))
    and evaluated for all (column, cut) pairs via elementwise ops + one axis=0 reduction;
  * ONE ``cp.asnumpy`` at the very end reads back the whole (K,) pass/fail vector.

Total: O(1) host<->device round trips for ALL K columns' precheck, replacing K separate multi-sync calls.
Columns that PASS still go through the EXACT UNCHANGED single-column ``detect_hinge_breakpoints_gpu`` for
the real multi-round greedy tau search (zero new code on that path -- the only new numerics is the
precheck PASS/FAIL decision, and only a boolean gate, not a value that propagates into the output). A
column whose batched precheck disagrees with the single-column precheck by a hair near the threshold can
only cost a redundant (harmless) extra full-scan call, never a missed/wrong breakpoint, since the full
scan re-validates from scratch.

Both this batched entry point AND the original per-column ``detect_hinge_breakpoints_gpu`` are kept
(REJECTED-is-not-DELETED discipline) -- see ``_hinge_basis_fe.generate_hinge_features`` for the dispatcher
that benches both on first use and defaults to whichever measured faster on THIS host/shape, with the
result cached per (n, K) so repeat calls at the same shape skip re-benching."""
from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["detect_hinge_breakpoints_gpu_batch"]


def _batched_precheck_gpu(cp, X_batch, yg, precheck_qs, precheck_min_sse_drop: float, min_seg_rows: int) -> np.ndarray:
    """Return a host ``(K,)`` bool array: which columns of ``X_batch`` (n, K) clear the plausibility
    precheck against the shared ``yg`` (n,). Closed-form batched twin of the single-column precheck in
    ``detect_hinge_breakpoints_gpu`` -- see module docstring for the derivation. Selection-equivalent (a
    PASS/FAIL gate, not a value that propagates into the emitted breakpoints)."""
    n, K = X_batch.shape
    xbar = X_batch.mean(axis=0)  # (K,)
    ybar = yg.mean()
    Xc = X_batch - xbar[None, :]  # (n, K)
    yc = yg - ybar  # (n,)
    sxx = (Xc * Xc).sum(axis=0)  # (K,)
    sxy = Xc.T @ yc  # (K,) -- one GEMV for every column's cov(x, y)
    syy = float(yc @ yc)
    sxx_safe = cp.where(sxx > 1e-24, sxx, 1.0)
    b = sxy / sxx_safe
    sse_lin = syy - sxy * sxy / sxx_safe  # (K,)
    degenerate = sxx <= 1e-24

    qs = np.asarray(precheck_qs, dtype=np.float64)
    cuts = cp.quantile(X_batch, cp.asarray(qs), axis=0)  # (C, K) -- per-column cut positions
    best_drop = cp.zeros(K, dtype=cp.float64)
    for c_idx in range(cuts.shape[0]):
        cut_k = cuts[c_idx]  # (K,)
        relu = cp.maximum(X_batch - cut_k[None, :], 0.0)  # (n, K)
        n_right = (X_batch > cut_k[None, :]).sum(axis=0)
        ok = (n_right >= min_seg_rows) & ((n - n_right) >= min_seg_rows)

        relu_mean = relu.mean(axis=0)
        cov_x_relu = (Xc * relu).sum(axis=0)
        b_v = cov_x_relu / sxx_safe
        r_relu = (relu - relu_mean[None, :]) - b_v[None, :] * Xc  # FWL residual of relu vs [1, x]
        denom = (r_relu * r_relu).sum(axis=0)
        y_resid = yc[:, None] - b[None, :] * Xc  # y residualised against each column's OWN linear fit
        num = (r_relu * y_resid).sum(axis=0)
        denom_safe = cp.where(denom > 1e-24, denom, 1.0)
        sse_relu = cp.where(denom > 1e-24, sse_lin - num * num / denom_safe, sse_lin)
        with_guard = cp.where(ok, sse_relu, sse_lin)
        drop = cp.where(sse_lin > 1e-24, 1.0 - with_guard / cp.where(sse_lin > 1e-24, sse_lin, 1.0), 0.0)
        best_drop = cp.maximum(best_drop, cp.where(cp.isfinite(drop), drop, 0.0))

    passes = (~degenerate) & (sse_lin > 1e-24) & (best_drop >= float(precheck_min_sse_drop))
    return np.asarray(cp.asnumpy(passes))


def detect_hinge_breakpoints_gpu_batch(
    x_cols: Sequence[np.ndarray],
    y: np.ndarray,
    *,
    max_breakpoints: int,
    min_heldout_r2_uplift: float,
    precheck_qs: Sequence[float],
    precheck_min_sse_drop: float,
    cand_q_lo: float,
    cand_q_hi: float,
    n_candidates: int,
    min_rows: int,
    min_seg_rows: int,
) -> Optional[list]:
    """Batched twin of ``detect_hinge_breakpoints_gpu``: runs the plausibility precheck for EVERY column
    in ``x_cols`` in ONE closed-form GPU pass, then only the columns that pass go through the exact
    unchanged single-column full-scan detector. Returns a list of tau-lists (one per input column, same
    order), or ``None`` on any cupy fault (caller falls back to the per-column host/GPU detectors).

    PRECONDITION (checked by the caller, ``generate_hinge_features``): every column in ``x_cols`` and
    ``y`` must already be finite and share the same length ``n`` -- the caller already enforces this (any
    column with a NaN is skipped before reaching the detector), so this function does not re-validate it
    beyond a shape assert; callers with ragged/NaN-bearing columns must use the per-column entry point."""
    if not x_cols:
        return []
    try:
        import cupy as cp

        from ._hinge_detect_gpu_resident import detect_hinge_breakpoints_gpu
        from ._fe_resident_operands import resident_operand

        n = int(np.asarray(x_cols[0]).shape[0])
        y_arr = np.asarray(y, dtype=np.float64).ravel()
        if y_arr.size != n or n < min_rows:
            return None
        X_host = np.empty((n, len(x_cols)), dtype=np.float64)
        for j, xc in enumerate(x_cols):
            xj = np.asarray(xc, dtype=np.float64).ravel()
            if xj.size != n:
                return None  # ragged input -- precondition violated, caller must use per-column path
            X_host[:, j] = xj
        X_batch = cp.asarray(X_host)
        yg = resident_operand(y_arr, "hinge_y", dtype=np.float64)

        passes = _batched_precheck_gpu(cp, X_batch, yg, precheck_qs, precheck_min_sse_drop, min_seg_rows)

        out: list = [[] for _ in x_cols]
        for j, ok in enumerate(passes):
            if not ok:
                continue
            taus = detect_hinge_breakpoints_gpu(
                x_cols[j], y_arr, max_breakpoints=max_breakpoints, min_heldout_r2_uplift=min_heldout_r2_uplift,
                precheck_qs=precheck_qs, precheck_min_sse_drop=precheck_min_sse_drop,
                cand_q_lo=cand_q_lo, cand_q_hi=cand_q_hi, n_candidates=n_candidates,
                min_rows=min_rows, min_seg_rows=min_seg_rows,
            )
            out[j] = taus if taus is not None else []
        return out
    except Exception as _exc:
        logger.debug("detect_hinge_breakpoints_gpu_batch failed (%s); per-column fallback", _exc)
        return None
