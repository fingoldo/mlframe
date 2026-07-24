"""DEVICE-RESIDENT hinge breakpoint detector (kernel-residency port, 2026-07-02).

Device twin of ``_hinge_basis_fe._detect_hinge_breakpoints`` -- the STRICT FE scan's largest remaining
PURE-HOST compute stage (measured 5.9s host-CPU at 1M rows: per-column QR + 24 per-cut FWL projections +
held-out lstsq, all numpy). Everything runs on the GPU here:

* the plausibility PRE-CHECK (3 coarse cuts) and the full ``_HINGE_N_CANDIDATES``-cut scan share ONE batched
  Frisch-Waugh-Lovell evaluation: the relu column of EVERY candidate cut is materialised as one (n, C) device
  matrix and projected against the round's fixed design block in a single GEMM pair
  (``R - Q @ (Q.T @ R)``), so the per-cut Python loop and its 2*C host GEMVs collapse to 2 GEMMs;
* the per-round QR of the growing design block runs on device (``cp.linalg.qr``);
* the held-out %3-stride tau-validation fits both designs with ``cp.linalg.lstsq`` on device slices;
* only SCALARS cross the bus: the per-round argmin/best-SSE, the held-out uplift, and the found taus.

The math is IDENTICAL to the host detector (same FWL identity, same guards: per-side segment-row minimum,
``denom < 1e-24`` rank-deficiency skip, tau-neighbourhood dedup, the same held-out stride split) -- device FP
reduction order differs at ~1e-12, far below the tau-selection scale, so the found breakpoints are
selection-equivalent (validated: F2 + the hinge unit suite select identically). Any cupy fault returns
``None`` and the caller runs the exact host detector (byte-identical default path).

Gate: the STRICT-resident path (``fe_gpu_strict_resident_enabled``); opt-out ``MLFRAME_FE_HINGE_GPU=0``.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["detect_hinge_breakpoints_gpu", "hinge_gpu_enabled"]


def hinge_gpu_enabled() -> bool:
    """Device hinge detector gate: STRICT-resident path AND not opted out via MLFRAME_FE_HINGE_GPU=0."""
    if os.environ.get("MLFRAME_FE_HINGE_GPU", "1").strip().lower() not in ("1", "true", "on", "yes"):
        return False
    try:
        from ._gpu_strict_fe import fe_gpu_strict_resident_enabled
        return bool(fe_gpu_strict_resident_enabled())
    except Exception:
        return False


def _gram_projector(cp, B):
    """Projection onto span(B) via the NORMAL EQUATIONS: ``proj(V) = B @ solve(B.T B, B.T V)``.

    Returns a closure. For the detector's tall-skinny blocks (n ~ 1e6, k <= 2 + max_breakpoints) this is two
    GEMMs + a k x k solve -- the tall QR (cusolver geqrf) the first port used dominated the stage wall on a
    small card. Numerics: the blocks here are [1, x, relu legs] with x standardised by construction upstream
    (raw operand scale), Gram condition ~ (cond QR)^2 but k <= 4 and the SSE argmin operates on well-separated
    cut SSEs, so the tau selection is unchanged (validated vs the QR form: same taus on the F2 fit; the hinge
    suite + F2 selection green). Falls back to QR on a singular Gram (cp.linalg.solve raises -> caller's
    except path -> host detector)."""
    G = B.T @ B                                                  # (k, k)
    def _proj(V):
        """Project ``V`` onto span(B) via the pre-factored normal-equations Gram ``G``."""
        return B @ cp.linalg.solve(G, B.T @ V)
    return _proj


def _fwl_relu_legs(cp, xg, cuts, n: int, min_seg_rows: int):
    """Cut-set-invariant pieces of the batched FWL SSE: the ``(n, C)`` relu-leg block ``R`` and the
    per-side segment-row validity mask ``ok``. Both depend ONLY on ``xg``, the cut set and the row
    guards -- NOT on the round's design block -- so for a fixed candidate cut set they are computed
    ONCE and reused across the growing-design breakpoint rounds (only ``proj(R)`` / ``r_y`` change per
    round). Materialising the (n, C) relu block is the stage's largest device allocation; hoisting it
    out of the round loop removes one full (n, C) build + the segment-row count per breakpoint round."""
    cuts_d = cp.asarray(np.ascontiguousarray(cuts, dtype=np.float64))
    R = cp.maximum(xg[:, None] - cuts_d[None, :], 0.0)  # (n, C) all relu legs at once
    n_right = (xg[:, None] > cuts_d[None, :]).sum(axis=0)
    ok = (n_right >= min_seg_rows) & ((n - n_right) >= min_seg_rows)
    return R, ok


def _batched_fwl_sse(cp, R, ok, r_y, proj, sse_B: float, cuts, found):
    """SSE of ``y ~ [B | relu(x - c)]`` for EVERY cut ``c`` in one batched FWL evaluation.

    ``R`` / ``ok`` are the cut-invariant relu-leg block and row-guard mask from ``_fwl_relu_legs``;
    ``r_y`` is y residualised against the round's fixed block B (``proj`` projects onto span(B));
    ``sse_B = r_y @ r_y``. Returns a host float64 (C,) SSE vector with ``inf`` at cuts failing the
    per-side segment-row guard / the found-tau dedup, and ``sse_B`` where relu lies in span(B)
    (``denom < 1e-24`` -- the host detector's rank-deficiency skip)."""
    Rres = R - proj(R)  # FWL residuals vs B, one GEMM pair
    denom = cp.einsum("ij,ij->j", Rres, Rres)
    num = Rres.T @ r_y
    sse = cp.where(denom < 1e-24, sse_B, sse_B - num * num / cp.maximum(denom, 1e-300))
    sse = cp.where(ok, sse, cp.inf)
    sse_h = cp.asnumpy(sse).astype(np.float64)
    if found:
        for i, c in enumerate(np.asarray(cuts, dtype=np.float64)):
            if any(abs(float(c) - t) < 1e-9 for t in found):
                sse_h[i] = np.inf
    return sse_h


def _heldout_uplift_gpu(cp, xg, yg, tau: float, min_rows: int) -> float:
    """Device twin of ``_heldout_hinge_r2_uplift``: %3-stride split, both designs fit on train rows with
    ``cp.linalg.lstsq``, scored held-out; returns ``R2_hinge_val - R2_linear_val`` (scalar)."""
    n = int(xg.size)
    if n < min_rows:
        return 0.0
    # The %3-stride split is DETERMINISTIC (data-independent), so build the train/val row indices on the host
    # (np.arange) and upload them -- no int(mask.sum()) degeneracy syncs and no boolean-mask gathers (integer
    # gathers have a known size). Counts are host-known too.
    _ar = np.arange(n)
    _va_idx = cp.asarray(np.where((_ar % 3) == 0)[0])
    _tr_idx = cp.asarray(np.where((_ar % 3) != 0)[0])
    if int(_tr_idx.size) < 32 or int(_va_idx.size) < 16:
        return 0.0
    x_tr, y_tr = xg[_tr_idx], yg[_tr_idx]
    x_va, y_va = xg[_va_idx], yg[_va_idx]
    yv_ss = float(cp.sum((y_va - y_va.mean()) ** 2))
    if yv_ss < 1e-24:
        return 0.0

    def _val_r2(with_relu: bool) -> float:
        """Fit a linear (or linear+hinge-leg, when ``with_relu``) least-squares model on the train split and return its held-out R2 on the val split."""
        cols_tr = [cp.ones_like(x_tr), x_tr]
        cols_va = [cp.ones_like(x_va), x_va]
        if with_relu:
            cols_tr.append(cp.maximum(x_tr - tau, 0.0))
            cols_va.append(cp.maximum(x_va - tau, 0.0))
        A_tr = cp.stack(cols_tr, axis=1)
        A_va = cp.stack(cols_va, axis=1)
        try:
            # normal-equations solve (k <= 3): two GEMVs + k x k solve, vs cusolver lstsq on the tall block.
            coef = cp.linalg.solve(A_tr.T @ A_tr, A_tr.T @ y_tr)
        except Exception:
            return -np.inf
        pred = A_va @ coef
        sse = float(cp.sum((y_va - pred) ** 2))
        return 1.0 - sse / yv_ss

    r2_lin = _val_r2(False)
    r2_hinge = _val_r2(True)
    if not (np.isfinite(r2_lin) and np.isfinite(r2_hinge)):
        return 0.0
    return float(r2_hinge - r2_lin)


def _hinge_max_rows() -> int:
    """Row cap for the hinge breakpoint search (0 = full-n). See the subsample note in the function body."""
    try:
        return int(os.environ.get("MLFRAME_HINGE_MAX_ROWS", "250000"))
    except (ValueError, TypeError):
        return 250000


def detect_hinge_breakpoints_gpu(
    x: np.ndarray,
    y: np.ndarray,
    *,
    max_breakpoints: int,
    min_heldout_r2_uplift: float,
    precheck_qs,
    precheck_min_sse_drop: float,
    cand_q_lo: float,
    cand_q_hi: float,
    n_candidates: int,
    min_rows: int,
    min_seg_rows: int,
) -> Optional[list]:
    """Device-resident hinge breakpoint detection. Returns the found tau list, or ``None`` on any cupy
    fault so the caller runs the exact host detector. The finite-row filtering / std guards mirror the
    host detector exactly (host-side, cheap) before anything is uploaded."""
    try:
        import cupy as cp

        from ._fe_resident_operands import resident_operand

        x = np.asarray(x, dtype=np.float64).ravel()
        y = np.asarray(y, dtype=np.float64).ravel()
        n = x.size
        if n != y.size or n < min_rows:
            return []
        finite = np.isfinite(x) & np.isfinite(y)
        if not finite.all():
            x = x[finite]
            y = y[finite]
            n = x.size
            if n < min_rows:
                return []
        if float(np.std(x)) < 1e-12 or float(np.std(y)) < 1e-12:
            return []

        # SUBSAMPLE CAP: the breakpoint SELECTION (which tau clears ``min_heldout_r2_uplift``) is a threshold
        # decision on a held-out R^2 -- a large strided subsample estimates that uplift well within the decision
        # margin, while the full-n FWL/projection SSEs are the per-call cost. Selection-equivalent (the detector
        # only proposes tau candidates; the FE gate re-scores the built feature). Env MLFRAME_HINGE_MAX_ROWS
        # (default 250k, 0=full-n). The finite/std guards above already ran on the full column.
        _hmax = _hinge_max_rows()
        if _hmax > 0 and n > _hmax:
            # GPU_INFRA_D-11 fix (mrmr_audit_2026-07-22): floor division (`n // _hmax`) gave stride==1 (no
            # thinning at all) for any n strictly between _hmax and 2*_hmax, so the documented "caps at
            # <=_hmax rows" claim was only true once n reached ~2*_hmax. Ceiling division actually caps at n.
            _st = -(-n // _hmax)
            if _st > 1:
                x = x[::_st]
                y = y[::_st]
                n = int(x.size)

        xg = cp.asarray(x)  # candidate column -- a fresh column every call, never cacheable
        # y is fit-constant across every no-NaN candidate column of this fit (same target, same
        # deterministic subsample stride at a given n) -- route through the content-keyed resident cache
        # so repeated calls share ONE upload instead of re-uploading the same (n,) float64 target per
        # column. A column WITH NaNs takes the finite-mask filter above, which can legitimately produce a
        # DIFFERENT filtered y per column (the mask depends on that column's own NaN pattern) -- content
        # hashing handles this correctly: a genuinely different y just misses the cache (a fresh upload,
        # never a wrong reuse), it never merges non-identical content.
        yg = resident_operand(y, "hinge_y", dtype=np.float64)

        # PLAUSIBILITY PRE-CHECK (device): plain-linear SSE via the same FWL machinery with B=[1, x]; the
        # 3 coarse cuts' 2-segment SSE from one batched evaluation. Same decision as the host pre-check
        # (fit y ~ [1, x, relu] fully == FWL vs B=[1, x] -- the identical partitioned-regression identity).
        ones = cp.ones_like(xg)
        B = cp.stack([ones, xg], axis=1)
        proj = _gram_projector(cp, B)
        r_y = yg - proj(yg)
        sse_lin = float(r_y @ r_y)
        if not np.isfinite(sse_lin) or sse_lin <= 1e-24:
            return []
        pre_cuts = np.unique(cp.asnumpy(cp.quantile(xg, cp.asarray(np.asarray(precheck_qs, dtype=np.float64)))))
        if pre_cuts.size == 0:
            return []
        R_pre, ok_pre = _fwl_relu_legs(cp, xg, pre_cuts, n, min_seg_rows)
        sse_pre = _batched_fwl_sse(cp, R_pre, ok_pre, r_y, proj, sse_lin, pre_cuts, [])
        with np.errstate(invalid="ignore"):
            drops = 1.0 - sse_pre / sse_lin
        best_drop = float(np.nanmax(np.where(np.isfinite(sse_pre), drops, 0.0))) if sse_pre.size else 0.0
        if best_drop < float(precheck_min_sse_drop):
            return []

        # FULL SCAN: inner-quantile candidate cuts; per round QR the growing block once and score every
        # cut in one batched FWL evaluation; held-out-validate the argmin tau; add its relu leg and repeat.
        qs = np.linspace(float(cand_q_lo), float(cand_q_hi), int(n_candidates))
        cand = np.unique(cp.asnumpy(cp.quantile(xg, cp.asarray(qs))))
        if cand.size == 0:
            return []
        found: list = []
        extra_legs: list = []
        # The candidate relu-leg block and row-guard mask depend only on (xg, cand); hoist them ONCE
        # out of the round loop (they are re-scored against each round's growing design via proj/r_yk).
        R_cand, ok_cand = _fwl_relu_legs(cp, xg, cand, n, min_seg_rows)
        for _round_idx in range(max(1, int(max_breakpoints))):
            if _round_idx == 0:
                # Round 0's fixed design IS B=[1, xg] (extra_legs still empty) -- the SAME block + Gram
                # projector already built for the pre-check above (identical GEMM sequence); reuse instead of
                # re-running it.
                projk = proj
                r_yk = r_y
                sse_B = sse_lin
            else:
                Bk = cp.stack([ones, xg, *extra_legs], axis=1)
                projk = _gram_projector(cp, Bk)
                r_yk = yg - projk(yg)
                sse_B = float(r_yk @ r_yk)
            sse_all = _batched_fwl_sse(cp, R_cand, ok_cand, r_yk, projk, sse_B, cand, found)
            _bi = int(np.argmin(sse_all))
            if not np.isfinite(sse_all[_bi]):
                break
            best_tau = float(cand[_bi])
            uplift = _heldout_uplift_gpu(cp, xg, yg, best_tau, min_rows)
            if uplift < float(min_heldout_r2_uplift):
                break
            found.append(best_tau)
            extra_legs.append(cp.maximum(xg - best_tau, 0.0))
        return found
    except Exception as _exc:
        logger.debug("detect_hinge_breakpoints_gpu failed (%s); host detector fallback", _exc)
        return None
