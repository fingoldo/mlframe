"""GPU-RESIDENT twin of :func:`_usability_aware_selection.usability_greedy` (REGRESSION path).

RESIDENCY CONTRACT (not a wall win). Gated on the resident flag
(``MLFRAME_FE_GPU_STRICT`` + ``MLFRAME_FE_GPU_STRICT_RESIDENT``); default OFF. On
this GTX 1050 Ti the usability greedy runs on a ~3000-row pool subsample, so the
cupy launch + reduction overhead dominates the tiny per-fold linear algebra and
this twin is EXPECTED to be SLOWER than the incremental-bordered numpy/sklearn CPU
path - and that is a PASS by the residency contract. The twin exists to KILL the
per-candidate value D2H/H2D churn the gated ``_usability_gpu`` primitives incur:
the prior GPU usability path (``MLFRAME_FE_GPU_USABILITY``) re-uploaded each
candidate's column and pulled each scalar back PER candidate PER fold PER round
(measured ~628 bulk H2D / ~80 bulk D2H over one F2 100k STRICT retention call). This
twin uploads the candidate value matrix + the target ONCE (one bulk H2D at entry)
and keeps EVERYTHING resident across all rounds/folds: the per-round held-out
residual + the |corr| shortlist (one device GEMV over all candidates), and the
per-candidate K-fold CV-MAE via the SAME incremental bordered normal equations as
the CPU path - pulling back only bounded per-round SCALARS (the shortlist order is
recovered on host from a small (P,) score vector; the per-candidate fold-MAE
matrix is a bounded (shortlist, n_folds) result, NOT per-candidate value data).

What is resident vs host control-flow (allowed by the contract):
  * RESIDENT (one bulk H2D at entry): the (n, P) candidate value matrix ``Vdev``
    (double) and the (n,) target ``ydev``. From them every round computes - on
    device - the selected-set design, the held-out residual, the centered |corr|
    of EVERY candidate vs that residual (the shortlist score), and for the
    shortlisted candidates the bordered (k+1)x(k+1) Gram solve per fold giving the
    per-fold MAE. NO per-candidate H2D and NO per-candidate value D2H.
  * HOST scalar / bounded D2H (allowed by the contract): the (P,) shortlist score
    vector (a per-ROUND result, used only to argsort host-side -> the same
    shortlist the CPU path picks), the (shortlist, n_folds) fold-MAE matrix (a
    per-ROUND result, bounded by ``shortlist`` not by the pool size), and the
    no-selection baseline fold MAEs. These mirror the CPU path's OWN per-round
    scalars; none is bulk per-candidate value data.

SELECTION-EQUIVALENCE IS THE BAR. The algorithm is byte-for-byte the SAME as the
CPU ``usability_greedy``: the same balanced ``arange % k`` fold partition seeded
the same way, the same ``(1-w)*mi/mi_max + w*|corr|`` shortlist pre-rank with the
same stable argsort, the same centered-OLS bordered normal equations (the unique
minimiser sklearn's StandardScaler+LinearRegression also finds), the same
majority-of-folds (>=75%) improvement gate, the same relative-MAE stop. Only the
float reduction ORDER differs between cupy and numpy (~1e-12), to which the
majority-of-folds gate is tolerant in practice (F2 selection-equivalent). NOTE: there
is NO explicit near-tie detector here - the only fall-throughs to the exact CPU path
are the singular-border ``_ResidentFallbackError`` (refits exactly) and any cupy/device
error (returns None); a pathological reduction-order tie that the gate cannot absorb
is a theoretical residual, not a guarded case.

CLASSIFICATION is handled by the resident logistic sibling
:mod:`_usability_greedy_clf_gpu_resident` (an L2 Newton / IRLS on the standardized
bordered design giving the same strictly-convex optimum sklearn's lbfgs finds, scored
by resident CV-logloss). This module's ``classification=True`` entry delegates to that
sibling so each module stays under the LOC budget; both keep the ``None``-on-failure /
flag-off byte-identical CPU contract.

Any cupy / device / import error returns ``None`` so the caller falls back to the
unchanged CPU greedy, keeping the default (flag-off) path byte-identical and a GPU
fault never breaking a fit.
"""
from __future__ import annotations

from mlframe.utils.log_throttle import log_throttle

import logging
import math
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


from ._usability_greedy_gpu_shared import _shrink_shortlist_to_ram_budget


def usability_greedy_gpu_resident(
    pool: list,
    y_cont: np.ndarray,
    *,
    w: float = 0.7,
    K: int = 8,
    seed: int = 0,
    n_folds: int = 4,
    mae_improve_rel: float = 0.01,
    shortlist: int = 40,
    shortlist_diversity_corr: float = 0.97,
    classification: bool = False,
) -> Optional[list]:
    """Resident twin of :func:`_usability_aware_selection.usability_greedy` (regression only).

    Returns the SAME selected ``UsableCandidate`` list (selection-equivalent: same
    indices in the same order) as the CPU greedy, computed with the candidate value
    matrix resident on the GPU. Returns ``None`` - so the caller falls back to the
    exact CPU path - for the classification scorer, an empty/degenerate pool, or
    any cupy/device/import error.

    The signature mirrors the CPU function exactly (the dispatcher forwards every
    kwarg unchanged); the RAM-governor ``K``-shrink the CPU path applies is also
    applied here so the resident selection matches under memory pressure."""
    if classification:
        # CLASSIFICATION is handled by the resident logistic twin (its own module so each stays
        # under the LOC budget). It returns ``None`` for a single-class target, a non-convergent /
        # singular / degenerate fold fit, or any cupy/device error -> the dispatcher then falls
        # through to the exact CPU logistic greedy, keeping flag-off + failures byte-identical CPU.
        try:
            from ._usability_greedy_clf_gpu_resident import usability_greedy_clf_gpu_resident
        except ImportError as e:
            logger.debug("usability_greedy_clf_gpu_resident import failed: %s", e)
            return None
        return usability_greedy_clf_gpu_resident(
            pool, y_cont, w=w, K=K, seed=seed, n_folds=n_folds,
            mae_improve_rel=mae_improve_rel, shortlist=shortlist, shortlist_diversity_corr=shortlist_diversity_corr,
        )
    if not pool:
        return None
    try:
        import cupy as cp
    except ImportError as e:
        logger.debug("cupy import failed, resident usability greedy unavailable: %s", e)
        return None
    try:
        from ._gpu_policy import gpu_globally_disabled
        if gpu_globally_disabled():
            return None
    except Exception as e:  # nosec B110 - optional dependency import guard
        logger.debug("gpu_globally_disabled() check failed, proceeding without the global-disable gate: %s", e)

    try:
        from ._usability_aware_selection import _scrub, _f64

        # Intentional float64: host copy of the float64 producer matrix the device kernel reads back
        y_host = _scrub(np.asarray(y_cont, dtype=np.float64))
        n = int(y_host.shape[0])
        if n < 2:
            return None  # cannot cross-validate on < 2 rows (CPU returns [], but [] != selection here)

        P = len(pool)
        # ---- ONE bulk H2D: the (n, P) candidate value matrix + the (n,) target. Resident hereafter. ----
        # Intentional float64: host copy of the float64 producer matrix the device kernel reads back
        Vhost = np.empty((n, P), dtype=np.float64)
        for j in range(P):
            col = _f64(pool[j].values)
            if col.shape[0] != n:
                return None  # ragged pool -> let the CPU path handle it exactly
            Vhost[:, j] = col
        Vdev = cp.asarray(Vhost)  # (n, P) resident
        ydev = cp.asarray(y_host)  # (n,) resident

        # RAM GOVERNOR (mirror the CPU path so the resident selection matches under memory pressure). The
        # CPU greedy caps K (and the shortlist with it) to the largest (n, K) float64 design the FE buffer
        # budget allows; replicate that here so a memory-pressured host selects the SAME set on both paths.
        K, shortlist = _shrink_shortlist_to_ram_budget(K, P, n, shortlist)

        # Balanced ``arange % k`` partition, seeded + shuffled IDENTICALLY to the CPU path (host RNG -> the
        # SAME fold vector, so the per-fold train/val splits match bit-for-bit).
        rng = np.random.default_rng(int(seed))
        nf = max(2, min(int(n_folds), n))
        folds_host = np.arange(n) % nf
        rng.shuffle(folds_host)
        folds_dev = cp.asarray(folds_host)
        mi_max = max((c.mi for c in pool), default=1.0)
        if mi_max == 0:
            mi_max = 1.0
        # Intentional float64: statistic vector (MI / score / fold error), exact ties decide the selection
        mi_dev = cp.asarray(np.asarray([float(c.mi) for c in pool], dtype=np.float64))

        # Resident per-fold train/val ROW INDICES (n,) - reused every round. Integer indices, not boolean
        # masks: a boolean-mask gather Vdev[mask] / ci[mask] re-syncs on the mask nonzero count for EVERY
        # candidate and fold (the dominant residency stall in this path), whereas an integer-index gather has a
        # known output size (no sync). The cp.where cost is paid once here (2*nf syncs) instead of per gather.
        tr_masks = [cp.where(folds_dev != fo)[0] for fo in range(nf)]
        va_masks = [cp.where(folds_dev == fo)[0] for fo in range(nf)]

        _abscorr_batch_resident = _usability_greedy_gpu_step1_known_output_size(Vdev, cp, P)

        _cv_baseline, _shortlist = _usability_greedy_gpu_step2_def_shortlist_sel(va_masks, tr_masks, Vdev, ydev, _abscorr_batch_resident, cp, mi_dev, P, w, mi_max, shortlist_diversity_corr, shortlist, nf)

        _cv_candidates = _usability_greedy_gpu_step3_def_cv_candidates(nf, tr_masks, va_masks, ydev, Vdev, cp)

        min_improving_folds = max(1, math.ceil(0.75 * nf))
        selected: list = []
        folds_cur = _cv_baseline()
        mae_cur = float(folds_cur.mean())
        _usability_greedy_g_range_min(K, P, _shortlist, selected, mae_cur, folds_cur, _cv_candidates, min_improving_folds, mae_improve_rel)
        return [pool[i] for i in selected]
    except _ResidentFallbackError as e:
        log_throttle(logger, "usability_greedy_gpu_resident.fallback", logging.WARNING, "usability_greedy_gpu_resident: singular border / degenerate device fit, falling back to the exact CPU greedy: %s", e)
        return None
    except Exception as e:
        logger.warning("usability_greedy_gpu_resident: cupy/device error, falling back to the exact CPU greedy: %s", e)
        return None


_ABSCORR_TAIL_KERNEL = None
_ABSCORR_CACHE_MAX_BYTES = 256 * 1024 * 1024  # the centered candidate matrix is kept per row set while it fits


def _abscorr_tail_kernel(cp):
    """Elementwise kernel of the last steps of the |corr| score: ``r = num / sqrt(ssc * ssv)`` where the column and the residual are usable, else 0, then ``|r|`` with non-finite -> 0 (one launch
    where the ``where`` / ``isfinite`` / ``abs`` chain was ten)."""
    global _ABSCORR_TAIL_KERNEL
    if _ABSCORR_TAIL_KERNEL is None:
        _ABSCORR_TAIL_KERNEL = cp.ElementwiseKernel(
            "float64 num, float64 ssc, float64 ssv, bool col_ok, bool resid_ok",  # Intentional float64: statistic vector, exact ties decide the selection
            "float64 out",  # Intentional float64: statistic vector, exact ties decide the selection
            "double denom = sqrt(ssc * ssv); double r = (col_ok && resid_ok && ssc > 0.0 && denom > 0.0) ? num / denom : 0.0; out = isfinite(r) ? fabs(r) : 0.0;",
            "usability_abscorr_tail",
        )
    return _ABSCORR_TAIL_KERNEL


def _usability_greedy_gpu_step1_known_output_size(Vdev, cp, P):
    """Step 1 of usability_greedy_gpu_resident: lines starting at ``def _abscorr_batch_resident(resid_dev, rows_mask) -> np.ndarray:``."""
    col_stats: dict = {}  # id(row set) -> (row set, centered candidate matrix, column ok mask): the candidates do not change between calls, the residual does

    def _abscorr_batch_resident_reference(resid_dev, rows_mask) -> np.ndarray:
        """|corr(candidate_j, resid)| for EVERY candidate over ``rows_mask`` rows, on device. Mirrors
        the per-candidate ``_abscorr`` (centered dot / sqrt(ss), std<1e-12 -> 0) batched into one GEMV.
        Returns a host (P,) double score vector (a per-ROUND result, not per-candidate value data)."""
        if rows_mask is None:
            M = Vdev
            rv = resid_dev
        else:
            M = Vdev[rows_mask]  # (m, P) resident view-copy
            rv = resid_dev
        m = int(M.shape[0])
        # Intentional float64: statistic vector (MI / score / fold error), exact ties decide the selection
        out = cp.zeros(P, dtype=cp.float64)
        if m == 0:
            return np.asarray(cp.asnumpy(out))
        col_std = M.std(axis=0)  # (P,)
        # The two degenerate-residual guards are folded into the device mask instead of read back as Python floats (two stream stalls per call, ~100 per
        # fit): a flat or zero-energy residual zeroes every score, exactly as the early returns did.
        v_std = rv.std()
        vm = rv - rv.mean()
        ssv = cp.dot(vm, vm)
        resid_ok = (v_std >= 1e-12) & (ssv > 0.0)
        Mc = M - M.mean(axis=0, keepdims=True)
        num = Mc.T @ vm  # (P,) centered dot
        ssc = (Mc * Mc).sum(axis=0)  # (P,)
        denom = cp.sqrt(ssc * ssv)
        valid = (col_std >= 1e-12) & (ssc > 0.0) & (denom > 0.0) & resid_ok
        r = cp.where(valid, num / cp.where(denom > 0.0, denom, 1.0), 0.0)
        r = cp.where(cp.isfinite(r), cp.abs(r), 0.0)
        return np.asarray(cp.asnumpy(r))

    def _abscorr_batch_resident(resid_dev, rows_mask) -> np.ndarray:
        """:func:`_abscorr_batch_resident_reference` with the candidate side computed once per row set (the diversity filter and every round score the same candidates against a new residual: the
        gather, the column standard deviations, the centred matrix and its squared sums were recomputed on every call) and the tail as one kernel; the same arithmetic, so the same scores."""
        key = id(rows_mask) if rows_mask is not None else None
        entry = col_stats.get(key)
        if entry is None or (rows_mask is not None and entry[0] is not rows_mask):
            M = Vdev if rows_mask is None else Vdev[rows_mask]
            if int(M.shape[0]) == 0:
                return np.zeros(P, dtype=np.float64)  # Intentional float64: statistic vector
            Mc = M - M.mean(axis=0, keepdims=True)
            ssc = (Mc * Mc).sum(axis=0)
            col_ok = (M.std(axis=0) >= 1e-12) & (ssc > 0.0)
            entry = (rows_mask, Mc, ssc, col_ok)
            if int(Mc.nbytes) <= _ABSCORR_CACHE_MAX_BYTES:
                col_stats[key] = entry
        _rows, Mc, ssc, col_ok = entry
        rv = resid_dev
        v_std = rv.std()
        vm = rv - rv.mean()
        ssv = cp.dot(vm, vm)
        resid_ok = (v_std >= 1e-12) & (ssv > 0.0)
        num = Mc.T @ vm
        return np.asarray(cp.asnumpy(_abscorr_tail_kernel(cp)(num, ssc, ssv, col_ok, resid_ok)))

    _abscorr_batch_resident.reference = _abscorr_batch_resident_reference  # type: ignore[attr-defined]
    return _abscorr_batch_resident


def _usability_greedy_gpu_step2_def_shortlist_sel(va_masks, tr_masks, Vdev, ydev, _abscorr_batch_resident, cp, mi_dev, P, w, mi_max, shortlist_diversity_corr, shortlist, nf):
    """Step 2 of usability_greedy_gpu_resident: lines starting at ``def _shortlist(sel_idx) -> list:``."""
    def _shortlist(sel_idx) -> list:
        """Rank unselected candidates by |corr| with the fold-0 held-out residual of the currently-selected set (leakage-safe: fit on fold-0 train rows only), then return the top-scored candidates for the greedy round; raises ``_ResidentFallbackError`` on a singular selected design so the caller falls back to the exact CPU path rather than risk a wrong selection."""
        # HELD-OUT residual on fold-0 (mirrors the CPU path's leakage-safe design): fit the selected set
        # on the ~fold-0 train rows, score the |corr| on the held-out fold-0 residual. No selection ->
        # the mean residual over all rows (no fit -> no leakage).
        if sel_idx:
            ho = va_masks[0]
            tr = tr_masks[0]
            beta, ybar, mu = _fit_selected(sel_idx, tr)
            if beta is None:
                # singular selected design on the train fold: fall back to the exact CPU shortlist for
                # this round (selection correctness never depends on the fast path).
                raise _ResidentFallbackError()
            Sho = Vdev[ho][:, sel_idx]
            pred = ybar + (Sho - mu) @ beta
            resid = ydev[ho] - pred
            uses = _abscorr_batch_resident(resid, ho)
        else:
            resid = ydev - ydev.mean()
            uses = _abscorr_batch_resident(resid, None)
        sel_set = set(sel_idx)
        scored = []
        mi_host = cp.asnumpy(mi_dev)
        # `uses` is already a host numpy array here -
        # _abscorr_batch_resident is type-hinted -> np.ndarray and returns np.asarray(cp.asnumpy(out))
        # on every branch - so the `cp.asnumpy(uses) if hasattr(...) else ...` dead branch (left over
        # from an earlier refactor where the function may have returned a device array) always took the
        # no-op np.asarray(uses) path. Removed; uses is used directly.
        uses_host = uses
        for i in range(P):
            if i in sel_set:
                continue
            use = float(uses_host[i])
            scored.append((i, (1.0 - w) * (mi_host[i] / mi_max) + w * use))
        scored.sort(key=lambda t: t[1], reverse=True)
        # Diversity filter (mirrors the CPU usability_greedy._shortlist): several top-scored
        # candidates can be near-duplicate views of the SAME underlying signal and would
        # otherwise occupy multiple shortlist slots, starving room for a genuinely different
        # candidate. Reuses _abscorr_batch_resident (one GEMV per KEPT candidate, computing every
        # remaining candidate's |corr| against it in one device pass) rather than a per-pair CPU
        # loop, to stay resident. Bounded cost: at most `shortlist` GEMVs per greedy round.
        out_idx: list[int] = []
        corr_vs_kept: list[np.ndarray] = []
        for i, _s in scored:
            if any(float(cvk[i]) > shortlist_diversity_corr for cvk in corr_vs_kept):
                continue
            out_idx.append(i)
            if len(out_idx) >= max(1, shortlist):
                break
            corr_vs_kept.append(_abscorr_batch_resident(Vdev[:, i], None))
        return out_idx

    def _fit_selected(sel_idx, tr_mask):
        """Centered-OLS fit of the SELECTED set on ``tr_mask`` rows. Returns (beta (k,), ybar, mu (k,))
        all resident, or (None, None, None) if the centered Gram is singular (caller falls back)."""
        Str = Vdev[tr_mask][:, sel_idx]
        ytr = ydev[tr_mask]
        ybar = ytr.mean()
        yc = ytr - ybar
        mu = Str.mean(axis=0)
        Sc = Str - mu
        G = Sc.T @ Sc
        b = Sc.T @ yc
        try:
            beta = cp.linalg.solve(G, b)
        except Exception as e:
            logger.debug("usability_greedy_gpu_resident._fit_selected: centered Gram is singular, caller falls back to the CPU refit: %s", e)
            return None, None, None
        if not bool(cp.all(cp.isfinite(beta))):
            return None, None, None  # cupy's solve does not raise on a singular matrix: it returns inf / nan
        return beta, ybar, mu

    def _cv_baseline() -> np.ndarray:
        """No-selection per-fold MAE: predict each val fold by its train-fold mean. Resident; returns a
        host (nf,) vector (the SAME quantity the CPU ``_cv_per_fold([])`` computes)."""
        # Intentional float64: statistic vector (MI / score / fold error), exact ties decide the selection
        errs = np.empty(nf, dtype=np.float64)
        for fo in range(nf):
            tr, va = tr_masks[fo], va_masks[fo]
            m = ydev[tr].mean()
            errs[fo] = float(cp.mean(cp.abs(ydev[va] - m)))
        return errs
    return _cv_baseline, _shortlist


def _usability_greedy_gpu_step3_def_cv_candidates(nf, tr_masks, va_masks, ydev, Vdev, cp):
    """Step 3 of usability_greedy_gpu_resident: lines starting at ``def _cv_candidates(sel_idx, cand_list) -> dict:``."""
    def _cv_candidates(sel_idx, cand_list) -> dict:
        """The batched bordered-normal-equation CV for the whole shortlist (:func:`_cv_candidates_block`)."""
        return _cv_candidates_block(cp, Vdev, tr_masks, va_masks, ydev, nf, sel_idx, cand_list)

    return _cv_candidates


def _bordered_fold_pred(cp, Vdev, tr, va, sel_idx, ybar, yc, cc_tr, cc_va, d, bnew, m):
    """Validation predictions of the ``m`` candidates of one fold when ``sel_idx`` is already selected: the ``m`` bordered normal systems solved in one batched ``cp.linalg.solve``; a failing solve raises ``_ResidentFallbackError``."""
    Str = Vdev[tr][:, sel_idx]
    mu = Str.mean(axis=0)
    Sc_tr = Str - mu
    Sc_va = Vdev[va][:, sel_idx] - mu
    Gs = Sc_tr.T @ Sc_tr
    bs = Sc_tr.T @ yc
    g = Sc_tr.T @ cc_tr  # (k, m)
    k = int(Gs.shape[0])
    # Intentional float64: normal equations solved in double, as the host solve this device path mirrors
    G = cp.empty((m, k + 1, k + 1), dtype=cp.float64)
    G[:, :k, :k] = Gs
    G[:, :k, k] = g.T
    G[:, k, :k] = g.T
    G[:, k, k] = d
    rhs = cp.empty((m, k + 1), dtype=cp.float64)  # Intentional float64: normal equations solved in double
    rhs[:, :k] = bs
    rhs[:, k] = bnew
    try:
        beta = cp.linalg.solve(G, rhs[:, :, None])[:, :, 0]
    except Exception as e:
        log_throttle(logger, "usability_greedy_gpu_resident.hot_loop_warning", logging.WARNING, "cp.linalg.solve failed (likely a singular Gram matrix), stopping the greedy: %s", e)
        raise _ResidentFallbackError() from e
    pred = ybar + Sc_va @ beta[:, :k].T + cc_va * beta[:, k]
    return pred


def _cv_candidates_block(cp, Vdev, tr_masks, va_masks, ydev, nf, sel_idx, cand_list) -> dict:
    """:func:`_cv_candidates_loop` for the whole shortlist at once: per fold the candidates are one ``(n, m)`` block, their borders are ``m`` systems of one batched solve and the fold errors one
    reduction, instead of about 25 small launches per (candidate, fold). The values are the loop's (reductions in another order, so equal to rounding); a singular border or a non-finite
    solution raises ``_ResidentFallbackError`` as the loop does."""
    cand_list = list(cand_list)
    if not cand_list:
        return {}
    m = len(cand_list)
    C = Vdev[:, cp.asarray(np.asarray(cand_list, dtype=np.int64))]  # (n, m)
    errs = cp.empty((nf, m), dtype=cp.float64)  # Intentional float64: statistic vector (fold error), exact ties decide the selection
    mind = None
    for fo in range(nf):
        tr, va = tr_masks[fo], va_masks[fo]
        ytr = ydev[tr]
        ybar = ytr.mean()
        yc = ytr - ybar
        yva = ydev[va]
        ctr = C[tr]
        cmu = ctr.mean(axis=0)
        cc_tr = ctr - cmu
        cc_va = C[va] - cmu
        d = (cc_tr * cc_tr).sum(axis=0)
        bnew = yc @ cc_tr
        if not sel_idx:
            mind = d if mind is None else cp.minimum(mind, d)
            pred = ybar + cc_va * (bnew / d)
        else:
            pred = _bordered_fold_pred(cp, Vdev, tr, va, sel_idx, ybar, yc, cc_tr, cc_va, d, bnew, m)
        errs[fo] = cp.mean(cp.abs(yva[:, None] - pred), axis=0)
    if mind is None:
        host = cp.asnumpy(errs)  # (nf, m)
        if not np.isfinite(host).all():
            raise _ResidentFallbackError()
    else:
        both = cp.asnumpy(cp.concatenate([errs.ravel(), mind.astype(cp.float64)]))  # ONE D2H: the errors and the round-1 borders. Intentional float64: the errors are double, so the borders join them at double
        host = both[: nf * m].reshape(nf, m)
        if float(both[nf * m :].min()) <= 1e-12:
            raise _ResidentFallbackError()
    by_cand = np.ascontiguousarray(host.T)
    return {i: by_cand[r] for r, i in enumerate(cand_list)}


def _usability_greedy_g_range_min(K, P, _shortlist, selected, mae_cur, folds_cur, _cv_candidates, min_improving_folds, mae_improve_rel):
    """Block of usability_greedy_gpu_resident starting at ``for _ in range(min(K, P)):``."""
    for _ in range(min(K, P)):
        cand_idx = _shortlist(selected)
        best_i, best_mean, best_folds = -1, mae_cur, folds_cur
        mf_by_i = _cv_candidates(selected, cand_idx)
        for i in cand_idx:
            mf = mf_by_i[i]
            if int(np.sum(mf < folds_cur)) < min_improving_folds:
                continue
            if float(mf.mean()) < best_mean:
                best_mean, best_i, best_folds = float(mf.mean()), i, mf
        if best_i < 0 or best_mean >= mae_cur * (1.0 - mae_improve_rel):
            break
        selected.append(best_i)
        folds_cur, mae_cur = best_folds, best_mean


class _ResidentFallbackError(Exception):
    """Internal signal: a singular/degenerate device fit was hit -> fall back to the exact CPU greedy
    (which has its own per-candidate refit fallback), so selection is never decided by the fast path."""


__all__ = ["usability_greedy_gpu_resident"]
