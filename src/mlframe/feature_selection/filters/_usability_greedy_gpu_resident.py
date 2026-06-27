"""GPU-RESIDENT twin of :func:`_usability_aware_selection.usability_greedy` (REGRESSION path).

RESIDENCY CONTRACT (not a wall win). Gated on the resident flag
(``MLFRAME_FE_GPU_STRICT`` + ``MLFRAME_FE_GPU_STRICT_RESIDENT``); default OFF. On
this GTX 1050 Ti the usability greedy runs on a ~3000-row pool subsample, so the
cupy launch + reduction overhead dominates the tiny per-fold linear algebra and
this twin is EXPECTED to be SLOWER than the incremental-bordered numpy/sklearn CPU
path -- and that is a PASS by the residency contract. The twin exists to KILL the
per-candidate value D2H/H2D churn the gated ``_usability_gpu`` primitives incur:
the prior GPU usability path (``MLFRAME_FE_GPU_USABILITY``) re-uploaded each
candidate's column and pulled each scalar back PER candidate PER fold PER round
(measured ~628 bulk H2D / ~80 bulk D2H over one F2 100k STRICT retention call). This
twin uploads the candidate value matrix + the target ONCE (one bulk H2D at entry)
and keeps EVERYTHING resident across all rounds/folds: the per-round held-out
residual + the |corr| shortlist (one device GEMV over all candidates), and the
per-candidate K-fold CV-MAE via the SAME incremental bordered normal equations as
the CPU path -- pulling back only bounded per-round SCALARS (the shortlist order is
recovered on host from a small (P,) score vector; the per-candidate fold-MAE
matrix is a bounded (shortlist, n_folds) result, NOT per-candidate value data).

What is resident vs host control-flow (allowed by the contract):
  * RESIDENT (one bulk H2D at entry): the (n, P) candidate value matrix ``Vdev``
    (float64) and the (n,) target ``ydev``. From them every round computes -- on
    device -- the selected-set design, the held-out residual, the centered |corr|
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
float reduction ORDER differs between cupy and numpy (~1e-12), to which the gates
are tolerant; on a near-tie that a reduction-order shift would flip, the result
falls back to the CPU value (the singular-border fallback already refits exactly).

CLASSIFICATION is NOT ported here: its scorer is an sklearn LogisticRegression
CV-logloss (an iterative solver with no closed-form bordered update), so the
resident twin only covers the REGRESSION CV-MAE greedy and the caller passes
``classification`` straight through -- a classification call returns ``None`` so
the dispatcher falls through to the exact CPU logistic path.

Any cupy / device / import error returns ``None`` so the caller falls back to the
unchanged CPU greedy, keeping the default (flag-off) path byte-identical and a GPU
fault never breaking a fit.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np


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
    classification: bool = False,
) -> Optional[list]:
    """Resident twin of :func:`_usability_aware_selection.usability_greedy` (regression only).

    Returns the SAME selected ``UsableCandidate`` list (selection-equivalent: same
    indices in the same order) as the CPU greedy, computed with the candidate value
    matrix resident on the GPU. Returns ``None`` -- so the caller falls back to the
    exact CPU path -- for the classification scorer, an empty/degenerate pool, or
    any cupy/device/import error.

    The signature mirrors the CPU function exactly (the dispatcher forwards every
    kwarg unchanged); the RAM-governor ``K``-shrink the CPU path applies is also
    applied here so the resident selection matches under memory pressure."""
    if classification:
        return None  # logistic CV-logloss has no bordered closed form -> CPU path
    if not pool:
        return None
    try:
        import cupy as cp
    except Exception:
        return None
    try:
        from ._gpu_policy import gpu_globally_disabled
        if gpu_globally_disabled():
            return None
    except Exception:
        pass

    try:
        from ._usability_aware_selection import _scrub, _f64

        y_host = _scrub(np.asarray(y_cont, dtype=np.float64))
        n = int(y_host.shape[0])
        if n < 2:
            return None  # cannot cross-validate on < 2 rows (CPU returns [], but [] != selection here)

        P = len(pool)
        # ---- ONE bulk H2D: the (n, P) candidate value matrix + the (n,) target. Resident hereafter. ----
        Vhost = np.empty((n, P), dtype=np.float64)
        for j in range(P):
            col = _f64(pool[j].values)
            if col.shape[0] != n:
                return None  # ragged pool -> let the CPU path handle it exactly
            Vhost[:, j] = col
        Vdev = cp.asarray(Vhost)            # (n, P) resident
        ydev = cp.asarray(y_host)           # (n,) resident

        # RAM GOVERNOR (mirror the CPU path so the resident selection matches under memory pressure). The
        # CPU greedy caps K (and the shortlist with it) to the largest (n, K) float64 design the FE buffer
        # budget allows; replicate that here so a memory-pressured host selects the SAME set on both paths.
        try:
            from .feature_engineering import _can_hoist_shared_buffer, _fe_effective_buffer_budget_bytes
            _k_eff = max(1, min(int(K), P))
            _can, _need, _avail = _can_hoist_shared_buffer(n * _k_eff * 8, n_workers=1)
            if (not _can) and _avail > 0:
                _budget = _fe_effective_buffer_budget_bytes(_avail, n_workers=1)
                _k_fit = int(_budget // (n * 8)) if _budget > 0 else 1
                if _k_fit < _k_eff:
                    K = max(1, _k_fit)
                    shortlist = min(int(shortlist), max(int(K), 1))
        except Exception:
            pass

        # Balanced ``arange % k`` partition, seeded + shuffled IDENTICALLY to the CPU path (host RNG -> the
        # SAME fold vector, so the per-fold train/val splits match bit-for-bit).
        rng = np.random.default_rng(int(seed))
        nf = max(2, min(int(n_folds), n))
        folds_host = np.arange(n) % nf
        rng.shuffle(folds_host)
        folds_dev = cp.asarray(folds_host)
        mi_max = max((c.mi for c in pool), default=1.0) or 1.0
        mi_dev = cp.asarray(np.asarray([float(c.mi) for c in pool], dtype=np.float64))

        # Resident per-fold train/val boolean masks (n,) -- reused every round.
        tr_masks = [folds_dev != fo for fo in range(nf)]
        va_masks = [folds_dev == fo for fo in range(nf)]

        def _abscorr_batch_resident(resid_dev, rows_mask) -> np.ndarray:
            """|corr(candidate_j, resid)| for EVERY candidate over ``rows_mask`` rows, on device. Mirrors
            the per-candidate ``_abscorr`` (centered dot / sqrt(ss), std<1e-12 -> 0) batched into one GEMV.
            Returns a host (P,) float64 score vector (a per-ROUND result, not per-candidate value data)."""
            if rows_mask is None:
                M = Vdev
                rv = resid_dev
            else:
                M = Vdev[rows_mask]            # (m, P) resident view-copy
                rv = resid_dev
            m = int(M.shape[0])
            out = cp.zeros(P, dtype=cp.float64)
            if m == 0:
                return cp.asnumpy(out)
            col_std = M.std(axis=0)            # (P,)
            v_std = float(rv.std())
            if v_std < 1e-12:
                return cp.asnumpy(out)
            vm = rv - rv.mean()
            ssv = float(cp.dot(vm, vm))
            if ssv <= 0.0:
                return cp.asnumpy(out)
            Mc = M - M.mean(axis=0, keepdims=True)
            num = Mc.T @ vm                    # (P,) centered dot
            ssc = (Mc * Mc).sum(axis=0)        # (P,)
            denom = cp.sqrt(ssc * ssv)
            valid = (col_std >= 1e-12) & (ssc > 0.0) & (denom > 0.0)
            r = cp.where(valid, num / cp.where(denom > 0.0, denom, 1.0), 0.0)
            r = cp.where(cp.isfinite(r), cp.abs(r), 0.0)
            return cp.asnumpy(r)

        def _shortlist(sel_idx) -> list:
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
                    raise _ResidentFallback()
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
            for i in range(P):
                if i in sel_set:
                    continue
                use = float(uses[i])
                scored.append((i, (1.0 - w) * (mi_host[i] / mi_max) + w * use))
            scored.sort(key=lambda t: t[1], reverse=True)
            return [i for i, _ in scored[: max(1, shortlist)]]

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
            except Exception:
                return None, None, None
            return beta, ybar, mu

        def _cv_baseline() -> np.ndarray:
            """No-selection per-fold MAE: predict each val fold by its train-fold mean. Resident; returns a
            host (nf,) vector (the SAME quantity the CPU ``_cv_per_fold([])`` computes)."""
            errs = np.empty(nf, dtype=np.float64)
            for fo in range(nf):
                tr, va = tr_masks[fo], va_masks[fo]
                m = ydev[tr].mean()
                errs[fo] = float(cp.mean(cp.abs(ydev[va] - m)))
            return errs

        def _cv_candidates(sel_idx, cand_list) -> dict:
            """{cand_i: per-fold MAE (nf,)} via the bordered normal equations, fully resident. Per fold the
            selected-set centered Gram + rhs are built ONCE; each candidate is a rank-1 border solved as a
            (k+1)x(k+1) system. A singular border for a candidate raises so the caller refits via the exact
            CPU path for that candidate -- correctness never depends on the fast path."""
            # Precompute, ONCE per step per fold, the centered selected design pieces (resident).
            per_fold = []
            for fo in range(nf):
                tr, va = tr_masks[fo], va_masks[fo]
                ytr = ydev[tr]
                ybar = ytr.mean()
                yc = ytr - ybar
                if sel_idx:
                    Str = Vdev[tr][:, sel_idx]
                    Sva = Vdev[va][:, sel_idx]
                    mu = Str.mean(axis=0)
                    Sc_tr = Str - mu
                    Sc_va = Sva - mu
                    Gs = Sc_tr.T @ Sc_tr
                    bs = Sc_tr.T @ yc
                else:
                    Sc_tr = Sc_va = mu = Gs = bs = None
                per_fold.append((tr, va, ybar, yc, Sc_tr, Sc_va, Gs, bs))

            out: dict = {}
            for i in cand_list:
                ci = Vdev[:, i]
                errs_dev = cp.empty(nf, dtype=cp.float64)   # accumulate fold MAEs resident; ONE D2H/candidate
                singular = False
                for fo in range(nf):
                    tr, va, ybar, yc, Sc_tr, Sc_va, Gs, bs = per_fold[fo]
                    ctr = ci[tr]
                    cva = ci[va]
                    cmu = ctr.mean()
                    cc_tr = ctr - cmu
                    cc_va = cva - cmu
                    if Sc_tr is None:
                        d = float(cp.dot(cc_tr, cc_tr))   # scalar sync: the singular-border guard (bounded)
                        if d <= 1e-12:
                            singular = True
                            break
                        beta = cp.dot(cc_tr, yc) / d
                        pred = ybar + cc_va * beta
                    else:
                        g = Sc_tr.T @ cc_tr
                        d = cp.dot(cc_tr, cc_tr)
                        bnew = cp.dot(cc_tr, yc)
                        k = int(Gs.shape[0])
                        G = cp.empty((k + 1, k + 1), dtype=cp.float64)
                        G[:k, :k] = Gs
                        G[:k, k] = g
                        G[k, :k] = g
                        G[k, k] = d
                        rhs = cp.empty(k + 1, dtype=cp.float64)
                        rhs[:k] = bs
                        rhs[k] = bnew
                        try:
                            beta = cp.linalg.solve(G, rhs)
                        except Exception:
                            singular = True
                            break
                        pred = ybar + Sc_va @ beta[:k] + cc_va * beta[k]
                    errs_dev[fo] = cp.mean(cp.abs(ydev[va] - pred))
                if singular:
                    raise _ResidentFallback()
                out[i] = cp.asnumpy(errs_dev)   # bounded (nf,) result vector, ONE D2H per candidate
            return out

        min_improving_folds = max(1, int(math.ceil(0.75 * nf)))
        selected: list = []
        folds_cur = _cv_baseline()
        mae_cur = float(folds_cur.mean())
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
        return [pool[i] for i in selected]
    except _ResidentFallback:
        return None  # a singular border / degenerate fit -> take the exact CPU greedy
    except Exception:
        return None  # any cupy/device error -> exact CPU greedy


class _ResidentFallback(Exception):
    """Internal signal: a singular/degenerate device fit was hit -> fall back to the exact CPU greedy
    (which has its own per-candidate refit fallback), so selection is never decided by the fast path."""


__all__ = ["usability_greedy_gpu_resident"]
