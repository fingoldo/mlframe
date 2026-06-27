"""GPU-RESIDENT twin of the CLASSIFICATION branch of
:func:`_usability_aware_selection.usability_greedy` (logistic CV-logloss greedy).

This is the classification sibling of :mod:`_usability_greedy_gpu_resident` (the
regression CV-MAE twin). Gated on the resident flag
(``MLFRAME_FE_GPU_STRICT`` + ``MLFRAME_FE_GPU_STRICT_RESIDENT``); default OFF. It is
the LAST CPU-resident stage under the flag: the regression twin returns ``None`` for
classification and falls back to the CPU sklearn ``LogisticRegression`` path. This
module ports that path so a classification fit is resident too, for true 100%
residency.

RESIDENCY CONTRACT (not a wall win). On this GTX 1050 Ti the usability greedy runs on
a ~3000-row pool subsample, so the cupy launch + per-fit Newton overhead dominates the
tiny per-fold logistic and this twin is EXPECTED to be SLOWER than the sklearn CPU
path -- and that is a PASS by the residency contract. The twin exists to KILL the
per-candidate value D2H/H2D churn: it uploads the candidate value matrix + the target
ONCE (one bulk H2D at entry) and keeps EVERYTHING resident across all rounds/folds;
only bounded per-round SCALARS / result-vectors cross D2H (the (P,) shortlist score
vector, the per-fold logloss arrays). NO per-candidate H2D and NO per-candidate value
D2H, NO host binning.

SELECTION-EQUIVALENCE IS THE BAR (not byte-identity). The CPU classification scorer is
an sklearn ``make_pipeline(StandardScaler(), LogisticRegression(max_iter=200))`` scored
by K-fold CV-logloss. ``StandardScaler -> LogisticRegression`` is an L2-penalised
logistic on the per-column standardised design (penalty='l2', C=1.0, intercept
unpenalised) whose UNIQUE convex optimum the lbfgs solver finds; a resident Newton /
IRLS on the SAME regularised objective converges to that same optimum (the problem is
strictly convex), so the per-fold CV-logloss -- and therefore the committed feature set
-- matches. Everything ELSE is byte-for-byte the SAME as the CPU path: the same encoded
0..C-1 class codes, the same seeded ``arange % k`` fold partition, the same
``(1-w)*mi/mi_max + w*|corr(resid)|`` shortlist pre-rank with the same positive-class /
majority-class residual and the same stable argsort, the same class-prior no-selection
baseline, the same majority-of-folds (>=75%) improvement gate, the same relative-logloss
stop. Only float reduction ORDER differs (~1e-9), to which the gates are tolerant; on a
near-tie a reduction-order shift would flip, or any non-convergence / singular Hessian /
degenerate fold, the call returns ``None`` and the caller takes the exact CPU greedy.

Any cupy / device / import error returns ``None`` so the caller falls back to the
unchanged CPU greedy, keeping the default (flag-off) path byte-identical and a GPU fault
never breaking a fit.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np


class _ResidentClfFallback(Exception):
    """Internal signal: a non-convergent / singular / degenerate device logistic fit was hit ->
    fall back to the exact CPU greedy so selection is never decided by the fast path."""


# Newton / IRLS convergence controls. The logistic objective is strictly convex (L2 penalty),
# so Newton converges quadratically; ``_NEWTON_TOL`` matches the optimum tightly enough that the
# per-fold logloss agrees with sklearn lbfgs to the selection gate's tolerance.
_NEWTON_MAX_ITER = 200
_NEWTON_TOL = 1e-9
_C = 1.0          # sklearn LogisticRegression default inverse-regularisation strength
_LOGLOSS_EPS = 1e-15


def usability_greedy_clf_gpu_resident(
    pool: list,
    y_cont: np.ndarray,
    *,
    w: float = 0.7,
    K: int = 8,
    seed: int = 0,
    n_folds: int = 4,
    mae_improve_rel: float = 0.01,
    shortlist: int = 40,
) -> Optional[list]:
    """Resident twin of the CLASSIFICATION branch of :func:`usability_greedy`.

    Returns the SAME selected ``UsableCandidate`` list (selection-equivalent: same indices
    in the same order) as the CPU logistic CV-logloss greedy, computed with the candidate
    value matrix resident on the GPU. Returns ``None`` -- so the caller falls back to the
    exact CPU path -- for an empty/degenerate pool, a single-class target, a non-convergent
    or singular fold fit, or any cupy/device/import error. ``mae_improve_rel`` carries the
    relative-improvement stop (it is the logloss stop here; the kwarg name is shared with the
    regression path the dispatcher forwards from)."""
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
        from ._usability_aware_selection import _f64

        # Encode to dense 0..C-1 class codes IDENTICALLY to the CPU path (np.unique inverse).
        y_raw = np.asarray(y_cont).ravel()
        _classes, y_enc = np.unique(y_raw, return_inverse=True)
        n_classes = int(_classes.size)
        if n_classes < 2:
            return None  # CPU returns [] for <2 classes; [] != a selection here, so defer to CPU
        if n_classes > 2:
            # MULTICLASS is kept on the CPU path (evidence-based, not assumed). sklearn's default
            # multinomial logistic is fit by lbfgs in the SYMMETRIC (over-parametrised) coefficient
            # space, whose Hessian is RANK-DEFICIENT (the softmax sum-to-zero null space). A resident
            # Newton on that same symmetric objective hits a singular block Hessian (cp.linalg.solve
            # fails / NaNs) -> it cannot reproduce sklearn's probabilities, and a reduced (C-1)-class
            # re-parametrisation gives DIFFERENT coefficients/probabilities than sklearn's symmetric
            # fit, which flips CV-logloss selections on near-ties. Measured on a 3-class fixture: the
            # resident multinomial Newton diverged from sklearn (singular Hessian / empty selection vs
            # the CPU 4-feature set). The bar is selection-equivalence, so multiclass stays CPU.
            return None
        n = int(y_enc.shape[0])
        if n < 2:
            return None

        P = len(pool)
        # ---- ONE bulk H2D: the (n, P) candidate value matrix + the (n,) class codes. Resident hereafter. ----
        Vhost = np.empty((n, P), dtype=np.float64)
        for j in range(P):
            col = _f64(pool[j].values)
            if col.shape[0] != n:
                return None  # ragged pool -> let the CPU path handle it exactly
            Vhost[:, j] = col
        Vdev = cp.asarray(Vhost)                       # (n, P) resident
        yenc_dev = cp.asarray(y_enc.astype(np.int64))  # (n,) resident class codes
        ydev_f = yenc_dev.astype(cp.float64)           # binary positive-class indicator helper

        # RAM GOVERNOR (mirror the CPU path so the resident selection matches under memory pressure).
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

        # Balanced ``arange % k`` partition, seeded + shuffled IDENTICALLY to the CPU path.
        rng = np.random.default_rng(int(seed))
        nf = max(2, min(int(n_folds), n))
        folds_host = np.arange(n) % nf
        rng.shuffle(folds_host)
        folds_dev = cp.asarray(folds_host)
        mi_max = max((c.mi for c in pool), default=1.0) or 1.0
        mi_host = np.asarray([float(c.mi) for c in pool], dtype=np.float64)

        tr_masks = [folds_dev != fo for fo in range(nf)]
        va_masks = [folds_dev == fo for fo in range(nf)]
        # host fold masks (for cheap unique-class checks without a per-fold D2H of codes)
        tr_masks_h = [folds_host != fo for fo in range(nf)]
        va_masks_h = [folds_host == fo for fo in range(nf)]

        # Majority class for the multiclass shortlist residual; positive class is class 1 for binary
        # (mirrors the CPU path: binary indicator 1{y==1}, multiclass indicator 1{y==argmax bincount}).
        _bincount = np.bincount(y_enc, minlength=n_classes)
        maj = int(np.argmax(_bincount))
        pos_cls = 1 if n_classes == 2 else maj
        pos_dev = (yenc_dev == pos_cls).astype(cp.float64)

        labels_dev = cp.arange(n_classes)

        # ---------- resident logistic fits (strictly-convex L2 Newton; the unique sklearn optimum) ----------
        def _standardize(Xtr, Xother):
            mu = Xtr.mean(axis=0)
            sd = Xtr.std(axis=0)
            sd = cp.where(sd < 1e-12, 1.0, sd)
            return (Xtr - mu) / sd, [(Xo - mu) / sd for Xo in Xother]

        def _fit_binary(Xs, yb):
            """L2 Newton for binary logistic on standardized design ``Xs`` (intercept appended,
            unpenalised). ``yb`` is a resident {0,1} float vector. Returns w (k+1,) or raises."""
            nn, k = Xs.shape
            A = cp.empty((nn, k + 1), dtype=cp.float64)
            A[:, :k] = Xs
            A[:, k] = 1.0
            wv = cp.zeros(k + 1, dtype=cp.float64)
            reg = cp.ones(k + 1, dtype=cp.float64)
            reg[k] = 0.0
            lam = 1.0 / _C
            for _ in range(_NEWTON_MAX_ITER):
                z = A @ wv
                p = 1.0 / (1.0 + cp.exp(-z))
                Wd = p * (1.0 - p)
                grad = A.T @ (p - yb) + lam * reg * wv
                H = (A * Wd[:, None]).T @ A
                H[cp.arange(k + 1), cp.arange(k + 1)] += lam * reg
                try:
                    step = cp.linalg.solve(H, grad)
                except Exception:
                    raise _ResidentClfFallback()
                wv = wv - step
                if not bool(cp.all(cp.isfinite(wv))):
                    raise _ResidentClfFallback()
                if float(cp.max(cp.abs(step))) < _NEWTON_TOL:
                    break
            else:
                raise _ResidentClfFallback()  # did not converge -> defer to CPU
            return wv

        def _proba_binary(Xs, wv):
            k = Xs.shape[1]
            z = Xs @ wv[:k] + wv[k]
            return 1.0 / (1.0 + cp.exp(-z))

        def _fit_multinomial(Xs, yc):
            """Symmetric multinomial L2 Newton (full block Hessian) on standardized design. ``yc`` is a
            resident int code vector. Returns W (k+1, C) or raises. sklearn 1.x multinomial default."""
            nn, k = Xs.shape
            d = k + 1
            A = cp.empty((nn, d), dtype=cp.float64)
            A[:, :k] = Xs
            A[:, k] = 1.0
            Y = cp.zeros((nn, n_classes), dtype=cp.float64)
            Y[cp.arange(nn), yc] = 1.0
            Wm = cp.zeros((d, n_classes), dtype=cp.float64)
            reg = cp.ones(d, dtype=cp.float64)
            reg[k] = 0.0
            lam = 1.0 / _C
            regdiag = lam * cp.diag(reg)
            for _ in range(_NEWTON_MAX_ITER):
                Z = A @ Wm
                Z = Z - Z.max(axis=1, keepdims=True)
                E = cp.exp(Z)
                Pm = E / E.sum(axis=1, keepdims=True)
                G = A.T @ (Pm - Y) + lam * (reg[:, None] * Wm)
                grad = G.reshape(-1)
                Hbig = cp.zeros((d * n_classes, d * n_classes), dtype=cp.float64)
                for c in range(n_classes):
                    for c2 in range(n_classes):
                        wgt = Pm[:, c] * ((1.0 if c == c2 else 0.0) - Pm[:, c2])
                        blk = (A * wgt[:, None]).T @ A
                        if c == c2:
                            blk = blk + regdiag
                        Hbig[c * d:(c + 1) * d, c2 * d:(c2 + 1) * d] = blk
                try:
                    step = cp.linalg.solve(Hbig, grad)
                except Exception:
                    raise _ResidentClfFallback()
                Wm = Wm - step.reshape(d, n_classes)
                if not bool(cp.all(cp.isfinite(Wm))):
                    raise _ResidentClfFallback()
                if float(cp.max(cp.abs(step))) < _NEWTON_TOL:
                    break
            else:
                raise _ResidentClfFallback()
            return Wm

        def _proba_multinomial(Xs, Wm):
            nn, k = Xs.shape
            A = cp.empty((nn, k + 1), dtype=cp.float64)
            A[:, :k] = Xs
            A[:, k] = 1.0
            Z = A @ Wm
            Z = Z - Z.max(axis=1, keepdims=True)
            E = cp.exp(Z)
            return E / E.sum(axis=1, keepdims=True)

        def _logloss(yc_dev, proba):
            """CV-logloss over ALL ``labels_dev`` classes (mirrors sklearn ``log_loss(..., labels)``).
            ``proba`` is (m,) for binary positive-class prob or (m, C) for multinomial."""
            if proba.ndim == 1:
                p1 = cp.clip(proba, _LOGLOSS_EPS, 1.0 - _LOGLOSS_EPS)
                p = cp.stack([1.0 - p1, p1], axis=1)
            else:
                p = cp.clip(proba, _LOGLOSS_EPS, 1.0)
                p = p / p.sum(axis=1, keepdims=True)
            m = int(yc_dev.shape[0])
            ll = -cp.mean(cp.log(p[cp.arange(m), yc_dev]))
            return float(ll)

        def _fit_proba(Xtr_s, Xeval_list, ytr_codes_dev):
            """Fit on standardized train design, return list of eval-set probabilities (positive-class
            (m,) for binary, (m,C) for multinomial). Raises _ResidentClfFallback on a bad fit."""
            if n_classes == 2:
                yb = (ytr_codes_dev == 1).astype(cp.float64)
                wv = _fit_binary(Xtr_s, yb)
                return [_proba_binary(Xe, wv) for Xe in Xeval_list]
            Wm = _fit_multinomial(Xtr_s, ytr_codes_dev)
            return [_proba_multinomial(Xe, Wm) for Xe in Xeval_list]

        # ---------------- shortlist (residual-aware pre-rank), fully resident ----------------
        def _abscorr_batch(resid_dev, rows_mask) -> np.ndarray:
            M = Vdev if rows_mask is None else Vdev[rows_mask]
            out = cp.zeros(P, dtype=cp.float64)
            m = int(M.shape[0])
            if m == 0:
                return cp.asnumpy(out)
            if float(resid_dev.std()) < 1e-12:
                return cp.asnumpy(out)
            vm = resid_dev - resid_dev.mean()
            ssv = float(cp.dot(vm, vm))
            if ssv <= 0.0:
                return cp.asnumpy(out)
            Mc = M - M.mean(axis=0, keepdims=True)
            num = Mc.T @ vm
            ssc = (Mc * Mc).sum(axis=0)
            denom = cp.sqrt(ssc * ssv)
            col_std = M.std(axis=0)
            valid = (col_std >= 1e-12) & (ssc > 0.0) & (denom > 0.0)
            r = cp.where(valid, num / cp.where(denom > 0.0, denom, 1.0), 0.0)
            r = cp.where(cp.isfinite(r), cp.abs(r), 0.0)
            return cp.asnumpy(r)

        def _shortlist(sel_idx) -> list:
            if sel_idx:
                ho = va_masks[0]
                tr = tr_masks[0]
                if int(np.unique(y_enc[tr_masks_h[0]]).size) >= 2:
                    Xtr = Vdev[tr][:, sel_idx]
                    Xho = Vdev[ho][:, sel_idx]
                    Xtr_s, (Xho_s,) = _standardize(Xtr, [Xho])
                    proba = _fit_proba(Xtr_s, [Xho_s], yenc_dev[tr])[0]
                    if proba.ndim == 1:
                        phat = proba                      # binary: P(class 1) == P(pos)
                    else:
                        phat = proba[:, pos_cls]          # multiclass: P(majority)
                    resid = pos_dev[ho] - phat
                else:
                    resid = pos_dev[ho] - pos_dev[tr].mean()
                uses = _abscorr_batch(resid, ho)
            else:
                resid = pos_dev - pos_dev.mean()
                uses = _abscorr_batch(resid, None)
            sel_set = set(sel_idx)
            scored = []
            for i in range(P):
                if i in sel_set:
                    continue
                scored.append((i, (1.0 - w) * (mi_host[i] / mi_max) + w * float(uses[i])))
            scored.sort(key=lambda t: t[1], reverse=True)
            return [i for i, _ in scored[: max(1, shortlist)]]

        # ---------------- per-fold CV-logloss, fully resident ----------------
        def _cv_baseline() -> np.ndarray:
            """No-selection per-fold logloss: predict each val fold by its train-fold class PRIOR. Binary
            (multiclass is deferred to CPU above), so the prior is the SCALAR train-fold positive rate; the
            per-val proba is BROADCAST resident from that scalar -- NO per-fold H2D tile (residency contract)."""
            errs = np.empty(nf, dtype=np.float64)
            for fo in range(nf):
                tr, va = tr_masks[fo], va_masks[fo]
                p1 = float(pos_dev[tr].mean())                 # bounded scalar D2H (the train-fold prior)
                p1 = min(max(p1, _LOGLOSS_EPS), 1.0 - _LOGLOSS_EPS)
                yv = ydev_f[va]                                # resident {0,1} val labels (class 1 == positive)
                errs[fo] = float(-cp.mean(yv * math.log(p1) + (1.0 - yv) * math.log(1.0 - p1)))
            return errs

        def _cv_candidate(sel_idx, cand) -> np.ndarray:
            """Per-fold logloss for the selected set + one candidate ``cand`` (a fresh fit per fold).
            Mirrors the CPU per-candidate refit; returns a bounded (nf,) result vector (one D2H/fold-scalar)."""
            full = sel_idx + [cand]
            errs = np.empty(nf, dtype=np.float64)
            for fo in range(nf):
                tr, va = tr_masks[fo], va_masks[fo]
                if int(np.unique(y_enc[tr_masks_h[fo]]).size) < 2:
                    errs[fo] = np.inf
                    continue
                Xtr = Vdev[tr][:, full]
                Xva = Vdev[va][:, full]
                Xtr_s, (Xva_s,) = _standardize(Xtr, [Xva])
                proba = _fit_proba(Xtr_s, [Xva_s], yenc_dev[tr])[0]
                errs[fo] = _logloss(yenc_dev[va], proba)
            return errs

        min_improving_folds = max(1, int(math.ceil(0.75 * nf)))
        selected: list = []
        folds_cur = _cv_baseline()
        cur = float(folds_cur.mean())
        for _ in range(min(K, P)):
            cand_idx = _shortlist(selected)
            best_i, best_mean, best_folds = -1, cur, folds_cur
            for i in cand_idx:
                mf = _cv_candidate(selected, i)
                if int(np.sum(mf < folds_cur)) < min_improving_folds:
                    continue
                if float(mf.mean()) < best_mean:
                    best_mean, best_i, best_folds = float(mf.mean()), i, mf
            if best_i < 0 or best_mean >= cur * (1.0 - mae_improve_rel):
                break
            selected.append(best_i)
            folds_cur, cur = best_folds, best_mean
        return [pool[i] for i in selected]
    except _ResidentClfFallback:
        return None  # non-convergent / singular / degenerate device fit -> exact CPU greedy
    except Exception:
        return None  # any cupy/device error -> exact CPU greedy


__all__ = ["usability_greedy_clf_gpu_resident"]
