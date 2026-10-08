"""Scoring leaves and fold-CV steps of the usability-aware greedy: scrub / float64 casts, the |corr| diversity gate and the four ``usability_greedy`` steps.

Carved out of ``_usability_aware_selection``, which re-exports every name so the historical import path keeps working.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger("mlframe.feature_selection.filters._usability_aware_selection")

def _scrub(v: np.ndarray, dtype: Any = np.float64) -> np.ndarray:
    """Cast ``v`` to ``dtype`` and replace every non-finite entry (NaN/+inf/-inf) with 0.0, returning a new array."""
    # ``np.where(isfinite, a, 0)`` is bit-identical to ``nan_to_num(nan=0, posinf=0, neginf=0)`` (isfinite
    # is False for exactly nan/+inf/-inf) but ~2.8x faster (no per-call isposinf/isneginf/_getmaxmin
    # machinery): 764us -> 269us on a 100k float32 column. _scrub is called ~17k+/retention fit on full-n
    # columns, so this is a direct cut to the pool-build cost. Verified bit-identical over float32/float64 +
    # nan/inf fuzz.
    # A cast to a narrower dtype (e.g. float64 -> float32) can overflow to +-inf on an extreme
    # input value -- numpy warns "overflow encountered in cast" even though that's exactly the
    # non-finite case this function's whole job is to zero out right below. Harmless, suppressed
    # locally rather than at every caller.
    with np.errstate(over="ignore"):
        a = np.asarray(v, dtype=dtype)
    return np.where(np.isfinite(a), a, 0)


def _f64(v: np.ndarray) -> np.ndarray:
    """Upcast a stored (possibly float32) candidate column to float64 for MI / correlation /
    recipe-edge computation where the heavy-tail precision matters (transient; not stored)."""
    return np.asarray(v, dtype=np.float64)


def _abscorr(u: np.ndarray, v: np.ndarray) -> float:
    """Absolute Pearson correlation ``|corr(u, v)|`` in float64, used as the diversity / near-duplicate gate. Returns
    0.0 if either input is empty or near-constant (std < 1e-12), or if the raw correlation is non-finite."""
    # GATED GPU PATH (MLFRAME_FE_GPU_USABILITY, default OFF). The cupy twin is float64 + the SAME
    # std<1e-12 guard, but a cupy reduction can reassociate the last bits vs numpy -> a |corr| drift
    # that, on the ULP-sensitive clean-form demotion, could flip a pin. So it is ENABLED only on a host
    # where the gate-on pytest verified the SAME selection; on ANY cupy/device error we fall through to
    # the exact numpy path (the fit is never broken by a GPU problem).
    if _GPU_USABILITY():
        try:
            from ._usability_gpu import gpu_abscorr
            return gpu_abscorr(u, v)
        except Exception as e:  # nosec B110 - optional/best-effort path, rationale documented
            logger.debug("_abscorr: GPU path failed, falling back to the exact CPU path: %s", e)
    u = _f64(u); v = _f64(v)  # precision for the heavy-tail correlation
    if u.size == 0 or float(np.std(u)) < 1e-12 or float(np.std(v)) < 1e-12:
        return 0.0
    r = np.corrcoef(u, v)[0, 1]
    return abs(float(r)) if np.isfinite(r) else 0.0


def _GPU_USABILITY() -> bool:
    """Whether the gated cupy usability-scoring path is active (``MLFRAME_FE_GPU_USABILITY`` + live
    cupy + global GPU not disabled). Default OFF; the CPU path is the proven, selection-exact default.
    Imported lazily so a no-cupy host never touches the GPU module."""
    try:
        from ._usability_gpu import fe_gpu_usability_enabled
        return fe_gpu_usability_enabled()
    except Exception as e:
        logger.debug("_GPU_USABILITY: fe_gpu_usability_enabled() check failed, staying on the CPU path: %s", e)
        return False


def _usability_greedy_step1_unchanged_selection_identical(K, pool, st, shortlist):
    """Step 1 of usability_greedy: lines starting at ``try:``."""
    try:
        from mlframe.feature_selection.filters.feature_engineering import _can_hoist_shared_buffer, _fe_effective_buffer_budget_bytes

        _k_eff = max(1, min(int(K), len(pool)))
        _can, _need, _avail = _can_hoist_shared_buffer(st.n * _k_eff * 8, n_workers=1)
        if (not _can) and _avail > 0:
            # Cap K to the largest float64 (n, K) design that fits the SAME overhead-aware budget the
            # gate used (not the raw available), flooring at 1 so the greedy always makes progress.
            _budget = _fe_effective_buffer_budget_bytes(_avail, n_workers=1)
            _k_fit = int(_budget // (st.n * 8)) if _budget > 0 else 1
            if _k_fit < _k_eff:
                K = max(1, _k_fit)
                shortlist = min(int(shortlist), max(int(K), 1))
    except Exception as e:  # nosec B110 - best-effort path
        logger.debug("shortlist auto-sizing failed, keeping the caller-provided shortlist: %s", e)
    return K, shortlist


def _usability_greedy_step2_def_cv_fold(classification, n_folds, folds, y_enc, n_classes, _logloss, _pv, _mk, y_cont):
    """Step 2 of usability_greedy: lines starting at ``def _cv_per_fold(sel_idx) -> np.ndarray:``."""
    def _cv_per_fold(sel_idx) -> np.ndarray:
        """Exact (full-refit) K-fold CV score of the candidate set ``sel_idx``: per fold, fit ``_mk()`` on the
        train rows and score the held-out rows, returning one error value per fold (CV log loss when
        ``classification``, else CV mean-absolute-error). With an empty ``sel_idx`` scores the no-selection
        baseline (train-fold class-prior probabilities for classification, train-fold mean for regression). This
        is the exact fallback for any fold where the incremental bordered-Gram solve in
        ``_cv_candidates_incremental`` is singular."""
        if classification:
            # CV LOGLOSS of a logistic model (lower-is-better, same gate semantics as MAE). The
            # no-selection baseline is the constant train-fold class-PRIOR probability.
            if not sel_idx:
                errs = []
                for fo in range(n_folds):
                    trm, vam = folds != fo, folds == fo
                    prior = np.bincount(y_enc[trm], minlength=n_classes).astype(np.float64)
                    prior = prior / max(prior.sum(), 1.0)
                    prior = np.clip(prior, 1e-12, 1.0)
                    proba = np.tile(prior, (int(vam.sum()), 1))
                    errs.append(_logloss(y_enc[vam], proba))
                return np.asarray(errs, dtype=np.float64)
            Xs = np.column_stack([_pv(i) for i in sel_idx])
            errs = []
            for fo in range(n_folds):
                trm, vam = folds != fo, folds == fo
                if np.unique(y_enc[trm]).size < 2:
                    errs.append(np.inf)
                    continue
                m = _mk().fit(Xs[trm], y_enc[trm])
                proba = m.predict_proba(Xs[vam])
                errs.append(_logloss(y_enc[vam], proba))
            return np.asarray(errs, dtype=np.float64)
        if not sel_idx:
            return np.array([float(np.mean(np.abs(y_cont[folds == fo] - float(np.mean(y_cont[folds != fo]))))) for fo in range(n_folds)])
        Xs = np.column_stack([_pv(i) for i in sel_idx])
        errs = []
        for fo in range(n_folds):
            trm, vam = folds != fo, folds == fo
            m = _mk().fit(Xs[trm], y_cont[trm])
            errs.append(float(np.mean(np.abs(y_cont[vam] - m.predict(Xs[vam])))))
        return np.asarray(errs, dtype=np.float64)
    return _cv_per_fold


def _usability_greedy_step3_cheap_residual_aware(classification, n_classes, y_enc, _pv, folds, _mk, y_cont, pool, w, mi_max, shortlist_diversity_corr, shortlist):
    """Step 3 of usability_greedy: lines starting at ``def _shortlist(sel_idx) -> list[int]:``."""
    def _shortlist(sel_idx) -> list[int]:
        """Rank every not-yet-selected pool candidate by a cheap pre-rank score - ``(1-w) * (mi / mi_max) + w *
        |corr(candidate, held-out residual)|`` - and return the indices of the top ``shortlist`` (default 40)
        candidates. The residual is computed on the fold-0 held-out rows only (fit on the other folds, to avoid
        leakage): for regression it is ``y - model.predict(X)``, for classification the positive/majority-class
        indicator minus its predicted probability; with no prior selection it falls back to the (train-fold) mean/
        prior-centered residual over all rows. This pre-rank only bounds the candidate pool the greedy's expensive
        per-step CV evaluates - the actual commit decision is always the CV-MAE/logloss improvement."""
        # HELD-OUT residual (audit fix): fit on the fold-0-out train rows but score the
        # candidate correlation on the HELD-OUT fold-0 residual only - the prior code predicted over
        # ALL rows (in-sample for the ~(k-1)/k training rows), which is the leakage the module's
        # "held-out residual" design explicitly avoids. The no-selection case uses the mean residual
        # over all rows (no model fit -> no leakage).
        if classification:
            # CLASSIFICATION residual: correlate each candidate with the POSITIVE-class indicator
            # residual (point-biserial-style). For binary y the indicator is 1{y==last class}; once a
            # logistic model is selected, the residual is indicator - P(positive). For multiclass we
            # fall back to the one-vs-rest indicator of the majority class (a cheap pre-rank only - the
            # COMMIT decision is always the CV-logloss improvement).
            if n_classes == 2:
                pos = (y_enc == 1).astype(np.float64)
            else:
                _maj = int(np.argmax(np.bincount(y_enc, minlength=n_classes)))
                pos = (y_enc == _maj).astype(np.float64)
            if sel_idx:
                Xs = np.column_stack([_pv(i) for i in sel_idx])
                ho = folds == 0
                tr = ~ho
                if np.unique(y_enc[tr]).size >= 2:
                    m = _mk().fit(Xs[tr], y_enc[tr])
                    proba = m.predict_proba(Xs[ho])
                    if n_classes == 2:
                        phat = proba[:, 1]
                    else:
                        phat = proba[:, _maj]
                    resid = pos[ho] - phat
                else:
                    resid = pos[ho] - float(np.mean(pos[tr]))
                rows = ho
            else:
                resid = pos - float(np.mean(pos))
                rows = slice(None)
        elif sel_idx:
            Xs = np.column_stack([_pv(i) for i in sel_idx])
            ho = folds == 0
            tr = ~ho
            m = _mk().fit(Xs[tr], y_cont[tr])
            resid = y_cont[ho] - m.predict(Xs[ho])
            rows = ho
        else:
            resid = y_cont - float(np.mean(y_cont))
            rows = slice(None)
        cand_ids = [i for i in range(len(pool)) if i not in sel_idx]
        # GATED GPU BATCH (MLFRAME_FE_GPU_USABILITY, default OFF): the per-candidate |corr(values,
        # resid)| is the n-scaling inner loop of the shortlist - batch all candidates' columns vs the
        # one resid in ONE cupy GEMV (centered dot / sqrt(ss)) instead of a Python loop of np.corrcoef.
        # BIT-FAITHFUL to the per-candidate ``_abscorr`` (same float64 estimator + std<1e-12 guard), so
        # the shortlist ORDER is unchanged. Any cupy/device error -> the exact per-candidate CPU loop.
        uses = None
        if _GPU_USABILITY() and cand_ids:
            try:
                from mlframe.feature_selection.filters._usability_gpu import gpu_abscorr_batch
                cols = np.column_stack([_pv(i)[rows] for i in cand_ids])
                uses = gpu_abscorr_batch(cols, _f64(np.asarray(resid)))
            except Exception as e:  # best-effort: the exact per-candidate CPU |corr| loop below reproduces the same shortlist order
                logger.debug("gpu_abscorr_batch failed, falling back to the exact CPU path: %s", e)
                uses = None  # fall back to the exact CPU path
        scored = []
        for k, i in enumerate(cand_ids):
            use = float(uses[k]) if uses is not None else _abscorr(pool[i].values[rows], resid)
            scored.append((i, (1.0 - w) * (pool[i].mi / mi_max) + w * use, pool[i].mi, use))
        scored.sort(key=lambda t: t[1], reverse=True)
        if logger.isEnabledFor(logging.DEBUG):
            top = ", ".join(f"{pool[i].name}(score={s:.4f},mi={m:.4f},use={u:.4f})" for i, s, m, u in scored[:10])
            logger.debug("usability_greedy._shortlist(sel=%r): top10=[%s]", [pool[j].name for j in sel_idx], top)
        # Greedy diversity filter (mirrors build_usability_candidate_pool's per-pair dedup): several
        # top-scored candidates can be near-duplicate views of the SAME underlying signal (e.g. one
        # coarse rint(d)-driven relationship expressed via 3+ algebraically-different c-side wraps),
        # which would otherwise occupy multiple `shortlist` slots and starve room for a genuinely
        # different-signal candidate the CV-MAE commit stage never gets a chance to evaluate. Walk the
        # score-sorted list, keep a candidate only if it is NOT a near-duplicate (|corr| >
        # shortlist_diversity_corr) of an already-kept one.
        out: list[int] = []
        kept_vals: list[np.ndarray] = []
        for i, _s, _m, _u in scored:
            v = pool[i].values[rows]
            if any(_abscorr(v, kv) > shortlist_diversity_corr for kv in kept_vals):
                continue
            out.append(i)
            kept_vals.append(v)
            if len(out) >= max(1, shortlist):
                break
        return out
    return _shortlist


def _usability_greedy_step4_consistency_across_folds(K, pool, _shortlist, st, classification, _cv_candidates_incremental, _cv_per_fold, mae_improve_rel):
    """Step 4 of usability_greedy: lines starting at ``for _ in range(min(K, len(pool))):``."""
    for _ in range(min(K, len(pool))):
        cand_idx = _shortlist(st.selected)
        best_i, best_mean, best_folds = -1, st.mae_cur, st.folds_cur
        # regression: score the whole shortlist via the incremental bordered solve (one selected-set
        # Gram per fold, reused across candidates). classification stays on the per-candidate refit.
        # ``_USAB_FORCE_FULL_REFIT`` (test-only) bypasses the incremental solve to A/B the selection.
        import os as _os
        _force_full = bool(_os.environ.get("_USAB_FORCE_FULL_REFIT"))
        _mf_by_i = None if (classification or _force_full) else _cv_candidates_incremental(st.selected, cand_idx)
        for i in cand_idx:
            mf = _mf_by_i[i] if _mf_by_i is not None else _cv_per_fold([*st.selected, i])
            if int(np.sum(mf < st.folds_cur)) < st.min_improving_folds:
                continue  # not a consistent improvement across folds
            if float(mf.mean()) < best_mean:
                best_mean, best_i, best_folds = float(mf.mean()), i, mf
        if best_i < 0 or best_mean >= st.mae_cur * (1.0 - mae_improve_rel):
            break
        st.selected.append(best_i)
        st.folds_cur, st.mae_cur = best_folds, best_mean
