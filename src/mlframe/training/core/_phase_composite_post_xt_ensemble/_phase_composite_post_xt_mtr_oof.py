"""Honest train-K-fold OOF weighting for the MULTI_TARGET_REGRESSION per-column ensemble.

The MTR per-column ensemble (``MTRPerColumnEqualMeanEnsemble``) can learn per-target NNLS weights. Fitting them on
the suite's val fold double-dips the early-stopping surface (the same leak as the single-target ensemble). This
helper computes an honest train-K-fold OOF prediction stack -- each component clone re-fit on K-1 folds, predicting
the held-out fold -- and returns the per-column NNLS weights derived from it, so the per-column weights never reuse
the surface the components were tuned against.

Bench: ``training/_benchmarks/bench_mtr_nnls_oof.py`` (honest-OOF NNLS beats equal_mean on 8/8 seeds, ~9% lower test
RMSE, and is leak-free vs val-fit).

cProfile (n=4000, K=3, 4 components, kfold=5; 2026-06-11): ``compute_mtr_oof_nnls_weights`` is ~0.14 s/call, of
which ~77% is the inner component ``fit`` re-refits (the leak-free K-fold's irreducible cost) and ~2% is the
wrapper's own work; the NNLS solves, the per-component finite check (``np.isfinite(oof).all(axis=(1,2))``) and the
exclusion bookkeeping are in the noise. The per-component-exclusion path solves a ``<= n_comp``-wide NNLS submatrix
(survivors only), so it is strictly ``<=`` the prior full-matrix work -- no actionable wrapper speedup; the hot path
is the inner refits, which exist by design.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger("mlframe.training.core._phase_composite_post")


def _slice_rows_by_idx(X: Any, idx: np.ndarray) -> Any:
    """Row-subset a pandas / polars / ndarray frame by integer indices without copying the whole frame.

    Each fold's ``tr_idx`` is distinct, so the K slices are irreducible (no shared materialization to hoist) and
    already done once per fold OUTSIDE the component loop. ``reset_index`` is ~2/3 of the slice wall (10->30 ms @
    n=100k) but is NOT droppable for free: removing it makes ``fit`` see a non-contiguous gather, perturbing the
    NNLS-input predictions by ~1 ULP (1e-17) -> not bit-identical, so kept (clean 0..n-1 index for the components).
    """
    if hasattr(X, "iloc"):
        sub = X.iloc[idx]
        return sub.reset_index(drop=True) if hasattr(sub, "reset_index") else sub
    if hasattr(X, "filter") and hasattr(X, "slice"):
        import polars as pl  # type: ignore
        return X[idx.tolist()] if hasattr(X, "__getitem__") else X.filter(pl.Series(np.isin(np.arange(len(X)), idx)))
    return X[idx]


def compute_mtr_oof_nnls_weights(
    components: list[Any],
    X_train: Any,
    y_train: np.ndarray,
    *,
    kfold: int = 5,
    random_state: int = 42,
) -> np.ndarray | None:
    """Return ``(n_components, n_targets)`` per-column NNLS weights from honest train-K-fold OOF, or ``None``.

    Each component is cloned and re-fit on K-1 folds to predict the held-out fold, assembling an OOF (n_comp, n,
    K) stack; per target column an independent NNLS solve recovers non-negative weights.

    Per-component exclusion (audit I7, implemented 2026-06-11)
    ----------------------------------------------------------
    A SINGLE bad component (its fold-refit raises, or it emits a non-finite OOF cell) no longer forfeits the
    WHOLE benched ~9% honest-OOF win (``bench_mtr_nnls_oof.py``). The bad component is EXCLUDED -- its row in the
    returned ``(n_components, n_targets)`` weight matrix is left all-zero -- and the NNLS solve runs over the
    SURVIVING components only, so the ensemble keeps the honest weighting whenever >=2 components survive. A
    zero weight-row contributes nothing to the caller's ``np.einsum("cnk,ck->nk", stack, weights)`` apply, so
    the return contract is preserved: the matrix is still shaped to ALL components (the caller at
    ``_post_xt_ensemble_mtr.py`` injects weights of exactly that shape), the excluded component simply gets
    weight 0 on every target. This is strictly better than the previous all-or-nothing ``None`` return, which
    silently degraded the ensemble to equal-mean (a documented numerical loss) on the first bad component.

    ``None`` is returned only when the honest weighting cannot be formed at all:

    * Not applicable -- fewer than 2 components, or fewer than ``max(50, kfold * 2)`` rows: K-fold OOF is not
      meaningful and equal-mean is the correct, non-degraded answer. Stays quiet (no WARNING).
    * Too few survivors -- after exclusion fewer than 2 components remain usable: there is no ensemble left to
      weight, so equal-mean is again correct. Logged at WARNING (some honest weighting was forfeited).
    * Catch-all -- an unexpected error before any per-component result is available. Logged at WARNING.

    Each EXCLUDED component is logged once at WARNING naming the partial forfeiture so the degradation is visible
    in prod logs; the success (all components survive) and not-applicable paths stay quiet.

    Bench note: per-component exclusion only ADDS rows to the survivor set relative to the previous code (which
    survived only when ALL components were clean); on the all-clean case it is bit-identical to the prior NNLS
    solve, so it cannot regress the ``bench_mtr_nnls_oof.py`` majority-win verdict (that bench has no failing
    component). It strictly salvages cases the old code dropped to equal-mean.
    """
    from sklearn.base import clone
    from sklearn.model_selection import KFold
    from scipy.optimize import nnls

    try:
        y_arr = np.asarray(y_train, dtype=np.float64)
        if y_arr.ndim == 1:
            y_arr = y_arr.reshape(-1, 1)
        n, k = y_arr.shape
        n_comp = len(components)
        if n_comp < 2 or n < max(50, kfold * 2):
            return None
        oof = np.full((n_comp, n, k), np.nan, dtype=np.float64)
        # Per-component health: a component is excluded if ANY fold-refit raises
        # (recorded in ``excluded``) or it emits a non-finite OOF cell (detected
        # after the stack is built). Excluded components keep a zero weight-row.
        excluded: dict[int, str] = {}
        kf = KFold(n_splits=int(kfold), shuffle=True, random_state=int(random_state))
        for tr_idx, ho_idx in kf.split(np.arange(n)):
            X_tr = _slice_rows_by_idx(X_train, tr_idx)
            X_ho = _slice_rows_by_idx(X_train, ho_idx)
            y_tr = y_arr[tr_idx]
            for ci, comp in enumerate(components):
                if ci in excluded:
                    continue  # already failed in an earlier fold -> skip the refit
                try:
                    cl = clone(comp)
                    cl.fit(X_tr, y_tr)
                    p = np.asarray(cl.predict(X_ho), dtype=np.float64)
                    if p.ndim == 1:
                        p = p.reshape(-1, 1)
                    oof[ci, ho_idx, :] = p
                except Exception as exc:
                    excluded[ci] = f"fold refit raised ({exc})"
        # Non-finite OOF cells exclude their component (not the whole weighting).
        # Components already excluded for a raise have NaN rows and are skipped here.
        finite_per_comp = np.isfinite(oof).all(axis=(1, 2))  # (n_comp,)
        for ci in range(n_comp):
            if ci not in excluded and not bool(finite_per_comp[ci]):
                n_bad = int((~np.isfinite(oof[ci])).sum())
                excluded[ci] = f"non-finite OOF cells ({n_bad}/{oof[ci].size})"
        survivors = [ci for ci in range(n_comp) if ci not in excluded]
        for ci in sorted(excluded):
            logger.warning(
                "[MTR CT_ENSEMBLE] honest-OOF excluding component %d (%s); weighting the remaining "
                "%d/%d component(s) with the benched ~9%% NNLS surface (excluded component gets weight 0).",
                ci, excluded[ci], len(survivors), n_comp,
            )
        if len(survivors) < 2:
            logger.warning(
                "[MTR CT_ENSEMBLE] honest-OOF left only %d usable component(s) of %d after exclusion; "
                "forfeiting the benched ~9%% NNLS win and falling back to equal-mean per-column weights.",
                len(survivors), n_comp,
            )
            return None
        # NNLS over the survivor sub-matrix only; scatter back into a full-width
        # (n_comp, k) matrix so excluded rows stay 0 and the caller's einsum apply
        # over ALL components is unchanged.
        weights = np.zeros((n_comp, k), dtype=np.float64)
        surv_idx = np.asarray(survivors, dtype=np.intp)
        oof_surv = oof[surv_idx]  # (n_surv, n, k); all-finite by construction
        for kk in range(k):
            A_k = oof_surv[:, :, kk].T  # (n, n_surv)
            w_k, _ = nnls(A_k, y_arr[:, kk], maxiter=200)
            if float(w_k.sum()) > 0:
                weights[surv_idx, kk] = w_k
            else:
                # Degenerate NNLS for this column -> equal-mean across SURVIVORS
                # only (excluded components stay 0).
                weights[surv_idx, kk] = 1.0 / len(survivors)
        return weights
    except Exception as exc:
        logger.warning(
            "[MTR CT_ENSEMBLE] honest-OOF NNLS weighting failed (%s); forfeiting the benched ~9%% NNLS win "
            "and falling back to equal-mean per-column weights.",
            exc,
        )
        return None
