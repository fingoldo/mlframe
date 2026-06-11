"""Honest train-K-fold OOF weighting for the MULTI_TARGET_REGRESSION per-column ensemble.

The MTR per-column ensemble (``MTRPerColumnEqualMeanEnsemble``) can learn per-target NNLS weights. Fitting them on
the suite's val fold double-dips the early-stopping surface (the same leak as the single-target ensemble). This
helper computes an honest train-K-fold OOF prediction stack -- each component clone re-fit on K-1 folds, predicting
the held-out fold -- and returns the per-column NNLS weights derived from it, so the per-column weights never reuse
the surface the components were tuned against.

Bench: ``training/_benchmarks/bench_mtr_nnls_oof.py`` (honest-OOF NNLS beats equal_mean on 8/8 seeds, ~9% lower test
RMSE, and is leak-free vs val-fit).
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger("mlframe.training.core._phase_composite_post")


def _slice_rows_by_idx(X: Any, idx: np.ndarray) -> Any:
    """Row-subset a pandas / polars / ndarray frame by integer indices without copying the whole frame."""
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
    K) stack; per target column an independent NNLS solve recovers non-negative weights. Returns ``None`` on any
    failure so the caller can fall back to equal-mean.

    Observability (audit I7, 2026-06-11): a ``None`` return forfeits the benched ~9% honest-OOF win
    (``bench_mtr_nnls_oof.py``) and silently degrades the MTR ensemble to equal-mean. The two *failure* exits
    (a component fold-refit raising, or a non-finite OOF cell) and the catch-all exit are therefore logged at
    WARNING -- not DEBUG -- so the forfeiture is visible in prod logs. The "not applicable" exit (too few
    components / too few rows for a meaningful K-fold) stays quiet: that path is expected on small data and
    equal-mean is the correct, non-degraded answer there.

    FUTURE (deferred per audit I7): the failure exits return ``None`` for the WHOLE weighting on a SINGLE bad
    component -- a per-component exclusion (drop the failing component, NNLS-weight the rest) would salvage the
    win in more cases. That is held back deliberately: it changes the return contract (the caller at
    ``_post_xt_ensemble_mtr.py`` injects weights shaped to ALL components) and must be re-validated against
    ``bench_mtr_nnls_oof.py`` before shipping, which cannot run in this environment.
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
        kf = KFold(n_splits=int(kfold), shuffle=True, random_state=int(random_state))
        for tr_idx, ho_idx in kf.split(np.arange(n)):
            X_tr = _slice_rows_by_idx(X_train, tr_idx)
            X_ho = _slice_rows_by_idx(X_train, ho_idx)
            y_tr = y_arr[tr_idx]
            for ci, comp in enumerate(components):
                try:
                    cl = clone(comp)
                    cl.fit(X_tr, y_tr)
                    p = np.asarray(cl.predict(X_ho), dtype=np.float64)
                    if p.ndim == 1:
                        p = p.reshape(-1, 1)
                    oof[ci, ho_idx, :] = p
                except Exception as exc:
                    logger.warning(
                        "[MTR CT_ENSEMBLE] honest-OOF fold refit failed for component %d (%s); forfeiting "
                        "the benched ~9%% NNLS win and falling back to equal-mean per-column weights.",
                        ci, exc,
                    )
                    return None
        if not np.all(np.isfinite(oof)):
            logger.warning(
                "[MTR CT_ENSEMBLE] honest-OOF stack contains non-finite predictions (%d/%d cells); "
                "forfeiting the benched ~9%% NNLS win and falling back to equal-mean per-column weights.",
                int((~np.isfinite(oof)).sum()), oof.size,
            )
            return None
        weights = np.zeros((n_comp, k), dtype=np.float64)
        for kk in range(k):
            A_k = oof[:, :, kk].T  # (n, n_comp)
            w_k, _ = nnls(A_k, y_arr[:, kk], maxiter=200)
            weights[:, kk] = w_k if float(w_k.sum()) > 0 else 1.0 / n_comp
        return weights
    except Exception as exc:
        logger.warning(
            "[MTR CT_ENSEMBLE] honest-OOF NNLS weighting failed (%s); forfeiting the benched ~9%% NNLS win "
            "and falling back to equal-mean per-column weights.",
            exc,
        )
        return None
