"""Leakage-safe Mode-A out-of-fold residual computation for ``compute_residual_attention``.

The residual target a row attends to must be produced by an auxiliary model that never saw that row's outer-fold partner rows. A flat aux KFold (independent of the
caller's outer ``splitter``) does NOT satisfy this: a val row in outer fold ``f`` attends to complement rows whose aux residual may have been produced by an aux model
that trained on fold ``f`` (including the val row itself), so the val row's own target leaks into its attention feature across the outer split.

The fix is NESTED: for each outer fold ``f``, the residual bank used to attend ``f``'s val rows is computed by an aux OOF restricted to ``f``'s train complement only.
No row of fold ``f`` ever participates in a complement-row residual, so the aux target for any bank row is out-of-fold w.r.t. the SAME partition the outer attention uses.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
from sklearn.model_selection import KFold


def compute_oof_yhat_within(
    X_sub: np.ndarray,
    y_sub: np.ndarray,
    *,
    task: str,
    make_aux: Callable[[], Any],
    aux_n_splits: int,
    seed: int,
) -> np.ndarray:
    """Aux OOF predictions ``y_hat`` for the rows of ``X_sub`` using only rows within ``X_sub`` — used per outer fold on the fold's train complement.

    The aux sub-partition is restricted to ``X_sub`` so no row outside it contributes to any row's prediction. ``aux_n_splits`` is capped to the subset size so tiny
    subsets still partition cleanly.
    """
    n = X_sub.shape[0]
    n_splits = min(aux_n_splits, n) if n >= 2 else 2
    n_splits = max(2, n_splits)
    y_hat = np.zeros(n, dtype=np.float32)
    aux_splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr_idx, va_idx in aux_splitter.split(X_sub):
        model = make_aux()
        model.fit(X_sub[tr_idx], y_sub[tr_idx])
        if task == "binary":
            y_hat[va_idx] = model.predict_proba(X_sub[va_idx])[:, 1].astype(np.float32, copy=False)
        else:
            y_hat[va_idx] = model.predict(X_sub[va_idx]).astype(np.float32, copy=False)
    return y_hat


def compute_oof_residual_within(
    X_sub: np.ndarray,
    y_sub: np.ndarray,
    *,
    task: str,
    make_aux: Callable[[], Any],
    aux_n_splits: int,
    seed: int,
) -> np.ndarray:
    """Aux OOF residuals for the rows of ``X_sub`` using only rows within ``X_sub`` — used per outer fold on the fold's train complement.

    The aux sub-partition is restricted to the complement so no outer-val row contributes to a complement-row residual. ``aux_n_splits`` is capped to the complement
    size so tiny complements (rare with sane fold counts) still partition cleanly.
    """
    y_hat = compute_oof_yhat_within(X_sub, y_sub, task=task, make_aux=make_aux, aux_n_splits=aux_n_splits, seed=seed)
    return (y_sub.astype(np.float32) - y_hat).astype(np.float32)
