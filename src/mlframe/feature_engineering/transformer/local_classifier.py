"""Locally-weighted classifier / regressor per row: fit a tiny linear model on each query's top-k neighbours, output its prediction.

Iter 30 mechanism. Differs from iter 7 (`compute_local_linear_attention`) which outputs residuals: iter 30 outputs the PREDICTION of a locally-fit linear classifier
(binary) or regressor (regression) at the query point, plus the model's local goodness-of-fit and coefficient norm.

For binary classification:
- Per query, fit a weighted logistic regression on its top-k=32 neighbors with weights = exp(-d²/h²).
- Outputs:
  1. ``local_proba`` — predicted P(y=1) at the query.
  2. ``local_logit`` — log(p/(1-p)).
  3. ``local_train_acc`` — accuracy of the local model on its own kNN training set (overfit indicator).
  4. ``local_coef_norm`` — L2 norm of the fit coefficient vector (sparsity / extreme-direction indicator).

For regression:
- Per query, fit a weighted linear regression on top-k neighbors.
- Outputs: ``local_pred`` (predicted y), ``local_resid_std`` (residual std on local training), ``local_coef_norm``, ``local_R2`` (local R²).

CB cannot fit per-query local models — its symmetric oblivious trees + TS encoding operate over the full training set. Each row gets a tiny LOCAL classifier fit
on its own neighborhood; the prediction encodes "what does a linear classifier think about this point given only its nearest few rows" — entirely outside CB's
representational capacity.

Leakage discipline: kNN refit per fold from train-fold rows; query rows fit their local model on train-fold neighbors only.

Cost: O(N_q · k · d²) for solving the small linear systems. At k=32, d=6, n_q=4000: ~5M ops per fold, sub-second.

References:
- Locally-weighted regression / LOWESS (Cleveland 1979).
- Local logistic regression for binary classification (Loader 1999).
"""
from __future__ import annotations

import logging
from typing import Any, Literal, Optional

import numpy as np
import polars as pl

from ._utils import require_seed, validate_numeric_input

logger = logging.getLogger(__name__)


def _solve_weighted_logreg(X_local: np.ndarray, y_local: np.ndarray, w: np.ndarray, x_query: np.ndarray, n_iter: int = 5, ridge: float = 0.1) -> tuple[float, float, float, float]:
    """Newton-Raphson weighted logistic regression with intercept + ridge regularisation.

    Solves: min Σ_i w_i [y_i log p_i + (1-y_i) log(1-p_i)] + ridge ||β||².

    Returns (proba, logit, train_acc, coef_norm) at the query point.
    """
    n, d = X_local.shape
    # Add intercept column.
    X_aug = np.hstack([np.ones((n, 1), dtype=np.float32), X_local])
    beta = np.zeros(d + 1, dtype=np.float32)
    w_safe = np.maximum(w, 1e-9)
    for _ in range(n_iter):
        eta = X_aug @ beta
        eta = np.clip(eta, -20.0, 20.0)
        p = 1.0 / (1.0 + np.exp(-eta))
        # Weighted gradient.
        grad = X_aug.T @ (w_safe * (p - y_local)) + ridge * beta
        # Weighted Hessian.
        W = w_safe * p * (1.0 - p) + 1e-9
        H = X_aug.T @ (W[:, None] * X_aug) + ridge * np.eye(d + 1, dtype=np.float32)
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            break
        beta = beta - step
    # Predict at query.
    x_query_aug = np.concatenate([[1.0], x_query]).astype(np.float32)
    logit_q = float(x_query_aug @ beta)
    logit_q = float(np.clip(logit_q, -20.0, 20.0))
    proba_q = 1.0 / (1.0 + np.exp(-logit_q))
    # Train accuracy on local set.
    p_train = 1.0 / (1.0 + np.exp(-np.clip(X_aug @ beta, -20.0, 20.0)))
    train_acc = float(((p_train >= 0.5).astype(np.float32) == y_local).mean())
    coef_norm = float(np.linalg.norm(beta[1:]))  # exclude intercept
    return proba_q, logit_q, train_acc, coef_norm


def _solve_weighted_linreg(X_local: np.ndarray, y_local: np.ndarray, w: np.ndarray, x_query: np.ndarray, ridge: float = 0.1) -> tuple[float, float, float, float]:
    """Weighted ridge linear regression with intercept.

    Solves: min Σ_i w_i (y_i - X β)² + ridge ||β||².

    Returns (pred, resid_std, coef_norm, r2) at the query point.
    """
    n, d = X_local.shape
    X_aug = np.hstack([np.ones((n, 1), dtype=np.float32), X_local])
    w_safe = np.maximum(w, 1e-9)
    Wx = w_safe[:, None] * X_aug
    A = X_aug.T @ Wx + ridge * np.eye(d + 1, dtype=np.float32)
    b = X_aug.T @ (w_safe * y_local)
    # lstsq tolerates an ill-conditioned local Hessian A (collinear neighbours)
    # that solve() either rejects or solves into exploded coefficients.
    try:
        beta = np.linalg.lstsq(A, b, rcond=None)[0]
    except np.linalg.LinAlgError:
        beta = np.zeros(d + 1, dtype=np.float32)
    x_query_aug = np.concatenate([[1.0], x_query]).astype(np.float32)
    pred_q = float(x_query_aug @ beta)
    # Local R² and residuals.
    y_hat = X_aug @ beta
    resid = y_local - y_hat
    resid_std = float(np.sqrt((w_safe * resid * resid).sum() / max(w_safe.sum(), 1e-9)))
    y_mean_w = float((w_safe * y_local).sum() / max(w_safe.sum(), 1e-9))
    ss_tot = float((w_safe * (y_local - y_mean_w) ** 2).sum()) + 1e-9
    ss_res = float((w_safe * resid * resid).sum())
    r2 = 1.0 - ss_res / ss_tot
    coef_norm = float(np.linalg.norm(beta[1:]))
    return pred_q, resid_std, coef_norm, r2


def compute_local_classifier_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: Optional[np.ndarray],
    splitter: Optional[Any] = None,
    *,
    seed: int,
    task: Literal["binary", "regression"] = "binary",
    k: int = 32,
    bandwidth_mult: float = 1.0,
    ridge: float = 0.1,
    standardize: bool = True,
    column_prefix: str = "loccls",
    dtype: type = np.float32,
) -> pl.DataFrame:
    """Locally-weighted classifier (binary) or regressor (regression) features per row.

    Output: 4 columns per row.
    Binary: ``proba``, ``logit``, ``train_acc``, ``coef_norm``.
    Regression: ``pred``, ``resid_std``, ``coef_norm``, ``r2``.
    """
    seed = require_seed(seed)
    validate_numeric_input(X_train, name="X_train", allow_fp16=False)
    if X_query is not None:
        validate_numeric_input(X_query, name="X_query", allow_fp16=False)
    if k < 4:
        raise ValueError(f"k must be >= 4; got {k}.")
    if task not in ("binary", "regression"):
        raise ValueError(f"task must be 'binary' or 'regression'; got {task!r}.")

    X_train_f = np.asarray(X_train, dtype=np.float32)
    y_train_f = np.asarray(y_train, dtype=np.float32).ravel()

    def _process(Xt: np.ndarray, Xq: np.ndarray, y_t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if standardize:
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler().fit(Xt)
            Xt_s = scaler.transform(Xt).astype(np.float32)
            Xq_s = scaler.transform(Xq).astype(np.float32)
        else:
            Xt_s = Xt
            Xq_s = Xq
        from sklearn.neighbors import NearestNeighbors
        k_used = min(k, Xt_s.shape[0])
        nn = NearestNeighbors(n_neighbors=k_used, algorithm="auto", n_jobs=-1).fit(Xt_s)
        dists, ids = nn.kneighbors(Xq_s)
        # Bandwidth = bandwidth_mult * median k-th NN distance.
        h = max(float(np.median(dists[:, -1])) * bandwidth_mult, 1e-3)
        n_q = Xq_s.shape[0]
        f1 = np.zeros(n_q, dtype=np.float32)
        f2 = np.zeros(n_q, dtype=np.float32)
        f3 = np.zeros(n_q, dtype=np.float32)
        f4 = np.zeros(n_q, dtype=np.float32)
        for i in range(n_q):
            x_q = Xq_s[i]
            neigh_ids = ids[i]
            X_local = Xt_s[neigh_ids]
            y_local = y_t[neigh_ids]
            w = np.exp(-(dists[i] ** 2) / (2 * h * h)).astype(np.float32)
            if task == "binary":
                # Degenerate-class handling: if all neighbours same class, fall back to majority vote.
                if y_local.min() == y_local.max():
                    f1[i] = float(y_local[0])
                    f2[i] = 20.0 if y_local[0] > 0.5 else -20.0
                    f3[i] = 1.0
                    f4[i] = 0.0
                else:
                    f1[i], f2[i], f3[i], f4[i] = _solve_weighted_logreg(X_local, y_local, w, x_q, ridge=ridge)
            else:
                f1[i], f2[i], f3[i], f4[i] = _solve_weighted_linreg(X_local, y_local, w, x_q, ridge=ridge)
        return f1, f2, f3, f4

    def _make_df(f1: np.ndarray, f2: np.ndarray, f3: np.ndarray, f4: np.ndarray) -> dict[str, np.ndarray]:
        if task == "binary":
            return {
                f"{column_prefix}_proba": f1.astype(dtype, copy=False),
                f"{column_prefix}_logit": f2.astype(dtype, copy=False),
                f"{column_prefix}_train_acc": f3.astype(dtype, copy=False),
                f"{column_prefix}_coef_norm": f4.astype(dtype, copy=False),
            }
        return {
            f"{column_prefix}_pred": f1.astype(dtype, copy=False),
            f"{column_prefix}_resid_std": f2.astype(dtype, copy=False),
            f"{column_prefix}_coef_norm": f3.astype(dtype, copy=False),
            f"{column_prefix}_r2": f4.astype(dtype, copy=False),
        }

    if X_query is not None:
        Xq = np.asarray(X_query, dtype=np.float32)
        f1, f2, f3, f4 = _process(X_train_f, Xq, y_train_f)
        return pl.DataFrame(_make_df(f1, f2, f3, f4))

    if splitter is None:
        raise ValueError("Mode A (X_query=None) requires a splitter.")
    n_train = X_train_f.shape[0]
    out_f1: np.ndarray = np.zeros(n_train, dtype=dtype)
    out_f2: np.ndarray = np.zeros(n_train, dtype=dtype)
    out_f3: np.ndarray = np.zeros(n_train, dtype=dtype)
    out_f4: np.ndarray = np.zeros(n_train, dtype=dtype)
    splits = list(splitter.split(X_train_f))
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        f1, f2, f3, f4 = _process(X_train_f[train_idx], X_train_f[val_idx], y_train_f[train_idx])
        out_f1[val_idx] = f1.astype(dtype, copy=False)
        out_f2[val_idx] = f2.astype(dtype, copy=False)
        out_f3[val_idx] = f3.astype(dtype, copy=False)
        out_f4[val_idx] = f4.astype(dtype, copy=False)
        logger.info("local_classifier: fold %d/%d done (n_q=%d, k=%d)", fold_idx + 1, len(splits), len(val_idx), k)

    return pl.DataFrame(_make_df(out_f1, out_f2, out_f3, out_f4))
