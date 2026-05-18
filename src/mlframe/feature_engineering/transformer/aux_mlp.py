"""Aux MLP classifier features: sklearn MLPClassifier OOF predictions as features.

Originally implemented as iter 38 attempt. The user redirected toward proper learned-attention (NCA-projection); this module is retained per the "never delete
feature-computing code" rule — we'll re-test on a broader dataset matrix later. Some datasets where MLP differs algorithmically from tree-boostings may benefit
from these features as a stacking signal even if mammography doesn't.

Mechanism (binary):
- Per fold, train sklearn MLPClassifier (1 hidden layer of `hidden_size` units, ReLU, Adam optimizer).
- OOF-predict for val fold.
- Expose 2 features: ``mlp_proba`` (sigmoid output) and ``mlp_logit`` (logit transform).

For regression: MLPRegressor with same architecture. Outputs (``mlp_pred``, placeholder ``mlp_resid``=0 at inference).

This is gradient-trained backprop via sklearn's Adam, so technically beyond-frozen — but it's just stacking, not structural learned-attention. Kept for completeness
and future broad-dataset retest.

Leakage discipline: MLP refit per fold; OOF predictions per fold; train-fold rows only.

Reference: Wolpert 1992 — stacked generalization. sklearn.neural_network.MLPClassifier.
"""
from __future__ import annotations

import logging
import warnings
from typing import Any, Literal, Optional

import numpy as np
import polars as pl

from ._utils import require_seed, validate_numeric_input

logger = logging.getLogger(__name__)


def _fit_aux_mlp(X: np.ndarray, y: np.ndarray, *, task: str, seed: int, hidden_size: int, max_iter: int):
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(X)
    X_s = scaler.transform(X)
    if task == "binary":
        model = MLPClassifier(
            hidden_layer_sizes=(hidden_size,), activation="relu", solver="adam",
            alpha=1e-3, learning_rate_init=0.01, max_iter=max_iter, random_state=seed,
            early_stopping=True, validation_fraction=0.1, n_iter_no_change=10, tol=1e-4,
        )
    else:
        model = MLPRegressor(
            hidden_layer_sizes=(hidden_size,), activation="relu", solver="adam",
            alpha=1e-3, learning_rate_init=0.01, max_iter=max_iter, random_state=seed,
            early_stopping=True, validation_fraction=0.1, n_iter_no_change=10, tol=1e-4,
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_s, y)
    return model, scaler


def _predict_aux_mlp(model, scaler, X: np.ndarray, task: str) -> tuple[np.ndarray, np.ndarray]:
    X_s = scaler.transform(X)
    if task == "binary":
        proba = model.predict_proba(X_s)[:, 1].astype(np.float32)
        proba_clipped = np.clip(proba, 1e-6, 1.0 - 1e-6)
        logit = np.log(proba_clipped / (1.0 - proba_clipped)).astype(np.float32)
        return proba, logit
    pred = model.predict(X_s).astype(np.float32)
    return pred, np.zeros_like(pred)


def compute_aux_mlp_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: Optional[np.ndarray],
    splitter: Optional[Any] = None,
    *,
    seed: int,
    task: Literal["binary", "regression"] = "binary",
    hidden_size: int = 16,
    max_iter: int = 300,
    column_prefix: str = "mlp",
    dtype: np.dtype = np.float32,
) -> pl.DataFrame:
    """Aux MLP OOF predictions + logit as features (2 columns)."""
    seed = require_seed(seed)
    validate_numeric_input(X_train, name="X_train", allow_fp16=False)
    if X_query is not None:
        validate_numeric_input(X_query, name="X_query", allow_fp16=False)
    if task not in ("binary", "regression"):
        raise ValueError(f"task must be 'binary' or 'regression'; got {task!r}.")

    X_train_f = np.asarray(X_train, dtype=np.float32)
    y_train_f = np.asarray(y_train, dtype=np.float32).ravel()

    if X_query is not None:
        Xq = np.asarray(X_query, dtype=np.float32)
        model, scaler = _fit_aux_mlp(X_train_f, y_train_f, task=task, seed=seed, hidden_size=hidden_size, max_iter=max_iter)
        f1, f2 = _predict_aux_mlp(model, scaler, Xq, task=task)
        if task == "binary":
            return pl.DataFrame({f"{column_prefix}_proba": f1.astype(dtype, copy=False), f"{column_prefix}_logit": f2.astype(dtype, copy=False)})
        return pl.DataFrame({f"{column_prefix}_pred": f1.astype(dtype, copy=False), f"{column_prefix}_resid": f2.astype(dtype, copy=False)})

    if splitter is None:
        raise ValueError("Mode A (X_query=None) requires a splitter.")
    n_train = X_train_f.shape[0]
    out_f1 = np.zeros(n_train, dtype=dtype)
    out_f2 = np.zeros(n_train, dtype=dtype)
    splits = list(splitter.split(X_train_f))
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        model, scaler = _fit_aux_mlp(X_train_f[train_idx], y_train_f[train_idx], task=task, seed=int(seed) + fold_idx * 17, hidden_size=hidden_size, max_iter=max_iter)
        f1, f2 = _predict_aux_mlp(model, scaler, X_train_f[val_idx], task=task)
        out_f1[val_idx] = f1.astype(dtype, copy=False)
        out_f2[val_idx] = f2.astype(dtype, copy=False)
        logger.info("aux_mlp: fold %d/%d done", fold_idx + 1, len(splits))

    if task == "binary":
        return pl.DataFrame({f"{column_prefix}_proba": out_f1, f"{column_prefix}_logit": out_f2})
    return pl.DataFrame({f"{column_prefix}_pred": out_f1, f"{column_prefix}_resid": out_f2})
