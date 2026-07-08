"""Baseline surprise features (leakage-free): -log p(y_train | baseline_pred) aggregated to queries via kNN.

Iter 73 mechanism. Information-theoretic agent's #1 ranked recommendation, redesigned for leakage-free OOF +
production-safe Mode B (queries get features via kNN-aggregation of train surprises, never use y_query).

Mechanism:
1. Fit 50-iter LGB baseline on train fold → predictions on both train and query.
2. Compute surprise PER TRAIN ROW:
   - Binary: -log p̂ if y=1 else -log(1-p̂)
   - Regression: (y - ŷ)² / (2σ²), with σ = std of train residuals
3. Build kNN index on standardized train X.
4. For each query row:
   - Find K=32 nearest train rows
   - Emit:
     - mean_surprise_neighbors
     - max_surprise_neighbors
     - std_surprise_neighbors
     - baseline_pred (query's own ŷ or p̂)
     - frac_high_surprise (fraction of neighbors above median train surprise)
5. 5 features per query. NO y_query used → leak-free for both OOF and Mode B.

Why this is structurally novel vs iter 60-71:
- Iter 60-63 |residual|-bands use y in training-fold surprise quantiles, query routes via band CENTROIDS.
- Iter 71 NN-target-mean uses y in OOF embedding space, but K=200/500 was too large for small-N.
- Iter 73 NN-surprise-mean uses smaller K=32 and per-row surprise (not target) → more localized signal.

Leakage discipline: iter 60-pattern — train y used to define surprise on train rows; query rows never see
their own y in any feature. K=32 is well below 614-row train fold size, no global-collapse failure mode.
"""
from __future__ import annotations

import logging
from typing import Any, Literal, Optional

import numpy as np
import polars as pl

from ._utils import require_seed, validate_numeric_input

logger = logging.getLogger(__name__)


def _fit_baseline_predict(Xt: np.ndarray, y_t: np.ndarray, Xq: np.ndarray, task: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ImportError("baseline_surprise requires lightgbm") from exc
    if task == "binary":
        m = lgb.LGBMClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=int(seed), verbose=-1, n_jobs=-1)
        m.fit(Xt, y_t.astype(np.int32))
        p_train = m.predict_proba(Xt)[:, 1].astype(np.float32)
        p_query = m.predict_proba(Xq)[:, 1].astype(np.float32)
    else:
        m = lgb.LGBMRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=int(seed), verbose=-1, n_jobs=-1)
        m.fit(Xt, y_t)
        p_train = m.predict(Xt).astype(np.float32)
        p_query = m.predict(Xq).astype(np.float32)
    return p_train, p_query


def compute_baseline_surprise_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: Optional[np.ndarray],
    splitter: Optional[Any] = None,
    *,
    seed: int,
    task: Literal["binary", "regression"] = "regression",
    k_neighbors: int = 32,
    standardize: bool = True,
    column_prefix: str = "surprise",
    dtype: type = np.float32,
) -> pl.DataFrame:
    """Per-query NN-aggregated train surprise. Leakage-free for both OOF and Mode B."""
    from sklearn.neighbors import NearestNeighbors

    seed = require_seed(seed)
    validate_numeric_input(X_train, name="X_train", allow_fp16=False)
    if X_query is not None:
        validate_numeric_input(X_query, name="X_query", allow_fp16=False)

    X_train_f = np.asarray(X_train, dtype=np.float32)
    y_train_f = np.asarray(y_train, dtype=np.float32).ravel()
    n_features = 5

    def _process(Xt: np.ndarray, Xq: np.ndarray, y_t: np.ndarray, fold_seed: int) -> np.ndarray:
        if standardize:
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler().fit(Xt)
            Xt_s = scaler.transform(Xt).astype(np.float32)
            Xq_s = scaler.transform(Xq).astype(np.float32)
        else:
            Xt_s = Xt
            Xq_s = Xq
        p_train, p_query = _fit_baseline_predict(Xt_s, y_t, Xq_s, task=task, seed=fold_seed)
        # Compute per-train-row surprise (uses y_train — OK).
        if task == "binary":
            p_train_c = np.clip(p_train, 1e-6, 1 - 1e-6)
            surprise_train = (-y_t * np.log(p_train_c) - (1.0 - y_t) * np.log(1.0 - p_train_c)).astype(np.float32)
        else:
            resid = (y_t - p_train).astype(np.float32)
            sigma = float(resid.std()) + 1e-6
            surprise_train = ((resid**2) / (2.0 * sigma * sigma)).astype(np.float32)

        # kNN aggregation of train surprises onto query rows.
        k_eff = min(k_neighbors, Xt_s.shape[0])
        nn = NearestNeighbors(n_neighbors=k_eff, n_jobs=-1).fit(Xt_s)
        _, idx = nn.kneighbors(Xq_s)
        nbr_surp = surprise_train[idx]  # (n_q, k_eff)
        median_train_surp = float(np.median(surprise_train))

        mean_surp = nbr_surp.mean(axis=1).astype(np.float32)
        max_surp = nbr_surp.max(axis=1).astype(np.float32)
        std_surp = nbr_surp.std(axis=1).astype(np.float32) + 1e-9
        frac_high = (nbr_surp > median_train_surp).mean(axis=1).astype(np.float32)
        return np.column_stack([mean_surp, max_surp, std_surp, p_query, frac_high])

    def _make_df(feats: np.ndarray) -> dict[str, np.ndarray]:
        cols: dict[str, np.ndarray] = {}
        cols[f"{column_prefix}_nbr_mean"] = feats[:, 0].astype(dtype, copy=False)
        cols[f"{column_prefix}_nbr_max"] = feats[:, 1].astype(dtype, copy=False)
        cols[f"{column_prefix}_nbr_std"] = feats[:, 2].astype(dtype, copy=False)
        cols[f"{column_prefix}_baseline_pred"] = feats[:, 3].astype(dtype, copy=False)
        cols[f"{column_prefix}_frac_high"] = feats[:, 4].astype(dtype, copy=False)
        return cols

    if X_query is not None:
        Xq = np.asarray(X_query, dtype=np.float32)
        feats = _process(X_train_f, Xq, y_train_f, seed)
        return pl.DataFrame(_make_df(feats))

    if splitter is None:
        raise ValueError("Mode A (X_query=None) requires a splitter.")
    n_train = X_train_f.shape[0]
    out: np.ndarray = np.zeros((n_train, n_features), dtype=dtype)
    splits = list(splitter.split(X_train_f))
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        feats = _process(X_train_f[train_idx], X_train_f[val_idx], y_train_f[train_idx], int(seed) + fold_idx * 100)
        out[val_idx] = feats.astype(dtype, copy=False)
        logger.info("baseline_surprise: fold %d/%d done", fold_idx + 1, len(splits))

    return pl.DataFrame(_make_df(out))
