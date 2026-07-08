"""Predictive info delta: H(y) - H(y | baseline_pred_bin) per row.

Iter 82 mechanism. Info-theoretic agent's #5 ranked.

Per row: emit predictive_info = H(y) - H(y | baseline_pred_bin). Bin baseline predictions into K=10
quantiles; per bin compute conditional entropy/variance of y; assign each query to its baseline-bin and
emit the entropy reduction.

5 features: pred_info_delta, baseline_pred, baseline_bin, H_y_given_bin, H_y_marginal.
"""
from __future__ import annotations

import logging
from typing import Any, Literal, Optional

import numpy as np
import polars as pl

from ._utils import require_seed, validate_numeric_input

logger = logging.getLogger(__name__)


def compute_predictive_info_delta_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: Optional[np.ndarray],
    splitter: Optional[Any] = None,
    *,
    seed: int,
    task: Literal["binary", "regression"] = "regression",
    n_bins: int = 10,
    standardize: bool = True,
    column_prefix: str = "pinfo",
    dtype: type = np.float32,
) -> pl.DataFrame:
    """Predictive info delta features.

    Output: 5 features — pred_info_delta, baseline_pred, baseline_bin, H_y_given_bin, H_y_marginal.
    """
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ImportError("predictive_info_delta requires lightgbm") from exc

    seed = require_seed(seed)
    validate_numeric_input(X_train, name="X_train", allow_fp16=False)
    if X_query is not None:
        validate_numeric_input(X_query, name="X_query", allow_fp16=False)

    X_train_f = np.asarray(X_train, dtype=np.float32)
    y_train_f = np.asarray(y_train, dtype=np.float32).ravel()
    n_features_out = 5

    def _process(Xt: np.ndarray, Xq: np.ndarray, y_t: np.ndarray, fold_seed: int) -> np.ndarray:
        """Per-fold feature block: fit a small baseline LightGBM, bin its train predictions into ``n_bins`` quantile buckets with a per-bin conditional entropy H(y|bin), assign query rows to bins, and return the marginal-minus-conditional information delta (H(y) - H(y|bin)) plus the baseline prediction, bin index, conditional entropy, and marginal entropy per query row."""
        if standardize:
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler().fit(Xt)
            Xt_s = scaler.transform(Xt).astype(np.float32)
            Xq_s = scaler.transform(Xq).astype(np.float32)
        else:
            Xt_s = Xt
            Xq_s = Xq
        if task == "binary":
            model = lgb.LGBMClassifier(n_estimators=50, max_depth=3, learning_rate=0.1,
                                       random_state=int(fold_seed), verbose=-1, n_jobs=-1).fit(Xt_s, y_t.astype(np.int32))
            p_train = model.predict_proba(Xt_s)[:, 1].astype(np.float32)
            p_query = model.predict_proba(Xq_s)[:, 1].astype(np.float32)
            # H(y) marginal: Bernoulli entropy of class prior.
            p_mean = float(y_t.mean())
            p_mean_c = max(min(p_mean, 1 - 1e-6), 1e-6)
            H_y = float(-p_mean_c * np.log(p_mean_c) - (1 - p_mean_c) * np.log(1 - p_mean_c))
        else:
            model = lgb.LGBMRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=int(fold_seed), verbose=-1, n_jobs=-1).fit(Xt_s, y_t)
            p_train = model.predict(Xt_s).astype(np.float32)
            p_query = model.predict(Xq_s).astype(np.float32)
            # H(y) marginal: 0.5 * log(2π e σ²) for Gaussian — use just log(σ²) per row scaling.
            H_y = float(np.log(np.var(y_t) + 1e-9))

        # Bin train baseline predictions.
        bin_edges = np.quantile(p_train, np.linspace(0.0, 1.0, n_bins + 1))
        train_bin = np.zeros(p_train.shape[0], dtype=np.int32)
        H_y_per_bin = np.zeros(n_bins, dtype=np.float32)
        for b in range(n_bins):
            if b == 0:
                mask = p_train <= bin_edges[b + 1]
            elif b == n_bins - 1:
                mask = p_train > bin_edges[b]
            else:
                mask = (p_train > bin_edges[b]) & (p_train <= bin_edges[b + 1])
            train_bin[mask] = b
            if mask.sum() > 0:
                if task == "binary":
                    p_bin = max(min(float(y_t[mask].mean()), 1 - 1e-6), 1e-6)
                    H_y_per_bin[b] = float(-p_bin * np.log(p_bin) - (1 - p_bin) * np.log(1 - p_bin))
                else:
                    H_y_per_bin[b] = float(np.log(np.var(y_t[mask]) + 1e-9))

        # Assign query rows to bins.
        query_bin = np.zeros(p_query.shape[0], dtype=np.int32)
        for b in range(n_bins):
            if b == 0:
                mask = p_query <= bin_edges[b + 1]
            elif b == n_bins - 1:
                mask = p_query > bin_edges[b]
            else:
                mask = (p_query > bin_edges[b]) & (p_query <= bin_edges[b + 1])
            query_bin[mask] = b
        query_H_given_bin = H_y_per_bin[query_bin]
        pred_info_delta = (H_y - query_H_given_bin).astype(np.float32)

        return np.column_stack([
            pred_info_delta,
            p_query,
            query_bin.astype(np.float32),
            query_H_given_bin,
            np.full(p_query.shape[0], H_y, dtype=np.float32),
        ])

    def _make_df(feats: np.ndarray) -> dict[str, np.ndarray]:
        """Reshape the flat ``_process`` output block into named ``{prefix}_delta`` / ``_baseline_pred`` / ``_bin`` / ... columns."""
        cols: dict[str, np.ndarray] = {}
        cols[f"{column_prefix}_delta"] = feats[:, 0].astype(dtype, copy=False)
        cols[f"{column_prefix}_baseline_pred"] = feats[:, 1].astype(dtype, copy=False)
        cols[f"{column_prefix}_bin"] = feats[:, 2].astype(dtype, copy=False)
        cols[f"{column_prefix}_H_given_bin"] = feats[:, 3].astype(dtype, copy=False)
        cols[f"{column_prefix}_H_marginal"] = feats[:, 4].astype(dtype, copy=False)
        return cols

    if X_query is not None:
        Xq = np.asarray(X_query, dtype=np.float32)
        feats = _process(X_train_f, Xq, y_train_f, seed)
        return pl.DataFrame(_make_df(feats))

    if splitter is None:
        raise ValueError("Mode A (X_query=None) requires a splitter.")
    n_train = X_train_f.shape[0]
    out: np.ndarray = np.zeros((n_train, n_features_out), dtype=dtype)
    splits = list(splitter.split(X_train_f))
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        feats = _process(X_train_f[train_idx], X_train_f[val_idx], y_train_f[train_idx], int(seed) + fold_idx * 100)
        out[val_idx] = feats.astype(dtype, copy=False)
        logger.info("predictive_info_delta: fold %d/%d done", fold_idx + 1, len(splits))

    return pl.DataFrame(_make_df(out))
