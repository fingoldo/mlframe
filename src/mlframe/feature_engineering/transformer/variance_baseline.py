"""Variance baseline: predict squared residual as a target.

Iter 87 mechanism. Agent C #3 ranked.

Two-stage OOF:
1. Fit mu_hat baseline (LGB d=3, 50 iter).
2. Fit sigma2_hat baseline on (y - mu_hat)² as target (LGB d=3, 50 iter).

Per query emit 5 leakage-free features:
- pred_mu (baseline prediction)
- pred_sigma2 (conditional variance)
- log_pred_sigma2 (log heteroscedasticity)
- snr_proxy = pred_mu / sqrt(pred_sigma2)
- relative_sigma2 = pred_sigma2 / global_train_var(y)

Structurally new: none of 86 mechanisms predicts residual² as a target.
"""
from __future__ import annotations

import logging
from typing import Any, Literal, Optional

import numpy as np
import polars as pl

from ._utils import require_seed, validate_numeric_input

logger = logging.getLogger(__name__)


def compute_variance_baseline_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: Optional[np.ndarray],
    splitter: Optional[Any] = None,
    *,
    seed: int,
    task: Literal["binary", "regression"] = "regression",
    standardize: bool = True,
    column_prefix: str = "varbase",
    dtype: type = np.float32,
) -> pl.DataFrame:
    """Variance-baseline features.

    Output: 5 features per row.
    """
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ImportError("variance_baseline requires lightgbm") from exc

    seed = require_seed(seed)
    validate_numeric_input(X_train, name="X_train", allow_fp16=False)
    if X_query is not None:
        validate_numeric_input(X_query, name="X_query", allow_fp16=False)

    X_train_f = np.asarray(X_train, dtype=np.float32)
    y_train_f = np.asarray(y_train, dtype=np.float32).ravel()
    n_features_out = 5

    def _process(Xt: np.ndarray, Xq: np.ndarray, y_t: np.ndarray, fold_seed: int) -> np.ndarray:
        """Fit the mu_hat baseline, fit a second baseline on the squared residual to get sigma2_hat, then derive the 5 heteroscedasticity features on the query rows."""
        if standardize:
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler().fit(Xt)
            Xt_s = scaler.transform(Xt).astype(np.float32)
            Xq_s = scaler.transform(Xq).astype(np.float32)
        else:
            Xt_s = Xt
            Xq_s = Xq
        # Stage 1: mu_hat
        if task == "binary":
            m_mu = lgb.LGBMClassifier(n_estimators=50, max_depth=3, learning_rate=0.1,
                                      random_state=int(fold_seed), verbose=-1, n_jobs=-1).fit(Xt_s, y_t.astype(np.int32))
            mu_train = m_mu.predict_proba(Xt_s)[:, 1].astype(np.float32)
            mu_query = m_mu.predict_proba(Xq_s)[:, 1].astype(np.float32)
        else:
            m_mu = lgb.LGBMRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=int(fold_seed), verbose=-1, n_jobs=-1).fit(Xt_s, y_t)
            mu_train = m_mu.predict(Xt_s).astype(np.float32)
            mu_query = m_mu.predict(Xq_s).astype(np.float32)

        # Stage 2: sigma2_hat
        squared_resid_train = ((y_t - mu_train) ** 2).astype(np.float32)
        m_sigma = lgb.LGBMRegressor(n_estimators=50, max_depth=3, learning_rate=0.1,
                                    random_state=int(fold_seed) + 1, verbose=-1, n_jobs=-1).fit(Xt_s, squared_resid_train)
        sigma2_query = np.clip(m_sigma.predict(Xq_s).astype(np.float32), 1e-9, None)

        log_sigma2 = np.log(sigma2_query)
        sigma_sqrt = np.sqrt(sigma2_query)
        snr_proxy = (mu_query / (sigma_sqrt + 1e-9)).astype(np.float32)
        global_var = float(y_t.var()) + 1e-9
        rel_sigma2 = (sigma2_query / global_var).astype(np.float32)

        return np.column_stack([mu_query, sigma2_query, log_sigma2, snr_proxy, rel_sigma2])

    def _make_df(feats: np.ndarray) -> dict[str, np.ndarray]:
        """Split the (n, 5) variance-baseline feature matrix into named, dtype-cast columns for the output polars frame."""
        cols: dict[str, np.ndarray] = {}
        cols[f"{column_prefix}_mu"] = feats[:, 0].astype(dtype, copy=False)
        cols[f"{column_prefix}_sigma2"] = feats[:, 1].astype(dtype, copy=False)
        cols[f"{column_prefix}_log_sigma2"] = feats[:, 2].astype(dtype, copy=False)
        cols[f"{column_prefix}_snr"] = feats[:, 3].astype(dtype, copy=False)
        cols[f"{column_prefix}_rel_sigma2"] = feats[:, 4].astype(dtype, copy=False)
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
        logger.info("variance_baseline: fold %d/%d done", fold_idx + 1, len(splits))

    return pl.DataFrame(_make_df(out))
