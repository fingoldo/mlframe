"""Pairwise KL/JS divergence features between 3 baselines.

Iter 76 mechanism. Info-theoretic agent's #2 ranked.

For each query: compute 3 pairwise KL divergences between 3 baseline predictive distributions + their
max + symmetric Jensen-Shannon divergence (mean over swaps). 5 features. Captures distributional
disagreement beyond point-prediction disagreement (iter 69).

Binary: Bernoulli KL.
Regression: Gaussian KL with per-baseline residual sigma estimated on train fold.
"""
from __future__ import annotations

import logging
from typing import Any, Literal, Optional

import numpy as np
import polars as pl

from ._utils import require_seed, validate_numeric_input

logger = logging.getLogger(__name__)


def _bernoulli_kl(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Elementwise KL(Bernoulli(p) || Bernoulli(q)), with both probabilities clipped away from 0/1 to avoid log(0)."""
    p_c = np.clip(p, 1e-6, 1 - 1e-6)
    q_c = np.clip(q, 1e-6, 1 - 1e-6)
    return np.asarray(p_c * np.log(p_c / q_c) + (1 - p_c) * np.log((1 - p_c) / (1 - q_c)))


def _gaussian_kl(mu_i: np.ndarray, mu_j: np.ndarray, sigma_i: float, sigma_j: float) -> np.ndarray:
    """Elementwise KL(N(mu_i, sigma_i^2) || N(mu_j, sigma_j^2)), variances tiny-epsilon-floored against zero-residual baselines."""
    var_i = sigma_i**2 + 1e-9
    var_j = sigma_j**2 + 1e-9
    return np.asarray(np.log(sigma_j / sigma_i) + (var_i + (mu_i - mu_j) ** 2) / (2.0 * var_j) - 0.5)


def _fit_3baselines_with_sigma(Xt: np.ndarray, y_t: np.ndarray, Xq: np.ndarray, task: str, seed: int):
    """Fit 3 deliberately-diverse baselines (shallow LGBM, deeper LGBM, linear/logistic) on the train fold and return their query-set predictions plus, for regression, each baseline's train-residual sigma (needed by the Gaussian KL)."""
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ImportError("pairwise_kl_divergence requires lightgbm") from exc
    from sklearn.linear_model import Ridge, LogisticRegression

    sigmas = [1.0, 1.0, 1.0]
    if task == "binary":
        m1 = lgb.LGBMClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=int(seed), verbose=-1, n_jobs=-1)
        m1.fit(Xt, y_t.astype(np.int32))
        p1_train = np.asarray(m1.predict_proba(Xt))[:, 1].astype(np.float32)
        p1_query = np.asarray(m1.predict_proba(Xq))[:, 1].astype(np.float32)
        m2 = lgb.LGBMClassifier(n_estimators=50, max_depth=5, learning_rate=0.1, random_state=int(seed) + 1, verbose=-1, n_jobs=-1)
        m2.fit(Xt, y_t.astype(np.int32))
        p2_train = np.asarray(m2.predict_proba(Xt))[:, 1].astype(np.float32)
        p2_query = np.asarray(m2.predict_proba(Xq))[:, 1].astype(np.float32)
        try:
            m3 = LogisticRegression(max_iter=200, solver="liblinear", random_state=int(seed) + 2)
            m3.fit(Xt, y_t.astype(np.int32))
            p3_train = m3.predict_proba(Xt)[:, 1].astype(np.float32)
            p3_query = m3.predict_proba(Xq)[:, 1].astype(np.float32)
        except Exception:
            prior = float(y_t.mean())
            p3_train = np.full(Xt.shape[0], prior, dtype=np.float32)
            p3_query = np.full(Xq.shape[0], prior, dtype=np.float32)
    else:
        m1 = lgb.LGBMRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=int(seed), verbose=-1, n_jobs=-1)
        m1.fit(Xt, y_t)
        p1_train = np.asarray(m1.predict(Xt)).astype(np.float32)
        p1_query = np.asarray(m1.predict(Xq)).astype(np.float32)
        m2 = lgb.LGBMRegressor(n_estimators=50, max_depth=5, learning_rate=0.1, random_state=int(seed) + 1, verbose=-1, n_jobs=-1)
        m2.fit(Xt, y_t)
        p2_train = np.asarray(m2.predict(Xt)).astype(np.float32)
        p2_query = np.asarray(m2.predict(Xq)).astype(np.float32)
        m3 = Ridge(alpha=1.0, random_state=int(seed) + 2)
        m3.fit(Xt, y_t)
        p3_train = m3.predict(Xt).astype(np.float32)
        p3_query = m3.predict(Xq).astype(np.float32)
        sigmas[0] = float((y_t - p1_train).std()) + 1e-6
        sigmas[1] = float((y_t - p2_train).std()) + 1e-6
        sigmas[2] = float((y_t - p3_train).std()) + 1e-6
    return (p1_query, p2_query, p3_query), sigmas


def compute_pairwise_kl_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: Optional[np.ndarray],
    splitter: Optional[Any] = None,
    *,
    seed: int,
    task: Literal["binary", "regression"] = "regression",
    standardize: bool = True,
    column_prefix: str = "pwkl",
    dtype: type = np.float32,
) -> pl.DataFrame:
    """Pairwise KL divergence features between 3 baselines.

    Output: 5 features per row — KL(1||2), KL(2||3), KL(1||3), max(KL), JS divergence.
    """
    seed = require_seed(seed)
    validate_numeric_input(X_train, name="X_train", allow_fp16=False)
    if X_query is not None:
        validate_numeric_input(X_query, name="X_query", allow_fp16=False)

    X_train_f = np.asarray(X_train, dtype=np.float32)
    y_train_f = np.asarray(y_train, dtype=np.float32).ravel()
    n_features = 5

    def _process(Xt: np.ndarray, Xq: np.ndarray, y_t: np.ndarray, fold_seed: int) -> np.ndarray:
        """Fit the 3 baselines on one (fold-local or global) train slice, then compute the 5 pairwise-KL/JS features for the query rows."""
        if standardize:
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler().fit(Xt)
            Xt_s = scaler.transform(Xt).astype(np.float32)
            Xq_s = scaler.transform(Xq).astype(np.float32)
        else:
            Xt_s = Xt
            Xq_s = Xq
        (p1, p2, p3), sigmas = _fit_3baselines_with_sigma(Xt_s, y_t, Xq_s, task=task, seed=fold_seed)
        if task == "binary":
            kl12 = _bernoulli_kl(p1, p2).astype(np.float32)
            kl23 = _bernoulli_kl(p2, p3).astype(np.float32)
            kl13 = _bernoulli_kl(p1, p3).astype(np.float32)
            mean_p = (p1 + p2 + p3) / 3.0
            js = ((_bernoulli_kl(p1, mean_p) + _bernoulli_kl(p2, mean_p) + _bernoulli_kl(p3, mean_p)) / 3.0).astype(np.float32)
        else:
            kl12 = _gaussian_kl(p1, p2, sigmas[0], sigmas[1]).astype(np.float32)
            kl23 = _gaussian_kl(p2, p3, sigmas[1], sigmas[2]).astype(np.float32)
            kl13 = _gaussian_kl(p1, p3, sigmas[0], sigmas[2]).astype(np.float32)
            # For Gaussian JS, use mixture mean.
            mean_mu = (p1 + p2 + p3) / 3.0
            mean_var = ((sigmas[0] ** 2 + sigmas[1] ** 2 + sigmas[2] ** 2) / 3.0) + ((p1 - mean_mu) ** 2 + (p2 - mean_mu) ** 2 + (p3 - mean_mu) ** 2) / 3.0
            mean_sigma = float(np.sqrt(mean_var.mean()))
            js = ((_gaussian_kl(p1, mean_mu, sigmas[0], mean_sigma) + _gaussian_kl(p2, mean_mu, sigmas[1], mean_sigma) + _gaussian_kl(p3, mean_mu, sigmas[2], mean_sigma)) / 3.0).astype(np.float32)
        max_kl = np.maximum(kl12, np.maximum(kl23, kl13)).astype(np.float32)
        return np.column_stack([kl12, kl23, kl13, max_kl, js])

    def _make_df(feats: np.ndarray) -> dict[str, np.ndarray]:
        """Cast the 5 raw feature columns to the requested output dtype and label them with the ``column_prefix``."""
        cols: dict[str, np.ndarray] = {}
        cols[f"{column_prefix}_kl_lgbd3_lgbd5"] = feats[:, 0].astype(dtype, copy=False)
        cols[f"{column_prefix}_kl_lgbd5_linear"] = feats[:, 1].astype(dtype, copy=False)
        cols[f"{column_prefix}_kl_lgbd3_linear"] = feats[:, 2].astype(dtype, copy=False)
        cols[f"{column_prefix}_max_kl"] = feats[:, 3].astype(dtype, copy=False)
        cols[f"{column_prefix}_js"] = feats[:, 4].astype(dtype, copy=False)
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
        logger.info("pairwise_kl_divergence: fold %d/%d done", fold_idx + 1, len(splits))

    return pl.DataFrame(_make_df(out))
