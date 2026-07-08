"""Fisher-weighted residual band features.

Iter 81 mechanism. Info-theoretic agent's #3 ranked. Curvature-aware residual bands.

For each row, compute Fisher information proxy = ||∇p̂(x)||² via finite-difference gradient norm
of baseline LGB prediction w.r.t. input features. Weight residuals by sqrt(Fisher); quintile-band the
weighted residuals; emit weighted band id + per-band agg_y + raw Fisher + raw residual.

5 features. Second-order curvature info (none of iter 60-80 used it).
"""
from __future__ import annotations

import logging
from typing import Any, Literal, Optional

import numpy as np
import polars as pl

from ._utils import require_seed, validate_numeric_input

logger = logging.getLogger(__name__)

# Cap on the vertically-stacked perturbation matrix (float32 elems). 64M ~ 256MB; above this
# fall back to the per-feature loop to bound peak RAM.
_MAX_STACK_ELEMS = 64_000_000


def compute_fisher_weighted_residual_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: Optional[np.ndarray],
    splitter: Optional[Any] = None,
    *,
    seed: int,
    task: Literal["binary", "regression"] = "regression",
    n_bands: int = 5,
    eps: float = 0.05,
    standardize: bool = True,
    column_prefix: str = "fishres",
    dtype: type = np.float32,
) -> pl.DataFrame:
    """Fisher-weighted residual band features.

    Output: 5 features — weighted_band_id, agg_y_at_band, fisher_norm, raw_residual_proxy, weighted_residual_proxy.
    """
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ImportError("fisher_weighted_residual requires lightgbm") from exc

    seed = require_seed(seed)
    validate_numeric_input(X_train, name="X_train", allow_fp16=False)
    if X_query is not None:
        validate_numeric_input(X_query, name="X_query", allow_fp16=False)

    X_train_f = np.asarray(X_train, dtype=np.float32)
    y_train_f = np.asarray(y_train, dtype=np.float32).ravel()
    n_features_out = 5

    def _gradient_norm(model, X: np.ndarray, is_binary: bool) -> np.ndarray:
        n, d = X.shape
        if is_binary:
            p_base = model.predict_proba(X)[:, 1].astype(np.float32)
        else:
            p_base = model.predict(X).astype(np.float32)
        grad_sq_sum = np.zeros(n, dtype=np.float32)
        # Batched finite-difference: stack d copies of X into one (d*n, d) matrix, add eps to
        # block j's column j, then a SINGLE predict over the stack instead of d separate calls.
        # Bit-identical to the per-feature predict (tree models predict per-row independently);
        # amortizes LightGBM per-call overhead. Gated on stack size to bound peak RAM.
        stack_elems = d * n * d
        if stack_elems <= _MAX_STACK_ELEMS and d > 1:
            stack = np.broadcast_to(X, (d, n, d)).reshape(d * n, d).copy()
            block_starts = np.arange(d) * n
            for j in range(d):
                stack[block_starts[j] : block_starts[j] + n, j] += eps
            if is_binary:
                p_plus_all = model.predict_proba(stack)[:, 1].astype(np.float32).reshape(d, n)
            else:
                p_plus_all = model.predict(stack).astype(np.float32).reshape(d, n)
            # Accumulate in the SAME sequential per-feature order as the fallback loop so the
            # float32 reduction order (and thus the result) is bit-identical, not pairwise-summed.
            contrib = (((p_plus_all - p_base[None, :]) / eps) ** 2).astype(np.float32)
            for j in range(d):
                grad_sq_sum += contrib[j]
        else:
            for j in range(d):
                X_plus = X.copy()
                X_plus[:, j] += eps
                if is_binary:
                    p_plus = model.predict_proba(X_plus)[:, 1].astype(np.float32)
                else:
                    p_plus = model.predict(X_plus).astype(np.float32)
                grad_sq_sum += ((p_plus - p_base) / eps) ** 2
        return np.sqrt(grad_sq_sum) + 1e-9

    def _process(Xt: np.ndarray, Xq: np.ndarray, y_t: np.ndarray, fold_seed: int) -> np.ndarray:
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
            is_binary = True
            # Residual proxy: -log p(y|x) per train row.
            p_train_c = np.clip(p_train, 1e-6, 1 - 1e-6)
            resid_train = (-y_t * np.log(p_train_c) - (1 - y_t) * np.log(1 - p_train_c)).astype(np.float32)
            # Query "residual proxy": entropy of prediction (uncertainty).
            p_query_c = np.clip(p_query, 1e-6, 1 - 1e-6)
            resid_query = (-p_query_c * np.log(p_query_c) - (1 - p_query_c) * np.log(1 - p_query_c)).astype(np.float32)
        else:
            model = lgb.LGBMRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=int(fold_seed), verbose=-1, n_jobs=-1).fit(Xt_s, y_t)
            p_train = model.predict(Xt_s).astype(np.float32)
            p_query = model.predict(Xq_s).astype(np.float32)
            is_binary = False
            resid_train = np.abs(y_t - p_train).astype(np.float32)
            # Query residual proxy: prediction variance proxy = |p_query - train_y_median|
            resid_query = np.abs(p_query - float(np.median(y_t))).astype(np.float32)

        # Fisher norm per train + query row.
        fisher_train = _gradient_norm(model, Xt_s, is_binary).astype(np.float32)
        fisher_query = _gradient_norm(model, Xq_s, is_binary).astype(np.float32)

        # Weighted residual = |residual| × sqrt(Fisher); use train to set band thresholds.
        weighted_train = resid_train * np.sqrt(fisher_train)
        weighted_query = resid_query * np.sqrt(fisher_query)

        quantiles = np.quantile(weighted_train, np.linspace(0.0, 1.0, n_bands + 1))
        band_y_mean = np.zeros(n_bands, dtype=np.float32)
        for b in range(n_bands):
            if b == 0:
                mask = weighted_train <= quantiles[b + 1]
            elif b == n_bands - 1:
                mask = weighted_train > quantiles[b]
            else:
                mask = (weighted_train > quantiles[b]) & (weighted_train <= quantiles[b + 1])
            if mask.sum() > 0:
                band_y_mean[b] = float(y_t[mask].mean())

        # Assign query rows to bands.
        query_band = np.zeros(weighted_query.shape[0], dtype=np.int32)
        for b in range(n_bands):
            if b == 0:
                mask = weighted_query <= quantiles[b + 1]
            elif b == n_bands - 1:
                mask = weighted_query > quantiles[b]
            else:
                mask = (weighted_query > quantiles[b]) & (weighted_query <= quantiles[b + 1])
            query_band[mask] = b
        agg_y = band_y_mean[query_band].astype(np.float32)

        return np.column_stack([
            query_band.astype(np.float32),
            agg_y,
            fisher_query,
            resid_query,
            weighted_query.astype(np.float32),
        ])

    def _make_df(feats: np.ndarray) -> dict[str, np.ndarray]:
        cols: dict[str, np.ndarray] = {}
        cols[f"{column_prefix}_band"] = feats[:, 0].astype(dtype, copy=False)
        cols[f"{column_prefix}_band_y_mean"] = feats[:, 1].astype(dtype, copy=False)
        cols[f"{column_prefix}_fisher_norm"] = feats[:, 2].astype(dtype, copy=False)
        cols[f"{column_prefix}_residual_proxy"] = feats[:, 3].astype(dtype, copy=False)
        cols[f"{column_prefix}_weighted_residual"] = feats[:, 4].astype(dtype, copy=False)
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
        logger.info("fisher_weighted_residual: fold %d/%d done", fold_idx + 1, len(splits))

    return pl.DataFrame(_make_df(out))
