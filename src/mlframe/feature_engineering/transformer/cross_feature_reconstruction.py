"""Cross-feature reconstruction residuals (boosting autoencoder leave-one-feature-out).

Iter 95 mechanism. Agent C #2 ranked.

For each feature x_j: fit OOF LGB predicting x_j from {x_k : k≠j}. Emit standardized residuals.
NO y used — unsupervised feature-space outlierness, complementary to geometric mechanisms.

5 aggregate features per query: sum_sq_z_residuals, max_z_residual, mean_z_residual, n_extreme, log_l2_norm.
"""
from __future__ import annotations
import logging
import numpy as np
import polars as pl
from ._utils import require_seed, validate_numeric_input

logger = logging.getLogger(__name__)


def compute_cross_feature_reconstruction_features(
    X_train, y_train, X_query=None, splitter=None, *, seed, task="regression",
    n_estimators=30, max_depth=3, standardize=True, column_prefix="xfeat", dtype=np.float32,
):
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ImportError("cross_feature_reconstruction requires lightgbm") from exc
    seed = require_seed(seed)
    validate_numeric_input(X_train, name="X_train", allow_fp16=False)
    if X_query is not None:
        validate_numeric_input(X_query, name="X_query", allow_fp16=False)
    X_train_f = np.asarray(X_train, dtype=np.float32)
    n_features_out = 5

    def _process(Xt, Xq, fold_seed):
        if standardize:
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler().fit(Xt)
            Xt_s = scaler.transform(Xt).astype(np.float32)
            Xq_s = scaler.transform(Xq).astype(np.float32)
        else:
            Xt_s, Xq_s = Xt, Xq
        d = Xt_s.shape[1]
        z_residuals_q = np.zeros((Xq_s.shape[0], d), dtype=np.float32)
        for j in range(d):
            mask = np.ones(d, dtype=bool); mask[j] = False
            Xt_j_in = Xt_s[:, mask]
            Xq_j_in = Xq_s[:, mask]
            m = lgb.LGBMRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=0.1,
                                  random_state=int(fold_seed) + j, verbose=-1, n_jobs=-1).fit(Xt_j_in, Xt_s[:, j])
            x_hat_train = m.predict(Xt_j_in).astype(np.float32)
            r_train = Xt_s[:, j] - x_hat_train
            mad_j = float(np.median(np.abs(r_train - float(np.median(r_train))))) + 1e-6
            x_hat_q = m.predict(Xq_j_in).astype(np.float32)
            r_q = Xq_s[:, j] - x_hat_q
            z_residuals_q[:, j] = r_q / mad_j
        abs_z = np.abs(z_residuals_q)
        sum_sq = (z_residuals_q ** 2).sum(axis=1).astype(np.float32)
        max_z = abs_z.max(axis=1).astype(np.float32)
        mean_abs_z = abs_z.mean(axis=1).astype(np.float32)
        n_extreme = (abs_z > 3.0).sum(axis=1).astype(np.float32)
        log_l2 = np.log(np.sqrt(sum_sq) + 1e-9).astype(np.float32)
        return np.column_stack([sum_sq, max_z, mean_abs_z, n_extreme, log_l2])

    def _make_df(feats):
        cols = {}
        cols[f"{column_prefix}_sum_sq_z"] = feats[:, 0].astype(dtype, copy=False)
        cols[f"{column_prefix}_max_abs_z"] = feats[:, 1].astype(dtype, copy=False)
        cols[f"{column_prefix}_mean_abs_z"] = feats[:, 2].astype(dtype, copy=False)
        cols[f"{column_prefix}_n_extreme"] = feats[:, 3].astype(dtype, copy=False)
        cols[f"{column_prefix}_log_l2"] = feats[:, 4].astype(dtype, copy=False)
        return cols

    if X_query is not None:
        Xq = np.asarray(X_query, dtype=np.float32)
        return pl.DataFrame(_make_df(_process(X_train_f, Xq, seed)))
    if splitter is None:
        raise ValueError("Mode A requires splitter.")
    n_train = X_train_f.shape[0]
    out = np.zeros((n_train, n_features_out), dtype=dtype)
    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X_train_f)):
        out[val_idx] = _process(X_train_f[train_idx], X_train_f[val_idx], int(seed) + fold_idx * 100).astype(dtype, copy=False)
        logger.info("cross_feature_reconstruction: fold %d done", fold_idx + 1)
    return pl.DataFrame(_make_df(out))
