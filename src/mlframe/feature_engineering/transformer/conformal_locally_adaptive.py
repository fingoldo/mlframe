"""Split-conformal interval width with locally-adaptive sigma_hat(x) (Mondrian-conformal).

Iter 93 mechanism. Agent B #1 ranked.

Per OOF fold: fit baseline on train-half, compute nonconformity = |y - ŷ| / sigma_hat(x) on calibration-half
where sigma_hat is kNN-local MAD of residuals. Emit locally-normalized conformal interval half-width
at α ∈ {0.1, 0.2}, plus the row's own sigma_hat and z-score-style ratio.

5 features.
"""
from __future__ import annotations
import logging
import numpy as np
import polars as pl
from ._utils import require_seed, validate_numeric_input

logger = logging.getLogger(__name__)


def compute_conformal_locally_adaptive_features(
    X_train, y_train, X_query=None, splitter=None, *, seed, task="regression",
    k_sigma=16, alphas=(0.1, 0.2), standardize=True, column_prefix="cla", dtype=np.float32,
):
    """Mondrian split-conformal interval-width features: fit a baseline on half the train fold, compute nonconformity scores locally normalized by a kNN-MAD sigma_hat on the other half, and emit the resulting interval half-widths at each alpha plus sigma_hat, the point prediction, and a width ratio."""
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ImportError("conformal_locally_adaptive requires lightgbm") from exc
    from sklearn.neighbors import NearestNeighbors

    seed = require_seed(seed)
    validate_numeric_input(X_train, name="X_train", allow_fp16=False)
    if X_query is not None:
        validate_numeric_input(X_query, name="X_query", allow_fp16=False)
    X_train_f = np.asarray(X_train, dtype=np.float32)
    y_train_f = np.asarray(y_train, dtype=np.float32).ravel()
    n_features_out = 5

    def _local_sigma(Xt_s, residuals, X_target, k):
        """kNN-local MAD of residuals: for each row in ``X_target``, find its ``k`` nearest neighbours in ``Xt_s`` and return the median absolute deviation of their residuals (epsilon-floored to avoid a zero denominator)."""
        k_eff = min(k, Xt_s.shape[0])
        nn = NearestNeighbors(n_neighbors=k_eff, n_jobs=-1).fit(Xt_s)
        _, idx = nn.kneighbors(X_target)
        nbr_resid = residuals[idx]
        nbr_med = np.median(nbr_resid, axis=1, keepdims=True)
        return np.median(np.abs(nbr_resid - nbr_med), axis=1).astype(np.float32) + 1e-6

    def _process(Xt, Xq, y_t, fold_seed):
        """Split the train fold in half, fit the baseline on h1, compute locally-normalized nonconformity scores on h2, then derive per-query conformal widths at each alpha, sigma_hat, the point prediction, and their ratio. Returns an all-zero block on a tiny (n<4) train fold since an empty half would otherwise fail the fit / quantile opaquely."""
        if standardize:
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler().fit(Xt)
            Xt_s = scaler.transform(Xt).astype(np.float32)
            Xq_s = scaler.transform(Xq).astype(np.float32)
        else:
            Xt_s, Xq_s = Xt, Xq
        n = Xt_s.shape[0]
        # Wave 39 (2026-05-20): tiny-train regime (n<4) empties h1 (n//2==0) or h2,
        # then lgb.fit on empty raises opaquely and downstream sigma/quantile produce garbage.
        # Return a zero-feature block matching the expected (Xq_s.shape[0], 5) shape.
        if n < 4:
            return np.zeros((Xq_s.shape[0], 5), dtype=np.float32)
        rng = np.random.default_rng(int(fold_seed))
        idx = np.arange(n); rng.shuffle(idx)
        h1, h2 = idx[: n // 2], idx[n // 2 :]
        if task == "binary":
            m = lgb.LGBMClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=int(fold_seed), verbose=-1, n_jobs=-1).fit(Xt_s[h1], y_t[h1].astype(np.int32))
            preds_h2 = m.predict_proba(Xt_s[h2])[:, 1].astype(np.float32)
            preds_q = m.predict_proba(Xq_s)[:, 1].astype(np.float32)
        else:
            m = lgb.LGBMRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=int(fold_seed), verbose=-1, n_jobs=-1).fit(Xt_s[h1], y_t[h1])
            preds_h2 = m.predict(Xt_s[h2]).astype(np.float32)
            preds_q = m.predict(Xq_s).astype(np.float32)
        abs_resid_h2 = np.abs(y_t[h2] - preds_h2)
        sigma_h2 = _local_sigma(Xt_s[h2], abs_resid_h2, Xt_s[h2], k_sigma)
        nonconf_scores = abs_resid_h2 / sigma_h2
        # Quantile of nonconformity at each alpha
        widths_q = np.zeros((Xq_s.shape[0], len(alphas)), dtype=np.float32)
        sigma_q = _local_sigma(Xt_s[h2], abs_resid_h2, Xq_s, k_sigma)
        for i, a in enumerate(alphas):
            q = float(np.quantile(nonconf_scores, 1.0 - a))
            widths_q[:, i] = q * sigma_q
        return np.column_stack([widths_q[:, 0], widths_q[:, 1], sigma_q, preds_q, widths_q[:, 0] / (widths_q[:, 1] + 1e-9)])

    def _make_df(feats):
        """Label the 5 raw feature columns with ``column_prefix`` and cast to the requested output dtype."""
        cols = {}
        cols[f"{column_prefix}_width_a01"] = feats[:, 0].astype(dtype, copy=False)
        cols[f"{column_prefix}_width_a02"] = feats[:, 1].astype(dtype, copy=False)
        cols[f"{column_prefix}_sigma_hat"] = feats[:, 2].astype(dtype, copy=False)
        cols[f"{column_prefix}_pred"] = feats[:, 3].astype(dtype, copy=False)
        cols[f"{column_prefix}_width_ratio"] = feats[:, 4].astype(dtype, copy=False)
        return cols

    if X_query is not None:
        Xq = np.asarray(X_query, dtype=np.float32)
        return pl.DataFrame(_make_df(_process(X_train_f, Xq, y_train_f, seed)))
    if splitter is None:
        raise ValueError("Mode A requires splitter.")
    n_train = X_train_f.shape[0]
    out = np.zeros((n_train, n_features_out), dtype=dtype)
    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X_train_f)):
        out[val_idx] = _process(X_train_f[train_idx], X_train_f[val_idx], y_train_f[train_idx], int(seed) + fold_idx * 100).astype(dtype, copy=False)
        logger.info("conformal_locally_adaptive: fold %d done", fold_idx + 1)
    return pl.DataFrame(_make_df(out))
