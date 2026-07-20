"""Target-augmented k-means codebook: KMeans on [X, ŷ_baseline] joint space.

Iter 99 mechanism. Agent A #4 ranked.

Per train fold: fit baseline → predictions; KMeans (K=20) on [X_standardized, ŷ_baseline]; per query
emit cluster id one-hot bin + log-distance to top-3 centroids + cluster size + cluster y_mean.

5 features.
"""
from __future__ import annotations
import logging
import numpy as np
import polars as pl
from ._utils import require_seed, validate_numeric_input

logger = logging.getLogger(__name__)


def compute_target_kmeans_codebook_features(
    X_train, y_train, X_query=None, splitter=None, *, seed, task="regression",
    n_clusters=20, standardize=True, column_prefix="tkmc", dtype=np.float32,
):
    """Fit a fast LightGBM baseline per train fold, cluster [X, ŷ_baseline] jointly via MiniBatchKMeans, and emit per-query cluster id + cluster y-mean + cluster size + distance-to-nearest-centroid + log-distance-to-3rd-nearest-centroid (5 columns). See module docstring for the mechanism rationale."""
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ImportError("target_kmeans_codebook requires lightgbm") from exc
    from sklearn.cluster import MiniBatchKMeans
    from scipy.spatial.distance import cdist as _cdist

    seed = require_seed(seed)
    validate_numeric_input(X_train, name="X_train", allow_fp16=False)
    if X_query is not None:
        validate_numeric_input(X_query, name="X_query", allow_fp16=False)
    X_train_f = np.asarray(X_train, dtype=np.float32)
    y_train_f = np.asarray(y_train, dtype=np.float32).ravel()
    n_features_out = 5

    def _process(Xt, Xq, y_t, fold_seed):
        """Core per-fold pipeline: scale, fit a small LightGBM baseline to get ŷ, cluster the [X, ŷ*sqrt(d)] joint space (ŷ weighted so it isn't swamped by d feature dims), then compute each query row's cluster stats and centroid distances."""
        if standardize:
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler().fit(Xt)
            Xt_s = scaler.transform(Xt).astype(np.float32)
            Xq_s = scaler.transform(Xq).astype(np.float32)
        else:
            Xt_s, Xq_s = Xt, Xq
        # Get baseline ŷ to augment X
        if task == "binary":
            m = lgb.LGBMClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=int(fold_seed), verbose=-1, n_jobs=-1).fit(Xt_s, y_t.astype(np.int32))
            y_hat_train = np.asarray(m.predict_proba(Xt_s))[:, 1].astype(np.float32)
            y_hat_query = np.asarray(m.predict_proba(Xq_s))[:, 1].astype(np.float32)
        else:
            m = lgb.LGBMRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=int(fold_seed), verbose=-1, n_jobs=-1).fit(Xt_s, y_t)
            y_hat_train = np.asarray(m.predict(Xt_s)).astype(np.float32)
            y_hat_query = np.asarray(m.predict(Xq_s)).astype(np.float32)
        # Joint [X, ŷ] space — ŷ gets weight = sqrt(d) to balance scale
        d = Xt_s.shape[1]
        weight = np.sqrt(d)
        Xt_aug = np.column_stack([Xt_s, y_hat_train[:, None] * weight]).astype(np.float32)
        Xq_aug = np.column_stack([Xq_s, y_hat_query[:, None] * weight]).astype(np.float32)
        n_clusters_eff = min(n_clusters, Xt_aug.shape[0] // 4 if Xt_aug.shape[0] >= 80 else 4)
        km = MiniBatchKMeans(n_clusters=n_clusters_eff, random_state=int(fold_seed), n_init=4, max_iter=100, batch_size=256).fit(Xt_aug)
        train_labels = km.labels_
        cluster_y_mean = np.zeros(n_clusters_eff, dtype=np.float32)
        cluster_size = np.zeros(n_clusters_eff, dtype=np.float32)
        for c in range(n_clusters_eff):
            mask = train_labels == c
            if mask.sum() > 0:
                cluster_y_mean[c] = float(y_t[mask].mean())
                cluster_size[c] = float(mask.sum())
        query_labels = km.predict(Xq_aug)
        agg_y = cluster_y_mean[query_labels].astype(np.float32)
        agg_size = cluster_size[query_labels].astype(np.float32)
        # Distance to nearest 3 centroids
        d_all = _cdist(Xq_aug, km.cluster_centers_)  # (n_q, n_clusters)
        d_sort = np.sort(d_all, axis=1)
        nearest_d = d_sort[:, 0].astype(np.float32)
        log_d_3rd = np.log(d_sort[:, min(2, n_clusters_eff - 1)] + 1e-9).astype(np.float32)
        return np.column_stack([query_labels.astype(np.float32), agg_y, agg_size, nearest_d, log_d_3rd])

    def _make_df(feats):
        """Split the flat ``_process`` output into the 5 named output columns, cast to the requested output ``dtype``."""
        cols = {}
        cols[f"{column_prefix}_cluster_id"] = feats[:, 0].astype(dtype, copy=False)
        cols[f"{column_prefix}_cluster_y_mean"] = feats[:, 1].astype(dtype, copy=False)
        cols[f"{column_prefix}_cluster_size"] = feats[:, 2].astype(dtype, copy=False)
        cols[f"{column_prefix}_nearest_d"] = feats[:, 3].astype(dtype, copy=False)
        cols[f"{column_prefix}_log_d_3rd"] = feats[:, 4].astype(dtype, copy=False)
        return cols

    if X_query is not None:
        Xq = np.asarray(X_query, dtype=np.float32)
        feats = _process(X_train_f, Xq, y_train_f, seed)
        return pl.DataFrame(_make_df(feats))
    if splitter is None:
        raise ValueError("Mode A requires splitter.")
    n_train = X_train_f.shape[0]
    out = np.zeros((n_train, n_features_out), dtype=dtype)
    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X_train_f)):
        out[val_idx] = _process(X_train_f[train_idx], X_train_f[val_idx], y_train_f[train_idx], int(seed) + fold_idx * 100).astype(dtype, copy=False)
        logger.info("target_kmeans_codebook: fold %d done", fold_idx + 1)
    return pl.DataFrame(_make_df(out))
