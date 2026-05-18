"""Conformal coverage-failure indicator on adjacent folds.

Iter 91 mechanism. Agent B #5 ranked.

Per query: find K=20 nearest OOF rows in train; for each neighbor, check if its true y fell in its
predicted α=0.2 conformal interval (binary). Emit fraction-covered + mean signed miscoverage direction.

5 features.
"""
from __future__ import annotations
import logging
from typing import Any, Literal, Optional
import numpy as np
import polars as pl
from ._utils import require_seed, validate_numeric_input

logger = logging.getLogger(__name__)


def compute_conformal_coverage_failure_features(
    X_train, y_train, X_query=None, splitter=None, *, seed, task="regression",
    k_neighbors=20, alpha=0.2, standardize=True, column_prefix="ccf", dtype=np.float32,
):
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ImportError("conformal_coverage_failure requires lightgbm") from exc
    from sklearn.neighbors import NearestNeighbors

    seed = require_seed(seed)
    validate_numeric_input(X_train, name="X_train", allow_fp16=False)
    if X_query is not None:
        validate_numeric_input(X_query, name="X_query", allow_fp16=False)
    X_train_f = np.asarray(X_train, dtype=np.float32)
    y_train_f = np.asarray(y_train, dtype=np.float32).ravel()
    n_features_out = 5

    def _process(Xt, Xq, y_t, fold_seed):
        if standardize:
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler().fit(Xt)
            Xt_s = scaler.transform(Xt).astype(np.float32)
            Xq_s = scaler.transform(Xq).astype(np.float32)
        else:
            Xt_s, Xq_s = Xt, Xq
        # Split train into 2 halves: fit on half1, get residuals on half2, compute α-quantile
        n = Xt_s.shape[0]
        idx = np.arange(n)
        rng = np.random.default_rng(int(fold_seed))
        rng.shuffle(idx)
        h1, h2 = idx[: n // 2], idx[n // 2 :]
        if task == "binary":
            m = lgb.LGBMClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=int(fold_seed), verbose=-1, n_jobs=-1).fit(Xt_s[h1], y_t[h1].astype(np.int32))
            preds_h2 = m.predict_proba(Xt_s[h2])[:, 1].astype(np.float32)
            scores_h2 = np.abs(y_t[h2] - preds_h2)
            preds_train_all = np.empty(n, dtype=np.float32)
            preds_train_all[h2] = preds_h2
            preds_train_all[h1] = m.predict_proba(Xt_s[h1])[:, 1].astype(np.float32)  # for symmetry, in-sample
            preds_query = m.predict_proba(Xq_s)[:, 1].astype(np.float32)
        else:
            m = lgb.LGBMRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=int(fold_seed), verbose=-1, n_jobs=-1).fit(Xt_s[h1], y_t[h1])
            preds_h2 = m.predict(Xt_s[h2]).astype(np.float32)
            scores_h2 = np.abs(y_t[h2] - preds_h2)
            preds_train_all = np.empty(n, dtype=np.float32)
            preds_train_all[h2] = preds_h2
            preds_train_all[h1] = m.predict(Xt_s[h1]).astype(np.float32)
            preds_query = m.predict(Xq_s).astype(np.float32)
        q = float(np.quantile(scores_h2, 1.0 - alpha))
        # Coverage status for each train row: did its true y fall within ±q of its prediction?
        train_covered = (np.abs(y_t - preds_train_all) <= q).astype(np.float32)
        train_signed_miscov = np.where(np.abs(y_t - preds_train_all) > q, np.sign(y_t - preds_train_all), 0.0).astype(np.float32)
        k_eff = min(k_neighbors, n)
        nn = NearestNeighbors(n_neighbors=k_eff, n_jobs=-1).fit(Xt_s)
        _, q_idx = nn.kneighbors(Xq_s)
        nbr_covered = train_covered[q_idx]
        nbr_miscov = train_signed_miscov[q_idx]
        frac_covered = nbr_covered.mean(axis=1).astype(np.float32)
        mean_signed_miscov = nbr_miscov.mean(axis=1).astype(np.float32)
        std_covered = nbr_covered.std(axis=1).astype(np.float32) + 1e-9
        n_uncov = (nbr_covered == 0).sum(axis=1).astype(np.float32)
        return np.column_stack([frac_covered, mean_signed_miscov, std_covered, n_uncov, preds_query])

    def _make_df(feats):
        cols = {}
        cols[f"{column_prefix}_frac_covered"] = feats[:, 0].astype(dtype, copy=False)
        cols[f"{column_prefix}_mean_signed_miscov"] = feats[:, 1].astype(dtype, copy=False)
        cols[f"{column_prefix}_std_covered"] = feats[:, 2].astype(dtype, copy=False)
        cols[f"{column_prefix}_n_uncovered"] = feats[:, 3].astype(dtype, copy=False)
        cols[f"{column_prefix}_baseline_pred"] = feats[:, 4].astype(dtype, copy=False)
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
        logger.info("conformal_coverage_failure: fold %d done", fold_idx + 1)
    return pl.DataFrame(_make_df(out))
