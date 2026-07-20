"""Trust score via correctness-density (Jiang et al. style).

Iter 90 mechanism.

Per row: compute log-ratio d_kNN(x, correct-rows-class-c) / d_kNN(x, correct-rows-other-classes).
For regression: "correct" = baseline residual within ±MAD; for binary: "correct" = baseline-pred-class matches y.

CORRECTNESS-MASK CAVEAT (audit4): the baseline model predicts on its OWN training rows to define the
"correct" set (``m.predict(Xt)`` on the fit fold), so the per-fold correctness mask is IN-SAMPLE, not
out-of-fold, despite the module name. This is TRAIN/SERVING SKEW (train rows get an optimistic-correctness
signal), NOT leakage into held-out metrics -- held-out / query rows are scored against the fitted state via
the standard Mode-B path, so honest holdout numbers DEGRADE (never inflate) from the skew. Opt-in only.

Per query emit 5 features:
- nearest_correct_distance (min distance to any OOF-correct row of dominant class)
- log_trust_ratio (binary: log d(other) - log d(same); regression: log d(high-resid) - log d(low-resid))
- raw_d_correct (any class)
- raw_d_incorrect
- frac_correct_in_topK
"""
from __future__ import annotations

import logging
from typing import Any, Literal, Optional

import numpy as np
import polars as pl

from ._utils import require_seed, validate_numeric_input

logger = logging.getLogger(__name__)


def compute_trust_score_oof_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: Optional[np.ndarray],
    splitter: Optional[Any] = None,
    *,
    seed: int,
    task: Literal["binary", "regression"] = "regression",
    k_neighbors: int = 16,
    standardize: bool = True,
    column_prefix: str = "trust",
    dtype: type = np.float32,
) -> pl.DataFrame:
    """Trust-score features. 5 outputs per row."""
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ImportError("trust_score_oof requires lightgbm") from exc
    from sklearn.neighbors import NearestNeighbors

    seed = require_seed(seed)
    validate_numeric_input(X_train, name="X_train", allow_fp16=False)
    if X_query is not None:
        validate_numeric_input(X_query, name="X_query", allow_fp16=False)

    X_train_f = np.asarray(X_train, dtype=np.float32)
    y_train_f = np.asarray(y_train, dtype=np.float32).ravel()
    n_features_out = 5

    def _process(Xt, Xq, y_t, fold_seed):
        """Fit a small in-sample baseline, split train rows into correct/incorrect (and positive-correct/negative-correct) subsets, then compute per-query kNN-distance and neighborhood-correctness-fraction trust features against those subsets."""
        if standardize:
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler().fit(Xt)
            Xt_s = scaler.transform(Xt).astype(np.float32)
            Xq_s = scaler.transform(Xq).astype(np.float32)
        else:
            Xt_s, Xq_s = Xt, Xq
        # Fit baseline + define correctness mask
        if task == "binary":
            m = lgb.LGBMClassifier(n_estimators=50, max_depth=3, learning_rate=0.1,
                                   random_state=int(fold_seed), verbose=-1, n_jobs=-1).fit(Xt_s, y_t.astype(np.int32))
            preds = np.asarray(m.predict_proba(Xt_s))[:, 1]
            pred_class = (preds > 0.5).astype(np.float32)
            correct_mask = pred_class == y_t
            pos_correct = correct_mask & (y_t > 0.5)
            neg_correct = correct_mask & (y_t <= 0.5)
        else:
            m = lgb.LGBMRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=int(fold_seed), verbose=-1, n_jobs=-1).fit(Xt_s, y_t)
            preds = np.asarray(m.predict(Xt_s)).astype(np.float32)
            abs_resid = np.abs(y_t - preds)
            mad = float(np.median(abs_resid)) + 1e-6
            correct_mask = abs_resid < mad
            # For regression "correct" subsets by y direction
            pos_correct = correct_mask & (y_t > float(np.median(y_t)))
            neg_correct = correct_mask & (y_t <= float(np.median(y_t)))

        # kNN distances to "correct" subsets
        def _nn_dist(target_idx_mask, X_query):
            """Mean and nearest-neighbor kNN distance from each row of ``X_query`` to the train subset selected by ``target_idx_mask``; returns sentinel fill values (mean=1.0, nearest=0.0) when the subset is empty."""
            idx = np.where(target_idx_mask)[0]
            if idx.size == 0:
                return np.full(X_query.shape[0], 1.0, dtype=np.float32), np.full(X_query.shape[0], 0.0, dtype=np.float32)
            k_eff = min(k_neighbors, idx.size)
            nn = NearestNeighbors(n_neighbors=k_eff, n_jobs=-1).fit(Xt_s[idx])
            d, _ = nn.kneighbors(X_query)
            return d.mean(axis=1).astype(np.float32), d[:, 0].astype(np.float32)

        d_pos_mean, d_pos_min = _nn_dist(pos_correct, Xq_s)
        d_neg_mean, _ = _nn_dist(neg_correct, Xq_s)
        d_corr_mean, _ = _nn_dist(correct_mask, Xq_s)
        d_incorr_mean, _ = _nn_dist(~correct_mask, Xq_s)
        log_trust_ratio = np.log((d_neg_mean + 1e-6) / (d_pos_mean + 1e-6)).astype(np.float32)
        # frac correct among top-k overall
        nn = NearestNeighbors(n_neighbors=min(k_neighbors, Xt_s.shape[0]), n_jobs=-1).fit(Xt_s)
        _, all_idx = nn.kneighbors(Xq_s)
        frac_correct = correct_mask[all_idx].mean(axis=1).astype(np.float32)
        return np.column_stack([d_pos_min, log_trust_ratio, d_corr_mean, d_incorr_mean, frac_correct])

    def _make_df(feats):
        """Name the 5 flat ``feats`` columns (nearest-positive dist, log trust ratio, correct/incorrect mean dist, top-k correctness fraction) as ``{prefix}_<name>`` and cast to the output ``dtype``."""
        cols = {}
        cols[f"{column_prefix}_nearest_pos_dist"] = feats[:, 0].astype(dtype, copy=False)
        cols[f"{column_prefix}_log_trust_ratio"] = feats[:, 1].astype(dtype, copy=False)
        cols[f"{column_prefix}_d_correct_mean"] = feats[:, 2].astype(dtype, copy=False)
        cols[f"{column_prefix}_d_incorrect_mean"] = feats[:, 3].astype(dtype, copy=False)
        cols[f"{column_prefix}_frac_correct_topk"] = feats[:, 4].astype(dtype, copy=False)
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
        logger.info("trust_score_oof: fold %d/%d done", fold_idx + 1, len(splits))

    return pl.DataFrame(_make_df(out))
