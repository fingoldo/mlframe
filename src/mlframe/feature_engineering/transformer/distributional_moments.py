"""Distributional moments: 9 quantile LGBs → skew + kurtosis + tail-mass per row.

Iter 94 mechanism. Agent B #3 ranked. 3rd/4th moments are mathematically orthogonal to std (2nd).
"""
from __future__ import annotations
import logging
from typing import Sequence
import numpy as np
import polars as pl
from ._utils import require_seed, validate_numeric_input

logger = logging.getLogger(__name__)
_QUANTILES = (0.05, 0.15, 0.25, 0.5, 0.75, 0.85, 0.95)


def compute_distributional_moments_features(
    X_train, y_train, X_query=None, splitter=None, *, seed, task="regression",
    quantiles: Sequence[float] = _QUANTILES, standardize=True, column_prefix="distmom", dtype=np.float32,
):
    """Fit one small LightGBM per quantile (7 quantile regressors, or 7 reweighted classifiers for binary), then derive skew/kurtosis/tail-asymmetry/tail-mass/median features from the predicted quantile spread per query row."""
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ImportError("distributional_moments requires lightgbm") from exc
    seed = require_seed(seed)
    validate_numeric_input(X_train, name="X_train", allow_fp16=False)
    if X_query is not None:
        validate_numeric_input(X_query, name="X_query", allow_fp16=False)
    X_train_f = np.asarray(X_train, dtype=np.float32)
    y_train_f = np.asarray(y_train, dtype=np.float32).ravel()
    n_features_out = 5

    def _process(Xt, Xq, y_t, fold_seed):
        """Per-fold feature block: fit the per-quantile models on the train fold, predict on the query rows, sort each row's predictions to resolve quantile crossing, then compute the skew/kurtosis/tail proxies from the resulting quantile spread."""
        if standardize:
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler().fit(Xt)
            Xt_s = scaler.transform(Xt).astype(np.float32)
            Xq_s = scaler.transform(Xq).astype(np.float32)
        else:
            Xt_s, Xq_s = Xt, Xq
        preds_q = np.zeros((Xq_s.shape[0], len(quantiles)), dtype=np.float32)
        if task == "binary":
            # For binary: predict baseline at K sample-weight reweightings — gamma in {0, 0.5, 1, 1.5, 2, 3, 5}
            gammas = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
            for i, g in enumerate(gammas[: len(quantiles)]):
                if g > 0:
                    p_mean = float(y_t.mean())
                    sw = np.where(y_t > 0.5, (1 - p_mean) ** g, p_mean**g).astype(np.float32)
                else:
                    sw = None
                m = lgb.LGBMClassifier(n_estimators=30, max_depth=3, learning_rate=0.1, random_state=int(fold_seed) + i, verbose=-1, n_jobs=-1)
                m.fit(Xt_s, y_t.astype(np.int32), sample_weight=sw)
                preds_q[:, i] = m.predict_proba(Xq_s)[:, 1].astype(np.float32)
        else:
            for i, a in enumerate(quantiles):
                m = lgb.LGBMRegressor(n_estimators=30, max_depth=3, learning_rate=0.1, objective="quantile", alpha=a, random_state=int(fold_seed) + i, verbose=-1, n_jobs=-1)
                m.fit(Xt_s, y_t)
                preds_q[:, i] = m.predict(Xq_s).astype(np.float32)
        # Sort each row's predictions (in case quantile crossing)
        preds_q.sort(axis=1)
        q05 = preds_q[:, 0]; q25 = preds_q[:, 2]
        q50 = preds_q[:, 3]; q75 = preds_q[:, 4]; q85 = preds_q[:, 5]; q95 = preds_q[:, 6]
        iqr = (q75 - q25).astype(np.float32) + 1e-9
        # Skewness (q95 + q05 - 2*q50) / IQR
        skew_proxy = ((q95 + q05 - 2 * q50) / iqr).astype(np.float32)
        # Kurtosis proxy (q95 - q05) / (q75 - q25)
        kurt_proxy = ((q95 - q05) / iqr).astype(np.float32)
        # Upper tail asymmetry
        upper_asym = ((q95 - q50) / (q50 - q05 + 1e-9)).astype(np.float32)
        # Tail mass (q95 - q85)
        tail_mass = (q95 - q85).astype(np.float32)
        return np.column_stack([skew_proxy, kurt_proxy, upper_asym, tail_mass, q50])

    def _make_df(feats):
        """Reshape the flat ``_process`` output block into named ``{prefix}_skew`` / ``_kurt`` / ``_upper_asym`` / ``_tail_mass`` / ``_q50`` columns."""
        cols = {}
        cols[f"{column_prefix}_skew"] = feats[:, 0].astype(dtype, copy=False)
        cols[f"{column_prefix}_kurt"] = feats[:, 1].astype(dtype, copy=False)
        cols[f"{column_prefix}_upper_asym"] = feats[:, 2].astype(dtype, copy=False)
        cols[f"{column_prefix}_tail_mass"] = feats[:, 3].astype(dtype, copy=False)
        cols[f"{column_prefix}_q50"] = feats[:, 4].astype(dtype, copy=False)
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
        logger.info("distributional_moments: fold %d done", fold_idx + 1)
    return pl.DataFrame(_make_df(out))
