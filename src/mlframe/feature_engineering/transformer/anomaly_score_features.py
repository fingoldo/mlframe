"""iter115 mechanism. IsolationForest anomaly-score features for rare-positive binary.

Pivot away from iter69-mechanism family (which fully fails on rare-positive binary, mammography
1.3% pos: 4 orthogonal interventions iter111-114 all negative on CB AUC).

Anomaly-detection signal source: rare-positive rows are by definition statistically unusual.
IsolationForest assigns an anomaly score (high = more anomalous) per row using random tree
partitioning. The score is computed PURELY ON X (no labels), so it generalises across regimes
and doesn't suffer from "tiny positive class per fold" problems that broke iter69 baselines.

Mechanism:
1. Per fold, fit two IsolationForest models on training X (both with different seeds for
   ensemble disagreement; n_estimators=100, contamination='auto').
2. Per query row, output:
   - anomaly_score_1, anomaly_score_2 (raw scores from each forest)
   - mean and std (disagreement) of the two scores
   - mean(score) - global_mean: relative-anomaly z-score in feature space

5 features per row. Pure-X mechanism — no labels touched.

Task-agnostic by design: same features regardless of binary / regression. Useful as additive
component to iter69 on regression too (anomaly score might capture outlier-y rows boostings
miss); useful AS a primary signal on rare-positive binary (where iter69 fails).

Leakage discipline:
- Mode A (X_query=None): per-fold fit on train_idx; predict on val_idx.
- Mode B (X_query given): fit on full X_train; predict on X_query.

Cost: 2 IsolationForest fits per fold; trivially fast (~0.5s on 100k×100 X).
"""
from __future__ import annotations

import logging
from typing import Any, Literal, Optional

import numpy as np
import polars as pl

from ._utils import require_seed, validate_numeric_input

logger = logging.getLogger(__name__)


def _fit_anomaly_predict(Xt: np.ndarray, Xq: np.ndarray, seed: int) -> np.ndarray:
    """Fit 2 IsolationForest models on Xt with different seeds, return 5 features per Xq row."""
    from sklearn.ensemble import IsolationForest

    iso1 = IsolationForest(n_estimators=100, contamination="auto", random_state=int(seed), n_jobs=-1, max_samples=min(256, Xt.shape[0]))
    iso1.fit(Xt)
    s1 = -iso1.score_samples(Xq).astype(np.float32)  # negate so higher = more anomalous

    iso2 = IsolationForest(n_estimators=100, contamination="auto", random_state=int(seed) + 41, n_jobs=-1, max_samples=min(256, Xt.shape[0]))
    iso2.fit(Xt)
    s2 = -iso2.score_samples(Xq).astype(np.float32)

    mean = ((s1 + s2) / 2.0).astype(np.float32)
    std = (np.abs(s1 - s2) / 2.0).astype(np.float32)  # |diff|/2 = std of 2 values
    global_mean_train = float(((-iso1.score_samples(Xt) + -iso2.score_samples(Xt)) / 2.0).mean())
    rel_z = (mean - global_mean_train).astype(np.float32)

    return np.column_stack([s1, s2, mean, std, rel_z])


def compute_anomaly_score_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: Optional[np.ndarray],
    splitter: Optional[Any] = None,
    *,
    seed: int,
    task: Literal["binary", "regression"] = "regression",
    standardize: bool = True,
    column_prefix: str = "anom",
    dtype: np.dtype = np.float32,
) -> pl.DataFrame:
    """Pure-X anomaly-score features via IsolationForest ensemble (iter115).

    Output: 5 features per row. Task-agnostic.
    """
    seed = require_seed(seed)
    validate_numeric_input(X_train, name="X_train", allow_fp16=False)
    if X_query is not None:
        validate_numeric_input(X_query, name="X_query", allow_fp16=False)

    X_train_f = np.asarray(X_train, dtype=np.float32)
    y_train_f = np.asarray(y_train, dtype=np.float32).ravel()  # not actually used; kept for API parity

    def _process(Xt: np.ndarray, Xq: np.ndarray, y_t: np.ndarray, fold_seed: int) -> np.ndarray:
        if standardize:
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler().fit(Xt)
            Xt_s = scaler.transform(Xt).astype(np.float32)
            Xq_s = scaler.transform(Xq).astype(np.float32)
        else:
            Xt_s = Xt
            Xq_s = Xq
        return _fit_anomaly_predict(Xt_s, Xq_s, seed=fold_seed)

    def _make_df(feats: np.ndarray) -> dict[str, np.ndarray]:
        cols: dict[str, np.ndarray] = {}
        cols[f"{column_prefix}_iso1"] = feats[:, 0].astype(dtype, copy=False)
        cols[f"{column_prefix}_iso2"] = feats[:, 1].astype(dtype, copy=False)
        cols[f"{column_prefix}_mean"] = feats[:, 2].astype(dtype, copy=False)
        cols[f"{column_prefix}_std"] = feats[:, 3].astype(dtype, copy=False)
        cols[f"{column_prefix}_rel_z"] = feats[:, 4].astype(dtype, copy=False)
        return cols

    n_features = 5

    if X_query is not None:
        Xq = np.asarray(X_query, dtype=np.float32)
        feats = _process(X_train_f, Xq, y_train_f, seed)
        return pl.DataFrame(_make_df(feats))

    if splitter is None:
        raise ValueError("Mode A (X_query=None) requires a splitter.")
    n_train = X_train_f.shape[0]
    out = np.zeros((n_train, n_features), dtype=dtype)
    splits = list(splitter.split(X_train_f))
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        feats = _process(X_train_f[train_idx], X_train_f[val_idx], y_train_f[train_idx], int(seed) + fold_idx * 100)
        out[val_idx] = feats.astype(dtype, copy=False)
        logger.info("anomaly_score: fold %d/%d done", fold_idx + 1, len(splits))

    return pl.DataFrame(_make_df(out))
