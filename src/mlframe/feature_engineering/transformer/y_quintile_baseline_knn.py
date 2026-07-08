"""iter106 mechanism. Y-quintile-conditioned baseline-prediction-at-kNN features.

Structurally different from:
- iter69/102 (baseline_disagreement): baseline prediction AT the query x.
- iter103 (residual_stratified_distance): distances to easy/hard sets (|residual| split).
- cdist (class_distance): distances to top/bottom y-quintile (no baseline).

This mechanism asks a different question per query x:
"What does the baseline predict for the nearest training rows IN EACH y-stratum?"

Hypothesis: For a query x, if the baseline's prediction-at-x is high but baseline's
predictions at nearest LOW-y training neighbours are also high (i.e. neighbours are
predicted-like-high-y by baseline despite being low-y in truth), there's a structured
confusion the boosting can exploit. Pure distance features (cdist) only show neighbour
LOCATIONS; pure baseline-prediction features (iter69) only show prediction-at-x;
this combines them: where does baseline predict things at the nearest neighbours of
each y-stratum?

Mechanism (per fold for Mode A, single-pass for Mode B):
1. Fit 1 LGB baseline (depth=5, 100 estimators) on training rows.
2. Stratify training rows by y-quintile q in {0, 1, 2, 3, 4}, where stratum_q is rows
   with y in [q*0.2, (q+1)*0.2] target-quantile.
3. For each query x, find k=8 nearest training rows in each stratum.
4. For each stratum, compute baseline's prediction MEAN and STD across those 8 neighbours.
5. Output 10 features per query: 5 strata x (mean, std).

Leakage discipline:
- Mode A (X_query=None): per fold, baseline fit on train_idx; y-strata derived from
  y_train[train_idx]; query rows (val_idx) score against fold's stratified neighbour banks.
- Mode B (X_query given): baseline fit on full X_train; y-strata from full y_train; queries
  scored.

Cost: 1 LGB fit + 5 sklearn NearestNeighbors fits per fold. ~3-5x cheaper than iter69
(which fits 3 baselines and 2 kNN sets).
"""
from __future__ import annotations

import logging
from typing import Any, Literal, Optional

import numpy as np
import polars as pl

from ._utils import require_seed, validate_numeric_input

logger = logging.getLogger(__name__)

_N_STRATA = 5
_K_NEIGHBOURS = 8


def _fit_baseline_predict(Xt: np.ndarray, y_t: np.ndarray, Xall: np.ndarray, task: str, seed: int) -> np.ndarray:
    """Fit LGB baseline on (Xt, y_t), return predictions on Xall. Shape (n_all,)."""
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ImportError("y_quintile_baseline_knn requires lightgbm") from exc

    if task == "binary":
        m = lgb.LGBMClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=int(seed), verbose=-1, n_jobs=-1)
        m.fit(Xt, y_t.astype(np.int32))
        return np.asarray(m.predict_proba(Xall)[:, 1].astype(np.float32))
    else:
        m = lgb.LGBMRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=int(seed), verbose=-1, n_jobs=-1)
        m.fit(Xt, y_t)
        return np.asarray(m.predict(Xall).astype(np.float32))


def _knn_pred_stats(X_stratum: np.ndarray, pred_stratum: np.ndarray, X_query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """For each query row, find k nearest neighbours in X_stratum, return (mean, std) of
    pred_stratum across those neighbours. Shape (n_q,) each."""
    from sklearn.neighbors import NearestNeighbors

    n_sub = X_stratum.shape[0]
    n_q = X_query.shape[0]
    if n_sub == 0:
        return np.zeros(n_q, dtype=np.float32), np.zeros(n_q, dtype=np.float32)
    k_request = min(k, n_sub)
    nn = NearestNeighbors(n_neighbors=k_request, algorithm="auto", n_jobs=-1).fit(X_stratum)
    _dists, ids = nn.kneighbors(X_query)
    pred_at_nn = pred_stratum[ids]
    return pred_at_nn.mean(axis=1).astype(np.float32), pred_at_nn.std(axis=1).astype(np.float32)


def compute_y_quintile_baseline_knn_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: Optional[np.ndarray],
    splitter: Optional[Any] = None,
    *,
    seed: int,
    task: Literal["binary", "regression"] = "regression",
    standardize: bool = True,
    column_prefix: str = "yqbk",
    dtype: type = np.float32,
) -> pl.DataFrame:
    """Y-quintile-conditioned baseline-prediction-at-kNN features (iter106).

    Output: 10 features per row (5 y-strata x mean + std of baseline pred at k=8 NN).
    """
    seed = require_seed(seed)
    validate_numeric_input(X_train, name="X_train", allow_fp16=False)
    if X_query is not None:
        validate_numeric_input(X_query, name="X_query", allow_fp16=False)

    X_train_f = np.asarray(X_train, dtype=np.float32)
    y_train_f = np.asarray(y_train, dtype=np.float32).ravel()

    def _process(Xt: np.ndarray, Xq: np.ndarray, y_t: np.ndarray, fold_seed: int) -> np.ndarray:
        """Stratify the train fold into ``_N_STRATA`` y-bands (quantile bands for regression, baseline-probability bands for binary), then for each stratum compute the mean/std of the in-band baseline predictions among each query row's k nearest in-band neighbours."""
        if standardize:
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler().fit(Xt)
            Xt_s = scaler.transform(Xt).astype(np.float32)
            Xq_s = scaler.transform(Xq).astype(np.float32)
        else:
            Xt_s = Xt
            Xq_s = Xq

        pred_train = _fit_baseline_predict(Xt_s, y_t, Xt_s, task=task, seed=fold_seed)

        if task == "binary":
            strata_edges = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]) - 1e-9
            y_for_strata = pred_train  # use baseline OOF prob for binary (target is 0/1)
        else:
            strata_edges = np.quantile(y_t, np.linspace(0.0, 1.0, _N_STRATA + 1))
            y_for_strata = y_t

        feats = np.zeros((Xq.shape[0], _N_STRATA * 2), dtype=np.float32)
        for q in range(_N_STRATA):
            lo = strata_edges[q]
            hi = strata_edges[q + 1]
            if q == _N_STRATA - 1:
                mask = (y_for_strata >= lo) & (y_for_strata <= hi)
            else:
                mask = (y_for_strata >= lo) & (y_for_strata < hi)
            X_q = Xt_s[mask]
            pred_q = pred_train[mask]
            m, s = _knn_pred_stats(X_q, pred_q, Xq_s, k=_K_NEIGHBOURS)
            feats[:, q * 2 + 0] = m
            feats[:, q * 2 + 1] = s
        return feats

    def _make_df(feats: np.ndarray) -> dict[str, np.ndarray]:
        """Label the per-stratum mean/std columns with ``column_prefix`` and cast to the requested output dtype."""
        cols: dict[str, np.ndarray] = {}
        for q in range(_N_STRATA):
            cols[f"{column_prefix}_q{q}_mean"] = feats[:, q * 2 + 0].astype(dtype, copy=False)
            cols[f"{column_prefix}_q{q}_std"] = feats[:, q * 2 + 1].astype(dtype, copy=False)
        return cols

    n_features = _N_STRATA * 2

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
        logger.info("y_quintile_baseline_knn: fold %d/%d done", fold_idx + 1, len(splits))

    return pl.DataFrame(_make_df(out))
