"""iter114 mechanism. SMOTE-augmented baseline disagreement for rare-positive binary.

Direct extension of [baseline_disagreement_balanced.py](baseline_disagreement_balanced.py) (iter113).
Class-balanced loss weighting (iter113) only marginally improved rare-positive results (mammography
CB AUC -0.70 -> -0.55 vs iter69-direct -1.05); class imbalance at the BASELINE-FITTING LEVEL needs
addressing more directly. SMOTE generates synthetic minority instances in feature space, increasing
positive count before fitting the LGB/Ridge baselines.

Mechanism:
1. For each fold, oversample the minority class via k-NN-based SMOTE to match majority size (or
   to a configurable target ratio). 5-NN interpolation in feature space.
2. Fit 3 baselines (LGB d=3, LGB d=5, Logistic) on the SMOTE-augmented (X', y') training set.
3. Predict on the ORIGINAL X_query (not augmented). Augmentation is a baseline-fitting trick only.
4. Output same 8-feature disagreement signature as iter69 / iter113.

Regression path: SMOTE doesn't apply; falls back to iter69's vanilla baselines (and downstream
identical to iter69).

Cost: SMOTE on (k-NN search over minority) is O(n_pos * d * log n_pos). For mammography 11k with
145 positives, trivially fast.
"""
from __future__ import annotations

import logging
from typing import Any, Literal, Optional

import numpy as np
import polars as pl

from ._utils import require_seed, validate_numeric_input

logger = logging.getLogger(__name__)


def _smote_oversample_minority(X: np.ndarray, y: np.ndarray, k_neighbors: int = 5, target_ratio: float = 1.0, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Inline SMOTE: oversample y=1 class to target_ratio of majority class via k-NN interpolation.

    target_ratio=1.0 means equal classes. Returns (X_aug, y_aug) with original + synthetic positives.
    """
    from sklearn.neighbors import NearestNeighbors

    rng = np.random.default_rng(int(seed))
    pos_mask = y == 1
    X_pos = X[pos_mask]
    n_pos = X_pos.shape[0]
    n_neg = (~pos_mask).sum()
    n_synth = max(0, int(n_neg * target_ratio) - n_pos)
    if n_pos < 2 or n_synth <= 0:
        return X, y

    k_eff = min(k_neighbors, n_pos - 1)
    nn = NearestNeighbors(n_neighbors=k_eff + 1, algorithm="auto", n_jobs=-1).fit(X_pos)
    _d, ids = nn.kneighbors(X_pos)
    ids = ids[:, 1:]  # drop self

    parent_idx = rng.integers(0, n_pos, size=n_synth)
    neighbour_pick = rng.integers(0, k_eff, size=n_synth)
    neighbour_idx = ids[parent_idx, neighbour_pick]
    alpha = rng.random(size=(n_synth, 1)).astype(np.float32)
    synth = X_pos[parent_idx] + alpha * (X_pos[neighbour_idx] - X_pos[parent_idx])
    X_aug = np.concatenate([X, synth.astype(np.float32)], axis=0)
    y_aug = np.concatenate([y, np.ones(n_synth, dtype=y.dtype)])
    return X_aug, y_aug


def _fit_3baselines_smote_predict_on_query(
    Xt: np.ndarray, y_t: np.ndarray, Xq: np.ndarray, task: str, seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit 3 baselines with SMOTE-augmented binary training data, return (p1, p2, p3)."""
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ImportError("baseline_disagreement_smote requires lightgbm") from exc
    from sklearn.linear_model import Ridge, LogisticRegression

    if task == "binary":
        y_t_int = y_t.astype(np.int32)
        if y_t_int.sum() >= 2 and y_t_int.sum() < y_t_int.shape[0] * 0.4:
            X_aug, y_aug = _smote_oversample_minority(Xt, y_t_int, k_neighbors=5, target_ratio=1.0, seed=seed)
        else:
            X_aug, y_aug = Xt, y_t_int

        m1 = lgb.LGBMClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=int(seed), verbose=-1, n_jobs=-1)
        m1.fit(X_aug, y_aug)
        p1 = np.asarray(m1.predict_proba(Xq))[:, 1].astype(np.float32)
        m2 = lgb.LGBMClassifier(n_estimators=50, max_depth=5, learning_rate=0.1, random_state=int(seed) + 1, verbose=-1, n_jobs=-1)
        m2.fit(X_aug, y_aug)
        p2 = np.asarray(m2.predict_proba(Xq))[:, 1].astype(np.float32)
        try:
            m3 = LogisticRegression(max_iter=200, solver="liblinear", random_state=int(seed) + 2)
            m3.fit(X_aug, y_aug)
            p3 = m3.predict_proba(Xq)[:, 1].astype(np.float32)
        except Exception:
            p3 = np.full(Xq.shape[0], float(y_t.mean()), dtype=np.float32)
    else:
        m1 = lgb.LGBMRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=int(seed), verbose=-1, n_jobs=-1)
        m1.fit(Xt, y_t)
        p1 = np.asarray(m1.predict(Xq)).astype(np.float32)
        m2 = lgb.LGBMRegressor(n_estimators=50, max_depth=5, learning_rate=0.1, random_state=int(seed) + 1, verbose=-1, n_jobs=-1)
        m2.fit(Xt, y_t)
        p2 = np.asarray(m2.predict(Xq)).astype(np.float32)
        m3 = Ridge(alpha=1.0, random_state=int(seed) + 2)
        m3.fit(Xt, y_t)
        p3 = m3.predict(Xq).astype(np.float32)
    return p1, p2, p3


def compute_baseline_disagreement_smote_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: Optional[np.ndarray],
    splitter: Optional[Any] = None,
    *,
    seed: int,
    task: Literal["binary", "regression"] = "regression",
    standardize: bool = True,
    column_prefix: str = "blagreementsmote",
    dtype: type = np.float32,
) -> pl.DataFrame:
    """SMOTE-augmented baseline disagreement (iter114). 8 features per row (same as iter69)."""
    seed = require_seed(seed)
    validate_numeric_input(X_train, name="X_train", allow_fp16=False)
    if X_query is not None:
        validate_numeric_input(X_query, name="X_query", allow_fp16=False)

    X_train_f = np.asarray(X_train, dtype=np.float32)
    y_train_f = np.asarray(y_train, dtype=np.float32).ravel()

    def _process(Xt: np.ndarray, Xq: np.ndarray, y_t: np.ndarray, fold_seed: int) -> np.ndarray:
        """Fit the three SMOTE-augmented baselines on ``Xt``/``y_t`` and derive the 8 predictions/disagreement statistics for ``Xq``."""
        if standardize:
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler().fit(Xt)
            Xt_s = scaler.transform(Xt).astype(np.float32)
            Xq_s = scaler.transform(Xq).astype(np.float32)
        else:
            Xt_s = Xt
            Xq_s = Xq
        p1, p2, p3 = _fit_3baselines_smote_predict_on_query(Xt_s, y_t, Xq_s, task=task, seed=fold_seed)
        stack = np.stack([p1, p2, p3], axis=1)
        mean = stack.mean(axis=1)
        std = stack.std(axis=1)
        rng = stack.max(axis=1) - stack.min(axis=1)
        lgb_diff = p1 - p2
        lgb_vs_linear = ((p1 + p2) / 2.0) - p3
        return np.column_stack([p1, p2, p3, mean, std, rng, lgb_diff, lgb_vs_linear])

    def _make_df(feats: np.ndarray) -> dict[str, np.ndarray]:
        """Map the fixed 8-column layout of ``feats`` (3 baseline predictions + 5 disagreement stats) to their ``{column_prefix}_*`` output names."""
        cols: dict[str, np.ndarray] = {}
        cols[f"{column_prefix}_p_lgbd3"] = feats[:, 0].astype(dtype, copy=False)
        cols[f"{column_prefix}_p_lgbd5"] = feats[:, 1].astype(dtype, copy=False)
        cols[f"{column_prefix}_p_linear"] = feats[:, 2].astype(dtype, copy=False)
        cols[f"{column_prefix}_mean"] = feats[:, 3].astype(dtype, copy=False)
        cols[f"{column_prefix}_std"] = feats[:, 4].astype(dtype, copy=False)
        cols[f"{column_prefix}_range"] = feats[:, 5].astype(dtype, copy=False)
        cols[f"{column_prefix}_depth_diff"] = feats[:, 6].astype(dtype, copy=False)
        cols[f"{column_prefix}_lgb_vs_linear"] = feats[:, 7].astype(dtype, copy=False)
        return cols

    n_features = 8

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
        logger.info("baseline_disagreement_smote: fold %d/%d done", fold_idx + 1, len(splits))

    return pl.DataFrame(_make_df(out))
