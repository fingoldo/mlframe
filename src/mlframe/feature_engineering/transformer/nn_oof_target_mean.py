"""NN target-mean in OOF embedding space: Home Credit 1st-place winning idea.

Iter 71 mechanism. From Kaggle Home Credit Default Risk 1st-place solution, where
`neighbors_target_mean_500` was the SINGLE highest-importance feature.

Mechanism:
1. Fit 3 baselines on train fold: LGB depth=3, LGB depth=5, Ridge/LogReg.
2. Build a 3D embedding per row: (p_lgbd3, p_lgbd5, p_linear). The baselines act as a learned
   nonlinear projection from raw X to a 3D y-aware space; rows close in this embedding have
   similar predicted y under multiple model classes → genuinely similar targets.
3. For each query row, find top-K nearest training rows in the 3D embedding for K ∈ {50, 200, 500}.
4. Emit per-K stats:
   - mean(y) over K neighbors
   - std(y) over K neighbors (uncertainty)
   - fraction_positive (binary) / fraction_above_median (regression)
5. 9 features total (3 stats × 3 K-values).

Why this is structurally different from iter 60-70:
- Iter 60-68 use baseline residuals → anchor-routing in raw X-space.
- Iter 69 uses baseline predictions DIRECTLY (8 disagreement statistics).
- Iter 71 uses baseline predictions as EMBEDDING for kNN target-encoding → captures local neighborhood
  of target manifold, not just point disagreement.

Local-mean smoothing in y-aware embedding space = the canonical "stacked target encoding" trick.

Leakage discipline:
- Mode A (X_query=None): per outer fold, fit 3 baselines on train_idx → predict embedding for train_idx
  rows. For each val_idx row, find K nearest train_idx rows in embedding → emit stats over their y. No
  val_idx row sees its own y or its embedding via in-sample baseline.
- Mode B (X_query given): fit on full X_train → embed both X_train (NN index) and X_query (queries).
  For each X_query row, find K nearest X_train rows → emit stats over X_train y.

Cost: 3× baseline fits + sklearn NearestNeighbors O(N log N) build + query. Sub-second per fold.
"""
from __future__ import annotations

import logging
from typing import Any, Literal, Optional, Sequence

import numpy as np
import polars as pl

from ._utils import require_seed, validate_numeric_input

logger = logging.getLogger(__name__)

_DEFAULT_K_VALUES: tuple[int, ...] = (50, 200, 500)


def _fit_3baselines_predict_two(
    Xt: np.ndarray, y_t: np.ndarray, Xq: np.ndarray, task: str, seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit 3 baselines on (Xt, y_t), predict on BOTH Xt and Xq.

    Returns (train_embedding, query_embedding) shape (n_train, 3) and (n_query, 3).
    """
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ImportError("nn_oof_target_mean requires lightgbm") from exc
    from sklearn.linear_model import Ridge, LogisticRegression

    train_emb = np.zeros((Xt.shape[0], 3), dtype=np.float32)
    query_emb = np.zeros((Xq.shape[0], 3), dtype=np.float32)
    if task == "binary":
        m1 = lgb.LGBMClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=int(seed), verbose=-1, n_jobs=-1)
        m1.fit(Xt, y_t.astype(np.int32))
        train_emb[:, 0] = m1.predict_proba(Xt)[:, 1].astype(np.float32)
        query_emb[:, 0] = m1.predict_proba(Xq)[:, 1].astype(np.float32)
        m2 = lgb.LGBMClassifier(n_estimators=50, max_depth=5, learning_rate=0.1, random_state=int(seed) + 1, verbose=-1, n_jobs=-1)
        m2.fit(Xt, y_t.astype(np.int32))
        train_emb[:, 1] = m2.predict_proba(Xt)[:, 1].astype(np.float32)
        query_emb[:, 1] = m2.predict_proba(Xq)[:, 1].astype(np.float32)
        try:
            m3 = LogisticRegression(max_iter=200, solver="liblinear", random_state=int(seed) + 2)
            m3.fit(Xt, y_t.astype(np.int32))
            train_emb[:, 2] = m3.predict_proba(Xt)[:, 1].astype(np.float32)
            query_emb[:, 2] = m3.predict_proba(Xq)[:, 1].astype(np.float32)
        except Exception:
            prior = float(y_t.mean())
            train_emb[:, 2] = prior
            query_emb[:, 2] = prior
    else:
        m1 = lgb.LGBMRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=int(seed), verbose=-1, n_jobs=-1)
        m1.fit(Xt, y_t)
        train_emb[:, 0] = m1.predict(Xt).astype(np.float32)
        query_emb[:, 0] = m1.predict(Xq).astype(np.float32)
        m2 = lgb.LGBMRegressor(n_estimators=50, max_depth=5, learning_rate=0.1, random_state=int(seed) + 1, verbose=-1, n_jobs=-1)
        m2.fit(Xt, y_t)
        train_emb[:, 1] = m2.predict(Xt).astype(np.float32)
        query_emb[:, 1] = m2.predict(Xq).astype(np.float32)
        m3 = Ridge(alpha=1.0, random_state=int(seed) + 2)
        m3.fit(Xt, y_t)
        train_emb[:, 2] = m3.predict(Xt).astype(np.float32)
        query_emb[:, 2] = m3.predict(Xq).astype(np.float32)
    return train_emb, query_emb


def compute_nn_oof_target_mean_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: Optional[np.ndarray],
    splitter: Optional[Any] = None,
    *,
    seed: int,
    task: Literal["binary", "regression"] = "regression",
    k_values: Sequence[int] = _DEFAULT_K_VALUES,
    standardize: bool = True,
    column_prefix: str = "nnoof",
    dtype: type = np.float32,
) -> pl.DataFrame:
    """NN target-mean in 3D OOF embedding space.

    Output: per K-value: mean_y, std_y, frac_positive_or_above_median = 3 features.
    Total: len(k_values) × 3 features.
    """
    from sklearn.neighbors import NearestNeighbors

    seed = require_seed(seed)
    validate_numeric_input(X_train, name="X_train", allow_fp16=False)
    if X_query is not None:
        validate_numeric_input(X_query, name="X_query", allow_fp16=False)

    X_train_f = np.asarray(X_train, dtype=np.float32)
    y_train_f = np.asarray(y_train, dtype=np.float32).ravel()
    k_list = tuple(int(k) for k in k_values)
    n_features = 3 * len(k_list)

    def _process(Xt: np.ndarray, Xq: np.ndarray, y_t: np.ndarray, fold_seed: int) -> np.ndarray:
        """Embed Xt/Xq via 3 baseline models, find each query's k nearest training neighbours in embedding space, then for every k in k_list compute the neighbours' target mean/std/above-threshold-fraction."""
        if standardize:
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler().fit(Xt)
            Xt_s = scaler.transform(Xt).astype(np.float32)
            Xq_s = scaler.transform(Xq).astype(np.float32)
        else:
            Xt_s = Xt
            Xq_s = Xq
        train_emb, query_emb = _fit_3baselines_predict_two(Xt_s, y_t, Xq_s, task=task, seed=fold_seed)
        # Build NN index on train embedding.
        k_max = min(max(k_list), Xt_s.shape[0])
        nn = NearestNeighbors(n_neighbors=k_max, n_jobs=-1).fit(train_emb)
        _, neighbor_idx = nn.kneighbors(query_emb)  # (n_q, k_max)
        # y of neighbors: gather.
        neighbor_y = y_t[neighbor_idx]  # (n_q, k_max)
        # Pivot threshold for "fraction above" feature: median for regression, 0.5 for binary.
        threshold = 0.5 if task == "binary" else float(np.median(y_t))
        # Compute per-K stats.
        out = np.zeros((Xq_s.shape[0], n_features), dtype=np.float32)
        for ki, k in enumerate(k_list):
            k_eff = min(k, k_max)
            slice_y = neighbor_y[:, :k_eff]
            mean_y = slice_y.mean(axis=1).astype(np.float32)
            std_y = slice_y.std(axis=1).astype(np.float32) + 1e-9
            frac = (slice_y > threshold).mean(axis=1).astype(np.float32)
            base = ki * 3
            out[:, base] = mean_y
            out[:, base + 1] = std_y
            out[:, base + 2] = frac
        return out

    def _make_df(feats: np.ndarray) -> dict[str, np.ndarray]:
        """Name the 3 stats (mean/std/fraction-above) per k value into distinct output columns."""
        cols: dict[str, np.ndarray] = {}
        for ki, k in enumerate(k_list):
            base = ki * 3
            cols[f"{column_prefix}_k{k}_mean_y"] = feats[:, base].astype(dtype, copy=False)
            cols[f"{column_prefix}_k{k}_std_y"] = feats[:, base + 1].astype(dtype, copy=False)
            tag = "frac_pos" if task == "binary" else "frac_above_med"
            cols[f"{column_prefix}_k{k}_{tag}"] = feats[:, base + 2].astype(dtype, copy=False)
        return cols

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
        logger.info("nn_oof_target_mean: fold %d/%d done", fold_idx + 1, len(splits))

    return pl.DataFrame(_make_df(out))
