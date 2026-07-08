"""iter103 mechanism. Residual-stratified nearest-neighbour distance features.

Structural shift from iter69/iter102: instead of "more baselines, more disagreement stats",
this mechanism exposes the LOCAL STRUCTURE of where a baseline-shallow regressor's residuals
are large vs small in feature space. For each query, it reports distances to nearest training
rows in the "easy" half (where the baseline already predicts well) and in the "hard" half
(where the baseline residual is large).

Why this is different from iter69/iter102:
- iter69/102 expose the baseline's PREDICTION at the query x.
- iter103 exposes the LOCAL DENSITY of baseline-easy vs baseline-hard training rows around
  the query x. Two queries with identical baseline-prediction can be in very different
  "neighbourhood-difficulty" regimes; iter103 surfaces that.

Mechanism (per fold for Mode A, single-pass for Mode B):
1. Fit 1 LGB baseline (depth=5, 100 estimators) on training rows.
2. Compute OOF residual r_i = y_i - oof_pred_i for each train row.
3. Split train into easy_set = {i : |r_i| < median |r|} and hard_set = {i : |r_i| >= median |r|}.
4. For each query x_q, kNN to easy_set and hard_set separately at k in {1, 3, 5}.
5. Output 11 features:
   - 3 distances to easy_set kNN (k=1, 3, 5)
   - 3 distances to hard_set kNN (k=1, 3, 5)
   - 3 log-ratios log(d_hard_k / (d_easy_k + 1e-6))  (Bayes-rule-aligned: > 0 -> easy region)
   - mean |residual| of nearest k=5 easy training rows (locally how easy is "easy")
   - mean |residual| of nearest k=5 hard training rows (locally how hard is "hard")

Hypothesis: tree boostings can compute pairwise feature comparisons but not "distance to
nearest baseline-confused training row" without explicit per-instance kNN lookups. Surfacing
this geometric difficulty signal lets the downstream boosting allocate split budget to
"hard regions" without rebuilding the kNN structure inside its own training loop.

Leakage discipline:
- Mode A (X_query=None): per fold, baseline fit on train_idx; easy/hard split derived from
  baseline's PREDICTION ERROR on train_idx via inner KFold(3) (so each train row's residual
  is honest OOF wrt its inclusion in easy/hard). Val rows queried against full easy/hard sets.
- Mode B (X_query given): baseline fit on full X_train via outer KFold(3) to get OOF residuals,
  easy/hard split computed once, queries scored against full sets.

Cost: 1 LGB fit (+ inner KFold(3) of 3 LGB fits for OOF residual) + 2 sklearn NearestNeighbors
fits per fold. ~5x cheaper than iter69 (which fits 3 baselines + does kNN).
"""
from __future__ import annotations

import logging
from typing import Any, Literal, Optional

import numpy as np
import polars as pl

from ._intel_patch import try_patch_sklearn
from ._knn_helper import knn_search
from ._utils import require_seed, validate_numeric_input

logger = logging.getLogger(__name__)

_K_SCALES = (1, 3, 5)


def _resolve_lgb_device() -> str:
    """Return 'cuda' if mlframe's LGB-GPU probe is opted-in via MLFRAME_TRUST_LGB_CUDA=1, else 'cpu'.

    Uses the same gating as mlframe.training.helpers (which sets device_type='cuda' when the
    LightGBM wheel was built with USE_CUDA=ON and the user opted in). Default pip wheels of
    LightGBM are CPU-only, so the env var is required to avoid silent fallback warnings
    per LGBMRegressor call.
    """
    try:
        from mlframe.training import LGB_GPU_AVAILABLE
        return "cuda" if LGB_GPU_AVAILABLE else "cpu"
    except ImportError:
        return "cpu"


def _compute_oof_residuals(Xt: np.ndarray, y_t: np.ndarray, task: str, seed: int) -> np.ndarray:
    """Return |OOF residual| per training row via inner KFold(3) LGB fits.

    Uses LGB GPU (device_type='cuda') when ``MLFRAME_TRUST_LGB_CUDA=1`` env var is set and the
    installed LightGBM wheel has CUDA support; otherwise CPU. Matches the same opt-in gating
    used in mlframe.training.helpers — see mlframe.training._gpu_probe for the probe pattern.
    """
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ImportError("residual_stratified_distance requires lightgbm") from exc
    from sklearn.model_selection import KFold

    lgb_device = _resolve_lgb_device()
    n = Xt.shape[0]
    oof = np.zeros(n, dtype=np.float32)
    inner_splitter = KFold(n_splits=3, shuffle=True, random_state=int(seed) + 11)
    for inner_idx, (in_tr, in_val) in enumerate(inner_splitter.split(Xt)):
        if task == "binary":
            m = lgb.LGBMClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.1, device_type=lgb_device, random_state=int(seed) + 7 + inner_idx, verbose=-1, n_jobs=-1
            )
            m.fit(Xt[in_tr], y_t[in_tr].astype(np.int32))
            oof[in_val] = m.predict_proba(Xt[in_val])[:, 1].astype(np.float32)
        else:
            m = lgb.LGBMRegressor(
                n_estimators=100, max_depth=5, learning_rate=0.1, device_type=lgb_device, random_state=int(seed) + 7 + inner_idx, verbose=-1, n_jobs=-1
            )
            m.fit(Xt[in_tr], y_t[in_tr])
            oof[in_val] = m.predict(Xt[in_val]).astype(np.float32)
    return np.abs(y_t - oof).astype(np.float32)


def _kth_nearest_dists_and_residuals(
    X_subset: np.ndarray,
    abs_resid_subset: np.ndarray,
    X_query: np.ndarray,
    residual_k: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Single kNN search returning both (dists at _K_SCALES, mean |residual| at k=residual_k).

    Replaces the prior pair of calls (`_kth_nearest_dists` + `_kth_nearest_residuals`) which
    fitted NearestNeighbors twice on the same X_subset just to get ids separately. ~2x faster
    on the geometry stage and a strict prerequisite for any later hnswlib path (which has
    higher per-index build cost than sklearn).

    Returns:
        dists: (n_query, len(_K_SCALES)) — distances at k in _K_SCALES
        mean_resid: (n_query,) — mean |residual| of nearest residual_k training rows
    """
    n_sub = X_subset.shape[0]
    n_q = X_query.shape[0]
    if n_sub == 0:
        return (
            np.full((n_q, len(_K_SCALES)), 1e6, dtype=np.float32),
            np.zeros(n_q, dtype=np.float32),
        )
    k_max = max(max(_K_SCALES), residual_k)
    # Single knn_search: returns both dists and ids; we use dists for _K_SCALES slicing and
    # ids for residual lookup. _knn_helper picks hnswlib at N>=50000, sklearn fallback else.
    dists, ids = knn_search(X_subset, X_query, k=k_max)
    n_returned = dists.shape[1]
    out_dists = np.empty((n_q, len(_K_SCALES)), dtype=np.float32)
    for i, k in enumerate(_K_SCALES):
        k_eff = min(k, n_returned) - 1
        out_dists[:, i] = dists[:, k_eff]
    r_k = min(residual_k, n_returned)
    mean_resid = abs_resid_subset[ids[:, :r_k]].mean(axis=1).astype(np.float32)
    return out_dists, mean_resid


def compute_residual_stratified_distance_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: Optional[np.ndarray],
    splitter: Optional[Any] = None,
    *,
    seed: int,
    task: Literal["binary", "regression"] = "regression",
    standardize: bool = True,
    column_prefix: str = "rsd",
    dtype: type = np.float32,
) -> pl.DataFrame:
    """Residual-stratified nearest-neighbour distance features (iter103).

    Output: 11 features per row.
    """
    seed = require_seed(seed)
    try_patch_sklearn()
    validate_numeric_input(X_train, name="X_train", allow_fp16=False)
    if X_query is not None:
        validate_numeric_input(X_query, name="X_query", allow_fp16=False)

    X_train_f = np.asarray(X_train, dtype=np.float32)
    y_train_f = np.asarray(y_train, dtype=np.float32).ravel()

    def _process(Xt: np.ndarray, Xq: np.ndarray, y_t: np.ndarray, fold_seed: int) -> np.ndarray:
        if standardize:
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler().fit(Xt)
            Xt_s = scaler.transform(Xt).astype(np.float32)
            Xq_s = scaler.transform(Xq).astype(np.float32)
        else:
            Xt_s = Xt
            Xq_s = Xq

        abs_r = _compute_oof_residuals(Xt_s, y_t, task=task, seed=fold_seed)
        median_r = float(np.median(abs_r))
        easy_mask = abs_r < median_r
        hard_mask = ~easy_mask
        X_easy = Xt_s[easy_mask]
        r_easy = abs_r[easy_mask]
        X_hard = Xt_s[hard_mask]
        r_hard = abs_r[hard_mask]

        # Single kNN search per band returns both dists at _K_SCALES and mean |residual| at k=5.
        # Replaces prior 2 NN fits (one for dists, one for residuals) per band.
        d_easy, mean_r_easy = _kth_nearest_dists_and_residuals(X_easy, r_easy, Xq_s, residual_k=5)
        d_hard, mean_r_hard = _kth_nearest_dists_and_residuals(X_hard, r_hard, Xq_s, residual_k=5)
        log_ratio = np.log((d_hard + 1e-6) / (d_easy + 1e-6))
        return np.column_stack([d_easy, d_hard, log_ratio, mean_r_easy, mean_r_hard])

    def _make_df(feats: np.ndarray) -> dict[str, np.ndarray]:
        cols: dict[str, np.ndarray] = {}
        cols[f"{column_prefix}_d_easy_k1"] = feats[:, 0].astype(dtype, copy=False)
        cols[f"{column_prefix}_d_easy_k3"] = feats[:, 1].astype(dtype, copy=False)
        cols[f"{column_prefix}_d_easy_k5"] = feats[:, 2].astype(dtype, copy=False)
        cols[f"{column_prefix}_d_hard_k1"] = feats[:, 3].astype(dtype, copy=False)
        cols[f"{column_prefix}_d_hard_k3"] = feats[:, 4].astype(dtype, copy=False)
        cols[f"{column_prefix}_d_hard_k5"] = feats[:, 5].astype(dtype, copy=False)
        cols[f"{column_prefix}_logratio_k1"] = feats[:, 6].astype(dtype, copy=False)
        cols[f"{column_prefix}_logratio_k3"] = feats[:, 7].astype(dtype, copy=False)
        cols[f"{column_prefix}_logratio_k5"] = feats[:, 8].astype(dtype, copy=False)
        cols[f"{column_prefix}_mean_r_easy_k5"] = feats[:, 9].astype(dtype, copy=False)
        cols[f"{column_prefix}_mean_r_hard_k5"] = feats[:, 10].astype(dtype, copy=False)
        return cols

    n_features = 11

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
        logger.info("residual_stratified_distance: fold %d/%d done", fold_idx + 1, len(splits))

    return pl.DataFrame(_make_df(out))
