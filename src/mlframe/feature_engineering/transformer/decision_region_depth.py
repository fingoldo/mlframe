"""Decision region depth: isotropic probes per query measure boundary distance.

Iter 83 mechanism. Adversarial agent's #5 ranked.

For binary: from each query row, draw 8 random unit directions; for each direction binary-search the
distance (in z-units, capped at 3σ) along which prediction flips. Emit min/median/max probe distance.

For regression: same but flip = move across train residual decile.

5 features per row.
"""
from __future__ import annotations

import logging
from typing import Any, Literal, Optional

import numpy as np
import polars as pl

from ._utils import require_seed, validate_numeric_input

logger = logging.getLogger(__name__)


def compute_decision_region_depth_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: Optional[np.ndarray],
    splitter: Optional[Any] = None,
    *,
    seed: int,
    task: Literal["binary", "regression"] = "regression",
    n_probes: int = 8,
    max_scale: float = 3.0,
    standardize: bool = True,
    column_prefix: str = "drd",
    dtype: type = np.float32,
) -> pl.DataFrame:
    """Decision region depth via isotropic random-direction probes."""
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ImportError("decision_region_depth requires lightgbm") from exc

    seed = require_seed(seed)
    validate_numeric_input(X_train, name="X_train", allow_fp16=False)
    if X_query is not None:
        validate_numeric_input(X_query, name="X_query", allow_fp16=False)

    X_train_f = np.asarray(X_train, dtype=np.float32)
    y_train_f = np.asarray(y_train, dtype=np.float32).ravel()
    n_features_out = 5

    def _process(Xt: np.ndarray, Xq: np.ndarray, y_t: np.ndarray, fold_seed: int) -> np.ndarray:
        """Fit a shallow model on (Xt, y_t), then for each query row and each of ``n_probes`` random directions binary/coarse-search (at scales 0.5/1/2/max_scale sigma) the smallest perturbation that flips the prediction, returning the min/max/median/mean/std of the flip distances."""
        if standardize:
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler().fit(Xt)
            Xt_s = scaler.transform(Xt).astype(np.float32)
            Xq_s = scaler.transform(Xq).astype(np.float32)
        else:
            Xt_s = Xt
            Xq_s = Xq
        d = Xt_s.shape[1]
        n_q = Xq_s.shape[0]
        feat_std = Xt_s.std(axis=0) + 1e-6
        if task == "binary":
            model = lgb.LGBMClassifier(n_estimators=50, max_depth=3, learning_rate=0.1,
                                       random_state=int(fold_seed), verbose=-1, n_jobs=-1).fit(Xt_s, y_t.astype(np.int32))
            pred_orig = model.predict_proba(Xq_s)[:, 1].astype(np.float32)
            orig_class = (pred_orig > 0.5).astype(np.float32)
        else:
            model = lgb.LGBMRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=int(fold_seed), verbose=-1, n_jobs=-1).fit(Xt_s, y_t)
            pred_orig = model.predict(Xq_s).astype(np.float32)
            train_resid_decile = float(np.quantile(np.abs(y_t - model.predict(Xt_s)), 0.10))
            orig_class = None

        rng = np.random.default_rng(int(fold_seed))
        # Sample n_probes random unit directions.
        directions = rng.standard_normal((n_probes, d)).astype(np.float32)
        directions = directions / (np.linalg.norm(directions, axis=1, keepdims=True) + 1e-9)
        # Scale directions by feature stds for isotropic-in-z perturbation
        directions_scaled = directions * feat_std[None, :]

        # For each direction, do coarse scale search at {0.5, 1, 2, 3} σ
        scales = np.array([0.5, 1.0, 2.0, max_scale], dtype=np.float32)
        # probe_dist[probe, q] = smallest scale that flips
        # CPX36 FUTURE: this loop does a FIXED n_probes*len(scales) (=32) full predicts independent
        # of d, unlike the sibling transformers whose predict count scales with d (where batching the
        # vertical perturbation stack paid 1.4-2.7x). Here batching 32 calls into one (32*n_q, d)
        # predict would amortize far less LightGBM per-call overhead (8 probes x 4 scales), and the
        # stack grows with n_probes*n_q*d which would need the same size gate for a smaller payoff.
        # Deferred as the lowest-yield of the five; revisit if n_probes is raised materially.
        flip_dists = np.full((n_probes, n_q), max_scale + 0.5, dtype=np.float32)
        for p in range(n_probes):
            for scale in scales:
                Xq_perturbed = Xq_s + scale * directions_scaled[p][None, :]
                if task == "binary":
                    pred_pert = model.predict_proba(Xq_perturbed)[:, 1].astype(np.float32)
                    flipped = (pred_pert > 0.5).astype(np.float32) != orig_class
                else:
                    pred_pert = model.predict(Xq_perturbed).astype(np.float32)
                    flipped = np.abs(pred_pert - pred_orig) > train_resid_decile
                update_mask = flipped & (flip_dists[p] > scale)
                flip_dists[p, update_mask] = scale

        min_d = flip_dists.min(axis=0).astype(np.float32)
        max_d = flip_dists.max(axis=0).astype(np.float32)
        median_d = np.median(flip_dists, axis=0).astype(np.float32)
        mean_d = flip_dists.mean(axis=0).astype(np.float32)
        std_d = flip_dists.std(axis=0).astype(np.float32) + 1e-9
        return np.column_stack([min_d, max_d, median_d, mean_d, std_d])

    def _make_df(feats: np.ndarray) -> dict[str, np.ndarray]:
        """Split the (n, 5) probe-distance matrix into named, dtype-cast columns for the output polars frame."""
        cols: dict[str, np.ndarray] = {}
        cols[f"{column_prefix}_min_dist"] = feats[:, 0].astype(dtype, copy=False)
        cols[f"{column_prefix}_max_dist"] = feats[:, 1].astype(dtype, copy=False)
        cols[f"{column_prefix}_median_dist"] = feats[:, 2].astype(dtype, copy=False)
        cols[f"{column_prefix}_mean_dist"] = feats[:, 3].astype(dtype, copy=False)
        cols[f"{column_prefix}_std_dist"] = feats[:, 4].astype(dtype, copy=False)
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
        logger.info("decision_region_depth: fold %d/%d done", fold_idx + 1, len(splits))

    return pl.DataFrame(_make_df(out))
