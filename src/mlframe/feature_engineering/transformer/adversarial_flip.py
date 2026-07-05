"""Adversarial flip distance features: smallest per-feature perturbation to flip baseline prediction.

Iter 79 mechanism. Adversarial agent's #1 ranked.

For each query row, for each feature j: coarse line search ε ∈ {0.5σ, 1σ, 2σ} in both ±directions
to find smallest |Δx_j| that:
- Binary: flips class prediction (crosses 0.5).
- Regression: moves prediction across one residual-decile boundary on train.

Emit per row:
- min_flip_dist — smallest |Δ| across features (∞ if no flip within 2σ)
- mean_flip_dist — mean across features
- max_flip_dist — max
- argmin_feature_id — which feature gives smallest flip distance
- frac_flippable — fraction of features with a flip within 2σ

5 features.
"""
from __future__ import annotations

import logging
from typing import Any, Literal, Optional

import numpy as np
import polars as pl

from ._utils import require_seed, validate_numeric_input

logger = logging.getLogger(__name__)

_SCALES = (0.5, 1.0, 2.0)

# Cap on the vertically-stacked perturbation matrix (float32 elems). 64M ~ 256MB; above this
# fall back to the per-combo loop to bound peak RAM.
_MAX_STACK_ELEMS = 64_000_000


def compute_adversarial_flip_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: Optional[np.ndarray],
    splitter: Optional[Any] = None,
    *,
    seed: int,
    task: Literal["binary", "regression"] = "regression",
    standardize: bool = True,
    column_prefix: str = "advflip",
    dtype: np.dtype = np.float32,
) -> pl.DataFrame:
    """Adversarial flip distance features per query row.

    Output: 5 features — min_dist, mean_dist, max_dist, argmin_feature_id, frac_flippable.
    """
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ImportError("adversarial_flip requires lightgbm") from exc

    seed = require_seed(seed)
    validate_numeric_input(X_train, name="X_train", allow_fp16=False)
    if X_query is not None:
        validate_numeric_input(X_query, name="X_query", allow_fp16=False)

    X_train_f = np.asarray(X_train, dtype=np.float32)
    y_train_f = np.asarray(y_train, dtype=np.float32).ravel()
    n_features_out = 5

    def _process(Xt: np.ndarray, Xq: np.ndarray, y_t: np.ndarray, fold_seed: int) -> np.ndarray:
        if standardize:
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler().fit(Xt)
            Xt_s = scaler.transform(Xt).astype(np.float32)
            Xq_s = scaler.transform(Xq).astype(np.float32)
        else:
            Xt_s = Xt
            Xq_s = Xq
        d = Xt_s.shape[1]
        feat_std = Xt_s.std(axis=0) + 1e-6
        if task == "binary":
            model = lgb.LGBMClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=int(fold_seed), verbose=-1, n_jobs=-1)
            model.fit(Xt_s, y_t.astype(np.int32))
            pred_orig = model.predict_proba(Xq_s)[:, 1].astype(np.float32)
            orig_class = (pred_orig > 0.5).astype(np.float32)
        else:
            model = lgb.LGBMRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=int(fold_seed), verbose=-1, n_jobs=-1)
            model.fit(Xt_s, y_t)
            pred_orig = model.predict(Xq_s).astype(np.float32)
            train_pred = model.predict(Xt_s).astype(np.float32)
            train_resid_decile = float(np.quantile(np.abs(y_t - train_pred), 0.10))
            orig_class = None  # for regression we compare pred-shift magnitude

        n_q = Xq_s.shape[0]
        # For each row, compute per-feature minimum scale that flips.
        # We try scales in _SCALES (0.5σ, 1σ, 2σ) in ±directions. Record smallest scale that flips.
        flip_dists = np.full((n_q, d), 2.5, dtype=np.float32)  # default = "no flip within 2σ"
        # (j, scale, sign) perturbation combos — each is one full predict in the per-feature loop.
        combos = [(j, scale, sign) for j in range(d) for scale in _SCALES for sign in (-1.0, 1.0)]
        n_combo = len(combos)
        # Batched path: stack all n_combo perturbations into one (n_combo*n_q, d) matrix and do a
        # SINGLE predict, then replay the per-combo flip-update in the SAME order. Bit-identical to
        # the per-combo predict for tree models. Gated on stack size to bound peak RAM.
        stack_elems = n_combo * n_q * d
        if stack_elems <= _MAX_STACK_ELEMS and n_combo > 0:
            stack = np.broadcast_to(Xq_s, (n_combo, n_q, d)).reshape(n_combo * n_q, d).copy()
            for c, (j, scale, sign) in enumerate(combos):
                stack[c * n_q : (c + 1) * n_q, j] += sign * scale * feat_std[j]
            if task == "binary":
                pred_all = model.predict_proba(stack)[:, 1].astype(np.float32).reshape(n_combo, n_q)
            else:
                pred_all = model.predict(stack).astype(np.float32).reshape(n_combo, n_q)
            for c, (j, scale, sign) in enumerate(combos):
                pred_pert = pred_all[c]
                if task == "binary":
                    flipped = (pred_pert > 0.5).astype(np.float32) != orig_class
                else:
                    flipped = np.abs(pred_pert - pred_orig) > train_resid_decile
                update_mask = flipped & (flip_dists[:, j] > scale)
                flip_dists[update_mask, j] = scale
        else:
            for j in range(d):
                for scale in _SCALES:
                    for sign in (-1.0, 1.0):
                        Xq_perturbed = Xq_s.copy()
                        Xq_perturbed[:, j] = Xq_perturbed[:, j] + sign * scale * feat_std[j]
                        if task == "binary":
                            pred_pert = model.predict_proba(Xq_perturbed)[:, 1].astype(np.float32)
                            flipped = (pred_pert > 0.5).astype(np.float32) != orig_class
                        else:
                            pred_pert = model.predict(Xq_perturbed).astype(np.float32)
                            flipped = np.abs(pred_pert - pred_orig) > train_resid_decile
                        # Update minimum scale where flip occurred.
                        update_mask = flipped & (flip_dists[:, j] > scale)
                        flip_dists[update_mask, j] = scale

        min_dist = flip_dists.min(axis=1).astype(np.float32)
        mean_dist = flip_dists.mean(axis=1).astype(np.float32)
        max_dist = flip_dists.max(axis=1).astype(np.float32)
        argmin_feat = flip_dists.argmin(axis=1).astype(np.float32)
        frac_flippable = (flip_dists < 2.5).mean(axis=1).astype(np.float32)
        return np.column_stack([min_dist, mean_dist, max_dist, argmin_feat, frac_flippable])

    def _make_df(feats: np.ndarray) -> dict[str, np.ndarray]:
        cols: dict[str, np.ndarray] = {}
        cols[f"{column_prefix}_min_dist"] = feats[:, 0].astype(dtype, copy=False)
        cols[f"{column_prefix}_mean_dist"] = feats[:, 1].astype(dtype, copy=False)
        cols[f"{column_prefix}_max_dist"] = feats[:, 2].astype(dtype, copy=False)
        cols[f"{column_prefix}_argmin_feat"] = feats[:, 3].astype(dtype, copy=False)
        cols[f"{column_prefix}_frac_flippable"] = feats[:, 4].astype(dtype, copy=False)
        return cols

    if X_query is not None:
        Xq = np.asarray(X_query, dtype=np.float32)
        feats = _process(X_train_f, Xq, y_train_f, seed)
        return pl.DataFrame(_make_df(feats))

    if splitter is None:
        raise ValueError("Mode A (X_query=None) requires a splitter.")
    n_train = X_train_f.shape[0]
    out = np.zeros((n_train, n_features_out), dtype=dtype)
    splits = list(splitter.split(X_train_f))
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        feats = _process(X_train_f[train_idx], X_train_f[val_idx], y_train_f[train_idx], int(seed) + fold_idx * 100)
        out[val_idx] = feats.astype(dtype, copy=False)
        logger.info("adversarial_flip: fold %d/%d done", fold_idx + 1, len(splits))

    return pl.DataFrame(_make_df(out))
