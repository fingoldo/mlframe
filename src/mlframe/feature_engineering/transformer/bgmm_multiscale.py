"""Multi-scale BGM virtual sampling: fit BayesianGaussianMixture at multiple component counts, expose distance features from each.

Iter 45 mechanism. Directly extends iter 41 BGM winner (which used a single n_components=5) by fitting multiple GMMs at different resolutions.

Different component counts give different density estimates:
- Low K (e.g. 3): coarse-grained — captures broad modes.
- Medium K (e.g. 5): the iter-41 winner — balances bias and variance.
- High K (e.g. 8): fine-grained — captures fine sub-modes but may overfit on rare positive class.

For each scale, sample virtuals from the learned posterior and expose distance features (k=1,3,5,10 + log-gap). Total: ``len(component_counts) * (8)`` features.

Why this should improve on iter 41:
- iter 41 fixed at K=5: forced choice of resolution.
- iter 45: exposes ALL resolutions; the boosting can split on the most-relevant scale for each region of X.

Tradeoff: more features (default 24 vs 8) — more chance of dilution. The boosting's feature-selection should handle this.

Leakage discipline: BGMs refit per fold per scale.

Cost: ~3 × iter-41 cost = ~3-6 sec per fold for 3 scales.
"""
from __future__ import annotations

import logging
import warnings
from typing import Any, Literal, Optional, Tuple

import numpy as np
import polars as pl

from ._utils import require_seed, validate_numeric_input

logger = logging.getLogger(__name__)

_K_SCALES = (1, 3, 5, 10)
_COMPONENT_COUNTS_DEFAULT = (3, 5, 8)


def _fit_bgmm_and_sample(X_minority: np.ndarray, n_synthetic: int, n_components: int, seed: int) -> np.ndarray:
    """Same as iter 41 helper — fit BayesianGaussianMixture and sample virtuals."""
    from sklearn.mixture import BayesianGaussianMixture
    n_min = X_minority.shape[0]
    if n_min < n_components + 1:
        return np.asarray(X_minority[np.random.default_rng(seed).integers(0, n_min, size=n_synthetic)].copy().astype(np.float32))
    bgm = BayesianGaussianMixture(
        n_components=n_components,
        covariance_type="full",
        max_iter=200,
        random_state=seed,
        reg_covar=1e-4,
        weight_concentration_prior_type="dirichlet_process",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            bgm.fit(X_minority)
            samples, _ = bgm.sample(n_synthetic)
        except Exception as exc:
            logger.info("bgmm_multiscale: BGM fit failed at K=%d (%s); falling back to bootstrap.", n_components, exc)
            rng = np.random.default_rng(seed)
            samples = X_minority[rng.integers(0, n_min, size=n_synthetic)]
    return np.asarray(samples.astype(np.float32))


def _kth_nearest_dists(X_subset: np.ndarray, X_query: np.ndarray, k_max: int) -> np.ndarray:
    """Compute, for each row of ``X_query``, the distance to its k-th nearest neighbor in ``X_subset`` for every k in ``_K_SCALES``.

    Falls back to a large constant (1e6) when ``X_subset`` is empty (e.g. no negatives/positives after slicing), and clips
    the effective k to the available subset size when ``X_subset`` has fewer than ``k_max`` rows so small folds don't error.
    """
    from sklearn.neighbors import NearestNeighbors
    n_sub = X_subset.shape[0]
    if n_sub == 0:
        return np.full((X_query.shape[0], len(_K_SCALES)), 1e6, dtype=np.float32)
    k_request = min(k_max, n_sub)
    nn = NearestNeighbors(n_neighbors=k_request, algorithm="auto", n_jobs=-1).fit(X_subset)
    dists, _ids = nn.kneighbors(X_query)
    out = np.zeros((X_query.shape[0], len(_K_SCALES)), dtype=np.float32)
    for col_idx, k in enumerate(_K_SCALES):
        eff_k = min(k, k_request)
        out[:, col_idx] = dists[:, eff_k - 1]
    return out


def compute_bgmm_multiscale_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: Optional[np.ndarray],
    splitter: Optional[Any] = None,
    *,
    seed: int,
    task: Literal["binary", "regression"] = "binary",
    component_counts: Tuple[int, ...] = _COMPONENT_COUNTS_DEFAULT,
    n_synthetic_multiplier: float = 5.0,
    q_high: float = 0.8,
    standardize: bool = True,
    column_prefix: str = "bgmms",
    dtype: type = np.float32,
) -> pl.DataFrame:
    """Multi-scale BGM virtual distance features.

    Output: ``len(component_counts) * 2 * len(_K_SCALES)`` columns. Default 3 × 2 × 4 = 24 features.
    """
    seed = require_seed(seed)
    validate_numeric_input(X_train, name="X_train", allow_fp16=False)
    if X_query is not None:
        validate_numeric_input(X_query, name="X_query", allow_fp16=False)

    X_train_f = np.asarray(X_train, dtype=np.float32)
    y_train_f = np.asarray(y_train, dtype=np.float32).ravel()

    def _slice(X_sub: np.ndarray, y_sub: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Split rows into (positive, negative) groups: label>0.5 for binary tasks, or the top/bottom ``q_high`` quantile tails for regression."""
        if task == "binary":
            pos = y_sub > 0.5
            return X_sub[pos], X_sub[~pos]
        y_hi = np.quantile(y_sub, q_high)
        y_lo = np.quantile(y_sub, 1.0 - q_high)
        return X_sub[y_sub >= y_hi], X_sub[y_sub <= y_lo]

    def _process(Xt: np.ndarray, Xq: np.ndarray, y_t: np.ndarray, fold_seed: int) -> np.ndarray:
        """Fit one BGM per scale on ``Xt``'s positives, then emit per-query [pos_dist, log_gap] blocks for each scale, concatenated in ``component_counts`` order.

        Optionally standardizes with a ``RobustScaler`` fit on ``Xt`` only (fold-local, no leakage). Returns an all-zeros
        block when either class has fewer than 2 rows in this fold (too little signal to fit a mixture or compute k-NN).
        """
        if standardize:
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler().fit(Xt)
            Xt_s = scaler.transform(Xt).astype(np.float32)
            Xq_s = scaler.transform(Xq).astype(np.float32)
        else:
            Xt_s = Xt
            Xq_s = Xq
        Xt_pos, Xt_neg = _slice(Xt_s, y_t)
        if Xt_pos.shape[0] < 2 or Xt_neg.shape[0] < 2:
            return np.zeros((Xq_s.shape[0], 2 * len(_K_SCALES) * len(component_counts)), dtype=np.float32)
        neg_d = _kth_nearest_dists(Xt_neg, Xq_s, max(_K_SCALES))
        n_synthetic = max(50, int(Xt_pos.shape[0] * n_synthetic_multiplier))
        all_feats = []
        for scale_idx, n_comp in enumerate(component_counts):
            k_mix = max(2, min(n_comp, Xt_pos.shape[0] // 3))
            X_synth = _fit_bgmm_and_sample(Xt_pos, n_synthetic=n_synthetic, n_components=k_mix, seed=fold_seed + scale_idx * 7)
            X_virtual_pos = np.concatenate([Xt_pos, X_synth], axis=0)
            pos_d = _kth_nearest_dists(X_virtual_pos, Xq_s, max(_K_SCALES))
            log_gap = np.log(np.maximum(neg_d, 1e-9)) - np.log(np.maximum(pos_d, 1e-9))
            all_feats.append(pos_d)
            all_feats.append(log_gap)
        return np.asarray(np.concatenate(all_feats, axis=1).astype(np.float32))

    def _make_df(feats: np.ndarray) -> dict[str, np.ndarray]:
        """Label the flat ``feats`` column block with ``{prefix}_K{n_comp}_{pos|loggap}_k{k}`` names, matching the emission order of ``_process``."""
        cols: dict[str, np.ndarray] = {}
        col_idx = 0
        for n_comp in component_counts:
            tag = f"K{n_comp}"
            for k in _K_SCALES:
                cols[f"{column_prefix}_{tag}_pos_k{k}"] = feats[:, col_idx].astype(dtype, copy=False)
                col_idx += 1
            for k in _K_SCALES:
                cols[f"{column_prefix}_{tag}_loggap_k{k}"] = feats[:, col_idx].astype(dtype, copy=False)
                col_idx += 1
        return cols

    if X_query is not None:
        Xq = np.asarray(X_query, dtype=np.float32)
        feats = _process(X_train_f, Xq, y_train_f, seed)
        return pl.DataFrame(_make_df(feats))

    if splitter is None:
        raise ValueError("Mode A (X_query=None) requires a splitter.")
    n_train = X_train_f.shape[0]
    n_features = 2 * len(_K_SCALES) * len(component_counts)
    out: np.ndarray = np.zeros((n_train, n_features), dtype=dtype)
    splits = list(splitter.split(X_train_f))
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        feats = _process(X_train_f[train_idx], X_train_f[val_idx], y_train_f[train_idx], int(seed) + fold_idx * 100)
        out[val_idx] = feats.astype(dtype, copy=False)
        logger.info("bgmm_multiscale: fold %d/%d done", fold_idx + 1, len(splits))

    return pl.DataFrame(_make_df(out))
