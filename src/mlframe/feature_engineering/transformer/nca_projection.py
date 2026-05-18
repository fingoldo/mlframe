"""NCA (Neighborhood Components Analysis) learned projection features.

Iter 38 mechanism. FIRST BEYOND-FROZEN mechanism — uses gradient-trained backprop to learn a target-aware linear projection.

NCA (Goldberger et al. 2005) fits a linear transformation matrix W ∈ R^(d × d_embed) by maximizing the expected leave-one-out kNN classification accuracy of the
training set under the Euclidean distance in the projected space ``W x``. The gradient is computed analytically; sklearn uses L-BFGS.

The learned W is fundamentally different from:
- Random projection (RFF): target-agnostic.
- PCA: target-agnostic, maximizes variance.
- LDA (iter 37): target-aware but linear and assumes Gaussian classes.
- Frozen kNN with raw features: target-agnostic geometry.

NCA learns the projection that makes same-class points MORE similar in the projected space and different-class points LESS similar — supervised metric learning via
backprop on a continuous relaxation of kNN-accuracy.

Mechanism:
1. Per fold, fit NCA on (X_train_fold, y_train_fold) → learns projection W_fold.
2. Project query rows: ``X_query_projected = X_query @ W_fold``.
3. Expose projected coordinates as features (d_embed=4 default).

For regression: convert y to top/bottom quintile binary labels for NCA fit; project on learned W.

Why "beyond frozen": the projection W is determined by gradient descent over a target-aware objective. The PROJECTION FUNCTION itself is learned, not a fixed
random/PCA/LDA transform.

Why CB-blind: CB cannot internally fit a multi-feature linear projection via gradient. It can split per-feature; it cannot construct a target-aware linear combination
of features as a single feature.

Leakage discipline: NCA refit per fold from train-fold rows + labels only.

Cost: sklearn NCA with default L-BFGS, d=6-10, n=4000 trains in ~5-30 sec per fold. Total: ~30s-3min per dataset.

References:
- Goldberger et al. 2005 — Neighborhood Components Analysis.
- Yang & Jin 2006 — distance metric learning survey.
- sklearn.neighbors.NeighborhoodComponentsAnalysis.
"""
from __future__ import annotations

import logging
import warnings
from typing import Any, Literal, Optional

import numpy as np
import polars as pl

from ._utils import require_seed, validate_numeric_input

logger = logging.getLogger(__name__)


def _fit_nca(X: np.ndarray, y_binary: np.ndarray, *, n_components: int, max_iter: int, seed: int):
    from sklearn.neighbors import NeighborhoodComponentsAnalysis
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(X)
    X_s = scaler.transform(X)
    n_components_eff = min(n_components, X.shape[1])
    nca = NeighborhoodComponentsAnalysis(
        n_components=n_components_eff,
        init="pca",  # initialise from PCA for stability
        max_iter=max_iter,
        random_state=seed,
        warm_start=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nca.fit(X_s, y_binary.astype(int))
    return nca, scaler


def compute_nca_projection_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: Optional[np.ndarray],
    splitter: Optional[Any] = None,
    *,
    seed: int,
    task: Literal["binary", "regression"] = "binary",
    n_components: int = 4,
    max_iter: int = 50,
    q_high: float = 0.8,
    column_prefix: str = "nca",
    dtype: np.dtype = np.float32,
) -> pl.DataFrame:
    """NCA learned-projection coordinates as features.

    Output: ``n_components`` columns per row, named ``{prefix}_c{j}``.

    Beyond-frozen mechanism: learns target-aware linear projection via gradient (L-BFGS).
    """
    seed = require_seed(seed)
    validate_numeric_input(X_train, name="X_train", allow_fp16=False)
    if X_query is not None:
        validate_numeric_input(X_query, name="X_query", allow_fp16=False)
    if task not in ("binary", "regression"):
        raise ValueError(f"task must be 'binary' or 'regression'; got {task!r}.")

    X_train_f = np.asarray(X_train, dtype=np.float32)
    y_train_f = np.asarray(y_train, dtype=np.float32).ravel()

    def _binarize(y_sub: np.ndarray) -> np.ndarray:
        if task == "binary":
            return (y_sub > 0.5).astype(int)
        threshold = np.quantile(y_sub, q_high)
        return (y_sub >= threshold).astype(int)

    def _process(Xt: np.ndarray, Xq: np.ndarray, y_t: np.ndarray, fold_seed: int) -> np.ndarray:
        y_bin = _binarize(y_t)
        # NCA needs at least 2 classes with non-empty members.
        if y_bin.sum() < 2 or (1 - y_bin).sum() < 2:
            return np.zeros((Xq.shape[0], n_components), dtype=np.float32)
        nca, scaler = _fit_nca(Xt, y_bin, n_components=n_components, max_iter=max_iter, seed=fold_seed)
        Xq_s = scaler.transform(Xq)
        proj = nca.transform(Xq_s).astype(np.float32)
        # Pad if NCA returned fewer components than requested.
        if proj.shape[1] < n_components:
            padded = np.zeros((Xq.shape[0], n_components), dtype=np.float32)
            padded[:, :proj.shape[1]] = proj
            proj = padded
        return proj

    def _make_df(feats: np.ndarray) -> dict[str, np.ndarray]:
        return {f"{column_prefix}_c{j}": feats[:, j].astype(dtype, copy=False) for j in range(feats.shape[1])}

    if X_query is not None:
        Xq = np.asarray(X_query, dtype=np.float32)
        feats = _process(X_train_f, Xq, y_train_f, seed)
        return pl.DataFrame(_make_df(feats))

    if splitter is None:
        raise ValueError("Mode A (X_query=None) requires a splitter.")
    n_train = X_train_f.shape[0]
    out = np.zeros((n_train, n_components), dtype=dtype)
    splits = list(splitter.split(X_train_f))
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        feats = _process(X_train_f[train_idx], X_train_f[val_idx], y_train_f[train_idx], int(seed) + fold_idx * 13)
        out[val_idx] = feats.astype(dtype, copy=False)
        logger.info("nca_projection: fold %d/%d done (n_train=%d, n_val=%d, n_components=%d)", fold_idx + 1, len(splits), len(train_idx), len(val_idx), n_components)

    return pl.DataFrame(_make_df(out))
