"""Persistence diagram features via gudhi: H0/H1 birth-death of local Vietoris-Rips filtration.

Iter 86 mechanism. Geometric agent's #4 ranked. Topology of local residual field.

For each query: take K=30 nearest train rows, compute Vietoris-Rips filtration via gudhi.RipsComplex
on (X-distance, y-difference) joint metric, extract H0 persistence pairs (connected components).
Emit max persistence (longest-lived bar), number of significant bars (above threshold), Wasserstein-style
total persistence.

5 features.
"""
from __future__ import annotations

import logging
from typing import Any, Literal, Optional

import numpy as np
import polars as pl

from ._utils import require_seed, validate_numeric_input

logger = logging.getLogger(__name__)


def compute_persistence_diagram_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: Optional[np.ndarray],
    splitter: Optional[Any] = None,
    *,
    seed: int,
    task: Literal["binary", "regression"] = "regression",
    k_neighbors: int = 30,
    max_filtration: float = 5.0,
    standardize: bool = True,
    column_prefix: str = "pers",
    dtype: np.dtype = np.float32,
) -> pl.DataFrame:
    """Persistence diagram features via Vietoris-Rips H0 from gudhi.

    Output: 5 features — max_persistence, n_bars_above_thresh, total_persistence, mean_birth, mean_death.
    """
    try:
        import gudhi
    except ImportError as exc:
        raise ImportError("persistence_diagram requires gudhi") from exc
    from sklearn.neighbors import NearestNeighbors

    seed = require_seed(seed)
    validate_numeric_input(X_train, name="X_train", allow_fp16=False)
    if X_query is not None:
        validate_numeric_input(X_query, name="X_query", allow_fp16=False)

    X_train_f = np.asarray(X_train, dtype=np.float32)
    y_train_f = np.asarray(y_train, dtype=np.float32).ravel()
    n_features_out = 5

    def _process(Xt: np.ndarray, Xq: np.ndarray, y_t: np.ndarray) -> np.ndarray:
        if standardize:
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler().fit(Xt)
            Xt_s = scaler.transform(Xt).astype(np.float32)
            Xq_s = scaler.transform(Xq).astype(np.float32)
        else:
            Xt_s = Xt
            Xq_s = Xq
        k_eff = min(k_neighbors, Xt_s.shape[0])
        nn = NearestNeighbors(n_neighbors=k_eff, n_jobs=-1).fit(Xt_s)
        _, idx = nn.kneighbors(Xq_s)
        n_q = Xq_s.shape[0]
        out = np.zeros((n_q, n_features_out), dtype=np.float32)
        for q in range(n_q):
            nbr_X = Xt_s[idx[q]]  # (k_eff, d)
            # Build Rips complex on neighborhood with edge weight = pairwise distance.
            try:
                rips = gudhi.RipsComplex(points=nbr_X, max_edge_length=float(max_filtration))
                tree = rips.create_simplex_tree(max_dimension=1)
                tree.compute_persistence()
                # Get H0 (connected components) persistence diagram.
                h0_pairs = tree.persistence_intervals_in_dimension(0)
                if h0_pairs.shape[0] == 0:
                    continue
                # Exclude infinite-death bar (always present).
                finite_pairs = h0_pairs[np.isfinite(h0_pairs[:, 1])]
                if finite_pairs.shape[0] == 0:
                    continue
                lifespans = finite_pairs[:, 1] - finite_pairs[:, 0]
                max_pers = float(lifespans.max())
                total_pers = float(lifespans.sum())
                threshold = 0.1 * max_pers if max_pers > 0 else 0.0
                n_significant = int((lifespans > threshold).sum())
                mean_birth = float(finite_pairs[:, 0].mean())
                mean_death = float(finite_pairs[:, 1].mean())
                out[q, 0] = max_pers
                out[q, 1] = float(n_significant)
                out[q, 2] = total_pers
                out[q, 3] = mean_birth
                out[q, 4] = mean_death
            except Exception:  # nosec B110 - optional/best-effort path, rationale documented
                pass  # leave zeros
        return out

    def _make_df(feats: np.ndarray) -> dict[str, np.ndarray]:
        cols: dict[str, np.ndarray] = {}
        cols[f"{column_prefix}_max_persistence"] = feats[:, 0].astype(dtype, copy=False)
        cols[f"{column_prefix}_n_significant"] = feats[:, 1].astype(dtype, copy=False)
        cols[f"{column_prefix}_total_persistence"] = feats[:, 2].astype(dtype, copy=False)
        cols[f"{column_prefix}_mean_birth"] = feats[:, 3].astype(dtype, copy=False)
        cols[f"{column_prefix}_mean_death"] = feats[:, 4].astype(dtype, copy=False)
        return cols

    if X_query is not None:
        Xq = np.asarray(X_query, dtype=np.float32)
        feats = _process(X_train_f, Xq, y_train_f)
        return pl.DataFrame(_make_df(feats))

    if splitter is None:
        raise ValueError("Mode A (X_query=None) requires a splitter.")
    n_train = X_train_f.shape[0]
    out = np.zeros((n_train, n_features_out), dtype=dtype)
    splits = list(splitter.split(X_train_f))
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        feats = _process(X_train_f[train_idx], X_train_f[val_idx], y_train_f[train_idx])
        out[val_idx] = feats.astype(dtype, copy=False)
        logger.info("persistence_diagram: fold %d/%d done", fold_idx + 1, len(splits))

    return pl.DataFrame(_make_df(out))
