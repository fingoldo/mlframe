"""Numba specification of the honest supervised null for the plain 'mdlp' binning strategy.

Built once per fit from the same frame and target the production categorisation uses. Columns with 1 < distinct values <= 32 are binned
midpoint-wise without the target (their observed codes are exact under permutation); the rest are supervised and are re-binned by the njit
chain on every permuted target. The target is encoded with the production ``_encode_mdlp_target`` (permutation-invariant), and the
target-free collapse fallbacks are precomputed per column.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_LOW_CARD_CAP = 32  # compute_col_edges default (kwargs 'low_card_cap')
_FALLBACK_NBINS = 5  # collapsed_fallback_nbins default


def _pad_rows(rows: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Stack ragged 1-D rows into a zero-padded 2-D matrix; returns the matrix and each row's length."""
    lengths = np.array([r.size for r in rows], dtype=np.int64)
    width = max(1, int(lengths.max()) if lengths.size else 1)
    # Intentional float64: MDLP cut search at the production binning width (cut points must match supervised_binning)
    out = np.zeros((len(rows), width), dtype=np.float64)
    for j, r in enumerate(rows):
        out[j, : r.size] = r
    return out, lengths


def build_honest_spec(X, cols, y_raw, max_depth: int, method: str = "mdlp") -> dict | None:
    """Spec dict for ``_mdlp_honest_njit.honest_null_columns`` over the first ``len(cols)`` factor columns, or None when a column is not
    usable (non-numeric or non-finite): the caller then keeps the categorise-per-permutation path."""
    from ._adaptive_nbins import _edges_from_quantiles
    from ._supervised_collapse_refine import _fallback_edges
    from .supervised_binning import _encode_mdlp_target

    n = len(y_raw)
    m = len(cols)
    raw = np.zeros((n, m), dtype=np.float64)  # Intentional float64: MDLP cut search at the production binning width (cut points must match supervised_binning)
    is_cont = np.zeros(m, dtype=np.bool_)
    q_fb_rows: list[np.ndarray] = []
    q_rf_rows: list[np.ndarray] = []
    for j, c in enumerate(cols):
        if c not in X.columns:
            # Intentional float64: MDLP cut search at the production binning width (cut points must match supervised_binning)
            q_fb_rows.append(np.empty(0, dtype=np.float64))
            # Intentional float64: MDLP cut search at the production binning width (cut points must match supervised_binning)
            q_rf_rows.append(np.empty(0, dtype=np.float64))
            continue
        if not pd.api.types.is_numeric_dtype(X[c]):
            # Categorical (non-numeric) columns are factorised without the target, so their codes are invariant under permutation of y:
            # they stay fixed (observed codes), exactly like the low-cardinality numeric columns.
            # Intentional float64: MDLP cut search at the production binning width (cut points must match supervised_binning)
            q_fb_rows.append(np.empty(0, dtype=np.float64))
            # Intentional float64: MDLP cut search at the production binning width (cut points must match supervised_binning)
            q_rf_rows.append(np.empty(0, dtype=np.float64))
            continue
        # Intentional float64: MDLP cut search at the production binning width (cut points must match supervised_binning)
        v = np.asarray(X[c].to_numpy(), dtype=np.float64)
        if not np.isfinite(v).all():
            return None
        raw[:, j] = v
        uniq = np.unique(v)
        if uniq.size > _LOW_CARD_CAP:
            is_cont[j] = True
            # Intentional float64: MDLP cut search at the production binning width (cut points must match supervised_binning)
            q_fb_rows.append(np.asarray(_edges_from_quantiles(v, _FALLBACK_NBINS), dtype=np.float64))
            # Intentional float64: MDLP cut search at the production binning width (cut points must match supervised_binning)
            q_rf_rows.append(np.asarray(_fallback_edges(v, "quantile", _FALLBACK_NBINS), dtype=np.float64))
        else:
            # Intentional float64: MDLP cut search at the production binning width (cut points must match supervised_binning)
            q_fb_rows.append(np.empty(0, dtype=np.float64))
            # Intentional float64: MDLP cut search at the production binning width (cut points must match supervised_binning)
            q_rf_rows.append(np.empty(0, dtype=np.float64))
    if method == "oos":
        # Held-out-validated MDLP: the cut search reads the raw (compact) class labels directly.
        _, enc = np.unique(np.asarray(y_raw), return_inverse=True)
        enc = enc.reshape(-1)
        depth_eff = int(max_depth)
    else:
        enc = _encode_mdlp_target(np.asarray(y_raw), 64, False, 16, int(max_depth))
    if method == "oos":
        labels = np.asarray(enc)
    elif isinstance(enc, tuple):
        labels, depth_eff = np.asarray(enc[0]), int(enc[1])
    else:
        labels, depth_eff = np.asarray(enc), int(max_depth)
    labels = np.ascontiguousarray(labels.ravel().astype(np.int64))
    q_fb, q_fb_len = _pad_rows(q_fb_rows)
    q_rf, q_rf_len = _pad_rows(q_rf_rows)
    return {
        "method": method,
        "raw": np.ascontiguousarray(raw),
        "is_cont": is_cont,
        "q_fb": q_fb,
        "q_fb_len": q_fb_len,
        "q_rf": q_rf,
        "q_rf_len": q_rf_len,
        "y_enc": labels,
        "n_classes_total": int(labels.max()) + 1,
        "max_depth": depth_eff,
        "min_split_size": 5,
        "val_min_split_size": 5,
        "oos_tolerance": 0.3,
        "holdout_frac": 0.3,
        "alpha": 0.05,
        "n_permutations": 15,
        "seed": 0,
    }


__all__ = ["build_honest_spec"]
