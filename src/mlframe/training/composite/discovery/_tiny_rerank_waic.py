"""WAIC tie-break for the tiny-model rerank, carved out of ``_tiny_rerank.py`` to keep that module under the
1000-LOC house limit. ``_apply_waic_tiebreak`` is imported back into ``_tiny_rerank`` and called as a plain function
(it takes ``self`` explicitly), so the rerank call site is unchanged.
"""
from __future__ import annotations

import math

import numpy as np

from ..transforms import get_transform


def _apply_waic_tiebreak(self, order, kept_specs, agg_scores, names, *, y_screen, per_base_cache, rel_tol: float = 0.02):
    """Re-order the RMSE-ascending ``order`` so that, within each relative-RMSE noise band, transforms are ranked by
    WAIC (higher = better out-of-fold generalisation). Only bands where every member has a valid WAIC are re-ordered;
    everything else keeps its RMSE+name position. Stores the per-spec WAIC on ``self._tiny_rerank_waic_scores``.
    Returns a new integer order array (the input ``order`` unchanged when no usable WAIC was produced)."""
    from ._eval_waic import compute_transform_waic

    n_folds = int(getattr(self.config, "transform_waic_n_folds", 4) or 4)
    rs = int(getattr(self, "random_seed", 0) or 0)
    yb = np.asarray(y_screen, dtype=np.float64).ravel()
    waic: dict[int, float] = {}
    for i, spec in enumerate(kept_specs):
        cached = per_base_cache.get(getattr(spec, "base_column", None))
        if cached is None:
            continue
        base_screen, x_mat = cached
        try:
            transform = get_transform(spec.transform_name)
        except Exception:
            continue
        bb = np.asarray(base_screen, dtype=np.float64).ravel()
        valid = np.isfinite(yb) & np.isfinite(bb)
        if int(valid.sum()) < 2 * n_folds:
            continue
        try:
            target = np.asarray(transform.forward(yb[valid], bb[valid], spec.fitted_params), dtype=np.float64).ravel()
        except Exception:
            continue
        xv = np.asarray(x_mat, dtype=np.float64)[valid]
        fin = np.isfinite(target)
        if int(fin.sum()) < 2 * n_folds or xv.shape[0] != target.shape[0]:
            continue
        score = compute_transform_waic(target[fin], xv[fin], n_folds=n_folds, random_state=rs)
        if getattr(score, "valid", False) and math.isfinite(score.waic):
            waic[i] = float(score.waic)
    self._tiny_rerank_waic_scores = {kept_specs[i].name: v for i, v in waic.items()}
    if not waic:
        return order

    idx = [int(i) for i in order]
    new_order: list[int] = []
    j = 0
    while j < len(idx):
        s0 = agg_scores[idx[j]]
        k = j + 1
        while (
            k < len(idx)
            and math.isfinite(s0)
            and math.isfinite(agg_scores[idx[k]])
            and (agg_scores[idx[k]] - s0) <= abs(s0) * rel_tol
        ):
            k += 1
        band = idx[j:k]
        if len(band) > 1 and all(b in waic for b in band):
            band = sorted(band, key=lambda b: (-waic[b], names[b]))
        new_order.extend(band)
        j = k
    return np.asarray(new_order, dtype=int)
