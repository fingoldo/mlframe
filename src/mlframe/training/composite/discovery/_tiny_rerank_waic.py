"""WAIC tie-break for the tiny-model rerank, carved out of ``_tiny_rerank.py`` to keep that module under the
1000-LOC house limit. ``_apply_waic_tiebreak`` is imported back into ``_tiny_rerank`` and called as a plain function
(it takes ``self`` explicitly), so the rerank call site is unchanged.
"""
from __future__ import annotations

import logging
import math

import numpy as np

from ..transforms import get_transform
from ._rejection_ledger import RejectStage, ledger_append

logger = logging.getLogger(__name__)


def apply_honest_oof_floor(self, kept_specs, agg_scores, honest_oof, honest_oof_baseline):
    """Enforce the honest-OOF floor as a REJECTION, not just a rank key: drop specs whose MEASURED honest reconstruction
    cannot beat ``min(raw-y, AR-lag)`` within tolerance. Without this honest-OOF only reorders (the raw-baseline gate is
    off by default) so a spec worse than the lag_predict failsafe we deploy anyway is still carried into the ensemble
    (prod: 13.30 ensemble vs 11.58 lag floor). Specs absent from ``honest_oof`` (degenerate measurement) keep their CV
    rank. Returns the (possibly filtered) ``(kept_specs, agg_scores)``; also refreshes ``self._tiny_rerank_scores``.
    """
    if not (math.isfinite(honest_oof_baseline) and bool(getattr(self.config, "honest_oof_floor_reject_enabled", True))):
        return kept_specs, agg_scores
    floor_tol = float(getattr(self.config, "honest_oof_selection_tolerance", 1.05))
    floor_thr = honest_oof_baseline * floor_tol
    kept_after, agg_after, dropped = [], [], []
    for i, spec in enumerate(kept_specs):
        hv = honest_oof.get(spec.name)
        if hv is not None and math.isfinite(hv) and hv >= floor_thr:
            dropped.append((spec.name, float(hv)))
            ledger_append(
                self, spec_name=spec.name, stage=RejectStage.HONEST_OOF_FLOOR,
                reason=f"honest-OOF reconstruction {hv:.4g} >= floor {floor_thr:.4g} "
                       f"(min(raw,lag)={honest_oof_baseline:.4g} * {floor_tol})",
                base_column=getattr(spec, "base_column", ""),
                transform_name=getattr(spec, "transform_name", ""),
                numbers={"honest_oof_rmse": float(hv), "floor": float(honest_oof_baseline), "threshold": float(floor_thr)},
            )
            continue
        kept_after.append(spec)
        agg_after.append(agg_scores[i])
    if dropped:
        logger.info(
            "[CompositeTargetDiscovery.honest_oof_select] floor gate rejected %d/%d spec(s) that lose to "
            "min(raw,lag)=%.4g (tol=%.2f): %s",
            len(dropped), len(kept_specs), honest_oof_baseline, floor_tol,
            ", ".join(f"{n}(RMSE={v:.4g})" for n, v in dropped[:5]),
        )
        self._tiny_rerank_scores = {kept_after[i].name: float(agg_after[i]) for i in range(len(kept_after))}
        return kept_after, agg_after
    return kept_specs, agg_scores


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
        except Exception as e:  # nosec B112 - swallow converted to debug-log, non-fatal by design
            logger.debug("suppressed in _tiny_rerank_waic.py:75: %s", e)
            continue
        bb = np.asarray(base_screen, dtype=np.float64).ravel()
        valid = np.isfinite(yb) & np.isfinite(bb)
        if int(valid.sum()) < 2 * n_folds:
            continue
        try:
            target = np.asarray(transform.forward(yb[valid], bb[valid], spec.fitted_params), dtype=np.float64).ravel()
        except Exception as e:  # nosec B112 - swallow converted to debug-log, non-fatal by design
            logger.debug("suppressed in _tiny_rerank_waic.py:83: %s", e)
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
        while k < len(idx) and math.isfinite(s0) and math.isfinite(agg_scores[idx[k]]) and (agg_scores[idx[k]] - s0) <= abs(s0) * rel_tol:
            k += 1
        band = idx[j:k]
        if len(band) > 1 and all(b in waic for b in band):
            band = sorted(band, key=lambda b: (-waic[b], names[b]))
        new_order.extend(band)
        j = k
    return np.asarray(new_order, dtype=int)
