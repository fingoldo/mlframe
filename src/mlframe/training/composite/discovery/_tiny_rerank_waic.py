"""WAIC tie-break for the tiny-model rerank, carved out of ``_tiny_rerank.py`` to keep that module under the
1000-LOC house limit. ``_apply_waic_tiebreak`` is imported back into ``_tiny_rerank`` and called as a plain function
(it takes ``self`` explicitly), so the rerank call site is unchanged.
"""
from __future__ import annotations

import logging
import math

import numpy as np

from ..transforms import UnknownTransformError, get_transform
from ._rejection_ledger import RejectStage, ledger_append
from ._score import Score, rank_specs
from mlframe.training.composite.transforms._call_gateway import call_transform

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


def _apply_waic_tiebreak(self, order, kept_specs, agg_scores, names, *, y_screen, per_base_cache, rel_tol: float = 0.02, groups=None, time_aware: bool = False):
    """Re-order the RMSE-ascending ``order`` so that, within each relative-RMSE noise band, transforms are ranked by
    WAIC (higher = better out-of-fold generalisation). Only bands where every member has a valid WAIC are re-ordered;
    everything else keeps its RMSE+name position. Stores the WAIC of every scored spec on ``self._tiny_rerank_waic_scores``:
    only members of multi-spec bands that reach the top-m window are scored, since no other score can change the result.
    Returns a new integer order array (the input ``order`` unchanged when no usable WAIC was produced)."""
    from ._eval_waic import compute_transform_waic

    n_folds = int(getattr(self.config, "transform_waic_n_folds", 4) or 4)
    # `self.random_seed` does not exist on CompositeTargetDiscovery -- the configured seed lives at
    # `self.config.random_state` (which every other call site in the sibling _tiny_rerank.py module
    # reads correctly). The old `getattr(self, "random_seed", 0)` always fell through to the
    # default, silently pinning this K-fold split to seed 0 regardless of the caller's random_state.
    rs = int(getattr(self.config, "random_state", 42) or 0)
    yb = np.asarray(y_screen, dtype=np.float64).ravel()
    # One float64 valid-row matrix per base, shared by every spec on it: the per-spec cast copied the same (n, F) block
    # each time, and handing every spec the same array object lets the WAIC folds reuse one binned dataset per fold. The
    # specs are scored grouped by base and only the latest two bases' matrices are kept: one per base held them all.
    from ._per_base_x import BoundedMemo, base_ordered

    _base_x = BoundedMemo(capacity=2)

    def _valid_x(base_col, base_screen, x_mat):
        """``(valid mask, float64 X on the valid rows)`` for this base, built once while it is in use."""
        def build():
            bb = np.asarray(base_screen, dtype=np.float64).ravel()
            valid = np.isfinite(yb) & np.isfinite(bb)
            return valid, bb, np.ascontiguousarray(np.asarray(x_mat, dtype=np.float64)[valid])

        return _base_x.get_or_build(base_col, build)

    def _waic_for(i: int):
        """WAIC of spec ``i`` on the screen sample, or None when it cannot be scored."""
        spec = kept_specs[i]
        cached = per_base_cache.get(getattr(spec, "base_column", None))
        if cached is None:
            return None
        base_screen, x_mat = cached
        try:
            transform = get_transform(spec.transform_name)
        except Exception as e:  # nosec B112 - swallow converted to debug-log, non-fatal by design
            logger.debug("suppressed: %s", e)
            return None
        valid, bb, xv_base = _valid_x(getattr(spec, "base_column", None), base_screen, x_mat)
        if int(valid.sum()) < 2 * n_folds:
            return None
        try:
            target = np.asarray(call_transform(transform, "forward", yb[valid], bb[valid], spec.fitted_params,
                                               groups=None if groups is None else np.asarray(groups)[valid]), dtype=np.float64).ravel()
        except Exception as e:  # nosec B112 - swallow converted to debug-log, non-fatal by design
            logger.debug("suppressed: %s", e)
            return None
        fin = np.isfinite(target)
        if int(fin.sum()) < 2 * n_folds or xv_base.shape[0] != target.shape[0]:
            return None
        g = None if groups is None else np.asarray(groups)[valid][fin]  # WAIC folds follow the rerank's groups and time order
        all_fin = bool(fin.all())
        score = compute_transform_waic(target if all_fin else target[fin], xv_base if all_fin else xv_base[fin], n_folds=n_folds,
                                       random_state=rs, groups=g, time_aware=time_aware)
        if getattr(score, "valid", False) and math.isfinite(score.waic):
            return float(score.waic)
        return None

    # Specs are independent and each runs 4 single-threaded tiny-GBM folds (LightGBM releases the GIL), so they go on
    # threads: the serial loop was 73s of a 139s discovery on a 300k-row tie-heavy target.
    from joblib import Parallel, delayed
    from pyutilz.parallel import cpu_count_physical

    # The bands depend only on the RMSE order, so find them first and score WAIC only where it can change the result:
    # a singleton band has nothing to re-order, and a band starting past the top-m cut is trimmed away whatever its order.
    # Scoring every spec ran 4 folds each for scores that were then discarded.
    bands = rmse_bands([int(i) for i in order], agg_scores, rel_tol)
    top_m = max(1, int(getattr(self.config, "top_m_after_tiny", len(order)) or len(order)))
    # A WAIC over T compares densities in T units, so it can only order transforms whose T is in y units: the additive ones
    # (T = y - g(base)). Between a residual and a compressive transform (log, cbrt, a ratio) it rewarded the smaller T scale
    # by ~log(scale_y / scale_T) nats per row, which is not generalisation; such bands keep their y-scale RMSE order.
    def _same_scale(band) -> bool:
        """True when every spec in the band keeps T in y units, which is what makes a WAIC comparison valid."""
        return all(_additive_in_t(kept_specs[b]) for b in band)

    to_score: list[int] = []
    pos = 0
    for band in bands:
        if len(band) > 1 and pos < top_m and _same_scale(band):
            to_score.extend(band)
        pos += len(band)

    to_score = base_ordered(to_score, lambda i: getattr(kept_specs[i], "base_column", None))
    n_jobs = max(1, min(len(to_score), cpu_count_physical()))
    if n_jobs > 1:
        results = Parallel(n_jobs=n_jobs, backend="threading", prefer="threads")(delayed(_waic_for)(i) for i in to_score)
    else:
        results = [_waic_for(i) for i in to_score]
    waic: dict[int, float] = {i: v for i, v in zip(to_score, results) if v is not None}
    self._tiny_rerank_waic_scores = {kept_specs[i].name: v for i, v in waic.items()}
    if not waic:
        return order

    new_order: list[int] = []
    for band in bands:
        if len(band) > 1 and all(b in waic for b in band):
            band = rank_specs(band, lambda b: Score(waic[b], "waic_nats", "screen_oof", "waic", kept_specs[b].transform_name), descending=True,
                              name=lambda b: names[b], transform=lambda b: kept_specs[b].transform_name)
        new_order.extend(band)
    return np.asarray(new_order, dtype=int)


def _additive_in_t(spec) -> bool:
    """True when the spec's transform is additive in T (its T is in y units)."""
    try:
        return bool(getattr(get_transform(spec.transform_name), "additive_in_t", False))
    except UnknownTransformError:  # an unregistered transform is simply not WAIC-comparable; anything else is a real bug
        return False


def rmse_bands(idx: list, agg_scores, rel_tol: float) -> list:
    """Split the RMSE-ascending ``idx`` into consecutive noise bands: members within ``rel_tol`` of the band's first score."""
    bands: list = []
    j = 0
    while j < len(idx):
        s0 = agg_scores[idx[j]]
        k = j + 1
        while k < len(idx) and math.isfinite(s0) and math.isfinite(agg_scores[idx[k]]) and (agg_scores[idx[k]] - s0) <= abs(s0) * rel_tol:
            k += 1
        bands.append(idx[j:k])
        j = k
    return bands
