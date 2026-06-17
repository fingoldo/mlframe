"""Distribution-driven composite-estimator recommendation (Workstream E3).

The target-distribution analyzer already detects pathologies (heavy-tail, multi-modal, skew, ...). This
maps that verdict to the composite estimator best suited to it, so the suite can recommend (and, in a
follow-up, auto-dispatch) the right specialised estimator instead of always using the default regressor.

Heavy-tail / skew -> TailCompositeEstimator (GPD tail). Multi-modal -> CompositeDistributionEstimator
(full predictive distribution / CRPS). Returns a recommendation dict or None when the default fits.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Optional


def recommend_composite_estimator(pathologies: Sequence[str]) -> Optional[dict[str, Any]]:
    """Map analyzer pathology strings to a recommended composite estimator (or None for the default).

    ``pathologies`` are the prefixed strings from the target-distribution analyzer (e.g.
    ``"heavy_tail(excess_kurt=12.3)"``, ``"multi_modal_target(peaks=3, ...)"``). Matching is by prefix.
    Returns ``{"estimator": <class name>, "module": <import path>, "reason": <pathology>}`` for the FIRST
    matching pathology by priority (heavy-tail/skew before multi-modal), else None.
    """
    pats = [str(p) for p in (pathologies or [])]

    def _first(prefixes):
        for p in pats:
            if any(p.startswith(pre) for pre in prefixes):
                return p
        return None

    heavy = _first(("heavy_tail", "skewed_target"))
    if heavy is not None:
        return {
            "estimator": "TailCompositeEstimator",
            "module": "mlframe.training.composite.extremes",
            "reason": heavy,
        }
    multimodal = _first(("multi_modal_target", "multimodal"))
    if multimodal is not None:
        return {
            "estimator": "CompositeDistributionEstimator",
            "module": "mlframe.training.composite.distributional",
            "reason": multimodal,
        }
    return None


def instantiate_recommended_estimator(recommendation: Optional[dict[str, Any]], **kwargs: Any):
    """Construct the recommended composite estimator from a ``recommend_composite_estimator`` dict (or None).

    Imports ``recommendation["module"].recommendation["estimator"]`` and instantiates it with ``kwargs``.
    Returns ``None`` when ``recommendation`` is None so the caller keeps its default estimator.

    IMPORTANT (proven empirically): ``TailCompositeEstimator`` and ``CompositeDistributionEstimator`` are
    NOT generic drop-in regressors -- they wrap a ``CompositeTargetEstimator`` and REQUIRE a ``base_column``
    (or ``base_columns``) kwarg; calling ``.fit(X, y)`` without one raises. So a live auto-swap to them is
    inseparable from the composite-discovery base-selection flow (which picks the dominant base feature) --
    it is a dedicated composite-integration step, not a strategy-layer estimator swap. Pass ``base_column=``
    here when you have one. (``BaggedCompositeEstimator`` is the only base-free drop-in, but no analyzer
    pathology currently recommends it.) The recommendation itself remains advisory until that integration.
    """
    if not recommendation:
        return None
    import importlib

    mod = importlib.import_module(recommendation["module"])
    cls = getattr(mod, recommendation["estimator"])
    return cls(**kwargs)
