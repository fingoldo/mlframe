"""Post-discovery pruning of redundant composite specs, run before any per-spec training.

Lives beside ``_phase_composite_discovery`` (which is near the module-size limit); called once per original target after
the full T columns are materialised.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ..composite.discovery._t_equivalence import DEFAULT_R2_TOL, find_equivalent_composite_specs, t_train_envelope

logger = logging.getLogger("mlframe.training.core._phase_composite_discovery")


def _spec_name(s: Any):
    """Name of a composite spec given either as a dict or as an object with a ``name`` attribute."""
    return s.get("name") if isinstance(s, dict) else getattr(s, "name", None)


def prune_equivalent_composite_specs(
    *,
    specs: list,
    t_by_name: dict[str, np.ndarray],
    y_full: np.ndarray,
    train_idx: Any,
    pending: list[dict],
    metadata: dict,
    target_type: str,
    target_name: str,
    r2_tol: float = DEFAULT_R2_TOL,
) -> dict[str, str]:
    """Stamp each spec's T-train envelope, then drop specs equivalent to raw y / to a better-ranked spec.

    Mutates ``pending`` (the global to-train buffer), ``metadata["composite_target_specs"]`` and ``_failures`` in place so a
    dropped spec is neither trained nor shown in the verdict. Returns ``{dropped_name: reason}``.
    """
    y_full = np.asarray(y_full, dtype=np.float64).reshape(-1)
    tr = np.arange(y_full.size) if train_idx is None else np.asarray(train_idx)
    exported = (metadata.get("composite_target_specs", {}).get(str(target_type), {}) or {}).get(target_name) or []
    exported_by_name = {_spec_name(s): s for s in exported}
    t_train_by_name: dict[str, np.ndarray] = {}
    for s in specs:
        n = _spec_name(s)
        if n not in t_by_name:
            continue
        t_tr = np.asarray(t_by_name[n], dtype=np.float64)[tr]
        t_train_by_name[n] = t_tr
        env = t_train_envelope(t_tr)
        if env is None:
            continue
        # The end-of-target wrapper has no base column to rebuild T from; hand it the exact train envelope.
        for holder in (getattr(s, "fitted_params", None), (exported_by_name.get(n) or {}).get("fitted_params")):
            if isinstance(holder, dict):
                holder["t_train_envelope_low"], holder["t_train_envelope_high"] = env
    # The budget's order, tier by tier: raw gains mix RMSE fractions and MI nats when some specs of the target were
    # measured on the honest holdout and others fell back to MI. Specs the budget does not list follow, in their order.
    from ._phase_composite_discovery_gates import rank_pending_composites

    ranked = [p["name"] for p in rank_pending_composites([p for p in pending if str(p.get("tt")) == str(target_type)])]
    priority = [n for n in ranked if n in t_train_by_name] + [n for n in t_train_by_name if n not in set(ranked)]
    drops = find_equivalent_composite_specs(y_full[tr], t_train_by_name, priority, r2_tol=r2_tol)
    if not drops:
        return drops
    for n, why in drops.items():
        logger.info("[CompositeTargetDiscovery] dropped redundant composite '%s' before training: %s.", n, why)
    pending[:] = [p for p in pending if not (str(p.get("tt")) == str(target_type) and p.get("name") in drops)]
    if exported:
        exported[:] = [s for s in exported if _spec_name(s) not in drops]
    fails = metadata.setdefault("composite_target_failures", {}).setdefault(str(target_type), {}).setdefault(target_name, [])
    fails.extend({"name": n, "kept": False, "rejected": True, "reason": f"redundant: {why}"} for n, why in drops.items())
    metadata.setdefault("composite_target_equivalence_drops", {}).setdefault(str(target_type), {})[target_name] = dict(drops)
    return drops


def forget_untrained_specs(metadata: dict, dropped: list[dict], reason: str) -> None:
    """Remove the exported specs of ``dropped`` pending entries from ``metadata['composite_target_specs']`` and record each in
    ``composite_target_failures`` with ``reason``, so saved metadata lists only the specs that were trained.

    Every reader of the spec list (the CT-ensemble builder, the suite-end summary, composite-feature stacking, the
    precomputed-bundle replay) treats a listed spec as shipped; a spec dropped by the global cap or the gain floor was not.
    """
    for p in dropped:
        tt, target, name = str(p.get("tt")), p.get("target"), p.get("name")
        if target is None:
            continue
        specs = metadata.get("composite_target_specs", {}).get(tt, {}).get(target)
        if isinstance(specs, list):
            specs[:] = [s for s in specs if _spec_name(s) != name]
        fails = metadata.setdefault("composite_target_failures", {}).setdefault(tt, {}).setdefault(target, [])
        fails.append({"name": name, "kept": False, "rejected": True, "reason": reason})
