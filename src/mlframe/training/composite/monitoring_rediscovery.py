"""Drift-aware rediscovery: wire ``CompositeDriftMonitor`` alarms into ``discover_incremental``.

Usage
-----
::

    from mlframe.training.composite.monitoring import CompositeDriftMonitor, check_and_rediscover

    monitor = CompositeDriftMonitor(fitted_estimator)
    outcome = check_and_rediscover(
        monitor, discovery, df_new, target_col, feature_cols,
        y_new=y_new, train_idx=np.arange(len(df_new)),
    )
    # outcome["drift"]      -- did the monitor recommend an update?
    # outcome["decision"]   -- the IncrementalDecision (None when no drift)
    # outcome["refitted"]   -- True when a full re-discovery was run
    # outcome["specs"]      -- the specs to use going forward

Flow: run the monitor on the new batch; when it recommends an update, probe the prior specs with
:func:`mlframe.training.composite.discovery.discover_incremental` (cheap per-spec MI re-score). A
REUSE verdict keeps the prior specs (the drift did not invalidate them); a REDISCOVER verdict runs a
full ``discovery.fit`` on the new frame (when ``refit=True`` and ``train_idx`` is supplied) so the
alarm automatically triggers rediscovery. Alternatively, construct the monitor with an ``on_drift``
callback -- it is invoked with the drift report whenever ``recommend_update`` fires inside
``monitor()``, letting callers plug this helper (or any other reaction) into a streaming loop.

Leakage / RAM discipline: nothing here copies a frame; the monitor pulls narrow columns, the
incremental probe reads a bounded row sample, and the optional refit is the standard train-only fit.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


def check_and_rediscover(
    monitor: Any,
    discovery: Any,
    df_new: Any,
    target_col: str,
    feature_cols: Sequence[str],
    *,
    y_new: Any = None,
    train_idx: Optional[np.ndarray] = None,
    refit: bool = True,
    reference: Any = None,
    y_reference: Any = None,
    **incremental_kwargs: Any,
) -> Dict[str, Any]:
    """Monitor a new batch and automatically rediscover composite specs on drift.

    Parameters
    ----------
    monitor
        A :class:`CompositeDriftMonitor` bound to the deployed estimator.
    discovery
        The prior fitted :class:`CompositeTargetDiscovery` (supplies prior specs + data signature
        for the incremental probe, and performs the full refit on a REDISCOVER verdict).
    df_new, target_col, feature_cols
        The new (appended / streaming) frame and the discovery identifiers.
    y_new
        Optional targets of the new batch; enables the monitor's residual signals and is required
        for a meaningful refit (``df_new`` must contain ``target_col`` for the fit itself).
    train_idx
        Row indices of ``df_new`` to fit on when a full rediscovery is triggered. ``None`` disables
        the automatic refit (the verdict is still returned so the caller can schedule one).
    refit
        Set False to only report the REDISCOVER verdict without running ``discovery.fit``.
    reference, y_reference
        Forwarded to ``monitor.monitor`` for the sketch-less estimator case.
    **incremental_kwargs
        Forwarded to :func:`discover_incremental` (``sample_n`` / ``min_surviving_fraction`` /
        ``eps_mi_gain``).

    Returns
    -------
    dict with keys ``drift`` (bool), ``report`` (the monitor report), ``decision``
    (:class:`IncrementalDecision` | None), ``refitted`` (bool) and ``specs`` (the spec list to use).
    """
    X_new = df_new[list(feature_cols)] if hasattr(df_new, "__getitem__") and hasattr(df_new, "columns") else df_new
    report = monitor.monitor(X_new, y=y_new, reference=reference, y_reference=y_reference)
    prior_specs = list(getattr(discovery, "specs_", []) or [])
    if not report.get("recommend_update"):
        return {"drift": False, "report": report, "decision": None, "refitted": False, "specs": prior_specs}

    from .discovery import discover_incremental

    decision = discover_incremental(discovery, df_new, target_col, feature_cols, **incremental_kwargs)
    if decision.reuse:
        logger.info(
            "[check_and_rediscover] drift alarm raised but the incremental probe kept the prior " "specs (%s); no rediscovery needed.",
            decision.reason,
        )
        return {"drift": True, "report": report, "decision": decision, "refitted": False, "specs": decision.specs or prior_specs}

    if not refit or train_idx is None:
        logger.warning(
            "[check_and_rediscover] drift confirmed (REDISCOVER verdict: %s) but no refit was run "
            "(refit=%s, train_idx=%s); caller must schedule a full discovery.fit.",
            decision.reason, refit, "given" if train_idx is not None else "None",
        )
        return {"drift": True, "report": report, "decision": decision, "refitted": False, "specs": prior_specs}

    logger.warning("[check_and_rediscover] drift confirmed -- running full re-discovery (%s).", decision.reason)
    discovery.fit(df_new, target_col, list(feature_cols), np.asarray(train_idx))
    return {
        "drift": True, "report": report, "decision": decision, "refitted": True,
        "specs": list(getattr(discovery, "specs_", []) or []),
    }
