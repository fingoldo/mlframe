"""Partial-dependence, ICE and interaction-strength diagnostics, split out of ``diagnostics_dispatch``.

The parent re-exports every name here; the parent's shared helpers are imported per call because the parent imports this
module at its own top level.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

import numpy as np

# The parent module's logger name: these lines predate the split, and log filters select them by that name.
logger = logging.getLogger("mlframe.reporting.diagnostics_dispatch")


def render_pdp_ice_diagnostic(
    *,
    model: Any,
    df: Any,
    feature_names: Optional[Sequence[str]],
    feature_importances: Optional[Sequence[float]],
    plot_outputs: str,
    base_path: str,
    metrics_dict: Optional[dict] = None,
    top_features: int = 4,
    sample: int = 2_000,
    grid: int = 20,
    seed: int = 0,
) -> bool:
    """PDP/ICE for the top feature-importance features. Default-ON when a fitted model + feature frame are present.

    The composer subsamples rows to ``sample`` before any predict, so cost is ``grid`` predicts independent of n
    (RAM-safe on 100GB frames -- the carrier frame is never copied, only a row view is sampled inside the composer).
    Skips cheaply when the model cannot predict, the frame is empty, or no features can be ranked.
    """
    from .diagnostics_dispatch import _column_names, _record, _record_path, _save_spec
    charts = metrics_dict.setdefault("charts", {"saved": [], "failed": []}) if isinstance(metrics_dict, dict) else None
    if model is None or df is None or not plot_outputs or not base_path:
        return False
    if not (hasattr(model, "predict") or hasattr(model, "predict_proba")):
        return False
    names = list(feature_names) if feature_names else _column_names(df)
    if not names:
        return False
    # Rank by importance when available, else take the first columns; cap to the top-N legible features.
    if feature_importances is not None and len(feature_importances) == len(names):
        order = np.argsort(np.asarray(feature_importances, dtype=np.float64))[::-1]
        ranked = [names[i] for i in order]
    else:
        ranked = names
    top = ranked[: max(1, int(top_features))]
    interaction = (top[0], top[1]) if len(top) >= 2 else None
    try:
        from mlframe.reporting.charts.pdp_ice import compose_pdp_figure

        spec = compose_pdp_figure(
            model, df, top, grid=grid, sample=sample, interaction_pair=interaction, seed=seed,
        )
        ok = _save_spec(spec, plot_outputs, base_path + "_pdp_ice")
        _record(charts, "pdp_ice", ok)
        if ok:
            _record_path(charts, base_path + "_pdp_ice")
        return ok
    except Exception:
        logger.exception("diagnostics_dispatch: pdp_ice failed; continuing.")
        _record(charts, "pdp_ice", False)
        return False


def render_pdp_2d_diagnostic(
    *,
    model: Any,
    df: Any,
    feature_names: Optional[Sequence[str]],
    feature_importances: Optional[Sequence[float]],
    plot_outputs: str,
    base_path: str,
    metrics_dict: Optional[dict] = None,
    sample: int = 2_000,
    grid: int = 20,
    seed: int = 0,
) -> bool:
    """2-D PDP surface for the top interacting feature pair (opt-in). The composer picks the pair (top SHAP-interaction
    pair when available, else top-2 importances) and caps sample_rows + grid internally, so cost is ``grid`` predicts
    independent of n. Best-effort: any failure is logged and swallowed so the report never aborts.
    """
    from .diagnostics_dispatch import _column_names, _record, _record_path, _save_figure
    charts = metrics_dict.setdefault("charts", {"saved": [], "failed": []}) if isinstance(metrics_dict, dict) else None
    if model is None or df is None or not plot_outputs or not base_path:
        return False
    if not (hasattr(model, "predict") or hasattr(model, "predict_proba")):
        return False
    names = list(feature_names) if feature_names else _column_names(df)
    if not names or len(names) < 2:
        return False
    # Rank by importance when available so the SHAP-less fallback pair is the top-2 most important, mirroring pdp_ice.
    feat_x = feat_y = None
    if feature_importances is not None and len(feature_importances) == len(names):
        order = np.argsort(np.asarray(feature_importances, dtype=np.float64))[::-1]
        feat_x, feat_y = names[int(order[0])], names[int(order[1])]
    try:
        from mlframe.reporting.charts.pdp_2d import compose_pdp_2d_figure

        fig = compose_pdp_2d_figure(model, df, feat_x, feat_y, grid=grid, sample_rows=sample, seed=seed)
        ok = _save_figure(fig, plot_outputs, base_path + "_pdp_2d")
        if ok is None:
            return False  # png not requested; nothing rendered, nothing to record either way
        _record(charts, "pdp_2d", ok)
        if ok:
            _record_path(charts, base_path + "_pdp_2d")
        return ok
    except Exception:
        logger.exception("diagnostics_dispatch: pdp_2d failed; continuing.")
        _record(charts, "pdp_2d", False)
        return False


def _interaction_cost_within_budget(model: Any, df: Any, k: int, grid: int, sample: int, max_seconds: float) -> bool:
    """Project the H-statistic 2-D PDP cost from one timed predict and compare to ``max_seconds``.

    The dominant work is ``C(k,2)`` 2-D PDP surfaces at ``grid^2`` predict-batches of ``sample`` rows each, plus ``k``
    1-D PDPs at ``grid`` batches. One predict on the sample gives the per-batch latency; the projection is model-agnostic
    (a slow deep net self-skips, a fast tree runs).

    A failure inside the probe INFRASTRUCTURE (row count, subsetting) allows the render, because the render is
    best-effort and swallows its own errors. A failure of the model's own ``predict`` does not: a model that cannot
    predict on a clean sample cannot produce a single PDP surface either, so charging it the full budget only buys a
    slow walk to the same failure.

    The probe's OUTPUT is deliberately discarded rather than threaded into the composer. Reusing it would save one
    batch out of ``n_pairs*grid^2 + k*grid`` -- at the defaults (k=8, grid=20) that is 1 of 11,360, under 0.01% of
    the projected work -- which does not justify widening the composer's signature to accept a precomputed batch.
    """
    from .diagnostics_dispatch import _row_count, _subset_rows
    import time as _time

    try:
        n = _row_count(df)
        if n == 0 or k < 2:
            return True
        probe = _subset_rows(df, np.arange(min(sample, n), dtype=np.int64))
        fn = getattr(model, "predict_proba", None) or getattr(model, "predict", None)
        if fn is None:
            return True
    except Exception:
        logger.debug("interaction_strength: cost probe setup failed; allowing render.", exc_info=True)
        return True
    try:
        t0 = _time.perf_counter()
        fn(probe)
        t_batch = _time.perf_counter() - t0
    except Exception:
        logger.info(
            "[diagnostics] interaction_strength: the model raised on a %d-row probe predict, so no PDP surface can "
            "be built -- skipping rather than spending the budget reaching the same failure.",
            min(sample, n),
        )
        return False
    n_pairs = k * (k - 1) // 2
    projected = (n_pairs * grid * grid + k * grid) * t_batch
    return projected <= float(max_seconds)


def render_interaction_strength_diagnostic(
    *,
    model: Any,
    df: Any,
    feature_names: Optional[Sequence[str]],
    feature_importances: Optional[Sequence[float]],
    plot_outputs: str,
    base_path: str,
    metrics_dict: Optional[dict] = None,
    max_features: int = 8,
    sample: int = 2_000,
    grid: int = 20,
    max_seconds: float = 20.0,
    seed: int = 0,
) -> bool:
    """Friedman-Popescu H-statistic heatmap over the top feature-importance features. Reuses the PDP machinery.

    Cost is C(k,2) 2-D PDP surfaces (grid^2 predicts each), so total time scales with the model's predict latency. A
    per-predict TIME probe projects the full cost and skips (logged) when it exceeds ``max_seconds`` -- so a fast/small
    model runs it and a slow one self-skips. Best-effort: any failure is logged and swallowed so the report never aborts.
    """
    from .diagnostics_dispatch import _column_names, _record, _record_path, _save_spec
    charts = metrics_dict.setdefault("charts", {"saved": [], "failed": []}) if isinstance(metrics_dict, dict) else None
    if model is None or df is None or not plot_outputs or not base_path:
        return False
    if not (hasattr(model, "predict") or hasattr(model, "predict_proba")):
        return False
    names = list(feature_names) if feature_names else _column_names(df)
    if not names or len(names) < 2:
        return False
    if feature_importances is not None and len(feature_importances) == len(names):
        order = np.argsort(np.asarray(feature_importances, dtype=np.float64))[::-1]
        ranked = [names[int(i)] for i in order]
    else:
        ranked = names
    top = ranked[: max(2, int(max_features))]
    if max_seconds and max_seconds > 0 and not _interaction_cost_within_budget(model, df, len(top), grid, sample, max_seconds):
        logger.info(
            "[diagnostics] interaction_strength: projected 2-D PDP cost over budget (%d features, grid=%d, "
            "%.0fs cap) -- skipping; raise ReportingConfig.interaction_strength_max_seconds to force.",
            len(top), grid, float(max_seconds),
        )
        return False
    try:
        from mlframe.reporting.charts.interaction_strength import compose_interaction_strength_figure

        spec = compose_interaction_strength_figure(
            model, df, top, max_features=max_features, grid=grid, sample=sample, seed=seed,
        )
        ok = _save_spec(spec, plot_outputs, base_path + "_interaction_strength")
        _record(charts, "interaction_strength", ok)
        if ok:
            _record_path(charts, base_path + "_interaction_strength")
        return ok
    except Exception:
        logger.exception("diagnostics_dispatch: interaction_strength failed; continuing.")
        _record(charts, "interaction_strength", False)
        return False
