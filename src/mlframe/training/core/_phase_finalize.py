"""Suite-end finalization: fairness reports, phase summaries, selected features."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ..phases import format_phase_summary
from ._setup_helpers import _finalize_and_save_metadata

if TYPE_CHECKING:
    from ._training_context import TrainingContext

logger = logging.getLogger(__name__)


def finalize_suite(ctx: TrainingContext) -> dict:
    """Aggregate fairness reports, save metadata, emit phase/rendering summaries, surface selected features.

    Returns ``ctx.metadata`` (also mutated in-place) so legacy callers keeping a ``metadata = finalize_suite(ctx)`` rebind keep working.
    """
    # Trainer stores fairness_report in model.metrics[split]; lift to top-level so callers don't re-walk the models dict.
    fairness_reports: dict[str, Any] = {}
    for _ttype, _targets in ctx.models.items():
        for _tname, _model_list in _targets.items():
            for _m in _model_list:
                _m_metrics = getattr(_m, "metrics", None)
                if not isinstance(_m_metrics, dict):
                    continue
                for _split in ("test", "val", "train"):
                    _split_metrics = _m_metrics.get(_split)
                    if isinstance(_split_metrics, dict) and "fairness_report" in _split_metrics:
                        _key = f"{_ttype}__{_tname}__{getattr(_m, 'model_name', type(getattr(_m, 'model', _m)).__name__)}__{_split}"
                        fairness_reports[_key] = _split_metrics["fairness_report"]
    if fairness_reports:
        ctx.metadata["fairness_report"] = fairness_reports

    # ``verbose=0`` silences the duplicate "Saved metadata to ..." log line; main.py already saved partway.
    _finalize_and_save_metadata(ctx, verbose=0)

    if ctx.verbose:
        logger.info("[phases] Top phases by wall-clock time:\n%s", format_phase_summary())

        # Wall-share percentages computed against the longest-running phase (suite root).
        try:
            from ..phases import phase_snapshot
            _snap = phase_snapshot()
            if _snap:
                _root_wall = _snap[0][1] if _snap else 0.0
                if _root_wall > 0:
                    _share_str = ", ".join(
                        f"{p}={tot/_root_wall*100:.1f}%"
                        for p, tot, _ in _snap[:8]
                    )
                    logger.info("[wall-share] top: %s", _share_str)
        except Exception:
            pass

        # Surface cumulative kaleido oneshot cost; per-call warning is suppressed (idempotent).
        try:
            from mlframe.reporting.renderers.plotly import (
                get_kaleido_oneshot_stats, reset_kaleido_oneshot_stats,
            )
            _kal_n, _kal_wall = get_kaleido_oneshot_stats()
            if _kal_n > 0:
                logger.info(
                    "[plotly-render] kaleido oneshot fallback fired %d times "
                    "(cumulative %.1fs wall, %.0fms/call avg). Persistent "
                    "sync-server path would be ~10-100x faster -- upgrade "
                    "kaleido (>=1.x ships ``start_sync_server``) to enable.",
                    _kal_n, _kal_wall, (_kal_wall / _kal_n) * 1000,
                )
            reset_kaleido_oneshot_stats()
        except Exception:
            pass

    # Surface selected-features per model (and a flat union) on metadata for caller introspection.
    _selected_features_per_model: dict = {}
    _selected_features_union: set = set()
    for _tt, _by_name in (ctx.models or {}).items():
        if not isinstance(_by_name, dict):
            continue
        for _tn, _entries in _by_name.items():
            if not isinstance(_entries, list):
                continue
            for _entry in _entries:
                _cols = getattr(_entry, "columns", None)
                _mn = getattr(_entry, "model_name", None) or ""
                if _cols is None:
                    continue
                _key = f"{_tt}/{_tn}/{_mn}" if _mn else f"{_tt}/{_tn}"
                _selected_features_per_model[_key] = list(_cols)
                _selected_features_union.update(_cols)
                try:
                    _entry.selected_features_ = list(_cols)
                except Exception:
                    pass
    if _selected_features_per_model:
        ctx.metadata["selected_features"] = sorted(_selected_features_union)
        ctx.metadata["selected_features_per_model"] = _selected_features_per_model

    return ctx.metadata
