"""
Suite-end finalization: fairness reports, phase summaries, selected features.

Runs after all model training completes, before composite post-processing.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from ..phases import format_phase_summary

logger = logging.getLogger(__name__)


def finalize_suite(
    *,
    models: Dict,
    metadata: Dict,
    outlier_detector,
    outlier_detection_result,
    trainset_features_stats,
    data_dir: str,
    models_dir: str,
    target_name: str,
    model_name: str,
    slug_to_original_target_type: Dict,
    slug_to_original_target_name: Dict,
    _finalize_and_save_metadata,
    verbose: bool,
) -> Dict:
    """Aggregate fairness reports, save metadata, emit phase/rendering summaries,
    and surface selected features on metadata.

    Returns updated metadata.
    """
    # Aggregate per-model fairness_report into metadata so callers can access it
    # without re-walking models dict. Trainer stores fairness_report in model.metrics[split].
    fairness_reports: Dict[str, Any] = {}
    for _ttype, _targets in models.items():
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
        metadata["fairness_report"] = fairness_reports

    # Save metadata again with slug-to-original name mappings for load_mlframe_suite
    _finalize_and_save_metadata(
        metadata=metadata,
        outlier_detector=outlier_detector,
        outlier_detection_result=outlier_detection_result,
        trainset_features_stats=trainset_features_stats,
        data_dir=data_dir,
        models_dir=models_dir,
        target_name=target_name,
        model_name=model_name,
        verbose=0,  # silent to avoid duplicate log messages
        slug_to_original_target_type=slug_to_original_target_type,
        slug_to_original_target_name=slug_to_original_target_name,
    )

    if verbose:
        logger.info("[phases] Top phases by wall-clock time:\n%s", format_phase_summary())

        # Top-N wall-share so the reader immediately sees where time went vs total.
        # Percentages computed against the longest-running phase (suite root).
        # Helps spot plot/render-bound vs train-bound runs at a glance.
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

        # Kaleido oneshot fallback summary: if any plotly PNG/SVG/PDF saves took
        # the slow oneshot path, surface cumulative cost so reader knows ROI of
        # upgrading kaleido. Per-call warning was suppressed (idempotent).
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

    # Surface selected-features list per trained model so callers can introspect
    # feature-selection outputs (MRMR / RFECV) without walking the nested namespace.
    # Also exposes flat union for "did any model keep the informative feature?" checks.
    _selected_features_per_model: dict = {}
    _selected_features_union: set = set()
    for _tt, _by_name in (models or {}).items():
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
        metadata["selected_features"] = sorted(_selected_features_union)
        metadata["selected_features_per_model"] = _selected_features_per_model

    return metadata
