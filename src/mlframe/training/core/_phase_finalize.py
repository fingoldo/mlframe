"""Suite-end finalization: fairness reports, phase summaries, selected features."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ..phases import format_phase_summary
from ._setup_helpers import _finalize_and_save_metadata

if TYPE_CHECKING:
    from ._training_context import TrainingContext

logger = logging.getLogger(__name__)


def _aggregate_discovery_cache_stats(metadata: dict) -> dict:
    """Collapse the per-target ``composite_target_cache`` map into hit / miss totals.

    ``_phase_composite_discovery.py`` writes ``metadata["composite_target_cache"][target_type]
    [target_name] = {"hit": bool, "key": str}`` for every target it tried to read from the disk-
    backed DiscoveryCache. Aggregating here keeps finalize as the single source of truth and avoids
    instrumenting DiscoveryCache itself (locked module, no counters of its own).
    """
    _by_target = metadata.get("composite_target_cache")
    if not isinstance(_by_target, dict):
        return {"hits": 0, "misses": 0}
    _hits = 0
    _misses = 0
    for _by_name in _by_target.values():
        if not isinstance(_by_name, dict):
            continue
        for _entry in _by_name.values():
            if not isinstance(_entry, dict):
                continue
            if _entry.get("hit") is True:
                _hits += 1
            elif _entry.get("hit") is False:
                _misses += 1
    return {"hits": _hits, "misses": _misses}


def _build_cache_stats(ctx) -> dict:
    """Assemble the ``metadata["cache_stats"]`` payload from ctx-side counters + discovery aggregate.

    Layout (per Wave-8 observability spec):
        pipeline_cache    : PipelineCache.n_hits / n_misses merged from each _train_one_target call.
        discovery_cache   : aggregated from metadata["composite_target_cache"]; the locked
                            DiscoveryCache exposes no native counter, but every read site already
                            stamps {hit: bool} into metadata so we tally there.
        fingerprint_cache : ctx._cache_stats["fingerprint_cache"], proxied at the if/else in
                            _train_one_target since the underlying dict has no counters.
        pandas_view_cache : ctx._cache_stats["pandas_view_cache"], proxied at the
                            ``_view_cache.get`` site in _train_one_target.

    Each block also carries ``hit_rate`` = hits / (hits + misses), or ``None`` when the denominator
    is zero (no accesses observed) so consumers can tell "0% hit" from "never used".
    """
    _ctx_stats = getattr(ctx, "_cache_stats", None) or {}
    _pipeline = dict(_ctx_stats.get("pipeline_cache", {"hits": 0, "misses": 0}))
    _fingerprint = dict(_ctx_stats.get("fingerprint_cache", {"hits": 0, "misses": 0}))
    _pandas_view = dict(_ctx_stats.get("pandas_view_cache", {"hits": 0, "misses": 0}))
    _discovery = _aggregate_discovery_cache_stats(ctx.metadata or {})

    def _with_rate(block: dict) -> dict:
        _h = int(block.get("hits", 0))
        _m = int(block.get("misses", 0))
        _total = _h + _m
        return {
            "hits": _h,
            "misses": _m,
            "hit_rate": (_h / _total) if _total > 0 else None,
        }

    return {
        "pipeline_cache": _with_rate(_pipeline),
        "discovery_cache": _with_rate(_discovery),
        "fingerprint_cache": _with_rate(_fingerprint),
        "pandas_view_cache": _with_rate(_pandas_view),
    }


def finalize_suite(ctx: TrainingContext) -> dict:
    """Aggregate fairness reports, save metadata, emit phase/rendering summaries, surface selected features.

    Returns ``ctx.metadata`` (also mutated in-place) so legacy callers keeping a ``metadata = finalize_suite(ctx)`` rebind keep working.
    """
    # Single pass over ctx.models that collects BOTH the per-split fairness reports
    # (lifted from model.metrics) AND the per-entry selected-features list (mirrored to
    # entry.selected_features_). The earlier code walked the same nested dict twice;
    # combining halves Python-level iteration cost for runs with hundreds of models.
    fairness_reports: dict[str, Any] = {}
    _selected_features_per_model: dict = {}
    _selected_features_union: set = set()
    for _ttype, _by_name in (ctx.models or {}).items():
        if not isinstance(_by_name, dict):
            continue
        for _tname, _entries in _by_name.items():
            if not isinstance(_entries, list):
                continue
            for _entry in _entries:
                # Fairness lift: model.metrics[split].fairness_report -> flat metadata key.
                _m_metrics = getattr(_entry, "metrics", None)
                if isinstance(_m_metrics, dict):
                    for _split in ("test", "val", "train"):
                        _split_metrics = _m_metrics.get(_split)
                        if isinstance(_split_metrics, dict) and "fairness_report" in _split_metrics:
                            _key = f"{_ttype}__{_tname}__{getattr(_entry, 'model_name', type(getattr(_entry, 'model', _entry)).__name__)}__{_split}"
                            fairness_reports[_key] = _split_metrics["fairness_report"]
                # Selected-features capture: entry.columns -> metadata + entry.selected_features_.
                _cols = getattr(_entry, "columns", None)
                if _cols is None:
                    continue
                _mn = getattr(_entry, "model_name", None) or ""
                _sf_key = f"{_ttype}/{_tname}/{_mn}" if _mn else f"{_ttype}/{_tname}"
                _selected_features_per_model[_sf_key] = list(_cols)
                _selected_features_union.update(_cols)
                try:
                    _entry.selected_features_ = list(_cols)
                except Exception:
                    pass
    if fairness_reports:
        ctx.metadata["fairness_report"] = fairness_reports

    # Cache observability: aggregate hit / miss / hit_rate per cache backend before the metadata
    # write so the persisted blob has the stats. Failures here must not block the suite from saving
    # its main metadata payload -- log + fall through.
    try:
        ctx.metadata["cache_stats"] = _build_cache_stats(ctx)
    except Exception as _cs_err:
        logger.warning("cache_stats aggregation failed: %s", _cs_err)

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

    # Selected-features surfacing populated during the combined walk above.
    if _selected_features_per_model:
        ctx.metadata["selected_features"] = sorted(_selected_features_union)
        ctx.metadata["selected_features_per_model"] = _selected_features_per_model

    return ctx.metadata
