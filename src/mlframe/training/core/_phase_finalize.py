"""Suite-end finalization: fairness reports, phase summaries, selected features."""
from __future__ import annotations

import logging
import math
import os
from os.path import join
from typing import TYPE_CHECKING, Any

from pyutilz.strings import slugify

from ..io import save_mlframe_model
from ..phases import format_phase_summary
from ._setup_helpers import _finalize_and_save_metadata

if TYPE_CHECKING:
    from ._training_context import TrainingContext

logger = logging.getLogger(__name__)


# Direction map for ensemble flavour selection. Min-is-better metrics like RMSE / MAE / log_loss / pinball / brier
# default to "minimise"; max-is-better metrics like AUC / NDCG / R2 / accuracy / F1 are detected by substring so
# prefixed variants (``val.integral_error``, ``oof.RMSE_y``) match without an exhaustive enumeration.
_MIN_IS_BETTER_KEYS = ("rmse", "mae", "log_loss", "logloss", "pinball", "brier", "error")
_MAX_IS_BETTER_KEYS = ("auc", "ndcg", "r2", "accuracy", "f1")


def _is_minimise_metric(metric_name: str) -> bool:
    """Return True when ``metric_name`` is a lower-is-better metric. Defaults to True (RMSE-like) when ambiguous; ensembles in mlframe default to selection-by-error."""
    _lower = metric_name.lower()
    if any(k in _lower for k in _MAX_IS_BETTER_KEYS):
        return False
    if any(k in _lower for k in _MIN_IS_BETTER_KEYS):
        return True
    return True


def _pick_best_flavour(ensembles_for_target: dict) -> str | None:
    """Pick the best ensemble flavour from ``{method_name: ens_result}`` by val / oof primary metric.

    ``ens_result.metrics`` is a nested dict ``{split: {metric_name: value}}``. OOF preferred (cross_val_predict
    held-out signal, never burned for ES); val acceptable for back-compat with single-fold suites without
    cross_val_predict. Returns the bare method name (e.g. ``"harm"`` / ``"arithm"`` / ``"rrf"``) or ``None`` when
    nothing scored; falls back to the first available key if all metrics are non-numeric so consumers always get
    a stamp.
    """
    if not ensembles_for_target:
        return None
    _scored: list[tuple[str, float, bool]] = []
    for _method, _ens in ensembles_for_target.items():
        _metrics = getattr(_ens, "metrics", None)
        if not isinstance(_metrics, dict):
            continue
        _split = _metrics.get("oof") or _metrics.get("val") or _metrics.get("test")
        if not isinstance(_split, dict):
            continue
        for _mn, _mv in _split.items():
            try:
                _val = float(_mv)
            except (TypeError, ValueError):
                continue
            if math.isnan(_val):
                continue
            _scored.append((str(_method), _val, _is_minimise_metric(_mn)))
            break
    if not _scored:
        return next(iter(ensembles_for_target.keys()), None)
    _minimise = _scored[0][2]
    if _minimise:
        return min(_scored, key=lambda r: r[1])[0]
    return max(_scored, key=lambda r: r[1])[0]


def _persist_ct_ensemble_entries(ctx: "TrainingContext") -> None:
    """Save every ``_CT_ENSEMBLE__*`` entry from ``ctx.models`` to disk so ``load_mlframe_suite`` rehydrates it.

    Layout mirrors the per-model files produced by ``_setup_model_directories`` -- the predict loader already
    discovers ``*.dump`` recursively, so appending here makes the cross-target ensemble visible without any
    reader-side changes. The literal target-name key ``_CT_ENSEMBLE__<orig>`` is also stamped into
    ``slug_to_original_target_name`` so the load-time round-trip (slugify strips the leading underscore;
    the mapping puts it back) resolves to the in-memory key consumers expect.
    """
    if not getattr(ctx, "data_dir", "") or not getattr(ctx, "models_dir", ""):
        return
    _base = join(
        ctx.data_dir, ctx.models_dir,
        slugify(ctx.target_name), slugify(ctx.model_name),
    )
    try:
        os.makedirs(_base, exist_ok=True)
    except OSError as _exc:
        logger.warning("[_CT_ENSEMBLE persist] could not ensure base dir %s: %s", _base, _exc)
        return
    _n_saved = 0
    for _tt, _by_name in (ctx.models or {}).items():
        if not isinstance(_by_name, dict):
            continue
        for _tname, _entries in _by_name.items():
            if not isinstance(_tname, str) or not _tname.startswith("_CT_ENSEMBLE__"):
                continue
            if not isinstance(_entries, list) or not _entries:
                continue
            _tt_slug = slugify(str(_tt).lower())
            # The outer guard already enforces ``_tname.startswith("_CT_ENSEMBLE__")``, so the value IS
            # the literal directory name we want. (The earlier slice-and-splice was a no-op rewrite of the
            # same string -- the comment about slugify dropping the leading underscore did not match the
            # actual code.)
            _dir_name = _tname
            _target_dir = join(_base, _tt_slug, _dir_name)
            try:
                os.makedirs(_target_dir, exist_ok=True)
            except OSError as _exc:
                logger.warning("[_CT_ENSEMBLE persist] mkdir %s failed: %s", _target_dir, _exc)
                continue
            for _i, _entry in enumerate(_entries):
                _fname = f"CT_ENSEMBLE_{_i}.dump" if len(_entries) > 1 else "CT_ENSEMBLE.dump"
                _fpath = join(_target_dir, _fname)
                # save_mlframe_model uses dill + zstd. The cross-target ensemble wraps each component in
                # _phase_composite_post's local ``_PrePipelinePredictShim`` closure -- dill captures the closure
                # cell so the round-trip works without a top-level definition.
                try:
                    save_mlframe_model(_entry, _fpath, verbose=0)
                    _n_saved += 1
                except Exception as _exc:
                    logger.warning(
                        "[_CT_ENSEMBLE persist] save failed for %s/%s: %s. Predict-from-disk for this target will fall back to component models without the ensemble combiner.",
                        _tt, _tname, _exc,
                    )
            ctx.slug_to_original_target_name[_dir_name] = _tname
            ctx.slug_to_original_target_name[slugify(_dir_name)] = _tname
    if _n_saved and getattr(ctx, "verbose", 0):
        logger.info("[_CT_ENSEMBLE persist] saved %d cross-target ensemble entry/entries to disk.", _n_saved)


def _persist_chosen_ensemble_flavours(ctx: "TrainingContext") -> None:
    """Walk ``ctx.ensembles`` and stamp ``metadata["ensembles_chosen"][target_type][target_name] = flavour``.

    ``ctx.ensembles`` is populated by ``score_ensemble`` (``use_mlframe_ensembles=True``) as
    ``{tt: {tname: {method: ens_result}}}``. We pick the best flavour by OOF / val primary metric per leaf;
    targets without ensembles contribute nothing -- predict falls back to arithmetic mean on those slots.
    """
    _ens_by_target = getattr(ctx, "ensembles", None)
    if not _ens_by_target:
        return
    _chosen: dict[str, dict[str, str]] = {}
    for _tt, _by_name in _ens_by_target.items():
        if not isinstance(_by_name, dict):
            continue
        for _tname, _methods in _by_name.items():
            if not isinstance(_methods, dict):
                continue
            _flavour = _pick_best_flavour(_methods)
            if _flavour:
                _chosen.setdefault(str(_tt), {})[str(_tname)] = _flavour
    if _chosen:
        ctx.metadata["ensembles_chosen"] = _chosen
        if getattr(ctx, "verbose", 0):
            logger.info("[ensembles_chosen] persisted flavours: %s", _chosen)


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

    Layout (observability spec):
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

    # Predict-path parity: stamp the per-target chosen ensemble flavour into metadata BEFORE the metadata
    # write so predict_mlframe_models_suite / predict_from_models can replay the right ``arithm`` / ``harm`` /
    # ``geo`` / ... combine instead of hard-coding np.mean.
    try:
        _persist_chosen_ensemble_flavours(ctx)
    except Exception as _ec_err:
        logger.warning("[ensembles_chosen] persist failed: %s", _ec_err)

    # Predict-path parity: dump the in-memory ``_CT_ENSEMBLE__*`` entries (cross-target ensembles built by
    # _phase_composite_post.py) to disk so the predict loader (load_mlframe_suite / predict_mlframe_models_suite)
    # actually sees them. Pre-fix they lived only in ctx.models and were silently lost across save / load.
    try:
        _persist_ct_ensemble_entries(ctx)
    except Exception as _ct_err:
        logger.warning("[_CT_ENSEMBLE persist] failed: %s", _ct_err)

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
