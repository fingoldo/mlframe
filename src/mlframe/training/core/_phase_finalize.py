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
    """Pick the best ensemble flavour from ``{method_name: ens_result}``.

    C-P0-5 / Arch-2: this function used to iterate ``_split.items()`` and break on the FIRST
    numeric metric (whichever key dict iteration produced first), then declare lower/higher-better
    by that single metric's name. That made the winner depend on metric registration order
    (``integral_error`` vs ``rmse`` vs ``auc``). Now it delegates to the canonical
    ``_choose_ensemble_flavour`` from ``_phase_train_one_target`` so finalize and the per-target
    stamper use the SAME canonical ranking (oof.integral_error -> oof.rmse -> val.integral_error
    -> val.rmse). Test split is deliberately not in the ranking.
    """
    if not ensembles_for_target:
        return None
    # Leaf import: ``_ensemble_chooser`` is the canonical home for the chooser; no cycle to dodge.
    from ._ensemble_chooser import _choose_ensemble_flavour as _canonical_chooser
    return _canonical_chooser(ensembles_for_target)


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
            # Wave 46 (2026-05-20): defence-in-depth: a key like "_CT_ENSEMBLE__../../evil" would
            # bypass the prefix gate and traverse out of _base; slugify keeps the prefix as a literal
            # marker while neutralising any path separators / parent-dir refs inside the rest.
            _dir_name = "_CT_ENSEMBLE__" + slugify(_tname[len("_CT_ENSEMBLE__"):])
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
                    # lean=True strips train/val/test preds/probs/targets and
                    # other forensics-only fields before dill.dump. The CT_ENSEMBLE
                    # bundle is consumed exclusively by ``predict_from_models`` (the
                    # composite cross-target combiner), which never reads
                    # ``train_preds`` etc. -- those attributes only exist as a
                    # side-product of the training loop. iter-344 cb+xgb regression
                    # save profile showed dill descending recursively through
                    # ~900K leaf-numpy arrays per per-target CT_ENSEMBLE bundle on
                    # 1M-row runs (181s wall for 63MB on disk = 0.35 MB/s).
                    # lean=True drops the dill descent to the inference path only,
                    # yielding ~30x save speedup (post-iter-344 commit 9295e67
                    # measured on the harness flip; same arithmetic applies here
                    # because the saved object graph is identical).
                    save_mlframe_model(_entry, _fpath, verbose=0, lean=True)
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
    """Walk ``ctx.ensembles`` and ADD missing chosen-flavour entries to ``metadata["ensembles_chosen"]["simple"]``.

    Arch-2 / Low-10: the per-target stamper in ``_phase_train_one_target`` already writes
    ``metadata["ensembles_chosen"]["simple"][target_type][target_name] = flavour`` immediately after
    ``score_ensemble`` returns. Finalize previously REBUILT the whole dict, silently overwriting
    each per-target decision. Now finalize only fills in slots the per-target stamper did NOT write
    (e.g. composite cross-target ensembles, or targets where the stamper threw and was caught),
    using the same canonical chooser as the per-target stamper.

    Arch-3: ``ensembles_chosen`` is sub-keyed per ensemble family. Backfill of simple per-target
    ensembles always lands under the ``"simple"`` bucket; cross-target ensembles are stamped by
    ``_phase_composite_post`` directly under the ``"cross_target"`` bucket and are not touched here.

    ``ctx.ensembles`` shape: ``{tt: {tname: {method: ens_result}}}``. Targets without ensembles
    contribute nothing -- predict falls back to arithmetic mean on those slots.
    """
    _ens_by_target = getattr(ctx, "ensembles", None)
    if not _ens_by_target:
        return
    _root = ctx.metadata.get("ensembles_chosen") if isinstance(ctx.metadata, dict) else None
    if not isinstance(_root, dict):
        _root = {}
    _simple = _root.setdefault("simple", {})
    if not isinstance(_simple, dict):
        # Sentinel guard: a foreign caller may have stamped a non-dict; reset to empty dict.
        _simple = {}
        _root["simple"] = _simple
    _added: list[tuple[str, str, str]] = []
    for _tt, _by_name in _ens_by_target.items():
        if not isinstance(_by_name, dict):
            continue
        _tt_str = str(_tt)
        _slot = _simple.setdefault(_tt_str, {})
        if not isinstance(_slot, dict):
            continue
        for _tname, _methods in _by_name.items():
            _tname_str = str(_tname)
            if _tname_str in _slot or _tname in _slot:
                # Per-target stamper already chose; don't overwrite (Arch-2 / Low-10).
                continue
            if not isinstance(_methods, dict):
                continue
            _flavour = _pick_best_flavour(_methods)
            if _flavour:
                _slot[_tname_str] = _flavour
                _added.append((_tt_str, _tname_str, _flavour))
    if _root:
        ctx.metadata["ensembles_chosen"] = _root
    if _added and getattr(ctx, "verbose", 0):
        logger.info("[ensembles_chosen] finalize backfilled %d flavour entry/entries: %s", len(_added), _added)


def _stamp_ensemble_composition(ctx: "TrainingContext") -> None:
    """Stamp ``metadata["ensemble_composition"]`` -- per-target snapshot of {flavour, members, fallback_reason}.

    Arch-6: a single observability dict that consumers (predict, evaluation, ops dashboards) can read
    without having to walk both ``ctx.ensembles`` (simple per-target) and ``ctx.models[tt][_CT_ENSEMBLE__*]``
    (cross-target). Reads from the stacker instance state (CompositeCrossTargetEnsemble.export_metadata),
    not from re-derivation, so member weights / strategy / fallback notes survive verbatim.

    Layout:
        metadata["ensemble_composition"]["simple"][tt][tname] = {
            "flavour": <str>, "members": [(name, weight), ...], "fallback_reason": <str|None>,
        }
        metadata["ensemble_composition"]["cross_target"][tt][tname] = {
            "strategy": <str>, "members": [(name, weight), ...], "fallback_reason": <str|None>,
        }
    """
    _composition: dict = {"simple": {}, "cross_target": {}}

    # Simple per-target ensembles: equal-weighted blend of every method in ``ctx.ensembles``.
    _ens_by_target = getattr(ctx, "ensembles", None) or {}
    _chosen = ctx.metadata.get("ensembles_chosen") if isinstance(ctx.metadata, dict) else None
    _simple_chosen = (_chosen or {}).get("simple", {}) if isinstance(_chosen, dict) else {}
    for _tt, _by_name in _ens_by_target.items():
        if not isinstance(_by_name, dict):
            continue
        _tt_str = str(_tt)
        for _tname, _methods in _by_name.items():
            if not isinstance(_methods, dict) or not _methods:
                continue
            _tname_str = str(_tname)
            _flavour = (_simple_chosen.get(_tt_str) or {}).get(_tname_str) if isinstance(_simple_chosen, dict) else None
            # Simple ensembles aggregate per-model probs uniformly inside ``score_ensemble``; the
            # "members" surfaced here are the underlying ensembling-method results (one per
            # flavour evaluated). Weight is 1/n -- this is reporting-grade not replay-grade.
            _members = [(str(_m), 1.0 / len(_methods)) for _m in sorted(_methods.keys())]
            _fallback_reason = None
            if _flavour is None:
                _fallback_reason = (
                    "no metric-driven winner; predict will use first-emitted flavour fallback "
                    "(deterministic via dict-insertion order)"
                )
            _composition["simple"].setdefault(_tt_str, {})[_tname_str] = {
                "flavour": _flavour,
                "members": _members,
                "fallback_reason": _fallback_reason,
            }

    # Cross-target ensembles: stored as ``_CT_ENSEMBLE__<orig>`` entries inside ctx.models[tt].
    for _tt, _by_name in (ctx.models or {}).items():
        if not isinstance(_by_name, dict):
            continue
        _tt_str = str(_tt)
        for _tname, _entries in _by_name.items():
            if not isinstance(_tname, str) or not _tname.startswith("_CT_ENSEMBLE__"):
                continue
            if not isinstance(_entries, list) or not _entries:
                continue
            _entry = _entries[0]
            _ensemble = getattr(_entry, "model", _entry)
            if not hasattr(_ensemble, "export_metadata"):
                continue
            try:
                _exp = _ensemble.export_metadata()
            except Exception as _exp_err:
                logger.warning("[ensemble_composition] export_metadata failed for %s/%s: %s", _tt_str, _tname, _exp_err)
                continue
            _strategy = _exp.get("strategy")
            _names = _exp.get("component_names") or []
            _weights = _exp.get("weights") or []
            _members = [
                (str(_n), float(_w))
                for _n, _w in zip(_names, _weights)
            ]
            _notes = _exp.get("notes") or {}
            _fallback_reason = None
            if _strategy == "single_best_fallback":
                _fallback_reason = "single component or non-finite OOF: fell back to best-single-target predictor"
            elif "fallback_reason" in _notes:
                _fallback_reason = str(_notes["fallback_reason"])
            elif _notes.get("capped_to_top_n"):
                _fallback_reason = f"capped to top {_notes['capped_to_top_n']} components"
            _composition["cross_target"].setdefault(_tt_str, {})[_tname] = {
                "strategy": _strategy,
                "members": _members,
                "fallback_reason": _fallback_reason,
            }

    if _composition["simple"] or _composition["cross_target"]:
        ctx.metadata["ensemble_composition"] = _composition


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


def _auto_calibrate_on_calib_slice(ctx: "TrainingContext") -> None:
    """Auto-fit post-hoc calibrators for every per-target model that carries a disjoint calib slice.

    Active only when ``TrainingSplitConfig.calib_size > 0`` carved a calib slice (``ctx.calib_idx``) and
    the trainer stamped ``entry.calib_probs`` / ``entry.calib_target`` (base-model predict_proba on the
    carved slice + aligned labels). The slice is leakage-free: carved from train, base model fit on
    train-minus-calib, disjoint from val/test by the splitter's hard asserts. Skips models without a
    stamped calib slice (no-op), so calib_size==0 runs are unaffected.
    """
    _calib_idx = getattr(ctx, "calib_idx", None)
    if _calib_idx is None or len(_calib_idx) == 0:
        return
    from .._calibration_models import calibrate_namespace_model
    _n = 0
    for _ttype, _by_name in (ctx.models or {}).items():
        if not isinstance(_by_name, dict):
            continue
        for _tname, _entries in _by_name.items():
            if not isinstance(_entries, list):
                continue
            for _entry in _entries:
                try:
                    if calibrate_namespace_model(_entry, target_type=_ttype):
                        _n += 1
                except Exception as _cal_err:
                    logger.warning("[calib] auto-calibration failed for %s/%s: %s", _ttype, _tname, _cal_err)
    if _n and getattr(ctx, "verbose", 0):
        logger.info("[calib] auto-calibrated %d per-target model(s) on the disjoint calib slice.", _n)


def _render_model_comparison_leaderboards(ctx: "TrainingContext") -> None:
    """Render the per-target model-comparison leaderboard once all models for a target are trained.

    Fires only when >=2 models on the same target carry a usable test score; the composer subsamples
    internally so assembly is bounded regardless of n. Best-effort -- a render failure never blocks finalize.
    """
    data_dir = getattr(ctx, "data_dir", "") or ""
    if not data_dir or not getattr(ctx, "save_charts", False):
        return
    _cfg = getattr(ctx, "reporting_config", None)
    if _cfg is None:
        _configs_root = getattr(ctx, "configs", None)
        _cfg = getattr(_configs_root, "reporting_config", None) if _configs_root is not None else None
    if _cfg is not None and not getattr(_cfg, "model_comparison_charts", True):
        return
    plot_outputs = (getattr(_cfg, "plot_outputs", "") or "") if _cfg is not None else ""
    if not plot_outputs:
        return
    from mlframe.reporting.diagnostics_dispatch import render_model_comparison_from_suite
    _n = 0
    for _tt, _by_name in (ctx.models or {}).items():
        if not isinstance(_by_name, dict):
            continue
        for _tname, _entries in _by_name.items():
            if not isinstance(_entries, list) or len(_entries) < 2:
                continue
            _base = join(
                data_dir, "charts", slugify(ctx.target_name), slugify(ctx.model_name),
                slugify(str(_tt).lower()), slugify(str(_tname)), "model_comparison",
            )
            try:
                os.makedirs(os.path.dirname(_base), exist_ok=True)
                if render_model_comparison_from_suite(
                    model_entries=_entries, target_type=str(_tt),
                    plot_outputs=plot_outputs, base_path=_base, metrics_dict=ctx.metadata,
                ):
                    _n += 1
            except Exception as _mc_err:
                logger.warning("[model_comparison] render failed for %s/%s: %s", _tt, _tname, _mc_err)
    if _n and getattr(ctx, "verbose", 0):
        logger.info("[model_comparison] rendered %d per-target leaderboard(s).", _n)


def _render_split_comparison_panels(ctx: "TrainingContext") -> None:
    """Render the per-model cross-split overfit panel once a model carries >=2 usable splits.

    One panel per (target, name, entry) -- unlike the leaderboard this is per-MODEL, not per-target. Best-effort;
    a render failure never blocks finalize. The composer subsamples internally so assembly stays bounded.
    """
    data_dir = getattr(ctx, "data_dir", "") or ""
    if not data_dir or not getattr(ctx, "save_charts", False):
        return
    _cfg = getattr(ctx, "reporting_config", None)
    if _cfg is None:
        _configs_root = getattr(ctx, "configs", None)
        _cfg = getattr(_configs_root, "reporting_config", None) if _configs_root is not None else None
    if _cfg is not None and not getattr(_cfg, "split_comparison_charts", True):
        return
    plot_outputs = (getattr(_cfg, "plot_outputs", "") or "") if _cfg is not None else ""
    if not plot_outputs:
        return
    from mlframe.reporting.diagnostics_dispatch import render_split_comparison_from_suite
    _n = 0
    for _tt, _by_name in (ctx.models or {}).items():
        if not isinstance(_by_name, dict):
            continue
        for _tname, _entries in _by_name.items():
            if not isinstance(_entries, list):
                continue
            for _i, _entry in enumerate(_entries):
                _mn = str(getattr(_entry, "model_name", None) or type(getattr(_entry, "model", None)).__name__ or f"model_{_i}")
                _base = join(
                    data_dir, "charts", slugify(ctx.target_name), slugify(ctx.model_name),
                    slugify(str(_tt).lower()), slugify(str(_tname)), slugify(_mn),
                )
                try:
                    os.makedirs(os.path.dirname(_base), exist_ok=True)
                    if render_split_comparison_from_suite(
                        entry=_entry, target_type=str(_tt), plot_outputs=plot_outputs,
                        base_path=_base, metrics_dict=ctx.metadata, model_name=_mn,
                    ):
                        _n += 1
                except Exception as _sc_err:
                    logger.warning("[split_comparison] render failed for %s/%s/%s: %s", _tt, _tname, _mn, _sc_err)
    if _n and getattr(ctx, "verbose", 0):
        logger.info("[split_comparison] rendered %d per-model overfit panel(s).", _n)


def _render_prediction_stability_panels(ctx: "TrainingContext") -> None:
    """Render the ensemble member-disagreement panel for any entry that stashed an ``(n, n_members)`` member matrix.

    The ensembling scorer stamps ``entry.member_test_preds`` when it combined >=2 members; this renders the spread /
    spread-vs-mean / uncertainty-calibration panels from it. Default-on; skipped cheaply when no ensemble entry exists.
    """
    data_dir = getattr(ctx, "data_dir", "") or ""
    if not data_dir or not getattr(ctx, "save_charts", False):
        return
    _cfg = getattr(ctx, "reporting_config", None)
    if _cfg is None:
        _configs_root = getattr(ctx, "configs", None)
        _cfg = getattr(_configs_root, "reporting_config", None) if _configs_root is not None else None
    if _cfg is not None and not getattr(_cfg, "prediction_stability", True):
        return
    plot_outputs = (getattr(_cfg, "plot_outputs", "") or "") if _cfg is not None else ""
    if not plot_outputs:
        return
    from mlframe.reporting.diagnostics_dispatch import render_prediction_stability_diagnostic
    _n = 0
    for _tt, _by_name in (ctx.models or {}).items():
        if not isinstance(_by_name, dict):
            continue
        for _tname, _entries in _by_name.items():
            if not isinstance(_entries, list):
                continue
            for _i, _entry in enumerate(_entries):
                # Ensemble entries arrive as the (namespace, train_df, val_df, test_df) tuple from train_and_evaluate_model.
                _e = _entry[0] if isinstance(_entry, tuple) and _entry else _entry
                _mp = getattr(_e, "member_test_preds", None)
                if _mp is None:
                    continue
                _model = getattr(_e, "model", None)
                _mn = str(getattr(_e, "model_name", None) or (type(_model).__name__ if _model is not None else f"ensemble_{_i}"))
                _yt = getattr(_e, "test_target", None)
                _base = join(
                    data_dir, "charts", slugify(ctx.target_name), slugify(ctx.model_name),
                    slugify(str(_tt).lower()), slugify(str(_tname)), slugify(_mn),
                )
                try:
                    os.makedirs(os.path.dirname(_base), exist_ok=True)
                    if render_prediction_stability_diagnostic(
                        member_preds=_mp, y_true=_yt, plot_outputs=plot_outputs,
                        base_path=_base, metrics_dict=ctx.metadata,
                    ):
                        _n += 1
                except Exception as _ps_err:
                    logger.warning("[prediction_stability] render failed for %s/%s/%s: %s", _tt, _tname, _mn, _ps_err)
    if _n and getattr(ctx, "verbose", 0):
        logger.info("[prediction_stability] rendered %d ensemble member-disagreement panel(s).", _n)


def finalize_suite(ctx: TrainingContext) -> dict:
    """Aggregate fairness reports, save metadata, emit phase/rendering summaries, surface selected features.

    Returns ``ctx.metadata`` (also mutated in-place) so legacy callers keeping a ``metadata = finalize_suite(ctx)`` rebind keep working.
    """
    # Auto-calibrate per-target models on the disjoint calib slice (calib_size>0) BEFORE the metadata /
    # ensemble-composition walks so they see the calibrated wrappers + stamped calibrated_<split>_probs.
    try:
        _auto_calibrate_on_calib_slice(ctx)
    except Exception as _cal_err:
        logger.warning("[calib] auto-calibration pass failed: %s", _cal_err)

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

    # Arch-6: per-target ensemble composition snapshot for ops / debugging. Must run AFTER
    # _persist_chosen_ensemble_flavours so the "simple" bucket reads the post-backfill chosen flavour.
    try:
        _stamp_ensemble_composition(ctx)
    except Exception as _comp_err:
        logger.warning("[ensemble_composition] stamp failed: %s", _comp_err)

    # Predict-path parity: dump the in-memory ``_CT_ENSEMBLE__*`` entries (cross-target ensembles built by
    # _phase_composite_post.py) to disk so the predict loader (load_mlframe_suite / predict_mlframe_models_suite)
    # actually sees them. Pre-fix they lived only in ctx.models and were silently lost across save / load.
    try:
        _persist_ct_ensemble_entries(ctx)
    except Exception as _ct_err:
        logger.warning("[_CT_ENSEMBLE persist] failed: %s", _ct_err)

    # Per-target model-comparison leaderboard (ROC/metric overlay + sorted metric bars + between-model
    # prediction-correlation) once every model for a target exists; default-on, best-effort.
    try:
        _render_model_comparison_leaderboards(ctx)
    except Exception as _mc_err:
        logger.warning("[model_comparison] leaderboard pass failed: %s", _mc_err)

    # Per-model cross-split overfit panel (train/val/test headline deltas + verdict); default-on, best-effort.
    try:
        _render_split_comparison_panels(ctx)
    except Exception as _sc_err:
        logger.warning("[split_comparison] panel pass failed: %s", _sc_err)

    # Ensemble member-disagreement panel from the stashed (n, n_members) test matrix; default-on, best-effort.
    try:
        _render_prediction_stability_panels(ctx)
    except Exception as _ps_err:
        logger.warning("[prediction_stability] panel pass failed: %s", _ps_err)

    # Honest-estimator diagnostics aggregator: stamps bootstrap CI per metric, categorical PSI drift, calibration plot, and the provenance disposition table into metadata so the persisted blob carries the four artefacts. Gated by ReportingConfig.honest_estimator_diagnostics (default True). Failures must not block the save.
    _reporting_cfg = getattr(ctx, "reporting_config", None)
    if _reporting_cfg is None:
        _configs_root = getattr(ctx, "configs", None)
        _reporting_cfg = getattr(_configs_root, "reporting_config", None) if _configs_root is not None else None
    _hd_on = True if _reporting_cfg is None else bool(getattr(_reporting_cfg, "honest_estimator_diagnostics", True))
    if _hd_on:
        try:
            from ..honest_diagnostics import run_honest_diagnostics
            run_honest_diagnostics(ctx, getattr(ctx, "models", {}) or {}, ctx.metadata)
        except Exception as _hd_err:
            logger.warning("[honest_diagnostics] aggregator failed: %s", _hd_err)

    # ``verbose=0`` silences the duplicate "Saved metadata to ..." log line; main.py already saved partway.
    _finalize_and_save_metadata(ctx, verbose=0)

    # One-line chart-accounting INFO at suite end (INV-14): independent of verbose so an operator always learns
    # whether diagnostics were saved, skipped by design (no data_dir), or lost to a render failure.
    try:
        from ._setup_helpers import log_chart_summary
        log_chart_summary(ctx.metadata, save_charts=bool(getattr(ctx, "save_charts", False)),
                          data_dir=getattr(ctx, "data_dir", "") or None)
    except Exception as _cs_err:
        logger.debug("[finalize] chart-summary log failed: %s", _cs_err)

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

    # Restore process-wide overrides flipped by setup_configuration. Pre-fix the
    # residual_audit + inline_display flags were set but NEVER restored, so two
    # back-to-back suite calls with different behavior_config values silently
    # inherited the first call's setting (the leading comment at the set site
    # promised restore but it was aspirational). Snapshot lives in ctx.artifacts.
    _artifacts = ctx.artifacts or {}
    _residual_audit_prior = _artifacts.pop("_process_flag_prior_residual_audit", None)
    if _residual_audit_prior is not None:
        try:
            from mlframe.training.evaluation import _set_residual_audit_enabled
            _set_residual_audit_enabled(_residual_audit_prior)
        except (ImportError, AttributeError) as _restore_err:
            logger.debug(
                "[finalize] residual_audit flag restore failed: %s: %s",
                type(_restore_err).__name__, _restore_err,
            )
    if "_process_flag_prior_inline_display" in _artifacts:
        _inline_display_prior = _artifacts.pop("_process_flag_prior_inline_display")
        try:
            from mlframe.reporting.renderers.save import set_inline_display_mode
            set_inline_display_mode(_inline_display_prior)
        except (ImportError, AttributeError) as _restore_err:
            logger.debug(
                "[finalize] inline_display flag restore failed: %s: %s",
                type(_restore_err).__name__, _restore_err,
            )

    return ctx.metadata
