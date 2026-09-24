"""Per-target simple-ensemble evaluation tail, carved out of
``_train_one_target`` (in ``_phase_train_one_target_body``).

Fires once per ``pre_pipeline`` after the inner model loop has populated
``ens_models``; if 2+ members survived the gate, score_ensemble blends them,
stamps the chosen flavour into metadata, persists the ensemble objects into
``ctx.ensembles`` and the per-target ``models`` slot, and records the
replay-critical ``rrf_k``.

Re-imported at the parent's module bottom so historical
``from ._phase_train_one_target import _finalize_per_target_ensembling``
keeps resolving transparently.
"""
from __future__ import annotations

import logging

from mlframe.models.ensembling import score_ensemble

# Top-level import of ``_choose_ensemble_flavour`` from the new leaf module ``_ensemble_chooser``.
# Pre-fix this lived in ``_phase_train_one_target`` (parent of this sibling) and had to be
# in-function imported on every per-target iteration to dodge the import cycle (parent re-exports
# this sibling at its bottom). The leaf move breaks the cycle so the import resolves once at module
# load, surfacing any typo / signature drift immediately rather than mid-suite.
from ._ensemble_chooser import _choose_ensemble_flavour

logger = logging.getLogger("mlframe.training.core._phase_train_one_target")


def _finalize_per_target_ensembling(
    *,
    ens_models,
    train_df_transformed,
    behavior_config,
    ctx,
    cur_target_name,
    current_common_params,
    common_params,
    pre_pipeline_name,
    models,
    target_type,
    metadata,
    verbose: bool,
    current_val_target=None,
):
    """Run ``score_ensemble`` on the surviving members and persist outputs.

    Mirrors the prior in-line block byte-for-byte: same dict spread order,
    same defensive ``pop`` of ``group_ids`` / ``sample_weight`` before the
    explicit kwargs, same K2-CATASTROPHIC-DROPOUT sentinel filter, same
    ``ensembles_chosen`` stamping with ``simple`` sub-key, same
    ``rrf_k`` persist into ``ensembles_chosen_params``.

    Parameters are kwargs-only to keep the long call-site readable and to
    avoid positional-arg drift when callers grow extra context.
    """
    if not (ens_models and len(ens_models) > 1):
        # A single model still needs its threshold: this early return used to skip tuning entirely, so a lone
        # CatBoost on a 2%-positive target kept 0.5 and predicted the negative class almost everywhere, while the same
        # config with two models was tuned.
        # With ensembling off there is no ens_models list at all; the target's trained models are the members.
        _members = ens_models or list((models or {}).get(target_type, {}).get(cur_target_name, []) or [])
        _tune_decision_thresholds(
            ens_models=_members, target_type=target_type, cur_target_name=cur_target_name,
            behavior_config=behavior_config, common_params=common_params, metadata=metadata, verbose=verbose,
            current_val_target=current_val_target, ensembles=None, chosen_flavour=None,
        )
        return

    if verbose:
        logger.info("evaluating simple ensembles...")
    ens_n_features = train_df_transformed.shape[1] if train_df_transformed is not None else None
    # Name the ensemble by its members so log grep shows which models actually participated;
    # cap to 4 to keep headers readable. short_model_tag strips internal shim suffixes
    # (WithDMatrixReuse / WithDatasetReuse) so the tag is the bare family name.
    from .._format import short_model_tag as _short_tag_fn
    _member_tags = [_short_tag_fn(getattr(m, "model", m)) for m in ens_models]
    if len(_member_tags) <= 4:
        _members_label = "[" + "+".join(_member_tags) + "]"
    else:
        _members_label = f"[N={len(_member_tags)}]"
    # confidence_ensemble_quantile=0.0 disables the Conf Ensemble output entirely (the default: see
    # BehaviorConfig.confidence_ensemble_quantile's docstring). The getattr fallback matches that same
    # default for a behavior_config stub that lacks the attribute entirely.
    _conf_q = float(getattr(behavior_config, "confidence_ensemble_quantile", 0.0))
    # Thread ctx.group_ids + per-target sample_weight into score_ensemble so the
    # gate / NNLS / RRF stages compute weighted + group-aware. Pre-fix these were
    # both silently absent here -- score_ensemble's docstring at models/ensembling.py
    # says "ctx auto-passes when available" but the suite never auto-passed; LTR /
    # weighted suites got member selection + RRF blend computed on i.i.d. rows.
    _ctx_sw_dict = getattr(ctx, "sample_weights", None) or {}
    _ens_sample_weight = (
        _ctx_sw_dict.get(cur_target_name)
        if isinstance(_ctx_sw_dict, dict) and _ctx_sw_dict
        else (current_common_params.get("sample_weight") if isinstance(current_common_params, dict) else None)
    )
    # Spread common_params first, then explicitly set group_ids/sample_weight.
    # If common_params happened to already carry either key (unlikely on the
    # current build but defensive against future schema drift), the explicit
    # set wins. Avoid TypeError "multiple values for kw" by removing first.
    _ens_kwargs = dict(common_params or {})
    _ens_kwargs.pop("group_ids", None)
    _ens_kwargs.pop("sample_weight", None)
    _ens_kwargs.pop("target_type", None)
    # W16D / A3#3: surface ``TrainingBehaviorConfig.use_ap12_calibrated_probs_in_ensemble`` as the
    # explicit ``use_ap12_calibrated_probs`` kwarg on ``score_ensemble``. Default True so the suite
    # default benefits from AP12-calibrated probs in arithm / harm / quad / qube / geo / median blends;
    # opt-out by setting False on the behavior config. RRF is rank-based and ignores the knob.
    _use_ap12_cal = bool(getattr(behavior_config, "use_ap12_calibrated_probs_in_ensemble", True))
    _ens_kwargs.pop("use_ap12_calibrated_probs", None)
    # PZAD blend knobs (behavior_config): Caruana metric-direct weights (alternative to NNLS) + extra flavours
    # (e.g. rank_average) appended to the default SIMPLE_ENSEMBLING_METHODS set. Both OFF/empty by default.
    if bool(getattr(behavior_config, "use_caruana_weights_in_ensemble", False)):
        _ens_kwargs["use_caruana_weights"] = True
    _extra_methods = tuple(getattr(behavior_config, "extra_ensembling_methods", ()) or ())
    if _extra_methods:
        from mlframe.models.ensembling.base import SIMPLE_ENSEMBLING_METHODS as _SIMPLE_METHODS
        _base_methods = _ens_kwargs.get("ensembling_methods")
        if not isinstance(_base_methods, (list, tuple)) or not _base_methods:
            _base_methods = list(_SIMPLE_METHODS)
        _ens_kwargs["ensembling_methods"] = list(dict.fromkeys(list(_base_methods) + list(_extra_methods)))
    _ensembles = score_ensemble(
        models_and_predictions=ens_models,
        ensemble_name=f"{pre_pipeline_name}{_members_label} ",
        n_features=ens_n_features,
        uncertainty_quantile=_conf_q,
        group_ids=getattr(ctx, "group_ids", None),
        sample_weight=_ens_sample_weight,
        use_ap12_calibrated_probs=_use_ap12_cal,
        # score_ensemble drops flavours invalid for this target type (rank fusion outside learning-to-rank) before building any.
        target_type=target_type,
        **_ens_kwargs,
    )
    _chosen = None  # the winning flavour, set below when there is one; the threshold tuner reads it
    # Persist the ensemble outputs so finalize_suite can serialise them and downstream
    # consumers (predict, reporting) see them. Pre-fix this return value was bound to a
    # local that nothing read, silently discarding every ensemble model the suite built.
    if _ensembles:
        ctx.ensembles.setdefault(target_type, {})[cur_target_name] = _ensembles
        # Mirror into the per-target model list (same slot the per-family training loop
        # uses) so any code iterating ``models[target_type][target_name]`` picks the
        # ensembles up without needing a separate dispatch.
        _target_models = models.setdefault(target_type, {}).setdefault(cur_target_name, [])
        for _ens_method, _ens_result in _ensembles.items():
            # K2-CATASTROPHIC-DROPOUT sentinel filter: ``score_ensemble`` short-circuits
            # with sentinel-only result entries (``_reason``, ``_n_members``,
            # ``_dropped_member``, ``_kept_member``, ``_k2_mae_ratio``) when the K=2
            # catastrophic-dropout fires (or any other early-exit branch returns a
            # leading-underscore key). Those are METADATA, not model entries; appending
            # them into the per-target model list pollutes downstream predict / metric /
            # ensemble code with strings / ints / floats where it expects model objects.
            # Skip any key starting with ``_`` to leave the model list clean.
            if isinstance(_ens_method, str) and _ens_method.startswith("_"):
                continue
            # ``score_ensemble``'s values are the raw ``train_and_evaluate_model`` return --
            # ``(entry_namespace, train_df, val_df, test_df)`` -- not a bare namespace (mirrors the
            # same unwrap ``process_method.py`` already does to stamp ``member_test_preds``). Every
            # OTHER entry in this per-target model list (the raw per-family models) is a bare
            # namespace, so appending the 4-tuple as-is here silently broke any consumer of
            # ``models[target_type][target_name]`` expecting namespace-like ``.model``/``.test_preds``
            # attributes on an ensemble entry. Unwrap to the namespace before appending; the full
            # tuple (incl. the split frames) stays available via ``ctx.ensembles``.
            _entry_ns = _ens_result[0] if isinstance(_ens_result, tuple) else _ens_result
            # Stamp the flavour name onto the result object itself. Before this, the dict key
            # (the only place the flavour lived) was dropped the moment the value went into the
            # flat per-target model list -- a caller holding one of these entries had no way to
            # tell WHICH ensemble flavour it was short of parsing the flavour token back out of a
            # saved chart filename (``metrics[...]['charts']['paths']``), which silently stopped
            # working the moment chart rendering was disabled (``OutputConfig.save_charts=False``).
            try:
                _entry_ns.name = _ens_method
                # Unique identity among the target's models (flavour + pre-pipeline + members, as in the chart name);
                # metadata blocks key results by ``model_name``, and without it every ensemble keyed as "NoneType".
                _entry_ns.model_name = f"Ens{str(_ens_method).upper()} {pre_pipeline_name}{_members_label}".strip()
            except Exception as _name_stamp_err:
                logger.debug("could not stamp flavour name %r onto ensemble result: %s", _ens_method, _name_stamp_err)
            _target_models.append(_entry_ns)
        # Stamp the winning ensemble flavour so the predict path picks the same flavour the
        # training selection rule would have picked. Predict reads ``ensembles_chosen``
        # (see core/predict.py::_resolve_chosen_flavour) which expects the nested layout
        # ``{target_type: {target_name: flavour}}``. A None winner (no candidate exposed a
        # ranking metric) is intentionally NOT stamped so the predict-side fallback fires.
        try:
            _chosen = _choose_ensemble_flavour(_ensembles)
            if _chosen is not None:
                # Sub-key per ensemble family: simple per-target ensembles live under
                # ``ensembles_chosen["simple"]``; cross-target ensembles are stamped by
                # _phase_composite_post under ``ensembles_chosen["cross_target"]``.
                metadata.setdefault("ensembles_chosen", {}).setdefault("simple", {}).setdefault(target_type, {})[cur_target_name] = _chosen
        except Exception as _choose_err:
            logger.warning("ensembles_chosen stamp failed for %s/%s: %s", target_type, cur_target_name, _choose_err)
        # Persist ``rrf_k`` only when RRF was actually iterated for this target -- otherwise
        # the metadata stamps a stale-but-default ``rrf_k`` for regression-only suites (where
        # score_ensemble filters RRF out) which pollutes regression-review diffs without ever
        # affecting predict (which only reads rrf_k for the rrf flavour). Detection: look at
        # ``ensembling_methods`` in common_params AND the keys actually emitted into
        # ``_ensembles`` -- a flavour is in the iteration when it appears in either.
        # Blend weights must survive into predict. Training used to fit NNLS/Caruana weights, score a WEIGHTED blend,
        # stamp that flavour as the winner -- and persist nothing, so deployment replayed the same members as an
        # unweighted mean. That is a different estimator from the one whose metrics were reported.
        try:
            _saw = _ensembles.get("_stacking_gate") if isinstance(_ensembles, dict) else None
            if isinstance(_saw, dict) and _saw.get("applied_to_blend") and _saw.get("aligned_weights"):
                metadata.setdefault("ensembles_chosen_params", {}).setdefault(str(target_type), {}).setdefault(str(cur_target_name), {})["blend_weights"] = [
                    float(w) for w in _saw["aligned_weights"]
                ]
        except Exception as _w_err:
            logger.warning("blend-weight stamp failed for %s/%s: %s", target_type, cur_target_name, _w_err)
        # And the member set those weights belong to, so predict blends exactly the models training scored.
        _members = _ensembles.get("_surviving_members") if isinstance(_ensembles, dict) else None
        if isinstance(_members, list) and _members:
            metadata.setdefault("ensembles_chosen_params", {}).setdefault(str(target_type), {}).setdefault(str(cur_target_name), {})["members"] = [
                str(_n) for _n in _members
            ]
        _ens_methods_used = common_params.get("ensembling_methods") if isinstance(common_params, dict) else None
        _rrf_in_iter = False
        if isinstance(_ens_methods_used, (list, tuple)):
            _rrf_in_iter = "rrf" in _ens_methods_used
        if not _rrf_in_iter:
            _rrf_in_iter = any(isinstance(k, str) and k == "rrf" for k in _ensembles.keys())
        if _rrf_in_iter:
            try:
                _rrf_k_used = int(common_params.get("rrf_k", 60))
            except (TypeError, ValueError):
                _rrf_k_used = 60
            metadata.setdefault("ensembles_chosen_params", {}).setdefault(str(target_type), {}).setdefault(str(cur_target_name), {})["rrf_k"] = _rrf_k_used

    # Per-target binary decision thresholds (val, never test): one per member, tuned on that member's own val
    # probabilities, and one for the ensemble, tuned on the BLEND's. See _tune_decision_thresholds.
    _tune_decision_thresholds(
        ens_models=ens_models, target_type=target_type, cur_target_name=cur_target_name, behavior_config=behavior_config,
        common_params=common_params, metadata=metadata, verbose=verbose, current_val_target=current_val_target,
        ensembles=_ensembles, chosen_flavour=_chosen,
    )

    # VOTENRANK-WIRE (diversity): rank the suite's own fitted-but-not-selected members for genuine
    # blend-additive diversity, over the same ``ens_models`` pool ``score_ensemble`` just blended above.
    # Observational-only (never changes which models/ensembles get used); default ON per
    # TrainingBehaviorConfig.recommend_diversity_additions_in_leaderboard.
    try:
        from ._diversity_recommendations import compute_diversity_recommendations
        _div_shortlist = compute_diversity_recommendations(
            ens_models=ens_models,
            target_type=target_type,
            behavior_config=behavior_config,
            verbose=verbose,
        )
        if _div_shortlist is not None:
            metadata.setdefault("diversity_recommendations", {}).setdefault(str(target_type), {})[str(cur_target_name)] = _div_shortlist
    except Exception as _div_err:
        logger.warning("diversity_recommendations wiring failed for %s/%s: %s", target_type, cur_target_name, _div_err)


def _member_name(member, index: int) -> str:
    """The name a member's predictions are keyed by at predict time (``model.model_name`` = its .dump basename)."""
    name = getattr(member, "model_name", None)
    return str(name) if name else f"member{index}"


def _member_val_probs(member):
    """Val probabilities of one member; the member tuple layout is (model, test_preds, test_probs, val_preds, val_probs, ...)."""
    if isinstance(member, (tuple, list)):
        return member[4] if len(member) > 4 else None
    return getattr(member, "val_probs", None)


def _tune_decision_thresholds(
    *, ens_models, target_type, cur_target_name, behavior_config, common_params, metadata, verbose,
    current_val_target=None, ensembles=None, chosen_flavour=None,
) -> None:
    """Tune a binary decision threshold for every member AND for the ensemble, each on its OWN val probabilities.

    One threshold per target used to be tuned on the FIRST member that exposed val probabilities and then applied to
    everything - to each other member and to the blend. A linear member's flatter probabilities put the tuned value
    (e.g. 0.31) far below the sharper blend's operating point, so the deployed hard labels over-predicted the positive
    class while the log said "tuned". Members are stamped under ``"{tt}|{tname}|{model_name}"`` and read that key first
    at predict; the target key ``"{tt}|{tname}"`` is the ensemble's (the blend's own val probabilities), or the lone
    model's when there is only one.

    Tri-state ``behavior_config.tune_decision_threshold``: "auto" tunes only an imbalanced val target, True always,
    False forces 0.5. Val is the early-stopping surface and an allowed tuning surface; test is never touched.
    """
    try:
        from ..configs import TargetTypes

        if target_type != TargetTypes.BINARY_CLASSIFICATION:
            return
        mode = getattr(behavior_config, "tune_decision_threshold", "auto")
        if mode is False:
            return
        val_target = current_val_target
        if val_target is None and isinstance(common_params, dict):
            val_target = common_params.get("val_target")
        if val_target is None:
            return
        import numpy as _np
        from ._setup_helpers import should_tune_decision_threshold as _should_tune, tune_decision_threshold as _tune

        y = _np.asarray(val_target).ravel()
        target_key = f"{target_type}|{cur_target_name}"
        metric = str(getattr(behavior_config, "tune_decision_threshold_metric", "balanced_accuracy"))
        tune = _should_tune(mode, y)
        thresholds = metadata.setdefault("decision_thresholds", {})
        paths = metadata.setdefault("decision_threshold_paths", {})

        def _stamp(key: str, probs, label: str) -> None:
            """Record the decision threshold and how it was chosen for ``key``: tuned on ``probs``, or the 0.5 default."""
            if not tune:
                thresholds[key], paths[key] = 0.5, "default_0.5"
                return
            arr = _np.asarray(probs)
            pos = arr[:, 1] if arr.ndim == 2 and arr.shape[1] >= 2 else arr.ravel()
            if pos.shape[0] != y.shape[0]:
                return  # a confidence-filtered subset cannot be scored against the full val target
            thresholds[key] = float(_tune(y, pos, metric=metric))
            paths[key] = "tuned"
            if verbose:
                logger.info("tuned decision threshold for %s: %.4f (metric=%s, val, %s)", key, thresholds[key], metric, label)

        member_probs = [(i, m, _member_val_probs(m)) for i, m in enumerate(ens_models or [])]
        member_probs = [(i, m, p) for i, m, p in member_probs if p is not None]
        for i, m, p in member_probs:
            _stamp(f"{target_key}|{_member_name(m, i)}", p, "member")

        blend_probs = None
        if ensembles is not None and chosen_flavour is not None:
            entry = ensembles.get(chosen_flavour)
            result = entry[0] if isinstance(entry, tuple) and entry else entry
            blend_probs = getattr(result, "val_probs", None)
        if blend_probs is not None:
            _stamp(target_key, blend_probs, f"ensemble '{chosen_flavour}'")
        elif len(member_probs) == 1:
            _stamp(target_key, member_probs[0][2], "single model")
        elif not tune:
            thresholds[target_key], paths[target_key] = 0.5, "default_0.5"
            if verbose:
                logger.info("decision threshold for %s: 0.5 (auto: balanced target, not tuned)", target_key)
    except Exception as _thr_err:
        logger.warning("decision-threshold tuning failed for %s/%s: %s", target_type, cur_target_name, _thr_err)
