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
        return

    if verbose:
        logger.info(f"evaluating simple ensembles...")
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
    # confidence_ensemble_quantile=0.0 disables the Conf Ensemble output entirely.
    _conf_q = float(getattr(behavior_config, "confidence_ensemble_quantile", 0.1))
    # Thread ctx.group_ids + per-target sample_weight into score_ensemble so the
    # gate / NNLS / RRF stages compute weighted + group-aware. Pre-fix these were
    # both silently absent here -- score_ensemble's docstring at models/ensembling.py
    # says "ctx auto-passes when available" but the suite never auto-passed; LTR /
    # weighted suites got member selection + RRF blend computed on i.i.d. rows.
    _ctx_sw_dict = getattr(ctx, "sample_weights", None) or {}
    _ens_sample_weight = (
        _ctx_sw_dict.get(cur_target_name)
        if isinstance(_ctx_sw_dict, dict) and _ctx_sw_dict
        else (
            current_common_params.get("sample_weight")
            if isinstance(current_common_params, dict)
            else None
        )
    )
    # Spread common_params first, then explicitly set group_ids/sample_weight.
    # If common_params happened to already carry either key (unlikely on the
    # current build but defensive against future schema drift), the explicit
    # set wins. Avoid TypeError "multiple values for kw" by removing first.
    _ens_kwargs = dict(common_params or {})
    _ens_kwargs.pop("group_ids", None)
    _ens_kwargs.pop("sample_weight", None)
    # W16D / A3#3: surface ``TrainingBehaviorConfig.use_ap12_calibrated_probs_in_ensemble`` as the
    # explicit ``use_ap12_calibrated_probs`` kwarg on ``score_ensemble``. Default True so the suite
    # default benefits from AP12-calibrated probs in arithm / harm / quad / qube / geo / median blends;
    # opt-out by setting False on the behavior config. RRF is rank-based and ignores the knob.
    _use_ap12_cal = bool(getattr(behavior_config, "use_ap12_calibrated_probs_in_ensemble", True))
    _ens_kwargs.pop("use_ap12_calibrated_probs", None)
    _ensembles = score_ensemble(
        models_and_predictions=ens_models,
        ensemble_name=f"{pre_pipeline_name}{_members_label} ",
        n_features=ens_n_features,
        uncertainty_quantile=_conf_q,
        group_ids=getattr(ctx, "group_ids", None),
        sample_weight=_ens_sample_weight,
        use_ap12_calibrated_probs=_use_ap12_cal,
        **_ens_kwargs,
    )
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
            _target_models.append(_ens_result)
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
                metadata.setdefault("ensembles_chosen", {}) \
                    .setdefault("simple", {}) \
                    .setdefault(target_type, {})[cur_target_name] = _chosen
        except Exception as _choose_err:
            logger.warning("ensembles_chosen stamp failed for %s/%s: %s", target_type, cur_target_name, _choose_err)
        # Persist ``rrf_k`` only when RRF was actually iterated for this target -- otherwise
        # the metadata stamps a stale-but-default ``rrf_k`` for regression-only suites (where
        # score_ensemble filters RRF out) which pollutes regression-review diffs without ever
        # affecting predict (which only reads rrf_k for the rrf flavour). Detection: look at
        # ``ensembling_methods`` in common_params AND the keys actually emitted into
        # ``_ensembles`` -- a flavour is in the iteration when it appears in either.
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
            metadata.setdefault("ensembles_chosen_params", {}) \
                .setdefault(str(target_type), {})[str(cur_target_name)] = {
                    "rrf_k": _rrf_k_used,
                }
