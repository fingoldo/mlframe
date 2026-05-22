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
    # Lazy import: parent re-exports this helper at its module bottom, so a
    # top-level import would form a cycle. Python's module cache makes this
    # sub-microsecond per call.
    from ._phase_train_one_target import _choose_ensemble_flavour

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
    _ensembles = score_ensemble(
        models_and_predictions=ens_models,
        ensemble_name=f"{pre_pipeline_name}{_members_label} ",
        n_features=ens_n_features,
        uncertainty_quantile=_conf_q,
        group_ids=getattr(ctx, "group_ids", None),
        sample_weight=_ens_sample_weight,
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
        # Persist ``rrf_k`` (and a couple of other replay-critical params) into metadata
        # so predict-side ``_combine_probs`` replays the exact same blend a non-default-k
        # train was scored with. Pre-fix predict hard-coded k=60 - a user setting rrf_k=10
        # at train time silently got k=60 at predict time. The default 60 mirrors
        # ``score_ensemble``'s default; common_params may override.
        try:
            _rrf_k_used = int(common_params.get("rrf_k", 60))
        except (TypeError, ValueError):
            _rrf_k_used = 60
        metadata.setdefault("ensembles_chosen_params", {}) \
            .setdefault(str(target_type), {})[str(cur_target_name)] = {
                "rrf_k": _rrf_k_used,
            }
