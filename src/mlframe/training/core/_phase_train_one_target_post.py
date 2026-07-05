"""Per-model post-train tail + extreme-AR gate carved from ``_phase_train_one_target_body``.

Holds two cohesive blocks lifted VERBATIM out of the nested pre_pipeline x model x weight
loop in ``_train_one_target`` so the parent stays under the 1k LOC ceiling:

* ``_evaluate_mlp_extreme_ar_gate`` -- the per-(target, model) extreme-AR + group-aware
  skip decision (mirrors the composite-discovery skip). Returns the skip flag + the MLP
  protection flags; the caller keeps the ``continue`` so phase ordering is unchanged.
* ``_run_per_model_post_train_tail`` -- the immediately-after-``process_model`` tail:
  TTA uncertainty eval, per-model composite y-scale emit, and the adaptive RAM reclaim.
  All three mutate ``metadata`` / ``ctx.models`` in place, so nothing is returned.

The blocks read/write only the locals passed in explicitly (no closure capture); their
own lazy imports moved with them so the parent-bottom re-export doesn't add hard imports.
"""

from __future__ import annotations

import logging

import numpy as np

from ._phase_train_one_target_mlp_helpers import extreme_ar_skip_decision

logger = logging.getLogger("mlframe.training.core._phase_train_one_target")

# mlframe-private selector markers applied to MRMR / RFECV via ``setattr`` in ``_build_pre_pipelines``; sklearn.clone()
# strips non-constructor attributes, so they must be re-asserted on the per-strategy clone (esp. the weight-aware flag,
# without which ``_passthrough_cols_fit_transform`` never forwards ``sample_weight`` and weight-aware FS is inert).
_SELECTOR_STICKY_ATTRS = ("_mlframe_use_sample_weights_in_fs_", "_mlframe_selector_kind_", "_mlframe_identity_cache_override_")


def _forward_selector_sticky_attrs(src, dst):
    """Copy the mlframe selector sticky markers from ``src`` onto ``dst`` (and any inner ``'pre'`` step)."""
    if src is None or dst is None:
        return
    src_step = src.named_steps["pre"] if hasattr(src, "named_steps") and "pre" in getattr(src, "named_steps", {}) else src
    dst_step = dst.named_steps["pre"] if hasattr(dst, "named_steps") and "pre" in getattr(dst, "named_steps", {}) else dst
    for _attr in _SELECTOR_STICKY_ATTRS:
        if hasattr(src_step, _attr):
            setattr(dst_step, _attr, getattr(src_step, _attr))


def _evaluate_mlp_extreme_ar_gate(
    *,
    mlframe_model_name: str,
    cur_target_name: str,
    behavior_config,
    metadata: dict,
    _model_idx_in_run: int,
    _total_models_in_run: int,
):
    """Per-(target, model) extreme-AR + group-aware gate.

    Extreme-AR + group-aware MLP skip (mirrors the composite-discovery
    extreme_ar_group_aware_skip). On AR(1)-dominated targets with a group-aware split, MLP
    cannot learn a transferable residual: the target is fully explained by the lag, and the
    MLP's nearly-linear decision surface extrapolates catastrophically on unseen-group test
    rows (observed in prod: very small pred_std vs target_std, strongly negative R2,
    predictions near-constant vs target_mean). The ensemble's quality gate catches it, but the
    wasted train time + multi-MB save dump is pure cost. Skip MLP in this regime; lag_predict +
    Ridge carry the AR signal.

    Returns ``(skip, mlp_extreme_ar_fired, mlp_ea_lag1)``. ``skip`` tells the caller to
    ``continue`` past this model; the two MLP flags drive the downstream weight-decay /
    output-activation protections. Behaviour is byte-for-byte the inline block it replaced.
    """
    # Extreme-AR + group-aware MLP trigger predicate (shared by
    # 3 protections: skip / drop per-group aggregate cols /
    # bump weight_decay 100x). Computed once per (target, model)
    # so the three protections agree on whether to fire.
    _mlp_extreme_ar_fired = False
    _mlp_ea_lag1 = None
    _mlp_ea_thr = float(
        getattr(
            behavior_config,
            "mlp_extreme_ar_threshold",
            0.99,
        )
    )
    # Target-level extreme-AR + group-aware signal (same for every
    # model; gate only UNBOUNDED-OUTPUT models on it).
    #
    # CRITICAL: the stored ``target_distribution_report`` describes the
    # RAW/picked target (lag1_corr ~1.0 here), computed ONCE before
    # composite expansion. It is NOT recomputed per composite target.
    # Composite targets (``TVT-diff-*`` / ``TVT-linres*`` / residuals)
    # deliberately REMOVE or BOUND the AR variance -- that is exactly
    # where an MLP belongs and works. So the skip must fire ONLY on the
    # raw target, NEVER on a composite. Gating on the raw lag1 for a
    # composite would wrongly drop the MLP from the one regime it can
    # actually win.
    _skip_models = tuple(
        getattr(
            behavior_config,
            "extreme_ar_group_aware_skip_models",
            ("mlp",),
        )
    )
    # Only read the (raw-target) AR signal when it could matter for
    # THIS model: a skip candidate, or the MLP (which uses the flag
    # for its weight_decay / output-activation protections even when
    # the hard skip is off).
    if mlframe_model_name in _skip_models or mlframe_model_name == "mlp":
        _td_report = metadata.get("target_distribution_report", {}) or {}
        _td_diag = _td_report.get("diagnostics", {}) or {}
        _td_knobs = _td_report.get("knob_overrides", {}) or {}
        _ea_lag1 = _td_diag.get("lag1_autocorr_per_group")
        _split_overrides = _td_knobs.get("split_config", {}) or {}
        _group_aware = bool(_split_overrides.get("prefer_group_aware", False))
        _skip, _extreme_ar_fired = extreme_ar_skip_decision(
            mlframe_model_name,
            cur_target_name,
            skip_models=_skip_models,
            skip_enabled=bool(
                getattr(
                    behavior_config,
                    "mlp_extreme_ar_group_aware_skip",
                    True,
                )
            ),
            lag1_autocorr_per_group=_ea_lag1,
            group_aware=_group_aware,
            threshold=_mlp_ea_thr,
        )
        # MLP-specific flag for downstream protections.
        if mlframe_model_name == "mlp":
            _mlp_extreme_ar_fired = _extreme_ar_fired
            _mlp_ea_lag1 = _ea_lag1
        if _skip:
            logger.warning(
                "Skipping %s training for raw target='%s' (model %d/%d): "
                "extreme-AR + group-aware skip fired "
                "(lag1_autocorr_per_group=%.4f >= %.2f). Neural net "
                "collapses on unseen test groups here and is dropped by "
                "the ensemble quality gate anyway. NOTE: composite "
                "targets are exempt (bounded variance -- neural nets "
                "train there). Disable via "
                "TrainingBehaviorConfig(mlp_extreme_ar_group_aware_skip=False) "
                "or drop %r from extreme_ar_group_aware_skip_models.",
                mlframe_model_name,
                cur_target_name,
                _model_idx_in_run + 1,
                _total_models_in_run,
                float(_ea_lag1),
                _mlp_ea_thr,
                mlframe_model_name,
            )
            return True, _mlp_extreme_ar_fired, _mlp_ea_lag1
    return False, _mlp_extreme_ar_fired, _mlp_ea_lag1


def _run_per_model_post_train_tail(
    *,
    behavior_config,
    test_df_transformed,
    current_test_target,
    ctx,
    target_type,
    cur_target_name: str,
    mlframe_model_name: str,
    metadata: dict,
    test_df_pd,
    _train_idx,
) -> None:
    """Run the per-model tail that fires right after a successful ``process_model``.

    Mutates ``metadata`` (uncertainty_eval / composite y-scale emit) and reclaims RAM in
    place; returns nothing. Behaviour is byte-for-byte the inline blocks it replaced.
    """
    # Opt-in (B): TTA predictive-uncertainty quality on the model-ready transformed test frame
    # (live here alongside the just-trained model entry). Regression + numeric features only.
    if getattr(behavior_config, "uncertainty_eval", False) and test_df_transformed is not None and current_test_target is not None:
        try:
            _ue_ents = (ctx.models.get(str(target_type)) or {}).get(cur_target_name) or []
            _ue_e = _ue_ents[-1] if _ue_ents else None
            _ue_e = _ue_e[0] if isinstance(_ue_e, tuple) and _ue_e else _ue_e
            _ue_model = getattr(_ue_e, "model", None) if _ue_e is not None else None
            if _ue_model is not None and hasattr(_ue_model, "predict") and getattr(_ue_e, "test_probs", None) is None:
                from .._uncertainty_eval import _narrow_numeric_frame, evaluate_tta_quality

                _fni = getattr(_ue_model, "feature_names_in_", None)
                _ue_cols = list(_fni) if _fni is not None else []
                if not _ue_cols:
                    _ue_cols = [c for c in list(getattr(test_df_transformed, "columns", []) or []) if c != cur_target_name]
                _ue_X = _narrow_numeric_frame(test_df_transformed, _ue_cols) if _ue_cols else None
                if _ue_X is not None:
                    _ue_y = np.asarray(current_test_target.values if hasattr(current_test_target, "values") else current_test_target, dtype=np.float64).reshape(
                        -1
                    )
                    if _ue_y.shape[0] == _ue_X.shape[0]:
                        import pandas as _ue_pd

                        _ue_rep = evaluate_tta_quality(
                            lambda Z, _m=_ue_model, _c=list(_ue_cols): np.asarray(_m.predict(_ue_pd.DataFrame(Z, columns=_c))).reshape(-1),
                            _ue_X,
                            _ue_y,
                        )
                        _ue_key = f"{target_type}/{cur_target_name}/{getattr(_ue_e, 'model_name', '') or mlframe_model_name}"
                        metadata.setdefault("uncertainty_eval", {})[_ue_key] = {"test": _ue_rep}
        except Exception as _ue_err:
            logger.warning("[uncertainty_eval] eval failed for %s/%s: %s", target_type, cur_target_name, _ue_err)
    # Per-model IMMEDIATE y-scale emit for composite targets:
    # composite per-model reporting is suppressed upstream (the
    # T-scale chart is skipped and the T-scale metric line emits
    # no numbers in the original scale). Without this hook the
    # operator gets no per-model feedback for composites until
    # the end-of-target wrap-pass. Idempotent + non-fatal: the
    # end-of-target pass still runs the full train/val/test
    # metrics block + watchdog on the (now wrapped) entry.
    try:
        from ..composite.transforms import is_composite_target_name as _is_comp

        if _is_comp(cur_target_name):
            _specs = (metadata.get("composite_target_specs") or {}).get(str(target_type)) or {}
            _spec_pair = None
            for _orig_n, _spec_list in _specs.items():
                for _s in _spec_list or ():
                    if isinstance(_s, dict) and _s.get("name") == cur_target_name:
                        _spec_pair = (_orig_n, _s)
                        break
                if _spec_pair is not None:
                    break
            _entries_for_target = (ctx.models.get(str(target_type)) or {}).get(cur_target_name) or []
            if _spec_pair is not None and _entries_for_target:
                _orig_n, _spec = _spec_pair
                _y_full = (getattr(ctx, "target_by_type", {}) or {}).get(str(target_type), {}).get(_orig_n)
                from ._phase_composite_wrapping import (
                    emit_per_model_composite_y_scale_test,
                )

                emit_per_model_composite_y_scale_test(
                    entry=_entries_for_target[-1],
                    composite_spec=_spec,
                    orig_target_name=_orig_n,
                    composite_name=cur_target_name,
                    target_name=cur_target_name,
                    y_full=_y_full,
                    test_idx=getattr(ctx, "test_idx", None),
                    test_df_pd=test_df_pd,
                    train_idx=_train_idx,
                    plot_file=getattr(ctx, "plot_file", None),
                    reporting_config=getattr(ctx, "reporting_config", None),
                )
    except Exception as _pmce:
        logger.warning(
            "per-model composite y-scale emit failed for " "target=%s model=%s (non-fatal): %s",
            cur_target_name,
            mlframe_model_name,
            _pmce,
        )
    # Reclaim this model's transient bloat (float64 pre_pipeline
    # copies, intermediate frames) before the next model so they
    # don't stack with the next fit + the PipelineCache toward OOM.
    # Adaptive: a no-op unless RSS actually grew past the threshold.
    try:
        from ..utils import maybe_clean_ram_adaptive as _mclean

        _mclean()
    except Exception:
        pass
