"""
Phase 6-7: composite-target post-processing.

1. Composite-target wrapping — wraps fitted T-scale models in ``CompositeTargetEstimator``
   so predictions are y-scale, then computes y-scale RMSE/MAE/R² per split.
2. Cross-target ensemble — opt-in ensemble over composite + raw components
   (mean / linear_stack / nnls_stack / oof_weighted).
3. Suite-end dummy-baselines summary — cross-target verdict block.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ..composite.transforms import is_composite_target_name

logger = logging.getLogger(__name__)

# T2#10 2026-05-18 Pack G universal watchdog threshold. ``wrapper.predict(X)``
# is compared against ``transform.inverse(inner.predict(X), base, params)``;
# divergence beyond this fraction of ``y_std`` fires a WARNING.
#
# Choice of 1%: the wrapper applies a y-train clip on inverse() output, so
# out-of-envelope rows can show tiny per-row differences (clip pulled the
# extreme back inside [y_min, y_max] while the reconstructed path didn't).
# 1% of y_std is well below the float64 round-off floor accumulated across
# a typical (n=10^5, transform=linear_residual) split, AND well above the
# clip-induced noise on a normally-distributed y (clip would have to bite
# ~3 sigma rows AND the inverse path miss them, both rare). Wrapper-math
# bugs (entry-mutation cache stale, double-inverse, base mismatch) produce
# divergence in the 5-50% range, comfortably above this threshold.
#
# Tune by raising if a healthy wrapper fires this warning in your data
# (consult the watchdog log line; %_of_y_std is included so the threshold
# can be set just above the observed noise floor).
_WATCHDOG_RELATIVE_THRESHOLD = 0.01


from ._phase_composite_post_lag_predict import _LagPredictDeployableModel  # noqa: E402, F401
from ._phase_composite_wrapping import _run_composite_target_wrapping  # noqa: F401, E402


def recover_composite_y_scale_metrics(
    *,
    models: dict,
    metadata: dict,
    target_by_type: dict,
    composite_specs_by_target_type: dict,
    filtered_train_idx,
    filtered_train_df,
    filtered_val_idx,
    filtered_val_df,
    test_idx,
    test_df_pd,
    enable_watchdog: bool = True,
) -> dict[tuple, np.ndarray]:
    """T1#7 2026-05-18 lazy recovery of composite-target y-scale metrics.

    When the suite runs with ``skip_wrap_pass_predict=True`` (default since
    2026-05-18), the wrap step still runs but the y-scale metric block is
    bypassed - ``metadata["composite_target_y_scale_metrics"]`` stays empty.

    Callers that subsequently need those metrics (notebooks, dashboards,
    downstream audits) invoke this helper. It walks the already-wrapped
    ``models`` dict and computes RMSE/MAE/R2 per (composite_name, split).
    The metadata dict is populated in place with the same shape as the
    eager path; the train-prediction cache is returned so subsequent
    cross-target ensemble work reuses the freshly-computed predictions.

    Idempotent: when the wrap step in ``_run_composite_target_wrapping``
    detects an entry whose inner is already a ``CompositeTargetEstimator``
    it skips re-wrapping, so callers can invoke this helper safely after
    the eager path has already run.
    """
    return _run_composite_target_wrapping(
        models=models,
        metadata=metadata,
        target_by_type=target_by_type,
        composite_specs_by_target_type=composite_specs_by_target_type,
        filtered_train_idx=filtered_train_idx,
        filtered_train_df=filtered_train_df,
        filtered_val_idx=filtered_val_idx,
        filtered_val_df=filtered_val_df,
        test_idx=test_idx,
        test_df_pd=test_df_pd,
        skip_predict=False,
        enable_watchdog=enable_watchdog,
    )


# _run_suite_end_dummy_baselines_summary moved to sibling; re-exported below.


def run_composite_post_processing(
    *,
    models: dict,
    metadata: dict,
    target_by_type: dict,
    composite_target_discovery_config,
    target_name: str,
    model_name: str,
    filtered_train_df,
    filtered_val_df,
    test_df_pd,
    filtered_train_idx,
    filtered_val_idx,
    test_idx,
    train_df_pd,
    val_df_pd,
    train_idx,
    val_idx,
    dummy_baselines_config,
    reporting_config,
    plot_file: str | None,
    verbose: bool,
    ctx: Any = None,
) -> tuple[dict, dict]:
    """Run composite wrapping, cross-target ensemble, and suite-end summary.

    ``ctx`` (the suite TrainingContext) carries ``timestamps`` / ``sample_weights`` / ``group_ids`` aligned to
    full-data row indices; the cross-target ensemble builder subsets them by ``filtered_train_idx`` so the honest
    OOF split can be time-aware, weighted, and group-aware. Returns updated (models, metadata).
    """
    # Composite-target wrapping: T-scale inner models get wrapped so predict() returns y-scale.
    composite_specs_by_target_type = metadata.get("composite_target_specs", {}) or {}
    # Train-prediction cache (key = id(wrapper)) populated by the wrapping block and reused by the cross-target ensemble block.
    _train_pred_cache: dict[tuple, np.ndarray] = {}
    if composite_specs_by_target_type:
        _skip_predict = bool(getattr(
            composite_target_discovery_config, "skip_wrap_pass_predict", False,
        ))
        _enable_watchdog = bool(getattr(
            composite_target_discovery_config, "enable_wrap_pass_watchdog", True,
        ))
        _train_pred_cache = _run_composite_target_wrapping(
            models=models,
            metadata=metadata,
            target_by_type=target_by_type,
            composite_specs_by_target_type=composite_specs_by_target_type,
            filtered_train_idx=filtered_train_idx,
            filtered_train_df=filtered_train_df,
            filtered_val_idx=filtered_val_idx,
            filtered_val_df=filtered_val_df,
            test_idx=test_idx,
            test_df_pd=test_df_pd,
            skip_predict=_skip_predict,
            enable_watchdog=_enable_watchdog,
            target_name=target_name,
            plot_file=plot_file,
            reporting_config=reporting_config,
        )

    # Cross-target ensemble (opt-in). Stored as a SimpleNamespace under models[type][f"_CT_ENSEMBLE__{original_target}"].
    _ce_strategy = getattr(
        composite_target_discovery_config, "cross_target_ensemble_strategy", "off",
    )
    # Unconditional banner when discovery is enabled so "no log lines" remains a debuggable signal.
    if composite_target_discovery_config.enabled:
        _n_specs_total = sum(
            sum(len(v) for v in _tt_specs.values())
            for _tt_specs in (composite_specs_by_target_type or {}).values()
        )
        logger.info(
            "[CompositeCrossTargetEnsemble] entry: strategy='%s', "
            "target_types=%d, composite_specs=%d",
            _ce_strategy,
            len(composite_specs_by_target_type or {}),
            _n_specs_total,
        )
    # Build CT_ENSEMBLE for raw-target models even when a target had 0
    # composite specs discovered. On extreme-AR + group-aware regression
    # (composite-discovery extreme_ar_group_aware_skip fires; see round
    # 5.3) the affected target gets a ``composite_target_failures`` entry
    # but NO ``composite_target_specs`` entry, so the entry guard below
    # silently bypasses the dummy-floor gate + lag_predict injection,
    # leaving the suite shipping a simple-arithmetic ensemble of the raw
    # models -- which is provably WORSE than the best single component
    # when 3 of 4 boosters are above the lag-predict floor (observed
    # in prod: EnsARITHM TEST=12.45 vs Ridge alone 11.63 vs
    # lag_predict 11.58).
    #
    # I2 fix (2026-06-10): the synthesis must be PER-TARGET, not gated on
    # the GLOBALLY-empty specs dict. In a mixed suite where one regression
    # target was discovered (specs dict non-empty) and a sibling was
    # AR-skipped (no specs entry), the old ``not composite_specs_by_target_type``
    # guard skipped synthesis entirely, so the AR-skipped sibling never got
    # its lag-floor ensemble. We now synthesise a per-target empty-spec
    # entry for EVERY regression target with at least one trained raw model
    # that lacks a specs entry, regardless of whether other targets were
    # discovered, so the loop runs for every such target, lag_predict is
    # injected, and the OOF + dummy-floor + AR(1)-failsafe gates pick the
    # right component.
    _build_for_raw_only = bool(getattr(
        composite_target_discovery_config,
        "always_build_ct_ensemble_for_raw", True,
    ))
    if (composite_target_discovery_config.enabled
            and _ce_strategy != "off"
            and _build_for_raw_only):
        from ..configs import TargetTypes as _TT

        # Merge into a FRESH local dict; never mutate the metadata-owned
        # ``composite_target_specs`` (that dict can back the on-disk discovery
        # cache -- see _phase_composite_discovery.py:439-442 -- and is read by
        # report()/predict()). Shallow-copy each per-target-type sub-dict so the
        # synthesised ``[]`` entries don't leak back into metadata either.
        _merged_specs: dict = {
            _tt_k: dict(_tt_v) for _tt_k, _tt_v in (composite_specs_by_target_type or {}).items()
        }
        # ``TargetTypes`` is a ``StrEnum`` so ``_TT.REGRESSION`` and the
        # ``str(...)``-flavoured key the discovery phase writes are
        # hash-equivalent; ``setdefault`` resolves to the existing regression
        # bucket (if any) rather than creating a duplicate.
        _reg_specs_bucket = _merged_specs.setdefault(_TT.REGRESSION, {})
        _reg_models = (models or {}).get(_TT.REGRESSION, {}) if models else {}
        _n_synth = 0
        for _raw_tname, _entries in _reg_models.items():
            if (_entries
                    and not is_composite_target_name(str(_raw_tname))
                    and _raw_tname not in _reg_specs_bucket):
                _reg_specs_bucket[_raw_tname] = []
                _n_synth += 1
        # Drop an empty regression bucket we created but never populated so the
        # downstream ``if not _tt_specs: continue`` guard isn't tripped by an
        # accidental empty key on a no-regression-model suite.
        if not _reg_specs_bucket and _TT.REGRESSION not in (composite_specs_by_target_type or {}):
            _merged_specs.pop(_TT.REGRESSION, None)
        if _n_synth:
            # Rebind to the merged dict so the loop below sees both the
            # discovered specs AND the synthesised raw-only ``[]`` entries.
            composite_specs_by_target_type = _merged_specs
            logger.info(
                "[CompositeCrossTargetEnsemble] always_build_ct_ensemble_for_raw=True: "
                "synthesised raw-only entries for %d regression target(s) "
                "with trained models but no discovered composite specs; "
                "ensemble loop will inject lag_predict and run the dummy-floor "
                "+ AR(1)-failsafe gates for each.",
                _n_synth,
            )
    if (composite_target_discovery_config.enabled
            and _ce_strategy != "off"
            and composite_specs_by_target_type):
        from ._phase_composite_post_xt_ensemble import _build_cross_target_ensemble_for_target

        for _tt_e, _tt_specs in composite_specs_by_target_type.items():
            if not _tt_specs:
                continue
            # StrEnum: models.get(str_key) is hash-equivalent to models.get(enum_key).
            if _tt_e not in (models or {}):
                logger.info(
                    "[CompositeCrossTargetEnsemble] target_type='%s': no models "
                    "registered; ensemble skipped.", _tt_e,
                )
                continue
            for _orig_tname, _spec_list in _tt_specs.items():
                _build_cross_target_ensemble_for_target(
                    _tt_e=_tt_e,
                    _orig_tname=_orig_tname,
                    _spec_list=_spec_list,
                    _ce_strategy=_ce_strategy,
                    models=models,
                    metadata=metadata,
                    target_by_type=target_by_type,
                    composite_target_discovery_config=composite_target_discovery_config,
                    target_name=target_name,
                    model_name=model_name,
                    filtered_train_df=filtered_train_df,
                    filtered_val_df=filtered_val_df,
                    test_df_pd=test_df_pd,
                    filtered_train_idx=filtered_train_idx,
                    filtered_val_idx=filtered_val_idx,
                    test_idx=test_idx,
                    train_df_pd=train_df_pd,
                    val_df_pd=val_df_pd,
                    train_idx=train_idx,
                    val_idx=val_idx,
                    reporting_config=reporting_config,
                    plot_file=plot_file,
                    _train_pred_cache=_train_pred_cache,
                    ctx=ctx,
                )

    # MoE selection gate + composite VALUE report: this is the one place where the deployed composite ensemble,
    # the raw-y model, the lag failsafe, true y and group_ids coexist on the honest val split. Both are flag-gated
    # and no-op cleanly (deploy byte-identical) when their inputs are missing.
    try:
        from ._phase_composite_post_moe import run_composite_moe_and_value_report
        run_composite_moe_and_value_report(
            models=models,
            metadata=metadata,
            target_by_type=target_by_type,
            composite_target_discovery_config=composite_target_discovery_config,
            filtered_train_df=filtered_train_df,
            filtered_val_df=filtered_val_df,
            filtered_train_idx=filtered_train_idx,
            filtered_val_idx=filtered_val_idx,
            ctx=ctx,
        )
    except Exception as _moe_err:
        logger.warning(
            "[CompositeMoE] value-report / MoE-gate stage failed (%s); deploy left unchanged.", _moe_err,
        )

    _run_suite_end_dummy_baselines_summary(
        models=models,
        metadata=metadata,
        dummy_baselines_config=dummy_baselines_config,
    )

    return models, metadata


from ._phase_composite_post_summary import _run_suite_end_dummy_baselines_summary  # noqa: E402, F401
