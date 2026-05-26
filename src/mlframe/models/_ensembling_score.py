"""Main ``score_ensemble`` entry point for ``mlframe.models.ensembling``.

Split out of ``ensembling.py`` to keep the parent below the 1k-line monolith
threshold. The parent re-exports ``score_ensemble`` so historical
``from mlframe.models.ensembling import score_ensemble`` imports continue
to resolve.

Most of the heavy work happens via parent-module helpers
(``compute_member_quality_gate``, ``_process_single_ensemble_method``,
``ensemble_probabilistic_predictions``, ``compute_high_correlation_pairs``,
``_build_votenrank_leaderboard_from_results``); imported lazily here to
dodge the ``ensembling -> _ensembling_score -> ensembling`` import cycle.
"""
from __future__ import annotations

import logging
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

# Shared helpers + constants come from the leaf; sibling-defined dispatchers
# (quality_gate / predict / process_method) come from their own siblings.
# No ``from .ensembling import`` here: the parent re-imports this module at
# its bottom; routing every dependency through leaves breaks the cycle.
from ._ensembling_base import (  # noqa: F401
    SIMPLE_ENSEMBLING_METHODS,
    compute_high_correlation_pairs,
)
from ._ensembling_predict import ensemble_probabilistic_predictions  # noqa: F401 -- re-exported for monkey-patch
from ._ensembling_process_method import _process_single_ensemble_method
from ._ensembling_quality_gate import compute_member_quality_gate  # noqa: F401 -- monkey-patched by tests
from ._ensembling_score_validate import _validate_score_ensemble_inputs
from ._ensembling_score_gate import (
    catastrophic_drop_k2,
    catastrophic_drop_kn,
    select_gate_source_split,
)
from ._ensembling_score_flavours import (
    apply_diversity_drop,
    apply_quality_gate_kn,
    build_member_tag_lists,
    collapse_to_single_flavour_if_identical,
    filter_sign_sensitive_flavours,
    maybe_build_votenrank_leaderboard,
    run_stacking_aware_gate,
)
# ``_build_votenrank_leaderboard_from_results`` lives in ``ensembling.py``
# (defined after this sibling is loaded), so it can only be imported lazily
# inside the call site that uses it.
from joblib import delayed
from pyutilz.parallel import cpu_count_physical, parallel_run
from pyutilz.pythonlib import is_jupyter_notebook

# Use the parent module's logger name so caplog filters on
# ``"mlframe.models.ensembling"`` continue to capture our records.
# The sibling lives at ``mlframe.models._ensembling_score`` but the public
# API surface (and the tests that assert on log lines) all reference the
# parent module name.
logger = logging.getLogger("mlframe.models.ensembling")


def score_ensemble(
    models_and_predictions: Sequence,
    ensemble_name: str,
    target: pd.Series = None,
    train_idx: np.ndarray = None,
    test_idx: np.ndarray = None,
    val_idx: np.ndarray = None,
    df: pd.DataFrame = None,
    train_target: pd.Series = None,
    test_target: pd.Series = None,
    val_target: pd.Series = None,
    target_label_encoder: object = None,
    # Outlier-member-filter thresholds. The historical absolute defaults
    # (``max_mae=0.05``, ``max_std=0.06``) excluded all 6 members of a
    # uniform tree-model suite (CB / XGB / LGB x 2 weight schemas) on
    # the 2026-04-24 prod log -- turning the filter into a no-op + 36
    # noisy WARN lines per ensemble. Defaults flipped to relative
    # (``2.5xmedian``); pass non-zero ``max_mae`` / ``max_std`` to keep
    # the legacy behaviour.
    max_mae: float = 0.0,
    max_std: float = 0.0,
    max_mae_relative: float = 2.5,
    max_std_relative: float = 2.5,
    ensure_prob_limits: bool = True,
    nbins: int = 100,
    ensembling_methods=SIMPLE_ENSEMBLING_METHODS,
    uncertainty_quantile: float = 0.1,
    normalize_stds_by_mean_preds: bool = False,
    custom_ice_metric: Callable = None,
    custom_rice_metric: Callable = None,
    subgroups: dict = None,
    max_ensembling_level: int = 1,
    n_features: int = None,
    n_jobs: int = None,
    min_samples_for_parallel: int = 10_000_000,
    verbose: bool = True,
    flag_degenerate_conf_subset: bool = True,
    degenerate_class_ratio: float = 0.01,
    diversity_corr_warn_threshold: float = 0.98,
    # NO-SW / NO-GROUPS: per-row weights and group identifiers, plumbed through the quality gate,
    # diversity check, member-quality metric aggregation, and downstream weight-fit. Both default
    # to None to preserve legacy unweighted-i.i.d. semantics; ctx auto-passes when available.
    sample_weight: Optional[np.ndarray] = None,
    group_ids: Optional[np.ndarray] = None,
    rrf_k: int = 60,
    # NO-GUARD-IDENTICAL: short-circuit when every member's predictions on the gate split match
    # numerically (Pearson corr == 1.0 AND elementwise close). One arithmetic-mean ensemble is
    # returned to skip every redundant flavour. Disabled by default so legacy reports keep their
    # shape; opt in via the suite caller.
    early_exit_if_identical: bool = False,
    # GATE-DOUBLE-DIP: when True, the quality-gate source is restricted to OOF predictions; legacy
    # callers that only stamped val_/test_/train_ preds fall through to the disabled gate path.
    # C-P1-1: default flipped to True. Pre-fix the gate silently fell through to ``val_preds``
    # (the same surface early-stopping already burned) for any member without OOF, biasing the
    # gate-survivors selection. The suite caller (_phase_train_one_target.py) NEVER overrode this
    # so every default suite ran with a val-biased gate. Setting to False explicitly re-enables
    # the legacy fallback chain (oof -> val -> test -> train) when the suite has not stamped OOF.
    require_oof_for_gate: bool = True,
    # COARSE-GATE-FALLBACK: when require_oof_for_gate=True AND OOF is unavailable, the strict gate
    # skips entirely. That's the right call for FINE thresholds (2.5x median), but it lets
    # CATASTROPHIC outliers survive: 2026-05-21 prod log had an MLP with R^2=-4.75 sitting in the
    # ensemble alongside three R^2~0.99 members because no member stamped OOF. This fallback runs
    # a SECOND gate at a much higher relative threshold (5x median by default) against the
    # val/test/train fallback chain -- enough to drop the catastrophic disasters while leaving
    # honest near-median members alone. Setting to <=0 disables the coarse fallback entirely.
    coarse_gate_max_mae_relative: float = 5.0,
    coarse_gate_max_std_relative: float = 5.0,
    # K2-CATASTROPHIC-DROPOUT: when K == 2, the peer-median gate is symmetric
    # (both members are equidistant from (a+b)/2 by construction), so the
    # legacy K=2 branch returned kept-all unconditionally. TVT-2026-05-21 had
    # Ridge MAE=7.89 alongside MLP MAE=11442 (ratio = 1450x); the ensemble
    # arithm-mean was MAE=5720 -- half-broken. When true target is available
    # for the gate-source split, this NEW gate compares per-member MAE-to-target
    # directly and drops the obvious catastrophic outlier (ratio >= threshold).
    # Conservative default 20.0 -- only catches disasters, not normal variance
    # between honest models (Ridge vs LightGBM typically differ by <2x MAE).
    # Set <= 1.0 to disable.
    k2_catastrophic_mae_ratio: float = 20.0,
    # VOTENRANK: build a votenrank.Leaderboard over the resulting per-flavour metrics and stamp it
    # in the returned dict under ``_leaderboard``. Defaults True for classification; regression-only
    # flavours skip rank-based methods automatically.
    build_votenrank_leaderboard: bool = True,
    # Stacking-aware gate hook. When True, runs the NNLS-weight gate from composite_stacking on the
    # ensemble's OOF predictions and persists the survivors / weights under ``_stacking_gate``. The
    # gate is observational unless the suite caller wires it into a follow-up linear stack.
    # C-P1-5: default flipped to True. The gate is observational (does not drop members) and
    # surfaces per-member NNLS-weight info operators want for audit; previously this required the
    # suite caller to opt in, so the default suite path lost the diagnostics entirely.
    enable_stacking_aware_gate: bool = True,
    stacking_gate_min_weight: float = 0.05,
    # AP7: when True, the NNLS weights computed by ``stacking_aware_gate`` are fed into
    # ``combine_probs`` as ``precomputed_weights`` (replacing the uniform 1/M weight on
    # arithm / harm / quad / qube / geo flavours). Default True -- the gate already runs
    # observationally so wiring its output into the blend is the natural finalisation. Set
    # False to restore the legacy uniform-mean behaviour while keeping the NNLS diagnostic.
    use_nnls_weights: bool = True,
    # P1-7: optional auto-drop of one member from each high-correlation pair.
    # ``None`` preserves the observational-only default (just WARNs + stamps to _diversity);
    # passing a float in (0, 1] activates auto-drop when any pair's |corr| exceeds the floor.
    # The MEMBER WITH HIGHER MEAN ABSOLUTE GATE-METRIC (mae from the gate) is dropped, so the
    # surviving member is the one closer to the median.
    auto_drop_diversity_above: Optional[float] = None,
    **kwargs,
):
    """Compares different ensembling methods for a list of models.

    Parameters
    ----------
    n_jobs : int, optional
        Number of parallel jobs. If None, automatically determined based on
        sample count and min_samples_for_parallel. Use 1 for sequential processing.
    min_samples_for_parallel : int, default=1_000_000
        Minimum number of samples required to enable parallel processing when n_jobs is None.
    """

    level_models_and_predictions = models_and_predictions
    res, is_regression, ensembling_methods, ensure_prob_limits = _validate_score_ensemble_inputs(
        level_models_and_predictions=level_models_and_predictions,
        ensembling_methods=ensembling_methods,
        ensure_prob_limits=ensure_prob_limits,
        max_ensembling_level=max_ensembling_level,
        verbose=verbose,
    )
    if res:
        return res

    # Determine sample count for parallelization decision
    first_pred = level_models_and_predictions[0]
    if first_pred.val_probs is not None:
        n_samples = len(first_pred.val_probs)
    elif first_pred.val_preds is not None:
        n_samples = len(first_pred.val_preds)
    else:
        n_samples = 0

    # Determine n_jobs if not specified
    effective_n_jobs = n_jobs
    if effective_n_jobs is None:
        if n_samples >= min_samples_for_parallel and not is_jupyter_notebook():
            effective_n_jobs = min(len(ensembling_methods), cpu_count_physical())
        else:
            effective_n_jobs = 1

    # Convert pandas Series to numpy arrays before parallel section to avoid pickling issues
    train_target_arr = train_target.to_numpy() if isinstance(train_target, pd.Series) else train_target
    test_target_arr = test_target.to_numpy() if isinstance(test_target, pd.Series) else test_target
    val_target_arr = val_target.to_numpy() if isinstance(val_target, pd.Series) else val_target
    target_arr = target.to_numpy() if isinstance(target, pd.Series) else target

    # ONE-pass member quality gate before iterating ensemble flavors. The previous behaviour ran the same outlier
    # filter inside ``ensemble_probabilistic_predictions`` once per flavor x split, which on a 4-model x 5-flavor x
    # (full+conf) x 2-split layout printed the same "ens member N excluded ..." line ~20x per suite call. Compute
    # ONCE here, log the decision once, then pass only kept members to the flavor loop and disable the embedded
    # filter so no duplicate prints fire.
    #
    # Source ordering: OOF preds/probs come FIRST -- the gate's job is to drop members whose preds are outliers vs
    # the ensemble median, and val_preds are already burned for early-stopping (gating on them double-dips val).
    # OOF preds are the only honest train-side signal (cross_val_predict held-out rows). Fallback chain: oof_* ->
    # val_* -> test_* -> train_* preserves the legacy behaviour for members trained without oof_n_splits.
    (
        _gate_preds_for_check,
        _gate_source_split,
        _coarse_gate_active,
        max_mae,
        max_std,
        max_mae_relative,
        max_std_relative,
    ) = select_gate_source_split(
        level_models_and_predictions=level_models_and_predictions,
        require_oof_for_gate=require_oof_for_gate,
        coarse_gate_max_mae_relative=coarse_gate_max_mae_relative,
        coarse_gate_max_std_relative=coarse_gate_max_std_relative,
        max_mae=max_mae,
        max_std=max_std,
        max_mae_relative=max_mae_relative,
        max_std_relative=max_std_relative,
        verbose=verbose,
    )

    # 2026-05-11 (user request): TWO tag lists:
    # 1. ``_ensemble_member_tags`` -- full (shim-stripped) class / model names for the per-member quality-gate log line (operators want to see which exact model class was excluded).
    # 2. ``_ensemble_short_tags`` -- collapsed short tags (``cb`` / ``xgb`` / ``lgb`` / ``hgb`` / non-tree class name) for the rebuilt ensemble label after the gate. Without the short-collapse, the rebuilt label reads ``[CatBoostRegressor+XGBRegressor+LGBMRegressor]`` (38 chars) instead of ``[cb+xgb+lgb]`` (12 chars) -- bloated chart titles + breaks the original short-label contract from core.py.
    _ensemble_member_tags, _ensemble_short_tags = build_member_tag_lists(level_models_and_predictions)

    (
        level_models_and_predictions,
        _gate_preds_for_check,
        _ensemble_member_tags,
        _ensemble_short_tags,
    ) = catastrophic_drop_kn(
        level_models_and_predictions=level_models_and_predictions,
        _gate_preds_for_check=_gate_preds_for_check,
        _gate_source_split=_gate_source_split,
        _ensemble_member_tags=_ensemble_member_tags,
        _ensemble_short_tags=_ensemble_short_tags,
        train_target_arr=train_target_arr,
        val_target_arr=val_target_arr,
        test_target_arr=test_target_arr,
        k2_catastrophic_mae_ratio=k2_catastrophic_mae_ratio,
        verbose=verbose,
        res=res,
    )

    (
        level_models_and_predictions,
        _ensemble_member_tags,
        _ensemble_short_tags,
        ensemble_name,
        _k2_early_return,
    ) = catastrophic_drop_k2(
        level_models_and_predictions=level_models_and_predictions,
        _gate_preds_for_check=_gate_preds_for_check,
        _gate_source_split=_gate_source_split,
        _ensemble_member_tags=_ensemble_member_tags,
        _ensemble_short_tags=_ensemble_short_tags,
        ensemble_name=ensemble_name,
        train_target_arr=train_target_arr,
        val_target_arr=val_target_arr,
        test_target_arr=test_target_arr,
        k2_catastrophic_mae_ratio=k2_catastrophic_mae_ratio,
        verbose=verbose,
        res=res,
    )
    if _k2_early_return:
        return res

    (
        level_models_and_predictions,
        _ensemble_member_tags,
        _ensemble_short_tags,
        ensemble_name,
        max_mae,
        max_std,
        max_mae_relative,
        max_std_relative,
        _gate_stats,
    ) = apply_quality_gate_kn(
        level_models_and_predictions=level_models_and_predictions,
        _gate_preds_for_check=_gate_preds_for_check,
        _gate_source_split=_gate_source_split,
        _ensemble_member_tags=_ensemble_member_tags,
        _ensemble_short_tags=_ensemble_short_tags,
        ensemble_name=ensemble_name,
        max_mae=max_mae,
        max_std=max_std,
        max_mae_relative=max_mae_relative,
        max_std_relative=max_std_relative,
        sample_weight=sample_weight,
        group_ids=group_ids,
        ensembling_methods=ensembling_methods,
        res=res,
        verbose=verbose,
        # Pass the parent-module bound name so tests that monkey-patch
        # ``mlframe.models._ensembling_score.compute_member_quality_gate`` still
        # see their spy hit. Resolved here per-call (not closed over) so the
        # patch can be installed AFTER the helper module has been imported.
        compute_member_quality_gate_fn=compute_member_quality_gate,
    )

    # Observational diversity check: pairs of kept members whose val-pred Pearson correlation exceeds the threshold are
    # surfaced via WARN + persisted to the returned dict under ``_diversity.high_correlation_pairs``. Defaults to
    # observational-only (no member removed); pass ``auto_drop_diversity_above`` to actually drop one of each pair.
    _high_corr_pairs, _div_split_used = compute_high_correlation_pairs(
        level_models_and_predictions,
        _ensemble_member_tags,
        threshold=diversity_corr_warn_threshold,
    )
    (
        level_models_and_predictions,
        _ensemble_member_tags,
        _ensemble_short_tags,
        _auto_dropped,
    ) = apply_diversity_drop(
        level_models_and_predictions=level_models_and_predictions,
        _ensemble_member_tags=_ensemble_member_tags,
        _ensemble_short_tags=_ensemble_short_tags,
        _high_corr_pairs=_high_corr_pairs,
        _gate_stats=_gate_stats,
        auto_drop_diversity_above=auto_drop_diversity_above,
        diversity_corr_warn_threshold=diversity_corr_warn_threshold,
        _div_split_used=_div_split_used,
        res=res,
    )

    ensembling_methods = filter_sign_sensitive_flavours(
        ensembling_methods=ensembling_methods,
        is_regression=is_regression,
        level_models_and_predictions=level_models_and_predictions,
        verbose=verbose,
    )

    ensembling_methods = collapse_to_single_flavour_if_identical(
        ensembling_methods=ensembling_methods,
        early_exit_if_identical=early_exit_if_identical,
        _gate_preds_for_check=_gate_preds_for_check,
        level_models_and_predictions=level_models_and_predictions,
        _gate_source_split=_gate_source_split,
        res=res,
        verbose=verbose,
    )

    _nnls_weights_for_blend = run_stacking_aware_gate(
        enable_stacking_aware_gate=enable_stacking_aware_gate,
        _gate_preds_for_check=_gate_preds_for_check,
        target_arr=target_arr,
        level_models_and_predictions=level_models_and_predictions,
        _ensemble_member_tags=_ensemble_member_tags,
        stacking_gate_min_weight=stacking_gate_min_weight,
        use_nnls_weights=use_nnls_weights,
        res=res,
        verbose=verbose,
    )

    for ensembling_level in range(max_ensembling_level):

        next_level_models_and_predictions = []

        # Common parameters for all ensemble methods
        common_params = dict(
            level_models_and_predictions=level_models_and_predictions,
            is_regression=is_regression,
            ensembling_level=ensembling_level,
            ensemble_name=ensemble_name,
            target=target_arr,
            train_idx=train_idx,
            test_idx=test_idx,
            val_idx=val_idx,
            train_target=train_target_arr,
            test_target=test_target_arr,
            val_target=val_target_arr,
            target_label_encoder=target_label_encoder,
            max_mae=max_mae,
            max_std=max_std,
            max_mae_relative=max_mae_relative,
            max_std_relative=max_std_relative,
            ensure_prob_limits=ensure_prob_limits,
            nbins=nbins,
            uncertainty_quantile=uncertainty_quantile,
            normalize_stds_by_mean_preds=normalize_stds_by_mean_preds,
            custom_ice_metric=custom_ice_metric,
            custom_rice_metric=custom_rice_metric,
            subgroups=subgroups,
            n_features=n_features,
            verbose=verbose,
            kwargs=kwargs,
            flag_degenerate_conf_subset=flag_degenerate_conf_subset,
            degenerate_class_ratio=degenerate_class_ratio,
            sample_weight=sample_weight,
            rrf_k=rrf_k,
            precomputed_weights=_nnls_weights_for_blend,
        )

        if len(ensembling_methods) > 1 and effective_n_jobs > 1:
            # loky pickles kwargs across worker boundaries; closure-captured metrics/lambdas
            # blow up in workers. Pre-check so we can fall back to sequential with a clear warning.
            try:
                import pickle

                pickle.dumps((custom_ice_metric, custom_rice_metric, kwargs))
            except (pickle.PicklingError, AttributeError, TypeError) as exc:
                logger.warning(
                    "ensembling: falling back to sequential -- one of " "custom_ice_metric / custom_rice_metric / kwargs is not picklable: %s",
                    exc,
                )
                effective_n_jobs = 1

        if len(ensembling_methods) > 1 and effective_n_jobs > 1:
            # Parallel processing -- loky + tiny max_nbytes keeps arrays in-memory (no spill) per pre-existing tuning
            results = parallel_run(
                [delayed(_process_single_ensemble_method)(ensemble_method=method, **common_params) for method in ensembling_methods],
                n_jobs=effective_n_jobs,
                backend="loky",
                max_nbytes="1K",
                verbose=0,
            )
            for internal_method, next_ens_results, conf_results in results:
                res[internal_method] = next_ens_results
                next_level_models_and_predictions.append(next_ens_results)
                if conf_results is not None:
                    res[internal_method + " conf"] = conf_results
        else:
            # Sequential processing
            for ensemble_method in ensembling_methods:
                internal_method, next_ens_results, conf_results = _process_single_ensemble_method(ensemble_method=ensemble_method, **common_params)
                res[internal_method] = next_ens_results
                next_level_models_and_predictions.append(next_ens_results)
                if conf_results is not None:
                    res[internal_method + " conf"] = conf_results

        level_models_and_predictions = next_level_models_and_predictions

    maybe_build_votenrank_leaderboard(res, is_regression=is_regression, build_votenrank_leaderboard_flag=build_votenrank_leaderboard)
    return res

