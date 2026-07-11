"""Composite target transforms, estimator wrapper, and discovery.

Building blocks for composite-target discovery. This module ships:

1. The transform registry (forward / inverse / fit / domain check).
2. ``CompositeTargetEstimator`` -- sklearn-compatible wrapper that
   hides the transform-and-invert loop from downstream callers.
3. ``CompositeTargetDiscovery`` -- auto-finds the best (base, transform)
   pairs by MI gain over the raw target, with strict train-only
   fitting and forbidden-base filtering.

Concept. A composite target is a transform ``T = f(y, base)`` such
that the model learns ``T`` from features ``X`` (typically excluding
the dominant feature ``base``), and a wrapper applies ``f^{-1}`` at
predict time to recover ``y`` in the original scale. The structural
example: ``y = target`` and ``base = lag_feature``, where the autoregressive
lag is captured natively by the transform and the model is forced to
explain the remaining residual.

Public surface
--------------
- :class:`Transform` -- frozen dataclass, one entry per transform.
- :data:`_TRANSFORMS_REGISTRY` / :func:`get_transform` /
  :func:`list_transforms` -- registry lookup.
- :class:`CompositeTargetEstimator` -- sklearn-compatible wrapper that
  fits an inner regressor on ``T`` and inverts at predict.
- :exc:`DomainViolationError`, :exc:`UnknownTransformError`.
- :class:`CompositeTargetDiscovery` -- auto-finds the best
  ``(base, transform)`` pairs by MI gain.
- :class:`CompositeCrossTargetEnsemble` -- cross-target ensembling.
- :class:`DiscoveryCache` -- caches discovery results across fits.
- :class:`CompositeProvenance` / :func:`report_to_markdown` -- audit
  trail + human-readable discovery report.
- :func:`streaming_alpha_check_and_refit` -- streaming drift refit.
- :func:`bayesian_alpha_fit` -- Bayesian alpha estimation.
- :func:`forward_stepwise_multi_base` -- multi-base forward stepwise.
- :func:`composite_predictions_as_feature` / :func:`composite_oof_predictions`
  -- feature stacking.

Design choices
--------------
- Transforms are looked up by **name** at fit/predict time, never
  stored as per-instance callables. This keeps :func:`sklearn.clone`
  semantics honest, makes pickle work with the standard library
  (no closure traps -> no PII leakage via captured DataFrames), and
  lets the wrapper survive process boundaries (joblib, Optuna).
- Transforms are **frozen**: ``forward``, ``inverse``, ``fit``,
  ``domain_check`` are pure module-level functions registered in
  :data:`_TRANSFORMS_REGISTRY` at import time. Adding a new transform =
  one dataclass entry + one parametrized test row.
- Fitted parameters (``alpha``, ``beta``, MAD floor, post-inverse
  y-clip bounds) are computed **only on training rows passed to
  ``fit``**. The wrapper never re-fits at predict time; downstream
  composite-target discovery is responsible for using the same
  ``train_idx`` discipline at the screening step.
- Numerical safety: MAD-soft-cap with floor (against degenerate
  ``T_train`` collapsing to a constant), post-inverse y-clip to the
  ``[Q001/10, Q999*10]`` bounds of ``y_train`` (against ``exp(...)``
  blow-up in ``logratio``), and ``np.isfinite`` guards on incoming
  ``base`` values at predict (against adversarial ``+inf`` injection).

Out of scope for this module
----------------------------
- ``base_margin`` / classification residuals: regression only here.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# CompositeSpec moved to composite_spec.py (broke circular import with
# composite_discovery). Re-export below preserves callers doing
# ``from mlframe.training.composite import CompositeSpec``.
# ----------------------------------------------------------------------
from .spec import CompositeSpec

# Re-export everything from composite_transforms for full back-compat.
# Existing callers ``from mlframe.training.composite import Transform,
# _TRANSFORMS_REGISTRY, get_transform, ...`` keep working unchanged.
# ----------------------------------------------------------------------
from .transforms import (
    DomainViolationError,
    UnknownTransformError,
    Transform,
    TAG_CORE,
    TAG_EXTENDED,
    TAG_REGRESSION,
    _MAD_FLOOR_FRAC,
    _MAD_SOFT_CAP_K,
    _MULTI_BASE_COND_NUMBER_MAX,
    _GROUPED_MIN_GROUP_SIZE,
    _QUANTILE_RESIDUAL_DEFAULT_N_BINS,
    _QUANTILE_RESIDUAL_DEFAULT_MIN_BIN_N,
    _MONOTONIC_RESIDUAL_DEFAULT_N_KNOTS,
    _MONOTONIC_RESIDUAL_DEFAULT_MIN_KNOT_N,
    _EWMA_RESIDUAL_DEFAULT_K,
    _FRAC_DIFF_DEFAULT_D,
    _FRAC_DIFF_DEFAULT_LAGS,
    _ROLLING_QUANTILE_DEFAULT_K,
    _TRANSFORMS_REGISTRY,
    TRANSFORMS_REGISTRY,
    TRANSFORM_NAME_SHORT,
    compose_target_name,
    get_transform,
    is_composite_target_name,
    list_transforms,
    # Shared helpers used by transforms (tests import some directly).
    _ewma_compute,
    _frac_diff_weights,
    _james_stein_shrinkage_factor,
    _monotonic_residual_g,
    _quantile_residual_assign_bins,
    _rolling_median,
    _row_alpha_beta,
    # 11 transform impls (private helpers; re-exported because tests
    # and a few internal call sites import them by name).
    _diff_forward, _diff_inverse, _diff_fit, _diff_domain,
    _ratio_forward, _ratio_inverse, _ratio_fit, _ratio_domain,
    _logratio_forward, _logratio_inverse, _logratio_fit, _logratio_domain,
    _linear_residual_forward, _linear_residual_inverse, _linear_residual_fit, _linear_residual_domain,
    _linear_residual_multi_forward, _linear_residual_multi_inverse, _linear_residual_multi_fit, _linear_residual_multi_domain,
    _linear_residual_grouped_forward, _linear_residual_grouped_inverse, _linear_residual_grouped_fit, _linear_residual_grouped_domain,
    _quantile_residual_forward, _quantile_residual_inverse, _quantile_residual_fit, _quantile_residual_domain,
    _monotonic_residual_forward, _monotonic_residual_inverse, _monotonic_residual_fit, _monotonic_residual_domain,
    _ewma_residual_forward, _ewma_residual_inverse, _ewma_residual_fit, _ewma_residual_domain,
    _rolling_quantile_ratio_forward, _rolling_quantile_ratio_inverse, _rolling_quantile_ratio_fit, _rolling_quantile_ratio_domain,
    _frac_diff_forward, _frac_diff_inverse, _frac_diff_fit, _frac_diff_domain,
)

# ----------------------------------------------------------------------
# Re-export CompositeTargetEstimator + helpers from composite_estimator
# for full back-compat (callers import from this module).
# ----------------------------------------------------------------------
from .estimator import (
    CompositeTargetEstimator,
    _Y_CLIP_LOW_FRAC,
    _Y_CLIP_HIGH_FRAC,
    _y_train_clip_bounds,
    _to_1d_numpy,
    _extract_base,
    _extract_groups,
    _extract_base_matrix,
    _is_polars_df,
    predict_quantile_ensemble,
)

# ----------------------------------------------------------------------
# Re-export CompositeProvenance + report_to_markdown.
# ----------------------------------------------------------------------
from .provenance import (
    CompositeProvenance,
    _format_transform_formulas,
    report_to_markdown,
)

# ----------------------------------------------------------------------
# Re-export ensemble + OOF + util symbols.
# ----------------------------------------------------------------------
from .ensemble import (
    CompositeCrossTargetEnsemble,
    compute_oof_holdout_predictions,
    derive_seeds,
    detect_gpu_in_use,
    env_signature,
    _is_monotone_nondecreasing,
)

# ----------------------------------------------------------------------
# Re-export screening helpers.
# ----------------------------------------------------------------------
from .discovery.screening import (
    _extract_column_array,
    _is_numeric_column,
    _safe_corr,
    _safe_abs_corr_all,
    _residualise,
    _mi_pair_bin,
    _mi_to_target,
    _silence_tiny_model_output,
    _build_tiny_model,
    _tiny_cv_rmse_raw_y,
    _tiny_cv_rmse_y_scale_multiseed,
    _tiny_cv_rmse_raw_y_multiseed,
    _per_bin_rmse,
    _tiny_cv_rmse_y_scale,
    _sample_indices,
)

# ----------------------------------------------------------------------
# Re-export independent + dependent helper modules.
# ----------------------------------------------------------------------
from .discovery.auto_detect import (
    detect_time_column_candidates,
    sort_df_by_time_column,
    detect_group_column_candidates,
    _GROUP_DETECT_DEFAULT_MIN_UNIQUE,
    _GROUP_DETECT_DEFAULT_MAX_UNIQUE,
    _GROUP_DETECT_DEFAULT_MIN_SIZE_RATIO,
)
from .cache import (
    DiscoveryCache,
    data_signature,
    make_discovery_cache_key,
)
from .ensemble.stacking import (
    residual_correlation_matrix,
    max_off_diagonal_correlation,
    stacking_aware_gate,
    residual_dedup_indices,
)
from .transforms.interaction_bases import (
    generate_interaction_bases,
)
from .streaming import (
    streaming_alpha_check_and_refit,
    _STREAMING_DEFAULT_Z_THRESHOLD,
    _STREAMING_DEFAULT_MIN_BUFFER_N,
)
from .discovery.bayesian import (
    bayesian_alpha_fit,
    bayesian_alpha_fit_bootstrap,
)
from .discovery.forward_stepwise import (
    forward_stepwise_multi_base,
    _MULTI_BASE_DEFAULT_MAX_K,
    _MULTI_BASE_DEFAULT_MIN_MARGINAL_GAIN,
)
from .ensemble.feature_stacking import (
    composite_predictions_as_feature,
    composite_oof_predictions,
)

# ----------------------------------------------------------------------
# Re-export CompositeTargetDiscovery.
# ----------------------------------------------------------------------
from .discovery import CompositeTargetDiscovery

# Re-export the config alongside the discovery class so a single import
# location serves both; same module path the lazy dict-config path uses.
from ..configs import CompositeTargetDiscoveryConfig

# Classification composite (base-margin residual modelling) + split-conformal
# prediction interval helper -- the two FUTURE extensions.
from .classification import CompositeClassificationEstimator
from .conformal import conformal_quantile

# Purged/embargoed time-series CV + base-target-leakage detection.
from .cv import PurgedTimeSeriesSplit, make_purged_cv
from .discovery._leakage import detect_base_target_leakage

# Unified explainability report + composite-vs-raw meta-stacker.
from .report import composite_report
from .meta import CompositeOrRawStacker

# Full predictive-distribution composite (CRPS), joint HPO, model card.
from .distributional import CompositeDistributionEstimator
from .hpo import optimize_composite
from .model_card import composite_model_card

# Auto temporal base engineering, bagged (epistemic) composite, survival/AFT.
from .discovery._base_engineering import engineer_temporal_bases
from .bagging import BaggedCompositeEstimator
from .survival import CompositeSurvivalEstimator
from .dual_direction import DualDirectionCompositeEstimator

# Dependency-light serving export of a fitted composite.
from .serving import export_serving_spec, load_serving_spec

# Champion-challenger comparison, double-ML orthogonalized composite, ranking.
from .compare import compare_models, should_promote
from .orthogonal import OrthogonalizedCompositeEstimator
from .ranking import CompositeRankEstimator

# Panel/longitudinal (entity fixed-effects) composite + extreme-value tail composite.
from .panel import CompositePanelEstimator
from .extremes import TailCompositeEstimator

# Classifier gate + regression blend for zero-inflated/point-mass targets.
from .gated_outlier import GatedOutlierEstimator

# Compute-saving advisory: prune additive-residual transforms for strictly-positive, scale-like pairs.
from .transform_priority import recommend_transform_candidates
from .feature_subset_bagging import FeatureSubsetBaggingEnsemble, correlation_cluster_feature_subsets

# Auxiliary-target OOF meta-features feeding a primary-target model (MoA-style multi-stage stacking).
from .stacking_multi_stage import MultiStageMetaFeatureStacker

# Data-sparse-segment specialist model, rank-spliced back into the main model's predictions.
from .segment_routed import SegmentRoutedEstimator

# One OOF submodel per redundant/correlated feature block (e.g. DCD cluster), stacked by a meta-model.
from .grouped_block_stacking import GroupedBlockStacker

# Row-level (unaggregated child-table) modeling + post-hoc entity-averaging, group-CV leakage-safe.
from .row_level_average import compute_row_level_then_average_predictions

# Rolling self-referential forecast chaining: stage-1 extrapolated onto the target window's own features.
from .chained_window_forecast import ChainedWindowForecaster

# Leakage-safe cross-sectional "macro factor" (group-aggregate prediction) as a per-row feature.
from .group_aggregate_macro import predicted_group_aggregate_feature

# Wide-to-long reshape (value, count, feature-identity) to suppress spurious cross-feature tree interactions.
from .long_format_gbm import melt_to_long_gbm_features

# Direct (non-recursive) multi-horizon forecasting: one model per horizon block, all from origin-time features.
from .direct_multi_horizon import DirectMultiHorizonEnsemble

# One model per market/operating regime, combined by routing or averaging.
from .regime_split_ensemble import RegimeSplitEnsemble

# Leakage-safe semi-supervised self-training: fold-ensemble pseudo-labels, confidence filtering, iterative rounds.
from .pseudo_labeling import PseudoLabelingLoop

# One model per (cross of) explicit categorical segment keys, with incremental add/update/remove lifecycle.
from .segmented_model_factory import SegmentedModelFactory

# Gate classifier hard-routes to branch regressors; gate probability stacked as a feature.
from .gated_regression_mixture import GatedRegressionMixture

# Blend an entity-specific model with a metadata/global model by per-entity observation count.
from .count_weighted_blend import CountWeightedBlendEnsemble

# Shared-trunk NN: primary regression head + weighted auxiliary classification/regression heads, jointly trained.
from .multitask_auxiliary_loss import MultiTaskAuxiliaryLossRegressor

# Missing-aware composite, OOF feature generator, spec stability selection.
from .missing import MissingAwareComposite
from .suite_features import CompositeFeatureGenerator
from .discovery._stability import stability_select_specs

# Compositional (simplex) target composite + quantile-regression-forest distribution.
from .simplex import CompositeSimplexEstimator, aitchison_distance
from .qrf import CompositeQRFEstimator

# Conformal classification prediction SETS (LAC / APS), base-vs-residual
# attribution, and deployed-model drift monitoring.
from .attribution import explain_prediction, attribution_summary
from .monitoring import CompositeDriftMonitor

# GLM-family composite (log-link Poisson / Gamma / Tweedie residual over a base mean).
from .glm import CompositeGLMEstimator

# Native pinball-quantile composite: one inner per quantile fit on the transform
# T, inverted to y-scale, non-crossing enforced per row.
from .quantile import CompositeQuantileEstimator

# Multi-output composite: one CompositeTargetEstimator per column of a vector
# target (n, K), each with its own transform + base; predict returns (n, K).
from .multi_output import (
    CompositeMultiOutputEstimator,
    make_per_column_specs,
)
from .sklearn_compat import (
    make_composite_regressor,
    CompositeTargetTransformer,
)
from .autoconfig import suggest_discovery_config
from .highlevel import discover_and_wrap, DiscoverAndWrapResult

# Composite VALUE report (per-group did-it-help), the not-worse-than-lag MoE selection gate, and the Winkler interval score.
from ._value_report import build_composite_value_report, render_composite_value_report
from ._moe_gate import MoESelectionGate
from ._regime_headroom import regime_headroom_map, render_regime_headroom_map
from ._winkler import (
    winkler_interval_score, winkler_score_per_row, winkler_score_per_group,
    mean_coverage, interval_quality_summary,
)
from ._heteroscedastic import HeteroscedasticCompositeEstimator
from ._pseudo_bma import pseudo_bma_weights, blend as pseudo_bma_blend
from .calendar_anomaly import detect_calendar_anomalies, apply_calendar_anomaly_flag

# Curated public surface for ``from ...composite import *`` -- excludes the
# submodule names + stdlib leakage (logging / annotations) that bare star-import
# would otherwise pull in. Direct ``from ...composite import <submodule>`` and
# any underscore-prefixed internal symbol still resolve; __all__ governs only
# the star-import set.
__all__ = [
    # estimators / discovery / ensemble
    "CompositeTargetEstimator", "CompositeClassificationEstimator",
    "CompositeGLMEstimator", "CompositeQuantileEstimator",
    "CompositeMultiOutputEstimator", "make_per_column_specs",
    "make_composite_regressor", "CompositeTargetTransformer",
    "suggest_discovery_config", "discover_and_wrap", "DiscoverAndWrapResult",
    "explain_prediction", "attribution_summary", "CompositeDriftMonitor",
    "PurgedTimeSeriesSplit", "make_purged_cv", "detect_base_target_leakage",
    "composite_report", "CompositeOrRawStacker",
    "CompositeDistributionEstimator", "optimize_composite", "composite_model_card",
    "GatedOutlierEstimator",
    "MultiStageMetaFeatureStacker",
    "SegmentRoutedEstimator",
    "GroupedBlockStacker",
    "compute_row_level_then_average_predictions",
    "ChainedWindowForecaster",
    "predicted_group_aggregate_feature",
    "melt_to_long_gbm_features",
    "DirectMultiHorizonEnsemble",
    "RegimeSplitEnsemble",
    "PseudoLabelingLoop",
    "SegmentedModelFactory",
    "GatedRegressionMixture",
    "CountWeightedBlendEnsemble",
    "MultiTaskAuxiliaryLossRegressor",
    "engineer_temporal_bases", "BaggedCompositeEstimator", "CompositeSurvivalEstimator",
    "export_serving_spec", "load_serving_spec",
    "compare_models", "should_promote", "OrthogonalizedCompositeEstimator",
    "CompositeRankEstimator", "CompositePanelEstimator", "TailCompositeEstimator",
    "MissingAwareComposite", "CompositeFeatureGenerator", "stability_select_specs",
    "CompositeSimplexEstimator", "CompositeQRFEstimator", "aitchison_distance",
    "build_composite_value_report", "render_composite_value_report", "MoESelectionGate",
    "regime_headroom_map", "render_regime_headroom_map",
    "winkler_interval_score", "winkler_score_per_row", "winkler_score_per_group",
    "mean_coverage", "interval_quality_summary",
    "HeteroscedasticCompositeEstimator", "pseudo_bma_weights", "pseudo_bma_blend",
    "detect_calendar_anomalies", "apply_calendar_anomaly_flag",
    "CompositeTargetDiscovery", "CompositeTargetDiscoveryConfig",
    "CompositeCrossTargetEnsemble", "CompositeSpec", "CompositeProvenance",
    "DiscoveryCache",
    # transforms + registry
    "Transform", "get_transform", "list_transforms", "TRANSFORMS_REGISTRY",
    "compose_target_name", "is_composite_target_name", "TRANSFORM_NAME_SHORT",
    "TAG_CORE", "TAG_EXTENDED", "TAG_REGRESSION", "generate_interaction_bases",
    # uncertainty
    "conformal_quantile",
    # ensemble / OOF / stacking
    "compute_oof_holdout_predictions", "composite_oof_predictions",
    "composite_predictions_as_feature", "predict_quantile_ensemble",
    "stacking_aware_gate",
    # discovery helpers
    "forward_stepwise_multi_base", "detect_group_column_candidates",
    "detect_time_column_candidates", "sort_df_by_time_column",
    "residual_dedup_indices", "residual_correlation_matrix",
    "max_off_diagonal_correlation", "bayesian_alpha_fit",
    "bayesian_alpha_fit_bootstrap", "streaming_alpha_check_and_refit",
    "derive_seeds",
    # cache / provenance / reporting
    "data_signature", "make_discovery_cache_key", "env_signature",
    "detect_gpu_in_use", "report_to_markdown",
    # errors
    "DomainViolationError", "UnknownTransformError",
    # compute-saving transform-candidate pruning
    "recommend_transform_candidates",
    # correlation-cluster-aware feature-subset bagging
    "FeatureSubsetBaggingEnsemble", "correlation_cluster_feature_subsets",
]
