"""
Configuration classes for mlframe training pipeline.

Uses Pydantic for validation while supporting dict-like instantiation for backward compatibility.
All config classes support lenient validation - inputs are normalized to canonical forms.

``configs.py`` was split into sibling modules to drop below the 1k-line monolith
threshold. Shared constants + the ``TargetTypes`` enum + ``BaseConfig`` live in
the leaf module ``_configs_base.py`` so the sibling config files can import
their dependencies from a leaf instead of from this module (which would close
a ``configs -> sibling -> configs`` import cycle). Re-exported here for
backward-compat: every historical
``from mlframe.training.configs import TargetTypes`` etc. resolves identity-equal.
"""

from __future__ import annotations


from typing import Optional, Dict, Any, Tuple, Literal, FrozenSet

# Shared constants + TargetTypes enum + BaseConfig live in the leaf module so
# sibling configs can import them without re-entering this module (which would
# close a cycle).
from ._configs_base import (  # noqa: F401
    DEFAULT_RANDOM_SEED,
    DEFAULT_TREE_ITERATIONS,
    DEFAULT_CALIBRATION_BINS,
    DEFAULT_FAIRNESS_MIN_POP_CAT_THRESH,
    DEFAULT_RFECV_MAX_RUNTIME_MINS,
    DEFAULT_RFECV_CV_SPLITS,
    DEFAULT_RFECV_MAX_NOIMPROVING_ITERS,
    VALID_MODEL_TYPES,
    VALID_LINEAR_MODEL_TYPES,
    VALID_SCALER_NAMES,
    VALID_TASK_TYPES,
    VALID_MATMUL_PRECISIONS,
    StrEnum,
    TargetTypes,
    BaseConfig,
)

# Preprocessing-side configs moved to ``_preprocessing_configs.py``;
# re-exported below so historical
# ``from mlframe.training.configs import PreprocessingConfig`` (and the
# other moved names) imports continue to resolve. See sibling for SSOT.
from ._preprocessing_configs import (  # noqa: F401
    PreprocessingConfig, TrainingSplitConfig, PreprocessingBackendConfig,
    PreprocessingExtensionsConfig, FeatureTypesConfig,
)
# FeatureSelectionConfig moved to ``_feature_selection_config.py`` so siblings
# that need it as a field type can import it without re-entering this module
# (which would close a ``configs <-> sibling`` import cycle).
from ._feature_selection_config import FeatureSelectionConfig  # noqa: E402,F401

# Model + hyperparameter + training-behavior configs moved to
# ``_model_configs.py``; re-exported below so historical
# ``from mlframe.training.configs import ModelConfig`` (and the other moved
# names) imports continue to resolve. See sibling for SSOT.
from ._model_configs import (  # noqa: F401
    ModelConfig, LinearModelConfig, TreeModelConfig, MLPConfig, NGBConfig,
    AutoMLConfig, ModelHyperparamsConfig, TrainingBehaviorConfig,
    MultilabelDispatchConfig, LearningToRankConfig, QuantileRegressionConfig,
    EnsemblingConfig,
)

# Training-runtime + IO configs moved to ``_training_runtime_configs.py``;
# re-exported below so historical
# ``from mlframe.training.configs import TrainingConfig`` (and the other
# moved names) imports continue to resolve. See sibling for SSOT.
from ._training_runtime_configs import (  # noqa: E402,F401
    TrainingConfig, DataConfig, TrainingControlConfig, MetricsConfig,
    FeatureImportanceConfig, OutputConfig, OutlierDetectionConfig,
    SliceStableESConfig,
)


class BaselineDiagnosticsConfig(BaseConfig):
    """Configuration for the auto-baseline diagnostics pass.

    Runs once per (target_type, target_name) before per-target training
    starts. Cheap (~30-90 s on a sampled view of train+val) and reports:

    1. Headline baseline metric (RMSE/MAE/R^2 for regression;
       AUC/logloss for binary). One quick model.
    2. Sequential ablation: drop top-K features by feature_importances_,
       retrain, measure metric delta. Surfaces dominant features.
    3. ``init_score`` baseline for regression: refits a quick LightGBM
       with the top-1 dominant feature passed via init_score (model
       learns only the residual). If this baseline already matches raw
       within ``init_score_optimal_threshold_pct``, composite-target
       discovery is unlikely to add value (recommendation downgrade).
    4. ``composite_recommendation`` flag in {high_potential, marginal,
       unlikely_to_help} consumed by composite-target discovery to gate
       expensive screening loops.

    Default ON for regression and binary classification; multiclass /
    multilabel / LtR / quantile_regression skipped (init_score semantics
    don't carry).
    """

    enabled: bool = True

    # Ablation: drop top-K features ranked by quick-model FI.
    ablation_top_k: int = 5

    # Quick-model knobs. LightGBM is the workhorse: fast, supports
    # init_score natively for regression, robust on cold caches.
    # n_estimators=100 (was 200): the ablation diagnostic exists to find the DOMINANT feature, and
    # _benchmarks/bench_ablation_n_estimators_provisioning.py (6 scenarios x 3 seeds: linear /
    # interaction / redundant / noisy-weak / high-card-cat / mixed-binary) shows the dominant-feature
    # verdict is IDENTICAL to the n_estimators=200 default on 18/18 cells at 0.75x/0.5x/0.25x, with
    # ground-truth recovery unchanged (15/18 at both 200 and 100). The top-k tail wobbles only among
    # effectively-tied weak features -- seed-noise present at the 200 default too (redundant/noisy/
    # high-card top3 already varies across seeds at 200). Per accuracy-then-speed the verdict is equal
    # so speed breaks the tie: 100 buys ~1.8x ablation wall (1.825x bench / 1.78x at 200k+sample_n=50k)
    # with no verdict change. sample_n reductions were ALSO tested and REJECTED -- they flip the
    # dominant feature (16/18, 12/18), so sample_n stays at 50000.
    quick_model_family: Literal["lightgbm"] = "lightgbm"
    quick_model_n_estimators: int = 100
    quick_model_num_leaves: int = 31
    quick_model_learning_rate: float = 0.05

    # init_score baseline: regression-only in MVP. Top-K dominant
    # features are summed; for K=1 the single base is passed as
    # init_score, for K>1 a quick OLS combines them first.
    init_score_top_k: int = 1
    # init_score baseline supports both regression and binary classification.
    # For binary the init_score lives in logit space: top-K dominant
    # features are LR-combined into a probability-scale score, then
    # converted to logit and passed as LightGBM's ``init_score=`` so the
    # booster learns the residual logit.
    init_score_apply_to_target_types: Tuple[str, ...] = (
        "regression", "binary_classification",
    )

    # Sample size for ablation/init-score fits. Capped well below the
    # full 4M-row regime so the diagnostic stays under ~1 minute on
    # large datasets. None means "use full train".
    sample_n: Optional[int] = 50_000

    # Recommendation thresholds (in PERCENT of headline metric).
    # Ablation Δ% is computed as (metric_after_drop / metric_raw - 1) * 100.
    # Higher Δ% means the dropped feature contributed more.
    high_potential_min_dominance_pct: float = 5.0  # >5pct from any top-K feature -> dominant
    init_score_optimal_threshold_pct: float = 1.0  # init_score within 1pct of raw -> already optimal
    marginal_threshold_pct: float = 2.0  # 2pct <= dominance < 5pct -> marginal

    # Higher-is-better metrics (AUC) flip the sign convention for
    # ablation Δ% computation. Auto-derived from target_type at runtime.
    apply_to_target_types: Tuple[str, ...] = ("regression", "binary_classification")

    random_state: int = DEFAULT_RANDOM_SEED


class DummyBaselinesConfig(BaseConfig):
    """Configuration for the pre-training Dummy-baseline report.

    Runs once per (target_type, target_name) AFTER ``BaselineDiagnostics``
    and BEFORE the per-model training loop. Computes a tabular comparison
    of trivial / dummy baselines (mean / median / prior / per_group_mean
    / TS-naive / seasonal-naive / ...) on val + test, picks the strongest
    by a target-type-specific primary metric, and emits one overlay plot
    for the strongest baseline only.

    Sit-alongside relationship with ``BaselineDiagnosticsConfig``: that
    class answers "is the target predictable from these features at
    all?" (LightGBM quick fit + feature ablation); this class answers
    "is the task even hard?" (vs trivial reference predictors).

    Operator contract (per plan v3):
    - Default INFO output: ≤ 2 lines per target (verdict + plot path);
      full table demoted to DEBUG.
    - Suite-end summary block with cross-target verdict table.
    - Four canonical UPPERCASE WARN tokens for grep-able alerts:
      ``BEST_MODEL_BELOW_DUMMY``, ``ALL_BASELINES_BELOW_RANDOM``,
      ``TS_BEATS_TREES``, ``PARTIAL_FAILURE``.
    """

    enabled: bool = True

    # Route the strongest dummy baseline through the standard report_model_perf pipeline
    # (the same scatter / residual / calibration charts + metric-title headers real models
    # get) so the no-model floor appears in the operator's familiar report format. Default ON.
    plot_strongest: bool = True

    # Render the dedicated single-figure pre-training overlay (predictions-vs-actual scatter +
    # residual histogram for regression; class-prior bar for classification) via
    # plot_best_dummy_baseline_overlay, saved next to the standard reports. Default OFF: the
    # standard per-model charts (plot_strongest) already cover the floor, so this extra PNG was
    # deliberately removed from the suite; opt in when a one-glance overlay is wanted.
    overlay_plot: bool = False

    # Per-target-type opt-out. Default: every supported target type
    # gets baselines. Operator can disable for specific types via
    # ``apply_to_target_types - {"learning_to_rank"}`` etc.
    apply_to_target_types: FrozenSet[str] = frozenset({
        "regression", "binary_classification", "multiclass_classification",
        "multilabel_classification", "learning_to_rank", "quantile_regression",
    })

    # Time-series baseline knobs (only fire when ``ts_field`` is set on
    # the FTE AND the train/val/test split is temporally monotonic).
    # ``ts_extra_periods`` lets the user inject domain-known seasonal
    # periods (e.g. 17-day biological cycles, 90-day quarterly cycles)
    # that the auto-step-size + ACF detector would miss.
    ts_extra_periods: Tuple[int, ...] = ()

    # Per-group baseline (per_group_mean / per_group_prior) leakage
    # defenses (round-3 audit D1).
    # - ``per_group_max_cardinality_ratio``: skip the baseline if the
    #   chosen categorical's unique-count > (n_train * this ratio).
    #   Default 0.5 catches row-id-like keys (user_id, transaction_id,
    #   hash) that would silently produce perfect-prediction oracles.
    # - ``per_group_min_val_coverage_pct``: exclude per_group_* from
    #   strongest-pick eligibility if val coverage of the chosen cat
    #   falls below this. Below 50%, the metric is dominated by
    #   unseen-category fallback (= train_y.mean()) and not by the
    #   group-conditioning effect.
    # - ``per_group_high_overlap_threshold``: if more than this fraction
    #   of val rows have a group with ≥5 train labels, log the
    #   row-label annotation "(high entity overlap — measures
    #   re-appearance, not generalization)".
    per_group_max_cardinality_ratio: float = 0.5
    per_group_min_val_coverage_pct: float = 50.0
    per_group_high_overlap_threshold: float = 0.5

    # n_repeats for stochastic baselines (round-3 audit C#2, C#5).
    # Single-realization variance dominates the AUC / NDCG estimate at
    # small n_val; reporting mean ± std across deterministic seeds
    # gives the operator a noise-floor anchor.
    stratified_n_repeats: int = 20
    random_within_query_n_repeats: int = 10

    # Strongest-pick robustness gate (round-3 D2).
    # Paired bootstrap on the strongest-vs-runner-up baseline pair;
    # if P(strongest beats runner-up) falls below this, annotate the
    # log line as "(TIE)" and suppress the overlay plot.
    paired_bootstrap_n_resamples: int = 1000
    strongest_min_beat_runner_up_prob: float = 0.7

    # Bootstrap CI on the strongest baseline's primary metric, fired
    # only when ``min(n_val, n_test) < bootstrap_ci_threshold`` (point
    # estimate is accurate to <1% above this threshold; CI suppressed
    # to keep output uncluttered).
    bootstrap_ci_threshold: int = 2000
    # 2000 resamples: percentile-CI bound MC jitter scales ~1/sqrt(B); bench
    # (bench_bootstrap_ci_n_resamples) shows B=2000 lowers seed-to-seed CI
    # wobble from ~4.2% to ~3.0% of half-width across 15 cells (all wins),
    # for ~+0.9ms once per small-n target. See CHANGELOG.
    bootstrap_ci_n_resamples: int = 2000

    # Auto-WARN trigger: model lift below this multiplier vs strongest
    # dummy baseline → ``BEST_MODEL_BELOW_DUMMY`` warning emitted in
    # the suite-end summary. 1.5x is the canonical "your model isn't
    # better than random" Kaggle threshold; can be tightened or
    # loosened per deployment.
    best_model_min_lift: float = 1.5

    # Random seed for stochastic baselines + bootstrap (combined with
    # per-target hash internally to ensure independence across
    # targets — round-3 D13).
    random_state: int = DEFAULT_RANDOM_SEED


# ``_REPORTING_ALLOWED_TITLE_TOKENS`` moved to ``_reporting_configs.py`` along
# with the ``ReportingConfig`` class that consumes it. Re-exported below so
# downstream consumers that imported the private constant from this module
# still resolve.


# Reporting / naming / fairness / container configs moved to
# ``_reporting_configs.py``; re-exported below so historical
# ``from mlframe.training.configs import ReportingConfig`` (and the other
# moved names) imports continue to resolve. See sibling for SSOT.
from ._reporting_configs import (  # noqa: F401
    ReportingConfig, ConfidenceAnalysisConfig, ConformalConfig, RegressionCalibrationConfig, NamingConfig,
    PredictionsContainer, FairnessConfig,
    _REPORTING_ALLOWED_TITLE_TOKENS,
)
# CompositeTargetDiscoveryConfig was the first symbol carved out to a
# sibling. Re-exported here so historical
# ``from mlframe.training.configs import CompositeTargetDiscoveryConfig``
# imports continue to resolve. See sibling for SSOT.
from ._composite_target_discovery_config import CompositeTargetDiscoveryConfig  # noqa: F401


# Helper function to create config from dict (backward compatibility)
def config_from_dict(config_class: type[BaseConfig], params: Dict[str, Any]) -> BaseConfig:
    """Create config from dict, handling nested dicts.

    Parameters
    ----------
    config_class : type[BaseConfig]
        The config class to instantiate.
    params : dict
        Dictionary of parameters to pass to the config.

    Returns
    -------
    BaseConfig
        Instantiated config object.
    """
    return config_class(**params)


# Export all configs and constants
__all__ = [
    # Constants
    "DEFAULT_RANDOM_SEED",
    "DEFAULT_TREE_ITERATIONS",
    "DEFAULT_CALIBRATION_BINS",
    "DEFAULT_FAIRNESS_MIN_POP_CAT_THRESH",
    "DEFAULT_RFECV_MAX_RUNTIME_MINS",
    "DEFAULT_RFECV_CV_SPLITS",
    "DEFAULT_RFECV_MAX_NOIMPROVING_ITERS",
    "VALID_MODEL_TYPES",
    "VALID_LINEAR_MODEL_TYPES",
    "VALID_SCALER_NAMES",
    "VALID_TASK_TYPES",
    "VALID_MATMUL_PRECISIONS",
    # Enums
    "TargetTypes",
    # Base
    "BaseConfig",
    "PreprocessingConfig",
    "TrainingSplitConfig",
    "PreprocessingBackendConfig",
    "FeatureTypesConfig",
    "FeatureSelectionConfig",
    "ModelConfig",
    "LinearModelConfig",
    "TreeModelConfig",
    "MLPConfig",
    "NGBConfig",
    "AutoMLConfig",
    "ModelHyperparamsConfig",
    "TrainingBehaviorConfig",
    "TrainingConfig",
    "config_from_dict",
    # train_and_evaluate_model configs
    "DataConfig",
    "TrainingControlConfig",
    "MetricsConfig",
    "ReportingConfig",
    "FeatureImportanceConfig",
    "OutputConfig",
    "OutlierDetectionConfig",
    "SliceStableESConfig",
    "ConfidenceAnalysisConfig",
    "ConformalConfig",
    "RegressionCalibrationConfig",
    "NamingConfig",
    "PredictionsContainer",
    "FairnessConfig",
]
