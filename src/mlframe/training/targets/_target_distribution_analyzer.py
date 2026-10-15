"""Per-target distribution analyzer (mini-HPT block).

Future-work follow-up to ``tests/training/test_suite_resiliency_synthetic_distributions.py``
(2026-05-21): the resiliency tests catch wrong-defaults failures AFTER training,
but the user asked for a layer that DETECTS pathological feature/target
distributions BEFORE training and tunes defaults accordingly.

Scope (first pass)
------------------
This analyzer covers the failure modes a real regression incident
surfaced + the closest synthetic siblings:

Regression targets
- Heavy-tail (high excess kurtosis): suggest robust regressor loss
- Multi-modal (>= 2 well-separated peaks): warn (LR/MLP single-mode bias)
- Strong AR (lag-1 autocorr > 0.7): a real root cause for the
  R^2=-4.75 MLP collapse -- suggest use_layernorm=False for MLP since
  per-row LayerNorm destroys inter-row absolute-scale signal under strong AR
- Clustered target (within-group std << between-group std when group_ids
  supplied): suggest group-aware split / target encoding
- Skewed (|skew| > 2): suggest log/sqrt transform
- Near-constant target (std/|mean| < 1e-3): hard warn (degenerate)

Classification targets
- Class imbalance (max/min freq > 10): suggest class_weight='balanced'
- Rare classes (any class < ``rare_class_min_n``): warn
- Near-singleton (>= 99% one class): hard warn (degenerate)

Out of scope (future expansion):
- Feature-side pathology (clustered features, redundant features) -- the
  existing ``compute_high_correlation_pairs`` covers the redundancy case.
- Automatic config mutation -- this module ONLY reports / recommends; the
  suite caller (or the user) is responsible for merging recommendations
  into the active config.

Public API
----------
``analyze_target_distribution(y, X=None, group_ids=None, target_type='auto')``
returns a ``TargetDistributionReport`` dataclass; the suite caller can merge
``report.knob_overrides`` into its hyperparams config.

Detection thresholds are conservative (high precision, may miss subtle cases)
because false positives here mean reshaping defaults under the user's feet --
under-detection is recoverable, over-detection silently changes behaviour.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


# Detection thresholds. Each surface as a module-level constant so tests can
# tighten / loosen them without monkey-patching the function body. The
# defaults below were chosen so the synthetic 8-scenario tests in
# test_suite_resiliency_synthetic_distributions.py classify cleanly.
from ._target_distribution_types import (  # noqa: F401 - re-exported from the facade
    _HEAVY_TAIL_EXCESS_KURT,
    _MULTI_MODAL_MIN_PEAK_SEP_STDS,
    _MULTI_MODAL_KDE_BINS,
    _MULTI_MODAL_VALLEY_DEPTH_RATIO,
    _STRONG_AR_PEARSON_LAG1,
    _CLUSTERED_TARGET_VARIANCE_RATIO,
    _SKEW_ABS_THRESHOLD,
    _NEAR_CONSTANT_REL_STD,
    _CLASS_IMBALANCE_MAX_MIN_RATIO,
    _RARE_CLASS_MIN_N,
    _NEAR_SINGLETON_MAX_FRACTION,
    _LOW_VAR_REL_STD,
    _LOW_VAR_NUNIQUE,
    _REDUNDANT_CORR_THRESHOLD,
    _HIGH_CARDINALITY_MAX,
    _NAN_FRACTION_THRESHOLD,
    _LEAKAGE_CORR_THRESHOLD,
    _LEAKAGE_MAX_CLASSES_FOR_AUC,
    _REDUNDANCY_MAX_NUMERIC_FEATURES,
    TargetDistributionReport,
)

# ---------------------------------------------------------------------------
# Feature-side detectors (mini-HPT v2).
#
# Inspect the FEATURE matrix for pathologies that distort downstream training:
#
# - Near-constant features (std/|mean| < 1e-3): contribute no information; the
#   pre-pipeline ``VarianceThreshold`` already drops these IF the operator
#   enabled it, but most callers don't. Flag the feature name so the operator
#   either drops it or sets preprocessing_config.drop_near_constant=True.
#
# - Redundant feature pairs (|Pearson| > 0.95): kept together they
#   over-weight the underlying signal AND inflate the linear-model condition
#   number. Pair-flag for operator review; auto-drop is risky (which member
#   to keep depends on downstream FE / target).
#
# - High-cardinality categorical features (n_unique > 100): one-hot blows up
#   the feature space; recommend target / hashing encoders.
#
# - NaN-heavy features (fraction >= _NAN_FRACTION_THRESHOLD, which is 0.99): random missingness or structural?
#   At that fraction the imputer is producing essentially the whole column, so the operator should pick a
#   strategy explicitly rather than rely on the default. The threshold is 0.99 and not the former 0.5 for the
#   structural-missingness reason written out at the constant's own definition above -- at 50% this rule alone
#   discarded whole feature families whose missingness is meaningful. This block said "> 50%" long after that
#   change, so a reader wondering why a 60%-NaN column was not flagged concluded the detector was broken.
#
# - Suspected target leakage (|Pearson(x, y)| > 0.99 for regression OR
#   per-class AUC > 0.99 for classification): a feature should NOT predict
#   the target almost perfectly unless it's the target itself or a leaked
#   sibling. Mark + WARN; do NOT auto-drop because legitimate features can
#   land here on tiny datasets.
# ---------------------------------------------------------------------------


# ----------------------------------------------------------------------
# Sibling-module re-exports. Statistical helpers + modal-detection +
# target/feature analyzer functions live in four sibling modules. The
# parent loads them at its bottom AFTER ``TargetDistributionReport`` and
# the threshold constants are bound, so the siblings can ``from
# ._target_distribution_analyzer import <constant>`` at their module top.
# ----------------------------------------------------------------------
from ._target_distribution_analyzer_stats import (  # noqa: F401
    _check_within_group_ordering,
    _excess_kurtosis,
    _lag1_autocorr,
    _lag1_autocorr_grouped,
    _lag_autocorr,
    _max_abs_lag_autocorr,
    _skewness,
)
from ._target_distribution_analyzer_modes import (  # noqa: F401
    _classify_target_type,
    _detect_multi_modal,
    _within_between_group_variance_ratio,
)
from ._target_distribution_analyzer_target_fn import (  # noqa: F401
    analyze_target_distribution,
)
from ._target_distribution_analyzer_features import (  # noqa: F401
    FeatureDistributionReport,
    _normalise_X,
    _pairwise_redundant_features,
    analyze_feature_distribution,
)
