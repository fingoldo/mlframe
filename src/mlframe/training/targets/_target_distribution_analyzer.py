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
from dataclasses import dataclass, field
from typing import Any, Literal

logger = logging.getLogger(__name__)


# Detection thresholds. Each surface as a module-level constant so tests can
# tighten / loosen them without monkey-patching the function body. The
# defaults below were chosen so the synthetic 8-scenario tests in
# test_suite_resiliency_synthetic_distributions.py classify cleanly.
_HEAVY_TAIL_EXCESS_KURT: float = 5.0
# Peak separation threshold uses GLOBAL std (cheap to compute). For a symmetric
# bimodal with cluster-std=sigma and peak spacing 2*d, the global std works out
# to ~d (peaks dominate the variance), so peak separation in global-stds caps
# at ~2.0 even for very clean bimodals. We set 1.8 to admit textbook bimodals
# while the valley-depth antimode check screens out unimodal-with-noise.
_MULTI_MODAL_MIN_PEAK_SEP_STDS: float = 1.8
_MULTI_MODAL_KDE_BINS: int = 128
_MULTI_MODAL_VALLEY_DEPTH_RATIO: float = 0.7  # antimode must drop to <= 0.7 * min(peak)
_STRONG_AR_PEARSON_LAG1: float = 0.7
_CLUSTERED_TARGET_VARIANCE_RATIO: float = 0.3  # within-std / between-std
_SKEW_ABS_THRESHOLD: float = 2.0
_NEAR_CONSTANT_REL_STD: float = 1e-3
_CLASS_IMBALANCE_MAX_MIN_RATIO: float = 10.0
_RARE_CLASS_MIN_N: int = 100
_NEAR_SINGLETON_MAX_FRACTION: float = 0.99

# Feature-side detector thresholds (used by analyze_feature_distribution).
_LOW_VAR_REL_STD: float = 1e-3       # std / (|mean|+eps) below this -> near-constant
_LOW_VAR_NUNIQUE: int = 2             # binary features with imbalance flagged separately
_REDUNDANT_CORR_THRESHOLD: float = 0.95   # |Pearson| above this -> redundant pair
_HIGH_CARDINALITY_MAX: int = 100      # categorical features above this -> recommend encoder
_NAN_FRACTION_THRESHOLD: float = 0.5  # 50%+ NaN rate -> structural issue, not random missingness
_LEAKAGE_CORR_THRESHOLD: float = 0.99 # feature-target |corr| above this -> suspected leakage
# Computing the full correlation matrix is O(n_features^2). Cap to keep the analyzer
# fast on wide frames; redundant-pair detection is a heuristic, not a guarantee.
_REDUNDANCY_MAX_NUMERIC_FEATURES: int = 500


@dataclass
class TargetDistributionReport:
    """Report from :func:`analyze_target_distribution`.

    ``pathologies`` is a list of short human-readable strings (one per detected
    pathology) intended for inclusion in the suite log. ``knob_overrides`` is a
    nested dict of recommended config overrides keyed by model-family slot
    (e.g. ``mlp_kwargs``, ``lgb_kwargs``, ``cb_kwargs``); the suite caller
    deep-merges this onto its hyperparams config, with caller-supplied values
    taking precedence (recommendation, not enforcement).

    ``diagnostics`` carries every measured statistic (kurtosis, skew, lag-1
    autocorr, etc.) so downstream observability dashboards can plot trends and
    operators can debug a recommendation they disagree with.
    """

    target_type: Literal["regression", "classification"]
    n_samples: int
    pathologies: list[str] = field(default_factory=list)
    knob_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    diagnostics: dict[str, float] = field(default_factory=dict)
    # Per-knob provenance stamps: {slot: {knob_name: {"value": v, "source": "analyzer", "reason": "<pathology>"}}}.
    # Parallel to ``knob_overrides`` so existing consumers (suite caller's deep-merge) keep working unchanged; consumers
    # that care about origin (which value came from the analyzer vs the user vs the project default) inspect this field.
    knob_overrides_provenance: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)

    def merge_into_config(self, base_config: dict, override_existing: bool = False) -> dict:
        """Deep-merge knob_overrides into a (possibly nested) hyperparams config.

        ``override_existing=False`` is the default and the safe choice: caller-supplied
        values are preserved; recommendations only fill in gaps. Pass True ONLY when
        the recommendation should win (rare; used for hard-pathology overrides).
        """
        out = {k: (v.copy() if isinstance(v, dict) else v) for k, v in base_config.items()}
        for slot, knobs in self.knob_overrides.items():
            if slot not in out:
                out[slot] = {}
            elif not isinstance(out[slot], dict):
                continue  # caller put a non-dict there; bail rather than clobber
            for k, v in knobs.items():
                if override_existing or k not in out[slot]:
                    out[slot][k] = v
        return out


# ---------------------------------------------------------------------------
# Feature-side detectors (mini-HPT v2, 2026-05-21).
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
# - NaN-heavy features (fraction > 50%): random missingness or structural?
#   At >=50% the imputer is dominating the column; the operator should pick a
#   strategy explicitly rather than rely on the default.
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
from ._target_distribution_analyzer_stats import (  # noqa: E402,F401
    _check_within_group_ordering,
    _excess_kurtosis,
    _lag1_autocorr,
    _lag1_autocorr_grouped,
    _lag_autocorr,
    _max_abs_lag_autocorr,
    _skewness,
)
from ._target_distribution_analyzer_modes import (  # noqa: E402,F401
    _classify_target_type,
    _detect_multi_modal,
    _within_between_group_variance_ratio,
)
from ._target_distribution_analyzer_target_fn import (  # noqa: E402,F401
    analyze_target_distribution,
)
from ._target_distribution_analyzer_features import (  # noqa: E402,F401
    FeatureDistributionReport,
    _normalise_X,
    _pairwise_redundant_features,
    analyze_feature_distribution,
)
