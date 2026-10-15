"""Shared target-distribution thresholds and the report record, kept in a leaf module.

Leaf module: the analyzer facade and its feature, mode and target-function siblings all read these, and this module imports nothing from the package.
"""

from dataclasses import dataclass, field
from typing import Any, Literal

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
_LOW_VAR_REL_STD: float = 1e-3  # std / (|mean|+eps) below this -> near-constant
_LOW_VAR_NUNIQUE: int = 2  # binary features with imbalance flagged separately
_REDUNDANT_CORR_THRESHOLD: float = 0.95  # |Pearson| above this -> redundant pair
_HIGH_CARDINALITY_MAX: int = 100  # categorical features above this -> recommend encoder
# Feature NaN rate at/above which a column is reported as nan-heavy (and, when auto-drop is on, removed).
# 0.99, not the former 0.5: at 50% this rule alone discarded whole feature families whose missingness is
# STRUCTURAL rather than random -- a field that only applies to a subset of rows (hourly rate on
# fixed-price jobs, AI-prompt stats on non-AI posts) is missing by construction, and its presence/absence
# is itself predictive. Observed on a production run: 35 of 118 columns (30%) dropped, every one of them
# for this rule and nothing else. Tree models consume NaN natively, so dropping such a column removes
# signal rather than noise. At 0.99 the rule keeps its genuine purpose -- catching a column that is
# essentially empty and cannot carry information -- without acting as a de-facto feature selector.
_NAN_FRACTION_THRESHOLD: float = 0.99
_LEAKAGE_CORR_THRESHOLD: float = 0.99  # feature-target |corr| (regression/binary) or per-class AUC (multiclass) above this -> suspected leakage
_LEAKAGE_MAX_CLASSES_FOR_AUC: int = (
    20  # multiclass leakage detector cap: avoids an O(n_features * n_classes) AUC sweep on high-cardinality integer-coded columns
)
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
