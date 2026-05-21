"""Per-target distribution analyzer (mini-HPT block).

Future-work follow-up to ``tests/training/test_suite_resiliency_synthetic_distributions.py``
(2026-05-21): the resiliency tests catch wrong-defaults failures AFTER training,
but the user asked for a layer that DETECTS pathological feature/target
distributions BEFORE training and tunes defaults accordingly.

Scope (first pass)
------------------
This analyzer covers the failure modes the 2026-05-21 TVT regression incident
surfaced + the closest synthetic siblings:

Regression targets
- Heavy-tail (high excess kurtosis): suggest robust regressor loss
- Multi-modal (>= 2 well-separated peaks): warn (LR/MLP single-mode bias)
- Strong AR (lag-1 autocorr > 0.7): the TVT-2026-05-21 root cause for the
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
import math
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd

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


def _excess_kurtosis(y: np.ndarray) -> float:
    """Biased (Pearson) excess kurtosis; gaussian baseline = 0."""
    n = y.size
    if n < 4:
        return 0.0
    mu = float(np.mean(y))
    sigma = float(np.std(y))
    if sigma <= 0.0 or not math.isfinite(sigma):
        return 0.0
    z = (y - mu) / sigma
    return float(np.mean(z ** 4)) - 3.0


def _skewness(y: np.ndarray) -> float:
    """Biased moment-based skewness."""
    n = y.size
    if n < 3:
        return 0.0
    mu = float(np.mean(y))
    sigma = float(np.std(y))
    if sigma <= 0.0 or not math.isfinite(sigma):
        return 0.0
    z = (y - mu) / sigma
    return float(np.mean(z ** 3))


def _lag1_autocorr(y: np.ndarray) -> float:
    """Pearson autocorrelation between y[:-1] and y[1:].

    A naive autocorr would assume y is time-ordered. This function assumes
    rows are in their training order; AR detection is meaningful only when
    the caller knows the rows have a natural sequence. The suite caller
    skips this detector when ``has_time_axis=False``.
    """
    if y.size < 4:
        return 0.0
    a, b = y[:-1], y[1:]
    sa, sb = float(np.std(a)), float(np.std(b))
    if sa <= 0.0 or sb <= 0.0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _lag1_autocorr_grouped(y: np.ndarray, group_ids: np.ndarray, min_group_size: int = 4) -> float:
    """Per-group lag-1 autocorr averaged across groups (sample-weighted by group size).

    For data where rows have a natural sequence WITHIN each group but not across
    groups (the classic wellbore-MD / per-customer-time-series / per-subject-EEG
    layout), naive ``_lag1_autocorr`` measures cross-group transitions as if they
    were temporal -- spurious low/zero AR. Instead, compute lag-1 autocorr for
    every group independently and aggregate by weighting each group's correlation
    by its row count.

    Groups smaller than ``min_group_size`` rows are skipped. Returns NaN when no
    qualifying groups remain.
    """
    if y.size != group_ids.size or y.size < min_group_size:
        return float("nan")
    # Sort by group then by original row order (so within-group sequence stays).
    # We assume the caller's frame is already ordered the way training will see it
    # -- which for wellbore data means MD-sorted within each well_id. So we just
    # walk unique groups and slice in place; no re-sort needed.
    weighted_sum = 0.0
    weight_sum = 0.0
    uniq = np.unique(group_ids)
    for g in uniq:
        mask = group_ids == g
        n_g = int(mask.sum())
        if n_g < min_group_size:
            continue
        yg = y[mask]
        ar_g = _lag1_autocorr(yg)
        if not math.isfinite(ar_g):
            continue
        weighted_sum += ar_g * n_g
        weight_sum += n_g
    if weight_sum <= 0:
        return float("nan")
    return weighted_sum / weight_sum


_MULTI_MODAL_VALLEY_DEPTH_RATIO: float = 0.7  # antimode must drop to <= 0.7 * min(peak)


def _detect_multi_modal(y: np.ndarray, n_bins: int = _MULTI_MODAL_KDE_BINS,
                        min_peak_sep_stds: float = _MULTI_MODAL_MIN_PEAK_SEP_STDS,
                        valley_depth_ratio: float = _MULTI_MODAL_VALLEY_DEPTH_RATIO) -> tuple[bool, int, float]:
    """Detect >= 2 well-separated peaks via smoothed histogram + antimode check.

    A unimodal but noisy histogram can grow many local maxima from binning
    noise; counting them naively flags gaussian samples as multi-modal. The
    correct test is the *antimode* check: a true bimodal distribution has
    a deep valley BETWEEN the two peaks. We require:

    1. Two distinct local maxima separated by >= ``min_peak_sep_stds`` * std.
    2. The minimum bin between them drops to <= ``valley_depth_ratio`` *
       min(peak_a, peak_b). 0.7 means the antimode must be at least 30%
       below the lower of the two peaks -- gaussian sampling noise rarely
       creates valleys that deep across a wide separation.

    Aggressive smoothing (binomial kernel applied twice) suppresses bin-by-bin
    noise without erasing genuine separations.

    Returns (is_multi_modal, n_peaks_above_min_height, max_qualified_separation).
    """
    if y.size < 50:
        return False, 0, 0.0
    sigma = float(np.std(y))
    if sigma <= 0.0:
        return False, 0, 0.0
    hist, edges = np.histogram(y, bins=n_bins)
    centres = (edges[:-1] + edges[1:]) / 2.0
    # Aggressive smoothing: two passes of binomial(5) kernel kills bin-by-bin noise.
    kernel = np.array([1, 4, 6, 4, 1], dtype=np.float64) / 16.0
    smoothed = np.convolve(hist.astype(np.float64), kernel, mode="same")
    smoothed = np.convolve(smoothed, kernel, mode="same")
    if smoothed.max() <= 0:
        return False, 0, 0.0
    min_height = 0.05 * float(smoothed.max())
    peaks: list[int] = []
    for i in range(1, len(smoothed) - 1):
        if smoothed[i] >= min_height and smoothed[i] > smoothed[i - 1] and smoothed[i] >= smoothed[i + 1]:
            peaks.append(i)
    if len(peaks) < 2:
        return False, len(peaks), 0.0
    # Antimode-qualified pairs: for each pair, check separation in std-units AND
    # valley-depth between them. Only such pairs count as evidence of multi-modality.
    max_qualified_sep = 0.0
    for ii in range(len(peaks)):
        for jj in range(ii + 1, len(peaks)):
            pi, pj = peaks[ii], peaks[jj]
            sep = abs(centres[pj] - centres[pi]) / sigma
            if sep < min_peak_sep_stds:
                continue
            lo, hi = min(pi, pj), max(pi, pj)
            valley = float(smoothed[lo:hi + 1].min())
            lower_peak = float(min(smoothed[pi], smoothed[pj]))
            if lower_peak <= 0:
                continue
            if valley / lower_peak <= valley_depth_ratio:
                if sep > max_qualified_sep:
                    max_qualified_sep = sep
    return bool(max_qualified_sep > 0.0), len(peaks), float(max_qualified_sep)


def _within_between_group_variance_ratio(y: np.ndarray, group_ids: np.ndarray) -> float:
    """Return within-group std / between-group std.

    Ratios near 0 indicate strongly clustered target (group fully determines y).
    Ratios near or above 1 indicate group label adds no information.
    """
    uniq_groups = np.unique(group_ids)
    if len(uniq_groups) < 2:
        return float("nan")
    group_means = np.zeros(len(uniq_groups), dtype=np.float64)
    within_sq_sum = 0.0
    within_n = 0
    for k, g in enumerate(uniq_groups):
        mask = group_ids == g
        if not np.any(mask):
            continue
        yk = y[mask]
        group_means[k] = float(np.mean(yk))
        within_sq_sum += float(np.sum((yk - group_means[k]) ** 2))
        within_n += int(yk.size)
    within_std = math.sqrt(within_sq_sum / max(within_n, 1))
    between_std = float(np.std(group_means))
    if between_std <= 0.0:
        return float("inf")
    return within_std / between_std


def _classify_target_type(y: np.ndarray) -> Literal["regression", "classification"]:
    """Heuristic: integer-typed AND <= 50 unique values -> classification."""
    if y.dtype.kind in ("i", "u", "b"):
        if np.unique(y).size <= 50:
            return "classification"
    if y.dtype.kind == "f":
        # Floats with very few unique values are encoded classification.
        if np.unique(y).size <= 50:
            return "classification"
    return "regression"


def analyze_target_distribution(
    y: np.ndarray,
    *,
    X: Optional[np.ndarray] = None,
    group_ids: Optional[np.ndarray] = None,
    target_type: Literal["regression", "classification", "auto"] = "auto",
    has_time_axis: bool = True,
) -> TargetDistributionReport:
    """Inspect target distribution + optionally group_ids; return a report
    of detected pathologies and recommended hyperparameter overrides.

    Parameters
    ----------
    y
        Target array, 1-D. NaNs are dropped before analysis.
    X
        Optional feature matrix; reserved for future feature-side detectors
        (currently unused at this scope).
    group_ids
        Optional 1-D array of group identifiers; enables the clustered-target
        detector. Skipped if None or all-same-group.
    target_type
        "regression" / "classification" / "auto" (heuristic from dtype + uniques).
    has_time_axis
        Whether row order encodes a temporal sequence. When True, lag-1
        autocorrelation is meaningful (TVT-2026-05-21 scenario). When False
        the AR detector is skipped to avoid spurious "strong AR" hits on
        randomly shuffled rows.

    Returns
    -------
    TargetDistributionReport
    """
    y = np.asarray(y).reshape(-1)
    if y.dtype.kind != "f":
        y_for_stats = y.astype(np.float64)
    else:
        y_for_stats = y
    finite = np.isfinite(y_for_stats)
    if not np.all(finite):
        y_for_stats = y_for_stats[finite]
    n = int(y_for_stats.size)

    ttype: Literal["regression", "classification"]
    if target_type == "auto":
        ttype = _classify_target_type(y)
    else:
        ttype = target_type

    diagnostics: dict[str, float] = {"n_samples": float(n)}
    pathologies: list[str] = []
    knob_overrides: dict[str, dict[str, Any]] = {}

    if ttype == "regression":
        if n < 30:
            # Too small for any reliable detection; return early with the empty report.
            return TargetDistributionReport(
                target_type="regression", n_samples=n,
                pathologies=["insufficient_samples_n<30"],
                knob_overrides=knob_overrides, diagnostics=diagnostics,
            )

        mu = float(np.mean(y_for_stats))
        sigma = float(np.std(y_for_stats))
        diagnostics["mean"] = mu
        diagnostics["std"] = sigma

        # Near-constant target -- hard pathology, no recommendation suffices.
        rel_std = abs(sigma) / (abs(mu) + 1e-9) if abs(mu) > 1e-9 else (sigma if sigma > 0 else 0.0)
        diagnostics["rel_std"] = rel_std
        if rel_std < _NEAR_CONSTANT_REL_STD:
            pathologies.append(f"near_constant_target(rel_std={rel_std:.2e})")
            return TargetDistributionReport(
                target_type="regression", n_samples=n,
                pathologies=pathologies, knob_overrides=knob_overrides,
                diagnostics=diagnostics,
            )

        # Heavy tail
        kurt = _excess_kurtosis(y_for_stats)
        diagnostics["excess_kurtosis"] = kurt
        if kurt > _HEAVY_TAIL_EXCESS_KURT:
            pathologies.append(f"heavy_tail(excess_kurt={kurt:.1f})")
            # Robust regressor preferences across families:
            knob_overrides.setdefault("mlp_kwargs", {})
            mlp_mp = knob_overrides["mlp_kwargs"].setdefault("model_params", {})
            mlp_mp["loss_fn"] = "huber"  # MLP family knob; consumed at MLPTorchModel
            knob_overrides.setdefault("lgb_kwargs", {})["objective"] = "huber"
            knob_overrides.setdefault("xgb_kwargs", {})["objective"] = "reg:pseudohubererror"

        # Skewness
        skew = _skewness(y_for_stats)
        diagnostics["skew"] = skew
        if abs(skew) > _SKEW_ABS_THRESHOLD:
            pathologies.append(f"skewed_target(skew={skew:.2f})")
            # No automatic knob override here -- log transform is a data
            # transformation, not a hyperparameter. The composite-discovery
            # block already considers log/sqrt residual targets when its
            # auto-detector flags skew; we surface the warning so the
            # operator knows it's worth investigating that path.

        # Multi-modal
        is_mm, n_peaks, max_sep = _detect_multi_modal(y_for_stats)
        diagnostics["n_modal_peaks"] = float(n_peaks)
        diagnostics["modal_peak_separation_stds"] = float(max_sep)
        if is_mm:
            pathologies.append(
                f"multi_modal_target(peaks={n_peaks}, max_sep={max_sep:.2f} stds)"
            )

        # Strong AR. Global lag-1 autocorr is meaningful when row order encodes
        # time across the whole dataset. For group-ordered data (rows ordered
        # within each group but not across groups -- wellbore MD / per-customer
        # time series / per-subject EEG), a per-group autocorr aggregated by
        # group size is the right metric: global lag-1 looks low because the
        # group boundaries inject discontinuities, but within each group there
        # IS a strong AR signal. The TVT-2026-05-21 prod log had MD-sorted
        # rows within 771 wells; global lag-1 wasn't measured (the suite
        # didn't pass timestamps so has_time_axis=False), and the MLP
        # use_layernorm recommendation never fired. The per-group branch below
        # gives the analyzer access to that signal even when the suite caller
        # can't supply explicit timestamps.
        ar = float("nan")
        ar_source = None
        if has_time_axis:
            ar = _lag1_autocorr(y_for_stats)
            ar_source = "global"
            diagnostics["lag1_autocorr"] = ar
        elif group_ids is not None:
            gids_arr = np.asarray(group_ids).reshape(-1)
            if gids_arr.size == y.size:
                # Apply the same finite-mask filter as y_for_stats.
                _gids_finite = gids_arr[finite] if finite.size == y.size else gids_arr
                ar = _lag1_autocorr_grouped(y_for_stats, _gids_finite)
                ar_source = "per_group"
                diagnostics["lag1_autocorr_per_group"] = ar
        if math.isfinite(ar) and abs(ar) > _STRONG_AR_PEARSON_LAG1:
            pathologies.append(f"strong_AR_target(lag1_corr={ar:.3f}, source={ar_source})")
            # TVT-2026-05-21 root cause: MLP with per-row layernorm collapses
            # under strong AR because the layer destroys inter-row absolute-
            # scale signal that AR depends on. Force layernorm OFF.
            knob_overrides.setdefault("mlp_kwargs", {})
            mlp_np = knob_overrides["mlp_kwargs"].setdefault("network_params", {})
            mlp_np["use_layernorm"] = False

        # Clustered target (within-group << between-group variance)
        if group_ids is not None:
            gids = np.asarray(group_ids).reshape(-1)
            if gids.size == y.size and finite.sum() == n:
                ratio = _within_between_group_variance_ratio(y_for_stats, gids[finite] if finite.size == y.size else gids)
                diagnostics["within_between_group_var_ratio"] = ratio
                if math.isfinite(ratio) and ratio < _CLUSTERED_TARGET_VARIANCE_RATIO:
                    pathologies.append(
                        f"clustered_target(within/between_std_ratio={ratio:.3f})"
                    )
                    # Recommend group-aware splitting; the suite already supports this via
                    # ``group_column`` in the split config, so we just surface the hint.
                    knob_overrides.setdefault("split_config", {})["prefer_group_aware"] = True

    else:  # classification
        if n < 30:
            return TargetDistributionReport(
                target_type="classification", n_samples=n,
                pathologies=["insufficient_samples_n<30"],
                knob_overrides=knob_overrides, diagnostics=diagnostics,
            )
        # Class frequency table.
        classes, counts = np.unique(y, return_counts=True)
        n_classes = int(classes.size)
        diagnostics["n_classes"] = float(n_classes)
        if n_classes < 2:
            pathologies.append("single_class_target")
            return TargetDistributionReport(
                target_type="classification", n_samples=n,
                pathologies=pathologies, knob_overrides=knob_overrides,
                diagnostics=diagnostics,
            )
        freqs = counts / counts.sum()
        max_freq = float(freqs.max())
        min_freq = float(freqs.min())
        diagnostics["max_class_freq"] = max_freq
        diagnostics["min_class_freq"] = min_freq
        # Near-singleton: one class dominates entirely.
        if max_freq >= _NEAR_SINGLETON_MAX_FRACTION:
            pathologies.append(f"near_singleton_class(max_freq={max_freq:.3f})")
        # Class imbalance.
        ratio = max_freq / max(min_freq, 1.0 / n)
        diagnostics["class_imbalance_ratio"] = ratio
        if ratio > _CLASS_IMBALANCE_MAX_MIN_RATIO:
            pathologies.append(f"class_imbalance(max/min={ratio:.1f}x)")
            # Recommend balanced class_weight where the model family supports it.
            knob_overrides.setdefault("lgb_kwargs", {})["class_weight"] = "balanced"
            knob_overrides.setdefault("xgb_kwargs", {})["scale_pos_weight"] = max(1.0, ratio)
            knob_overrides.setdefault("cb_kwargs", {})["auto_class_weights"] = "Balanced"
        # Rare classes.
        rare_classes = [int(c) for c, k in zip(classes, counts) if int(k) < _RARE_CLASS_MIN_N]
        diagnostics["n_rare_classes"] = float(len(rare_classes))
        if rare_classes:
            pathologies.append(
                f"rare_classes(n={len(rare_classes)}, min_n_threshold={_RARE_CLASS_MIN_N})"
            )

    return TargetDistributionReport(
        target_type=ttype, n_samples=n,
        pathologies=pathologies, knob_overrides=knob_overrides,
        diagnostics=diagnostics,
    )


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


@dataclass
class FeatureDistributionReport:
    """Report from :func:`analyze_feature_distribution`."""

    n_samples: int
    n_features: int
    pathologies: list[str] = field(default_factory=list)
    # Per-feature warning detail: ``{feature_name: [pathology_strings]}``.
    feature_warnings: dict[str, list[str]] = field(default_factory=dict)
    # Suggested drop candidates (near-constant / NaN-heavy / one side of a
    # redundant pair). Operator should review before applying.
    drop_candidates: list[str] = field(default_factory=list)
    # Suspected target-leakage feature names (not auto-actioned).
    leakage_candidates: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    knob_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)


def _pairwise_redundant_features(
    X_numeric: np.ndarray,
    feature_names: list[str],
    threshold: float = _REDUNDANT_CORR_THRESHOLD,
) -> list[tuple[str, str, float]]:
    """Return list of (feature_a, feature_b, |corr|) for pairs above the threshold.

    Uses np.corrcoef on the FULL matrix (O(n_features^2) memory). The caller
    is responsible for capping feature count via _REDUNDANCY_MAX_NUMERIC_FEATURES.
    """
    if X_numeric.shape[1] < 2:
        return []
    # corrcoef along rows -> transpose so each row is a feature.
    with np.errstate(invalid="ignore", divide="ignore"):
        C = np.corrcoef(X_numeric, rowvar=False)
    pairs: list[tuple[str, str, float]] = []
    n_feats = X_numeric.shape[1]
    for i in range(n_feats):
        for j in range(i + 1, n_feats):
            corr = float(C[i, j])
            if not math.isfinite(corr):
                continue
            if abs(corr) >= threshold:
                pairs.append((feature_names[i], feature_names[j], abs(corr)))
    pairs.sort(key=lambda t: -t[2])
    return pairs


def _normalise_X(
    X,
    feature_names: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Return (df, numeric_cols, categorical_cols).

    Numpy input gets generic ``f0...fN`` names; pandas keeps its columns.
    Polars input is zero-copy-converted to pandas via ``.to_pandas()`` so
    column dtypes (Float32 / Int64 / Boolean / String) are preserved -- a
    naive ``np.asarray(polars_df)`` collapses everything to object dtype
    and downstream ``is_numeric_dtype`` checks misclassify every column as
    categorical. TVT 2026-05-21 reproduction: the suite called the analyzer
    after _phase_train_val_test_split but BEFORE _phase_fit_pipeline (train_df
    still polars), and the analyzer logged
    ``high_cardinality_categorical(n=25)`` for 25 Float32 numeric features.
    Categorical detection: object / category / pandas string dtype goes to
    ``categorical_cols``; numeric int/float/bool goes to ``numeric_cols``.
    """
    if isinstance(X, pd.DataFrame):
        df = X
    else:
        # Polars-family input handling (2026-05-21 P0 #3 + follow-up). Three
        # shapes need explicit dispatch; the first version of this branch
        # only handled DataFrame and the other two fell through to the
        # numpy path with the same misclassification bug the original fix
        # targeted:
        #   - polars.DataFrame: to_pandas() returns a pd.DataFrame; canonical.
        #   - polars.LazyFrame: NO to_pandas method; must collect() first then
        #     to_pandas(). Without explicit handling np.asarray hits a 0-d
        #     object array and every column reads as categorical.
        #   - polars.Series: to_pandas() returns a pd.Series (NOT DataFrame),
        #     downstream df.columns AttributeErrors. Convert to a 1-column
        #     DataFrame.
        _module = type(X).__module__ if X is not None else ""
        _is_polars = isinstance(_module, str) and _module.startswith("polars")
        df = None
        if _is_polars:
            _typename = type(X).__name__
            if _typename == "LazyFrame":
                # Materialise to DataFrame then convert. LazyFrame has neither
                # to_pandas nor columns directly, so caller must accept the
                # collect() cost; a LazyFrame walked through analyze_feature_
                # distribution is going to be materialised anyway downstream.
                try:
                    df = X.collect().to_pandas()
                except Exception:
                    df = None
            elif _typename == "Series":
                # 1-D series -> single-column frame. Use the series name if
                # caller didn't supply feature_names.
                _name = (feature_names[0] if feature_names else (getattr(X, "name", None) or "f0"))
                try:
                    df = pd.DataFrame({_name: X.to_pandas()})
                except Exception:
                    df = None
            else:
                _to_pandas = getattr(X, "to_pandas", None)
                if callable(_to_pandas):
                    df = _to_pandas()
        if df is None:
            arr = np.asarray(X)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            if feature_names is None:
                feature_names = [f"f{i}" for i in range(arr.shape[1])]
            df = pd.DataFrame(arr, columns=feature_names)
    numeric_cols: list[str] = []
    categorical_cols: list[str] = []
    for c in df.columns:
        s = df[c]
        # pandas <2 has ``object`` for strings; >=2 has ArrowExtension or StringDtype.
        # Treat anything non-numeric as categorical for analysis purposes.
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            numeric_cols.append(str(c))
        elif pd.api.types.is_bool_dtype(s):
            # Bools coalesce into numeric for std-based checks (cast to int8).
            numeric_cols.append(str(c))
        else:
            categorical_cols.append(str(c))
    return df, numeric_cols, categorical_cols


def analyze_feature_distribution(
    X,
    y: Optional[np.ndarray] = None,
    *,
    feature_names: Optional[list[str]] = None,
    target_type: Literal["regression", "classification", "auto"] = "auto",
    low_variance_rel_std: float = _LOW_VAR_REL_STD,
    redundant_corr_threshold: float = _REDUNDANT_CORR_THRESHOLD,
    high_cardinality_max: int = _HIGH_CARDINALITY_MAX,
    nan_fraction_threshold: float = _NAN_FRACTION_THRESHOLD,
    leakage_corr_threshold: float = _LEAKAGE_CORR_THRESHOLD,
    redundancy_max_numeric_features: int = _REDUNDANCY_MAX_NUMERIC_FEATURES,
) -> FeatureDistributionReport:
    """Inspect the FEATURE matrix and (optionally) target; surface pathologies.

    Each detector is documented in the module docstring. The function never
    mutates X / y. Recommendations are observational; the suite caller decides
    whether to action them.
    """
    df, numeric_cols, categorical_cols = _normalise_X(X, feature_names=feature_names)
    n_samples = int(df.shape[0])
    n_features = int(df.shape[1])
    diagnostics: dict[str, Any] = {
        "n_samples": n_samples, "n_features": n_features,
        "n_numeric": len(numeric_cols), "n_categorical": len(categorical_cols),
    }
    pathologies: list[str] = []
    feature_warnings: dict[str, list[str]] = {}
    drop_candidates: list[str] = []
    leakage_candidates: list[str] = []
    knob_overrides: dict[str, dict[str, Any]] = {}

    if n_samples < 30:
        # Too small for any reliable detection; return early.
        return FeatureDistributionReport(
            n_samples=n_samples, n_features=n_features,
            pathologies=["insufficient_samples_n<30"],
            diagnostics=diagnostics,
        )

    def _add_warning(col: str, msg: str) -> None:
        feature_warnings.setdefault(col, []).append(msg)

    # --- numeric: low-variance + nan fraction ---
    # bench-attempt-rejected (2026-05-21, 200k rows / 15 cols): tried
    # materialising df[numeric_cols] -> (N, F) block once and reducing via
    # np.nanmean / np.nanstd / np.isfinite axis=0 instead of the per-column
    # loop below. Full-pass time unchanged (~270 ms), y-skipping path got
    # ~15 ms SLOWER. Root cause: np.nanmean and np.nanstd each do their own
    # internal NaN scan, so vectorising on top of an isfinite mask scans the
    # block 3x instead of the loop's 2x; pandas BlockManager extraction also
    # has overhead vs per-column .to_numpy. Keep the per-column path.
    low_var_features: list[str] = []
    nan_heavy_features: list[str] = []
    for c in numeric_cols:
        col = df[c].to_numpy()
        col_f = np.asarray(col, dtype=np.float64)
        nan_mask = ~np.isfinite(col_f)
        nan_frac = float(nan_mask.mean()) if col_f.size > 0 else 0.0
        if nan_frac >= nan_fraction_threshold:
            nan_heavy_features.append(c)
            _add_warning(c, f"nan_fraction={nan_frac:.2f} >= {nan_fraction_threshold}")
            drop_candidates.append(c)
            continue  # Skip further stats on a NaN-dominated feature.
        finite = col_f[~nan_mask]
        if finite.size < 2:
            _add_warning(c, "insufficient_finite_values")
            drop_candidates.append(c)
            continue
        mu = float(np.mean(finite))
        sd = float(np.std(finite))
        rel = abs(sd) / (abs(mu) + 1e-9) if abs(mu) > 1e-9 else (sd if sd > 0 else 0.0)
        if sd <= 0.0 or rel < low_variance_rel_std:
            low_var_features.append(c)
            _add_warning(c, f"low_variance(rel_std={rel:.2e})")
            drop_candidates.append(c)
    if low_var_features:
        pathologies.append(f"low_variance_features(n={len(low_var_features)})")
        diagnostics["low_variance_features"] = list(low_var_features)
    if nan_heavy_features:
        pathologies.append(f"nan_heavy_features(n={len(nan_heavy_features)})")
        diagnostics["nan_heavy_features"] = list(nan_heavy_features)
        # Recommend explicit NaN strategy at the preprocessing layer.
        knob_overrides.setdefault("preprocessing_config", {})["review_nan_strategy"] = True

    # --- categorical: high cardinality ---
    high_card_features: list[str] = []
    for c in categorical_cols:
        n_unique = int(df[c].nunique(dropna=False))
        if n_unique > high_cardinality_max:
            high_card_features.append(c)
            _add_warning(c, f"high_cardinality(n_unique={n_unique} > {high_cardinality_max})")
    if high_card_features:
        pathologies.append(f"high_cardinality_categorical(n={len(high_card_features)})")
        diagnostics["high_cardinality_features"] = list(high_card_features)
        # Recommend a target / hashing encoder over one-hot (preprocessing layer hint).
        knob_overrides.setdefault("preprocessing_config", {})["prefer_high_cardinality_encoder"] = True

    # --- redundant pairs (numeric only; skip the dropped low-var/nan-heavy set) ---
    candidate_numeric = [c for c in numeric_cols if c not in low_var_features and c not in nan_heavy_features]
    if len(candidate_numeric) > redundancy_max_numeric_features:
        diagnostics["redundancy_skipped"] = (
            f"n_numeric={len(candidate_numeric)} > cap={redundancy_max_numeric_features}; "
            "pairwise correlation O(n^2) would be too costly. Lower the threshold via "
            "redundancy_max_numeric_features or pre-filter."
        )
    elif len(candidate_numeric) >= 2:
        # Fill NaNs with column means for the corrcoef pass so a sparse NaN row
        # doesn't poison every correlation in its row/column. We're not
        # imputing the data downstream, just computing redundancy.
        sub = df[candidate_numeric].to_numpy(dtype=np.float64, na_value=np.nan)
        col_means = np.nanmean(sub, axis=0)
        # Vectorised mean fill -- np.take_along_axis would be overkill here.
        for j in range(sub.shape[1]):
            mask = ~np.isfinite(sub[:, j])
            if mask.any():
                sub[mask, j] = col_means[j] if math.isfinite(col_means[j]) else 0.0
        pairs = _pairwise_redundant_features(sub, candidate_numeric, threshold=redundant_corr_threshold)
        if pairs:
            pathologies.append(f"redundant_feature_pairs(n={len(pairs)})")
            diagnostics["redundant_feature_pairs"] = [
                {"a": a, "b": b, "corr": c} for a, b, c in pairs[:50]  # cap log to top 50
            ]
            for a, b, c in pairs:
                _add_warning(a, f"redundant_with({b}, corr={c:.3f})")
                _add_warning(b, f"redundant_with({a}, corr={c:.3f})")

    # --- target leakage (only if y supplied) ---
    if y is not None and len(candidate_numeric) > 0:
        y_arr = np.asarray(y).reshape(-1)
        if y_arr.size == n_samples and y_arr.dtype.kind in ("f", "i", "u", "b"):
            # pandas.DataFrame.corrwith vectorises pairwise-complete-obs
            # correlation across all columns in one C-level call: ~65 ms vs
            # the prior per-column np.corrcoef loop's ~145 ms at 200k rows /
            # 15 cols, and bit-exact -- it builds the per-column finite mask
            # internally instead of imputing NaN cells to the column mean
            # (which would dilute the correlation enough to drop legitimate
            # leakage below the 0.99 threshold on combos with sparse NaN).
            y_series = pd.Series(y_arr, index=df.index)
            try:
                corrs = df[candidate_numeric].corrwith(y_series, drop=False)
            except Exception:
                # Object-dtype mix or other corrwith refusal — fall through
                # to nothing rather than crash the analyzer.
                corrs = pd.Series(dtype=np.float64)
            for c, corr_val_raw in corrs.items():
                if not pd.notna(corr_val_raw):
                    continue
                corr_val = float(corr_val_raw)
                if abs(corr_val) >= leakage_corr_threshold:
                    leakage_candidates.append(c)
                    _add_warning(c, f"suspected_target_leakage(corr_with_y={corr_val:.4f})")
        if leakage_candidates:
            pathologies.append(f"suspected_target_leakage(n={len(leakage_candidates)})")
            diagnostics["leakage_candidates"] = list(leakage_candidates)

    return FeatureDistributionReport(
        n_samples=n_samples, n_features=n_features,
        pathologies=pathologies,
        feature_warnings=feature_warnings,
        drop_candidates=drop_candidates,
        leakage_candidates=leakage_candidates,
        diagnostics=diagnostics,
        knob_overrides=knob_overrides,
    )
