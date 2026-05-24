"""``analyze_target_distribution`` carved out of
``mlframe.training._target_distribution_analyzer``.

The function consumes statistical helpers from
``_target_distribution_analyzer_stats`` + classifier helpers from
``_target_distribution_analyzer_modes``, plus the ``TargetDistributionReport``
dataclass + the detection threshold constants at the parent's top.

Bound back into the parent's namespace via re-export at the parent's
module bottom so historical
``from mlframe.training._target_distribution_analyzer import analyze_target_distribution``
resolves transparently.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Literal, Optional

import numpy as np

from ._target_distribution_analyzer import (
    TargetDistributionReport,
    _CLASS_IMBALANCE_MAX_MIN_RATIO,
    _CLUSTERED_TARGET_VARIANCE_RATIO,
    _HEAVY_TAIL_EXCESS_KURT,
    _NEAR_CONSTANT_REL_STD,
    _NEAR_SINGLETON_MAX_FRACTION,
    _RARE_CLASS_MIN_N,
    _SKEW_ABS_THRESHOLD,
    _STRONG_AR_PEARSON_LAG1,
)
from ._target_distribution_analyzer_modes import (
    _classify_target_type,
    _detect_multi_modal,
    _within_between_group_variance_ratio,
)
from ._target_distribution_analyzer_stats import (
    _check_within_group_ordering,
    _excess_kurtosis,
    _lag1_autocorr_grouped,
    _lag_autocorr,
    _max_abs_lag_autocorr,
    _skewness,
)

logger = logging.getLogger(__name__)


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
            # E5.1: scan lags 1/2/3/5 and take the strongest |autocorr|. Long-memory
            # series can hit lag-2/3 strongly with weak lag-1 -- both shapes feed
            # the same MLP-LayerNorm collapse mode the AR detector exists to flag.
            ar, ar_lag = _max_abs_lag_autocorr(y_for_stats)
            ar_source = f"global_lag{ar_lag}" if ar_lag else "global"
            diagnostics["lag1_autocorr"] = _lag_autocorr(y_for_stats, lag=1)
            diagnostics["max_abs_autocorr"] = float(ar)
            diagnostics["max_abs_autocorr_lag"] = int(ar_lag) if ar_lag else 0
        elif group_ids is not None:
            gids_arr = np.asarray(group_ids).reshape(-1)
            if gids_arr.size == y.size:
                # Apply the same finite-mask filter as y_for_stats.
                _gids_finite = gids_arr[finite] if finite.size == y.size else gids_arr
                # E5 (2026-05-21) ordering-check addition: warn when within-group
                # sequence is destroyed by post-FTE shuffling. The per-group AR
                # detector assumes rows of the same group are contiguous AND in
                # their natural order (MD-sorted for wellbore, time-ordered for
                # per-customer logs). When the suite caller shuffles BEFORE
                # passing to analyze_target_distribution, the within-group
                # autocorr drops to ~0 and the detector silently false-negatives
                # the same prod-relevant pathology it exists to catch.
                _ordered = _check_within_group_ordering(_gids_finite)
                diagnostics["group_ordering_check"] = bool(_ordered)
                if not _ordered:
                    logger.warning(
                        "_lag1_autocorr_grouped: rows do not appear sorted by group "
                        "(only %.0f%% consecutive transitions are within-group). The "
                        "per-group AR detector assumes within-group sequence is preserved; "
                        "if your data IS group-sorted, ignore this warning. Otherwise "
                        "the AR signal is being destroyed by post-FTE row shuffling.",
                        100.0 * float(np.mean(_gids_finite[:-1] == _gids_finite[1:])),
                    )
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
                    # E5.2 (2026-05-21): clustered targets share the SAME MLP-LayerNorm collapse mode
                    # as strong-AR (per-row LayerNorm destroys the between-row absolute-scale signal
                    # that the group label encodes). The strong-AR detector above already turns
                    # LayerNorm off when has_time_axis is True OR per-group AR fires; this branch
                    # closes the third path (group-clustered target where AR is weak but the
                    # absolute group level dominates -- e.g. wellbore TVT, customer LTV by tenure).
                    knob_overrides.setdefault("mlp_kwargs", {}).setdefault("network_params", {})["use_layernorm"] = False

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
