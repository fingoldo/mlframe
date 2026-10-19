"""Stages of ``analyze_target_distribution``."""

from __future__ import annotations

# --- imports (managed) ---
import math
import numpy as np
from mlframe.training.loss_recommendation import huber_delta_for
from mlframe.training.targets._target_distribution_analyzer_modes import _within_between_group_variance_ratio
from mlframe.training.targets._target_distribution_analyzer_target_fn import _apply_heavy_tail_huber_overrides
from mlframe.training.targets._target_distribution_types import _CLASS_IMBALANCE_MAX_MIN_RATIO
from mlframe.training.targets._target_distribution_types import _CLUSTERED_TARGET_VARIANCE_RATIO
from mlframe.training.targets._target_distribution_types import _HEAVY_TAIL_EXCESS_KURT
from mlframe.training.targets._target_distribution_types import _NEAR_SINGLETON_MAX_FRACTION
from mlframe.training.targets._target_distribution_types import _RARE_CLASS_MIN_N
from mlframe.training.targets._target_distribution_types import _SKEW_ABS_THRESHOLD
# --- end imports ---


def _classification_balance_pathologies(counts, diagnostics, pathologies, n, knob_overrides, _stamp_prov, classes):
    """Stage of ``analyze_target_distribution``."""
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
        _stamp_prov("lgb_kwargs", "class_weight", "balanced", "class_imbalance")
        _stamp_prov("xgb_kwargs", "scale_pos_weight", max(1.0, ratio), "class_imbalance")
        _stamp_prov("cb_kwargs", "auto_class_weights", "Balanced", "class_imbalance")
    # Rare classes. Keep class labels in their native dtype (not force-cast to int):
    # labels can legitimately be strings, and only the count is used below -- forcing
    # int() here previously both mis-displayed float labels (e.g. 1.4/1.6 both -> 1,
    # not that the truncated value is ever actually surfaced) and would crash outright
    # on string-labeled classification targets.
    rare_classes = [c for c, k in zip(classes, counts) if int(k) < _RARE_CLASS_MIN_N]
    diagnostics["n_rare_classes"] = float(len(rare_classes))
    if rare_classes:
        pathologies.append(f"rare_classes(n={len(rare_classes)}, min_n_threshold={_RARE_CLASS_MIN_N})")


def _group_ids_pathologies(group_ids, y, finite, n, y_for_stats, diagnostics, pathologies, knob_overrides, _stamp_prov):
    """Stage of ``analyze_target_distribution``."""
    if group_ids is not None:
        gids = np.asarray(group_ids).reshape(-1)
        if gids.size == y.size and finite.sum() == n:
            ratio = _within_between_group_variance_ratio(y_for_stats, gids[finite] if finite.size == y.size else gids)
            diagnostics["within_between_group_var_ratio"] = ratio
            if math.isfinite(ratio) and ratio < _CLUSTERED_TARGET_VARIANCE_RATIO:
                pathologies.append(f"clustered_target(within/between_std_ratio={ratio:.3f})")
                # Recommend group-aware splitting; the suite already supports this via
                # ``group_column`` in the split config, so we just surface the hint.
                knob_overrides.setdefault("split_config", {})["prefer_group_aware"] = True
                _stamp_prov("split_config", "prefer_group_aware", True, "clustered_target")
                # E5.2: clustered targets share the SAME MLP-LayerNorm collapse mode
                # as strong-AR (per-row LayerNorm destroys the between-row absolute-scale signal
                # that the group label encodes). The strong-AR detector above already turns
                # LayerNorm off when has_time_axis is True OR per-group AR fires; this branch
                # closes the third path (group-clustered target where AR is weak but the
                # absolute group level dominates -- e.g. group-level target magnitude, customer LTV by tenure).
                knob_overrides.setdefault("mlp_kwargs", {}).setdefault("network_params", {})["use_layernorm"] = False
                _stamp_prov("mlp_kwargs", "network_params.use_layernorm", False, "clustered_target")


def _tail_shape_pathologies(kurt, pathologies, knob_overrides, _stamp_prov, y_for_stats, _z2, _z, diagnostics):
    """Stage of ``analyze_target_distribution``."""
    if kurt > _HEAVY_TAIL_EXCESS_KURT:
        pathologies.append(f"heavy_tail(excess_kurt={kurt:.1f})")
        _apply_heavy_tail_huber_overrides(kurt, knob_overrides, _stamp_prov, huber_delta_for(y_for_stats))

    # Skewness (from the same standardised z as the kurtosis above; z*z*z == z2*z elementwise).
    skew = float(np.mean(_z2 * _z))
    diagnostics["skew"] = skew
    if abs(skew) > _SKEW_ABS_THRESHOLD:
        pathologies.append(f"skewed_target(skew={skew:.2f})")
        # No automatic knob override here -- log transform is a data
        # transformation, not a hyperparameter. The composite-discovery
        # block already considers log/sqrt residual targets when its
        # auto-detector flags skew; we surface the warning so the
        # operator knows it's worth investigating that path.

    # Multi-modal
