"""Label distribution drift report — train/val/test prior shift detection.

Catches the most common silent failure mode of forward-mode (temporal)
splits: the marginal P(y) shifts between splits and the model trains in
one regime, gets evaluated in another, and posts a confusing AUC drop
that is read as "the model overfits" when in fact the calibration is
broken because the prior moved.

The classic example: positive class is rare in train (e.g. 30%),
selection bias inflates val and test (e.g. 80% positive). Naive
classifier learns "P(y=1)≈0.30" prior, gets blown out on val/test.
Computing this drift takes <1ms and tells you up front, BEFORE you
invest 5 hours of training, that you have a label-shift problem.

Public surface:
- compute_label_distribution_drift(...)
- format_drift_report(...)
- DEFAULT_BINARY_DRIFT_WARN_THRESHOLD_PP — 5 percentage points
- DEFAULT_REGRESSION_MEAN_Z_WARN_THRESHOLD — 0.5 sigma

Auto-emitted via train_mlframe_models_suite right after the
train/val/test split materialises, with the report tucked into
``metadata["label_distribution_drift"][target_type][cur_target_name]``
for retrospective inspection (and so test code can assert it fired).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


DEFAULT_BINARY_DRIFT_WARN_THRESHOLD_PP: float = 5.0
"""Default warn threshold for binary classification: emit if any split's
P(y=1) differs from train's by more than this many percentage points."""

DEFAULT_REGRESSION_MEAN_Z_WARN_THRESHOLD: float = 0.5
"""Default warn threshold for regression targets: emit if any split's
mean differs from train's by more than this many train-target sigma."""

DEFAULT_MULTI_DRIFT_WARN_THRESHOLD_PP: float = 5.0
"""Default warn threshold for multiclass / multilabel: emit if any class's
rate in any split differs from train's by more than this many pp."""


def _to_1d_numpy(arr: Any) -> Optional[np.ndarray]:
    """Coerce target to a numpy array; return None for None inputs."""
    if arr is None:
        return None
    if hasattr(arr, "to_numpy"):
        out = arr.to_numpy()
    elif hasattr(arr, "values"):
        out = arr.values
    else:
        out = np.asarray(arr)
    return out


def _binary_split_summary(arr: np.ndarray) -> Dict[str, float]:
    n = int(arr.shape[0])
    if n == 0:
        return {"n": 0, "n_positive": 0, "p_positive": float("nan")}
    n_pos = int((arr == 1).sum())
    return {
        "n": n,
        "n_positive": n_pos,
        "p_positive": n_pos / n,
    }


def _multiclass_split_summary(arr: np.ndarray, classes: Sequence) -> Dict[str, Any]:
    n = int(arr.shape[0])
    counts = {int(c) if isinstance(c, (np.integer, int)) else c:
              int((arr == c).sum()) for c in classes}
    rates = {k: (v / n if n else float("nan")) for k, v in counts.items()}
    return {"n": n, "counts": counts, "rates": rates}


def _multilabel_split_summary(arr: np.ndarray) -> Dict[str, Any]:
    """For (N, K) label matrix, compute per-label positive rate."""
    n, k = (int(arr.shape[0]), int(arr.shape[1])) if arr.ndim == 2 else (int(arr.shape[0]), 1)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    pos_per_label = arr.sum(axis=0).astype(int).tolist()
    return {
        "n": n,
        "n_labels": k,
        "n_positive_per_label": pos_per_label,
        "p_positive_per_label": [
            (p / n if n else float("nan")) for p in pos_per_label
        ],
    }


def _regression_split_summary(arr: np.ndarray) -> Dict[str, float]:
    n = int(arr.shape[0])
    if n == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"),
                "median": float("nan"), "p01": float("nan"), "p99": float("nan")}
    arr = arr.astype(np.float64, copy=False)
    return {
        "n": n,
        "mean": float(np.nanmean(arr)),
        "std": float(np.nanstd(arr, ddof=1)) if n > 1 else 0.0,
        "median": float(np.nanmedian(arr)),
        "p01": float(np.nanquantile(arr, 0.01)),
        "p99": float(np.nanquantile(arr, 0.99)),
    }


def compute_label_distribution_drift(
    train_target: Any,
    val_target: Any,
    test_target: Any,
    target_type: str,
    *,
    warn_threshold_pp: float = DEFAULT_BINARY_DRIFT_WARN_THRESHOLD_PP,
    regression_mean_z_threshold: float = DEFAULT_REGRESSION_MEAN_Z_WARN_THRESHOLD,
    multi_warn_threshold_pp: float = DEFAULT_MULTI_DRIFT_WARN_THRESHOLD_PP,
) -> Dict[str, Any]:
    """Compute label-distribution drift between train, val, test splits.

    The routine is type-aware:

    * binary — reports n / n_positive / p_positive per split, cross-split
      Δ in percentage points, and warns when |max Δpp| exceeds the
      threshold (default 5pp).
    * multiclass — same per-class. Warns when ANY class's rate drifts
      beyond the threshold in any split.
    * multilabel — per-label P(y_k=1). Warns when ANY label drifts.
    * regression — mean / std / median / p01 / p99. Warns when val or
      test mean is more than ``regression_mean_z_threshold`` sigma away
      from train mean (sigma estimated from train).

    The warn threshold is intentionally aggressive (5pp / 0.5σ) because
    the cost of a false positive is one log line, while the cost of
    missing a real shift is hours of compute on a miscalibrated model.

    Parameters
    ----------
    train_target, val_target, test_target
        Target arrays for each split; coerced via ``.to_numpy()`` or
        ``.values`` if available. ``val_target`` or ``test_target`` may
        be ``None`` (returns the corresponding entries as ``None``).
    target_type
        TargetTypes string value: "binary_classification",
        "multiclass_classification", "multilabel_classification", or
        "regression". Other strings → behaves as binary.
    warn_threshold_pp
        Threshold in percentage points for binary drift warnings.
    regression_mean_z_threshold
        Threshold in train-sigma units for regression drift warnings.
    multi_warn_threshold_pp
        Threshold for multiclass / multilabel per-class warnings.

    Returns
    -------
    dict
        Structured report with keys ``target_type``, ``splits`` (per-split
        summary), ``drifts`` (cross-split deltas), ``warnings`` (list of
        human-readable strings; empty when everything is within
        threshold), ``warn_threshold_pp`` (the threshold used).
    """
    train = _to_1d_numpy(train_target)
    val = _to_1d_numpy(val_target)
    test = _to_1d_numpy(test_target)

    if train is None:
        return {
            "target_type": target_type,
            "splits": {},
            "drifts": {},
            "warnings": ["train_target is None — no drift report computed."],
            "warn_threshold_pp": warn_threshold_pp,
        }

    is_multilabel = (
        target_type == "multilabel_classification"
        or (hasattr(train, "ndim") and train.ndim == 2)
    )
    is_regression = (target_type == "regression")
    is_multiclass = (target_type == "multiclass_classification")
    # Default fall-through: binary classification.

    splits: Dict[str, Any] = {}
    warnings: List[str] = []
    drifts: Dict[str, Any] = {}

    if is_multilabel:
        for name, arr in (("train", train), ("val", val), ("test", test)):
            splits[name] = _multilabel_split_summary(arr) if arr is not None else None
        # Per-label drift in pp
        train_rates = splits["train"]["p_positive_per_label"]
        for split_name in ("val", "test"):
            if splits.get(split_name) is None:
                continue
            split_rates = splits[split_name]["p_positive_per_label"]
            deltas = [(s - t) * 100 for s, t in zip(split_rates, train_rates)]
            drifts[f"{split_name}_minus_train_pp_per_label"] = deltas
            for k, d in enumerate(deltas):
                if abs(d) > multi_warn_threshold_pp:
                    warnings.append(
                        f"{split_name.upper()} P(y_{k}=1)={split_rates[k]:.3f} vs "
                        f"train {train_rates[k]:.3f} (Δ={d:+.1f}pp); "
                        f"label-shift suspected on label {k}."
                    )

    elif is_regression:
        for name, arr in (("train", train), ("val", val), ("test", test)):
            splits[name] = _regression_split_summary(arr) if arr is not None else None
        train_mean = splits["train"]["mean"]
        train_std = splits["train"]["std"] or float("nan")
        for split_name in ("val", "test"):
            if splits.get(split_name) is None:
                continue
            split_mean = splits[split_name]["mean"]
            delta = split_mean - train_mean
            z = (delta / train_std) if (train_std and not np.isnan(train_std) and train_std > 0) else float("nan")
            drifts[f"{split_name}_mean_minus_train"] = float(delta)
            drifts[f"{split_name}_mean_z_vs_train"] = float(z)
            if not np.isnan(z) and abs(z) > regression_mean_z_threshold:
                warnings.append(
                    f"{split_name.upper()} mean={split_mean:.4g} vs "
                    f"train {train_mean:.4g} (Δ={delta:+.4g}, z={z:+.2f}σ "
                    f"vs train σ={train_std:.4g}); regression target shift suspected."
                )

    elif is_multiclass:
        # Discover class labels from the union of all three splits.
        all_arr = np.concatenate(
            [a for a in (train, val, test) if a is not None and a.size > 0]
        )
        classes = list(np.unique(all_arr).tolist())
        for name, arr in (("train", train), ("val", val), ("test", test)):
            splits[name] = _multiclass_split_summary(arr, classes) if arr is not None else None
        train_rates = splits["train"]["rates"]
        for split_name in ("val", "test"):
            if splits.get(split_name) is None:
                continue
            split_rates = splits[split_name]["rates"]
            per_class_pp = {
                c: (split_rates[c] - train_rates[c]) * 100 for c in classes
            }
            drifts[f"{split_name}_minus_train_pp_per_class"] = per_class_pp
            for c, d in per_class_pp.items():
                if abs(d) > multi_warn_threshold_pp:
                    warnings.append(
                        f"{split_name.upper()} P(y={c})={split_rates[c]:.3f} vs "
                        f"train {train_rates[c]:.3f} (Δ={d:+.1f}pp); "
                        f"class-prior shift suspected for class {c}."
                    )

    else:
        # Binary classification (default).
        for name, arr in (("train", train), ("val", val), ("test", test)):
            splits[name] = _binary_split_summary(arr) if arr is not None else None
        train_p = splits["train"]["p_positive"]
        for split_name in ("val", "test"):
            if splits.get(split_name) is None:
                continue
            split_p = splits[split_name]["p_positive"]
            delta_pp = (split_p - train_p) * 100
            drifts[f"{split_name}_minus_train_pp"] = float(delta_pp)
            if abs(delta_pp) > warn_threshold_pp:
                warnings.append(
                    f"{split_name.upper()} P(y=1)={split_p:.3f} vs train "
                    f"{train_p:.3f} (Δ={delta_pp:+.1f}pp); selection-bias / "
                    f"prior-shift suspected — model will be miscalibrated on "
                    f"{split_name}."
                )
        # Track val-vs-test as a separate diagnostic (val-test mismatch
        # is common with shuffled val + temporal test).
        if splits.get("val") is not None and splits.get("test") is not None:
            v = splits["val"]["p_positive"]
            t = splits["test"]["p_positive"]
            drifts["test_minus_val_pp"] = float((t - v) * 100)
        drifts["max_abs_drift_pp"] = max(
            (abs(d) for k, d in drifts.items()
             if k.endswith("_minus_train_pp")),
            default=0.0,
        )

    return {
        "target_type": target_type,
        "splits": splits,
        "drifts": drifts,
        "warnings": warnings,
        "warn_threshold_pp": warn_threshold_pp,
    }


def format_drift_report(report: Dict[str, Any], target_name: str = "") -> str:
    """One-line-per-split human-readable rendering for log output.

    Compact enough to stamp into the log right before training; verbose
    enough to spot the problem at a glance without re-loading metadata.
    """
    target_type = report["target_type"]
    splits = report["splits"]
    warnings = report["warnings"]

    label = f" target={target_name}" if target_name else ""
    lines = [f"label_distribution_drift report (target_type={target_type}{label}):"]

    if target_type == "regression":
        for name in ("train", "val", "test"):
            s = splits.get(name)
            if s is None:
                continue
            lines.append(
                f"  {name:<5} n={s['n']:>10_} mean={s['mean']:.4g} "
                f"std={s['std']:.4g} median={s['median']:.4g} "
                f"p01={s['p01']:.4g} p99={s['p99']:.4g}"
            )
    elif target_type == "multilabel_classification":
        for name in ("train", "val", "test"):
            s = splits.get(name)
            if s is None:
                continue
            rates_str = ", ".join(f"{r:.3f}" for r in s["p_positive_per_label"])
            lines.append(
                f"  {name:<5} n={s['n']:>10_} p_positive_per_label=[{rates_str}]"
            )
    elif target_type == "multiclass_classification":
        for name in ("train", "val", "test"):
            s = splits.get(name)
            if s is None:
                continue
            rates_str = ", ".join(f"{c}:{r:.3f}" for c, r in s["rates"].items())
            lines.append(f"  {name:<5} n={s['n']:>10_} rates={{{rates_str}}}")
    else:  # binary
        for name in ("train", "val", "test"):
            s = splits.get(name)
            if s is None:
                continue
            lines.append(
                f"  {name:<5} n={s['n']:>10_} n_positive={s['n_positive']:>10_} "
                f"P(y=1)={s['p_positive']:.4f}"
            )

    if warnings:
        for w in warnings:
            lines.append(f"  WARN: {w}")
    else:
        lines.append("  (no drift warnings — splits within threshold)")

    return "\n".join(lines)
