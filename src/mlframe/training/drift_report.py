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
- DEFAULT_REGRESSION_REL_SHIFT_WARN_THRESHOLD — 20% relative move in level / dispersion / upper tail

Auto-emitted via train_mlframe_models_suite right after the
train/val/test split materialises, with the report tucked into
``metadata["label_distribution_drift"][target_type][cur_target_name]``
for retrospective inspection (and so test code can assert it fired).
"""
from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np

logger = logging.getLogger(__name__)


DEFAULT_BINARY_DRIFT_WARN_THRESHOLD_PP: float = 5.0
"""Default warn threshold for binary classification: emit if any split's
P(y=1) differs from train's by more than this many percentage points."""

DEFAULT_REGRESSION_MEAN_Z_WARN_THRESHOLD: float = 0.5
"""Default warn threshold for regression targets: emit if any split's
mean differs from train's by more than this many train-target sigma."""

DEFAULT_REGRESSION_REL_SHIFT_WARN_THRESHOLD: float = 0.20
"""Default warn threshold for a regression split's RELATIVE shift in level (mean), dispersion (std) or upper tail
(p99) against train. Scale-free, so it stays reachable on a heavy-tailed target where the sigma test cannot fire: a
production run whose target mean fell 59% and p99 fell 63% scored 0.09 sigma and reported no drift at all."""

_MIN_RESOLVABLE_SE_MULTIPLE: float = 5.0
"""A train statistic smaller than this many standard errors of the train mean is inside sampling noise, so its ratio
against a split carries no information and the relative check skips it."""

DEFAULT_MULTI_DRIFT_WARN_THRESHOLD_PP: float = 5.0
"""Default warn threshold for multiclass / multilabel: emit if any class's
rate in any split differs from train's by more than this many pp."""


from .utils import coerce_to_numpy as _coerce_to_numpy


def _to_numpy_or_none(arr: Any) -> np.ndarray | None:
    """Coerce target to a numpy array; return ``None`` for ``None`` inputs.

    Shape is preserved (no reshape). Use ``coerce_to_1d_numpy`` from
    ``training.utils`` directly if a 1-D contract is required.
    """
    _result = _coerce_to_numpy(arr, allow_none=True)
    return np.asarray(_result) if _result is not None else None


def _binary_split_summary(arr: np.ndarray) -> dict[str, float]:
    """Summarize one split of a binary target: sample count, positive count, and positive rate.

    NaN-robust like the regression/multiclass/multilabel summarizers in this module: rows with a
    missing label are excluded from ``n`` (and therefore from ``p_positive``'s denominator) rather than
    silently counted as negative -- ``arr == 1`` is False for NaN, so an un-filtered denominator would
    quietly deflate ``p_positive`` with no signal that labels were missing. ``n_missing`` surfaces the count.
    """
    arr = np.asarray(arr)
    try:
        arr_float = arr.astype(np.float64, copy=False)
        finite_mask = np.isfinite(arr_float)
    except (TypeError, ValueError):
        # Non-float-castable dtype (e.g. object array of bools/strings): no NaN concept, treat as all-finite.
        finite_mask = np.ones(arr.shape[0], dtype=bool)
    n_missing = int(arr.shape[0] - finite_mask.sum())
    arr = arr[finite_mask]
    n = int(arr.shape[0])
    if n == 0:
        return {"n": 0, "n_positive": 0, "p_positive": float("nan"), "n_missing": n_missing}
    n_pos = int((arr == 1).sum())
    return {
        "n": n,
        "n_positive": n_pos,
        "p_positive": n_pos / n,
        "n_missing": n_missing,
    }


def _multiclass_split_summary(arr: np.ndarray, classes: Sequence) -> dict[str, Any]:
    """Summarize one split of a multiclass target: per-class count and rate, computed via a single ``np.unique`` pass and reindexed to ``classes``' order/dtype so missing classes report zero rather than raising a KeyError."""
    n = int(arr.shape[0])
    # Single sort-based pass via np.unique(return_counts) instead of one ``(arr == c).sum()`` full scan per class.
    # Bit-identical (exact integer counts); the per-class dict is rebuilt in the caller's ``classes`` order/dtype.
    # Bench n=10M: 1.18x @K=5, 4.92x @K=20, 10.6x @K=100 (2026-06-15).
    if n:
        uvals, ucounts = np.unique(arr, return_counts=True)
        count_lookup = dict(zip(uvals.tolist(), ucounts.tolist()))
    else:
        count_lookup = {}
    counts = {int(c) if isinstance(c, (np.integer, int)) else c: int(count_lookup.get(c, 0)) for c in classes}
    rates = {k: (v / n if n else float("nan")) for k, v in counts.items()}
    return {"n": n, "counts": counts, "rates": rates}


def _multilabel_split_summary(arr: np.ndarray) -> dict[str, Any]:
    """For (N, K) label matrix, compute per-label positive rate.

    Handles three input shapes:
      - 2-D ndarray (N, K) - canonical
      - 1-D ndarray (N,) - single-label, reshaped to (N, 1)
      - 1-D object ndarray of per-row arrays - stacked to (N, K).
        The polars ``pl.List(pl.Int8)`` roundtrip lands here when a
        multilabel target column survives the polars->pandas conversion
        as object dtype with nested ``np.ndarray`` cells.
    """
    if hasattr(arr, "dtype") and arr.dtype == object and arr.ndim == 1 and arr.shape[0] > 0:
        try:
            arr = np.stack([np.asarray(c) for c in arr], axis=0)
        except (ValueError, TypeError):
            # Jagged / mixed-shape object array; leave as-is and let the
            # subsequent ndim check route via the 1-D fallback.
            pass
    n, k = (int(arr.shape[0]), int(arr.shape[1])) if arr.ndim == 2 else (int(arr.shape[0]), 1)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    pos_per_label = arr.sum(axis=0).astype(int).tolist()
    return {
        "n": n,
        "n_labels": k,
        "n_positive_per_label": pos_per_label,
        "p_positive_per_label": [(p / n if n else float("nan")) for p in pos_per_label],
    }


def _regression_split_summary(arr: np.ndarray) -> dict[str, float]:
    """Summarize one split of a regression target: count, NaN-robust mean/std/median, and 1st/99th percentiles."""
    n = int(arr.shape[0])
    nan = float("nan")
    if n == 0:
        return {"n": 0, "mean": nan, "std": nan, "median": nan, "p01": nan, "p99": nan, "min": nan, "max": nan,
                "atom": nan, "atom_share": 0.0, "n_below_atom": 0}
    arr = arr.astype(np.float64, copy=False)
    median = float(np.nanmedian(arr))
    finite = arr[np.isfinite(arr)]
    # A value holding at least half the rows is necessarily the median: one comparison pass finds a point mass (a 0 for
    # "no event", a filled constant) without a sort. Rows below it are what p01 hides: a target 74% zeros with a few
    # refunds at -2.17 printed p01=0, and the zero-inflation check that needs 0 to be the minimum silently declined.
    atom_share = float(np.count_nonzero(finite == median)) / finite.size if finite.size and np.isfinite(median) else 0.0
    has_atom = atom_share >= 0.5
    return {
        "n": n,
        "mean": float(np.nanmean(arr)),
        "std": float(np.nanstd(arr, ddof=1)) if n > 1 else 0.0,
        "median": median,
        "p01": float(np.nanquantile(arr, 0.01)),
        "p99": float(np.nanquantile(arr, 0.99)),
        "min": float(finite.min()) if finite.size else nan,
        "max": float(finite.max()) if finite.size else nan,
        "atom": median if has_atom else nan,
        "atom_share": atom_share if has_atom else 0.0,
        "n_below_atom": int(np.count_nonzero(finite < median)) if has_atom else 0,
    }


def _regression_shape_warnings(split_name: str, splits: dict[str, Any], drifts: dict[str, Any], rel_threshold: float) -> list[str]:
    """Relative level / dispersion / upper-tail shift warnings for ``splits[split_name]`` vs ``splits["train"]``, recording every ratio into ``drifts``.

    The sigma test alone is unreachable on a heavy-tailed target: sigma is set by the tail while the shift happens in
    the bulk, so a production target whose mean fell 59% and whose p99 fell 63% scored 0.09 sigma and reported no
    drift. These three ratios are scale-free and each catches a shift the others miss -- level (mean), dispersion
    (std) and upper tail (p99). A near-zero train reference makes a ratio meaningless rather than large, so those are
    skipped rather than warned on.
    """
    train_summary, split_summary = splits["train"], splits[split_name]
    out: list[str] = []
    checks = (
        ("mean", "level", "mean"),
        ("std", "dispersion", "std"),
        ("p99", "upper tail", "p99"),
    )
    # A ratio is only meaningful when its denominator is resolvable above sampling noise. A target centred on ~0 has
    # a train mean of a few standard errors, so ANY split produces a large ratio out of pure noise -- which would
    # replace one useless verdict with another. The standard error of the train mean is the resolution limit for
    # every statistic here, so one rule covers all three.
    train_std = float(train_summary.get("std") or 0.0)
    n_train = int(train_summary.get("n") or 0)
    resolvable = _MIN_RESOLVABLE_SE_MULTIPLE * train_std / np.sqrt(n_train) if train_std > 0 and n_train > 1 else 0.0
    for key, label, stat_name in checks:
        train_v = train_summary.get(key)
        split_v = split_summary.get(key)
        if train_v is None or split_v is None or not np.isfinite(train_v) or not np.isfinite(split_v):
            continue
        if train_v == 0 or abs(train_v) < resolvable:
            continue
        ratio = float(split_v) / float(train_v)
        drifts[f"{split_name}_{stat_name}_ratio_vs_train"] = ratio
        rel_shift = ratio - 1.0
        if abs(rel_shift) > rel_threshold:
            out.append(
                f"{split_name.upper()} {stat_name}={split_v:.4g} vs train {train_v:.4g} "
                f"({rel_shift:+.1%}, threshold +/-{rel_threshold:.0%}); regression target {label} shift suspected. "
                f"A sigma-scaled mean test can miss this on a heavy-tailed target."
            )
    return out


def compute_label_distribution_drift(
    train_target: Any,
    val_target: Any,
    test_target: Any,
    target_type: str,
    *,
    warn_threshold_pp: float = DEFAULT_BINARY_DRIFT_WARN_THRESHOLD_PP,
    regression_mean_z_threshold: float = DEFAULT_REGRESSION_MEAN_Z_WARN_THRESHOLD,
    regression_rel_shift_threshold: float = DEFAULT_REGRESSION_REL_SHIFT_WARN_THRESHOLD,
    multi_warn_threshold_pp: float = DEFAULT_MULTI_DRIFT_WARN_THRESHOLD_PP,
) -> dict[str, Any]:
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
    train = _to_numpy_or_none(train_target)
    val = _to_numpy_or_none(val_target)
    test = _to_numpy_or_none(test_target)

    if train is None or train.size == 0:
        # A legitimate-but-empty train split (0-row split, non-None) hit the multiclass branch's
        # ``np.concatenate([a for a in (train, val, test) if a is not None and a.size > 0])`` with an
        # empty list whenever val/test were also None -- raised a bare, context-free
        # "need at least one array to concatenate" instead of this same graceful no-op the `train is
        # None` case already gets. An empty train split carries no more baseline signal than a None one.
        return {
            "target_type": target_type,
            "splits": {},
            "drifts": {},
            "warnings": [f"train_target is {'None' if train is None else 'empty (0 rows)'} - no drift report computed."],
            "warn_threshold_pp": warn_threshold_pp,
        }

    # Detect multilabel: explicit target_type label, or 2-D ndarray, or
    # 1-D object dtype where each cell is itself an array (the polars
    # ``pl.List(pl.Int8)`` -> object-cell roundtrip surfacing in 3-way
    # fuzz: target_type sometimes arrives as ``"binary_classification"``
    # but the actual values came through as a 1-D object array of
    # per-row label vectors). Without the third clause, the routing
    # fell into ``_binary_split_summary``'s ``arr == 1`` and raised
    # ``truth value of an array ambiguous``.
    def _is_object_of_arrays(_a) -> bool:
        """Detect a 1-D object-dtype array whose cells are themselves array-likes (the polars ``pl.List`` -> object-cell roundtrip signature), so multilabel routing catches it even when ``target_type`` misreports "binary_classification"."""
        try:
            if not hasattr(_a, "dtype"):
                return False
            if _a.dtype != object or _a.ndim != 1 or _a.shape[0] == 0:
                return False
            _first = _a[0]
            return hasattr(_first, "shape") or (hasattr(_first, "__len__") and not isinstance(_first, (str, bytes)))
        except Exception as exc:
            logger.debug("drift: object-of-arrays probe failed, treating as not object-of-arrays: %s", exc)
            return False

    is_multilabel = target_type == "multilabel_classification" or (hasattr(train, "ndim") and train.ndim == 2) or _is_object_of_arrays(train)
    is_regression = target_type == "regression"
    is_multiclass = target_type == "multiclass_classification"
    # Default fall-through: binary classification.

    splits: dict[str, Any] = {}
    warnings: list[str] = []
    drifts: dict[str, Any] = {}

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
                    f"{split_name.upper()} mean={split_mean:.4g} vs train {train_mean:.4g} (Δ={delta:+.4g}, z={z:+.2f}σ "
                    f"vs train σ={train_std:.4g}); regression target shift suspected."
                )
            warnings.extend(_regression_shape_warnings(split_name, splits, drifts, regression_rel_shift_threshold))

    elif is_multiclass:
        # Discover class labels from the union of all three splits.
        all_arr = np.concatenate([a for a in (train, val, test) if a is not None and a.size > 0])
        classes = np.unique(all_arr).tolist()
        for name, arr in (("train", train), ("val", val), ("test", test)):
            splits[name] = _multiclass_split_summary(arr, classes) if arr is not None else None
        train_rates = splits["train"]["rates"]
        for split_name in ("val", "test"):
            if splits.get(split_name) is None:
                continue
            split_rates = splits[split_name]["rates"]
            per_class_pp = {c: (split_rates[c] - train_rates[c]) * 100 for c in classes}
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
                    f"{split_name.upper()} P(y=1)={split_p:.3f} vs train {train_p:.3f} (Δ={delta_pp:+.1f}pp); selection-bias / "
                    f"prior-shift suspected - model will be miscalibrated on {split_name}. The remedy is a held-out calibration split "
                    f"(TrainingSplitConfig.calib_size); without one the suite can measure the miscalibration but cannot correct it."
                )
        # Track val-vs-test as a separate diagnostic (val-test mismatch
        # is common with shuffled val + temporal test).
        if splits.get("val") is not None and splits.get("test") is not None:
            v = splits["val"]["p_positive"]
            t = splits["test"]["p_positive"]
            drifts["test_minus_val_pp"] = float((t - v) * 100)
        drifts["max_abs_drift_pp"] = max(
            (abs(d) for k, d in drifts.items() if k.endswith("_minus_train_pp")),
            default=0.0,
        )

    # Report target_type that matches the splits format the formatter
    # expects: when shape detection routed to the multilabel branch but
    # the caller-supplied ``target_type`` was something else (binary /
    # unlabelled / mismatched), reflect the actual format used so
    # ``format_label_distribution_drift_report`` reads the right keys.
    effective_target_type = "multilabel_classification" if is_multilabel else target_type
    return {
        "target_type": effective_target_type,
        "splits": splits,
        "drifts": drifts,
        "warnings": warnings,
        "warn_threshold_pp": warn_threshold_pp,
    }


def format_drift_report(report: dict[str, Any], target_name: str = "") -> str:
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
            line = (f"  {name:<5} n={s['n']:>10_} mean={s['mean']:.4g} std={s['std']:.4g} median={s['median']:.4g} "
                    f"p01={s['p01']:.4g} p99={s['p99']:.4g} min={s.get('min', float('nan')):.4g} max={s.get('max', float('nan')):.4g}")
            if s.get("atom_share", 0.0) >= 0.5:
                line += f" point_mass={s['atom']:.4g} ({s['atom_share']:.0%})"
                if s.get("n_below_atom"):
                    line += f", {s['n_below_atom']:_} row(s) below it"
            lines.append(line)
    elif target_type == "multilabel_classification":
        for name in ("train", "val", "test"):
            s = splits.get(name)
            if s is None:
                continue
            rates_str = ", ".join(f"{r:.3f}" for r in s["p_positive_per_label"])
            lines.append(f"  {name:<5} n={s['n']:>10_} p_positive_per_label=[{rates_str}]")
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
            lines.append(f"  {name:<5} n={s['n']:>10_} n_positive={s['n_positive']:>10_} " f"P(y=1)={s['p_positive']:.4f}")

    if warnings:
        lines.extend(f"  WARN: {w}" for w in warnings)
    else:
        lines.append("  (no drift warnings - splits within threshold)")

    return "\n".join(lines)
