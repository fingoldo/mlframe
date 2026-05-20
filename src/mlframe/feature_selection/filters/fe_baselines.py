"""Honest non-polynomial baselines for pair feature engineering.

Before claiming a polynomial-engineered feature is useful, quantify what TRIVIAL pair operations (multiplication, ratio, log, min/max) deliver.
The polynomial may add ~0% over those.

Public API:
* ``trivial_pair_features(x_a, x_b)`` -- dict of named trivial pair features (numpy arrays, same length as x_a).
* ``score_trivial_baselines(x_a, x_b, y, discrete_target)`` -- runs each trivial feature through the configured MI estimator and returns
  ``{name: mi_value}`` sorted descending.
* ``best_trivial_pair(x_a, x_b, y, discrete_target)`` -- single best ``(name, feature_array, mi_value)`` to use as a comparison baseline
  for ``optimise_hermite_pair``.

Use as the FE baseline instead of (or in addition to) the ``MI(x_a, x_b)`` joint MI. If your polynomial-engineered MI doesn't clear
``best_trivial * 1.05``, the polynomial is not worth it -- emit the trivial feature instead.
"""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np


def trivial_pair_features(x_a: np.ndarray, x_b: np.ndarray) -> dict:
    """Return a dict of trivial pair-feature transforms.

    All produce 1-D arrays the same length as ``x_a``. NaN/inf cells are masked by the consuming code; here we emit raw values plus a
    ratio with epsilon for safety.
    """
    x_a = np.asarray(x_a, dtype=np.float64)
    x_b = np.asarray(x_b, dtype=np.float64)
    eps = 1e-9
    feats = {}
    feats["mul"] = x_a * x_b
    feats["add"] = x_a + x_b
    feats["sub"] = x_a - x_b
    # Ratio with stable epsilon: eps in denominator avoids div-by-zero and is small relative to z-score / minmax preprocessed scales.
    feats["ratio_ab"] = x_a / (x_b + np.sign(x_b) * eps + eps)
    feats["ratio_ba"] = x_b / (x_a + np.sign(x_a) * eps + eps)
    # Distance / max / min -- common in gradient-boosting FE libraries.
    feats["sq_dist"] = (x_a - x_b) ** 2
    feats["sum_sq"] = x_a ** 2 + x_b ** 2
    feats["maxab"] = np.maximum(x_a, x_b)
    feats["minab"] = np.minimum(x_a, x_b)
    # Log-magnitude with sign retained -- captures multiplicative structure (log(|a*b|) = log|a| + log|b|).
    feats["log_abs_mul"] = (np.log(np.abs(x_a) + eps) + np.log(np.abs(x_b) + eps)) * np.sign(x_a * x_b + eps)
    # Atan2 -- captures angular interactions on 2D inputs.
    feats["atan2"] = np.arctan2(x_a, x_b)
    # Geometric mean (sign-aware).
    feats["geo_mean"] = np.sign(x_a * x_b) * np.sqrt(np.abs(x_a * x_b) + eps)
    return feats


def _mi_1d(x: np.ndarray, y: np.ndarray, *, discrete_target: bool,
            mi_estimator: str = "plugin", plugin_n_bins: int = 20,
            n_neighbors: int = 3) -> float:
    """1-D MI(x, y) using the configured estimator (fast plug-in or slower KSG)."""
    if not np.all(np.isfinite(x)):
        return 0.0
    if mi_estimator == "plugin":
        # Lazy-import to avoid circular dep.
        from .hermite_fe import (
            _plugin_mi_classif_njit,
            _plugin_mi_regression_njit,
        )
        x_njit = np.ascontiguousarray(x, dtype=np.float64)
        if discrete_target:
            y_njit = np.asarray(y, dtype=np.int64)
            return float(_plugin_mi_classif_njit(x_njit, y_njit, plugin_n_bins))
        y_njit = np.asarray(y, dtype=np.float64)
        return float(_plugin_mi_regression_njit(x_njit, y_njit, plugin_n_bins))
    # ksg path
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if discrete_target:
            return float(mutual_info_classif(
                x.reshape(-1, 1), y, n_neighbors=n_neighbors,
                random_state=42, discrete_features=False,
            )[0])
        return float(mutual_info_regression(
            x.reshape(-1, 1), y, n_neighbors=n_neighbors,
            random_state=42, discrete_features=False,
        )[0])


def score_trivial_baselines(
    x_a: np.ndarray, x_b: np.ndarray, y: np.ndarray, *,
    discrete_target: bool = True,
    mi_estimator: str = "plugin",
    plugin_n_bins: int = 20,
    n_neighbors: int = 3,
) -> dict:
    """Return ``{trivial_feature_name: mi_value}`` sorted descending.

    The dict's iteration order is the rank order, so ``next(iter(scores))`` is the winner.

    2026-05-18 PERFORMANCE: previously called ``_mi_1d`` 12x serially (~25ms each = 300ms total on n=200k).
    Now batches all 12 trivial features into a single
    ``_plugin_mi_classif_batch_njit`` / ``_plugin_mi_regression_batch_njit``
    call which parallelises over columns via ``@njit(parallel=True)``.
    Measured ~12x speedup on this hot path (cProfile 2026-05-18 showed
    best_trivial_pair was 25% of polynom-FE wall time).
    """
    feats = trivial_pair_features(x_a, x_b)
    # Filter finite-only features ONCE; stack into (n, k) for batch MI.
    valid_names = []
    valid_cols = []
    for name, f in feats.items():
        if np.all(np.isfinite(f)):
            valid_names.append(name)
            valid_cols.append(f)
    if not valid_cols:
        return {}
    if mi_estimator == "plugin":
        # Lazy-import to break circular dep.
        from .hermite_fe import (
            _plugin_mi_classif_batch_njit,
            _plugin_mi_regression_batch_njit,
        )
        X_batch = np.ascontiguousarray(
            np.column_stack(valid_cols), dtype=np.float64,
        )
        if discrete_target:
            y_njit = np.asarray(y, dtype=np.int64)
            mi_arr = _plugin_mi_classif_batch_njit(X_batch, y_njit, plugin_n_bins)
        else:
            y_njit = np.asarray(y, dtype=np.float64)
            mi_arr = _plugin_mi_regression_batch_njit(X_batch, y_njit, plugin_n_bins)
        scores = {name: float(mi_arr[i]) for i, name in enumerate(valid_names)}
    else:
        # KSG path: no batch kernel exists, fall back to per-feature loop.
        scores = {}
        for name, f in zip(valid_names, valid_cols):
            scores[name] = _mi_1d(
                f, y, discrete_target=discrete_target,
                mi_estimator=mi_estimator, plugin_n_bins=plugin_n_bins,
                n_neighbors=n_neighbors,
            )
    return dict(sorted(scores.items(), key=lambda kv: -kv[1]))


def auto_unary_transforms(x: np.ndarray, y: np.ndarray, *,
                            discrete_target: bool = True,
                            mi_estimator: str = "plugin",
                            plugin_n_bins: int = 20,
                            n_neighbors: int = 3,
                            min_uplift: float = 1.05) -> dict:
    """Probe a small set of unary transforms (log, sqrt, 1/x, exp clipping) and return ``{name: (transformed_x, mi)}`` for those that beat
    the identity ``MI(x, y)`` by ``min_uplift``.

    Useful as a pre-step before the polynomial-pair search: replace each input feature by the unary transform that best correlates with y.
    Linearises multiplicative / log-normal / exponential relationships; e.g. ``y = sign(x_a * x_b)`` becomes ``log|x_a| + log|x_b|`` after
    the unary step, which the pair optimizer can then express trivially.
    """
    eps = 1e-9
    x = np.asarray(x, dtype=np.float64)
    transforms = {
        "identity": x,
        "log_abs": np.log(np.abs(x) + eps),
        "sqrt_abs_signed": np.sign(x) * np.sqrt(np.abs(x)),
        "inv": np.sign(x) / (np.abs(x) + eps),
        "square": x ** 2,
        "cube": x ** 3,
        "tanh": np.tanh(x),
    }
    base = _mi_1d(x, y, discrete_target=discrete_target,
                   mi_estimator=mi_estimator, plugin_n_bins=plugin_n_bins,
                   n_neighbors=n_neighbors)
    out = {}
    for name, arr in transforms.items():
        if not np.all(np.isfinite(arr)):
            continue
        mi = _mi_1d(arr, y, discrete_target=discrete_target,
                    mi_estimator=mi_estimator, plugin_n_bins=plugin_n_bins,
                    n_neighbors=n_neighbors)
        if name == "identity" or mi >= base * min_uplift:
            out[name] = (arr, float(mi))
    return out


def best_unary_transform(x: np.ndarray, y: np.ndarray, **kwargs) -> tuple:
    """Single best unary: ``(name, transformed_x, mi)``."""
    candidates = auto_unary_transforms(x, y, min_uplift=0.0, **kwargs)
    if not candidates:
        return "identity", x, 0.0
    name, (arr, mi) = max(candidates.items(), key=lambda kv: kv[1][1])
    return name, arr, mi


def triplet_pair_features(x_a: np.ndarray, x_b: np.ndarray,
                            x_c: np.ndarray) -> dict:
    """3-way pair-style features. Captures ``y = sign(x_a * x_b * x_c)`` and similar 3-way interactions that pair-FE cannot represent."""
    x_a = np.asarray(x_a, dtype=np.float64)
    x_b = np.asarray(x_b, dtype=np.float64)
    x_c = np.asarray(x_c, dtype=np.float64)
    eps = 1e-9
    return {
        "abc_mul": x_a * x_b * x_c,
        "ab_minus_c": x_a * x_b - x_c,
        "ab_div_c": (x_a * x_b) / (x_c + np.sign(x_c) * eps + eps),
        "a_plus_bc": x_a + x_b * x_c,
        "a_times_bc": x_a * (x_b + x_c),
        "ab_plus_ac": x_a * x_b + x_a * x_c,
        "sum_of_squares": x_a ** 2 + x_b ** 2 + x_c ** 2,
        "atan2_ab_c": np.arctan2(x_a * x_b, x_c),
        "geo_mean3": np.sign(x_a * x_b * x_c) * np.cbrt(np.abs(x_a * x_b * x_c) + eps),
    }


def score_triplet_baselines(x_a, x_b, x_c, y, *,
                              discrete_target: bool = True,
                              mi_estimator: str = "plugin",
                              plugin_n_bins: int = 20,
                              n_neighbors: int = 3) -> dict:
    """Rank 3-way trivial features by MI."""
    feats = triplet_pair_features(x_a, x_b, x_c)
    scores = {}
    for name, f in feats.items():
        if not np.all(np.isfinite(f)):
            continue
        scores[name] = _mi_1d(f, y, discrete_target=discrete_target,
                              mi_estimator=mi_estimator,
                              plugin_n_bins=plugin_n_bins,
                              n_neighbors=n_neighbors)
    return dict(sorted(scores.items(), key=lambda kv: -kv[1]))


def best_trivial_pair(
    x_a: np.ndarray, x_b: np.ndarray, y: np.ndarray, *,
    discrete_target: bool = True,
    mi_estimator: str = "plugin",
    plugin_n_bins: int = 20,
    n_neighbors: int = 3,
) -> tuple:
    """Return ``(name, feature_array, mi_value)`` for the best trivial pair feature. ``None`` if all trivial features are non-finite.

    2026-05-18 PERFORMANCE: shares the same batch MI path as
    ``score_trivial_baselines`` (parallel over columns via numba).
    Pre-fix: 12 serial single-MI calls = 300ms on n=200k. Post-fix:
    1 batched call ~ 30ms (~10x speedup on this hot path).
    """
    feats = trivial_pair_features(x_a, x_b)
    valid_names = []
    valid_cols = []
    for name, f in feats.items():
        if np.all(np.isfinite(f)):
            valid_names.append(name)
            valid_cols.append(f)
    if not valid_cols:
        return None
    if mi_estimator == "plugin":
        from .hermite_fe import (
            _plugin_mi_classif_batch_njit,
            _plugin_mi_regression_batch_njit,
        )
        X_batch = np.ascontiguousarray(
            np.column_stack(valid_cols), dtype=np.float64,
        )
        if discrete_target:
            y_njit = np.asarray(y, dtype=np.int64)
            mi_arr = _plugin_mi_classif_batch_njit(X_batch, y_njit, plugin_n_bins)
        else:
            y_njit = np.asarray(y, dtype=np.float64)
            mi_arr = _plugin_mi_regression_batch_njit(X_batch, y_njit, plugin_n_bins)
        # Wave 21 P0: MI batch kernel may return NaN for degenerate
        # features. Pre-fix np.argmax picks the NaN's index -> wrong "best
        # baseline" + bogus float(mi_arr[best_idx]) feeds downstream MI gate.
        _finite_mask = np.isfinite(mi_arr)
        if not _finite_mask.any():
            # All-NaN: no valid baseline -- caller treats None as
            # "no baseline beat the gate" which is the correct semantic.
            return None, None, float("nan")
        if not _finite_mask.all():
            _finite_idx = np.where(_finite_mask)[0]
            best_idx = int(_finite_idx[np.argmax(mi_arr[_finite_idx])])
        else:
            best_idx = int(np.argmax(mi_arr))
        return valid_names[best_idx], valid_cols[best_idx], float(mi_arr[best_idx])
    # KSG path: no batch kernel, fall back to per-feature loop.
    best_name, best_arr, best_mi = None, None, -np.inf
    for name, f in zip(valid_names, valid_cols):
        mi = _mi_1d(
            f, y, discrete_target=discrete_target,
            mi_estimator=mi_estimator, plugin_n_bins=plugin_n_bins,
            n_neighbors=n_neighbors,
        )
        if mi > best_mi:
            best_mi = mi
            best_name = name
            best_arr = f
    if best_name is None:
        return None
    return best_name, best_arr, float(best_mi)
