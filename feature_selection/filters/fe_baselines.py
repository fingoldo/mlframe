"""Honest non-polynomial baselines for pair feature engineering.

Per the brainstorm-agent meta-finding "honest non-poly baselines are
critical": before claiming a polynomial-engineered feature is useful,
quantify what TRIVIAL pair operations (multiplication, ratio, log,
min/max) deliver. The polynomial may add ~0% over those.

Public API:
* ``trivial_pair_features(x_a, x_b)`` -- returns a dict of named
  trivial pair features (numpy arrays, same length as x_a).
* ``score_trivial_baselines(x_a, x_b, y, discrete_target)`` -- runs
  each trivial feature through the configured MI estimator and
  returns ``{name: mi_value}`` sorted descending.
* ``best_trivial_pair(x_a, x_b, y, discrete_target)`` -- single best
  ``(name, feature_array, mi_value)`` to use as a comparison baseline
  for ``optimise_hermite_pair``.

Use this AS THE FE BASELINE instead of (or in addition to) the
``MI(x_a, x_b)`` joint MI. If your polynomial-engineered MI doesn't
clear ``best_trivial * 1.05``, the polynomial is not worth it -- emit
the trivial feature instead.
"""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np


def trivial_pair_features(x_a: np.ndarray, x_b: np.ndarray) -> dict:
    """Return a dict of trivial pair-feature transforms. All produce
    1-D arrays the same length as ``x_a``. NaN/inf cells are masked
    by the consuming code; here we emit raw values + ratio with
    epsilon for safety."""
    x_a = np.asarray(x_a, dtype=np.float64)
    x_b = np.asarray(x_b, dtype=np.float64)
    eps = 1e-9
    feats = {}
    feats["mul"] = x_a * x_b
    feats["add"] = x_a + x_b
    feats["sub"] = x_a - x_b
    # Ratio with stable epsilon. We add eps to denominator to avoid
    # division by zero; the chosen epsilon is small relative to typical
    # feature scales after the standard z-score / minmax preprocessing.
    feats["ratio_ab"] = x_a / (x_b + np.sign(x_b) * eps + eps)
    feats["ratio_ba"] = x_b / (x_a + np.sign(x_a) * eps + eps)
    # Distance / max / min -- common in gradient-boosting feature
    # engineering libraries.
    feats["sq_dist"] = (x_a - x_b) ** 2
    feats["sum_sq"] = x_a ** 2 + x_b ** 2
    feats["maxab"] = np.maximum(x_a, x_b)
    feats["minab"] = np.minimum(x_a, x_b)
    # Log-magnitude with sign retained -- captures multiplicative
    # structure (log(|a*b|) = log|a| + log|b|).
    feats["log_abs_mul"] = (np.log(np.abs(x_a) + eps)
                              + np.log(np.abs(x_b) + eps)
                              ) * np.sign(x_a * x_b + eps)
    # Atan2 -- captures angular interactions on 2D inputs.
    feats["atan2"] = np.arctan2(x_a, x_b)
    # Geometric mean (sign-aware).
    feats["geo_mean"] = np.sign(x_a * x_b) * np.sqrt(np.abs(x_a * x_b) + eps)
    return feats


def _mi_1d(x: np.ndarray, y: np.ndarray, *, discrete_target: bool,
            mi_estimator: str = "plugin", plugin_n_bins: int = 20,
            n_neighbors: int = 3) -> float:
    """1-D MI(x, y) using the configured estimator. Wraps both the
    fast plug-in and the slower KSG paths."""
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

    The dict's iteration order is the rank order, so
    ``next(iter(scores))`` is the winner."""
    feats = trivial_pair_features(x_a, x_b)
    scores = {}
    for name, f in feats.items():
        if not np.all(np.isfinite(f)):
            continue
        scores[name] = _mi_1d(
            f, y, discrete_target=discrete_target,
            mi_estimator=mi_estimator, plugin_n_bins=plugin_n_bins,
            n_neighbors=n_neighbors,
        )
    return dict(sorted(scores.items(), key=lambda kv: -kv[1]))


def best_trivial_pair(
    x_a: np.ndarray, x_b: np.ndarray, y: np.ndarray, *,
    discrete_target: bool = True,
    mi_estimator: str = "plugin",
    plugin_n_bins: int = 20,
    n_neighbors: int = 3,
) -> tuple:
    """Return ``(name, feature_array, mi_value)`` for the best trivial
    pair feature. ``None`` if all trivial features are non-finite."""
    feats = trivial_pair_features(x_a, x_b)
    best_name = None
    best_arr = None
    best_mi = -np.inf
    for name, f in feats.items():
        if not np.all(np.isfinite(f)):
            continue
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
