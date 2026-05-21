"""Feature-distribution drift report -- train/val/test feature-side shift detection.

Complements ``label_distribution_drift`` (which catches target-side prior
shift) by reporting per-feature mean / std drift across train / val / test.
Catches the MLP-collapse failure mode the TVT-2026-05-21 prod log surfaced:
Ridge gets R^2=1.00 (linear extrapolation handles group-shift fine) while
MLP gets R^2=-314 (non-linear extrapolation produces garbage when test
features sit several stds away from train).

Implementation notes
--------------------
- Per-numeric-feature drift = (test_mean - train_mean) / train_std (z-score).
- We only consider numeric columns; categorical drift via PSI is a
  separate (more expensive) pass.
- Threshold default 3.0 sigma matches the empirical "MLP starts failing
  beyond ~3 SDs out-of-distribution" rule of thumb. WARN-only -- the
  sensor doesn't change model behaviour, just surfaces the risk so the
  operator can choose to (a) add the feature to the no-extrapolate list,
  (b) switch the model family, or (c) accept the K=2 catastrophic-dropout
  protection downstream.
- Report stamps into ``metadata["feature_distribution_drift"]`` so
  downstream observability tooling can plot per-feature drift trends.

Public API
----------
``compute_feature_distribution_drift(train_df, val_df, test_df, *,
warn_threshold_z=3.0) -> dict``
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


DEFAULT_FEATURE_DRIFT_WARN_THRESHOLD_Z: float = 3.0
"""Default warn threshold: emit if any feature's val or test mean is more
than this many train-std away from the train mean."""


def _numeric_columns(df: Any) -> List[str]:
    """Return the list of numeric column names from a pandas/polars DataFrame."""
    if df is None:
        return []
    # pandas path
    if hasattr(df, "select_dtypes"):
        try:
            return list(df.select_dtypes(include="number").columns)
        except Exception:
            return []
    # polars path -- duck-type the schema walk
    if hasattr(df, "schema") and hasattr(df, "columns"):
        try:
            import polars as pl
            return [name for name, dt in df.schema.items() if dt.is_numeric()]
        except Exception:
            return []
    return []


def _col_to_numpy(df: Any, col: str) -> Optional[np.ndarray]:
    """Best-effort 1-D numpy array for a single column across pandas / polars."""
    try:
        if hasattr(df, "loc"):
            return df[col].to_numpy()
        if hasattr(df, "to_numpy") and hasattr(df, "columns"):
            # polars frame
            return df[col].to_numpy()
    except Exception:
        return None
    return None


def compute_feature_distribution_drift(
    train_df: Any,
    val_df: Any,
    test_df: Any,
    *,
    warn_threshold_z: float = DEFAULT_FEATURE_DRIFT_WARN_THRESHOLD_Z,
    feature_names: Optional[List[str]] = None,
    max_features_in_log: int = 10,
) -> Dict[str, Any]:
    """Compute per-feature mean drift across train / val / test.

    Returns a dict with:
      - ``per_feature``: ``{col: {"train_mean", "train_std", "val_z",
        "test_z"}}`` for each numeric feature.
      - ``drift_candidates``: list of (col, max_abs_z) where max_abs_z
        exceeds ``warn_threshold_z``, sorted descending.
      - ``threshold``: the z-threshold that fired warnings.
      - ``n_numeric_features``: count of numeric features inspected.

    The function never raises on missing val/test (returns z=NaN for the
    missing slot). Per-feature z is NaN when train_std==0 (constant
    feature -- no drift signal can be extracted).

    WARN-logs the top ``max_features_in_log`` drift candidates so the
    operator sees the worst-offender list in the run log; the full
    per_feature dict is in the returned report for retrospective use.
    """
    cols = feature_names or _numeric_columns(train_df)
    per_feature: Dict[str, Dict[str, float]] = {}
    candidates: List[tuple[str, float]] = []
    for col in cols:
        train_vals = _col_to_numpy(train_df, col)
        if train_vals is None:
            continue
        train_vals = np.asarray(train_vals, dtype=np.float64)
        train_vals = train_vals[np.isfinite(train_vals)]
        if train_vals.size < 2:
            continue
        train_mean = float(np.mean(train_vals))
        train_std = float(np.std(train_vals))
        if train_std <= 0.0 or not math.isfinite(train_std):
            # Constant feature -- no drift signal computable.
            per_feature[col] = {
                "train_mean": train_mean,
                "train_std": 0.0,
                "val_z": float("nan"),
                "test_z": float("nan"),
            }
            continue

        def _z_for(other_df):
            if other_df is None:
                return float("nan")
            other = _col_to_numpy(other_df, col)
            if other is None:
                return float("nan")
            other = np.asarray(other, dtype=np.float64)
            other = other[np.isfinite(other)]
            if other.size < 2:
                return float("nan")
            return float((np.mean(other) - train_mean) / train_std)

        val_z = _z_for(val_df)
        test_z = _z_for(test_df)
        per_feature[col] = {
            "train_mean": train_mean,
            "train_std": train_std,
            "val_z": val_z,
            "test_z": test_z,
        }
        max_abs_z = max(
            abs(val_z) if math.isfinite(val_z) else 0.0,
            abs(test_z) if math.isfinite(test_z) else 0.0,
        )
        if max_abs_z > warn_threshold_z:
            candidates.append((col, max_abs_z))
    candidates.sort(key=lambda pair: -pair[1])
    if candidates:
        _shown = candidates[:max_features_in_log]
        _detail = ", ".join(f"{c}(|z|={z:.2f})" for c, z in _shown)
        if len(candidates) > max_features_in_log:
            _detail += f", +{len(candidates) - max_features_in_log} more"
        logger.warning(
            "[feature-distribution-drift] %d numeric feature(s) drift > %.1f sigma "
            "between train and val/test. Non-linear models (MLP) extrapolate "
            "poorly here -- K=2 ensemble catastrophic-dropout will catch the "
            "worst offenders, but switching to linear / tree models is the "
            "honest fix. Top drifters: %s",
            len(candidates), warn_threshold_z, _detail,
        )
    return {
        "per_feature": per_feature,
        "drift_candidates": [(c, z) for c, z in candidates],
        "threshold": warn_threshold_z,
        "n_numeric_features": len(per_feature),
    }
