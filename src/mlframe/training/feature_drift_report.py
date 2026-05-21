"""Feature-distribution drift report -- per-feature train/val/test shift signal.

OBSERVATIONAL ONLY. The 2026-05-22 review explicitly flagged 3 weaknesses
in any "feature drift -> auto-action" coupling:

1. **There is no operator**. The mlframe suite runs autonomously; "operator
   reads the WARN and decides" is the wrong framing for an autonomous
   pipeline. The sensor must produce signal the SYSTEM can use, not text
   a human will read.

2. **Drift does NOT prove model harm.** A 5-sigma drift on a feature with
   FI=0 is harmless; a 2-sigma drift on a feature with FI=0.8 (dominant)
   is potentially catastrophic. Per-feature z-score alone is NOT a
   grounded "the model will fail" signal -- pairing with feature
   importance is required for any actionable claim, and even then it's
   a correlation not a guarantee.

3. **Feature selection runs inside per-model pre_pipeline.** MRMR / RFECV
   drop features before the model sees them; this sensor flags features
   the model NEVER receives. Drift on a dropped feature is observationally
   uninteresting.

So the actionable layer in the TVT-2026-05-21 protection stack is the K=2
ensemble catastrophic-dropout (target-aware, measures actual MAE
divergence after training). This sensor is COMPLEMENTARY: it stamps
per-feature drift stats into metadata for post-mortem correlation
("the K=2 dropout fired -- here are the drift candidates that MIGHT
explain it") and surfaces an INFO line on extreme drift (>= 10x default
threshold) where the correlation is strong enough to log without false-
positive risk.

Implementation notes
--------------------
- Per-numeric-feature drift = (other_mean - train_mean) / train_std (z-score).
- We only consider numeric columns; categorical drift via PSI is a
  separate (more expensive) pass.
- Default threshold 3.0 sigma for the metadata stamp; INFO-log fires at
  3.0 sigma, WARN-log only at >= 10x sigma (extreme outliers where the
  signal is strong enough to act on independently).
- Report stamps into ``metadata["feature_distribution_drift"]`` so
  observability tooling can plot per-feature drift trends.

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
    feature_importance: Optional[Dict[str, float]] = None,
    max_features_in_log: int = 10,
) -> Dict[str, Any]:
    """Compute per-feature mean drift across train / val / test.

    Returns a dict with:
      - ``per_feature``: ``{col: {"train_mean", "train_std", "val_z",
        "test_z"}}`` for each numeric feature.
      - ``drift_candidates``: list of (col, max_abs_z) where max_abs_z
        exceeds ``warn_threshold_z``, sorted descending.
      - ``weighted_drift_score``: optional importance-weighted aggregate
        = sum(|z_i| * |fi_i|) / sum(|fi_i|). Populated only when
        ``feature_importance`` is supplied. Higher = more risk because
        the IMPORTANT features (high FI) are drifting; a non-FI-weighted
        high-z on irrelevant features is much less worrying.
      - ``threshold``: the z-threshold that fired log lines.
      - ``n_numeric_features``: count of numeric features inspected.

    The function never raises on missing val/test (returns z=NaN for the
    missing slot). Per-feature z is NaN when train_std==0 (constant
    feature -- no drift signal can be extracted).

    Log level depends on the magnitude AND the FI-weighted score:
      - <3 sigma: silent.
      - 3-10 sigma: INFO log (observational; per-model FS may drop these).
      - >=10 sigma OR weighted_drift_score >= 1.0: WARN log (correlation
        with model harm is strong enough to escalate).

    ``feature_importance`` is an optional dict ``{col: fi}`` from any
    reasonable source (sklearn coef_ magnitudes, mlframe FI top-K, MRMR
    scores). Missing columns get 0 weight. The aggregate skips features
    with NaN z (constant features).
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
    # Compute the FI-weighted aggregate when feature_importance is provided.
    # Without FI this is None -- the per-feature z-scores alone aren't strong
    # enough to be a grounded harm signal (drift on a fi=0 feature is
    # harmless; drift on a dominant feature is potentially catastrophic).
    weighted_score: Optional[float] = None
    if feature_importance is not None:
        _num = 0.0
        _den = 0.0
        for col, entry in per_feature.items():
            fi = float(abs(feature_importance.get(col, 0.0)))
            if fi == 0.0:
                continue
            _z = max(
                abs(entry["val_z"]) if math.isfinite(entry["val_z"]) else 0.0,
                abs(entry["test_z"]) if math.isfinite(entry["test_z"]) else 0.0,
            )
            _num += _z * fi
            _den += fi
        if _den > 0:
            weighted_score = float(_num / _den)

    if candidates:
        _shown = candidates[:max_features_in_log]
        _detail = ", ".join(f"{c}(|z|={z:.2f})" for c, z in _shown)
        if len(candidates) > max_features_in_log:
            _detail += f", +{len(candidates) - max_features_in_log} more"
        # 2026-05-22: WARN only when (a) extreme drift (>=10x threshold) OR
        # (b) FI-weighted score >= 1.0 (important features are drifting). The
        # plain "drift > 3 sigma" case is INFO-only because it's not a
        # grounded harm signal -- per-model FS may drop the feature, or it
        # may be FI~0 and the drift is irrelevant.
        _max_abs_z = candidates[0][1] if candidates else 0.0
        _escalate = (
            _max_abs_z >= 10.0 * warn_threshold_z
            or (weighted_score is not None and weighted_score >= 1.0)
        )
        _level = "warning" if _escalate else "info"
        _ws_str = (
            f", weighted_drift={weighted_score:.2f}" if weighted_score is not None else ""
        )
        getattr(logger, _level)(
            "[feature-distribution-drift] %d numeric feature(s) drift > %.1f sigma "
            "between train and val/test (max |z|=%.2f%s). Observational only -- "
            "drift does NOT guarantee model harm; per-model FS may drop these "
            "before training; K=2 ensemble catastrophic-dropout is the "
            "actionable target-aware protection downstream. Top drifters: %s",
            len(candidates), warn_threshold_z, _max_abs_z, _ws_str, _detail,
        )
    return {
        "per_feature": per_feature,
        "drift_candidates": [(c, z) for c, z in candidates],
        "weighted_drift_score": weighted_score,
        "threshold": warn_threshold_z,
        "n_numeric_features": len(per_feature),
    }
