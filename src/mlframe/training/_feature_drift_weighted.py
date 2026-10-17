"""Importance-weighted aggregate of per-feature drift z-scores (helper for ``feature_drift_report``)."""

from __future__ import annotations

import math
from typing import Any, Optional


def fi_weighted_drift_score(per_feature: dict[str, Any], feature_importance: Optional[dict[str, float]]) -> Optional[float]:
    """Return ``sum(|z_i| * |fi_i|) / sum(|fi_i|)`` over features with non-zero importance, or ``None`` without importance / weight."""
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
    return weighted_score
