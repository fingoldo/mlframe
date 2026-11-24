"""Cluster 41 regression: low-variance detection at zero mean is scale-free (audit 2026-11-20, finding 41-F32).

A non-constant feature with mean zero and a tiny spread used to be flagged low-variance, because the absolute std was used as the relative
measure. Only an exactly constant column is low-variance at a zero mean now; a real low-relative-std column with a non-zero mean is still flagged.
"""

import numpy as np
import pandas as pd

from mlframe.training.targets._target_distribution_analyzer_features import analyze_feature_distribution


def test_zero_mean_tiny_spread_is_not_low_variance():
    """A non-constant zero-mean feature at std 2e-4 is not reported low-variance (41-F32)."""
    X = pd.DataFrame({"a": np.array([2e-4, -2e-4] * 20), "b": np.arange(40.0)})
    report = analyze_feature_distribution(X)
    assert not any(p.startswith("low_variance") for p in report.pathologies)


def test_constant_column_is_still_low_variance():
    """Control: an exactly constant column is still flagged low-variance (41-F32)."""
    X = pd.DataFrame({"a": np.zeros(40), "b": np.arange(40.0)})
    report = analyze_feature_distribution(X)
    assert any(p.startswith("low_variance") for p in report.pathologies)
