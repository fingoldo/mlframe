"""Regression: the preflight redundancy probe must detect a highly-correlated pair under missingness.

Pre-fix, ``dataset_diagnostics`` zero-filled NaN via ``nan_to_num`` before ``corrcoef``; on a
high-missingness column the injected zeros collapse its spread and bias |corr| toward 0, so a pair
that is near-perfectly correlated on its complete rows reads as non-redundant and mis-routes the
redundancy gate. The fix uses pandas pairwise-complete correlation (both-present rows only).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("xgboost")

from mlframe.feature_selection.shap_proxied_fs._shap_proxy_preflight import dataset_diagnostics


def test_highly_correlated_but_missing_pair_detected_as_redundant():
    """dataset_diagnostics uses pairwise-complete correlation so a 70%-missing-but-near-perfectly-correlated pair is still flagged redundant, not diluted by zero-fill."""
    rng = np.random.default_rng(0)
    n = 1000
    base = rng.normal(size=n)
    a = base + 0.01 * rng.normal(size=n)
    b = base + 0.01 * rng.normal(size=n)  # ~perfectly correlated with a on complete rows

    # Knock out 70% of b at random; complete-pair correlation stays ~1.0.
    mask = rng.random(n) < 0.7
    b_missing = b.copy()
    b_missing[mask] = np.nan

    y = (base > 0).astype(int)
    X = pd.DataFrame({"a": a, "b": b_missing})

    d = dataset_diagnostics(X, y, classification=True)
    assert d["max_abs_corr"] >= 0.9, f"highly-correlated-but-missing pair not detected as redundant: max_abs_corr={d['max_abs_corr']:.3f}"
