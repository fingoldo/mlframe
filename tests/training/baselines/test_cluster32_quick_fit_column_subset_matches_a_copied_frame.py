"""Cluster 32 regression: fitting the quick model on a column subset of the full frame equals fitting on a copied sub-frame (audit 2026-11-20, finding 32-F14)."""

import numpy as np
import pandas as pd
import pytest

from mlframe.training.baselines.diagnostics import BaselineDiagnostics
from mlframe.training.configs import BaselineDiagnosticsConfig


@pytest.mark.parametrize("n", [40, 400])
def test_column_subset_mode_gives_the_same_metric_as_the_copied_frame(n):
    """Same split, same model: ``columns=kept`` on the full frame returns exactly the metric and importances of ``X[kept]`` without ``columns``."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(n, 5)), columns=list("abcde"))
    y = (X["a"] + 0.3 * rng.normal(size=n) > 0).astype(int).to_numpy()
    kept = ["a", "c", "d", "e"]
    diag = BaselineDiagnostics(BaselineDiagnosticsConfig())
    copied = diag._fit_quick_and_score(X[kept], y, kept, [], "binary_classification", "AUC", inner_n_jobs=1)
    subset = diag._fit_quick_and_score(X, y, kept, [], "binary_classification", "AUC", inner_n_jobs=1, columns=kept)
    assert copied[0] == subset[0]
    np.testing.assert_array_equal(copied[1], subset[1])
