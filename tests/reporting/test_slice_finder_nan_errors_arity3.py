"""Slice finding with NaN errors and 3-way combinations keeps the feature rows aligned with the errors (audit 2026-11-20, finding 21-F2)."""
import numpy as np
import pandas as pd

from mlframe.reporting.charts.slice_finder import find_weak_slices


def test_nan_errors_with_arity_three_run_on_aligned_rows():
    """A missing prediction removes its row from the errors; the 3-way search must read the same rows (it used to read the unfiltered matrix)."""
    rng = np.random.default_rng(0)
    n = 600
    X = pd.DataFrame(rng.standard_normal((n, 5)), columns=list("abcde"))
    y_true = rng.standard_normal(n)
    y_pred = y_true + 0.1 * rng.standard_normal(n)
    y_pred[::7] = np.nan
    result = find_weak_slices(X, y_true, y_pred, max_arity=3)
    assert result is not None
