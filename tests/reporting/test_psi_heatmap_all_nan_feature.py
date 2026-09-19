"""A feature with no computable PSI must not make the drift heatmap print "All-NaN slice encountered".

Suite logs showed ``drift.py: RuntimeWarning: All-NaN slice encountered`` from the heatmap's one-line verdict: a
column that is entirely missing gets a PSI row of NaN, and ``np.nanmax`` over that row warns even though the verdict
already skips such rows. The row still counts as "no computable PSI"; the other features keep their verdict.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from mlframe.reporting.charts.drift import compute_psi_matrix, psi_heatmap

N = 400


def _frame():
    rng = np.random.default_rng(7)
    return pd.DataFrame({
        "empty": np.full(N, np.nan),
        "drifting": np.concatenate([rng.normal(0, 1, N // 2), rng.normal(4, 1, N // 2)]),
    })


def test_all_nan_feature_renders_without_all_nan_warning():
    ts = np.arange(N, dtype=np.int64)
    matrix, rows, _ = compute_psi_matrix(_frame(), ts, n_time_buckets=4)
    assert "empty" in list(rows) and not np.isfinite(matrix[list(rows).index("empty")]).any()
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="All-NaN slice encountered")
        fig = psi_heatmap(_frame(), ts, n_time_buckets=4)
    assert fig is not None
