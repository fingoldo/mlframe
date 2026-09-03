"""One categorical column took the whole drift heatmap down.

A production run logged ``psi_heatmap failed; continuing`` with ``ValueError: could not convert string to float:
'FIXED'`` -- ``compute_psi_matrix`` casts every column to float64 because its PSI is quantile-binned, which only
means anything for an ordered numeric column. The frame it is handed contains the model's categorical features
too, so one unusable column cost every usable one its chart. Categorical drift has its own report; here such a
column is skipped and named.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from mlframe.reporting.charts.drift import compute_psi_matrix, psi_heatmap

N = 400


@pytest.fixture
def mixed_frame():
    """Two numeric columns that drift, plus the string column that used to crash the chart."""
    rng = np.random.default_rng(3)
    drifting = np.concatenate([rng.normal(0, 1, N // 2), rng.normal(4, 1, N // 2)])
    return pd.DataFrame({
        "stable_num": rng.normal(0, 1, N),
        "job_type": np.where(np.arange(N) % 2 == 0, "FIXED", "HOURLY"),
        "drifting_num": drifting,
    })


@pytest.fixture
def timestamps():
    """Monotonic timestamps so the time bucketing is well defined."""
    return np.arange(N, dtype=np.int64)


class TestItSurvivesANonNumericColumn:
    """The defect: an exception instead of a chart."""

    def test_the_matrix_is_computed(self, mixed_frame, timestamps):
        """Both numeric columns keep their row; only the string column drops out."""
        matrix, _rows, _ = compute_psi_matrix(mixed_frame, timestamps, n_time_buckets=4)
        assert matrix.shape[0] == 2

    def test_the_heatmap_renders(self, mixed_frame, timestamps):
        """The production symptom was the whole figure being dropped."""
        assert psi_heatmap(mixed_frame, timestamps, n_time_buckets=4) is not None

    def test_row_labels_stay_aligned_with_their_rows(self, mixed_frame, timestamps):
        """A skipped column must not shift every later name onto the wrong row -- worse than crashing."""
        matrix, rows, _ = compute_psi_matrix(mixed_frame, timestamps, n_time_buckets=4)
        assert list(rows) == ["stable_num", "drifting_num"]
        drift_row = matrix[list(rows).index("drifting_num")]
        stable_row = matrix[list(rows).index("stable_num")]
        assert np.nanmax(drift_row) > np.nanmax(stable_row), "labels do not match the data they name"

    def test_the_skip_is_reported(self, mixed_frame, timestamps, caplog):
        """A silently missing feature row reads as "this feature is fine"."""
        with caplog.at_level(logging.INFO, logger="mlframe.reporting.charts.drift"):
            compute_psi_matrix(mixed_frame, timestamps, n_time_buckets=4)
        text = " ".join(r.getMessage() for r in caplog.records)
        assert "job_type" in text

    def test_an_all_categorical_frame_returns_an_empty_matrix(self, timestamps):
        """The caller already renders an explanatory panel for an empty matrix; it must not get an exception."""
        frame = pd.DataFrame({"a": ["x"] * N, "b": ["y"] * N})
        matrix, rows, _ = compute_psi_matrix(frame, timestamps, n_time_buckets=4)
        assert matrix.shape[0] == 0
        assert rows == ()

    def test_a_fully_numeric_frame_is_unchanged(self, timestamps):
        """The skip must not alter the path that already worked."""
        rng = np.random.default_rng(5)
        frame = pd.DataFrame({"a": rng.normal(size=N), "b": rng.normal(size=N)})
        matrix, rows, _ = compute_psi_matrix(frame, timestamps, n_time_buckets=4)
        assert matrix.shape[0] == 2
        assert list(rows) == ["a", "b"]
