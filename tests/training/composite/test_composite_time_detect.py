"""Tests for ``detect_time_column_candidates`` + ``sort_df_by_time_column`` (OPEN-3 from R10c follow-up; auto-detect time column for EWMA / rolling / frac_diff transforms)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.training.composite import (
    detect_time_column_candidates,
    sort_df_by_time_column,
)


class TestDetectTimeColumn:
    """Groups tests covering detect time column."""
    def test_datetime_column_top(self) -> None:
        """Datetime-dtype column wins over any numeric monotonic column."""
        df = pd.DataFrame(
            {
                "ts": pd.date_range("2024-01-01", periods=100, freq="h"),
                "row_id": np.arange(100),
                "x_num": np.random.default_rng(0).normal(size=100),
            }
        )
        results = detect_time_column_candidates(df)
        assert results, "expected at least one candidate"
        top_name, top_info = results[0]
        assert top_name == "ts"
        assert top_info["is_datetime"] is True
        assert top_info["score"] >= 100

    def test_monotonic_int_column(self) -> None:
        """Monotonic int column qualifies even without datetime dtype."""
        df = pd.DataFrame({"row_id": np.arange(50), "x_num": np.random.default_rng(0).normal(size=50)})
        results = detect_time_column_candidates(df)
        names = {name for name, _ in results}
        assert "row_id" in names
        # Confirm monotonic detection.
        info = dict(results)["row_id"]
        assert info["is_monotonic"] is True
        assert info["monotonic_direction"] == "asc"

    def test_monotonic_descending(self) -> None:
        """Strictly decreasing numeric column also qualifies as time-like."""
        df = pd.DataFrame({"countdown": np.arange(100, 0, -1)})
        results = detect_time_column_candidates(df)
        names = {name for name, _ in results}
        assert "countdown" in names
        assert dict(results)["countdown"]["monotonic_direction"] == "desc"

    def test_non_monotonic_numeric_rejected(self) -> None:
        """Random gaussian column is not monotonic -> not a time candidate."""
        df = pd.DataFrame({"random": np.random.default_rng(0).normal(size=200)})
        results = detect_time_column_candidates(df)
        assert all(name != "random" for name, _ in results)

    def test_empty_df_returns_empty(self) -> None:
        """Empty df returns empty."""
        df = pd.DataFrame()
        results = detect_time_column_candidates(df)
        assert results == []

    def test_explicit_candidate_columns(self) -> None:
        """Explicit candidate columns."""
        df = pd.DataFrame(
            {
                "a": pd.date_range("2024-01-01", periods=10, freq="h"),
                "b": np.arange(10),
            }
        )
        results = detect_time_column_candidates(df, candidate_columns=["b"])
        names = [name for name, _ in results]
        assert "b" in names
        assert "a" not in names

    def test_score_ordering(self) -> None:
        """Datetime (score=100) ranks above monotonic numeric (score=50)."""
        df = pd.DataFrame(
            {
                "row_id": np.arange(50),
                "ts": pd.date_range("2024-01-01", periods=50, freq="h"),
            }
        )
        results = detect_time_column_candidates(df)
        # ts (datetime, score=100) is first; row_id (monotonic, score=50) is second.
        scores = [info["score"] for _, info in results]
        assert scores == sorted(scores, reverse=True)
        assert results[0][0] == "ts"


class TestSortDfByTime:
    """Groups tests covering sort df by time."""
    def test_pandas_sort_ascending(self) -> None:
        """Pandas sort ascending."""
        df = pd.DataFrame(
            {
                "ts": [3, 1, 4, 1, 5, 9, 2],
                "x": ["a", "b", "c", "d", "e", "f", "g"],
            }
        )
        sorted_df = sort_df_by_time_column(df, "ts")
        assert list(sorted_df["ts"]) == [1, 1, 2, 3, 4, 5, 9]
        # Original NOT mutated.
        assert list(df["ts"]) == [3, 1, 4, 1, 5, 9, 2]

    def test_pandas_sort_descending(self) -> None:
        """Pandas sort descending."""
        df = pd.DataFrame({"ts": [1, 5, 2, 4, 3]})
        sorted_df = sort_df_by_time_column(df, "ts", ascending=False)
        assert list(sorted_df["ts"]) == [5, 4, 3, 2, 1]

    def test_index_reset_after_sort(self) -> None:
        """Sorting reorders rows; the returned df must have a fresh 0..N-1 index so downstream callers can assume sequential access."""
        df = pd.DataFrame({"ts": [3, 1, 2]}, index=["x", "y", "z"])
        sorted_df = sort_df_by_time_column(df, "ts")
        assert list(sorted_df.index) == [0, 1, 2]
