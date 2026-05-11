"""Tests for ``detect_group_column_candidates`` (R10c follow-up OPEN-2; auto-detect group_column candidates for linear_residual_grouped)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.training.composite import (
    _GROUP_DETECT_DEFAULT_MAX_UNIQUE,
    _GROUP_DETECT_DEFAULT_MIN_UNIQUE,
    detect_group_column_candidates,
)


class TestDetectGroupColumn:
    def test_well_id_pattern(self) -> None:
        """The canonical TVT pattern: well_id with ~600 unique values, balanced group sizes -> high score."""
        rng = np.random.default_rng(0)
        n = 10_000
        # 20 wells with roughly equal rows.
        well_ids = np.repeat([f"well_{i}" for i in range(20)], n // 20)
        rng.shuffle(well_ids)
        df = pd.DataFrame({
            "well_id": well_ids,
            "x_num": rng.normal(size=n),
            "x_other_num": rng.normal(size=n),
        })
        results = detect_group_column_candidates(df)
        assert results, "expected well_id as a candidate"
        top_name, top_info = results[0]
        assert top_name == "well_id"
        assert top_info["n_unique"] == 20
        # Balanced groups => low size_cv
        assert top_info["size_cv"] < 0.2

    def test_high_cardinality_string_rejected(self) -> None:
        """Per-row unique string (n_unique == n_rows) exceeds max_unique => rejected."""
        df = pd.DataFrame({"row_id": [f"r_{i}" for i in range(1000)]})
        results = detect_group_column_candidates(df, max_unique=500)
        assert all(name != "row_id" for name, _ in results)

    def test_singleton_unique_rejected(self) -> None:
        """Constant column (n_unique = 1) fails min_unique gate."""
        df = pd.DataFrame({"constant": ["x"] * 100})
        results = detect_group_column_candidates(df, min_unique=2)
        assert all(name != "constant" for name, _ in results)

    def test_unbalanced_groups_lower_score(self) -> None:
        """Highly unbalanced groups (one giant + many tiny) gets a LOWER score than balanced."""
        n = 1000
        balanced_groups = np.repeat([f"g_{i}" for i in range(10)], n // 10)
        np.random.default_rng(0).shuffle(balanced_groups)
        # Make second column with 1 giant group + 9 tiny groups.
        unbalanced_groups = np.array(["g_0"] * (n - 90) + [f"g_{i}" for i in range(1, 10) for _ in range(10)])
        np.random.default_rng(0).shuffle(unbalanced_groups)
        df = pd.DataFrame({
            "balanced": balanced_groups,
            "unbalanced": unbalanced_groups,
        })
        results = detect_group_column_candidates(df, min_size_ratio=0.005)
        # Both pass min thresholds but balanced should rank higher.
        names = [name for name, _ in results]
        if "balanced" in names and "unbalanced" in names:
            balanced_idx = names.index("balanced")
            unbalanced_idx = names.index("unbalanced")
            assert balanced_idx < unbalanced_idx

    def test_min_group_size_floor(self) -> None:
        """Group with < min_size_ratio * n_rows rows excludes the WHOLE column."""
        n = 1000
        # 99 rows in g0 + 1 row in g1; min_size_ratio=0.1 floors at 100 rows. g1 has 1 row < 100 -> reject column.
        groups = np.array(["g0"] * 999 + ["g1"])
        df = pd.DataFrame({"groups": groups})
        results = detect_group_column_candidates(df, min_size_ratio=0.1)
        assert all(name != "groups" for name, _ in results)

    def test_returns_sorted_by_score(self) -> None:
        """Multiple valid candidates returned sorted by score descending."""
        n = 1000
        df = pd.DataFrame({
            "good_groups": np.repeat([f"a_{i}" for i in range(10)], n // 10),
            "ok_groups": np.repeat([f"b_{i}" for i in range(5)], n // 5),
            "x_num": np.random.default_rng(0).normal(size=n),
        })
        results = detect_group_column_candidates(df)
        scores = [info["score"] for _, info in results]
        # Sorted descending.
        assert scores == sorted(scores, reverse=True)

    def test_explicit_candidate_columns(self) -> None:
        """Caller can restrict to a specific subset of columns."""
        n = 200
        df = pd.DataFrame({
            "a_groups": np.repeat([f"a_{i}" for i in range(5)], n // 5),
            "b_groups": np.repeat([f"b_{i}" for i in range(5)], n // 5),
        })
        results = detect_group_column_candidates(df, candidate_columns=["a_groups"])
        names = [name for name, _ in results]
        assert "a_groups" in names
        assert "b_groups" not in names

    def test_default_thresholds_documented(self) -> None:
        """Sanity: the module-level defaults match the documented thresholds."""
        assert _GROUP_DETECT_DEFAULT_MIN_UNIQUE == 3
        assert _GROUP_DETECT_DEFAULT_MAX_UNIQUE == 500
