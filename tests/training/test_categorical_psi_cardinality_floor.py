"""A level-wise PSI was reported for a column with 981_873 levels, and it read as catastrophic drift.

The production log: ``skills_text(PSI=42.666)`` on the same 0.10 / 0.25 credit-risk scale as a five-level column.
With 2.18M train rows over 981_873 levels that is 2.2 rows per level -- nearly every level a singleton, so the
number measures sampling noise, not drift, and inviting an operator to act on it is worse than saying nothing.
The floor reuses ``bin_min_count``: a level-wise frequency needs at least that many rows per level on average.
"""

from __future__ import annotations

import logging

import pandas as pd
import pytest

from mlframe.training.feature_drift_report import compute_categorical_drift_psi


def _frames(n_levels: int, n_rows: int = 2000):
    """train/val frames whose single column has ``n_levels`` roughly equally-sized levels."""
    train = pd.DataFrame({"col": [f"v{i % n_levels}" for i in range(n_rows)]})
    val = pd.DataFrame({"col": [f"v{(i * 7) % n_levels}" for i in range(n_rows // 4)]})
    return train, val


class TestTheCardinalityFloor:
    """Which columns get a level-wise PSI at all."""

    def test_a_near_singleton_column_is_skipped(self):
        """2000 rows over 1800 levels is 1.1 rows/level -- there is no frequency to compare."""
        train, val = _frames(n_levels=1800)
        out = compute_categorical_drift_psi(train, val, None, feature_names=["col"])
        assert [c for c, _ in out["skipped_high_cardinality"]] == ["col"]
        assert "col" not in out["per_feature"]

    def test_a_skipped_column_is_not_a_drift_candidate(self):
        """The production symptom: a meaningless number at the top of the drifters list."""
        train, val = _frames(n_levels=1800)
        out = compute_categorical_drift_psi(train, val, None, feature_names=["col"])
        assert [c for c, _ in out["drift_candidates"]] == []

    def test_an_ordinary_categorical_still_gets_its_psi(self):
        """2000 rows over 20 levels is 100 rows/level; this is exactly what the metric is for."""
        train, val = _frames(n_levels=20)
        out = compute_categorical_drift_psi(train, val, None, feature_names=["col"])
        assert "col" in out["per_feature"]
        assert out["skipped_high_cardinality"] == []

    @pytest.mark.parametrize("n_levels, kept", [(400, True), (500, False)])
    def test_the_floor_sits_where_bin_min_count_puts_it(self, n_levels, kept):
        """2000 rows: 400 levels is 5.0 rows/level (kept), 500 is 4.0 (skipped) at the default bin_min_count=5."""
        train, val = _frames(n_levels=n_levels)
        out = compute_categorical_drift_psi(train, val, None, feature_names=["col"])
        assert ("col" in out["per_feature"]) is kept

    def test_a_caller_can_lower_the_floor(self):
        """``bin_min_count`` is the knob; the floor must follow it rather than being hardcoded."""
        train, val = _frames(n_levels=1000)
        out = compute_categorical_drift_psi(train, val, None, feature_names=["col"], bin_min_count=1)
        assert "col" in out["per_feature"]


class TestTheSkipIsVisible:
    """A dropped column must not read as a clean one."""

    def test_the_skip_is_logged_with_the_level_count(self, caplog):
        """The reader needs the level count to see WHY the metric could not be computed."""
        with caplog.at_level(logging.INFO, logger="mlframe.training.feature_drift_report"):
            train, val = _frames(n_levels=1800)
            compute_categorical_drift_psi(train, val, None, feature_names=["col"])
        text = " ".join(r.getMessage() for r in caplog.records)
        assert "1_800 levels" in text
        assert "rows/level" in text
