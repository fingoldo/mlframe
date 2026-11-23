"""Audit 2026-11-20 finding 35-F40: the polars group-candidate path uses the same helper as the default-candidate path.

A temporal column is never a categorical candidate (target-encoding it leaks time), so the group path must not offer one either.
"""

from datetime import date

import polars as pl

from mlframe.training.composite.discovery import auto_detect


def test_polars_group_path_calls_the_shared_helper(monkeypatch):
    """With no explicit candidate list, detect_group_column_candidates asks the shared helper for the candidates."""
    calls = []
    real = auto_detect._polars_default_cat_candidates

    def spy(df, max_unique):
        """Record the call and delegate."""
        calls.append(max_unique)
        return real(df, max_unique)

    monkeypatch.setattr(auto_detect, "_polars_default_cat_candidates", spy)
    df = pl.DataFrame({"g": [i % 5 for i in range(200)], "d": [date(2020, 1, 1 + (i % 28)) for i in range(200)]})
    auto_detect.detect_group_column_candidates(df)
    assert calls, "the group path must use the shared candidate helper"


def test_temporal_column_is_not_a_candidate():
    """The shared helper leaves a Date column out of the candidates."""
    df = pl.DataFrame({"g": [i % 5 for i in range(200)], "d": [date(2020, 1, 1 + (i % 28)) for i in range(200)]})
    cands = auto_detect._polars_default_cat_candidates(df, 500)
    assert "d" not in cands
    assert "g" in cands
