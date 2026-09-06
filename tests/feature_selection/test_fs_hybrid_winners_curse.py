"""Tests for the winner's-curse column.

The one that matters most is the refusal: this repository's RFECV, driven with `scoring=None`, optimises a
probabilistic error internally while the report reads AUC. Subtracting one from the other yields a number
with the shape of optimism and the content of a unit mismatch, so the table must decline to take that
difference and say why.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from mlframe.feature_selection._benchmarks.fs_hybrid._winners_curse import optimism_rows, optimism_table


def _record(arm: str, seed: int, claimed: Optional[float], holdout: float, metric: Optional[str] = "roc_auc") -> Dict[str, Any]:
    """One ok cell with an internal optimum and a self-K holdout AUC."""
    record: Dict[str, Any] = {
        "status": "ok",
        "arm": arm,
        "scenario": "bed",
        "dataset_seed": seed,
        "cv_seed": 0,
        "selection_score": claimed,
        "selection_metric": metric,
        "scores": {"self": {"models": {"lightgbm": {"roc_auc": holdout}}}},
    }
    return record


class TestOptimism:
    """When the two sides measure the same thing."""

    def test_an_optimistic_arm_shows_a_positive_gap(self) -> None:
        """Claiming 0.90 and holding 0.80 is the effect this column exists to surface."""
        records = [_record("wrapper", seed, 0.90, 0.80) for seed in range(4)]
        (row,) = optimism_rows(records)
        assert row.comparable and row.optimism == pytest.approx(0.10)

    def test_a_pessimistic_arm_shows_a_negative_gap(self) -> None:
        """The column is a signed difference, not an absolute one: under-claiming is a different fact."""
        (row,) = optimism_rows([_record("wrapper", seed, 0.70, 0.78) for seed in range(3)])
        assert row.optimism == pytest.approx(-0.08)

    def test_the_gap_is_averaged_over_seeds_not_pooled(self) -> None:
        """One seed's wide gap must not be carried by another's; each seed contributes its own difference."""
        records = [_record("wrapper", 0, 0.90, 0.80), _record("wrapper", 1, 0.85, 0.85)]
        (row,) = optimism_rows(records)
        assert row.optimism == pytest.approx(0.05)
        assert row.n_seeds == 2


class TestMetricMismatch:
    """The refusal, which is the load-bearing behaviour."""

    def test_a_different_internal_metric_is_not_differenced(self) -> None:
        """An error and an AUC differ by a unit; a subtraction here would be nonsense with a plausible sign."""
        records = [_record("rfecv", seed, -0.42, 0.78, metric="compute_probabilistic_multiclass_error") for seed in range(3)]
        (row,) = optimism_rows(records)
        assert not row.comparable
        assert row.optimism is None
        assert row.selection_metric == "compute_probabilistic_multiclass_error"

    def test_an_unnamed_internal_metric_is_not_differenced(self) -> None:
        """An arm that does not say what it optimised cannot be checked, and silence is not agreement."""
        (row,) = optimism_rows([_record("mystery", 0, 0.9, 0.8, metric=None)])
        assert not row.comparable and row.optimism is None

    def test_the_table_separates_the_two_groups_and_names_the_metric(self) -> None:
        """A reader must be able to tell a measured gap from a comparison that was declined."""
        records = [_record("wrapper", 0, 0.90, 0.80), _record("rfecv", 0, -0.42, 0.78, metric="compute_probabilistic_multiclass_error")]
        rendered = "\n".join(optimism_table(records))
        assert "optimism=+0.1000" in rendered
        assert "NOT COMPARABLE" in rendered
        assert "compute_probabilistic_multiclass_error" in rendered


class TestAbsence:
    """What the table must not imply about arms it does not list."""

    def test_arms_without_an_internal_optimum_are_absent(self) -> None:
        """A filter never searches, so it claims nothing; absence is not a clean bill of health."""
        records: List[Dict[str, Any]] = [_record("filter", 0, None, 0.80)]
        assert optimism_rows(records) == []

    def test_an_empty_table_says_so_rather_than_printing_nothing(self) -> None:
        """A blank block reads as 'no optimism found', which is a different claim from 'nothing was claimed'."""
        rendered = "\n".join(optimism_table([_record("filter", 0, None, 0.8)]))
        assert "no arm in this run reported an internal optimum" in rendered
        assert "absence is not evidence of honesty" in rendered.lower()

    def test_failed_cells_do_not_contribute(self) -> None:
        """A crashed cell may still carry a stale claim; it is not a measurement."""
        bad = _record("wrapper", 0, 0.9, 0.8)
        bad["status"] = "error"
        assert optimism_rows([bad]) == []
