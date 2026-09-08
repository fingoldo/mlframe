"""Tests for scoring the pre-registration's own predictions.

The four outcome cells have to be distinguishable, and the two that matter are the ones a lazy
implementation collapses: a prediction that FAILED (the bed expected to defeat an arm and the arm paid) is
a finding about our priors, and an unpredicted failure is either a discovery or a broken bed. Folding
either into "as expected" would leave the scorecard flattering by construction.
"""

from __future__ import annotations

from typing import Any, Dict, List

from mlframe.feature_selection._benchmarks.fs_hybrid._forecast import forecast_rows, forecast_table


def _record(arm: str, seed: int, auc: float, expected: List[str], scenario: str = "bed") -> Dict[str, Any]:
    """One ok cell carrying the bed's prediction alongside its score."""
    return {
        "status": "ok",
        "arm": arm,
        "scenario": scenario,
        "dataset_seed": seed,
        "cv_seed": 0,
        "expected_to_break": list(expected),
        "scores": {"1k": {"models": {"lightgbm": {"roc_auc": auc}}}},
    }


def _cells(spec: Dict[str, float], expected: List[str], seeds: int = 8, scenario: str = "bed") -> List[Dict[str, Any]]:
    """Records for one bed: ``{arm: delta from the null}``, null at 0.70 with a per-seed drift."""
    out: List[Dict[str, Any]] = []
    for seed in range(seeds):
        drift = 0.001 * seed
        out.append(_record("all-features", seed, 0.70 + drift, expected, scenario))
        for arm, delta in spec.items():
            out.append(_record(arm, seed, 0.70 + drift + delta, expected, scenario))
    return out


class TestOutcomes:
    """The four cells."""

    def test_a_bed_that_defeats_the_arm_it_named_is_confirmed(self) -> None:
        """The ordinary success: the prediction was made in advance and it held."""
        (row,) = [r for r in forecast_rows(_cells({"weak": -0.05}, ["weak"])) if r.arm == "weak"]
        assert row.predicted_to_break and row.broke and row.outcome == "confirmed"

    def test_an_arm_that_pays_despite_the_prediction_is_a_failed_prediction(self) -> None:
        """The interesting cell: our prior about the method was wrong, and the report must say so."""
        (row,) = [r for r in forecast_rows(_cells({"strong": +0.05}, ["strong"])) if r.arm == "strong"]
        assert row.predicted_to_break and not row.broke and row.outcome == "prediction failed"

    def test_an_unnamed_arm_that_loses_is_an_unpredicted_failure(self) -> None:
        """A bed defeating an arm nobody expected it to is a discovery or a bug; either way it surfaces."""
        (row,) = [r for r in forecast_rows(_cells({"surprise": -0.05}, ["other"])) if r.arm == "surprise"]
        assert not row.predicted_to_break and row.broke and row.outcome == "unpredicted failure"

    def test_an_unnamed_arm_that_pays_is_as_expected(self) -> None:
        """The quiet cell, and the only one that needs no explanation."""
        (row,) = [r for r in forecast_rows(_cells({"fine": +0.05}, ["other"])) if r.arm == "fine"]
        assert row.outcome == "as expected"


class TestBrokeMeansFailedToBeatTheNull:
    """What counts as breaking."""

    def test_a_gain_that_does_not_separate_from_zero_counts_as_broken(self) -> None:
        """An arm that gains without separating has not demonstrated a gain; noise must not refute a prediction."""
        records: List[Dict[str, Any]] = []
        for seed in range(8):
            # A difference that alternates sign: mean near zero, and no paired test can call it positive.
            offset = 0.02 if seed % 2 == 0 else -0.02
            records.append(_record("all-features", seed, 0.70, ["noisy"]))
            records.append(_record("noisy", seed, 0.70 + offset, ["noisy"]))
        (row,) = [r for r in forecast_rows(records) if r.arm == "noisy"]
        assert row.broke

    def test_a_clear_separated_gain_does_not_count_as_broken(self) -> None:
        """The rule must not be so strict that every arm 'breaks' and the scorecard says nothing."""
        (row,) = [r for r in forecast_rows(_cells({"clear": +0.05}, [])) if r.arm == "clear"]
        assert not row.broke


class TestProvenance:
    """Where the predictions came from is part of the claim."""

    def test_predictions_carried_by_the_cells_are_used_and_named(self) -> None:
        """A prediction that travelled with the run cannot have been edited between the run and the scoring."""
        rendered = "\n".join(forecast_table(_cells({"weak": -0.05}, ["weak"])))
        assert "predictions read from: the cells themselves" in rendered

    def test_a_run_without_carried_predictions_says_where_it_looked_instead(self) -> None:
        """'Declared before this run' and 'declared before some run' are different claims."""
        records = _cells({"weak": -0.05}, [])
        for record in records:
            record.pop("expected_to_break")
            record["scenario"] = "xor3"
        rendered = "\n".join(forecast_table(records))
        assert "committed registry" in rendered


class TestTable:
    """What the report shows."""

    def test_the_hit_rate_counts_only_the_predicted_pairs(self) -> None:
        """An unpredicted pair is not a prediction and must not dilute the score in either direction."""
        records = _cells({"weak": -0.05, "strong": +0.05, "unnamed": +0.05}, ["weak", "strong"])
        rendered = "\n".join(forecast_table(records))
        assert "hit rate: 1/2" in rendered

    def test_failed_predictions_are_listed_with_their_effect(self) -> None:
        """A hit rate without the misses named is a grade with no way to check it."""
        rendered = "\n".join(forecast_table(_cells({"strong": +0.05}, ["strong"])))
        assert "PREDICTION FAILED" in rendered and "strong" in rendered

    def test_a_run_where_no_bed_named_an_arm_reports_no_rate_rather_than_nan(self) -> None:
        """A rate over zero predictions is neither 0% nor 100%, and `nan%` invites being read as one."""
        rendered = "\n".join(forecast_table(_cells({"a": 0.01}, [])))
        assert "nothing to score" in rendered
        assert "nan" not in rendered
        # The unpredicted side still reports: a bed that predicted nothing can still be defeated by one.
        assert "unpredicted failures:" in rendered
