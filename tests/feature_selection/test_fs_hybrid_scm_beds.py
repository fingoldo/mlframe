"""Tests for the SCM beds' adaptation to the benchmark harness.

The translation has one job: hand the runner a truth dictionary that means what the runner thinks it means.
The matched-K grid is derived from `truth["base"]`, support recovery is scored against `truth["relevant"]`,
and the coverage meta-test reads `expected_to_break` -- so a translation that quietly disagreed with the
scenario's own answer key would misreport every arm on every one of these beds while looking healthy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.data.datasets import scenarios
from mlframe.feature_selection._benchmarks.fs_hybrid._scm_beds import build_scm_bed, scm_bed_scenarios

ROWS = 1200


class TestBedShape:
    """What the runner needs to be able to consume a bed at all."""

    def test_returns_a_frame_labels_and_truth(self) -> None:
        """The harness's bed contract: a pandas frame, an integer label array, a truth dictionary."""
        frame, labels, truth = build_scm_bed("linear_k5_p50", seed=0, n_samples=ROWS)
        assert isinstance(frame, pd.DataFrame) and len(frame) == ROWS
        assert isinstance(labels, np.ndarray) and labels.shape == (ROWS,)
        assert set(np.unique(labels)) <= {0, 1}
        assert isinstance(truth, dict)

    def test_truth_partitions_the_columns(self) -> None:
        """Every column is either in the answer key or in the noise list, and none is in both."""
        frame, _labels, truth = build_scm_bed("linear_k5_p50", seed=0, n_samples=ROWS)
        base, noise = set(truth["base"]), set(truth["noise"])
        assert base & noise == set()
        assert base | noise == {str(column) for column in frame.columns}

    def test_null_beds_are_excluded_by_default(self) -> None:
        """With no relevant column, 'did any arm beat all-features' has no meaning on a null bed."""
        default = {name for name, _ in scm_bed_scenarios()}
        with_null = {name for name, _ in scm_bed_scenarios(include_null=True)}
        assert not any(name.startswith("null_") for name in default)
        assert any(name.startswith("null_") for name in with_null)


class TestTruthMatchesTheScenario:
    """The translation must not invent, drop or reorder the scenario's own answer key."""

    @pytest.mark.parametrize("name", ["linear_k5_p50", "xor3", "mb_spouse_collider", "latent_replicates_private_delta"])
    def test_the_answer_key_is_the_markov_blanket(self, name: str) -> None:
        """`base` drives the matched-K grid, so it must be the pre-registered primary target set."""
        from mlframe.data.datasets.generator import generate

        spec = scenarios.get(name).build(seed=0).model_copy(update={"n_samples": ROWS})
        expected = list(generate(spec).truth.primary_target_set().members)
        _frame, _labels, truth = build_scm_bed(name, seed=0, n_samples=ROWS)
        assert truth["base"] == expected
        assert truth["relevant"] == expected

    @pytest.mark.parametrize("name", ["xor3", "mediator_chain_with_proxy"])
    def test_expectations_travel_with_the_bed(self, name: str) -> None:
        """The coverage meta-test reads these; a bed that lost them would silently stop being falsifiable."""
        _frame, _labels, truth = build_scm_bed(name, seed=0, n_samples=ROWS)
        assert truth["expected_to_break"] == scenarios.get(name).expected_to_break

    def test_the_ceiling_travels_with_the_bed(self) -> None:
        """A downstream AUC means nothing without the AUC that was achievable on the same data."""
        _frame, _labels, truth = build_scm_bed("linear_k5_p50", seed=0, n_samples=ROWS)
        assert truth["bayes_auc"] == pytest.approx(0.80, abs=0.03)
        assert "calibrated Bayes AUC" in truth["notes"]

    def test_the_spec_hash_is_recorded(self) -> None:
        """The bed a cell ran on must be identifiable afterwards, not merely named."""
        _frame, _labels, truth = build_scm_bed("xor3", seed=0, n_samples=ROWS)
        assert truth["spec_hash"] == scenarios.get("xor3").build(seed=0).model_copy(update={"n_samples": ROWS}).content_hash()


class TestSeeds:
    """Seeds are data, not structure."""

    def test_two_seeds_give_different_data_and_the_same_answer_key(self) -> None:
        """Paired statistics need independent redraws of one bed, not two different beds."""
        first_frame, _f, first_truth = build_scm_bed("linear_k5_p50", seed=0, n_samples=ROWS)
        second_frame, _s, second_truth = build_scm_bed("linear_k5_p50", seed=1, n_samples=ROWS)
        assert first_truth["base"] == second_truth["base"]
        assert not np.array_equal(first_frame.to_numpy(), second_frame.to_numpy())

    def test_the_same_seed_reproduces_the_bed(self) -> None:
        """A resumed run must land on the same data it started with."""
        first, labels_a, _t = build_scm_bed("xor3", seed=3, n_samples=ROWS)
        second, labels_b, _u = build_scm_bed("xor3", seed=3, n_samples=ROWS)
        np.testing.assert_array_equal(first.to_numpy(), second.to_numpy())
        np.testing.assert_array_equal(labels_a, labels_b)


class TestRunsThroughTheHarness:
    """One real cell, because a bed that only satisfies the contract on paper is not wired in."""

    def test_a_cell_completes_and_scores_at_the_matched_grid(self) -> None:
        """The null arm on an SCM bed produces an ok record whose K labels come from the blanket size."""
        from sklearn.model_selection import train_test_split

        from mlframe.feature_selection._benchmarks.fs_hybrid.run_experiment import CellSpec, build_arm_roster, run_cell

        frame, labels, truth = build_scm_bed("linear_k5_p50", seed=0, n_samples=ROWS)
        x_train, x_test, y_train, y_test = train_test_split(frame, labels, test_size=0.4, random_state=0, stratify=labels)
        roster = build_arm_roster(int(frame.shape[1]), random_state=0)
        spec = CellSpec(scenario="linear_k5_p50", arm="all-features", dataset_seed=0, cv_seed=0, protocol_version="test", config={})

        record = run_cell(spec, roster["all-features"], x_train, np.asarray(y_train), x_test, np.asarray(y_test), truth)
        assert record["status"] == "ok", record.get("error")
        assert record["target_size"] == len(truth["base"])
        assert {"1k", "2k", "5k"} <= set(record["scores"])
        assert record["truth_relevant"] == sorted(truth["base"])
