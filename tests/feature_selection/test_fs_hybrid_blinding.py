"""Tests for blind aggregation.

The mechanism is only worth having if it actually hides what it claims to: a mapping that is alphabetical,
or stable across runs, or leaks the author's arm through an unblinded field, provides the appearance of a
control and none of the substance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pytest

from mlframe.feature_selection._benchmarks.fs_hybrid._blinding import (
    NEVER_BLINDED,
    apply_blinding,
    blind_labels,
    load_mapping,
    unblind_text,
    write_mapping,
)

ARMS = ["all-features", "mrmr", "rfecv", "boruta", "ace", "univariate-mi", "variance-sort"]


def _records(arms: List[str]) -> List[Dict[str, Any]]:
    """One minimal record per arm."""
    return [{"arm": arm, "scenario": "bed", "dataset_seed": 0, "status": "ok"} for arm in arms]


class TestMapping:
    """What the mapping must and must not do."""

    def test_covers_every_arm_exactly_once(self) -> None:
        """A partial mapping would leave some rows readable and defeat the whole exercise."""
        mapping = blind_labels(ARMS)
        assert set(mapping) == set(ARMS)
        assert len(set(mapping.values())) == len(ARMS)

    def test_the_null_hypothesis_stays_visible(self) -> None:
        """Every contrast is stated against it and it is nobody's method, so hiding it only hurts reading."""
        mapping = blind_labels(ARMS)
        for name in NEVER_BLINDED:
            assert mapping[name] == name

    def test_labels_are_not_alphabetical(self) -> None:
        """Alphabetical labels are no blinding at all: the first label would be the same arm every time."""
        mapping = blind_labels(ARMS)
        blinded = sorted(name for name in ARMS if name not in NEVER_BLINDED)
        assert [mapping[name] for name in blinded] != [f"arm_{chr(ord('a') + i)}" for i in range(len(blinded))]

    def test_the_salt_reshuffles_the_labels(self) -> None:
        """Without this, a reader carries yesterday's mapping into today's table from memory."""
        assert blind_labels(ARMS, salt="run-1") != blind_labels(ARMS, salt="run-2")

    def test_the_same_salt_reproduces_the_mapping(self) -> None:
        """A blinded table has to be regenerable exactly, or it cannot be checked."""
        assert blind_labels(ARMS, salt="run-1") == blind_labels(ARMS, salt="run-1")

    def test_more_arms_than_letters_still_get_distinct_labels(self) -> None:
        """A wide roster must not collide two arms onto one label."""
        many = [f"arm-{i:03d}" for i in range(40)]
        mapping = blind_labels(many)
        assert len(set(mapping.values())) == 40


class TestApplying:
    """Applying the mapping to records."""

    def test_records_are_relabelled_and_the_originals_are_untouched(self) -> None:
        """The caller keeps the real records, so reading them is a visible act rather than a silent one."""
        records = _records(ARMS)
        mapping = blind_labels(ARMS)
        blinded = apply_blinding(records, mapping)
        assert [record["arm"] for record in records] == ARMS
        assert {record["arm"] for record in blinded} == set(mapping.values())

    def test_an_unmapped_arm_is_refused(self) -> None:
        """A mapping from a different run would blind the table inconsistently, which is worse than not at all."""
        mapping = blind_labels(["mrmr", "all-features"])
        with pytest.raises(KeyError, match="not in the blinding mapping"):
            apply_blinding(_records(["mrmr", "rfecv"]), mapping)

    def test_non_arm_fields_survive(self) -> None:
        """Only the arm name is hidden; the data the analysis needs must pass through unchanged."""
        blinded = apply_blinding(_records(["mrmr", "all-features"]), blind_labels(["mrmr", "all-features"]))
        assert all(record["scenario"] == "bed" and record["status"] == "ok" for record in blinded)


class TestRoundTrip:
    """Storing the key apart from the data, and putting it back at the end."""

    def test_mapping_is_written_beside_the_results_and_read_back(self, tmp_path: Path) -> None:
        """Its own file, so the aggregation code can be handed the records without the key."""
        results = str(tmp_path / "cells.jsonl")
        mapping = blind_labels(ARMS, salt="x")
        write_mapping(results, mapping)
        assert load_mapping(results) == mapping

    def test_a_run_without_blinding_reads_as_none(self, tmp_path: Path) -> None:
        """Absence is not an error; most runs are not blinded."""
        assert load_mapping(str(tmp_path / "nothing.jsonl")) is None

    def test_unblinding_a_report_restores_every_name(self) -> None:
        """The reveal happens once, on a committed report, after the numbers are fixed."""
        mapping = blind_labels(ARMS, salt="x")
        report = " ".join(f"| {mapping[name]} | 0.5 |" for name in ARMS)
        revealed = unblind_text(report, mapping)
        for name in ARMS:
            assert name in revealed

    def test_long_labels_are_not_eaten_by_short_ones(self) -> None:
        """`arm_a` must not consume the prefix of `arm_aa`, which would rename the wrong row."""
        many = [f"arm-{i:03d}" for i in range(30)]
        mapping = blind_labels(many, salt="x")
        two_letter = [name for name, label in mapping.items() if len(label) > len("arm_a")]
        assert two_letter, "this fixture must produce at least one two-letter label"
        text = " ".join(sorted(mapping.values()))
        revealed = unblind_text(text, mapping)
        assert all(name in revealed for name in many)
        assert "arm_" not in revealed
