"""Behavioural tests for the run manifest and the checks a report runs against it.

The manifest exists so a reader does not have to take the run's discipline on trust, which means the checks
have to fire on the specific ways a resumable, author-run benchmark drifts: extra seeds appearing after the
first tables were read, an arm or scenario that was never declared, a pre-registration edited afterwards.
Each of those has a test that fails if the check is removed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from mlframe.feature_selection._benchmarks.fs_hybrid._manifest import (
    MANIFEST_SCHEMA_VERSION,
    build_manifest,
    check_run_against_manifest,
    load_manifest,
    manifest_path_for,
    write_manifest,
)


def _records(seeds: List[int], arm: str = "mrmr", scenario: str = "bed") -> List[Dict[str, Any]]:
    """Build minimal records for the declaration checks, which read only the index fields."""
    return [{"status": "ok", "arm": arm, "scenario": scenario, "dataset_seed": s} for s in seeds]


def _manifest(tmp_path: Path, seeds: List[int]) -> Dict[str, Any]:
    """Build a manifest declaring one bed, one arm and the given seeds."""
    return build_manifest(
        results_path=str(tmp_path / "cells.jsonl"),
        scenarios=["bed"],
        arms=["mrmr", "all-features"],
        dataset_seeds=seeds,
        cv_seeds=[0],
        protocol_version="test-v1",
        mode="dev",
    )


class TestRoundTrip:
    """Writing and reading back."""

    def test_written_beside_its_results_file(self, tmp_path: Path) -> None:
        """The manifest lives next to the results it describes, so the pair cannot be separated by accident."""
        results = str(tmp_path / "cells.jsonl")
        path = write_manifest(results, _manifest(tmp_path, [0, 1]))
        assert path == manifest_path_for(results)
        assert Path(path).exists()

    def test_reads_back_what_was_declared(self, tmp_path: Path) -> None:
        """Every field a check depends on survives the round trip."""
        results = str(tmp_path / "cells.jsonl")
        write_manifest(results, _manifest(tmp_path, [0, 1, 2]))
        loaded = load_manifest(results)
        assert loaded is not None
        assert loaded["schema_version"] == MANIFEST_SCHEMA_VERSION
        assert loaded["n_seeds_declared"] == 3
        assert loaded["dataset_seeds"] == [0, 1, 2]
        assert set(loaded["arms"]) == {"mrmr", "all-features"}

    def test_a_missing_manifest_reads_as_none(self, tmp_path: Path) -> None:
        """An older run without a manifest is absence, not an exception to handle at every call site."""
        assert load_manifest(str(tmp_path / "nothing.jsonl")) is None

    def test_a_corrupt_manifest_reads_as_none(self, tmp_path: Path) -> None:
        """A truncated file is treated as no declaration rather than crashing the report."""
        results = str(tmp_path / "cells.jsonl")
        Path(manifest_path_for(results)).write_text("{ not json", encoding="utf-8")
        assert load_manifest(results) is None


class TestDeclarationChecks:
    """What the report must say when the run and its declaration disagree."""

    def test_a_matching_run_produces_no_notes(self, tmp_path: Path) -> None:
        """A run that stayed inside its declaration is reported as clean, with nothing to explain."""
        assert check_run_against_manifest(_manifest(tmp_path, [0, 1, 2]), _records([0, 1, 2])) == []

    def test_extra_seeds_are_reported_as_optional_stopping(self, tmp_path: Path) -> None:
        """Seeds beyond the declared set are the resumable runner's characteristic drift, so they are named."""
        notes = check_run_against_manifest(_manifest(tmp_path, [0, 1]), _records([0, 1, 2, 3]))
        assert any("OPTIONAL STOPPING" in note for note in notes)
        assert any("[2, 3]" in note for note in notes)

    def test_fewer_seeds_than_declared_is_not_flagged(self, tmp_path: Path) -> None:
        """An unfinished run is incomplete, not undeclared; only running MORE than declared is drift."""
        assert check_run_against_manifest(_manifest(tmp_path, [0, 1, 2]), _records([0, 1])) == []

    def test_an_undeclared_arm_is_named(self, tmp_path: Path) -> None:
        """An arm nobody declared cannot enter a leaderboard silently."""
        notes = check_run_against_manifest(_manifest(tmp_path, [0]), _records([0], arm="late-addition"))
        assert any("UNDECLARED ARMS" in note and "late-addition" in note for note in notes)

    def test_a_matched_cardinality_control_is_not_reported_as_undeclared(self, tmp_path: Path) -> None:
        """`random-<k>` is named after the bed's answer-key size, so its literal name varies per bed.

        Comparing raw names would report every bed but the first as carrying an undeclared arm, and a check
        that cries wolf on every run teaches a reader to skip the line where a real one appears.
        """
        manifest = _manifest(tmp_path, [0])
        manifest["arms"] = ["mrmr", "all-features", "random-5"]
        manifest["arm_families"] = ["all-features", "mrmr", "random-<k>"]
        notes = check_run_against_manifest(manifest, _records([0], arm="random-250"))
        assert not any("UNDECLARED ARMS" in note for note in notes), notes

    def test_a_genuinely_new_arm_is_still_reported(self, tmp_path: Path) -> None:
        """The family rule must not become a blanket amnesty: an arm from no declared family still shows."""
        manifest = _manifest(tmp_path, [0])
        manifest["arm_families"] = ["all-features", "mrmr", "random-<k>"]
        notes = check_run_against_manifest(manifest, _records([0], arm="late-addition"))
        assert any("UNDECLARED ARMS" in note for note in notes)

    def test_an_undeclared_scenario_is_named(self, tmp_path: Path) -> None:
        """A scenario added after the fact is the cheapest way to change a conclusion, so it is surfaced."""
        notes = check_run_against_manifest(_manifest(tmp_path, [0]), _records([0], scenario="added-later"))
        assert any("UNDECLARED SCENARIOS" in note and "added-later" in note for note in notes)

    def test_a_changed_preregistration_invalidates_the_binding(self, tmp_path: Path) -> None:
        """A document edited after the run did not bind it, and the report says so rather than citing it."""
        manifest = _manifest(tmp_path, [0])
        manifest["preregistration_sha256"] = "0" * 64
        notes = check_run_against_manifest(manifest, _records([0]))
        assert any("PRE-REGISTRATION CHANGED" in note for note in notes)

    def test_an_unhashed_preregistration_is_reported(self, tmp_path: Path) -> None:
        """A run not tied to a version of the document is not a run bound by it."""
        manifest = _manifest(tmp_path, [0])
        manifest["preregistration_sha256"] = None
        notes = check_run_against_manifest(manifest, _records([0]))
        assert any("not hashed at run time" in note for note in notes)

    def test_a_different_environment_warns_about_timings_only(self, tmp_path: Path) -> None:
        """Quality metrics survive a machine change; wall-clock does not, and the note says which."""
        manifest = _manifest(tmp_path, [0])
        manifest["environment"] = {"platform": "some-other-box"}
        notes = check_run_against_manifest(manifest, _records([0]))
        assert any("ENVIRONMENT DIFFERS" in note and "timings do not" in note for note in notes)

    def test_no_manifest_is_itself_a_note(self, tmp_path: Path) -> None:
        """An undeclared run is not a clean run, and reads as neither silent nor fine."""
        notes = check_run_against_manifest(None, _records([0]))
        assert len(notes) == 1 and "nothing about the run was declared" in notes[0]


class TestContent:
    """What the manifest has to carry for the checks to mean anything."""

    def test_records_the_environment_tuple_timings_are_compared_within(self, tmp_path: Path) -> None:
        """Without the platform and library versions, a timing comparison has no denominator."""
        manifest = _manifest(tmp_path, [0])
        env = manifest["environment"]
        assert env["platform"]
        assert "python" in env["versions"]

    def test_serializes_to_stable_json(self, tmp_path: Path) -> None:
        """Keys are sorted on write, so a manifest can be hashed or diffed without spurious churn."""
        results = str(tmp_path / "cells.jsonl")
        write_manifest(results, _manifest(tmp_path, [0]))
        text = Path(manifest_path_for(results)).read_text(encoding="utf-8")
        keys = list(json.loads(text))
        assert keys == sorted(keys)
