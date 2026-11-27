"""Cluster 12 regression: the phase-0 runner refuses an unknown ``--arms`` name instead of running a silently smaller roster (audit 2026-11-20, 12-F76)."""

import numpy as np
import pytest

from mlframe.feature_selection._benchmarks.fs_hybrid import _arms, _real_beds, run_experiment, run_phase0


def _patch(monkeypatch, ran):
    """One tiny bed, a three-arm roster, and a run_grid that records its roster instead of running."""
    monkeypatch.setattr(_real_beds, "real_bed_scenarios", lambda include_ineligible=False: [("bed", lambda seed: (np.zeros((5, 4)), np.zeros(5)))])
    monkeypatch.setattr(_arms, "build_arm_roster", lambda width, random_state=0: {"oracle-informative": object, "all-features": object, "random-1": object})

    def fake_run_grid(**kwargs):
        """Record the roster and report one executed cell."""
        ran.append(kwargs["roster"])
        return 1

    monkeypatch.setattr(run_experiment, "run_grid", fake_run_grid)


def test_misspelt_arm_name_raises_and_runs_nothing(monkeypatch, tmp_path):
    """A typo in --arms raises ValueError naming it; no grid is run (12-F76)."""
    ran: list = []
    _patch(monkeypatch, ran)
    with pytest.raises(ValueError, match="oracle-informatve"):
        run_phase0.main(["--arms", "oracle-informatve", "--out", str(tmp_path / "o.jsonl")])
    assert ran == []


def test_known_arm_name_builds_the_subset_roster(monkeypatch, tmp_path):
    """A valid name keeps that arm plus the random arms and drops the rest (12-F76)."""
    ran: list = []
    _patch(monkeypatch, ran)
    assert run_phase0.main(["--arms", "oracle-informative", "--out", str(tmp_path / "o.jsonl")]) == 1
    assert sorted(ran[0]) == ["oracle-informative", "random-1"]
