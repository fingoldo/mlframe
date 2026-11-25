"""Cluster 12 regression: the parity bench summary survives --seeds 0 (audit 2026-11-20, finding 12-F86).

With no seeds there are no rows; the per-estimator summary used to call min/max on an empty array and crash. The run itself is
stubbed out so the test exercises only the summary path.
"""

from mlframe.feature_selection._benchmarks.fs_hybrid import ablation_rfecv_parity as par


def test_zero_seeds_prints_no_runs_instead_of_crashing(monkeypatch, tmp_path, capsys):
    """--seeds 0 completes and reports n=0 for each estimator (12-F86)."""

    def _never(*args, **kwargs):
        """Fail if a run is attempted at zero seeds."""
        raise AssertionError("no run should happen at --seeds 0")

    monkeypatch.setattr(par, "_run_one", _never)
    out = tmp_path / "parity.json"
    par.main(["--seeds", "0", "--out", str(out)])
    text = capsys.readouterr().out
    assert "no runs (n=0)" in text
