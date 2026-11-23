"""Cluster 12 regression: the ablation's main() returns the number of failed runs, so a fully failed grid is non-zero (audit 2026-11-20, finding 12-F85).

Driven through main() with run_one replaced by a failing stub, and with a one-seed, one-bed grid written to a temp file.
"""

from mlframe.feature_selection._benchmarks.fs_hybrid import ablation_rfecv_aggregation as ab


def _always_fails(*args, **kwargs):
    """Stand-in for a run that raises on every configuration."""
    raise RuntimeError("boom")


def test_all_runs_failing_returns_nonzero(monkeypatch, tmp_path):
    """Every configuration errors, so main() reports that many failures rather than a success count (12-F85)."""
    monkeypatch.setattr(ab, "run_one", _always_fails)
    out = tmp_path / "ablation.json"
    failures = ab.main(["--seeds", "1", "--beds", "bed_a", "--out", str(out)])
    assert failures > 0
    assert failures == len(ab.configurations())


def test_no_failed_runs_returns_zero(monkeypatch, tmp_path):
    """Control: when every run succeeds the count is zero (12-F85)."""

    def _ok(bed, seed, rule, policy, rows):
        """Return a healthy record."""
        return {"bed": bed, "seed": seed, "voting_rule": rule, "missing_policy": policy, "recall": 1.0, "precision": 1.0, "n_selected": 1}

    monkeypatch.setattr(ab, "run_one", _ok)
    out = tmp_path / "ablation.json"
    assert ab.main(["--seeds", "1", "--beds", "bed_a", "--out", str(out)]) == 0
