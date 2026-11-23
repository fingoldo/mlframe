"""Cluster 42 regression: the minimax benchmark times the shipped class, not a second frozen copy (audit 2026-11-20, finding 42-F10)."""

from mlframe.votenrank._benchmarks import bench_minimax_winning_votes as bench


def test_no_second_frozen_copy_remains():
    """Only the old formulation is kept as a fixture; the duplicate of the shipped logic is gone (42-F10)."""
    assert not hasattr(bench, "_new_minimax")
    assert hasattr(bench, "_old_minimax")


def test_bench_identity_check_passes_against_shipped_class(capsys):
    """Running the bench on a small table passes its identity check against Leaderboard.minimax_ranking (42-F10)."""
    bench.bench(6, 4, reps=1)
    assert "identity=OK" in capsys.readouterr().out
