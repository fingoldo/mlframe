"""Audit 2026-11-20 finding 12-F28: a size with a failed cell must not get a family mean computed over its surviving cells."""

from mlframe.feature_selection._benchmarks.fs_hybrid.sweep_recovery_vs_n import _family_curve


def test_size_with_a_failed_cell_is_excluded_from_the_curve():
    """Size 100 has one failed (NaN) cell, so it has no mean; size 200 is complete and keeps its mean."""
    rows = [
        {"arm": "a", "n_samples": 100, "recall": 0.8},
        {"arm": "a", "n_samples": 100, "recall": float("nan")},
        {"arm": "a", "n_samples": 200, "recall": 0.6},
        {"arm": "a", "n_samples": 200, "recall": 0.8},
    ]
    assert _family_curve(rows, ["a"]) == {200: 0.7}


def test_all_finite_cells_average_as_before():
    """With no failures the curve is the plain per-size mean."""
    rows = [{"arm": "a", "n_samples": 50, "recall": 0.5}, {"arm": "a", "n_samples": 50, "recall": 1.0}]
    assert _family_curve(rows, ["a"]) == {50: 0.75}
