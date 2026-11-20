"""Audit 2026-11-20 findings 12-F36 and 12-F37: a failed run must not poison a configuration's mean or its selection sizes."""

import sys
from pathlib import Path

import pytest

_FS_HYBRID = Path(__file__).resolve().parents[2] / "src/mlframe/feature_selection/_benchmarks/fs_hybrid"


@pytest.fixture(scope="module")
def ablation():
    """Import the ablation bench with its sibling modules on the path."""
    sys.path.insert(0, str(_FS_HYBRID))
    try:
        import ablation_rfecv_aggregation as mod
    finally:
        sys.path.remove(str(_FS_HYBRID))
    return mod


def _row(rule, seed, recall, n_selected=5):
    """A synthetic result row for one configuration and seed."""
    return {"bed": "b", "seed": seed, "voting_rule": rule, "missing_policy": "worst", "recall": recall, "precision": 0.5,
            "n_selected": n_selected, "n_features": 10}


def test_failed_seed_marks_configuration_incomplete_and_leaves_no_nan(ablation):
    """Borda has a failed seed: it is flagged INCOMPLETE and excluded from the ranking, which stays finite."""
    rows = [_row("Borda", 0, 0.8), _row("Borda", 1, float("nan")), _row("Copeland", 0, 0.6), _row("Copeland", 1, 0.7)]
    text = "\n".join(ablation.summarize(rows))
    assert "Borda/worst INCOMPLETE" in text
    assert "nan" not in text.lower()


def test_failed_run_does_not_count_toward_kept_share(ablation):
    """A failed run carries n_selected=0; it must not drag the kept-column average down."""
    rows = [_row("Borda", 0, 0.8, n_selected=8), _row("Borda", 1, 0.8, n_selected=8), _row("Borda", 2, float("nan"), n_selected=0),
            _row("Copeland", 0, 0.6, n_selected=8), _row("Copeland", 1, 0.6, n_selected=8)]
    text = "\n".join(ablation.summarize(rows))
    assert "kept 8.0 of 10" in text
