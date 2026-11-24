"""Cluster 32 regression: the group-target cast is a single assignment (audit 2026-11-20, finding 32-F16).

Both branches of the binary/regression test cast the target to float, so the test was dead. It is now one assignment; the source no longer
contains the identical branches.
"""

from pathlib import Path

from mlframe.training.baselines import _dummy_baseline_compute as dbc


def test_identical_branches_collapsed():
    """The target cast is not duplicated across a binary-classification branch (32-F16)."""
    src = Path(dbc.__file__).resolve().read_text(encoding="utf-8")
    assert 'if target_type == "binary_classification":\n        # For DummyClassifier-style pred' not in src
    assert src.count("y_series = pd.Series(train_y).astype(float)") == 1
