"""Audit 2026-11-20 finding 12-F30: every preset of the cost bench fits on the SAME train/holdout split."""

import sys
from pathlib import Path

import numpy as np
import pytest

_FS_HYBRID = Path(__file__).resolve().parents[2] / "src/mlframe/feature_selection/_benchmarks/fs_hybrid"


@pytest.fixture(scope="module")
def cost_bench():
    """Import the cost bench with its sibling modules on the path."""
    sys.path.insert(0, str(_FS_HYBRID))
    try:
        import round4_fe_richops_cost_bench as mod
    finally:
        sys.path.remove(str(_FS_HYBRID))
    return mod


def test_split_is_the_same_on_every_call(cost_bench):
    """Repeated calls on the same bed size give identical train and holdout rows, so presets are comparable."""
    tr1, te1 = cost_bench.split_indices(500)
    tr2, te2 = cost_bench.split_indices(500)
    np.testing.assert_array_equal(tr1, tr2)
    np.testing.assert_array_equal(te1, te2)


def test_split_partitions_rows_at_sixty_percent(cost_bench):
    """The split is a partition of all rows with 60% for training."""
    tr, te = cost_bench.split_indices(500)
    assert len(tr) == 300 and len(te) == 200
    assert set(tr.tolist()) | set(te.tolist()) == set(range(500))
