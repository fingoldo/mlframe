"""Audit 2026-11-20 finding 12-F33: the MRMR default flip is judged on paired deltas over several splits, not one split and a fixed tolerance."""

import sys
from pathlib import Path

import pytest

_FS_HYBRID = Path(__file__).resolve().parents[2] / "src/mlframe/feature_selection/_benchmarks/fs_hybrid"


@pytest.fixture(scope="module")
def realdata_bench():
    """Import the realdata bench with its sibling modules on the path."""
    sys.path.insert(0, str(_FS_HYBRID))
    try:
        import round3_realdata_bench as mod
    finally:
        sys.path.remove(str(_FS_HYBRID))
    return mod


def test_consistent_small_loss_within_noise_is_not_a_proof_of_holding(realdata_bench):
    """Deltas around -0.002 with spread: the lower 95% bound falls below -0.003, so the variant is not shown to hold."""
    holds, mean, _se = realdata_bench.flip_verdict([-0.002, -0.004, 0.001, -0.006, -0.001])
    assert mean < 0
    assert holds is False


def test_clear_non_negative_deltas_hold(realdata_bench):
    """Deltas that are all clearly non-negative hold."""
    holds, mean, _se = realdata_bench.flip_verdict([0.01, 0.012, 0.009, 0.011, 0.01])
    assert holds is True and mean > 0


def test_single_split_cannot_hold(realdata_bench):
    """With one split there is no spread estimate, so the verdict cannot claim the variant holds."""
    holds, _mean, se = realdata_bench.flip_verdict([0.5])
    assert holds is False and se == float("inf")
