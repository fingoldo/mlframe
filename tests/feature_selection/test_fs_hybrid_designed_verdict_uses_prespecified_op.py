"""Audit 2026-11-20 finding 12-F29: the designed-bed verdict reads the pre-specified right operator, not the best test AUC."""

import sys
from pathlib import Path

import pytest

_FS_HYBRID = Path(__file__).resolve().parents[2] / "src/mlframe/feature_selection/_benchmarks/fs_hybrid"


@pytest.fixture(scope="module")
def bench():
    """Import the bench with its sibling modules on the path."""
    sys.path.insert(0, str(_FS_HYBRID))
    try:
        import fe_richops_bench
    finally:
        sys.path.remove(str(_FS_HYBRID))
    return fe_richops_bench


def test_designed_recovery_uses_the_prespecified_operator_not_the_max(bench):
    """A higher-scoring other operator must not be chosen: the gated bed is judged on gated0 only."""
    op, val = bench.designed_recovery({"gated0": 0.5, "ratio": 0.9}, "gated")
    assert (op, val) == ("gated0", 0.5)


def test_designed_recovery_is_nan_when_the_operator_is_absent(bench):
    """A missing operator gives NaN, not a substitute from the others."""
    op, val = bench.designed_recovery({"ratio": 0.9}, "gated")
    assert op == "gated0" and val != val  # NaN
