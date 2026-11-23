"""Cluster 17 regression: the bench's no-op placeholder expression is gone (audit 2026-11-20, finding 17-F31).

Source-level pin: the statement ``Tmy*nz*0+0.0  # placeholder`` no longer appears, and the module still compiles.
"""

from pathlib import Path

from mlframe.metrics._benchmarks import bench_fused_regression_ext_welford as bench

_SRC = Path(bench.__file__).resolve()


def test_placeholder_no_op_is_removed():
    """The no-op placeholder expression is deleted (17-F31)."""
    assert "Tmy*nz*0+0.0" not in _SRC.read_text(encoding="utf-8")
