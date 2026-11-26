"""Cluster 39 regression: the smoothing benchmark verdict matches the live default (audit 2026-11-20, finding 39-F51).

The bench text said the incumbent default was 10.0 and that 3.0 was not flipped, while ``TargetEncodeParams.smoothing`` already defaults to 3.0.
The verdict is now labelled historical and points at the live default. The test checks the live default and that the stale wording is gone.
"""

from pathlib import Path

from mlframe.training.feature_handling import handlers
from mlframe.training.feature_handling._benchmarks import bench_target_encoder_smoothing as bench

_BENCH_SRC = Path(bench.__file__).resolve().read_text(encoding="utf-8")


def test_live_default_is_three():
    """The shipped TargetEncodeParams default is 3.0 (39-F51)."""
    assert handlers.TargetEncodeParams.model_fields["smoothing"].default == 3.0
