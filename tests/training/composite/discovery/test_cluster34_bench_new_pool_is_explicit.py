"""Audit 2026-11-20 finding 34-F32: the bench's NEW arm contains the widened transforms, not the config default.

The config default excludes the four widened transforms, so the "NEW" arm of the full-pool comparison did not test them.
"""

from mlframe.training.composite.discovery._benchmarks import bench_widened_default_transforms as bench


def test_new_pool_includes_the_widened_transforms():
    """The NEW pool is the old pool followed by the four widened transforms."""
    pool = bench._full_new_pool()
    for name in bench._NEW_EXTRA_TRANSFORMS:
        assert name in pool
    assert pool == list(bench._OLD_TRANSFORMS) + list(bench._NEW_EXTRA_TRANSFORMS)
