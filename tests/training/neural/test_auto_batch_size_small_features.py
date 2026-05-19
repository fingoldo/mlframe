"""Pre-fix the auto-batch ceiling was 1024, which forced 4M-row narrow-feature
runs into ~4000 batches per epoch (single-thread CPU-bound DataLoader = idle
GPU). New ceiling lets the memory-budget computation actually bind, so
narrow-feature frames get ~thousands at a time.
"""
from __future__ import annotations

import pytest


@pytest.mark.fast
def test_auto_batch_ceiling_is_permissive_for_small_features():
    from mlframe.training.mlp_runtime_defaults import resolve_mlp_train_batch_size

    # 25 features, plenty of memory: resolver should pick well above old 1024 ceiling.
    bs = resolve_mlp_train_batch_size(
        n_features=25,
        available_memory_bytes=8 * 1024 ** 3,  # 8 GB budget
    )
    assert bs >= 8192, (
        f"With 25 features and 8 GB budget auto batch_size should be >= 8192, "
        f"got {bs}. Old ceiling 1024 made narrow-feature training catastrophically slow."
    )


@pytest.mark.fast
def test_auto_batch_floor_respected():
    """When memory budget is tiny, resolver clamps to min (32), not below."""
    from mlframe.training.mlp_runtime_defaults import resolve_mlp_train_batch_size

    bs = resolve_mlp_train_batch_size(
        n_features=1000,
        available_memory_bytes=1024,  # 1 KB - absurdly small
    )
    assert bs >= 32, f"batch_size floor (32) violated: got {bs}"


@pytest.mark.fast
def test_auto_batch_wide_features_shrinks():
    """Memory budget still binds on wide-feature frames."""
    from mlframe.training.mlp_runtime_defaults import resolve_mlp_train_batch_size

    bs_narrow = resolve_mlp_train_batch_size(
        n_features=25, available_memory_bytes=8 * 1024 ** 3,
    )
    bs_wide = resolve_mlp_train_batch_size(
        n_features=10_000, available_memory_bytes=8 * 1024 ** 3,
    )
    assert bs_wide < bs_narrow, (
        f"Wider features should yield smaller batch_size at fixed memory budget. "
        f"narrow(25)={bs_narrow}, wide(10000)={bs_wide}"
    )


@pytest.mark.fast
def test_auto_batch_ceiling_not_unbounded():
    """Ceiling exists so we don't OOM on absurdly-cheap rows."""
    from mlframe.training.mlp_runtime_defaults import resolve_mlp_train_batch_size

    bs = resolve_mlp_train_batch_size(
        n_features=1,
        available_memory_bytes=128 * 1024 ** 3,  # 128 GB
    )
    assert bs <= 65536, f"batch_size ceiling (65536) violated: got {bs}"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "--no-cov", "-x", "-s", "--tb=short"]))
