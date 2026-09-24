"""The random-features module must not carry a GPU-decision cache that nothing reads or writes."""

import mlframe.feature_engineering.transformer.random_features as rf


def test_the_unused_gpu_auto_cache_is_gone():
    """Readers took it for a memo of the auto-GPU probe, so the per-call probe cost went unnoticed."""
    assert not hasattr(rf, "_GPU_AUTO_CACHE")
