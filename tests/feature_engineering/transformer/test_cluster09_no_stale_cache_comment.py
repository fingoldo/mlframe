"""Cluster 09 regression: the random-features module no longer documents a removed GPU cache (audit 2026-11-20, finding 09-F37).

Source-level pin: the stale note about the never-read _GPU_AUTO_CACHE is gone, and the module keeps no reference to that name.
"""

from pathlib import Path

from mlframe.feature_engineering.transformer import random_features as rf

_SRC = Path(rf.__file__).resolve()


def test_stale_gpu_cache_note_is_removed():
    """No reference to the removed _GPU_AUTO_CACHE remains in the module (09-F37)."""
    assert "_GPU_AUTO_CACHE" not in _SRC.read_text(encoding="utf-8")
