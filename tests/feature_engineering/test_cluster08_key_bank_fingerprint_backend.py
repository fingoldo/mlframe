"""Cluster 08 regression: the key-bank cache fingerprint includes the ANN backend (audit 2026-11-20, finding 08-F8).

Indices built by one backend were reused for another because the fingerprint ignored it. The backend is now hashed, so a different backend
addresses a different cache entry, and the same backend keeps its key.
"""

import numpy as np

from mlframe.feature_engineering.transformer._key_bank import _key_bank_fingerprint


def _fp(**overrides):
    """Fingerprint of a small fixed bank with optional overrides."""
    kwargs = dict(
        X_train=np.arange(40, dtype=np.float32).reshape(10, 4),
        seed=0,
        n_heads=2,
        head_dim=3,
        metric="cosine",
        standardize=True,
        ann_M=8,
        ann_ef_construction=50,
        projection="random",
        dtype=np.float32,
    )
    kwargs.update(overrides)
    return _key_bank_fingerprint(**kwargs)


def test_different_ann_backend_changes_the_key():
    """Two backends give two different cache keys (08-F8)."""
    assert _fp(ann_backend="pynndescent") != _fp(ann_backend="hnswlib")


def test_same_ann_backend_keeps_the_key():
    """Control: the same backend and inputs give the same key (08-F8)."""
    assert _fp(ann_backend="pynndescent") == _fp(ann_backend="pynndescent")
