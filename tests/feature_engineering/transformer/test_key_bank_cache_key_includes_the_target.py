"""The key-bank cache key changes with the target, so another y never loads a bank built for the first one."""

from __future__ import annotations

import numpy as np

from mlframe.feature_engineering.transformer._key_bank import _key_bank_fingerprint


def _kw():
    """Build parameters shared by the fingerprints below."""
    rng = np.random.default_rng(0)
    return dict(X_train=rng.normal(size=(60, 4)).astype(np.float32), seed=0, n_heads=2, head_dim=2, metric="cosine", standardize=True,
                ann_M=16, ann_ef_construction=100, projection="pls", dtype=np.float32)


def test_different_targets_give_different_fingerprints():
    """Same X, two targets: before, both mapped to one cache entry holding the first target's y_train and projections."""
    kw = _kw()
    y1 = np.arange(60, dtype=np.float64)
    y2 = y1[::-1].copy()
    assert _key_bank_fingerprint(**kw, y_train=y1) != _key_bank_fingerprint(**kw, y_train=y2)


def test_same_target_gives_the_same_fingerprint():
    """The fingerprint stays deterministic for equal content."""
    kw = _kw()
    y = np.arange(60, dtype=np.float64)
    assert _key_bank_fingerprint(**kw, y_train=y) == _key_bank_fingerprint(**kw, y_train=y.copy())


def test_omitting_the_target_keeps_the_legacy_key():
    """Callers that do not pass y_train get the previous X-only key."""
    kw = _kw()
    assert _key_bank_fingerprint(**kw) == _key_bank_fingerprint(**kw, y_train=None)
