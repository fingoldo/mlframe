"""Cluster 28 regression: a cached validation Dataset hit clears a stale eval init_score (audit 2026-11-20, finding 28-F16)."""

import numpy as np
import pytest

lgb = pytest.importorskip("lightgbm")

from mlframe.training.lgb_shim import LGBMRegressorWithDatasetReuse


def test_val_cache_hit_drops_init_score_when_none_is_passed():
    """Second fit on the same val set with eval_init_score=None must not keep the first fit's offsets (28-F16)."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    y = X[:, 0] + rng.normal(scale=0.1, size=200)
    Xv = rng.normal(size=(60, 3))
    yv = Xv[:, 0]
    stale_offset = np.full(60, 5.0)

    m = LGBMRegressorWithDatasetReuse(n_estimators=3, verbose=-1)
    m.fit(X, y, eval_set=[(Xv, yv)], eval_init_score=[stale_offset])
    m.fit(X, y, eval_set=[(Xv, yv)], eval_init_score=None)

    cached = list(m._cached_val_datasets.values())
    assert cached, "no cached validation Dataset to inspect"
    assert cached[0].get_init_score() is None, "cache hit kept the stale eval init_score"
