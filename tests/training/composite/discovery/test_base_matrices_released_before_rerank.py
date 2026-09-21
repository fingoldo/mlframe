"""The per-base screening matrices must be released before the rerank, not held until ``fit`` returns.

The candidate loop builds a full feature matrix plus a base-dropped float and prebinned copy per base. Nothing reads
them afterwards, yet they stayed referenced through the tiny rerank, the holdout gates, auto-chain and the honest
re-score - an estimated 1.2 GB at 100k x 500 with three bases, during the most memory-heavy phases.
"""

from __future__ import annotations

import gc
import weakref

import numpy as np
import pandas as pd

from mlframe.training.composite import CompositeTargetDiscovery
from mlframe.training.composite.discovery import CompositeTargetDiscovery as _Disc
from mlframe.training.configs import CompositeTargetDiscoveryConfig


def _frame(n: int = 1500, seed: int = 0) -> pd.DataFrame:
    """A frame with a clear additive base so the rerank has specs to score."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(10.0, 50.0, n)
    x0, x1 = rng.normal(size=n), rng.normal(size=n)
    return pd.DataFrame({"base": base, "x0": x0, "x1": x1, "y": 1.5 * base + 2.0 * x0 + rng.normal(size=n)})


def test_the_full_feature_matrix_is_gone_when_the_rerank_starts(monkeypatch):
    """A weak reference to the built matrix must be dead by the time the tiny rerank runs."""
    refs: list = []
    real_build = _Disc._build_feature_matrix

    def tracking_build(self, *args, **kwargs):
        """Build the matrix as usual and keep only a weak reference to it."""
        out = real_build(self, *args, **kwargs)
        refs.append(weakref.ref(out))
        return out

    alive_at_rerank: list[bool] = []
    real_rerank = _Disc._tiny_model_rerank

    def observing_rerank(self, *args, **kwargs):
        """Record whether the screening matrix is still referenced, then rerank as usual."""
        gc.collect()
        alive_at_rerank.append(any(r() is not None for r in refs))
        return real_rerank(self, *args, **kwargs)

    monkeypatch.setattr(_Disc, "_build_feature_matrix", tracking_build)
    monkeypatch.setattr(_Disc, "_tiny_model_rerank", observing_rerank)
    df = _frame()
    cfg = CompositeTargetDiscoveryConfig(enabled=True, random_state=0, screening="tiny_model", base_candidates=["base"], tiny_model_n_estimators=20)
    disc = CompositeTargetDiscovery(cfg)
    disc.fit(df, "y", ["base", "x0", "x1"], np.arange(len(df)))
    assert refs, "the full feature matrix was never built on this path, so the check below would prove nothing"
    assert alive_at_rerank, "the tiny rerank never ran, so the release point was not reached"
    assert not any(alive_at_rerank), "the screening matrix was still referenced when the rerank started"
