"""The fit's data signature must be computed only when something asks for it, and must survive pickling.

``discover_incremental`` is the only reader of the signature, yet every ``fit`` - and every stability replicate and
per-group fit - computed it: 84 ms on pandas and 308 ms on polars at 200k x 50, seconds on wide polars frames. It is now
computed on first request, and pickling computes it before the frame reference is dropped, so a persisted result keeps
the byte-identical warm-start fast path.
"""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from mlframe.training.composite import CompositeTargetDiscovery
from mlframe.training.composite import cache as cache_mod
from mlframe.training.composite.discovery import discover_incremental
from mlframe.training.configs import CompositeTargetDiscoveryConfig

_FEATS = ["base", "x1", "x2"]


def _frame(n: int = 800, seed: int = 0) -> pd.DataFrame:
    """A frame with a clear additive base."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(10.0, 50.0, n)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    return pd.DataFrame({"base": base, "x1": x1, "x2": x2, "y": 1.5 * base + 2.0 * x1 + rng.normal(size=n)})


def _fit(df: pd.DataFrame) -> CompositeTargetDiscovery:
    """A small discovery fit."""
    disc = CompositeTargetDiscovery(CompositeTargetDiscoveryConfig(
        enabled=True, random_state=0, base_candidates=["base"], transforms=["diff", "linear_residual"], screening="mi",
    ))
    disc.fit(df, "y", _FEATS, np.arange(len(df)))
    return disc


def _count_signatures(monkeypatch) -> dict:
    """Count every ``data_signature`` computation reachable from discovery."""
    seen = {"n": 0}
    real = cache_mod.data_signature

    def counting(*args, **kwargs):
        """Count, then compute."""
        seen["n"] += 1
        return real(*args, **kwargs)

    import mlframe.training.composite.discovery as disc_pkg

    monkeypatch.setattr(disc_pkg, "data_signature", counting)
    monkeypatch.setattr(cache_mod, "data_signature", counting)  # the call-time ``from ..cache import`` path
    return seen


def test_fit_does_not_compute_the_signature(monkeypatch):
    """A plain fit must not pay for a signature nobody has asked for."""
    seen = _count_signatures(monkeypatch)
    _fit(_frame())
    assert seen["n"] == 0, f"fit computed the data signature {seen['n']} time(s)"


def test_the_signature_matches_the_eager_value_and_is_computed_once(monkeypatch):
    """On request it equals what ``data_signature`` gives the same frame, and a second request is free."""
    df = _frame()
    disc = _fit(df)
    expected = cache_mod.data_signature(df, "y", _FEATS)
    seen = _count_signatures(monkeypatch)
    first = disc.fit_data_signature()
    second = disc.fit_data_signature()
    assert first == expected and first
    assert first == second and seen["n"] == 1


def test_a_pickled_result_keeps_the_signature_for_warm_start():
    """Pickling drops the frame, so the signature is materialised first; the warm-start still takes its fast path."""
    df = _frame()
    restored = pickle.loads(pickle.dumps(_fit(df)))  # nosec B301 -- round-trip of a locally-created, trusted object
    assert restored.fit_data_signature() == cache_mod.data_signature(df, "y", _FEATS)
    decision = discover_incremental(restored, df, "y", _FEATS)
    assert decision.reuse, "an identical frame must be reused on the byte-identical fast path"
