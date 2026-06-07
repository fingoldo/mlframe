"""Regression tests for DiscoveryCache wiring (P0 ship-or-strip -> ship).

Validates:
- The discovery config signature embeds library versions so a mocked
  ``mlframe.__version__`` bump invalidates the cache.
- Two consecutive lookups with identical inputs return the cached payload.
- The signature is deterministic across repeated calls (sort_keys JSON).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from mlframe.training.composite.cache import (
    DiscoveryCache,
    data_signature,
    make_discovery_cache_key,
)
from mlframe.training.core._phase_composite_discovery import (
    _discovery_config_signature,
)


class _FakeConfig:
    """Minimal pydantic-like stand-in: model_dump returns a stable dict."""

    def __init__(self, **kwargs: Any) -> None:
        self._fields = dict(kwargs)

    def model_dump(self, mode: str = "python") -> dict[str, Any]:
        return dict(self._fields)


def _fake_df(seed: int = 0, n: int = 64) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "target": rng.normal(size=n),
        "base": rng.normal(size=n),
        "f1": rng.normal(size=n),
        "f2": rng.normal(size=n),
    })


def test_discovery_config_signature_is_deterministic():
    cfg = _FakeConfig(enabled=True, top_k=3, random_state=42)
    s1 = _discovery_config_signature(cfg)
    s2 = _discovery_config_signature(cfg)
    assert s1 == s2
    assert isinstance(s1, str) and len(s1) > 0


def test_discovery_config_signature_changes_when_mlframe_version_bumps():
    """Cache-poisoning protection: version change must invalidate."""
    cfg = _FakeConfig(enabled=True, top_k=3)
    with mock.patch("mlframe.__version__", "0.0.0-fake-A"):
        s_a = _discovery_config_signature(cfg)
    with mock.patch("mlframe.__version__", "0.0.0-fake-B"):
        s_b = _discovery_config_signature(cfg)
    assert s_a != s_b


def test_discovery_cache_round_trip_skips_recompute(tmp_path):
    """Two consecutive get() calls on the same key return identical payloads."""
    cfg = _FakeConfig(enabled=True, top_k=3, random_state=42)
    df = _fake_df()
    feature_cols = ["base", "f1", "f2"]

    df_sig = data_signature(df, "target", feature_cols, random_state=42)
    cfg_sig = _discovery_config_signature(cfg)
    cache_key = make_discovery_cache_key(df_sig, "target", cfg_sig, random_state=42)

    cache = DiscoveryCache(str(tmp_path))
    assert cache.get(cache_key) is None
    payload = {"specs_export": [{"name": "fake_spec"}], "failures": [], "filter_drops": {}}
    cache.set(cache_key, payload)
    # Second discovery call would look this up and skip recompute.
    roundtrip = cache.get(cache_key)
    assert roundtrip == payload


def test_discovery_cache_miss_after_version_bump(tmp_path):
    """Same df + same logical config but bumped mlframe version -> miss."""
    cfg = _FakeConfig(enabled=True, top_k=3)
    df = _fake_df()
    feature_cols = ["base", "f1", "f2"]
    df_sig = data_signature(df, "target", feature_cols, random_state=42)

    cache = DiscoveryCache(str(tmp_path))
    with mock.patch("mlframe.__version__", "1.0.0-A"):
        key_a = make_discovery_cache_key(
            df_sig, "target", _discovery_config_signature(cfg), random_state=42,
        )
        cache.set(key_a, {"specs_export": [{"name": "a"}]})

    with mock.patch("mlframe.__version__", "1.0.0-B"):
        key_b = make_discovery_cache_key(
            df_sig, "target", _discovery_config_signature(cfg), random_state=42,
        )
        # Bumped version must yield a different key (cache MISS).
        assert key_a != key_b
        assert cache.get(key_b) is None
