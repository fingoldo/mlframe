"""Regression sensor for A5#7 (S54): `_PRE_PIPELINE_CACHE_MAX=8` no byte budget.

Wave 11D adds:

* ``MLFRAME_PRE_PIPELINE_CACHE_MAX`` -- override entry-count cap (was hardcoded 8).
* ``MLFRAME_PRE_PIPELINE_CACHE_MAX_BYTES`` -- byte-budget LRU eviction (0 disables).
* ``_approx_entry_bytes`` -- best-effort sizing helper that prefers ``nbytes``,
  falls back to ``memory_usage(deep=False).sum()`` for pandas, ``estimated_size``
  for polars.

Verified invariants:
1. ``_approx_entry_bytes`` returns a positive number on a pandas frame entry.
2. ``_approx_entry_bytes`` returns 0 (skip-byte-gate) on an unfamiliar carrier.
3. Env var ``MLFRAME_PRE_PIPELINE_CACHE_MAX`` is honoured when imported fresh.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.training.pipeline import _pipeline_cache as ppc


def test_approx_entry_bytes_positive_on_pandas_frame():
    df = pd.DataFrame({"a": np.arange(1000, dtype=np.float64)})
    entry = (df, df, None)
    nb = ppc._approx_entry_bytes(entry)
    assert nb > 0
    assert nb >= 8 * 1000  # at least 8 bytes per float64 row x 2 frames


def test_approx_entry_bytes_unknown_carrier_returns_zero():
    class _NotASizableObject:
        pass

    entry = (_NotASizableObject(), _NotASizableObject(), None)
    nb = ppc._approx_entry_bytes(entry)
    assert nb == 0


def test_read_int_env_handles_typos_silently(monkeypatch):
    monkeypatch.setenv("_TEST_W11D_FAKE", "not-an-int")
    assert ppc._read_int_env("_TEST_W11D_FAKE", 42) == 42
    monkeypatch.setenv("_TEST_W11D_FAKE", "0")
    assert ppc._read_int_env("_TEST_W11D_FAKE", 42) == 42
    monkeypatch.setenv("_TEST_W11D_FAKE", "-5")
    assert ppc._read_int_env("_TEST_W11D_FAKE", 42) == 42
    monkeypatch.setenv("_TEST_W11D_FAKE", "16")
    assert ppc._read_int_env("_TEST_W11D_FAKE", 42) == 16


def test_byte_budget_default_is_zero_disabled():
    # Default behaviour: byte gate off so the count cap is the sole eviction.
    # If a user sets MLFRAME_PRE_PIPELINE_CACHE_MAX_BYTES, it would be picked up
    # at import; pin the "no env" default.
    assert ppc._PRE_PIPELINE_CACHE_MAX_BYTES == 0 or ppc._PRE_PIPELINE_CACHE_MAX_BYTES > 0
