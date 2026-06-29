"""Unit tests for the RAM-relative polars->pandas view cache budget resolver.

The budget gates whether a large transformed-feature view is REUSED across the per-target loop (10
composite targets -> one conversion) vs evicted-and-recomputed each target. It is configurable by
TYPE x SIZE so it scales with the host instead of a fixed 2 GB cap that self-evicts a 10 GB view.
"""
from __future__ import annotations

import psutil
import pytest

from mlframe.training.core._phase_train_one_target_polars_fastpath import (
    resolve_pandas_view_cache_budget_bytes as _resolve,
)

_VARS = (
    "MLFRAME_PANDAS_VIEW_CACHE_TYPE",
    "MLFRAME_PANDAS_VIEW_CACHE_SIZE",
    "MLFRAME_PANDAS_VIEW_CACHE_MAX_MB",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for v in _VARS:
        monkeypatch.delenv(v, raising=False)
    yield


def test_default_is_free_ram_share_0p2():
    expected = 0.2 * float(psutil.virtual_memory().available)
    got = _resolve()
    assert got == pytest.approx(expected, rel=0.05)  # free RAM drifts slightly between the two reads


def test_total_ram_share(monkeypatch):
    monkeypatch.setenv("MLFRAME_PANDAS_VIEW_CACHE_TYPE", "TOTAL_RAM_SHARE")
    monkeypatch.setenv("MLFRAME_PANDAS_VIEW_CACHE_SIZE", "0.1")
    expected = 0.1 * float(psutil.virtual_memory().total)
    assert _resolve() == pytest.approx(expected, rel=0.02)


def test_absolute_mb(monkeypatch):
    monkeypatch.setenv("MLFRAME_PANDAS_VIEW_CACHE_TYPE", "ABSOLUTE_MB")
    monkeypatch.setenv("MLFRAME_PANDAS_VIEW_CACHE_SIZE", "3000")
    assert _resolve() == 3000 * (1024 ** 2)


def test_legacy_max_mb_alias_when_new_vars_unset(monkeypatch):
    monkeypatch.setenv("MLFRAME_PANDAS_VIEW_CACHE_MAX_MB", "5000")
    assert _resolve() == 5000 * (1024 ** 2)


def test_new_vars_take_precedence_over_legacy(monkeypatch):
    monkeypatch.setenv("MLFRAME_PANDAS_VIEW_CACHE_MAX_MB", "5000")
    monkeypatch.setenv("MLFRAME_PANDAS_VIEW_CACHE_TYPE", "ABSOLUTE_MB")
    monkeypatch.setenv("MLFRAME_PANDAS_VIEW_CACHE_SIZE", "1000")
    assert _resolve() == 1000 * (1024 ** 2)  # not the 5000 legacy value


def test_malformed_size_falls_back_to_2gb(monkeypatch):
    monkeypatch.setenv("MLFRAME_PANDAS_VIEW_CACHE_TYPE", "ABSOLUTE_MB")
    monkeypatch.setenv("MLFRAME_PANDAS_VIEW_CACHE_SIZE", "not-a-number")
    assert _resolve() == 2048.0 * (1024 ** 2)


def test_free_ram_share_budget_admits_a_10gb_view_on_a_high_ram_host():
    """On a host with >=60 GB free, the default 0.2 share must exceed a ~10 GB view so it is reused."""
    free_gb = psutil.virtual_memory().available / (1024 ** 3)
    if free_gb < 60:
        pytest.skip(f"host has only {free_gb:.0f} GB free; the >=10 GB reuse assertion needs >=60 GB free")
    assert _resolve() > 10 * (1024 ** 3)
