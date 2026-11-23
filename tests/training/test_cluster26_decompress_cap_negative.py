"""Cluster 26 regression: a negative decompression cap falls back to the default rather than disabling the cap (audit 2026-11-20, finding 26-F32)."""

import pytest

from mlframe.training import _bounded_zstd as bz


@pytest.fixture
def _env(monkeypatch):
    """Expose a setter for the cap environment variable."""

    def _set(value):
        """Set or clear the environment variable."""
        if value is None:
            monkeypatch.delenv(bz.MAX_DECOMPRESSED_ENV, raising=False)
        else:
            monkeypatch.setenv(bz.MAX_DECOMPRESSED_ENV, value)

    return _set


def test_negative_cap_uses_default_not_disabled(_env, monkeypatch):
    """A typo like -1 yields the default ceiling, not 0 (no cap) (26-F32)."""
    monkeypatch.setattr(bz, "_default_max_decompressed_bytes", lambda: 12345)
    _env("-1")
    assert bz.max_decompressed_bytes() == 12345


def test_zero_still_disables_explicitly(_env):
    """Control: 0 remains the documented explicit disable (26-F32)."""
    _env("0")
    assert bz.max_decompressed_bytes() == 0


def test_positive_cap_is_used(_env):
    """Control: a positive value is the cap (26-F32)."""
    _env("1000")
    assert bz.max_decompressed_bytes() == 1000
